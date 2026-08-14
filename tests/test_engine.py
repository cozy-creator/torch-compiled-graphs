from __future__ import annotations

import errno
import inspect
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from hashrepo import LocalCAS

import torch_compiled_graphs.engine as engine_module
import torch_compiled_graphs.host_isa as host_isa_module
from torch_compiled_graphs import (
    AdmissionError,
    Engine,
    EnsureOutcome,
    GraphClassSpec,
    QuarantinedArtifact,
    RuntimeCompatibility,
    StorageError,
    StoreOutcome,
)
from torch_compiled_graphs.artifact import pack_artifact, read_metadata
from torch_compiled_graphs.identity import from_axes
from torch_compiled_graphs.storage import _CompiledGraphStore, _quarantine_ref

torch: Any = pytest.importorskip("torch")


class Double(torch.nn.Module):  # type: ignore[misc]
    def forward(self, value: Any) -> Any:
        return value * 2


class WithLiteral(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.table = torch.tensor([2.0, 3.0])

    def forward(self, value: Any) -> Any:
        return value * self.table


def _elf() -> bytes:
    names = b"\0.shstrtab\0.lrodata\0"
    section_offset = 64
    section_size = 64
    string_offset = section_offset + section_size * 3
    image = bytearray(string_offset + len(names))
    image[:4] = b"\x7fELF"
    image[4:7] = bytes((2, 1, 1))
    struct.pack_into("<Q", image, 0x28, section_offset)
    struct.pack_into("<HHH", image, 0x3A, section_size, 3, 1)
    struct.pack_into("<II", image, section_offset + section_size, 1, 3)
    struct.pack_into("<QQ", image, section_offset + section_size + 0x18, string_offset, len(names))
    struct.pack_into("<II", image, section_offset + 2 * section_size, 11, 1)
    struct.pack_into("<QQ", image, section_offset + 2 * section_size + 0x18, len(image), 0)
    image[string_offset:] = names
    return bytes(image)


def _zip_entry(name: str, data: str | bytes) -> tuple[zipfile.ZipInfo, str | bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    return info, data


def packager(output: str, files: Mapping[str, Sequence[object]]) -> object:
    assert len(files) == 1
    entry = next(iter(files))
    with zipfile.ZipFile(output, "w") as archive:
        for info, data in (
            _zip_entry(
                f"data/aotinductor/{entry}/model.wrapper.cpp",
                "AOTInductorModelBase(1, 1, 0, device_str, std::move(cubin_dir), false)",
            ),
            _zip_entry(f"data/aotinductor/{entry}/model.so", _elf()),
        ):
            archive.writestr(info, data)
    return output


def literal_packager(output: str, files: Mapping[str, Sequence[object]]) -> object:
    assert len(files) == 1
    entry = next(iter(files))
    wrapper = "\n".join(
        (
            "AOTInductorModelBase(1, 1, 1, device_str, std::move(cubin_dir), false)",
            'constants_info_[0].name = "table";',
            'constants_info_[0].original_fqn = "table";',
            "constants_info_[0].data_size = 8;",
            "constants_info_[0].from_folded = false;",
            "constants_info_[0].type = static_cast<int32_t>(",
            "    torch::aot_inductor::ConstantType::TensorConstant);",
            "constants_info_[0].dtype = cached_torch_dtype_float32;",
            "constants_info_[0].shape = {2};",
        )
    )
    with zipfile.ZipFile(output, "w") as archive:
        for info, data in (
            _zip_entry(f"data/aotinductor/{entry}/model.wrapper.cpp", wrapper),
            _zip_entry(f"data/aotinductor/{entry}/model.so", _elf()),
        ):
            archive.writestr(info, data)
    return output


def _fake_compile_package(
    package: Callable[[str, Mapping[str, Sequence[object]]], object] = packager,
    *,
    mutate: Callable[[], None] | None = None,
) -> Callable[[Any, Path], Path]:
    def compile_package(plan: Any, workspace: Path) -> Path:
        if mutate is not None:
            mutate()
        result = package(
            str(workspace / "model.pt2"),
            {plan.declaration.graph_class: ["wrapper.cpp", "model.so"]},
        )
        return Path(str(result))

    return compile_package


def _alternate_artifact(package: Path, source_artifact: Path, output: Path) -> Path:
    alternate_package = output.with_suffix(".pt2")
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(alternate_package, "w") as target:
        for info in source.infolist():
            data = source.read(info)
            if info.filename.endswith(".so"):
                data += b"different but valid package bytes"
            target.writestr(info, data)
    return pack_artifact(alternate_package, output, read_metadata(source_artifact))


@pytest.fixture(autouse=True)
def _default_fake_compile_package(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("real_aoti") is None:
        monkeypatch.setattr(engine_module, "_compile_package", _fake_compile_package())


def _spec_for(program: object) -> GraphClassSpec:
    return GraphClassSpec(
        "model",
        "denoiser",
        program,
        {
            "v": 3,
            "lifted_inputs": [],
            "pytree": {"in": "leaf", "out": "leaf"},
            "specialization": {},
        },
        "0" * 32,
    )


def _spec() -> GraphClassSpec:
    program = torch.export.export(Double(), (torch.ones(4096),))
    return _spec_for(program)


def _instruction_prefixes(disassembly: str) -> tuple[str, ...]:
    return tuple(
        match.group(1)
        for match in re.finditer(r"^\s*[0-9a-f]+:\s+([0-9a-f]{2})(?:\s|$)", disassembly, re.M)
    )


def test_disassembly_prefix_classifier_distinguishes_vex_and_evex() -> None:
    disassembly = """
       0: c5 fc 58 c1           vaddps %ymm1,%ymm0,%ymm0
       4: 62 f1 7c 48 58 c1     vaddps %zmm1,%zmm0,%zmm0
    """
    assert _instruction_prefixes(disassembly) == ("c5", "62")


def _runtime() -> RuntimeCompatibility:
    return RuntimeCompatibility("cpu", toolchain=_toolchain())


def _toolchain(*, torch_digest: str = "torch-record-v1") -> dict[str, str]:
    return {
        "settings_declaration": "settings-v1",
        "loaded_libs": "loaded-libs-v1",
        "torch": torch_digest,
        "triton": "triton-record-v1",
    }


def test_engine_signatures_expose_no_output_changing_compile_seam() -> None:
    forbidden = {"compiler", "packager", "context", "options", "before_compile"}
    assert forbidden.isdisjoint(inspect.signature(Engine.ensure).parameters)
    assert forbidden.isdisjoint(inspect.signature(Engine._mint).parameters)


def test_fresh_engine_reuses_local_hashrepo_without_compiling(tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    spec = _spec()
    key = _runtime().key(spec.declare())
    first = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "first",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    assert first.outcome == EnsureOutcome.MINTED
    assert first.compiled_graph.package.is_file()

    def must_not_construct() -> GraphClassSpec:
        raise AssertionError("restart reuse invoked the lazy recipe")

    second = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "second",
        target="cpu",
        toolchain=_toolchain(),
        recipe=must_not_construct,
    )
    assert second.outcome == EnsureOutcome.REUSED
    assert second.compiled_graph.package.read_bytes() == first.compiled_graph.package.read_bytes()

    cold_destination = tmp_path / "cold-process"
    cold = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from hashrepo import LocalCAS
from torch_compiled_graphs import Engine, EnsureOutcome
assert "torch" not in sys.modules
def forbidden_recipe():
    raise AssertionError("cache hit invoked recipe")
result = Engine(LocalCAS(sys.argv[1])).ensure(
    sys.argv[2], Path(sys.argv[3]), target="cpu",
    toolchain={"settings_declaration":"settings-v1","loaded_libs":"loaded-libs-v1","torch":"torch-record-v1","triton":"triton-record-v1"},
    recipe=forbidden_recipe,
)
assert result.outcome == EnsureOutcome.REUSED
assert "torch" not in sys.modules
""",
            str(cas_root),
            str(key),
            str(cold_destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert cold.returncode == 0
    assert (cold_destination / "model.pt2").is_file()

    process_destination = tmp_path / "process"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch_compiled_graphs",
            "resolve",
            "--cas-root",
            str(cas_root),
            first.compiled_graph.key,
            str(process_destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (process_destination / "model.pt2").is_file()


@pytest.mark.parametrize(
    ("target", "torch_digest", "pattern"),
    [
        ("sm_90", "torch-record-v1", "stored target"),
        ("cpu", "different-record", "stored toolchain axis"),
    ],
)
def test_cache_hit_rejects_requested_runtime_mismatch_without_recipe(
    target: str,
    torch_digest: str,
    pattern: str,
    tmp_path: Path,
) -> None:
    cas_root = tmp_path / "cas"
    spec = _spec()
    key = _runtime().key(spec.declare())
    Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "seed",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )

    def forbidden_recipe() -> GraphClassSpec:
        raise AssertionError("mismatched cache hit invoked recipe")

    with pytest.raises(AdmissionError, match=pattern):
        Engine(LocalCAS(cas_root)).ensure(
            key,
            tmp_path / "mismatch",
            target=target,
            toolchain=_toolchain(torch_digest=torch_digest),
            recipe=forbidden_recipe,
        )


def test_cache_hit_ignores_trace_only_model_library_records(tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    spec = _spec()
    key = _runtime().key(spec.declare())
    recorded = _toolchain()
    recorded["diffusers"] = "record-a"
    Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "seed",
        target="cpu",
        toolchain=recorded,
        recipe=lambda: spec,
    )
    current = _toolchain()
    current["diffusers"] = "record-b"
    reused = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "reused",
        target="cpu",
        toolchain=current,
        recipe=lambda: (_ for _ in ()).throw(AssertionError("cache hit invoked recipe")),
    )
    assert reused.outcome == EnsureOutcome.REUSED


def test_corrupt_local_object_is_quarantined_and_repaired(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    spec = _spec()
    key = _runtime().key(spec.declare())
    first = Engine(cas).ensure(
        key,
        tmp_path / "first",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    manifest = cas.load_manifest(first.compiled_graph.manifest)
    cas.object_path(manifest.files[0].digest).write_bytes(b"corrupt")

    repaired = Engine(LocalCAS(tmp_path / "cas")).ensure(
        key,
        tmp_path / "repaired",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    assert repaired.outcome == EnsureOutcome.MINTED
    assert repaired.compiled_graph.package.is_file()


def test_repairing_with_a_previously_divergent_manifest_retires_its_stale_quarantine(
    tmp_path: Path,
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    engine = Engine(cas)
    spec = _spec()
    key = _runtime().key(spec.declare())
    minted = engine.ensure(
        key,
        tmp_path / "minted",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    artifact_a = engine.export_artifact(key, tmp_path / "artifact-a.tar.gz")
    artifact_b = _alternate_artifact(
        minted.compiled_graph.package,
        artifact_a,
        tmp_path / "artifact-b.tar.gz",
    )
    divergent = engine.import_artifact(key, artifact_b)
    assert divergent.outcome == StoreOutcome.DIVERGENT

    manifest_a = cas.load_manifest(minted.compiled_graph.manifest)
    cas.object_path(manifest_a.files[0].digest).write_bytes(b"corrupt-a")
    with pytest.raises(QuarantinedArtifact, match="CAS verification"):
        engine.resolve(key, tmp_path / "corrupt-a")

    repaired = engine.import_artifact(key, artifact_b)
    assert repaired.outcome == StoreOutcome.REPAIRED
    assert repaired.manifest == divergent.manifest
    resolved = engine.resolve(key, tmp_path / "resolved-b")
    assert resolved is not None
    assert resolved.manifest == divergent.manifest


def test_stale_quarantine_clear_never_removes_a_concurrent_fresh_marker(
    tmp_path: Path,
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    store = _CompiledGraphStore(cas)
    key = str(from_axes({"graph": "graph", "sm": "sm_89", "toolchain": "toolchain"}))
    manifest = cas.put_bytes(b"manifest identity")
    store.quarantine(key, manifest)
    name = _quarantine_ref(key, manifest)
    stale = cas.read_ref(name)
    assert stale is not None
    fresh = cas.put_bytes(b"fresh quarantine marker")
    cas.compare_and_swap_ref(name, fresh, expected=stale)

    assert not store._clear_quarantine(key, manifest, stale)
    assert cas.read_ref(name) == fresh


def test_named_literal_bytes_survive_hashrepo_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safetensors.torch import load_file

    spec = _spec_for(torch.export.export(WithLiteral(), (torch.ones(2),)))
    monkeypatch.setattr(engine_module, "_compile_package", _fake_compile_package(literal_packager))
    result = Engine(LocalCAS(tmp_path / "cas")).ensure(
        _runtime().key(spec.declare()),
        tmp_path / "graph",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    assert result.compiled_graph.literals is not None
    assert load_file(result.compiled_graph.literals)["table"].tolist() == [2.0, 3.0]


@pytest.mark.real_aoti
def test_real_aoti_package_survives_restart_reuse(tmp_path: Path) -> None:
    """Exercise torch.export -> AOTInductor -> HashRepo without test doubles."""

    spec = _spec()
    runtime = _runtime()
    key = runtime.key(spec.declare())
    cas_root = tmp_path / "cas"
    minted = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "minted",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    assert minted.outcome == EnsureOutcome.MINTED

    reused = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "reused",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: (_ for _ in ()).throw(AssertionError("restart reuse rebuilt recipe")),
    )
    assert reused.outcome == EnsureOutcome.REUSED
    assert reused.compiled_graph.package.read_bytes() == minted.compiled_graph.package.read_bytes()
    assert torch._inductor.aoti_load_package(str(reused.compiled_graph.package)) is not None

    if platform.machine() == "x86_64" and shutil.which("objdump") is not None:
        saw_vex_vector = False
        with zipfile.ZipFile(reused.compiled_graph.package) as archive:
            objects = [name for name in archive.namelist() if name.endswith(".so")]
            assert objects
            for name in objects:
                extracted = Path(archive.extract(name, tmp_path / "objects"))
                disassembly = subprocess.run(
                    ["objdump", "-d", str(extracted)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                prefixes = _instruction_prefixes(disassembly)
                assert "62" not in prefixes, "generated host code contains an EVEX instruction"
                saw_vex_vector = saw_vex_vector or any(
                    prefix in {"c4", "c5"} for prefix in prefixes
                )
        assert saw_vex_vector, "vectorizing graph emitted no VEX instruction evidence"


@pytest.mark.real_aoti
def test_real_aoti_refuses_a_literal_whose_fqn_the_compiler_erases(tmp_path: Path) -> None:
    """Never guess which exported value an anonymous package constant means."""

    spec = _spec_for(torch.export.export(WithLiteral(), (torch.ones(2),)))
    with pytest.raises(AdmissionError, match="_tensor_constant0"):
        Engine(LocalCAS(tmp_path / "cas")).ensure(
            _runtime().key(spec.declare()),
            tmp_path / "refused",
            target="cpu",
            toolchain=_toolchain(),
            recipe=lambda: spec,
        )


def test_lazy_recipe_must_restate_the_requested_key(tmp_path: Path) -> None:
    expected = _spec()
    changed = _spec_for(torch.export.export(WithLiteral(), (torch.ones(2),)))
    key = _runtime().key(expected.declare())
    with pytest.raises(AdmissionError, match="lazy recipe derives"):
        Engine(LocalCAS(tmp_path / "cas")).ensure(
            key,
            tmp_path / "graph",
            target="cpu",
            toolchain=_toolchain(),
            recipe=lambda: changed,
        )


def test_program_mutation_during_compile_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    key = _runtime().key(spec.declare())

    def mutate() -> None:
        program = cast(Any, spec.program)
        placeholder = next(
            node for node in program.graph_module.graph.nodes if node.op == "placeholder"
        )
        placeholder.meta["val"] = torch.ones(3)

    monkeypatch.setattr(engine_module, "_compile_package", _fake_compile_package(mutate=mutate))

    with pytest.raises(AdmissionError, match="changed during compilation"):
        Engine(LocalCAS(tmp_path / "cas")).ensure(
            key,
            tmp_path / "graph",
            target="cpu",
            toolchain=_toolchain(),
            recipe=lambda: spec,
        )


def test_imported_artifact_restarts_and_wrong_key_is_refused(tmp_path: Path) -> None:
    source_cas = LocalCAS(tmp_path / "source-cas")
    source_engine = Engine(source_cas)
    spec = _spec()
    key = _runtime().key(spec.declare())
    minted = source_engine.ensure(
        key,
        tmp_path / "minted",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    fetched = source_engine.export_artifact(key, tmp_path / "fetched.tar.gz")
    assert source_engine.export_artifact(key, fetched) == fetched
    occupied = tmp_path / "occupied.tar.gz"
    occupied.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="not the selected"):
        source_engine.export_artifact(key, occupied)
    assert occupied.read_bytes() == b"do not overwrite"

    divergent = _alternate_artifact(
        minted.compiled_graph.package,
        fetched,
        tmp_path / "divergent.tar.gz",
    )
    divergent_bytes = divergent.read_bytes()
    with pytest.raises(FileExistsError, match="not the selected"):
        source_engine.export_artifact(key, divergent)
    assert divergent.read_bytes() == divergent_bytes

    destination_cas = LocalCAS(tmp_path / "destination-cas")
    imported = Engine(destination_cas).import_artifact(key, fetched)
    resolved = Engine(LocalCAS(tmp_path / "destination-cas")).resolve(key, tmp_path / "resolved")
    assert str(imported.key) == str(key)
    assert resolved is not None
    assert resolved.package.is_file()

    wrong = from_axes({"graph": "wrong", "sm": "cpu", "toolchain": "wrong"})
    with pytest.raises(StorageError, match="expected"):
        Engine(destination_cas).import_artifact(wrong, fetched)


def test_export_refuses_a_corrupt_cas_object(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    engine = Engine(cas)
    spec = _spec()
    key = _runtime().key(spec.declare())
    result = engine.ensure(
        key,
        tmp_path / "minted",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    manifest = cas.load_manifest(result.compiled_graph.manifest)
    cas.object_path(manifest.files[0].digest).write_bytes(b"corrupt")
    with pytest.raises(StorageError, match="export verification"):
        engine.export_artifact(key, tmp_path / "corrupt.tar.gz")
    assert not (tmp_path / "corrupt.tar.gz").exists()


def test_export_race_refuses_a_different_same_key_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    engine = Engine(cas)
    spec = _spec()
    key = _runtime().key(spec.declare())
    minted = engine.ensure(
        key,
        tmp_path / "minted",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    selected = engine.export_artifact(key, tmp_path / "selected.tar.gz")
    divergent = _alternate_artifact(
        minted.compiled_graph.package,
        selected,
        tmp_path / "divergent.tar.gz",
    )
    destination = tmp_path / "racing.tar.gz"
    divergent_bytes = divergent.read_bytes()

    def lose_race(source: object, target: object) -> None:
        destination.write_bytes(divergent_bytes)
        raise FileExistsError(errno.EEXIST, "simulated winner", target)

    monkeypatch.setattr(os, "link", lose_race)
    with pytest.raises(FileExistsError, match="not the selected"):
        engine.export_artifact(key, destination)
    assert destination.read_bytes() == divergent_bytes


def test_host_isa_admission_refuses_cross_machine_and_missing_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    engine = Engine(cas)
    spec = _spec()
    runtime = _runtime()
    key = runtime.key(spec.declare())
    engine.ensure(
        key,
        tmp_path / "minted",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )

    actual_machine = platform.machine().lower()
    other_machine = "aarch64" if actual_machine != "aarch64" else "x86_64"
    monkeypatch.setattr(platform, "machine", lambda: other_machine)
    with pytest.raises(AdmissionError, match="artifact host code"):
        engine.resolve(key, tmp_path / "cross-machine")

    monkeypatch.setattr(platform, "machine", lambda: actual_machine)
    required = runtime.host_isa["host_isa_features"]
    if required != "none":
        monkeypatch.setattr(host_isa_module, "_cpu_features", frozenset)
        with pytest.raises(AdmissionError, match="this host lacks"):
            engine.resolve(key, tmp_path / "missing-features")


def test_two_process_resolves_converge_on_one_destination(tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    spec = _spec()
    key = _runtime().key(spec.declare())
    Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "seed",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    destination = tmp_path / "shared"
    command = [
        sys.executable,
        "-m",
        "torch_compiled_graphs",
        "resolve",
        "--cas-root",
        str(cas_root),
        str(key),
        str(destination),
    ]
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    results = [process.communicate(timeout=30) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], results
    assert (destination / "model.pt2").is_file()


def test_destination_io_error_does_not_quarantine_valid_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    spec = _spec()
    key = _runtime().key(spec.declare())
    Engine(cas).ensure(
        key,
        tmp_path / "seed",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    original = LocalCAS.materialize

    def no_space(*args: object, **kwargs: object) -> object:
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(LocalCAS, "materialize", no_space)
    with pytest.raises(OSError) as raised:
        Engine(cas).resolve(key, tmp_path / "failed")
    assert raised.value.errno == errno.ENOSPC
    monkeypatch.setattr(LocalCAS, "materialize", original)
    assert Engine(cas).resolve(key, tmp_path / "healthy") is not None
