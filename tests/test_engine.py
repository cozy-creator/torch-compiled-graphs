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
from tensorfs import CASRef, LocalCAS

import torchcg._wrapper_split as wrapper_split_module
import torchcg.engine as engine_module
import torchcg.host_isa as host_isa_module
from torchcg import (
    AdmissionError,
    ConstantBindingError,
    Engine,
    EnsureOutcome,
    GraphSpecialization,
    QuarantinedArtifact,
    RuntimeCompatibility,
    StorageError,
    StoreOutcome,
    build_call_ingress,
)
from torchcg.artifact import pack_artifact, read_metadata, unpack_artifact
from torchcg.identity import from_axes
from torchcg.storage import _CompiledGraphStore, _quarantine_ref

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


class WithBuffer(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0, 3.0]))

    def forward(self, value: Any) -> Any:
        return value * self.scale


class WithTwoLiterals(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.tensor([2.0])
        self.second = torch.tensor([3.0])

    def forward(self, value: Any) -> Any:
        return value * self.first + self.second


class NestedRuntime(torch.nn.Module):  # type: ignore[misc]
    def forward(
        self,
        sample: Any,
        conditioning: dict[str, Any],
        shape: list[Any],
        return_dict: bool,
        tail: Any,
    ) -> Any:
        return (
            sample
            + conditioning["zeta"]
            + conditioning["alpha"]
            + tail
            + float(shape[0][0] + shape[0][1])
            + int(return_dict)
        )


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


def computed_packager(output: str, files: Mapping[str, Sequence[object]]) -> object:
    assert len(files) == 1
    entry = next(iter(files))
    wrapper = "\n".join(
        (
            "AOTInductorModelBase(1, 1, 1, device_str, std::move(cubin_dir), false)",
            'constants_info_[0].name = "folded";',
            'constants_info_[0].original_fqn = "folded";',
            "constants_info_[0].data_size = 8;",
            "constants_info_[0].from_folded = true;",
            "constants_info_[0].type = static_cast<int32_t>(",
            "    torch::aot_inductor::ConstantType::FoldedConstant);",
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


def first_literal_packager(output: str, files: Mapping[str, Sequence[object]]) -> object:
    assert len(files) == 1
    entry = next(iter(files))
    wrapper = "\n".join(
        (
            "AOTInductorModelBase(1, 1, 1, device_str, std::move(cubin_dir), false)",
            'constants_info_[0].name = "first";',
            'constants_info_[0].original_fqn = "first";',
            "constants_info_[0].data_size = 4;",
            "constants_info_[0].from_folded = false;",
            "constants_info_[0].type = static_cast<int32_t>(",
            "    torch::aot_inductor::ConstantType::TensorConstant);",
            "constants_info_[0].dtype = cached_torch_dtype_float32;",
            "constants_info_[0].shape = {1};",
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
            {plan.declaration.name: ["wrapper.cpp", "model.so"]},
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


def _spec_for(program: object) -> GraphSpecialization:
    example_args, example_kwargs = program.example_inputs  # type: ignore[attr-defined]
    ingress = build_call_ingress(program, ("value",), example_args, example_kwargs)
    return GraphSpecialization("model", "denoiser", program, ingress)


def _spec() -> GraphSpecialization:
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


def test_fresh_engine_reuses_the_local_store_without_compiling(tmp_path: Path) -> None:
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

    def must_not_construct() -> GraphSpecialization:
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
from tensorfs import LocalCAS
from torchcg import Engine, EnsureOutcome
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
            "torchcg",
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

    def forbidden_recipe() -> GraphSpecialization:
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
    cas.object_path(first.compiled_graph.artifact).write_bytes(b"corrupt")

    repaired = Engine(LocalCAS(tmp_path / "cas")).ensure(
        key,
        tmp_path / "repaired",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    assert repaired.outcome == EnsureOutcome.MINTED
    assert repaired.compiled_graph.package.is_file()


def test_repairing_with_a_previously_divergent_artifact_retires_its_stale_quarantine(
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

    cas.object_path(minted.compiled_graph.artifact).write_bytes(b"corrupt-a")
    with pytest.raises(QuarantinedArtifact, match="CAS verification"):
        engine.resolve(key, tmp_path / "corrupt-a")

    repaired = engine.import_artifact(key, artifact_b)
    assert repaired.outcome == StoreOutcome.REPAIRED
    assert repaired.artifact == divergent.artifact
    resolved = engine.resolve(key, tmp_path / "resolved-b")
    assert resolved is not None
    assert resolved.artifact == divergent.artifact


def test_each_quarantine_event_replaces_the_previous_marker_generation(
    tmp_path: Path,
) -> None:
    cas = LocalCAS(tmp_path / "cas")
    store = _CompiledGraphStore(cas)
    key = str(
        from_axes(
            {
                "compile_policy": "policy",
                "graph": "graph",
                "sm": "sm_89",
                "toolchain": "toolchain",
            }
        )
    )
    artifact = cas.put_bytes(b"artifact identity")
    store.quarantine(key, artifact)
    name = _quarantine_ref(key, artifact)
    stale = cas.read_ref(name)
    assert stale is not None
    store.quarantine(key, artifact)
    fresh = cas.read_ref(name)

    assert fresh is not None
    assert fresh != stale
    assert not store._clear_quarantine(key, artifact, stale)
    assert cas.read_ref(name) == fresh


def test_repair_cannot_clear_a_concurrent_production_quarantine(
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
    artifact_a = engine.export_artifact(key, tmp_path / "artifact-a.tar.gz")
    artifact_b = _alternate_artifact(
        minted.compiled_graph.package,
        artifact_a,
        tmp_path / "artifact-b.tar.gz",
    )
    divergent = engine.import_artifact(key, artifact_b)
    assert divergent.outcome == StoreOutcome.DIVERGENT

    cas.object_path(minted.compiled_graph.artifact).write_bytes(b"corrupt-a")
    with pytest.raises(QuarantinedArtifact, match="CAS verification"):
        engine.resolve(key, tmp_path / "corrupt-a")

    name = _quarantine_ref(str(key), divergent.artifact)
    stale = cas.read_ref(name)
    assert stale is not None
    original_clear = _CompiledGraphStore._clear_quarantine
    interleaved = False

    def quarantine_before_clear(
        store: _CompiledGraphStore,
        inner_key: str,
        artifact: CASRef,
        expected: CASRef,
    ) -> bool:
        nonlocal interleaved
        if not interleaved and inner_key == str(key) and artifact == divergent.artifact:
            interleaved = True
            _CompiledGraphStore(cas).quarantine(inner_key, divergent.artifact)
        return original_clear(store, inner_key, artifact, expected)

    monkeypatch.setattr(_CompiledGraphStore, "_clear_quarantine", quarantine_before_clear)
    with pytest.raises(QuarantinedArtifact, match="fresh quarantine"):
        engine.import_artifact(key, artifact_b)

    fresh = cas.read_ref(name)
    assert interleaved
    assert fresh is not None
    assert fresh != stale
    with pytest.raises(QuarantinedArtifact, match="is quarantined"):
        engine.resolve(key, tmp_path / "still-quarantined")


def test_named_literal_bytes_survive_store_reuse(
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


def test_compile_derives_its_own_key_and_reuses_without_recompiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    runtime = _runtime()
    calls = 0

    def compile_once(plan: object, workspace: Path) -> Path:
        nonlocal calls
        calls += 1
        return _fake_compile_package(packager)(plan, workspace)

    monkeypatch.setattr(engine_module, "_compile_package", compile_once)
    engine = Engine(LocalCAS(tmp_path / "cas"))
    minted = engine.compile(spec, runtime, tmp_path / "minted")
    reused = engine.compile(spec, runtime, tmp_path / "reused")
    assert minted.outcome == EnsureOutcome.MINTED
    assert reused.outcome == EnsureOutcome.REUSED
    assert minted.compiled_graph.key == str(runtime.key(spec.declare()))
    assert reused.compiled_graph.artifact == minted.compiled_graph.artifact
    assert calls == 1


def test_compile_refuses_a_state_dict_constant_eliminated_by_the_package(
    tmp_path: Path,
) -> None:
    program = torch.export.export(WithBuffer(), (torch.ones(2),))
    spec = _spec_for(program)

    with pytest.raises(AdmissionError, match="eliminated.*state-dict.*scale"):
        Engine(LocalCAS(tmp_path / "cas")).compile(spec, _runtime(), tmp_path / "refused")

    assert not (tmp_path / "refused").exists()


def test_compile_refuses_a_package_constant_the_program_never_lifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "_compile_package", _fake_compile_package(literal_packager))

    with pytest.raises(AdmissionError, match="not in the exported program.*table"):
        Engine(LocalCAS(tmp_path / "cas")).compile(_spec(), _runtime(), tmp_path / "refused")


def test_compile_allows_an_eliminated_literal_whose_value_is_keyed(tmp_path: Path) -> None:
    spec = _spec_for(torch.export.export(WithLiteral(), (torch.ones(2),)))

    result = Engine(LocalCAS(tmp_path / "cas")).compile(
        spec, _runtime(), tmp_path / "compiled"
    )

    graph_specialization = result.compiled_graph.metadata["graph_specialization"]
    assert isinstance(graph_specialization, dict)
    assert graph_specialization["literal_values"] == spec.declare().literal_values
    assert graph_specialization["literal_payload_values"] == ""
    assert graph_specialization["constants"] == []
    assert result.compiled_graph.literals is None


def test_compile_authenticates_only_the_surviving_literal_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safetensors.torch import load_file

    spec = _spec_for(torch.export.export(WithTwoLiterals(), (torch.ones(1),)))
    monkeypatch.setattr(
        engine_module, "_compile_package", _fake_compile_package(first_literal_packager)
    )

    result = Engine(LocalCAS(tmp_path / "cas")).compile(
        spec, _runtime(), tmp_path / "compiled"
    )

    graph_specialization = result.compiled_graph.metadata["graph_specialization"]
    assert isinstance(graph_specialization, dict)
    assert graph_specialization["literal_values"] == spec.declare().literal_values
    assert graph_specialization["literal_payload_values"]
    assert graph_specialization["literal_payload_values"] != graph_specialization["literal_values"]
    assert result.compiled_graph.literals is not None
    assert set(load_file(result.compiled_graph.literals)) == {"first"}


def test_compile_allows_package_only_computed_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_module, "_compile_package", _fake_compile_package(computed_packager))

    result = Engine(LocalCAS(tmp_path / "cas")).compile(_spec(), _runtime(), tmp_path / "compiled")

    graph_specialization = result.compiled_graph.metadata["graph_specialization"]
    assert isinstance(graph_specialization, dict)
    assert graph_specialization["constants"] == [
        {"fqn": "folded", "source": "computed", "dtype": "float32", "shape": [2]}
    ]


@pytest.mark.real_aoti
def test_real_aoti_package_survives_restart_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise CPU AOTI reuse and prove the private CUDA transforms decline."""

    emitted: list[wrapper_split_module.SplitOutcome] = []

    def record(outcome: wrapper_split_module.SplitOutcome, lever: str = "") -> None:
        del lever
        emitted.append(outcome)

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor-cache"))
    monkeypatch.setattr(wrapper_split_module, "_emit", record)

    spec = _spec()
    runtime = _runtime()
    key = runtime.key(spec.declare())
    cas_root = tmp_path / "cas"
    minted = Engine(LocalCAS(cas_root)).compile(spec, runtime, tmp_path / "minted")
    assert minted.outcome == EnsureOutcome.MINTED

    restarted = Engine(LocalCAS(cas_root))
    reused = restarted.compile(spec, runtime, tmp_path / "reused")
    assert reused.outcome == EnsureOutcome.REUSED
    assert reused.compiled_graph.package.read_bytes() == minted.compiled_graph.package.read_bytes()
    loaded = restarted.runner(key, tmp_path / "loaded")
    assert loaded is not None
    with pytest.raises(ConstantBindingError, match="before complete binding"):
        loaded(torch.ones(4096))
    loaded.bind({}, device="cpu")
    representative = torch.linspace(-3.0, 3.0, 4096)
    actual = loaded(representative)
    expected = Double()(representative)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert emitted
    assert not any(outcome.applied for outcome in emitted)
    assert any("CUDA_DRIVER_CHECK" in outcome.reason for outcome in emitted)

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
def test_real_aoti_runner_binds_named_state_by_reference(tmp_path: Path) -> None:
    module = WithBuffer()
    program = torch.export.export(module, (torch.ones(2),))
    spec = _spec_for(program)
    runtime = _runtime()
    engine = Engine(LocalCAS(tmp_path / "cas"))
    result = engine.compile(spec, runtime, tmp_path / "compiled")
    runner = engine.runner(result.compiled_graph.key, tmp_path / "loaded")
    assert runner is not None
    runner.bind({"scale": module.scale}, device="cpu")
    representative = torch.tensor([4.0, 5.0])
    torch.testing.assert_close(runner(representative), module(representative))
    assert runner.bound_fqns == ("scale",)


@pytest.mark.real_aoti
def test_real_aoti_named_buffer_runs_after_interpreter_restart(tmp_path: Path) -> None:
    module = WithBuffer()
    spec = _spec_for(torch.export.export(module, (torch.ones(2),)))
    runtime = _runtime()
    cas_root = tmp_path / "cas"
    result = Engine(LocalCAS(cas_root)).compile(spec, runtime, tmp_path / "compiled")
    code = """
import sys
from pathlib import Path

import torch
from tensorfs import LocalCAS
from torchcg import Engine

runner = Engine(LocalCAS(Path(sys.argv[1]))).runner(sys.argv[2], Path(sys.argv[3]))
assert runner is not None
runner.bind({"scale": torch.tensor([7.0, 11.0])}, device="cpu")
actual = runner(torch.tensor([4.0, 5.0]))
torch.testing.assert_close(actual, torch.tensor([28.0, 55.0]))
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(cas_root),
            result.compiled_graph.key,
            str(tmp_path / "restarted"),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "subprocess-cache")},
    )


@pytest.mark.real_aoti
def test_real_nested_call_ingress_runs_after_interpreter_restart(tmp_path: Path) -> None:
    args = (
        torch.ones(2, 3),
        {"zeta": torch.full((2, 3), 2.0), "alpha": torch.full((2, 3), 3.0)},
        [[2, 3]],
        False,
        torch.full((2, 3), 4.0),
    )
    params = ("sample", "conditioning", "shape", "return_dict", "tail")
    program = torch.export.export(NestedRuntime(), args, {}, strict=True)
    ingress = build_call_ingress(program, params, args, {})
    spec = GraphSpecialization("nested", "denoiser", program, ingress)
    cas_root = tmp_path / "cas"
    result = Engine(LocalCAS(cas_root)).compile(spec, _runtime(), tmp_path / "compiled")

    code = """
import sys
from pathlib import Path

import torch
from tensorfs import LocalCAS
from torchcg import CallIngress, Engine

engine = Engine(LocalCAS(Path(sys.argv[1])))
resolved = engine.resolve(sys.argv[2], Path(sys.argv[3]) / "resolved")
assert resolved is not None
graph_specialization = resolved.metadata["graph_specialization"]
assert isinstance(graph_specialization, dict)
graph = graph_specialization["graph"]
assert isinstance(graph, dict)
ingress = CallIngress.from_graph(graph)
runner = engine.runner(sys.argv[2], Path(sys.argv[3]) / "runner")
assert runner is not None
runner.bind({}, device="cpu")
args = (
    torch.ones(2, 3),
    {"zeta": torch.full((2, 3), 2.0), "alpha": torch.full((2, 3), 3.0)},
    [[2, 3]],
    False,
    torch.full((2, 3), 4.0),
)
actual = runner(*ingress.feeds(args, {}))
torch.testing.assert_close(actual, torch.full((2, 3), 15.0))
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(cas_root),
            result.compiled_graph.key,
            str(tmp_path / "restarted-nested"),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "nested-cache")},
    )


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


def test_literal_mutation_during_payload_serialization_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec_for(torch.export.export(WithLiteral(), (torch.ones(2),)))
    monkeypatch.setattr(
        engine_module, "_compile_package", _fake_compile_package(literal_packager)
    )
    write_literals = engine_module._write_literals

    def mutate_before_write(
        program: object, constants: tuple[object, ...], target: Path
    ) -> str:
        cast(Any, program).constants["table"].add_(1)
        return write_literals(program, cast(Any, constants), target)

    monkeypatch.setattr(engine_module, "_write_literals", mutate_before_write)

    with pytest.raises(AdmissionError, match="literal serialization"):
        Engine(LocalCAS(tmp_path / "cas")).compile(spec, _runtime(), tmp_path / "refused")


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

    wrong = from_axes(
        {
            "compile_policy": "wrong",
            "graph": "wrong",
            "sm": "cpu",
            "toolchain": "wrong",
        }
    )
    with pytest.raises(StorageError, match="expected"):
        Engine(destination_cas).import_artifact(wrong, fetched)


def test_resolve_refuses_an_occupied_same_key_divergent_directory(tmp_path: Path) -> None:
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
    destination = tmp_path / "occupied"
    unpack_artifact(divergent, destination)
    divergent_package = (destination / "model.pt2").read_bytes()

    with pytest.raises(FileExistsError, match="not the selected"):
        engine.resolve(key, destination)
    assert (destination / "model.pt2").read_bytes() == divergent_package


def test_resolve_race_refuses_a_same_key_divergent_winner(
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
    destination = tmp_path / "racing"

    def lose_race(source: object, target: object) -> None:
        unpack_artifact(divergent, destination)
        raise OSError(errno.EEXIST, "simulated winner", target)

    monkeypatch.setattr(os, "rename", lose_race)
    with pytest.raises(FileExistsError, match="not the selected"):
        engine.resolve(key, destination)
    assert (destination / "model.pt2").read_bytes() != minted.compiled_graph.package.read_bytes()


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
    cas.object_path(result.compiled_graph.artifact).write_bytes(b"corrupt")
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
        "torchcg",
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
    # Acquisition no longer writes the archive anywhere: resolve untars straight
    # out of the store. The only destination write left is the unpack itself, so
    # that is where a full disk now surfaces -- at the fsync of an unpacked
    # member, which is exactly how delayed allocation reports ENOSPC.
    #
    # Only the caller's subtree is allowed to fail. Failing every fsync would
    # also break the quarantine marker's own write, and the test would then pass
    # whichever branch ran. tests/test_storage.py holds the same property with
    # the mutation evidence behind it.
    destinations = tmp_path / "out"
    destinations.mkdir()
    original = os.fsync

    def no_space(descriptor: int) -> None:
        try:
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            return original(descriptor)
        if destinations == target or destinations in target.parents:
            raise OSError(errno.ENOSPC, "no space")
        return original(descriptor)

    monkeypatch.setattr(os, "fsync", no_space)
    with pytest.raises(OSError) as raised:
        Engine(cas).resolve(key, destinations / "failed")
    assert raised.value.errno == errno.ENOSPC
    assert not isinstance(raised.value, QuarantinedArtifact)
    monkeypatch.setattr(os, "fsync", original)
    # The bytes were never in question, so nothing may have been quarantined.
    assert Engine(cas).resolve(key, tmp_path / "healthy") is not None


def test_unreadable_stored_bytes_do_quarantine(tmp_path: Path) -> None:
    """The other half of the ENOSPC property: bad stored bytes DO quarantine.

    Without this, the test above passes vacuously against a resolve that never
    quarantines anything at all.
    """

    cas = LocalCAS(tmp_path / "cas")
    spec = _spec()
    key = _runtime().key(spec.declare())
    stored = Engine(cas).ensure(
        key,
        tmp_path / "seed",
        target="cpu",
        toolchain=_toolchain(),
        recipe=lambda: spec,
    )
    cas.object_path(stored.compiled_graph.artifact).write_bytes(b"not a tarball")
    with pytest.raises(QuarantinedArtifact):
        Engine(cas).resolve(key, tmp_path / "failed")
