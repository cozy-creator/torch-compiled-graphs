from __future__ import annotations

import errno
import struct
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from hashrepo import LocalCAS

from torch_compiled_graphs import (
    AdmissionError,
    Engine,
    EnsureOutcome,
    GraphSpec,
    RuntimeCompatibility,
    StorageError,
    from_axes,
)
from torch_compiled_graphs.compiler import Compiler

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


def compiler(*args: object, **kwargs: object) -> object:
    return ["wrapper.cpp", "model.so"]


def _spec() -> GraphSpec:
    program = torch.export.export(Double(), (torch.ones(2),))
    return GraphSpec("model", "denoiser", program)


def _runtime() -> RuntimeCompatibility:
    return RuntimeCompatibility("cpu", deployment_compatibility="test-image-v1")


def test_fresh_engine_reuses_local_hashrepo_without_compiling(tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    spec = _spec()
    key = _runtime().key(spec.declare())
    first = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "first",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
        compiler=cast(Compiler, compiler),
        packager=packager,
    )
    assert first.outcome == EnsureOutcome.MINTED
    assert first.graph.package.is_file()

    def must_not_construct() -> GraphSpec:
        raise AssertionError("restart reuse invoked the lazy recipe")

    second = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "second",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=must_not_construct,
    )
    assert second.outcome == EnsureOutcome.REUSED
    assert second.graph.package.read_bytes() == first.graph.package.read_bytes()

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
    deployment_compatibility="test-image-v1", recipe=forbidden_recipe,
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
            first.graph.key,
            str(process_destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (process_destination / "model.pt2").is_file()


def test_corrupt_local_object_is_quarantined_and_repaired(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "cas")
    spec = _spec()
    key = _runtime().key(spec.declare())
    first = Engine(cas).ensure(
        key,
        tmp_path / "first",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
        compiler=cast(Compiler, compiler),
        packager=packager,
    )
    manifest = cas.load_manifest(first.graph.manifest)
    cas.object_path(manifest.files[0].digest).write_bytes(b"corrupt")

    repaired = Engine(LocalCAS(tmp_path / "cas")).ensure(
        key,
        tmp_path / "repaired",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
        compiler=cast(Compiler, compiler),
        packager=packager,
    )
    assert repaired.outcome == EnsureOutcome.MINTED
    assert repaired.graph.package.is_file()


def test_named_literal_bytes_survive_hashrepo_reuse(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    spec = GraphSpec(
        "model",
        "denoiser",
        torch.export.export(WithLiteral(), (torch.ones(2),)),
    )
    result = Engine(LocalCAS(tmp_path / "cas")).ensure(
        _runtime().key(spec.declare()),
        tmp_path / "graph",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
        compiler=cast(Compiler, compiler),
        packager=literal_packager,
    )
    assert result.graph.literals is not None
    assert load_file(result.graph.literals)["table"].tolist() == [2.0, 3.0]


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
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
    )
    assert minted.outcome == EnsureOutcome.MINTED

    reused = Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "reused",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=lambda: (_ for _ in ()).throw(AssertionError("restart reuse rebuilt recipe")),
    )
    assert reused.outcome == EnsureOutcome.REUSED
    assert reused.graph.package.read_bytes() == minted.graph.package.read_bytes()


def test_real_aoti_refuses_a_literal_whose_fqn_the_compiler_erases(tmp_path: Path) -> None:
    """Never guess which exported value an anonymous package constant means."""

    spec = GraphSpec(
        "model",
        "denoiser",
        torch.export.export(WithLiteral(), (torch.ones(2),)),
    )
    with pytest.raises(AdmissionError, match="_tensor_constant0"):
        Engine(LocalCAS(tmp_path / "cas")).ensure(
            _runtime().key(spec.declare()),
            tmp_path / "refused",
            target="cpu",
            deployment_compatibility="test-image-v1",
            recipe=lambda: spec,
        )


def test_lazy_recipe_must_restate_the_requested_key(tmp_path: Path) -> None:
    expected = _spec()
    changed = GraphSpec(
        "model",
        "denoiser",
        torch.export.export(WithLiteral(), (torch.ones(2),)),
    )
    key = _runtime().key(expected.declare())
    with pytest.raises(AdmissionError, match="lazy recipe derives"):
        Engine(LocalCAS(tmp_path / "cas")).ensure(
            key,
            tmp_path / "graph",
            target="cpu",
            deployment_compatibility="test-image-v1",
            recipe=lambda: changed,
            compiler=cast(Compiler, compiler),
            packager=packager,
        )


def test_program_mutation_during_compile_is_refused(tmp_path: Path) -> None:
    spec = _spec()
    key = _runtime().key(spec.declare())

    def mutating_compiler(*args: object, **kwargs: object) -> object:
        program = cast(Any, spec.program)
        placeholder = next(
            node for node in program.graph_module.graph.nodes if node.op == "placeholder"
        )
        placeholder.meta["val"] = torch.ones(3)
        return compiler(*args, **kwargs)

    with pytest.raises(AdmissionError, match="changed during compilation"):
        Engine(LocalCAS(tmp_path / "cas")).ensure(
            key,
            tmp_path / "graph",
            target="cpu",
            deployment_compatibility="test-image-v1",
            recipe=lambda: spec,
            compiler=cast(Compiler, mutating_compiler),
            packager=packager,
        )


def test_imported_artifact_restarts_and_wrong_key_is_refused(tmp_path: Path) -> None:
    source_cas = LocalCAS(tmp_path / "source-cas")
    spec = _spec()
    key = _runtime().key(spec.declare())
    minted = Engine(source_cas).ensure(
        key,
        tmp_path / "minted",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
        compiler=cast(Compiler, compiler),
        packager=packager,
    )
    manifest = source_cas.load_manifest(minted.graph.manifest)
    fetched = source_cas.materialize(manifest.files[0], tmp_path / "fetched.tar.gz")

    destination_cas = LocalCAS(tmp_path / "destination-cas")
    imported = Engine(destination_cas).import_artifact(key, fetched)
    resolved = Engine(LocalCAS(tmp_path / "destination-cas")).resolve(key, tmp_path / "resolved")
    assert str(imported.key) == str(key)
    assert resolved is not None
    assert resolved.package.is_file()

    wrong = from_axes({"graph": "wrong", "sm": "cpu", "toolchain": "wrong"})
    with pytest.raises(StorageError, match="expected"):
        Engine(destination_cas).import_artifact(wrong, fetched)


def test_two_process_resolves_converge_on_one_destination(tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    spec = _spec()
    key = _runtime().key(spec.declare())
    Engine(LocalCAS(cas_root)).ensure(
        key,
        tmp_path / "seed",
        target="cpu",
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
        compiler=cast(Compiler, compiler),
        packager=packager,
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
        deployment_compatibility="test-image-v1",
        recipe=lambda: spec,
        compiler=cast(Compiler, compiler),
        packager=packager,
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
