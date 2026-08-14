from __future__ import annotations

import hashlib
import io
import json
import struct
import tarfile
import zipfile
from pathlib import Path
from typing import cast

import pytest

import torch_compiled_graphs.artifact as artifact_module
from torch_compiled_graphs import COMPILED_GRAPH_FORMAT, ArtifactError, GraphClassDeclaration
from torch_compiled_graphs.artifact import (
    build_metadata,
    pack_artifact,
    read_metadata,
    unpack_artifact,
    validate_metadata,
)
from torch_compiled_graphs.cli import main


def literal_digest(
    data: bytes,
    *,
    name: str = "table",
    dtype: str = "torch.float32",
    shape: tuple[int, ...] = (2, 2),
) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(dtype.encode())
    digest.update(str(tuple(shape)).encode())
    digest.update(data)
    return digest.hexdigest()[:32]


def safetensors_file(
    path: Path,
    tensors: list[tuple[str, str, tuple[int, ...], bytes]],
) -> Path:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def metadata(*, literal: bytes | None = None) -> dict[str, object]:
    constants = (
        [{"fqn": "table", "source": "literal", "dtype": "float32", "shape": [2, 2]}]
        if literal is not None
        else []
    )
    digest = literal_digest(literal) if literal is not None else ""
    graph: dict[str, object] = {
        "v": 3,
        "constant_fqns": ["table"] if literal is not None else [],
        "lifted_inputs": [],
        "pytree": {"in": "leaf", "out": "leaf"},
        "specialization": {},
    }
    if digest:
        graph["literal_values"] = digest
    declaration = GraphClassDeclaration(
        graph_class="denoiser/h=64,w=64",
        target="unet",
        graph=graph,
        graph_witness="fedcba9876543210",
        range_digest="0123456789abcdef" * 2,
        literal_values=digest,
    )
    return build_metadata(
        graph_class={
            "name": declaration.graph_class,
            "target": declaration.target,
            "class_hash": declaration.class_hash,
            "graph": dict(declaration.graph),
            "graph_witness": declaration.graph_witness,
            "range_digest": declaration.range_digest,
            "fork": [],
            "class_dims": [],
            "strict": True,
            "lora_bucket": 0,
            "literal_values": declaration.literal_values,
            "placement": list(declaration.placement),
            "constants": constants,
        },
        sm="sm_89",
        toolchain={
            "torch": "record-digest",
            "triton": "compiler-digest",
        },
        host_isa={
            "machine": "x86_64",
            "host_isa_level": "x86-64-v3",
            "host_isa_features": (
                "abm,avx,avx2,bmi1,bmi2,cx16,f16c,fma,lahf_lm,movbe,popcnt,"
                "sse4_1,sse4_2,ssse3,xsave"
            ),
            "cpp_march": "x86-64-v3",
            "cpp_simdlen": "256",
        },
    )


def aoti_package(
    path: Path, *, baked: bool = False, lrodata: int = 0, literal: bool = False
) -> Path:
    names = b"\0.shstrtab\0.lrodata\0"
    section_offset = 64
    section_size = 64
    section_count = 3
    string_offset = section_offset + section_size * section_count
    payload_offset = string_offset + len(names)
    shared_object = bytearray(payload_offset + lrodata)
    shared_object[:4] = b"\x7fELF"
    shared_object[4:7] = bytes((2, 1, 1))
    struct.pack_into("<Q", shared_object, 0x28, section_offset)
    struct.pack_into("<HHH", shared_object, 0x3A, section_size, section_count, 1)
    struct.pack_into("<II", shared_object, section_offset + section_size, 1, 3)
    struct.pack_into(
        "<QQ", shared_object, section_offset + section_size + 0x18, string_offset, len(names)
    )
    struct.pack_into("<II", shared_object, section_offset + 2 * section_size, 11, 1)
    struct.pack_into(
        "<QQ", shared_object, section_offset + 2 * section_size + 0x18, payload_offset, lrodata
    )
    shared_object[string_offset:payload_offset] = names
    flag = str(baked).lower()
    count = 1 if literal else 0
    wrapper = f"AOTInductorModelBase(1, 1, {count}, device_str, std::move(cubin_dir), {flag})"
    if literal:
        wrapper += "\n" + "\n".join(
            (
                'constants_info_[0].name = "table";',
                'constants_info_[0].original_fqn = "table";',
                "constants_info_[0].data_size = 16;",
                "constants_info_[0].from_folded = false;",
                "constants_info_[0].type = static_cast<int32_t>(",
                "    torch::aot_inductor::ConstantType::TensorConstant);",
                "constants_info_[0].dtype = cached_torch_dtype_float32;",
                "constants_info_[0].shape = {2, 2};",
            )
        )
    with zipfile.ZipFile(path, "w") as archive:
        root = "data/aotinductor/denoiser/h=64,w=64"
        archive.writestr(f"{root}/model.wrapper.cpp", wrapper)
        archive.writestr(f"{root}/model.so", shared_object)
    return path


def test_artifact_is_deterministic_and_unpacks_atomically(tmp_path: Path) -> None:
    package = tmp_path / "source.pt2"
    aoti_package(package)
    first = pack_artifact(package, tmp_path / "first.tar.gz", metadata())
    package.touch()
    second = pack_artifact(package, tmp_path / "second.tar.gz", metadata())
    assert first.read_bytes() == second.read_bytes()

    destination = tmp_path / "materialized"
    unpacked = unpack_artifact(first, destination)
    assert unpacked == metadata()
    assert (destination / "model.pt2").read_bytes() == package.read_bytes()
    assert read_metadata(first)["compiled_graph_format"] == COMPILED_GRAPH_FORMAT == 1

    assert main(["verify", str(first)]) == 0


def test_literal_declaration_requires_payload(tmp_path: Path) -> None:
    package = tmp_path / "source.pt2"
    aoti_package(package)
    with pytest.raises(ArtifactError, match="literal constants"):
        pack_artifact(package, tmp_path / "artifact.tar.gz", metadata(literal=b"\0" * 16))


@pytest.mark.parametrize(
    ("case", "pattern"),
    [
        ("garbage", "header"),
        ("missing_name", "names do not match"),
        ("extra_name", "names do not match"),
        ("dtype", "dtype does not match"),
        ("shape", "shape does not match"),
        ("value", "literal_values"),
    ],
)
def test_literal_payload_is_exactly_verified(case: str, pattern: str, tmp_path: Path) -> None:
    expected = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    package = aoti_package(tmp_path / "source.pt2", literal=True)
    literals = tmp_path / "constants.safetensors"
    if case == "garbage":
        literals.write_bytes(b"garbage")
    elif case == "missing_name":
        safetensors_file(literals, [("other", "F32", (2, 2), expected)])
    elif case == "extra_name":
        safetensors_file(
            literals,
            [("table", "F32", (2, 2), expected), ("extra", "F32", (), b"\0" * 4)],
        )
    elif case == "dtype":
        safetensors_file(literals, [("table", "I32", (2, 2), expected)])
    elif case == "shape":
        safetensors_file(literals, [("table", "F32", (4,), expected)])
    else:
        safetensors_file(literals, [("table", "F32", (2, 2), b"\0" * 16)])
    with pytest.raises(ArtifactError, match=pattern):
        pack_artifact(
            package,
            tmp_path / "artifact.tar.gz",
            metadata(literal=expected),
            literals=literals,
        )


def test_payload_is_forbidden_without_declared_literals(tmp_path: Path) -> None:
    package = aoti_package(tmp_path / "source.pt2")
    literals = safetensors_file(tmp_path / "constants.safetensors", [])
    with pytest.raises(ArtifactError, match="declares no literals"):
        pack_artifact(package, tmp_path / "artifact.tar.gz", metadata(), literals=literals)


def test_unpack_rejects_corrupted_literal_bytes(tmp_path: Path) -> None:
    expected = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    package = aoti_package(tmp_path / "source.pt2", literal=True)
    literals = safetensors_file(
        tmp_path / "constants.safetensors", [("table", "F32", (2, 2), expected)]
    )
    artifact = pack_artifact(
        package,
        tmp_path / "artifact.tar.gz",
        metadata(literal=expected),
        literals=literals,
    )
    with tarfile.open(artifact, "r:gz") as source:
        members = {
            row.name: source.extractfile(row).read()  # type: ignore[union-attr]
            for row in source.getmembers()
        }
    members["constants.safetensors"] = members["constants.safetensors"][:-1] + b"\0"
    with tarfile.open(artifact, "w:gz") as output:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            output.addfile(info, io.BytesIO(data))
    destination = tmp_path / "materialized"
    with pytest.raises(ArtifactError, match="literal_values"):
        unpack_artifact(artifact, destination)
    assert not destination.exists()


def test_pack_refuses_a_package_that_bakes_constants(tmp_path: Path) -> None:
    package = aoti_package(tmp_path / "source.pt2", baked=True)
    with pytest.raises(ArtifactError, match="not code-only"):
        pack_artifact(package, tmp_path / "artifact.tar.gz", metadata())


def test_stamped_key_must_restate_recorded_facts() -> None:
    raw = metadata()
    raw["sm"] = "sm_90"
    with pytest.raises(ArtifactError, match="does not restate"):
        validate_metadata(raw)


def test_one_format_symbol_controls_stamp_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_module, "COMPILED_GRAPH_FORMAT", 7)
    stamped = metadata()
    assert stamped["compiled_graph_format"] == 7
    retired = dict(stamped)
    retired["compiled_graph_format"] = 1
    with pytest.raises(ArtifactError, match="compiled_graph_format must be 7"):
        validate_metadata(retired)


def test_unstamped_host_isa_is_refused() -> None:
    raw = metadata()
    del cast(dict[str, object], raw["host_isa"])["host_isa_level"]
    with pytest.raises(ArtifactError, match="missing host ISA facts"):
        validate_metadata(raw)


def test_above_v3_host_isa_is_refused() -> None:
    raw = metadata()
    host_isa = cast(dict[str, object], raw["host_isa"])
    host_isa["host_isa_level"] = "x86-64-v4"
    host_isa["cpp_march"] = "x86-64-v4"
    with pytest.raises(ArtifactError, match="exceeds the v3 cap"):
        validate_metadata(raw)


def test_class_hash_must_restate_declaration_facts() -> None:
    raw = metadata()
    raw["graph_class"]["graph_witness"] = "0" * 16  # type: ignore[index]
    with pytest.raises(ArtifactError, match="class_hash does not restate"):
        validate_metadata(raw)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"fqn": "weight", "source": "state_dict", "dtype": "float32"},
        {"fqn": "weight", "source": "unknown", "dtype": "float32", "shape": [1]},
        {"fqn": "weight", "source": "state_dict", "dtype": "float32", "shape": [True]},
    ],
)
def test_constant_manifest_rows_fail_closed(row: dict[str, object]) -> None:
    raw = metadata()
    raw["graph_class"]["constants"] = [row]  # type: ignore[index]
    with pytest.raises(ArtifactError, match="graph_class constant"):
        validate_metadata(raw)


@pytest.mark.parametrize("retired", ["format", "entry", "entries"])
def test_v1_refuses_retired_artifact_shapes(retired: str) -> None:
    raw = metadata()
    raw[retired] = 3 if retired == "format" else {}
    with pytest.raises(ArtifactError, match="metadata fields must be exactly"):
        validate_metadata(raw)


def test_unexpected_archive_member_is_refused_without_destination(tmp_path: Path) -> None:
    artifact = tmp_path / "bad.tar"
    with tarfile.open(artifact, "w") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "output"
    with pytest.raises(ArtifactError, match="unexpected"):
        unpack_artifact(artifact, destination)
    assert not destination.exists()
