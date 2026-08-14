from __future__ import annotations

import io
import struct
import tarfile
import zipfile
from pathlib import Path

import pytest

from torch_compiled_graphs import (
    ArtifactError,
    GraphDeclaration,
    build_metadata,
    pack_artifact,
    read_metadata,
    unpack_artifact,
    validate_metadata,
)
from torch_compiled_graphs.cli import main


def metadata(*, literal: bool = False) -> dict[str, object]:
    constants = (
        [{"fqn": "table", "source": "literal", "dtype": "float32", "shape": [2, 2]}]
        if literal
        else []
    )
    declaration = GraphDeclaration(
        "denoiser/h=64,w=64",
        "unet",
        "fedcba9876543210",
        literal_values="a" * 32 if literal else "",
    )
    return build_metadata(
        entry={
            "name": declaration.entry,
            "target": declaration.target,
            "class_hash": declaration.class_hash,
            "graph": declaration.graph,
            "literal_values": declaration.literal_values,
            "placement": list(declaration.placement),
            "constants": constants,
        },
        sm="sm_89",
        toolchain={"torch": "record-digest", "triton": "compiler-digest"},
    )


def aoti_package(path: Path, *, baked: bool = False, lrodata: int = 0) -> Path:
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
    wrapper = f"AOTInductorModelBase(1, 1, 0, device_str, std::move(cubin_dir), {flag})"
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
    assert read_metadata(first)["compiled_graph_format"] == 1

    assert main(["verify", str(first)]) == 0


def test_literal_declaration_requires_payload(tmp_path: Path) -> None:
    package = tmp_path / "source.pt2"
    aoti_package(package)
    with pytest.raises(ArtifactError, match="literal constants"):
        pack_artifact(package, tmp_path / "artifact.tar.gz", metadata(literal=True))


def test_pack_refuses_a_package_that_bakes_constants(tmp_path: Path) -> None:
    package = aoti_package(tmp_path / "source.pt2", baked=True)
    with pytest.raises(ArtifactError, match="not code-only"):
        pack_artifact(package, tmp_path / "artifact.tar.gz", metadata())


def test_stamped_key_must_restate_recorded_facts() -> None:
    raw = metadata()
    raw["sm"] = "sm_90"
    with pytest.raises(ArtifactError, match="does not restate"):
        validate_metadata(raw)


def test_class_hash_must_restate_declaration_facts() -> None:
    raw = metadata()
    raw["entry"]["graph"] = "0" * 16  # type: ignore[index]
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
    raw["entry"]["constants"] = [row]  # type: ignore[index]
    with pytest.raises(ArtifactError, match="entry constant"):
        validate_metadata(raw)


@pytest.mark.parametrize("retired", ["format", "entries"])
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
