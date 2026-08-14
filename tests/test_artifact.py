from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from compiled_graphs import (
    ArtifactError,
    build_metadata,
    pack_artifact,
    read_metadata,
    unpack_artifact,
    validate_metadata,
)


def metadata(*, literal: bool = False) -> dict[str, object]:
    constants = [{"fqn": "table", "source": "literal"}] if literal else []
    return build_metadata(
        entry={
            "name": "denoiser/h=64,w=64",
            "target": "unet",
            "class_hash": "0123456789abcdef",
            "constants": constants,
        },
        sm="sm_89",
        toolchain={"torch": "record-digest", "diffusers": "trace-time-only"},
        extra={"family": "sdxl"},
    )


def test_artifact_is_deterministic_and_unpacks_atomically(tmp_path: Path) -> None:
    package = tmp_path / "source.pt2"
    package.write_bytes(b"compiled package")
    first = pack_artifact(package, tmp_path / "first.tar.gz", metadata())
    package.touch()
    second = pack_artifact(package, tmp_path / "second.tar.gz", metadata())
    assert first.read_bytes() == second.read_bytes()

    destination = tmp_path / "materialized"
    unpacked = unpack_artifact(first, destination)
    assert unpacked == metadata()
    assert (destination / "model.pt2").read_bytes() == b"compiled package"
    assert read_metadata(first)["compiled_graph_format"] == 1


def test_literal_declaration_requires_payload(tmp_path: Path) -> None:
    package = tmp_path / "source.pt2"
    package.write_bytes(b"compiled package")
    with pytest.raises(ArtifactError, match="literal constants"):
        pack_artifact(package, tmp_path / "artifact.tar.gz", metadata(literal=True))


def test_stamped_key_must_restate_recorded_facts() -> None:
    raw = metadata()
    raw["sm"] = "sm_90"
    with pytest.raises(ArtifactError, match="does not restate"):
        validate_metadata(raw)


@pytest.mark.parametrize("retired", ["format", "entries"])
def test_v1_refuses_retired_artifact_shapes(retired: str) -> None:
    raw = metadata()
    raw[retired] = 3 if retired == "format" else {}
    with pytest.raises(ArtifactError, match="retired artifact fields"):
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
