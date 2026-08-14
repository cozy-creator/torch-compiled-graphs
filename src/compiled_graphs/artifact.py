from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .identity import from_artifact_metadata, is_compiled_graph_key

COMPILED_GRAPH_FORMAT = 1
COMPILED_GRAPH_FORMAT_KEY = "compiled_graph_format"
ARTIFACT_KIND = "aot-inductor"
METADATA_NAME = "metadata.json"
PACKAGE_NAME = "model.pt2"
LITERALS_NAME = "constants.safetensors"
_REQUIRED_MEMBERS = frozenset((METADATA_NAME, PACKAGE_NAME))
_MEMBERS = _REQUIRED_MEMBERS | {LITERALS_NAME}
_MAX_METADATA_BYTES = 8 << 20


class ArtifactError(ValueError):
    """A compiled-graph envelope is malformed or internally inconsistent."""


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _entry_requires_literals(entry: Mapping[str, Any]) -> bool:
    constants = entry.get("constants", ())
    if not isinstance(constants, list):
        raise ArtifactError("entry constants must be an array")
    for row in constants:
        if not isinstance(row, Mapping):
            raise ArtifactError("entry constant must be an object")
        if row.get("source") == "literal":
            return True
    return False


def validate_metadata(
    raw: Mapping[str, Any], *, has_literals: bool | None = None
) -> dict[str, Any]:
    """Validate, detach, and return the sole v1 metadata representation."""

    try:
        decoded: object = json.loads(json.dumps(dict(raw), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"metadata is not finite JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ArtifactError("metadata root must be an object")
    metadata = cast(dict[str, Any], decoded)
    if metadata.get(COMPILED_GRAPH_FORMAT_KEY) != COMPILED_GRAPH_FORMAT:
        raise ArtifactError(f"{COMPILED_GRAPH_FORMAT_KEY} must be {COMPILED_GRAPH_FORMAT}")
    retired = sorted(set(metadata) & {"format", "entries"})
    if retired:
        raise ArtifactError(f"retired artifact fields are not valid in v1: {retired!r}")
    if metadata.get("kind") != ARTIFACT_KIND:
        raise ArtifactError(f"kind must be {ARTIFACT_KIND!r}")
    if metadata.get("package_constants_in_so") is not False:
        raise ArtifactError("package_constants_in_so must be false")
    if metadata.get("constant_folding_fenced") is not True:
        raise ArtifactError("constant_folding_fenced must be true")
    entry = metadata.get("entry")
    if not isinstance(entry, dict):
        raise ArtifactError("metadata must contain one entry object")
    for field in ("name", "target", "class_hash"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise ArtifactError(f"entry {field!r} must be a non-empty string")
    expected = from_artifact_metadata(metadata).value
    stamped = metadata.get("cell_key")
    if not isinstance(stamped, str) or not is_compiled_graph_key(stamped):
        raise ArtifactError("cell_key is missing or malformed")
    if stamped != expected:
        raise ArtifactError(f"cell_key {stamped!r} does not restate artifact facts ({expected})")
    needs_literals = _entry_requires_literals(entry)
    if has_literals is False and needs_literals:
        raise ArtifactError("entry declares literal constants but carries no literal payload")
    return metadata


def build_metadata(
    *,
    entry: Mapping[str, Any],
    sm: str,
    toolchain: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata from recorded facts and stamp the key derived from them."""

    reserved = {
        COMPILED_GRAPH_FORMAT_KEY,
        "kind",
        "cell_key",
        "entry",
        "sm",
        "toolchain",
        "package_constants_in_so",
        "constant_folding_fenced",
    }
    collision = sorted(set(extra or {}) & reserved)
    if collision:
        raise ArtifactError(f"extra metadata cannot override reserved fields {collision!r}")
    metadata: dict[str, Any] = {
        COMPILED_GRAPH_FORMAT_KEY: COMPILED_GRAPH_FORMAT,
        "kind": ARTIFACT_KIND,
        "entry": dict(entry),
        "sm": str(sm),
        "toolchain": dict(toolchain),
        "package_constants_in_so": False,
        "constant_folding_fenced": True,
        **dict(extra or {}),
    }
    metadata["cell_key"] = from_artifact_metadata(metadata).value
    return validate_metadata(metadata)


def _metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    checked = validate_metadata(metadata)
    return json.dumps(
        checked, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def pack_artifact(
    package: str | Path,
    output: str | Path,
    metadata: Mapping[str, Any],
    *,
    literals: str | Path | None = None,
) -> Path:
    """Create the deterministic v1 tar+gzip envelope."""

    package_path = Path(package)
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    literal_path = Path(literals) if literals is not None else None
    if literal_path is not None and not literal_path.is_file():
        raise FileNotFoundError(literal_path)
    checked = validate_metadata(metadata, has_literals=literal_path is not None)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive:
                    encoded = _metadata_bytes(checked)
                    archive.addfile(_tarinfo(METADATA_NAME, len(encoded)), io.BytesIO(encoded))
                    for name, source in (
                        (PACKAGE_NAME, package_path),
                        (LITERALS_NAME, literal_path),
                    ):
                        if source is None:
                            continue
                        with source.open("rb") as handle:
                            archive.addfile(_tarinfo(name, source.stat().st_size), handle)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_name, target)
        _fsync_dir(target.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def _members(artifact: Path) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
    try:
        archive = tarfile.open(artifact, mode="r:*")
        rows = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ArtifactError(f"cannot read artifact {artifact}: {exc}") from exc
    members: dict[str, tarfile.TarInfo] = {}
    for row in rows:
        if row.name not in _MEMBERS or not row.isfile() or row.name in members:
            archive.close()
            raise ArtifactError(f"unexpected or duplicate artifact member {row.name!r}")
        members[row.name] = row
    missing = _REQUIRED_MEMBERS - set(members)
    if missing:
        archive.close()
        raise ArtifactError(f"artifact is missing {sorted(missing)!r}")
    return archive, members


def read_metadata(artifact: str | Path) -> dict[str, Any]:
    path = Path(artifact)
    archive, members = _members(path)
    try:
        info = members[METADATA_NAME]
        if info.size > _MAX_METADATA_BYTES:
            raise ArtifactError(f"metadata exceeds {_MAX_METADATA_BYTES} bytes")
        handle = archive.extractfile(info)
        if handle is None:
            raise ArtifactError("metadata member is unreadable")
        try:
            raw = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"metadata is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ArtifactError("metadata root must be an object")
        return validate_metadata(raw, has_literals=LITERALS_NAME in members)
    finally:
        archive.close()


def unpack_artifact(artifact: str | Path, destination: str | Path) -> dict[str, Any]:
    """Validate and atomically materialize an artifact into a new directory."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive, members = _members(Path(artifact))
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name, info in members.items():
            extracted = archive.extractfile(info)
            if extracted is None:
                raise ArtifactError(f"artifact member {name!r} is unreadable")
            with extracted as source:
                with (stage / name).open("wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                    output.flush()
                    os.fsync(output.fileno())
        metadata = read_metadata(artifact)
        os.replace(stage, target)
        _fsync_dir(target.parent)
        return metadata
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        archive.close()
