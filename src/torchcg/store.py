"""Where the artifact lives: content-addressed get/put over one tensorfs CAS.

An artifact is a compressed tarball, so it enters the store as one whole
unchunked blob named by the sha256 of its bytes, and an exact-key ref points
straight at it. There is no per-artifact manifest: a manifest holding exactly
one file entry is an indirection with nothing to describe.

Mint-once, first-writer-wins. The key IS the artifact's content address, so two
different byte strings under one key mean an axis the key does not carry decided
the output -- never something to overwrite.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorfs import CASRef, DigestMismatch, LocalCAS

from .identity import is_artifact_key
from .refuse import DivergentArtifact, StoreError

_REF_PREFIX = "torchcg/v1/graphs"
_BUFFER = 1 << 16
_MEMBERS = ("metadata.json", "model.pt2", "constants.safetensors")


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """One verified artifact unpacked for a caller."""

    key: str
    directory: Path
    metadata: dict[str, Any]
    ref: CASRef

    @property
    def package(self) -> Path:
        return self.directory / "model.pt2"

    @property
    def literals(self) -> Path | None:
        path = self.directory / "constants.safetensors"
        return path if path.is_file() else None


def _require_key(key: object) -> str:
    value = str(key)
    if not is_artifact_key(value):
        raise StoreError(f"{value!r} is not an artifact key")
    return value


def _ref_name(key: str) -> str:
    return f"{_REF_PREFIX}/{key}"


def _digest_file(path: Path) -> CASRef:
    """Name a file by its bytes without writing them anywhere."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_BUFFER):
            digest.update(chunk)
    return CASRef(digest.hexdigest())


class Store:
    """Artifact naming and immutability over one ``LocalCAS``."""

    def __init__(self, cas: LocalCAS) -> None:
        self.cas = cas

    def _admit(self, artifact: Path) -> CASRef:
        """Admit one whole-file blob, writing nothing when the store holds it.

        `verify_object`, not `contains`: presence is not integrity. `contains`
        is one lstat and answers True for bytes corrupted IN PLACE -- fine for a
        resume journal, wrong here, where an artifact whose blob went bad must be
        REPLACED rather than re-registered, or the corruption survives admission
        and surfaces later as a divergence on a key that never diverged.
        """

        ref = _digest_file(artifact)
        try:
            self.cas.verify_object(ref)
            return ref
        except (DigestMismatch, FileNotFoundError):
            return self.cas.put_file(artifact, expected=ref)

    def put(self, key: str | Any, artifact: Path) -> CASRef:
        """Store one artifact under its key. Idempotent; divergence REFUSES."""

        name = _require_key(key)
        if not artifact.is_file():
            raise StoreError(f"artifact {str(artifact)!r} is not a file")
        ref = self._admit(artifact)
        known = self.cas.read_ref(_ref_name(name))
        if known is None:
            self.cas.write_ref(_ref_name(name), ref)
            return ref
        if known.digest != ref.digest:
            raise DivergentArtifact(
                f"key {name} already holds {known.digest[:16]}, and these bytes "
                f"are {ref.digest[:16]}. The key IS the content address, so two "
                f"byte strings under one key mean an axis the key does not carry "
                f"decided the output -- find that axis rather than overwriting."
            )
        return ref

    def get(self, key: str | Any, destination: Path) -> StoredArtifact | None:
        """Unpack and verify one artifact, or None when the key is unknown."""

        name = _require_key(key)
        ref = self.cas.read_ref(_ref_name(name))
        if ref is None:
            return None
        try:
            self.cas.verify_object(ref)
            path = self.cas.object_path(ref)
        except (DigestMismatch, FileNotFoundError) as exc:
            raise StoreError(f"key {name} points at unreadable bytes: {exc}") from exc
        directory = unpack(path, destination)
        stamped = read_metadata(directory)
        if stamped.get("key") != name:
            raise StoreError(
                f"artifact under key {name} states key {stamped.get('key')!r}; the "
                f"bytes and the address disagree"
            )
        return StoredArtifact(key=name, directory=directory, metadata=stamped, ref=ref)

    def keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name.removeprefix(f"{_REF_PREFIX}/")
                for name in self.cas.list_refs(_REF_PREFIX)
            )
        )


def unpack(artifact: Path, destination: Path) -> Path:
    """Extract one artifact tarball, refusing any member it does not name.

    A closed member list rather than a path check: an archive is untrusted
    input, and "reject what escapes the directory" is a filter somebody has to
    keep correct, while "accept exactly these three names" cannot be walked out
    of at all.
    """

    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(artifact, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name not in _MEMBERS or not member.isfile():
                    raise StoreError(
                        f"artifact carries member {member.name!r}; a v1 artifact is "
                        f"exactly {list(_MEMBERS)!r}"
                    )
                tar.extract(member, destination, filter="data")
    except tarfile.TarError as exc:
        raise StoreError(f"artifact could not be read: {exc}") from exc
    if not (destination / "model.pt2").is_file():
        raise StoreError("artifact carries no model.pt2")
    return destination


def read_metadata(directory: Path) -> dict[str, Any]:
    try:
        stamped = json.loads((directory / "metadata.json").read_text())
    except (OSError, ValueError) as exc:
        raise StoreError(f"artifact metadata is unreadable: {exc}") from exc
    if not isinstance(stamped, dict):
        raise StoreError("artifact metadata is not an object")
    if stamped.get("kind") != "aot-inductor":
        raise StoreError(f"artifact states kind {stamped.get('kind')!r}")
    policy = stamped.get("compile_policy")
    if not isinstance(policy, dict) or not policy:
        raise StoreError(
            "artifact records no compile_policy; the options are half the "
            "compiler's input signature and a key without them collides"
        )
    if policy.get("aot_inductor.package_constants_in_so") is not False:
        raise StoreError(
            "package_constants_in_so must be false: an artifact is CODE, and its "
            "constants are raw pointers into the live module's weights"
        )
    if not (
        policy.get("always_keep_tensor_constants")
        or policy.get("aot_inductor.use_runtime_constant_folding")
    ):
        raise StoreError(
            "constant folding is unfenced: with neither always_keep_tensor_"
            "constants nor use_runtime_constant_folding, inductor inlines small "
            "lifted constants into generated code and the bindable table is short"
        )
    return stamped


__all__ = ["Store", "StoredArtifact", "read_metadata", "unpack"]
