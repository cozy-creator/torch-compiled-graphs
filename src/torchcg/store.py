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

from tensorfs import CASRef, DigestMismatch, LocalCAS, RefConflict

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
        # Compare-and-swap rather than read-then-write: first-writer-wins is a
        # claim about concurrent minters, and a read-then-write cannot make it.
        try:
            self.cas.compare_and_swap_ref(_ref_name(name), ref, expected=None)
            return ref
        except RefConflict:
            pass
        known = self.cas.read_ref(_ref_name(name))
        if known is not None and known.digest != ref.digest:
            raise DivergentArtifact(
                f"key {name} already holds {known.digest[:16]}, and these bytes "
                f"are {ref.digest[:16]}. The FIRST artifact stands and is still "
                f"servable; nothing was overwritten. Two byte strings under one "
                f"key mean either an axis the key does not carry decided the "
                f"output, or AOTI simply did not emit the same bytes twice -- "
                f"MEASURED not reproducible for one graph on one host, so this "
                f"refusal cannot tell the two apart. Reaching it at all means a "
                f"minter skipped `get` and re-minted a key that already existed."
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


def pack(directory: Path, destination: Path) -> Path:
    """Pack one artifact DETERMINISTICALLY.

    The envelope carries no time and no ownership: members are added in a fixed
    order with mtime 0, uid/gid 0 and no names, and gzip is given `mtime=0`.

    This is not tidiness. `put` reports divergence by comparing whole-file
    digests, so an envelope that varied with the clock would make two identical
    mints look like a key collision -- a false alarm on the one guard whose job
    is to catch a real one. With the envelope fixed, the only thing that can
    differ is the compiler's own output, which is exactly the question.
    """

    import gzip

    destination.parent.mkdir(parents=True, exist_ok=True)
    # `filename=""` because gzip stores the OUTPUT PATH in its header, so
    # packing the same artifact to two scratch names would digest differently.
    with destination.open("wb") as handle, gzip.GzipFile(
        filename="", mode="wb", fileobj=handle, mtime=0
    ) as raw:
        with tarfile.open(fileobj=raw, mode="w") as tar:  # type: ignore[arg-type]
            for member in _MEMBERS:
                path = directory / member
                if not path.is_file():
                    continue
                info = tar.gettarinfo(str(path), arcname=member)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
    return destination


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


__all__ = ["Store", "StoredArtifact", "pack", "read_metadata", "unpack"]
