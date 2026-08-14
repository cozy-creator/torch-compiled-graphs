"""Exact-key compiled-graph policy over HashRepo's generic local CAS."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hashrepo import CASRef, DigestMismatch, FileEntry, LocalCAS, RefConflict, RepositoryManifest

from .artifact import ArtifactError, _fsync_dir, _verify_materialized, unpack_artifact
from .identity import KEY_SCHEME, CompiledGraphKey, is_compiled_graph_key

_ARTIFACT_PATH = "compiled-graph.tar.gz"
_REF_PREFIX = "torch-compiled-graphs/v1"


class StorageError(RuntimeError):
    """A local compiled-graph record is malformed or unavailable."""


class QuarantinedArtifact(StorageError):
    """An exact-key record failed integrity or admission and cannot be reused."""


class StoreOutcome(str, Enum):
    STORED = "stored"
    PRESENT = "present"
    REPAIRED = "repaired"
    DIVERGENT = "divergent"


@dataclass(frozen=True, slots=True)
class StoreResult:
    outcome: StoreOutcome
    key: str
    manifest: CASRef


@dataclass(frozen=True, slots=True)
class StoredGraph:
    """One verified artifact materialized for a caller."""

    key: str
    directory: Path
    metadata: dict[str, object]
    manifest: CASRef

    @property
    def package(self) -> Path:
        return self.directory / "model.pt2"

    @property
    def literals(self) -> Path | None:
        path = self.directory / "constants.safetensors"
        return path if path.is_file() else None


def _key_value(key: str | CompiledGraphKey) -> str:
    value = str(key)
    if not is_compiled_graph_key(value):
        raise StorageError(f"v1 storage requires a {KEY_SCHEME} key")
    return value


def _graph_ref(key: str) -> str:
    return f"{_REF_PREFIX}/graphs/{key}"


def _quarantine_ref(key: str, manifest: CASRef) -> str:
    return f"{_REF_PREFIX}/quarantine/{key}/{manifest.digest}"


class _GraphStore:
    """Compiled-graph naming, immutability, and quarantine over one ``LocalCAS``."""

    def __init__(self, cas: LocalCAS) -> None:
        self.cas = cas

    @staticmethod
    def _file(manifest: RepositoryManifest) -> FileEntry:
        if len(manifest.files) != 1 or manifest.files[0].path != _ARTIFACT_PATH:
            raise StorageError("compiled-graph manifest must contain exactly its v1 artifact")
        return manifest.files[0]

    def _is_quarantined(self, key: str, manifest: CASRef) -> bool:
        return self.cas.read_ref(_quarantine_ref(key, manifest)) is not None

    def _clear_quarantine(self, key: str, manifest: CASRef) -> None:
        name = _quarantine_ref(key, manifest)
        marker = self.cas.read_ref(name)
        if marker is None:
            return
        try:
            self.cas.compare_and_swap_ref(name, None, expected=marker)
        except RefConflict:
            pass

    def quarantine(self, key: str | CompiledGraphKey, manifest: CASRef) -> None:
        value = _key_value(key)
        name = _quarantine_ref(value, manifest)
        marker = self.cas.put_bytes(
            json.dumps(
                {"format": 1, "key": value, "manifest": str(manifest)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        try:
            self.cas.compare_and_swap_ref(name, marker, expected=None)
        except RefConflict:
            pass

    def store(self, key: str | CompiledGraphKey, artifact: str | Path) -> StoreResult:
        """Verify and keep the first bytes published under one exact graph key."""

        value = _key_value(key)
        source = Path(artifact)
        with tempfile.TemporaryDirectory(prefix="torch-compiled-graphs-import-") as raw:
            owned = Path(raw) / _ARTIFACT_PATH
            shutil.copyfile(source, owned)
            with owned.open("rb") as handle:
                os.fsync(handle.fileno())
            metadata = unpack_artifact(owned, Path(raw) / "verified")
            if metadata.get("compiled_graph_key") != value:
                raise StorageError(
                    f"artifact states key {metadata.get('compiled_graph_key')!r}, "
                    f"expected {value!r}"
                )
            file = self.cas.ingest_file(owned, manifest_path=_ARTIFACT_PATH)
        candidate = self.cas.store_manifest(RepositoryManifest((file,)))
        ref_name = _graph_ref(value)
        current = self.cas.read_ref(ref_name)
        if current is None:
            try:
                self.cas.compare_and_swap_ref(ref_name, candidate, expected=None)
                return StoreResult(StoreOutcome.STORED, value, candidate)
            except RefConflict:
                current = self.cas.read_ref(ref_name)
        if current is None:
            raise StorageError(f"exact-key ref {value} disappeared during publication")
        if current == candidate and self._is_quarantined(value, current):
            self._clear_quarantine(value, current)
            return StoreResult(StoreOutcome.REPAIRED, value, current)
        if current == candidate:
            return StoreResult(StoreOutcome.PRESENT, value, current)
        if self._is_quarantined(value, current):
            try:
                self.cas.compare_and_swap_ref(ref_name, candidate, expected=current)
                return StoreResult(StoreOutcome.REPAIRED, value, candidate)
            except RefConflict:
                current = self.cas.read_ref(ref_name)
                if current == candidate:
                    return StoreResult(StoreOutcome.PRESENT, value, candidate)
                if current is None:
                    raise StorageError(f"exact-key ref {value} disappeared during repair") from None
        self.quarantine(value, candidate)
        return StoreResult(StoreOutcome.DIVERGENT, value, candidate)

    @staticmethod
    def _verify_archive(artifact: Path, key: str) -> None:
        with tempfile.TemporaryDirectory(prefix="torch-compiled-graphs-export-verify-") as raw:
            metadata = unpack_artifact(artifact, Path(raw) / "artifact")
        if metadata.get("compiled_graph_key") != key:
            raise ArtifactError(
                f"artifact states key {metadata.get('compiled_graph_key')!r}, expected {key!r}"
            )

    def export_artifact(self, key: str | CompiledGraphKey, destination: str | Path) -> Path:
        """Export one exact CAS record as a fully verified artifact envelope."""

        value = _key_value(key)
        manifest_ref = self.cas.read_ref(_graph_ref(value))
        if manifest_ref is None:
            raise StorageError(f"compiled graph {value} is not present")
        if self._is_quarantined(value, manifest_ref):
            raise QuarantinedArtifact(f"compiled graph {value} is quarantined")

        target = Path(destination)
        if target.exists():
            if not target.is_file():
                raise FileExistsError(f"destination {target} is not a regular file")
            self._verify_archive(target, value)
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            try:
                manifest = self.cas.load_manifest(manifest_ref)
                self.cas.materialize(self._file(manifest), temporary)
                self._verify_archive(temporary, value)
            except (
                ArtifactError,
                DigestMismatch,
                FileNotFoundError,
                StorageError,
                ValueError,
            ) as exc:
                self.quarantine(value, manifest_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed export verification: {exc}"
                ) from exc
            try:
                os.link(temporary, target)
            except FileExistsError:
                if not target.is_file():
                    raise
                self._verify_archive(target, value)
            _fsync_dir(target.parent)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    def resolve(self, key: str | CompiledGraphKey, destination: str | Path) -> StoredGraph | None:
        """Materialize and fully verify one exact key, or return a clean miss."""

        value = _key_value(key)
        manifest_ref = self.cas.read_ref(_graph_ref(value))
        if manifest_ref is None:
            return None
        if self._is_quarantined(value, manifest_ref):
            raise QuarantinedArtifact(f"compiled graph {value} is quarantined")

        target = Path(destination)
        if target.exists():
            metadata = _verify_materialized(target)
            if metadata.get("compiled_graph_key") != value:
                raise FileExistsError(f"destination {target} contains a different graph")
            return StoredGraph(value, target, metadata, manifest_ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_archive = tempfile.mkstemp(
            prefix="graph-", suffix=".tar.gz", dir=target.parent
        )
        os.close(descriptor)
        archive = Path(raw_archive)
        candidate = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        candidate.rmdir()
        try:
            try:
                manifest = self.cas.load_manifest(manifest_ref)
                file = self._file(manifest)
                self.cas.materialize(file, archive)
            except (DigestMismatch, FileNotFoundError, ValueError, StorageError) as exc:
                self.quarantine(value, manifest_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed CAS verification: {exc}"
                ) from exc
            try:
                metadata = unpack_artifact(archive, candidate)
            except ArtifactError as exc:
                self.quarantine(value, manifest_ref)
                raise QuarantinedArtifact(
                    f"compiled graph {value} failed artifact verification: {exc}"
                ) from exc
            if metadata.get("compiled_graph_key") != value:
                self.quarantine(value, manifest_ref)
                raise QuarantinedArtifact(
                    f"stored artifact states key {metadata.get('compiled_graph_key')!r}, "
                    f"expected {value!r}"
                )
            try:
                os.rename(candidate, target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not target.is_dir():
                    raise
                winner = _verify_materialized(target)
                if winner.get("compiled_graph_key") != value:
                    raise FileExistsError(
                        f"destination {target} contains a different graph"
                    ) from exc
                return StoredGraph(value, target, winner, manifest_ref)
            _fsync_dir(target.parent)
            return StoredGraph(value, target, metadata, manifest_ref)
        except QuarantinedArtifact:
            raise
        except (OSError, ArtifactError):
            raise
        except ValueError as exc:
            self.quarantine(value, manifest_ref)
            raise QuarantinedArtifact(f"compiled graph {value} failed verification: {exc}") from exc
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(candidate, ignore_errors=True)
