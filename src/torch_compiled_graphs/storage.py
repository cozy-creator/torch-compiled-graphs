"""Exact-key compiled-graph policy over HashRepo's generic local CAS."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hashrepo import CASRef, FileEntry, LocalCAS, RefConflict, RepositoryManifest

from .artifact import read_metadata, unpack_artifact
from .identity import KEY_SCHEME, CompiledGraphKey

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
    if not value.startswith(f"{KEY_SCHEME}-"):
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
        metadata = read_metadata(source)
        if metadata.get("cell_key") != value:
            raise StorageError(
                f"artifact states key {metadata.get('cell_key')!r}, expected {value!r}"
            )

        file = self.cas.ingest_file(source, manifest_path=_ARTIFACT_PATH)
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
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_archive = tempfile.mkstemp(
            prefix="graph-", suffix=".tar.gz", dir=target.parent
        )
        os.close(descriptor)
        archive = Path(raw_archive)
        try:
            manifest = self.cas.load_manifest(manifest_ref)
            self.cas.materialize(self._file(manifest), archive)
            metadata = unpack_artifact(archive, target)
            if metadata.get("cell_key") != value:
                raise StorageError(
                    f"stored artifact states key {metadata.get('cell_key')!r}, expected {value!r}"
                )
            return StoredGraph(value, target, metadata, manifest_ref)
        except (OSError, ValueError, StorageError) as exc:
            shutil.rmtree(target, ignore_errors=True)
            self.quarantine(value, manifest_ref)
            raise QuarantinedArtifact(f"compiled graph {value} failed verification: {exc}") from exc
        finally:
            archive.unlink(missing_ok=True)
