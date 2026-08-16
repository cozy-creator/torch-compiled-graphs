"""Blob custody: what the store does with an artifact's bytes.

These exercise ``_CompiledGraphStore`` directly against a real tensorfs store and
a real artifact envelope, with no torch anywhere -- the engine-level tests that
cover the same seam all sit behind ``importorskip("torch")`` and skip wherever
torch is absent, which left byte custody with no unconditional proof.

The properties under test are the ones the blob cutover is responsible for:
admission converges, a repeat admission writes nothing, a corrupt blob is
refused at lookup, and acquiring an artifact never writes through the store's
own path.
"""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
from tensorfs import CASRef, LocalCAS
from test_artifact import aoti_package, metadata

import torchcg.storage as storage_module
from torchcg.artifact import pack_artifact, read_metadata, unpack_artifact
from torchcg.storage import (
    QuarantinedArtifact,
    StoreOutcome,
    _CompiledGraphStore,
    _graph_ref,
)

# Every gate below is asserted this many times over independent stores. One pass
# of a convergence or zero-write property is an anecdote; the repeat is the gate.
_GATE_RUNS = 5


def _artifact(tmp_path: Path, name: str = "graph") -> tuple[Path, str]:
    """One valid v1 envelope and the exact key it states."""

    package = aoti_package(tmp_path / f"{name}.pt2")
    built = pack_artifact(package, tmp_path / f"{name}.tar.gz", metadata())
    key = read_metadata(built)["compiled_graph_key"]
    assert isinstance(key, str)
    return built, key


def _store(root: Path) -> tuple[_CompiledGraphStore, LocalCAS]:
    cas = LocalCAS(root)
    return _CompiledGraphStore(cas), cas


def _objects(cas: LocalCAS) -> dict[Path, tuple[int, int, int]]:
    """Identity of every object in the store, so a rewrite cannot hide."""

    identities: dict[Path, tuple[int, int, int]] = {}
    for path in sorted(cas.objects.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        identities[path] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return identities


@pytest.mark.parametrize("run", range(_GATE_RUNS))
def test_admission_converges_on_one_blob(run: int, tmp_path: Path) -> None:
    """Storing the same artifact twice converges on the same blob, once."""

    artifact, key = _artifact(tmp_path)
    store, cas = _store(tmp_path / "cas")

    first = store.store(key, artifact)
    assert first.outcome is StoreOutcome.STORED

    second = store.store(key, artifact)
    assert second.outcome is StoreOutcome.PRESENT
    assert second.artifact == first.artifact

    # The ref names the blob itself, not a manifest wrapping it: the digest the
    # store published must be the digest of the artifact's own bytes.
    assert cas.read_ref(_graph_ref(key)) == first.artifact
    assert first.artifact == CASRef(hashlib.sha256(artifact.read_bytes()).hexdigest())
    assert cas.object_path(first.artifact).read_bytes() == artifact.read_bytes()


@pytest.mark.parametrize("run", range(_GATE_RUNS))
def test_a_second_admission_moves_zero_bytes(
    run: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-admitting bytes the store already holds must not write them again."""

    artifact, key = _artifact(tmp_path)
    store, cas = _store(tmp_path / "cas")
    store.store(key, artifact)

    before = _objects(cas)
    assert before, "the first admission stored nothing"

    def refuse(*args: object, **kwargs: object) -> CASRef:
        raise AssertionError("a repeat admission wrote the blob a second time")

    monkeypatch.setattr(LocalCAS, "put_file", refuse)
    repeat = store.store(key, artifact)

    assert repeat.outcome is StoreOutcome.PRESENT
    # Nothing created, nothing rewritten -- same inodes, sizes and mtimes.
    assert _objects(cas) == before


@pytest.mark.parametrize("run", range(_GATE_RUNS))
def test_lookup_refuses_a_blob_that_is_a_different_valid_artifact(
    run: int, tmp_path: Path
) -> None:
    """Only the digest catches a swap for OTHER well-formed, same-key bytes.

    Garbage in the blob is caught downstream by the unpacker, so corrupting it
    with junk tests the tar reader, not the store -- that version of this gate
    passed even with verification removed. The substitute here unpacks cleanly
    and states the very key being resolved, so the sole thing standing between
    it and the caller is re-hashing the blob at lookup.
    """

    artifact, key = _artifact(tmp_path, "genuine")
    # Same metadata, hence the same key; a larger .lrodata section, hence
    # different bytes.
    other = pack_artifact(
        aoti_package(tmp_path / "other.pt2", lrodata=64),
        tmp_path / "other.tar.gz",
        metadata(),
    )
    assert other.read_bytes() != artifact.read_bytes()
    assert read_metadata(other)["compiled_graph_key"] == key

    store, cas = _store(tmp_path / "cas")
    stored = store.store(key, artifact)
    cas.object_path(stored.artifact).write_bytes(other.read_bytes())

    with pytest.raises(QuarantinedArtifact):
        store.resolve(key, tmp_path / "resolved")
    with pytest.raises(QuarantinedArtifact):
        store.export_artifact(key, tmp_path / "exported.tar.gz")


@pytest.mark.parametrize("run", range(_GATE_RUNS))
def test_lookup_refuses_a_blob_of_garbage(run: int, tmp_path: Path) -> None:
    """The coarse case: bytes that are not an artifact at all are refused."""

    artifact, key = _artifact(tmp_path)
    store, cas = _store(tmp_path / "cas")
    stored = store.store(key, artifact)
    cas.object_path(stored.artifact).write_bytes(b"not the artifact")

    with pytest.raises(QuarantinedArtifact):
        store.resolve(key, tmp_path / "resolved")
    with pytest.raises(QuarantinedArtifact):
        store.export_artifact(key, tmp_path / "exported.tar.gz")


@pytest.mark.parametrize("run", range(_GATE_RUNS))
def test_acquiring_an_artifact_never_writes_through_the_blob(
    run: int, tmp_path: Path
) -> None:
    """The store's own path is read-only to torchcg, whatever its mode bits.

    tensorfs installs objects 0600 today (0444 is designed but unlanded, see
    tcg#28), so the filesystem will not stop a careless write. This asserts the
    half torchcg owns: neither resolving nor exporting mutates the blob.
    """

    artifact, key = _artifact(tmp_path)
    store, cas = _store(tmp_path / "cas")
    stored = store.store(key, artifact)
    blob = cas.object_path(stored.artifact)

    before = blob.stat()
    original = blob.read_bytes()

    store.resolve(key, tmp_path / "resolved")
    store.export_artifact(key, tmp_path / "exported.tar.gz")

    after = blob.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert blob.read_bytes() == original
    # The export is an independent file, never a hardlink aliasing the blob.
    assert (tmp_path / "exported.tar.gz").stat().st_ino != before.st_ino


def test_resolve_untars_straight_out_of_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acquisition copies nothing: the unpacker is handed the blob itself.

    This is the property that replaced ``cas.materialize`` at the resolve site.
    Asserting the unpack source IS the object path is what would fail if a
    staging copy were ever reintroduced.
    """

    artifact, key = _artifact(tmp_path)
    store, cas = _store(tmp_path / "cas")
    stored = store.store(key, artifact)

    seen: list[Path] = []

    def record(source: str | Path, destination: str | Path) -> dict[str, Any]:
        seen.append(Path(source))
        return unpack_artifact(source, destination)

    monkeypatch.setattr(storage_module, "unpack_artifact", record)
    resolved = store.resolve(key, tmp_path / "resolved")

    assert resolved is not None
    assert seen == [cas.object_path(stored.artifact)]
    # No archive-shaped temp was staged beside the destination.
    assert not list(tmp_path.glob("*.tar.gz.*"))
    assert not list(tmp_path.glob("graph-*.tar.gz"))


def test_export_is_verified_and_republishable(tmp_path: Path) -> None:
    """Export hands back the exact artifact bytes, and is idempotent."""

    artifact, key = _artifact(tmp_path)
    store, _ = _store(tmp_path / "cas")
    store.store(key, artifact)

    destination = tmp_path / "out" / "exported.tar.gz"
    assert store.export_artifact(key, destination) == destination
    assert destination.read_bytes() == artifact.read_bytes()

    # A second export onto the occupied destination accepts it as already-correct.
    assert store.export_artifact(key, destination) == destination
    assert destination.read_bytes() == artifact.read_bytes()


def _fail_fsync_under(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """Make fsync report ENOSPC for one subtree only.

    Failing every fsync would be useless: the quarantine marker is itself written
    through the store, so a blanket failure makes quarantining raise OSError too
    and the test passes no matter which branch ran. Only the caller's disk is
    allowed to be full here.
    """

    original = os.fsync

    def no_space(descriptor: int) -> None:
        try:
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            return original(descriptor)
        if root == target or root in target.parents:
            raise OSError(errno.ENOSPC, "no space")
        return original(descriptor)

    monkeypatch.setattr(os, "fsync", no_space)


def test_a_destination_write_failure_does_not_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENOSPC on the caller's disk is no evidence about the stored bytes."""

    artifact, key = _artifact(tmp_path)
    store, _ = _store(tmp_path / "cas")
    store.store(key, artifact)

    destinations = tmp_path / "out"
    destinations.mkdir()
    _fail_fsync_under(monkeypatch, destinations)

    with pytest.raises(OSError) as raised:
        store.resolve(key, destinations / "failed")
    assert raised.value.errno == errno.ENOSPC
    assert not isinstance(raised.value, QuarantinedArtifact)

    monkeypatch.undo()
    # The bytes were never in question, so nothing may have been quarantined.
    assert store.resolve(key, tmp_path / "healthy") is not None
