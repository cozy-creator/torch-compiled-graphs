"""Banked facts about the STORE: mint-once, first-writer-wins (tcg#84)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from tensorfs import LocalCAS

from torchcg.identity import artifact_key, contiguous_handle
from torchcg.refuse import DivergentArtifact, StoreError
from torchcg.store import Store, pack

pytest.importorskip("torch")

GRAPH = "cg-graph-v1-" + "a" * 56
POLICY = {"always_keep_tensor_constants": True,
          "aot_inductor.package_constants_in_so": False}


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(LocalCAS(tmp_path / "cas"))


def key(graph: str = GRAPH) -> str:
    return artifact_key(
        graph,
        sm="sm_89",
        env={"torch": "2.13.0"},
        policy=POLICY,
        layout=contiguous_handle(),
    ).value


def make_artifact(tmp_path: Path, name: str, *, key_value: str, body: bytes) -> Path:
    directory = tmp_path / f"build-{name}"
    directory.mkdir()
    (directory / "model.pt2").write_bytes(body)
    (directory / "metadata.json").write_text(
        json.dumps({
            "kind": "aot-inductor",
            "key": key_value,
            "graph": GRAPH,
            "name": GRAPH,
            "compile_policy": POLICY,
            "declared_input_layout": contiguous_handle(),
        })
    )
    return pack(directory, tmp_path / f"{name}.tar.gz")


def test_put_then_get_round_trips(store: Store, tmp_path: Path) -> None:
    value = key()
    artifact = make_artifact(tmp_path, "a", key_value=value, body=b"so-bytes")
    store.put(value, artifact)
    stored = store.get(value, tmp_path / "out")
    assert stored is not None
    assert stored.key == value
    assert stored.package.read_bytes() == b"so-bytes"


def test_an_unknown_key_is_None_not_an_error(store: Store, tmp_path: Path) -> None:
    assert store.get(key(), tmp_path / "out") is None


def test_storing_the_SAME_bytes_twice_is_idempotent(store: Store, tmp_path: Path) -> None:
    value = key()
    first = make_artifact(tmp_path, "a", key_value=value, body=b"so-bytes")
    second = make_artifact(tmp_path, "b", key_value=value, body=b"so-bytes")
    assert store.put(value, first).digest == store.put(value, second).digest


def test_DIVERGENT_bytes_under_one_key_REFUSE(store: Store, tmp_path: Path) -> None:
    """tcg#84, mint-once / first-writer-wins.

    The key IS the artifact's content address, so two different byte strings
    under one key mean an axis the key does not carry decided the output. The
    remedy is to find that axis -- never to overwrite, which would make the
    store's answer depend on who published last.
    """

    value = key()
    store.put(value, make_artifact(tmp_path, "a", key_value=value, body=b"first"))
    with pytest.raises(DivergentArtifact, match="content address"):
        store.put(value, make_artifact(tmp_path, "b", key_value=value, body=b"second"))
    # The FIRST writer still wins: the store was not left half-updated.
    stored = store.get(value, tmp_path / "out")
    assert stored is not None and stored.package.read_bytes() == b"first"


def test_bytes_that_disagree_with_their_own_ADDRESS_refuse(
    store: Store, tmp_path: Path
) -> None:
    """The artifact states a key; it must be the key it was fetched under.
    Without this, a mis-filed artifact serves silently under the wrong identity."""

    value, other = key(), key("cg-graph-v1-" + "b" * 56)
    store.put(value, make_artifact(tmp_path, "a", key_value=other, body=b"so"))
    with pytest.raises(StoreError, match="bytes and the address disagree"):
        store.get(value, tmp_path / "out")


def test_an_artifact_carrying_an_unnamed_MEMBER_refuses(
    store: Store, tmp_path: Path
) -> None:
    """A closed member list rather than a path check: an archive is untrusted
    input, and "reject what escapes the directory" is a filter somebody has to
    keep correct, while "accept exactly these three names" cannot be walked out
    of at all."""

    value = key()
    artifact = make_artifact(tmp_path, "a", key_value=value, body=b"so")
    smuggled = tmp_path / "smuggled.tar.gz"
    (tmp_path / "evil.py").write_text("# anything")
    with tarfile.open(artifact, "r:gz") as source, tarfile.open(smuggled, "w:gz") as out:
        for member in source.getmembers():
            out.addfile(member, source.extractfile(member))
        out.add(tmp_path / "evil.py", arcname="../evil.py")
    store.put(value, smuggled)
    with pytest.raises(StoreError, match="carries member"):
        store.get(value, tmp_path / "out")


def test_a_key_shaped_wrong_never_reaches_the_store(store: Store, tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="is not an artifact key"):
        store.get("not-a-key", tmp_path / "out")


def test_the_ENVELOPE_is_deterministic(tmp_path: Path) -> None:
    """Found while writing the divergence arm: two tarballs of identical content
    hashed differently, so `put` reported a key collision where nothing had
    collided -- a false alarm on the one guard whose job is a real one.

    tar records mtime/uid/gid and gzip records the OUTPUT PATH in its header, so
    an envelope built the ordinary way varies with the clock and the scratch
    name. Fixed, the only thing that can differ between two mints of one key is
    the compiler's own output, which is exactly the question being asked.
    """

    directory = tmp_path / "build"
    directory.mkdir()
    (directory / "model.pt2").write_bytes(b"so-bytes")
    (directory / "metadata.json").write_text("{}")
    first = pack(directory, tmp_path / "first.tar.gz").read_bytes()
    (directory / "model.pt2").touch()  # move the mtime
    second = pack(directory, tmp_path / "second-different-name.tar.gz").read_bytes()
    assert first == second
