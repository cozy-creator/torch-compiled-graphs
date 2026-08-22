"""Banked facts about the STORE: mint-once, first-writer-wins (tcg#84)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from tensorfs import LocalCAS

from torchcg.identity import artifact_key, contiguous_handle
from torchcg.refuse import KeyAlreadyMinted, StoreError
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


def test_a_key_MINTS_ONCE_whatever_the_bytes_say(store: Store, tmp_path: Path) -> None:
    """tcg#84 as ruled 2026-08-21: the KEY is the identity, the bytes are not.

    The refusal is on the key being PRESENT. It is not a byte comparison and
    must not become one -- AOTI does not emit the same bytes twice for one graph
    on one host, so a comparison here could not tell a key collision from
    compiler nondeterminism, and a verdict a check cannot prove is worse than no
    check because it will be believed.

    Both arms refuse identically, which is the whole point: the store does not
    know, and does not claim to know, whether these bytes match.
    """

    for name, body in (("same", b"first"), ("different", b"second")):
        substore = Store(LocalCAS(tmp_path / f"cas-{name}"))
        value = key()
        substore.put(value, make_artifact(tmp_path, f"{name}-1", key_value=value, body=b"first"))
        with pytest.raises(KeyAlreadyMinted, match="already minted"):
            substore.put(value, make_artifact(tmp_path, f"{name}-2", key_value=value, body=body))
        # The first artifact stands and is still servable.
        stored = substore.get(value, tmp_path / f"out-{name}")
        assert stored is not None and stored.package.read_bytes() == b"first"


def test_IDENTICAL_bytes_are_refused_too_which_is_how_we_know(
    store: Store, tmp_path: Path
) -> None:
    """The falsifier for the ruling itself.

    A byte-comparing `put` ACCEPTS identical bytes -- that is what makes it a
    comparison. So refusing them is the observable difference between "the store
    checks the key" and "the store checks the bytes", and it is the arm that
    goes red the day someone reintroduces the compare.

    This is not hypothetical: relying on `compare_and_swap_ref` alone did exactly
    that, because tensorfs treats a swap to the value already stored as a no-op
    success. The refusal fired only when the bytes differed.
    """

    value = key()
    body = b"identical"
    store.put(value, make_artifact(tmp_path, "a", key_value=value, body=body))
    with pytest.raises(KeyAlreadyMinted, match="already minted"):
        store.put(value, make_artifact(tmp_path, "b", key_value=value, body=body))


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
    """tar records mtime/uid/gid and gzip records the OUTPUT PATH in its header,
    so an envelope built the ordinary way varies with the clock and the scratch
    name.

    This was found when it made `put`'s byte comparison fire on a clock. That
    comparison is now gone (tcg#84 ruling), so the property is no longer
    load-bearing -- it is kept because a digest that means "these bytes" rather
    than "these bytes, packed at that moment" is the cheaper thing to have, and
    because it is a precondition of any future reproducibility work.
    """

    directory = tmp_path / "build"
    directory.mkdir()
    (directory / "model.pt2").write_bytes(b"so-bytes")
    (directory / "metadata.json").write_text("{}")
    first = pack(directory, tmp_path / "first.tar.gz").read_bytes()
    (directory / "model.pt2").touch()  # move the mtime
    second = pack(directory, tmp_path / "second-different-name.tar.gz").read_bytes()
    assert first == second
