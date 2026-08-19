"""Lanes, the graph-set document, and the store lifecycle over a local CAS.

The lifecycle here runs against ``LocalGraphStore`` with zero gen-worker
knowledge -- the tcg#41 boundary -- and a structural fence asserts the whole
package keeps it that way.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tensorfs import LocalCAS

from torchcg.document import DocumentError, GraphRecord, GraphSetDocument, LaneGraphs
from torchcg.graph_identity import EnvIdentity
from torchcg.ingress import CallIngress, CallInput
from torchcg.lane import LaneError, LaneRef, require_targets, resolve_target
from torchcg.requirements import RequirementsManifest
from torchcg.store import LocalGraphStore, PublishOutcome, StoreError, holes

STACK = (("torch", "2.13.0"),)
ENV = EnvIdentity(stack=STACK, sm="sm_89")


def ingress() -> CallIngress:
    return CallIngress(
        parameters=("sample",),
        flat_arity=1,
        inputs=(
            CallInput(
                name="sample",
                position=0,
                param="sample",
                param_position=0,
                path=(),
                exported_name="sample",
                dtype="float32",
                shape=(1, 4),
            ),
        ),
    )


def graph(letter: str) -> str:
    return "cg-graph-v1-" + letter * 56


def lane_graphs(*letters: str) -> LaneGraphs:
    return LaneGraphs(
        contract="sd15.diffusers-bf16@1",
        targets=("pipe.unet", "pipe.vae.decoder"),
        graphs=tuple(
            GraphRecord(graph=graph(letter), target="pipe.unet", ingress=ingress())
            for letter in letters
        ),
        unobserved_targets=("pipe.vae.decoder",),
    )


class TestLane:
    def test_accepts_the_contract_file_spelling(self) -> None:
        lane = LaneRef("sdxl.diffusers-bf16@1", dtype="bf16-stand-in")
        assert lane.contract == "sdxl.diffusers-bf16@1"
        assert lane.dtype == "bf16-stand-in"

    def test_refuses_noncanonical_declarations(self) -> None:
        with pytest.raises(LaneError):
            LaneRef("not a contract")
        with pytest.raises(LaneError):
            LaneRef("missing-version")
        with pytest.raises(LaneError):
            LaneRef("producer.format@0")
        with pytest.raises(LaneError):
            require_targets(())
        with pytest.raises(LaneError):
            require_targets(("vae.1bad",))
        with pytest.raises(LaneError):
            require_targets(("unet", "unet"))

    def test_resolves_paths_and_names_the_failing_segment(self) -> None:
        class Leaf:
            pass

        class Node:
            leaf = Leaf()

        assert isinstance(resolve_target({"root": Node()}, "root.leaf"), Leaf)
        with pytest.raises(LaneError, match="'missing' does not exist"):
            resolve_target({"root": Node()}, "root.leaf.missing")
        with pytest.raises(LaneError, match="not in the author namespace"):
            resolve_target({"root": Node()}, "other.leaf")


class TestDocument:
    def test_document_bytes_are_canonical_and_roundtrip(self) -> None:
        document = GraphSetDocument(stack=STACK, lanes=(lane_graphs("a", "b"),))
        assert GraphSetDocument.decode(document.encode()) == document
        assert GraphSetDocument.decode(document.encode()).encode() == document.encode()

    def test_empty_document_is_the_stated_eager_marker(self) -> None:
        document = GraphSetDocument(stack=STACK)
        assert document.eager_permanent
        assert GraphSetDocument.decode(document.encode()).eager_permanent

    def test_every_declared_target_must_be_stated(self) -> None:
        with pytest.raises(DocumentError, match="says nothing about"):
            LaneGraphs(
                contract="unit.c@1",
                targets=("pipe.unet", "pipe.vae.decoder"),
                graphs=(),
                unobserved_targets=("pipe.unet",),
            )

    def test_duplicate_graph_rows_are_a_producer_bug(self) -> None:
        with pytest.raises(DocumentError, match="repeats a graph hash"):
            lane_graphs("a", "a")


class TestLocalGraphStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> LocalGraphStore:
        return LocalGraphStore(LocalCAS(tmp_path / "cas"))

    def test_graphset_roundtrip_and_replace(self, store: LocalGraphStore) -> None:
        assert store.get_graphs("release-1") is None
        document = GraphSetDocument(stack=STACK, lanes=(lane_graphs("a"),))
        store.put_graphs("release-1", document)
        fetched = store.get_graphs("release-1")
        assert fetched is not None and fetched.encode() == document.encode()
        replacement = GraphSetDocument(stack=STACK, lanes=(lane_graphs("b"),))
        store.put_graphs("release-1", replacement)
        fetched = store.get_graphs("release-1")
        assert fetched is not None and fetched.encode() == replacement.encode()

    def test_artifact_lifecycle_publish_fetch_holes(
        self, store: LocalGraphStore, tmp_path: Path
    ) -> None:
        lane = lane_graphs("a", "b")
        assert holes(store, lane, ENV) == (graph("a"), graph("b"))

        artifact = tmp_path / "compiled_graph.tar.gz"
        artifact.write_bytes(b"minted-bytes")
        manifest = RequirementsManifest(
            include_set=(("torch", "2.13.0"),), sm_compiled="sm_89"
        )
        assert (
            store.publish_artifact(graph("a"), ENV, artifact, manifest)
            is PublishOutcome.PUBLISHED
        )
        assert (
            store.publish_artifact(graph("a"), ENV, artifact, manifest)
            is PublishOutcome.PRESENT
        )
        # Partial-hit: only the unminted graph remains a hole.
        assert holes(store, lane, ENV) == (graph("b"),)
        # A different env is a different position entirely.
        other_env = EnvIdentity(stack=STACK, sm="sm_86")
        assert holes(store, lane, other_env) == (graph("a"), graph("b"))

        fetched = store.fetch_artifact(graph("a"), ENV, tmp_path / "out" / "a.tar.gz")
        assert fetched is not None and fetched.read_bytes() == b"minted-bytes"
        assert store.fetch_artifact(graph("b"), ENV, tmp_path / "b.tar.gz") is None
        assert store.get_manifest(graph("a"), ENV) == manifest
        assert store.get_manifest(graph("b"), ENV) is None

    def test_divergent_bytes_under_an_occupied_position_refuse(
        self, store: LocalGraphStore, tmp_path: Path
    ) -> None:
        manifest = RequirementsManifest(
            include_set=(("torch", "2.13.0"),), sm_compiled="sm_89"
        )
        first = tmp_path / "first.tar.gz"
        first.write_bytes(b"one")
        second = tmp_path / "second.tar.gz"
        second.write_bytes(b"two")
        store.publish_artifact(graph("a"), ENV, first, manifest)
        with pytest.raises(StoreError, match="diverged"):
            store.publish_artifact(graph("a"), ENV, second, manifest)

    def test_manifest_sm_must_match_the_publish_position(
        self, store: LocalGraphStore, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "a.tar.gz"
        artifact.write_bytes(b"bytes")
        manifest = RequirementsManifest(
            include_set=(("torch", "2.13.0"),), sm_compiled="sm_86"
        )
        with pytest.raises(StoreError, match="wrote one of them wrong"):
            store.publish_artifact(graph("a"), ENV, artifact, manifest)


def test_torchcg_knows_no_worker_and_no_hub() -> None:
    """The tcg#41 boundary as a fence: no gen-worker, hub, or wire imports."""

    package = Path(__file__).resolve().parents[1] / "src" / "torchcg"
    forbidden = ("gen_worker", "tensorhub", "import requests", "import httpx", "jwt")
    for source in sorted(package.rglob("*.py")):
        text = source.read_text()
        for name in forbidden:
            assert name not in text, f"{source.name} references {name!r}"
