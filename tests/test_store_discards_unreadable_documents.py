"""A graph-set document this build cannot read is ABSENT, and gets discarded.

Lineage: tcg#69, filed from pgw#1525. The blocking observation was real and on
disk: a box's ``~/.cache/cozy/graph-cas`` held ``sd15.main`` and ``sdxl.main``
written at ``DOCUMENT_FORMAT = 2``; a build vendoring ``DOCUMENT_FORMAT = 3``
raised ``StoreError: graph-set document 'sd15.main' is unreadable: document v
must be 3`` at ADOPT time, so the first cold start after an upgrade failed on a
machine that could simply have rebuilt. Compiled graphs are derived and
disposable (Paul, 2026-08-19: "weights migrate/normalize once at ingest; graphs
regenerate"), so the store self-heals by re-derive and discard, never by migration.

Two properties the fix must have and the obvious fix does not:

* it must not special-case ``v``. The live documents were ``v=2``; the issue as
  filed said ``v=1``. A reader that translated one spelling would have passed a
  green test over the defect actually on disk. Every unreadable shape is a miss.
* it must not discard a document it CAN read. "Treat everything as absent" also
  turns the StoreError green, and would silently retire every compiled artifact
  on the box on every boot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from tensorfs import LocalCAS

from torchcg.document import GraphRecord, GraphSetDocument, LaneGraphs
from torchcg.graph_identity import EnvIdentity
from torchcg.ingress import CallIngress, CallInput
from torchcg.requirements import RequirementsManifest
from torchcg.store import LocalGraphStore, StoreError

STACK = (("torch", "2.13.0"),)
NAME = "sd15.main"
REF = "torchcg/v2/graphsets/" + NAME


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


def a_readable_document() -> GraphSetDocument:
    return GraphSetDocument(
        stack=STACK,
        lanes=(
            LaneGraphs(
                contract="sd15.diffusers@1+plain.bf16@1",
                targets=("pipe.unet",),
                graphs=(
                    GraphRecord(
                        graph="cg-graph-v1-" + "a" * 56,
                        target="pipe.unet",
                        ingress=ingress(),
                    ),
                ),
                unobserved_targets=(),
            ),
        ),
    )


def store_at(root: Path) -> tuple[LocalCAS, LocalGraphStore]:
    cas = LocalCAS(root)
    return cas, LocalGraphStore(cas)


def plant(cas: LocalCAS, raw: bytes) -> str:
    """Put ``raw`` at the ``sd15.main`` graph-set document position."""

    ref = cas.put_bytes(raw)
    cas.compare_and_swap_ref(REF, ref, expected=cas.read_ref(REF))
    return str(ref)


def a_v2_document() -> bytes:
    """The shape that was actually on disk: today's document, stamped v=2."""

    body = a_readable_document().as_dict()
    body["v"] = 2
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")


#: Every way a document can fail to decode. The parametrization is the point:
#: the policy is about UNREADABILITY, not about one version number.
UNREADABLE = {
    "an older document format (the shape found on disk)": a_v2_document(),
    "a v1 spelling (the shape the issue was filed as)": json.dumps(
        {"v": 1, "stack": [], "lanes": []}, separators=(",", ":")
    ).encode("ascii"),
    # tcg#79 moved DOCUMENT_FORMAT 3 -> 4, so these two move with it: a case
    # named "ahead of this one" that is actually THIS build's format would still
    # go red (its stack is empty) while proving something else entirely.
    "a newer format from a build ahead of this one": json.dumps(
        {"v": 5, "stack": [], "lanes": []}, separators=(",", ":")
    ).encode("ascii"),
    "a field set this grammar does not have": json.dumps(
        {"v": 4, "stack": [], "lanes": [], "closure": "deadbeef"},
        separators=(",", ":"),
    ).encode("ascii"),
    "bytes that are not JSON at all": b"\x00\x01 not a document",
    "JSON that is not an object": b"[1,2,3]",
}


class TestAnUnreadableDocumentIsAMiss:
    @pytest.mark.parametrize("shape", sorted(UNREADABLE), ids=lambda s: s)
    def test_every_unreadable_shape_reads_as_absent_never_as_an_error(
        self, tmp_path: Path, shape: str
    ) -> None:
        cas, store = store_at(tmp_path)
        plant(cas, UNREADABLE[shape])
        assert store.get_graphs(NAME) is None

    @pytest.mark.parametrize("shape", sorted(UNREADABLE), ids=lambda s: s)
    def test_the_unreadable_position_is_emptied_not_left_to_fail_again(
        self, tmp_path: Path, shape: str
    ) -> None:
        cas, store = store_at(tmp_path)
        plant(cas, UNREADABLE[shape])
        store.get_graphs(NAME)
        assert cas.read_ref(REF) is None

    def test_a_tampered_document_is_absent_too_not_a_verification_error(
        self, tmp_path: Path
    ) -> None:
        """Digest mismatch is the same class of fact: bytes we cannot trust.

        It reached the same ``StoreError`` before, and it deserves the same
        answer -- a derived artifact whose bytes went bad is re-derivable.
        """

        cas, store = store_at(tmp_path)
        ref = plant(cas, a_readable_document().encode())
        cas.object_path(ref).write_bytes(b"corrupted, not its own digest")
        assert store.get_graphs(NAME) is None
        assert cas.read_ref(REF) is None

    def test_the_stranded_bytes_are_reclaimed_not_carried_forever(self, tmp_path: Path) -> None:
        cas, store = store_at(tmp_path)
        ref = plant(cas, a_v2_document())
        blob = cas.object_path(ref)
        assert blob.exists()
        store.get_graphs(NAME)
        assert not blob.exists()

    def test_the_second_call_is_an_ordinary_miss(self, tmp_path: Path) -> None:
        cas, store = store_at(tmp_path)
        plant(cas, a_v2_document())
        assert store.get_graphs(NAME) is None
        assert store.get_graphs(NAME) is None

    def test_a_remint_lands_on_the_emptied_position_and_reads_back(self, tmp_path: Path) -> None:
        """The whole point: absent means the ordinary refill path runs."""

        cas, store = store_at(tmp_path)
        plant(cas, a_v2_document())
        assert store.get_graphs(NAME) is None
        fresh = a_readable_document()
        store.put_graphs(NAME, fresh)
        assert store.get_graphs(NAME) == fresh


class TestTheDiscardIsAnnouncedOnce:
    def test_one_warning_names_the_graphset_the_bytes_and_the_cause(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cas, store = store_at(tmp_path)
        ref = plant(cas, a_v2_document())
        with caplog.at_level(logging.WARNING, logger="torchcg.store"):
            store.get_graphs(NAME)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert NAME in message
        assert ref in message
        assert "document v must be 4" in message

    def test_the_ordinary_miss_says_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _cas, store = store_at(tmp_path)
        with caplog.at_level(logging.WARNING, logger="torchcg.store"):
            assert store.get_graphs(NAME) is None
        assert not caplog.records


class TestTheDiscardIsNarrow:
    """The tests that fail a fix which just swallows everything."""

    def test_a_readable_document_is_returned_and_its_position_survives(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cas, store = store_at(tmp_path)
        document = a_readable_document()
        store.put_graphs(NAME, document)
        with caplog.at_level(logging.WARNING, logger="torchcg.store"):
            assert store.get_graphs(NAME) == document
        assert cas.read_ref(REF) is not None
        assert not caplog.records

    def test_a_sibling_graphset_is_untouched_by_a_discard(self, tmp_path: Path) -> None:
        cas, store = store_at(tmp_path)
        sibling = a_readable_document()
        store.put_graphs("sdxl.main", sibling)
        plant(cas, a_v2_document())
        store.get_graphs(NAME)
        assert store.get_graphs("sdxl.main") == sibling

    def test_the_discard_spares_bytes_a_concurrent_mint_has_not_yet_named(
        self, tmp_path: Path
    ) -> None:
        """A discard reclaims ONE object, never everything unreferenced.

        A mint installs its object BEFORE the logical ref that makes it
        reachable. A mark-and-sweep run inside that gap deletes a live
        artifact, which is why the discard is targeted at the bytes it just
        failed to read and at nothing else.
        """

        cas, store = store_at(tmp_path)
        in_flight = cas.put_bytes(b"an artifact a peer just minted")
        plant(cas, a_v2_document())
        store.get_graphs(NAME)
        assert cas.object_path(in_flight).exists()

    def test_shared_bytes_survive_because_another_position_still_names_them(
        self, tmp_path: Path
    ) -> None:
        """Two positions holding identical bytes share ONE object.

        Deleting a position's object without checking the other positions
        would punch a hole under a live one -- which is the failure mode a
        content-addressed store makes possible and a path-addressed one does
        not.
        """

        cas, store = store_at(tmp_path)
        stale = a_v2_document()
        ref = plant(cas, stale)
        cas.compare_and_swap_ref("torchcg/v2/graphsets/sdxl.main", ref, expected=None)
        store.get_graphs(NAME)
        assert cas.read_ref(REF) is None
        assert cas.object_path(ref).exists()


class TestTheErrorBranchSurvives:
    """Widening the MISS branch must not delete the branch that reports lies."""

    def test_a_divergent_artifact_publish_still_refuses(self, tmp_path: Path) -> None:
        cas, store = store_at(tmp_path)
        graph = "cg-graph-v1-" + "b" * 56
        env = EnvIdentity(stack=STACK, sm="sm_89")
        manifest = RequirementsManifest(
            include_set=(("torch", ">=2.13.0,<3"),),
            sm_compiled="sm_89",
            cuda_floor=">=13.0",
            autotuned_on=None,
        )
        first = tmp_path / "first.so"
        first.write_bytes(b"the artifact one mint produced")
        store.publish_artifact(graph, env, first, manifest)
        second = tmp_path / "second.so"
        second.write_bytes(b"different bytes at the same position")
        with pytest.raises(StoreError, match="diverged"):
            store.publish_artifact(graph, env, second, manifest)
        del cas

    def test_an_unsafe_graphset_name_still_refuses(self, tmp_path: Path) -> None:
        _cas, store = store_at(tmp_path)
        with pytest.raises(StoreError, match="unsafe graph-set name"):
            store.get_graphs("../escape")
