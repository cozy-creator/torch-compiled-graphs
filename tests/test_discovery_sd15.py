"""tcg#41 acceptance: author code as-is, one lane, discovered, reproducible.

A stock diffusers ``StableDiffusionPipeline`` (tiny config, generated weights,
nothing downloaded) is declared with one execution lane. The instrumented run
discovers the unet + vae-decoder graph set; the resulting document is
byte-identical across two in-process runs and across a subprocess fence, and
the lifecycle continues into the store: publish one artifact, and only the
holes remain.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

import sd15_tiny  # noqa: E402
from tensorfs import LocalCAS  # noqa: E402

from torchcg.discovery import discover_lane  # noqa: E402
from torchcg.document import GraphSetDocument  # noqa: E402
from torchcg.graph_identity import EnvIdentity, closure_hash  # noqa: E402
from torchcg.requirements import RequirementsManifest  # noqa: E402
from torchcg.store import LocalGraphStore, PublishOutcome, holes  # noqa: E402


@pytest.fixture(scope="module")
def document() -> GraphSetDocument:
    return sd15_tiny.discover_document()


def test_one_lane_discovers_the_observed_graph_set(document: GraphSetDocument) -> None:
    (lane,) = document.lanes
    assert lane.contract == sd15_tiny.LANE_CONTRACT
    assert lane.unobserved_targets == ()
    by_target = {record.target for record in lane.graphs}
    assert by_target == {"unet", "vae.decoder"}
    # Two denoising steps at one shape dedup to ONE unet graph class.
    assert len(lane.graphs) == 2
    for record in lane.graphs:
        assert record.graph.startswith("cg-graph-v1-")
        assert record.ingress.inputs  # a real observed tensor contract


def test_document_is_byte_reproducible_in_process(document: GraphSetDocument) -> None:
    again = sd15_tiny.discover_document()
    assert again.encode() == document.encode()


def test_document_is_stable_across_processes(document: GraphSetDocument) -> None:
    """The subprocess fence: a fresh interpreter derives the same bytes."""

    script = Path(__file__).resolve().parent / "sd15_tiny.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == document.digest()


def test_a_second_lane_stamps_a_distinct_graph_set() -> None:
    pipe = sd15_tiny.build_pipe()
    lane = discover_lane(
        "sd15.decoder-only-fp32@1",
        ("vae.decoder",),
        pipe.components,
        lambda: sd15_tiny.drive(pipe),
    )
    assert {record.target for record in lane.graphs} == {"vae.decoder"}


def test_an_undriven_target_is_stated_unobserved() -> None:
    pipe = sd15_tiny.build_pipe()
    lane = discover_lane(
        sd15_tiny.LANE_CONTRACT,
        ("unet", "vae.encoder"),
        pipe.components,
        lambda: sd15_tiny.drive(pipe),  # text-to-image never encodes
    )
    assert lane.unobserved_targets == ("vae.encoder",)
    assert {record.target for record in lane.graphs} == {"unet"}


def test_lifecycle_continues_into_the_store(
    document: GraphSetDocument, tmp_path: Path
) -> None:
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    store.put_graphs("release-1", document)
    fetched = store.get_graphs("release-1")
    assert fetched is not None and fetched.encode() == document.encode()

    (lane,) = fetched.lanes
    env = EnvIdentity(closure=fetched.closure, sm="sm_89")
    assert holes(store, lane, env) == tuple(record.graph for record in lane.graphs)

    minted = tmp_path / "minted.tar.gz"
    minted.write_bytes(b"stand-in artifact bytes")
    manifest = RequirementsManifest(
        include_set=(("torch", "0.0.0"),), sm_compiled="sm_89"
    )
    first = lane.graphs[0].graph
    assert (
        store.publish_artifact(first, env, minted, manifest)
        is PublishOutcome.PUBLISHED
    )
    # Partial-hit: exactly the unminted graphs remain holes.
    assert holes(store, lane, env) == tuple(
        record.graph for record in lane.graphs[1:]
    )


def test_closure_hash_matches_between_lock_statement_and_installed() -> None:
    """Same entries, any spelling of the names: one env hash."""

    entries = {"Foo_Bar": "1.0", "qux": "2.0"}
    assert closure_hash(entries) == closure_hash(
        {"foo-bar": "1.0", "qux": "2.0"}
    )
