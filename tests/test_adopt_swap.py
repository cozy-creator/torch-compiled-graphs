"""ctx.compile's engine: AdoptSession over a real discovered document (pgw#1372).

Everything here runs the real lifecycle on CPU: real ``discover_lane`` output
(real graph hashes, real ingress), a real ``LocalGraphStore`` over a tensorfs
CAS, and real module-forward swaps on live ``torch.nn`` modules driven the
imperative way (``module = session.adopt(module)`` — the serve half of
``ctx.compile``). Only the artifact LOADER is a stub -- turning bytes into a
callable is the AOTInductor runtime's job and needs the target GPU; the seam
is the point.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensorfs import LocalCAS

from torchcg.adopt import HOLE_MISS, AdoptError, AdoptSession
from torchcg.discovery import discover_lane
from torchcg.document import GraphRecord, GraphSetDocument
from torchcg.graph_identity import EnvIdentity
from torchcg.requirements import EnvironmentMismatch, RequirementsManifest
from torchcg.store import LocalGraphStore, PublishOutcome, StoreError

SM = "sm_89"
STACK: tuple[tuple[str, str], ...] = (("torch", torch.__version__),)


class TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, sample: torch.Tensor, doubled: bool = False) -> torch.Tensor:
        out: torch.Tensor = self.proj(sample)
        return out * 2 if doubled else out


class TinyPipe:
    """A pipeline-shaped container: ``ctx.compile(pipe)`` walks components."""

    def __init__(self) -> None:
        self.unet = TinyUnet()

    @property
    def components(self) -> dict[str, object]:
        return {"unet": self.unet}


@pytest.fixture()
def pipe() -> SimpleNamespace:
    torch.manual_seed(0)
    return SimpleNamespace(unet=TinyUnet())


CONTRACT = "tiny.plain-fp32@1"


def discover(pipe: SimpleNamespace, *, flags: tuple[bool, ...] = (False,)) -> GraphSetDocument:
    """One real derive pass: each (shape, flag) sample is a graph specialization."""

    def drive() -> None:
        for flag in flags:
            pipe.unet(torch.zeros(1, 4), doubled=flag)
            pipe.unet(torch.zeros(2, 4), doubled=flag)

    lane_graphs = discover_lane(CONTRACT, ("pipe.unet",), {"pipe": pipe}, drive)
    return GraphSetDocument(stack=STACK, lanes=(lane_graphs,))


def manifest() -> RequirementsManifest:
    return RequirementsManifest(include_set=(("torch", torch.__version__),), sm_compiled=SM)


@pytest.fixture()
def store(tmp_path: Path) -> LocalGraphStore:
    return LocalGraphStore(LocalCAS(tmp_path / "cas"))


def publish(store: LocalGraphStore, tmp_path: Path, graph: str, payload: bytes) -> Path:
    artifact = tmp_path / f"mint-{graph[-8:]}.so"
    artifact.write_bytes(payload)
    store.publish_artifact(graph, EnvIdentity(stack=STACK, sm=SM), artifact, manifest())
    return artifact


def stub_loader(
    markers: dict[str, torch.Tensor],
) -> Callable[[Path, GraphRecord, object], Callable[..., torch.Tensor]]:
    """A loader whose 'compiled graph' returns a per-graph sentinel tensor.

    Three arguments since tcg#58: the third is the LIVE MODULE the compiled
    graph binds its constants from. This stub asserts it is the module that
    claimed the record rather than ignoring it -- the production loader cannot
    work without it, so a double that drops it models the wrong thing.
    """

    def load(
        path: Path, record: GraphRecord, module: object
    ) -> Callable[..., torch.Tensor]:
        assert isinstance(module, torch.nn.Module)
        sentinel = markers.setdefault(record.graph, torch.full((1,), float(len(markers) + 1)))

        def compiled(*args: object, **kwargs: object) -> torch.Tensor:
            return sentinel

        return compiled

    return load


def session_for(
    store: LocalGraphStore | None,
    document: GraphSetDocument,
    markers: dict[str, torch.Tensor],
    tmp_path: Path,
    stack: tuple[tuple[str, str], ...] | None = None,
) -> AdoptSession:
    return AdoptSession(
        store, document, CONTRACT, SM,
        loader=stub_loader(markers), artifacts_dir=tmp_path / "adopted",
        stack=stack if stack is not None else STACK,
    )


def test_partial_hit_arms_the_marked_module_and_states_the_holes(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    lane = document.lanes[0]
    assert len(lane.graphs) == 2  # two shapes -> two graph specializations
    hit, hole = lane.graphs[0], lane.graphs[1]
    publish(store, tmp_path, hit.graph, b"compiled-bytes")

    markers: dict[str, torch.Tensor] = {}
    session = session_for(store, document, markers, tmp_path)
    # The torch.compile idiom, verbatim: the module comes BACK, armed.
    pipe.unet = session.adopt(pipe.unet)

    # The armed graph serves through the swap on the module the author marked.
    hit_shape = tuple(d for d in hit.ingress.inputs[0].shape if isinstance(d, int))
    out = pipe.unet(torch.zeros(hit_shape))
    assert torch.equal(out, markers[hit.graph])

    # The hole's shape stays the author's own eager forward.
    hole_shape = tuple(d for d in hole.ingress.inputs[0].shape if isinstance(d, int))
    eager = pipe.unet(torch.zeros(hole_shape))
    assert eager.shape == torch.Size(hole_shape)
    assert not any(torch.equal(eager, m) for m in markers.values())

    # THE mint handoff: ordered holes carrying full GraphRecords, reasons stated.
    assert [h.record.graph for h in session.holes] == [hole.graph]
    assert session.holes[0].reason == HOLE_MISS
    assert session.holes[0].record.ingress is hole.ingress
    assert session.adopted == (hit,)
    assert session.unclaimed == ()


def test_container_walk_adopts_every_component(
    store: LocalGraphStore, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    container = TinyPipe()
    document = discover(SimpleNamespace(unet=container.unet))
    for record in document.lanes[0].graphs:
        publish(store, tmp_path, record.graph, record.graph.encode())

    markers: dict[str, torch.Tensor] = {}
    session = session_for(store, document, markers, tmp_path)
    back = session.adopt(container)  # ctx.compile(pipe): walks components
    assert back is container
    assert len(session.adopted) == 2
    out = container.unet(torch.zeros(1, 4))
    assert any(torch.equal(out, m) for m in markers.values())
    with pytest.raises(AdoptError, match="nn.Module or a pipeline"):
        session.adopt(object())


def test_arming_a_late_mint_flips_dispatch_without_a_reboot(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    first, second = document.lanes[0].graphs
    publish(store, tmp_path, first.graph, b"first")

    markers: dict[str, torch.Tensor] = {}
    session = session_for(store, document, markers, tmp_path)
    pipe.unet = session.adopt(pipe.unet)
    assert [h.record.graph for h in session.holes] == [second.graph]

    # The background mint lands the hole and arms it -- per graph, live.
    minted = publish(store, tmp_path, second.graph, b"second")
    session.arm(second, minted)

    second_shape = tuple(d for d in second.ingress.inputs[0].shape if isinstance(d, int))
    assert torch.equal(pipe.unet(torch.zeros(second_shape)), markers[second.graph])
    assert session.holes == ()
    assert set(session.adopted) == {first, second}


def test_a_store_less_session_states_every_claimed_graph_as_mint_work(
    pipe: SimpleNamespace, tmp_path: Path
) -> None:
    document = discover(pipe)
    session = session_for(None, document, {}, tmp_path)
    pipe.unet = session.adopt(pipe.unet)
    assert [h.reason for h in session.holes] == [HOLE_MISS, HOLE_MISS]
    assert session.adopted == ()
    # Eager serves regardless — the bridge is unconditional.
    assert pipe.unet(torch.zeros(1, 4)).shape == torch.Size((1, 4))


def test_exact_env_mismatch_refuses_loudly_at_session_construction(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    with pytest.raises(EnvironmentMismatch) as excinfo:
        session_for(store, document, {}, tmp_path,
                    stack=(("torch", "0.0.0-divergent"),))
    assert "torch 0.0.0-divergent != stamped" in str(excinfo.value)
    # Refused BEFORE any adopt call could run: no instance forward installed.
    assert "forward" not in pipe.unet.__dict__


def test_an_unreadable_store_row_is_a_hole_not_a_boot_failure(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    first, second = document.lanes[0].graphs
    publish(store, tmp_path, first.graph, b"good")

    class OneBadRow:
        """The real store with exactly one unreadable artifact row."""

        def get_graphs(self, name: str) -> GraphSetDocument | None:
            return store.get_graphs(name)

        def put_graphs(self, name: str, document: GraphSetDocument) -> None:
            store.put_graphs(name, document)

        def has_artifact(self, graph: str, env: EnvIdentity) -> bool:
            return store.has_artifact(graph, env)

        def fetch_artifact(
            self, graph: str, env: EnvIdentity, destination: str | Path
        ) -> Path | None:
            if graph == second.graph:
                raise StoreError("synthetic corruption")
            return store.fetch_artifact(graph, env, destination)

        def publish_artifact(
            self,
            graph: str,
            env: EnvIdentity,
            artifact: str | Path,
            manifest: RequirementsManifest,
        ) -> PublishOutcome:
            return store.publish_artifact(graph, env, artifact, manifest)

        def get_manifest(
            self, graph: str, env: EnvIdentity
        ) -> RequirementsManifest | None:
            return store.get_manifest(graph, env)

    markers: dict[str, torch.Tensor] = {}
    session = AdoptSession(
        OneBadRow(), document, CONTRACT, SM,
        loader=stub_loader(markers), artifacts_dir=tmp_path / "adopted",
        stack=STACK,
    )
    pipe.unet = session.adopt(pipe.unet)
    assert session.adopted == (first,)
    assert [h.record.graph for h in session.holes] == [second.graph]
    assert session.holes[0].reason.startswith("store_error:")


def test_missing_lane_and_eager_permanent_documents_refuse_typed(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    with pytest.raises(AdoptError, match="no lane 'other.fp8@1'"):
        AdoptSession(store, document, "other.fp8@1", SM, loader=stub_loader({}),
                     artifacts_dir=tmp_path / "a", stack=STACK)
    eager = GraphSetDocument(stack=STACK, lanes=())
    with pytest.raises(AdoptError, match="eager-permanent"):
        AdoptSession(store, eager, CONTRACT, SM, loader=stub_loader({}),
                     artifacts_dir=tmp_path / "a", stack=STACK)


def test_two_graphs_with_one_tensor_structure_disarm_each_other(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    # Same tensor feeds, different baked-in literal (`doubled`): two real
    # graph specializations the dispatcher cannot tell apart at call time.
    document = discover(pipe, flags=(False, True))
    lane = document.lanes[0]
    assert len(lane.graphs) == 4
    for record in lane.graphs:
        publish(store, tmp_path, record.graph, record.graph.encode())

    markers: dict[str, torch.Tensor] = {}
    session = session_for(store, document, markers, tmp_path)
    pipe.unet = session.adopt(pipe.unet)
    # Every structure collides with its literal-twin: nothing stays armed,
    # every pair is reported, and the calls all serve the author's eager path.
    assert session.adopted == ()
    assert len(session.ambiguous) == 4
    assert session.holes == ()
    out = pipe.unet(torch.zeros(1, 4), doubled=True)
    assert out.shape == torch.Size((1, 4))
    assert not any(torch.equal(out, m) for m in markers.values())


def test_a_record_no_marked_module_claims_is_stated_not_dropped(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)

    class Stranger(torch.nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

    session = session_for(store, document, {}, tmp_path)
    session.adopt(Stranger())  # signature admits none of the records
    assert len(session.unclaimed) == 2
    assert session.holes == ()  # unclaimed is not mint work: nothing to arm onto
    with pytest.raises(AdoptError, match="never claimed"):
        session.arm(document.lanes[0].graphs[0], tmp_path / "x.so")


def test_a_marked_module_that_matches_nothing_is_recorded_with_the_gap(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    """tcg#70: the MODULE side of an unclaimed record, and the reason.

    `unclaimed` says which graphs found no home. Without the other half — which
    MARKS found no graph — a boot cannot tell "the author stopped marking a
    module" from "the module the author marks no longer has the forward the
    derive traced", and both print as adopted=0, holes=0.
    """
    document = discover(pipe)

    class Stranger(torch.nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

    session = session_for(store, document, {}, tmp_path)
    session.adopt(Stranger())

    marks = session.unclaimed_marks
    assert len(marks) == 1
    assert marks[0].module == "Stranger"
    assert marks[0].parameters == ("latents",)
    # The gap is the diagnosis: the nearest record wants `sample`, which this
    # module's forward does not accept.
    assert "sample" in marks[0].missing
    assert marks[0].nearest in {record.graph for record in document.lanes[0].graphs}
    assert "matched NO graph" in marks[0].describe()
    assert "sample" in marks[0].describe()


def test_nothing_armed_nothing_to_mint_and_a_mark_that_matched_is_not_silence(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    """`adopted=0, holes=0` has TWO causes and they must be distinguishable.

    Nothing-to-do is a quiet success. A mark that matched nothing over a lane
    full of records is a compiled path switched off, and it printed identically.
    """
    document = discover(pipe)

    class Stranger(torch.nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

    # (a) nobody marked anything: two zeros, and that IS nothing to do.
    quiet = session_for(store, document, {}, tmp_path)
    assert quiet.adopted == () and quiet.holes == ()
    assert quiet.silently_eager() is False

    # (b) a mark landed and fit nothing: the same two zeros, a different fact.
    loud = session_for(store, document, {}, tmp_path)
    loud.adopt(Stranger())
    assert loud.adopted == () and loud.holes == ()
    assert loud.silently_eager() is True


def test_a_mark_that_claims_its_records_is_not_reported_as_unclaimed(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    """The green arm: the instrument must not fire on a working adoption."""
    document = discover(pipe)
    session = session_for(store, document, {}, tmp_path)
    session.adopt(pipe.unet)

    assert session.unclaimed_marks == ()
    assert session.silently_eager() is False
    assert session.holes != (), "a store-less lane still forms mint work"
