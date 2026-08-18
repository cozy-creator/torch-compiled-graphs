"""Adopt-first swap-in over a real discovered document (pgw#1372's mechanism).

Everything here runs the real lifecycle on CPU: real ``discover_lane`` output
(real graph hashes, real ingress), a real ``LocalGraphStore`` over a tensorfs
CAS, and real module-forward swaps on live ``torch.nn`` modules. Only the
artifact LOADER is a stub -- turning bytes into a callable is the AOTInductor
runtime's job and needs the target GPU; the seam is the point.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensorfs import LocalCAS

from torchcg.adopt import HOLE_MISS, AdoptError, adopt_lane
from torchcg.discovery import discover_lane
from torchcg.document import GraphSetDocument
from torchcg.graph_identity import EnvIdentity, closure_hash
from torchcg.lane import Lane
from torchcg.requirements import EnvironmentMismatch, RequirementsManifest
from torchcg.store import LocalGraphStore, StoreError

SM = "sm_89"
INSTALLED = {"torch": torch.__version__, "example-lib": "1.0.0"}
CLOSURE = closure_hash(INSTALLED)


class TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, sample: torch.Tensor, doubled: bool = False) -> torch.Tensor:
        out = self.proj(sample)
        return out * 2 if doubled else out


@pytest.fixture()
def pipe() -> SimpleNamespace:
    torch.manual_seed(0)
    return SimpleNamespace(unet=TinyUnet())


LANE = Lane("bf16", compile=("pipe.unet",), contract="plain.fp32@1", dtype=torch.float32)


def discover(pipe: SimpleNamespace, *, flags: tuple[bool, ...] = (False,)) -> GraphSetDocument:
    """One real discovery pass: each (shape, flag) sample is a graph class."""

    def drive() -> None:
        for flag in flags:
            pipe.unet(torch.zeros(1, 4), doubled=flag)
            pipe.unet(torch.zeros(2, 4), doubled=flag)

    lane_graphs = discover_lane(LANE, {"pipe": pipe}, drive)
    return GraphSetDocument(closure=CLOSURE, lanes=(lane_graphs,))


def manifest() -> RequirementsManifest:
    return RequirementsManifest(include_set=(("torch", torch.__version__),), sm_compiled=SM)


@pytest.fixture()
def store(tmp_path: Path) -> LocalGraphStore:
    return LocalGraphStore(LocalCAS(tmp_path / "cas"))


def publish(store: LocalGraphStore, tmp_path: Path, graph: str, payload: bytes) -> Path:
    artifact = tmp_path / f"mint-{graph[-8:]}.so"
    artifact.write_bytes(payload)
    store.publish_artifact(graph, EnvIdentity(closure=CLOSURE, sm=SM), artifact, manifest())
    return artifact


def stub_loader(markers: dict[str, torch.Tensor]):
    """A loader whose 'compiled graph' returns a per-graph sentinel tensor."""

    def load(path: Path, record):
        sentinel = markers.setdefault(record.graph, torch.full((1,), float(len(markers) + 1)))

        def compiled(*args, **kwargs):
            return sentinel

        return compiled

    return load


def test_partial_hit_arms_the_stored_graph_and_states_the_holes(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    lane = document.lanes[0]
    assert len(lane.graphs) == 2  # two shapes -> two graph classes
    hit, hole = lane.graphs[0], lane.graphs[1]
    publish(store, tmp_path, hit.graph, b"compiled-bytes")

    markers: dict[str, torch.Tensor] = {}
    adoption = adopt_lane(
        store, document, "bf16", {"pipe": pipe}, SM,
        loader=stub_loader(markers), artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )

    # The armed graph serves through the swap; the module is the one CALLED,
    # so the author's ordinary `pipe.unet(...)` call routes through it.
    hit_shape = tuple(
        d for d in hit.ingress.inputs[0].shape if isinstance(d, int)
    )
    out = pipe.unet(torch.zeros(hit_shape))
    assert torch.equal(out, markers[hit.graph])

    # The hole's shape stays the author's own eager forward.
    hole_shape = tuple(d for d in hole.ingress.inputs[0].shape if isinstance(d, int))
    eager = pipe.unet(torch.zeros(hole_shape))
    assert eager.shape == torch.Size(hole_shape)
    assert not any(torch.equal(eager, m) for m in markers.values())

    # The ordered hole list is THE mint handoff: canonical document order,
    # full GraphRecord per row, reason stated.
    assert [h.record.graph for h in adoption.holes] == [hole.graph]
    assert adoption.holes[0].reason == HOLE_MISS
    assert adoption.holes[0].record.ingress == hole.ingress
    assert adoption.adopted == (hit,)


def test_arming_a_late_mint_flips_dispatch_without_a_reboot(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    lane = document.lanes[0]
    first, second = lane.graphs
    publish(store, tmp_path, first.graph, b"first")

    markers: dict[str, torch.Tensor] = {}
    adoption = adopt_lane(
        store, document, "bf16", {"pipe": pipe}, SM,
        loader=stub_loader(markers), artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )
    assert [h.record.graph for h in adoption.holes] == [second.graph]

    # The background mint lands the hole and arms it -- per graph, live.
    minted = publish(store, tmp_path, second.graph, b"second")
    adoption.arm(second, minted)

    second_shape = tuple(d for d in second.ingress.inputs[0].shape if isinstance(d, int))
    assert torch.equal(pipe.unet(torch.zeros(second_shape)), markers[second.graph])
    assert adoption.holes == ()
    assert set(adoption.adopted) == {first, second}


def test_exact_env_mismatch_refuses_loudly_before_touching_the_modules(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    with pytest.raises(EnvironmentMismatch) as excinfo:
        adopt_lane(
            store, document, "bf16", {"pipe": pipe}, SM,
            loader=stub_loader({}), artifacts_dir=tmp_path / "adopted",
            installed={**INSTALLED, "torch": "0.0.0-divergent"},
        )
    assert "build-system bug" in str(excinfo.value)
    # Refused BEFORE any swap: no instance forward was installed.
    assert "forward" not in pipe.unet.__dict__


def test_an_unreadable_store_row_is_a_hole_not_a_boot_failure(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    first, second = document.lanes[0].graphs
    publish(store, tmp_path, first.graph, b"good")

    class OneBadRow:
        def __getattr__(self, name):
            return getattr(store, name)

        def fetch_artifact(self, graph, env, destination):
            if graph == second.graph:
                raise StoreError("synthetic corruption")
            return store.fetch_artifact(graph, env, destination)

    markers: dict[str, torch.Tensor] = {}
    adoption = adopt_lane(
        OneBadRow(), document, "bf16", {"pipe": pipe}, SM,
        loader=stub_loader(markers), artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )
    assert adoption.adopted == (first,)
    assert [h.record.graph for h in adoption.holes] == [second.graph]
    assert adoption.holes[0].reason.startswith("store_error:")


def test_missing_lane_and_eager_permanent_documents_refuse_typed(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    document = discover(pipe)
    with pytest.raises(AdoptError, match="no lane 'fp8'"):
        adopt_lane(
            store, document, "fp8", {"pipe": pipe}, SM,
            loader=stub_loader({}), artifacts_dir=tmp_path / "a", installed=INSTALLED,
        )
    eager = GraphSetDocument(closure=CLOSURE, lanes=())
    with pytest.raises(AdoptError, match="eager-permanent"):
        adopt_lane(
            store, eager, "bf16", {"pipe": pipe}, SM,
            loader=stub_loader({}), artifacts_dir=tmp_path / "a", installed=INSTALLED,
        )


def test_two_graphs_with_one_tensor_structure_disarm_each_other(
    pipe: SimpleNamespace, store: LocalGraphStore, tmp_path: Path
) -> None:
    # Same tensor feeds, different baked-in literal (`doubled`): two real
    # graph classes the dispatcher cannot tell apart at call time.
    document = discover(pipe, flags=(False, True))
    lane = document.lanes[0]
    assert len(lane.graphs) == 4
    for record in lane.graphs:
        publish(store, tmp_path, record.graph, record.graph.encode())

    markers: dict[str, torch.Tensor] = {}
    adoption = adopt_lane(
        store, document, "bf16", {"pipe": pipe}, SM,
        loader=stub_loader(markers), artifacts_dir=tmp_path / "adopted",
        installed=INSTALLED,
    )
    # Every structure collides with its literal-twin: nothing stays armed,
    # every pair is reported, and the calls all serve the author's eager path.
    assert adoption.adopted == ()
    assert len(adoption.ambiguous) == 4
    assert adoption.holes == ()
    out = pipe.unet(torch.zeros(1, 4), doubled=True)
    assert out.shape == torch.Size((1, 4))
    assert not any(torch.equal(out, m) for m in markers.values())
