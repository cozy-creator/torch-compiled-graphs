"""The transform-pass mechanism, driven end to end on CPU (tcg#52).

Everything here is the real lifecycle on generated fake weights: a real
``TransformSession``, a real ``PrecomputeAndFree`` evaluating a real
submodule over a real domain, real safetensors buckets through a real
``LocalGraphStore`` over a tensorfs CAS, real ``torch.export`` discovery, and
a real ``AdoptSession``. Nothing here is mocked -- the tiny model IS the
production code path with a 32-wide checkpoint instead of a 66 GB one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from tensorfs import LocalCAS
from torch import nn

from torchcg.adopt import AdoptError, AdoptSession
from torchcg.discovery import DiscoveryError, discover_modules
from torchcg.document import GraphSetDocument
from torchcg.graph_identity import closure_hash, graph_hash
from torchcg.ingress import build_call_ingress
from torchcg.lane import LaneError, LaneRef
from torchcg.store import LocalGraphStore, StoreError
from torchcg.transform import (
    OffDomain,
    PrecomputeAndFree,
    TransformError,
    TransformMode,
    TransformOrderError,
    TransformSession,
    canonical_key,
    handle,
    installed_passes,
    module_bytes,
    registered_passes,
    resolve_pass,
    side_table_class,
)

WIDTH = 32
BLOCKS = 3
PRESETS: tuple[int, ...] = (4, 6)
PASS = PrecomputeAndFree.NAME
FOLDED_LANE = "tiny.folded-fp32@1"
PLAIN_LANE = "tiny.plain-fp32@1"
SM = "sm_89"
INSTALLED = {"torch": torch.__version__, "example-lib": "1.0.0"}
CLOSURE = closure_hash(INSTALLED)


# -- the author's model ----------------------------------------------------


class Modulation(nn.Module):
    """The foldable submodule: a pure function of the domain point."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 3 * width)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.proj(temb).chunk(3, dim=-1))


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.adaln = Modulation(width)
        self.linear = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaln(temb)
        modulated: torch.Tensor = self.linear(hidden * (1 + scale) + shift)
        scaled: torch.Tensor = modulated * gate
        return scaled


class TinyDiT(nn.Module):
    """A miniature of the shape adaln_skip.py was written for."""

    def __init__(self, width: int = WIDTH, blocks: int = BLOCKS) -> None:
        super().__init__()
        self.time_embedder = nn.Linear(1, width)
        self.blocks = nn.ModuleList(Block(width) for _ in range(blocks))

    def forward(self, hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        temb = self.time_embedder(timestep)
        for block in self.blocks:
            hidden = block(hidden, temb)
        return hidden


def schedule(steps: int) -> torch.Tensor:
    return torch.linspace(1.0, 0.1, steps, dtype=torch.float32).reshape(-1, 1)


def hidden_for(steps: int, width: int = WIDTH) -> torch.Tensor:
    return torch.full((steps, width), 0.5, dtype=torch.float32)


def domain_key(point: Any) -> tuple[str, int]:
    """The endpoint's half: what a domain point IS."""

    return ("steps", int(point))


def temb_for(root: Any, module: Any, point: Any) -> torch.Tensor:
    """The endpoint's half: how to CALL the folded submodule."""

    embedded: torch.Tensor = root.time_embedder(schedule(int(point)))
    return embedded


def make_model(*, blocks: int = BLOCKS, seed: int = 0) -> TinyDiT:
    torch.manual_seed(seed)
    return TinyDiT(blocks=blocks)


def make_pass(
    *,
    domain: tuple[int, ...] = PRESETS,
    select: str = "blocks.*.adaln",
    call: Any = temb_for,
) -> PrecomputeAndFree:
    return PrecomputeAndFree(
        select=select,
        domain=domain,
        key=domain_key,
        call=call,
        removes="adaln_projections",
    )


def folded_lane(passes: tuple[str, ...] = (PASS,)) -> LaneRef:
    return LaneRef(FOLDED_LANE, passes=passes)


@pytest.fixture()
def store(tmp_path: Path) -> LocalGraphStore:
    return LocalGraphStore(LocalCAS(tmp_path / "cas"))


# -- the first instance: compute, free, measure ----------------------------


def test_precompute_and_free_matches_the_untransformed_forward() -> None:
    """The whole claim in one case: same answer, fewer bytes, measured."""

    model = make_model()
    reference = {
        steps: model(hidden_for(steps), schedule(steps)).detach().clone()
        for steps in PRESETS
    }
    adaln_bytes = sum(module_bytes(block.adaln) for block in model.blocks)
    assert adaln_bytes > 0

    session = TransformSession(folded_lane())
    report = session.run(model, make_pass())

    assert report.pass_name == "precompute-and-free@1"
    assert report.mode is TransformMode.COMPUTED
    assert report.plan.removes == ("adaln_projections",)
    assert report.plan.rewrites == tuple(
        f"blocks.{index}.adaln" for index in range(BLOCKS)
    )
    assert report.plan.domain == tuple(canonical_key(domain_key(p)) for p in PRESETS)
    assert report.plan.scope_bytes == adaln_bytes
    assert report.freed_bytes == adaln_bytes
    # cache_bytes is MEASURED off the installed side tables, not projected.
    assert report.cache_bytes == sum(
        module_bytes(block.adaln) for block in model.blocks
    )
    assert report.net_bytes == report.freed_bytes - report.cache_bytes
    assert report.net_bytes > 0
    # Inherited honesty: no numerical-equivalence gate on real weights has run.
    assert report.validated is False

    # The projection weights are GONE: nothing under an adaln path is a Linear.
    assert not [
        name
        for name, module in model.named_modules()
        if ".adaln." in f"{name}." and isinstance(module, nn.Linear)
    ]
    assert installed_passes(model) == (PASS,)

    bound = handle(model, PASS)
    for steps in PRESETS:
        bound.bind(steps)
        torch.testing.assert_close(
            model(hidden_for(steps), schedule(steps)), reference[steps]
        )

    print(
        f"\n[precompute-and-free] blocks={BLOCKS} domain={len(PRESETS)} "
        f"freed={report.freed_bytes}B cache={report.cache_bytes}B "
        f"net={report.net_bytes}B mode={report.mode} validated={report.validated}"
    )


def test_the_side_table_moves_with_the_module() -> None:
    """ie#615's device-migration bug, pinned -- and its cause, demonstrated.

    The endpoint's plain-dict table was invisible to ``nn.Module._apply``, so
    ``transformer.to("cuda")`` moved every parameter and left the cache on the
    host. Registering the table as non-persistent buffers makes it module
    state, so it moves by construction. The control below is the plain-dict
    shape: it does NOT move, which is what makes this test's pass a positive
    observation rather than a tautology.
    """

    model = make_model()
    TransformSession(folded_lane()).run(model, make_pass())

    def cached() -> list[torch.Tensor]:
        return [
            tensor
            for name, tensor in model.named_buffers()
            if ".adaln._st_" in f".{name}"
        ]

    assert len(cached()) == BLOCKS * len(PRESETS) * 3

    model.to(torch.float64)
    assert {tensor.dtype for tensor in cached()} == {torch.float64}

    model.to(device="meta")
    assert {tensor.device.type for tensor in cached()} == {"meta"}
    assert {tensor.device.type for tensor in model.parameters()} == {"meta"}

    # The side table is DERIVED state: it must not enter state_dict() and so
    # must not perturb the lane contract's weight names.
    assert not [key for key in model.state_dict() if "_st_" in key]

    # CONTROL -- the dict-held table the bug shipped with.
    class DictTable(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.table = {"k": (torch.zeros(2, 2),)}

    control = DictTable()
    control.to(torch.float64)
    assert control.anchor.dtype is torch.float64  # the parameter moved
    assert control.table["k"][0].dtype is torch.float32  # the table did NOT


def test_off_domain_and_unbound_are_typed_refusals() -> None:
    model = make_model()
    TransformSession(folded_lane()).run(model, make_pass())
    bound = handle(model, PASS)

    with pytest.raises(OffDomain) as refusal:
        bound.bind(99)
    assert "cannot be recomputed at request time" in str(refusal.value)
    assert "blocks.0.adaln" in str(refusal.value)

    unbound = side_table_class()(
        name="probe", pass_name=PASS, table={"k": (torch.zeros(2),)}, unwrap=False
    )
    with pytest.raises(OffDomain) as never_bound:
        unbound()
    assert "no domain point bound" in str(never_bound.value)


def test_a_plan_that_frees_nothing_is_refused() -> None:
    model = make_model()
    lane = folded_lane()

    with pytest.raises(TransformError) as empty_select:
        make_pass(select="blocks.*.nope").plan(model, lane=lane)
    assert "matched no submodule" in str(empty_select.value)
    assert "the module shape moved" in str(empty_select.value)

    with pytest.raises(TransformError) as empty_domain:
        make_pass(domain=()).plan(model, lane=lane)
    assert "empty domain refuses every request" in str(empty_domain.value)

    with pytest.raises(TransformError) as collision:
        make_pass(domain=(4, 4)).plan(model, lane=lane)
    assert "address the same bucket" in str(collision.value)


def test_registration_is_how_a_lane_names_a_pass() -> None:
    assert PASS in registered_passes()
    assert resolve_pass(PASS) is PrecomputeAndFree

    with pytest.raises(TransformError) as unknown:
        resolve_pass("no-such-pass@1")
    assert "known passes" in str(unknown.value)

    with pytest.raises(TransformError):
        resolve_pass("Not A Pass Name")
    with pytest.raises(LaneError):
        LaneRef(FOLDED_LANE, passes=("unversioned",))
    with pytest.raises(LaneError):
        LaneRef(FOLDED_LANE, passes=(PASS, PASS))


# -- ordering: PASS -> DISCOVERY -> EXPORT ---------------------------------


def drive_for(model: TinyDiT) -> Any:
    def drive() -> None:
        model(hidden_for(PRESETS[0]), schedule(PRESETS[0]))

    return drive


def test_a_pass_after_discovery_is_refused() -> None:
    model = make_model()
    discover_modules(LaneRef(PLAIN_LANE), {"dit": model}, drive_for(model))

    with pytest.raises(TransformOrderError) as late:
        TransformSession(folded_lane()).run(model, make_pass())
    assert "already export-sealed" in str(late.value)
    assert "no longer exists" in str(late.value)


def test_a_pass_after_the_seal_is_refused() -> None:
    session = TransformSession(folded_lane())
    session.run(make_model(), make_pass())
    session.seal()
    with pytest.raises(TransformOrderError) as sealed:
        session.run(make_model(), make_pass())
    assert "PASS -> DISCOVERY -> EXPORT" in str(sealed.value)


def test_a_lane_that_declares_a_pass_needs_the_sealed_set() -> None:
    model = make_model()
    with pytest.raises(TransformOrderError) as missing:
        discover_modules(folded_lane(), {"dit": model}, drive_for(model))
    assert "declares passes" in str(missing.value)

    other = make_model()
    session = TransformSession(folded_lane())
    session.run(other, make_pass())
    transforms = session.seal()
    with pytest.raises(TransformOrderError) as undeclared:
        discover_modules(
            LaneRef(PLAIN_LANE), {"dit": other}, drive_for(other), transforms=transforms
        )
    assert "sealed for lane" in str(undeclared.value)


def test_a_pass_on_an_unmarked_tree_is_refused() -> None:
    """The graph identified must be the graph the pass rewrote."""

    transformed = make_model()
    marked = make_model(seed=1)
    session = TransformSession(folded_lane())
    session.run(transformed, make_pass())
    transforms = session.seal()

    with pytest.raises(DiscoveryError) as stray:
        discover_modules(
            folded_lane(),
            {"dit": marked},
            drive_for(marked),
            transforms=transforms,
        )
    assert "on no marked module's tree" in str(stray.value)


# -- identity -------------------------------------------------------------


def test_the_pass_lane_and_the_pass_less_lane_are_different_graphs() -> None:
    plain = make_model()
    plain_graphs = discover_modules(
        LaneRef(PLAIN_LANE), {"dit": plain}, drive_for(plain)
    )

    folded = make_model()
    session = TransformSession(folded_lane())
    session.run(folded, make_pass())
    transforms = session.seal()
    folded_graphs = discover_modules(
        folded_lane(), {"dit": folded}, drive_for(folded), transforms=transforms
    )

    assert plain_graphs.passes == ()
    assert folded_graphs.passes == (PASS,)
    assert plain_graphs.graphs and folded_graphs.graphs
    plain_hashes = {record.graph for record in plain_graphs.graphs}
    folded_hashes = {record.graph for record in folded_graphs.graphs}
    assert plain_hashes.isdisjoint(folded_hashes)

    # And the pass NAME alone changes identity, on one identical program --
    # so a pass that did not change the trace still cannot share a hash with
    # the pass-less lane. This is the derivation-input claim, isolated.
    probe = nn.Linear(4, 4)
    program = torch.export.export(probe, (torch.zeros(1, 4),))
    ingress = build_call_ingress(program, ["input"], (torch.zeros(1, 4),), {})
    bare = graph_hash(program, ingress)
    one = graph_hash(program, ingress, passes=(PASS,))
    two = graph_hash(program, ingress, passes=(PASS, "quantize-in-place@1"))
    assert len({bare, one, two}) == 3

    print(
        f"\n[identity] plain={sorted(plain_hashes)[0][:24]}... "
        f"folded={sorted(folded_hashes)[0][:24]}... "
        f"plain_graphs={len(plain_hashes)} folded_graphs={len(folded_hashes)}"
    )


def test_the_document_carries_the_lane_passes_byte_stably() -> None:
    folded = make_model()
    session = TransformSession(folded_lane())
    session.run(folded, make_pass())
    lane_graphs = discover_modules(
        folded_lane(), {"dit": folded}, drive_for(folded), transforms=session.seal()
    )
    document = GraphSetDocument(closure=CLOSURE, lanes=(lane_graphs,))
    assert document.encode() == GraphSetDocument.decode(document.encode()).encode()
    assert GraphSetDocument.decode(document.encode()).lanes[0].passes == (PASS,)


# -- the store round trip --------------------------------------------------


class CountingCall:
    """Records every evaluation, so "never evaluated" is an OBSERVATION."""

    def __init__(self) -> None:
        self.points: list[Any] = []

    def __call__(self, root: Any, module: Any, point: Any) -> torch.Tensor:
        self.points.append(point)
        return temb_for(root, module, point)


def test_buckets_round_trip_and_a_loaded_lane_never_evaluates(
    store: LocalGraphStore,
) -> None:
    """``ensure`` semantics: mint once, then read -- one key, both directions."""

    minted = make_model()
    mint_call = CountingCall()
    minted_report = TransformSession(folded_lane(), store=store).run(
        minted, make_pass(call=mint_call)
    )
    assert minted_report.mode is TransformMode.COMPUTED
    assert len(minted_report.produced) == len(PRESETS)
    assert all(row.bytes > 0 and row.address for row in minted_report.produced)
    assert len(mint_call.points) == len(PRESETS) * BLOCKS
    for key in minted_report.plan.domain:
        assert store.has_side_table(PASS, key)

    served = make_model()
    serve_call = CountingCall()
    served_report = TransformSession(folded_lane(), store=store).run(
        served, make_pass(call=serve_call)
    )
    assert served_report.mode is TransformMode.LOADED
    assert served_report.loaded == minted_report.plan.domain
    assert served_report.produced == ()
    # The projection was never EVALUATED on the serving side: the bucket
    # already holds its output. This is the whole point of the artifact path.
    assert serve_call.points == []
    assert served_report.freed_bytes == minted_report.freed_bytes
    assert served_report.cache_bytes == minted_report.cache_bytes

    for steps in PRESETS:
        handle(minted, PASS).bind(steps)
        handle(served, PASS).bind(steps)
        torch.testing.assert_close(
            served(hidden_for(steps), schedule(steps)),
            minted(hidden_for(steps), schedule(steps)),
        )

    print(
        f"\n[buckets] minted={len(minted_report.produced)} "
        f"loaded={len(served_report.loaded)} evaluations_on_mint="
        f"{len(mint_call.points)} evaluations_on_serve={len(serve_call.points)}"
    )


def test_a_partly_minted_domain_self_mints_the_rest(store: LocalGraphStore) -> None:
    partial = make_model()
    TransformSession(folded_lane(), store=store).run(
        partial, make_pass(domain=(PRESETS[0],))
    )

    whole = make_model()
    call = CountingCall()
    report = TransformSession(folded_lane(), store=store).run(
        whole, make_pass(call=call)
    )
    assert report.mode is TransformMode.MIXED
    assert report.loaded == (canonical_key(domain_key(PRESETS[0])),)
    assert len(report.produced) == 1
    # Exactly the missing point was recomputed -- the fallback is a
    # performance difference, never a correctness one.
    assert call.points == [PRESETS[1]] * BLOCKS


def test_a_bucket_from_another_checkpoint_fails_by_name(
    store: LocalGraphStore,
) -> None:
    TransformSession(folded_lane(), store=store).run(make_model(blocks=3), make_pass())

    other = make_model(blocks=2)
    with pytest.raises(TransformError) as mismatch:
        TransformSession(folded_lane(), store=store).run(other, make_pass())
    assert "not this checkpoint's" in str(mismatch.value)
    assert "minted for 3 module(s)" in str(mismatch.value)


def test_a_divergent_bucket_under_one_address_is_refused(
    store: LocalGraphStore, tmp_path: Path
) -> None:
    report = TransformSession(folded_lane(), store=store).run(make_model(), make_pass())
    occupied = report.plan.domain[0]
    impostor = tmp_path / "impostor.safetensors"
    impostor.write_bytes(b"not the bucket that was minted")
    with pytest.raises(StoreError) as diverged:
        store.publish_side_table(PASS, occupied, impostor)
    assert "deterministic pass diverged" in str(diverged.value)
    with pytest.raises(StoreError):
        store.publish_side_table("Not A Pass", occupied, impostor)


# -- the serve side --------------------------------------------------------


def loader(path: Path, record: Any) -> Any:  # pragma: no cover - never armed here
    raise AssertionError("no artifact should be fetched in these cases")


def folded_document() -> tuple[GraphSetDocument, Any]:
    folded = make_model()
    session = TransformSession(folded_lane())
    session.run(folded, make_pass())
    transforms = session.seal()
    lane_graphs = discover_modules(
        folded_lane(), {"dit": folded}, drive_for(folded), transforms=transforms
    )
    return GraphSetDocument(closure=CLOSURE, lanes=(lane_graphs,)), transforms


def test_adoption_refuses_a_boot_that_did_not_run_the_lane_passes(
    tmp_path: Path,
) -> None:
    document, transforms = folded_document()

    with pytest.raises(AdoptError) as unran:
        AdoptSession(
            None,
            document,
            FOLDED_LANE,
            SM,
            loader=loader,
            artifacts_dir=tmp_path,
            installed=INSTALLED,
        )
    assert "declares passes" in str(unran.value)
    assert PASS in str(unran.value)

    session = AdoptSession(
        None,
        document,
        FOLDED_LANE,
        SM,
        loader=loader,
        artifacts_dir=tmp_path,
        installed=INSTALLED,
        transforms=transforms,
    )
    assert session.passes == (PASS,)
    # And the adopted module is export-sealed, so the ordering also holds
    # from the serve end: a pass handed this module now is refused.
    served = make_model()
    TransformSession(folded_lane()).run(served, make_pass())
    session.adopt(served)
    with pytest.raises(TransformOrderError):
        TransformSession(folded_lane()).run(served, make_pass())
