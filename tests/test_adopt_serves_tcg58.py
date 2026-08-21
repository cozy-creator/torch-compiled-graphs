"""tcg#58: an adopted graph must SERVE — real compile, real bind, real numbers.

``test_adopt_swap`` proves the SWAP (which module, which structure, which
hole) against a stub loader, and that is the right shape for what it tests.
What no test asserted is the half the stub stands in for: that the thing
armed behind the dispatcher **computes the same answer the module did**.

That gap is not theoretical. Every consumer wrote the loader as a bare
``aoti_load_package(path)`` and shipped it, because a stub loader returning a
sentinel satisfies every swap assertion there is. The artifacts are WEIGHTLESS
by sealed policy, so a raw handle runs with an empty constant buffer, and the
author's nested kwargs never become the flat feeds the package was exported
with. Both defects are invisible to a structural test and fatal to a request.

So this file compiles a real graph (CPU AOTInductor — no GPU), publishes it
into a real ``LocalGraphStore``, adopts it through the real ``AdoptSession``
with **no loader argument at all**, and then compares the adopted module's
output against the eager output it recorded first. An empty constant buffer
cannot pass that; neither can an unflattened call.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensorfs import LocalCAS

from torchcg import serve
from torchcg.adopt import AdoptSession
from torchcg.declaration import GraphSpecialization, RuntimeCompatibility
from torchcg.discovery import discover_lane
from torchcg.document import GraphSetDocument
from torchcg.engine import Engine
from torchcg.graph_identity import EnvIdentity
from torchcg.requirements import RequirementsManifest
from torchcg.runner import ConstantBindingError
from torchcg.store import LocalGraphStore

CONTRACT = "tiny.plain@1+plain.f32@1"
STACK: tuple[tuple[str, str], ...] = (("torch", torch.__version__),)
TOOLCHAIN = {
    "settings_declaration": "settings-v1",
    "loaded_libs": "loaded-libs-v1",
    "torch": "torch-record-v1",
    "triton": "triton-record-v1",
}
SAMPLE = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
CONTEXT = torch.tensor([[0.5, 0.25, -0.5, 2.0]])


class TinyUnet(torch.nn.Module):
    """Real parameters and a KEYWORD argument, both deliberately.

    * Real parameters, because an empty constant buffer is only detectable
      against a module whose weights actually move the answer.
    * A keyword argument, because tcg#61: the loaded package is torch's
      ``AOTICompiledModel`` and its ``__call__`` re-flattens with the
      export's own ``in_spec``, so it wants the author's call SHAPE. A module
      whose forward takes one positional tensor cannot tell a correct
      implementation from one that pre-flattens the call itself — both work.
      The first real artifact this ran against had five keyword arguments and
      answered with ``ValueError: Ran into a kwarg keyword mismatch``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(
        self, sample: torch.Tensor, *, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        out: torch.Tensor = self.proj(sample) + encoder_hidden_states
        return out * 3.0


@pytest.fixture()
def adopted(tmp_path: Path) -> Any:
    """One real compiled graph, published and adoptable — the whole lifecycle.

    Returns the namespace the author code lives in plus everything a test
    needs to say what happened: the eager answer, the document, the store and
    the session.
    """

    torch.manual_seed(0)
    pipe = SimpleNamespace(unet=TinyUnet())
    eager = pipe.unet(SAMPLE, encoder_hidden_states=CONTEXT).detach().clone()

    programs: dict[str, Any] = {}

    def sink(graph: str, program: Any) -> str:
        programs[graph] = program
        return f"sha256:{graph[-16:]:0>64}"

    def drive() -> None:
        pipe.unet(SAMPLE, encoder_hidden_states=CONTEXT)

    lane = discover_lane(
        CONTRACT, ("pipe.unet",), {"pipe": pipe}, drive, program_sink=sink
    )
    assert len(lane.graphs) == 1, "one driven shape must derive exactly one graph"
    record = lane.graphs[0]

    runtime = RuntimeCompatibility("cpu", toolchain=TOOLCHAIN)
    engine = Engine(LocalCAS(tmp_path / "graph-cas"))
    spec = GraphSpecialization(
        record.graph, record.target, programs[record.graph], record.ingress
    )
    result = engine.compile(spec, runtime, tmp_path / "minted")
    sm = str(result.compiled_graph.metadata["sm"])

    packed = engine.export_artifact(result.compiled_graph.key, tmp_path / "artifact.tgz")
    env = EnvIdentity(stack=STACK, sm=sm)
    store = LocalGraphStore(LocalCAS(tmp_path / "store-cas"))
    store.publish_artifact(
        record.graph,
        env,
        packed,
        RequirementsManifest(
            include_set=(("torch", torch.__version__),), sm_compiled=sm
        ),
    )

    document = GraphSetDocument(stack=STACK, lanes=(lane,))
    # NO `loader=`. The production loader is the default precisely so a
    # consumer cannot supply a raw one by accident.
    session = AdoptSession(
        store,
        document,
        CONTRACT,
        sm,
        artifacts_dir=tmp_path / "adopted",
        stack=STACK,
    )
    return SimpleNamespace(
        pipe=pipe, eager=eager, record=record, session=session, store=store,
        env=env, packed=packed, tmp_path=tmp_path,
    )


def test_the_adopted_graph_returns_the_eager_answer(adopted: Any) -> None:
    """THE assertion the stub loader could never make."""

    adopted.session.adopt(adopted.pipe.unet)
    assert adopted.session.holes == (), f"unexpected holes: {adopted.session.holes}"
    assert [row.graph for row in adopted.session.adopted] == [adopted.record.graph]

    out = adopted.pipe.unet(SAMPLE, encoder_hidden_states=CONTEXT)
    torch.testing.assert_close(out, adopted.eager)


def test_the_call_went_through_the_compiled_graph_and_not_the_eager_forward(
    adopted: Any,
) -> None:
    """Same numbers could also mean the eager forward answered. It did not."""

    adopted.session.adopt(adopted.pipe.unet)
    armed = adopted.pipe.unet.forward._entries[0][1]
    assert isinstance(armed, serve.CompiledGraphCall)
    assert armed.runner.calls == 0

    adopted.pipe.unet(SAMPLE, encoder_hidden_states=CONTEXT)

    assert armed.runner.calls == 1


def test_the_constants_are_bound_from_the_live_module_by_reference(
    adopted: Any,
) -> None:
    """The weights the graph reads ARE the module's — one copy, not two."""

    adopted.session.adopt(adopted.pipe.unet)
    armed = adopted.pipe.unet.forward._entries[0][1]
    runner = armed.runner

    assert runner.bound is True
    assert set(runner.bound_fqns) == {"proj.weight", "proj.bias"}
    # And the declared set is EXACTLY the bound set (tcg#80): with runtime
    # constant folding off there is no `source=computed` row for the artifact
    # to own, so every declared constant is a raw pointer into the live
    # module. That equality IS the memory fix -- a `computed` row is a second
    # copy of a weight, materialized on the first call by a `cudaMalloc`
    # outside the caching allocator. Binding is still by SOURCE rather than
    # by "everything declared"; there is simply nothing left to skip.
    assert set(runner.declared_fqns) == set(runner.bound_fqns)
    assert not [row for row in runner._constants if row.source == "computed"]
    for fqn, value in runner._bound_values.items():
        assert value.data_ptr() == adopted.pipe.unet.state_dict()[fqn].data_ptr()


def test_a_module_on_the_wrong_device_is_a_hole_and_not_a_wrong_answer(
    adopted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The independent cross-check: recorded placement vs the live module.

    Binding host pointers into a device graph (or the reverse) is accepted by
    AOTI and answered with garbage, so it has to be refused BY NAME before the
    bind. The placement is what the mint recorded from the program it
    compiled; the device is read off the module's own tensors — two sources,
    which is the only kind of cross-check worth writing.
    """

    monkeypatch.setattr(serve, "module_device", lambda module: "cuda:0")
    adopted.session.adopt(adopted.pipe.unet)

    assert adopted.session.adopted == ()
    assert [hole.record.graph for hole in adopted.session.holes] == [
        adopted.record.graph
    ]
    reason = adopted.session.holes[0].reason
    assert reason.startswith("ConstantBindingError:"), reason
    assert "placement" in reason, reason
    # And the module still serves, eagerly, with the right answer.
    torch.testing.assert_close(
        adopted.pipe.unet(SAMPLE, encoder_hidden_states=CONTEXT), adopted.eager)


def test_a_raw_unbound_load_refuses_instead_of_serving_garbage(
    adopted: Any,
) -> None:
    """What the shipped loader did, run for real: it cannot even be called.

    The runner gate is what makes the whole class loud rather than silent —
    the raw path's failure mode WOULD have been an empty constant buffer, and
    this is the refusal standing between that and a served request.
    """

    graph = serve.materialize(adopted.packed, adopted.tmp_path / "raw")
    from torchcg.runner import CompiledGraphRunner

    runner = CompiledGraphRunner._from_verified_graph(graph)
    with pytest.raises(ConstantBindingError) as caught:
        runner(SAMPLE, encoder_hidden_states=CONTEXT)
    assert caught.value.reason == "constants_unbound"


def test_the_fetched_artifact_verifies_before_anything_is_loaded(
    adopted: Any, tmp_path: Path
) -> None:
    """A corrupt fetch is a hole with a stated reason, never a dead boot."""

    corrupt = tmp_path / "corrupt.tgz"
    corrupt.write_bytes(b"not an artifact")
    with pytest.raises(Exception) as caught:
        serve.aoti_loader(corrupt, adopted.record, adopted.pipe.unet)
    assert "artifact" in str(caught.value).lower() or "gzip" in str(caught.value).lower()
