"""Graph content identity through the discovery primitive (pgw#1368/#1031).

Identity is the DERIVED graph: two code spellings that trace identically are
one graph; an edit that changes the trace changes the hash by construction;
and observation is structure-only, so a drive under fake tensors states the
same identity as a real drive.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from torchcg.discovery import discover_lane  # noqa: E402
from torchcg.document import LaneGraphs  # noqa: E402

  # noqa: E402

LANE = "unit.identity-fp32@1"
TARGETS = ("host.inner",)


class Scale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return sample * self.weight + 1.0


class ScaleRenamedElsewhere(torch.nn.Module):
    """A different class, module path and author -- the same trace."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return sample * self.weight + 1.0


class ScaleEdited(torch.nn.Module):
    """One-op edit: the ingress is identical, the derived graph is not."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return sample * self.weight + 2.0


class Host(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.inner(sample)
        return result


def discovered(inner: torch.nn.Module, sample: torch.Tensor) -> LaneGraphs:
    host = Host(inner)
    return discover_lane(LANE, TARGETS, {"host": host}, lambda: host(sample))


def test_identical_traces_from_different_code_spellings_dedup() -> None:
    sample = torch.zeros(2, 4)
    first = discovered(Scale(), sample)
    second = discovered(ScaleRenamedElsewhere(), sample)
    assert [record.graph for record in first.graphs] == [
        record.graph for record in second.graphs
    ]


def test_an_edit_that_changes_the_trace_changes_the_hash() -> None:
    """The pgw#1031 red case: same ingress, different graph, different key."""

    sample = torch.zeros(2, 4)
    first = discovered(Scale(), sample)
    edited = discovered(ScaleEdited(), sample)
    assert first.graphs[0].ingress.digest() == edited.graphs[0].ingress.digest()
    assert first.graphs[0].graph != edited.graphs[0].graph


def test_two_observed_shapes_are_two_graph_specializations() -> None:
    host = Host(Scale())

    def drive() -> None:
        host(torch.zeros(1, 4))
        host(torch.zeros(2, 4))
        host(torch.zeros(2, 4))  # repeat: dedup, not a third class

    lane = discover_lane(LANE, TARGETS, {"host": host}, drive)
    assert len(lane.graphs) == 2
    assert len({record.graph for record in lane.graphs}) == 2


def test_fake_tensor_observation_states_the_same_identity() -> None:
    """Observation is structure-only: a fake-tensor drive hashes identically."""

    from torch._subclasses.fake_tensor import FakeTensorMode

    sample = torch.zeros(2, 4)
    real = discovered(Scale(), sample)

    host = Host(Scale())
    mode = FakeTensorMode(allow_non_fake_inputs=True)

    def drive() -> None:
        with mode:
            host(mode.from_tensor(sample))

    fake = discover_lane(LANE, TARGETS, {"host": host}, drive)
    assert [record.graph for record in fake.graphs] == [
        record.graph for record in real.graphs
    ]
