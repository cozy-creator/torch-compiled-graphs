"""tcg#79: respelling the LANE moves no ``cg-graph-v1`` hash.

The question this file answers, because it decided whether the tcg#79 cut needed
a graph-identity version bump: does changing what a lane is NAMED BY re-key the
graphs? No. ``graph_hash`` is (canonical trace, ingress, passes, literals) --
the lane contract is not among its inputs and never was. So every graph in an
old document keeps its identity and every already-minted artifact stays
addressable; only the ROW KEY the graphs hang off is respelled.

That is why ``GRAPH_SCHEME`` stays ``cg-graph-v1`` and ``ENV_SCHEME`` stays
``cg-env-v2``, and why only ``DOCUMENT_FORMAT`` moves (3 -> 4).

This is measured through the real discovery primitive rather than asserted from
the signature: a signature fence would still pass if some caller started
folding the contract into the payload.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from torchcg.discovery import discover_modules  # noqa: E402
from torchcg.lane import LaneRef  # noqa: E402

V2_STAMP = "sdxl.diffusers@1+plain.bf16@1"
OTHER_V2_STAMP = "sdxl-inpainting.diffusers@1+cozy.fp8-rowwise@1"


class TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.proj(sample) * timestep
        return result


def _graphs_for(stamp: str) -> tuple[str, ...]:
    unet = TinyUnet().to(torch.float32)

    def drive() -> torch.Tensor:
        out: torch.Tensor = unet(
            torch.zeros(1, 4, dtype=torch.float32), torch.ones((), dtype=torch.float32)
        )
        return out

    lane = discover_modules(LaneRef(stamp, dtype=torch.float32), {"unet": unet}, drive)
    assert lane.contract == stamp
    return tuple(record.graph for record in lane.graphs)


def test_two_differently_spelled_lanes_derive_the_same_graph_hashes() -> None:
    first = _graphs_for(V2_STAMP)
    assert first, "the fixture derived no graph, so this proves nothing"
    assert first == _graphs_for(OTHER_V2_STAMP)
