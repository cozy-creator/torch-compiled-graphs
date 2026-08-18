"""The Lane API is pinned to the contract file Paul line-reviews.

The target-state SDXL endpoint (`serverless-endpoints/sdxl/src/sdxl/main_v2.py`)
is the authority on the author-facing spelling::

    Lane("bf16", compile=("unet",), contract="plain.bf16@1",
         dtype=torch.bfloat16)

Two fences, different in kind:

1. ALWAYS-RUN -- a fixture endpoint shaped exactly like the contract file
   (lanes in a class-level declaration, targets as component-attribute paths,
   ``lane.dtype`` consumed by setup) drives the real discovery primitive.
2. SIBLING-CHECKOUT -- when the contract file itself is present on this
   machine, every ``Lane(...)`` call in it is EXECUTED against this package's
   constructor, so a spelling change in the reviewed file breaks here before
   it breaks an author. CI has no sibling checkouts; the skip names the
   absent path rather than printing green (the rename-hazard lesson: read
   the count, not the verdict).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from torchcg.discovery import discover_lane  # noqa: E402
from torchcg.lane import Lane  # noqa: E402

CONTRACT_FILE = Path.home() / "cozy/serverless-endpoints/sdxl/src/sdxl/main_v2.py"


class TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.proj(sample) * timestep
        return result


class TinyPipe:
    """The component-bearing shape discovery resolves paths against."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.unet = TinyUnet().to(dtype)
        self.dtype = dtype

    @property
    def components(self) -> dict[str, object]:
        return {"unet": self.unet}

    def __call__(self, prompt: str) -> torch.Tensor:
        sample = torch.zeros(1, 4, dtype=self.dtype)
        out: torch.Tensor = self.unet(sample, torch.ones((), dtype=self.dtype))
        return out


class FixtureEndpoint:
    """The contract file's shape: lanes declared beside unchanged serve code."""

    lanes = (
        Lane("bf16", compile=("unet",), contract="plain.bf16@1", dtype=torch.bfloat16),
        Lane("fp8", compile=("unet",), contract="cozy.fp8-rowwise@1", dtype=torch.bfloat16),
    )

    def setup(self, lane: Lane) -> None:
        # main_v2: from_pretrained(..., torch_dtype=ctx.lane.dtype)
        self.pipe = TinyPipe(dtype=torch.float32 if lane.dtype is None else lane.dtype)

    def generate(self, prompt: str) -> torch.Tensor:
        result: torch.Tensor = self.pipe(prompt)
        return result


def test_fixture_endpoint_discovers_through_the_contract_shape() -> None:
    endpoint = FixtureEndpoint()
    for lane in endpoint.lanes:
        assert lane.dtype is torch.bfloat16  # ctx.lane.dtype is real, per lane
        endpoint.setup(lane)
        graphs = discover_lane(
            lane, endpoint.pipe.components, lambda: endpoint.generate("trace")
        )
        assert graphs.contract == lane.contract
        assert graphs.unobserved_targets == ()
        assert [record.target for record in graphs.graphs] == ["unet"]


def test_two_lanes_share_targets_but_keep_their_own_contract() -> None:
    bf16, fp8 = FixtureEndpoint.lanes
    assert bf16.compile == fp8.compile == ("unet",)
    assert bf16.contract != fp8.contract


def test_every_lane_call_in_the_reviewed_contract_file_constructs() -> None:
    if not CONTRACT_FILE.is_file():
        pytest.skip(f"contract file absent on this machine: {CONTRACT_FILE}")
    tree = ast.parse(CONTRACT_FILE.read_text())
    lane_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Lane"
    ]
    assert lane_calls, "the contract file spells no Lane(...) calls"
    for call in lane_calls:
        expression = ast.Expression(body=call)
        ast.fix_missing_locations(expression)
        lane = eval(  # noqa: S307 -- executing the reviewed file's own spelling
            compile(expression, str(CONTRACT_FILE), "eval"),
            {"Lane": Lane, "torch": torch},
        )
        assert isinstance(lane, Lane)
        assert lane.compile and lane.contract
