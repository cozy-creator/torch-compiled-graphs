"""The lane API is pinned to the contract file Paul line-reviews.

The target-state SDXL endpoint (`serverless-endpoints/sdxl/src/sdxl/main_v2.py`)
is the authority on the author-facing spelling. Paul's review ruling: the
named ``Lane(...)`` class is DEAD -- a lane IS a tensorfs contract reference::

    @endpoint(lanes=("sdxl.diffusers-bf16@1", "cozy.sdxl-fp8-rowwise@1"))
    class SdxlEndpoint(Endpoint):
        def setup(self, ctx):
            ...
            self.pipe.unet = ctx.compile(self.pipe.unet)

The decorator carries ``lanes=`` ONLY; the compile hint is IMPERATIVE,
torch.compile-style, in setup (``self.pipe.unet = ctx.compile(self.pipe.unet)``)
-- real objects, no strings. dtype comes FROM resolution (``ctx.lane.dtype``
on the resolved ``LaneRef``), never from the author.

Two fences, different in kind:

1. ALWAYS-RUN -- a fixture endpoint shaped exactly like the contract file
   (contract-handle lanes, imperative ctx.compile marking in setup, resolved
   ``LaneRef.dtype`` consumed by setup) drives the real discovery primitive
   over the MARKED modules.
2. SIBLING-CHECKOUT -- when the contract file itself is present on this
   machine, its decorator's ``lanes=`` handles are validated through this
   package's ``require_contract_ref``, the imperative ``ctx.compile(``
   marking is asserted PRESENT, and the dead spellings (``Lane(``,
   decorator ``compile=``) are asserted ABSENT.
   CI has no sibling checkouts; the skip names the absent path rather than
   printing green (the rename-hazard lesson: read the count, not the
   verdict).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeVar

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from torchcg.discovery import discover_modules  # noqa: E402
from torchcg.lane import LaneRef, require_contract_ref  # noqa: E402

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


_M = TypeVar("_M")


class _MarkingCtx:
    """The trace-side half of the ctx.compile contract, in miniature.

    ``compile`` is identity-typed: at serve it may return the adopted
    compiled graph, but the annotation the author sees is module-in,
    module-out -- the swap point is the author's marked line.
    """

    def __init__(self, lane: LaneRef) -> None:
        self.lane = lane
        self.marked: list[object] = []

    def compile(self, module: _M) -> _M:
        self.marked.append(module)
        return module


class FixtureEndpoint:
    """The contract file's shape: contract-handle lanes beside unchanged code."""

    lanes = ("sdxl.diffusers-bf16@1", "cozy.sdxl-fp8-rowwise@1")

    def setup(self, ctx: _MarkingCtx) -> None:
        # main_v2: from_pretrained(..., torch_dtype=ctx.lane.dtype), then
        # self.pipe.unet = ctx.compile(self.pipe.unet)
        lane = ctx.lane
        self.pipe = TinyPipe(dtype=torch.float32 if lane.dtype is None else lane.dtype)
        self.pipe.unet = ctx.compile(self.pipe.unet)

    def generate(self, prompt: str) -> torch.Tensor:
        result: torch.Tensor = self.pipe(prompt)
        return result


def _resolved(contract: str) -> LaneRef:
    # dtype comes from RESOLUTION (registry entry / model-type pointer),
    # never from the author's spelling.
    return LaneRef(contract, dtype=torch.bfloat16)


def test_fixture_endpoint_discovers_through_the_contract_shape() -> None:
    endpoint = FixtureEndpoint()
    for contract in endpoint.lanes:
        ctx = _MarkingCtx(_resolved(contract))
        assert ctx.lane.dtype is torch.bfloat16  # ctx.lane.dtype is real, per lane
        endpoint.setup(ctx)
        assert ctx.marked == [endpoint.pipe.unet]
        graphs = discover_modules(
            ctx.lane,
            {"unet": endpoint.pipe.unet},
            lambda: endpoint.generate("trace"),
        )
        assert graphs.contract == contract
        assert graphs.unobserved_targets == ()
        assert [record.target for record in graphs.graphs] == ["unet"]


def test_two_lanes_share_targets_but_keep_their_own_contract() -> None:
    bf16, fp8 = FixtureEndpoint.lanes
    assert bf16 != fp8
    assert require_contract_ref(bf16) and require_contract_ref(fp8)


def _decorator_kwarg_elements(tree: ast.Module, kwarg: str) -> list[ast.expr]:
    """Elements of `kwarg=(...)` on a class -- CLASS KEYWORDS first (the
    ratified `class M(Model[MT], lanes=(...))` spelling), decorator calls as
    the historical fallback."""

    out: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for keyword in node.keywords:
            if keyword.arg == kwarg and isinstance(keyword.value, ast.Tuple):
                out.extend(keyword.value.elts)
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            for keyword in decorator.keywords:
                if keyword.arg != kwarg or not isinstance(keyword.value, ast.Tuple):
                    continue
                out.extend(keyword.value.elts)
    return out


def test_the_reviewed_contract_file_spells_lanes_as_contract_refs() -> None:
    """Lanes are contract REFERENCES: registry objects or handle strings.

    Either spelling resolves to a tensorfs layout contract; what is fenced
    OUT is the dead vocabulary -- a named Lane class, a decorator-level
    compile= -- and a lanes= entry that is neither a registry reference nor
    a handle.
    """

    if not CONTRACT_FILE.is_file():
        pytest.skip(f"contract file absent on this machine: {CONTRACT_FILE}")
    source = CONTRACT_FILE.read_text()
    assert "Lane(" not in source, "the dead named-Lane spelling is back"
    assert "ctx.compile(" in source, "the imperative compile marking is absent"
    tree = ast.parse(source)
    lanes = _decorator_kwarg_elements(tree, "lanes")
    assert lanes, "the contract file's decorator spells no lanes=(...)"
    for element in lanes:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            # Handle-string spelling: must be a registry contract handle.
            assert require_contract_ref(element.value) == element.value
            assert LaneRef(element.value).contract == element.value
        else:
            # Registry-object spelling: an imported reference or an inline
            # tensorfs.Contract(...); a bare name would be the dead class.
            assert isinstance(element, (ast.Attribute, ast.Call)), ast.dump(element)
    assert not _decorator_kwarg_elements(tree, "compile"), (
        "the decorator-level compile= spelling is dead; marking is imperative"
    )
