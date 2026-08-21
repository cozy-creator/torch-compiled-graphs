"""A quant recipe as a lane property, through the tcg#52 mechanism (tcg#53).

The SECOND instance, which is the argument that tcg#52 built a mechanism and
not one model's special case: a completely different rewrite -- no domain, no
side table, no artifact -- runs through the same ``TransformSession``, the
same ``TransformPlan``/``TransformReport``, the same ordering rules and the
same graph-identity input.

The converter here is a stand-in for ``torchao.quantize_``: torchao is not a
torchcg dependency and the kernel is not what this owns. What IS under test is
the vocabulary (a closed recipe object with a PINNED granularity), the plan's
REASONS, and the refusals -- which are the product, because every failure this
guards is otherwise silent.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from torchcg.lane import LaneRef
from torchcg.quantize import (
    KEPT_DTYPE,
    KEPT_OUTSIDE,
    KEPT_SKIPPED,
    KEPT_UNALIGNED,
    ModuleSelect,
    QuantPlan,
    QuantRecipe,
    RecipeError,
    RecipeQuantize,
    applied_lane_row,
    is_resident,
    plan_quant,
    scope_census,
)
from torchcg.transform import (
    TransformMode,
    TransformSession,
    dtype_label,
    registered_passes,
    resolve_pass,
    set_submodule,
    tensor_bytes,
)

WIDTH = 64
LANE = "tiny.plain@1+cozy.fp8-rowwise@1"
QUANT_PASS = RecipeQuantize.NAME


def narrow_dtype() -> torch.dtype:
    """The 1-byte storage dtype this box can actually make on CPU."""

    try:
        torch.zeros(1).to(torch.float8_e4m3fn)
        return torch.float8_e4m3fn
    except Exception:  # pragma: no cover - old torch
        return torch.int8


NARROW = narrow_dtype()


class RowwiseFp8:
    """Stand-in for a torchao rowwise tensor subclass.

    It advertises the LOGICAL dtype it emulates and holds narrow components,
    which is exactly the shape that makes a naive ``numel * element_size``
    census read "no fp8 here".
    """

    def __init__(self, weight: torch.Tensor) -> None:
        self.scale = (
            weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-6).to(torch.float32)
        )
        self.qdata = (weight.to(torch.float32) / self.scale).to(NARROW)
        self.dtype = weight.dtype
        self._numel = int(weight.numel())

    def __tensor_flatten__(self) -> tuple[list[str], dict[str, Any]]:
        return ["qdata", "scale"], {}

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return int(torch.empty((), dtype=self.dtype).element_size())

    def dequant(self) -> torch.Tensor:
        return (self.qdata.to(torch.float32) * self.scale).to(self.dtype)


class Fp8Linear(nn.Linear):
    """What the converter leaves behind: still an ``nn.Linear``, narrow weight."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight: Any = self.weight
        out: torch.Tensor = nn.functional.linear(value, weight.dequant(), self.bias)
        return out


class Layer(nn.Module):
    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, width)
        self.mlp = nn.Linear(width, width)
        self.norm_out = nn.Linear(width, width)  # skip-list hit
        self.gate = nn.Linear(width, 40)  # 40 % 16 -> unaligned
        self.aux = nn.Linear(width, width)  # left fp32 -> dtype refusal

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.mlp(self.qkv(value))
        return out


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class Conditioner(nn.Module):
    """Decoder stack plus the siblings the scope must exclude BY DEFAULT."""

    def __init__(self) -> None:
        super().__init__()
        self.model = Decoder()
        self.visual = nn.Linear(WIDTH, WIDTH)
        self.lm_head = nn.Linear(WIDTH, WIDTH)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.lm_head(self.model(value))
        return out


def make_conditioner(seed: int = 0) -> Conditioner:
    torch.manual_seed(seed)
    model = Conditioner().to(torch.bfloat16)
    # In-scope leaves left in fp32: the dtype refusal needs a real case.
    for layer in model.model.layers:
        layer.aux.to(torch.float32)
    return model


def recipe(*, pays_only_under_compile: bool = False) -> QuantRecipe:
    return QuantRecipe(
        name="fp8-w8a8-dynamic-rowwise@1",
        scheme="float8 dynamic activation x float8 weight",
        granularity=("per_row", "per_row"),
        select=ModuleSelect(root="model.layers", skip=("norm",), align=16),
        min_sm=(8, 9),
        pays_only_under_compile=pays_only_under_compile,
    )


def convert_rowwise(root: Any, plan: QuantPlan, *, only: int | None = None) -> None:
    """The stand-in converter. ``only`` truncates it, to prove the refusal."""

    for fqn in plan.convert[: only if only is not None else len(plan.convert)]:
        old = root.get_submodule(fqn)
        new = Fp8Linear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            dtype=old.weight.dtype,
        )
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.detach().clone())
        quantized = RowwiseFp8(old.weight.detach())
        del new._parameters["weight"]
        new.weight = quantized  # type: ignore[assignment]
        set_submodule(root, fqn, new)


# -- the vocabulary --------------------------------------------------------


def test_granularity_has_no_default() -> None:
    with pytest.raises(TypeError):
        QuantRecipe(  # type: ignore[call-arg]
            name="fp8-w8a8-dynamic-rowwise@1",
            scheme="float8",
            select=ModuleSelect(),
            min_sm=(8, 9),
            pays_only_under_compile=False,
        )
    with pytest.raises(RecipeError) as half:
        QuantRecipe(
            name="fp8-w8a8-dynamic-rowwise@1",
            scheme="float8",
            granularity=("per_row", ""),
            select=ModuleSelect(),
            min_sm=(8, 9),
            pays_only_under_compile=False,
        )
    assert "there is no default" in str(half.value)


def test_the_plan_states_a_reason_for_every_module_it_keeps() -> None:
    model = make_conditioner()
    plan = plan_quant(model, recipe())

    assert plan.convert == (
        "model.layers.0.qkv",
        "model.layers.0.mlp",
        "model.layers.1.qkv",
        "model.layers.1.mlp",
    )
    reasons = plan.kept_by_reason
    assert reasons[KEPT_SKIPPED] == ("model.layers.0.norm_out", "model.layers.1.norm_out")
    assert reasons[KEPT_UNALIGNED] == ("model.layers.0.gate", "model.layers.1.gate")
    assert reasons[KEPT_DTYPE] == ("model.layers.0.aux", "model.layers.1.aux")
    # The vision tower and the head are outside the scope BY DEFAULT -- the
    # scope is rooted at the decoder stack rather than enumerating exclusions.
    assert set(reasons[KEPT_OUTSIDE]) == {"visual", "lm_head"}
    assert sum(len(rows) for rows in reasons.values()) == len(plan.kept)

    print(
        f"\n[quant plan] convert={len(plan.convert)} "
        f"kept={ {name: len(rows) for name, rows in reasons.items()} }"
    )


def test_an_empty_plan_is_refused() -> None:
    model = make_conditioner()
    blind = QuantRecipe(
        name="fp8-w8a8-dynamic-rowwise@1",
        scheme="float8",
        granularity=("per_row", "per_row"),
        select=ModuleSelect(root="model.moved_away", skip=(), align=16),
        min_sm=(8, 9),
        pays_only_under_compile=False,
    )
    with pytest.raises(RecipeError) as empty:
        plan_quant(model, blind)
    assert "the module shape moved under the recipe" in str(empty.value)
    assert "silently-bfloat16" in str(empty.value)


# -- the pass --------------------------------------------------------------


def test_the_quant_recipe_runs_through_the_transform_mechanism() -> None:
    model = make_conditioner()
    probe = torch.ones(2, WIDTH, dtype=torch.bfloat16)
    before_out = model(probe).detach().clone()
    plan = plan_quant(model, recipe())
    before = scope_census(model, plan.convert)

    session = TransformSession(LaneRef(LANE, passes=(QUANT_PASS,)))
    report = session.run(
        model, RecipeQuantize(recipe=recipe(), convert=convert_rowwise, sm=(8, 9))
    )

    assert report.pass_name == QUANT_PASS
    assert report.mode is TransformMode.CONVERTED
    assert report.plan.removes == ("fp8-w8a8-dynamic-rowwise@1",)
    assert report.plan.scope_bytes == before["bytes"]
    assert report.freed_bytes > 0
    assert report.cache_bytes < before["bytes"]
    assert report.validated is False
    assert session.seal().passes == (QUANT_PASS,)

    after = scope_census(model, plan.convert)
    assert after["bytes"] < before["bytes"]
    assert before["dtypes"] == ["bfloat16"]
    # The census SEES THROUGH the subclass: a naive numel*element_size would
    # still price these as bfloat16 and read as "no fp8 here".
    assert all("bfloat16[" in label for label in after["dtypes"])
    assert all(is_resident(model.get_submodule(fqn)) for fqn in plan.convert)

    # Still a usable module: the stand-in dequantizes, so this is a real
    # forward and not a structural assertion.
    torch.testing.assert_close(model(probe), before_out, rtol=0.2, atol=0.2)

    row = applied_lane_row("conditioner", recipe(), plan)
    assert row["lane"] == "fp8-w8a8-dynamic-rowwise@1"
    assert row["granularity"] == ["per_row", "per_row"]
    assert row["modules"] == len(plan.convert)
    assert row["kept"][KEPT_UNALIGNED] == 2

    print(
        f"\n[quantize-in-place] modules={len(plan.convert)} "
        f"scope_before={before['bytes']}B scope_after={after['bytes']}B "
        f"freed={report.freed_bytes}B dtypes={after['dtypes']}"
    )


def test_a_speed_arm_refuses_a_pod_that_cannot_compile() -> None:
    model = make_conditioner()
    lane = LaneRef(LANE, passes=(QUANT_PASS,))
    with pytest.raises(RecipeError) as eager:
        RecipeQuantize(
            recipe=recipe(pays_only_under_compile=True),
            convert=convert_rowwise,
            sm=(8, 9),
            compiled=False,
        ).plan(model, lane=lane)
    assert "0.82x -- a LOSS" in str(eager.value)

    # The MEMORY arm has the opposite compile dependence and must still arm.
    memory = RecipeQuantize(
        recipe=recipe(pays_only_under_compile=False),
        convert=convert_rowwise,
        sm=(8, 9),
        compiled=False,
    ).plan(model, lane=lane)
    assert memory.rewrites

    with pytest.raises(RecipeError) as old_card:
        RecipeQuantize(
            recipe=recipe(), convert=convert_rowwise, sm=(8, 6)
        ).plan(model, lane=lane)
    assert "needs sm>=(8, 9)" in str(old_card.value)


def test_a_half_converted_model_is_refused() -> None:
    model = make_conditioner()
    session = TransformSession(LaneRef(LANE, passes=(QUANT_PASS,)))
    with pytest.raises(RecipeError) as half:
        session.run(
            model,
            RecipeQuantize(
                recipe=recipe(),
                convert=lambda root, plan: convert_rowwise(root, plan, only=1),
                sm=(8, 9),
            ),
        )
    assert "refusing to serve a half-quantized model" in str(half.value)
    assert "3/4" in str(half.value)  # one converted, three still bf16


def test_a_converter_that_matched_nothing_is_refused() -> None:
    model = make_conditioner()
    session = TransformSession(LaneRef(LANE, passes=(QUANT_PASS,)))
    with pytest.raises(RecipeError) as nothing:
        session.run(
            model,
            RecipeQuantize(
                recipe=recipe(),
                convert=lambda root, plan: None,
                sm=(8, 9),
            ),
        )
    assert "the converter matched NOTHING" in str(nothing.value)


def test_an_already_converted_module_is_never_re_run() -> None:
    model = make_conditioner()
    TransformSession(LaneRef(LANE, passes=(QUANT_PASS,))).run(
        model, RecipeQuantize(recipe=recipe(), convert=convert_rowwise, sm=(8, 9))
    )
    # A second plan over the same tree sees the converted modules as resident
    # and refuses for having nothing left -- double quantization cannot happen.
    with pytest.raises(RecipeError) as again:
        plan_quant(model, recipe())
    assert "nothing in scope" in str(again.value)


def test_the_census_sees_through_the_subclass() -> None:
    weight = torch.ones(WIDTH, WIDTH, dtype=torch.bfloat16)
    quantized = RowwiseFp8(weight)
    naive = quantized.numel() * quantized.element_size()
    assert tensor_bytes(weight) == naive
    assert tensor_bytes(quantized) < naive
    assert dtype_label(weight) == "bfloat16"
    assert dtype_label(quantized).startswith("bfloat16[")


def test_both_passes_are_one_mechanism() -> None:
    """Two unrelated rewrites, one session, one sealed set."""

    assert set(registered_passes()) >= {"precompute-and-free@1", QUANT_PASS}
    assert resolve_pass(QUANT_PASS) is RecipeQuantize

    model = make_conditioner()
    session = TransformSession(LaneRef(LANE, passes=(QUANT_PASS,)))
    session.run(
        model, RecipeQuantize(recipe=recipe(), convert=convert_rowwise, sm=(8, 9))
    )
    transforms = session.seal()
    assert transforms.contract == LANE
    assert transforms.passes == (QUANT_PASS,)
    assert transforms.reports[0].mode is TransformMode.CONVERTED
