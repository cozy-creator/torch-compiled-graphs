"""tcg#85 -- a policy requesting what the target will DROP is a refusal, by name.

The measured case ([[tcg#82]] phase 2): `max_autotune: True` was set and the
sdxl UNet was minted on this box's RTX 4070 Laptop. The policy digest moved, so
every existing artifact became a store miss; the mint took 622.5 s against an
85.2 s baseline; and `torch/_inductor/utils.py` logged **`Not enough SMs to use
max_autotune_gemm mode`** because the card has 36 SMs against torch's internal
`min_sms = 68`. The single most valuable half of the lever -- the Triton GEMM
templates, the only thing that reaches the 88.6% of device time measured in
extern cuBLAS -- never ran, while the key split and the 7.3x mint were paid in
full. Nothing in production would have noticed: the artifact is correct, it
serves, and it is merely slower and differently-addressed than its policy says.

The arms drive the SM COUNT at the source of truth -- torch's own
`DeviceProperties.create`, which is what `is_big_gpu` reads -- and never a
predicate of ours. No card is required or used.
"""

from __future__ import annotations

from typing import Any

import pytest
from torch._inductor.runtime.hints import DeviceProperties

from torchcg import compiler as compiler_module
from torchcg.compiler import DroppedOptimization

#: What ships, restated so a change to the shipped policy shows up in a test
#: about the shipped policy (the tcg#80 convention, kept).
SHIPPED: dict[str, object] = {
    "compile_threads": 4,
    "aot_inductor.package_constants_in_so": False,
    "always_keep_tensor_constants": True,
    "aot_inductor.use_runtime_constant_folding": False,
    "aot_inductor.package": True,
}

#: The 4070 Laptop this was measured on, and an H100. Every fleet card clears
#: the gate comfortably (A100 108, H100 132, H200 132); this box does not.
SMALL_CARD = 36
FLEET_CARD = 132


def _policy(monkeypatch: pytest.MonkeyPatch, options: dict[str, object]) -> None:
    monkeypatch.setattr(compiler_module, "_COMPILER_OPTIONS", dict(options))


def _card(monkeypatch: pytest.MonkeyPatch, multi_processor_count: int) -> None:
    """Put a card of one SM count in front of torch's OWN device gate.

    `DeviceProperties.create` is what `torch._inductor.utils.is_big_gpu` reads,
    so moving it moves torch's verdict and the number our refusal quotes
    together -- one seam, not a mocked verdict beside a mocked reason.
    """

    properties = DeviceProperties(
        type="cuda",
        index=0,
        multi_processor_count=multi_processor_count,
        cc=89,
        major=8,
    )
    monkeypatch.setattr(
        DeviceProperties, "create", classmethod(lambda cls, device: properties)
    )


def test_the_shipped_policy_requests_nothing_a_card_can_decline() -> None:
    """Today's five options have no hardware precondition at all."""

    assert compiler_module._compiler_options() == SHIPPED
    assert compiler_module.dropped_options("cuda") == {}
    assert compiler_module.executed_compile_policy("cuda") == (
        compiler_module.compile_policy()
    )


def test_max_autotune_on_a_36_sm_card_is_a_refusal_naming_the_dropped_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured defect, refused before a 622-second mint buys half a lever."""

    _policy(monkeypatch, {**SHIPPED, "max_autotune": True})
    _card(monkeypatch, SMALL_CARD)

    dropped = compiler_module.dropped_options("cuda")
    assert set(dropped) == {"max_autotune"}
    assert "36 SMs" in dropped["max_autotune"].reason
    assert "Triton GEMM templates" in dropped["max_autotune"].reason

    with pytest.raises(DroppedOptimization, match="max_autotune") as refusal:
        compiler_module.executed_compile_policy("cuda")
    assert "36 SMs" in str(refusal.value)
    assert "is_big_gpu" in str(refusal.value)


def test_the_same_policy_on_a_fleet_card_runs_and_keys_on_the_lever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same policy, same process, bigger card: the option EXECUTES and keys."""

    _policy(monkeypatch, {**SHIPPED, "max_autotune": True})
    _card(monkeypatch, FLEET_CARD)

    assert compiler_module.dropped_options("cuda") == {}
    assert compiler_module.executed_compile_policy("cuda")["max_autotune"] is True
    with_lever = compiler_module.compile_policy_digest("cuda")

    _policy(monkeypatch, SHIPPED)
    assert compiler_module.compile_policy_digest("cuda") != with_lever


def test_a_sanctioned_drop_removes_the_option_from_the_KEY_not_just_from_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tcg#85's core clause, and the reason the escape hatch is not a no-op.

    A caller that genuinely wants the pointwise/conv half of `max_autotune` on
    a small card says so. Then the key must be a function of the optimizations
    that EXECUTED: the sanctioned-dropped option is gone from the policy, so
    the artifact addresses as what it is -- identical to the artifact the same
    graph produces without the lever -- instead of claiming a lever that never
    ran. Otherwise two cards running the same emitted code disagree on its
    address, and one card running two different codes agrees on one.
    """

    _policy(monkeypatch, {**SHIPPED, "max_autotune": True})
    _card(monkeypatch, SMALL_CARD)
    monkeypatch.setattr(
        compiler_module, "_ACCEPT_DROPPED_OPTIONS", frozenset({"max_autotune"})
    )

    executed = compiler_module.executed_compile_policy("cuda")
    assert executed["max_autotune"] == "no-gemm-templates"
    partial_digest = compiler_module.compile_policy_digest("cuda")

    # THREE codes, THREE addresses. The partial mint is not the no-lever mint:
    # torch drops the GEMM templates and still benchmarks the pointwise
    # kernels, so its bytes differ from both neighbours. A key that said
    # "absent" would collide with the no-lever artifact in the CAS.
    _policy(monkeypatch, SHIPPED)
    no_lever_digest = compiler_module.compile_policy_digest("cuda")

    _policy(monkeypatch, {**SHIPPED, "max_autotune": True})
    _card(monkeypatch, FLEET_CARD)
    full_lever_digest = compiler_module.compile_policy_digest("cuda")

    assert len({partial_digest, no_lever_digest, full_lever_digest}) == 3


def test_a_cpu_target_asks_the_card_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch keeps Triton templates on CPU without an SM gate, so there is no drop.

    This also keeps every CPU mint -- and every key a cardless box derives --
    free of any question that needs a device to answer.
    """

    _policy(monkeypatch, {**SHIPPED, "max_autotune": True})

    def _explode(cls: Any, device: Any) -> Any:  # pragma: no cover - must not run
        raise AssertionError("a cpu target must not read GPU device properties")

    monkeypatch.setattr(DeviceProperties, "create", classmethod(_explode))
    assert compiler_module.dropped_options("cpu") == {}
    assert compiler_module.executed_compile_policy("cpu")["max_autotune"] is True


def test_a_falsy_request_has_nothing_to_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    """`max_autotune: False` asks for nothing, so no card can decline it."""

    _policy(monkeypatch, {**SHIPPED, "max_autotune": False})
    _card(monkeypatch, SMALL_CARD)
    assert compiler_module.dropped_options("cuda") == {}


def test_cudagraphs_is_classified_INERT_and_cannot_reach_any_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AOTI discards it, so keying on it would split the pool on nothing.

    `get_cpp_wrapper_config` computes `triton.cudagraphs and not
    V.aot_compilation` (torch 2.13, `compile_fx.py:2246`), which is False for
    every artifact this package mints. CUDA-graph capture is a SERVING lever
    around `runner.__call__`; it is not a compile-time fact and must not be an
    address (tcg#82, `0-DESIGN.md` §9).
    """

    baseline = compiler_module.compile_policy_digest("cpu")
    _policy(monkeypatch, {**SHIPPED, "triton.cudagraphs": True})
    policy = compiler_module.compile_policy()
    assert "triton.cudagraphs" not in policy
    assert compiler_module.compile_policy_digest("cpu") == baseline


def test_the_device_gate_is_asked_UNCACHED_so_it_can_go_red_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`is_big_gpu` is `functools.cache`d on the device, and a cached verdict
    is a guard that cannot go red: the first answer in a process would decide
    every later question about every later card. The precondition calls the
    uncached function, so two different cards in one process give two answers.
    """

    from torch._inductor.utils import is_big_gpu

    assert hasattr(is_big_gpu, "__wrapped__")
    _policy(monkeypatch, {**SHIPPED, "max_autotune": True})
    _card(monkeypatch, FLEET_CARD)
    assert compiler_module.dropped_options("cuda") == {}
    _card(monkeypatch, SMALL_CARD)
    assert set(compiler_module.dropped_options("cuda")) == {"max_autotune"}
