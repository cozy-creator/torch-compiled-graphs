"""Banked facts about DISPATCH: the selector IS the dispatcher (se#837).

The normalization is not tested by asserting a plan object has the right shape
-- that is exactly the defect this replaces, a plan that was computed and never
executed. It is tested by asserting the CALL the compiled thing received.
"""

from __future__ import annotations

from typing import Any

import pytest

from torchcg.adopt import Dispatcher, Record, fit
from torchcg.identity import CallIngress, CallInput

torch = pytest.importorskip("torch")


def _row(
    name: str,
    position: int,
    param: str,
    param_position: int,
    dtype: str,
    shape: tuple[int | str, ...],
    path: tuple[int | str, ...] = (),
) -> CallInput:
    from torchcg.identity import exported_input_name

    return CallInput(
        name=name,
        position=position,
        param=param,
        param_position=param_position,
        path=path,
        exported_name=exported_input_name(param, path),
        dtype=dtype,
        shape=shape,
    )


def unet_ingress(*, timestep_dtype: str = "float32", batch: int = 2) -> CallIngress:
    """The sd15 UNet call shape: a 4-D latent, a RANK-0 timestep, an embedding."""

    return CallIngress(
        parameters=("sample", "timestep", "encoder_hidden_states"),
        flat_arity=3,
        inputs=(
            _row("sample", 0, "sample", 0, "float32", (batch, 4, 8, 8)),
            _row("timestep", 1, "timestep", 1, timestep_dtype, ()),
            _row("encoder_hidden_states", 2, "encoder_hidden_states", 2,
                 "float32", (batch, 77, 16)),
        ),
    )


class _Spy:
    """Stands in for the compiled package: records exactly what it was called
    with. The .so's own behavior is torch's; what is under test here is the
    call this library hands it."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append((args, kwargs))
        return "compiled"


def _call(
    batch: int = 2, *, timestep_dtype: str = "float32"
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (
        torch.zeros(batch, 4, 8, 8),
        torch.zeros((), dtype=getattr(torch, timestep_dtype)),
        torch.zeros(batch, 77, 16),
    ), {}


# ---------------------------------------------------------------------------
# se#837 -- the recast is EXECUTED, not merely computable
# ---------------------------------------------------------------------------


def test_an_exact_call_dispatches_compiled_with_no_normalization() -> None:
    args, kwargs = _call()
    gap, recasts, passed = fit(unet_ingress(), args, kwargs)
    assert gap == ""
    assert recasts == ()
    assert passed == 3


def test_an_int64_timestep_against_a_float32_record_RUNS_COMPILED() -> None:
    """The whole point. ddim/dpmpp/unipc end `set_timesteps` in `.to(int64)`;
    euler-family present float32. The sampler is deliberately not a compile
    axis, so an int64 timestep must reach the SAME artifact -- recast on the
    way in, at the boundary.

    Before this, `_row_mismatch` refused on dtype and the request served eager
    while the contract said it was covered.
    """

    spy = _Spy()
    eager = _Spy()
    dispatcher = Dispatcher(eager=eager)
    dispatcher.arm([Record("g", unet_ingress(), spy)])

    args, kwargs = _call(timestep_dtype="int64")
    assert dispatcher(*args, **kwargs) == "compiled"

    assert eager.calls == [], "the call fell through to eager"
    assert dispatcher.compiled_calls == 1
    assert dispatcher.recast_calls == 1
    # The EXECUTED normalization: the .so saw float32, not int64.
    (seen_args, _), = spy.calls
    assert seen_args[1].dtype is torch.float32
    # ...and the value survived it. 999 is the timestep the bf16 exclusion
    # exists for; float32 holds it exactly.
    assert int(seen_args[1]) == 0


def test_the_recast_preserves_the_value() -> None:
    spy = _Spy()
    dispatcher = Dispatcher(eager=_Spy())
    dispatcher.arm([Record("g", unet_ingress(), spy)])
    args, kwargs = _call(timestep_dtype="int64")
    args = (args[0], torch.tensor(999, dtype=torch.int64), args[2])
    dispatcher(*args, **kwargs)
    (seen_args, _), = spy.calls
    assert seen_args[1].dtype is torch.float32
    assert int(seen_args[1]) == 999


def test_the_caller_s_own_tensors_are_not_mutated() -> None:
    """A sampler that reuses its kwargs across steps must not see a dtype it
    did not set."""

    spy = _Spy()
    dispatcher = Dispatcher(eager=_Spy())
    dispatcher.arm([Record("g", unet_ingress(), spy)])
    args, kwargs = _call(timestep_dtype="int64")
    dispatcher(*args, **kwargs)
    assert args[1].dtype is torch.int64


def test_disabling_the_normalization_reproduces_the_PRE_FIX_REFUSAL(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The red arm the requirement asks for by name. With the recast targets
    emptied, the int64 request is refused on dtype and serves eager -- which is
    precisely the behavior se#837 measured in the old tree."""

    import torchcg.adopt as adopt

    monkeypatch.setattr(adopt, "RECAST_TARGETS", ())
    spy, eager = _Spy(), _Spy()
    dispatcher = Dispatcher(eager=eager)
    dispatcher.arm([Record("g", unet_ingress(), spy)])
    args, kwargs = _call(timestep_dtype="int64")
    dispatcher(*args, **kwargs)
    assert spy.calls == []
    assert len(eager.calls) == 1
    assert dispatcher.eager_calls == 1


def test_a_recast_is_ONLY_for_a_rank_0_integral_feed() -> None:
    """Not a general dtype coercion. A tensor of values is a different graph,
    and silently recasting it would serve the wrong numerics at speed."""

    # A rank-4 float16 latent against a float32 record: refused, not recast.
    args, kwargs = _call()
    args = (args[0].to(torch.float16), args[1], args[2])
    gap, recasts, _ = fit(unet_ingress(), args, kwargs)
    assert "dtype float16 != expected float32" in gap
    assert recasts == ()


def test_bfloat16_is_NOT_a_recast_target() -> None:
    """bf16's 8 mantissa bits round timestep 999 to 1000 -- a numeric change,
    not a normalization."""

    import torchcg.adopt as adopt

    assert "bfloat16" not in adopt.RECAST_TARGETS
    assert "float16" not in adopt.RECAST_TARGETS
    assert torch.tensor(999, dtype=torch.int64).to(torch.bfloat16).item() == 1000.0
    assert torch.tensor(999, dtype=torch.int64).to(torch.float32).item() == 999.0


def test_the_gap_sentence_and_the_match_decision_are_ONE_predicate() -> None:
    """The guard and its explanation cannot drift into telling two stories,
    because the sentence IS the refusal."""

    args, kwargs = _call(batch=3)
    gap, _, passed = fit(unet_ingress(batch=2), args, kwargs)
    assert gap and "sample" in gap
    assert passed == 0


# ---------------------------------------------------------------------------
# Exact-match-or-eager
# ---------------------------------------------------------------------------


def test_a_call_that_fits_nothing_serves_EAGER_and_raises_nothing() -> None:
    spy, eager = _Spy(), _Spy()
    dispatcher = Dispatcher(eager=eager)
    dispatcher.arm([Record("g", unet_ingress(batch=2), spy)])
    args, kwargs = _call(batch=5)
    assert dispatcher(*args, **kwargs) == "compiled"  # the eager spy's return
    assert spy.calls == []
    assert dispatcher.eager_calls == 1


def test_the_first_gap_is_reported_ONCE_per_module() -> None:
    spy, eager = _Spy(), _Spy()
    dispatcher = Dispatcher(eager=eager)
    dispatcher.arm([Record("g", unet_ingress(batch=2), spy)])
    args, kwargs = _call(batch=5)
    for _ in range(4):
        dispatcher(*args, **kwargs)
    assert dispatcher.eager_calls == 4
    assert dispatcher._reported is True


def test_a_concrete_record_is_tried_before_a_dynamic_one() -> None:
    """An exact bucket beats a dynamic range that also spans it."""

    concrete, dynamic = _Spy(), _Spy()
    dynamic_ingress = CallIngress(
        parameters=("sample", "timestep", "encoder_hidden_states"),
        flat_arity=3,
        inputs=(
            _row("sample", 0, "sample", 0, "float32", ("s0", 4, 8, 8)),
            _row("timestep", 1, "timestep", 1, "float32", ()),
            _row("encoder_hidden_states", 2, "encoder_hidden_states", 2,
                 "float32", ("s0", 77, 16)),
        ),
        symbols=(("s0", (2, 8)),),
    )
    dispatcher = Dispatcher(eager=_Spy())
    dispatcher.arm([
        Record("dyn", dynamic_ingress, dynamic),
        Record("fixed", unet_ingress(batch=2), concrete),
    ])
    args, kwargs = _call(batch=2)
    dispatcher(*args, **kwargs)
    assert len(concrete.calls) == 1 and dynamic.calls == []


def test_a_symbolic_dim_is_not_a_wildcard() -> None:
    """A dynamic record guards exactly what it was exported for: inside the
    range, and one symbol takes one value everywhere it appears."""

    ingress = CallIngress(
        parameters=("sample", "timestep", "encoder_hidden_states"),
        flat_arity=3,
        inputs=(
            _row("sample", 0, "sample", 0, "float32", ("s0", 4, 8, 8)),
            _row("timestep", 1, "timestep", 1, "float32", ()),
            _row("encoder_hidden_states", 2, "encoder_hidden_states", 2,
                 "float32", ("s0", 77, 16)),
        ),
        symbols=(("s0", (2, 8)),),
    )
    args, kwargs = _call(batch=4)
    assert fit(ingress, args, kwargs)[0] == ""
    # Outside the exported range.
    assert "outside the exported range" in fit(ingress, _call(batch=9)[0], {})[0]
    # The two feeds disagree about the shared symbol.
    args = (torch.zeros(4, 4, 8, 8), torch.zeros(()), torch.zeros(2, 77, 16))
    assert "put it at" in fit(ingress, args, {})[0]


def test_a_stride_bearing_symbol_refuses_a_non_multiple() -> None:
    """A latent side a UNet accepts only in multiples of eight exports as
    `8*s1`, and the coefficient is carried by the symbol's own name."""

    ingress = CallIngress(
        parameters=("sample",),
        flat_arity=1,
        inputs=(_row("sample", 0, "sample", 0, "float32", (1, 4, "8*s1", 8)),),
        symbols=(("8*s1", (16, 64)),),
    )
    assert fit(ingress, (torch.zeros(1, 4, 24, 8),), {})[0] == ""
    assert "not a multiple of 8" in fit(ingress, (torch.zeros(1, 4, 20, 8),), {})[0]


def test_a_recast_reaches_a_NESTED_feed() -> None:
    """`added_cond_kwargs.text_embeds` is a real sd-family call path, so the
    normalization must rewrite through the nesting, not only at the top."""

    ingress = CallIngress(
        parameters=("sample", "conditioning"),
        flat_arity=2,
        inputs=(
            _row("sample", 0, "sample", 0, "float32", (2, 4)),
            _row("step", 1, "conditioning", 1, "float32", (), path=("step",)),
        ),
    )
    spy = _Spy()
    dispatcher = Dispatcher(eager=_Spy())
    dispatcher.arm([Record("g", ingress, spy)])
    conditioning = {"step": torch.tensor(7, dtype=torch.int64)}
    dispatcher(torch.zeros(2, 4), conditioning)
    (seen_args, _), = spy.calls
    assert seen_args[1]["step"].dtype is torch.float32
    assert int(seen_args[1]["step"]) == 7
    # The caller's own dict is untouched.
    assert conditioning["step"].dtype is torch.int64


# ---------------------------------------------------------------------------
# rebind -- the constant table after a WEIGHT WRITE
# ---------------------------------------------------------------------------


class _Package:
    """Stands in for torch's AOTICompiledModel: records what it was bound to."""

    def __init__(self) -> None:
        self.installs: list[dict[str, Any]] = []
        # The fields `adopt.load` stamps and `rebind` reads back.
        self._torchcg_retained: dict[str, Any] = {}
        self._torchcg_state_fqns: tuple[str, ...] = ()
        self._torchcg_literals: dict[str, Any] = {}

    def load_constants(self, values: Any, **_kw: Any) -> None:
        self.installs.append(dict(values))


class _Module:
    def __init__(self, weight: Any) -> None:
        self._weight = weight

    def state_dict(self) -> dict[str, Any]:
        return {"w.weight": self._weight}

    def named_buffers(self) -> list[tuple[str, Any]]:
        return []


def test_rebind_re_resolves_the_state_dict_and_KEEPS_the_literals() -> None:
    """Constants bind `user_managed=True`, so the artifact holds raw POINTERS
    into the module's tensors. Folding a LoRA REPLACES a tensor rather than
    writing through it, which leaves those pointers aimed at storage the module
    no longer uses — and the graph then serves stale weights with nothing
    raising."""

    from torchcg.adopt import rebind

    original, folded = torch.zeros(2, 2), torch.ones(2, 2)
    module = _Module(original)
    package = _Package()
    package._torchcg_retained = {"w.weight": original, "baked": torch.full((1,), 7.0)}
    package._torchcg_state_fqns = ("w.weight",)
    package._torchcg_literals = {"baked": torch.full((1,), 7.0)}

    module._weight = folded
    assert rebind(package, module) == 2

    installed = package.installs[-1]
    assert installed["w.weight"] is folded, "rebind kept the STALE pointer"
    assert torch.equal(installed["baked"], torch.full((1,), 7.0)), (
        "the trace-baked literal was dropped — it was never the module's to change"
    )


def test_rebind_refuses_a_package_it_did_not_load() -> None:
    from torchcg.adopt import rebind
    from torchcg.refuse import AdoptError

    class _Foreign:
        """A package torchcg never loaded — no stamped table at all."""

    with pytest.raises(AdoptError, match="not loaded by torchcg.adopt"):
        rebind(_Foreign(), _Module(torch.zeros(1)))


def test_rebind_refuses_when_the_module_LOST_a_declared_constant() -> None:
    """Silently binding a short table is how a graph ends up reading whatever
    happens to be at an address."""

    from torchcg.adopt import rebind
    from torchcg.refuse import AdoptError

    package = _Package()
    package._torchcg_retained = {"gone": torch.zeros(1)}
    package._torchcg_state_fqns = ("gone",)
    package._torchcg_literals = {}

    with pytest.raises(AdoptError, match="no longer holds"):
        rebind(package, _Module(torch.zeros(1)))
