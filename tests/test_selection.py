from __future__ import annotations

import json
from typing import Any

import pytest

from torchcg.contracts import read_contract
from torchcg.ingress import CallIngress, CallInput, exported_input_name
from torchcg.selection import (
    AOTI_ALIGNMENT,
    MISS_RUNGS,
    SELECTION_CONTRACT_FILE,
    SELECTION_CONTRACT_VERSION,
    GraphSpecializationCandidate,
    IngressMiss,
    MissReason,
    NormalizationKind,
    PresentedCall,
    PresentedValue,
    Selection,
    SelectionError,
    SelectionOutcome,
    SpecializationReport,
    class_report,
    decode_selection_corpus,
    describe_call,
    miss_distance,
    normalization_plan,
    realign_gap,
    recast_gap,
    select,
    selection_vectors,
)

# ---------------------------------------------------------------------------
# The corpus IS the contract: every vector is replayed through the reference
# implementation, and the corpus is refused unless it states this version and
# this implementation's constants.
# ---------------------------------------------------------------------------


VECTORS = selection_vectors()


@pytest.mark.parametrize("vector", VECTORS, ids=lambda vector: vector.name)
def test_every_vector_is_reproduced_by_the_reference_implementation(vector: Any) -> None:
    assert select(list(vector.candidates), vector.call) == vector.expect


def test_the_corpus_covers_every_outcome_reason_and_normalization() -> None:
    outcomes = {vector.expect.outcome for vector in VECTORS}
    assert outcomes == set(SelectionOutcome)
    reasons = {
        miss.reason
        for vector in VECTORS
        for report in vector.expect.ranked
        for miss in report.misses
    }
    assert reasons == set(MissReason)
    kinds = {
        row.kind for vector in VECTORS for row in vector.expect.normalizations
    }
    assert kinds == set(NormalizationKind)


def test_the_corpus_is_the_packaged_contract_file() -> None:
    raw = json.loads(read_contract(SELECTION_CONTRACT_FILE))
    assert raw["contract"] == "ingress_selection_v1"
    assert raw["v"] == SELECTION_CONTRACT_VERSION
    assert decode_selection_corpus(raw) == VECTORS


def _corpus() -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(read_contract(SELECTION_CONTRACT_FILE))
    return raw


def test_an_unknown_corpus_version_is_refused() -> None:
    raw = _corpus()
    raw["v"] = 2
    with pytest.raises(SelectionError, match="is not v1"):
        decode_selection_corpus(raw)


def test_an_unknown_corpus_contract_is_refused() -> None:
    raw = _corpus()
    raw["contract"] = "ingress_selection_v2"
    with pytest.raises(SelectionError, match="names contract"):
        decode_selection_corpus(raw)


def test_an_unknown_corpus_field_is_refused() -> None:
    raw = _corpus()
    raw["extra"] = 1
    with pytest.raises(SelectionError, match="fields must be exactly"):
        decode_selection_corpus(raw)


def test_a_corpus_stating_a_different_rung_table_is_refused() -> None:
    raw = _corpus()
    raw["rungs"]["dtype_mismatch"] = 7
    with pytest.raises(SelectionError, match="rung table"):
        decode_selection_corpus(raw)


def test_a_corpus_stating_a_different_recast_domain_is_refused() -> None:
    raw = _corpus()
    raw["normalizations"]["recast"]["targets"] = ["float32", "float64", "bfloat16"]
    with pytest.raises(SelectionError, match="recast domain"):
        decode_selection_corpus(raw)


def test_a_corpus_stating_a_different_alignment_is_refused() -> None:
    raw = _corpus()
    raw["normalizations"]["realign"]["alignment"] = 32
    with pytest.raises(SelectionError, match="realign domain"):
        decode_selection_corpus(raw)


def test_a_vector_stating_a_distance_its_misses_do_not_produce_is_refused() -> None:
    raw = _corpus()
    for vector in raw["vectors"]:
        if vector["expect"]["ranked"]:
            vector["expect"]["ranked"][0]["distance"] = [0]
            break
    with pytest.raises(SelectionError, match="states a distance"):
        decode_selection_corpus(raw)


# ---------------------------------------------------------------------------
# The rest is the surface the corpus cannot state: early exit, the adapter from
# a real call, and the refusals that keep the typed values canonical.
# ---------------------------------------------------------------------------


def _input(
    position: int,
    param: str,
    param_position: int,
    dtype: str,
    shape: tuple[int | str, ...],
    path: tuple[str, ...] = (),
) -> CallInput:
    name = param
    for step in path:
        name = step
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


BASE = CallIngress(
    parameters=("sample", "timestep", "encoder"),
    flat_arity=3,
    inputs=(
        _input(0, "sample", 0, "float16", ("batch", 4, 64, 64)),
        _input(1, "timestep", 1, "float32", ()),
        _input(2, "encoder", 2, "float16", ("batch", 77, 2048), ("text_embeds",)),
    ),
    symbols=(("batch", (1, 8)),),
    excluded_inputs=("lora_a", "lora_b"),
)


def _tensorish(dtype: str, shape: tuple[int, ...], *, ptr: int = 0, contig: bool = True) -> Any:
    class _Fake:
        def __init__(self) -> None:
            self.dtype = f"torch.{dtype}"
            self.shape = shape

        def is_contiguous(self) -> bool:
            return contig

        def data_ptr(self) -> int:
            return ptr

    return _Fake()


def _call(**over: PresentedValue) -> PresentedCall:
    feeds = {
        "sample": PresentedValue("float16", (2, 4, 64, 64)),
        "timestep": PresentedValue("float32", ()),
        "text_embeds": PresentedValue("float16", (2, 77, 2048)),
    }
    feeds.update(over)
    return PresentedCall(feeds=tuple(sorted(feeds.items())))


def test_first_only_is_an_early_exit_from_the_same_walk() -> None:
    """Not a second rule: the truncated report is a prefix of the full one."""
    call = _call(
        sample=PresentedValue("float32", (2, 4, 32, 32)),
        timestep=PresentedValue("bfloat16", ()),
    )
    full = class_report("denoiser", BASE, call)
    first = class_report("denoiser", BASE, call, first_only=True)
    assert len(full.misses) > 1
    assert first.misses == full.misses[:1]
    assert not first.admits and not full.admits


def test_an_admitting_class_reports_identically_under_both_walks() -> None:
    call = _call()
    assert class_report("denoiser", BASE, call) == class_report(
        "denoiser", BASE, call, first_only=True
    )


def test_miss_distance_orders_a_prefix_before_its_extension() -> None:
    shallow = (IngressMiss(MissReason.DTYPE_MISMATCH),)
    deep = (IngressMiss(MissReason.DTYPE_MISMATCH), IngressMiss(MissReason.RANK_MISMATCH))
    assert miss_distance(shallow) < miss_distance(deep)
    assert miss_distance(deep) < miss_distance((IngressMiss(MissReason.INPUT_MISSING),))


def test_the_rung_table_is_total_over_the_closed_reason_enum() -> None:
    assert set(MISS_RUNGS) == set(MissReason)


def test_candidate_order_does_not_change_the_outcome() -> None:
    call = _call(timestep=PresentedValue("bfloat16", ()))
    forward = [
        GraphSpecializationCandidate("denoiser.a", BASE),
        GraphSpecializationCandidate("denoiser.z", BASE),
    ]
    assert select(forward, call) == select(list(reversed(forward)), call)


def test_duplicate_candidate_names_are_refused() -> None:
    rows = [
        GraphSpecializationCandidate("denoiser", BASE),
        GraphSpecializationCandidate("denoiser", BASE),
    ]
    with pytest.raises(SelectionError, match="share one graph specialization name"):
        select(rows, _call())


def test_realignment_never_moves_admission_and_recast_does() -> None:
    misaligned = _call(timestep=PresentedValue("float32", (), alignment_offset=4))
    recastable = _call(timestep=PresentedValue("int64", ()))
    refused = _call(timestep=PresentedValue("bfloat16", ()))
    rows = [GraphSpecializationCandidate("denoiser", BASE)]
    assert select(rows, misaligned).outcome is SelectionOutcome.ADMITTED
    assert select(rows, recastable).outcome is SelectionOutcome.ADMITTED
    assert select(rows, refused).outcome is SelectionOutcome.NO_CLASS_ADMITS


def test_recast_subsumes_realignment_on_one_feed() -> None:
    call = _call(
        timestep=PresentedValue("int64", (), contiguous=False, alignment_offset=9)
    )
    plan = normalization_plan(BASE, call)
    assert len(plan) == 1
    assert plan[0].kind is NormalizationKind.RECAST
    assert plan[0].dtype == "float32"


def test_recast_and_realign_gaps_are_named_not_boolean() -> None:
    scalar = BASE.inputs[1]
    assert recast_gap(scalar, PresentedValue("int64", ())) == "int64_to_float32"
    assert recast_gap(scalar, PresentedValue("float32", ())) == ""
    assert recast_gap(BASE.inputs[0], PresentedValue("int64", (2, 4, 64, 64))) == ""
    assert realign_gap(PresentedValue("float32", (), alignment_offset=1)) == "unaligned_16b"
    assert realign_gap(PresentedValue("float32", (), contiguous=False)) == "non_contiguous"
    assert realign_gap(PresentedValue()) == ""


def test_describe_call_replays_the_recorded_coordinate() -> None:
    rows = [GraphSpecializationCandidate("denoiser", BASE)]
    call = describe_call(
        rows,
        (_tensorish("float16", (2, 4, 64, 64)), _tensorish("float32", (), ptr=4)),
        {"encoder": {"text_embeds": _tensorish("float16", (2, 77, 2048))}},
    )
    assert call.excluded_present == ()
    assert call.feed("timestep") == PresentedValue("float32", (), alignment_offset=4)
    assert call.feed("text_embeds") == PresentedValue("float16", (2, 77, 2048))
    outcome = select(rows, call)
    assert outcome.outcome is SelectionOutcome.ADMITTED
    assert outcome.symbols == (("batch", 2),)
    assert [row.reason for row in outcome.normalizations] == ["unaligned_16b"]


def test_describe_call_finds_a_nested_excluded_input() -> None:
    rows = [GraphSpecializationCandidate("denoiser", BASE)]
    call = describe_call(
        rows,
        (_tensorish("float16", (2, 4, 64, 64)), _tensorish("float32", ())),
        {
            "encoder": {"text_embeds": _tensorish("float16", (2, 77, 2048))},
            "cross_attention_kwargs": {"lora_a": _tensorish("float16", (8, 320))},
        },
    )
    assert call.excluded_present == ("lora_a",)
    assert select(rows, call).outcome is SelectionOutcome.NO_CLASS_ADMITS


def test_describe_call_reports_a_non_tensor_rather_than_omitting_it() -> None:
    rows = [GraphSpecializationCandidate("denoiser", BASE)]
    call = describe_call(
        rows,
        (_tensorish("float16", (2, 4, 64, 64)), 999),
        {"encoder": {"text_embeds": _tensorish("float16", (2, 77, 2048))}},
    )
    assert call.feed("timestep") == PresentedValue()
    ranked = select(rows, call).ranked
    assert [miss.reason for miss in ranked[0].misses] == [MissReason.INPUT_NOT_TENSOR]


def test_describe_call_omits_an_input_the_call_does_not_carry() -> None:
    rows = [GraphSpecializationCandidate("denoiser", BASE)]
    call = describe_call(rows, (_tensorish("float16", (2, 4, 64, 64)),), {})
    assert call.feed("timestep") is None
    ranked = select(rows, call).ranked
    assert [miss.reason for miss in ranked[0].misses] == [MissReason.INPUT_MISSING]


def test_describe_call_refuses_two_classes_that_spell_one_name_differently() -> None:
    other = CallIngress(
        parameters=("text_embeds",),
        flat_arity=1,
        inputs=(_input(0, "text_embeds", 0, "float16", ("batch", 77, 2048)),),
        symbols=(("batch", (1, 8)),),
    )
    rows = [
        GraphSpecializationCandidate("denoiser.a", BASE),
        GraphSpecializationCandidate("denoiser.b", other),
    ]
    with pytest.raises(SelectionError, match="different call coordinate"):
        describe_call(rows, (), {})


def test_a_selection_may_not_carry_a_field_its_outcome_forbids() -> None:
    with pytest.raises(SelectionError, match="only an admitted selection"):
        Selection(outcome=SelectionOutcome.NO_CLASS_ADMITS, selected="denoiser")
    with pytest.raises(SelectionError, match="exhaustive ranking"):
        Selection(
            outcome=SelectionOutcome.ADMITTED,
            selected="denoiser",
            ranked=(SpecializationReport("denoiser"),),
        )
    with pytest.raises(SelectionError, match="at least two classes"):
        Selection(outcome=SelectionOutcome.SPECIALIZATION_AMBIGUOUS, ambiguous=("denoiser",))


def test_a_presented_value_refuses_an_out_of_range_alignment_offset() -> None:
    with pytest.raises(SelectionError, match="alignment_offset"):
        PresentedValue("float32", (), alignment_offset=AOTI_ALIGNMENT)


def test_a_presented_call_refuses_unsorted_feeds() -> None:
    with pytest.raises(SelectionError, match="sorted and unique"):
        PresentedCall(
            feeds=(("z", PresentedValue("float32", ())), ("a", PresentedValue("float32", ())))
        )


def test_a_realign_normalization_may_not_declare_a_dtype() -> None:
    from torchcg.selection import FeedNormalization

    with pytest.raises(SelectionError, match="may not change dtype"):
        FeedNormalization(
            input="timestep",
            position=1,
            kind=NormalizationKind.REALIGN,
            reason="unaligned_16b",
            dtype="float32",
        )
    with pytest.raises(SelectionError, match="may only target"):
        FeedNormalization(
            input="timestep",
            position=1,
            kind=NormalizationKind.RECAST,
            reason="int64_to_bfloat16",
            dtype="bfloat16",
        )
