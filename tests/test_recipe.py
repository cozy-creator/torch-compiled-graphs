from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from torchcg.contracts import read_contract
from torchcg.identity import from_axes, toolchain_axis_digest
from torchcg.ingress import CallIngress, CallInput
from torchcg.recipe import (
    IDENTIFIER_GRAMMAR,
    RESERVED_IDENTIFIERS,
    BucketAxis,
    DeclaredRunner,
    GraphClassVariant,
    Loop,
    LoopKind,
    LoopStep,
    ParameterKind,
    Recipe,
    RecipeError,
    RecipeParameter,
    RecipeReference,
    RecipeRefusal,
    RecipeRunner,
    RepeatKind,
    Scheduler,
    SessionState,
    bucket_of,
    call_signature,
    parse_bucket_axis_name,
    parse_class_hash,
    parse_family_name,
    parse_ingress_digest,
    parse_layout_contract,
    parse_parameter_name,
    parse_recipe_digest,
    parse_runner_name,
    parse_scheduler_name,
)


class _Runtime:
    """A RuntimeCompatibility-shaped stand-in; the real one needs a live device."""

    def __init__(self, sm: str, toolchain: dict[str, str]) -> None:
        self.sm = sm
        self.toolchain = toolchain


def _ingress(
    parameters: tuple[str, ...],
    rows: tuple[tuple[str, tuple[str | int, ...], str, tuple[int | str, ...]], ...],
    symbols: tuple[tuple[str, tuple[int, int]], ...] = (),
) -> CallIngress:
    inputs: list[CallInput] = []
    for position, (param, path, dtype, shape) in enumerate(rows):
        name = param
        for step in path:
            name = step if isinstance(step, str) else f"{name}.{step}"
        inputs.append(
            CallInput(
                name=name,
                position=position,
                param=param,
                param_position=parameters.index(param),
                path=path,
                exported_name="_".join((param, *(str(step) for step in path))),
                dtype=dtype,
                shape=shape,
            )
        )
    return CallIngress(
        parameters=parameters,
        flat_arity=len(rows),
        inputs=tuple(inputs),
        symbols=symbols,
    )


_TOKENS = (("tokens", (1, 77)),)


def _text_encoder_ingress() -> CallIngress:
    return _ingress(("input_ids",), (("input_ids", (), "int64", (1, "tokens")),), _TOKENS)


def _denoiser_ingress(side: int) -> CallIngress:
    return _ingress(
        ("latents", "timestep", "conditioning"),
        (
            ("latents", (), "float16", (1, 4, side, side)),
            ("timestep", (), "float32", (1,)),
            ("conditioning", ("prompt",), "float16", (1, "tokens", 768)),
            ("conditioning", ("negative_prompt",), "float16", (1, "tokens", 768)),
        ),
        _TOKENS,
    )


def _decoder_ingress(side: int) -> CallIngress:
    return _ingress(("latents",), (("latents", (), "float16", (1, 4, side, side)),))


_CLASS_HASHES = {
    ("text_encoder", 0, "bf16"): "1c0e7a4b39d5f682",
    ("denoiser", 64, "bf16"): "2f91b8c40ae7d135",
    ("denoiser", 128, "bf16"): "3ab47de205c1986f",
    ("denoiser", 64, "fp8_rowwise"): "8b1d5f36c07a294e",
    ("denoiser", 128, "fp8_rowwise"): "9c2e604718fb35a0",
    ("decoder", 64, "bf16"): "4d5c1e93b6720af8",
    ("decoder", 128, "bf16"): "5e6f2a08c9143b7d",
}

_BF16 = parse_layout_contract("bf16")
_FP8 = parse_layout_contract("fp8_rowwise")


def _variant(
    runner: str, side: int, ingress: CallIngress, layout: str = "bf16"
) -> GraphClassVariant:
    return GraphClassVariant(
        class_hash=parse_class_hash(_CLASS_HASHES[(runner, side, layout)]),
        ingress_digest=parse_ingress_digest(ingress.digest()),
        ingress=ingress,
        layout=parse_layout_contract(layout),
        bucket=bucket_of((("resolution", side),)) if side else (),
    )


def canonical_recipe() -> Recipe:
    """The exact document shipped as ``contracts/recipe_v1.json``.

    This builder is the corpus's source. Regenerate the corpus by replacing its
    ``recipe``/``digest``/``reference`` members from this recipe and writing the
    file as ``json.dumps(corpus, indent=2, sort_keys=True) + "\\n"`` in ASCII;
    the ``digest`` covers only the ``recipe`` member, so prose members may move
    without rekeying a consumer's pin.
    """

    return Recipe(
        family=parse_family_name("toy_diffusion"),
        buckets=(BucketAxis(name=parse_bucket_axis_name("resolution"), values=(64, 128)),),
        runners=(
            RecipeRunner(
                name=parse_runner_name("decoder"),
                axes=(parse_bucket_axis_name("resolution"),),
                variants=tuple(
                    _variant("decoder", side, _decoder_ingress(side)) for side in (64, 128)
                ),
            ),
            RecipeRunner(
                name=parse_runner_name("denoiser"),
                axes=(parse_bucket_axis_name("resolution"),),
                variants=tuple(
                    _variant("denoiser", side, _denoiser_ingress(side), layout)
                    for side in (64, 128)
                    for layout in ("bf16", "fp8_rowwise")
                ),
            ),
            RecipeRunner(
                name=parse_runner_name("text_encoder"),
                axes=(),
                variants=(_variant("text_encoder", 0, _text_encoder_ingress()),),
            ),
        ),
        loop=Loop(
            kind=LoopKind.STAGED,
            stages=(
                LoopStep(runner=parse_runner_name("text_encoder")),
                LoopStep(
                    runner=parse_runner_name("denoiser"),
                    repeat=RepeatKind.COUNTED,
                    parameter=parse_parameter_name("steps"),
                ),
                LoopStep(runner=parse_runner_name("decoder")),
            ),
        ),
        parameters=(
            RecipeParameter(name=parse_parameter_name("steps"), minimum=1, maximum=100),
        ),
        scheduler=Scheduler(
            name=parse_scheduler_name("euler_discrete"),
            parameters=(
                (parse_parameter_name("num_train_timesteps"), 1000),
                (parse_parameter_name("shift"), 3.0),
                (parse_parameter_name("use_karras_sigmas"), False),
            ),
        ),
    )


_AR_CLASS_HASHES = {"prefill": "6f7a3b1c8d20e594", "decode": "7809c4d21be3f6a5"}


def autoregressive_recipe() -> Recipe:
    """The second document in the corpus: an AR family whose loop is host-owned.

    LLMs and VLMs are first-class, and their iteration is data-dependent — it
    runs until the model says stop.  No count in a document can say that, so the
    recipe says ``kind: host`` instead of pretending.
    """

    prefill = _ingress(
        ("input_ids",),
        (("input_ids", (), "int64", (1, "prompt_tokens")),),
        (("prompt_tokens", (1, 4096)),),
    )
    decode = _ingress(
        ("input_ids", "kv_cache"),
        (
            ("input_ids", (), "int64", (1, 1)),
            ("kv_cache", (0,), "float16", (1, 8, "context", 64)),
            ("kv_cache", (1,), "float16", (1, 8, "context", 64)),
        ),
        (("context", (1, 4096)),),
    )
    return Recipe(
        family=parse_family_name("toy_llm"),
        buckets=(),
        runners=(
            RecipeRunner(
                name=parse_runner_name("decode"),
                axes=(),
                variants=(
                    GraphClassVariant(
                        class_hash=parse_class_hash(_AR_CLASS_HASHES["decode"]),
                        ingress_digest=parse_ingress_digest(decode.digest()),
                        ingress=decode,
                        layout=_BF16,
                    ),
                ),
            ),
            RecipeRunner(
                name=parse_runner_name("prefill"),
                axes=(),
                variants=(
                    GraphClassVariant(
                        class_hash=parse_class_hash(_AR_CLASS_HASHES["prefill"]),
                        ingress_digest=parse_ingress_digest(prefill.digest()),
                        ingress=prefill,
                        layout=_BF16,
                    ),
                ),
            ),
        ),
        loop=Loop(
            kind=LoopKind.HOST,
            stages=(
                LoopStep(runner=parse_runner_name("prefill")),
                LoopStep(runner=parse_runner_name("decode")),
            ),
            session_state=SessionState.HOST,
        ),
    )


def _corpus() -> dict[str, Any]:
    corpus = json.loads(read_contract("recipe_v1.json"))
    assert isinstance(corpus, dict)
    return corpus


def _document() -> dict[str, Any]:
    document = json.loads(json.dumps(canonical_recipe().as_dict()))
    assert isinstance(document, dict)
    return document


def test_shipped_corpus_is_the_canonical_document() -> None:
    corpus = _corpus()
    recipe = canonical_recipe()
    assert corpus["authority"] == "torchcg recipe v1"
    assert corpus["recipe"] == recipe.as_dict()
    assert corpus["digest"] == str(recipe.digest())
    assert corpus["reference"] == recipe.reference().as_dict()
    autoregressive = autoregressive_recipe()
    assert corpus["host_loop_recipe"] == autoregressive.as_dict()
    assert corpus["host_loop_digest"] == str(autoregressive.digest())
    assert corpus["identifier_grammar"] == IDENTIFIER_GRAMMAR
    assert corpus["reserved_identifiers"] == sorted(RESERVED_IDENTIFIERS)
    assert [row["reason"] for row in corpus["refusals"]] == sorted(
        reason.value for reason in RecipeRefusal
    )


def test_document_round_trips_through_decode_and_bytes() -> None:
    recipe = canonical_recipe()
    assert Recipe.decode(recipe.as_dict()) == recipe
    assert Recipe.loads(recipe.canonical()) == recipe
    assert Recipe.loads(recipe.canonical()).digest() == recipe.digest()


def test_reference_pins_the_document_and_refuses_a_stranger() -> None:
    recipe = canonical_recipe()
    reference = recipe.reference()
    assert RecipeReference.decode(reference.as_dict()) == reference
    recipe.verify(reference)
    stranger = RecipeReference(
        family=recipe.family, digest=parse_recipe_digest("0" * 32)
    )
    with pytest.raises(RecipeError) as caught:
        recipe.verify(stranger)
    assert caught.value.reason is RecipeRefusal.REFERENCE_MISMATCH


def test_recipe_digest_is_machine_independent_and_the_key_is_not() -> None:
    recipe = canonical_recipe()
    variant = recipe.runner(parse_runner_name("denoiser")).variant(
        {parse_bucket_axis_name("resolution"): 64}, _BF16
    )
    ampere = _Runtime("sm_86", {"torch": "a" * 16})
    hopper = _Runtime("sm_90", {"torch": "a" * 16})
    rebuilt = _Runtime("sm_86", {"torch": "b" * 16})
    assert variant.key(ampere) != variant.key(hopper)
    assert variant.key(ampere) != variant.key(rebuilt)
    assert variant.key(ampere) == from_axes(
        {
            "graph": str(variant.class_hash),
            "sm": "sm_86",
            "toolchain": toolchain_axis_digest({"torch": "a" * 16}),
        }
    )
    assert str(variant.key(ampere)).startswith("cg-key-v1-")
    # The document carries no machine axis at all, so no SKU can rekey the pin.
    assert "sm_" not in recipe.canonical().decode("ascii")


def test_the_document_cannot_state_a_checkpoint_level_fact() -> None:
    """The class-level layer stays checkpoint-free by closed field sets, not by habit."""

    recipe = canonical_recipe()
    text = recipe.canonical().decode("ascii")
    for forbidden in ("checkpoint", "weights", "state_dict", "tuned", "default", "lora"):
        assert forbidden not in text
    intrusions: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("document", lambda document: document.update({"checkpoint": "flux/dev"})),
        ("runner", lambda document: document["runners"][0].update({"weights": "a.safetensors"})),
        ("variant", lambda document: document["runners"][0]["variants"][0].update({"tuned": {}})),
        ("scheduler", lambda document: document["scheduler"].update({"defaults": {}})),
    )
    for level, intrude in intrusions:
        document = _document()
        intrude(document)
        with pytest.raises(RecipeError) as caught:
            Recipe.decode(document)
        assert caught.value.reason is RecipeRefusal.RECIPE_FIELDS_INVALID, level


def test_a_host_owned_loop_is_a_legal_declaration() -> None:
    recipe = autoregressive_recipe()
    assert recipe.loop.kind is LoopKind.HOST
    assert recipe.loop.session_state is SessionState.HOST
    assert [step.runner for step in recipe.loop.stages] == ["prefill", "decode"]
    assert Recipe.loads(recipe.canonical()) == recipe
    assert recipe.digest() != canonical_recipe().digest()
    # The per-step classes and the session-state owner are stated; the iteration
    # is not, and the vocabulary refuses to fake it with a count.
    with pytest.raises(RecipeError) as caught:
        Loop(
            kind=LoopKind.HOST,
            stages=(
                LoopStep(
                    runner=parse_runner_name("decode"),
                    repeat=RepeatKind.COUNTED,
                    parameter=parse_parameter_name("steps"),
                ),
            ),
            session_state=SessionState.HOST,
        )
    assert caught.value.reason is RecipeRefusal.LOOP_INVALID


def test_bucket_lookup_is_exact_and_never_ranks() -> None:
    runner = canonical_recipe().runner(parse_runner_name("denoiser"))
    resolution = parse_bucket_axis_name("resolution")
    assert runner.variant({resolution: 128}, _BF16).class_hash == _CLASS_HASHES[
        ("denoiser", 128, "bf16")
    ]
    assert runner.variant({resolution: 128}, _FP8).class_hash == _CLASS_HASHES[
        ("denoiser", 128, "fp8_rowwise")
    ]
    with pytest.raises(RecipeError) as caught:
        runner.variant({resolution: 96}, _BF16)
    assert caught.value.reason is RecipeRefusal.BUCKET_INVALID


def test_call_signature_is_the_generation_projection() -> None:
    runner = canonical_recipe().runner(parse_runner_name("denoiser"))
    signature = runner.signature
    assert signature.flat_arity == 4
    assert [parameter.name for parameter in signature.parameters] == [
        "latents",
        "timestep",
        "conditioning",
    ]
    assert [parameter.kind for parameter in signature.parameters] == [
        ParameterKind.TENSOR,
        ParameterKind.TENSOR,
        ParameterKind.MAPPING,
    ]
    conditioning = signature.parameters[2]
    assert [leaf.name for leaf in conditioning.leaves] == ["prompt", "negative_prompt"]
    assert {leaf.rank for leaf in conditioning.leaves} == {3}
    # Every variant of one runner projects to one signature: that is what makes a
    # single generated binding well defined across buckets.
    assert {call_signature(variant.ingress) for variant in runner.variants} == {signature}
    absent = call_signature(
        CallIngress(
            parameters=("latents", "generator"),
            flat_arity=1,
            inputs=_denoiser_ingress(64).inputs[:1],
        )
    )
    assert absent.parameters[1].kind is ParameterKind.ABSENT
    assert absent.parameters[1].leaves == ()


def test_sequence_parameters_project_as_sequences() -> None:
    signature = call_signature(
        _ingress(
            ("prompt_embeds",),
            (
                ("prompt_embeds", (0,), "float16", (1, 8)),
                ("prompt_embeds", (1,), "float16", (1, 8)),
            ),
        )
    )
    assert signature.parameters[0].kind is ParameterKind.SEQUENCE


def test_every_generated_symbol_is_a_legal_identifier_in_both_languages() -> None:
    for reserved in ("type", "match", "impl", "loop", "fn"):
        assert reserved in RESERVED_IDENTIFIERS
        with pytest.raises(RecipeError) as caught:
            parse_runner_name(reserved)
        assert caught.value.reason is RecipeRefusal.IDENTIFIER_RESERVED
    for illegal in ("Denoiser", "de-noiser", "2denoiser", "", "de noiser", "x" * 49):
        with pytest.raises(RecipeError) as caught:
            parse_runner_name(illegal)
        assert caught.value.reason is RecipeRefusal.IDENTIFIER_INVALID


def test_layout_is_an_axis_of_the_class_row() -> None:
    """fp8-rowwise and bf16 traces are different graphs, so the row names which."""

    runner = canonical_recipe().runner(parse_runner_name("denoiser"))
    assert runner.layouts == ("bf16", "fp8_rowwise")
    resolution = parse_bucket_axis_name("resolution")
    assert runner.variant({resolution: 64}, _BF16).class_hash != runner.variant(
        {resolution: 64}, _FP8
    ).class_hash
    # Two layouts and no layout named is a refusal, never a guess: choosing one
    # is the hub's join with per-checkpoint layout metadata, a separate document.
    with pytest.raises(RecipeError) as caught:
        runner.variant({resolution: 64})
    assert caught.value.reason is RecipeRefusal.LAYOUT_INVALID
    # A runner with one layout resolves without naming it; that is not a choice.
    decoder = canonical_recipe().runner(parse_runner_name("decoder"))
    assert decoder.layouts == ("bf16",)
    assert decoder.variant({resolution: 64}).layout == _BF16


def test_the_recipe_asserts_the_declaration_rather_than_replacing_it() -> None:
    """Bindings generate from the declaration; this document proves no drift."""

    recipe = canonical_recipe()
    declared = [
        DeclaredRunner(
            name=runner.name,
            ingress_digests=tuple(variant.ingress_digest for variant in runner.variants),
        )
        for runner in recipe.runners
    ]
    recipe.assert_declaration(declared)
    drifted = [
        DeclaredRunner(
            name=runner.name,
            ingress_digests=(
                (parse_ingress_digest("0" * 32),)
                if runner.name == "denoiser"
                else tuple(variant.ingress_digest for variant in runner.variants)
            ),
        )
        for runner in recipe.runners
    ]
    with pytest.raises(RecipeError) as caught:
        recipe.assert_declaration(drifted)
    assert caught.value.reason is RecipeRefusal.DECLARATION_DRIFT
    assert "denoiser" in str(caught.value)
    with pytest.raises(RecipeError) as absent:
        recipe.assert_declaration(declared[:2])
    assert absent.value.reason is RecipeRefusal.DECLARATION_DRIFT


def _mutate(**changes: Any) -> Callable[[], object]:
    def build() -> object:
        document = _document()
        document.update(changes)
        return Recipe.decode(document)

    return build


def _patch(mutate: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
    def build() -> object:
        document = _document()
        mutate(document)
        return Recipe.decode(document)

    return build


def _drop_loop_stage(document: dict[str, Any]) -> None:
    document["loop"]["stages"] = [
        step for step in document["loop"]["stages"] if step["runner"] != "decoder"
    ]


def _unknown_stage(document: dict[str, Any]) -> None:
    document["loop"]["stages"][2]["runner"] = "vae"


def _swap_runners(document: dict[str, Any]) -> None:
    document["runners"][0], document["runners"][1] = (
        document["runners"][1],
        document["runners"][0],
    )


def _undeclared_bucket_value(document: dict[str, Any]) -> None:
    for variant in document["runners"][1]["variants"][2:]:
        variant["bucket"]["resolution"] = 96


def _broken_ingress(document: dict[str, Any]) -> None:
    document["runners"][0]["variants"][0]["ingress"]["v"] = 2


def _short_digest(document: dict[str, Any]) -> None:
    document["runners"][0]["variants"][0]["ingress_digest"] = "abc"


def _wrong_digest(document: dict[str, Any]) -> None:
    document["runners"][0]["variants"][0]["ingress_digest"] = "0" * 32


def _unknown_repeat(document: dict[str, Any]) -> None:
    document["loop"]["stages"][1]["repeat"] = "twice"


def _bad_bounds(document: dict[str, Any]) -> None:
    document["parameters"][0]["minimum"] = 0


def _unknown_parameter(document: dict[str, Any]) -> None:
    document["loop"]["stages"][1]["parameter"] = "cycles"


def _idle_parameter(document: dict[str, Any]) -> None:
    document["loop"]["stages"][1] = {"runner": "denoiser", "repeat": "once"}


def _bad_scheduler(document: dict[str, Any]) -> None:
    document["scheduler"]["parameters"] = []


def _unsorted_axis(document: dict[str, Any]) -> None:
    document["buckets"][0]["values"] = [128, 64]


def _missing_field(document: dict[str, Any]) -> None:
    del document["loop"]


def _disagreeing_variants() -> object:
    left = _denoiser_ingress(64)
    right = _ingress(
        ("latents", "timestep", "conditioning"),
        (
            ("latents", (), "float32", (1, 4, 128, 128)),
            ("timestep", (), "float32", (1,)),
            ("conditioning", ("prompt",), "float16", (1, "tokens", 768)),
            ("conditioning", ("negative_prompt",), "float16", (1, "tokens", 768)),
        ),
        _TOKENS,
    )
    return RecipeRunner(
        name=parse_runner_name("denoiser"),
        axes=(parse_bucket_axis_name("resolution"),),
        variants=(_variant("denoiser", 64, left), _variant("denoiser", 128, right)),
    )


def _drop_one_layout(document: dict[str, Any]) -> None:
    variants = document["runners"][1]["variants"]
    document["runners"][1]["variants"] = [variants[0], variants[1], variants[2]]


_REFUSALS: dict[RecipeRefusal, Callable[[], object]] = {
    RecipeRefusal.LAYOUT_INVALID: lambda: canonical_recipe()
    .runner(parse_runner_name("denoiser"))
    .variant({parse_bucket_axis_name("resolution"): 64}),
    RecipeRefusal.DECLARATION_DRIFT: lambda: canonical_recipe().assert_declaration(
        [
            DeclaredRunner(
                name=parse_runner_name("denoiser"),
                ingress_digests=(parse_ingress_digest("0" * 32),),
            )
        ]
    ),
    RecipeRefusal.RECIPE_VERSION_UNSUPPORTED: _mutate(v=2),
    RecipeRefusal.RECIPE_FIELDS_INVALID: _patch(_missing_field),
    RecipeRefusal.IDENTIFIER_INVALID: _mutate(family="Toy Diffusion"),
    RecipeRefusal.IDENTIFIER_RESERVED: _mutate(family="type"),
    RecipeRefusal.DIGEST_INVALID: _patch(_short_digest),
    RecipeRefusal.BUCKET_AXIS_INVALID: _patch(_unsorted_axis),
    RecipeRefusal.BUCKET_INVALID: _patch(_undeclared_bucket_value),
    RecipeRefusal.BUCKET_COVERAGE_INCOMPLETE: _patch(_drop_one_layout),
    RecipeRefusal.INGRESS_INVALID: _patch(_broken_ingress),
    RecipeRefusal.INGRESS_DIGEST_MISMATCH: _patch(_wrong_digest),
    RecipeRefusal.SIGNATURE_DISAGREEMENT: _disagreeing_variants,
    RecipeRefusal.RUNNER_INVALID: _patch(_swap_runners),
    RecipeRefusal.RUNNER_UNKNOWN: _patch(_unknown_stage),
    RecipeRefusal.RUNNER_UNUSED: _patch(_drop_loop_stage),
    RecipeRefusal.LOOP_INVALID: _patch(_unknown_repeat),
    RecipeRefusal.PARAMETER_INVALID: _patch(_bad_bounds),
    RecipeRefusal.PARAMETER_UNKNOWN: _patch(_unknown_parameter),
    RecipeRefusal.PARAMETER_UNUSED: _patch(_idle_parameter),
    RecipeRefusal.SCHEDULER_INVALID: _patch(_bad_scheduler),
    RecipeRefusal.REFERENCE_MISMATCH: lambda: canonical_recipe().verify(
        RecipeReference(
            family=canonical_recipe().family, digest=parse_recipe_digest("f" * 32)
        )
    ),
}


def test_every_declared_refusal_is_reachable() -> None:
    assert set(_REFUSALS) == set(RecipeRefusal)


@pytest.mark.parametrize("reason", sorted(RecipeRefusal))
def test_refusal_reason_is_exact(reason: RecipeRefusal) -> None:
    with pytest.raises(RecipeError) as caught:
        _REFUSALS[reason]()
    assert caught.value.reason is reason
    assert str(caught.value).startswith(f"{reason.value}: ")


def test_reading_a_recipe_never_imports_torch() -> None:
    probe = (
        "import sys, torchcg.recipe;"
        "assert 'torch' not in sys.modules, sorted(sys.modules)[:0] or 'torch imported'"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
