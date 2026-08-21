"""tcg#80 -- the compile policy is a KEY AXIS and a DECLARED FACT, not a literal.

Two defects, one root. The compiler ran with
``aot_inductor.use_runtime_constant_folding=True``, which materializes a
SECOND full constant set by direct ``cudaMalloc`` on the first compiled call
(``aoti_runtime/model_base.h:345``, ``model_container.h:169-186``) -- ~4.8 GiB
of first-call transient on sdxl UNet-only, which is why an 8 GiB card could
not serve it. And the two axes that describe that configuration were written
as literals by the same function that validated them, so a fold-on and a
fold-off artifact of the same graph shared one key and collided in the CAS.

Every red arm below is driven by making the SOURCE OF TRUTH disagree --
`compiler._COMPILER_OPTIONS`, the one place the policy is stated -- never by
handing a predicate an argument. That is the rule the three "guard that cannot
go red" findings earned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from tensorfs import LocalCAS

# The one real v1 metadata fixture, reused rather than re-spelled.
from test_artifact import metadata as _artifact_metadata

from torchcg import Engine, GraphSpecialization, RuntimeCompatibility, build_call_ingress
from torchcg import compiler as compiler_module
from torchcg.artifact import ArtifactError, validate_metadata
from torchcg.engine import AdmissionError
from torchcg.graph_identity import EnvIdentity

#: What ships. Stated here so a change to the shipped policy has to be made
#: twice, deliberately, and shows up as a diff in a file about the policy.
SHIPPED: dict[str, object] = {
    "compile_threads": 4,
    "aot_inductor.package_constants_in_so": False,
    "always_keep_tensor_constants": True,
    "aot_inductor.use_runtime_constant_folding": False,
    "aot_inductor.package": True,
}

#: The configuration this issue removes: the fold is deferred to LOAD, which
#: keeps every constant bindable but buys a second constant set on the card.
RUNTIME_FOLD: dict[str, object] = {**SHIPPED, "always_keep_tensor_constants": False,
                "aot_inductor.use_runtime_constant_folding": True}

#: Torch's default, and the one configuration no artifact may ship under: with
#: both flags off inductor INLINES a lifted constant's values into the kernel
#: whenever its shape is `()` or 1-D with <= 8 elements, and the artifact is
#: silently specific to the checkpoint it was minted from.
UNFENCED: dict[str, object] = {**SHIPPED, "always_keep_tensor_constants": False,
            "aot_inductor.use_runtime_constant_folding": False}

STACK = (("nvidia-cublas-cu13", "13.0.0"), ("torch", "2.13.0"), ("triton", "3.7.1"))


def _toolchain() -> dict[str, str]:
    return {"torch": "torch-record-v1", "triton": "triton-record-v1"}


def _policy(monkeypatch: pytest.MonkeyPatch, options: dict[str, object]) -> None:
    """Move the ONE place the compile policy is stated."""

    monkeypatch.setattr(compiler_module, "_COMPILER_OPTIONS", dict(options))


class SmallBias(torch.nn.Module):
    """The sdxl shape, in miniature.

    `conv_out.bias` is 1-D with 4 elements -- exactly the tensor the sdxl UNet
    mint reported as its ONE eliminated state-dict constant, and exactly what
    `GraphLowering.can_inline_constant` accepts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_out = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.conv_out(sample)
        return result


def _spec() -> GraphSpecialization:
    sample = torch.randn(1, 4, 8, 8)
    program = torch.export.export(SmallBias(), (sample,))
    ingress = build_call_ingress(program, ("sample",), (sample,), {})
    return GraphSpecialization("small-bias", "denoiser", program, ingress)


def _metadata(**overrides: Any) -> dict[str, Any]:
    """Real v1 metadata through the real builder -- `test_artifact`'s fixture.

    Reused rather than re-spelled: a second hand-built metadata shape is a
    second thing to drift, and this test is about the two fields the builder
    stamps.
    """

    return {**_artifact_metadata(), **overrides}


# -- the axes are READ from the build ----------------------------------------


def test_the_shipped_policy_is_weightless_and_leaves_every_constant_bindable() -> None:
    assert compiler_module._compiler_options() == SHIPPED
    assert compiler_module.constant_folding_fenced() is True
    assert compiler_module.package_constants_in_so() is False
    # `compile_threads` is a scheduling knob, not codegen: keying on it would
    # split the artifact pool on a value that cannot change the bytes.
    assert "compile_threads" not in compiler_module.compile_policy()


def test_a_stamped_axis_follows_the_config_and_cannot_be_a_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect, stated as a test: move the config, the stamp must move.

    Before tcg#80 this passed with the policy untouched AND with the policy
    inverted, because `build_metadata` wrote `False`/`True` verbatim.
    """

    shipped = _metadata()
    assert shipped["constant_folding_fenced"] is True
    assert shipped["compile_policy"] == compiler_module.compile_policy()

    _policy(monkeypatch, RUNTIME_FOLD)
    folded = _metadata()
    assert folded["constant_folding_fenced"] is True
    assert folded["compile_policy"]["aot_inductor.use_runtime_constant_folding"] is True
    assert folded["compile_policy"] != shipped["compile_policy"]


def test_an_unfenced_build_cannot_produce_a_valid_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pgw#1097 fence, now readable off the build that produced it."""

    _policy(monkeypatch, UNFENCED)
    with pytest.raises(ArtifactError, match="constant_folding_fenced must be true"):
        _metadata()


def test_a_build_that_packages_its_own_constants_cannot_produce_an_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the double residency, refused the same way."""

    _policy(monkeypatch, {**SHIPPED, "aot_inductor.package_constants_in_so": True})
    with pytest.raises(ArtifactError, match="package_constants_in_so must be false"):
        _metadata()


def test_a_stamp_that_contradicts_the_artifacts_own_policy_is_refused() -> None:
    """Two independent facts in one artifact, and they must agree."""

    metadata = _metadata()
    with pytest.raises(ArtifactError, match="but this artifact's own compile_policy"):
        validate_metadata({**metadata, "constant_folding_fenced": False})
    with pytest.raises(ArtifactError, match="but this artifact's own compile_policy"):
        validate_metadata({**metadata, "package_constants_in_so": True})


def test_an_unclassified_compile_option_is_a_refusal_not_a_silent_omission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new option nobody classified would leave the key blind to it."""

    _policy(monkeypatch, {**SHIPPED, "aot_inductor.emit_current_arch_binary": True})
    with pytest.raises(compiler_module.CompileError, match="neither codegen nor topology"):
        compiler_module.compile_policy()


# -- the policy reaches every key --------------------------------------------


def test_two_compile_policies_cannot_share_one_artifact_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured collision: identical graph, sm and toolchain; one key.

    Nothing here is hand-fed. The two arms differ only in
    `compiler._COMPILER_OPTIONS`, which is what a mint actually reads, and the
    keys are asked for exactly as the engine asks for them.
    """

    spec = _spec()
    declaration = spec.declare()
    runtime = RuntimeCompatibility("cpu", toolchain=_toolchain())

    shipped_key = runtime.key(declaration)
    shipped_env = EnvIdentity(stack=STACK, sm="sm_89")
    shipped_metadata = _metadata()

    _policy(monkeypatch, RUNTIME_FOLD)
    folded_key = RuntimeCompatibility("cpu", toolchain=_toolchain()).key(declaration)
    folded_env = EnvIdentity(stack=STACK, sm="sm_89")
    folded_metadata = _metadata()

    assert shipped_key != folded_key
    assert shipped_env.value != folded_env.value
    assert shipped_metadata["compiled_graph_key"] != folded_metadata["compiled_graph_key"]
    # ...and the graph, sm and toolchain halves are IDENTICAL, so the policy is
    # demonstrably the only thing that moved.
    assert shipped_key.as_dict()["graph"] == folded_key.as_dict()["graph"]
    assert shipped_key.as_dict()["sm"] == folded_key.as_dict()["sm"]
    assert shipped_key.as_dict()["toolchain"] == folded_key.as_dict()["toolchain"]
    assert dict(shipped_env.stack) == dict(folded_env.stack)


def test_an_env_names_its_options_in_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal that can say which option differs beats two hashes."""

    _policy(monkeypatch, RUNTIME_FOLD)
    described = EnvIdentity(stack=STACK, sm="sm_89").describe()
    assert "aot_inductor.use_runtime_constant_folding=True" in described


# -- the mechanism, on the installed compiler --------------------------------


@pytest.mark.real_aoti
def test_real_aoti_the_shipped_policy_bakes_no_weight_and_folds_nothing(
    tmp_path: Path,
) -> None:
    """Every state-dict constant survives as a BINDABLE row, and none is computed.

    The equality is the memory fix: a `computed` (`_FOLDED_CONST_*`) row is a
    second copy of a weight that the first compiled call materializes outside
    the caching allocator. There are none.
    """

    engine = Engine(LocalCAS(tmp_path / "cas"))
    module = SmallBias()
    sample = torch.randn(1, 4, 8, 8)
    program = torch.export.export(module, (sample,))
    spec = GraphSpecialization(
        "small-bias", "denoiser", program,
        build_call_ingress(program, ("sample",), (sample,), {}),
    )
    result = engine.compile(spec, RuntimeCompatibility("cpu", toolchain=_toolchain()),
                            tmp_path / "compiled")

    graph_specialization = result.compiled_graph.metadata["graph_specialization"]
    assert isinstance(graph_specialization, dict)
    rows = graph_specialization["constants"]
    assert {str(row["fqn"]) for row in rows} == {"conv_out.weight", "conv_out.bias"}
    assert {str(row["source"]) for row in rows} == {"state_dict"}

    runner = engine.runner(result.compiled_graph.key, tmp_path / "loaded")
    assert runner is not None
    runner.bind(dict(module.state_dict()), device="cpu")
    torch.testing.assert_close(runner(sample), module(sample))


@pytest.mark.real_aoti
def test_real_aoti_an_unfenced_policy_compiles_a_weight_into_the_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RED arm of the same mint, driven by the config alone.

    This is the on-card measurement reproduced at micro scale: with both flags
    off, inductor eliminates the 4-element `conv_out.bias` -- the sdxl mint
    named that exact fqn -- and torchcg's admission refuses the package before
    it can be packed. The refusal is what makes "just set the flag to False"
    the wrong fix and this issue's policy the right one.
    """

    _policy(monkeypatch, UNFENCED)
    engine = Engine(LocalCAS(tmp_path / "cas"))
    with pytest.raises(AdmissionError, match="conv_out.bias"):
        engine.compile(_spec(), RuntimeCompatibility("cpu", toolchain=_toolchain()),
                       tmp_path / "compiled")
