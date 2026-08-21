"""tcg#83 -- the DECLARED INPUT LAYOUT is a key axis, a declared fact, and a bind contract.

The graph axis is stride-BLIND: `declaration._canonical_graph` renders every
tensor as ``t(dtype|[shape]|device)`` and never a stride. So a contiguous mint
and a channels_last mint of ONE module agree on graph, sm, toolchain and
compile policy while emitting different `.so` bytes -- one wrapper carries a
per-call permute kernel for the conv weight and the other does not. That is the
tcg#80 collision again, with strides instead of flags.

Every arm below moves the SOURCE OF TRUTH -- `compiler._DECLARED_INPUT_LAYOUT`
for the declaration, and for the vocabulary the RATIFIED RECORDS THEMSELVES
(tcg#87: `layout.py` holds no handle, no rank and no permutation of its own, so
the only way to move the vocabulary is to move a document) -- and never hands a
predicate an argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from tensorfs import LocalCAS

# The one real v1 metadata fixture, reused rather than re-spelled.
from test_artifact import metadata as _artifact_metadata

from torchcg import Engine, GraphSpecialization, RuntimeCompatibility, build_call_ingress
from torchcg import compiler as compiler_module
from torchcg import layout as layout_module
from torchcg.artifact import ArtifactError
from torchcg.engine import AdmissionError
from torchcg.graph_identity import EnvIdentity
from torchcg.layout import LayoutError, LayoutUndeliverableError, contiguous_handle
from torchcg.runner import ConstantBindingError

#: Read from the corpus at call time, never spelled: see `contiguous_handle`.
CONTIGUOUS = contiguous_handle()
CHANNELS_LAST = "torch.channels_last-2d@1"
STACK = (("nvidia-cublas-cu13", "13.0.0"), ("torch", "2.13.0"), ("triton", "3.7.1"))


_CORPUS = Path(__file__).resolve().parent / "testdata" / "spec" / "v2"


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the ratified corpus, so an arm can move the SOURCE.

    Never the real one: an arm that edited the checked-in records would leave
    the suite's later tests reading a mutated vocabulary.
    """

    root = tmp_path / "corpus" / "v2"
    (root / "layouts").mkdir(parents=True)
    for path in (_CORPUS / "layouts").glob("*.json"):
        (root / "layouts" / path.name).write_bytes(path.read_bytes())
    monkeypatch.setenv("TENSORFS_SPEC_V2", str(root))
    layout_module._paired.cache_clear()
    return root / "layouts"


def _toolchain() -> dict[str, str]:
    return {"torch": "torch-record-v1", "triton": "triton-record-v1"}


def _declare(monkeypatch: pytest.MonkeyPatch, handle: str) -> None:
    """Move the ONE place the declared input layout is stated."""

    monkeypatch.setattr(compiler_module, "_DECLARED_INPUT_LAYOUT", handle)


class Conv(torch.nn.Module):
    """A 4-D weight is the whole subject: 3-D and 1-D tensors have one layout."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = torch.nn.functional.silu(self.conv(sample))
        return result


def _module_and_sample(*, channels_last: bool) -> tuple[Conv, torch.Tensor]:
    module = Conv()
    sample = torch.randn(2, 8, 16, 16)
    if channels_last:
        # `Module.to` accepts a memory format at runtime; its stubs do not.
        module = cast(Conv, cast(Any, module).to(memory_format=torch.channels_last))
        sample = sample.to(memory_format=torch.channels_last)
    return module, sample


def _spec(*, channels_last: bool = False) -> tuple[GraphSpecialization, Conv, torch.Tensor]:
    module, sample = _module_and_sample(channels_last=channels_last)
    program = torch.export.export(module, (sample,))
    ingress = build_call_ingress(program, ("sample",), (sample,), {})
    return GraphSpecialization("conv", "denoiser", program, ingress), module, sample


def _metadata(**overrides: Any) -> dict[str, Any]:
    return {**_artifact_metadata(), **overrides}


# -- the vocabulary is CLOSED, and a wish outside it is never invented --------


def test_the_shipped_declaration_is_the_stored_layout() -> None:
    """Row-major, which is what every existing tree and artifact is."""

    assert compiler_module.declared_input_layout() == CONTIGUOUS
    assert sorted(layout_module.catalog()) == [
        "torch.channels_last-2d@1",
        "torch.channels_last-3d@1",
        "torch.contiguous@1",
    ]


def test_the_deliverable_catalog_is_a_SUBSET_of_what_is_ratified() -> None:
    """tcg#87. The corpus ratifies seven arrangements; this compiler can deliver
    three of them, because a torch memory format produces those three and
    nothing in torch produces the rest. The other four are not errors and not
    candidates -- they are ratified layouts that reach a tensor through the
    tensorfs fill path instead of through a declaration here."""

    from tensorfs import layouts

    ratified = set(layouts(_CORPUS))
    deliverable = set(layout_module.catalog())
    assert deliverable < ratified
    undeliverable = ratified - deliverable
    assert undeliverable == {
        "cublas.blockscale-128x4@1",
        "nunchaku.micro-scale@1",
        "torch.stride-padding-16@1",
        "torch.transposed@1",
    }

    # And the refusal SAYS WHICH KIND OF MISS IT IS, because the remedies
    # differ: ratify a record, versus deliver it in flight.
    with pytest.raises(LayoutUndeliverableError, match="IS ratified"):
        layout_module.require_morphism("nunchaku.micro-scale@1")
    with pytest.raises(LayoutError, match="not a ratified layout morphism"):
        layout_module.require_morphism("cozy.invented@1")


def test_the_pairing_is_ARITHMETIC_and_a_moved_record_stops_pairing(
    corpus: Path,
) -> None:
    """THE ONE-PRODUCER FENCE, red-proven by moving the SOURCE.

    torchcg pairs a torch memory format with a record only when the permutation
    torch produces at that rank EQUALS the permutation the record states. So
    perturbing the record -- the ratifying document, not a predicate's argument
    -- must drop the pairing, not be quietly absorbed by a literal that says
    channels_last anyway.
    """

    import json

    record = corpus / "torch.channels_last-2d.v1.json"
    document = json.loads(record.read_text(encoding="utf-8"))
    assert document["permutation"] == [0, 2, 3, 1]
    document["permutation"] = [0, 3, 2, 1]
    record.write_text(json.dumps(document), encoding="utf-8")
    layout_module._paired.cache_clear()

    assert CHANNELS_LAST not in layout_module.catalog()
    with pytest.raises(LayoutUndeliverableError):
        layout_module.require_morphism(CHANNELS_LAST)


def test_no_corpus_is_a_NAMED_refusal_never_an_empty_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty catalog reads as "nothing is ratified", and the caller's next
    move is to fall back to row-major -- silently undoing the delivery this
    whole axis exists to make possible."""

    from torchcg.layout import CORPUS_ENV, LayoutCorpusError

    monkeypatch.setenv(CORPUS_ENV, str(tmp_path / "nowhere"))
    layout_module._paired.cache_clear()
    with pytest.raises(LayoutCorpusError, match="no ratified layout corpus"):
        layout_module.catalog()
    with pytest.raises(LayoutCorpusError):
        layout_module.require_morphism("torch.contiguous@1")


def test_an_unratified_layout_is_a_refusal_not_a_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint-declared class is real, and torchcg may not invent it."""

    from torchcg.artifact import validate_metadata

    stored = _metadata()
    _declare(monkeypatch, "bfl.nvfp4-preswizzled@1")
    # A build cannot state it...
    with pytest.raises(LayoutError, match="not a ratified layout morphism"):
        compiler_module.declared_input_layout()
    with pytest.raises(LayoutError, match="not a ratified layout morphism"):
        _metadata()
    # ...and an artifact from elsewhere that carries it cannot be read.
    with pytest.raises(ArtifactError, match="not a ratified layout morphism"):
        validate_metadata({**stored, "declared_input_layout": "bfl.nvfp4-preswizzled@1"})


def test_a_catalog_morphism_is_torchs_own_memory_format_not_a_copied_order() -> None:
    """The permutation lives in torch; the catalog only NAMES it."""

    from torch._inductor.ir import NHWC_STRIDE_ORDER, NHWDC_STRIDE_ORDER

    catalog = layout_module.catalog()
    assert catalog[CHANNELS_LAST].stride_order(4) == tuple(NHWC_STRIDE_ORDER)
    assert catalog["torch.channels_last-3d@1"].stride_order(5) == tuple(NHWDC_STRIDE_ORDER)
    assert layout_module.morphism_for_stride_order(NHWC_STRIDE_ORDER) == catalog[CHANNELS_LAST]
    # A rank the morphism is not defined for leaves the tensor row-major, and
    # a stride order no catalog entry names is a CANDIDATE, never a guess.
    assert catalog[CHANNELS_LAST].stride_order(2) is None
    assert layout_module.morphism_for_stride_order([1, 0, 2, 3]) is None


# -- the axis moves with the declaration -------------------------------------


def test_two_declared_layouts_cannot_share_one_artifact_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collision this axis exists to remove, with every other axis held."""

    spec, _, _ = _spec()
    declaration = spec.declare()
    runtime = RuntimeCompatibility("cpu", toolchain=_toolchain())

    stored_key = runtime.key(declaration)
    stored_env = EnvIdentity(stack=STACK, sm="sm_89")

    _declare(monkeypatch, CHANNELS_LAST)
    ideal_key = RuntimeCompatibility("cpu", toolchain=_toolchain()).key(declaration)
    ideal_env = EnvIdentity(stack=STACK, sm="sm_89")

    assert stored_key != ideal_key
    assert stored_env.value != ideal_env.value
    # ...and every other axis is IDENTICAL, so the layout is demonstrably the
    # only thing that moved. The graph axis in particular cannot see a stride.
    for axis in ("compile_policy", "graph", "sm", "toolchain"):
        assert stored_key.as_dict()[axis] == ideal_key.as_dict()[axis]
    assert stored_key.as_dict()["declared_input_layout"] == CONTIGUOUS
    assert ideal_key.as_dict()["declared_input_layout"] == CHANNELS_LAST
    assert CHANNELS_LAST in ideal_env.describe()


def test_the_artifact_confesses_the_layout_its_build_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read from the build, never restated: move the config, the stamp moves."""

    stored = _metadata()
    assert stored["declared_input_layout"] == CONTIGUOUS
    _declare(monkeypatch, CHANNELS_LAST)
    ideal = _metadata()
    assert ideal["declared_input_layout"] == CHANNELS_LAST
    assert ideal["compiled_graph_key"] != stored["compiled_graph_key"]


def test_a_wishlist_row_is_a_ratified_handle_or_nothing() -> None:
    """A candidate carries the ORDER inductor asked for and no invented name."""

    metadata = _metadata()
    assert metadata["layout_wishlist"] == []
    from torchcg.artifact import validate_metadata

    ratified = {"fqn": "conv.weight", "morphism": CHANNELS_LAST, "stride_order": [3, 0, 2, 1]}
    candidate = {"fqn": "conv.weight", "morphism": "", "stride_order": [3, 0, 2, 1]}
    for row in (ratified, candidate):
        validate_metadata({**metadata, "layout_wishlist": [row]})
    with pytest.raises(ArtifactError, match="ratified catalog handle"):
        validate_metadata(
            {
                **metadata,
                "layout_wishlist": [{**candidate, "morphism": "bfl.nvfp4-preswizzled@1"}],
            }
        )
    with pytest.raises(ArtifactError, match="permutation"):
        validate_metadata(
            {**metadata, "layout_wishlist": [{**candidate, "stride_order": [3, 0, 2, 9]}]}
        )


# -- the mint may not declare a layout its own tensors do not have ------------


def test_a_mint_whose_tensors_contradict_its_declaration_refuses_before_compiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An NCHW module under a channels_last declaration is named, not repacked."""

    _declare(monkeypatch, CHANNELS_LAST)
    spec, _, _ = _spec()
    engine = Engine(LocalCAS(tmp_path / "cas"))
    with pytest.raises(AdmissionError, match=r"declares input layout .*channels_last"):
        engine.compile(
            spec, RuntimeCompatibility("cpu", toolchain=_toolchain()), tmp_path / "compiled"
        )


def test_a_channels_last_module_satisfies_a_channels_last_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same check, green, and it is the whole point of the contract."""

    _declare(monkeypatch, CHANNELS_LAST)
    spec, _, _ = _spec(channels_last=True)
    assert compiler_module.layout_violations(spec.program) == []
    _declare(monkeypatch, CONTIGUOUS)
    violations = compiler_module.layout_violations(spec.program)
    assert any("conv.weight" in violation for violation in violations)


# -- the mint, the wishlist and the binder, on the installed compiler ---------


def _mint(engine: Engine, spec: GraphSpecialization, tmp_path: Path, name: str) -> Any:
    return engine.compile(
        spec, RuntimeCompatibility("cpu", toolchain=_toolchain()), tmp_path / name
    )


@pytest.mark.real_aoti
def test_real_aoti_the_mint_emits_the_layout_inductor_actually_wanted(
    tmp_path: Path,
) -> None:
    """The wish is READ from inductor's own `require_strides`, at micro scale.

    Contiguous conv weights: inductor asks for stride order [3, 0, 2, 1] and
    inserts the copy -- 40 such launches and 1.19 GiB/step of DRAM traffic on
    the real sdxl UNet (tcg#82). The wish is emitted in the ratified
    vocabulary, against the artifact's own fqn, and the artifact still serves.
    """

    engine = Engine(LocalCAS(tmp_path / "cas"))
    spec, module, sample = _spec()
    result = _mint(engine, spec, tmp_path, "compiled")
    metadata = result.compiled_graph.metadata
    assert metadata["declared_input_layout"] == CONTIGUOUS
    assert metadata["layout_wishlist"] == [
        {"fqn": "conv.weight", "morphism": CHANNELS_LAST, "stride_order": [3, 0, 2, 1]}
    ]

    runner = engine.runner(result.compiled_graph.key, tmp_path / "loaded")
    assert runner is not None
    runner.bind(dict(module.state_dict()), device="cpu")
    torch.testing.assert_close(runner(sample), module(sample))


@pytest.mark.real_aoti
def test_real_aoti_declaring_the_wanted_layout_deletes_the_permute_and_the_wish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliver the weights channels_last and the per-call permute is never generated.

    `require_strides` returns x UNCHANGED when the strides already match (torch
    `_inductor/ir.py`), so an empty wishlist is the artifact stating, from the
    compiler's own decisions, that nothing is re-laid-out per call. This is the
    acceptance for `0-DESIGN.md` §9, measured by execution rather than read.
    """

    _declare(monkeypatch, CHANNELS_LAST)
    engine = Engine(LocalCAS(tmp_path / "cas"))
    spec, module, sample = _spec(channels_last=True)
    result = _mint(engine, spec, tmp_path, "compiled")
    metadata = result.compiled_graph.metadata
    assert metadata["declared_input_layout"] == CHANNELS_LAST
    assert metadata["layout_wishlist"] == []

    runner = engine.runner(result.compiled_graph.key, tmp_path / "loaded")
    assert runner is not None
    runner.bind(dict(module.state_dict()), device="cpu")
    torch.testing.assert_close(runner(sample), module(sample))


@pytest.mark.real_aoti
def test_real_aoti_nchw_bytes_offered_to_a_channels_last_artifact_are_a_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The named first blocker, as a test.

    What stood at `runner.py:246` was an unconditional `.contiguous()`, which
    would have taken these NCHW weights, copied them at full weight cost, and
    bound them to an artifact compiled for NHWC -- silently. Layout is contract
    identity: a wrong pairing refuses by name.
    """

    _declare(monkeypatch, CHANNELS_LAST)
    engine = Engine(LocalCAS(tmp_path / "cas"))
    spec, module, _ = _spec(channels_last=True)
    result = _mint(engine, spec, tmp_path, "compiled")

    runner = engine.runner(result.compiled_graph.key, tmp_path / "loaded")
    assert runner is not None
    nchw = {
        name: value.detach().contiguous() for name, value in module.state_dict().items()
    }
    with pytest.raises(ConstantBindingError, match="conv.weight") as refusal:
        runner.bind(nchw, device="cpu")
    assert refusal.value.reason == "layout_mismatch"
    assert CHANNELS_LAST in str(refusal.value)


@pytest.mark.real_aoti
def test_real_aoti_an_out_of_catalog_wish_falls_back_and_emits_an_unratified_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corpus: Path
) -> None:
    """The PERMANENT fallback path, driven by moving the catalog itself.

    With `torch.channels_last-2d@1` absent from the RATIFYING CORPUS, inductor still
    wants stride order [3, 0, 2, 1] for the conv weight. The mint compiles
    against the STORED layout, serves, and emits the order as a candidate with
    no morphism name attached. Machines derive along ratified morphisms; they
    never invent one.
    """

    (corpus / "torch.channels_last-2d.v1.json").unlink()
    layout_module._paired.cache_clear()
    engine = Engine(LocalCAS(tmp_path / "cas"))
    spec, module, sample = _spec()
    result = _mint(engine, spec, tmp_path, "compiled")
    metadata = result.compiled_graph.metadata
    assert metadata["declared_input_layout"] == CONTIGUOUS
    assert metadata["layout_wishlist"] == [
        {"fqn": "conv.weight", "morphism": "", "stride_order": [3, 0, 2, 1]}
    ]

    runner = engine.runner(result.compiled_graph.key, tmp_path / "loaded")
    assert runner is not None
    runner.bind(dict(module.state_dict()), device="cpu")
    torch.testing.assert_close(runner(sample), module(sample))


def test_resolving_the_IDENTITY_never_imports_torch(tmp_path: Path) -> None:
    """tcg#87 regression fence, and it is load-bearing.

    `torchcg resolve` and the cold-process reuse path exist to hand an artifact
    back out of the store in a process that never imports torch, and
    `require_morphism` sits on that path because EVERY artifact declares a
    layout. Pairing a record to a `memory_format` needs a torch probe, so
    making the catalog corpus-derived came within one line of importing torch
    on every resolve -- caught by `test_engine`'s cold-process arm, whose whole
    assertion is `"torch" not in sys.modules`.

    The identity is resolvable from the corpus ALONE (unique rankless record,
    permutes nothing), which is what keeps that property. Driven in a real cold
    subprocess, because the parent process has torch imported already and
    cannot answer this question about itself.
    """

    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from torchcg.layout import contiguous_handle, require_morphism\n"
            "handle = contiguous_handle()\n"
            "assert require_morphism(handle).handle == handle\n"
            "assert 'torch' not in sys.modules, sorted(sys.modules)[:0] or 'torch imported'\n"
            "print(handle)\n",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == CONTIGUOUS
