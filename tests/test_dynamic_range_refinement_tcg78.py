"""tcg#78: a declared symbolic range the graph's own guards contradict.

The defect these tests pin, measured on the real sd15 UNet before the fix: a
batch ``Dim`` over ``[1, 2]`` exports cleanly, compiles for 84-129 s of GPU,
and is then rejected by the mint's admission check -- and when that check is
bypassed, the artifact it wanted to publish **answers a batch-1 call with a
(2, 4, 64, 64) tensor of garbage and raises nothing**.

The chain, every link measured on CPU with this fixture:

1. ``torch.export`` accepts the Dim. ``range_constraints`` says ``[1, 2]``.
2. The AOTI compile decomposes first, and only the decomposed graph asks the
   size-1 contiguity question -- ``torch._prims.convert_element_type`` ->
   ``guard_or_false(prod(sizes) < 2)``. It guards ``s53 >= 2``.
3. ``_refine_ranges`` narrows ``[1, 2]`` to ``[2, 2]``; a singleton range makes
   the ShapeEnv install ``s53 -> 2`` as a REPLACEMENT.
4. Every downstream read of the symbol now sees the constant 2.

Three seams are covered: the declaration must not move under (4), the derive
must refuse at (2) instead of paying for (3), and the mint must refuse a
narrowing whether or not it reached a singleton.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

import sd15_tiny  # noqa: E402
import sympy  # type: ignore[import-untyped]  # noqa: E402
import torch  # noqa: E402
from tensorfs import LocalCAS  # noqa: E402
from torch.utils._sympy.value_ranges import ValueRanges  # noqa: E402

from torchcg import Engine, EnsureOutcome  # noqa: E402
from torchcg.declaration import (  # noqa: E402
    GraphSpecialization,
    RuntimeCompatibility,
)
from torchcg.discovery import discover_lane  # noqa: E402
from torchcg.dynamic import (  # noqa: E402
    declared_symbol_ranges,
    decomposition_narrowing,
    live_symbol_ranges,
    narrowed_symbols,
    shape_env_of,
)
from torchcg.engine import AdmissionError  # noqa: E402
from torchcg.ingress import build_call_ingress  # noqa: E402

TOOLCHAIN = {
    "settings_declaration": "settings-v1",
    "loaded_libs": "loaded-libs-v1",
    "torch": "torch-record-v1",
    "triton": "triton-record-v1",
}

#: Three aspects x both CFG modes -- the fan tcg#77 collapses.
SHAPES = ((64, 64), (48, 80), (80, 48))


def _drive(pipe: object, shapes: tuple[tuple[int, int], ...] = SHAPES) -> None:
    for height, width in shapes:
        for guidance in (7.5, 1.0):
            pipe(  # type: ignore[operator]
                prompt="a cat",
                num_inference_steps=1,
                guidance_scale=guidance,
                height=height,
                width=width,
                output_type="pt",
            )


def _lane(dynamic_dims: object, shapes: tuple[tuple[int, int], ...] = SHAPES) -> Any:
    pipe = sd15_tiny.build_pipe()
    return discover_lane(
        sd15_tiny.LANE_CONTRACT,
        ("unet",),
        pipe.components,
        lambda: _drive(pipe, shapes),
        dynamic_dims=dynamic_dims,  # type: ignore[arg-type]
    )


def _batch_only(_target: str, _name: str, axis: int) -> bool:
    return axis == 0


def _spatial_only(_target: str, _name: str, axis: int) -> bool:
    return axis in (2, 3)


@pytest.fixture(scope="module")
def unet() -> Any:
    return sd15_tiny.build_pipe().unet


def _export_batch_dynamic(module: Any, max_batch: int) -> GraphSpecialization:
    """The subject, exported directly so the RANGE is the only variable."""

    dim = module.config.cross_attention_dim
    sample = torch.zeros(2, 4, 64, 64)
    timestep = torch.zeros((), dtype=torch.float32)
    cond = torch.zeros(2, 77, dim)
    batch = torch.export.Dim("s53", min=1, max=max_batch)
    program = torch.export.export(
        module,
        (sample, timestep, cond),
        {"return_dict": False},
        dynamic_shapes=({0: batch}, {}, {0: batch}, None),
    )
    ingress = build_call_ingress(
        program,
        ("sample", "timestep", "encoder_hidden_states", "return_dict"),
        (sample, timestep, cond),
        {"return_dict": False},
    )
    return GraphSpecialization("unet", "unet", program, ingress)


# --------------------------------------------------------------------------
# 1. The declaration is a function of the exported program, not of the
#    compiler's later opinion about it.
# --------------------------------------------------------------------------


def test_a_declaration_does_not_move_when_the_shape_env_does(unet: Any) -> None:
    """RED ARM. On the unfixed tree this witness moved with the ShapeEnv.

    `_render_symbol` read ``SymNode.expr`` -- a property that folds in every
    replacement the environment has been told about since. A compile installs
    replacements there, so the SAME program declared twice around one compile
    produced two different witnesses and the mint refused its own output with
    "exported program changed during compilation". Nothing about the program
    changed; the compiler had merely made up its mind.
    """

    spec = _export_batch_dynamic(unet, max_batch=2)
    before = spec.declare()

    environment = shape_env_of(spec.program)
    assert environment is not None, "a dynamic program must carry a ShapeEnv"
    symbols = [
        symbol for symbol in environment.var_to_range if str(symbol).startswith("s")
    ]
    assert symbols, "the export must have produced at least one symbol"
    for symbol in symbols:
        environment.var_to_range[symbol] = ValueRanges(
            sympy.Integer(2), sympy.Integer(2)
        )
        environment._set_replacement(symbol, sympy.Integer(2), "tcg78_test")

    assert spec.declare() == before
    assert spec.declare().graph_witness == before.graph_witness


# --------------------------------------------------------------------------
# 2. The derive refuses the axis, and says which guard refused it.
# --------------------------------------------------------------------------


def test_the_probe_names_the_contradicted_range(unet: Any) -> None:
    """The program declares [1, 2]; its own ShapeEnv already says [2, 2].

    Both statements come out of ONE export that raised nothing. Torch
    specializes the sizes 0 and 1 rather than reason about broadcasting
    symbolically, so every dynamic dim is guarded `>= 2` -- and
    ``range_constraints``, the field the lock is built from, keeps the
    author's request unamended.
    """

    spec = _export_batch_dynamic(unet, max_batch=2)
    assert declared_symbol_ranges(spec.program)["s53"] == (1, 2)
    assert live_symbol_ranges(spec.program)["s53"] == (2, 2)
    moved = decomposition_narrowing(spec.program)
    assert moved["s53"] == ((1, 2), (2, 2))


def test_a_narrowing_short_of_a_singleton_is_still_a_narrowing(unet: Any) -> None:
    """``>= 2`` against ``[1, 4]`` leaves ``[2, 4]``: no symbol is replaced,
    no digest moves, and before tcg#78 this shipped GREEN."""

    spec = _export_batch_dynamic(unet, max_batch=4)
    moved = decomposition_narrowing(spec.program)
    assert moved["s53"] == ((1, 4), (2, 4)), (
        "a narrowing that stops short of a singleton is still a narrowing, and "
        "is exactly the case that shipped GREEN before tcg#78"
    )


def test_the_derive_refuses_the_batch_axis_and_returns_the_static_fan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RED ARM. Before tcg#78 this collapsed 6 -> 3 and the refusal came
    84-129 s into a GPU compile, if it came at all."""

    with caplog.at_level(logging.WARNING, logger="torchcg.discovery"):
        lane = _lane(_batch_only)

    assert len(lane.graphs) == len(SHAPES) * 2, (
        "a refused axis leaves the fan it was going to collapse INTACT"
    )
    for record in lane.graphs:
        assert record.ingress.symbols == ()
        for row in record.ingress.inputs:
            assert all(isinstance(dim, int) for dim in row.shape)

    refusals = [
        record.getMessage()
        for record in caplog.records
        if "REFUSED" in record.getMessage()
    ]
    assert refusals, "a silent fallback is how a lane believes a fan collapsed"
    assert any("guards contradict the declared range" in text for text in refusals)
    assert any("allow only [2, 2]" in text for text in refusals), (
        "the refusal must quote the range the graph actually supports"
    )


def test_the_derive_still_collapses_the_spatial_axes() -> None:
    """The axis the dynamic-dims program is actually FOR (pgw#1548).

    Spatial dims are derived (``16*s3``) with roots over 3..5; the same
    ``>= 2`` guard fires against them and takes nothing away, so nothing is
    refused and the fan still collapses.
    """

    lane = _lane(_spatial_only)
    assert len(lane.graphs) == 2, "CFG stays a fan; the three aspects become one"
    for record in lane.graphs:
        sample = next(row for row in record.ingress.inputs if row.name == "sample")
        assert isinstance(sample.shape[0], int), "batch was not offered"
        assert isinstance(sample.shape[2], str), "the aspect axes must be symbolic"
        assert record.ingress.symbols


# --------------------------------------------------------------------------
# 3. The mint refuses a narrowed artifact by name -- including the narrowing
#    that moves no digest.
# --------------------------------------------------------------------------


def _mint(spec: GraphSpecialization) -> tuple[Any, bool]:
    """``(result, the artifact was really written)`` -- the workspace dies with
    the temporary directory, so the file is checked while it still exists."""

    runtime = RuntimeCompatibility("cpu", toolchain=TOOLCHAIN)
    with tempfile.TemporaryDirectory(prefix="tcg78-") as raw:
        root = Path(raw)
        result = Engine(LocalCAS(root / "cas")).compile(spec, runtime, root / "dest")
        return result, result.compiled_graph.package.is_file()


def test_the_mint_refuses_an_artifact_narrower_than_its_declaration(
    unet: Any,
) -> None:
    """RED ARM for the QUIET case: [1, 4] narrows to [2, 4], no symbol is
    replaced, no witness moves -- and before tcg#78 this minted GREEN."""

    spec = _export_batch_dynamic(unet, max_batch=4)
    with pytest.raises(AdmissionError) as raised:
        _mint(spec)
    message = str(raised.value)
    assert "NARROWER range than it declares" in message
    assert "s53 declared [1, 4] but the graph's guards allow only [2, 4]" in message


def test_the_mint_still_publishes_a_graph_whose_ranges_hold(unet: Any) -> None:
    """The control: a static export mints, and nothing above touches it."""

    sample = torch.zeros(2, 4, 64, 64)
    timestep = torch.zeros((), dtype=torch.float32)
    cond = torch.zeros(2, 77, unet.config.cross_attention_dim)
    program = torch.export.export(
        unet, (sample, timestep, cond), {"return_dict": False}
    )
    ingress = build_call_ingress(
        program,
        ("sample", "timestep", "encoder_hidden_states", "return_dict"),
        (sample, timestep, cond),
        {"return_dict": False},
    )
    spec = GraphSpecialization("unet", "unet", program, ingress)
    assert declared_symbol_ranges(program) == {}
    result, published = _mint(spec)
    assert result.outcome == EnsureOutcome.MINTED
    assert published


# --------------------------------------------------------------------------
# The comparison itself, without a model in the way.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "live", "expected"),
    [
        ({"s0": (1, 8)}, {"s0": (1, 8)}, {}),
        ({"s0": (1, 8)}, {"s0": (2, 8)}, {"s0": ((1, 8), (2, 8))}),
        ({"s0": (1, 8)}, {"s0": (1, 4)}, {"s0": ((1, 8), (1, 4))}),
        ({"s0": (1, 8)}, {"s0": (4, 4)}, {"s0": ((1, 8), (4, 4))}),
        # A range that WIDENED still covers everything the lock promised.
        ({"s0": (2, 4)}, {"s0": (1, 8)}, {}),
        # A symbol the environment no longer carries states nothing.
        ({"s0": (1, 8)}, {}, {}),
    ],
)
def test_only_a_narrowing_is_a_refusal(
    declared: dict[str, tuple[int, int]],
    live: dict[str, tuple[int, int]],
    expected: dict[str, object],
) -> None:
    assert narrowed_symbols(declared, live) == expected


def test_a_replaced_symbol_reads_as_its_singleton(unet: Any) -> None:
    """A replacement is the strongest narrowing there is, and the live view
    must not report it as "unchanged" just because var_to_range lags."""

    spec = _export_batch_dynamic(unet, max_batch=8)
    environment = shape_env_of(spec.program)
    assert environment is not None
    assert live_symbol_ranges(spec.program)["s53"] == (2, 8)
    environment._set_replacement(
        sympy.Symbol("s53", positive=True), sympy.Integer(2), "tcg78_test"
    )
    assert live_symbol_ranges(spec.program)["s53"] == (2, 2)
