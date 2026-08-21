"""tcg#77: the shape fan collapses into ONE graph whose ingress states ranges.

The subject is the same tiny SD1.5-class pipeline the tcg#41 acceptance uses
(real diffusers author code, generated weights, nothing downloaded). It is
driven at several resolutions and both CFG modes -- the exact fan pgw#1548
measured at 14 specializations on the real sd15 endpoint -- and the assertions
are about the COUNT, the guards, and what the dispatcher does with them.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

import sd15_tiny  # noqa: E402
import torch  # noqa: E402

from torchcg.adopt import _matches  # noqa: E402
from torchcg.discovery import discover_lane  # noqa: E402
from torchcg.document import GraphRecord  # noqa: E402

#: The tiny VAE has one block, so its scale factor is 1 and these pixel
#: sizes ARE the latent sides. Three aspects x both CFG modes = the six-call
#: fan this issue exists to collapse.
SHAPES = ((64, 64), (48, 80), (80, 48))


def _drive(pipe: object) -> None:
    for height, width in SHAPES:
        for guidance in (7.5, 1.0):
            pipe(  # type: ignore[operator]
                prompt="a cat",
                num_inference_steps=1,
                guidance_scale=guidance,
                height=height,
                width=width,
                output_type="pt",
            )


def _lane(dynamic_dims: object) -> object:
    pipe = sd15_tiny.build_pipe()
    return discover_lane(
        sd15_tiny.LANE_CONTRACT,
        ("unet",),
        pipe.components,
        lambda: _drive(pipe),
        dynamic_dims=dynamic_dims,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def static() -> object:
    return _lane(None)


@pytest.fixture(scope="module")
def dynamic() -> object:
    return _lane(True)


def test_the_static_fan_is_one_graph_per_observed_shape(static: object) -> None:
    """The control: six drives, six graph specializations, all concrete."""

    assert len(static.graphs) == len(SHAPES) * 2  # type: ignore[attr-defined]
    for record in static.graphs:  # type: ignore[attr-defined]
        assert record.ingress.symbols == ()
        for row in record.ingress.inputs:
            assert all(isinstance(dim, int) for dim in row.shape)


def test_dynamic_dims_collapse_the_aspect_fan(dynamic: object) -> None:
    """Six observed calls become TWO: the three aspects collapse, CFG does not.

    AMENDED BY tcg#78. This asserted ONE graph, with batch symbolic over
    [1, 2] -- and that graph could not be minted. Torch specializes the sizes
    0 and 1 rather than reason about them symbolically, so it guards every
    dynamic dim ``>= 2``; an axis observed at 1 and 2 is therefore contradicted
    by the graph's own guards the moment it is exported, and the artifact that
    came out of compiling it answered a batch-1 call with a batch-2 tensor of
    garbage. The derive now refuses that axis by name.

    What matters is that refusing it costs the ASPECT collapse nothing --
    3 aspects x 2 CFG modes goes 6 -> 2, not 6 -> 6.
    """

    assert len(dynamic.graphs) == 2  # type: ignore[attr-defined]
    for record in dynamic.graphs:  # type: ignore[attr-defined]
        assert record.ingress.symbols, "a collapsed record must state its ranges"
        sample = next(row for row in record.ingress.inputs if row.name == "sample")
        batch, channels, height, width = sample.shape
        assert isinstance(height, str) and isinstance(width, str)
        assert channels == 4
        assert batch in (1, 2), "the CFG axis is concrete in each record"
        bounds = record.ingress.symbol_bounds
        assert bounds[height] == (48, 80)
        assert bounds[width] == (48, 80)
        # A derived dim: the UNet's downsamples make 16 the stride, measured
        # off the observed values rather than assumed.
        assert height.startswith("16*") and width.startswith("16*")
        assert height != width, "H and W are two degrees of freedom, not one"
        text = next(
            row for row in record.ingress.inputs if row.name == "encoder_hidden_states"
        )
        assert text.shape[0] == batch, "one CFG mode, one batch, both rows"

    assert {
        next(row for row in record.ingress.inputs if row.name == "sample").shape[0]
        for record in dynamic.graphs  # type: ignore[attr-defined]
    } == {1, 2}


def test_a_per_axis_policy_collapses_only_the_axis_it_admits() -> None:
    """pgw#1548's gate: adopt one axis without adopting the other.

    AMENDED BY tcg#78: offering ONLY the batch axis now collapses nothing,
    because that axis is the one the guards refuse. The fan comes back whole
    and loudly -- which is also what pgw#1548 measured on the real endpoint
    from the other direction (14 -> 14: the batch axis removed no
    specializations even when it appeared to work).
    """

    lane = _lane(lambda _target, _name, axis: axis == 0)
    assert len(lane.graphs) == len(SHAPES) * 2  # type: ignore[attr-defined]
    for record in lane.graphs:  # type: ignore[attr-defined]
        assert record.ingress.symbols == ()
        sample = next(row for row in record.ingress.inputs if row.name == "sample")
        assert all(isinstance(dim, int) for dim in sample.shape)


def test_offering_only_the_aspect_axes_is_the_axis_that_pays() -> None:
    """The other half of the gate, and the one the program is FOR."""

    lane = _lane(lambda _target, _name, axis: axis in (2, 3))
    assert len(lane.graphs) == 2  # type: ignore[attr-defined]
    for record in lane.graphs:  # type: ignore[attr-defined]
        sample = next(row for row in record.ingress.inputs if row.name == "sample")
        assert isinstance(sample.shape[0], int), "batch was never offered"
        assert isinstance(sample.shape[2], str)


def _call(
    record: GraphRecord, batch: int, height: int, width: int
) -> tuple[bool, dict[str, Any]]:
    rows = {row.name: row for row in record.ingress.inputs}
    sample = rows["sample"]
    text = rows["encoder_hidden_states"]
    kwargs = {
        "sample": torch.zeros(
            (batch, 4, height, width), dtype=getattr(torch, sample.dtype)
        ),
        "timestep": torch.zeros((), dtype=getattr(torch, rows["timestep"].dtype)),
        "encoder_hidden_states": torch.zeros(
            (batch, int(text.shape[1]), int(text.shape[2])),
            dtype=getattr(torch, text.dtype),
        ),
    }
    return _matches(record, (), kwargs), kwargs


def _record_for(lane: object, batch: int) -> GraphRecord:
    """The record whose CFG mode is this batch -- the axis stayed concrete."""

    record: GraphRecord = next(
        candidate
        for candidate in lane.graphs  # type: ignore[attr-defined]
        if next(
            row for row in candidate.ingress.inputs if row.name == "sample"
        ).shape[0] == batch
    )
    return record


def test_the_dispatcher_admits_every_observed_shape(dynamic: object) -> None:
    for height, width in SHAPES:
        for batch in (1, 2):
            record = _record_for(dynamic, batch)
            matched, _ = _call(record, batch, height, width)
            assert matched, f"{batch}x{height}x{width} must enter the dynamic graph"


def test_the_dispatcher_refuses_shapes_outside_the_exported_range(
    dynamic: object,
) -> None:
    """A symbolic dim is a RANGE, never a wildcard."""

    record = _record_for(dynamic, 2)
    assert not _call(record, 4, 64, 64)[0], "batch 4 is not this record's batch"
    assert not _call(record, 2, 96, 64)[0], "a 96 side is outside [48, 80]"
    assert not _call(record, 2, 32, 64)[0], "a 32 side is outside [48, 80]"
    # In range and still refused: the graph's own guards accept this axis only
    # in multiples of 16, and the record says so.
    assert not _call(record, 2, 56, 64)[0], "56 is in range but off-stride"


def test_an_inconsistent_batch_is_refused(dynamic: object) -> None:
    """A record's rows agree: batch 2 on sample, batch 1 on conditioning is
    not this record's call, whether the axis is symbolic or concrete."""

    record = _record_for(dynamic, 2)
    rows = {row.name: row for row in record.ingress.inputs}
    text = rows["encoder_hidden_states"]
    kwargs = {
        "sample": torch.zeros(
            (2, 4, 64, 64), dtype=getattr(torch, rows["sample"].dtype)
        ),
        "timestep": torch.zeros((), dtype=getattr(torch, rows["timestep"].dtype)),
        "encoder_hidden_states": torch.zeros(
            (1, int(text.shape[1]), int(text.shape[2])),
            dtype=getattr(torch, text.dtype),
        ),
    }
    assert not _matches(record, (), kwargs)


def test_a_refused_dynamic_export_falls_back_loudly(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The red arm: when export refuses the Dims, the fan comes back INTACT."""

    real = torch.export.export

    def refuse(*args: object, **kwargs: object) -> object:
        if kwargs.get("dynamic_shapes"):
            raise RuntimeError("Constraints violated (synthetic)")
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.export, "export", refuse)
    with caplog.at_level(logging.WARNING, logger="torchcg.discovery"):
        lane = _lane(True)
    assert len(lane.graphs) == len(SHAPES) * 2  # type: ignore[attr-defined]
    assert any("was REFUSED" in record.message for record in caplog.records)


def test_the_entry_runner_path_guards_the_same_derived_dim() -> None:
    """The OTHER dispatch seam (`selection`) reads the stride identically.

    Two selectors exist -- the per-module dispatcher above and the
    entrypoint-level selection the AOTI entry runner uses -- and a range they
    disagree about is a graph one of them enters and the other refuses.
    """

    from torchcg.ingress import CallIngress, CallInput, exported_input_name
    from torchcg.selection import MissReason, PresentedCall, PresentedValue, class_report

    def row(
        position: int, param: str, dtype: str, shape: tuple[int | str, ...]
    ) -> CallInput:
        return CallInput(
            name=param,
            position=position,
            param=param,
            param_position=position,
            path=(),
            exported_name=exported_input_name(param),
            dtype=dtype,
            shape=shape,
        )

    ingress = CallIngress(
        parameters=("sample", "encoder_hidden_states"),
        flat_arity=2,
        inputs=(
            row(0, "sample", "bfloat16", ("s0", 4, "8*s1", "8*s1")),
            row(1, "encoder_hidden_states", "bfloat16", ("s0", 77, 768)),
        ),
        symbols=(("8*s1", (48, 80)), ("s0", (1, 2))),
    )

    def report(
        sample: tuple[int, ...], text: tuple[int, ...]
    ) -> tuple[Any, ...]:
        call = PresentedCall(
            feeds=(
                ("encoder_hidden_states", PresentedValue("bfloat16", text)),
                ("sample", PresentedValue("bfloat16", sample)),
            )
        )
        return class_report("cg", ingress, call).misses

    assert report((2, 4, 64, 64), (2, 77, 768)) == ()
    assert report((2, 4, 52, 52), (2, 77, 768))[0].reason is MissReason.RANGE_VIOLATION
    assert report((2, 4, 96, 96), (2, 77, 768))[0].reason is MissReason.RANGE_VIOLATION
    assert (
        report((2, 4, 64, 64), (1, 77, 768))[0].reason is MissReason.SYMBOL_INCONSISTENT
    )
