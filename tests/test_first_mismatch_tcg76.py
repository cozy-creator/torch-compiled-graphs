"""tcg#76: an armed dispatcher that matches nothing NAMES its first divergence.

The night this pays for (2026-08-20, sd15 head-to-head): `armed_graphs=14,
compiled_graph_calls=0, displaced=[]` — and THREE confident root causes
(format skew, destination occupancy, timestep dtype) each died against a
counter reading, because the one object that knew the exact divergent input
on every call — the dispatcher, whose guard refuses row by row — said
nothing. Eager fall-through per call is correct and stays silent; the FIRST
all-miss call on a module now prints, once, at WARNING (the level a bare
``gen-worker up`` surfaces), the best-matching record's first failing row:
expected name/dtype/shape vs received py-type/dtype/shape.

``_row_mismatch`` is the single source for the guard AND the diagnosis, so
the two cannot drift into telling different stories.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

from torchcg.adopt import _ForwardDispatcher, _row_mismatch
from torchcg.document import GraphRecord
from torchcg.ingress import CallIngress, CallInput


def _record(graph_letter: str, *, sample_dtype: str = "float16") -> GraphRecord:
    ingress = CallIngress(
        parameters=("sample", "timestep", "encoder_hidden_states"),
        flat_arity=7,
        inputs=(
            CallInput("sample", 0, "sample", 0, (), "sample", sample_dtype, (2, 4, 64, 64)),
            CallInput("timestep", 1, "timestep", 1, (), "timestep", "int64", ()),
            CallInput(
                "encoder_hidden_states", 2, "encoder_hidden_states", 2, (),
                "encoder_hidden_states", "float16", (2, 77, 768),
            ),
        ),
    )
    return GraphRecord(
        graph="cg-graph-v1-" + graph_letter * 56, target="unet", ingress=ingress
    )


class _Module(torch.nn.Module):
    def forward(
        self, sample: Any, timestep: Any, encoder_hidden_states: Any = None
    ) -> str:
        return "eager"


def _armed() -> tuple[_ForwardDispatcher, list[Any]]:
    module = _Module()
    dispatcher = _ForwardDispatcher(module)
    entered: list[Any] = []

    def compiled(*args: Any, **kwargs: Any) -> str:
        entered.append(args)
        return "compiled"

    dispatcher.arm(_record("a"), compiled)
    return dispatcher, entered


SAMPLE = torch.zeros(2, 4, 64, 64, dtype=torch.float16)
EHS = torch.zeros(2, 77, 768, dtype=torch.float16)


def test_the_first_all_miss_call_names_the_divergent_input_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dispatcher, entered = _armed()
    with caplog.at_level(logging.WARNING, logger="torchcg.adopt"):
        # A python int where the guard wants an int64 0-d tensor — the exact
        # class of divergence a document read can never show.
        result = dispatcher(SAMPLE, 981, encoder_hidden_states=EHS)
        again = dispatcher(SAMPLE, 981, encoder_hidden_states=EHS)

    assert result == "eager" and again == "eager" and not entered
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "once per module, not per call"
    message = warnings[0].getMessage()
    assert "NO armed graph matched" in message
    assert "'timestep'" in message and "py-type int" in message
    assert "1/3 input row(s) matched" in message
    assert "expected a torch.Tensor int64 ()" in message


def test_a_matching_call_enters_compiled_and_prints_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dispatcher, entered = _armed()
    with caplog.at_level(logging.WARNING, logger="torchcg.adopt"):
        result = dispatcher(
            SAMPLE, torch.tensor(981, dtype=torch.int64), encoder_hidden_states=EHS
        )
    assert result == "compiled" and entered
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_the_best_matching_record_is_the_one_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Record A dies on row 0 (bf16 sample); record B passes the sample row
    and dies on the timestep — B's divergence is the story, and B prints."""
    module = _Module()
    dispatcher = _ForwardDispatcher(module)
    dispatcher.arm(_record("a", sample_dtype="bfloat16"), lambda *a, **k: "A")
    ehs_record = _record("b")
    dispatcher.arm(ehs_record, lambda *a, **k: "B")

    with caplog.at_level(logging.WARNING, logger="torchcg.adopt"):
        dispatcher(SAMPLE, torch.tensor([981], dtype=torch.int64),
                   encoder_hidden_states=EHS)

    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    message = warning.getMessage()
    assert ehs_record.graph[-16:] in message
    assert "1/3 input row(s) matched" in message
    # The [1]-shaped timestep is a RANK divergence against the 0-D guard —
    # named as such, not as a dtype problem.
    assert "rank 1 (shape (1,))" in message and "expected rank 0" in message


def test_row_mismatch_is_the_single_source_for_guard_and_story() -> None:
    record = _record("a")
    row = record.ingress.inputs[1]  # timestep int64 ()
    assert _row_mismatch(row, (SAMPLE, torch.tensor(1, dtype=torch.int64)), {}) is None
    assert "py-type float" in (_row_mismatch(row, (SAMPLE, 1.0), {}) or "")
    assert "dtype float32" in (
        _row_mismatch(row, (SAMPLE, torch.tensor(1.0)), {}) or ""
    )
    missing = _row_mismatch(row, (SAMPLE,), {})
    assert missing is not None and "never resolved" in missing
