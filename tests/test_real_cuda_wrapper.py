"""tcg#4: the transforms, anchored to REAL CUDA inductor output.

Every other test in this suite drives `_wrapper_split` / `_run_impl_split`
against a shape-faithful SYNTHETIC replica of inductor's emission. This file
pins them to the real thing: `realdata/aoti_cuda_wrapper_torch2130_sm86.cpp`
is the verbatim wrapper torch 2.13.0+cu130 (git cf30153c) emitted on
2026-08-16 for a 12-layer mm+relu buffer model, compiled through the sealed
`Engine` on a RunPod RTX A4000 (sm_86, ubuntu-22.04, g++ 11.4) under this
package's production inductor options. On that pod the split artifact was
also EXECUTED: output bitwise-identical to the unsplit compile and to eager.

If inductor's emission drifts from what these matchers recognise, the
transforms silently decline fleet-wide — this file is what turns that silent
decline into a red test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from torchcg import _run_impl_split as ris
from torchcg import _wrapper_split as ws

FIXTURE = Path(__file__).parent / "realdata" / "aoti_cuda_wrapper_torch2130_sm86.cpp"


@pytest.fixture(scope="module")
def source() -> str:
    return FIXTURE.read_text()


@pytest.fixture
def _small_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capture model is small on purpose (a pod-minutes compile); the
    production bars stay real and only the threshold is relaxed here."""
    monkeypatch.setattr(ris, "MIN_COMPUTE_LINES", 10)
    monkeypatch.setattr(ws, "MIN_STATEMENTS", 10)


def test_run_impl_split_applies_to_real_output(source: str, _small_enough: None) -> None:
    split = ris.split_run_impl(source)  # production DEFAULT_PARTS
    assert split.applied, split.reason
    assert ris.reconstruct(split.main, split.parts) == source


def test_real_output_exercises_the_declaration_forms(source: str, _small_enough: None) -> None:
    """The kinds the real compute region uses must all be recognised ones —
    an unknown form would surface as a Declined resolve_type, but assert the
    coverage explicitly so drift names the missing kind."""
    lines = source.split("\n")
    anchors = ris.find_anchors(lines)
    body = ris.parse_body(lines, anchors)
    kinds = {decl.kind for decl in body.decls.values()}
    assert kinds == {"steal", "move", "const_bind", "handle", "kernels_bind", "array", "raii"}
    assert "proxy_executor" in body.params  # real signature exceeds the replica's


def test_constants_ctor_split_applies_to_real_output(source: str, _small_enough: None) -> None:
    result, outcome = ws.split_constants_constructor(source)
    assert outcome.applied, outcome.reason
    assert outcome.statements == 264  # 24 constants x 11 fields
    assert result != source


def test_production_thresholds_decline_this_small_capture(source: str) -> None:
    """The fixture is far below the pathological shape; unpatched, both
    transforms must decline on SIZE — any other reason means the shape
    recognition itself regressed against real output."""
    split = ris.split_run_impl(source)
    assert not split.applied
    assert "not the pathological shape" in split.reason
    assert "compute region is 111 lines" in split.reason

    _, outcome = ws.split_constants_constructor(source)
    assert not outcome.applied
    assert "only 264 statements" in outcome.reason
