"""The tcg#4 matrix fold rules, provable without torch or a compiled probe."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "host_fingerprint"


def _load(name: str) -> ModuleType:
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    specification = importlib.util.spec_from_file_location(name, _HARNESS / f"{name}.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_axes = _load("axes")
_report = _load("matrix_report")


def _row(
    *,
    diff: dict[str, str] | None = None,
    load_ok: bool = True,
    exec_ok: bool = True,
    output_ok: bool = True,
) -> dict[str, Any]:
    build = {name: f"value-{name}" for name in _axes.AXIS_NAMES}
    host = dict(build)
    host.update(diff or {})
    return {
        "schema": _axes.AXES_SCHEMA_VERSION,
        "build_axes": build,
        "host_axes": host,
        "diff_axes": sorted(name for name in build if host[name] != build[name]),
        "load_ok": load_ok,
        "exec_ok": exec_ok,
        "output_ok": output_ok,
    }


def test_row_validation_refuses_missing_axes_and_dishonest_diffs() -> None:
    good = _row()
    _axes.validate_row(good)

    breakages: dict[str, Callable[[dict[str, Any]], object]] = {
        "missing axis": lambda row: row["host_axes"].pop("glibc"),
        "dishonest diff": lambda row: row.update(diff_axes=["torch_version"]),
        "wrong schema": lambda row: row.update(schema=0),
        "non-bool flag": lambda row: row.update(load_ok="yes"),
    }
    for breakage, mutate in breakages.items():
        row = _row()
        mutate(row)
        with pytest.raises(ValueError):
            _axes.validate_row(row)
        del breakage


def test_contingency_classifies_candidates_retain_and_unmeasured() -> None:
    rows = [
        _row(diff={"triton": "other"}),
        _row(diff={"triton": "third"}),
        _row(diff={"glibc": "older"}, load_ok=False),
        _row(),
    ]
    folded = _report.report(rows)
    assert folded["rows"] == 4
    assert folded["failures"] == 1
    assert folded["coarsening_candidates"] == ["triton"]
    assert "glibc" in folded["retain"]
    assert "torch_version" in folded["unmeasured"]
    contingency = folded["contingency"]
    assert contingency["triton"] == {
        "differed_failed": 0,
        "differed_passed": 2,
        "same_failed": 1,
        "same_passed": 1,
    }


def test_an_unmeasured_axis_is_never_a_coarsening_candidate() -> None:
    folded = _report.report([_row(), _row()])
    assert folded["coarsening_candidates"] == []
    assert folded["retain"] == []
    assert sorted(folded["unmeasured"]) == sorted(_axes.AXIS_NAMES)


def test_cache_hit_improvement_counts_newly_shared_pairs_only() -> None:
    rows = [
        _row(diff={"triton": "other"}),
        _row(diff={"triton": "other", "glibc": "older"}),
    ]
    improvement = _report.cache_hit_improvement(rows, "triton")
    assert improvement["insufficient_data"] is False
    assert improvement["hosts"] == 3
    assert improvement["pairs"] == 3
    # build<->host1 disagree only on triton; the pair with host2 still
    # disagrees on glibc after the drop; host1<->host2 likewise.
    assert improvement["newly_shared"] == 1
    assert improvement["already_shared"] == 0
    assert improvement["improvement"] == pytest.approx(1 / 3)


def test_a_single_host_matrix_reports_insufficient_data() -> None:
    folded = _report.report([_row()])
    assert "insufficient data" in folded["verdict"]
    assert folded["cache_hit_improvement"] == []


_RESULTS = _HARNESS / "results"
_RETAINED_BOUNDARY = ["cxx_compiler", "glibc", "libstdcxx_max_glibcxx", "os_release"]


def _campaign(bundle: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(path.read_text())
        for path in sorted(_RESULTS.glob(f"pod-20260816-{bundle}-*.json"))
    ]
    assert rows, f"the committed 2026-08-16 {bundle} rows are missing"
    return rows


@pytest.mark.parametrize("bundle,expected_rows", [("b12", 7), ("b13", 3)])
def test_the_measured_pod_campaign_still_retains_its_boundary(
    bundle: str, expected_rows: int
) -> None:
    """The 2026-08-16 rows are the evidence tcg#4 rests on: fold them, not prose."""

    folded = _report.report(_campaign(bundle))
    assert folded["rows"] == expected_rows
    assert folded["inconclusive_rows"] == 0
    assert folded["failures"] == 1
    assert folded["retain"] == _RETAINED_BOUNDARY
    assert "insufficient data" not in folded["verdict"]
    for axis in ("machine", "host_isa_level", "host_isa_features", "torch_cxx11_abi"):
        assert axis in folded["unmeasured"]


def test_both_load_directions_across_the_retained_boundary_really_failed() -> None:
    """Each direction failed for its OWN reason; one direction proves neither."""

    failures = {
        row["build_axes"]["os_release"]: row
        for row in _campaign("b12") + _campaign("b13")
        if not row["load_ok"]
    }
    assert sorted(failures) == ["debian-12", "debian-13"]
    assert "executable stack" in failures["debian-12"]["error"]
    assert "GLIBC_2.38" in failures["debian-13"]["error"]
    for row in failures.values():
        assert sorted(row["diff_axes"]) == _RETAINED_BOUNDARY


def test_the_axes_measured_inert_on_a_cpu_target_are_candidates_not_changes() -> None:
    folded = _report.report(_campaign("b12"))
    for axis in ("python_abi", "triton", "torch_version"):
        assert axis in folded["coarsening_candidates"]
        assert folded["contingency"][axis]["differed_failed"] == 0
