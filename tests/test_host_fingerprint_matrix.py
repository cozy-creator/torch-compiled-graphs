"""tcg#4 phase-2 logic: inconclusive rows, the census projection, and the
runner's plan. All provable without torch, docker, a network or a pod."""

from __future__ import annotations

import importlib.util
import json
import sys
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
    # `dataclasses` resolves annotations through sys.modules[cls.__module__],
    # so a module executed outside it cannot define a dataclass.
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


_axes = _load("axes")
_report = _load("matrix_report")
_census = _load("fleet_census")
_runner = _load("run_matrix")


def _row(
    *,
    diff: dict[str, str] | None = None,
    load_ok: bool = True,
    exec_ok: bool = True,
    output_ok: bool = True,
    inconclusive: bool = False,
) -> dict[str, Any]:
    build = {name: f"value-{name}" for name in _axes.AXIS_NAMES}
    host = dict(build)
    host.update(diff or {})
    row: dict[str, Any] = {
        "schema": _axes.AXES_SCHEMA_VERSION,
        "build_axes": build,
        "host_axes": host,
        "diff_axes": sorted(name for name in build if host[name] != build[name]),
        "load_ok": load_ok,
        "exec_ok": exec_ok,
        "output_ok": output_ok,
    }
    if inconclusive:
        row["inconclusive"] = True
    return row


# --- inconclusive rows -------------------------------------------------------


def test_an_inconclusive_row_never_marks_its_axes_incompatible() -> None:
    """The whole point: 'could not try' must not read as 'incompatible'."""

    rows = [
        _row(diff={"glibc": "older"}, load_ok=False, exec_ok=False, output_ok=False,
             inconclusive=True),
        _row(diff={"triton": "other"}),
    ]
    folded = _report.report(rows)
    assert folded["rows"] == 2
    assert folded["conclusive_rows"] == 1
    assert folded["inconclusive_rows"] == 1
    assert folded["failures"] == 0
    # glibc differed only on the inconclusive row, so it is UNMEASURED, and
    # emphatically not RETAIN.
    assert "glibc" not in folded["retain"]
    assert "glibc" in folded["unmeasured"]
    assert folded["coarsening_candidates"] == ["triton"]


def test_the_same_row_without_the_flag_would_have_condemned_the_axis() -> None:
    """Proves the flag is load-bearing, not decorative."""

    condemning = _row(diff={"glibc": "older"}, load_ok=False, exec_ok=False, output_ok=False)
    assert "glibc" in _report.report([condemning, _row()])["retain"]


def test_validation_refuses_an_inconclusive_row_that_claims_it_loaded() -> None:
    row = _row(load_ok=True, inconclusive=True)
    with pytest.raises(ValueError):
        _axes.validate_row(row)


def test_an_all_inconclusive_matrix_reports_insufficient_data() -> None:
    rows = [
        _row(diff={"glibc": "a"}, load_ok=False, exec_ok=False, output_ok=False,
             inconclusive=True),
        _row(diff={"glibc": "b"}, load_ok=False, exec_ok=False, output_ok=False,
             inconclusive=True),
    ]
    folded = _report.report(rows)
    assert folded["conclusive_rows"] == 0
    assert "insufficient data" in folded["verdict"]
    assert folded["coarsening_candidates"] == []


# --- fleet census ------------------------------------------------------------


def test_census_counts_each_host_once_however_chatty() -> None:
    export = [
        json.dumps({"host_id": "a", "axes": {"glibc": "2.39", "triton": "3.1.0"}}),
        json.dumps({"host_id": "a", "axes": {"glibc": "2.39", "triton": "3.1.0"}}),
        json.dumps({"host_id": "a", "axes": {"glibc": "2.39", "triton": "3.1.0"}}),
        json.dumps({"host_id": "b", "axes": {"glibc": "2.35", "triton": "3.1.0"}}),
    ]
    census = _census.parse_export(export, "test")
    assert census["hosts"] == 2
    assert census["axis_values"]["glibc"] == {"2.39": 1, "2.35": 1}
    assert census["axis_values"]["triton"] == {"3.1.0": 2}


def test_census_surfaces_malformed_and_unknown_rather_than_swallowing_them() -> None:
    export = [
        json.dumps({"host_id": "a", "axes": {"glibc": "2.39", "not_an_axis": "x"}}),
        "{ not json",
        json.dumps({"missing": "host_id"}),
    ]
    census = _census.parse_export(export, "test")
    assert census["hosts"] == 1
    assert census["malformed_lines"] == 2
    assert census["unknown_axes"] == ["not_an_axis"]
    assert "not_an_axis" not in census["axis_values"]
    assert "triton" in census["missing_axes"]


# --- projection --------------------------------------------------------------


def test_projection_gain_is_zero_for_an_axis_the_whole_fleet_agrees_on() -> None:
    census = {
        "hosts": 10,
        "axis_values": {"glibc": {"2.39": 10}, "triton": {"3.1.0": 6, "absent": 4}},
    }
    uniform = _report.projected_cache_hit_improvement(census, "glibc")
    assert uniform["projected_gain"] == pytest.approx(0.0)
    # A varying axis is where the gain lives.
    varying = _report.projected_cache_hit_improvement(census, "triton")
    assert varying["projected_gain"] > 0.0


def test_projection_arithmetic_matches_the_stated_model() -> None:
    census = {"hosts": 4, "axis_values": {"a": {"x": 2, "y": 2}, "b": {"p": 2, "q": 2}}}
    # agreement per axis = 0.5^2 + 0.5^2 = 0.5; before = 0.25; after dropping a = 0.5
    projected = _report.projected_cache_hit_improvement(census, "a")
    assert projected["hit_rate_before"] == pytest.approx(0.25)
    assert projected["hit_rate_after"] == pytest.approx(0.5)
    assert projected["projected_gain"] == pytest.approx(0.25)


def test_projection_always_declares_itself_a_projection() -> None:
    census = {"hosts": 2, "axis_values": {"triton": {"a": 1, "b": 1}}}
    projected = _report.projected_cache_hit_improvement(census, "triton")
    assert projected["projection_not_measurement"] is True
    assert projected["assumes_axis_independence"] is True


def test_projection_refuses_an_axis_the_census_never_saw() -> None:
    projected = _report.projected_cache_hit_improvement({"axis_values": {}}, "glibc")
    assert projected["insufficient_data"] is True
    assert projected["projection_not_measurement"] is True


def test_report_keeps_measurement_and_projection_separate() -> None:
    census = {"hosts": 3, "axis_values": {"triton": {"a": 2, "b": 1}}}
    folded = _report.report([_row(diff={"triton": "other"}), _row()], census)
    assert "cache_hit_improvement" in folded  # measured, over observed hosts
    assert folded["projection"]["per_axis"][0]["projection_not_measurement"] is True
    assert "projection" in folded and "note" in folded["projection"]


def test_synthetic_fleet_is_correlated_like_a_real_one() -> None:
    census = _census.parse_export(_census.synthetic_export(200), "synthetic")
    assert census["hosts"] == 200
    # The two image generations move glibc and torch together, so those axes
    # have the same shape — which is exactly what the independence assumption
    # cannot see.
    assert len(census["axis_values"]["glibc"]) == 2
    assert len(census["axis_values"]["torch_version"]) == 2


# --- runner plan -------------------------------------------------------------


def test_every_transport_plans_without_executing_anything() -> None:
    bundle, out = Path("/tmp/bundle"), Path("/tmp/out")
    for raw, expect in (
        ({"name": "l", "transport": "local", "axis_intent": "i", "obtain": "o"}, "probe_run.py"),
        (
            {
                "name": "d",
                "transport": "docker",
                "image": "python:3.13-slim",
                "axis_intent": "i",
                "obtain": "o",
            },
            "docker",
        ),
        (
            {
                "name": "s",
                "transport": "ssh",
                "endpoint": "root@host",
                "axis_intent": "i",
                "obtain": "o",
            },
            "rsync",
        ),
    ):
        commands = _runner.plan_commands(_runner.Target.from_dict(raw), bundle, out)
        assert commands, "a target must plan at least one command"
        rendered = " ".join(command.shell() for command in commands)
        assert expect in rendered
        assert "probe_run.py" in rendered


def test_a_malformed_target_is_refused_at_load() -> None:
    for raw in (
        {"name": "x", "transport": "docker", "axis_intent": "i", "obtain": "o"},  # no image
        {"name": "x", "transport": "ssh", "axis_intent": "i", "obtain": "o"},  # no endpoint
        {"name": "x", "transport": "carrier-pigeon", "axis_intent": "i", "obtain": "o"},
        {"name": "x", "transport": "local"},  # no intent/obtain
    ):
        with pytest.raises(ValueError):
            _runner.Target.from_dict(raw)


def test_the_shipped_inventory_parses_and_declares_its_pod_costs() -> None:
    targets = _runner.load_targets(_HARNESS / "targets.example.json")
    assert len(targets) >= 4
    assert any(not target.pod_only for target in targets), "some targets must be free"
    assert any(target.pod_only for target in targets), "the real matrix needs pods"
    for target in targets:
        assert target.axis_intent, f"{target.name} must say which axis it varies"
        assert target.obtain, f"{target.name} must say how to obtain it"
