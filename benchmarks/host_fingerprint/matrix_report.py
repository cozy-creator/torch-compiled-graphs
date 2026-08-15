"""Fold portability-matrix rows into the per-axis report tcg#4 asks for.

Pure logic, importable without torch, so the fold rules are unit-testable.

Run:  python matrix_report.py results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_HOSTS_FOR_A_VERDICT = 2


def _failed(row: dict[str, Any]) -> bool:
    return not (row["load_ok"] and row["exec_ok"] and row["output_ok"])


def axis_contingency(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per axis: how often did a difference from the build host co-occur with
    a failure? Counts are per row (one candidate host each)."""

    axes = sorted(rows[0]["build_axes"]) if rows else []
    table = {
        axis: {"differed_failed": 0, "differed_passed": 0, "same_failed": 0, "same_passed": 0}
        for axis in axes
    }
    for row in rows:
        failed = _failed(row)
        for axis in axes:
            differed = axis in row["diff_axes"]
            bucket = ("differed" if differed else "same") + ("_failed" if failed else "_passed")
            table[axis][bucket] += 1
    return table


def classify(table: dict[str, dict[str, int]]) -> dict[str, list[str]]:
    """The issue's two lists, honestly derived.

    An axis is a COARSENING CANDIDATE only when it was actually observed to
    differ and no differing row ever failed. An axis is RETAIN when any
    differing row failed. An axis nobody ever saw differ is UNMEASURED —
    fail-closed means it is never a candidate.
    """

    candidates, retain, unmeasured = [], [], []
    for axis, counts in sorted(table.items()):
        if counts["differed_failed"] > 0:
            retain.append(axis)
        elif counts["differed_passed"] > 0:
            candidates.append(axis)
        else:
            unmeasured.append(axis)
    return {"coarsening_candidates": candidates, "retain": retain, "unmeasured": unmeasured}


def _host_identities(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Distinct hosts in the matrix: every candidate host plus the build host."""

    hosts: list[dict[str, str]] = []
    seen: set[str] = set()
    for axes in [rows[0]["build_axes"], *[row["host_axes"] for row in rows]] if rows else []:
        fingerprint = json.dumps(axes, sort_keys=True)
        if fingerprint not in seen:
            seen.add(fingerprint)
            hosts.append(axes)
    return hosts


def cache_hit_improvement(rows: list[dict[str, Any]], dropped_axis: str) -> dict[str, Any]:
    """Expected improvement from dropping ONE axis, over observed hosts.

    Of all unordered host pairs, how many newly share a cache key: they agree
    on every retained axis and disagreed only on the dropped one. Reported as
    counts plus the fraction; a one-host matrix states insufficient data
    instead of inventing a number.
    """

    hosts = _host_identities(rows)
    if len(hosts) < MIN_HOSTS_FOR_A_VERDICT:
        return {"dropped_axis": dropped_axis, "insufficient_data": True, "hosts": len(hosts)}
    total = newly_shared = already_shared = 0
    axes = sorted(hosts[0])
    for left, right in combinations(hosts, 2):
        total += 1
        disagreements = [axis for axis in axes if left[axis] != right[axis]]
        if not disagreements:
            already_shared += 1
        elif disagreements == [dropped_axis]:
            newly_shared += 1
    return {
        "dropped_axis": dropped_axis,
        "insufficient_data": False,
        "hosts": len(hosts),
        "pairs": total,
        "already_shared": already_shared,
        "newly_shared": newly_shared,
        "improvement": newly_shared / total if total else 0.0,
    }


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from axes import validate_row

    for row in rows:
        validate_row(row)
    table = axis_contingency(rows)
    classes = classify(table)
    return {
        "rows": len(rows),
        "failures": sum(1 for row in rows if _failed(row)),
        "contingency": table,
        **classes,
        "cache_hit_improvement": [
            cache_hit_improvement(rows, axis) for axis in classes["coarsening_candidates"]
        ],
        "verdict": (
            "insufficient data: a fingerprint change needs rows from at least "
            f"{MIN_HOSTS_FOR_A_VERDICT} distinct hosts"
            if len(_host_identities(rows)) < MIN_HOSTS_FOR_A_VERDICT
            else "measured; candidates above differed without a single failure"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="+", type=Path)
    arguments = parser.parse_args()
    loaded = [json.loads(path.read_text()) for path in arguments.rows]
    print(json.dumps(report(loaded), indent=2))


if __name__ == "__main__":
    main()
