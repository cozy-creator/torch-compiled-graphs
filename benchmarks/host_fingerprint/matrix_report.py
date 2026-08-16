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


def inconclusive(row: dict[str, Any]) -> bool:
    """Did the host fail to ATTEMPT the probe (no torch, no package)?

    Such a row carries real axis values but no compatibility evidence. It is
    excluded from the contingency: treating "could not try" as "incompatible"
    would mark every differing axis RETAIN for the wrong reason, which is
    exactly the misattribution this matrix exists to prevent.
    """

    return bool(row.get("inconclusive", False))


def conclusive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not inconclusive(row)]


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


def projected_cache_hit_improvement(
    census: dict[str, Any], dropped_axis: str
) -> dict[str, Any]:
    """Projected gain from dropping ONE axis, over a FLEET CENSUS.

    This is a PROJECTION, not a measurement. The census gives per-axis value
    distributions (marginals), not the joint distribution, so the arithmetic
    below assumes the axes vary independently across the fleet — stated in
    every returned record because a correlated fleet (one image pinning
    glibc, libstdc++ and torch together) makes the true gain smaller.

    For two hosts drawn at random, P(agree on axis X) = sum_v p_v^2. The
    cache-hit rate is the product over retained axes; dropping A removes its
    factor, so the gain is the difference.
    """

    distributions: dict[str, dict[str, int]] = census.get("axis_values", {})
    if dropped_axis not in distributions:
        return {
            "dropped_axis": dropped_axis,
            "projection_not_measurement": True,
            "insufficient_data": True,
            "reason": f"census records no distribution for {dropped_axis!r}",
        }

    def agreement(axis: str) -> float:
        counts = distributions[axis]
        total = sum(counts.values())
        if total <= 0:
            return 1.0
        return sum((count / total) ** 2 for count in counts.values())

    before = 1.0
    for axis in distributions:
        before *= agreement(axis)
    after = before / agreement(dropped_axis) if agreement(dropped_axis) else before
    return {
        "dropped_axis": dropped_axis,
        "projection_not_measurement": True,
        "assumes_axis_independence": True,
        "insufficient_data": False,
        "census_hosts": census.get("hosts"),
        "census_source": census.get("source"),
        "axes_in_census": len(distributions),
        "hit_rate_before": before,
        "hit_rate_after": after,
        "projected_gain": after - before,
    }


def report(
    rows: list[dict[str, Any]], census: dict[str, Any] | None = None
) -> dict[str, Any]:
    from axes import validate_row

    for row in rows:
        validate_row(row)
    conclusive = conclusive_rows(rows)
    table = axis_contingency(conclusive)
    classes = classify(table)
    folded: dict[str, Any] = {
        "rows": len(rows),
        "conclusive_rows": len(conclusive),
        "inconclusive_rows": len(rows) - len(conclusive),
        "failures": sum(1 for row in conclusive if _failed(row)),
        "contingency": table,
        **classes,
        "cache_hit_improvement": [
            cache_hit_improvement(conclusive, axis) for axis in classes["coarsening_candidates"]
        ],
        "verdict": (
            "insufficient data: a fingerprint change needs rows from at least "
            f"{MIN_HOSTS_FOR_A_VERDICT} distinct hosts that could ATTEMPT the probe"
            if len(_host_identities(conclusive)) < MIN_HOSTS_FOR_A_VERDICT
            else "measured; candidates above differed without a single failure"
        ),
    }
    if census is not None:
        folded["projection"] = {
            "note": (
                "projection over a fleet census, NOT measured portability; "
                "assumes axes vary independently, which a fleet of pinned "
                "images does not"
            ),
            "per_axis": [
                projected_cache_hit_improvement(census, axis)
                for axis in sorted(census.get("axis_values", {}))
            ],
        }
    return folded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="+", type=Path)
    parser.add_argument(
        "--census",
        type=Path,
        help="fleet census (fleet_census.py output) for the PROJECTED gain",
    )
    arguments = parser.parse_args()
    loaded = [json.loads(path.read_text()) for path in arguments.rows]
    census = json.loads(arguments.census.read_text()) if arguments.census else None
    print(json.dumps(report(loaded, census), indent=2))


if __name__ == "__main__":
    main()
