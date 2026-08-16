"""Candidate-host half of the tcg#4 portability matrix.

Takes a probe bundle from `probe_build.py`, records the SAME per-axis facts
for this host, then attempts import -> runner -> bind -> execute -> output
comparison, and emits exactly one matrix row.

Run:  python probe_run.py <bundle-dir> --output <row.json>

The row never hides a failure: load/exec/output are separate booleans and the
first failing stage carries its error text.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from axes import AXES_SCHEMA_VERSION, AXIS_NAMES, record_axes, validate_row


def run_probe(bundle: Path) -> dict[str, Any]:
    probe = json.loads((bundle / "probe.json").read_text())
    if probe["schema"] != AXES_SCHEMA_VERSION:
        raise SystemExit(f"bundle schema {probe['schema']} != {AXES_SCHEMA_VERSION}")
    host_axes = record_axes()
    build_axes = probe["build_axes"]
    row: dict[str, Any] = {
        "schema": AXES_SCHEMA_VERSION,
        "key": probe["key"],
        "build_axes": build_axes,
        "host_axes": host_axes,
        "diff_axes": sorted(
            name for name in AXIS_NAMES if host_axes[name] != build_axes[name]
        ),
        "load_ok": False,
        "exec_ok": False,
        "output_ok": False,
    }

    try:
        import torch
        from tensorfs import LocalCAS

        from torchcg import Engine
    except Exception as error:  # noqa: BLE001 - the row reports, never raises
        # The host could not ATTEMPT the probe. That is not evidence that its
        # axes are incompatible, and counting it as a failure would mark every
        # differing axis RETAIN for a reason that has nothing to do with
        # portability. Inconclusive rows are excluded from the contingency.
        row["error"] = f"import: {error}"
        row["inconclusive"] = True
        return row

    with tempfile.TemporaryDirectory(prefix="tfs-probe-run-") as scratch:
        scratch_path = Path(scratch)
        try:
            engine = Engine(LocalCAS(scratch_path / "cas"))
            engine.import_artifact(probe["key"], bundle / "artifact.tar.gz")
            runner = engine.runner(probe["key"], scratch_path / "loaded")
            if runner is None:
                row["error"] = "load: runner resolved to None for the probe key"
                return row
            constants = {
                name: torch.tensor(values) for name, values in probe["constants"].items()
            }
            runner.bind(constants, device="cpu")
            row["load_ok"] = True
        except Exception as error:  # noqa: BLE001
            row["error"] = f"load: {error}"
            return row

        try:
            actual = runner(torch.tensor(probe["input"]))
            row["exec_ok"] = True
        except Exception as error:  # noqa: BLE001
            row["error"] = f"exec: {error}"
            return row

        expected = torch.tensor(probe["expected"])
        try:
            torch.testing.assert_close(
                actual, expected, rtol=probe["rtol"], atol=probe["atol"]
            )
            row["output_ok"] = True
        except AssertionError as error:
            row["error"] = f"output: {error}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    row = run_probe(arguments.bundle)
    validate_row(row)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(row, indent=2) + "\n")
    flags = {name: row[name] for name in ("load_ok", "exec_ok", "output_ok")}
    print(f"row: {arguments.output}")
    print(f"  diff_axes: {row['diff_axes']}")
    print(f"  {flags}" + (f"  error: {row.get('error')}" if "error" in row else ""))


if __name__ == "__main__":
    main()
