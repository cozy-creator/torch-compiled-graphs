"""Orchestrate the tcg#4 portability matrix across many hosts.

One command ships the probe bundle to every target in a manifest, runs
`probe_run.py` there, collects the rows back, folds them with
`matrix_report.py`, and writes a dated report.

    python run_matrix.py --targets targets.example.json --bundle bundle/ \
        --out results/ [--dry-run] [--only name,name] [--include-pod-only]

Transports are pluggable and every one of them is dry-runnable: `--dry-run`
prints the exact command sequence per target and executes nothing, so a pod
campaign can be reviewed before a cent is spent.

    local   run on this host (the build host's own row)
    docker  run in a local container from an image (costs nothing)
    ssh     ship to a remote endpoint, run, fetch the row back (pods)

A target that cannot even attempt the probe — no torch, no package — yields
an INCONCLUSIVE row, never a failure. Conflating "this host is incompatible"
with "this host could not try" would silently mark innocent axes as
incompatible, which is the one mistake this matrix exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parents[1]
TARGETS_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class Target:
    """One host class in the matrix, with the reason it is in the matrix."""

    name: str
    transport: str
    axis_intent: str
    obtain: str
    pod_only: bool = False
    image: str | None = None
    endpoint: str | None = None
    python: str = "python3"
    setup: tuple[str, ...] = ()
    remote_dir: str = "/tmp/tcg4-probe"

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> Target:
        missing = {"name", "transport", "axis_intent", "obtain"} - set(raw)
        if missing:
            raise ValueError(f"target is missing {sorted(missing)}")
        if raw["transport"] not in {"local", "docker", "ssh"}:
            raise ValueError(f"unknown transport {raw['transport']!r} for {raw['name']!r}")
        if raw["transport"] == "docker" and not raw.get("image"):
            raise ValueError(f"docker target {raw['name']!r} needs an image")
        if raw["transport"] == "ssh" and not raw.get("endpoint"):
            raise ValueError(f"ssh target {raw['name']!r} needs an endpoint")
        return Target(
            name=raw["name"],
            transport=raw["transport"],
            axis_intent=raw["axis_intent"],
            obtain=raw["obtain"],
            pod_only=bool(raw.get("pod_only", False)),
            image=raw.get("image"),
            endpoint=raw.get("endpoint"),
            python=raw.get("python", "python3"),
            setup=tuple(raw.get("setup", ())),
            remote_dir=raw.get("remote_dir", "/tmp/tcg4-probe"),
        )


@dataclass(frozen=True)
class Command:
    """One step of a target's plan: what it is, and exactly what runs."""

    description: str
    argv: tuple[str, ...]
    # Steps that may fail without failing the target (teardown, best effort).
    tolerate_failure: bool = False

    def shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


@dataclass
class TargetOutcome:
    target: Target
    row_path: Path | None = None
    error: str | None = None
    commands: list[Command] = field(default_factory=list)


def load_targets(path: Path) -> list[Target]:
    raw = json.loads(path.read_text())
    if raw.get("schema") != TARGETS_SCHEMA_VERSION:
        raise ValueError(f"targets schema {raw.get('schema')!r} != {TARGETS_SCHEMA_VERSION}")
    targets = [Target.from_dict(entry) for entry in raw["targets"]]
    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise ValueError("target names must be unique")
    return targets


def _probe_argv(target: Target, bundle: str, row: str) -> str:
    """The one command every transport ultimately runs on its host."""

    script = "{repo}/benchmarks/host_fingerprint/probe_run.py"
    return f"{target.python} {script} {bundle} --output {row}"


def plan_commands(target: Target, bundle: Path, out_dir: Path) -> list[Command]:
    """Every step for one target, as data. Pure: builds no files, runs nothing.

    This is what `--dry-run` prints, which makes a pod campaign reviewable
    before it is paid for.
    """

    row_name = f"{target.name}.json"
    if target.transport == "local":
        inner = _probe_argv(target, str(bundle), str(out_dir / row_name)).replace(
            "{repo}", str(REPO)
        )
        return [
            *(
                Command(f"setup: {step}", ("bash", "-lc", step))
                for step in target.setup
            ),
            Command("probe", ("bash", "-lc", inner)),
        ]

    if target.transport == "docker":
        assert target.image is not None
        inner_probe = _probe_argv(target, "/bundle", f"/out/{row_name}").replace(
            "{repo}", "/repo"
        )
        script = " && ".join([*target.setup, inner_probe])
        return [
            Command(
                f"pull {target.image}",
                ("docker", "pull", target.image),
                tolerate_failure=True,
            ),
            Command(
                f"run probe in {target.image}",
                (
                    "docker",
                    "run",
                    "--rm",
                    "--network=host",
                    "-v",
                    f"{REPO}:/repo:ro",
                    "-v",
                    f"{bundle}:/bundle:ro",
                    "-v",
                    f"{out_dir}:/out",
                    target.image,
                    "bash",
                    "-lc",
                    script,
                ),
            ),
        ]

    assert target.endpoint is not None
    remote = target.remote_dir
    inner_probe = _probe_argv(target, f"{remote}/bundle", f"{remote}/{row_name}").replace(
        "{repo}", f"{remote}/repo"
    )
    script = " && ".join([*target.setup, inner_probe])
    return [
        Command("make remote dir", ("ssh", target.endpoint, f"mkdir -p {remote}")),
        Command(
            "ship harness",
            (
                "rsync",
                "-a",
                "--delete",
                f"{REPO}/benchmarks",
                f"{REPO}/src",
                f"{target.endpoint}:{remote}/repo/",
            ),
        ),
        Command(
            "ship bundle",
            ("rsync", "-a", "--delete", f"{bundle}/", f"{target.endpoint}:{remote}/bundle/"),
        ),
        Command("probe", ("ssh", target.endpoint, script)),
        Command(
            "collect row",
            ("rsync", "-a", f"{target.endpoint}:{remote}/{row_name}", str(out_dir / row_name)),
        ),
        Command(
            "teardown",
            ("ssh", target.endpoint, f"rm -rf {remote}"),
            tolerate_failure=True,
        ),
    ]


def run_target(
    target: Target, bundle: Path, out_dir: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> TargetOutcome:
    commands = plan_commands(target, bundle, out_dir)
    outcome = TargetOutcome(target=target, commands=commands)
    for command in commands:
        print(f"  [{target.name}] {command.description}", flush=True)
        try:
            finished = subprocess.run(
                command.argv, capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            if command.tolerate_failure:
                continue
            outcome.error = f"{command.description}: {error}"
            return outcome
        if finished.returncode != 0 and not command.tolerate_failure:
            tail = (finished.stderr or finished.stdout or "").strip().splitlines()[-6:]
            outcome.error = f"{command.description}: exit {finished.returncode}: " + " | ".join(
                tail
            )
            return outcome
    row_path = out_dir / f"{target.name}.json"
    if not row_path.is_file():
        outcome.error = "probe produced no row"
        return outcome
    outcome.row_path = row_path
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated target names")
    parser.add_argument("--include-pod-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    arguments = parser.parse_args()

    targets = load_targets(arguments.targets)
    if arguments.only:
        wanted = {name.strip() for name in arguments.only.split(",") if name.strip()}
        unknown = wanted - {target.name for target in targets}
        if unknown:
            raise SystemExit(f"unknown target(s): {sorted(unknown)}")
        targets = [target for target in targets if target.name in wanted]
    if not arguments.include_pod_only:
        skipped = [target.name for target in targets if target.pod_only]
        targets = [target for target in targets if not target.pod_only]
        if skipped:
            print(f"skipping pod-only targets (pass --include-pod-only): {skipped}")

    if arguments.dry_run:
        for target in targets:
            print(f"\n== {target.name} [{target.transport}] varies: {target.axis_intent}")
            for command in plan_commands(target, arguments.bundle, arguments.out):
                marker = "  (tolerated)" if command.tolerate_failure else ""
                print(f"  # {command.description}{marker}\n  {command.shell()}")
        print(f"\ndry run: {len(targets)} target(s), nothing executed")
        return 0

    if not (arguments.bundle / "probe.json").is_file():
        raise SystemExit(f"no probe bundle at {arguments.bundle} — run probe_build.py first")
    arguments.out.mkdir(parents=True, exist_ok=True)

    outcomes = [
        run_target(target, arguments.bundle, arguments.out, arguments.timeout)
        for target in targets
    ]
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome.error:
            print(f"  [{outcome.target.name}] FAILED TO PROBE: {outcome.error}")
            continue
        assert outcome.row_path is not None
        rows.append(json.loads(outcome.row_path.read_text()))

    if not rows:
        print("no rows collected; nothing to fold")
        return 1

    from matrix_report import report

    folded = report(rows)
    report_path = arguments.out / f"report-{date.today().isoformat()}.json"
    report_path.write_text(json.dumps(folded, indent=2) + "\n")
    print(f"\nrows: {len(rows)}   report: {report_path}")
    print(f"verdict: {folded['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
