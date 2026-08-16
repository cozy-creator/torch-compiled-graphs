"""Turn a fleet telemetry export into the axis-value census the projection needs.

    python fleet_census.py --input export.jsonl --output census.json
    python fleet_census.py --synthetic 400 --output census.json   # no prod access

INPUT FORMAT (JSON Lines, one host per line). Produce it from whatever the
fleet already records — worker activity events, a boot-time inventory, an
image manifest — as long as each line is one HOST OBSERVATION:

    {"host_id": "<stable id>", "axes": {"glibc": "glibc-2.35", "triton": "3.1.0", ...}}

`host_id` deduplicates repeat observations of the same host: a fleet where one
chatty worker reports hourly must not out-vote a hundred quiet ones. Axis
names must match `axes.AXIS_NAMES`; unknown names are reported and ignored so
a telemetry schema drift is visible instead of silently skewing the census.

THIS SCRIPT NEVER QUERIES PRODUCTION. It parses an export you supply. The
projection built on its output is only as good as the export: a census taken
from one region, one image generation, or one week of traffic will overstate
agreement and therefore understate the gain from dropping an axis.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from axes import AXIS_NAMES

CENSUS_SCHEMA_VERSION = 1


def parse_export(lines: list[str], source: str) -> dict[str, Any]:
    """Count distinct axis values per axis, one vote per distinct host."""

    seen_hosts: dict[str, dict[str, str]] = {}
    unknown_axes: set[str] = set()
    malformed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            host_id = str(record["host_id"])
            axes = dict(record["axes"])
        except (ValueError, KeyError, TypeError):
            malformed += 1
            continue
        unknown_axes |= set(axes) - set(AXIS_NAMES)
        # Last observation of a host wins; hosts are counted once.
        seen_hosts[host_id] = {name: str(value) for name, value in axes.items()}

    distributions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for axes in seen_hosts.values():
        for name in AXIS_NAMES:
            if name in axes:
                distributions[name][axes[name]] += 1

    return {
        "schema": CENSUS_SCHEMA_VERSION,
        "source": source,
        "hosts": len(seen_hosts),
        "malformed_lines": malformed,
        "unknown_axes": sorted(unknown_axes),
        "missing_axes": sorted(set(AXIS_NAMES) - set(distributions)),
        "axis_values": {name: dict(values) for name, values in sorted(distributions.items())},
    }


def synthetic_export(hosts: int, seed: int = 1960) -> list[str]:
    """A believable-shaped fleet for demonstrating the pipeline.

    Deliberately correlated: most hosts run one of two images, so glibc,
    libstdcxx and torch move together. That is what a real fleet looks like,
    and it is precisely the correlation the projection's independence
    assumption cannot see — the synthetic data makes that caveat visible
    rather than theoretical.
    """

    generator = random.Random(seed)
    images = [
        {
            "os_release": "ubuntu-24.04",
            "glibc": "glibc-2.39",
            "libstdcxx_max_glibcxx": "GLIBCXX_3.4.33",
            "cxx_compiler": "c++ (Ubuntu 13.3.0) 13.3.0",
            "torch_version": "2.13.0+cu130",
            "torch_git": "cf30153c",
            "torch_config_digest": "5f31e143bbf0b632",
        },
        {
            "os_release": "ubuntu-22.04",
            "glibc": "glibc-2.35",
            "libstdcxx_max_glibcxx": "GLIBCXX_3.4.30",
            "cxx_compiler": "c++ (Ubuntu 11.4.0) 11.4.0",
            "torch_version": "2.12.1+cu124",
            "torch_git": "a1b2c3d4",
            "torch_config_digest": "9c02aa71de44bb18",
        },
    ]
    lines = []
    for index in range(hosts):
        image = images[0] if generator.random() < 0.72 else images[1]
        axes = {
            **image,
            "machine": "x86_64",
            "python_abi": "cpython-3.13-cpython-313-x86_64-linux-gnu",
            "torch_cxx11_abi": "1",
            "triton": "3.1.0" if generator.random() < 0.95 else "absent",
            "host_isa_level": "x86-64-v3" if generator.random() < 0.6 else "x86-64-v4",
            "host_isa_features": "abm,avx,avx2,bmi1,bmi2,f16c,fma,movbe,xsave",
        }
        lines.append(json.dumps({"host_id": f"worker-{index:04d}", "axes": axes}))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL export (one host per line)")
    parser.add_argument("--synthetic", type=int, metavar="HOSTS", help="emit a synthetic fleet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-out", type=Path, help="also write the synthetic export here")
    arguments = parser.parse_args()

    if bool(arguments.input) == bool(arguments.synthetic):
        raise SystemExit("pass exactly one of --input or --synthetic")

    if arguments.synthetic:
        lines = synthetic_export(arguments.synthetic)
        source = f"synthetic fleet of {arguments.synthetic} hosts (NOT production)"
        if arguments.export_out:
            arguments.export_out.write_text("\n".join(lines) + "\n")
    else:
        lines = arguments.input.read_text().splitlines()
        source = str(arguments.input)

    census = parse_export(lines, source)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(census, indent=2) + "\n")
    print(f"census: {arguments.output}")
    print(f"  hosts: {census['hosts']}   axes: {len(census['axis_values'])}")
    if census["missing_axes"]:
        print(f"  missing axes (no projection for these): {census['missing_axes']}")
    if census["unknown_axes"]:
        print(f"  unknown axes ignored: {census['unknown_axes']}")


if __name__ == "__main__":
    main()
