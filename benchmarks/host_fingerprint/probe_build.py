"""Build-host half of the tcg#4 portability matrix.

Compiles ONE small real AOTI graph through the sealed Engine, exports the
verified artifact, and writes a self-contained probe bundle:

    bundle/
      artifact.tar.gz   the exact exported artifact envelope
      probe.json        key, per-axis build-host record, pinned input,
                        expected output, tolerances, binding constants

Run:  python probe_build.py --output <bundle-dir> [--target sm_NN]

`--target` defaults to `cpu`; a concrete `sm_NN` builds the same probe as a
CUDA graph on the visible device, and the bundle records the target so
`probe_run.py` binds and executes on the matching device.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from axes import AXES_SCHEMA_VERSION, record_axes, worker_style_toolchain
from tensorfs import LocalCAS

from torchcg import Engine, GraphClassSpec, RuntimeCompatibility, build_call_ingress

PROBE_LENGTH = 64
RTOL = 1e-5
ATOL = 1e-6


class ProbeBlock(torch.nn.Module):  # type: ignore[misc]
    """Small but multi-op: scale is a durable named buffer, so the remote
    half exercises real constant binding rather than an empty bind."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.linspace(0.5, 1.5, PROBE_LENGTH))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(value * 2.0 + 1.0) * self.scale


def build_bundle(output: Path, target: str = "cpu") -> None:
    output.mkdir(parents=True, exist_ok=True)
    axes = record_axes()
    device = "cuda" if target.startswith("sm_") else "cpu"

    module = ProbeBlock().to(device)
    example = torch.linspace(-3.0, 3.0, PROBE_LENGTH, device=device)
    program = torch.export.export(module, (example,))
    ingress = build_call_ingress(program, ("value",), (example,), {})
    spec = GraphClassSpec(
        "host-fingerprint-probe",
        "portability",
        program,
        {
            "v": 3,
            "lifted_inputs": [],
            "pytree": {"in": "leaf", "out": "leaf", "ingress": ingress.as_dict()},
            "specialization": {},
        },
    )
    runtime = RuntimeCompatibility(target, toolchain=worker_style_toolchain(axes))

    with tempfile.TemporaryDirectory(prefix="tfs-probe-") as scratch:
        scratch_path = Path(scratch)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(scratch_path / "inductor-cache"))
        engine = Engine(LocalCAS(scratch_path / "cas"))
        result = engine.compile(spec, runtime, scratch_path / "minted")
        key = result.compiled_graph.key
        engine.export_artifact(key, output / "artifact.tar.gz")

    expected = module(example)
    probe = {
        "schema": AXES_SCHEMA_VERSION,
        "key": key,
        "target": target,
        "build_axes": axes,
        "input": example.cpu().tolist(),
        "expected": expected.cpu().tolist(),
        "rtol": RTOL,
        "atol": ATOL,
        "constants": {"scale": module.scale.cpu().tolist()},
    }
    (output / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    print(f"probe bundle: {output}")
    print(f"  key: {key}")
    print(f"  axes recorded: {len(axes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="bundle directory")
    parser.add_argument("--target", default="cpu", help="'cpu' (default) or a concrete 'sm_NN'")
    arguments = parser.parse_args()
    build_bundle(arguments.output, arguments.target)


if __name__ == "__main__":
    main()
