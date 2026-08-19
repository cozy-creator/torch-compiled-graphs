"""Cold AOTI compile comparison on a diffusion-shaped graph specialization.

Run each arm in a fresh process with a fresh ``TORCHINDUCTOR_CACHE_DIR``.
This harness is intentionally outside the wheel: it measures the library's
private policy without creating a supported compiler-options surface.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Mapping
from typing import Any

import torch

from torchcg import _wrapper_split
from torchcg.compiler import (
    _aot_compile,
    _compiler_options,
    _compiling_under_export_context,
)
from torchcg.host_isa import _impose_host_policy


class _TimestepEmbedding(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )

    def forward(self, timestep: Any) -> Any:
        return self.proj(timestep.reshape(-1, 1))


class _CrossAttention(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, width: int, context_width: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.to_q = torch.nn.Linear(width, width, bias=False)
        self.to_k = torch.nn.Linear(context_width, width, bias=False)
        self.to_v = torch.nn.Linear(context_width, width, bias=False)
        self.to_out = torch.nn.Linear(width, width, bias=False)
        self.scale = width**-0.5

    def forward(self, value: Any, context: Any) -> Any:
        batch, channels, height, width = value.shape
        flat = self.norm(value.reshape(batch, channels, height * width).transpose(1, 2))
        query = self.to_q(flat)
        key = self.to_k(context)
        projected = self.to_v(context)
        attention = torch.softmax((query @ key.transpose(1, 2)) * self.scale, dim=-1)
        output = self.to_out(attention @ projected)
        return value + output.transpose(1, 2).reshape(batch, channels, height, width)


class RepresentativeDenoiser(torch.nn.Module):  # type: ignore[misc]
    """Small enough for CI hardware, but shaped like a diffusers UNet class."""

    def __init__(self, width: int = 64, context_width: int = 64) -> None:
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, width, 3, padding=1)
        self.time = _TimestepEmbedding(width)
        self.down = torch.nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.mid_norm = torch.nn.GroupNorm(8, width)
        self.mid = torch.nn.Conv2d(width, width, 3, padding=1)
        self.attention = _CrossAttention(width, context_width)
        self.up = torch.nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.out_norm = torch.nn.GroupNorm(8, width)
        self.conv_out = torch.nn.Conv2d(width, 4, 3, padding=1)

    def forward(self, sample: Any, timestep: Any, context: Any) -> Any:
        hidden = self.conv_in(sample)
        skip = hidden
        hidden = hidden + self.time(timestep)[:, :, None, None]
        hidden = self.down(hidden)
        hidden = torch.nn.functional.silu(self.mid_norm(self.mid(hidden)))
        hidden = self.attention(hidden, context)
        hidden = self.up(hidden) + skip
        return self.conv_out(torch.nn.functional.silu(self.out_norm(hidden)))


_PARTITION_KEYS: Mapping[str, tuple[str, ...]] = {
    "lowering_s": ("GraphLowering.run",),
    "codegen_s": ("GraphLowering.codegen",),
    "host_compile_s": ("AotCodeCompiler.compile",),
    "graph_passes_s": (
        "_recursive_pre_grad_passes",
        "_recursive_joint_graph_passes",
        "_recursive_post_grad_passes",
    ),
}


def _phase_snapshot() -> dict[str, float]:
    from torch._dynamo import utils

    return {
        str(name): float(sum(values)) for name, values in utils.compilation_time_metrics.items()
    }


def _partition_problems(wall: float, partition: Mapping[str, float]) -> list[str]:
    expected = {*_PARTITION_KEYS, "compile_other_s"}
    missing = sorted(expected - set(partition))
    problems = [f"missing partition members {missing!r}"] if missing else []
    total = sum(partition.get(name, 0.0) for name in expected)
    if abs(total - wall) > 0.05:
        problems.append(f"partition sums to {total:.3f}s, wall is {wall:.3f}s")
    if partition.get("compile_other_s", 0.0) < -0.05:
        problems.append("compile_other_s is negative; named members overlap")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 4), required=True)
    arguments = parser.parse_args()

    torch.manual_seed(7)
    model = RepresentativeDenoiser().eval()
    inputs = (
        torch.randn(1, 4, 32, 32),
        torch.tensor([500.0]),
        torch.randn(1, 77, 64),
    )
    program = torch.export.export(model, inputs)
    options = _compiler_options()
    options["compile_threads"] = arguments.threads

    _impose_host_policy()
    _wrapper_split.install()
    before = _phase_snapshot()
    started = time.monotonic()
    with _compiling_under_export_context(program):
        files = _aot_compile(
            program.module(check_guards=False),
            inputs,
            {},
            options=options,
        )
    wall = time.monotonic() - started
    after = _phase_snapshot()
    raw = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in set(before) | set(after)}
    partition = {
        label: sum(raw.get(name, 0.0) for name in names) for label, names in _PARTITION_KEYS.items()
    }
    partition["compile_other_s"] = wall - sum(partition.values())
    problems = _partition_problems(wall, partition)
    if problems:
        raise RuntimeError("compile partition does not close: " + "; ".join(problems))

    print(
        json.dumps(
            {
                "threads": arguments.threads,
                "wall_s": round(wall, 3),
                "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                "files": len(files),
                "partition": {name: round(value, 3) for name, value in partition.items()},
                "partition_problems": problems,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
