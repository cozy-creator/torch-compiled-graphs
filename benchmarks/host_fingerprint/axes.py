"""Per-axis host/toolchain facts for the tcg#4 portability matrix.

Each axis is recorded separately so a matrix row can say WHICH facts differed
between the build host and a candidate host, not merely that the folded
16-hex toolchain digest changed. Nothing here alters the production
fingerprint; this is measurement vocabulary only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import sysconfig
from ctypes import util as ctypes_util
from pathlib import Path
from typing import Any

AXES_SCHEMA_VERSION = 1

# The axes the matrix records, in report order. `sm`, `graph` and the packaged
# host_isa facts ride the artifact itself; these are the host-side facts the
# worker's toolchain fold and deployment identity are built from.
AXIS_NAMES = (
    "machine",
    "os_release",
    "glibc",
    "libstdcxx_max_glibcxx",
    "cxx_compiler",
    "python_abi",
    "torch_version",
    "torch_git",
    "torch_cxx11_abi",
    "torch_config_digest",
    "triton",
    "host_isa_level",
    "host_isa_features",
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _os_release() -> str:
    path = Path("/etc/os-release")
    if not path.is_file():
        return f"{platform.system()}-{platform.release().split('-')[0]}".lower()
    facts = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            facts[key] = value.strip('"')
    return f"{facts.get('ID', 'unknown')}-{facts.get('VERSION_ID', 'unknown')}"


def _libstdcxx_max_glibcxx() -> str:
    name = ctypes_util.find_library("stdc++")
    if not name:
        return "absent"
    candidates = [
        Path(prefix) / name
        for prefix in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/lib")
    ]
    for candidate in candidates:
        if candidate.is_file():
            raw = candidate.read_bytes()
            versions = sorted(
                {
                    tuple(int(part) if part else 0 for part in match.groups())
                    for match in re.finditer(rb"GLIBCXX_(\d+)\.(\d+)(?:\.(\d+))?", raw)
                }
            )
            if versions:
                return "GLIBCXX_" + ".".join(str(part) for part in versions[-1])
    return "unlocated"


def _cxx_compiler() -> str:
    for compiler in ("c++", "g++", "clang++"):
        try:
            output = subprocess.run(
                [compiler, "--version"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if output.returncode == 0 and output.stdout:
            return output.stdout.splitlines()[0].strip()
    return "absent"


def _torch_axes() -> dict[str, str]:
    try:
        import torch
    except ImportError:
        return {
            "torch_version": "absent",
            "torch_git": "absent",
            "torch_cxx11_abi": "absent",
            "torch_config_digest": "absent",
        }
    return {
        "torch_version": str(torch.__version__),
        "torch_git": str(getattr(torch.version, "git_version", "unknown")),
        "torch_cxx11_abi": str(int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", -1))),
        "torch_config_digest": _digest(torch.__config__.show()),
    }


def _triton_axis() -> str:
    try:
        import triton  # type: ignore[import-not-found]
    except ImportError:
        return "absent"
    return str(getattr(triton, "__version__", "unknown"))


def _host_isa_axes() -> dict[str, str]:
    # The production requirement derivation, read for measurement only. This
    # is a deliberate private import inside the harness (never the wheel).
    from torchcg.host_isa import _host_requirement

    facts = _host_requirement().facts()
    return {
        "host_isa_level": facts["host_isa_level"],
        "host_isa_features": facts["host_isa_features"],
    }


def record_axes() -> dict[str, str]:
    """Record every matrix axis for the current host, one value per axis."""

    axes: dict[str, str] = {
        "machine": platform.machine().lower(),
        "os_release": _os_release(),
        "glibc": "-".join(platform.libc_ver()) or "unknown",
        "libstdcxx_max_glibcxx": _libstdcxx_max_glibcxx(),
        "cxx_compiler": _cxx_compiler(),
        "python_abi": (
            f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"
            f"-{sysconfig.get_config_var('SOABI') or 'unknown'}"
        ),
        **_torch_axes(),
        "triton": _triton_axis(),
        **_host_isa_axes(),
    }
    assert tuple(sorted(axes)) == tuple(sorted(AXIS_NAMES)), "axis vocabulary drifted"
    return axes


def worker_style_toolchain(axes: dict[str, str]) -> dict[str, str]:
    """Fold host axes into the worker-shaped toolchain content block.

    Same member names the worker records (settings/libs/torch/triton); each
    value is a digest over the underlying axes so the folded 16-hex toolchain
    identity moves exactly when a member's facts move.
    """

    return {
        "settings_declaration": _digest("host-fingerprint-probe-v1"),
        "loaded_libs": _digest(
            json.dumps(
                {name: axes[name] for name in ("glibc", "libstdcxx_max_glibcxx", "cxx_compiler")},
                sort_keys=True,
            )
        ),
        "torch": _digest(
            json.dumps(
                {
                    name: axes[name]
                    for name in (
                        "torch_version",
                        "torch_git",
                        "torch_cxx11_abi",
                        "torch_config_digest",
                    )
                },
                sort_keys=True,
            )
        ),
        "triton": _digest(axes["triton"]),
    }


def validate_row(row: dict[str, Any]) -> None:
    """Refuse a malformed matrix row before it can pollute a report."""

    required = {"schema", "host_axes", "build_axes", "diff_axes", "load_ok", "exec_ok", "output_ok"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"matrix row is missing {sorted(missing)}")
    if row["schema"] != AXES_SCHEMA_VERSION:
        raise ValueError(f"matrix row schema {row['schema']!r} is not {AXES_SCHEMA_VERSION}")
    for side in ("host_axes", "build_axes"):
        axes = row[side]
        if not isinstance(axes, dict) or set(axes) != set(AXIS_NAMES):
            raise ValueError(f"{side} must record exactly the {len(AXIS_NAMES)} matrix axes")
    expected_diff = sorted(
        name for name in AXIS_NAMES if row["host_axes"][name] != row["build_axes"][name]
    )
    if sorted(row["diff_axes"]) != expected_diff:
        raise ValueError("diff_axes must restate exactly the axes whose values differ")
    for flag in ("load_ok", "exec_ok", "output_ok"):
        if not isinstance(row[flag], bool):
            raise ValueError(f"{flag} must be a boolean")
    if "inconclusive" in row:
        if not isinstance(row["inconclusive"], bool):
            raise ValueError("inconclusive must be a boolean")
        if row["inconclusive"] and row["load_ok"]:
            raise ValueError("an inconclusive row cannot also report a successful load")
