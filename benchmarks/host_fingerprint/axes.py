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

# 1 -> 2 (tcg#26): `torch_config_digest` changed derivation, so a v1 build_axes
# compared against a v2 host_axes would report a diff on an axis that did not
# move. The bump makes that mix a REFUSAL in `validate_row` instead of a false
# reading; the v1 rows under `results/` stay as the record of what was measured.
AXES_SCHEMA_VERSION = 2

#: Every schema version whose rows are still READABLE. A recorded row stays
#: valid evidence after a bump; only mixing derivations across a bump is unsafe.
KNOWN_SCHEMA_VERSIONS: tuple[int, ...] = (1, 2)

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


# tcg#26 — THE RULE: this axis states BUILD identity, never HOST identity.
#
# `torch.__config__.show()` is not a build manifest. ATen's `show_config()`
# interleaves compile-time facts with two kinds of line that describe the
# machine the process happens to be running on:
#
#   * `CPU capability usage: <ISA>` — the ISA the running process dispatched
#     to. The measured symptom: one 2.13.0+cpu wheel, git `cf30153c`, digests
#     `5f31e143bbf0b632` on an AVX2 host and `76efcb828dca63fb` on AVX512.
#   * the accelerator hook blocks — `show_config()` appends
#     `getCUDAHooks().showConfig()` (and the XPU/MAIA equivalents) behind
#     `hasCUDA()`, which is a live DRIVER probe, not a build flag. A `+cu130`
#     wheel on a host with a too-old driver emits no CUDA block at all, while
#     the same wheel on a working GPU host emits CUDA Runtime / NVCC
#     architecture flags / CuDNN / Magma. That is a far wider swing than the
#     CPU-capability line and it moves on driver state alone.
#
# ALLOWLIST, deliberately, rather than stripping the known-bad lines: an
# unrecognised line is DROPPED, so a future torch release that adds another
# runtime-probed line cannot silently re-fragment the axis. The build half is
# not weakened by that default — `Build settings:` carries COMMIT_SHA,
# TORCH_VERSION, CUDA_VERSION, CUDNN_VERSION, CXX_COMPILER, CXX_FLAGS and every
# USE_* toggle, and `torch_git` pins the same commit on an axis of its own.
#
# KEPT (each `#if`-guarded at BUILD time in aten/src/ATen/Version.cpp):
#   GCC / clang / MSVC, C++ Version, oneAPI MKL, MKL-DNN, OpenMP,
#   LAPACK is enabled, NNPACK is enabled, Cross compiling on MacOSX,
#   Build settings.
# DROPPED: `CPU capability usage:`, the CUDA/XPU/MAIA hook blocks, the
#   "PyTorch built with:" header (constant, carries no fact), and anything new.
_BUILD_IDENTITY_PREFIXES = (
    "GCC ",
    "clang ",
    "MSVC ",
    "C++ Version:",
    "Intel(R) oneAPI Math Kernel Library",
    "Intel(R) MKL-DNN",
    "OpenMP ",
    "LAPACK is enabled",
    "NNPACK is enabled",
    "Cross compiling on MacOSX",
    "Build settings:",
)


def build_identity_lines(show: str) -> list[str]:
    """The build-identifying lines of a `torch.__config__.show()` string.

    Source order is preserved and is stable: the runtime blocks ATen may or may
    not emit are appended between the kept lines, never interleaved among them.
    """

    kept = []
    for raw in show.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        fact = line[2:].strip()
        if fact.startswith(_BUILD_IDENTITY_PREFIXES):
            kept.append(fact)
    return kept


def torch_config_digest(show: str) -> str:
    """Digest the BUILD identity carried by a `show()` string, so the same
    wheel digests identically on every host that can run it."""

    return _digest("\n".join(build_identity_lines(show)))


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
        "torch_config_digest": torch_config_digest(torch.__config__.show()),
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

    FIDELITY, stated so the matrix is not read as production (tcg#26): the
    member NAMES match the worker's, the INPUTS do not. The real worker's
    `torch` member is the dist-info RECORD digest of the installed wheel
    (`gen_worker.compile_cache.toolchain_digest`) — whole-package content
    identity, host-independent by construction. This harness substitutes the
    version/git/abi/config axes it can probe without an install manifest. Read
    a diff here as "these hosts differ", never as "production would re-key".
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
    # Any KNOWN version, not just the current one (tcg#26): a row is internally
    # consistent whatever derivation recorded it, because `diff_axes` is
    # computed within the row from its own two sides. Pinning this to the
    # current version would make every schema bump retroactively unread the
    # archived campaign under `results/`, which is the evidence, not a cache.
    # The hazard a bump actually guards — a build bundle recorded under one
    # derivation compared against live host axes recorded under another — is
    # refused where it arises, in `probe_run.py`'s bundle check.
    if row["schema"] not in KNOWN_SCHEMA_VERSIONS:
        raise ValueError(
            f"matrix row schema {row['schema']!r} is not one of {list(KNOWN_SCHEMA_VERSIONS)}"
        )
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
