"""Execution lanes: the author-facing compile declaration (tcg#41).

A lane is the WHOLE author surface for compilation under the 2026-08-18
ship-code-as-is ruling: the author's serving code runs as-is, and a lane names
(1) the modules on the author's own objects that are compile targets, spelled
as attribute paths, and (2) the tensor-layout contract that lane expects
checkpoints in -- the same contract name tensorfs negotiates checkpoints
toward. Architecture forks (inpaint channels, refiner, fp8 vs bf16) are simply
more lanes. No lanes means eager forever, stated rather than implied.

The SPELLING here is a prototype: semantics are ruled, the exact author-typed
syntax is pgw#1367 open question 2 and stays unfrozen until Paul reviews it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_PATH_SEPARATOR = "."


class LaneError(ValueError):
    """An execution-lane declaration is incomplete or noncanonical."""


def _require_canonical(field: str, value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LaneError(f"lane {field} must be a non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionLane:
    """One compile lane: target module paths plus the checkpoint contract.

    ``targets`` are attribute paths on the author's own objects, resolved
    against the root namespace the author hands to discovery -- for example
    ``("pipe.unet", "pipe.vae.decoder")`` against ``{"pipe": pipe}``. The path
    names the module the author means, not a family the library knows.
    """

    name: str
    targets: tuple[str, ...]
    contract: str

    def __post_init__(self) -> None:
        _require_canonical("name", self.name)
        _require_canonical("contract", self.contract)
        if not isinstance(self.targets, tuple) or not self.targets:
            raise LaneError(
                f"lane {self.name!r} must declare at least one compile-target path"
            )
        for path in self.targets:
            _require_canonical("target path", path)
            segments = path.split(_PATH_SEPARATOR)
            if len(segments) < 2 or any(not seg.isidentifier() for seg in segments):
                raise LaneError(
                    f"lane {self.name!r} target {path!r} must be a dotted attribute "
                    f"path of identifiers rooted in the author namespace, "
                    f"e.g. 'pipe.unet'"
                )
        if len(set(self.targets)) != len(self.targets):
            raise LaneError(f"lane {self.name!r} declares a duplicate target path")


def resolve_target(roots: Mapping[str, object], path: str) -> object:
    """Resolve one target path against the author's root namespace.

    The first segment names a root object; the rest are attribute accesses on
    the author's own graph of objects. A miss is refused with the exact
    segment that failed, because a silently skipped target would turn a typo
    into an eager-forever lane.
    """

    segments = _require_canonical("target path", path).split(_PATH_SEPARATOR)
    root_name = segments[0]
    if root_name not in roots:
        raise LaneError(
            f"target {path!r} names root {root_name!r}, which is not in the "
            f"author namespace {sorted(roots)!r}"
        )
    value: object = roots[root_name]
    for depth, segment in enumerate(segments[1:], start=1):
        try:
            value = getattr(value, segment)
        except AttributeError as exc:
            raise LaneError(
                f"target {path!r}: {segment!r} does not exist on the object at "
                f"{'.'.join(segments[:depth])!r}"
            ) from exc
    return value


__all__ = ["ExecutionLane", "LaneError", "resolve_target"]
