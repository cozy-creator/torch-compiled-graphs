from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

KEY_SCHEME = "cg-key-v1"
MAX_KEY_LENGTH = 96
_DIGEST_HEX = 64
_REQUIRED_AXES = ("graph", "sm", "toolchain")
_KEY_RE = re.compile(rf"{KEY_SCHEME}-[0-9a-f]{{{_DIGEST_HEX}}}\Z")


class IdentityError(ValueError):
    """A compiled graph cannot state its complete, canonical identity."""


@dataclass(frozen=True, slots=True)
class CompiledGraphKey:
    """The three identity axes and their deterministic v1 address."""

    axes: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, str]:
        return dict(self.axes)

    def canonical(self) -> bytes:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    @property
    def value(self) -> str:
        digest = hashlib.sha256(self.canonical()).hexdigest()[:_DIGEST_HEX]
        return f"{KEY_SCHEME}-{digest}"

    def __str__(self) -> str:
        return self.value


def is_compiled_graph_key(value: object) -> bool:
    """Return whether a boundary value has the versioned key shape.

    Pre-launch v1 has one accepted key representation and no compatibility
    reader for abandoned or future schemes.
    """

    text = str(value or "")
    return len(text) <= MAX_KEY_LENGTH and _KEY_RE.fullmatch(text) is not None


def _refuse_key_as_fact(name: str, value: str) -> None:
    if is_compiled_graph_key(value):
        raise IdentityError(f"{name} is a compiled-graph key, not an identity fact")


def from_axes(axes: Mapping[str, str]) -> CompiledGraphKey:
    unknown = sorted(set(axes) - set(_REQUIRED_AXES))
    if unknown:
        raise IdentityError(
            f"unknown identity axes {unknown!r}; v1 is exactly {list(_REQUIRED_AXES)!r}"
        )
    clean: dict[str, str] = {}
    for name in _REQUIRED_AXES:
        raw = axes.get(name)
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise IdentityError(f"compiled graph identity requires canonical string {name!r}")
        value = raw
        _refuse_key_as_fact(name, value)
        clean[name] = value
    return CompiledGraphKey(tuple(sorted(clean.items())))


def _facts_digest(facts: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(facts), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def toolchain_axis_digest(block: Mapping[str, Any]) -> str:
    """Digest explicit compiler components/settings; callers supply no model facts."""

    return _facts_digest({str(name): str(value) for name, value in block.items()})


def from_artifact_metadata(metadata: Mapping[str, Any]) -> CompiledGraphKey:
    if metadata.get("kind") != "aot-inductor":
        raise IdentityError("only an aot-inductor artifact has compiled-graph identity")
    sm = metadata.get("sm")
    if not isinstance(sm, str) or not sm.strip():
        raise IdentityError("artifact records no GPU compute capability")
    entry = metadata.get("entry")
    if not isinstance(entry, Mapping):
        raise IdentityError("artifact records no entry object")
    graph = entry.get("class_hash")
    if not isinstance(graph, str) or not graph.strip():
        raise IdentityError("artifact entry records no class_hash")
    toolchain = metadata.get("toolchain")
    if (
        not isinstance(toolchain, Mapping)
        or not toolchain
        or any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not isinstance(value, str)
            or not value
            or value != value.strip()
            for name, value in toolchain.items()
        )
    ):
        raise IdentityError("artifact records no toolchain object")
    return from_axes({"graph": graph, "sm": sm, "toolchain": toolchain_axis_digest(toolchain)})
