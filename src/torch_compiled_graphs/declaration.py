"""Canonical declarations for one exported AOTInductor graph class."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .identity import CompiledGraphKey, _facts_digest, from_axes, toolchain_axis_digest

CANONICAL_GRAPH_FORMAT = 1


class DeclarationError(ValueError):
    """A graph or runtime cannot state a complete v1 declaration."""


class _Symbols:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}

    def name(self, symbol: Any) -> str:
        raw = str(symbol)
        known = self._ids.get(raw)
        if known is None:
            known = f"S{len(self._ids)}"
            self._ids[raw] = known
        return known

    def known(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._ids.items(), key=lambda item: item[1]))


def _render_symbol(value: Any, symbols: _Symbols) -> str:
    expression = getattr(getattr(value, "node", None), "expr", None)
    free = getattr(expression, "free_symbols", None) or ()
    if expression is None or not free:
        return str(expression if expression is not None else value)
    rendered = str(expression)
    for symbol in sorted(free, key=lambda item: len(str(item)), reverse=True):
        rendered = rendered.replace(str(symbol), symbols.name(symbol))
    return rendered


def _render_scalar(value: Any) -> str:
    import torch

    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        rendered = str(value)
        return rendered.split(":", 1)[0] if isinstance(value, torch.device) else rendered
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (int, str, bytes)):
        return repr(value)
    if isinstance(value, slice):
        return f"slice({value.start!r},{value.stop!r},{value.step!r})"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _render_value(value: Any, symbols: _Symbols) -> str:
    import torch

    if value is None:
        return "-"
    if isinstance(value, torch.Tensor):
        shape = ",".join(_render_symbol(dimension, symbols) for dimension in value.shape)
        return f"t({value.dtype}|[{shape}]|{value.device.type})"
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return f"sym({_render_symbol(value, symbols)})"
    if isinstance(value, (list, tuple)):
        body = ",".join(_render_value(item, symbols) for item in value)
        return f"[{body}]" if isinstance(value, list) else f"({body})"
    if isinstance(value, dict):
        body = ",".join(
            f"{key!r}:{_render_value(item, symbols)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + body + "}"
    return _render_scalar(value)


def _render_argument(value: Any, names: dict[Any, str], symbols: _Symbols) -> str:
    import torch

    if isinstance(value, torch.fx.Node):
        return names.get(value, "%?")
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_render_argument(item, names, symbols) for item in value) + ")"
    if isinstance(value, dict):
        body = ",".join(
            f"{key!r}:{_render_argument(item, names, symbols)}"
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return "{" + body + "}"
    if isinstance(value, torch.Tensor):
        shape = ",".join(_render_symbol(dimension, symbols) for dimension in value.shape)
        return f"t({value.dtype}|[{shape}])"
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return f"sym({_render_symbol(value, symbols)})"
    return _render_scalar(value)


def _target(node: Any) -> str:
    if node.op == "placeholder":
        return ""
    if node.op in ("call_method", "get_attr", "output"):
        return str(node.target)
    module = getattr(node.target, "__module__", "")
    name = getattr(node.target, "__qualname__", "")
    return f"{module}.{name}" if module and name else str(node.target)


def _node_value(node: Any) -> tuple[bool, Any]:
    for key in ("val", "example_value", "tensor_meta"):
        if key in node.meta:
            return True, node.meta[key]
    return False, None


def _graph_lines(graph: Any, symbols: _Symbols) -> list[str]:
    names = {node: f"%{index}" for index, node in enumerate(graph.nodes)}
    lines: list[str] = []
    for index, node in enumerate(graph.nodes):
        args = ",".join(_render_argument(item, names, symbols) for item in node.args)
        kwargs = ",".join(
            f"{key}={_render_argument(item, names, symbols)}"
            for key, item in sorted(node.kwargs.items())
        )
        present, raw_value = _node_value(node)
        value = _render_value(raw_value, symbols) if present else "-"
        placeholder = f"arg{index}" if node.op == "placeholder" else ""
        lines.append(
            f"node {index} {node.op} {_target(node)} {placeholder} "
            f"args=({args}) kwargs=({kwargs}) val={value}"
        )
    return lines


def _bound(value: Any) -> str:
    if value is None:
        return "oo"
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return str(value).replace("+", "")


def _range_lines(ranges: Any, symbols: _Symbols) -> list[str]:
    if not ranges:
        return []
    known = dict(symbols.known())
    lines = []
    for symbol, value_range in ranges.items():
        canonical = known.get(str(symbol))
        if canonical is not None:
            lines.append(
                f"sym {canonical} range=[{_bound(getattr(value_range, 'lower', None))},"
                f"{_bound(getattr(value_range, 'upper', None))}]"
            )
    return sorted(lines)


def _signature_lines(program: Any) -> list[str]:
    signature = getattr(program, "graph_signature", None)
    lines: list[str] = []
    for direction, specs in (
        ("in", getattr(signature, "input_specs", None) or ()),
        ("out", getattr(signature, "output_specs", None) or ()),
    ):
        for index, spec in enumerate(specs):
            kind_object = getattr(spec, "kind", None)
            kind = getattr(kind_object, "name", str(kind_object or ""))
            target = getattr(spec, "target", None) or "-"
            lines.append(f"sig {direction} {index} kind={kind} target={target}")
    call_spec = getattr(program, "call_spec", None)
    for direction, spec in (
        ("in", getattr(call_spec, "in_spec", None)),
        ("out", getattr(call_spec, "out_spec", None)),
    ):
        if spec is not None:
            try:
                import torch.utils._pytree as pytree

                rendered = str(pytree.treespec_dumps(spec))
            except Exception:
                rendered = repr(spec)
            lines.append(f"spec {direction} {rendered}".replace("\n", " "))
    return lines


def _canonical_graph(program: object) -> tuple[str, ...]:
    """Return the sole v1 canonical form for a ``torch.export.ExportedProgram``."""

    import torch

    if not isinstance(program, torch.export.ExportedProgram):
        raise DeclarationError("v1 declarations require a torch.export.ExportedProgram")
    symbols = _Symbols()
    lines = [f"v={CANONICAL_GRAPH_FORMAT} ir=export"]
    lines.extend(_graph_lines(program.graph_module.graph, symbols))
    lines.extend(_range_lines(program.range_constraints, symbols))
    lines.extend(_signature_lines(program))
    return tuple(lines)


def _graph_digest(program: object) -> str:
    payload = "\n".join(_canonical_graph(program)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _literal_names(program: object) -> tuple[str, ...]:
    signature = getattr(program, "graph_signature", None)
    state = {
        str(name)
        for field in ("parameters", "buffers")
        for name in (getattr(signature, field, ()) or ())
    }
    names = {str(name) for name in getattr(signature, "lifted_tensor_constants", ()) or ()}
    names.update(str(name) for name in (getattr(program, "constants", {}) or {}))
    return tuple(sorted(names - state))


def _literal_digest(program: object) -> str:
    names = _literal_names(program)
    if not names:
        return ""
    values = getattr(program, "constants", {}) or {}
    digest = hashlib.sha256()
    for name in names:
        value = values.get(name)
        if value is None:
            raise DeclarationError(f"literal constant {name!r} carries no value")
        digest.update(name.encode("utf-8") + b"\0")
        try:
            import torch

            tensor = value.detach().cpu().contiguous().reshape(-1)
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        except Exception as exc:
            raise DeclarationError(
                f"literal constant {name!r} could not be digested: {type(exc).__name__}: {exc}"
            ) from exc
    return digest.hexdigest()[:32]


def _placement(program: object) -> tuple[str, ...]:
    import torch

    devices: set[str] = set()

    def note(value: Any) -> None:
        if isinstance(value, torch.device):
            devices.add(f"{value.type}:{value.index}" if value.index is not None else value.type)
        elif isinstance(value, torch.Tensor):
            note(value.device)
        elif isinstance(value, (list, tuple)):
            for item in value:
                note(item)
        elif isinstance(value, dict):
            for item in value.values():
                note(item)

    for node in program.graph_module.graph.nodes:  # type: ignore[attr-defined]
        present, value = _node_value(node)
        if present:
            note(value)
        note(node.args)
        note(node.kwargs)
    return tuple(sorted(devices))


@dataclass(frozen=True, slots=True)
class GraphDeclaration:
    """Immutable graph facts shared by lookup, mint, and admission."""

    entry: str
    target: str
    graph: str
    literal_values: str = ""
    placement: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        entry = self.entry.strip()
        target = self.target.strip()
        if not entry or not target:
            raise DeclarationError("entry and target must be non-empty")
        if "\\" in entry or any(part in ("", ".", "..") for part in entry.split("/")):
            raise DeclarationError(f"unsafe entry name {entry!r}")
        if len(self.graph) != 16 or any(
            character not in "0123456789abcdef" for character in self.graph
        ):
            raise DeclarationError("graph digest must be 16 lowercase hexadecimal characters")
        if self.literal_values and (
            len(self.literal_values) != 32
            or any(character not in "0123456789abcdef" for character in self.literal_values)
        ):
            raise DeclarationError(
                "literal-values digest must be 32 lowercase hexadecimal characters"
            )
        object.__setattr__(self, "entry", entry)
        object.__setattr__(self, "target", target)
        canonical_placement = tuple(sorted(set(self.placement)))
        if len(canonical_placement) == 1:
            raise DeclarationError("single-device placement must be omitted")
        object.__setattr__(self, "placement", canonical_placement)

    def facts(self) -> dict[str, object]:
        facts: dict[str, object] = {
            "v": 1,
            "entry": self.entry,
            "target": self.target,
            "graph": self.graph,
        }
        if self.literal_values:
            facts["literal_values"] = self.literal_values
        if len(self.placement) > 1:
            facts["placement"] = list(self.placement)
        return facts

    @property
    def class_hash(self) -> str:
        return _facts_digest(self.facts())


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """A named exported program ready for local resolution or minting."""

    entry: str
    target: str
    program: object

    def declare(self) -> GraphDeclaration:
        graph = _graph_digest(self.program)
        literals = _literal_digest(self.program)
        placement = _placement(self.program)
        return GraphDeclaration(
            self.entry,
            self.target,
            graph,
            literal_values=literals,
            placement=placement if len(placement) > 1 else (),
        )


@dataclass(frozen=True, slots=True, init=False)
class RuntimeCompatibility:
    """Compiler/runtime facts that determine whether an artifact can execute."""

    sm: str
    _toolchain: tuple[tuple[str, str], ...]

    def __init__(self, sm: str, toolchain: Mapping[str, str]) -> None:
        architecture = str(sm).strip()
        if not architecture:
            raise DeclarationError("runtime requires a GPU compute capability")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in toolchain.items()
        ):
            raise DeclarationError("toolchain names and values must be strings")
        cleaned = tuple(sorted(toolchain.items()))
        if not cleaned or any(not key.strip() or not value.strip() for key, value in cleaned):
            raise DeclarationError("runtime requires non-empty toolchain facts")
        object.__setattr__(self, "sm", architecture)
        object.__setattr__(self, "_toolchain", cleaned)

    @property
    def toolchain(self) -> dict[str, str]:
        return dict(self._toolchain)

    def key(self, declaration: GraphDeclaration) -> CompiledGraphKey:
        return from_axes(
            {
                "graph": declaration.class_hash,
                "sm": self.sm,
                "toolchain": toolchain_axis_digest(self.toolchain),
            }
        )

    def canonical(self) -> bytes:
        return json.dumps(
            {"sm": self.sm, "toolchain": self.toolchain},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
