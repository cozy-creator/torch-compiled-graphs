"""Canonical declarations for one exported AOTInductor graph class."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .host_isa import HostISAError, _impose_host_policy
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
        return _render_tensor(value, symbols, include_device=True)
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
        return _render_tensor(value, symbols, include_device=False)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return f"sym({_render_symbol(value, symbols)})"
    return _render_scalar(value)


def _render_tensor(value: Any, symbols: _Symbols, *, include_device: bool) -> str:
    shape = ",".join(_render_symbol(dimension, symbols) for dimension in value.shape)
    layout = str(value.layout)
    stride = "-"
    if layout == "torch.strided":
        stride = ",".join(_render_symbol(dimension, symbols) for dimension in value.stride())
    device = f"|device={value.device.type}" if include_device else ""
    return f"t({value.dtype}|shape=[{shape}]|stride=[{stride}]|layout={layout}{device})"


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
    return hashlib.sha256(payload).hexdigest()


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
        try:
            import torch

            tensor = value.detach().cpu().contiguous().reshape(-1)
            _update_literal_digest(
                digest,
                name=name,
                dtype=str(value.dtype).removeprefix("torch."),
                shape=tuple(value.shape),
                chunks=(bytes(tensor.view(torch.uint8)),),
            )
        except Exception as exc:
            raise DeclarationError(
                f"literal constant {name!r} could not be digested: {type(exc).__name__}: {exc}"
            ) from exc
    return digest.hexdigest()


def _update_literal_digest(
    digest: Any,
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    chunks: Iterable[bytes],
) -> None:
    """Update a v1 literal digest from canonical facts and bounded byte chunks."""

    digest.update(name.encode("utf-8") + b"\0")
    digest.update(dtype.encode("ascii") + b"\0")
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii") + b"\0")
    for chunk in chunks:
        digest.update(chunk)


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
        if len(self.graph) != 64 or any(
            character not in "0123456789abcdef" for character in self.graph
        ):
            raise DeclarationError("graph digest must be 64 lowercase hexadecimal characters")
        if self.literal_values and (
            len(self.literal_values) != 64
            or any(character not in "0123456789abcdef" for character in self.literal_values)
        ):
            raise DeclarationError(
                "literal-values digest must be 64 lowercase hexadecimal characters"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _command_fingerprint(command: str) -> tuple[str, str, str]:
    resolved = shutil.which(command)
    if resolved is None:
        raise DeclarationError(f"AOTInductor C++ compiler {command!r} is unavailable")
    try:
        version = subprocess.run(
            [resolved, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        library = subprocess.run(
            [resolved, "-print-file-name=libstdc++.so.6"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeclarationError(
            f"cannot fingerprint AOTInductor compiler {resolved}: {exc}"
        ) from exc
    library_path = Path(library)
    if not library_path.is_file():
        raise DeclarationError(f"compiler did not resolve libstdc++.so.6: {library!r}")
    return (
        Path(resolved).name,
        hashlib.sha256(version.encode("utf-8")).hexdigest(),
        _sha256_file(library_path),
    )


def _cuda_driver_version() -> str:
    import ctypes

    try:
        driver = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        result = driver.cuDriverGetVersion(ctypes.byref(version))
    except (AttributeError, OSError) as exc:
        raise DeclarationError(f"CUDA target requires a readable driver version: {exc}") from exc
    if result != 0 or version.value <= 0:
        raise DeclarationError(f"CUDA driver version query failed with status {result}")
    return str(version.value)


@lru_cache(maxsize=8)
def _detected_toolchain(
    target: str,
    deployment_compatibility: str,
    host_facts: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    import torch

    try:
        import torch._inductor.config as inductor_config
    except ImportError as exc:
        raise DeclarationError("PyTorch AOTInductor is required") from exc
    try:
        triton_version = version("triton")
    except PackageNotFoundError as exc:
        raise DeclarationError("Triton is required") from exc
    from .compiler import _compiler_options

    abi = sysconfig.get_config_var("SOABI")
    multiarch = sysconfig.get_config_var("MULTIARCH")
    libc_name, libc_version = platform.libc_ver()
    cache_tag = sys.implementation.cache_tag
    torch_git = getattr(torch.version, "git_version", None)
    if not all((abi, multiarch, libc_name, libc_version, cache_tag, torch_git)):
        raise DeclarationError(
            "runtime cannot derive complete Python, platform, libc, and Torch build facts"
        )

    configured = getattr(inductor_config.cpp, "cxx", ())
    candidates = configured if isinstance(configured, (list, tuple)) else (configured,)
    compiler = next((str(item) for item in candidates if item), "g++")
    compiler_name, compiler_version, libstdcxx = _command_fingerprint(compiler)
    facts = {
        "compile_settings": json.dumps(
            _compiler_options(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ),
        "compiler": compiler_name,
        "compiler_version_sha256": compiler_version,
        "deployment_compatibility": deployment_compatibility,
        "inductor": str(torch_git),
        "libc": f"{libc_name}-{libc_version}",
        "libstdcxx_sha256": libstdcxx,
        "platform": sys.platform,
        "python_abi": str(abi),
        "python_cache_tag": str(cache_tag),
        "python_multiarch": str(multiarch),
        "torch": str(torch.__version__),
        "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode("utf-8")).hexdigest(),
        "torch_cxx11_abi": str(torch.compiled_with_cxx11_abi()),
        "triton": triton_version,
    }
    facts.update(host_facts)
    if target.startswith("sm_"):
        cuda_runtime = getattr(torch.version, "cuda", None)
        if not cuda_runtime or not torch.cuda.is_available():
            raise DeclarationError("CUDA target requires a compatible visible CUDA device")
        cuda_capability = torch.cuda.get_device_capability()
        actual = f"sm_{cuda_capability[0]}{cuda_capability[1]}"
        if target != actual:
            raise DeclarationError(f"requested CUDA target {target!r} does not match {actual!r}")
        facts["cuda_driver"] = _cuda_driver_version()
        facts["cuda_runtime"] = str(cuda_runtime)
    else:
        cpu_capability = torch.backends.cpu.get_cpu_capability()
        if target != f"cpu-{cpu_capability.lower()}":
            raise DeclarationError(
                f"CPU target {target!r} does not match capability {cpu_capability!r}"
            )
        facts["cpu_target"] = cpu_capability
    if any(not value for value in facts.values()):
        raise DeclarationError("runtime derived an empty compatibility fact")
    return tuple(sorted(facts.items()))


@dataclass(frozen=True, slots=True, init=False)
class RuntimeCompatibility:
    """Library-derived compiler/runtime facts for one current execution target."""

    sm: str
    _toolchain: tuple[tuple[str, str], ...]

    def __init__(self, target: str, *, deployment_compatibility: str) -> None:
        requested = str(target).strip().lower()
        deployment = str(deployment_compatibility).strip()
        if not deployment:
            raise DeclarationError("runtime requires an explicit deployment_compatibility axis")
        if requested == "cpu":
            import torch

            architecture = f"cpu-{torch.backends.cpu.get_cpu_capability().lower()}"
        elif re.fullmatch(r"sm_[0-9]+", requested):
            architecture = requested
        else:
            raise DeclarationError("runtime target must be 'cpu' or a concrete 'sm_NN'")
        try:
            host_facts = tuple(sorted(_impose_host_policy().items()))
        except HostISAError as exc:
            raise DeclarationError(f"cannot establish host ISA policy: {exc}") from exc
        cleaned = _detected_toolchain(architecture, deployment, host_facts)
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
