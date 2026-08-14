from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

_COMPILER_OPTIONS: dict[str, object] = {
    "compile_threads": 1,
    "aot_inductor.package_constants_in_so": False,
    "aot_inductor.use_runtime_constant_folding": True,
    "aot_inductor.package": True,
}


class CompileError(RuntimeError):
    """AOTInductor did not produce a packageable code-only graph."""


class Compiler(Protocol):
    def __call__(
        self,
        graph_module: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        options: Mapping[str, object],
    ) -> object: ...


class Packager(Protocol):
    def __call__(self, output: str, files: Mapping[str, Sequence[object]]) -> object: ...


def _compiler_options() -> dict[str, object]:
    """Return the sole compile policy accepted by pre-launch v1."""

    return dict(_COMPILER_OPTIONS)


def _fake_mode(program: object) -> object | None:
    for holder in ("state_dict", "constants"):
        values = getattr(program, holder, None)
        if not isinstance(values, Mapping):
            continue
        for tensor in values.values():
            mode = getattr(tensor, "fake_mode", None)
            if mode is not None:
                return cast(object, mode)
    return None


def _shape_env(program: object) -> object | None:
    graph_module = getattr(program, "graph_module", None)
    graph = getattr(graph_module, "graph", None)
    for node in getattr(graph, "nodes", ()):
        value = getattr(node, "meta", {}).get("val")
        for dimension in getattr(value, "shape", ()) or ():
            environment = getattr(getattr(dimension, "node", None), "shape_env", None)
            if environment is not None:
                return cast(object, environment)
    return None


@contextmanager
def _compiling_under_export_context(program: object) -> Iterator[None]:
    """Restore the FakeTensor tracing context carried by one exported program."""

    mode = _fake_mode(program)
    if mode is None:
        yield
        return
    try:
        guards = import_module("torch._guards")
        tracing_context = vars(guards)["TracingContext"]
        tracing = vars(guards)["tracing"]
    except (ImportError, KeyError) as exc:
        raise CompileError("PyTorch FakeTensor tracing support is unavailable") from exc

    environment = _shape_env(program)
    previous = getattr(mode, "shape_env", None)
    if environment is not None:
        cast(Any, mode).shape_env = environment
    try:
        with cast(Any, tracing)(cast(Any, tracing_context)(mode)):
            yield
    finally:
        if environment is not None:
            cast(Any, mode).shape_env = previous


def compile_exported_program(
    program: object,
    *,
    compiler: Compiler | None = None,
) -> tuple[object, ...]:
    """Compile one ExportedProgram to the loose files used by `package_aoti`.

    FakeTensor and ShapeEnv state is derived only from ``program``. Compile
    settings are fixed because every setting that can affect generated code
    must have exactly one v1 identity.
    """
    if compiler is None:
        try:
            module = import_module("torch._inductor")
        except ImportError as exc:  # pragma: no cover - exercised with torch extra
            raise CompileError("PyTorch with AOTInductor is required to compile") from exc
        compiler = cast(Compiler, vars(module)["aot_compile"])

    exported_module = getattr(program, "module", None)
    if not callable(exported_module):
        raise CompileError("program has no callable module(check_guards=False)")
    try:
        args, kwargs = program.example_inputs  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise CompileError("program has no (args, kwargs) example_inputs") from exc

    try:
        with _compiling_under_export_context(program):
            result = compiler(
                exported_module(check_guards=False),
                tuple(args),
                dict(kwargs or {}),
                options=_compiler_options(),
            )
    except Exception as exc:
        raise CompileError(f"AOTInductor compile failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(result, list) or not result:
        raise CompileError("AOTInductor did not return a non-empty loose-file list")
    return tuple(result)


def package_compiled_files(
    entry: str,
    files: Sequence[object],
    output: str | Path,
    *,
    packager: Packager | None = None,
) -> Path:
    """Package one named graph class into a `.pt2` file."""

    name = str(entry).strip()
    if not name:
        raise CompileError("entry name must not be empty")
    if not files:
        raise CompileError("cannot package an entry with no compiled files")
    if packager is None:
        try:
            module = import_module("torch._inductor.package")
        except ImportError as exc:  # pragma: no cover - exercised with torch extra
            raise CompileError("PyTorch with AOTInductor is required to package") from exc
        packager = cast(Packager, vars(module)["package_aoti"])
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = packager(str(target), {name: list(files)})
    except Exception as exc:
        raise CompileError(f"AOTInductor packaging failed: {type(exc).__name__}: {exc}") from exc
    return Path(str(result))
