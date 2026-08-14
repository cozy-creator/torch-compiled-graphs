from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

_REQUIRED_OPTIONS: dict[str, object] = {
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


def compiler_options(options: Mapping[str, object] | None = None) -> dict[str, object]:
    """Resolve caller tuning while enforcing the v1 correctness settings."""

    resolved = dict(options or {})
    resolved.setdefault("compile_threads", 4)
    resolved.update(_REQUIRED_OPTIONS)
    return resolved


def compile_exported_program(
    program: object,
    *,
    options: Mapping[str, object] | None = None,
    compiler: Compiler | None = None,
    before_compile: Callable[[], None] | None = None,
    context: AbstractContextManager[None] | None = None,
) -> tuple[object, ...]:
    """Compile one ExportedProgram to the loose files used by `package_aoti`.

    `before_compile` and `context` are narrow seams for worker-specific wrapper
    preparation and fake/meta tensor handling. They do not alter identity or
    packaging policy.
    """

    if before_compile is not None:
        before_compile()
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
        with context if context is not None else nullcontext():
            result = compiler(
                exported_module(check_guards=False),
                tuple(args),
                dict(kwargs or {}),
                options=compiler_options(options),
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
