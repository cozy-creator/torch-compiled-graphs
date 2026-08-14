from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from torch_compiled_graphs import CompileError, compile_exported_program, package_compiled_files
from torch_compiled_graphs.compiler import Compiler


class Program:
    example_inputs = (("input",), {"scale": 2})

    def module(self, *, check_guards: bool) -> str:
        assert check_guards is False
        return "graph"


def test_compile_uses_the_one_fixed_v1_policy() -> None:
    seen: dict[str, object] = {}

    def compiler(
        graph: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        options: Mapping[str, object],
    ) -> object:
        seen.update(graph=graph, args=args, kwargs=kwargs, options=dict(options))
        return ["wrapper.cpp", "kernel.so"]

    files = compile_exported_program(
        Program(),
        compiler=cast(Compiler, compiler),
    )
    assert files == ("wrapper.cpp", "kernel.so")
    assert seen["options"] == {
        "compile_threads": 1,
        "aot_inductor.package_constants_in_so": False,
        "aot_inductor.use_runtime_constant_folding": True,
        "aot_inductor.package": True,
    }


def test_compile_context_is_internal_and_derived_from_the_program() -> None:
    import torch._guards as guards
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    assert "context" not in inspect.signature(compile_exported_program).parameters
    previous = ShapeEnv()
    exported = ShapeEnv()
    mode = FakeTensorMode(shape_env=previous)
    program = cast(Any, Program())
    program.state_dict = {"weight": SimpleNamespace(fake_mode=mode)}
    dimension = SimpleNamespace(node=SimpleNamespace(shape_env=exported))
    value = SimpleNamespace(shape=(dimension,))
    node = SimpleNamespace(meta={"val": value})
    program.graph_module = SimpleNamespace(graph=SimpleNamespace(nodes=(node,)))

    def compiler(*args: object, **kwargs: object) -> object:
        context = guards.TracingContext.try_get()
        assert context is not None
        assert context.fake_mode is mode
        assert mode.shape_env is exported
        return ["wrapper.cpp", "kernel.so"]

    compile_exported_program(program, compiler=cast(Compiler, compiler))
    assert mode.shape_env is previous


def test_compile_refuses_non_file_list_result() -> None:
    with pytest.raises(CompileError, match="non-empty loose-file list"):
        compile_exported_program(Program(), compiler=cast(Compiler, lambda *args, **kwargs: ()))


def test_packager_receives_exactly_one_named_graph(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def packager(output: str, files: Mapping[str, Sequence[object]]) -> object:
        seen.update(output=output, files=files)
        return output

    output = tmp_path / "model.pt2"
    assert package_compiled_files("denoiser/h=64", ["a.so"], output, packager=packager) == output
    assert seen["files"] == {"denoiser/h=64": ["a.so"]}
