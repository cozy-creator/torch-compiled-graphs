from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

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
