from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import torch_compiled_graphs
import torch_compiled_graphs.compiler as compiler_module
from torch_compiled_graphs import CompileError

torch: Any = pytest.importorskip("torch")


class Sine(torch.nn.Module):  # type: ignore[misc]
    def forward(self, value: Any) -> Any:
        return value.sin()


def _program(*, dynamic: bool = False) -> object:
    inputs = (torch.ones(2, 3),)
    if not dynamic:
        return torch.export.export(Sine(), inputs)
    batch = torch.export.Dim("batch", min=1, max=8)
    return torch.export.export(Sine(), inputs, dynamic_shapes={"value": {0: batch}})


def test_compile_uses_the_one_fixed_v1_policy(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(compiler_module, "_aot_compile", compiler)
    files = compiler_module._compile_exported_program(_program())
    assert files == ("wrapper.cpp", "kernel.so")
    assert seen["options"] == {
        "compile_threads": 1,
        "aot_inductor.package_constants_in_so": False,
        "aot_inductor.use_runtime_constant_folding": True,
        "aot_inductor.package": True,
    }


@pytest.mark.parametrize("dynamic", [False, True])
def test_compile_context_is_derived_from_real_graph_metadata(dynamic: bool) -> None:
    import torch._guards as guards

    program: Any = _program(dynamic=dynamic)
    modes: dict[int, Any] = {
        id(candidate): candidate
        for node in program.graph_module.graph.nodes
        if (candidate := getattr(node.meta.get("val"), "fake_mode", None)) is not None
    }
    assert len(modes) == 1
    mode: Any = next(iter(modes.values()))
    original_environment = mode.shape_env
    assert original_environment is not None

    with compiler_module._compiling_under_export_context(program):
        context = guards.TracingContext.try_get()
        assert context is not None
        assert context.fake_mode is not None
        assert id(context.fake_mode) == id(mode)
        assert mode.shape_env is original_environment

    assert mode.shape_env is original_environment


def test_compile_surface_has_no_output_changing_callbacks() -> None:
    assert set(torch_compiled_graphs.__all__) == {
        "AdmissionError",
        "ArtifactError",
        "CompileError",
        "CompiledGraphKey",
        "DeclarationError",
        "Engine",
        "EnsureOutcome",
        "EnsureResult",
        "GraphDeclaration",
        "GraphSpec",
        "IdentityError",
        "QuarantinedArtifact",
        "RuntimeCompatibility",
        "StorageError",
        "StoredGraph",
        "StoreOutcome",
        "StoreResult",
        "is_compiled_graph_key",
    }
    assert tuple(inspect.signature(compiler_module._compile_exported_program).parameters) == (
        "program",
    )
    assert tuple(inspect.signature(compiler_module._package_compiled_files).parameters) == (
        "entry",
        "files",
        "output",
    )


def test_compile_refuses_missing_or_conflicting_graph_context() -> None:
    class Missing:
        graph_module = type("GraphModule", (), {"graph": type("Graph", (), {"nodes": ()})()})()

    with pytest.raises(CompileError, match="exactly one FakeTensorMode, found 0"):
        compiler_module._export_context(Missing())

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    first = FakeTensorMode(shape_env=ShapeEnv()).from_tensor(torch.ones(1))
    second = FakeTensorMode(shape_env=ShapeEnv()).from_tensor(torch.ones(1))
    node = SimpleNamespace(meta={"val": {"nested": (first, second)}})
    conflicting = SimpleNamespace(
        graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=(node,)))
    )
    with pytest.raises(CompileError, match="exactly one FakeTensorMode, found 2"):
        compiler_module._export_context(conflicting)


def test_compile_refuses_non_file_list_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compiler_module, "_aot_compile", lambda *args, **kwargs: ())
    with pytest.raises(CompileError, match="non-empty loose-file list"):
        compiler_module._compile_exported_program(_program())


def test_packager_receives_exactly_one_named_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def packager(output: str, files: Mapping[str, Sequence[object]]) -> object:
        seen.update(output=output, files=files)
        return output

    monkeypatch.setattr(compiler_module, "_package_aoti", packager)
    output = tmp_path / "model.pt2"
    assert (
        compiler_module._package_compiled_files(
            "denoiser/h=64", ["a.so"], output
        )
        == output
    )
    assert seen["files"] == {"denoiser/h=64": ["a.so"]}
