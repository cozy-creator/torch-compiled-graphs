from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import torchcg
import torchcg._wrapper_split as wrapper_split
import torchcg.compiler as compiler_module
from torchcg import CompileError

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
        "compile_threads": 4,
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
    assert set(torchcg.__all__) == {
        # tcg#52/#53 -- the transform-pass mechanism and its two instances.
        # `call`/`key` on PrecomputeAndFree are DOMAIN callbacks (how to
        # evaluate the folded submodule, what a domain point is), and they run
        # in the PASS phase before any graph exists; they cannot change a
        # compiled graph's output because the graph is derived AFTER them.
        "SIDE_TABLE_FORMAT",
        "CallInputs",
        "KeptModule",
        "ModuleSelect",
        "OffDomain",
        "PrecomputeAndFree",
        "Precomputed",
        "Produced",
        "QuantPlan",
        "QuantRecipe",
        "RecipeError",
        "RecipeQuantize",
        "SideTableStore",
        "TransformError",
        "TransformMode",
        "TransformOrderError",
        "TransformPass",
        "TransformPlan",
        "TransformReport",
        "TransformSession",
        "TransformSet",
        "applied_lane_row",
        "canonical_key",
        "handle",
        "installed_passes",
        "is_resident",
        "module_bytes",
        "plan_quant",
        "register_pass",
        "registered_passes",
        "require_pass_ref",
        "require_passes",
        "resolve_pass",
        "tensor_bytes",
        "ARTIFACT_KIND",
        "ARTIFACT_METADATA_FIELDS",
        "AdmissionError",
        "AdoptError",
        "AdoptSession",
        "ArtifactError",
        "ArtifactFormatSkew",
        "ArtifactLoader",
        "CompileError",
        "COMPILED_GRAPH_FORMAT",
        "ArtifactCandidate",
        "CallIngress",
        "CallInput",
        "SpecializationReport",
        "CompiledGraphKey",
        "CompiledGraphRunner",
        "DOCUMENT_FORMAT",
        # tcg#77 -- the caller's per-axis dynamic-dim policy.
        "DimPolicy",
        "DiscoveryError",
        "DocumentError",
        "ENV_SCHEME",
        "EnvIdentity",
        "EnvironmentMismatch",
        "LaneRef",
        "FeedNormalization",
        "GRAPH_SPECIALIZATION_BLOCK",
        "GRAPH_SCHEME",
        "GraphIdentityError",
        "GraphRecord",
        "GraphSetDocument",
        "GraphStore",
        "Hole",
        "LaneError",
        "LaneGraphs",
        "LocalGraphStore",
        "PublishOutcome",
        "REQUIRED_AXES",
        "RequirementsError",
        "RequirementsManifest",
        "ConstantBindingError",
        "DeclarationError",
        "RetiredGraphInterface",
        "Engine",
        "EnsureOutcome",
        "EnsureResult",
        "GraphSpecializationCandidate",
        "GraphSpecializationDeclaration",
        "GRAPH_INTERFACE_FORMAT",
        "GraphSpecialization",
        "IdentityError",
        "IngressError",
        "IngressMiss",
        "MissReason",
        "NormalizationKind",
        "PresentedCall",
        "PresentedValue",
        "QuarantinedArtifact",
        "RealignReason",
        "RuntimeCompatibility",
        "Selection",
        "SelectionError",
        "SelectionOutcome",
        "StorageError",
        "StoreError",
        "StoredCompiledGraph",
        "StoreOutcome",
        # tcg#70: the MODULE side of an unclaimed record — a ctx.compile mark
        # that fit no graph. Exported because the boot verdict that reports
        # adoption has to name it; state nobody can read is the defect.
        "UnclaimedMark",
        "StoreResult",
        "assert_exact_env",
        "build_call_ingress",
        "compile_stack",
        "describe_call",
        "discover_lane",
        "discover_modules",
        "exported_input_name",
        "graph_hash",
        "holes",
        "is_compile_relevant",
        "is_compiled_graph_key",
        "is_graph_hash",
        "rank",
        "require_contract_ref",
        "require_targets",
        "resolve_target",
        "select",
        "CompiledGraphCall",
        "MaterializedGraph",
        "VerifiedGraph",
        "aoti_loader",
        "materialize",
        "module_device",
        "resident_constants",
        "STACK_NAMES",
        "STACK_PREFIXES",
        "require_stack",
    }
    assert tuple(inspect.signature(compiler_module._compile_exported_program).parameters) == (
        "program",
    )
    assert tuple(inspect.signature(compiler_module._package_compiled_files).parameters) == (
        "name",
        "files",
        "output",
    )


def test_compile_installs_the_sealed_host_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: list[bool] = []

    def install() -> bool:
        installed.append(True)
        return True

    monkeypatch.setattr(compiler_module, "_impose_host_policy", lambda: {})
    monkeypatch.setattr(wrapper_split, "install", install)
    monkeypatch.setattr(compiler_module, "_aot_compile", lambda *args, **kwargs: ["model.so"])

    assert compiler_module._compile_exported_program(_program()) == ("model.so",)
    assert installed == [True]


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
    assert compiler_module._package_compiled_files("denoiser/h=64", ["a.so"], output) == output
    assert seen["files"] == {"denoiser/h=64": ["a.so"]}


def test_compile_inputs_are_derived_so_a_demoted_program_still_compiles() -> None:
    """pgw#1465: the derive drops example_inputs, and the compile must not care.

    This is the red arm for a P0 that took all 14 sd1.5 graph specializations to 0/14.
    `_demote_example_inputs` was right -- a hollow trace's example inputs pickle
    `_reconstruct_fake_tensor`, which torch's OWN loader refuses under
    `weights_only=True`, so a blob that kept them could not be reloaded on ANY
    device -- but `compiler.py` was the one reader a cross-repo grep missed,
    because it lives in torchcg itself.

    The fix is the same move tcg#55 made everywhere else: derive it. The flat
    call is `graph_signature.user_inputs`, the values are the placeholders'
    `meta['val']`, and `call_spec.in_spec` restores the nesting -- all three
    already in the program, none of them supplied.
    """

    torch = pytest.importorskip("torch")

    from torchcg.compiler import _compile_inputs
    from torchcg.discovery import _demote_example_inputs

    class Nested(torch.nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, sample: Any, conditioning: Any, return_dict: bool = False) -> Any:
            out = self.lin(sample) + conditioning["a"] + conditioning["b"]
            return {"o": out} if return_dict else (out,)

        # A nested dict AND a specialized non-tensor: both are things a flat
        # list of placeholder values alone could not put back.

    program = torch.export.export(
        Nested(),
        (torch.zeros(2, 4), {"a": torch.zeros(2, 4), "b": torch.zeros(2, 4)}),
        {"return_dict": False},
    )
    original_args, original_kwargs = program.example_inputs

    args, kwargs = _compile_inputs(program)
    assert len(args) == len(original_args)
    assert isinstance(args[1], dict) and set(args[1]) == {"a", "b"}, (
        "in_spec must restore the nesting, not hand back loose tensors"
    )
    assert kwargs == original_kwargs == {"return_dict": False}, (
        "a specialized constant is part of the graph and is replayed verbatim"
    )

    _demote_example_inputs(program)
    assert program.example_inputs is None
    # The whole point: still derivable with the field gone.
    again_args, again_kwargs = _compile_inputs(program)
    assert len(again_args) == len(args) and again_kwargs == kwargs
