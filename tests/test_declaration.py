from __future__ import annotations

from typing import Any

import pytest

from torch_compiled_graphs import GraphSpec, RuntimeCompatibility

torch: Any = pytest.importorskip("torch")


class Operation(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, *, add: bool = False, weight: float = 2.0) -> None:
        super().__init__()
        self.add = add
        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def forward(self, value: Any) -> Any:
        return value + self.weight if self.add else value * self.weight


class LayoutOperation(torch.nn.Module):  # type: ignore[misc]
    def forward(self, value: Any) -> Any:
        return value.relu()


def exported(module: Any) -> Any:
    return torch.export.export(module, (torch.ones(2),))


def test_declaration_keys_graph_structure_but_not_weight_values() -> None:
    first = GraphSpec("model", "denoiser", exported(Operation(weight=2.0))).declare()
    fine_tune = GraphSpec("model", "denoiser", exported(Operation(weight=9.0))).declare()
    changed_body = GraphSpec("model", "denoiser", exported(Operation(add=True))).declare()

    assert first == fine_tune
    assert first.class_hash != changed_body.class_hash


def test_entry_target_and_runtime_are_exact_key_facts() -> None:
    program = exported(Operation())
    first = GraphSpec("model", "denoiser", program).declare()
    renamed = GraphSpec("other", "denoiser", program).declare()
    runtime = RuntimeCompatibility("cpu", deployment_compatibility="test-image-a")

    assert first.class_hash != renamed.class_hash
    assert str(runtime.key(first)).startswith("cg-key-v1-")
    assert runtime.key(first) != RuntimeCompatibility(
        "cpu", deployment_compatibility="test-image-b"
    ).key(first)


def test_tensor_strides_and_layout_are_graph_identity() -> None:
    contiguous = torch.ones(2, 3)
    transposed = torch.ones(3, 2).transpose(0, 1)
    assert contiguous.shape == transposed.shape
    assert contiguous.stride() != transposed.stride()

    first = GraphSpec(
        "model", "layout", torch.export.export(LayoutOperation(), (contiguous,))
    ).declare()
    second = GraphSpec(
        "model", "layout", torch.export.export(LayoutOperation(), (transposed,))
    ).declare()
    assert first.graph != second.graph
    assert first.class_hash != second.class_hash
