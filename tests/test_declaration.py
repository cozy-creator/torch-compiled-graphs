from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from torchcg import (
    CallIngress,
    CallInput,
    DeclarationError,
    GraphClassDeclaration,
    GraphClassSpec,
    RuntimeCompatibility,
)
from torchcg.contracts import read_contract
from torchcg.declaration import _graph_digest

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


class LiteralOperation(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, value: float) -> None:
        super().__init__()
        self.literal = torch.tensor(value)

    def forward(self, value: Any) -> Any:
        return value + self.literal


class LiteralMatrixOperation(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.table = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    def forward(self, value: Any) -> Any:
        return value + self.table


def exported(module: Any) -> Any:
    return torch.export.export(module, (torch.ones(2),))


GRAPH_INTERFACE: dict[str, Any] = {
    "v": 3,
    "lifted_inputs": [],
    "pytree": {
        "in": "leaf",
        "out": "leaf",
        "ingress": CallIngress(
            parameters=("value",),
            flat_arity=1,
            inputs=(CallInput("value", 0, "value", 0, (), "value", "float32", (2,)),),
        ).as_dict(),
    },
    "specialization": {},
}


def spec(graph_class: str, target: str, program: object) -> GraphClassSpec:
    return GraphClassSpec(
        graph_class,
        target,
        program,
        GRAPH_INTERFACE,
    )


def test_declaration_keys_graph_structure_but_not_weight_values() -> None:
    first = spec("model", "denoiser", exported(Operation(weight=2.0))).declare()
    fine_tune = spec("model", "denoiser", exported(Operation(weight=9.0))).declare()
    changed_body = spec("model", "denoiser", exported(Operation(add=True))).declare()

    assert first == fine_tune
    assert len(first.graph_witness) == 16
    assert first.class_hash != changed_body.class_hash
    assert len(first.class_hash) == 16


def test_call_ingress_is_required_rekeys_and_owns_range_digest() -> None:
    program = exported(Operation())
    first = spec("model", "denoiser", program).declare()
    changed_graph = copy.deepcopy(GRAPH_INTERFACE)
    changed_ingress = CallIngress.decode(changed_graph["pytree"]["ingress"])
    changed_graph["pytree"]["ingress"] = CallIngress(
        parameters=changed_ingress.parameters,
        flat_arity=changed_ingress.flat_arity,
        inputs=(
            CallInput("value", 0, "value", 0, (), "value", "float32", (3,)),
        ),
    ).as_dict()
    changed = GraphClassSpec("model", "denoiser", program, changed_graph).declare()

    assert first.range_digest == CallIngress.from_graph(first.graph).digest()
    assert changed.range_digest != first.range_digest
    assert changed.class_hash != first.class_hash

    missing = copy.deepcopy(GRAPH_INTERFACE)
    del missing["pytree"]["ingress"]
    with pytest.raises(DeclarationError, match="declares no call ingress"):
        GraphClassSpec("model", "denoiser", program, missing).declare()

    numeric_alias = copy.deepcopy(GRAPH_INTERFACE)
    numeric_alias["v"] = 3.0
    with pytest.raises(DeclarationError, match="graph interface v must be 3"):
        GraphClassSpec("model", "denoiser", program, numeric_alias).declare()


def test_lifted_literal_values_ride_inside_graph_interface_and_rekey() -> None:
    first = spec("model", "denoiser", exported(LiteralOperation(2.0))).declare()
    changed = spec("model", "denoiser", exported(LiteralOperation(9.0))).declare()
    assert first.literal_values
    assert first.graph["literal_values"] == first.literal_values
    assert changed.graph["literal_values"] == changed.literal_values
    assert first.class_hash != changed.class_hash


def test_literal_digest_and_constant_names_match_current_worker_golden_vector() -> None:
    vector = json.loads(read_contract("literal_identity_v1.json"))
    program = torch.export.export(LiteralMatrixOperation(), (torch.ones(2, 2),))
    declaration = spec("model", "denoiser", program).declare()
    assert declaration.graph["constant_fqns"] == vector["constant_fqns"]
    assert declaration.graph["literal_values"] == vector["literal_values"]
    assert declaration.literal_values == vector["literal_values"]


def test_graph_class_target_and_runtime_are_exact_key_facts() -> None:
    program = exported(Operation())
    first = spec("model", "denoiser", program).declare()
    renamed = spec("other", "denoiser", program).declare()
    retargeted = spec("model", "vae", program).declare()
    runtime = RuntimeCompatibility("cpu", toolchain={"torch": "build-a"})

    assert first.class_hash == renamed.class_hash
    assert first.class_hash != retargeted.class_hash
    assert str(runtime.key(first)).startswith("cg-key-v1-")
    assert runtime.key(first) != RuntimeCompatibility(
        "cpu", toolchain={"torch": "build-b"}
    ).key(first)


def test_graph_witness_matches_current_worker_canonical_form() -> None:
    contiguous = torch.ones(2, 3)
    transposed = torch.ones(3, 2).transpose(0, 1)
    assert contiguous.shape == transposed.shape
    assert contiguous.stride() != transposed.stride()

    first = spec(
        "model", "layout", torch.export.export(LayoutOperation(), (contiguous,))
    ).declare()
    second = spec(
        "model", "layout", torch.export.export(LayoutOperation(), (transposed,))
    ).declare()
    assert first.graph_witness == second.graph_witness


def test_graph_witness_and_class_hash_match_current_worker_golden_vector() -> None:
    vector = json.loads(read_contract("graph_class_identity_v3.json"))
    block = vector["block"]
    program = torch.export.export(SineOperation(), (torch.ones(2, 3),))
    assert _graph_digest(program) == block["graph_witness"]
    declaration = GraphClassDeclaration(
        graph_class="display-name-does-not-key",
        target=block["target"],
        graph=block["graph"],
        graph_witness=block["graph_witness"],
        range_digest=block["range_digest"],
        fork=tuple((name, value) for name, value in block["fork"]),
        class_dims=tuple((name, value) for name, value in block["class_dims"]),
        strict=vector["strict"],
        lora_bucket=vector["lora_bucket"],
        placement=tuple(block["placement"]),
    )
    assert declaration.class_hash == vector["class_hash"]


class SineOperation(torch.nn.Module):  # type: ignore[misc]
    def forward(self, value: Any) -> Any:
        return value.sin()
