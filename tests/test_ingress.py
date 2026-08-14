from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from torch_compiled_graphs import CallIngress, IngressError, build_call_ingress
from torch_compiled_graphs.contracts import read_contract

torch: Any = pytest.importorskip("torch")


class NestedCall(torch.nn.Module):  # type: ignore[misc]
    def forward(
        self,
        sample: Any,
        conditioning: dict[str, Any],
        shape: list[Any],
        return_dict: bool,
        tail: Any,
    ) -> Any:
        height = shape[0][0]
        width = shape[0][1]
        return (
            sample
            + conditioning["zeta"]
            + conditioning["alpha"]
            + tail
            + float(height + width)
            + int(return_dict)
        )


class VectorGraph(torch.nn.Module):  # type: ignore[misc]
    def forward(self, value: Any) -> Any:
        return value.sin()


def _exported() -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    args = (
        torch.ones(2, 3),
        {"zeta": torch.full((2, 3), 2.0), "alpha": torch.full((2, 3), 3.0)},
        [[2, 3]],
        False,
        torch.full((2, 3), 4.0),
    )
    return torch.export.export(NestedCall(), args, {}, strict=True), args, {}


def _ingress(*, excluded: tuple[str, ...] = ()) -> CallIngress:
    program, args, kwargs = _exported()
    return build_call_ingress(
        program,
        ("sample", "conditioning", "shape", "return_dict", "tail"),
        args,
        kwargs,
        excluded_inputs=excluded,
    )


def test_builder_matches_torch_export_order_and_preserves_constant_positions() -> None:
    ingress = _ingress(excluded=("adapter_b", "adapter_a"))

    assert [row.exported_name for row in ingress.inputs] == [
        "sample",
        "conditioning_zeta",
        "conditioning_alpha",
        "tail",
    ]
    assert [row.position for row in ingress.inputs] == [0, 1, 2, 6]
    assert ingress.flat_arity == 7
    assert ingress.inputs[-1].param == "tail"
    assert ingress.inputs[-1].param_position == 4
    assert ingress.inputs[-1].path == ()
    assert ingress.excluded_inputs == ("adapter_a", "adapter_b")


def test_mapping_insertion_order_is_the_export_order() -> None:
    ingress = _ingress()
    by_position = [row.name for row in ingress.inputs]
    assert by_position[1:3] == ["zeta", "alpha"]


def test_decoder_round_trips_only_the_closed_canonical_shape() -> None:
    ingress = _ingress()
    encoded = ingress.as_dict()

    assert CallIngress.decode(encoded) == ingress
    assert CallIngress.decode(copy.deepcopy(encoded)).as_dict() == encoded

    unknown = copy.deepcopy(encoded)
    unknown["alias"] = "legacy"
    with pytest.raises(IngressError, match="fields must be exactly"):
        CallIngress.decode(unknown)

    duplicate = copy.deepcopy(encoded)
    for field in ("name", "param", "param_position", "path", "exported_name"):
        duplicate["inputs"][1][field] = copy.deepcopy(duplicate["inputs"][0][field])
    with pytest.raises(IngressError, match="duplicate input name"):
        CallIngress.decode(duplicate)

    noncanonical = copy.deepcopy(encoded)
    noncanonical["inputs"] = list(reversed(noncanonical["inputs"]))
    with pytest.raises(IngressError, match="strictly increasing"):
        CallIngress.decode(noncanonical)


def test_bind_replays_the_recorded_paths_without_searching() -> None:
    ingress = _ingress()
    _, args, _ = _exported()
    decoy = {"zeta": torch.zeros(9, 9), "alpha": torch.zeros(9, 9)}

    bound = ingress.bind(args[:1], {
        "decoy": decoy,
        "conditioning": args[1],
        "shape": args[2],
        "return_dict": args[3],
        "tail": args[4],
    })
    assert bound["zeta"] is args[1]["zeta"]
    assert bound["alpha"] is args[1]["alpha"]
    assert ingress.feeds(args, {}) == (args[0], args[1]["zeta"], args[1]["alpha"], args[4])

    with pytest.raises(IngressError, match="conditioning"):
        ingress.bind(args[:1], {"decoy": decoy, "tail": args[4]})


def test_decoder_refuses_unbounded_or_unreferenced_symbols() -> None:
    encoded = _ingress().as_dict()
    encoded["symbols"] = {"unused": [1, 2]}
    with pytest.raises(IngressError, match="unreferenced"):
        CallIngress.decode(encoded)

    encoded = _ingress().as_dict()
    encoded["inputs"][0]["shape"][0] = "missing"
    with pytest.raises(IngressError, match="no declared bounds"):
        CallIngress.decode(encoded)


def test_importable_call_ingress_vector_is_exact() -> None:
    vector = json.loads(read_contract("call_ingress_v1.json"))
    ingress = CallIngress.decode(vector["ingress"])
    assert ingress.digest() == vector["digest"]
    program = torch.export.export(VectorGraph(), (torch.ones(2, 3),), strict=True)
    built = build_call_ingress(program, ("value",), (torch.ones(2, 3),), {})
    assert built.as_dict() == vector["ingress"]
