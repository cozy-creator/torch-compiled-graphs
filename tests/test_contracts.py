from __future__ import annotations

import json
from pathlib import Path

import pytest

from torch_compiled_graphs.contracts import CONTRACT_FILES, read_contract

_LEGACY_PEER_PROJECTION = (
    "KEY_GRAMMAR_DIGEST",
    "compiled_graph_key_vectors.json",
)


def test_canonical_contracts_are_importable_package_data() -> None:
    assert CONTRACT_FILES == (
        "KEY_GRAMMAR_DIGEST",
        "call_ingress_v1.json",
        "compiled_graph_key_vectors.json",
        "graph_class_identity_v3.json",
        "literal_identity_v1.json",
        "toolchain_identity_v1.json",
    )
    for name in CONTRACT_FILES:
        data = read_contract(name)
        assert data
        if name.endswith(".json"):
            assert json.loads(data)


def test_unknown_contract_name_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown compiled-graph contract"):
        read_contract("../pyproject.toml")


def test_legacy_peer_projection_is_exactly_canonical() -> None:
    """Remove this projection when Tensorhub th#1947 consumes the wheel."""
    projected = Path(__file__).parent / "testdata"
    assert {path.name for path in projected.iterdir()} == set(_LEGACY_PEER_PROJECTION)
    for name in _LEGACY_PEER_PROJECTION:
        assert (projected / name).read_bytes() == read_contract(name)
