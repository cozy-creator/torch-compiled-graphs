from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from hashrepo import LocalCAS

from torch_compiled_graphs import (
    CompiledGraphKey,
    Engine,
    IdentityError,
    StorageError,
    is_compiled_graph_key,
)
from torch_compiled_graphs.contracts import read_contract
from torch_compiled_graphs.identity import from_axes, toolchain_axis_digest


def test_key_is_stable_and_has_only_the_three_compilation_axes() -> None:
    key = from_axes({"sm": "sm_89", "toolchain": "0123456789abcdef", "graph": "fedcba9876543210"})
    assert key.canonical() == (
        b'{"graph":"fedcba9876543210","sm":"sm_89","toolchain":"0123456789abcdef"}'
    )
    assert str(key) == (
        "cg-key-v1-aefe4c6d52d304f8ef7cc6f9ffae296113b1546defe761e1c12ac2cf"
    )


def test_toolchain_axis_and_final_key_match_current_worker_golden_vector() -> None:
    vector = json.loads(read_contract("toolchain_identity_v1.json"))
    axis = toolchain_axis_digest(vector["block"])
    assert axis == vector["toolchain_axis"]
    assert str(
        from_axes({"graph": vector["graph"], "sm": vector["sm"], "toolchain": axis})
    ) == vector["key"]
    changed_model_libraries = dict(vector["block"])
    changed_model_libraries.update(diffusers="changed", transformers="changed", peft="changed")
    assert toolchain_axis_digest(changed_model_libraries) == axis


def test_unknown_or_missing_axes_are_refused() -> None:
    with pytest.raises(IdentityError, match="unknown identity"):
        from_axes({"graph": "g", "sm": "s", "toolchain": "t", "family": "sdxl"})
    with pytest.raises(IdentityError, match="toolchain"):
        from_axes({"graph": "g", "sm": "s"})


def test_a_key_cannot_be_hashed_as_an_input_fact() -> None:
    key = str(from_axes({"graph": "g", "sm": "s", "toolchain": "t"}))
    with pytest.raises(IdentityError, match="not an identity fact"):
        from_axes({"graph": key, "sm": "s", "toolchain": "t"})


def test_public_boundary_validator_accepts_only_the_key_shape() -> None:
    key = str(from_axes({"graph": "g", "sm": "s", "toolchain": "t"}))
    assert is_compiled_graph_key(key)
    assert not is_compiled_graph_key(key.upper())
    assert not is_compiled_graph_key("not-a-key")

    class StringLike:
        def __str__(self) -> str:
            return key

    assert not is_compiled_graph_key(StringLike())


def test_public_boundary_validator_matches_shared_key_corpus() -> None:
    corpus = read_contract("compiled_graph_key_vectors.json")
    recorded = next(
        line.strip()
        for line in read_contract("KEY_GRAMMAR_DIGEST").decode().splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert hashlib.sha256(corpus).hexdigest() == recorded
    vectors = json.loads(corpus)["vectors"]
    assert all(is_compiled_graph_key(row["key"]) is row["valid"] for row in vectors)


def test_compiled_key_constructor_enforces_the_three_canonical_axes() -> None:
    with pytest.raises(IdentityError, match="axes must be exactly"):
        CompiledGraphKey((("bogus", "axis"),))
    with pytest.raises(IdentityError, match="canonical string 'sm'"):
        CompiledGraphKey((("graph", "g"), ("sm", " cpu"), ("toolchain", "t")))


@pytest.mark.parametrize(
    "value",
    [
        "cg-key-v1-" + "A" * 56,
        "cg-key-v1-" + "0" * 55,
        "cg-key-v1-" + "0" * 56 + "\n",
    ],
)
def test_public_resolve_refuses_noncanonical_keys(value: str, tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="compiled-graph key"):
        Engine(LocalCAS(tmp_path / "cas")).resolve(value, tmp_path / "graph")


def test_public_resolve_accepts_a_future_scheme_as_a_clean_miss(tmp_path: Path) -> None:
    key = "future-scheme-" + "0" * 56
    assert Engine(LocalCAS(tmp_path / "cas")).resolve(key, tmp_path / "graph") is None
