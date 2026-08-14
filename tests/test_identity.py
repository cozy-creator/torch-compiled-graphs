from __future__ import annotations

from pathlib import Path

import pytest
from hashrepo import LocalCAS

from torch_compiled_graphs import (
    Engine,
    IdentityError,
    StorageError,
    from_axes,
    is_compiled_graph_key,
)


def test_key_is_stable_and_has_only_the_three_compilation_axes() -> None:
    key = from_axes({"sm": "sm_89", "toolchain": "0123456789abcdef", "graph": "fedcba9876543210"})
    assert key.canonical() == (
        b'{"graph":"fedcba9876543210","sm":"sm_89","toolchain":"0123456789abcdef"}'
    )
    assert str(key) == (
        "cg-key-v1-aefe4c6d52d304f8ef7cc6f9ffae296113b1546defe761e1c12ac2cf521e8fce"
    )


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


@pytest.mark.parametrize(
    "value",
    [
        "future-scheme-" + "0" * 64,
        "cg-key-v1-" + "A" * 64,
        "cg-key-v1-" + "0" * 63,
        "cg-key-v1-" + "0" * 64 + "\n",
    ],
)
def test_public_resolve_refuses_noncanonical_keys(value: str, tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="cg-key-v1"):
        Engine(LocalCAS(tmp_path / "cas")).resolve(value, tmp_path / "graph")
