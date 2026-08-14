from __future__ import annotations

import pytest

from compiled_graphs import (
    IdentityError,
    contract_digest,
    from_axes,
    is_compiled_graph_key,
)


def test_key_is_stable_and_has_only_the_three_compilation_axes() -> None:
    key = from_axes({"sm": "sm_89", "toolchain": "0123456789abcdef", "graph": "fedcba9876543210"})
    assert key.canonical() == (
        b'{"graph":"fedcba9876543210","sm":"sm_89","toolchain":"0123456789abcdef"}'
    )
    assert str(key) == "cg-key-v1-aefe4c6d52d304f8ef7cc6f9ffae296113b1546defe761e1c12ac2cf"


def test_unknown_or_missing_axes_are_refused() -> None:
    with pytest.raises(IdentityError, match="unknown identity"):
        from_axes({"graph": "g", "sm": "s", "toolchain": "t", "family": "sdxl"})
    with pytest.raises(IdentityError, match="toolchain"):
        from_axes({"graph": "g", "sm": "s"})


def test_a_key_cannot_be_hashed_as_an_input_fact() -> None:
    key = str(from_axes({"graph": "g", "sm": "s", "toolchain": "t"}))
    with pytest.raises(IdentityError, match="not an identity fact"):
        from_axes({"graph": key, "sm": "s", "toolchain": "t"})
    with pytest.raises(IdentityError, match="not an identity fact"):
        contract_digest([key])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cg-key-v1-" + "a" * 56, True),
        ("future-scheme-" + "0" * 56, True),
        ("cg-key-v1-" + "A" * 56, False),
        ("cg-key-v1-" + "0" * 55, False),
        ("cg-key-v1-" + "0" * 56 + "\n", False),
    ],
)
def test_key_shape_is_scheme_agnostic_and_right_anchored(value: str, expected: bool) -> None:
    assert is_compiled_graph_key(value) is expected


def test_contract_digest_is_order_independent() -> None:
    assert contract_digest(["b", "a"]) == contract_digest(["a", "b"])
