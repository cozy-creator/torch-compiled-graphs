"""Banked facts about the KEY: policy-in-key, and layout as a morphism handle."""

from __future__ import annotations

from typing import Any

import pytest

from torchcg.identity import (
    KEY_SCHEME,
    ArtifactKey,
    artifact_key,
    contiguous_handle,
    is_artifact_key,
    require_morphism,
)
from torchcg.refuse import IdentityError, LayoutError, LayoutUndeliverableError

pytest.importorskip("torch")

ENV = {"torch": "2.13.0", "triton": "3.7.1", "glibc": "2.36", "os_release": "debian-12"}
FOLD_ON = {"always_keep_tensor_constants": True, "aot_inductor.package": True}
FOLD_OFF = {"always_keep_tensor_constants": False, "aot_inductor.package": True}
GRAPH = "cg-graph-v1-" + "a" * 56


def key(**overrides: Any) -> ArtifactKey:
    fields = dict(overrides)
    graph = fields.pop("graph", GRAPH)
    return artifact_key(
        graph,
        sm=fields.pop("sm", "sm_89"),
        env=fields.pop("env", ENV),
        policy=fields.pop("policy", FOLD_ON),
        layout=fields.pop("layout", contiguous_handle()),
    )


def test_the_key_states_its_scheme_and_shape() -> None:
    value = key().value
    assert value.startswith(f"{KEY_SCHEME}-")
    assert is_artifact_key(value)


def test_policy_is_in_the_key_so_differently_configured_mints_never_collide() -> None:
    """tcg#80, measured twice on 2026-08-21 as a `FileExistsError`.

    The graph axis is blind to compile options: a fold-on mint and a fold-off
    mint of ONE graph render identically and emit different `.so` bytes. Leaving
    the options out was not minimalism -- it made the two one address, and the
    second publish died.
    """

    assert key(policy=FOLD_ON).value != key(policy=FOLD_OFF).value


def test_every_axis_moves_the_key() -> None:
    """The red arm for the key as a whole: an axis that cannot move the address
    is an axis the key does not actually carry."""

    base = key().value
    assert key(sm="sm_86").value != base
    assert key(env={**ENV, "torch": "2.14.0"}).value != base
    assert key(policy=FOLD_OFF).value != base
    assert key(graph="cg-graph-v1-" + "b" * 56).value != base


def test_the_key_is_stable_across_restatement() -> None:
    assert key().value == key().value
    # Block ordering is not an axis.
    assert key(env=dict(reversed(list(ENV.items())))).value == key().value


def test_declared_input_layout_is_a_MORPHISM_HANDLE_not_a_free_string() -> None:
    """tcg#83/#85 as amended by tcg#87.

    The axis carries the ratified handle ITSELF rather than a digest, so a
    refusal can say which two layouts disagree. A value nobody ratified is a
    refusal, never a coercion to row-major: a declaration no loader can verify
    would let an artifact claim a layout its loader silently ignores, which is
    the defect this axis exists to remove.
    """

    handle = contiguous_handle()
    assert key(layout=handle).axes()["layout"] == handle
    with pytest.raises(LayoutError):
        key(layout="torch.contiguous")
    with pytest.raises(LayoutError):
        key(layout="cozy.invented-yesterday@1")
    with pytest.raises(LayoutError):
        key(layout=None)


def test_the_handle_is_READ_from_tensorfs_not_authored_here() -> None:
    """tcg#87: two producers of a key axis value is tcg#80's disease one level
    up. The old tree spelled `torch.channels-last@1` while the ratifying corpus
    said `torch.channels_last-2d@1`, and both were live."""

    import torchcg.identity as identity

    source = (
        identity.__file__ and open(identity.__file__).read()  # noqa: SIM115
    )
    for authored in ("torch.contiguous@1", "torch.channels-last@1", "channels_last-2d@1"):
        assert authored not in source, (
            f"identity.py spells the layout handle {authored!r}; handles have ONE "
            f"producer and it is tensorfs' spec/v2/layouts"
        )
    assert contiguous_handle() == require_morphism(contiguous_handle()).handle


def test_a_ratified_layout_torch_cannot_deliver_refuses_DIFFERENTLY() -> None:
    """"Nobody ratified this" is answered by writing a record; "torch has no
    memory format for it" is answered by the tensorfs fill path. Collapsing the
    two into one message sends a reader to the wrong remedy."""

    with pytest.raises(LayoutUndeliverableError):
        require_morphism("torch.transposed@1")
    # ...and it is a LayoutError, so a caller that only knows the base class
    # still catches it.
    with pytest.raises(LayoutError):
        require_morphism("torch.transposed@1")


def test_an_unstatable_axis_refuses() -> None:
    with pytest.raises(IdentityError):
        key(graph="not-a-graph-hash")
    with pytest.raises(IdentityError):
        key(sm="ampere")
    with pytest.raises(IdentityError):
        key(env={})
    with pytest.raises(IdentityError):
        key(policy={})
    # A float cannot be canonically restated across languages.
    with pytest.raises(IdentityError):
        key(policy={"threshold": 0.5})


def test_the_compile_stack_SELECTS_and_refuses_the_whole_installed_set() -> None:
    """pgw#1489: handing an env fingerprint every installed package splits the
    artifact pool on drift that cannot reach the compiler — a docs extra, a
    pillow bump, a dev tool (43-package diffs measured between environments that
    serve identically). Only torch, triton and the nvidia-* wheels can change
    what inductor emits."""

    from torchcg.identity import compile_stack, is_compile_relevant

    lock = {
        "torch": "2.13.0", "triton": "3.7.1", "nvidia-cublas-cu13": "13.0.0",
        "pillow": "11.0.0", "sphinx": "8.0.0", "gen-worker": "0.1.0",
    }
    assert compile_stack(lock) == {
        "torch": "2.13.0", "triton": "3.7.1", "nvidia-cublas-cu13": "13.0.0",
    }
    # Spelling-tolerant on SELECTION only; the pair goes in verbatim.
    assert is_compile_relevant("nvidia_cublas_cu12")
    assert compile_stack({"torch": "2.13.0", "nvidia_cudnn_cu13": "9.0"})[
        "nvidia_cudnn_cu13"
    ] == "9.0"


def test_a_stack_that_cannot_name_torch_is_refused() -> None:
    """torch IS the compiler; an env block without it describes a machine that
    cannot mint."""

    from torchcg.identity import compile_stack

    with pytest.raises(IdentityError, match="states torch"):
        compile_stack({"triton": "3.7.1"})
    with pytest.raises(IdentityError, match="one distribution"):
        compile_stack({"torch": "2.13.0", "Torch": "2.14.0"})
    # Two SPELLINGS of one library are two sources for one fact, even when the
    # versions agree — that is the pgw#1472 disease, not a harmless alias.
    with pytest.raises(IdentityError, match="one distribution"):
        compile_stack({
            "torch": "2.13.0",
            "nvidia_cublas_cu13": "13.0.0",
            "nvidia-cublas-cu13": "13.0.0",
        })
