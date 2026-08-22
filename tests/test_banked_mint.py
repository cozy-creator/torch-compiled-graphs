"""Banked facts about the MINT: fold policy, the SM gate, range narrowing,
device uniformity."""

from __future__ import annotations

import ast
from typing import Any

import pytest

import torchcg.mint as mint
from torchcg.identity import build_call_ingress, placement
from torchcg.refuse import DroppedOptimization, MintError, RangeNarrowed

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from fixture import UNET_PARAMS, tiny_unet  # noqa: E402

# ---------------------------------------------------------------------------
# tcg#80 -- the fold policy
# ---------------------------------------------------------------------------


def test_always_keep_tensor_constants_is_on_and_runtime_folding_is_off() -> None:
    """The measured pair. `use_runtime_constant_folding` buys the same bindable
    constant table AND splits a const-fold subgraph that materializes a SECOND
    full constant set by direct cudaMalloc outside the caching allocator --
    +4782 MiB first-call transient on sdxl UNet-only.
    `always_keep_tensor_constants` buys the table with no second set."""

    assert mint.POLICY["always_keep_tensor_constants"] is True
    assert mint.POLICY["aot_inductor.use_runtime_constant_folding"] is False
    assert mint.POLICY["aot_inductor.package_constants_in_so"] is False


def test_the_fold_policy_is_KEYED() -> None:
    """A codegen option that never reaches the key is an option the fleet can
    change without re-minting -- which is how a fold-on and a fold-off artifact
    got one address."""

    policy = mint.compile_policy("cpu")
    assert policy["always_keep_tensor_constants"] is True
    assert "aot_inductor.use_runtime_constant_folding" in policy
    # ...and a topology option is NOT keyed: it changes compile speed, never
    # what inductor emits.
    assert "compile_threads" not in policy


def test_an_unclassified_option_REFUSES(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classify or refuse. An option nobody has decided about is one nobody
    decided whether the key must carry."""

    monkeypatch.setitem(mint.POLICY, "some.new_lever", True)
    with pytest.raises(MintError, match="classified neither"):
        mint.compile_policy("cpu")


def test_the_constant_table_must_be_fenced_at_READ_too() -> None:
    """The independent witness: an artifact whose stamped policy folds nothing
    is refused on load, whatever this build's own policy says."""

    import json
    import tempfile
    from pathlib import Path

    from torchcg.refuse import StoreError
    from torchcg.store import read_metadata

    for policy, expected in (
        ({"always_keep_tensor_constants": True,
          "aot_inductor.package_constants_in_so": False}, None),
        ({"always_keep_tensor_constants": False,
          "aot_inductor.use_runtime_constant_folding": False,
          "aot_inductor.package_constants_in_so": False}, "unfenced"),
        ({"always_keep_tensor_constants": True,
          "aot_inductor.package_constants_in_so": True}, "package_constants_in_so"),
    ):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            (directory / "metadata.json").write_text(
                json.dumps({"kind": "aot-inductor", "compile_policy": policy})
            )
            if expected is None:
                assert read_metadata(directory)["kind"] == "aot-inductor"
            else:
                with pytest.raises(StoreError, match=expected):
                    read_metadata(directory)


# ---------------------------------------------------------------------------
# tcg#85 -- the SM gate
# ---------------------------------------------------------------------------


def test_the_sm_gate_is_INERT_under_the_shipped_policy() -> None:
    """`max_autotune` is absent entirely, so the gate asks nothing. A guard
    that fires under the shipped configuration would be a bug, not a guard."""

    assert "max_autotune" not in mint.POLICY
    assert mint.compile_policy("cpu")


def test_the_sm_gate_is_read_from_torch_UNCACHED() -> None:
    """A `functools.cache`d verdict is a guard that cannot go red: the first
    device asked becomes the answer for every device after it."""

    import ast
    import inspect

    source = inspect.getsource(mint._gemm_templates_available)
    tree = ast.parse(source.lstrip()).body[0]
    assert isinstance(tree, ast.FunctionDef)
    # The docstring NAMES torch's threshold as the measured motivation; the
    # question here is whether the CODE restates it.
    body = ast.unparse(
        ast.Module(body=[n for n in tree.body if not _is_docstring(n)], type_ignores=[])
    )
    assert "__wrapped__" in body, "the gate must unwrap torch's cache"
    assert "min_sms" not in body and "68" not in body, (
        "the SM threshold is torch's to state; a copied number drifts silently"
    )


def _is_docstring(node: ast.stmt) -> bool:
    import ast

    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")
def test_the_sm_gate_REFUSES_a_requested_lever_the_target_would_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured on an RTX 4070 Laptop: `max_autotune: True` moved the key, cost
    622.5 s against an 85.2 s baseline, and torch logged `Not enough SMs to use
    max_autotune_gemm mode` -- 36 SMs against its internal min_sms = 68.

    Minting anyway produces an artifact that keys as if the lever ran and
    compiles as if it did not, so the fleet re-mints for nothing.
    """

    monkeypatch.setitem(mint.POLICY, "max_autotune", True)
    shortfall = mint._gemm_templates_available("cuda")
    if not shortfall:
        pytest.skip("this card clears torch's is_big_gpu; the gate cannot fire here")
    with pytest.raises(DroppedOptimization, match="SILENTLY DROP"):
        mint.compile_policy("cuda")
    # ...and a SANCTIONED drop is admitted, so the refusal is a policy and not
    # a wall.
    monkeypatch.setattr(mint, "ACCEPT_DROPPED", frozenset({"max_autotune"}))
    assert mint.compile_policy("cuda")["max_autotune"] is True


# ---------------------------------------------------------------------------
# tcg#78 -- a batch Dim that straddles 1
# ---------------------------------------------------------------------------


def _export_batch_dim(low: int, high: int) -> tuple[Any, Any]:
    unet = tiny_unet()
    batch = torch.export.Dim("batch", min=low, max=high)
    args = (torch.zeros(2, 4, 8, 8), torch.zeros(()), torch.zeros(2, 77, 16))
    kwargs = {"return_dict": False}
    program = torch.export.export(
        unet, args, kwargs, strict=False,
        dynamic_shapes=({0: batch}, None, {0: batch}, None),
    )
    return program, build_call_ingress(program, UNET_PARAMS, args, kwargs)


def test_a_batch_range_the_guards_support_is_ADMITTED() -> None:
    """The green half. Without it the refusal below proves only that something
    always raises."""

    program, _ = _export_batch_dim(2, 8)
    mint.assert_ranges_hold(program, "cg-graph-v1-" + "a" * 56)
    assert not mint.narrowed_symbols(
        mint.declared_ranges(program), mint.live_ranges(program)
    )


def test_a_batch_Dim_that_straddles_1_is_REFUSED() -> None:
    """Every dynamic dim is guarded `>= 2` -- torch specializes the sizes 0 and
    1 rather than reason about broadcasting and contiguity symbolically -- so an
    axis whose minimum is 1 is contradicted the moment it is exported.

    The measured consequence of NOT refusing: an sd15 UNet exported with a batch
    Dim over [1, 2] compiles for 84-129 s of GPU and produces an artifact that,
    called at batch 1, returns a (2, 4, 64, 64) tensor of garbage and raises
    nothing.
    """

    try:
        program, _ = _export_batch_dim(1, 2)
    except Exception as exc:  # torch may refuse the range at export
        assert "1" in str(exc) or "constrain" in str(exc).lower()
        return
    declared = mint.declared_ranges(program)
    live = mint.live_ranges(program)
    moved = mint.narrowed_symbols(declared, live)
    assert moved, (
        f"a batch Dim over [1, 2] left every declared range intact "
        f"(declared={declared}, live={live}); the narrowing detector is blind"
    )
    with pytest.raises(RangeNarrowed, match="NARROWER"):
        mint.assert_ranges_hold(program, "cg-graph-v1-" + "a" * 56)


def test_a_replacement_reads_as_the_STRONGEST_narrowing() -> None:
    """A symbol the ShapeEnv has replaced with a constant must not read as
    "unchanged" -- a singleton range is the narrowest possible answer, and the
    old digest-comparison guard caught it only by accident."""

    declared = {"s0": (1, 8)}
    assert mint.narrowed_symbols(declared, {"s0": (2, 2)}) == {"s0": ((1, 8), (2, 2))}
    # Widening is not narrowing: it still serves every declared shape.
    assert mint.narrowed_symbols(declared, {"s0": (1, 16)}) == {}


# ---------------------------------------------------------------------------
# tcg#89 -- device uniformity, asked as the two questions it is
# ---------------------------------------------------------------------------


class _Node:
    def __init__(self, value: Any) -> None:
        self.meta: dict[str, Any] = {"val": value}
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}


class _Program:
    def __init__(self, devices: list[str]) -> None:
        nodes = [_Node(torch.zeros(1, device=d)) for d in devices]
        self.graph_module = type(
            "GM", (), {"graph": type("G", (), {"nodes": nodes})()}
        )()


def test_a_uniform_placement_is_ADMITTED() -> None:
    program, _ = _export_batch_dim(2, 8)
    assert placement(program) == ("cpu",)
    mint.assert_device_uniform(program, "cg-graph-v1-" + "a" * 56)


def test_MIXED_DEVICE_TYPES_are_refused() -> None:
    """The fact the old test meant to guard. It compared a device STRING to a
    device TYPE (`{'cuda:0'} == {'cuda'}`), so it passed only where it could not
    fire: green on a cardless runner, red on every machine with a card."""

    with pytest.raises(MintError, match="straddles device types"):
        mint.assert_device_uniform(_Program(["cpu", "meta"]), "g")


def test_MIXED_DEVICE_INDICES_are_refused_SEPARATELY() -> None:
    """tcg#89's open question, answered: the index IS worth asserting on its
    own. One type and two cards is a real defect that the type check cannot
    see, and nothing else in the library looks for it."""

    program = _Program(["cpu"])
    program.graph_module.graph.nodes[0].meta["val"] = None
    program.graph_module.graph.nodes[0].args = (
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    )
    with pytest.raises(MintError, match="several INDICES"):
        mint.assert_device_uniform(program, "g")


def test_the_uniformity_check_compares_TYPE_against_TYPE() -> None:
    """The red arm for the tcg#89 defect itself: a single-card trace stamps
    `cuda:0`, and that must be ADMITTED. The broken comparison refused it."""

    program = _Program(["cpu"])
    program.graph_module.graph.nodes[0].meta["val"] = None
    program.graph_module.graph.nodes[0].args = (torch.device("cuda", 0),)
    mint.assert_device_uniform(program, "g")
