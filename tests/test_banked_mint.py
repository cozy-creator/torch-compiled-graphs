"""Banked facts about the MINT: fold policy, the SM gate, range narrowing,
device uniformity."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

import torchcg.identity as identity
import torchcg.mint as mint
from torchcg.identity import build_call_ingress, placement
from torchcg.refuse import (
    DroppedOptimization,
    IdentityError,
    MintError,
    RangeNarrowed,
)

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


# ---------------------------------------------------------------------------
# Host ISA -- the classifier reads /proc/cpuinfo's vocabulary, not the psABI's
# ---------------------------------------------------------------------------

#: The flags a 13th-gen Intel Core prints, trimmed to the ones the levels use.
#: Note `abm` and the ABSENCE of `lzcnt`: Linux never emits the latter.
_INTEL_V3_FLAGS = frozenset({
    "cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3",
    "abm", "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "xsave",
})


def test_a_full_v3_host_classifies_as_v3() -> None:
    """The regression this file exists for. A level table written from the psABI
    document requires `lzcnt`; `/proc/cpuinfo` reports that capability as `abm`
    and never prints `lzcnt`, so the spec-spelled table matches nothing and
    demotes every Intel host one level -- costing AVX2 and FMA in every artifact
    it mints, silently, with no error anywhere.

    Measured on this box (i9-13950HX): full v3 feature set, classified v2.
    """

    assert identity.isa_level(_INTEL_V3_FLAGS) == "x86-64-v3"
    assert "lzcnt" not in _INTEL_V3_FLAGS
    # ...and the table must not be ASKING for a name Linux does not print.
    for _, required in identity._X86_LEVELS:
        assert "lzcnt" not in required, (
            "the level table spells LZCNT the psABI way; /proc/cpuinfo says `abm`"
        )


def test_the_levels_are_ordered_and_each_one_is_reachable() -> None:
    v2 = frozenset({"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"})
    assert identity.isa_level(frozenset()) == "x86-64"
    assert identity.isa_level(v2) == "x86-64-v2"
    assert identity.isa_level(_INTEL_V3_FLAGS) == "x86-64-v3"
    # One flag short of v3 falls back to v2 rather than claiming v3.
    assert identity.isa_level(_INTEL_V3_FLAGS - {"avx2"}) == "x86-64-v2"


def test_v3_widens_the_vector_register_budget() -> None:
    """simdlen follows the level: 256-bit vectors are exactly what v3 buys."""

    machine, march, simdlen = identity._host_isa()
    if machine != "x86_64":
        pytest.skip("simdlen policy is x86-only")
    assert (simdlen == 256) == (march == "x86-64-v3")


def test_THIS_host_is_classified_correctly() -> None:
    """Not a substitute for the stated-flag-set arms above -- it can only agree
    with whatever this CPU is. It is here because the demotion was found on a
    real host and would have been caught here first."""

    import platform

    if platform.machine() != "x86_64":
        pytest.skip("x86-only")
    features = {
        flag
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.split(":", 1)[0].strip() in ("flags", "Features")
        for flag in line.split(":", 1)[1].split()
    }
    expected = identity.isa_level(frozenset(features))
    assert identity._host_isa()[1] == expected
    if {"avx2", "fma", "bmi2"} <= features:
        assert expected == "x86-64-v3", (
            f"this CPU has AVX2/FMA/BMI2 and classified {expected}"
        )


def test_the_host_ISA_is_IMPOSED_into_the_key_not_left_to_the_caller() -> None:
    """The ISA is the one env fact this package decides, so it cannot be
    optional. A v2-compiled and a v3-compiled artifact of one graph differ in
    emitted instructions and in nothing else the key can see."""

    merged = identity.imposed_env({"torch": "2.13.0"})
    assert merged["torch"] == "2.13.0"
    for name in ("machine", "cpp_march", "cpp_simdlen"):
        assert name in merged, f"{name} is not in the keyed env fingerprint"
    # ...and it is the value this host actually imposes.
    assert merged["cpp_march"] == (identity._host_isa()[1] or "")


def test_two_ISA_LEVELS_cannot_share_one_key() -> None:
    """The collision the merge exists to prevent, asked directly."""

    from torchcg.identity import artifact_key, contiguous_handle

    graph = "cg-graph-v1-" + "a" * 56

    # `artifact_key` imposes the LOCAL host's facts, so the two levels are
    # compared through the block the key actually digests rather than by lying
    # to the constructor -- which it would (correctly) refuse.
    from torchcg.identity import _block_digest

    def axes(march: str) -> str:
        return _block_digest(
            "env fingerprint",
            {"torch": "2.13.0", "machine": "x86_64", "cpp_march": march,
             "cpp_simdlen": "256"},
            (str,),
        )

    assert axes("x86-64-v2") != axes("x86-64-v3")
    # ...and the imposed facts really are inside the keyed block.
    minted = artifact_key(
        graph, sm="sm_89", env={"torch": "2.13.0"},
        policy={"always_keep_tensor_constants": True}, layout=contiguous_handle(),
    )
    assert minted.env["cpp_march"] == identity.host_facts()["cpp_march"]


def test_a_caller_that_states_a_DIFFERENT_isa_is_refused() -> None:
    """Disagreement means the caller believes something false about the machine
    it is minting on -- a refusal, never a silent overwrite."""

    with pytest.raises(IdentityError, match="ISA facts are measured here"):
        identity.imposed_env({"torch": "2.13.0", "cpp_march": "x86-64-v4"})
    # Restating it CORRECTLY is fine: the caller is allowed to know.
    imposed = mint.impose_host_policy()
    merged = identity.imposed_env({"torch": "2.13.0", **imposed})
    assert merged["cpp_march"] == imposed["cpp_march"]
