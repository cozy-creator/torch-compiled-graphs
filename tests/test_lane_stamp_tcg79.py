"""tcg#79: a lane is the v2 (topology, quant) STAMP, and only that.

The tensor-layout contract v2 cut (tensorfs#151 -> th#2250 -> pgw#1621) changed
what a serve lane is NAMED BY. It was one contract handle::

    sdxl.diffusers-bf16@1

It is now the stamp PAIR, rendered ``<topology>+<quant>``::

    sdxl.diffusers@1+plain.bf16@1

Three things this file holds down, each of which cost somebody a night:

1. **The hyphenated PRODUCER.** tensorfs' ``isTFM1ContractName`` admits ``-``
   in the producer segment but never leads with one. Seven of the twenty-three
   v2 topologies need it. A grammar written by eye refuses every one of them,
   and pgw's did -- five migrated endpoints could not declare a lane at all.
2. **The ``+`` join.** tensorfs ``LayoutID.String``, the tensorfs SDK's
   ``render()`` and the derived-artifact CAS address all spell it. A drift here
   is a CAS-address fork, so both a MISSING and a DOUBLED join are refusals.
3. **The v1 spelling is refused BY NAME**, not coerced and not merely
   rejected: there is no rule that recovers the pair from a pre-crossed handle,
   and a generic refusal reads like a typo instead of a migration.

The accepted corpus is tensorfs' own banked answer key,
``spec/v2/baselines/stamps.json``. It is restated here so the fence always
runs (CI has no sibling checkouts) and cross-checked against the sibling when
one IS on the box -- the skip names the absent path rather than printing green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torchcg.document import DOCUMENT_FORMAT, DocumentError, GraphSetDocument, LaneGraphs
from torchcg.lane import LANE_JOIN, LaneError, LaneRef, parse_lane_id, require_lane_id

#: tensorfs `spec/v2/baselines/stamps.json` -> `stamps`, verbatim (2026-08-21).
TENSORFS_STAMPS = {
    "anima-base": "anima.net@1+plain.bf16@1",
    "anima-turbo": "anima.diffusion-model@1+plain.bf16@1",
    "ernie": "ernie.diffusers@1+plain.bf16@1",
    "flux2-klein": "flux2-klein.diffusers@1+plain.bf16@1",
    "hidream-o1": "hidream-o1.diffusers@1+plain.f32@1",
    "joycaption": "joycaption.llava@1+plain.bf16@1",
    "ltx-2": "ltx2.diffusers@1+plain.bf16@1",
    "ltx-2-upsampler": "ltx2-upsampler.diffusers@1+plain.bf16@1",
    "minimax-h3-diffusers": "minimax-h3.diffusers@1+plain.bf16@1",
    "musicgen": "musicgen.transformers@1+plain.f16@1",
    "qwen3.6-35b-a3b": "qwen3-6-35b-a3b.transformers@1+plain.bf16@1",
    "sd-turbo": "sd2.diffusers@1+plain.f32@1",
    "sd15": "sd15.diffusers@1+plain.f32@1",
    "sdxl-base": "sdxl.diffusers@1+plain.f32@1",
    "sdxl-base-fp16": "sdxl.diffusers@1+plain.f16@1",
    "sdxl-base-sharded": "sdxl.diffusers@1+plain.bf16@1",
    "sdxl-inpainting": "sdxl-inpainting.diffusers@1+plain.f32@1",
    "sdxl-inpainting-bf16": "sdxl-inpainting.diffusers@1+plain.bf16@1",
    "wan22": "wan22.diffusers@1+plain.f32@1",
    "z-image": "z-image.diffusers@1+plain.bf16@1",
}

#: The topologies whose producer segment carries a hyphen. THESE are what a
#: hand-written `[a-z0-9]+\.` producer class silently refuses.
HYPHENATED_PRODUCERS = (
    "flux2-klein.diffusers@1",
    "hidream-o1.diffusers@1",
    "minimax-h3.diffusers@1",
    "minimax-h3.native@1",
    "z-image.diffusers@1",
    "ltx2-upsampler.diffusers@1",
    "sdxl-inpainting.diffusers@1",
    "qwen3-6-35b-a3b.transformers@1",
    "qwen3-6-35b-a3b.transformers-split@1",
)

#: The quant rules v2 ships; the right half of every stamp above.
QUANT_RULES = (
    "plain.bf16@1",
    "plain.f16@1",
    "plain.f32@1",
    "cozy.fp8-rowwise@1",
    "cozy.fp8-storage@1",
    "cozy.nvfp4-flat@1",
    "hf.fp8-blockwise@1",
    "bfl.nvfp4-preswizzled@1",
)

TENSORFS_BASELINE = Path.home() / "cozy/tensorfs/spec/v2/baselines/stamps.json"

STAMP = "sdxl.diffusers@1+plain.bf16@1"


# -- accepted -----------------------------------------------------------------


@pytest.mark.parametrize("stamp", sorted(set(TENSORFS_STAMPS.values())))
def test_every_banked_v2_stamp_is_a_legal_lane(stamp: str) -> None:
    assert require_lane_id(stamp) == stamp
    assert LaneRef(stamp).contract == stamp


@pytest.mark.parametrize("topology", HYPHENATED_PRODUCERS)
@pytest.mark.parametrize("quant", QUANT_RULES)
def test_a_hyphenated_producer_crossed_with_every_rule_is_legal(
    topology: str, quant: str
) -> None:
    """The exact break that took five migrated endpoints off the board."""

    stamp = f"{topology}{LANE_JOIN}{quant}"
    assert require_lane_id(stamp) == stamp
    assert parse_lane_id(stamp) == (topology, quant)


def test_parse_splits_at_the_join_and_keeps_both_halves_whole() -> None:
    assert parse_lane_id(STAMP) == ("sdxl.diffusers@1", "plain.bf16@1")
    # A format segment may hold `.`, `-` and `_`; the producer cut is the FIRST
    # dot, exactly as tensorfs' `strings.Cut(name, ".")` does.
    assert parse_lane_id("a.b.c@1+d.e_f-g@2") == ("a.b.c@1", "d.e_f-g@2")


def test_a_large_but_uint32_version_is_legal() -> None:
    stamp = "sdxl.diffusers@4294967295+plain.bf16@4294967295"
    assert require_lane_id(stamp) == stamp


def test_the_restated_corpus_matches_the_tensorfs_baseline() -> None:
    """Drift detector for the table above. Skips BY NAME, never green-by-absence."""

    if not TENSORFS_BASELINE.is_file():
        pytest.skip(f"tensorfs sibling checkout absent on this machine: {TENSORFS_BASELINE}")
    banked = json.loads(TENSORFS_BASELINE.read_text())["stamps"]
    assert banked == TENSORFS_STAMPS


# -- refused ------------------------------------------------------------------


def test_the_retired_v1_spelling_is_refused_BY_NAME() -> None:
    """A bare handle is not a lane, and the refusal says which migration it is."""

    with pytest.raises(LaneError) as caught:
        require_lane_id("sdxl.diffusers-bf16@1")
    message = str(caught.value)
    assert "RETIRED v1" in message
    assert "sdxl.diffusers-bf16@1" in message
    # It must name the shape to write instead, or the refusal is a dead end.
    assert "sdxl.diffusers@1+plain.bf16@1" in message


@pytest.mark.parametrize(
    "spelling",
    [
        # Bare handles of every real shape: a v1 product, a topology alone, a
        # quant rule alone. NONE of them is a lane -- a lane is a pair.
        "sdxl.diffusers-bf16@1",
        "sdxl.diffusers@1",
        "plain.bf16@1",
        "cozy.sdxl-fp8-rowwise@1",
    ],
)
def test_a_bare_handle_is_never_a_lane(spelling: str) -> None:
    with pytest.raises(LaneError):
        require_lane_id(spelling)


@pytest.mark.parametrize(
    ("spelling", "why"),
    [
        ("-sdxl.diffusers@1+plain.bf16@1", "a producer may hold a hyphen, never lead with one"),
        ("sdxl.-diffusers@1+plain.bf16@1", "nor may a format lead with one"),
        ("sdxl.diffusers@0+plain.bf16@1", "version 0 is refused, as tensorfs ParseHandle does"),
        ("sdxl.diffusers@1+plain.bf16@0", "on the quant axis too"),
        ("sdxl.diffusers@01+plain.bf16@1", "a leading zero is a second spelling of one stamp"),
        ("sdxl.diffusers@1+plain.bf16@1+extra.thing@1", "a doubled join is not a triple"),
        ("sdxl.diffusers@1+", "a stated-but-empty quant half"),
        ("+plain.bf16@1", "a stated-but-empty topology half"),
        ("sdxl.diffusers@1plain.bf16@1", "the join cannot be dropped"),
        ("sdxl.diffusers@1 + plain.bf16@1", "nor padded -- the bytes are the address"),
        ("SDXL.diffusers@1+plain.bf16@1", "uppercase is not the tensorfs name grammar"),
        ("sdxl@1+plain.bf16@1", "a handle names <producer>.<format>, and the dot is required"),
        ("sdxl.diffusers+plain.bf16", "an unversioned handle"),
        ("sha256:" + "a" * 64, "an anonymous contract digest is not a stamp"),
        ("", "the empty string"),
    ],
)
def test_a_noncanonical_spelling_is_refused(spelling: str, why: str) -> None:
    with pytest.raises(LaneError):
        require_lane_id(spelling)


def test_an_overlong_contract_name_is_refused_at_the_manifest_bound() -> None:
    """64 bytes is tensorfs' TFM1MaxContractNameBytes, not a taste."""

    at_bound = "a" * 55 + ".diffusers"  # 65 chars: 55 + 1 + 9? -> compute below
    assert len(at_bound) == 65
    with pytest.raises(LaneError, match="TFM1MaxContractNameBytes"):
        require_lane_id(f"{at_bound}@1+plain.bf16@1")
    just_inside = "a" * 54 + ".diffusers"
    assert len(just_inside) == 64
    assert require_lane_id(f"{just_inside}@1+plain.bf16@1")


@pytest.mark.parametrize("value", [None, 1, b"sdxl.diffusers@1+plain.bf16@1", ("a", "b")])
def test_a_non_string_is_refused(value: object) -> None:
    with pytest.raises(LaneError):
        require_lane_id(value)


# -- the document restates the same grammar -----------------------------------


def _lane(contract: str) -> LaneGraphs:
    return LaneGraphs(contract=contract, targets=("unet",), graphs=(), unobserved_targets=("unet",))


def test_lane_graphs_takes_a_stamp_and_refuses_a_v1_handle() -> None:
    assert _lane(STAMP).contract == STAMP
    with pytest.raises(DocumentError, match="RETIRED v1"):
        _lane("sdxl.diffusers-bf16@1")


def test_the_document_format_is_bumped_and_an_older_document_refuses_by_VERSION() -> None:
    """A v3 document's lane keys resolve against no v2 stamp, so it is a
    document this build CANNOT READ -- named as a version, not as a lane typo.
    `store.py` already treats that as absent (tcg#69): one re-derive, no
    migration machinery."""

    assert DOCUMENT_FORMAT == 4
    document = GraphSetDocument(stack=(("torch", "2.13.0"),), lanes=(_lane(STAMP),))
    encoded = json.loads(document.encode())
    assert encoded["v"] == 4
    assert encoded["lanes"][0]["contract"] == STAMP

    stale = dict(encoded, v=3)
    stale["lanes"] = [dict(encoded["lanes"][0], contract="sdxl.diffusers-bf16@1")]
    with pytest.raises(DocumentError, match="document v must be 4"):
        GraphSetDocument.decode(stale)
