from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from tensorfs import LocalCAS

from torchcg import (
    ARTIFACT_KIND,
    GRAPH_SPECIALIZATION_BLOCK,
    REQUIRED_AXES,
    CompiledGraphKey,
    Engine,
    IdentityError,
    StorageError,
    is_compiled_graph_key,
)
from torchcg.contracts import read_contract
from torchcg.identity import (
    from_artifact_metadata,
    from_axes,
    policy_axis_digest,
    toolchain_axis_digest,
)

#: One complete axis set, so a test about ONE axis states only that axis.
_AXES = {
    "compile_policy": "p",
    "declared_input_layout": "torch.contiguous@1",
    "graph": "g",
    "sm": "s",
    "toolchain": "t",
}


def test_key_is_stable_and_has_only_the_five_compilation_axes() -> None:
    key = from_axes(
        {
            "sm": "sm_89",
            "toolchain": "0123456789abcdef",
            "graph": "fedcba9876543210",
            "compile_policy": "89abcdef01234567",
            "declared_input_layout": "torch.contiguous@1",
        }
    )
    assert key.canonical() == (
        b'{"compile_policy":"89abcdef01234567","declared_input_layout":"torch.contiguous@1","graph":"fedcba9876543210","sm":"sm_89","toolchain":"0123456789abcdef"}'
    )
    assert str(key) == (
        "cg-key-v3-c18bfdb53993bfbfda3dde3334757f1705627eb6e59761ab85436179"
    )


def test_toolchain_axis_and_final_key_match_current_worker_golden_vector() -> None:
    vector = json.loads(read_contract("toolchain_identity_v1.json"))
    axis = toolchain_axis_digest(vector["block"])
    assert axis == vector["toolchain_axis"]
    assert str(
        from_axes(
            {
                "graph": vector["graph"],
                "sm": vector["sm"],
                "toolchain": axis,
                "compile_policy": vector["compile_policy_axis"],
                "declared_input_layout": vector["declared_input_layout"],
            }
        )
    ) == vector["key"]
    assert policy_axis_digest(vector["compile_policy"]) == vector["compile_policy_axis"]
    changed_model_libraries = dict(vector["block"])
    changed_model_libraries.update(diffusers="changed", transformers="changed", peft="changed")
    assert toolchain_axis_digest(changed_model_libraries) == axis


def test_unknown_or_missing_axes_are_refused() -> None:
    with pytest.raises(IdentityError, match="unknown identity"):
        from_axes({**_AXES, "family": "sdxl"})
    with pytest.raises(IdentityError, match="toolchain"):
        from_axes({name: value for name, value in _AXES.items() if name != "toolchain"})


def test_a_key_cannot_be_hashed_as_an_input_fact() -> None:
    key = str(from_axes(_AXES))
    with pytest.raises(IdentityError, match="not an identity fact"):
        from_axes({**_AXES, "graph": key})


def test_public_boundary_validator_accepts_only_the_key_shape() -> None:
    key = str(from_axes(_AXES))
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


def test_compiled_key_constructor_enforces_the_canonical_axes() -> None:
    with pytest.raises(IdentityError, match="axes must be exactly"):
        CompiledGraphKey((("bogus", "axis"),))
    with pytest.raises(IdentityError, match="canonical string 'sm'"):
        CompiledGraphKey(tuple(sorted({**_AXES, "sm": " cpu"}.items())))


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


def _artifact_metadata(block_name: str) -> dict[str, object]:
    """Minimal aot-inductor metadata with the graph-specialization facts under ``block_name``."""

    return {
        "kind": ARTIFACT_KIND,
        "sm": "sm_89",
        block_name: {"specialization_hash": "fedcba9876543210"},
        "toolchain": {"torch": "2.13.0"},
        "compile_policy": {"aot_inductor.package": True},
        "declared_input_layout": "torch.contiguous@1",
    }


def test_exported_required_axes_is_the_tuple_the_derivation_enforces() -> None:
    """The export must BE the axis set in force, not a copy that can drift.

    Every input here is derived from ``REQUIRED_AXES`` rather than spelled out,
    so the test follows a deliberate axis change but fails the moment the
    exported tuple and the code that consumes it disagree.
    """

    facts = {name: f"{name}fact" for name in REQUIRED_AXES}
    key = from_axes(facts)
    assert set(key.as_dict()) == set(REQUIRED_AXES)

    with pytest.raises(IdentityError, match="unknown identity axes"):
        from_axes({**facts, "notanaxis": "value"})

    for missing in REQUIRED_AXES:
        with pytest.raises(IdentityError, match=re.escape(repr(missing))):
            from_axes({name: value for name, value in facts.items() if name != missing})


def test_exported_graph_specialization_block_is_the_block_the_reader_reads() -> None:
    """The same facts under any other block name must refuse.

    This is the pgw regression in miniature: the worker read a block named
    ``entry`` while this package wrote ``graph_specialization``, and every fixture built
    the obsolete shape, so nothing went red until production.
    """

    key = from_artifact_metadata(_artifact_metadata(GRAPH_SPECIALIZATION_BLOCK))
    assert key.as_dict()["graph"] == "fedcba9876543210"

    with pytest.raises(IdentityError, match=re.escape(GRAPH_SPECIALIZATION_BLOCK)):
        from_artifact_metadata(_artifact_metadata("entry"))


def test_exported_artifact_kind_is_the_kind_identity_refuses_on() -> None:
    """The export must BE the kind this reader enforces, not a copy.

    ``identity`` spelled the literal while ``artifact`` used the constant, so a
    rename would have split the two readers silently — the same shape as the
    ``entry``/``graph_specialization`` outage, one field over. Both the accept and the
    refusal are derived from the export, so this follows a deliberate rename
    and fails only when the export and its reader disagree. ``artifact``'s
    half of the pair is fenced in ``test_artifact.py``.
    """

    metadata = _artifact_metadata(GRAPH_SPECIALIZATION_BLOCK)
    assert metadata["kind"] == ARTIFACT_KIND
    assert from_artifact_metadata(metadata).as_dict()["graph"] == "fedcba9876543210"

    with pytest.raises(IdentityError, match=re.escape(ARTIFACT_KIND)):
        from_artifact_metadata({**metadata, "kind": ARTIFACT_KIND + "-not"})
