"""Two-level identity: env hashes, manifests as audit, the miss-path ladder."""

from __future__ import annotations

import pytest

from torchcg.graph_identity import (
    EnvIdentity,
    GraphIdentityError,
    closure_hash,
    installed_closure,
)
from torchcg.requirements import (
    ArtifactCandidate,
    EnvironmentMismatch,
    RequirementsError,
    RequirementsManifest,
    assert_exact_env,
    rank,
)

CLOSURE = closure_hash({"torch": "2.13.0", "nvidia-cublas-cu12": "12.8.4"})


def test_closure_hash_normalizes_names_and_orders() -> None:
    assert closure_hash({"Foo_Bar": "1.0", "baz": "2.0"}) == closure_hash(
        {"baz": "2.0", "foo-bar": "1.0"}
    )


def test_closure_hash_refuses_conflicting_duplicates_and_emptiness() -> None:
    with pytest.raises(GraphIdentityError):
        closure_hash({"foo-bar": "1.0", "Foo_Bar": "2.0"})
    with pytest.raises(GraphIdentityError):
        closure_hash({})


def test_installed_closure_hashes_this_process_env() -> None:
    entries = installed_closure()
    assert "torch" in {name.lower().replace("_", "-") for name in entries}
    assert len(closure_hash(entries)) == 64


def test_env_identity_is_deterministic_and_axis_sensitive() -> None:
    a = EnvIdentity(closure=CLOSURE, sm="sm_89")
    assert a.value == EnvIdentity(closure=CLOSURE, sm="sm_89").value
    assert a.value.startswith("cg-env-v1-")
    assert a.value != EnvIdentity(closure=CLOSURE, sm="sm_86").value
    other = closure_hash({"torch": "2.14.0"})
    assert a.value != EnvIdentity(closure=other, sm="sm_89").value


def test_env_identity_refuses_noncanonical_axes() -> None:
    with pytest.raises(GraphIdentityError):
        EnvIdentity(closure="short", sm="sm_89")
    with pytest.raises(GraphIdentityError):
        EnvIdentity(closure=CLOSURE, sm="whatever")


def manifest(**overrides: object) -> RequirementsManifest:
    base: dict[str, object] = {
        "include_set": (("torch", "2.13.0"), ("nvidia-cublas-cu12", "12.8.4")),
        "sm_compiled": "sm_89",
        "cuda_floor": "12.8",
        "autotuned_on": "sm_89",
    }
    base.update(overrides)
    return RequirementsManifest(**base)  # type: ignore[arg-type]


def test_manifest_roundtrips_and_refuses_emptiness() -> None:
    row = manifest()
    assert RequirementsManifest.decode(row.as_dict()) == row
    with pytest.raises(RequirementsError):
        manifest(include_set=())


def test_manifest_audit_passes_the_exact_env() -> None:
    manifest().assert_environment(
        {"Torch": "2.13.0", "nvidia_cublas_cu12": "12.8.4", "extra": "9"}, sm="sm_89"
    )


def test_manifest_audit_refuses_loudly_naming_every_divergence() -> None:
    with pytest.raises(EnvironmentMismatch) as caught:
        manifest().assert_environment({"torch": "2.12.0"}, sm="sm_75")
    message = str(caught.value)
    assert "torch: linked 2.13.0, installed 2.12.0" in message
    assert "nvidia-cublas-cu12" in message
    assert "sm: compiled sm_89, running sm_75" in message


def test_exact_env_audit_is_exact() -> None:
    installed = {"torch": "2.13.0", "nvidia-cublas-cu12": "12.8.4"}
    stamped = EnvIdentity(closure=closure_hash(installed), sm="sm_89")
    assert_exact_env(stamped, installed=installed, sm="sm_89")
    with pytest.raises(EnvironmentMismatch, match="sm"):
        assert_exact_env(stamped, installed=installed, sm="sm_86")
    with pytest.raises(EnvironmentMismatch, match="closure"):
        assert_exact_env(stamped, installed={"torch": "2.14.0"}, sm="sm_89")


GRAPH = "cg-graph-v1-" + "a" * 56


def candidate(sm_compiled: str, autotuned_on: str | None, digest: str) -> ArtifactCandidate:
    return ArtifactCandidate(
        graph=GRAPH,
        env=EnvIdentity(closure=CLOSURE, sm=sm_compiled),
        digest=digest,
        manifest=manifest(sm_compiled=sm_compiled, autotuned_on=autotuned_on),
    )


def test_rank_truth_table_with_sm_compat_boundaries() -> None:
    native = candidate("sm_89", "sm_89", "d1")
    native_heuristic = candidate("sm_89", None, "d2")
    compat_low = candidate("sm_86", "sm_86", "d3")
    compat_lower = candidate("sm_80", None, "d4")
    wrong_major = candidate("sm_90", "sm_90", "d5")
    above_host = candidate("sm_120", None, "d6")

    ordered = rank(
        [above_host, compat_lower, wrong_major, compat_low, native_heuristic, native],
        sm="sm_89",
    )
    # Runnable: same sm, or same major with host minor >= compiled minor.
    assert [row.digest for row in ordered] == ["d1", "d2", "d3", "d4"]

    # Boundary: compiled minor == host minor is runnable; a host below the
    # compiled minor is not; a different major never is (sm_120 = 12.0).
    assert rank([candidate("sm_89", None, "x")], sm="sm_89")
    assert not rank([candidate("sm_89", None, "x")], sm="sm_86")
    assert not rank([candidate("sm_120", None, "x")], sm="sm_89")
    # Deterministic tie-break on digest.
    tie = rank([candidate("sm_86", None, "z"), candidate("sm_86", None, "a")], sm="sm_89")
    assert [row.digest for row in tie] == ["a", "z"]
