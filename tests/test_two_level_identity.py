"""Two-level identity: the compile-stack env key, manifests as audit, the ladder."""

from __future__ import annotations

import pytest

from torchcg.graph_identity import (
    ENV_SCHEME,
    EnvIdentity,
    GraphIdentityError,
    compile_stack,
    is_compile_relevant,
)
from torchcg.requirements import (
    ArtifactCandidate,
    EnvironmentMismatch,
    RequirementsError,
    RequirementsManifest,
    assert_exact_env,
    rank,
)

STACK = (("nvidia-cublas-cu12", "12.8.4"), ("torch", "2.13.0"))

#: One endpoint lockfile's rows: the compile stack plus everything that
#: cannot reach the compiler. pgw#1472 measured 43-package diffs between envs
#: that serve identically; these are that shape, minimized.
LOCK_ROWS = {
    "torch": "2.13.0",
    "nvidia-cublas-cu12": "12.8.4",
    "triton": "3.6.0",
    "pillow": "11.1.0",
    "colorama": "0.4.6",
    "diffusers": "0.36.0",
}


def test_compile_stack_selects_the_compiler_and_nothing_else() -> None:
    assert compile_stack(LOCK_ROWS) == (
        ("nvidia-cublas-cu12", "12.8.4"),
        ("torch", "2.13.0"),
        ("triton", "3.6.0"),
    )
    # THE pgw#1489 PROPERTY: irrelevant drift does not move the key.
    drifted = {**LOCK_ROWS, "pillow": "12.0.0", "colorama": "0.4.7"}
    del drifted["diffusers"]
    drifted["rich"] = "14.0.0"
    assert compile_stack(drifted) == compile_stack(LOCK_ROWS)
    assert EnvIdentity(stack=compile_stack(drifted), sm="sm_89").value == (
        EnvIdentity(stack=compile_stack(LOCK_ROWS), sm="sm_89").value
    )


def test_compile_stack_moves_when_the_compiler_moves() -> None:
    bumped = {**LOCK_ROWS, "torch": "2.14.0"}
    assert compile_stack(bumped) != compile_stack(LOCK_ROWS)
    nvidia = {**LOCK_ROWS, "nvidia-cublas-cu12": "12.9.0"}
    assert compile_stack(nvidia) != compile_stack(LOCK_ROWS)


def test_is_compile_relevant_spelling() -> None:
    assert is_compile_relevant("nvidia_cuda_runtime_cu12")
    assert is_compile_relevant("Torch")
    assert not is_compile_relevant("torchvision")
    assert not is_compile_relevant("nvidiablah")


def test_env_identity_refuses_a_whole_installed_set() -> None:
    """The regression guard: the key cannot be handed the closure again."""

    with pytest.raises(GraphIdentityError, match="not a compile-stack package"):
        EnvIdentity(stack={**LOCK_ROWS}, sm="sm_89")


def test_env_identity_is_deterministic_and_axis_sensitive() -> None:
    a = EnvIdentity(stack=STACK, sm="sm_89")
    assert a.value == EnvIdentity(stack=dict(STACK), sm="sm_89").value
    assert a.value.startswith(ENV_SCHEME + "-")
    assert a.value != EnvIdentity(stack=STACK, sm="sm_86").value
    assert a.value != EnvIdentity(stack=(("torch", "2.14.0"),), sm="sm_89").value
    assert "torch 2.13.0" in a.describe() and "sm_89" in a.describe()


def test_env_identity_refuses_noncanonical_axes() -> None:
    with pytest.raises(GraphIdentityError, match="states torch"):
        EnvIdentity(stack=(("triton", "3.6.0"),), sm="sm_89")
    with pytest.raises(GraphIdentityError, match="states no version"):
        EnvIdentity(stack=(("torch", ""),), sm="sm_89")
    with pytest.raises(GraphIdentityError):
        EnvIdentity(stack=STACK, sm="whatever")


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


def test_exact_env_audit_is_exact_and_names_the_package() -> None:
    stamped = EnvIdentity(stack=STACK, sm="sm_89")
    assert_exact_env(stamped, stack=dict(STACK), sm="sm_89")
    # Irrelevant drift is not a divergence any more: the same lock rows plus a
    # different set of non-compile packages select the same stack.
    assert_exact_env(
        stamped,
        stack=compile_stack({**dict(STACK), "pillow": "12.0.0", "rich": "14.0.0"}),
        sm="sm_89",
    )
    with pytest.raises(EnvironmentMismatch, match="sm sm_86 != stamped sm_89"):
        assert_exact_env(stamped, stack=dict(STACK), sm="sm_86")
    with pytest.raises(EnvironmentMismatch, match=r"torch 2.14.0 != stamped 2.13.0"):
        assert_exact_env(
            stamped, stack={**dict(STACK), "torch": "2.14.0"}, sm="sm_89"
        )


GRAPH = "cg-graph-v1-" + "a" * 56


def candidate(sm_compiled: str, autotuned_on: str | None, digest: str) -> ArtifactCandidate:
    return ArtifactCandidate(
        graph=GRAPH,
        env=EnvIdentity(stack=STACK, sm=sm_compiled),
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
