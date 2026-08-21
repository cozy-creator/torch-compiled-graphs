"""tcg#86: the constants that DECIDE now derive or justify, and each fact has
ONE producer.

Every value below is unchanged by that work — this file exists so that stays
true, and so a future edit that re-picks one of them has to argue with a test
instead of with a comment. The red arms are the derivations: break the link
between a bound and what it is derived FROM and the equality stops holding.
"""

from __future__ import annotations

import pytest

import torchcg
from torchcg import _run_impl_split as runimpl
from torchcg import _wrapper_split as wrapper
from torchcg import artifact, identity, quantize, selection, spans, storage, store


def test_the_key_length_bound_DERIVES_from_the_key_it_admits() -> None:
    """66 of the 96 bytes are the package's own key; the other 30 are the
    foreign-scheme allowance the grammar needs and never named."""
    assert identity._OWN_KEY_LENGTH == len(f"{identity.KEY_SCHEME}-") + identity._DIGEST_HEX
    assert identity.MAX_KEY_LENGTH == identity._OWN_KEY_LENGTH + identity._FOREIGN_SCHEME_ALLOWANCE
    assert identity.MAX_KEY_LENGTH == 96, "the bound must not move silently"


def test_a_key_this_package_MINTS_can_never_exceed_its_own_admission_bound() -> None:
    """The property the derivation buys: widen the digest and the bound
    follows, instead of the package refusing keys it just made."""
    assert callable(identity.from_axes)
    assert identity._OWN_KEY_LENGTH <= identity.MAX_KEY_LENGTH
    assert identity._FOREIGN_SCHEME_ALLOWANCE >= 0


def test_the_ledger_has_ONE_precision_and_three_things_follow_from_it() -> None:
    """`round(, 3)`, the drop floor and the closure tolerance were three
    independent literals that happened to agree."""
    assert spans.LEDGER_PRECISION == 3
    assert spans._DROP_FLOOR_S == 10.0 ** -spans.LEDGER_PRECISION / 2 == 0.0005
    assert spans._closure_tolerance_s() == 0.05, (
        "the tolerance must not move: it is rounding-bound + declared slack, "
        "and no banked ledger has been measured against a tighter one"
    )
    assert spans._rounding_bound_s() < spans._closure_tolerance_s(), (
        "the declared slack is the part that is UNMEASURED; if it ever reaches "
        "zero the tolerance became a derivation and this comment is wrong"
    )


def test_the_closure_tolerance_GROWS_with_the_widest_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flat literal could not do this: add members to a partition and the
    rounding a closure check must forgive grows with it."""
    before = spans._closure_tolerance_s()
    widened = dict(spans.PARTITIONS)
    key = next(iter(widened))
    widened[key] = widened[key] + tuple(f"_probe_{i}_s" for i in range(20))
    monkeypatch.setattr(spans, "PARTITIONS", widened)
    assert spans._closure_tolerance_s() > before


def test_the_one_MiB_io_buffer_has_ONE_producer() -> None:
    """It lived under three names in three files."""
    assert artifact.IO_BUFFER_BYTES == 1 << 20
    assert storage._READ_BUFFER is artifact.IO_BUFFER_BYTES
    assert store._READ_BUFFER is artifact.IO_BUFFER_BYTES


def test_the_two_8_MiB_caps_are_SEPARATE_policies_that_may_diverge() -> None:
    """Same value today, unrelated subjects — metadata.json's size against a
    safetensors header length field. They are declared apart so tuning one
    cannot move the other."""
    assert artifact._MAX_METADATA_BYTES == artifact._MAX_SAFETENSORS_HEADER_BYTES == 8 << 20


def test_the_decline_gates_DERIVE_from_the_shapes_they_are_about() -> None:
    """Both justified their existence and never their value; the run_impl gate
    additionally looked fitted to its one observation (4231)."""
    assert wrapper.MIN_STATEMENTS == 8 * wrapper.CHUNK == 2000
    assert runimpl.MIN_COMPUTE_LINES == runimpl.DEFAULT_PARTS * 500 == 4000


def test_the_two_SIXTEENS_are_different_quantities_and_stay_apart() -> None:
    """`ModuleSelect.align` is 16 ELEMENTS of feature alignment for the scaled
    GEMM; `AOTI_ALIGNMENT` is 16 BYTES of address alignment for an input
    pointer. A future edit that unifies them on the strength of the shared
    literal is the defect this asserts against."""
    assert quantize.ModuleSelect().align == 16
    assert selection.AOTI_ALIGNMENT == 16
    assert torchcg is not None
