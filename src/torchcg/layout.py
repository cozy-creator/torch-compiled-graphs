"""Layout morphisms -- the byte layout an artifact DECLARES for its inputs.

tcg#83, the torchcg half of ``research/layout-morphisms/0-DESIGN.md``. The
compiler decides a byte layout for every tensor it touches; today the platform
delivers row-major bytes and inductor re-derives its preferred layout on every
forward call (sdxl UNet: 40 Triton permute launches, 1.19 GiB/step of DRAM
traffic, measured in tcg#82). The fix is a CONTRACT: the artifact declares the
layout it was compiled against, the key carries that declaration, and the
loader delivers bytes in it.

**THIS MODULE PRODUCES NO VOCABULARY** (tcg#87). tcg#83 shipped before
:mod:`tensorfs`' layout catalog existed, so it authored the closed inductor
class here as three literals -- and the two copies immediately disagreed on two
of the three handles. A layout handle is a KEY AXIS VALUE, so two producers of
it is tcg#80's disease one level up. There is now exactly one producer, the
ratified records under ``spec/v2/layouts``, and this module holds no handle, no
rank and no permutation of its own.

What it DOES hold is a CAPABILITY, which is a different kind of fact: which of
torch's own memory formats this compiler can deliver. A record is paired with a
memory format only when the permutation torch produces at that rank EQUALS the
permutation the record states, computed both ways rather than asserted -- so a
record that moves and a torch that changes are both caught, and neither can be
silently ignored.

A ratified record with no torch memory format behind it -- ``torch.transposed``,
``torch.stride-padding-16``, and the whole endpoint-declared /
kernel-library class where the real wins are (``bfl.nvfp4-preswizzled`` vs
``cozy.nvfp4-flat``) -- is UNDELIVERABLE BY THIS COMPILER, which is a different
refusal from unratified, because the remedies differ: ratify it, versus deliver
it through the tensorfs fill path. A layout nobody ratified is never invented
here: it is a CANDIDATE, emitted for ratification, and the mint falls back to
the identity (:mod:`torchcg.compiler`).

**The corpus is the CALLER'S to supply**, as tensorfs states: the wheel carries
no corpus, so a consumer vendors one. In production torchcg is vendored beside a
vendored tensorfs and the records are found automatically. Absent, every call
here refuses BY NAME -- never an empty catalog, which reads as "nothing is
ratified" and whose next move is a silent fall back to row-major.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .lane import _HANDLE_RE

#: Where the ratified corpus is, when it is not simply beside the installed
#: `tensorfs`. A PATH, which is configuration -- never a switch: setting it
#: cannot change what any code DOES, only which directory the one producer's
#: records are read from. tensorfs' wheel deliberately carries no corpus, so a
#: consumer that is not vendoring one (torchcg's own suite, a bare checkout)
#: needs some way to say where it put it, and inventing a second default
#: location inside this file would make this module an author again.
CORPUS_ENV = "TENSORFS_SPEC_V2"


class LayoutError(ValueError):
    """A layout declaration or wish is not a ratified catalog morphism."""


class LayoutUndeliverableError(LayoutError):
    """The layout IS ratified -- this compiler cannot deliver it.

    Deliberately distinct from the unratified case. "Nobody has ratified this
    arrangement" is answered by writing a record; "torch has no memory format
    that produces this arrangement" is answered by the tensorfs fill path, which
    applies the morphism in flight on the way to the device. Collapsing the two
    into one message sends a reader to the wrong remedy.
    """


class LayoutCorpusError(LayoutError):
    """No ratified layout corpus is reachable, so nothing can be resolved."""


#: The memory formats THIS COMPILER can deliver. A capability of torch, not a
#: vocabulary: no handle, no rank, no permutation is stated here. Which ratified
#: record each one IS gets computed, by comparing permutations.
_DELIVERABLE_FORMATS: tuple[str, ...] = (
    "contiguous_format",
    "channels_last",
    "channels_last_3d",
)


@dataclass(frozen=True, slots=True)
class LayoutMorphism:
    """One ratified arrangement this compiler can deliver.

    ``handle`` and ``rank`` are the RECORD's, read from the corpus.
    ``memory_format`` is the name of the ``torch.memory_format`` that was proven
    to produce that record's permutation -- the pairing, not a second statement
    of the arrangement.

    A morphism that does not apply to a tensor's rank leaves it row-major: a 1-D
    bias under a ``channels_last`` declaration is contiguous, not an error,
    because the permutation is the identity there.
    """

    handle: str
    memory_format: str
    rank: int | None

    def __post_init__(self) -> None:
        if _HANDLE_RE.fullmatch(self.handle) is None:
            raise LayoutError(
                f"a layout morphism handle is '<producer>.<format>@<version>'; "
                f"got {self.handle!r}"
            )

    def format(self) -> Any:
        import torch

        return getattr(torch, self.memory_format)

    def applies_to(self, rank: int) -> bool:
        return self.rank is None or self.rank == rank

    def matches(self, tensor: Any) -> bool:
        """Whether these bytes are ALREADY in this layout -- no copy needed.

        This is the predicate the whole design turns on: inductor's
        ``require_strides`` emits no copy when the strides already match
        (torch ``_inductor/ir.py``), so a tree delivered in the declared layout
        costs zero permute kernels. Read as a question about the tensor, never
        as an instruction to transform it.
        """

        import torch

        rank = tensor.dim()
        if not self.applies_to(rank):
            return bool(tensor.is_contiguous(memory_format=torch.contiguous_format))
        return bool(tensor.is_contiguous(memory_format=self.format()))

    def describe(self, tensor: Any) -> str:
        return (
            f"shape {tuple(tensor.shape)} strides {tuple(tensor.stride())} "
            f"(declared {self.handle})"
        )

    def stride_order(self, rank: int) -> tuple[int, ...] | None:
        """This morphism's stride order at one rank, computed FROM torch."""

        if not self.applies_to(rank):
            return None
        return _probe_stride_order(self.memory_format, rank)


def _resolve(root: Path | None) -> Path | None:
    """The corpus root a call will actually read.

    Resolved BEFORE the cache, never inside it: caching on the argument while
    reading the environment underneath would make the first call's environment
    the answer forever, which is the kind of guard that cannot go red.
    """

    if root is not None:
        return root
    stated = os.environ.get(CORPUS_ENV)
    return Path(stated) if stated else None


def _corpus(root: Path | None) -> dict[str, Any]:
    """The ratified records, or a refusal naming why there are none."""

    try:
        from tensorfs import layouts
    except ImportError as exc:  # pragma: no cover - tensorfs is a hard dependency
        raise LayoutCorpusError(
            f"the ratified layout corpus is tensorfs' (spec/v2/layouts) and "
            f"tensorfs is not importable: {exc}"
        ) from exc
    try:
        return dict(layouts(root))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise LayoutCorpusError(
            f"no ratified layout corpus is reachable, so no layout can be "
            f"resolved: {exc}. tensorfs' wheel deliberately carries no corpus; "
            f"a consumer vendors one beside the package, passes root=, or "
            f"names one in ${CORPUS_ENV}"
        ) from exc


@cache
def _paired(root: Path | None = None) -> tuple[
    dict[str, LayoutMorphism], tuple[str, ...]
]:
    """(deliverable morphisms, ratified-but-undeliverable handles).

    The pairing is arithmetic and runs in BOTH directions: every deliverable
    format must find exactly one record, and every record is deliverable only if
    some format reproduces its permutation. So a record whose permutation moves
    stops pairing, and a torch whose memory format changes stops pairing, rather
    than either being silently absorbed.
    """

    records = _corpus(root)
    deliverable: dict[str, LayoutMorphism] = {}
    for arrangement in records.values():
        memory_format = _format_producing(arrangement)
        if memory_format is None:
            continue
        deliverable[arrangement.handle] = LayoutMorphism(
            arrangement.handle, memory_format, arrangement.rank
        )
    undeliverable = tuple(
        sorted(handle for handle in records if handle not in deliverable)
    )
    return deliverable, undeliverable


def _format_producing(arrangement: Any) -> str | None:
    """The torch memory format whose permutation IS this record's, or None.

    The identity record is rankless and permutes nothing, which no probe can
    express as a permutation -- it is matched on those two properties instead,
    and it is the only record for which that is true.
    """

    if arrangement.rank is None:
        return "contiguous_format" if arrangement.is_identity else None
    if arrangement.is_identity:
        return None
    wanted = tuple(arrangement.permutation)
    # A memory format PERMUTES axes; it never SPLITS one. A record that
    # factors an axis (`torch.stride-padding-16@1` cuts axis 1 into
    # ceil(d1/16) x 16, the kernel-library class routinely cuts into five or
    # seven) is arithmetic no `memory_format` can express, and comparing a
    # rank-length probe against a longer permutation would answer "no" for the
    # right reason by accident. Say it on purpose.
    if len(wanted) != arrangement.rank or len(arrangement.sub_axes) != arrangement.rank:
        return None
    for memory_format in _DELIVERABLE_FORMATS:
        if _probe_permutation(memory_format, arrangement.rank) == wanted:
            return memory_format
    return None


def catalog(root: Path | None = None) -> dict[str, LayoutMorphism]:
    """handle -> morphism, for every ratified record this compiler can deliver."""

    return dict(_paired(_resolve(root))[0])


def contiguous_handle(root: Path | None = None) -> str:
    """The identity handle, READ from the corpus rather than spelled.

    It is the permanent fallback the design forbids removing: a wish outside the
    catalog compiles against this. Deriving it is the point -- spelling it would
    make this module a second author of the one handle every artifact that
    predates the layout axis carries.
    """

    from tensorfs import identity_arrangement

    try:
        return str(identity_arrangement(_resolve(root)).handle)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise LayoutCorpusError(
            f"the identity layout is the corpus' unique rankless record and it "
            f"could not be read: {exc}"
        ) from exc


def require_morphism(handle: object, root: Path | None = None) -> LayoutMorphism:
    """Resolve one declared layout, or refuse naming which kind of miss it is.

    An unclassified layout value is a refusal, never a coercion to row-major:
    a declaration nobody can verify would let an artifact claim a layout its
    loader silently ignores, which is the defect this axis exists to remove.
    """

    deliverable, undeliverable = _paired(_resolve(root))
    if isinstance(handle, str) and handle in deliverable:
        return deliverable[handle]
    if isinstance(handle, str) and handle in undeliverable:
        raise LayoutUndeliverableError(
            f"layout morphism {handle!r} IS ratified, and this compiler cannot "
            f"deliver it: no torch memory format produces its arrangement. It "
            f"reaches a tensor through the tensorfs fill path, which applies "
            f"the morphism in flight (tensorfs#154) -- not by being declared "
            f"here. Deliverable: {sorted(deliverable)!r}"
        )
    raise LayoutError(
        f"declared input layout {handle!r} is not a ratified layout morphism; "
        f"the ratified catalog is "
        f"{sorted([*deliverable, *undeliverable])!r}. Machines derive along "
        f"ratified morphisms and never invent one: an unknown layout is a "
        f"CANDIDATE for ratification, and the mint falls back to the identity"
    )


def is_catalog_handle(value: object, root: Path | None = None) -> bool:
    return isinstance(value, str) and value in _paired(_resolve(root))[0]


@cache
def _probe_stride_order(memory_format: str, rank: int) -> tuple[int, ...]:
    """The stride order one memory format produces at one rank, per torch.

    Read rather than written: a meta-device probe with pairwise-distinct
    dimensions is materialized in the format, and its strides are converted by
    inductor's OWN ``get_stride_order``. That is the same conversion inductor
    performs when it decides what layout it wants, so the two cannot drift.
    """

    import torch
    from torch._inductor.ir import get_stride_order

    size = tuple(range(2, 2 + rank))
    probe = torch.empty(size, device="meta").to(
        memory_format=getattr(torch, memory_format)
    )
    return tuple(int(value) for value in get_stride_order(probe.stride()))


@cache
def _probe_permutation(memory_format: str, rank: int) -> tuple[int, ...] | None:
    """Storage order over logical axes, outermost first -- the RECORD's spelling.

    The same probe :func:`_probe_stride_order` takes, converted into the
    vocabulary the corpus writes in: the axes sorted by descending stride. The
    two spellings are the same fact, and pairing compares in this one so the
    comparison is against what the record actually says rather than against a
    transformation of it.

    ``None`` when torch declines the format at that rank (``channels_last``
    requires rank 4) -- a NEGATIVE ANSWER, not a swallowed error: that format
    does not produce any arrangement there, which is exactly what the pairing
    needs to know.
    """

    import torch

    size = tuple(range(2, 2 + rank))
    try:
        probe = torch.empty(size, device="meta").to(
            memory_format=getattr(torch, memory_format)
        )
    except RuntimeError:
        return None
    strides = [int(value) for value in probe.stride()]
    return tuple(sorted(range(rank), key=lambda axis: -strides[axis]))


def morphism_for_stride_order(
    order: Sequence[int], root: Path | None = None
) -> LayoutMorphism | None:
    """The catalog morphism one stride order names, or None (a CANDIDATE).

    ``None`` is the design's permanent fallback path, not a failure: the mint
    compiles against the stored layout and emits the order as an unratified
    candidate. Never invent a name for it.
    """

    wanted = tuple(int(value) for value in order)
    for morphism in _paired(_resolve(root))[0].values():
        if morphism.stride_order(len(wanted)) == wanted:
            return morphism
    return None


__all__ = [
    "CORPUS_ENV",
    "LayoutCorpusError",
    "LayoutError",
    "LayoutMorphism",
    "LayoutUndeliverableError",
    "catalog",
    "contiguous_handle",
    "is_catalog_handle",
    "morphism_for_stride_order",
    "require_morphism",
]
