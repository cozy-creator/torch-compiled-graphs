"""Every refusal this library can raise, in one place.

A refusal names what went wrong and what the remedy is. There is no generic
error: a caller that cannot tell "nobody ratified this layout" from "torch
cannot deliver this layout" is sent to the wrong fix.
"""

from __future__ import annotations


class TorchCGError(Exception):
    """Root of every refusal."""


class IdentityError(TorchCGError):
    """A graph, ingress or environment cannot state its canonical identity."""


class IngressError(IdentityError):
    """A call-ingress declaration is incomplete or noncanonical."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = str(reason)
        super().__init__(detail if detail is not None else reason)


class LayoutError(IdentityError):
    """A declared layout is not a ratified morphism."""


class LayoutUndeliverableError(LayoutError):
    """The layout IS ratified; no torch memory format produces it.

    Distinct from unratified on purpose. "Nobody ratified this arrangement" is
    answered by writing a record; "torch has no memory format for it" is
    answered by the tensorfs fill path, which applies the morphism in flight.
    """


class LayoutCorpusError(LayoutError):
    """No ratified layout corpus is reachable, so nothing can be resolved."""


class MintError(TorchCGError):
    """A graph could not be exported, bound, compiled or packaged."""


class BindError(MintError):
    """Binding the symbolic parent does not re-derive the requested graph."""


class DroppedOptimization(MintError):
    """The target will SILENTLY DROP a requested compile option.

    Minting anyway produces an artifact that keys as if the lever ran and
    compiles as if it did not, so the fleet re-mints for nothing (tcg#85).
    """


class RangeNarrowed(MintError):
    """The compile narrowed a declared symbol range (tcg#78).

    Every dynamic dim is guarded ``>= 2`` -- torch specializes 0 and 1 rather
    than reason about broadcasting symbolically -- so an axis whose observed
    minimum is 1 is contradicted the moment it is exported. The artifact then
    cannot serve every shape its declaration admits while the dispatcher still
    guards on the declaration: an sd15 UNet exported with a batch ``Dim`` over
    ``[1, 2]`` returns a ``(2, 4, 64, 64)`` tensor of garbage at batch 1 and
    raises nothing.
    """


class StoreError(TorchCGError):
    """A stored artifact is malformed, unreadable or unavailable."""


class DivergentArtifact(StoreError):
    """One key already holds different bytes (mint-once, first-writer-wins).

    tcg#84. The key IS the artifact's content address, so two different byte
    strings under one key means an axis the key does not carry decided the
    output -- never something to overwrite.
    """


class AdoptError(TorchCGError):
    """A stored artifact cannot be adopted for this call."""


class KeyMismatch(AdoptError):
    """The artifact's stated identity is not the key it was fetched under."""


__all__ = [
    "AdoptError",
    "BindError",
    "DivergentArtifact",
    "DroppedOptimization",
    "IdentityError",
    "IngressError",
    "KeyMismatch",
    "LayoutCorpusError",
    "LayoutError",
    "LayoutUndeliverableError",
    "MintError",
    "RangeNarrowed",
    "StoreError",
    "TorchCGError",
]
