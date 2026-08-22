"""Mint, store, and adopt PyTorch AOTInductor graphs.

Five modules, and each answers one question:

* :mod:`torchcg.identity` -- what a compiled graph IS. One graph hash, one key.
* :mod:`torchcg.mint`     -- making the artifact: bind, compile, package.
* :mod:`torchcg.store`    -- where it lives: content-addressed, mint-once.
* :mod:`torchcg.adopt`    -- serving it: load, verify, dispatch-or-eager.
* :mod:`torchcg.refuse`   -- every way this can say no.

Torch does the compiling. What this library owns is identity, storage and
adoption -- and the refusals that keep an artifact from claiming to be
something it is not.
"""

from __future__ import annotations

from .adopt import Dispatcher, Recast, Record, adopt, fit, load, release
from .identity import (
    GRAPH_SCHEME,
    KEY_SCHEME,
    ArtifactKey,
    CallIngress,
    CallInput,
    LayoutMorphism,
    artifact_key,
    build_call_ingress,
    canonical_graph,
    contiguous_handle,
    graph_hash,
    is_artifact_key,
    is_graph_hash,
    require_morphism,
)
from .mint import GraphSpec, bind_static_spec, compile_policy, mint, strip_diagnostics
from .refuse import (
    AdoptError,
    BindError,
    DivergentArtifact,
    DroppedOptimization,
    IdentityError,
    IngressError,
    KeyMismatch,
    LayoutError,
    LayoutUndeliverableError,
    MintError,
    RangeNarrowed,
    StoreError,
    TorchCGError,
)
from .store import Store, StoredArtifact

__all__ = [
    "GRAPH_SCHEME",
    "KEY_SCHEME",
    "AdoptError",
    "ArtifactKey",
    "BindError",
    "CallIngress",
    "CallInput",
    "Dispatcher",
    "DivergentArtifact",
    "DroppedOptimization",
    "GraphSpec",
    "IdentityError",
    "IngressError",
    "KeyMismatch",
    "LayoutError",
    "LayoutMorphism",
    "LayoutUndeliverableError",
    "MintError",
    "RangeNarrowed",
    "Recast",
    "Record",
    "Store",
    "StoreError",
    "StoredArtifact",
    "TorchCGError",
    "adopt",
    "artifact_key",
    "bind_static_spec",
    "build_call_ingress",
    "canonical_graph",
    "compile_policy",
    "contiguous_handle",
    "fit",
    "graph_hash",
    "is_artifact_key",
    "is_graph_hash",
    "load",
    "mint",
    "release",
    "require_morphism",
    "strip_diagnostics",
]
