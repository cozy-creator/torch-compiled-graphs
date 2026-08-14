"""Local-first compile, store, resolve, and admission lifecycle."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from hashrepo import LocalCAS

from .artifact import build_metadata, pack_artifact, read_metadata
from .compiler import compile_exported_program, package_compiled_files
from .declaration import GraphDeclaration, GraphSpec, RuntimeCompatibility
from .host_isa import HostISAError, _admit_host
from .identity import CompiledGraphKey
from .introspection import DeclaredConstant, declared_constants
from .storage import (
    QuarantinedArtifact,
    StorageError,
    StoredGraph,
    StoreOutcome,
    StoreResult,
    _GraphStore,
)


class AdmissionError(StorageError):
    """Materialized bytes do not exactly satisfy their requested graph plan."""


class EnsureOutcome(str, Enum):
    REUSED = "reused"
    MINTED = "minted"


@dataclass(frozen=True, slots=True)
class _GraphPlan:
    """The one declaration and exact key used by both resolve and mint."""

    spec: GraphSpec
    declaration: GraphDeclaration
    runtime: RuntimeCompatibility
    key: CompiledGraphKey


@dataclass(frozen=True, slots=True)
class EnsureResult:
    outcome: EnsureOutcome
    graph: StoredGraph
    publication: StoreOutcome | None = None


def _literal_value(program: object, constant: DeclaredConstant) -> Any:
    values = getattr(program, "constants", {}) or {}
    if constant.fqn in values:
        return values[constant.fqn]
    if constant.name in values:
        return values[constant.name]
    raise AdmissionError(f"declared literal constant {constant.fqn!r} has no exported value")


def _write_literals(program: object, constants: tuple[DeclaredConstant, ...], target: Path) -> None:
    literals = [constant for constant in constants if constant.source == "literal"]
    if not literals:
        return
    try:
        from safetensors import TensorSpec, serialize_file
    except ImportError as exc:  # pragma: no cover - package dependency
        raise AdmissionError("safetensors is required for literal graph constants") from exc
    tensors = {
        constant.fqn: _literal_value(program, constant).detach().cpu().contiguous()
        for constant in literals
    }
    try:
        specs = {
            name: TensorSpec(
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=list(tensor.shape),
                data_ptr=tensor.data_ptr(),
                data_len=tensor.numel() * tensor.element_size(),
            )
            for name, tensor in tensors.items()
        }
        serialize_file(specs, str(target))
    except Exception as exc:
        raise AdmissionError(f"cannot serialize literal graph constants: {exc}") from exc


def _compile_package(plan: _GraphPlan, workspace: Path) -> Path:
    files = compile_exported_program(plan.spec.program)
    return package_compiled_files(
        plan.declaration.entry,
        files,
        workspace / "model.pt2",
    )


class Engine:
    """One local compiled-graph engine backed exclusively by HashRepo."""

    def __init__(self, cas: LocalCAS) -> None:
        self._store = _GraphStore(cas)

    @staticmethod
    def _plan(spec: GraphSpec, runtime: RuntimeCompatibility) -> _GraphPlan:
        declaration = spec.declare()
        return _GraphPlan(spec, declaration, runtime, runtime.key(declaration))

    @staticmethod
    def _admit_host_metadata(metadata: Mapping[str, object]) -> None:
        toolchain = metadata.get("toolchain")
        if not isinstance(toolchain, Mapping):
            raise AdmissionError("artifact records no toolchain")
        try:
            _admit_host(cast(Mapping[str, str], toolchain))
        except HostISAError as exc:
            raise AdmissionError(f"artifact host ISA is unsupported: {exc}") from exc

    @staticmethod
    def _admit(plan: _GraphPlan, graph: StoredGraph) -> None:
        metadata = graph.metadata
        entry = metadata.get("entry")
        if not isinstance(entry, Mapping):
            raise AdmissionError("artifact records no graph entry")
        expected_entry: dict[str, object] = {
            "name": plan.declaration.entry,
            "target": plan.declaration.target,
            "class_hash": plan.declaration.class_hash,
            "graph": plan.declaration.graph,
            "literal_values": plan.declaration.literal_values,
            "placement": list(plan.declaration.placement),
        }
        mismatches = [
            field for field, expected in expected_entry.items() if entry.get(field) != expected
        ]
        if metadata.get("sm") != plan.runtime.sm:
            mismatches.append("sm")
        if metadata.get("toolchain") != plan.runtime.toolchain:
            mismatches.append("toolchain")
        if metadata.get("compiled_graph_key") != str(plan.key):
            mismatches.append("compiled_graph_key")
        if mismatches:
            raise AdmissionError(
                "artifact does not satisfy the exact graph plan: " + ", ".join(mismatches)
            )

    def resolve(self, key: str | CompiledGraphKey, destination: str | Path) -> StoredGraph | None:
        """Resolve and verify an exact key without importing Torch or building a program."""

        graph = self._store.resolve(key, destination)
        if graph is None:
            return None
        self._admit_host_metadata(graph.metadata)
        return graph

    @staticmethod
    def _admit_request(graph: StoredGraph, *, target: str, deployment_compatibility: str) -> None:
        requested_target = str(target).strip().lower()
        requested_deployment = str(deployment_compatibility).strip()
        if not requested_target or not requested_deployment:
            raise AdmissionError("ensure requires target and deployment_compatibility")
        stored_target = graph.metadata.get("sm")
        target_matches = stored_target == requested_target or (
            requested_target == "cpu"
            and isinstance(stored_target, str)
            and stored_target.startswith("cpu-")
        )
        if not target_matches:
            raise AdmissionError(
                f"stored target {stored_target!r} does not match requested {requested_target!r}"
            )
        toolchain = graph.metadata.get("toolchain")
        stored_deployment = (
            toolchain.get("deployment_compatibility") if isinstance(toolchain, Mapping) else None
        )
        if stored_deployment != requested_deployment:
            raise AdmissionError(
                "stored deployment_compatibility "
                f"{stored_deployment!r} does not match requested {requested_deployment!r}"
            )

    def import_artifact(self, key: str | CompiledGraphKey, artifact: str | Path) -> StoreResult:
        """Fully verify and attach bytes fetched by HashRepo under one exact key."""

        self._admit_host_metadata(read_metadata(artifact))
        return self._store.store(key, artifact)

    def export_artifact(self, key: str | CompiledGraphKey, destination: str | Path) -> Path:
        """Export a fully verified artifact without exposing HashRepo ref layout."""

        return self._store.export_artifact(key, destination)

    def _mint(self, plan: _GraphPlan) -> StoreResult:
        """Compile, package, verify, and publish one plan into the local CAS."""

        with tempfile.TemporaryDirectory(prefix="torch-compiled-graphs-mint-") as raw:
            workspace = Path(raw)
            package = _compile_package(plan, workspace)
            if plan.spec.declare() != plan.declaration:
                raise AdmissionError("exported program changed during compilation or packaging")
            constants = declared_constants(package, plan.declaration.entry)
            literals = workspace / "constants.safetensors"
            _write_literals(plan.spec.program, constants, literals)
            metadata = build_metadata(
                entry={
                    "name": plan.declaration.entry,
                    "target": plan.declaration.target,
                    "class_hash": plan.declaration.class_hash,
                    "graph": plan.declaration.graph,
                    "literal_values": plan.declaration.literal_values,
                    "placement": list(plan.declaration.placement),
                    "constants": [constant.as_manifest_row() for constant in constants],
                },
                sm=plan.runtime.sm,
                toolchain=plan.runtime.toolchain,
            )
            artifact = pack_artifact(
                package,
                workspace / "compiled-graph.tar.gz",
                metadata,
                literals=literals if literals.is_file() else None,
            )
            return self._store.store(plan.key, artifact)

    def ensure(
        self,
        key: str | CompiledGraphKey,
        destination: str | Path,
        *,
        target: str,
        deployment_compatibility: str,
        recipe: Callable[[], GraphSpec],
    ) -> EnsureResult:
        """Reuse an admitted exact key, compiling only on a miss or quarantine."""

        requested = str(key)
        try:
            existing = self.resolve(requested, destination)
        except QuarantinedArtifact:
            existing = None
        if existing is not None:
            self._admit_request(
                existing,
                target=target,
                deployment_compatibility=deployment_compatibility,
            )
            return EnsureResult(EnsureOutcome.REUSED, existing)

        spec = recipe()
        runtime = RuntimeCompatibility(target, deployment_compatibility=deployment_compatibility)
        plan = self._plan(spec, runtime)
        if str(plan.key) != requested:
            raise AdmissionError(f"lazy recipe derives {plan.key}, not requested key {requested!r}")
        publication = self._mint(plan)
        resolved = self.resolve(plan.key, destination)
        if resolved is None:
            raise StorageError(f"compiled graph {plan.key} disappeared after publication")
        self._admit(plan, resolved)
        outcome = (
            EnsureOutcome.REUSED
            if publication.outcome == StoreOutcome.DIVERGENT
            else EnsureOutcome.MINTED
        )
        return EnsureResult(outcome, resolved, publication.outcome)
