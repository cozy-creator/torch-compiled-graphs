"""Local-first compile, store, resolve, and admission lifecycle."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from hashrepo import LocalCAS

from .artifact import build_metadata, pack_artifact
from .compiler import Compiler, Packager, compile_exported_program, package_compiled_files
from .declaration import GraphDeclaration, GraphSpec, RuntimeCompatibility
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
class GraphPlan:
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
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - package dependency
        raise AdmissionError("safetensors is required for literal graph constants") from exc
    tensors = {
        constant.fqn: _literal_value(program, constant).detach().cpu().contiguous()
        for constant in literals
    }
    save_file(tensors, str(target))


class Engine:
    """One local compiled-graph engine backed exclusively by HashRepo."""

    def __init__(self, cas: LocalCAS) -> None:
        self._store = _GraphStore(cas)

    def plan(self, spec: GraphSpec, runtime: RuntimeCompatibility) -> GraphPlan:
        declaration = spec.declare()
        return GraphPlan(spec, declaration, runtime, runtime.key(declaration))

    @staticmethod
    def _admit(plan: GraphPlan, graph: StoredGraph) -> None:
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
        if metadata.get("cell_key") != str(plan.key):
            mismatches.append("cell_key")
        if mismatches:
            raise AdmissionError(
                "artifact does not satisfy the exact graph plan: " + ", ".join(mismatches)
            )

    def resolve(self, plan: GraphPlan, destination: str | Path) -> StoredGraph | None:
        graph = self._store.resolve(plan.key, destination)
        if graph is None:
            return None
        try:
            self._admit(plan, graph)
        except AdmissionError:
            shutil.rmtree(graph.directory, ignore_errors=True)
            self._store.quarantine(plan.key, graph.manifest)
            raise
        return graph

    def mint(
        self,
        plan: GraphPlan,
        *,
        options: Mapping[str, object] | None = None,
        compiler: Compiler | None = None,
        packager: Packager | None = None,
        before_compile: Callable[[], None] | None = None,
        context: AbstractContextManager[None] | None = None,
    ) -> StoreResult:
        """Compile, package, verify, and publish one plan into the local CAS."""

        with tempfile.TemporaryDirectory(prefix="torch-compiled-graphs-mint-") as raw:
            workspace = Path(raw)
            files = compile_exported_program(
                plan.spec.program,
                options=options,
                compiler=compiler,
                before_compile=before_compile,
                context=context,
            )
            package = package_compiled_files(
                plan.declaration.entry,
                files,
                workspace / "model.pt2",
                packager=packager,
            )
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
        spec: GraphSpec,
        runtime: RuntimeCompatibility,
        destination: str | Path,
        *,
        options: Mapping[str, object] | None = None,
        compiler: Compiler | None = None,
        packager: Packager | None = None,
        before_compile: Callable[[], None] | None = None,
        context: AbstractContextManager[None] | None = None,
    ) -> EnsureResult:
        """Reuse an admitted exact key, compiling only on a miss or quarantine."""

        plan = self.plan(spec, runtime)
        try:
            existing = self.resolve(plan, destination)
        except (AdmissionError, QuarantinedArtifact):
            existing = None
        if existing is not None:
            return EnsureResult(EnsureOutcome.REUSED, existing)

        publication = self.mint(
            plan,
            options=options,
            compiler=compiler,
            packager=packager,
            before_compile=before_compile,
            context=context,
        )
        resolved = self.resolve(plan, destination)
        if resolved is None:
            raise StorageError(f"compiled graph {plan.key} disappeared after publication")
        outcome = (
            EnsureOutcome.REUSED
            if publication.outcome == StoreOutcome.DIVERGENT
            else EnsureOutcome.MINTED
        )
        return EnsureResult(outcome, resolved, publication.outcome)
