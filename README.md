# torchcg

`torchcg` mints and reuses verified PyTorch AOTInductor graphs. It is
independent of `python-gen-worker`: applications supply one exported program
plus the worker-recorded compiler-content block, and `tensorfs` is the sole
local content-addressed store.

Proven on GPU: a split-compiled artifact compiles, partial-links, loads and
executes on a real sm_86 pod with output bitwise-identical to the unsplit
compile and to eager (PR #33, `d7a0beb`).

## Install

Not on PyPI. Depend on it by git pin, at an exact rev:

```toml
# pyproject.toml -- resolves under uv, pip, and an exported requirements file
dependencies = [
    "torchcg @ git+https://github.com/cozy-creator/torchcg@<rev>",
]
```

Prefer that PEP 508 direct reference over a `[tool.uv.sources]` entry: a
`uv.sources` pin binds only to the project that declares it, so it is dropped
by `uv export`, by `pip install`, and by anything consuming your built wheel.
`torch` is an extra (`torchcg[torch]`), and resolving the lock builds `tensorfs`
from source, which needs a Rust toolchain on the resolving machine.

## V1 lifecycle

```python
from tensorfs import LocalCAS
from torchcg import (
    Engine,
    GraphClassSpec,
    RuntimeCompatibility,
    build_call_ingress,
)

ingress = build_call_ingress(
    exported_program,
    call_parameter_names,
    example_args,
    example_kwargs,
)
graph_interface["pytree"]["ingress"] = ingress.as_dict()

spec = GraphClassSpec(
    "denoiser/h=64,w=64",
    "unet",
    exported_program,
    graph=graph_interface,
)
runtime = RuntimeCompatibility(
    "sm_89",
    toolchain=recorded_toolchain,
)
engine = Engine(LocalCAS("/var/cache/graphs"))
result = engine.compile(
    spec,
    runtime,
    "/run/graphs/denoiser-h64-w64",
)
runner = engine.runner(result.compiled_graph.key, "/run/graphs/loaded")
assert runner is not None
runner.bind(resident_constants, device="cuda")
outputs = runner(*positional_inputs)
```

`graph_interface` carries the current v3 graph-class facts. Its required
`pytree.ingress` value is built and decoded only by this package; its digest is
the graph class's ingress-range digest rather than a caller-supplied second
identity. The same declaration derives the `cg-key-v1` lookup, mint stamp, and
admission expectation.

`Engine.compile(spec, runtime, destination)` is the sealed one-class operation
used by a compile child: it derives its own key and reuses an admitted exact
record before doing compiler work. A miss compiles one code-only graph under
the sole v1 compile policy, packages it as a verified artifact, stores it in
`tensorfs`, and materializes it. A later process pointed at the same store
reuses it without compiling. There is no public plan, compiler callback, option
map, or caller-supplied identity.

`Engine.resolve(key, destination)` is the known-key hit path: it neither
constructs an `ExportedProgram` nor imports Torch. It does read the local CPU
feature surface and fails before returning a package if the artifact's host-code
requirement is incomplete or unsupported. `Engine.ensure` resolves first and
invokes its lazy recipe exactly once, only on a miss.

`Engine.runner(key, destination)` resolves the exact selected bytes, loads the
named AOTI model, and returns a `CompiledGraphRunner`. The runner refuses every
call until `bind` proves the package's own constant table equals the artifact
manifest and completes one by-reference update. Missing, partial, or
out-of-memory binding leaves that runner permanently unusable; a caller may load
a fresh instance on a pod with capacity. Ingress selection, eager fallback,
family composition, process scheduling, Hub receipts, and telemetry remain
application policy and are intentionally absent.

## Compiler policy

The sealed policy uses four Inductor compile workers: pgw#757's measured
contention ceiling and the current worker value. A balanced 16-run
one-versus-four comparison found no material wall-time difference, so this is
parity and a ceiling, not a claim that four is faster.

The compiler privately rewrites the generated C++ wrapper's pathological
constructor and `run_impl` functions into reconstruction-checked smaller
functions and translation units. Unrecognized shapes compile unchanged. A failed
`run_impl` split retries the constructor-transformed monolith when that first
transform applied; if that build also fails, or the constructor transform was
the failure, the compiler retries PyTorch's original source. Neither transform
is a caller option, environment switch, public API, or identity axis.

## Toolchain identity

`recorded_toolchain` is an explicit adapter input, not a guessed version or host
fingerprint. A worker records its settings declaration, loaded-library digest,
installed Torch/Triton/NVIDIA wheel RECORD digests, and bundled CUDA binary
digests. This core applies the current worker membership rule and folds that
content to the 16-hex `toolchain` axis; trace-only `diffusers`, `transformers`,
and `peft` records are excluded because their effects already ride the graph.
Host ISA requirements are recorded and admitted separately.

**Every toolchain member must be a BUILD fact, never a HOST fact** (tcg#26).
The block's members are the caller's to choose, so this is a contract on the
caller: a member whose value moves when the same artifacts run on a different
machine fragments the cache by machine, and it does so on a dimension the ISA
axis deliberately clamps flat — the fingerprint then contradicts itself. Content
digests of installed files (wheel RECORDs, library bytes, tool binaries) satisfy
this by construction. Strings a runtime *probes* do not: `torch.__config__.show()`
is the measured example, since ATen interleaves build settings with the
dispatched CPU ISA and with accelerator blocks emitted only when a live driver
probe succeeds.

The measured portability matrix behind the retained axes lives in
`benchmarks/host_fingerprint/`, with the recorded rows beside `REQUIRED_AXES`.

## Storage boundary

`tensorfs` owns immutable objects, chunk manifests, durability, GC reachability,
and transfer primitives. This package owns only graph declaration, compilation,
artifact policy, exact-key refs, admission, and quarantine. A divergent second
artifact never overwrites an admitted key. A corrupt or inadmissible manifest is
quarantined and can be repaired only by a newly verified mint.

`Engine.export_artifact(key, path)` emits the exact verified artifact envelope
for a remote adapter. `Engine.import_artifact(key, path)` validates its metadata
and host requirement before attaching the fully verified bytes to the local
store. Neither operation exposes the package's private ref layout.

A package literal whose AOTI wrapper names no FQN lifted by the exported program
is refused rather than matched by tensor order or shape. Register durable module
tensors as buffers; guessing an anonymous constant mapping can silently run the
wrong computation. The opposite direction is safe: a program literal the
compiler eliminates remains covered by the key-bearing `literal_values` digest.
`literal_payload_values` separately authenticates the subset the package still
consumes, so partial elimination never weakens either identity or payload
verification.

Tensorhub is the first intended remote service, but no speculative registry or
plugin interface lives here. Its adapter will obtain byte-operation grants,
populate the same objects, then call `Engine.import_artifact(key, path)` for
verified attachment; compilation remains local-first and transport-independent.

## Contracts and spans

The versioned identity corpora used by non-Python consumers ship under
`torchcg.contracts`. `read_contract(name)` reads their canonical bytes from an
installed wheel; consumers pin the package version and corpus SHA-256 rather
than fetching a moving source branch. Their `authority` strings still read
`torch-compiled-graphs` — the pre-rename name is the historical authority, and
renaming it would rekey corpora that peers pin byte-for-byte.

`torchcg.recipe` is the reference implementation of `recipe_v1`, the versioned
vocabulary for one family's composition: which graph classes make one endpoint's
compiled pipeline, the loop between them, and the scheduler block that loop runs
under — including an autoregressive family's `loop.kind: host`, which states the
per-step classes and the session-state owner and says outright that the
data-dependent iteration is the host's. It is a vocabulary, not a DSL, and it is
deliberately class-level — it
pins each class by class hash, its exact `CallIngress` value, and the
tensor-layout contract it was traced against, never by a `cg-key-v1` value and
never by a checkpoint, so one machine-independent digest is valid on every SKU
and the key is folded at adopt time. Typed bindings are generated from a
declaration-time export rather than from the recipe; the recipe is the drift
assertion against that declaration and the adopt-time name-to-identity
reference. `docs/RECIPE.md` states the document, the digest rule, how it rides
beside `endpoint.lock`, and the numbered requirements a binding generator
implements against.

The versioned compile-span partition lives in `torchcg.spans`. Its three totals
each have one explicit residual, and `check()` must be run by the measurement
owner before emitting a table. Triton, autotune, and device-lock timing are
overlays, never partition members.

## CLI

```bash
torchcg inspect compiled_graph.tar.gz
torchcg verify compiled_graph.tar.gz
torchcg resolve --cas-root /var/cache/graphs CG_KEY DESTINATION
```

`inspect` validates and prints metadata. `verify` additionally checks the
artifact envelope, generated AOTI wrapper, ELF structure, constant manifest, and
code-only policy. `resolve` fully verifies a local exact-key artifact while
materializing it.

## Development

```bash
uv sync --all-extras
uv run ruff check src tests
uv run mypy
uv run pytest
```

PyTorch is an optional install extra because production workers control their
exact compiler build; the default compiler imports it only when minting.
`tensorfs` is git-pinned rather than installed from an index, so resolving the
lock builds a Rust extension — CI pins a stable toolchain for it.

CI installs all extras and runs a real CPU `torch.export` → AOTInductor →
`tensorfs` → restart-reuse test. CPU AOTI exercises the transforms' installed
decline/fallback path; the applied-transform tests compile and run the generated
CUDA-wrapper shapes as real C++, pinned against a verbatim captured sm_86
wrapper in `tests/realdata/`. The applied path on a real GPU was proven
separately on rented pods (PR #33) rather than in CI.

## Versioning and release

Package releases use SemVer, beginning with `0.1.0`; the first release tag is
`v0.1.0`. Internal artifact/key formats are independently named v1. Before
launch an internal v1 may be replaced in place: there are no dual readers,
compatibility aliases, or migration paths for abandoned pre-launch formats.

Internal consumers git-pin this repository. Nothing is published to PyPI as a
side effect of development — `.github/workflows/publish.yaml` is
`workflow_dispatch` only, and a release happens when someone deliberately asks
for one (DESIGN-RULINGS "Release policy", 2026-08-16). When dispatched, it
requires a `v<project version>` tag on `master`, rebuilds and smoke-tests the
wheel, and publishes through PyPI Trusted Publishing.

### 0.1.0 public API

- Compilation is owned by `Engine`; no public compiler, packager, context, or
  options callback can replace the fixed output-producing path. The engine
  derives the sole FakeTensorMode, ShapeEnv, and tracing context recursively
  from the `ExportedProgram` graph metadata and refuses incomplete or
  conflicting context.
- `is_compiled_graph_key` validates the shared scheme-agnostic
  `<scheme>-<56 lowercase hex>` boundary grammar. The core derives only
  `cg-key-v1` keys.
- Graph-class declarations use the current worker facts-v3 fold. The 16-hex
  canonical body witness and 16-hex class hash are paired collision chokepoints;
  the graph-class display name does not key.
- `CallIngress` is the closed v1 identity for one exported call. Its builder
  preserves mapping insertion order, flattened sequence positions (including
  non-tensor gaps), the ordered parameter axis (including zero-leaf arguments),
  parameter paths, exported placeholder names, finite symbol bounds, and
  excluded inputs. It is stamped at `graph.pytree.ingress`; its digest is derived
  and verified by `GraphClassDeclaration`.
- Literal identity is the worker's exact 32-hex v1 value digest and rides inside
  the graph-interface block. Weight values remain excluded. Toolchain identity
  accepts the worker-recorded content block through an explicit adapter seam.
- Durable values are `StoredCompiledGraph` objects and metadata carries one
  `graph_class` object. The retired `entry`, `GraphSpec`, `GraphDeclaration`,
  and `StoredGraph` shapes have no aliases or readers.
- `Engine.export_artifact(key, destination)` exports a fully verified artifact
  envelope without exposing ref layout. An occupied or racing destination is
  accepted only when its size and full SHA-256 match the exact selected manifest
  file; it is never overwritten.
- Host ISA facts cover every CPU and CUDA artifact. x86-64 compilation is
  process-wide capped at `x86-64-v3`; other architectures carry a conservative
  native feature requirement. Unstamped or unsupported artifacts fail closed.
- `Engine.runner` returns a gated `CompiledGraphRunner`; exact constant-table
  binding, by-reference lifetime, and one-class call binding are library-owned,
  while multi-class selection and eager fallback remain worker policy.
- `torchcg.spans` owns the compile attribution vocabulary and closure invariant
  used across the worker child boundary.
- `torchcg.recipe` owns the `recipe_v1` composition vocabulary: validated
  identifier types rather than bare strings, one closed refusal enum, a document
  that refuses any unknown version or field, and a `CallSignature` projection
  that is a pure function of `CallIngress`. It imports neither Torch nor the
  declaration module, and it decides nothing: bucket lookup is exact rather than
  ranked, and selection stays with `ingress_selection_v1`.

The package root exposes only the engine lifecycle, graph-class declarations,
result/value types, error types, the single `COMPILED_GRAPH_FORMAT` authority,
and `is_compiled_graph_key`. Introspection helpers remain in their owning
modules rather than being re-exported as a second facade.

## License

MIT
