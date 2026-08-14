# torch-compiled-graphs

`torch-compiled-graphs` mints and reuses verified PyTorch AOTInductor graphs.
It is independent of `python-gen-worker`: applications supply one exported
program plus the worker-recorded compiler-content block, while HashRepo is the
sole local content-addressed storage and chunking layer.

## V1 lifecycle

```python
from hashrepo import LocalCAS
from torch_compiled_graphs import (
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
identity. The same declaration
derives the `cg-key-v1` lookup, mint stamp, and
admission expectation. Future workers call `Engine.resolve(key, destination)`
with the known key: that hit path neither constructs an ExportedProgram nor
imports Torch. It does read the local CPU feature surface and fails before
returning a package if the artifact's host-code requirement is incomplete or
unsupported. `ensure` also resolves first and invokes its lazy recipe exactly
once only on a miss. A miss compiles one code-only graph under the sole v1
compile policy, packages it as a
verified artifact, stores it through HashRepo, and materializes it. A later
process pointed at the same HashRepo root reuses it without compiling.
`Engine.compile(spec, runtime, destination)` is the sealed one-class operation
used by a compile child: it derives its own key and reuses an admitted exact
record before doing compiler work. There is no public plan, compiler callback,
option map, or caller-supplied identity.

`Engine.runner(key, destination)` resolves the exact HashRepo-selected bytes,
loads the named AOTI model, and returns a `CompiledGraphRunner`. The runner
refuses every call until `bind` proves the package's own constant table equals
the artifact manifest and completes one by-reference update. Missing, partial,
or out-of-memory binding leaves that runner permanently unusable; a caller may
load a fresh instance on a pod with capacity. Ingress selection, eager fallback,
family composition, process scheduling, Hub receipts, and telemetry remain
application policy and are intentionally absent from this package.

The sealed compiler policy uses four Inductor compile workers: pgw#757's
measured contention ceiling and the current worker value. A balanced 16-run
one-versus-four comparison found no material wall-time difference, so this is
parity and a ceiling, not a claim that four is intrinsically faster. The
compiler privately rewrites the generated C++ wrapper's pathological
constructor and `run_impl` functions into reconstruction-checked smaller
functions/translation units. Unrecognized shapes compile unchanged. A failed
run_impl split retries the constructor-transformed monolith when that first
transform applied; if that build also fails, or the constructor transform was
the failure, the compiler retries PyTorch's original source. Neither transform
is a caller option, environment switch, public API, or identity axis.

`recorded_toolchain` is an explicit adapter input, not a guessed version or
host fingerprint. A worker records its settings declaration, loaded-library
digest, installed Torch/Triton/NVIDIA wheel RECORD digests, and bundled CUDA
binary digests. This core applies the current worker membership rule and folds
that content to the 16-hex `toolchain` axis; trace-only `diffusers`,
`transformers`, and `peft` records are excluded because their effects already
ride the graph. Host ISA requirements are recorded and admitted separately.

`Engine.export_artifact(key, path)` emits the exact verified artifact envelope
selected by the local HashRepo manifest for a remote adapter.
`Engine.import_artifact(key, path)` validates its metadata and host requirement
before attaching the fully verified bytes to the local CAS. Neither operation
exposes the package's private HashRepo ref layout.

HashRepo owns immutable objects, chunk manifests, durability, GC reachability,
and transfer primitives. This package owns only graph declaration, compilation,
artifact policy, exact-key refs, admission, and quarantine. A divergent second
artifact never overwrites an admitted key. A corrupt or inadmissible manifest
is quarantined and can be repaired only by a newly verified mint.

A package literal whose AOTI wrapper names no FQN lifted by the exported
program is refused rather than matched by tensor order or shape. Register
durable module tensors as buffers; guessing an anonymous constant mapping can
silently run the wrong computation. The opposite direction is safe: a program
literal the compiler eliminates remains covered by the key-bearing
`literal_values` digest. `literal_payload_values` separately authenticates the
subset the package still consumes, so partial elimination never weakens either
identity or payload verification.

Tensorhub is the first intended remote service, but no speculative registry or
plugin interface lives here. Its adapter will obtain byte-operation grants and
populate the same HashRepo objects/manifests, then call
`Engine.import_artifact(key, path)` for verified attachment; compilation remains
local-first and transport-independent.

The versioned identity corpora used by non-Python consumers ship under
`torch_compiled_graphs.contracts`. `read_contract(name)` reads their canonical
bytes from an installed wheel; consumers pin the package version and corpus
SHA-256 rather than fetching a moving source branch.

The versioned compile-span partition lives in
`torch_compiled_graphs.spans`. Its three totals each have one explicit
residual, and `check()` must be run by the measurement owner before emitting a
table. Triton, autotune, and device-lock timing are overlays, never partition
members.

## CLI

```bash
torch-compiled-graphs inspect compiled_graph.tar.gz
torch-compiled-graphs verify compiled_graph.tar.gz
torch-compiled-graphs resolve --cas-root /var/cache/graphs CG_KEY DESTINATION
```

`inspect` validates and prints metadata. `verify` additionally checks the
artifact envelope, generated AOTI wrapper, ELF structure, constant manifest,
and code-only policy. `resolve` fully verifies a local exact-key artifact while
materializing it.

## Development

```bash
uv sync --all-extras
uv run ruff check src tests
uv run mypy
uv run pytest
```

PyTorch is an optional install extra because production workers control their
exact compiler build. The default compiler imports it only when minting. CI
installs all extras and runs a real CPU `torch.export` to AOTInductor to
HashRepo to restart-reuse test. CPU AOTI currently exercises the transforms'
installed decline/fallback path; the applied-transform tests compile and run
the generated CUDA-wrapper shapes as real C++. A CUDA-generated applied-path
proof remains an explicit worker integration gate rather than being implied by
the CPU test.

## Versioning and release

Package releases use SemVer, beginning with `0.1.0`; the first release tag is
`v0.1.0`. Internal artifact/key formats are independently named v1. Before
launch an internal v1 may be replaced in place: there are no dual readers,
compatibility aliases, or migration paths for abandoned pre-launch formats.

### 0.4.0 public API

- Compilation is owned by `Engine`; no public compiler, packager, context, or
  options callback can replace the fixed output-producing path. The engine
  derives the sole FakeTensorMode, ShapeEnv, and tracing context recursively
  from the `ExportedProgram` graph metadata and refuses incomplete or
  conflicting context.
- `is_compiled_graph_key` validates the shared scheme-agnostic
  `<scheme>-<56 lowercase hex>` boundary grammar. The core derives only
  `cg-key-v1` keys.
- Graph-class declarations use the current worker facts-v3 fold. The 16-hex
  canonical body witness and 16-hex class hash are paired collision
  chokepoints; the graph-class display name does not key.
- `CallIngress` is the closed v1 identity for one exported call. Its builder
  preserves mapping insertion order, flattened sequence positions (including
  non-tensor gaps), the ordered parameter axis (including zero-leaf arguments),
  parameter paths, exported placeholder names, finite symbol bounds, and
  excluded inputs. It is stamped at `graph.pytree.ingress`; its
  digest is derived and verified by `GraphClassDeclaration`.
- Literal identity is the worker's exact 32-hex v1 value digest and rides
  inside the graph-interface block. Weight values remain excluded. Toolchain
  identity accepts the worker-recorded content block through an explicit
  adapter seam and matches the current 16-hex worker fold and final key.
- Durable values are `StoredCompiledGraph` objects and metadata carries one
  `graph_class` object. The retired `entry`, `GraphSpec`, `GraphDeclaration`,
  and `StoredGraph` shapes have no aliases or readers.
- `Engine.export_artifact(key, destination)` exports a fully verified artifact
  envelope without exposing HashRepo ref layout. An occupied or racing
  destination is accepted only when its size and full SHA-256 match the exact
  selected manifest file; it is never overwritten.
- Host ISA facts now cover every CPU and CUDA artifact. x86-64 compilation is
  process-wide capped at `x86-64-v3`; other architectures carry a conservative
  native feature requirement. Unstamped or unsupported artifacts fail closed.
- `Engine.compile` is the one public one-class compiler lifecycle. It computes
  declaration and key itself, admits an exact prior record, and otherwise runs
  the sole sealed compiler/package/store path.
- `Engine.runner` returns a gated `CompiledGraphRunner`; exact constant-table
  binding, by-reference lifetime, and one-class call binding are library-owned,
  while multi-class selection and eager fallback remain worker policy.
- `torch_compiled_graphs.spans` owns the compile attribution vocabulary and
  closure invariant used across the worker child boundary.

The package root exposes only the engine lifecycle, graph-class declarations,
result/value types, error types, the single `COMPILED_GRAPH_FORMAT` authority,
and `is_compiled_graph_key`. Introspection helpers remain in their owning
modules rather than being re-exported as a second facade.

The release workflow is exactly `.github/workflows/publish.yaml`. It requires
a `v<project version>` tag on `main`, rebuilds and smoke-tests the wheel, uses
PyPI Trusted Publishing through the `pypi` environment, and verifies the exact
`torch-compiled-graphs` version endpoint.

## License

MIT
