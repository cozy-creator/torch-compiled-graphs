# torch-compiled-graphs

`torch-compiled-graphs` mints and reuses verified PyTorch AOTInductor graphs.
It is independent of `python-gen-worker`: applications supply one exported
program plus the worker-recorded compiler-content block, while HashRepo is the
sole local content-addressed storage and chunking layer.

## V1 lifecycle

```python
from hashrepo import LocalCAS
from torch_compiled_graphs import Engine, GraphClassSpec, RuntimeCompatibility

spec = GraphClassSpec(
    "denoiser/h=64,w=64",
    "unet",
    exported_program,
    graph=graph_interface,
    range_digest=declared_ingress_range_digest,
)
runtime = RuntimeCompatibility(
    "sm_89",
    toolchain=recorded_toolchain,
)
key = runtime.key(spec.declare())
result = Engine(LocalCAS("/var/cache/graphs")).ensure(
    key,
    "/run/graphs/denoiser-h64-w64",
    target="sm_89",
    toolchain=recorded_toolchain,
    recipe=lambda: spec,
)
runner_package = result.compiled_graph.package
```

`graph_interface` and `declared_ingress_range_digest` are the current v3
graph-class facts produced by the declaration adapter. The same declaration
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

The sealed compiler policy uses four Inductor compile workers: pgw#757's
measured contention ceiling and the current worker value. A balanced 16-run
one-versus-four comparison found no material wall-time difference, so this is
parity and a ceiling, not a claim that four is intrinsically faster. The
compiler privately rewrites the generated C++ wrapper's pathological
constructor and `run_impl` functions into reconstruction-checked smaller
functions/translation units. Unrecognized shapes compile unchanged; failed
transformed builds retry PyTorch's original source. Neither transform is a
caller option, environment switch, public API, or identity axis.

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

A literal whose AOTI wrapper erases its exported FQN is refused rather than
matched by tensor order or shape. Register durable module tensors as buffers;
guessing an anonymous constant mapping can silently run the wrong computation.

Tensorhub is the first intended remote service, but no speculative registry or
plugin interface lives here. Its adapter will obtain byte-operation grants and
populate the same HashRepo objects/manifests, then call
`Engine.import_artifact(key, path)` for verified attachment; compilation remains
local-first and transport-independent.

The versioned identity corpora used by non-Python consumers ship under
`torch_compiled_graphs.contracts`. `read_contract(name)` reads their canonical
bytes from an installed wheel; consumers pin the package version and corpus
SHA-256 rather than fetching a moving source branch.

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

### 0.1.3 public API

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
