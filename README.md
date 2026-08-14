# torch-compiled-graphs

`torch-compiled-graphs` mints and reuses verified PyTorch AOTInductor graphs.
It is independent of `python-gen-worker`: applications supply one exported
program plus an explicit deployment-compatibility axis, while the library
derives the concrete compiler/runtime fingerprint and HashRepo is the sole local
content-addressed storage and chunking layer.

## V1 lifecycle

```python
from hashrepo import LocalCAS
from torch_compiled_graphs import Engine, GraphSpec, RuntimeCompatibility

spec = GraphSpec("denoiser/h=64,w=64", "unet", exported_program)
runtime = RuntimeCompatibility(
    "sm_89",
    deployment_compatibility="worker-image-sha256:...",
)
key = runtime.key(spec.declare())
result = Engine(LocalCAS("/var/cache/graphs")).ensure(
    key,
    "/run/graphs/denoiser-h64-w64",
    target="sm_89",
    deployment_compatibility="worker-image-sha256:...",
    recipe=lambda: spec,
)
runner_package = result.graph.package
```

The same declaration derives the `cg-key-v1` lookup, mint stamp, and
admission expectation. Future workers call `Engine.resolve(key, destination)`
with the known key: that hit path neither constructs an ExportedProgram nor
imports Torch. `ensure` also resolves first and invokes its lazy recipe exactly
once only on a miss. A miss compiles one code-only graph under the sole v1
compile policy, packages it as a
verified artifact, stores it through HashRepo, and materializes it. A later
process pointed at the same HashRepo root reuses it without compiling.

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

## CLI

```bash
torch-compiled-graphs inspect graph.tar.gz
torch-compiled-graphs verify graph.tar.gz
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
HashRepo to restart-reuse test.

## Versioning and release

Package releases use SemVer, beginning with `0.1.0`; the first release tag is
`v0.1.0`. Internal artifact/key formats are independently named v1. Before
launch an internal v1 may be replaced in place: there are no dual readers,
compatibility aliases, or migration paths for abandoned pre-launch formats.

The release workflow is exactly `.github/workflows/publish.yaml`. It requires
a `v<project version>` tag on `main`, rebuilds and smoke-tests the wheel, uses
PyPI Trusted Publishing through the `pypi` environment, and verifies the exact
`torch-compiled-graphs` version endpoint.

## License

MIT
