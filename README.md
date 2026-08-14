# compiled-graphs

`compiled-graphs` is the worker-independent core for minting, identifying,
packing, verifying, storing, and reusing PyTorch AOTInductor graphs.

The v1 boundary is deliberately narrow:

- one `torch.export.ExportedProgram` graph class per artifact;
- one `cg-key-v1` identity derived only from graph, GPU architecture, and
  compiler-toolchain facts;
- one deterministic artifact envelope containing `metadata.json`, `model.pt2`,
  and an optional literal-constant payload;
- code-only AOTI compilation with weight folding kept bindable; and
- storage behind `chunked-cas`, with Tensorhub as the first remote adapter.

This repository does not own endpoint composition, model loading, GPU
scheduling, worker telemetry, or Tensorhub policy. Those remain application
concerns in `python-gen-worker` and Tensorhub.

## Status

This is the first pre-release extraction slice. It establishes and tests the
format-v1 identity, compilation seam, deterministic package boundary, and
fail-closed AOTInductor package introspection. The `chunked-cas` adapter and the
migrations that delete the old worker copies are dependency-ordered follow-ups.

The repository is private during package and license review.

## Development

```bash
uv sync --all-extras
uv run ruff check src tests
uv run mypy
uv run pytest
```

PyTorch is imported only when the default compiler or packager is invoked.
Unit tests inject those two callables, so the format and policy core stays
cheap to test without CUDA or a multi-gigabyte torch installation.

## License

MIT
