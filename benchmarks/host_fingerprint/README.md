# tcg#4 — host-compatibility fingerprint portability matrix

Measures which fingerprint axes actually predict AOTI load/execution
incompatibility. **No fingerprint change happens until the matrix holds real
rows from at least two distinct hosts** — this harness produces the evidence
the issue requires before any axis is removed or coarsened.

## Halves

```bash
# Build host: compile one small real CPU AOTI graph, export the artifact,
# record every axis, pin input/expected/constants.
python probe_build.py --output bundle/

# Any candidate host: record the same axes, import -> load -> bind ->
# execute -> compare, emit one matrix row.
python probe_run.py bundle/ --output results/<host>.json

# Fold rows into the per-axis contingency + cache-hit improvement report.
python matrix_report.py results/*.json
```

Both halves need the package importable (`PYTHONPATH=../../src` or an
installed wheel) plus torch; `matrix_report.py` needs neither torch nor a GPU.

## Axes recorded (one value each, diffable per axis)

machine, os_release, glibc, libstdcxx_max_glibcxx, cxx_compiler, python_abi,
torch_version, torch_git, torch_cxx11_abi, torch_config_digest, triton,
host_isa_level, host_isa_features.

The probe's `RuntimeCompatibility` toolchain block folds these into the same
worker-shaped members (settings/libs/torch/triton), so the folded 16-hex axis
moves exactly when an underlying fact moves.

## Pod matrix runbook (the real measurement)

Build ONE bundle per compiler environment worth distinguishing, then run
`probe_run.py` on every representative image. Cover at minimum:

- the production worker CUDA images (current and previous release);
- an image with a different libstdc++/glibc pair (older LTS base);
- a different Torch patch build of the same minor (and one different minor);
- a Triton-present vs Triton-absent environment;
- an x86-64-v3-only host vs an avx512 host (`host_isa_*` axes).

CPU pods (~$0.03–0.50) suffice for CPU-target probes; a CUDA-target matrix
needs GPU pods and is a separate campaign. Collect every row, run
`matrix_report.py`, and only then open the follow-up issue that removes or
coarsens axes — with regression cases per retained boundary, per tcg#4.

## results/

`local-<hostname-hash>.json` is the single-host arm proving the harness end
to end (all axes equal, all stages ok). A one-host matrix decides nothing and
the report says so.
