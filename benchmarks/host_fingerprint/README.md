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

## Running the matrix (one command)

`run_matrix.py` ships the bundle to every target in a manifest, runs
`probe_run.py` there, collects the rows, folds them, and writes a dated
report. Transports: `local`, `docker` (free), `ssh` (pods).

```bash
# Review the entire campaign before spending a cent — prints the exact
# command sequence per target and executes nothing.
python run_matrix.py --targets targets.example.json --bundle bundle/ \
    --out results/ --dry-run

# Free targets only (pod targets are skipped unless you ask for them).
python run_matrix.py --targets targets.local-demo.json --bundle bundle/ --out results/

# The real campaign, once pods are booted and endpoints filled in.
python run_matrix.py --targets targets.json --bundle bundle/ --out results/ \
    --include-pod-only
```

`targets.example.json` is the host inventory **as data**: every target states
which axis it exists to vary, how to obtain it, and whether it needs a paid
pod. Pod targets are skipped by default, so no command in this directory can
start billing by accident.

## INCONCLUSIVE rows — the distinction that keeps the matrix honest

A host that cannot even attempt the probe (no torch, no package) yields a row
flagged `inconclusive`, and the report **excludes it from the contingency**.
Treating "could not try" as "incompatible" would mark every differing axis
RETAIN for a reason that has nothing to do with portability.

This is not hypothetical. The committed `local-demo-*` rows are exactly that
case: two containers whose glibc/libstdc++/os_release/cxx_compiler genuinely
differ from the build host, but with no torch installed. Scored naively they
would have condemned **eight** axes; scored honestly they condemn none and
the verdict stays `insufficient data`.

## Projected cache-hit improvement (tcg#4's "report before you change")

The issue asks for the expected cache-hit gain *before* any axis is touched.
Measured improvement needs probed hosts; a projection needs only a fleet
census:

```bash
# Build a census from a fleet export you supply (this never queries prod),
# or generate a synthetic one to exercise the pipeline.
python fleet_census.py --input export.jsonl --output results/census.json
python fleet_census.py --synthetic 400 --output results/census-synthetic.json

python matrix_report.py results/*.json --census results/census.json
```

The projection assumes axes vary **independently**, which a fleet of pinned
images does not — one image pins glibc, libstdc++ and torch together, so the
true gain from dropping any one of them is smaller than the model says. Every
projected record carries `projection_not_measurement` and
`assumes_axis_independence` so a number can never be quoted as a measurement.
It is only as good as the export behind it: a census from one region or one
image generation overstates agreement.

## Pod matrix runbook (the real measurement)

Build ONE bundle per compiler environment worth distinguishing, then run the
matrix against every representative image. Cover at minimum:

- the production worker CUDA images (current and previous release);
- an image with a different libstdc++/glibc pair (older LTS base);
- a different Torch patch build of the same minor (and one different minor);
- a Triton-present vs Triton-absent environment;
- an x86-64-v3-only host vs an avx512 host (`host_isa_*` axes).

Keep torch identical to the build host wherever the intent is to isolate an
OS-level axis; if torch moves too, several axes differ at once and the
contingency can only fail closed across all of them.

CPU pods (~$0.03–0.50) suffice for CPU-target probes; a CUDA-target matrix
needs GPU pods and is a separate campaign. Collect every row, run
`matrix_report.py`, and only then open the follow-up issue that removes or
coarsens axes — with regression cases per retained boundary, per tcg#4.

## The 2026-08-16 pod campaign — what the matrix actually measured

Three RunPod CPU pods (cpu3c, 2 vCPU, $0.06/hr; **$0.035 for the campaign**),
inventory in `targets.pod-2026-08-16.json`, rows in `results/pod-20260816-*`.
**10 conclusive rows, 0 inconclusive, 2 real failures, 3 distinct hosts** — the
two-host bar is cleared. Every verdict below is CPU-target only; the CUDA
images were not run.

Two bundles were built, one per OS, because a single bundle proves one
direction and **the two directions fail for different reasons**:

| build host | load host | result |
|---|---|---|
| debian-12 / glibc-2.36 / g++ 12.2 | debian-13 / glibc-2.41 | **FAIL** `dlopen: cannot enable executable stack as shared object requires` |
| debian-13 / glibc-2.41 / g++ 14.2 | debian-12 / glibc-2.36 | **FAIL** `libc.so.6: version 'GLIBC_2.38' not found` |
| either | ubuntu-24.04 / glibc-2.39 | pass, exact output |

**RETAIN — `os_release`, `glibc`, `libstdcxx_max_glibcxx`, `cxx_compiler`.**
A differing row failed, in both directions, with two unrelated mechanisms. The
boundary is real but narrow: the intermediate glibc-2.39 host loaded *both*
bundles, so this is not "any glibc step breaks". The four axes move together
because a base image pins them together and the matrix cannot separate them —
and `cxx_compiler` is a *build-time* fact recorded on the *load* host, so a row
varying it alone would prove nothing about the artifact. Fail closed on all four.

**COARSENING CANDIDATES — differed, never failed:**

- `python_abi` — cp313→cp311 alone (same OS, same torch wheel), and cp312 on
  the ubuntu host. Loads, executes, matches. The AOTI `.so` links libtorch, not
  libpython.
- `triton` — absent→3.7.1 alone. No effect on a CPU target. Triton's own
  targets are **unmeasured**; this is not licence to drop it for CUDA.
- `torch_version` / `torch_git` / `torch_config_digest` — a 2.13.0-built
  artifact loaded, executed and matched on torch 2.12.1, with those three axes
  differing alone. **One direction, one minor step.** The reverse is
  unobtainable: torch 2.12.1 cannot compile with this package at all
  (`torch._inductor.config` has no `cpp.march`, so the host ISA policy raises),
  and no second 2.13 patch is published on the cpu index. Do not read this as
  "torch does not matter".

**`torch_config_digest` is host-dependent, not build-dependent.**
`torch.__config__.show()` carries a runtime-detected `CPU capability usage:`
line. The same wheel (2.13.0+cpu, git `cf30153c`) digests `5f31e143bbf0b632` on
an EPYC 7713 (AVX2) and `76efcb828dca63fb` on an EPYC 9655P (AVX512), so the
axis fragments the cache by CPU model for no portability reason. Meanwhile
`host_isa_*` is blind to exactly that difference: compiles clamp to
`x86-64-v3`, so every v3-or-better x86 host records identical ISA facts (the
v3 artifact ran correctly on the AVX512 host). The fix is to stabilise the
digest, not to drop the axis.

**UNMEASURED, fail-closed — `machine`, `host_isa_level`, `host_isa_features`,
`torch_cxx11_abi`.** No row ever saw them differ, and the ISA pair *cannot*
differ between two v3+ x86 hosts, which is why no avx512-vs-v3 pod pair was
rented.

Measured cache-hit improvement over the 7 observed host identities (21 pairs,
a sample of hosts and not a fleet): dropping `triton` newly shares 2 pairs
(+9.5%), `python_abi` 1 pair (+4.8%), each torch axis 0 pairs — no two hosts
here are separated by torch alone.

**No axis changed on this evidence, deliberately.** This package's canonical
key is exactly `("graph", "sm", "toolchain")`; the toolchain block's *members*
are supplied by the caller. Every coarsening candidate above is a
worker-supplied member, and the only fingerprint facts this package owns —
`machine` and `host_isa_*` — are unmeasured. The coarsening decision therefore
belongs to the worker lane; this repo owns the evidence.

## results/

- `local-<hostname-hash>.json` — the phase-1 single-host arm (all axes equal,
  all stages ok).
- `local-demo-*.json` + `report-local-demo-*.json` — the free 3-host runner
  demonstration described above. It proves the runner and the inconclusive
  rule on real hosts; it proves **nothing** about portability.
- `census-synthetic.json`, `fleet-export-synthetic.jsonl` — the projection
  pipeline's synthetic input and output. Synthetic, deliberately correlated,
  and not production data.
- `pod-20260816-b12-*.json` / `pod-20260816-b13-*.json` — the campaign above,
  named for the bundle's build host (b12 = debian-12, b13 = debian-13), with
  `report-pod-20260816-b12.json` and `-b13.json` the folds. These are the only
  committed rows that say anything about portability.

**No fingerprint axis changes until the matrix holds conclusive rows from at
least two distinct hosts.** The pod campaign clears that bar; the axes it
cleared are not this package's to change, and the ones this package owns stay
fail-closed.
