# Compile-policy evidence

`compare_compile_threads.py` exports and AOTI-compiles a diffusion-shaped UNet
graph class: convolutions, timestep MLP, normalization, cross-attention,
downsample and upsample. It is deliberately outside the wheel and accepts only
the two measurement arms; the production compiler has no options surface.

## 2026-08-14 cold-cache comparison

- Host: Intel i9-13950HX, 24 cores / 32 threads, Linux 6.17.
- Compiler: PyTorch 2.13.0+cu130; CPU AOTI; four loose package files plus the
  manifest row returned in both arms.
- Each arm used a fresh process, fresh `TORCHINDUCTOR_CACHE_DIR`, fixed seed,
  and `nice -n 10`. Three balanced passes used orders 1/4/4/1, 4/1/1/4, and
  1/4/4/1/4/1/1/4. No other compiler was active.
- The harness ran its partition check itself. Every arm recorded
  `partition_problems=[]`; lowering, codegen, graph passes, host compile and
  the explicit residual summed to the measured compile wall, and the residual
  was non-negative.

| compile threads | eight cold walls (s) | mean (s) | median (s) | sample stdev (s) |
|---:|---|---:|---:|---:|
| 1 | 20.594, 21.176, 19.050, 20.563, 19.977, 18.516, 19.308, 19.245 | 19.804 | 19.643 | 0.918 |
| 4 | 18.943, 18.211, 21.479, 20.654, 19.351, 20.490, 18.789, 18.857 | 19.597 | 19.147 | 1.138 |

Four threads won five of eight adjacent balanced comparisons. Its aggregate
mean was 1.0% lower and its median 2.5% lower, both smaller than the observed
run-to-run variation; peak RSS stayed in the same roughly 730--738 MiB band.
The honest result is wall-time equivalence on this graph, not the first
four-sample run's apparent 11.1% win. Four remains sealed because it preserves
the current worker policy and pgw#757's measured 32-to-4 contention ceiling;
this comparison found no reason to create a second value or to claim a speedup.

The two host-code transforms have separate evidence: the test suite compiles
and executes original and transformed C++ translation units, compares their
outputs byte-for-byte, and mutates the original statements and generated
dispatch chain to prove both gates can fail red. The full suite also compiles,
packages, loads and output-checks two real CPU AOTI artifacts under the selected
policy; those wrappers decline the CUDA-oriented transforms and therefore
prove fallback, not an applied transform. A real CUDA-generated applied-path
proof is deliberately left as a worker integration gate.
