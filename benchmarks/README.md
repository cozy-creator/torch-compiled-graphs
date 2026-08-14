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
  and `nice -n 10`. Order was 1, 4, 4, 1. Load average was about 6 before the
  run; no other compiler was active.
- The harness ran its partition check itself. Every arm recorded
  `partition_problems=[]`; lowering, codegen, graph passes, host compile and
  the explicit residual summed to the measured compile wall, and the residual
  was non-negative.

| compile threads | cold wall (s) | peak RSS (MiB) |
|---:|---:|---:|
| 1 | 20.594 | 730 |
| 4 | 18.943 | 736 |
| 4 | 18.211 | 736 |
| 1 | 21.176 | 737 |

Four threads won in both orders: 8.0% and 14.0%. The two-run mean was 18.577s
versus 20.885s, an 11.1% wall reduction, with no observed peak-RSS increase
(736MiB versus the one-thread maximum of 737MiB). This agrees with pgw#757's
larger production measurement and selects four as the sealed policy.

The two host-code transforms have separate evidence: the test suite compiles
and executes original and transformed C++ translation units, compares their
outputs byte-for-byte, and mutates each generated shape to prove the
reconstruction gate can fail red. The full suite also compiles, packages,
loads and output-checks two real AOTI artifacts under the selected policy.
