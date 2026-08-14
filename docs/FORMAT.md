# Compiled-graph artifact v1

V1 has one writer shape and no compatibility reader for abandoned pre-launch
formats.

## Identity

An artifact carries one `compiled_graph_key` with this canonical derivation:

```text
cg-key-v1- + full lowercase hexadecimal
SHA-256(canonical JSON({graph, sm, toolchain}))
```

The three axes are exhaustive:

- `graph`: the full SHA-256 class hash of the canonical traced computation,
  including input tensor shape, stride, layout, dtype, and placement facts;
- `sm`: the concrete CUDA compute capability or CPU target; and
- `toolchain`: a full SHA-256 digest of library-derived Torch/Inductor, Triton,
  Python ABI/platform/arch, CUDA or CPU, compiler, libc/libstdc++, fixed compile
  settings, host ISA requirement, and the required deployment-compatibility
  axis.

Family, checkpoint, weights, declaration-wide coverage, and trace-time model
libraries are not axes. Facts that alter the trace arrive through `graph`;
facts that alter compilation arrive through `toolchain`.

## Envelope

The artifact is a deterministic gzip-compressed USTAR archive with exactly:

```text
metadata.json
model.pt2
constants.safetensors  # present only when the entry declares literals
```

All tar timestamps and ownership fields and the gzip timestamp are zero.
`metadata.json` is sorted compact ASCII JSON. The package is one graph class,
not a bundle.

Required top-level metadata includes:

- `compiled_graph_format: 1`;
- `kind: "aot-inductor"`;
- `compiled_graph_key`, which must be exactly derivable from the recorded facts;
- `entry`, with non-empty `name`, `target`, `class_hash`, and `graph`, plus
  `literal_values`, `placement`, and a `constants` array. `class_hash` must be
  recomputable from those declaration facts;
- non-empty `sm` and `toolchain` facts;
- `package_constants_in_so: false`; and
- `constant_folding_fenced: true`.

Those are the exact v1 top-level and entry fields; extensions and abandoned
pre-launch shapes are refused. Readers reject missing, duplicate, non-file, or
unexpected archive members and materialize only into a new directory.

The toolchain must include `machine`, `host_isa_level`, `host_isa_features`,
`cpp_march`, and `cpp_simdlen`. x86-64 writers cap `cpp_march` at
`x86-64-v3`, record only the cumulative features required by that level, and
use a matching 128- or 256-bit SIMD width. Other machines record `native` plus
the CPU features common to every processor reported by Linux. Readers reject
missing facts, cross-machine artifacts, unknown levels, above-v3 x86 code, and
requirements the current host cannot satisfy before returning the package.

Each constant row has exactly `fqn`, `source`, `dtype`, and `shape`. `source`
is one of `state_dict`, `computed`, or `literal`; `shape` is an array of
non-negative integer dimensions. Unknown, incomplete, or duplicate rows are
refused. A `literal` row requires `constants.safetensors`, and that payload is
forbidden otherwise. Readers validate its bounded header and exact names,
dtypes, shapes, byte ranges, and full value digest before admission. The
package's own AOTInductor wrapper remains the authority for classification.

Package release versions are ordinary SemVer and start at `0.1.0`. They are
independent from this internal v1. Before launch, this one accepted v1 may be
replaced in place; the package does not carry dual readers or writers.
