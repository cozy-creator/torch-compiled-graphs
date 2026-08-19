# Compiled-graph artifact v1

V1 has one writer shape and no compatibility reader for abandoned pre-launch
formats.

## Identity

An artifact carries one `compiled_graph_key` with this canonical derivation:

```text
cg-key-v1- + first 56 lowercase hexadecimal characters of
SHA-256(canonical JSON({graph, sm, toolchain}))
```

The three axes are exhaustive:

- `graph`: the 16-hex graph-specialization hash of the current facts-v3 fold. It binds
  target, fork, specialization dimensions, the derived call-ingress digest,
  graph-interface facts,
  the 16-hex canonical body witness, strictness, LoRA bucket, and multi-device
  placement when present. The display graph-specialization name does not key;
- `sm`: the concrete CUDA compute capability or CPU target; and
- `toolchain`: the 16-hex current-worker digest of an explicit recorded
  compiler-content block. The block carries the settings declaration,
  loaded-library digest, installed Torch/Triton/NVIDIA distribution RECORD
  digests, and bundled CUDA binary digests. Trace-only `diffusers`,
  `transformers`, and `peft` records are removed by the one membership rule.

Family, checkpoint, weights, declaration-wide coverage, and trace-time model
libraries are not axes. Facts that alter the trace arrive through `graph`;
facts that alter compilation arrive through `toolchain`.

## Envelope

The artifact is a deterministic gzip-compressed USTAR archive with exactly:

```text
metadata.json
model.pt2
constants.safetensors  # present only when the graph specialization declares literals
```

All tar timestamps and ownership fields and the gzip timestamp are zero.
`metadata.json` is sorted compact ASCII JSON. The package is one graph specialization,
not a bundle.

Required top-level metadata includes:

- `compiled_graph_format: 1`;
- `kind: "aot-inductor"`;
- `compiled_graph_key`, which must be exactly derivable from the recorded facts;
- `graph_specialization`, with non-empty `name`, `target`, `specialization_hash`,
  `graph_witness`, and `range_digest`; a non-empty `graph` interface object
  whose `pytree.ingress` is the exact closed CallIngress v1 value;
  canonical `fork`, `specialization_dims`, `strict`, `lora_bucket`, and `placement`
  facts; plus `literal_values` and a `constants` array. `specialization_hash` must be
  recomputable from the exact facts-v3 fold;
- non-empty `sm` and worker-recorded `toolchain` facts;
- a separate `host_isa` requirement;
- `package_constants_in_so: false`; and
- `constant_folding_fenced: true`.

Those are the exact v1 top-level and graph-specialization fields; extensions and abandoned
pre-launch shapes are refused. Readers reject missing, duplicate, non-file, or
unexpected archive members and materialize only into a new directory.

Archive admission is streaming and bounded before publication: `metadata.json`
is limited to 8 MiB, `model.pt2` to 16 GiB, and `constants.safetensors` to
4 GiB, with a total uncompressed ceiling equal to their sum. The package limit
is four times ZIP32's 4 GiB boundary so unusually large generated host or CUDA
code remains representable. Literal bytes above the ZIP32 boundary are treated
as model state and belong in the separately bound repository snapshot. Declared
oversize members, a compressed input above the total plus a one-percent deflate
margin, multiple gzip members, non-canonical tar extensions or tails, truncated
compression streams, and bytes beyond the uncompressed ceiling fail as
`ArtifactError`; an unsuccessful import publishes neither a destination nor an
exact-key ref.

The `host_isa` object must include `machine`, `host_isa_level`,
`host_isa_features`, `cpp_march`, and `cpp_simdlen`. x86-64 writers cap `cpp_march` at
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
Literal identity is the first 32 lowercase hexadecimal characters of SHA-256
over each sorted FQN, a NUL byte, `str(torch.dtype)`, `str(tuple(shape))`, and
the contiguous raw uint8 tensor bytes. There are no separators after dtype or
shape. The same routine validates `constants.safetensors`; changing name,
dtype, shape, or value rekeys, while state-dict weight bytes do not.

## Store namespace — frozen

A stored compiled graph is addressed by two ref names, and both are v1 identity:

```text
torchcg/v1/graphs/<compiled_graph_key>
torchcg/v1/quarantine/<compiled_graph_key>/<artifact digest>
```

`torchcg/v1` is frozen. It is the address every published compiled graph is
filed under, so moving it does not rename anything — it orphans every ref in
every store at once and costs a store wipe plus a fleet-wide re-mint. The
package once learned this the expensive way: a 2026-08-17 rename from
`torch-compiled-graphs/v1` orphaned the whole fleet's cache and was paid for by
purging and re-minting (tcg#39). `tests/test_storage.py` pins the prefix, both
ref shapes, the quarantine marker's keys, and the `artifact` field on
`StoreResult`/`StoredCompiledGraph` for the same reason. A rename sweep over
this package must exempt storage and wire identity strings rather than update
them.

Package release versions are ordinary SemVer and start at `0.1.0`. They are
independent from this internal v1. Before launch, this one accepted v1 may be
replaced in place; the package does not carry dual readers or writers.

A compiled graph contains one graph specialization. This format therefore emits no
bundle list. Any layer that groups several compiled graphs names that list
`compiled_graph_manifest`; tensorfs's generic `RepositoryManifest` remains a
byte-storage record and is not the compiled-graph bundle contract.
