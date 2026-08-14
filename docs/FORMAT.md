# Compiled-graph artifact v1

V1 has one writer shape and no compatibility reader for abandoned pre-launch
formats.

## Identity

An artifact carries one `cell_key` with this canonical derivation:

```text
cg-key-v1- + first 56 lowercase hex characters of
SHA-256(canonical JSON({graph, sm, toolchain}))
```

The three axes are exhaustive:

- `graph`: the one entry's traced-computation class hash;
- `sm`: the target GPU compute capability; and
- `toolchain`: the digest of compiler components and settings.

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
- `cell_key`, which must be exactly derivable from the recorded facts;
- `entry`, with non-empty `name`, `target`, `class_hash`, and a `constants`
  array;
- non-empty `sm` and `toolchain` facts;
- `package_constants_in_so: false`; and
- `constant_folding_fenced: true`.

The generic `format` field and plural `entries` bundle are retired and refused.
Readers reject missing, duplicate, non-file, or unexpected archive members and
materialize only into a new directory.

Each constant row has exactly `fqn`, `source`, `dtype`, and `shape`. `source`
is one of `state_dict`, `computed`, or `literal`; `shape` is an array of
non-negative integer dimensions. Unknown, incomplete, or duplicate rows are
refused. A `literal` row requires `constants.safetensors`; the package's own
AOTInductor wrapper remains the authority for the constant classification.
