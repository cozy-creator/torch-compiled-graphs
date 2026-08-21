# Transform passes

A **pass** is what makes a lane's format. A lane is a tensor-layout STAMP —
the `(topology, quant)` pair, `sdxl.diffusers@1+cozy.fp8-rowwise@1`;
an fp8 lane's weights are fp8 because something converted them, and a
precompute lane's blocks hold a side table because something folded them.
A pass gives that "something" a name, a version, an identity and an artifact.
Author code still runs untouched over the result.

## Ordering: PASS → DISCOVERY → EXPORT

The graph that gets identity must be the graph that serves. Enforced from both
ends:

- a pass runs inside a `TransformSession`, which `seal()`s once; a `run` after
  the seal refuses;
- `discover_modules` and `AdoptSession.adopt` stamp every module they touch
  export-sealed, and `TransformSession.run` refuses a target holding a sealed
  submodule;
- `discover_modules(..., transforms=sealed_set)` reconciles the lane's declared
  passes with the set that ran — either direction is a refusal — and
  `AdoptSession` does the same at construction, beside the exact-env audit.

The pass names are a `cg-graph-v1` derivation input, exactly as pins are: two
lanes differing only by a pass are different graphs, and the lane row in the
release document carries them.

## The interface

```python
class TransformPass(Protocol):
    NAME: str                                    # "<name>@<major>"
    def plan(self, target, *, lane: LaneRef) -> TransformPlan: ...
    def apply(self, target, plan, *, artifacts: SideTableStore | None) -> TransformReport: ...
```

`TransformPlan` states what will happen: `rewrites` (module FQNs), `removes`
(the tensorfs removable-set names the rewrite frees, so a trimmed model stops
downloading those bytes), `domain`, and `scope_bytes` — the bytes the targets
hold now, the only byte count knowable before the pass runs. `TransformReport`
states what did happen: measured `freed_bytes`/`cache_bytes`, the buckets
`produced` and `loaded`, and `validated`, which is `False` until a numerical
equivalence gate has actually run on real weights.

Passes register by name (`register_pass`, `resolve_pass`), which is how a lane
can name one and an unknown name can be refused rather than silently skipped.

## `precompute-and-free@1`

Evaluates a submodule over a declared domain, installs the outputs as a side
table, releases the weights.

```python
session = TransformSession(lane, store=store)
report = session.run(pipe.transformer, PrecomputeAndFree(
    select="transformer_blocks.*.adaln_proj",
    domain=[schedule(n) for n in STEP_PRESETS],
    key=timestep_key,          # a domain point's canonical identity
    call=temb_for,             # (root, submodule, point) -> the submodule's input
    removes="adaln_projections",
))
transforms = session.seal()
...
torchcg.transform.handle(pipe.transformer, report.pass_name).bind(schedule)
```

`apply` is always `ensure`: buckets the store already holds are read (the
submodule is never evaluated for them), the rest are computed and published.
The same key addresses both directions, so the fallback is a performance
difference and never a correctness one. Off-domain is a typed `OffDomain`
refusal — the weights are released and cannot be recomputed — so the legal
domain must be an API-boundary type on the consumer side.

The side table is registered as **non-persistent buffers**, not a dict: it
moves with `.to()`/`.cuda()` because it is module state, `torch.export` lifts
it as named inputs the arm binds by name instead of baking constants into the
`.so`, and it stays out of `state_dict()`.

What stays with the consumer: what a domain point IS, how to call the folded
submodule, and any row remapping its packing needs.

## `quantize-in-place@1`

`QuantRecipe` is a closed vocabulary a lane contract names — pinned
granularity (no default), `min_sm`, `pays_only_under_compile`, and a
`ModuleSelect` scope. `plan_quant` returns the convert set plus a **reason for
every module it keeps**. Refusals: an empty plan, a converter that matched
nothing, and a converter that matched some (a half-quantized model is never
served under the recipe's name). `scope_census` measures through tensor
subclasses, so a per-row fp8 weight is not priced as the dtype it advertises.

The preferred producer is ingest conversion — bytes at rest under the fp8
contract — which is a separate lane. This pass is the in-place fallback for a
lane still served from wide bytes, and it shares the recipe vocabulary with the
converter so the two cannot drift into two definitions of the same lane.
