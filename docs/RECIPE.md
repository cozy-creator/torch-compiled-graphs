# Recipe v1

A recipe states one family's **composition**: which graph specializations an endpoint's
compiled pipeline is made of, the loop between them, and the scheduler block
that loop runs under. It is a **vocabulary, not a DSL** — it names composition,
it does not program it. Payload parsing, CFG policy, residency, retries,
telemetry, ingress ranking, and eager fallback are host code by definition, not
vocabulary extensions. Anything the vocabulary cannot say is host code.

`torchcg.recipe` is the reference implementation; `contracts/recipe_v1.json`
is the corpus a non-Python consumer reads.

## The document

```json
{
  "v": 1,
  "family": "toy_diffusion",
  "buckets": [{"name": "resolution", "values": [64, 128]}],
  "runners": [
    {"name": "denoiser", "axes": ["resolution"], "variants": [
      {"bucket": {"resolution": 64},
       "layout": "bf16",
       "specialization_hash": "2f91b8c40ae7d135",
       "ingress_digest": "…32 hex…",
       "ingress": { /* the exact CallIngress v1 value */ }}
    ]}
  ],
  "loop": {
    "kind": "staged",
    "session_state": "none",
    "stages": [
      {"runner": "text_encoder", "repeat": "once"},
      {"runner": "denoiser", "repeat": "counted", "parameter": "steps"},
      {"runner": "decoder", "repeat": "once"}
    ]
  },
  "parameters": [{"name": "steps", "minimum": 1, "maximum": 100}],
  "scheduler": {"name": "euler_discrete", "parameters": {"shift": 3.0}}
}
```

- **Family composition** is `runners`: an author-facing handle, the bucket axes
  it varies on, and one variant per bucket per layout. A variant pins a class by
  its 16-hex `specialization_hash` — the key's `graph` axis — plus the exact `CallIngress`
  v1 value, its digest, and the tensor-layout contract it was traced against.
- **Loop structure** is `loop`: a `kind`, a `session_state` owner, and an
  ordered list of stages, each naming a runner and running `once` or `counted`
  by a declared integer parameter. There are no conditionals, expressions, or
  data-flow wiring; a stage's inputs and outputs are host code. The order is
  significant and is the only thing the loop states besides repetition.
- **The scheduler block** is a name plus finite JSON scalars. torchcg validates
  its shape and never interprets it: the host implements the named scheduler.
  It rides inside the digest because two otherwise identical compositions under
  different schedulers are different pipelines.

Display class names are deliberately absent. Identity is `specialization_hash` +
`ingress_digest`; a cosmetic rename must not move a pin that consumers build
against.

## Tensor layout is an axis of the specialization row

An fp8-rowwise trace and a bf16 trace are **different graphs**, so `layout` is a
fact of the specialization row, not a property of weights bound to it later. Every variant
names the layout contract it was traced against; a runner may offer several, and
bucket coverage must be total **per layout**, so a generated closed type stays
exhaustive whichever layout is in play.

**What this contract does with a layout: records it.** It does not enumerate the
token set, interpret it, or fork §1.32/§1.33's vocabulary, and it produces no
verdict. The recipe and per-checkpoint layout metadata are deliberately
**separate hub documents** — they have different axes and different lifecycles: a
re-mint regenerates recipes, a new checkpoint adds layout rows, and merging them
would rekey every family document on every upload. The hub **joins** them at
rebind and at invoke to answer COMPATIBLE / CONVERTIBLE / PRODUCIBLE /
INCOMPATIBLE ahead of time, extending §1.32(b) to the compiled path.

The runtime consequence, stated here because it is what makes the row load-
bearing: **a compiled backing accepts exactly its traced layout.** An eager
backing follows the fit ladder, which is host policy. `variant(bucket, layout)`
therefore refuses when a runner has more than one layout and none is named — the
choice belongs to that join, not to a lookup.

## `loop.kind: host` — the loop the vocabulary refuses to fake

LLMs and VLMs are first-class families, and their iteration is **data-dependent**:
it runs until the model says stop. No count in a document can say that. So a
loop declares one of two kinds:

- **`staged`** — the composition the vocabulary fully describes. Stages run
  `once` or `counted` by a bounded parameter, in the stated order.
- **`host`** — an autoregressive family. The recipe states everything it *can*:
  the per-step graph specializations in order (`prefill`, then `decode`), and
  `session_state`, which names who owns the state threaded between steps
  (`none`, `host` when the caller passes the KV cache in and out as tensors, or
  `graph` when the artifact carries it). It then says outright that the
  iteration itself is the host's. **A repeat count under a host loop is refused**
  (`loop_invalid`).

The corpus ships one of each: `recipe` is staged, `host_loop_recipe` is an AR
family. The vocabulary never pretends to describe a loop it cannot express —
that refusal is the feature, because a fabricated count would be read by a
second implementation as a real bound.

## Identity and the key

**A recipe never carries a `cg-key-v1` value.** It pins `graph` only. The exact
key is folded at adopt time from the pod's own axes:

```text
cg-key-v1 = key({graph: <variant specialization_hash>, sm: <pod>, toolchain: <pod>})
```

so one recipe digest is valid on every SKU and toolchain, and a re-mint changes
keys without touching author source (§4.27, §4.29). `GraphSpecializationVariant.key()`
takes any object with `sm` and `toolchain` — `RuntimeCompatibility` satisfies it
structurally, which is why this module never imports Torch.

## Digest, and how it rides beside `endpoint.lock`

```text
recipe_digest = first 32 lowercase hex of SHA-256(canonical JSON(document))
canonical JSON = sorted keys, compact separators, ASCII, finite
```

the same rule `CallIngress` v1 uses. The digest covers every field of the
document, including the embedded ingress values, and covers no machine axis.

**The recipe SITS BESIDE the lock; it never rides inside it.** `endpoint.lock`
states discovery and is rewritten per deployment; the recipe is
machine-independent and content-addressed. Restating its body in the lock would
create a second identity that can disagree with the first. The lock carries one
reference:

```json
{"family": "toy_diffusion", "v": 1, "digest": "…32 hex…"}
```

`Recipe.reference()` emits it and `Recipe.verify(reference)` refuses a stranger.
Where that field lives in the lock is the worker's schema decision, not this
package's.

## Typed-binding generation requirements

These are the contract for a binding generator (pgw#1332 in Python, `build.rs`
in Rust). They are numbered so a generator can cite them.

**Read G16 first — it fixes the direction.** Typed bindings are generated from a
**declaration-time fake-tensor export**, not from this document. The recipe a
mint emits is the **drift assertion** against that declaration and the
**adopt-time reference** from a name to a class identity. Everything below is
therefore two things at once: the facts a binding needs, and the facts this
document must state identically for the assertion to mean anything.

**G1 — Names are generatable by construction.** Every family, runner,
bucket-axis, call-parameter, scheduler, and scheduler-parameter name matches
`[a-z][a-z0-9_]*` in 1..48 bytes and is not reserved in Python or Rust. The
reserved list is frozen data in the corpus, not the running interpreter's
keyword list. A generator therefore owns **no** mangling or escaping rule; a
name it cannot emit was refused at the recipe.

**G2 — One runner generates one binding.** All variants of a runner share one
`CallSignature`: identical parameter axis, flat arity, excluded inputs, leaf
names, positions, paths, exported names, dtypes, and ranks. They may differ
**only** in concrete dimensions and symbol bounds. A recipe that violates this
is refused as `signature_disagreement`.

**G3 — Parameter order is the ingress order.** The generated callable's
parameters are exactly `ingress.parameters`, in that order; leaves within a
parameter are ordered by `position`. Generators never sort by name.

**G4 — Leaf kinds are projected, not re-derived.** `call_signature()` gives each
parameter one of `tensor`, `mapping`, `sequence`, `absent`, derived from its
leaves' paths. `mapping` keys and `sequence` indices come from the path steps.
An `absent` parameter contributed zero leaves at export: it is **still emitted**
in the generated signature (a dual-backed callable's eager and compiled backings
must be one callable), keyword-only, non-feeding. **Known limit of ingress v1:**
it does not record the pinned value of a zero-leaf argument, so a generator
cannot constrain that parameter's type and emits it unconstrained. Do not invent
a value for it.

**G5 — Shapes are runtime facts; ranks and dtypes are compile-time facts.**
Concrete dimensions and symbol names live per variant in its ingress; every
symbol carries finite inclusive bounds (`CallIngress` refuses an unbounded or
unreferenced symbol). Generators emit dtype and rank into the type and may emit
bounds as runtime assertions. A symbolic dimension is never a type.

**G6 — Bucket sets generate closed types, and coverage is total.** A runner's
axes are a sorted subset of the family axes; a variant pins exactly those axes
with values the axis declares; and every combination of its axes' values has a
variant **at every layout that runner offers**. So `Literal[64, 128]` in Python
or an enum in Rust is exhaustive and every selection resolves. A gap is refused
as `bucket_coverage_incomplete`.

**G7 — Names never key, so no generated code holds a string lookup.** A runner
handle is resolved to `specialization_hash` + `ingress_digest` **before anything runs** —
by the declaration at generation time, and by this document at adopt time — so
generated code carries the identity, never the name. `Recipe.runner(name)` exists
for that resolution and for diagnostics; a handler that reaches a runner through
a string it typed is exactly what codegen makes unrepresentable.

**G8 — Lookup is exact, never ranked.** `RecipeRunner.variant(bucket, layout)`
matches both exactly or refuses. Choosing which bucket serves a live call is
`ingress_selection_v1` (tcg#37) and choosing a layout is the hub's join — both
separate contracts. This one never approximates.

**G9 — Incompatible re-mints break builds, not pods.** The recipe digest moves
whenever any specialization hash, ingress, bucket set, loop, or scheduler block moves. A
consumer embeds the digest-pinned reference at build; regenerating against a
changed recipe fails compilation. Nothing arms wrong on a pod because the
generated signature and the artifact's admission expectation derive from the
same ingress digest — the type checker and admission enforce one identity from
two directions.

**G10 — Refusals are a closed, named set.** `RecipeRefusal` enumerates every
reason a document is refused; the corpus ships the list with a note per reason,
and each is reachable from a test. A second implementation reports the same
reason for the same document.

**G11 — The recipe is the CLASS-LEVEL layer and is structurally
checkpoint-free.** Buckets, graph specializations, signatures, the loop, and the
scheduler block are family facts, identical for every checkpoint that family
serves — which is what lets one graph serve sixteen fine-tunes (§4.27). Weight
sets, checkpoint refs, tuned values, and any per-request default are
**unrepresentable**, not merely discouraged: every object in v1 has a closed
field set, so a document carrying one is refused as `recipe_fields_invalid`.
A generator must not synthesize such a field from anything here.

**G12 — The generated family class is the TYPE; the injected value is an
INSTANCE.** Codegen emits, per family, one class holding only class-level facts
— runner call signatures, bucket literal types, class identities, the loop and
scheduler blocks. The typed runner callables are reached through a
**FamilyInstance** (family × weight-set), because a call needs bound constants
and this document deliberately names none. A handler parameter is annotated with
the family class and receives a fully resolved instance; the generator emits no
constructor that would let a family be called without one.

**G13 — Two parameters of one family are two independent instances.** Nothing
codegen emits may hold per-instance state at module or class level: class-level
output is immutable identity (hashes, digests, literal bucket sets, signatures),
and everything checkpoint-level lives on the instance the SDK constructs. So
`flux_a: Flux1Dev, flux_b: Flux1Dev` are two checkpoints with independent
weights and independent tuned values, sharing one artifact and one generated
type. A generated singleton, module-level registry, or process-global default
breaks this and is a codegen defect, not a recipe extension.

**G14 — A host-owned loop generates the same typed callables, and no iteration.**
`loop.kind` is a generated class-level fact, not a behaviour: for `host`, a
generator emits the per-step runners' typed callables and the `session_state`
owner, and emits **no** driver, no step count, and no termination condition. An
AR family's iteration is host code. A generator that synthesizes a loop bound
for a host loop has invented a fact the recipe deliberately refused to state.

**G15 — Layout is generated as a second closed axis, never inferred.** A runner's
layouts are a closed set exactly as its bucket values are, and coverage is total
per layout, so `Literal["bf16", "fp8_rowwise"]` or a Rust enum is exhaustive.
A generator emits the traced layout on each class and emits **no** conversion,
fallback, or preference between layouts: a compiled backing accepts exactly its
traced layout, and choosing one is the hub's join with per-checkpoint layout
metadata.

**G16 — The declaration is the binding source; the recipe is the assertion.**
Bindings generate from a declaration-time fake-tensor export, so a family types
correctly before anything is minted and codegen never waits on a compile. The
mint emits this document, and `Recipe.assert_declaration(...)` refuses
(`declaration_drift`) when the runners or their ingress digests differ from what
the bindings were generated against — a missing runner, an extra one, or a class
minted against a different call. Equal ingress digests imply equal signatures,
because the signature is a projection of the ingress. A generator that reads this
document *instead of* the declaration has inverted the dependency and loses the
assertion: there is then nothing left to compare against.

## What this contract deliberately does not say

Ingress ranking and feed normalization (`ingress_selection_v1`, tcg#37), eager
fallback and sticky de-arm, residency and execution groups, hub communication,
scheduler mathematics, an autoregressive loop's termination condition, and
anything else that would make this a language.

It also says nothing **checkpoint-level**, by construction (G11): weight sets and
checkpoint refs, per-checkpoint tensor-layout metadata (a separate hub document
it is joined with, never merged into), the tuned-value struct and its schema
(declared on the family in the SDK, stamped per release slot), and per-request
defaults. A family is the graph level; an instance is family × weight-set; a
request is neither. Those three axes do not cross, and this document is entirely
the first one.
