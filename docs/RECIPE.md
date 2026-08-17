# Recipe v1

A recipe states one family's **composition**: which graph classes an endpoint's
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
       "class_hash": "2f91b8c40ae7d135",
       "ingress_digest": "…32 hex…",
       "ingress": { /* the exact CallIngress v1 value */ }}
    ]}
  ],
  "loop": [
    {"runner": "text_encoder", "repeat": "once"},
    {"runner": "denoiser", "repeat": "counted", "parameter": "steps"},
    {"runner": "decoder", "repeat": "once"}
  ],
  "parameters": [{"name": "steps", "minimum": 1, "maximum": 100}],
  "scheduler": {"name": "euler_discrete", "parameters": {"shift": 3.0}}
}
```

- **Family composition** is `runners`: an author-facing handle, the bucket axes
  it varies on, and one variant per bucket. A variant pins a class by its 16-hex
  `class_hash` — the key's `graph` axis — plus the exact `CallIngress` v1 value
  and its digest.
- **Loop structure** is `loop`: an ordered list of stages, each naming a runner
  and running `once` or `counted` by a declared integer parameter. There are no
  conditionals, expressions, or data-flow wiring; a stage's inputs and outputs
  are host code. The order is significant and is the only thing the loop states
  besides repetition.
- **The scheduler block** is a name plus finite JSON scalars. torchcg validates
  its shape and never interprets it: the host implements the named scheduler.
  It rides inside the digest because two otherwise identical compositions under
  different schedulers are different pipelines.

Display class names are deliberately absent. Identity is `class_hash` +
`ingress_digest`; a cosmetic rename must not move a pin that consumers build
against.

## Identity and the key

**A recipe never carries a `cg-key-v1` value.** It pins `graph` only. The exact
key is folded at adopt time from the pod's own axes:

```text
cg-key-v1 = key({graph: <variant class_hash>, sm: <pod>, toolchain: <pod>})
```

so one recipe digest is valid on every SKU and toolchain, and a re-mint changes
keys without touching author source (§4.27, §4.29). `GraphClassVariant.key()`
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
variant. So `Literal[64, 128]` in Python or an enum in Rust is exhaustive and
every selection resolves. A gap is refused as `bucket_coverage_incomplete`.

**G7 — Names never key, so no generated code holds a string lookup.** Runner
handles resolve through the recipe at **generation** time to `class_hash` +
`ingress_digest`. Generated code carries the identity, not the name.
`Recipe.runner(name)` exists for the generator and for diagnostics; a handler
that reaches a runner through a string it typed is exactly what codegen makes
unrepresentable.

**G8 — Bucket lookup is exact, never ranked.** `RecipeRunner.variant(bucket)`
matches a bucket exactly or refuses. Choosing which bucket serves a live call is
`ingress_selection_v1` (tcg#37), a separate contract — this one never
approximates.

**G9 — Incompatible re-mints break builds, not pods.** The recipe digest moves
whenever any class hash, ingress, bucket set, loop, or scheduler block moves. A
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
checkpoint-free.** Buckets, graph classes, signatures, the loop, and the
scheduler block are family facts, identical for every checkpoint that family
serves — which is what lets one cell serve sixteen fine-tunes (§4.27). Weight
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

## What this contract deliberately does not say

Ingress ranking and feed normalization (`ingress_selection_v1`, tcg#37), eager
fallback and sticky de-arm, residency and execution groups, hub communication,
scheduler mathematics, and anything that would make this a language.

It also says nothing **checkpoint-level**, by construction (G11): weight sets and
checkpoint refs, the tuned-value struct and its schema (declared on the family in
the SDK, stamped per release slot), and per-request defaults. A family is the
graph level; an instance is family × weight-set; a request is neither. Those
three axes do not cross, and this document is entirely the first one.
