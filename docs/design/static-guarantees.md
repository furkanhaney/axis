# Moving ML assumptions upward

The framework should progressively move research mistakes out of researcher
memory. The target is the highest **sound** enforcement layer for each fact:

```text
unrepresentable state
compile-time type error
model-build or configuration error
runtime assertion
periodic audit
documented assumption
```

Moving a rule upward is valuable only when the stronger layer proves the same
claim. A type that merely looks reassuring is worse than an honest runtime
receipt.

## Structural axes, dynamic extents

Axis role and axis extent are different facts. `Time` can be structural while
its current length remains runtime data. An eventual typed façade could express
`Tensor<(Batch, Time, Model)>`, reject `softmax::<Vocab>()`, and derive a named
reduction's output axes without requiring static batch or sequence lengths.

The current runtime `Axis`/`Shape` algebra comes first because it establishes
the semantics the type layer must preserve: identity rather than display name,
explicit roles for query/key time, unrelated-axis preservation, deterministic
result ordering, and no implicit outer products. Stable Rust makes arbitrary
tuple set operations, useful compile errors, and generic axis unions difficult.
An opt-in macro-generated or typed façade should follow concrete programs; it
must not shrink valid dynamic models merely to satisfy the type system.

Static extents can prove additional local equations such as
`model = heads * head_feature`, while dynamic extents should fail during
`model.build` before allocation or training. Both are stronger than discovering
the mismatch inside a CUDA kernel.

## Graph and experiment checks

Some claims concern a built computation or a set of run configurations rather
than one Rust expression. Examples include causal masking, next-token target
shift, compatible tokenizer vocabularies, warmup below the total step budget,
and equal-compute comparisons. These belong in model-build validation or an
ML-specific graph/configuration analyzer.

Declarations are not evidence by themselves. A `causal_language_model` label
must be connected to a graph witness, such as logits carrying a causal-attention
provenance state, and equal-compute comparison must be checked against resolved
token/FLOP budgets. Checks should report what artifact established each PASS.

## Runtime facts stay runtime

Shard reopening, worker overlap, resume position, observed sample reuse,
finite-corpus coverage, loss/gradient finiteness, and evaluation contamination
depend on execution. They require runtime ledgers. A required scientific regime
uses an aborting assertion; a diagnostic-only check must not be presented as a
guarantee. Expensive semantic deduplication can be a periodic audit with its
sampling and error model stated in the receipt.

`Idr` and `TrainEvalDisjoint` are current examples. They exactly prove claims
about caller-defined IDs. They do not prove independence, content novelty, or
latent support.

## Ownership is useful but not proof of uniqueness

A non-`Clone` `Fresh<Sample>` prevents one wrapper value from being submitted
twice along an ordinary ownership path. It cannot stop code from copying the
payload, reconstructing the sample, or receiving the same content under a new
identity.

A stronger future API can make `DataLoader` return a non-cloneable guarded batch
with private payload access and make `Trainer::step` consume it. The loader can
commit its IDs before releasing optimizer effects, so the supported path cannot
silently reuse a guarded batch. Distributed and resumed runs still need a
shared or persisted identity ledger. The receipt must continue to say
“observed ID nonreuse,” not “fresh information.”

## Promotion rule

Start with the cheapest exact assertion that protects a real experiment. When
multiple programs reveal a stable structural rule, move it into model build or
the type system and delete the weaker assertion. Keep empirical claims at
runtime. This lets the framework accumulate research methodology without
turning speculative rules into restrictive abstractions.

How these narrow checks compose into evidence and constrain scientific claims
is described in [run certificates](../contracts/run-certificates.md).
