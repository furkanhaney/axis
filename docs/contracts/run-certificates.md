# Runs as evidence

A training process can execute successfully after it has stopped supporting the
researcher's intended interpretation. The framework therefore treats tensor
correctness, execution-history correctness, and scientific claim support as
different layers.

## Intended and realized run space

An experiment declaration identifies a constrained region of possible runs:
architecture, data history, objective, optimizer, compute budget, precision,
evaluation protocol, seed policy, and other coordinates. Runtime evidence asks
whether the realized run remained inside that region. Leaving it is meaningful
even when loss continues to decrease.

Named axes remove ambiguous tensor programs. Data-regime guards constrain
sample history. Future graph/configuration checks can constrain target shifts,
masking, budgets, and evaluator provenance. These mechanisms are
interpretation constraints: they remove alternative explanations for an
observed result.

## Certificates are composed from narrow receipts

A run certificate should combine evidence with explicit proof boundaries:

```text
objective
    categorical class axis reduced        VERIFIED by graph operation

data history
    finite population                     60,000 stable IDs
    complete shuffled passes              20 VERIFIED
    total observations                    1,200,000

optimizer
    Adam equations/defaults               DECLARED + IMPLEMENTATION TESTED
    update state remained device-resident VERIFIED by execution path

evaluation
    train/evaluation ID overlap           0 VERIFIED
```

`IdrReceipt`, `FinitePassesReceipt`, and `DisjointnessReceipt` are the first
certificate fragments. None uses a broad PASS to imply unmeasured properties.
Disjointness records the versioned identity scheme and says
`VerifiedObservations`; exact canonical-key separation does not prove
distributional independence, and finite-pass accounting does not prove shuffle
quality.

## Claims declare prerequisites

The framework cannot safely derive broad scientific conclusions from metrics.
Instead, a claim should name the evidence it requires. A future claim checker
can reject “architecture B wins at equal compute” when the two certificates do
not establish equal compute, while retaining the narrower observation that one
realized run achieved lower evaluation loss.

This is claim degradation rather than binary run validity. Violating a
comparison prerequisite can invalidate a causal comparison without erasing all
measurements. Diagnostics should report:

1. the observed fact;
2. the declared prerequisite it contradicts;
3. the interpretation that is no longer supported;
4. the narrower evidence that remains;
5. the explicit mechanism for changing the declaration when intentional.

Severity follows meaning: logical contradictions are errors, violated declared
assumptions abort, unusual conditions warn, useful measurements are notes, and
expensive empirical checks are audits. The framework must not promote a style
preference into an invariant.

## Defaults come from the research corpus

The research repository contains a large population of real training programs.
That is the source for defaults. Each migration records:

- repeated choices that worked across programs;
- scientifically meaningful variations that must stay explicit;
- one-off mechanics that should disappear into the library;
- overrides and the reason each experiment needed them.

A choice becomes a default after repeated consumers establish a stable center
and the exceptions are understood. Standard Adam beta/epsilon values are a
reasonable constructor default while learning rate remains explicit. Loss
reduction axes remain explicit because changing them changes the objective.
Finite reuse and IDR remain separate APIs because neither is a harmless
spelling preference.

The goal is few overrides for ordinary experiments and conspicuous declarations
for choices that change interpretation. A census across migrations should
precede new presets; frequency alone does not make a scientifically meaningful
choice safe to hide.

The current descriptive baseline is generated at
[data/default-census.md](../../data/evidence/default-census.md). It covers more than 433,000
active Python lines and reports extraction limits beside the counts.
