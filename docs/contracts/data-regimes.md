# Executable data-regime assumptions

Research code changes meaning when its data regime changes. A step-count edit
can turn a fresh-sample experiment into repeated training over a finite corpus
without changing the model code. `axis` therefore exposes two small
invariants underneath the batching and training-loop APIs.

## Single pass is a budget

```rust
let mut pass = SinglePass::new(dataset.len())?;

for batch in loader {
    pass.consume(batch.len())?;
    // train on the batch
}
```

`SinglePass` proves `samples_consumed <= samples_available`. It cannot prove
that the loader supplied distinct rows inside that budget, so its documentation
says exactly that. Use it when loader uniqueness is established elsewhere and
the dangerous change is silently crossing an epoch boundary.

## IDR observes identities

The infinite-data regime (IDR) is an operational approximation, not a physical
property. A finite corpus can declare limits for coverage and observed reuse:

```rust
let limits = IdrLimits::fixed(0.10, 0.001)?;
let mut idr = Idr::fixed(dataset.len(), limits)?;

for batch in loader {
    idr.observe(batch.sample_ids.iter().copied())?;
    // The assertion runs before these samples affect an optimizer step.
    train(batch)?;
}
```

This means at most 10% consumed samples per available corpus row and at most
0.1% repeated observed IDs. Coverage counts every consumed example, including
repeats. Repeat rate is `repeated observations / consumed observations`.
The first observation of an ID is unique; every later observation is a repeat.

For generated streams, corpus coverage has no honest denominator:

```rust
let mut idr = Idr::generated(IdrLimits::generated(0.0)?)?;
idr.observe(trajectory_ids)?;
```

The caller defines identity. A dataset normally supplies stable row or example
IDs. A simulator might use a trajectory ID plus step, while a synthetic
generator might use a unique draw counter. Hashes are acceptable only when the
experiment accepts their collision model. The library does not hash tensors or
guess whether two semantically identical examples count as reuse.

Tracking is exact and stores every distinct `u128` ID, so memory grows with the
number of unique observations. Generated sources can namespace the draw counter
with a seed, worker, or shard identity. Approximate cardinality or duplicate
sketches should be added only for a concrete experiment that cannot afford
exact tracking, with their false-positive and false-negative behavior in the
declared contract.

## Sources and batches

`DataLoader<S>` batches a `DataSource`; the source defines whether it can end
and what each sample ID means. `InMemoryDataset` is finite by default and emits
a short final batch. Calling `.repeat()` is the explicit act that makes it wrap.

`FinitePassesLoader` represents intentional finite-data reuse without calling
it IDR. It declares a population, batch size, exact pass count, and shuffle seed.
Each pass contains every stable sample ID exactly once, short final batches do
not cross pass boundaries, and later passes must contain the same population as
the first. Exhaustion verifies that all declared passes completed:

```rust
let mut batches = FinitePassesLoader::new(samples, 512, 20, seed)?;
while let Some(batch) = batches.next_batch()? {
    train(batch)?;
}
println!("{}", batches.receipt());
```

This receipt makes reuse explicit: a 60,000-example population over 20 passes
means 1,200,000 observations. It proves identity coverage and pass boundaries,
not randomness quality or independence between shuffled orders.

`AdditionDataset` is the first generated source. It deterministically produces
fresh addition problems forever from a seed and a monotonically increasing
draw counter. Its `u128` sample ID is `(seed << 64) | counter`:

```rust
let source = AdditionDataset::new(42);
let limits = IdrLimits::generated(0.0)?;
let mut batches = DataLoader::new(source, 256)?.assert_idr(limits)?;

let batch = batches.next_batch()?.expect("generated source never ends");
println!("{}", batch.regime); // IDR STATUS ... PASS
```

The stream has no epoch and no finite coverage denominator. Its IDs do not
repeat before counter exhaustion, but projected f32 operand values can coincide;
unique draw identity is not a proof of rich latent support. `DataLoader`
selects finite or generated IDR semantics from `source.available()`. A batch is
returned only after its regime assertion passes, and carries the corresponding
structured receipt.

`Disjointness` is a separate multi-population contract. The addition program
canonicalizes a problem as its ordered pair of operand bit patterns, registers
the evaluation population first, then registers every fresh training batch
before optimization. This catches a repeated problem even when it arrives under
a different generator draw ID. The versioned identity scheme and exact evidence
boundary are described in [semantic disjointness](disjointness.md).

`observe` records the batch and then checks the limits. A failing snapshot
therefore remains available for diagnostics. The error reports availability
when known, consumption, unique and repeated IDs, coverage, repeat rate, and
the declared limits. Empty observations are allowed; zero-sized corpora and
limits outside `0..=1` are rejected.

The addition demonstration consumes a new generated batch on every optimizer
step and runs `assert_idr` in its loader. The existing MLP, CNN, and attention
demonstrations intentionally optimize the same small synthetic batches
repeatedly. They are finite-data learning checks, not IDR experiments, and do
not instantiate these guards.

`Trainer` removes only error-prone update ordering. Its loss closure remains
explicit and must return a scalar after named reductions:

```rust
let mut trainer = Trainer::new(SGD::new(0.5)?);
let report = trainer.step(&mut model, |model| {
    model.forward(&x)?.squared_error(&y)?.mean([batch, output])
})?;
println!("{} {}", report.step(), report.pre_update_loss()?);
```

It performs zero-grad, a fresh forward/loss closure, backward, and optimizer
step in that order. `TrainStep` names its value `pre_update_loss`; reading it
is optional and synchronizes the GPU. Data regime evidence belongs to the
delivered batch rather than the Trainer, so generated and finite sources remain
independent of optimizer choice.

## IDR is relative to a budget and a unit

“Infinite” is not an intrinsic label on a corpus. A corpus is operationally in
IDR for a particular experiment when its usable population is large relative
to the experiment's declared consumption budget and reuse stays within the
declared limit. A language model may therefore need several simultaneous
ledgers: tokens consumed, sequence IDs, document IDs, shard leases, and worker
assignments. One counter cannot turn freshness into independence or information
novelty.

The current receipt proves only the claims it measures: finite coverage when a
denominator is known and exact reuse of caller-defined sample IDs. It does not
claim IID sampling, semantic deduplication, rich latent support, worker
coordination, or train/evaluation disjointness. Common Crawl adapters should add
those as explicit source-level evidence rather than make `assert_idr` imply
them. In particular, distinct windows from one document may be physically fresh
and strongly correlated, while copied pages may carry distinct IDs and little
new information.
