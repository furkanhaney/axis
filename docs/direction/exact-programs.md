# Exact programs: what Axis needs next

Updated 2026-09-24. This spec lists what Axis is missing for its next class
of consumer: programs with exact discrete state and audits over declared
finite domains. Differentiability depends on the construction. Each item names its evidence, what
"done" means, and its status against `main`.

## The consumer

Three outside studies built this class of program on Axis 0.11.0 in
September 2026: a calculator, a text editor, and an app router over 100 small
apps. They share finite-domain cells and tensor wiring, but differ in how
they produce exact discrete outputs:

- **The app router's hand-set tables.** Each cell is a table of logits over a
  finite input domain, set by hand to `±1000` (one winner per row). A step is
  `contract` (a one-hot input selects its row exactly) followed by `softmax`,
  which at that scale returns an exact one-hot in f32 because `e^-2000`
  underflows to zero. The state outputs are exactly 0 or 1; logits and
  frame values are not restricted to those values. This path uses
  differentiable operations, but saturated softmax has zero gradients in
  f32; exact forward execution does not establish useful learning gradients.
- **The calculator and editor's trained cells and hard decisions.** The
  calculator uses trained cells followed by `max` and `ge(0.0)` to produce
  exact one-hots. The editor also uses hard comparisons, including `eq(0.0)`
  for cursor searches. Their audits support exact discrete behavior within
  their declared domains, not an end-to-end differentiable construction.
- **Fixed wiring.** Tables compose through `contract`, `select`-by-one-hot,
  shifts (`narrow` plus `pad_zeros`), masks and scans, written inside
  `forward` as ordinary tensor algebra.
- **The app router's exhaustive audit.** Every table row is checked for its margin (winner
  beats every loser by 2000), every cell is run on every member of its
  domain against a plain-Rust spec, and frames are compared byte for byte.
  When the joint state space is too large (about 10^35 at 100 apps), the
  audit is compositional: a step touches the router state and at most one
  expert. Composition also requires that inactive experts preserve their
  state and that rendering depends only on the router and selected expert;
  local audits alone do not prove those wiring properties.
- **Sparse dispatch.** The app router selects one expert. Exact gating,
  delivery only to the selected expert, and identity transitions for every
  inactive expert together justify equivalence to dense execution.

The consumer studies report: the calculator matched the real engine on
1,317,308 key presses; the text editor on 1,000,000 of 1,000,000 edit
events; the router audits 20,800 routing cases, 1,142 expert transitions,
153,600 clicks and 1,142 frames at 100 apps in 15.2s on the desk RTX 5060.
The calculator and editor event counts are sampled sequence checks, not
exhaustive enumeration of their joint state spaces.

Every gap below was hit by that work or is required by the next step:
`axis compile`, which takes a model graph and emits a certified executable.

## Status legend

- **fixed on main**: merged, not yet in a published release.
- **partial**: improved on main; needs re-measuring on the consumer's shape.
- **missing**: not started.

## Part 1: measured gaps

### E1. Gather and dispatch by a device-resident index

- **Need.** Select rows, or a whole expert's step, by an index or one-hot
  that lives on the device.
- **Evidence.** `Tensor::gather` takes `&[usize]` host indices. The app
  router reads the focus one-hot back to the host every step to pick the
  expert to run. The readback is exact (the focus is an exact one-hot), but
  it is a synchronization per step and blocks keeping dispatch on the device.
- **Done when.** `gather` (or a sibling) accepts a device tensor of indices,
  with gradients to the gathered rows, and the router's `forward` runs with
  no host readback.
- **Status.** missing.

### E2. Readback at memory speed

- **Need.** `to_vec` at copy speed, and readback in a caller-chosen axis
  order.
- **Evidence.** In 0.11.0 `to_vec` computed a coordinate `Vec` per element:
  about 150 MB/s, 6.8s of the router's 15.2s audit. `to_vec` also ignores
  `with_layout` (it returns declared order), and `Tensor::stack`'s position
  sets logical order, so both the calculator and the router wrote their own
  "read in this axis order" helper.
- **Done when.** Contiguous tensors copy straight through (done on main,
  #153); strided tensors use an incremental offset walk (done on main); a
  `to_vec_in(order)` (or equivalent) returns values in a requested axis order
  so consumers stop reimplementing it.
- **Status.** fixed on main for speed (#153); order API missing.

### E3. Fast large reductions

- **Need.** Reducing a large tensor to a few numbers at bandwidth speed.
- **Evidence.** On 0.11.0, reducing a `[20, 240, 320, 3]` tensor took about
  0.17s whether by `sum` or by `contract` against ones. Comparing frames on
  the device (one number back per batch) therefore measured slower (27.8s)
  than reading every frame back to the host.
- **Done when.** The same reduction runs at memory bandwidth, and the
  router's frame audit compares on the device faster than on the host.
- **Status.** partial: `da74ad4` parallelized the grouped reduction across
  chunks; re-measure on this shape.

### E4. Buffer lifetime without caller-inserted synchronization

- **Need.** Intermediate buffers freed when their last use completes, not
  only at `Device::synchronize`.
- **Evidence.** In 0.11.0, `Device::track` pushes every buffer to a pending list that
  only `synchronize` clears. The calculator grew about 30 MB per step until
  `forward` synchronized every step. The router's window held 3.4 GB after
  its startup audit, and the next process on the same 8 GB card failed with
  `Driver(DriverError(2, "out of memory"))`.
- **Done when.** A long loop of steps with no explicit synchronize holds
  flat memory, measured.
- **Status.** fixed on main for unused intermediate retention: completion
  events retire buffers and inference backpressure bounds the pending backlog.
  Trainer retains its single step boundary. An 8,192-step synthetic witness
  holds bounded memory; the consumer transformer remains to be re-measured.

### E5. Persistent kernel cache by default

- **Need.** No per-process recompilation of every tile shape.
- **Evidence.** 0.11.0 never enabled cuTile's disk cache, so each process
  recompiled every shape through a `tileiras` child: 29.1s before the
  calculator's first forward. Every consumer called
  `cutile::jit_cache::enable_default()` itself, pinning `cutile` just for it.
- **Done when.** Enabled by Axis, with stats exposed.
- **Status.** fixed on main (#154, `jit_cache_stats`). Consumers drop their
  own call after the next release.

### E6. The index-plan contribution limit

- **Need.** Gathers and index plans larger than 2^24 contributions.
- **Evidence.** Full-batch training of the calculator's display tile net
  (301,860 rows) needed about 309 million contributions, 18x over the
  `16,777,216` limit, and had to be split into mini-batches.
- **Done when.** The limit is lifted, or chunked transparently with the
  same results.
- **Status.** missing.

### E7. A host backend

- **Need.** Run the same tensor programs without CUDA.
- **Evidence.** Axis is CUDA-only, so every consumer test needs a GPU.
  Tests fail when another job holds the desk card (E4's out-of-memory was
  one), and CI needs a GPU runner for logic that is table lookups. A native
  code backend (X8) needs a host path regardless.
- **Done when.** `Device::cpu()` runs the exact-program operations
  (`contract`, `softmax`, `select`, `narrow`, `pad_zeros`, `rename`,
  `broadcast_to`, elementwise) with results equal to CUDA on the
  consumers' audits.
- **Status.** missing.

## Part 2: exact programs

### X1. Finite-domain axes

- **Need.** An axis can declare its domain: `digit` in 10 values, `key` in
  15, `flag` in 2, with one-hot as the carried representation.
- **Why.** Every item below needs to know which inputs are finite and how
  large their product is: table folding (X6), exhaustive audits (X2), gather
  lowering (X6), and noise-margin checks.
- **Done when.** A declared domain survives `contract`, `rename` and `role`,
  and an audit can enumerate it without the consumer spelling it out.

### X2. Audits and their receipts

- **Need.** First-class exhaustive audits: margin assertion over every table
  row, cell-versus-spec over a whole domain, compositional audits (router
  plus per-expert), byte or f32 frame equality, and a receipt.
- **Why.** The calculator and the router each reimplement this. Receipts
  are already Axis's pattern (`IdrReceipt`, `FinitePassesReceipt`,
  `LearningProgressReceipt`; [run certificates](../contracts/run-certificates.md)).
- **Receipt key.** The router's receipt cache (a pass is reused only when
  its key matches) hashes every uploaded constant, the composing and
  checking code, and the lock file. The Axis version should key on the
  captured graph (X4), constants, reference spec and checker identities,
  declared audit domain and configuration, and execution dependencies
  (Axis, backend/compiler versions, precision and relevant target settings).
  A change to any of these must invalidate reuse. Record rows checked, minimum
  margin, and failures (always zero, since failures are never stored).
- **Done when.** A consumer's audit is a declaration plus a spec function,
  and its receipt composes into a run certificate. Changing only the spec,
  checker, domain or execution dependencies must force a fresh audit.

### X3. A step module

- **Need.** A standard `(state, event) -> (state, frame)` recurrence with
  explicit state fields, per-step detach, and buffer lifetimes (E4).
- **Why.** The calculator's `CalculatorState` and the router's `State` are
  the same hand-written pattern, including a `detach` that exists only
  because the backward graph otherwise grows across steps.
- **Done when.** Both consumers express their machines through it with no
  hand-written detach.

### X4. Graph capture

- **Need.** Trace a `forward` into a graph Axis can inspect and rewrite.
- **Why.** Axis executes eagerly. Every compiler pass in X6 rewrites a graph,
  and a receipt key (X2) is best taken over a graph, not over source text.
- **Done when.** A captured graph replays with results equal to the eager
  run, and exposes op, axis and domain information to passes.

### X5. Max-plus contraction

- **Need.** `contract` in the log semiring: addition in place of multiplication
  and `logsumexp` in place of summation. Max-plus uses `max` instead of
  `logsumexp` and is a distinct operation.
- **Why.** "Contract then softmax" is exact only because the consumers'
  inputs are exact one-hots. Soft inputs, and cells trained at a lower logit
  scale, can use contraction in log space. Scale 1000 alone does not make
  log-sum-exp equal max: `logsumexp(1000, 1000)` is about `1000.6932` in f32.
  Replacing it with max requires a verified unique-winner gap sufficient
  for the actual term count, precision and reduction implementation to
  eliminate the correction. Log-sum-exp gradients distribute over terms;
  max-plus follows a unique winning path and needs an explicit tie-gradient
  policy. Forward equality alone does not certify backward equivalence.
- **Done when.** A log-semiring `contract` with gradients, equal to its
  reference on the consumers' tables at scale 1 and scale 1000, including
  tied and near-tied inputs. Any max-plus lowering separately verifies its
  margin precondition and states its forward and backward guarantees.

### X6. Compiler passes

- **Need.** Passes over captured graphs (X4) with declared domains (X1):
  - exhaustive evaluation: a subgraph over a small finite domain is run once
    on every input and replaced by its table;
  - one-hot times table to gather: the largest single speedup available;
  - contraction ordering: contract one factor at a time, never materializing
    the product of all input domains (the calculator's out-of-memory came
    from exactly that);
  - scan recognition: an associative chain, such as carries, becomes a
    parallel prefix scan;
  - batching, liveness, fusion.
- **Why.** Each pass was found by hand in the calculator, one measured
  failure at a time. A compiler applies them every time.
- **Done when.** The hand-built calculator module is reproduced from its
  engine's semantics by passes, matching its tables, speed and equivalence
  run.

### X7. Importers

- **Need.** ONNX and `torch.export` (`.pt2`) graphs into Axis, with names
  assigned to positional dimensions by shape inference.
- **Why.** `axis compile model.onnx` starts here. Plain `.pt` files are
  pickles, which execute code on load; refuse them.
- **Done when.** An exported calculator graph imports and passes the same
  audit as the Axis-built module.

### X8. A native code backend

- **Need.** Emit Rust (or C) from a certified graph: tables as arrays,
  one-hot contractions as indexed loads.
- **Why.** The calculator module takes about 130 ms per key on a GPU at batch
  one; the same tables as array lookups are microseconds. This closes the
  loop: a program becomes a graph for learning and checking, and a program
  again for running.
- **Done when.** The emitted calculator passes the same audit and runs at
  native speed.

### X9. Per-row scale

- **Need.** A table whose rows carry their own logit scale.
- **Why.** One router can hold hand-set exact rows (scale 1000) beside
  learnable rows (low scale) for fuzzy intent, so a new route learns while
  certified routes stay exact and keep their receipts.
- **Done when.** Gradients reach low-scale rows and the audit's margin check
  reads per-row scale.

## Order

1. **Release what main already fixed** (E2 speed, E5), then **E1 and E4**:
   small, and both block the router past 100 apps. Re-measure **E3**.
2. **X1, X2, X3**: the consumers' hand-written audits, receipts and state
   machines become Axis features.
3. **X4, X6, X7, X8**: `axis compile`. Each depends on X1 and X2.
4. **E7** before X8 at the latest; **E6** and **X5, X9** when a consumer
   needs them.

## Out of scope

- Arbitrary Rust as compiler input. A program enters as a graph (X4, X7),
  or as a restricted form with bounded loops, fixed-size state and declared
  domains; outside its declared domain its behavior is undefined, and the
  receipt says so.
- Learning programs end to end through long discrete chains. Cells are
  learned (or set) locally against their own domains and composed frozen;
  gradient-trained program induction through long chains has a poor record.
