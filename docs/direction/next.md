# Continue from the working examples

Updated 2026-09-22. This note preserves the decisions and next useful work;
[library.md](../design/library.md) is the contract, and [api-sketch.md](../design/api-sketch.md) is the
owner's larger, partially unimplemented API sketch. The
[module backlog](module-backlog.md) uses the `torch.nn` catalog to prevent
ordinary layer families from being forgotten while retaining Axis's named-axis
contracts and evidence bar.

## What exists and why

- `src/library/` is the reusable crate. Introductory programs live under
  `src/examples/getting-started/`; ordinary model and optimizer flows live under
  `src/examples/training/`; research contracts and ports live under `src/studies/`.
  Launch from the Axis node, not a `src/`
  corridor. Shared CUDA setup and
  Cargo/check runners stay at the workspace's `scripts/`.
- MLP, attention, and CNN consume the library. CNN also retains its explicit
  cuTile baseline; the two programs share only their independent scalar oracle.
  Each consumer owns its independent scalar reference and program-specific
  tests. The library must not import its consumers to obtain an oracle.
- `Axis` is identity, `Dim` adds a tensor-local extent, and Layout is separate.
  Same-name axes are distinct. `role` creates a fresh identity with ancestry;
  it does not authorize implicit alignment. Attention now exercises this.
- Preserve the explicit loss: forward, unreduced loss, and named `mean` remain
  in each program. `Trainer` owns the invariant update ordering: zero gradients,
  construct a fresh scalar loss, backward, then optimizer step.
- `build` binds extents, validates, allocates, and initializes without a dummy
  forward. A compatible second build preserves parameters. Structural names
  such as `0.weight` and `query.weight` identify slots; runtime `ParamId`
  identifies shared storage. Names are not yet a serialization format.
- `SinglePass` and `Idr` make data-regime assumptions executable without
  pretending exact identity proves statistical independence. IDR observes caller-defined stable IDs; it does not
  infer semantic duplicates from tensor values. Exact tracking grows with the
  number of unique samples. See [data regimes](../contracts/data-regimes.md).
- `DataLoader` accepts finite or generated `DataSource`s and withholds a batch
  when its regime fails. `AdditionDataset` is the first inexhaustible source.
  `Trainer` captures only update ordering; MLP, attention, and CNN still state
  their forward, loss, and named reductions directly.
- Attention composition stays in `src/examples/training/attention/src/attention.rs`. Only its
  required tensor operations, causal mask and softmax, entered the library.
  A second consumer can justify promoting the composed module later.

## Evidence and limits

- MLP's 500-step held-out MSE matched the retained baseline at `5.74e-2`;
  [MLP records](../../src/examples/training/mlp/README.md#library-mlp). Named loading and pre-update
  logging passed the subsequent [feedback checks](../../data/evidence/feedback-verification.log).
- Attention's independent f64 forward and central differences pass for every
  Q/K/V element, including reordered physical storage and a leading logical
  feature axis. Causality and large-logit softmax checks pass. Its 200-step
  held-out MSE falls from `5.12e-1` to `2.39e-2` on toy prefix means;
  [training record](../../src/examples/training/attention/data/runs/training.log).
- CNN matches 4,096 scalar activation values, pooled features, logits, every
  parameter gradient, overlapping input gradients, and one update. At 100
  steps its held-out BCE falls from `7.04e-1` to `7.40e-2` with `100.00%`
  accuracy; [CNN record](../../src/examples/training/cnn/data/runs/library-training.log). Its Conv2d is
  `unfold2d` plus Linear, so `col2im` is the reverse of the gather rather than
  a separate special-case kernel.
- Addition consumes 128,000 generated samples with zero observed identity
  reuse and zero train/evaluation overlap. Its held-out MSE falls from
  `8.06e-1` to `3.88e-3`; [addition record](../../src/studies/contracts/generated-addition/data/runs/training.log).
- The first outside consumer now preserves categorical cross-entropy, Adam, and
  exact finite passes; its smoke reaches `33.98%` held-out MNIST accuracy.
  The population migration now runs independent parameters, gradients, Adam
  state, and learning rates along one named population axis. It exposed the
  correctness-first contraction lowering as the remaining throughput limit.
  The panel migration drove device-resident AdamW and preserves its
  preprocessing/split mechanics without claiming a completed GDP fit.
- Current cuTile lowering enqueues one eager step on a CUDA stream, retains its
  buffers, and synchronizes once at the Trainer boundary. GPU arithmetic and
  derivatives are real, but no broad speed claim
  follows: single-axis contractions use batched tiled matrix multiplication;
  multi-axis contractions still use gather/reduce plans. Softmax performs one
  tiled forward or backward reduction per row, including non-power-of-two
  widths, instead of recomputing the row reduction for every output. Generic
  plans retain their 16,777,216-contribution cap. BF16 matrix products with
  FP32 accumulation/state are explicit; there is no retained graph, CPU
  fallback, or higher-order differentiation.
- The outside Sudoku acceptance passed every host and CUDA oracle on an RTX
  5090, then trained over 1,600 fresh boards with zero observed reuse or
  train/evaluation overlap. That run measured only 8% peak GPU utilization and
  exposed the former 16,777,216-contribution convolution-plan cap. Convolution
  now computes unfold/col2im indices from compact geometry; its exact
  128x32x32x32 depthwise acceptance completes forward and backward without a
  per-contribution index allocation. The materialized patch tensor and eager
  launch/composition costs remain measured performance work. A later Perm
  scale probe found repeated host construction of Conv2d merge plans rather
  than the Trainer boundary as its dominant steady gap. Caching those plans
  reduced the exact two-method smoke from 85.55 to 25.07 seconds on an RTX
  5060 while preserving the recorded numerics; [profile](../backend/execution.md).
- The outside Chess acceptance composes bidirectional attention,
  input-dependent geometric bias, policy and value heads, AdamW, exact finite
  passes, and game-disjoint evaluation. Its bounded smoke passes; it makes no
  chess-ability claim.
- Causal masking is square and zero-offset; `prefix_causal_mask` additionally
  keeps a leading block of memory keys visible to every query. Softmax requires a finite entry
  in each row and permits negative infinity elsewhere. Padding/all-masked
  rows and cached decoding still need explicit contracts and witnesses.

## Next useful implementation

The byte autoencoder (`research/src/learning/bae`) is the next outside consumer: 125 bytes
to a 256-bit sign code to an autoregressive byte decoder over memory tokens,
with prefix (nested) dropout on the code. Its first slice landed `Embedding`,
`PositionEmbedding`, and `prefix_causal_mask` with scalar oracles. Its second
slice landed `Tensor::sign_straight_through` and the `SignStraightThrough`
module: forward `+1` where `x > 0` else `-1` (so `x == 0` maps to `-1`,
reproducing `bitae.quantize`'s `torch.where(z > 0, 1.0, -1.0)` rather than
`torch.sign`), backward the identity, matching `z + (hard - z).detach()`. The
remaining pull, in the order it bites: prefix dropout as a named-axis module
with a horizon-weighted loss, Bernoulli input dropout, and therefore explicit
random-state and train/evaluation semantics recorded in the receipt. Its held-out residual tables
(`bae/docs/NESTED.md`) are the replication oracle. A related finding: the
shared xorshift initializer emits exactly `-scale` as its first sample for any
seed below 2^40, so every module's first entry sits at the boundary; changing
the generator would move recorded baselines and is deferred.


Rerun the recorded Sudoku shapes and report utilization and elapsed time after
the tiled contraction and softmax changes. The next implementation should be
chosen from that profile: fuse attention if launch/materialization overhead now
dominates, or improve GEMM tiling if the matrix kernels remain inefficient.

After that, use the next consumer and the module backlog to choose between
production-transformer work (mixed precision, fused normalization and
attention, serialization), recurrent foundations, and the still useful
stacked-convolution composition witness. The external catalog supplies
discovery; each accepting program and independent oracle supply the reason to
land an Axis abstraction.

Atlas's `rbc-point` runtime (Studio `codex/axis-endtoend`) left two framework
items after PR #59, recorded here so the issue tracker can stay empty:

- Bound pending-buffer retention automatically for long inference forwards
  (issue #54). The consumer's explicit stream synchronizations at encoder and
  member boundaries are verified and remain; the framework still retires
  pending buffers only at the Trainer boundary.
- The cached click is kernel-bound at 39.6 ms of a 55 ms click against a 50 ms
  p95 contract: FP32 matmul 22.7, layout copies 7.3, plan-based means 4.8,
  zero-fills 2.9 (issue #58, closed). Candidates: FP16/BF16 matmul for
  inference, compact reductions for the means, and no zero-fill before a
  full overwrite. Outputs must stay bit-exact against the Studio receipts.

Run `bash scripts/check.sh` from the Axis root for formatting,
Clippy and CPU/GPU verification. Smoke a new program before 100/full-step
runs. Record actual evidence in the owning member's `data/evidence/` or `data/runs/`; update the
contract and this note when a capability or next step changes.
