# Continue from the working examples

Updated 2026-09-20. This note preserves the decisions and next useful work;
[library.md](library.md) is the contract, and [sample.rs](sample.rs) is the
owner's larger, partially unimplemented API sketch.

## What exists and why

- `src/axis/` is the reusable crate, with its own manifest. Every sibling under
  `src/` is a separate program crate. Launch from the Axis node, not a `src/`
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
  number of unique samples. See [data regimes](data-regimes.md).
- `DataLoader` accepts finite or generated `DataSource`s and withholds a batch
  when its regime fails. `AdditionDataset` is the first inexhaustible source.
  `Trainer` captures only update ordering; MLP, attention, and CNN still state
  their forward, loss, and named reductions directly.
- Attention composition stays in `src/attention/src/attention.rs`. Only its
  required tensor operations, causal mask and softmax, entered the library.
  A second consumer can justify promoting the composed module later.

## Evidence and limits

- MLP's 500-step held-out MSE matched the retained baseline at `5.74e-2`;
  [MLP records](../src/mlp/README.md#library-mlp). Named loading and pre-update
  logging passed the subsequent [feedback checks](../data/feedback-verification.log).
- Attention's independent f64 forward and central differences pass for every
  Q/K/V element, including reordered physical storage and a leading logical
  feature axis. Causality and large-logit softmax checks pass. Its 200-step
  held-out MSE falls from `5.12e-1` to `2.39e-2` on toy prefix means;
  [training record](../src/attention/data/training.log).
- CNN matches 4,096 scalar activation values, pooled features, logits, every
  parameter gradient, overlapping input gradients, and one update. At 100
  steps its held-out BCE falls from `7.04e-1` to `7.40e-2` with `100.00%`
  accuracy; [CNN record](../src/cnn/data/library-training.log). Its Conv2d is
  `unfold2d` plus Linear, so `col2im` is the reverse of the gather rather than
  a separate special-case kernel.
- Addition consumes 128,000 generated samples with zero observed identity
  reuse and zero train/evaluation overlap. Its held-out MSE falls from
  `8.06e-1` to `3.88e-3`; [addition record](../src/addition/data/training.log).
- The first outside consumer now preserves categorical cross-entropy, Adam, and
  exact finite passes; its smoke reaches `33.98%` held-out MNIST accuracy.
  The population migration now runs independent parameters, gradients, Adam
  state, and learning rates along one named population axis. It exposed the
  correctness-first contraction lowering as the remaining throughput limit.
  The panel migration drove device-resident AdamW and preserves its
  preprocessing/split mechanics without claiming a completed GDP fit.
- Current cuTile lowering uses synchronous f32 operations and CPU-built index
  plans. GPU arithmetic and derivatives are real, but no broad speed claim
  follows: single-axis contractions use batched tiled matrix multiplication;
  multi-axis contractions still use gather/reduce plans. Softmax performs one
  tiled forward or backward reduction per row, including non-power-of-two
  widths, instead of recomputing the row reduction for every output. Generic
  plans retain their 16,777,216-contribution cap. There is no retained graph,
  CPU fallback, mixed precision, or higher-order differentiation.
- The outside Sudoku acceptance passed every host and CUDA oracle on an RTX
  5090, then trained over 1,600 fresh boards with zero observed reuse or
  train/evaluation overlap. The same run measured only 8% peak GPU utilization
  and failed evaluation batches of 16 and 64 at the 16,777,216-contribution
  plan cap. This turns contraction planning and launch count into measured
  framework limits rather than inferred performance concerns.
- The outside Chess acceptance composes bidirectional attention,
  input-dependent geometric bias, policy and value heads, AdamW, exact finite
  passes, and game-disjoint evaluation. Its bounded smoke passes; it makes no
  chess-ability claim.
- Causal masking is square and zero-offset. Softmax requires a finite entry
  in each row and permits negative infinity elsewhere. Padding/all-masked
  rows and cached decoding still need explicit contracts and witnesses.

## Next useful implementation

Rerun the recorded Sudoku shapes and report utilization and elapsed time after
the tiled contraction and softmax changes. The next implementation should be
chosen from that profile: fuse attention if launch/materialization overhead now
dominates, or improve GEMM tiling if the matrix kernels remain inefficient.

After that, use the next consumer to choose between production-transformer work
(mixed precision, fused normalization and attention, serialization) and the
still useful stacked-convolution composition witness. Do not grow either
surface from an operator checklist: the accepting program must own the need and
the scalar or trusted reference.

Run `bash scripts/check.sh` from the Axis root for formatting,
Clippy and CPU/GPU verification. Smoke a new program before 100/full-step
runs. Record actual evidence in the owning member's `data/`; update the
contract and this note when a capability or next step changes.
