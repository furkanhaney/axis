# Acceptance ladder

Axis grows against outside programs that force several abstractions to compose.
A small isolated operator is a unit witness; these programs are framework
acceptance tests.

## Paired Muon training — current

The in-repository [Muon acceptance](../../src/examples/muon/README.md) trains
identically initialized MLPs on the same deterministic regression populations.
One uses AdamW throughout; the other explicitly assigns its first hidden
weight to Muon and the exact remainder to AdamW. Both arms must emit separate
held-out `LearningProgress` receipts. This is a composition and learning
witness, not an optimizer ranking. In the recorded 100-step run, each arm
reduced held-out MSE by more than 99% from the shared initialization.

## Sudoku transformer — current

The sibling [`sudoku-transformer`](https://github.com/furkanhaney/sudoku-transformer) repo is the first complete transformer acceptance.
It requires generated structured data, an executable IDR declaration, learned
token and position embeddings, bidirectional multi-head attention, LayerNorm,
GELU, residual paths, masked categorical loss, AdamW, and both cell and
whole-example metrics. A human can inspect the task and its answers directly.

The acceptance has two levels:

1. The bounded configuration must validate generated boards, preserve shapes,
   complete forward/backward/update, keep loss finite, emit passing IDR and
   train/evaluation-disjoint receipts, and report blank and full-board accuracy.
2. Larger runs must produce a held-out learning curve and solve-rate evidence
   before the public README claims that Axis learns Sudoku.

An RTX 5090 run completed 200 updates over 1,600 unique boards with zero reuse
or evaluation overlap. Loss moved from `2.873419` to `2.170745`, while blank
accuracy reached only `14.93%` and exact solves remained zero. This is a scaled
mechanics witness, not yet the second acceptance level.

## Chess transformer — current

The sibling [`chess-transformer`](https://gitlab.com/furkanhaney/chess-transformer)
repo exercises a different regime: a finite corpus split by whole game before
positions are derived. It adds history-aware 64-square inputs, geometric
attention bias, a bilinear 4,096-move policy, a win/draw/loss head, a joint
objective, exact pass accounting, and game-disjoint identities.

Its bounded smoke completes two AdamW updates and moves held-out total loss from
`8.689576` to `7.676713` over two positions. The tiny corpus measures mechanics
only. A future ability claim requires the versioned Lichess recipe and an
independent game-level evaluation corpus; the migration explicitly rejects the
old Python run's overlapping validation recipe.

## ask_bro — long-range

`ask_bro` deliberately means the useful sub-30B model, not a particular host or
vendor. Running its model through Axis would require the production transformer
stack that Sudoku does not:

- the real tokenizer and checkpoint format;
- rotary position encoding and grouped-query attention;
- RMSNorm, gated feed-forward blocks, and the model's exact nonlinearities;
- bf16/fp16 and quantized weights with measured numerical error;
- KV caching and autoregressive generation;
- bounded GPU memory planning, weight streaming, and multi-device placement;
- deterministic prompts and output comparisons against the current runtime;
- vision input if the selected little brother is multimodal; and
- throughput and memory measurements at the actual model size.

The first honest milestone is not “27B compiles.” It is one frozen layer matching
a trusted implementation on recorded inputs, then a complete forward pass, then
generation, then enough performance to replace the existing capability. Sudoku
and Chess should supply the shared math; ask_bro should drive only the machinery
that appears when the model becomes real-sized.
