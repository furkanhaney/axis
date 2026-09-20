# Give coordinates to the experiment

Axis is not organized around the goal of accumulating tensor operators. Its
technical bet is that an ML program should carry enough semantic structure for
the framework to reason about what the experiment means.

An anonymous shape such as `[32, 81, 10]` leaves `batch`, `position`, and `token`
in comments, dimension numbers, and programmer memory. Axis makes those names
identities in the program. The same information then participates in alignment,
broadcasting, contraction, autograd, diagnostics, and eventually compilation.

The second coordinate is the experimental regime. A run can declare and check
properties such as finite passes, no observed identity reuse, train/evaluation
disjointness, and an operational approximation to the infinite-data regime.
Receipts report exactly what was observed. They do not inflate non-reuse into a
claim of independent samples or infinite informational diversity.

The third coordinate is the objective. `masked_mean` is the first small example:
a consumer no longer hand-normalizes `loss * mask`. Axis checks that the mask is
constant and binary, rejects an empty selection, aligns it by axis identity, and
normalizes by the selected elements. This is still a low-level contract. The
longer path is to let a program declare statements such as “Sudoku loss applies
only to blanks” and reject a reduction over all cells.

The project therefore grows against consumer acceptances rather than an operator
checklist:

```text
MLP                  tensors, reverse mode, optimizers
CNN                  spatial structure and overlapping gradients
Sudoku transformer   embeddings, bidirectional attention, masked objectives, IDR
ask_bro              real sub-30B checkpoints, precision, caching, memory, generation
```

An outside consumer must express a real workload through the public API, train
it, evaluate it, and preserve its scientific contract. That is harder to game
than hundreds of isolated operator tests.

The intended destination is a run whose location is explicit across several
spaces:

```text
tensor semantics
model semantics
data regime
optimization regime
evaluation regime
```

Axis should keep execution inside the region the researcher declared, or stop
with a useful account of which boundary was crossed.

## Verification is the scarce resource

Code generation makes plausible ML programs cheap. It does not make it cheap to
know whether a run answered the question its author intended. A loss curve can
look healthy after a loader wraps, an evaluation population leaks into training,
a target shifts by the wrong offset, or supposedly equal-compute runs diverge.
Axis is aimed at that verification bottleneck:

```text
research intent
    -> constrained experiment
    -> tensor program
    -> GPU
    -> evidence about what actually ran
```

Named axes help because they expose more of the mathematics to the compiler and
reviewer. They are not the entire proposition. The same principle should move a
known failure left over time: from a surprising result, to a receipt, to a
runtime assertion, to a static check, and finally to a state the API cannot
express.

That makes Axis a place to encode scientific culture. Each real migration can
contribute one durable check for a mistake that an experienced researcher would
otherwise have to remember. Most checks remain dormant; declaring a causal
objective, generated-data regime, or equal-compute comparison activates the
ones that give those words operational meaning. The framework should prevent a
program from presenting persuasive evidence after its declared regime has
already failed.
