# Migration contract

## Scientific question retained

The source asks two questions: how accuracy varies across independently
initialized roughly one-thousand-parameter MNIST MLPs with log-uniform learning
rates, and how quickly a fused population can be trained. This migration
preserves the first question. Every population member has:

- the canonical MNIST 60,000/10,000 train/test split;
- non-overlapping 4x4 average pooling from 28x28 to 7x7;
- scalar normalization measured from training pixels and applied to both
  splits;
- a 49 -> 16 -> 10 ReLU MLP with exactly 970 trainable values;
- independent Kaiming-normal weights and zero biases;
- Adam with beta1 0.9, beta2 0.999, epsilon 1e-8;
- twenty independently shuffled complete finite-corpus passes; and
- held-out argmax accuracy, ordered and summarized against the sampled rate.

All population members see the same shuffle for a given pass, matching the
source's shared minibatches. `FinitePassesLoader` makes the declared reuse
executable and rejects within-pass duplicates or an incomplete pass. Applying
`assert_idr()` here would misstate the experiment.

The Rust random stream is deterministic from seed 0 but is not Torch's random
stream, so exact initial tensors, learning rates, and per-model accuracies do
not match bit for bit. The initialization distribution, rate interval, model,
data, optimizer, and evaluation contract are retained.

## Capability the migration exposed and closed

The Python implementation stores weights as `[population, input, output]` and
computes every independent model in one batched contraction. That concrete
consumer produced `PopulationLinear`: the population is a named axis on weights,
biases, activations, gradients, and optimizer state. Shared input batches and
targets broadcast over it explicitly. `Adam::with_axis_learning_rates` expands
one rate per member over each parameter without changing moment estimation.

This closes the semantic migration gap. It does not close the performance gap.
Axis still lowers contractions through correctness-first gather/reduce plans;
the fused four-member smoke measured `0.31` complete runs/s, below the former
sequential witness's `0.43` runs/s and far from the specialized source. The
result identifies kernel lowering as the next bottleneck and supports no source
throughput claim.

`--smoke` reduces the population, corpus, held-out split, and pass count. It
checks that the population contains finite accuracies, spans more than one
learning rate, and that the best final accuracy exceeds the best untrained
accuracy. It is a mechanics check, not a benchmark result.

The 2026-09-20 fused smoke trained four members for five passes over 2,048
examples. The sampled rates spanned `1.00e-4` to `2.37e-2`; best held-out
accuracy rose from an untrained population maximum of 8.20% to 60.55%.
Throughput was 0.31 complete runs/s. These numbers establish independent fused
semantics and explicitly do not establish competitive population throughput.
