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

## Capability the migration exposes

The Python implementation stores weights as `[population, input, output]` and
computes every independent model in one batched contraction. The current
library can attach unrelated axes to activations, but `Linear` owns one weight
matrix and therefore shares it across those axes. Merely adding a `population`
axis to inputs would silently train one model, not a population.

This executable consequently trains one model after another and labels its
timing `sequential fallback`. It must not be used to support the source claim
about millions of runs per hour. A faithful performance migration needs a
parameter-population axis whose initialization, gradients, Adam state, and
evaluation remain independent while kernels batch the contractions. That is a
concrete next abstraction derived from a real consumer rather than a generic
vectorization feature.

`--smoke` reduces the population, corpus, held-out split, and pass count. It
checks that the population contains finite accuracies, spans more than one
learning rate, and that the best final accuracy exceeds the best untrained
accuracy. It is a mechanics check, not a benchmark result.

The 2026-09-20 smoke trained four members for five passes over 2,048 examples.
The sampled rates spanned `1.00e-4` to `2.37e-2`; best held-out accuracy rose
from an untrained population maximum of 9.18% to 74.61%. Sequential fallback
throughput was 0.40 complete runs/s. These numbers establish an executable
migration and explicitly do not establish fused-population performance.
