# Axis MNIST population

This program assumes familiarity with ordinary MNIST training. It exists to exercise independent
models, optimizer state, and learning rates on one named population axis; it is not the introductory
MNIST example.

Rust mechanics migration of [`research/src/learning/training-dynamics/train_mlp_pop.py`](../../../../../research/src/learning/training-dynamics/train_mlp_pop.py), the
population MNIST probe. It trains independently initialized 970-parameter
classifiers at log-uniform learning rates, then reports held-out accuracy by
learning-rate quartile.

```sh
src/studies/contracts/mnist-population/scripts/train.sh --smoke
src/studies/contracts/mnist-population/scripts/train.sh --pop 512
```

The input is the canonical torchvision IDX data already stored under
`../research/src/learning/training-dynamics/data/MNIST/raw/`. Each model consumes the same declared
number of independently shuffled complete training passes. This is explicitly
a finite-data experiment, not IDR.

The migration now represents every independent model with a named population
axis. `PopulationLinear` carries independent weights, biases, gradients, and
Adam state, while `Adam::with_axis_learning_rates` broadcasts one sampled rate
over each member's parameters. The implementation is fused semantically, but
the correctness-first contraction lowering is still slower than the source and
does not reproduce its throughput claim. See
[docs/migration.md](docs/migration.md).

The fused four-member smoke raised the best held-out accuracy from `8.20%`
before training to `60.55%`; [data/runs/smoke.log](data/runs/smoke.log) records the full
learning-rate sweep. Its measured `0.31` runs/s is evidence that kernel lowering,
not the public population model, is now the throughput bottleneck.
