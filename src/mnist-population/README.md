# Axis MNIST population

Rust mechanics migration of [`research/training-dynamics/train_mlp_pop.py`](../../../research/training-dynamics/train_mlp_pop.py), the
population MNIST probe. It trains independently initialized 970-parameter
classifiers at log-uniform learning rates, then reports held-out accuracy by
learning-rate quartile.

```sh
scripts/train.sh --smoke
scripts/train.sh --pop 512
```

The input is the canonical torchvision IDX data already stored under
`training-dynamics/data/MNIST/raw/`. Each model consumes the same declared
number of independently shuffled complete training passes. This is explicitly
a finite-data experiment, not IDR.

The source fuses the population into a GPU tensor and primarily asks how many
complete runs fit in a second. `axis` does not yet have a model-population
axis, so this migration executes models sequentially. Its accuracy-versus-rate
result is meaningful; its timing is a fallback measurement and is not a
reproduction of the source throughput result. See
[docs/migration.md](docs/migration.md).

The bounded four-member smoke raised the best held-out accuracy from `9.18%`
before training to `74.61%`; [data/smoke.log](data/smoke.log) records the full
learning-rate sweep and labels its sequential throughput.
