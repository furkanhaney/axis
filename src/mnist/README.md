# Axis MNIST

Rust mechanics and architecture migration of
[`research/training-dynamics/train_mlp_1k.py`](../../../research/training-dynamics/train_mlp_1k.py), the small MNIST baseline used to
check the training-dynamics machinery. It keeps the 4x4 average pool,
train-only normalization, 49 -> 16 -> 10 network, 970-parameter budget,
shuffled finite passes, and held-out digit accuracy. It does not claim numerical
parity because its deterministic shuffler and initialization differ from
PyTorch, while its objective and Adam defaults now match the source.

```sh
training-dynamics/mlp-1k-rust/scripts/train.sh --smoke
training-dynamics/mlp-1k-rust/scripts/train.sh
```

The input files are the same torchvision IDX files already downloaded beneath
`../research/training-dynamics/data/MNIST/raw/`. The port deliberately does not claim IDR:
twenty epochs mean each of the 60,000 training examples is reused twenty times.
See [docs/migration.md](docs/migration.md) for the preserved contract and API
findings.

The categorical-loss smoke consumed five verified passes over 2,048 examples.
Held-out accuracy rose from `10.16%` to `33.98%`; the complete receipt and curve
are in [data/runs/smoke.log](data/runs/smoke.log).

The launcher points Cargo at the study-local CUDA toolkit provisioned by
`scripts/setup_cuda.py`; run that setup once if `.cuda/` is absent.
