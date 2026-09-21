# Axis MNIST: start here

This is the approachable end-to-end Axis example: recognizable data, one small MLP, categorical
loss, finite shuffled passes, and held-out accuracy. It is a Rust mechanics and architecture migration of
[`research/training-dynamics/train_mlp_1k.py`](../../../../../research/training-dynamics/train_mlp_1k.py), the small MNIST baseline used to
check the training-dynamics machinery. It keeps the 4x4 average pool,
train-only normalization, 49 -> 16 -> 10 network, 970-parameter budget,
shuffled finite passes, and held-out digit accuracy. It does not claim numerical
parity because its deterministic shuffler and initialization differ from
PyTorch, while its objective and Adam defaults now match the source.

```sh
bash src/examples/getting-started/mnist/scripts/train.sh --smoke
bash src/examples/getting-started/mnist/scripts/train.sh
```

Pass `--data /path/to/MNIST/raw` or place the four uncompressed IDX files in
this example's ignored `data/raw/` directory. The port deliberately does not claim IDR:
twenty epochs mean each of the 60,000 training examples is reused twenty times.
See [docs/migration.md](docs/migration.md) for the preserved contract and API
findings.

The categorical-loss smoke consumed five verified passes over 2,048 examples.
Held-out accuracy rose from `10.16%` to `33.98%`; the complete receipt and curve
are in [data/runs/smoke.log](data/runs/smoke.log).

The launcher points Cargo at the repository-local CUDA toolkit provisioned by
`scripts/setup_cuda.sh`; run that setup once if `build/cuda/` is absent.
