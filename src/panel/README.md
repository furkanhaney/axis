# Axis country panel

A bounded Rust mechanics migration of the ReLU path in
[`research/energy-output/fit_panel_torch.py`](../../../research/energy-output/fit_panel_torch.py). It exercises the same
country-year representation with `axis`: four logged capacity inputs,
training-only missing-value imputation, four missingness indicators, linear
time, training-only standardization, an 8-unit ReLU MLP, full-batch MSE,
device-resident AdamW, and a held-out validation checkpoint.

```sh
scripts/train.sh --smoke --split country
scripts/train.sh --smoke --split future
```

These commands are migration witnesses. They do not produce or replace any
reported real-GDP fit. See [docs/migration.md](docs/migration.md) for the
preserved contracts and current differences.

With AdamW enabled, the bounded country split reduced normalized validation MSE
from `1.657922` to `0.178666`; the future split reduced it from `1.962119` to
`0.107304`. These are mechanics witnesses recorded in [run evidence](data/runs/), not study
results.
