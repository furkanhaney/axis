# RTX 5090 long controls — 2026-09-20

These controls ran on public Axis commit `542b4f6` on one NVIDIA RTX 5090.
They establish longer generated-data training behavior before the later
Sudoku-driven lowering work.

## Generated addition

```bash
bash src/addition/scripts/train.sh --steps 5000
```

The run consumed 1,280,000 unique generated training IDs with zero observed
reuse and zero overlap with 256 evaluation IDs. Held-out MSE fell from
`0.80632222` to `0.00122140`. Axis reported `15.57 s`; complete command wall
time including startup was `48.78 s`.

## Causal attention

```bash
bash src/attention/scripts/train.sh --steps 2000
```

Training MSE fell from `4.30e-1` to `1.97e-4`; held-out MSE fell from
`5.12e-1` to `6.50e-4`. Complete command wall time was `21.12 s`.

The `.log` files contain the exact metrics and receipts. The corresponding
`-gpu.csv` files contain one-second `nvidia-smi` telemetry.
