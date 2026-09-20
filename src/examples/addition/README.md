# Addition stream

This is the smallest end-to-end IDR experiment in the workspace. A deterministic
`AdditionDataset` emits fresh `(left, right, sum)` draws without exhaustion. A
`DataLoader` rejects repeated draw identities before a batch can reach the
optimizer, and `Trainer` performs the explicit scalar-loss update.

```bash
scripts/train.sh --smoke
scripts/train.sh --steps 500
```

Training is measured in samples and optimizer steps. There are no epochs. The
final output includes a generated-stream IDR receipt for raw draw IDs. A separate
`Disjointness` ledger defines the scientific identity as the ordered pair of
operand `f32` bit patterns and rejects that canonical problem if it crosses the
observed training/evaluation boundary, even under a distinct draw ID. The
receipt proves separation among delivered operand pairs; it does not imply IID
sampling or a rich latent distribution. Evaluation identities are retained and
sealed before training begins; training identities are checked and discarded,
so ledger memory does not grow with the generated stream.

After training, the program holds the other operand fixed and perturbs each
operand upward around all 256 evaluation contexts. `EmpiricalMonotonicity` requires
the predicted sum to be empirically increasing for every ordered pair within
an absolute tolerance of `1e-6`. The receipt deliberately says this sampled
audit is not a global architectural guarantee.

The measured 500-step run consumed 128,000 unique draw IDs. Every canonical
training problem was checked against 256 retained evaluation problems with zero
overlap. Within-training semantic reuse was not measured by this bounded-memory
ledger. Held-out MSE fell from `0.80632222` to `0.00387844`; see
[data/runs/training.log](data/runs/training.log).
