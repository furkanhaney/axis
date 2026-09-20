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
evaluation source uses a separate seed namespace, and the final output includes
the generated-stream IDR receipt. `TrainEvalDisjoint` also proves that no exact
draw ID crossed between the observed training and evaluation populations. These
receipts prove exact draw nonreuse; they do not claim that floating-point operand
pairs cannot coincide.

After training, the program holds the other operand fixed and perturbs each
operand upward around all 256 evaluation contexts. `EmpiricalMonotonicity` requires
the predicted sum to be empirically increasing for every ordered pair within
an absolute tolerance of `1e-6`. The receipt deliberately says this sampled
audit is not a global architectural guarantee.

The measured 500-step run consumed 128,000 unique training draws with zero
observed reuse and zero overlap with 256 evaluation draws. Held-out MSE fell
from `0.80632222` to `0.00387844`; see [data/runs/training.log](data/runs/training.log).
