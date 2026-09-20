# Muon acceptance

This paired regression run exercises Axis's explicit optimizer partition. The
hybrid arm sends one hidden `Linear` weight to Muon using its explicit
`[fan_in, fan_out]` orientation and sends every other parameter to AdamW. The
control arm sends the same initialized model to AdamW. Both arms train and
evaluate on the same deterministic populations.

```bash
scripts/train.sh --smoke
scripts/train.sh --steps 100
```

The acceptance requires separate `LearningProgress` receipts from both arms.
It establishes that each optimizer can learn through `Trainer`; it does not
claim that Muon beats AdamW. The scalar library oracle separately checks the
exact one-step and two-step Muon updates for tall and wide matrices.

In the recorded 100-step run, held-out MSE moved from `0.26941565` to
`0.00241553` for AdamW and from the same baseline to `0.00045361` for the
explicit Muon/AdamW partition. These are two learning witnesses on one small
task, not a comparative optimizer result; see
[the run receipt](data/runs/training.log).
