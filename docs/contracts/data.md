# Data, evidence, and runs

`data/` is local by default. Downloaded datasets, generated inputs, caches,
checkpoints, and exploratory output do not enter Git.

Two lanes are deliberate repository artifacts:

- `data/evidence/` contains compact verification receipts, audits, and oracle
  comparisons that support a code or research claim.
- `data/runs/` contains selected reproducible training or profiling records
  whose command, revision, and interpretation remain useful after the run.

Each consumer owns its records at `src/<consumer>/data/evidence/` or
`src/<consumer>/data/runs/`. Cross-workspace checks live in the root lanes.
Moving a log into a tracked lane is an editorial decision: transient debugging
output stays local, while a record cited by documentation belongs beside the
consumer whose claim it supports.

The structure check rejects Rust source under any `docs/` directory, a retired
top-level `runs/` directory, and tracked `data/` files outside these two lanes.
The same check runs through `scripts/check.sh` and GitHub CI.
