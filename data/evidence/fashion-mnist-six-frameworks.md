# Fashion-MNIST matched smoke receipt

| Measure | Lightning | PyTorch | JAX | Axis | Burn | Candle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final accuracy | 0.5918 | 0.5918 | 0.5918 | 0.5918 | 0.5938 | 0.5918 |
| training seconds (total, median of 3) | 0.198 | 0.167 | 2.935 | 7.407 | 2.855 | 0.030 |
| training seconds (steady, median of 3) | 0.071 | 0.016 | 0.042 | 0.035 | 0.082 | 0.018 |
| task lines | 77 | 73 | 82 | 155 | 118 | 114 |
| total lines | 120 | 90 | 105 | 229 | 193 | 189 |
| identity overlap (default training) | not_checked | not_checked | not_checked | rejects | not_checked | not_checked |
| data reuse (default training) | not_checked | not_checked | not_checked | rejects | not_checked | not_checked |


**Shared GPU during timing**: Lightning, PyTorch, JAX, Axis, Burn, Candle had another session's process on the GPU immediately before or after at least one timed repeat (see `gpu_compute_apps_before_runs`/`gpu_compute_apps_after_runs` in smoke.json). The numbers above are kept as measured, on a shared GPU.

train_seconds_total includes the first training step (framework warmup: JIT
trace/compile, CUDA context and kernel planning); train_seconds_steady
excludes it. Both synchronize the device before reading the clock. Each
figure is the median of 3 full runs; every repeat is kept in
`smoke.json`.

| Target | Result |
| --- | --- |
| accuracy_within_0.02 | PASS |
| axis_runtime_at_most_lightning | MISS |
| axis_task_lines_at_most_lightning | MISS |
| axis_rejects_identity_overlap | PASS |
| axis_rejects_data_reuse | PASS |
| lightning_default_accepts_identity_overlap | PASS |
| pytorch_default_accepts_identity_overlap | PASS |
| jax_default_accepts_identity_overlap | PASS |
| lightning_default_accepts_data_reuse | PASS |
| pytorch_default_accepts_data_reuse | PASS |
| jax_default_accepts_data_reuse | PASS |
| burn_default_accepts_identity_overlap | PASS |
| candle_default_accepts_identity_overlap | PASS |
| burn_default_accepts_data_reuse | PASS |
| candle_default_accepts_data_reuse | PASS |

This receipt covers one bounded run on the executing host. It is not a framework-wide performance claim.
