# Fashion-MNIST matched smoke receipt

| Measure | Lightning | PyTorch | JAX | Axis |
| --- | ---: | ---: | ---: | ---: |
| final accuracy | 0.5918 | 0.5918 | 0.5918 | 0.5918 |
| training seconds (total, median of 3) | 0.183 | 0.161 | 2.751 | 6.584 |
| training seconds (steady, median of 3) | 0.066 | 0.013 | 0.038 | 0.030 |
| task lines | 77 | 73 | 82 | 155 |
| total lines | 120 | 90 | 105 | 229 |
| identity overlap (default training) | not_checked | not_checked | not_checked | rejects |
| data reuse (default training) | not_checked | not_checked | not_checked | rejects |

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

This receipt covers one bounded run on the executing host. It is not a framework-wide performance claim.
