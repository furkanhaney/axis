# ImageNet64 matched run receipt

| Measure | Lightning | PyTorch | JAX | Axis |
| --- | ---: | ---: | ---: | ---: |
| initial top-1 | 0.0010 | 0.0010 | 0.0010 | 0.0010 |
| initial top-5 | 0.0055 | 0.0055 | 0.0055 | 0.0055 |
| final top-1 | 0.0552 | 0.0554 | 0.0548 | 0.0540 |
| final top-5 | 0.1504 | 0.1511 | 0.1499 | 0.1485 |
| training seconds (total) | 183.31 | 186.17 | 45.20 | 1647.20 |
| training seconds (steady) | 49.05 | 42.33 | 40.60 | 1638.17 |
| images/second (steady) | 26119.9 | 30268.8 | 31559.2 | 782.1 |
| peak GPU memory (MiB) | 504 | 508 | 1304 | 1078 |
| task lines | 116 | 101 | 111 | 173 |
| total lines | 206 | 105 | 155 | 462 |
| identity overlap (default training) | not_checked | not_checked | not_checked | rejects |
| data reuse (default training) | not_checked | not_checked | not_checked | rejects |

train_seconds_total includes the first training step (framework warmup: JIT trace/compile, CUDA context and kernel planning); train_seconds_steady excludes it. Both synchronize the device before reading the clock. Each arm ran once at this budget; there is no repeat median at this scale.

| Target | Result |
| --- | --- |
| axis_rejects_identity_overlap | PASS |
| axis_rejects_data_reuse | PASS |
| lightning_default_accepts_identity_overlap | PASS |
| pytorch_default_accepts_identity_overlap | PASS |
| jax_default_accepts_identity_overlap | PASS |
| lightning_default_accepts_data_reuse | PASS |
| pytorch_default_accepts_data_reuse | PASS |
| jax_default_accepts_data_reuse | PASS |

This receipt covers one bounded run on one host at one budget. It is not a framework-wide performance claim.
