# Eager execution profile

Axis submits one eager training step to one CUDA stream and synchronizes at the
Trainer boundary. `AXIS_PROFILE=1` reports the major host stages as
`train_zero_grad`, `train_loss`, `train_backward`, `train_optimizer`, and
`train_boundary`. Tensor layout operations also report their own submission or
planning time. These timings are diagnostic wall times on the calling thread;
they are not CUDA kernel durations.

The Perm scale smoke exposed why the distinction matters. On an otherwise idle
RTX 5060, three ordered and three exact-set Adam steps over a
`[batch=48, channel=1, height=32, width=32]` input took 85.55 seconds at Axis
`7a14357`. Each steady loss construction spent about 5.90 seconds on the host,
while stream synchronization took at most 0.04 seconds. `Tensor::merge` was
rebuilding the same million-entry forward and reverse index plans on every
Conv2d forward.

Axis now reuses merge plans for a stable named shape, physical layout, output
axis, and ordered list of merged axes. The ordered axes are part of the key:
merging `[head, depth]` and `[depth, head]` can produce the same output shape
with different values. Host retention is bounded to 128 signatures and 256 MiB
of plan-vector payload; keys and container overhead are additional. The cache
does not evict: signatures encountered after either bound remain step-local.
The existing per-device plan cache remains independently bounded to 256 MiB.

The identical probe then took 25.07 seconds. Ordered/set run timers changed from
`47.770/37.554` seconds to `17.317/7.529` seconds, a 70.70% reduction across
the complete two-method command. All printed losses and validation metrics were
bit-for-bit unchanged at their recorded precision. Cache-hit loss construction
fell below 1.5 milliseconds. One-second GPU samples moved from 10.27% mean and
31% peak to 13.23% mean and 97% peak. The raw traces and exact command scope
are retained in [`data/runs/rtx-5060-merge-cache/`](../../data/runs/rtx-5060-merge-cache/).

This is one stable-shape convolution witness on a shared desktop GPU. Cold plan
construction still costs about 5.84 seconds per newly bound model in this
probe, and mean utilization remains low. The result does not establish
competitive convolution throughput or a general CUDA performance ratio.

A separate temporary consumer run observed one complete ordered `run()` over
1,000 Adam steps at the same batch, image, and feature sizes. Its internal timer,
which includes training and evaluation, reported 120.820 seconds; loss moved
from `0.285610` to `0.036525`. The subsequent exact-set run was intentionally
stopped, so the command has no successful process-exit receipt. This bounded
observation shows the ordered workload inside 30 minutes; it does not claim a
completed two-method experiment or an accuracy or throughput frontier.
