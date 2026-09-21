# Eager execution profile

## Recurrent correctness path

The first `Lstm` lowering projects the complete input sequence in one contraction,
with the named time axis intact, and adds input bias once. It then walks time in
logical coordinate order and submits one recurrent projection and `LstmCell`
transition per coordinate. Each operation is ordinary Axis algebra, so reverse
mode retains the connected hidden/cell chain and an explicit `LstmState::detach`
is the truncation boundary. Named selection uses rank-sized device metadata
rather than a host index for every element; stacking stores each hidden state as
one contiguous physical slice while preserving the declared logical axis order.

The graph and retained transition activations grow linearly with sequence length,
and kernel-launch count does too. This establishes the bounded eager semantics
tested by the independent scalar oracle. It does not establish competitive
sequence throughput. A later measured consumer can justify a fused recurrent
scan and backward lowering.

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

## Remaining Perm profile work

A later Axis 0.6.0 trace on an RTX 5070 Ti measured 2,113 CUDA launches across
six Perm optimizer updates, or about 352 launches per update including startup
and JIT work. It also recorded 1,215 `cuMemAllocAsync`/`cuMemFreeAsync` pairs
and 18 stream synchronizations. A steady-state trace still needs to attribute
those launches to graph operations before Axis can justify fusion, allocation
reuse, batched tiny work, or CUDA graph replay. Acceptance is fewer launches
per update and lower unprofiled update time with the frozen scientific metrics
unchanged; profiler wall time is diagnostic only.

In that trace, 341 generated `grouped_entry` kernels accounted for 77.6% of
sampled GPU kernel time. One selected launch used 168 registers per thread,
reached 22.58% achieved occupancy against a 25% theoretical limit, 22.79%
compute throughput, and 2.34% memory throughput. NVIDIA's profiler identified
registers as that launch's occupancy limiter. This one selected kernel does not
support a framework-wide Tensor Core or throughput claim. The next useful
experiment is to inspect generated code, live ranges, inlining, spills, and
tile choices, then repeat both `ncu` and unprofiled end-to-end measurements.

These are documented optimization targets rather than release correctness
failures. Axis 0.7 records them as explicit backend limits; it does not claim
that launch fragmentation or `grouped_entry` register pressure is solved.

A separate temporary consumer run observed one complete ordered `run()` over
1,000 Adam steps at the same batch, image, and feature sizes. Its internal timer,
which includes training and evaluation, reported 120.820 seconds; loss moved
from `0.285610` to `0.036525`. The subsequent exact-set run was intentionally
stopped, so the command has no successful process-exit receipt. This bounded
observation shows the ordered workload inside 30 minutes; it does not claim a
completed two-method experiment or an accuracy or throughput frontier.
