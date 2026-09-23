# Eager execution profile

## Translated spatial windows

`Tensor::pad_zeros` and `Tensor::narrow` map output coordinates to source
coordinates with one integer translation along a named axis. One 128-lane cuTile
kernel computes offsets from five integers per logical dimension: destination
extent/stride, source extent/stride, and translation. Reverse mode invokes the
same copy kernel with source/destination geometry exchanged and translation
negated. Each output element has at most one source, so no reduction or atomic
scatter is needed. Arbitrary physical input layouts are preserved by the inverse.

Missing coordinates use masked loads with literal zero fill, not multiplication
by a mask; NaN and infinity outside a retained interval cannot leak into padding.
Per-dimension invalid coordinates are clamped before address arithmetic. Checked
shape products and interval arithmetic enforce the signed 32-bit kernel limits
before allocating output. Neither direction builds an element-sized index table
or inherits the generic 16,777,216-contribution ceiling. Identity operations share
the original tensor node and storage. Nonidentity operations materialize their
output; multi-axis padding currently composes one operation per axis.

The consumer motivating this addition is MobileSAM TinyViT window attention:
128×128 is padded bottom/right to 133×133 for 7×7 windows; 64×64 is padded to
70×70 for 14×14 or 7×7 windows, then cropped after attention. Model-specific
partitioning, relative biases and masking policy remain in Atlas. The tests
establish copying and derivative correctness, not complete model parity or
an inference speedup. These operations are not present in released Axis 0.9.0.

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

The first optimization reused merge plans for a stable named shape, physical layout, output
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

## Compact layout lowering after the Atlas inference profile

The merge-cache result above is historical. A later two-image MobileSAM ensemble
trace recorded 293 `merge` spans totaling 249.818 calling-thread seconds and
16.824 seconds for the slowest call. The bounded cache did not avoid expensive
cold/uncached element-wise map construction. Nsight Systems recorded only 6.095
seconds of kernels and 8.462 seconds of combined kernel/copy activity over a
340.821-second traced GPU span. The longest interkernel gap was 16.764 seconds;
8,082 H2D copies transferred 30.791 GB. This is a profiling diagnosis, not an
unprofiled performance benchmark. CPU sampling was unavailable under host policy.

`with_layout` now reuses the compact device selection copier with no dimension
removed. Its forward and inverse require three metadata integers per logical
axis, not vectors proportional to element count. `split` materializes canonical
order only if necessary and creates a shape view; `merge` expands the specified
selected-axis order into physical order before creating its view. Identity maps,
including extent-one axes with irrelevant stride differences, remain storage
aliases. The element-sized merge cache and dead helpers are removed, not enlarged.
Broadcast/reduction planning and pending-buffer lifetimes remain separate limits.

The focused CUDA tests cover 72 independently indexed permutation/merge value and
gradient cases, noncontiguous split, signed-zero/nonfinite copying, invalid axes,
identity storage sharing and an actual 16,789,506-element tensor above the generic
plan ceiling. Evidence and the consumer comparison belong in
[`data/runs/compact-layout/`](../../data/runs/compact-layout/).

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

## cuTile trap: `constant()` silently truncates large `u64` literals

`backend::kernels::uniform_device`'s SplitMix64 mixer needs three 64-bit
constants with the top bit set (the golden-ratio constant and its own two
multipliers). Passing one of those literals straight to `cutile::core::constant`
-- `constant(0x9E37_79B9_7F4A_7C15u64, shape![128])` -- compiles cleanly and
runs, but every element reads back as exactly `0`, with no error at compile or
launch time. Any `u64` literal at or above `2^63` hits this in cuTile 0.3.1;
values below it are unaffected (measured with a throwaway probe kernel: a
literal one bit under the threshold read back correctly, the same literal
with that bit set read back as `0`). The fix used here is to split each
constant into two `u32`-range halves (each `constant()` call now argues a
value comfortably under `2^63`) and recombine with a shift and an or --
`(hi << 32) | lo` -- rather than one `constant()` call with the full 64-bit
literal. `broadcast_scalar` of a runtime kernel argument (this kernel's own
`seed: u64` parameter) is unaffected; the bug is specific to `constant()`'s
literal-parsing path, not to `u64` tiles or arithmetic in general.

Also worth knowing for future integer-tile kernels: `cutile::core::convert_tile`
does not implement integer-to-integer width changes at all in this version
(only the identity case and a few same-width bitcasts), despite the crate's
own doc comment claiming "all conversions supported" between every `iN`/`uN`
pair. `i32 -> u64`, `i32 -> u32`, and `u32 -> u64` all fail with "Unsupported
conversion" at kernel-compile time (a real compile error, unlike the silent
`constant()` truncation above). Integer-to-float and float-to-integer
conversions have no such restriction at any width, so routing an integer
width change through `f64` (exact for every value below `2^53`) works where a
direct integer widen does not. And the explicit arithmetic functions
(`muli`/`addi`/`xori`/`shri`/... with an `overflow::None` mode) fail to
serialize ("missing attribute 'overflow' on op MulI") where the plain Rust
operators (`*`, `+`, `^`, `>>`, ...) on the same tiles lower and run
correctly with ordinary wraparound semantics -- prefer the operators.

## ImageNet64 profile: the bias-broadcast gradient plan was never cached

A bounded `nsys`/`ncu`/`AXIS_PROFILE=1` pass training a 3-layer CNN on
ImageNet64 at batch 256 (Axis 0.11.0 against a hand-written PyTorch loop,
full method and numbers in the axis-benchmarks repo's
`data/evidence/imagenet64/profile.md`) found 71% of each step (213.6ms) in
`Tensor::align`'s general (rank-changing) path and another 22% (64.7ms) in
the `grouped_submit` reduction it feeds, for a combined 92.8% of the whole
308ms/step gap against PyTorch. Both costs came from the same place: every
layer's `.add(&bias)` broadcasts a `[channel]` tensor up to the full
activation shape, and because the bias requires a gradient, `align` built an
output-sized host index map and a fresh `Plan::reverse` scatter-add plan on
every training step, even though the shapes never change once a model is
built. `Plan::reverse` mints a new id on every call (`NEXT_PLAN.fetch_add`),
so `Device::device_plan`'s own upload cache -- which already exists and
already works for permutations -- always missed too, and the plan's
`offsets`/`left` arrays (up to 16.78MB for the first conv layer) were
re-uploaded every step in addition to being rebuilt every step.

The fix mirrors `Tensor::reduction_plans`'s existing cache (the one behind
`sum`/`mean`) exactly: a new `align_reverse_plan` method keys a thread-local
`HashMap<(Shape, Layout, Shape), Rc<Plan>>` by (the broadcasting tensor's own
shape, its physical layout, the target shape), matching `reduction_plans`'
`(u8, Shape, Layout, Vec<Axis>)` key one axis-set short (a broadcast target
is already a full `Shape`, so there is no separate axis list to carry). A
repeated broadcast now costs one hash lookup instead of an O(output size)
Rust loop, and because the cached `Plan` keeps the one `id` it was built
with, `device_plan`'s own cache then also hits on every repeat, for free,
with no change to `device_plan` itself. A microbenchmark on the desk RTX
5060 (100 steps of `[256,16,32,32] + [16]` broadcast-add, `mean`, backward,
matching the profile's first-layer bias shape) moved from 1594.7ms/step to
410.2ms/step, a 74.3% reduction -- consistent with this being the larger of
the profile's two findings but not the whole gap, since the reduction
kernel's own block-per-group launch shape (below) is a separate cost this
change does not touch.

This is a targeted fix at `align`'s own call site, not a change to
`Plan::reverse`'s other callers (`merge`, `gather`, `scatter_add`): their
`map` arguments are index data, not a pure function of shape and layout (an
embedding lookup's indices differ every call even when the shapes match), so
caching by signature would be silently wrong there.
