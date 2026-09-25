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

## Persistent JIT disk cache

cuTile JIT-compiles every kernel specialization through its `tileiras`
subprocess on first use with a new shape, at roughly 290 ms per new shape
(#152); a model with dozens of shape-distinct kernels can spend most of a
short process's wall time compiling rather than computing. cuTile 0.3.1
ships a content-addressed persistent cubin cache (`cutile::jit_cache`, keyed
by SHA-256 of the serialized Tile IR bytecode, target, opt level, and the
`tileiras` fingerprint) but leaves it off by default with no environment
switch of its own. `Device::cuda`/`Device::cuda_bf16` enable it the first
time any CUDA device is created in the process (`ensure_jit_cache_enabled`
in `runtime/backend.rs`, guarded by a `OnceLock` so later devices are a
no-op), at cuTile's own default location (`~/.cache/cutile/kernels`, or
`$XDG_CACHE_HOME/cutile/kernels`). `AXIS_JIT_CACHE=off` opts out for the
whole process; `AXIS_JIT_CACHE_DIR` points the store at an explicit
directory instead. Every cuTile store I/O failure is already soft (a miss,
not an error); enabling the store itself can also fail (an unwritable
directory, no resolvable per-user cache path), and that failure is logged
once to stderr and leaves the cache disabled rather than failing device
creation. `jit_cache_stats()` returns the process's cumulative disk
hit/miss counts for a receipt to record.

Measured on the desk RTX 5060 with `axis-mlp --steps 1` (context init, model
build, and one `Trainer::step`), a fresh empty `AXIS_JIT_CACHE_DIR` for the
cold run and the same populated directory reused unmodified for the warm run:

| | wall clock |
|---|---:|
| cold (empty cache, one process) | 5.07 s |
| warm (same cache directory, next process) | 1.35 s |
| `AXIS_JIT_CACHE=off`, run 1 | 5.31 s |
| `AXIS_JIT_CACHE=off`, run 2 | 5.35 s |

`AXIS_PROFILE=1`'s `train_*` stages isolate the first `Trainer::step` alone
(excluding the two pre-loop loss evaluations, which already warm the
forward-pass kernels): `train_zero_grad` plus `train_loss` plus the
loss/backward submission plus `train_backward` plus `train_optimizer` plus
`synchronize` plus `train_boundary` sum to 3.58 s cold and 0.68 s warm, a
5.3x reduction, consistent with `train_backward` alone (2.12 s cold, 0.43 s
warm) compiling several new-shape kernels once and never again.
`AXIS_JIT_CACHE=off` costs the same roughly 5.3 s on both runs: without the
cache every process recompiles from nothing, matching Axis's behavior before
this cache was wired in. The warm run wrote zero new cubins (still the 16
the cold run wrote), confirming every kernel the step needs was already
served from disk.

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

## `grouped()`'s one-block-per-group launch, and giving it more blocks

The same ImageNet64 profile's `ncu` trace found the other half of the gap in
the kernel `Device::grouped` launches, not just how often it ran: the bias
gradient's plan groups by output channel (16/32/64/1000 for the four
layers), and `grouped`'s launch used exactly one CUDA block per group, so
conv1's 4,194,304 contributions ran as 16 blocks on an 84-SM GPU -- 2.08%
occupancy, 34.5ms for that one kernel alone. The mechanism is not bad in
general (the same kernel doing a global-average-pool reduction, 1,048,576
contributions into 16,384 groups, measured at 24.86%/23.60%
compute/occupancy); it is bad specifically when the group count is tiny,
which a channel-wise bias gradient always is.

`Device::grouped` now runs in two stages. `chunked_offsets` (top of
`backend.rs`) splits each CSR group's contribution range into chunks of at
most `GROUPED_CHUNK` (512) -- a chunk never crosses a group boundary, so a
group at or under that size still gets exactly the one chunk it always had.
Stage one (`kernels::grouped_partial`) launches one block per chunk across
every group at once and writes each chunk's own sequential partial sum;
stage two (`kernels::grouped_combine`) launches one block per group and
sums that group's partials, in increasing chunk order, into the final
scaled output. Both stages sum in a fixed order determined entirely by the
plan, not by GPU scheduling, so the result is deterministic and
reproducible -- a reassociation of the same terms the single-block kernel
summed, not a race; a group of contributions large enough to matter (like
a bias gradient's, but not most other `grouped` consumers' typical group
sizes) trades one long serial sum for many short ones plus one short
combine. `grouped_reduction_spans_multiple_chunks_for_a_large_group`
exercises the actual multi-chunk path (600 contributions per group, over
`GROUPED_CHUNK`) for both directions `grouped` is used in: `align`'s
backward Group plan (the bias gradient) and `reduce_grouped`'s forward
Group plan (`sum`/`mean`), checking an exact-integer gradient a correct
reassociation still has to land on exactly.

Chained onto the plan-caching fix above, the same desk RTX 5060
microbenchmark (100 steps of `[256,16,32,32] + [16]` broadcast-add, mean,
backward) moved from 410.2ms/step (plan caching alone) to 170.0ms/step, a
further 58.5% reduction and 89.3% off the original 1594.7ms/step baseline
(9.4x). Real ImageNet64 numbers on the tower are in the PR.

## Declared forward-kernel preparation

`Device::prepare_kernels(&[KernelSpec])` compiles a declared list of contiguous
matrix-product and softmax shapes, including their output zero-fill kernels,
without allocating tensor storage or executing those kernels. Metadata-only
cuTile tensors produce the same specialization keys as the actual execution
path. Matrix preparation follows the device's FP32/BF16 mode. Every declaration
is validated before any compilation; zero or overflowing dimensions are errors.

Use this synchronous API during startup, before the latency-sensitive first
step. It front-loads compilation and warm-cache module loading; it does not
reduce total startup time or compile in parallel. Layout copies, other operators
and backward kernels remain lazy. Consumers must declare their actual lowered
shapes; this is not graph capture or automatic whole-model discovery.

The RTX 5060 synthetic witness evaluates 30 matrix-product/softmax pairs (47
unique kernel specializations, including zero-fill). Independent scalar outputs
pass, and exact output fingerprints agree across all four fresh-process runs:

| Process cache state | Preparation | First step | Cache hits / misses |
| --- | ---: | ---: | ---: |
| Cold, lazy | 0 ms | 17,712.551 ms | 0 / 47 |
| Cold, prepared | 17,632.633 ms | 10.917 ms | 0 / 47 |
| Warm, lazy | 0 ms | 3,637.746 ms | 47 / 0 |
| Warm, prepared | 3,485.925 ms | 10.299 ms | 47 / 0 |

Prepared first steps invoke no further cache lookup or compilation; warm
processes still use the same disk cache and compile nothing. The witness ran
on a shared workstation while another CUDA suite was running, so timings are
observations, not an isolated throughput comparison. BF16 and all-declarations-
validated-before-compilation tests pass separately. These numbers do not claim
a complete beat_this preparation recipe or its two-second song target. Raw
receipt: [kernel preparation](../../data/evidence/kernel-preparation.log).

## Inference buffer retirement

Outside `Trainer`, the backend checks completion every 32 allocations or 32 MiB of requested
allocation, whichever comes first, and
moves buffers whose only remaining owner is its pending list into an event
batch. Recording after the last submitted use matters: recording only at
allocation would permit a later use to race a free. Live tensor/gradient/plan
owners keep their references. Host upload arrays and temporary integer buffers
follow the same rule. Completed batches are freed; if uncompleted batches exceed
256 MiB, inference waits for the oldest event. The allocation interval and live
buffers are additional to this backlog budget. Live autograd graphs cannot be
reclaimed until their owners release them.

The Trainer step uses an unwind-safe thread-local scope to skip this maintenance,
so it neither polls nor records/waits for retirement events during its existing
single-boundary execution. Reads and explicit synchronization still clear all
pending batches. No API change or consumer synchronization is required.

The RTX 5060 synthetic witness repeats a 1 MiB FP32 operation. Retaining every
intermediate (the prior policy) held 512 MiB after 512 steps; automatic retirement
held at most 97 MiB over 8,192 steps, with exact outputs and no caller-inserted
synchronization in either loop. Driver-used memory increased by 480 MiB in the
baseline and about 1.7 MiB during the subsequent automatic-retirement run; the
allocator reused the baseline's pool, so these are incremental observations,
not comparable absolute process peaks. Live aliases and later gradient inputs
also pass. See [the raw receipt](../../data/evidence/buffer-retirement.log).
This witnesses bounded unused-intermediate retention, not a measured beat_this
transformer speedup or a bound on caller-owned activations.

A warmed MLP Trainer comparison (500 steps, seed/data/model/SGD unchanged)
reported 0.73 and 0.81 seconds before retirement, and 0.78 and 0.73 seconds
after it, in before/after/after/before order. Every final loss was identical
at printed precision. This short probe shows no observed regression; it is
not a general throughput guarantee. Binary hashes and full outputs are in
[the training receipt](../../data/evidence/retirement-training.log).

A separate large-intermediate witness runs 80 operations with 128 MiB outputs
(10 GiB allocated over the loop), with exact final values and at most 640 MiB
retained. The byte-triggered poll prevents waiting for 32 such allocations
before beginning retirement. The event backlog budget excludes the current
batch and caller-owned inputs/outputs. See
[the large-buffer receipt](../../data/evidence/large-buffer-retirement.log).
