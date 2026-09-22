---
title: An axis-based training crate, grown from concrete programs
status: experimental implementation
---

# An axis-based training crate

The intended interface began with the owner's [API sketch](api-sketch.md). It is an API
sketch with four target programs, not a compiled library. The first two
concrete implementations are [the MLP](../../src/examples/training/mlp/src/baseline.rs) and
[the CNN](../../src/examples/training/cnn/src/train.rs). Their jobs are to establish working math,
gradients, and execution behavior before those mechanics move behind the API.

The central boundary is **users manipulate named axes; the backend maps those
axes to dimensions, strides, and tiles**. This is a semantic tensor algebra.
The training convenience follows from that algebra and automatic derivatives.
The enforcement ladder from typed structure through empirical assertions is
documented in [static guarantees](static-guarantees.md).

## Implemented first slice

The `axis` package now exposes `axis` from [src/library/src/lib.rs](../../src/library/src/lib.rs).
[src/examples/training/mlp/src/train.rs](../../src/examples/training/mlp/src/train.rs) is the first executable consumer: the
original MLP expressed with named axes, automatic derivatives, Linear/ReLU,
explicit mean reduction, and SGD. The original standalone trainers and scalar
oracles remain unchanged for comparison.

[Causal attention](../../src/examples/training/attention/README.md) is the second executable
consumer. Its query/key/value and output projections reuse Linear, Module
parameter traversal, and SGD. Attention composition stays in that program;
only named-axis softmax and causal masking join the tensor algebra.

[The CNN](../../src/examples/training/cnn/README.md) is the third consumer. Conv2d and
Conv3d compose named patch extraction with a channel-grouped contraction;
global pooling is the existing named mean. Its explicit cuTile program remains
beside it as a lower-level baseline.

Implemented algebra includes identity/extent binding, strict subset-axis
broadcasting, contraction with shared batch axes, physical layout changes,
rename/role axes, split/merge, explicit outer products, causal and
prefix-causal masking, named-axis softmax, and differentiable named-axis minimum. Parameters have
stable IDs and version checks; repeated uses accumulate gradients and SGD
updates each shared parameter once. Backward releases its saved graph; a
second backward through it reports an error. Leaf gradients accumulate until
cleared, and replacing a parameter starts a fresh leaf.

Transforms whose physical index map is the identity are metadata views: an
already-satisfied `with_layout`, a contiguous `split`, and the corresponding
`merge` share storage and add no CUDA launch. Extent-one dimensions do not force
a copy merely because their irrelevant strides differ. A real permutation
materializes through the compact device selection copier with no axis removed;
its inverse uses the same rank-sized extent/stride specification for backward.
`split` and `merge` describe the required physical order, materialize it only
when needed, then change shape metadata. No element-sized layout or merge index
plan is built, cached or uploaded. Selected-axis order remains significant.

Research assumptions can also fail executable checks. `SinglePass` guards the
finite-corpus consumption budget. `Idr` consumes stable sample IDs and enforces
declared coverage and repeat-rate limits for finite or generated streams.
`DataLoader` applies these guards before releasing batches, while the counters
remain independent of optimizer and tensor execution; see
[data regimes](../contracts/data-regimes.md).

`Adam` and `AdamW` keep first/second moments and parameter updates on the GPU.
Their constructor defaults use beta1 `0.9`, beta2 `0.999`, and epsilon `1e-8`;
learning rate remains explicit, and AdamW also requires explicit decoupled
weight decay. `SGD`, `Adam`, and `AdamW` each expose `set_learning_rate` to
overwrite that rate before a step; `cosine_annealing_lr` and `one_cycle_lr`
are pure functions of the step index that reproduce PyTorch's own
`CosineAnnealingLR` and default-configuration `OneCycleLR` value sequences
(`optim.rs` doc comments cite both closed forms) for a consumer's training
loop to call once per step and feed to the setter — Axis has no stateful
scheduler object.

`clip_grad_norm` matches PyTorch's `clip_grad_norm_`: one global L2 norm over
every listed parameter's gradient together (never a per-tensor norm), scaling
all of them by the same `max_norm / (total_norm + 1e-6)` factor only when the
norm exceeds `max_norm`, and returning the pre-clip norm either way. It is a
free function over `impl IntoIterator<Item = Parameter>`, not a `Trainer`
option or an `Optimizer`: consumers that clip call it once themselves between
`loss.backward()` and the optimizer's own step, exactly where every sampled
consumer's own Python loop already calls `clip_grad_norm_`. Each parameter's
squared-sum reuses `Tensor::mean_square` over every axis, scaled back up by
the element count it divided by, since the crate has no direct sum reduction
yet; every reduction and the cross-parameter accumulation stay on device, and
the whole parameter set costs exactly one host read (for one `sqrt`), not one
per parameter.

`Muon` applies EMA momentum, optional Nesterov interpolation, five-step
Newton-Schulz orthogonalization, original rectangular-matrix scaling, and
decoupled decay to explicitly oriented rank-2 parameters. `MuonWithAuxAdamW`
validates that every selected `ParamId` belongs to the model and routes the
exact deduplicated remainder to AdamW. Both partitions prepare before either
commits. The orientation, scalar oracles, mixed-precision behavior, and current
limits are documented in [the Muon contract](muon.md).

`Tensor::gelu()` and the `GELU` module retain the tanh formulation. The explicit
`Tensor::gelu_exact()` operation and `ExactGELU` module instead evaluate
`x * Phi(x)` (erf-form GELU),
with analytic derivative `Phi(x) + x * phi(x)`, for imported models such as
MobileSAM that were trained with exact GELU. This is the distinction documented by
[PyTorch's GELU contract](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.activation.GELU.html).
It does not promise bitwise identity with a particular PyTorch/libm implementation.

cuTile 0.3.1 exposes no erf primitive. The implementation evaluates the normal CDF
using the five-coefficient rational approximation in Abramowitz and Stegun 26.2.17
(also reproduced in [NASA's normal-distribution reference](https://ntrs.nasa.gov/api/citations/19980045313/downloads/19980045313.pdf)).
Its mathematical CDF approximation error is below 7.5e-8; FP32 arithmetic adds
rounding error. Negative inputs use the small tail directly, avoiding cancellation.
“Exact” names the GELU formulation, not exact real arithmetic. The CUDA test uses
independent f64 Simpson integration, not the kernel's polynomial, and requires
forward error below 2e-6 and derivative error below 5e-7 over 8,193 inputs spanning
[-8,8], signed zero and large finite tails, with noncontiguous storage and partial
kernel tiles. Nonfinite inputs and bitwise cross-backend parity are not certified.
The existing tanh-GELU regression remains unchanged.

`SiLU` and `LeakyReLU::new(slope)?` each lower to one cuTile kernel in forward
and reverse mode. Leaky-ReLU slopes must be finite and non-negative; the
derivative at exactly zero is the negative slope, matching PyTorch. The
independent f64 oracle
covers 2,051 values, reordered storage, odd partial tiles, both forward paths and
both reverse-mode derivatives.

An additional ignored parity test can compare those three activations directly
with an installed PyTorch at runtime. Run it with
`AXIS_PYTHON=/path/to/python bash scripts/cargo.sh test -p axis pytorch_activation_forward_and_gradient_parity -- --ignored --nocapture`.
This dynamic comparison supplements the independent scalar oracles; it does not
replace them or make PyTorch a build dependency.
The adjacent `pytorch_activation_forward_performance` test also emits a warmed,
synchronized forward-latency receipt for one million elements. It is a local
regression signal rather than a framework-wide throughput claim.

The matched fragments use each framework's direct public spelling:

| Operation | PyTorch | Axis |
| --- | --- | --- |
| exact GELU | `F.gelu(x, approximate="none")` | `x.gelu_exact()?` |
| SiLU | `F.silu(x)` | `x.silu()?` |
| leaky ReLU | `F.leaky_relu(x, negative_slope=0.125)` | `x.leaky_relu(0.125)?` |

Parity cases must hold dtype, device, input, reduction, and differentiation
semantics constant. Axis should improve the spelling or the executable
guarantees and should meet or beat the measured PyTorch path before documentation
calls it faster.

Normalization follows the named axes that define the statistic rather than a
rank suffix. `LayerNorm::new(feature)?` remains the ordinary single-axis form;
`LayerNorm::new([row, feature])?` learns scale and bias over the complete
declared shape. `RmsNorm` has the same axis forms and learns scale without a
bias. Their defaults are epsilon `1e-5` and `1e-6`, respectively.

`GroupNorm::new(channel, groups, [height, width])?` divides the named channel
extent into contiguous groups, computes population moments over each group's
channels and the declared sample axes, then applies per-channel scale and bias.
The channel extent must divide evenly by the positive group count.
`InstanceNorm::new(channel, [height, width])?` is the groups-equal-channels
case. It requires at least one sample axis and defaults to no affine parameters.
Both use epsilon `1e-5`. `.epsilon(value)?` and `.affine(bool)` make departures
from those defaults visible. Any unrelated axes remain independent instances;
physical storage order does not change the statistic.

All four modules are stateless: training and evaluation have identical
behavior, and there are no running estimates. `Tensor::moments` and
`Tensor::mean_square` provide their reusable population reductions. Batch
normalization remains deliberately absent until `Module` can represent an
explicit train/evaluation mode and persistent non-parameter state; silently
substituting batch-local statistics would give it the wrong experimental
meaning.

`Embedding::new(vocabulary, feature.of(d))` looks up one learned vector per
vocabulary entry. Axis tensors are floating point, so a token arrives as a
one-hot coordinate on the named vocabulary axis rather than as an integer
index; the lookup is a contraction of that axis with a `[vocabulary, feature]`
table and has the same value and derivative as an index gather, so a repeated
token accumulates every contribution into its row. The device does not check
that the input is one-hot; a soft input is a weighted mixture of rows.
`PositionEmbedding::new(position, feature)` adds a `[position, feature]` table
whose extents are read from the input at build and broadcast over every other
axis. Both start uniform in `[-0.02, 0.02]`, expose one `table` parameter,
and reject an extent that differs from the one they were built with. The byte
autoencoder migration is their first consumer; Sudoku previously spelled both
inline as a bias-carrying Linear and a raw Parameter.

`Tensor::prefix_causal_mask(query, key, prefix)` keeps the first `prefix` key
positions visible to every query and masks the remainder causally, which is
the one pattern a plain causal mask cannot express: a block of memory or code
tokens in front of an autoregressive sequence. `causal_mask` is the
`prefix == 0` case. Both remain square, zero-offset, logical dense masks;
cached decoding and padding masks still need their own contracts.

`Tensor::sign_straight_through` and the `SignStraightThrough` module quantize
to `{-1, +1}`: `+1` where `x > 0`, otherwise `-1`, so `x == 0` maps to `-1`.
This reproduces the byte autoencoder's `bitae.quantize` two-level branch,
`torch.where(z > 0, 1.0, -1.0)`, deliberately rather than `torch.sign`, which
gives exactly `0` at `x == 0`. The backward is the identity: bae's
`z + (hard - z).detach()` stops the hard threshold's own gradient, so the
whole local Jacobian is the straight-through pass-through, `Rule::Identity`.
Axis leaves any squashing (bae applies `tanh` first) to the caller rather than
folding it into this op, matching bae's own composition of the two steps.

`Tensor::uniform(dims, seed, low, high, device)` and `Tensor::normal(dims,
seed, mean, std, device)` are the public, seeded random tensor constructors:
`world/energy-output` needs a fresh reproducible Bernoulli mask drawn every
epoch from one seed, and `world/fluid` needs seeded Gaussian noise added to
training inputs. Both draw from the same shared xorshift64 stream that
`Linear`/`Embedding`/`PositionEmbedding` initialization already uses
(`model/nn.rs`'s `uniform_values`, itself now a thin wrapper over the shared
`crate::tensor::xorshift_unit_stream`), so a seed reproduces bit-exact values
everywhere it is used, on any run or machine. `uniform` rescales each raw
`[0, 1)` sample to `[low, high)`; `normal` consumes successive raw pairs
`(u1, u2)` through the Box-Muller transform, `z0 = sqrt(-2 * ln(1 - u1)) *
cos(2*pi*u2)`, `z1 = sqrt(-2 * ln(1 - u1)) * sin(2*pi*u2)`, each scaled to
`mean + std * z`; an odd element count drops the unused second value of the
final pair. Using `1 - u1` rather than `u1` keeps the logarithm defined on
the shared stream's documented first-sample-exactly-zero seeds, at the cost
of collapsing that pair's first two normal draws to exactly `mean`. Both
constructors generate their values host-side, then upload them exactly like
`Tensor::from_slice` — the simplest honest implementation, and adequate for
the small per-epoch draws both consumers need. A random draw has no upstream
input, so neither constructor produces a gradient edge; it is a constant a
program can then compose with `mul`/`add` like any other tensor, not a
trainable parameter. Neither is dropout (a train/eval-mode masking module,
tracked separately) or an `RNG`-state object: each call is deterministic in
its seed alone, with no mutable generator to advance or restore.

`DataLoader` batches any `DataSource` and checks its regime before releasing a
batch. Finite `InMemoryDataset` sources report corpus size; generated sources
such as `AdditionDataset` do not invent one. Each guarded batch carries a
printable PASS receipt. `Trainer` standardizes zero-grad, fresh scalar-loss
construction, backward, and optimizer update while leaving the mathematical
loss expression in the program.

`model.parameter("0.weight")?` and `named_parameters()` expose structural
paths, including nested Sequential indices. Paths identify slots, while
`ParamId` identifies shared storage within a process; two tied slots can
name the same parameter. Missing or ambiguous paths return errors. The MLP
uses these paths to load its oracle weights without depending on iteration
order. This is not yet a checkpoint format.

The first backend is stream-ordered CUDA using cuTile kernels. An eager training
step enqueues allocations, forward arithmetic, derivatives, and optimizer
updates without per-operation synchronization, retains every device/host buffer
until the work completes, then synchronizes once at the Trainer step boundary.
Explicit host reads also synchronize. CPU code builds index plans; these
generic gather/reduction plans prioritize verifiable semantics. They are
limited to 16,777,216 contributions per operation and are not a competitive
GEMM implementation. Single-axis contractions take a batched tiled cuTile
matrix path. Axis caches host plans by named shape and layout, and retains up
to 256 MiB of uploaded plans per device; larger and one-off plans remain
step-local so a stable-shape speedup cannot grow device memory without bound.
Layout permutations and split/merge no longer use these generic plans. Their
metadata contains three integers per logical input axis; no merge cache remains.
Broadcast and reduction planning still have their own cache and contribution
limits. This distinction matters for cold shapes and large convolution patches.
The measured execution profile and claim boundary are recorded in
[the eager execution profile](../backend/execution.md).
Axis reordering is materialized when the batch, row, reduction, and column
groups are not already contiguous; this covers both `Linear` and attention.
Tensors participating in an operation must share the same `Device` handle.
There is no CPU fallback, higher-order differentiation, or retained-graph
mode. Softmax forward and backward use one tiled reduction per contiguous row,
padding non-power-of-two widths inside the tile; they no longer inherit the
generic index-plan contribution bound. Attention still composes separate
contraction, mask, softmax, and value-contraction operations rather than using
a fused attention kernel. Multi-axis contractions still use the generic plan
path.

Configured 2D and 3D convolution lower padding and stride through one compact,
rank-tagged unfold specification, then lower each channel group to the batched
single-axis contraction. Rank-specialized 128-lane patch kernels compute source
indices from O(rank) metadata and write group-major, matrix-ready storage while
preserving the logical named shape. Reverse mode is an input-centric
deterministic col2im: each physical input element enumerates its contributing
windows and is written once, without atomics or a reverse index allocation.
The contraction and both of its derivatives use tiled matrix multiplication.

This removes convolution's former 16,777,216-contribution plan limit and the
much larger host/device index allocations behind it. The materialized patch
tensor remains: for example, `[batch=128, channel=32, height=32, width=32]`
with a 3x3 depthwise kernel contains 37,748,736 FP32 patch values (144 MiB).
The path establishes exact semantics and autodiff; it is not a fused or direct
convolution implementation and has not established competitive convolution
throughput. A 3D patch tensor grows with output volume times input channels and
kernel volume, so it can dominate memory. Patch extraction itself bypasses the
generic plan ceiling, as do layout permutations and split/merge, but following
broadcast/reduction operations retain their own 16,777,216-contribution limit.
Fully padded windows produce exact zeros without indexing the input. Because
the cuTile kernels use signed 32-bit coordinates, every composite padded
spatial extent must fit `i32`; larger geometry is rejected before a plan is
cached or a patch output is allocated.

`MaxPool2d`/`MaxPool3d` reduce each channel independently, materializing the
same rank-tagged patch as convolution but with the patch fill set to negative
infinity instead of zero, so a padded position can never win the window; the
maximum is then `-min(-patch)`, reusing `Tensor::min`'s existing tie-break
(the first logical coordinate) and finite-only comparison unchanged. Because
patch materialization is itself a proper partition of the (larger) patch-space
tensor, `min`'s ordinary bijective backward is exact there, and convolution's
own col2im already sums overlapping window contributions back onto the shared
source exactly, so no new backend kernel was needed for either padding or a
stride smaller than the kernel. Padding must be at most half the kernel extent
per axis, PyTorch's own `MaxPool` constraint, which also guarantees every
window keeps at least one real element. The channel axis is appended after the
(in-place, resized) spatial axes, exactly as `Conv2d`/`Conv3d` append their
output-channel axis. `Tensor::adaptive_avg_pool3d` instead reduces each spatial
axis to an explicit target extent using PyTorch's own start/end bin formula
(`floor(i * in / out)`, `ceil((i + 1) * in / out)`); adjacent bins can share
one boundary element when the input extent does not divide evenly, so unlike a
fixed-kernel pool its reduction is not a strict partition. It is implemented as
a single weighted grouped reduction (the same CSR product-sum kernel
`contract` already exercises) built from both directions at once: forward sums
each bin's members scaled by its own `1 / size`, and backward gathers, per
input element, every bin that contains it, scaled the same way -- exact for
overlap without a new kernel or a host round trip.

`Device::cuda_bf16` is an explicit mixed-precision policy: matrix-product
inputs are rounded to BF16 inside the kernel and accumulated into FP32. Stored
parameters, activations outside matrix products, reductions, gradients, and
Adam state remain FP32. `Device::cuda` retains full FP32 matrix products. This
keeps precision choice visible in the experiment rather than changing global
semantics silently.

The [ImageNet64 starter](../../src/examples/getting-started/imagenet64/README.md)
adds the first large, access-controlled vision path. A Rust preparer validates
the published NPZ arrays and writes labeled fixed-size records, allowing one
sample to be sought without retaining a multi-gigabyte shard. Its compact
three-block CNN is an end-to-end API witness. Current convolution still
materializes patch tensors, so this is not a throughput or ImageNet-accuracy
claim. The unlabeled RGB byte benchmark in the sibling research repository can
verify pixels but cannot verify classification.

### Hosted API documentation

Normal Axis builds require CUDA because the cuTile dependency generates its
bindings from the installed toolkit. docs.rs deliberately provides no CUDA
toolkit, so its build disables Axis's default `cuda` feature and compiles a
type-compatible, non-operational backend solely to let rustdoc traverse the
same public tensor, model, training, and research-contract modules. Axis's
build script selects that backend only when docs.rs sets `DOCS_RS`; disabling
default features anywhere else is a compile error. The documentation path is
therefore not a CPU runtime or an alternate installation mode.

`tests/docsrs.sh` packages Axis first, then tests the artifact that crates.io
will receive. It proves three boundaries in fresh target directories: rustdoc
succeeds with both CUDA variables absent under `DOCS_RS`, the default build
still enters cuTile and reports a missing toolkit, and a no-default-features
build outside docs.rs is rejected. CI runs this contract on an ordinary Ubuntu
runner without installing CUDA.

Convolution dilation, built-in asymmetric convolution padding, and a full Transformer remain future
slices. The single-layer forward LSTM is an eager correctness path with explicit
state; it is not a fused recurrent kernel or a sequence-throughput result.

Run from the research node:

```bash
bash scripts/check.sh
bash src/examples/training/mlp/scripts/train.sh --smoke
bash src/examples/training/mlp/scripts/train.sh
bash src/studies/contracts/generated-addition/scripts/train.sh --smoke
bash src/examples/training/muon/scripts/train.sh --smoke
```

The verification script runs formatting, Clippy, the CPU shape test, and the
explicitly enabled GPU tests in the library, MLP, attention, and CNN consumers.
Algebra checks are in [src/library/src/tests.rs](../../src/library/src/tests.rs),
and the [MLP checks](../../src/examples/training/mlp/src/tests.rs) compare
all MLP predictions, parameter gradients, input gradients, and one update to
the scalar f64 oracle. Separate checks cover odd extents, short batches, extra
time axes, noncontiguous storage, both contraction derivatives, split/merge,
broadcast gradients, tied modules, stale parameter versions, and invalid axes.
The [measured MLP run](../../src/examples/training/mlp/README.md#library-mlp) establishes the first
slice. [Attention checks](../../src/examples/training/attention/src/tests.rs) additionally compare
every Q/K/V gradient against f64 central differences, verify causality and
layout invariance, and exercise stable softmax with extreme finite logits.

## Identity, extent, and layout

Use the sample's later formulation:

- `Axis`: opaque identity plus a diagnostic name. Creating two axes named
  `"feature"` creates two distinct identities. Pass the original handle to
  connect operations; avoid a global `Axis::named` lookup.
- `Dim`: one axis bound to one extent on a particular tensor.
- `Shape`: an ordered collection of unique axis bindings. Logical order makes
  iteration, serialization, and diagnostics reproducible.
- `Layout`: the mapping to physical strides and storage. Changing layout
  preserves the logical value and axis identities.

Pooling can preserve `height` while changing its tensor-local extent from 32
to 8. That does not authorize adding two tensors whose shared `height` axes
currently have different extents. Check compatibility at each operation.
An unbound axis has no `.extent()`; attention obtains the scale from
`q.extent(head_feature)` or its bound module specification.

For ordinary stable Rust, use `batch.of(128)` to construct a `Dim`. The
sample's `batch(128)` would require a callable representation or extra syntax
machinery; directly implementing the call traits on an Axis struct uses
unstable Rust APIs ([Fn documentation](https://doc.rust-lang.org/std/ops/trait.Fn.html)).
Start with an ordinary method and keep the semantic contract independent of
that spelling. Dynamic extents and fresh runtime identities suggest runtime
shape checks first; compile-time axis types can be evaluated later.

## Introductory vision boundary

The getting-started vision programs share an internal `axis-vision-data`
crate. It owns byte-format validation, train-only channel normalization,
one-hot packing, exact finite-pass training, and batched held-out evaluation.
Each program still constructs its own model and declares its own dataset path,
budget, and claim. This keeps the examples consistent while leaving the
scientific and architectural choices visible to a new reader.

The helper is not published as framework API. A dataset adapter should move
into `axis` only after outside consumers establish a stable representation and
provenance contract; reading a file is not itself a research guarantee.

## Operation contracts to settle first

| Operation | Contract |
|---|---|
| `Linear(input, hidden.of(32))` | Contract the named input axis, introduce the output axis, preserve every unrelated axis. Bind the input extent when building the model. Carries a learned `[output]` bias by default; `.bias(false)` before `build` omits it entirely, so `named_parameters` has only `weight`. |
| `x.squared_error(y)` | Align identical axis sets by identity, require equal extents, preserve the unreduced shape. |
| `x.abs()` | Elementwise absolute value, layout and axes unchanged. Backward is `gradient * sign(x)`; matching PyTorch's `abs` backward, the gradient is exactly zero at `x == 0`, not a one-sided slope or `NaN`. |
| `x.absolute_error(y)` | `(x - y).abs()` (PyTorch's unreduced `F.l1_loss`). Same axis-agreement contract as `x.squared_error(y)`, leaving the reduction to the caller. |
| `x.mean(axes)` | Remove precisely those axes; backward broadcasts and divides by their extent product. Scalar `.backward()` requires all loss axes to have been reduced. |
| `x.sum(axes)` | Remove precisely those axes; backward broadcasts the upstream gradient across them unchanged (unit local derivative per contributing element). Runs `mean`'s own group-sum kernel with a constant scale of `1.0` in place of `mean`'s `1 / extent`, so it recovers a caller's own division (e.g. by a data-dependent count) exactly where the divisor is not a compile-time constant. |
| `x.moments(axes)` / `x.mean_square(axes)` | Require at least one named axis and return population statistics with precisely those axes removed. Gradients broadcast through the original logical axes. |
| `x.sin()` | Apply elementwise sine in radians without changing axes or layout; backward multiplies by cosine of the saved input. |
| `x.exp()` | Elementwise `exp(x)`; backward multiplies by the saved output (`g * exp(x)`), the same trade `tanh` makes. |
| `x.ln()` | Elementwise natural logarithm; backward is `g / x`. IEEE behaviour, no clamping: `ln` of a non-positive input is `-inf` at `x == 0` and `NaN` below it, matching `f32::ln`/`torch.log`. |
| `x.softplus(beta, threshold)` | PyTorch-exact `Softplus`: `(1/beta) * ln(1 + exp(beta*x))`, except `x` itself where `beta*x > threshold` (`torch.nn.functional.softplus`'s own linear seam, computed exactly rather than through the logarithm so neither branch can overflow). Backward is `sigmoid(beta*x)` on the logarithmic branch and exactly `1` on the linear branch. `beta` must be finite and positive; `threshold` must be finite. |
| `CentralDifference(coordinate, step)` | Record which coordinate was shifted, require a finite positive step and identically ordered shapes on one device, and construct differentiable first or second centered stencils. It does not prove the caller's sampling or discretization method. |
| `x.min(axis)` | Remove one named axis and preserve unrelated axes. Ignore NaN and infinities, choose the first logical coordinate on finite ties, and route backward only to that winner. A group with no finite value returns NaN with zero derivative even under a non-finite upstream derivative. Forward and backward remain device-resident. |
| `x.max(axis)` | Exact mirror of `x.min(axis)`: same axis-removal contract, the same first-logical-coordinate tie rule, and the same NaN/zero-derivative empty-group behavior, over the maximum instead of the minimum. No `argmax`. |
| `x.logsumexp(axis)` | `m + ln(sum(exp(x - m)))` with `m = x.max(axis)` detached before the subtraction, matching `torch.logsumexp`; composed entirely from `max`/`sub`/`exp`/`sum`/`ln`/`add`, no dedicated kernel. Backward is exactly `softmax(x)` along `axis`, since the detached `m` carries no gradient edge. A group with no finite candidate inherits `max`'s `NaN` (not PyTorch's `-infinity`), and -- unlike `max`, which has a dedicated rule to zero that case's gradient -- the `NaN` propagates through the ordinary composition instead of landing on zero. |
| Elementwise add/multiply/divide | Align shared identities with equal extents. Permit scalar or subset-axis broadcasting, such as a `[hidden]` bias on `[batch, hidden]`. `x.div(y)` applies no epsilon or clamping; division by zero yields IEEE `inf`/`nan`, as in PyTorch, and callers that need a safe denominator (such as a masked-mean count) must clamp it themselves before dividing. |
| `x.gt(scalar)` / `.ge(scalar)` / `.lt(scalar)` / `.le(scalar)` / `.eq(scalar)` | Elementwise comparison against a scalar into a `{0.0, 1.0}` mask, IEEE-ordered so any comparison against `NaN` is `false`. Axis is floating point only, so this is the boolean dtype: no separate bool tensor exists. The mask carries no autograd edge at all -- not even a zero one -- matching PyTorch, where comparisons are non-differentiable; a downstream `x.mul(&mask)` still differentiates correctly with respect to `x`, treating the mask as a constant. Only the scalar form exists: every migrated consumer compares against a scalar, never against another tensor. `x.logical_and(&mask)` (`mul`) and `x.logical_not()` (`1 - x`) compose masks the way PyTorch's `&`/`~` do; logical OR has no named method since nothing calls it, but on 0/1 masks it is an elementwise maximum. |
| `x.clamp(min, max)` | Elementwise bound into `[min, max]`; either `Option<f32>` bound may be `None` to leave that side unbounded. Rejects a `NaN` bound or `min > max` before launch. `NaN` elements of `x` propagate unclamped (every ordered comparison against `NaN` is `false`, so neither branch fires). Gradient passes through unchanged where `min <= x && x <= max` -- including exactly at either bound -- and is zero elsewhere, matching PyTorch's `clamp` rather than a "zero at the boundary too" convention; a `NaN` input therefore also carries a zero gradient. |
| Incomparable axis sets | Require explicit expansion. `[batch, time] + [batch, hidden]` must not silently create `[batch, time, hidden]`. `x.broadcast_to(shape)` is that explicit expansion: `shape` must carry every axis `x` already has, at `x`'s own extent, and may add axes `x` lacks entirely (read with stride zero) or reorder `x`'s existing axes. Two operands whose axis sets are genuinely disjoint -- an outer op, such as a `torch.cdist`-style pairwise distance between a `pixel`-indexed and a `site`-indexed tensor -- each broadcast onto one shared `[pixel, site, ...]` shape first, then combine with an ordinary elementwise op: `a.broadcast_to(&shape)?.sub(&b.broadcast_to(&shape)?)`. Backward sums the upstream gradient over every axis a call added, the same rule every other broadcast in the crate already follows. |
| `contract(rhs, axes)` | Sum over the specified shared axes. Align remaining shared axes and preserve distinct axes in a deterministic logical order. |
| `split` / `merge` | Validate extent products and axis uniqueness; preserve the mapping needed to undo the operation during backward. |
| `select(axis, coordinate)` | Remove one named axis at a checked logical coordinate. Compute offsets from compact rank-sized geometry and scatter its derivative back into the original physical layout. |
| `gather(axis, index, output)` | Replace one named axis with a new named axis sized by a host-side `&[usize]` index (checked against the axis extent before any device work); every other axis and the physical layout it reads carry through unchanged. Forward and backward reuse the same host-built, CSR-grouped index plan `mean`/`min` already use for reduction and broadcast (`Plan::gather`/`Plan::reverse`), so backward is an exact, deterministic scatter-add — a repeated index accumulates every contribution, an unpicked row gets zero — without `Embedding`'s dense one-hot contraction. Shares that plan's 16,777,216-contribution limit, counted against the gathered output's size, not the table's row count. |
| `argmin(axis)` | Host-side `Vec<usize>` of the coordinate `min(axis)` itself picked at each remaining position, in the same first-logical-coordinate-on-ties, finite-only order `min` already documents. Non-differentiable and entirely host-side: it reuses `min`'s own device-computed value as the sole comparison target rather than a second comparator. A group with no finite candidate is an error, since no coordinate names a NaN result. No `argmax`, matching `min`/`max`'s own asymmetry. |
| `scatter_add(axis, index, bucket, bucket_count)` | Sum values into `bucket_count` buckets named by a host-side `&[usize]` label per position of `axis` (checked against `bucket_count` before any device work); a bucket no position names is an exact zero. The exact transpose of `gather`: same `Plan::gather`/`Plan::reverse` CSR pair, built from the same host index, with forward and backward swapped relative to `gather`'s own use of them. `Tensor::bincount(index, bucket, bucket_count, device)` is its constant-ones case, for `torch.bincount` call sites that have no values tensor to scatter. |
| `pad_zeros(axis, before, after)` | Preserve logical axis order and add exact zero-valued coordinates independently on each side. Backward crops to the original extent and layout. Zero/zero padding shares storage. |
| `narrow(axis, start, length)` | Preserve the named axis and select a checked nonempty contiguous interval. Backward inserts exact zeros outside the interval. A full-axis interval shares storage. |
| `Tensor::stack(values, axis, position)` | Require identical named input shapes and devices; insert the new logical axis at the declared position. Store sources contiguously under a stack-major physical layout and slice each derivative back to its source. |
| `Tensor::concat(values, axis)` | Require an axis every operand already has, with every other axis identical by identity and extent; sum each operand's extent on that axis and preserve the first operand's axis order. Composed from `pad_zeros` and `add`: each operand is zero-padded into its own slice, then summed, so backward narrows the incoming gradient to each operand without a dedicated rule. |
| `x.roll(axis, shift)` | Cyclically shift one named axis by an integer `shift`, wrapping at the boundary; PyTorch's sign convention, element `i` moves to `(i + shift).rem_euclid(extent)`. Shift 0 or a multiple of the extent is the identity. Multi-axis roll is one call per axis, matching `torch.roll`'s `dims=`. Composed from two `narrow` slices reassembled with `concat`, so backward is the same call with `shift` negated, exact for free without a dedicated rule. |
| `Tensor::upsample_nearest(axis, factor)` | One named axis, exact integer scale factor only; every other axis preserved. Composed from `stack` (duplicate `factor` times along a fresh axis) and `merge` (fold that axis into the named one, spatial-axis major); no dedicated backward rule, since `stack`/`merge` are already differentiable. |
| `Tensor::resample_bilinear(axis, out_extent)` | One named axis, any positive output extent; every other axis preserved. PyTorch's `align_corners=False` half-pixel weights, built host-side as a fixed (non-learned) `[axis, resampled]` matrix and applied with `contract`; backward is `contract`'s existing transpose-weighted gradient, exact for free. |
| `causal_mask(query, key)` | Require distinct axes with equal extents; replace key positions greater than query positions with negative infinity and give them zero derivative. Square, zero-offset self-attention only. |
| `softmax(axis)` | Normalize along one named axis without changing logical shape. Subtract each row's maximum. Rows need at least one finite value; other values may be finite or negative infinity. |
| `masked_softmax(axis, mask)` | `softmax(axis)` restricted to a constant `{0.0, 1.0}` validity mask sharing every named axis (same contract as `masked_mean`'s mask, but an all-zero group along `axis` is the expected fully-masked case, not an error). Masked positions get exactly zero probability and exactly zero gradient; a group whose mask is entirely zero returns all zeros for that group rather than the `NaN` `torch.softmax(x.masked_fill(~keep, -inf), dim)` produces there, folding the caller's own `nan_to_num(neginf=0.0)` cleanup into the op. |
| `unfold2d(channels, spatial, patch, kernel)` | Extract valid stride-one patches, preserve unrelated axes, and replace channels with one flattened patch axis. Backward sums overlapping contributions into the input. |
| `Conv2d(input, output, spatial, kernel)` | Cross-correlate named spatial axes with configurable positive stride, finite symmetric zero-padding, and positive channel groups. Require both channel extents to divide evenly by groups; preserve unrelated axes and append the output-channel axis. |
| `Conv3d(input, output, spatial, kernel)` | Apply the same contract to three ordered named spatial axes. Flatten patches by input channel, then the three kernel coordinates with the final coordinate fastest. |
| `MaxPool2d(channels, spatial, kernel)` / `MaxPool3d` | Reduce each configured spatial window to its maximum, independently per channel; stride defaults to the kernel extent, padding to zero, and padding must be at most half the kernel extent per axis. A padded position never wins; ties route the gradient to the window's first logical coordinate. Preserve unrelated axes (spatial axes resized in place) and append the channel axis, matching `Conv2d`/`Conv3d`. |
| `Tensor::adaptive_avg_pool3d(spatial, target)` | Reduce three named spatial axes to explicit target extents using PyTorch's own per-axis bin formula (`floor(i * in / out)`, `ceil((i + 1) * in / out)`); adjacent bins may share one boundary element, which is averaged into each bin it falls in. Preserve every other axis in place. |
| Named normalization | Normalize only the declared axes. Layer/RMS affine parameters span the declared normalized shape; group/instance affine parameters span the channel axis. Preserve every input axis and reject changed built extents or group geometry. |
| `Lstm(input, hidden, time)` | Apply a standard IFGO transition in logical time-coordinate order. Preserve unrelated stream axes, replace input with hidden, accept explicit hidden/cell state, and return both the complete sequence and connected terminal state. |
| `binary_cross_entropy_with_logits(target)` | Return stable unreduced elementwise losses for identical axis sets. Targets are constants; the caller names every reduction axis. |
| `binary_cross_entropy_with_logits_weighted(target, pos_weight)` | Same contract, plus a positive-class weight (`torch.nn.BCEWithLogitsLoss(pos_weight=...)`) that scales the positive term before the negative term is added, changing the loss value itself. `pos_weight` follows the same subset-axis broadcasting as elementwise `add`/`mul` (a scalar or, e.g., one weight per class); targets and `pos_weight` are both constants. |
| `categorical_cross_entropy_with_logits(target, class)` | Accept constant one-hot/probability targets over the same axes, stably reduce the named class axis, and preserve all other axes. Backward is `softmax(logits) - target`. |
| `Conv(input, output, spatial)` | Transform channels and spatial extents by the stated stride/padding/dilation rules. Preserve all unrelated axes. |
| `cross_entropy(target, class)` | Consume the named class axis; target axes must exactly match the remaining axes. Return their unreduced losses; integer targets have no gradient. |

The sample's second convolution, `feature=32 -> feature=64`, exposes an
important distinction. A parameter tensor cannot contain the same Axis twice.
The layer therefore needs distinct internal input/output role axes, maps the
input role to the incoming `feature`, and restores the public `feature` on the
result. The same requirement appears in square linear projections and recurrent
weights. Parameter shapes cannot simply inherit both public handles unchanged.

`Conv2d::new` defaults to stride `[1, 1]`, padding `[0, 0]`, and one group.
`stride([sy, sx])`, `padding([py, px])`, and `groups(g)` follow the order of
the two supplied spatial axes. Each output extent is
`floor((input + 2 * padding - kernel) / stride) + 1`; a kernel that does not fit
the padded extent is rejected. Grouped weights have logical shape
`[group, input-channel-in-group × kernel-y × kernel-x, output-channel-in-group]`,
with the flattened patch ordered by input channel, kernel y, then kernel x.
Depthwise convolution is the grouped case where `g` equals the input channel
count (and commonly the output channel count). Reconfiguring a built layer to
a different group geometry is rejected before execution. Dilation and
asymmetric padding are not yet built into convolution. Explicit `pad_zeros`
can supply asymmetric input padding before a convolution with zero built-in padding.

`Conv3d::new` applies the same contract to three supplied spatial axes,
conventionally `[depth, height, width]`. Its kernel, stride, and padding arrays
follow that exact order, and its flattened grouped weights order each patch as
input channel, kernel depth, kernel height, then kernel width. Axis identity,
not the conventional name, selects each coordinate.

Attention makes role axes public: query-time and key-time must be distinct
identities even when both derive from time. Prefer a name such as
`time.role("query")` over an alias that might imply equality. Ancestry is useful
for diagnostics; identity alone determines alignment. Cached attention will
also need explicit positional offsets for the causal relation.

## Parameters and differentiation

Begin with reverse-mode differentiation of the operations the MLP needs:
contraction, bias broadcasting, ReLU, squared error, and named mean reduction.
Each recorded operation owns its saved values until backward releases them.
Broadcast backward reduces over the axes introduced by the forward operation;
contraction backward follows the named axes, regardless of physical layout.

Give parameters stable `ParamId`s. Gradients and Adam moments are keyed by those
IDs. `optimizer.step(&mut model)` obtains mutable access only for the update,
and a parameter shared by multiple modules is visited once. Gradients from all
uses accumulate before the step. Saved parameter versions must reject an
update between forward and backward, which otherwise silently differentiates
the wrong weights. This preserves the explicit update ordering in both trainers.

Constructors can describe layers before input extents are known. An explicit
`model.build(input_shape, device, seed)` binds unresolved extents, validates
axis contracts, allocates parameters, and initializes them. It never runs a
dummy forward. Calling it again with compatible extents and device preserves
the parameters and ignores the new seed. Later batch extents can vary;
a different bound feature extent requires a new model.
Fallible tensor/model operations should return `Result`, including shape,
device, and JIT errors. The sketch's omission of `?` is not an error policy.

Evaluation metrics belong to Axis when their semantics are independent of a
dataset. `Tensor::categorical_accuracy` performs named-class argmax comparison
on the device and returns exact `(correct, total)` counts, so chunked evaluators
merge counts instead of averaging percentages. `Standardizer` fits finite,
train-only values with an explicit variance correction (sample standard
deviation by default) and applies the frozen transform to any later split.

Scientific residual checks follow the same ownership rule. `EmpiricalResidual`
is equation-independent: it records a versioned law identity, residual
expression, region, evaluator, limits, and stable observed statistics. The
equation and its boundary conditions remain in the consumer. The receipt
verifies supplied sampled discretized values and explicitly does not prove a
global law. See [the PINN design](pinn.md).

The first backend submits the existing cuTile kernels in stream order and
synchronizes at explicit step/read boundaries. A small layout plan maps the
semantic operation to its kernel specialization.
Specialization keys describe shapes, layouts, and dtype, rather than arbitrary
new axis IDs or display names. Kernel tiling and padding remain backend choices;
valid extents cannot be silently truncated to a multiple of 16.

## What the two trainers establish

The [overlap census](overlap.md) identifies shared math and support functions.
They are evidence for a backend substrate. The two standalone trainers do not
implement named axes, autodiff, general layouts, or module composition; the
new library supplies these to its first MLP consumer.

The CNN adds patch extraction, spatial reduction, a one-output linear layer,
and BCE. Its head uses a dot reduction while the MLP uses GEMM: one logical
Linear can reasonably lower to different kernels. Its fixed-input patch cache
belongs to execution planning; it is not a rule that all tensors keep an
im2col copy. Likewise, MLP loss averages over batch and output while binary
classification averages only over batch. Explicit loss reductions capture this
difference directly.

The trained CNN still has one trainable convolution. Its input-gradient oracle
proves overlapping patch accumulation (`col2im`) for that operation. A second
operator oracle covers padded, strided, grouped forward values and all
derivatives from a noncontiguous input. A compact depthwise-ReLU-pointwise
block supplies the first two-convolution composition witness. It establishes
MobileNet-style mechanics, not MobileNetV2 architecture or performance.

## Build order and acceptance

Keep the initial crate experimental inside this research study. Add modules
as working cases require them; a separate family-wide repo can follow an
actual outside consumer.

1. **Axis algebra:** shape binding, alignment, contraction, explicit reductions,
   split/merge, and role renaming. Verify duplicate-axis rejection, same-name
   distinct identities, extent mismatches, and forbidden implicit outer products.
2. **Executable MLP slice:** cuTile tensor operations, parameters, tape,
   Linear/ReLU, squared error, mean, and SGD. Rewrite the current MLP using the
   proposed API and compare predictions, gradients, updates, and learning
   against its retained scalar oracle. Include gradients for input tensors.
3. **Attention primitive (implemented):** split heads, role axes, scaled
   contraction, causal mask, softmax, value contraction, and merge. Verify
   against independent f64 forward/finite differences before considering a
   reusable attention module or a full Transformer.
4. **Executable CNN slice (implemented semantics):** convolution with stride,
   symmetric padding and grouped/depthwise channels, ReLU, named global mean,
   Linear, and stable binary loss reproduce the concrete CNN and a compact
   depthwise-separable block. `MaxPool2d`/`MaxPool3d` and
   `Tensor::adaptive_avg_pool3d` (below) round out the pooling family. Collapse,
   categorical cross-entropy, dilation, and an optimized convolution kernel
   remain.
5. **LSTM (implemented correctness slice):** stable sigmoid/tanh, compact named
   selection, stack, explicit connected or detached state, and an eager IFGO
   recurrence. Its independent f64 oracle covers every input, initial-state,
   weight, and bias derivative. The O(T) transition/activation graph is not a
   fused scan and establishes no sequence-scale performance claim.
6. **Transformer:** admit a concrete program before promoting a composed module.
   `Repeat` must create independent blocks unless weight tying is explicit.

The first two library trainers must work with reordered physical axes and
with unrelated axes added. For example, the same Linear must accept
`[batch, time, input]` and a layout storing `[input, batch, time]`, preserve
batch/time, and produce matching gradients. Include odd extents such as 17
features and a final short batch so that tile-friendly examples do not become
an accidental public restriction.

Success is the existing training behavior expressed through short programs
with these contracts enforced. The four golden programs remain acceptance
targets, and each capability gains its status from a running witness.
