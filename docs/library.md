---
title: An axis-based training crate, grown from concrete programs
status: experimental implementation
---

# An axis-based training crate

The intended interface is the owner's [sample.rs](sample.rs). It is an API
sketch with four target programs, not a compiled library. The first two
concrete implementations are [the MLP](../src/mlp/src/baseline.rs) and
[the CNN](../src/cnn/src/train.rs). Their jobs are to establish working math,
gradients, and execution behavior before those mechanics move behind the API.

The central boundary is **users manipulate named axes; the backend maps those
axes to dimensions, strides, and tiles**. This is a semantic tensor algebra.
The training convenience follows from that algebra and automatic derivatives.
The enforcement ladder from typed structure through empirical assertions is
documented in [static guarantees](static-guarantees.md).

## Implemented first slice

The `axis` package now exposes `axis` from [src/axis/src/lib.rs](../src/axis/src/lib.rs).
[src/mlp/src/train.rs](../src/mlp/src/train.rs) is the first executable consumer: the
original MLP expressed with named axes, automatic derivatives, Linear/ReLU,
explicit mean reduction, and SGD. The original standalone trainers and scalar
oracles remain unchanged for comparison.

[Causal attention](../src/attention/README.md) is the second executable
consumer. Its query/key/value and output projections reuse Linear, Module
parameter traversal, and SGD. Attention composition stays in that program;
only named-axis softmax and causal masking join the tensor algebra.

[The CNN](../src/cnn/README.md) is the third consumer. Conv2d composes valid
patch extraction with Linear; global pooling is the existing named mean. Its
explicit cuTile program remains beside it as a lower-level baseline.

Implemented algebra includes identity/extent binding, strict subset-axis
broadcasting, contraction with shared batch axes, physical layout changes,
rename/role axes, split/merge, explicit outer products, causal masking, and
named-axis softmax. Parameters have
stable IDs and version checks; repeated uses accumulate gradients and SGD
updates each shared parameter once. Backward releases its saved graph; a
second backward through it reports an error. Leaf gradients accumulate until
cleared, and replacing a parameter starts a fresh leaf.

Research assumptions can also fail executable checks. `SinglePass` guards the
finite-corpus consumption budget. `Idr` consumes stable sample IDs and enforces
declared coverage and repeat-rate limits for finite or generated streams.
`DataLoader` applies these guards before releasing batches, while the counters
remain independent of optimizer and tensor execution; see
[data regimes](data-regimes.md).

`Adam` and `AdamW` keep first/second moments and parameter updates on the GPU.
Their constructor defaults use beta1 `0.9`, beta2 `0.999`, and epsilon `1e-8`;
learning rate remains explicit, and AdamW also requires explicit decoupled
weight decay.

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

The first backend is synchronous CUDA f32 using cuTile kernels. CPU code builds
index plans; GPU kernels do forward arithmetic, derivatives, and SGD. These
generic gather/reduction plans prioritize verifiable semantics. They are
limited to 16,777,216 contributions per operation and are not a competitive
GEMM implementation. Tensors participating in an operation must share the
same `Device` handle. There is no CPU fallback, higher-order differentiation,
or retained-graph mode. Softmax currently repeats scalar row reductions per
output, with the same contribution bound; it is not a fused attention kernel.
Padding/stride variants, LSTM, and a full Transformer remain future
slices; their presence in the owner's sketch does not imply implementation.

Run from the research node:

```bash
bash cutile-mlp/scripts/check.sh
bash cutile-mlp/src/mlp/scripts/train.sh --smoke
bash cutile-mlp/src/mlp/scripts/train.sh
bash cutile-mlp/src/addition/scripts/train.sh --smoke
```

The verification script runs formatting, Clippy, the CPU shape test, and the
explicitly enabled GPU tests in the library, MLP, attention, and CNN consumers.
Algebra checks are in [src/axis/src/tests.rs](../src/axis/src/tests.rs),
and the [MLP checks](../src/mlp/src/tests.rs) compare
all MLP predictions, parameter gradients, input gradients, and one update to
the scalar f64 oracle. Separate checks cover odd extents, short batches, extra
time axes, noncontiguous storage, both contraction derivatives, split/merge,
broadcast gradients, tied modules, stale parameter versions, and invalid axes.
The [measured MLP run](../src/mlp/README.md#library-mlp) establishes the first
slice. [Attention checks](../src/attention/src/tests.rs) additionally compare
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

## Operation contracts to settle first

| Operation | Contract |
|---|---|
| `Linear(input, hidden.of(32))` | Contract the named input axis, introduce the output axis, preserve every unrelated axis. Bind the input extent when building the model. |
| `x.squared_error(y)` | Align identical axis sets by identity, require equal extents, preserve the unreduced shape. |
| `x.mean(axes)` | Remove precisely those axes; backward broadcasts and divides by their extent product. Scalar `.backward()` requires all loss axes to have been reduced. |
| Elementwise add/multiply | Align shared identities with equal extents. Permit scalar or subset-axis broadcasting, such as a `[hidden]` bias on `[batch, hidden]`. |
| Incomparable axis sets | Require explicit expansion. `[batch, time] + [batch, hidden]` must not silently create `[batch, time, hidden]`. |
| `contract(rhs, axes)` | Sum over the specified shared axes. Align remaining shared axes and preserve distinct axes in a deterministic logical order. |
| `split` / `merge` | Validate extent products and axis uniqueness; preserve the mapping needed to undo the operation during backward. |
| `causal_mask(query, key)` | Require distinct axes with equal extents; replace key positions greater than query positions with negative infinity and give them zero derivative. Square, zero-offset self-attention only. |
| `softmax(axis)` | Normalize along one named axis without changing logical shape. Subtract each row's maximum. Rows need at least one finite value; other values may be finite or negative infinity. |
| `unfold2d(channels, spatial, patch, kernel)` | Extract valid stride-one patches, preserve unrelated axes, and replace channels with one flattened patch axis. Backward sums overlapping contributions into the input. |
| `Conv2d(input, output, spatial, kernel)` | Compose `unfold2d` with Linear; shrink the named spatial extents, replace the input-channel axis, and preserve unrelated axes. |
| `binary_cross_entropy_with_logits(target)` | Return stable unreduced elementwise losses for identical axis sets. Targets are constants; the caller names every reduction axis. |
| `categorical_cross_entropy_with_logits(target, class)` | Accept constant one-hot/probability targets over the same axes, stably reduce the named class axis, and preserve all other axes. Backward is `softmax(logits) - target`. |
| `Conv(input, output, spatial)` | Transform channels and spatial extents by the stated stride/padding/dilation rules. Preserve all unrelated axes. |
| `cross_entropy(target, class)` | Consume the named class axis; target axes must exactly match the remaining axes. Return their unreduced losses; integer targets have no gradient. |

The sample's second convolution, `feature=32 -> feature=64`, exposes an
important distinction. A parameter tensor cannot contain the same Axis twice.
The layer therefore needs distinct internal input/output role axes, maps the
input role to the incoming `feature`, and restores the public `feature` on the
result. The same requirement appears in square linear projections and recurrent
weights. Parameter shapes cannot simply inherit both public handles unchanged.

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

The first backend can execute the existing cuTile kernels synchronously. A
small layout plan maps the semantic operation to its kernel specialization.
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

The current CNN has one trainable convolution. Its input-gradient oracle proves
overlapping patch accumulation (`col2im`) for that operation, including a
noncontiguous input layout. An actual two-convolution composition remains the
next witness before the sample's stacked CNN is considered established.

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
4. **Executable CNN slice (first stage implemented):** valid stride-one
   convolution, ReLU, named global mean, Linear, and stable binary loss now
   reproduce the concrete CNN, including overlapping input gradients. A
   stacked-convolution witness remains before same-padding, stride, adaptive
   pooling, collapse, and categorical cross-entropy.
5. **LSTM and Transformer:** add each as a concrete next program. Recurrence
   drives state/sequence lifetime decisions; attention drives role axes,
   masked softmax, split/merge, and shared-parameter behavior. `Repeat` must
   create independent blocks unless weight tying is explicit.

The first two library trainers must work with reordered physical axes and
with unrelated axes added. For example, the same Linear must accept
`[batch, time, input]` and a layout storing `[input, batch, time]`, preserve
batch/time, and produce matching gradients. Include odd extents such as 17
features and a final short batch so that tile-friendly examples do not become
an accidental public restriction.

Success is the existing training behavior expressed through short programs
with these contracts enforced. The four golden programs remain acceptance
targets, and each capability gains its status from a running witness.
