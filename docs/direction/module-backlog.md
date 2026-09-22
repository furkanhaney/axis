# The module catalog is a discovery backlog

Updated 2026-09-20. PyTorch's
[`torch.nn` catalog](https://docs.pytorch.org/docs/2.14/nn.html) is a useful
inventory of module families that working researchers expect. Axis uses it as
a backlog, not as an API specification or a promise of positional-shape
compatibility.

An Axis module is complete when it has:

1. a named-axis contract with invalid configurations rejected before launch;
2. parameters, persistent state, and training/evaluation behavior made explicit;
3. an independent host or trusted oracle for forward values and every gradient;
4. CUDA coverage for reordered storage, asymmetric geometry, and edge cases;
5. a real example or outside consumer that makes the abstraction necessary; and
6. documented defaults, evidence, and remaining performance limits.

The requested module can itself be the consumer when the operation is a common
primitive and its reference test exercises a representative composition. More
specialized abstractions should wait for a second use rather than enter as
unexercised surface area.

The catalog is also a number: [docs/nn/catalog.md](../nn/catalog.md) freezes
every `torch.nn` class as a row with a verdict, and `scripts/checks/nn_gap.sh
--check` holds the floor (1351 bp at the 2026-09-22 freeze). This file says in
what order rows should flip; that table says which have.

## Current surface

Axis already has `Linear` (bias on by default; `.bias(false)` before `build`
omits the bias parameter entirely, so `named_parameters` then has only
`weight`), `PopulationLinear` (bias always present; not yet given the same
option), `Conv2d`, `Conv3d`, `MaxPool2d`, `MaxPool3d`,
`Tensor::adaptive_avg_pool3d`, named-axis
`LayerNorm`, `RmsNorm`, `GroupNorm`, stateless `InstanceNorm`, one-hot
`Embedding`, `PositionEmbedding`, `ReLU`, tanh-form
`GELU` and erf-form `ExactGELU` (PyTorch's default `nn.GELU` is the exact form), `SiLU`, `LeakyReLU`, `Tanh`, `SignStraightThrough`, and
`Sequential`, with tensor-level losses (including `abs`/`absolute_error`, PyTorch's
unreduced `F.l1_loss`, zero gradient at exactly zero matching its `abs` backward
convention), sine, elementwise `exp`/`ln`, a PyTorch-exact
`Tensor::softplus(beta, threshold)`, a numerically stable named-axis
`Tensor::logsumexp(axis)` (`m + ln(sum(exp(x - m)))` with the shift `m = max(axis)`
detached, so the gradient is exactly `softmax`; an all-non-finite group inherits `max`'s
`NaN` rather than PyTorch's `-infinity`), differentiable central
differences, softmax, causal masking, `Tensor::masked_softmax(axis, mask)`
(a variable-length validity mask rather than causal masking's fixed
triangular shape; masked positions get exactly zero probability and
gradient, and a group whose mask is entirely zero returns all zeros instead
of the `NaN` a literal `masked_fill(-inf)` composition would produce),
named-axis nearest and bilinear
resampling (`Tensor::upsample_nearest`, `Tensor::resample_bilinear`), attention
composition, named reductions, an arbitrary-index `gather` (a host-side integer
index, not a one-hot contraction, so it scales to a large table where
`Embedding` cannot), `Tensor::argmin` (a host-side, non-differentiable index of
`min`'s own winning coordinate), `Tensor::scatter_add` (`gather`'s exact
transpose: sums values into host-index-named buckets, with an exact gradient)
and its constant-ones case `Tensor::bincount`, data regimes, metrics,
trainers, and optimizers.
`gt`/`ge`/`lt`/`le`/`eq` compare a tensor against a scalar into a `{0.0, 1.0}`
mask with no gradient of its own; `logical_and`/`logical_not` compose masks the
same way PyTorch's `&`/`~` do on 0/1 tensors (`mul`, `1 - x`).
`clamp(min, max)` bounds a tensor elementwise into `[min, max]` with either
side an optional `f32` (`None` leaves it unbounded), propagating `NaN` inputs
unclamped and passing the gradient through at either bound as well as
strictly inside it, matching PyTorch's `clamp`, not a "zero at the boundary
too" convention.
`SGD`, `Adam`, and `AdamW` take a per-step learning rate through
`set_learning_rate`, driven by the pure `cosine_annealing_lr` and
`one_cycle_lr` schedule functions (`optim.rs`), and a global-norm
`clip_grad_norm` matching `torch.nn.utils.clip_grad_norm_` (`optim.rs`);
there is still no mixed precision. `Tensor::uniform` and `Tensor::normal` are the
public, seeded random tensor constructors: generated host-side from the same
shared xorshift stream that initializes parameters, then uploaded, with no
gradient edge, since a random draw is a constant, not a parameter.
`Module` now also has an explicit training/evaluation mode: `forward` stays
evaluation semantics, and a provided `forward_training(input, pass: &mut
TrainingPass)` defaults to calling `forward`, so every existing module is
unchanged, while `Sequential` threads one `TrainingPass` through its children
in order. `TrainingPass` and `Trainer::with_seed`/`step_training` make the
per-step random-draw context and its seed explicit and recorded
(`TrainStep::pass_seed`); `Dropout` is its first consumer, identity in
`forward` and inverted dropout drawn from `pass.next_seed()` in
`forward_training`.
This is enough to train the existing MLP, CNN, attention, MNIST, Sudoku, chess,
and panel acceptances, but it is not yet a comfortable general module library.
`Tensor::broadcast_to(shape)` explicitly broadcasts a tensor onto a target
shape carrying every one of its axes (at its own extent) plus any axes it
lacks entirely; it is the outer-broadcast primitive elementwise
add/sub/mul/div deliberately refuse (they only ever align one operand's axis
set onto the other's when it is already a subset), so two operands with
genuinely disjoint axis sets -- `vision/image-encode`'s pixel-indexed and
site-indexed tensors in its `torch.cdist`-style pairwise distance -- compose
an outer op from `broadcast_to` and an ordinary elementwise op instead of a
dedicated outer-product method.

## Ordered backlog

| Stage | Families | Why this order |
| --- | --- | --- |
| Active | physics-informed residual consumers and independent numerical oracles | Central differences and empirical residual receipts now provide the first honest path; ODE and pendulum studies must establish defaults and expose missing composition. |
| Recurrent foundation | `LstmCell` and `Lstm` correctness are implemented; fused recurrence, direction, and layer composition remain | Independent forward and complete gradient oracles protect the eager IFGO implementation before performance work. |
| Stateful foundation | persistent non-parameter state, then named-axis `BatchNorm` | Running statistics cannot be represented honestly until `Module` can hold and update them (#76). |
| Common composition | `Conv1d` or rank-general convolution, `ELU`, dropout, prefix (nested) dropout, common losses | These unlock many ordinary ports once mode and random-state semantics exist. Exact GELU, `SiLU`, and `LeakyReLU` landed from Atlas and vision consumer pressure; `Embedding`, the prefix causal mask, and `SignStraightThrough` landed from the byte autoencoder migration, whose ordered binary code is also the consumer for prefix dropout. `MaxPool2d`/`MaxPool3d` and `Tensor::adaptive_avg_pool3d` landed from `fluid`'s U-Net encoder, `morpheus`'s peak-NMS decode, and `gastric`'s interface-region pooler. |
| Architecture families | recurrent variants, transpose convolution, reusable transformer encoder/decoder modules | Add them around measured consumers after the lower-level contracts settle. |
| Specialized | sparse, quantized, distributed, fractional pooling, lazy initialization | Each needs its own representation or execution contract; names alone would provide false parity. |

Dimensions are expressed by named axes rather than suffixing every concept with
`1d`, `2d`, and `3d`. Axis may still expose a dimension-specific kernel where
geometry or performance requires it, as convolution does, but normalization
should describe its feature and observation axes directly. Defaults should
match common research practice when the meaning is stable, and remain explicit
when a hidden default could change an experiment.

## Reading the upstream inventory

The catalog is reviewed family by family: containers, convolution, pooling,
padding, activations, normalization, recurrence, transformers, linear and
sparse layers, dropout, losses, vision helpers, shuffling, parallelism,
quantization, and lazy modules. Missing names become candidates. Actual Axis
priority comes from consumer frequency, shared primitives, scientific risk,
and the cost of producing a complete oracle.

This avoids two failure modes: overlooking boring but essential layers, and
claiming parity because a type name compiles. The backlog is satisfied by
working contracts and evidence, not by matching the length of another
framework's index.
