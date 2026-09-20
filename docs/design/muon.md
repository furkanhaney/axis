---
title: Explicit Muon parameter partitions
status: implemented rank-2 first slice
---

# Muon is an explicit optimizer partition

Axis implements Muon for rank-2 parameters and pairs it with AdamW for the
remainder of a model. The caller selects each Muon parameter with an explicit
storage orientation; Axis never guesses from tensor rank or positional shape.
This keeps the scientific and architectural choice visible:

```rust,ignore
let hidden_weight = model.parameter("0.weight")?;
let optimizer = MuonWithAuxAdamW::new(
    [MuonMatrix::axis_linear(&hidden_weight)],
    0.05,  // Muon learning rate
    0.002, // auxiliary AdamW learning rate
    0.01,  // decoupled weight decay in both arms
)?;
let mut trainer = Trainer::new(optimizer);
```

At every step, Axis enumerates the model's unique `ParamId`s, checks that every
selected ID belongs to that model, checks that selected tensors have rank two,
and partitions the exact remainder into AdamW. Tied structural paths are
deduplicated by storage identity. Repeating one ID with the same orientation is
harmless; selecting it with conflicting orientations is an error. Both arms
prepare their complete update before either commits parameters, state, or step
counters. A missing gradient is an error, including on the auxiliary arm.

Pure `Muon` also requires explicit `MuonMatrix` selections and requires those
selections to cover every unique parameter of the model passed to `step`.
Models with a bias, head, embedding, gain, or any other remainder should use
`MuonWithAuxAdamW`.

## Update definition

Muon first applies exponential-moving-average momentum and the source
implementation's Nesterov interpolation

```text
B_t = momentum * B_(t-1) + (1 - momentum) * G_t
U_t = (1 - momentum) * G_t + momentum * B_t
                                      when Nesterov is enabled
U_t = B_t                         otherwise
```

and normalizes `U_t` by its Frobenius norm plus `1e-7`. It then performs five
Newton-Schulz iterations with coefficients `a=3.4445`, `b=-4.7750`, and
`c=2.0315`:

```text
A = X X^T
B = -4.7750 A + 2.0315 A^2
X = 3.4445 X + B X
```

The smaller matrix dimension is the row dimension during the iteration. The
result is restored to the parameter's original named axes and multiplied by
`sqrt(max(1, fan_out / fan_in))`, matching the original rectangular-matrix
scaling. `MuonMatrix::axis_linear` and `fan_in_fan_out` interpret storage as
`[fan_in, fan_out]`; `fan_out_fan_in` interprets it as `[fan_out, fan_in]`.
Decoupled weight decay is applied as
`parameter *= 1 - learning_rate * weight_decay` before subtracting the Muon
direction.

Axis `Linear` weights are stored as `[fan_in, fan_out]`, while PyTorch stores
the corresponding matrix as `[fan_out, fan_in]`. Axis therefore computes the
source rule as `sqrt(max(1, fan_out / fan_in))`, using its second extent over
its first. A custom PyTorch-layout matrix must select `fan_out_fan_in`; an
unlabelled rank-2 matrix cannot enter Muon. Deterministic scalar tests cover
both orientations and tall and wide matrices after one and two steps, so a
transpose or scaling reversal fails visibly. They also compare the stored EMA
buffer after each step. Output-only checks are insufficient here: multiplying
both momentum and the Nesterov direction by one global constant is erased by
Frobenius normalization, even though the optimizer state is noncanonical.

Momentum, normalization, Newton-Schulz products, decay, and updates remain on
the CUDA stream. With `Device::cuda_bf16`, matrix-product inputs in the
Newton-Schulz iteration are rounded to BF16 with FP32 accumulation under the
same explicit policy as other Axis contractions; optimizer state and stored
parameters remain FP32. `Device::cuda` keeps those products FP32.

Frobenius normalization uses a dedicated device-resident tiled sum-of-squares
reduction and recursively reduces its partials. It does not construct the
general indexed reduction plan, whose experimental contribution bound is too
small for common hidden matrices above 16,777,216 elements. A regression
executes normalization beyond that boundary without allocating a host index
plan.

The algorithm follows Keller Jordan's reference implementation pinned at
revision `f98f1cacc0263b04290753e32be8d498c1efc806`. The source link,
copyright, and MIT notice ship with the crate in
[`THIRD_PARTY.md`](../../src/library/THIRD_PARTY.md).

## Acceptance and limits

The [paired acceptance](../../src/examples/muon/README.md) initializes the same
MLP twice, trains one arm with AdamW and one with an explicit
`MuonWithAuxAdamW` partition, and requires independent held-out
`LearningProgress` receipts. It establishes that both paths learn. It does not
claim that Muon wins.

This first slice deliberately excludes automatic rank-based selection,
convolution flattening, `PopulationLinear`, distributed update exchange, and
optimizer-state serialization. Each needs a consumer and an explicit identity
or layout contract before it joins the public surface.
