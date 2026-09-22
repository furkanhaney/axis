# Axis

[![Latest version](https://img.shields.io/crates/v/axis.svg)](https://crates.io/crates/axis)
[![Documentation](https://docs.rs/axis/badge.svg)](https://docs.rs/axis)

Axis is an experimental Rust machine-learning library built on NVIDIA
[cuTile Rust](https://github.com/NVlabs/cutile-rs). Tensor dimensions have
identities, not positions: a `batch` axis cannot silently become a `class`
axis because both happen to have the same extent.

The crate currently provides:

- named-axis tensor algebra and reverse-mode differentiation, including
  deterministic finite minimum reductions, asymmetric zero-padding, contiguous
  slicing, compact device copies for layout, merge, and alignment, and
  integer-factor nearest and half-pixel bilinear resampling along a named
  axis;
- `Linear`, `Conv2d`, and `Conv3d` with stride, symmetric padding, and
  grouped/depthwise channels, named-axis `LayerNorm`, `RmsNorm`, `GroupNorm`,
  and stateless `InstanceNorm`, one-hot `Embedding` and `PositionEmbedding`,
  explicit-state `LstmCell`/`Lstm`, activations, attention primitives with
  causal and prefix-causal masks, and sequential composition;
- device-resident SGD, Adam, AdamW, and explicitly oriented rank-2 Muon with an
  exact AdamW remainder;
- generated and finite data loaders with executable single-pass, finite-pass,
  and IDR assertions;
- versioned semantic identities with exact retained-population and
  bounded-memory streaming disjointness;
- exact, batch-composable categorical accuracy counts;
- train-fitted standardization with an explicit variance correction; and
- empirical monotonicity checks over explicitly ordered input pairs, with
  receipts that distinguish sampled evidence from a global guarantee; and
- empirical learning-progress checks that bind a metric and evaluation
  population to ordered budget observations without claiming convergence.

```rust,no_run
use axis::prelude::*;

fn main() -> Result<()> {
    let device = Device::cuda(0)?;
    let batch = Axis::new("batch");
    let input = Axis::new("input");
    let hidden = Axis::new("hidden");
    let class = Axis::new("class");

    let mut model = Sequential::new((
        Linear::new(input, hidden.of(32)),
        GELU,
        Linear::new(hidden, class.of(10)),
    ));
    model.build(&Shape::new([batch.of(64), input.of(49)])?, &device, 42)?;
    Ok(())
}
```

Axis requires Linux, Rust 1.89 or newer, an NVIDIA GPU supported by cuTile,
`libclang`, and CUDA 13.2 or newer. It is early research software: APIs may
change as real training programs expose better defaults and abstractions.

`Tensor::gelu_exact()` and `ExactGELU` provide erf-form GELU for pretrained-model
parity; `Tensor::gelu()` and `GELU` retain their tanh formulation. `SiLU` and
configurable `LeakyReLU` compose existing differentiable tensor primitives.
The exact-form operation uses FP32 normal-CDF evaluation and its analytic
derivative, not bitwise libm equivalence. `gelu` is the tanh form; a model
ported from PyTorch's default `nn.GELU` wants `gelu_exact`.

The Muon implementation's pinned upstream revision and MIT attribution are in
[THIRD_PARTY.md](THIRD_PARTY.md), which is included in every published crate.

`Conv2d::new` and `Conv3d::new` default to stride one, no padding, and one
group. Configure a depthwise layer by setting `groups` to the input channel
count; both input and output channel extents must be divisible by that value.
Kernel, stride, and padding entries correspond positionally to the supplied
named spatial axes. Built-in padding is symmetric per axis. Dilation and built-in
asymmetric convolution padding are not implemented. Both paths materialize FP32 patches before
contraction; 3D kernel volumes can make that intermediate large, while later
generic layout and broadcast operations retain their index-plan limits.

`tensor.pad_zeros(axis, before, after)` and `tensor.narrow(axis, start, length)`
give asymmetric zero-padding and contiguous nonempty slicing. Both preserve
named axes, keep values and gradients on-device, and use rank-sized metadata
instead of element-sized index tables; padding, slicing, layout changes, merge,
and alignment run as compact device copies. Identity operations share storage.

`Embedding` looks a one-hot `vocabulary` axis up in a learned
`[vocabulary, feature]` table, so a token is a coordinate rather than an integer
index; `PositionEmbedding` adds a learned `[position, feature]` table.
`Tensor::prefix_causal_mask` masks attention so a prefix attends freely and the
remainder attends causally.

`Lstm` names its time, input, and hidden axes and preserves every unrelated
stream axis. `run` starts from device-resident zero state; `run_from` accepts an
explicit `LstmState` and returns both the complete sequence and terminal hidden
and cell state. Returned states stay connected to reverse mode until the caller
uses `LstmState::detach`. The current implementation projects the full input
sequence once, then submits one eager recurrent transition per time coordinate
and retains its activations. It is a numerical-correctness path, not a fused scan
or a sequence-throughput claim.

```bash
cargo add axis@0.10.0
```

The [repository](https://github.com/furkanhaney/axis) contains complete MLP,
CNN, attention, generated-data, paired Muon, MNIST, and research-script
migrations with independent numerical oracles. Outside consumers pin the
published crate: a generated-data Sudoku transformer, a game-disjoint chess
policy-and-value transformer, an in-process cell-segmentation click in the Atlas
labeler, and a set of research studies each carrying a small Axis port checked
against a PyTorch oracle. What those consumers still lack is tracked as issues
and in `docs/direction/next.md`. Contributions from humans and agents are both
welcome under the repository's contribution contract. The project name and
branding are covered by its
[trademark policy](https://github.com/furkanhaney/axis/blob/main/TRADEMARK.md).
