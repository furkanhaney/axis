# Axis

[![Latest version](https://img.shields.io/crates/v/axis.svg)](https://crates.io/crates/axis)
[![Documentation](https://docs.rs/axis/badge.svg)](https://docs.rs/axis)

Axis is an experimental Rust machine-learning library built on NVIDIA
[cuTile Rust](https://github.com/NVlabs/cutile-rs). Tensor dimensions have
identities, not positions: a `batch` axis cannot silently become a `class`
axis because both happen to have the same extent.

The crate currently provides:

- named-axis tensor algebra and reverse-mode differentiation, including
  deterministic finite minimum reductions;
- `Linear`, `Conv2d` with stride, symmetric padding, and grouped/depthwise
  channels, `LayerNorm`, activations, attention primitives, and sequential
  composition;
- device-resident SGD, Adam, and AdamW;
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

`Conv2d::new` defaults to stride one, no padding, and one group. Configure a
depthwise layer by setting `groups` to the input channel count; both input and
output channel extents must be divisible by that value. Padding is symmetric
per named spatial axis. Dilation and asymmetric padding are not implemented.

```bash
cargo add axis@0.4.0
```

The [repository](https://github.com/furkanhaney/axis) contains complete MLP,
CNN, attention, generated-data, MNIST, and research-script migrations with
independent numerical oracles. Contributions from humans and agents are both
welcome under the repository's contribution contract. The project name and
branding are covered by its
[trademark policy](https://github.com/furkanhaney/axis/blob/main/TRADEMARK.md).
