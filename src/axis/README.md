# Axis

[![Latest version](https://img.shields.io/crates/v/axis.svg)](https://crates.io/crates/axis)

Axis is an experimental Rust machine-learning library built on NVIDIA
[cuTile Rust](https://github.com/NVlabs/cutile-rs). Tensor dimensions have
identities, not positions: a `batch` axis cannot silently become a `class`
axis because both happen to have the same extent.

The crate currently provides:

- named-axis tensor algebra and reverse-mode differentiation;
- `Linear`, `Conv2d`, `LayerNorm`, activations, attention primitives, and
  sequential composition;
- device-resident SGD, Adam, and AdamW;
- generated and finite data loaders with executable single-pass, finite-pass,
  IDR, and train/evaluation-disjointness assertions;
- exact, batch-composable categorical accuracy counts; and
- train-fitted standardization with an explicit variance correction.

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

```bash
cargo add axis@0.1.0
```

The [repository](https://github.com/furkanhaney/axis) contains complete MLP,
CNN, attention, generated-data, MNIST, and research-script migrations with
independent numerical oracles. Contributions from humans and agents are both
welcome under the repository's contribution contract.
