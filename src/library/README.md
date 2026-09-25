# Axis

[![Latest version](https://img.shields.io/crates/v/axis.svg)](https://crates.io/crates/axis)
[![Documentation](https://docs.rs/axis/badge.svg)](https://docs.rs/axis)

Deep-learning programs are unusually good at being wrong while continuing to
run. A loader wraps around, and a fresh-data study silently becomes a repeated
one. A test example slips into training under a new seed or a new file name. A
class axis is averaged away by accident. The loss still falls, the job still
exits zero, and the number that gets reported describes a different experiment
from the one on paper. No error is raised, because nothing was ever checked.

Axis is a Rust framework for training neural networks on
[NVIDIA cuTile](https://github.com/NVlabs/cutile-rs) that checks. The
assumptions a result depends on are executable: when one stops being true, the
run stops before the bad step reaches the optimizer, and when it holds, the run
hands back a receipt that says exactly what was verified. Named tensor axes do
the same for the math, so a `batch` axis can never quietly become a `class`
axis.

## Executable research checks

| Assumption | Check | What it catches |
|---|---|---|
| Every sample is new | `SinglePass`, `FinitePasses` | a step-count edit that turns a fresh-sample run into repeated passes over a finite corpus |
| Repetition stays within a declared rate | `Idr` with `IdrLimits` | a generator or loader that repeats examples more than the experiment allows, caught before the optimizer step |
| Train and evaluation never share an example | `Disjointness` over an `IdentityScheme` | overlap by *meaning*, not by file or seed: the same puzzle after relabeling, the same document through two shards |
| The model actually learned | `LearningProgress` | a loop that runs every forward, backward and step while one declared metric on one declared evaluation population never improves |
| A property holds on ordered inputs | `EmpiricalMonotonicity` | violations of a claimed monotone relation, recorded as sampled evidence rather than a global guarantee |
| A physical law is satisfied | `EmpiricalResidual` | a model whose residual against a stated law exceeds its declared limit |
| The run can be reproduced | `Trainer::step_training` with a `TrainingPass` | nondeterministic randomness: every random layer draws from a seed derived from the run seed and the step, and the step receipt names it |

Each check is narrow on purpose. It returns a receipt (`IdrReceipt`,
`DisjointnessReceipt`, `LearningProgressReceipt`, ...) naming the population,
the identity scheme or metric, and the limits it was held to, and receipts are
the fragments a run certificate is built from. A receipt says what was
checked, never more: a sampled monotonicity check says "sampled", and learning
progress says "improved by this much on this population", not "converged".

```rust,ignore
use axis::prelude::*;

// Train, tuning and audit populations must not share a puzzle, where "the same
// puzzle" is defined by the experiment, not by a file path or a seed.
let scheme = IdentityScheme::new(
    "sudoku-puzzle",
    "1",
    "81 cells after first-occurrence digit relabeling; positions retained",
)?;
let mut split = Disjointness::new(
    scheme,
    [
        PopulationSpec::streaming("training"),
        PopulationSpec::retained("tuning"),
        PopulationSpec::retained("audit"),
    ],
)?;
split.observe("tuning", tuning.iter().map(canonical_puzzle))?;
split.observe("audit", audit.iter().map(canonical_puzzle))?;
split.observe("training", batch.iter().map(canonical_puzzle))?;
let receipt = split.assert_disjoint()?; // fails the run on the first shared puzzle
```

These checks are used by programs outside this repository that pin the
published crate: a generated-data Sudoku transformer that asserts zero
repeated training puzzles (`IdrLimits::generated(0.0)`) and no puzzle shared
with evaluation, and a chess policy-and-value transformer that asserts its
train and evaluation positions come from disjoint games and that training ran
the declared number of passes.

## Named axes

Tensor dimensions have identities, not positions: a `batch` axis cannot
silently become a `class` axis because both happen to have the same extent,
and a reduction names the axis it removes.

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

## Compatibility

Axis implements 72% of PyTorch 2.14's `torch.nn` catalog, graded class by class
against its own definition of a finished module, with every operator checked
against an independent numerical oracle for its values and its gradients
([the catalog](https://github.com/furkanhaney/axis/blob/main/docs/nn/catalog.md)).
`axis::architectures` provides ResNet-34, ResNet-50, VGG16 and DCGAN, each
matching its reference's parameter count exactly and its outputs within `1e-7`
when loaded with the reference's weights. The operator reference is on
[docs.rs](https://docs.rs/axis).

## Requirements

Linux, Rust 1.89 or newer, an NVIDIA GPU supported by cuTile, `libclang`, and
CUDA 13.2 or newer. Axis is early research software: APIs may change as real
training programs expose better defaults and abstractions. The first CUDA
device also enables cuTile's persistent kernel cache at its default location
(`~/.cache/cutile/kernels`), overridable with `AXIS_JIT_CACHE=off` or
`AXIS_JIT_CACHE_DIR`.

```bash
cargo add axis@0.12.0
```

The Muon implementation's pinned upstream revision and MIT attribution are in
[THIRD_PARTY.md](THIRD_PARTY.md), which is included in every published crate.
The [repository](https://github.com/furkanhaney/axis) holds the contracts
behind each check, the acceptance programs, and the contribution contract;
contributions from humans and agents are both welcome. The project name and
branding are covered by its
[trademark policy](https://github.com/furkanhaney/axis/blob/main/TRADEMARK.md).
