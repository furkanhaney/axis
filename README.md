---
title: cuTile training workspace and axis
status: measured
---

# cuTile training workspace

An experimental Rust training library emerging from concrete programs. The
filesystem separates the reusable `axis` crate from its MLP, attention,
CNN, and generated-addition consumers. The CNN also retains its standalone kernel baseline. Named
axes express the math; cuTile handles GPU execution. Each program owns its
code, scalar reference, scripts, and run data.

```text
Cargo.toml                workspace, shared versions
Cargo.lock                one resolved dependency graph
src/
  axis/                    reusable library crate
    Cargo.toml
    README.md
    src/                  axis algebra, tensors, autograd, modules, CUDA backend
  mlp/                    MLP experiment: library consumer + explicit baseline
    Cargo.toml
    README.md
    src/                  train.rs, baseline.rs, reference.rs, tests.rs
    scripts/              train.sh, baseline.sh
    data/                 measured runs
  cnn/                    named-axis CNN consumer + explicit kernel baseline
    Cargo.toml
    README.md
    src/                  train.rs, reference.rs
    scripts/              train.sh
    docs/                 training contract and results
    data/                 measured runs
  attention/              causal attention consumer and prefix-mean training
    Cargo.toml
    README.md
    src/                  train.rs, attention.rs, reference.rs, tests.rs
    scripts/              train.sh
    data/                 measured runs
  addition/               endless generated samples + IDR-guarded training
    Cargo.toml
    README.md
    src/                  train.rs
    scripts/              train.sh
    data/                 measured IDR training run
  mnist/                  migrated finite-pass MNIST baseline
  mnist-population/       migrated learning-rate population probe
  panel/                  migrated country-year regression mechanics
scripts/                  shared CUDA setup, Cargo runner, workspace checks
docs/                    cross-program contracts, API sketch, overlap census
data/                    verification spanning multiple members
```

The crate physically lives at [src/axis](src/axis/README.md), with its
own [Cargo.toml](src/axis/Cargo.toml). Consumers declare a workspace path
dependency; the library cannot import experiment code.

- [MLP programs and measured learning](src/mlp/README.md)
- [CNN consumer and explicit baseline](src/cnn/README.md)
- [Causal attention and gradient verification](src/attention/README.md)
- [Generated addition under IDR](src/addition/README.md)
- [Migrated finite-pass MNIST baseline](src/mnist/README.md)
- [Migrated population probe](src/mnist-population/README.md)
- [Migrated country panel MLP](src/panel/README.md)
- [Library contracts and implementation scope](docs/library.md)
- [Continuation note: decisions, evidence, and next useful work](docs/next.md)
- [Executable single-pass and infinite-data-regime assumptions](docs/data-regimes.md)
- [Which ML assumptions belong in types, build checks, or runtime](docs/static-guarantees.md)
- [Run receipts, claim prerequisites, and corpus-derived defaults](docs/run-certificates.md)
- [Measured Python research defaults census](data/default-census.md)
- [Human-written motivation](docs/motivation.md)
- [Owner's four target programs](docs/sample.rs)
- [Shared-code census](docs/overlap.md)

## Run

From the Axis repo root:

```bash
# Optional, when no suitable system toolkit is installed:
python3 scripts/setup_cuda.py
bash scripts/check.sh
bash src/mlp/scripts/train.sh --smoke
bash src/mlp/scripts/train.sh
bash src/mlp/scripts/baseline.sh --smoke
bash src/cnn/scripts/train.sh --smoke
bash src/attention/scripts/train.sh --smoke
bash src/addition/scripts/train.sh --smoke
bash src/mnist/scripts/train.sh --smoke
bash src/mnist-population/scripts/train.sh --smoke
bash src/panel/scripts/train.sh --smoke --split country
python3 scripts/compare_trainers.py
```

Requires stable Rust 1.89+, Linux, libclang, and a cuTile-supported NVIDIA GPU.
The shared helper installs CUDA 13.2 into the ignored `.cuda/` folder; explicit
`CUDA_TOOLKIT_PATH` or `CUDA_HOME` wins. Builds use one ignored `target/` at
this workspace. No compatibility scripts remain at the former program paths.
The only external direct dependency is pinned `cutile = "=0.3.1"`.

Verified on `furkan-linux`, RTX 5060, driver 595.84, CUDA 13.2.51, Rust 1.96.1.
The first library MLP reproduces the baseline's 500-step held-out MSE of
`5.74e-2`; [run details](src/mlp/README.md#library-mlp). Causal attention and
the CNN also learn, with independent forward/gradient witnesses. The backend
remains experimental and unoptimized.

The generated addition run consumed 128,000 unique samples with zero observed
reuse or train/evaluation identity overlap; held-out MSE fell from `8.06e-1`
to `3.88e-3`.

Upstream: [cuTile Rust](https://github.com/NVlabs/cutile-rs).
