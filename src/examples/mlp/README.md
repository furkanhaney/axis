# MLP experiment

Two implementations of the same regression task: `axis-mlp` uses the sibling
`axis` library; `cutile-mlp` is the original explicit cuTile baseline.
Both share this node's scalar reference and deterministic data. The library
has no dependency on this experiment; its full MLP oracle test lives here.

From the Axis repo root:

```bash
bash src/examples/mlp/scripts/train.sh --smoke
bash src/examples/mlp/scripts/train.sh
bash src/examples/mlp/scripts/baseline.sh --verify-only
bash src/examples/mlp/scripts/baseline.sh
```

## Library MLP

The training loop in [src/train.rs](src/train.rs) is now:

```rust
let mut model = Sequential::new((
    Linear::new(input, hidden.of(32)),
    ReLU,
    Linear::new(hidden, output.of(16)),
));
model.build(x.shape(), &device, 42)?;
let mut optimizer = SGD::new(0.5)?;

for _ in 0..500 {
    model.zero_grad();
    let loss = model.forward(&x)?.squared_error(&y)?.mean([batch, output])?;
    loss.backward()?;
    optimizer.step(&mut model)?;
}
```

The complete script also loads the baseline's exact initial weights and data,
accepts `--steps`/`--smoke`, and reports held-out loss. Users name the axes to
contract or reduce; the backend handles storage and backward propagation.

Verified on the RTX 5060 with the toolchain below: one CPU shape test and five
GPU tests pass, including every parameter and input gradient against independent
f64 arithmetic. The largest MLP parameter-gradient error was `1.24e-8`, and the
largest input-gradient error was `2.32e-9`. The same Linear passes with 17 input
features, a three-sample batch, an extra time axis, a leading logical feature
axis, and reordered physical storage. Tied parameters and stale graphs are
checked explicitly. [Captured checks](../../../data/evidence/library-verification.log).

The 10-, 100-, and 500-step library runs passed. After 500 steps, training MSE
was `4.10e-2` (from `7.51e-1`), and held-out MSE was `5.74e-2` (from
`6.93e-1`). This agrees with the retained standalone trainer at its reported
precision. The library loop took 10.05 seconds in this single run,
including its first backward/update JIT and final evaluation. Intermediate
`step` lines report the loss used for that update; `final` evaluates the updated
weights. [Full run](data/runs/library-training.log), [100 steps](data/runs/library-100.log).
The standalone MLP's 500-step run and the CNN's 10-step smoke run also pass
after adding the library. [Regression output](../../../data/evidence/library-baseline-regression.log).

This is an experimental CUDA f32 backend using explicit index plans, with a
16,777,216-contribution limit per plan. It is substantially slower than the
baseline's specialized tiled matrix multiplies. CNN layers and optimizers
beyond SGD remain the next concrete additions; see the [scope and contracts](../../../docs/design/library.md).

## Standalone MLP baseline

- Architecture: 16 inputs, 32 ReLU hidden units, 16 linear outputs, both biases.
- Data: 256 training samples and 256 independent validation samples, sampled
  uniformly from `[-1, 1]`. A fixed randomly initialized teacher MLP generates
  noise-free targets. No downloads or external datasets are needed.
- Loss: mean squared error over **both samples and outputs**.
- Optimizer: full-batch SGD, default learning rate 0.50, 500 updates.
- GPU: tiled matrix multiplies (including transposed loads for backprop), bias
  addition, ReLU, loss derivative, parameter gradients, and parameter updates.
- CPU: reproducible initialization/data, scalar correctness oracle, and MSE
  reporting from predictions copied back only at logging points.

Backpropagation is explicit; this is not an autodiff framework. All gradients
are computed before any parameter is updated. GPU arrays are reused throughout
training. Each kernel launch synchronizes for clarity; this is a small training
demonstration, not a throughput benchmark or optimized trainer. Storage is f32;
cuTile's `mma` controls the tensor-core multiplication arithmetic.

The shape choices are intentional: matrix dimensions and batch size are
multiples of the 16-element tile width. Change the constants in `src/baseline.rs`
together if experimenting with other sizes. Partial tiles are not implemented.

## MLP correctness

Every run first checks a separate 32-sample batch:

1. Central finite differences check 16 weights/biases against scalar f64
   analytical derivatives.
2. GPU predictions and **every element** of all four parameter gradients are
   compared against the scalar reference.
3. Every parameter after one GPU SGD step is compared against the CPU update.

GPU comparison tolerance is `2e-4 + 2e-3 * abs(reference)`; finite-difference
tolerance is `2e-5 + 2e-3 * abs(gradient)`. Non-finite values fail. Training exits
unsuccessfully unless both training and held-out MSE decrease. Logged losses
evaluate the model **after** each reported update, including the last one.
Setup/JIT/verification time is reported separately from the measured training
loop, which includes synchronization and periodic evaluation.

## MLP measured run

Verified on 2026-09-20 on `furkan-linux`, RTX 5060, NVIDIA driver 595.84,
CUDA toolkit 13.2.51, Rust 1.96.1, cuTile 0.3.1. The 10-step smoke run,
100-step run, and default 500-step run all passed. Default seed: 42.

| MSE | Before training | After 500 steps | Reduction |
|---|---:|---:|---:|
| Training | 7.51e-1 | 4.10e-2 | 94.54% |
| Held-out validation | 6.93e-1 | 5.74e-2 | 91.71% |

The largest absolute GPU gradient error against the CPU oracle was 7.63e-9;
the finite-difference error was at most 3.59e-11. Setup, verification, and JIT
took 1.79 seconds; the small 500-step loop including evaluation took 0.10
seconds in this run. These are single-run wall times, not a performance
comparison. [Captured default-run output](data/evidence/verification.log).
