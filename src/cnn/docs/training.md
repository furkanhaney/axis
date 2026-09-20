# CNN training prototype

[src/train.rs](../src/train.rs) is a second concrete trainer beside the
MLP. It uses the same pinned cuTile 0.3.1 crate, lockfile, and local CUDA setup.
It compiles as `cutile-cnn` in this node's own crate.

From the Axis repo root:

```bash
bash src/cnn/scripts/train.sh --smoke
bash src/cnn/scripts/train.sh
bash src/cnn/scripts/train.sh --steps 1000 --lr 0.5 --seed 7
bash src/cnn/scripts/train.sh --verify-only
```

## Model and task

`1 × 10 × 10 image -> 3 × 3 valid conv, 16 channels -> ReLU -> global mean -> linear -> 1 logit`

There are 177 trainable scalar parameters. The filter bank is stored as a
16 × 16 matrix: nine filter coefficients per output channel plus seven
zero-padded rows whose gradients remain zero. Stride is one. The image has
one input channel, and the spatial output is 8 × 8.

The dataset contains 128 training images and 128 separately generated
validation images. Each has one horizontal or vertical stripe with randomized
position, brightness, background, and pixel noise. Both sets are balanced.
The CPU generates these small synthetic images; no external data is needed.

The GPU extracts the sliding patches, applies shared convolution weights,
adds biases, applies ReLU, averages spatial positions, and computes a linear
logit. Backpropagation and all SGD updates run on the GPU. Sigmoid/BCE
gradients use a stable exponential formulation. Reported BCE and accuracy are
computed from copied logits at logging points.

The default is 500 full-batch SGD steps, learning rate 1.00, model seed 42.
Patch tensors are computed once because these images stay fixed. Changing
batches or augmenting images would require rebuilding those patches.
There is one trainable convolution: the code computes filter/bias gradients
but does not compute gradients with respect to the raw input image. Stacking
convolutional layers will require that additional backward operation.

## Checks

Every run verifies a separate 16-image batch against
[a scalar f64 reference](../src/reference.rs) that performs direct
convolution over image coordinates, without the GPU's im2col construction:

- Convolution/ReLU activations, pooled features, and logits.
- Every element of the four parameter gradients, including padded filter rows.
- Every parameter after one SGD step.
- Central finite differences for 13 distinct parameter locations.

GPU comparisons use `2e-4 + 2e-3 * abs(reference)`; finite differences use
`2e-5 + 2e-3 * abs(gradient)`. The finite-difference perturbation is `1e-6`,
with the actual representable f32 perturbation in the denominator, to stay
within this fixture's local ReLU region. Non-finite values fail. Training
must reduce both training and held-out BCE; all reported metrics evaluate
the updated model, including after the final step.

## Measured result

2026-09-20, `furkan-linux`, RTX 5060, driver 595.84, CUDA 13.2.51, Rust 1.96.1.
The 10-step smoke, 100-step, and default 500-step runs passed.

| Metric | Initial | After 500 steps |
|---|---:|---:|
| Training BCE | 7.01e-1 | 4.50e-3 |
| Validation BCE | 7.04e-1 | 4.87e-3 |
| Training accuracy | 50.00% | 100.00% |
| Validation accuracy | 50.00% | 100.00% |

Training BCE decreased 99.36%; validation BCE decreased 99.31%. Maximum
absolute parameter-gradient error against the CPU reference was 2.27e-8;
finite-difference error was at most 8.08e-11. Setup/JIT/verification took
4.01 seconds; the small 500-step loop with evaluation took 0.22 seconds in
this run. [Captured output](../data/cnn-verification.log).

These results establish learning on this simple synthetic classification
task. They do not measure natural-image accuracy or comparative throughput.
Geometry and tile sizes remain explicit constraints of this prototype.
The [overlap census](../../../docs/overlap.md) and [library proposal](../../../docs/library.md) describe
what these two programs suggest extracting next.
