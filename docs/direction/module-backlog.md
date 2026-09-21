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

## Current surface

Axis already has `Linear`, `PopulationLinear`, `Conv2d`, named-axis
`LayerNorm`, `RmsNorm`, `GroupNorm`, stateless `InstanceNorm`, `ReLU`, `GELU`,
and `Sequential`, with tensor-level losses, softmax, causal masking, attention
composition, named reductions, data regimes, metrics, trainers, and optimizers.
This is enough to train the existing MLP, CNN, attention, MNIST, Sudoku, chess,
and panel acceptances, but it is not yet a comfortable general module library.

## Ordered backlog

| Stage | Families | Why this order |
| --- | --- | --- |
| Active | `Conv3d` | It generalizes established convolution geometry and current research consumers already use it. |
| Recurrent foundation | tensor `sigmoid`/`tanh`, sequence selection and stacking, `LstmCell`, then `Lstm` | The cell supplies a small complete gradient oracle; the sequence module then owns recurrence, state shape, direction, and layer composition. |
| Stateful foundation | explicit training/evaluation mode and persistent non-parameter state, then named-axis `BatchNorm` | Running statistics cannot be represented honestly by the current stateless `Module` contract. |
| Common composition | `Conv1d` or rank-general convolution, pooling, `SiLU`, `LeakyReLU`, `ELU`, `Embedding`, dropout, common losses | These unlock many ordinary ports once mode and random-state semantics exist. |
| Architecture families | recurrent variants, transpose convolution, reusable transformer encoder/decoder modules | Add them around measured consumers after the lower-level contracts settle. |
| Specialized | sparse, quantized, distributed, adaptive and fractional pooling, lazy initialization | Each needs its own representation or execution contract; names alone would provide false parity. |

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
