# MLP and CNN: overlap observed in working code

Both baseline trainers are separate crates in the same workspace. Their forward/backward
implementations remain explicit so that a library can emerge from the
operations these programs actually need. The explicit CNN deliberately carries
forward working primitives from the MLP; this is a reuse opportunity census,
not a claim that two independent authors arrived at the same design. Both
baselines now remain as oracles beside library-based consumers.

Run from the research root:

```bash
python3 cutile-mlp/scripts/compare_trainers.py
```

The initial census finds **8 identical function definitions, containing 93
nonblank lines in each trainer**. Matching ignores whitespace and includes
signatures, braces, and comments; attributes and blank lines are excluded.
The census counts whole matching definitions. It excludes shared fragments
inside differing functions and the separate CPU-reference files.

| Identical definition | What both programs use it for |
|---|---|
| `kernels::matmul` | Dense forward/backward in the MLP; convolution over patches and filter gradients in the CNN. |
| `kernels::bias_activation` | Bias addition and ReLU on activations. |
| `kernels::relu_backward` | Gate incoming gradients by saved activations. |
| `kernels::sgd` | Update a parameter tensor from its gradient. |
| `host::upload` | Convert host initialization/data into a shaped GPU tensor. |
| `host::zeros` | Allocate and initialize reusable work buffers. |
| `host::matmul` | Bind transpose specializations and launch matrix multiplication. |
| `host::compare` | Compare GPU arrays against an independent f64 oracle. |

## Same pattern with a concrete difference

| Area | MLP | CNN | Boundary suggested by the evidence |
|---|---|---|---|
| Bias gradients | Fixed 16-column reduction | Same reduction parameterized for 16 or 1 columns | Axis reduction, with shape-aware lowering. |
| Linear layer | GEMM, 16 outputs | Dot reduction, 1 logit | One mathematical operation with multiple backend implementations. |
| Loss derivative | MSE, normalized by samples × outputs | Stable sigmoid/BCE, normalized by samples | Loss remains unreduced until the caller names its reduction axes. |
| Parameters | Two weights and two biases | Filter bank, conv bias, head weight, head bias | Stable parameter identity plus gradient/update traversal. |
| Work buffers | Hidden state and dense gradients | Patches, spatial activations, pooled state, and gradients | Operation-specific saved values, released by backward. |
| Loop | Zero/init, verify, forward, backward, step, evaluate | Same stages with a classification metric | Training policy can remain explicit in the script. |
| Validation | MSE | BCE and accuracy | Metrics belong to the task; the evaluator can share execution plumbing. |
| Oracle | Scalar dense products | Direct spatial convolution | Keep independent implementations when extracting GPU code. |

## CNN-specific work

`im2col`, mean pooling and its backward operation, binary loss, and the scalar
output head are new here. The convolution gradient reduces contributions
from **every spatial position of every image** into the same filter weights.
Global mean pooling's backward divides by the 64 positions, then broadcasts.
Those two reductions are different from averaging the batch loss.

The CNN caches patches because its input images stay fixed. A changing batch,
data augmentation, or a second convolution would require a different lifetime.
The explicit implementation also has fixed geometry and tile widths; those
constraints must be backend checks or handled edge tiles in a public library.

## What this supports extracting next

The unchanged kernel primitives and allocation/launch helpers are the first
concrete implementation candidates. The owner's sample adds the public
semantic contract they need to serve: named axes, explicit reductions,
composable modules, and automatic differentiation. The implementation order
and acceptance criteria are in [library.md](library.md).

This census describes the retained explicit baselines. The `axis` member
now supplies named axes, autodiff, layout handling, Linear, and valid
stride-one Conv2d; see the [implementation scope](library.md). The library CNN
matches its direct scalar oracle through overlapping input gradients. Stacked
convolution and comparative performance remain unestablished.
