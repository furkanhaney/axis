# cuda-oxide: measured escape hatch, not a second backend

Axis uses cuTile because its tile-level model fits the framework's named tensor
operations and stable Rust toolchain. NVIDIA's cuda-oxide exposes lower-level
SIMT CUDA programming through nightly Rust and LLVM. That makes it useful for a
small fused kernel whose launch boundaries dominate, but a poor replacement for
Axis's whole backend today.

The first probe is deliberately an optimizer primitive: an out-of-place fused
AdamW update over 4,194,304 FP32 parameters. On an RTX 5060, five independent
processes each ran ten warmups and 100 CUDA-event-timed iterations. The fused
cuda-oxide kernel had a 0.340940 ms median; the equivalent three-kernel path had
a 0.473156 ms median, a 1.388x speedup. CPU and split-kernel references matched,
a non-divisible 4,099-element tail passed, and Compute Sanitizer reported zero
errors.

The probe lives in the sibling `cuda-oxide-axis-probe` repository at commit
`787e51301a908c63adab0e2dc7d359b37b851053`. It needs nightly-2026-08-28,
CUDA 13+, LLVM 21+, and an R580+ driver.

The narrow integration path is to compile a pinned cuda-oxide sidecar to PTX,
load it through Axis's existing CUDA driver layer, and launch it on the same
stream over the same buffers. Fused AdamW is the first candidate. Adoption
requires an end-to-end training benchmark: a faster isolated kernel does not
establish a faster step. Keep cuTile as the primary backend unless repeated
measurements show that a small set of sidecar kernels closes a material gap.
