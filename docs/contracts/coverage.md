# Test coverage

Axis reports two coverage scopes because a CPU-only number cannot exercise the
cuTile backend honestly.

```bash
# CPU-safe unit tests; CUDA tests remain ignored.
bash scripts/coverage.sh host

# The same unit tests plus serialized tests on a real NVIDIA device.
bash scripts/coverage.sh cuda
```

Both commands measure only the reusable `axis` library target. Cargo build
output, registry dependencies, generated CUDA bindings, and the separate
consumer programs are outside that target. The CUDA command includes ignored
tests explicitly and forces one test thread because they share the device.

On 2026-09-20, the host suite passed 27 tests and reported **33.30% line
coverage**. The low whole-crate percentage is expected: the backend, tensors,
modules, and trainer allocate or execute on CUDA. Within code the host can
exercise, line coverage was 99.44% for axes and shapes, 98.89% for data, 100%
for preprocessing, 97.73% for data-regime contracts, and 95.90% for empirical
monotonicity.

The combined RTX 5060 run passed all 41 tests and reported **85.29% line
coverage** and **90.93% function coverage**. Tensor line coverage is 91.56%,
including hand-checked forward and reverse-mode witnesses for binary
cross-entropy, causal softmax, overlapping Conv2d patches, and multiple-axis
contraction. Neural-module line coverage is 88.56%, including shape validation
that runs before parameter allocation.

The largest reported gap is inside functions annotated as cuTile device
kernels. They execute on the GPU, but LLVM's host coverage profile does not
record their device-side source lines; the CUDA tests instead observe their
outputs and gradients. The remaining host-observable gaps are mostly bounded
error branches and less common module construction paths.

The percentages are measurements, not a target to game. A new test should
protect behavior or a failure mode that matters to a consumer. Reaching 100%
in this report requires device-side source coverage support in addition to
more CUDA witnesses and error-path cases; excluding backend files would only
hide that limitation.
