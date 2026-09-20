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

On 2026-09-20, the host suite passed 26 tests and reported **31.28% line
coverage**. The low whole-crate percentage is expected: the backend, tensors,
modules, and trainer allocate or execute on CUDA. Within code the host can
exercise, line coverage was 99.44% for axes and shapes, 98.89% for data, 100%
for preprocessing, 97.73% for data-regime contracts, and 95.90% for empirical
monotonicity.

The combined RTX 5060 run passed all 36 tests and reported **78.67% line
coverage** and **84.89% function coverage**. It covered 100% of functions in
axes/shapes, data, metrics, preprocessing, monotonicity, data-regime contracts,
and the trainer. The largest remaining gaps are backend lowering branches for
operation shapes the current acceptance suite does not yet construct, followed
by tensor and neural-module error paths.

The percentages are measurements, not a target to game. A new test should
protect behavior or a failure mode that matters to a consumer. Reaching 100%
requires additional CUDA witnesses and backend-shape cases; excluding those
files from the report would only hide the work.
