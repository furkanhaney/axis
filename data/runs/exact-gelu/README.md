# Erf-form GELU witness (2026-09-20)

Issue: https://github.com/furkanhaney/axis/issues/46

Command (from this checkout, CUDA 13.2 toolkit supplied in the environment):

```sh
cargo test -p axis --release exact_gelu_matches_independent_quadrature_and_gradient \
  -- --ignored --exact tests::exact_gelu_matches_independent_quadrature_and_gradient --nocapture
```

Rust 1.96.1, cuTile 0.3.1, FP32 default Device. Test run on the desk RTX 5060;
the same release test executable was copied to `atlas-ci` and run on RTX 5070 Ti,
driver 595.84, with the tower CUDA toolkit. Both returned exit 0 and:

```text
exact GELU PASS n=8193 forward=3.920e-7 derivative=2.615e-7 tanh_gap=4.732e-4
test tests::exact_gelu_matches_independent_quadrature_and_gradient ... ok
```

The independent reference integrates the standard-normal density in f64 with
2,048 Simpson panels. Samples span [-8,8], signed zero and ±1000, in a 3×2731
logical tensor stored in the reverse physical order. Output, analytic derivative
through reverse mode, shape preservation and oracle discrimination against tanh
are checked. Limits were 2e-6 forward and 5e-7 derivative, not fitted afterward.

The tower had another GPU workload (68% utilization, 1,940 MiB allocated) when
the test started. This is correctness evidence only; neither elapsed test time
nor this record is an inference-performance benchmark. Model-level MobileSAM
parity, nonfinite inputs and bitwise cross-library identity are not established.
