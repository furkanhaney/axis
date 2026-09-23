# Device-side spatial windows (2026-09-20)

Issue: https://github.com/furkanhaney/axis/issues/52
Base: `948f9ca` (Axis 0.9.0). New API: `Tensor::pad_zeros(axis, before, after)`
and `Tensor::narrow(axis, start, length)`. These are not in published 0.9.0.

Commands from this checkout, CUDA toolkit 13.2 supplied in the environment:

```sh
bash scripts/cargo.sh test -p axis --release window_tests -- \
  --ignored --test-threads=1 --nocapture
bash scripts/check.sh
```

Rust 1.96.1, cuTile 0.3.1, FP32 `Device::cuda`. The three focused CUDA tests
passed on the desk RTX 5060 and the same executable passed on `atlas-ci` RTX
5070 Ti (driver 595.84), exit 0 on both. The full `scripts/check.sh` subsequently
passed, including workspace host/CUDA tests, Clippy, formatting, packaging,
docs.rs boundary and repository contracts. This is correctness evidence; no
throughput measurement or full-model parity is claimed.

The independent value/gradient oracle covers 30 cases across all three named
axes and two physical layouts. Values must match FP32 bits exactly; derivative
error must be below 2e-7. Additional checks preserve signed zero, infinities and
a quiet-NaN payload, prevent masked-away nonfinite derivatives from leaking,
reject invalid axes/intervals/overflow and cover the three actual TinyViT spatial
shapes. A 16,777,217-element source proves the copy path bypasses the generic
plan-contribution limit. Both directions use only five metadata integers per
logical dimension; model-specific attention behavior remains in Atlas.

```text
TinyViT window PASS 128x128x128, bottom/right padding=5
TinyViT window PASS 64x64x160, bottom/right padding=6
TinyViT window PASS 64x64x320, bottom/right padding=6
window compact geometry PASS beyond generic plan ceiling
window values and gradients PASS: 30 cases, 3 selected axes, 2 layouts
test result: ok. 3 passed; 0 failed
```

Focused executable SHA-256:
`c36c8c60d33abc6d3b996d1bcaf76a26f7e64d92d05812a7c1f2a52a0d2e750c`.
Tower execution copy: `/home/furkan/build/axis-spatial-window/window-tests`.
The desk was shared with graphical work. The tower's preflight sample was idle
(2 MiB, 0% utilization); this does not establish continuous isolation.
