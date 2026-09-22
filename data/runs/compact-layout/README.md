# Compact layout/split/merge — 2026-09-21

Issue: https://github.com/furkanhaney/axis/issues/56.
The consumer is the Atlas RBC five-decoder migration, not a changed model or
precision policy. This record separates profiling from clean elapsed-time runs.

## Motivation and baseline profile

Frozen synthetic fixture payload:
`18834bcd85d3df1f2c1be1f37bcbcca72d6973129642d0a1b05e08398d193e17`.
Atlas source `72bd1a93533bc698288d7eaf4be50e374171e3ca`, binary
`5ea85c8bf64c20c6a02f246893fe9571696826f64b9172d3b265009de33af192`.
Framework baseline: published 0.9.0 plus spatial-window
`869dc66323c51acd4eca244e3297445226e2896d`. Two images, one positive click each,
shared TinyViT encoder/prompt and five production mask decoders. Only final mean
masks/quality are read back. Stream fences bound queued-buffer retention at
encoder boundaries and between decoders. This patch does not change those fences.

Tower: RTX 5070 Ti 16 GiB, driver 595.84, CUDA 13.2, cuTile 0.3.1, Rust 1.96.1,
host release build, FP32. Preflight 2 MiB/0% GPU. No power/clock normalization or
continuous isolation claim. No environment changes. Nsight Systems 2026.3.2;
CPU sampling unavailable (`perf_event_paranoid=4`, `perf_event_open` check failed).
Profiling used `--trace=cuda,osrt --sample=none --cpuctxsw=none` plus
`AXIS_PROFILE=1`, without changing that policy. Both frozen mean-output gates
still passed. CPU operation spans are nested and must not all be added together.

| Profile observation | Value |
| --- | --- |
| `merge` calling-thread spans | 293 calls, 249.817897 s total, max 16.823849 s |
| `align` calling-thread spans | 1,314 calls, 28.949022 s total |
| GPU kernels | 12,779, total 6.095389261 s |
| `grouped_entry` kernel time | 5.871294978 s, 96.3% of kernel time |
| GPU kernel + copy union | 8.462096599 s |
| First-to-last traced kernel/copy span | 340.820546803 s |
| Kernel + copy busy fraction of that span | 2.48286% |
| Largest gap between consecutive kernels | 16.763855774 s |
| Host-to-device traffic | 8,082 copies, 30,790.908 MB |
| Device-to-host traffic | 4 copies, 0.524 MB |

The busy fraction is a timestamp-union measure for this application, not a
`nvidia-smi` sample, SM throughput or utilization of the machine in general.
All traced kernels used stream 13. This diagnosis justifies removing host index
construction before tuning one hot kernel. No `ncu` replay was used as timing.

Retained tower artifacts are under
`/home/furkan/build/atlas-axis-ensemble/profile-001/`:
`run.log`, `timeline.nsys-rep`, `timeline.sqlite`, and the four `stats_cuda_*.csv`
reports. No model weights or private images are redistributed here.

## Framework correctness

From the Axis checkout, with its CUDA toolkit supplied:

```sh
bash scripts/cargo.sh test -p axis --release compact_layout \
  -- --ignored --test-threads=1 --nocapture
```

The focused tests cover 36 layout-to-layout pairs and 36 ordered adjacent or
nonadjacent merge cases against independent coordinate indexing and gradients;
noncontiguous split; signed zero, infinity and NaN payload copying; invalid
axes/products; and identity buffer sharing including singleton dimensions.
The large test actually allocates 16,789,506 values and exercises layout, split,
merge and selected reverse-mode gradients above the generic contribution ceiling.
Permutation metadata is at most nine integers for its three-axis source.

Initial scoped Clippy exposed dead generic helpers after the migration. They
were removed rather than suppressed. The complete `bash scripts/check.sh` finished
with exit 0: structure, release readiness, docs.rs/package, formatting, Clippy,
host tests and workspace GPU witnesses. All three hosted PR #57 checks also pass.
The four focused tests pass on both RTX 5060 and RTX 5070 Ti. The tower executable
SHA-256 is `b005620c9997f23c499b602e9b3ec34ba2325255ca08c47caf6512d3af5b4171`;
its `focused.log` is `84b1975ecb384e9e30cec7b942b7c07a8397ff923363a1df82f7a60450473649`.

## Matched unprofiled consumer replay

Atlas combines implementation `2084ef0` with spatial-window `869dc66` as
`2e763ac813078804a18ab062714f06ce4c8a9b90`, branch `codex/atlas-layout-preview`.
No weights, precision, gates, model arithmetic or synchronization policy changed.
The only Atlas source change is its provenance string. The binaries run sequentially
on the same tower with the same fixture and `--mean-only`, without profiling.
There was no listed compute process before candidate execution. Clocks/thermals
are not normalized; these two observations are not a latency distribution.

| Image | Baseline seconds | Candidate seconds | Ratio |
| --- | --- | --- | --- |
| First | 266.514451365 | 50.084398984 | 5.32x |
| Second | 79.567415640 | 6.367503304 | 12.50x |

Both means pass unchanged limits, with identical reported errors/sign agreement.
The timer includes forward, first-use uploads and final checks, but not model
construction or native image preparation/geometry. This is not cached-click latency
or end-to-end product acceptance. Generic broadcast/reduction plans remain.

Retained tower `ab-layout-001` SHA-256 identities:

```text
candidate binary ca0341d260edcc8ae1328cbd1f0b0bc54625a37df4c17555ebc548f7599f5f32
reference.log    dcea66f8317009bdb57ae522b7c449b73f81d3ed5b8a44516c1d11d1c33d7b8b
candidate.log    1aa812340eff11a76ad5cf6682097c8d3a75070cdb7eadf9c68bddfac6c19909
```

Candidate exited 0. Baseline's completed log records both passing outputs/timings;
its old tool-session handle is unavailable for fresh exit-code observation.
Member and encoder-boundary replay follows separately from this timing lane.
