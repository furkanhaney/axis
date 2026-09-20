# axis

The reusable Rust library crate. Its public entry point is [src/lib.rs](src/lib.rs);
its manifest is [Cargo.toml](Cargo.toml). It depends only on cuTile, and is a
member of the enclosing research workspace.

- `axis.rs`: opaque axis identity, tensor-local extent binding, shapes/layouts.
- `tensor.rs`: named-axis algebra and reverse-mode differentiation.
- `nn.rs`: parameters, Linear/ReLU, Sequential, SGD, and device-resident Adam/AdamW.
- `data.rs`: finite/generated sources, exact shuffled passes, identity-aware batches.
- `regime.rs`: single-pass, finite-pass, IDR, and train/evaluation assertions.
- `train.rs`: ordered training steps with explicit scalar loss closures.
- `backend.rs`: synchronous cuTile f32 kernels and index plans.
- `tests.rs`: algebra, layout, gradients, parameter sharing and invalidation.

The MLP and attention consumers and their independent scalar oracles live in
[../mlp](../mlp/README.md) and [../attention](../attention/README.md).
The library has no dependency back to those programs. Contracts and current
limits are in [the workspace design](../../docs/library.md). The scientific
meaning and limits of the data guards are in [data regimes](../../docs/data-regimes.md).

From the workspace node:

```bash
bash scripts/cargo.sh test --release --locked -p axis
bash scripts/cargo.sh test --release --locked -p axis -- --ignored --test-threads=1
```
