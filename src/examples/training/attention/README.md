# Causal attention

A second consumer of `axis`, after the MLP. It learns a synthetic sequence's
prefix means with query/key/value projections, causal multi-head attention,
and an output projection. This is an attention block, not a full Transformer.

The [composition](src/attention.rs) stays here until another program needs it.
Its mathematical core is:

```rust
let probabilities = q.contract(&k, head_feature)?
    .scale(1.0 / width.sqrt())?
    .causal_mask(query_time, key_time)?
    .softmax(key_time)?;

probabilities.contract(&v, key_time)?
    .merge([head, head_feature], feature)?
    .rename(query_time, time)
```

Q/K/V come from splitting the feature axis into head and head-feature axes.
Query and key time are distinct `time.role(...)` identities. The new reusable
operations are only `Tensor::causal_mask(query, key)` and `softmax(axis)`,
including their derivatives. Projections, parameter traversal, gradient
clearing, loss reduction, and SGD are the same library machinery as the MLP.
Named parameter slots include `query.weight`, `value.bias`, and `output.weight`.

From the Axis repo root:

```bash
bash src/examples/training/attention/scripts/train.sh --smoke
bash src/examples/training/attention/scripts/train.sh
bash scripts/cargo.sh test --release --locked -p attention -- --ignored --test-threads=1 --nocapture
```

[The training script](src/train.rs) uses 16 sequences of length 5, two heads,
and three features per head. Training and held-out inputs come from distinct
fixed seeds. At 200 SGD steps, training MSE falls from `4.30e-1` to `1.81e-2`,
and held-out MSE from `5.12e-1` to `2.39e-2`. The 10-step smoke and 100-step
intermediate run also pass. [Recorded runs](data/runs/training.log).

[Verification](src/tests.rs) uses an independent [scalar f64 forward](src/reference.rs)
and central differences for every Q/K/V element. It covers odd sequence/head
extents, an extra batch axis, multiple physical layouts, a leading logical
feature axis, nonuniform upstream gradients, future-token isolation, and
zero gradients through masked positions. Separate checks cover large positive
and negative logits, softmax on a nonfinal axis, and invalid axis contracts.

The backend remains synchronous and unoptimized. Softmax subtracts the row
maximum; each row must have at least one finite value and may otherwise
contain finite values or negative infinity. All-masked rows, NaN and positive
infinity are outside its contract. The mask supports square, zero-offset
self-attention only. Cached decoding, padding masks, dropout, layer norm,
positional encoding, and fused attention are not implemented.
