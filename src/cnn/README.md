# CNN experiment

This node now contains two implementations of the same valid-convolution
classifier. `axis-cnn` consumes `axis`; `cutile-cnn` retains the original
explicit kernels as an implementation baseline. Both use the independent
direct scalar convolution in `src/reference.rs`.

The library program is short because convolution is composed from named patch
extraction and the existing Linear contraction. Reverse-mode patch extraction
scatters and sums overlapping contributions, providing the `col2im` behavior
needed to stack convolutions. Stable binary cross-entropy remains elementwise;
the program explicitly averages batch and output axes.

From the research root:

```bash
bash cutile-mlp/src/cnn/scripts/train.sh --smoke
bash cutile-mlp/src/cnn/scripts/train.sh
bash cutile-mlp/src/cnn/scripts/baseline.sh --verify-only
```

The library oracle test compares 4,096 activation values, pooled features,
logits, all parameter gradients, overlapping input gradients, and one SGD
update. It also checks stable BCE at logits `-1000` and `1000`. Physical image
storage is reordered independently of the logical axes.

At 100 steps on 16 training and 16 held-out images, held-out BCE falls from
`7.04e-1` to `7.40e-2` and accuracy reaches `100.00%`.
[Measured library run](data/library-training.log). See the original
[baseline training contract and results](docs/training.md).
