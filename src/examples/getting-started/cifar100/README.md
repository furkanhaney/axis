# CIFAR-100 with Axis

This example moves from grayscale silhouettes to 100-class color images. It
keeps the model deliberately small enough to read: a strided named-axis
convolution, ReLU, global spatial mean, and a categorical linear head.

Download the CIFAR-100 binary archive from the [dataset
site](https://www.cs.toronto.edu/~kriz/cifar.html), extract it, and point Axis
at the directory containing `train.bin` and `test.bin`:

```sh
bash src/examples/getting-started/cifar100/scripts/train.sh --data /path/to/cifar-100-binary --smoke
bash src/examples/getting-started/cifar100/scripts/train.sh --data /path/to/cifar-100-binary
```

The parser uses the fine label (100 categories), preserves CIFAR's planar RGB
layout, and fits one normalization transform per channel on training pixels
only. `--smoke` deterministically trains on the first 2,048 images and audits
the first 512 held-out images for five complete passes. It is a bounded
end-to-end check, not a competitive CIFAR architecture or accuracy claim.

The downloaded archive and extracted data stay local and are ignored by Git.
Strict IDX/CIFAR parsing, train-fitted preprocessing, tensor packing, exact
finite-pass receipts, and held-out evaluation are shared with the other
getting-started programs through `../vision-data`.

On the local RTX 5060, the bounded smoke completed in 5.93 seconds. Training
loss fell from 4.6292 in the first pass to 4.4359 in the fifth; held-out
accuracy moved from 2.15% to 2.54%, peaking at 3.32%. The exact receipts are in
[data/runs/smoke.log](data/runs/smoke.log). This small global-mean model is a
mechanics witness and teaching example, not a competitive CIFAR-100 result.
