# ImageNet64 with Axis

This example is the large-data vision rung after MNIST and CIFAR-100: a
three-block convolutional classifier, global average pooling, 1,000 logits,
categorical loss, exact top-1 counts, BF16 matrix products, and AdamW. The
program samples without replacement for the requested bounded run, enforces
the finite-corpus `SinglePass` budget, and reports work in samples and optimizer
steps.

The default command is a two-image, one-step mechanics witness and needs no
dataset:

```sh
bash src/examples/getting-started/imagenet64/scripts/train.sh --smoke
```

It proves that the complete Axis model builds, runs forward and backward, and
updates on the GPU. Its synthetic colors are not an ImageNet result.

## Prepare ImageNet64

Axis does not download or redistribute ImageNet. Obtain access from the
[ImageNet project](https://www.image-net.org/download.php), accept its current
terms, and download the downsampled 64x64 NPZ archives. The dataset authors'
format has ten training shards named `train_data_batch_1.npz` through
`train_data_batch_10.npz` and one `val_data.npz`. Each archive contains a
rank-2 uint8 `data` array in planar RGB order and one-based `labels`.

After extracting the outer downloads, prepare seekable files without Python:

```sh
bash src/examples/getting-started/imagenet64/scripts/prepare.sh \
  src/examples/getting-started/imagenet64/data/train.bin \
  /path/to/train_data_batch_{1..10}.npz

bash src/examples/getting-started/imagenet64/scripts/prepare.sh \
  src/examples/getting-started/imagenet64/data/validation.bin \
  /path/to/val_data.npz
```

Preparation validates NPY versions, dtypes, shapes, C-order layout, labels,
member ambiguity, and final file length. It converts labels to zero-based
`0..1000` and interleaves each label with its 12,288 image bytes so training
can seek to one sample without loading a multi-gigabyte shard. Partial output
is removed if validation fails. The prepared files remain local and ignored.
Named vision axes and byte-to-tensor packing come from the same internal
`vision-data` support crate used by the smaller getting-started programs; the
seekable ImageNet64 record format stays local to this example.

Train a bounded slice:

```sh
bash src/examples/getting-started/imagenet64/scripts/train.sh \
  --data src/examples/getting-started/imagenet64/data/train.bin \
  --steps 100 --batch 8
```

`--steps × --batch` may not exceed the file's sample count. The deterministic
affine permutation visits unique positions for that run; changing `--seed`
changes their order. Pixels are mapped from bytes to `[-1, 1]`. This starter
does not yet include random crops, flips, a learning-rate schedule, distributed
training, or a competitive ImageNet recipe.

The 64x64 variant preserves ILSVRC's 1,281,167 training images, 50,000
validation images, and 1,000 classes while downsampling pixels. ImageNet's
current access terms restrict use to non-commercial research and education;
those terms govern the images independently of Axis's MIT-licensed code. Cite
both ImageNet and *A Downsampled Variant of ImageNet as an Alternative to the
CIFAR Datasets* when publishing results.

## What is verified

Unit tests construct small, deterministic NPZ archives in memory and verify
preparation, random-access reads, one-based label conversion, invalid-label
rejection, and atomic failure. No ImageNet image is committed.

A [compact real-pixel receipt](data/evidence/real-validation-smoke.txt) records
the source revision and hashes for a two-image validation witness. It exercises
the Rust preparer and one GPU update while making no learning or accuracy claim.

The network is intentionally compact:

```text
RGB 64x64
  -> Conv2d(3 -> 16, 3x3, stride 2) + ReLU
  -> Conv2d(16 -> 32, 3x3, stride 2) + ReLU
  -> Conv2d(32 -> 64, 3x3, stride 2) + ReLU
  -> mean(height, width)
  -> Linear(64 -> 1000)
```

Axis's current convolution path materializes patch tensors. This example is an
API and data-pipeline starting point; it is not evidence of throughput parity
with a fused PyTorch/cuDNN ImageNet implementation.
