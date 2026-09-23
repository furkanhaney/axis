# DCGAN witness

Trains `axis::architectures::{DcganGenerator, DcganDiscriminator}` end to end on real 64x64
data: Fashion-MNIST resized (nearest-neighbor upscale, then zero-padded to the background) from
28x28 to 64x64 and replicated to three channels, the CelebA-free choice for a bounded desk-GPU
run. Both networks train through the exact alternating recipe documented on
`axis::architectures::gan`: a discriminator step on a real batch and a detached fake batch, then
a generator step backpropagated through the (otherwise untouched) discriminator, each through its
own `Adam(lr=2e-4, beta1=0.5, beta2=0.999)` and its own `Trainer`. Both networks start from the
paper's own `weights_init` (`dcgan_tutorial_init`).

```bash
scripts/train.sh --smoke --data /path/to/FashionMNIST/raw
scripts/train.sh --data /path/to/FashionMNIST/raw
```

`--data` points at a directory containing the four uncompressed Fashion-MNIST IDX files
(`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, and their `t10k-` counterparts are not
needed here), downloaded separately and left local; Git ignores `data/raw`. `--smoke` runs 40
steps at batch 16 for a fast sanity check; the default is 400 steps at batch 32 with `ngf=ndf=32`
(half the paper's width, chosen to fit comfortably alongside other work on an 8 GB desk GPU while
still exercising every layer). `--steps N` and `--batch N` override either.

**This is a mechanics witness, not an image-quality or convergence claim.** It establishes that
both networks actually train (parameters and BatchNorm running statistics for both the generator
and the discriminator change every step, `data/runs/dcgan-witness/loss.csv` records both losses
moving), not that the generator produces convincing images in this few-minutes budget -- DCGAN's
own paper trains for many epochs over much larger, curated datasets.

On the local RTX 5060, a 250-step run (batch 32, ngf=ndf=32, the default configuration)
completed in 273.5 seconds (about 4.6 minutes). `data/runs/dcgan-witness/loss.csv` has the
per-step discriminator and generator loss: `d_loss` collapses from `1.50` to near zero within
the first ~20 steps and stays there (the discriminator wins early and mostly stays ahead, as
DCGAN commonly does without the paper's full epoch count and dataset scale), while `g_loss`
swings widely (`0.14` to `6.3`) rather than converging smoothly -- exactly the dynamic this
witness is honest about: **the mechanics run correctly; the training balance itself is not
tuned or claimed to converge in this budget.** `data/runs/dcgan-witness/before.ppm`/`.png` and
`after.ppm`/`.png` are an 8x8 grid of the same 64 fixed latent vectors before training and after
the run: `before` is uniform mid-gray (the tutorial init's near-constant output at step 0);
`after` shows a dark background with a centered, blob-shaped brighter region in every tile --
the generator has visibly picked up Fashion-MNIST's black background and centered-garment
silhouette, even though per-pixel texture stays noisy and no tile resembles a specific garment.
That is the intended reading: motion and structure, not image quality. (PPM is P6, no image-crate
dependency for one witness artifact; the PNGs are a one-off local re-encode for convenient
viewing, not produced by this program.)
