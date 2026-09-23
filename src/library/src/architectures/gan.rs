//! DCGAN (Radford, Metz and Chintala, 2015), exactly as the PyTorch "DCGAN Tutorial" builds it
//! for 64x64 images: a fractionally-strided [`ConvTranspose2d`] generator and a strided
//! [`Conv2d`] discriminator, both plain [`Sequential`] compositions of existing library modules.
//! No new kernel was needed; [`Conv2d`]/[`ConvTranspose2d`] already existed, and this file adds
//! only the `bias(bool)` builder they needed to match the tutorial's `bias=False` convolutions
//! exactly (see `model/convolution.rs`, mirroring [`crate::Linear::bias`]).
//!
//! # Alternating training recipe
//!
//! The tutorial's training loop alternates one discriminator step and one generator step per
//! batch, each through its own [`Adam`] optimizer (`lr=2e-4`, `beta1=0.5`, `beta2=0.999`, the
//! paper's own hyperparameters, via `Adam::with_hyperparameters`) and its own [`Trainer`], so
//! `Trainer::step_training`'s `model.zero_grad()` only ever touches that one network's
//! parameters:
//!
//! 1. **Discriminator step.** Run `discriminator.forward_training` on a real batch against the
//!    real label (`1`) and on `generator.forward(noise)?.detach()` (a plain forward, then
//!    [`Tensor::detach`] so no gradient reaches the generator) against the fake label (`0`),
//!    both through [`Tensor::binary_cross_entropy_with_logits`], summed and driven through
//!    `d_trainer.step_training(&mut discriminator, ...)`. `detach` is exactly PyTorch's own
//!    `fake.detach()` in the tutorial's `netD` block.
//! 2. **Generator step.** Run `discriminator.forward(&generator.forward_training(noise, pass)?)`
//!    (this time keeping the generator's graph) against the real label -- the generator's own
//!    non-saturating objective, `-log(D(G(z)))` -- and drive it through
//!    `g_trainer.step_training(&mut generator, ...)`. The discriminator's own parameters still
//!    receive gradients from this backward pass (backward always flows through every tensor on
//!    the path), but only `g_trainer`'s optimizer -- scoped to `generator.parameters()` -- ever
//!    applies an update, so the discriminator is read-only here exactly as the tutorial's
//!    `netD.zero_grad()`-then-`netG` block leaves it.
//!
//! Each `step_training` call builds its own [`TrainingPass`] and commits it only after its own
//! optimizer step succeeds (`Trainer::step_training`'s documented ordering), so BatchNorm's
//! running statistics for the discriminator and the generator are staged and committed
//! independently, once per network per batch -- never mixed, never partially applied. See
//! `dcgan_forward_training_step_updates_both_networks_and_batch_norm_state` in `tests.rs` for
//! one complete alternating step run this way on the GPU, and
//! `src/examples/training/dcgan/README.md` for a bounded end-to-end witness.

use crate::{
    Axis, BatchNorm, Conv2d, ConvTranspose2d, Device, LeakyReLU, Module, Parameter, ReLU, Result,
    Sequential, Shape, State, Tanh, Tensor, TrainingPass,
};

/// PyTorch DCGAN tutorial default latent size (`nz`).
pub const DCGAN_LATENT: usize = 100;
/// PyTorch DCGAN tutorial default generator feature width (`ngf`).
pub const DCGAN_GENERATOR_FEATURES: usize = 64;
/// PyTorch DCGAN tutorial default discriminator feature width (`ndf`).
pub const DCGAN_DISCRIMINATOR_FEATURES: usize = 64;
/// PyTorch DCGAN tutorial default output/input image channel count (`nc`), RGB.
pub const DCGAN_IMAGE_CHANNELS: usize = 3;

const KERNEL: [usize; 2] = [4, 4];
const STRIDE: [usize; 2] = [2, 2];
const PADDING: [usize; 2] = [1, 1];

/// DCGAN's generator: five [`ConvTranspose2d`] layers taking a `[batch, channel(nz), 1, 1]`
/// latent map to a `[batch, height(64), width(64), channel(nc)]` image, exactly PyTorch's
/// `Generator` class in the DCGAN tutorial --
/// `ConvTranspose2d(nz, ngf*8, 4, 1, 0) -> BatchNorm -> ReLU`, three more
/// `ConvTranspose2d(_, _, 4, 2, 1) -> BatchNorm -> ReLU` halvings of `ngf`, then
/// `ConvTranspose2d(ngf, nc, 4, 2, 1) -> Tanh`. Every convolution has `bias=False`, matching the
/// tutorial (its bias would be redundant with the following BatchNorm's own bias, and the final
/// layer's tutorial bias is `False` too). `nz` is inferred from the input's `channel` extent at
/// `build`, exactly like every other named-axis module; only `ngf` and the output image channel
/// count are constructor hyperparameters. `channel` is reused as the same axis identity at every
/// stage (its extent alone changes, `nz -> ngf*8 -> ngf*4 -> ngf*2 -> ngf -> image_channels`),
/// the same reuse [`Conv2d`]'s own depthwise tests already rely on.
pub struct DcganGenerator {
    layers: Sequential,
}

impl DcganGenerator {
    /// `ngf=64`, output channels `3` (PyTorch tutorial defaults).
    pub fn new(channel: Axis, spatial: [Axis; 2]) -> Self {
        Self::with_features(
            channel,
            spatial,
            DCGAN_IMAGE_CHANNELS,
            DCGAN_GENERATOR_FEATURES,
        )
    }

    /// Explicit output image channel count and generator feature width.
    pub fn with_features(
        channel: Axis,
        spatial: [Axis; 2],
        image_channels: usize,
        ngf: usize,
    ) -> Self {
        let transpose = |output_extent: usize, stride: [usize; 2], padding: [usize; 2]| {
            ConvTranspose2d::new(channel, channel.of(output_extent), spatial, KERNEL)
                .stride(stride)
                .padding(padding)
                .bias(false)
        };
        let layers: Vec<Box<dyn Module>> = vec![
            Box::new(transpose(ngf * 8, [1, 1], [0, 0])),
            Box::new(BatchNorm::new(channel)),
            Box::new(ReLU),
            Box::new(transpose(ngf * 4, STRIDE, PADDING)),
            Box::new(BatchNorm::new(channel)),
            Box::new(ReLU),
            Box::new(transpose(ngf * 2, STRIDE, PADDING)),
            Box::new(BatchNorm::new(channel)),
            Box::new(ReLU),
            Box::new(transpose(ngf, STRIDE, PADDING)),
            Box::new(BatchNorm::new(channel)),
            Box::new(ReLU),
            Box::new(transpose(image_channels, STRIDE, PADDING)),
            Box::new(Tanh),
        ];
        Self {
            layers: Sequential::new(layers),
        }
    }
}

impl Module for DcganGenerator {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.layers.output_shape(input)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.layers.build(input, device, seed)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layers.forward(input)
    }
    fn forward_training(&self, input: &Tensor, pass: &mut TrainingPass) -> Result<Tensor> {
        self.layers.forward_training(input, pass)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.layers.named_parameters()
    }
    fn named_states(&self) -> Vec<(String, State)> {
        self.layers.named_states()
    }
}

/// DCGAN's discriminator: five [`Conv2d`] layers taking a `[batch, height(64), width(64),
/// channel(nc)]` image to one logit per sample, exactly PyTorch's `Discriminator` class in the
/// DCGAN tutorial -- `Conv2d(nc, ndf, 4, 2, 1) -> LeakyReLU(0.2)`, three more
/// `Conv2d(_, _, 4, 2, 1) -> BatchNorm -> LeakyReLU(0.2)` doublings of `ndf`, then
/// `Conv2d(ndf*8, 1, 4, 1, 0)`. Every convolution has `bias=False`. **This returns raw logits,
/// never a probability**: the tutorial's final `nn.Sigmoid()` is deliberately omitted in favor
/// of [`Tensor::binary_cross_entropy_with_logits`] (`BCEWithLogitsLoss`), the numerically stable
/// combined form -- call `.sigmoid()` on the result only if an explicit probability is needed
/// outside training. `ndf` is the only constructor hyperparameter; every extent, including the
/// input image channel count, the batch size, and the final singleton spatial axes, is inferred
/// from the input at `build`.
pub struct DcganDiscriminator {
    layers: Sequential,
}

impl DcganDiscriminator {
    /// `ndf=64` (the PyTorch tutorial default). The input image channel count is inferred from
    /// the input at `build`, like every other convolution stage; it is never a constructor
    /// argument, exactly as [`Conv2d::new`] never takes its own input extent.
    pub fn new(channel: Axis, spatial: [Axis; 2]) -> Self {
        Self::with_features(channel, spatial, DCGAN_DISCRIMINATOR_FEATURES)
    }

    /// Explicit discriminator feature width.
    pub fn with_features(channel: Axis, spatial: [Axis; 2], ndf: usize) -> Self {
        let leak = || LeakyReLU::new(0.2).expect("0.2 is finite and non-negative");
        let conv = |output_extent: usize, stride: [usize; 2], padding: [usize; 2]| {
            Conv2d::new(channel, channel.of(output_extent), spatial, KERNEL)
                .stride(stride)
                .padding(padding)
                .bias(false)
        };
        let layers: Vec<Box<dyn Module>> = vec![
            Box::new(conv(ndf, STRIDE, PADDING)),
            Box::new(leak()),
            Box::new(conv(ndf * 2, STRIDE, PADDING)),
            Box::new(BatchNorm::new(channel)),
            Box::new(leak()),
            Box::new(conv(ndf * 4, STRIDE, PADDING)),
            Box::new(BatchNorm::new(channel)),
            Box::new(leak()),
            Box::new(conv(ndf * 8, STRIDE, PADDING)),
            Box::new(BatchNorm::new(channel)),
            Box::new(leak()),
            Box::new(conv(1, [1, 1], [0, 0])),
        ];
        Self {
            layers: Sequential::new(layers),
        }
    }
}

impl Module for DcganDiscriminator {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.layers.output_shape(input)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.layers.build(input, device, seed)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layers.forward(input)
    }
    fn forward_training(&self, input: &Tensor, pass: &mut TrainingPass) -> Result<Tensor> {
        self.layers.forward_training(input, pass)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.layers.named_parameters()
    }
    fn named_states(&self) -> Vec<(String, State)> {
        self.layers.named_states()
    }
}

/// The DCGAN paper/tutorial's `weights_init`: every convolution (`ConvTranspose2d`/[`Conv2d`])
/// weight drawn `N(0, 0.02)`, every `BatchNorm` scale drawn `N(1, 0.02)`, and every bias (any
/// module) set exactly to `0`. An explicit opt-in, applied AFTER `build` to any already-built
/// [`Module`] by its own parameter-path suffix (`*.weight` for a convolution,
/// `*.scale` for BatchNorm's affine scale -- see `crate::BatchNorm`'s own "scale"/"bias" naming
/// -- `*.bias` for either), since Axis's own default convolution initialization is
/// Xavier/Glorot-uniform-shaped, not this tutorial's normal init, and [`BatchNorm`]'s own default
/// scale starts at exactly `1.0` with no spread. Draws come from [`Tensor::normal`]'s
/// deterministic host stream at `seed + parameter_index`, one distinct seed per parameter in
/// `named_parameters` order, so the same seed reproduces identical values on any run or machine.
pub fn dcgan_tutorial_init(model: &impl Module, device: &Device, seed: u64) -> Result<()> {
    for (index, (path, parameter)) in model.named_parameters().into_iter().enumerate() {
        let draw_seed = seed.wrapping_add(index as u64);
        let shape = parameter.tensor().shape().clone();
        let extent = shape.len();
        if path.ends_with(".bias") {
            parameter.set_values(&vec![0.0; extent])?;
        } else if path.ends_with(".scale") {
            let values =
                Tensor::normal(shape.dims().iter().copied(), draw_seed, 1.0, 0.02, device)?
                    .to_vec()?;
            parameter.set_values(&values)?;
        } else if path.ends_with(".weight") {
            let values =
                Tensor::normal(shape.dims().iter().copied(), draw_seed, 0.0, 0.02, device)?
                    .to_vec()?;
            parameter.set_values(&values)?;
        }
    }
    Ok(())
}
