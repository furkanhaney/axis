//! ResNet-34 and ResNet-50 (He, Zhang, Ren, Sun, "Deep Residual Learning for
//! Image Recognition", 2015), matching torchvision 0.26's `resnet34` and
//! `resnet50` exactly: the same 7x7 stride-2 stem, four stages of
//! `[3, 4, 6, 3]` blocks, projection (1x1 conv plus BatchNorm) shortcuts
//! wherever a stage changes channel count or spatial stride, and torchvision
//! v1.5's placement of the Bottleneck's stride on its 3x3 conv rather than
//! its first 1x1 conv. Every conv immediately followed by a BatchNorm carries
//! no bias of its own, since BatchNorm's own shift already absorbs one.
//!
//! `resnet18` (`BasicBlock`, stages `[2, 2, 2, 2]`) is not offered; wire one
//! up the same way [`resnet34`] does if a consumer needs it.
//!
//! Axis modules never take a batch axis (every layer here passes through
//! whatever axes it does not itself name), so these constructors take only
//! the channel and two spatial axes; a caller's input tensor carries a batch
//! axis of its own choosing alongside them.
//!
//! `resnet34_small_input`/`resnet50_small_input` offer a stem Axis-only
//! option with no torchvision equivalent: a 3x3 stride-1 conv and no max
//! pool, for 64x64-scale inputs where the standard 7x7 stride-2 conv
//! followed by a stride-2 max pool would collapse the image before the first
//! stage even runs. `resnet34`/`resnet50` match torchvision's stem exactly
//! and are the default choice.

use crate::{
    AdaptiveAvgPool2d, Axis, BatchNorm, Conv2d, Device, Flatten, Linear, MaxPool2d, Module,
    Parameter, ReLU, Result, Sequential, Shape, State, Tensor, TrainingPass,
};

/// The channel and two spatial axes every ResNet layer reads and writes.
#[derive(Clone, Copy)]
pub struct ResNetAxes {
    pub channel: Axis,
    pub height: Axis,
    pub width: Axis,
}

/// Which stem a ResNet built here starts with. See the module doc for the
/// exact contract; [`Stem::Standard`] is what `resnet34`/`resnet50` use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stem {
    Standard,
    SmallInput,
}

const BOTTLENECK_EXPANSION: usize = 4;
const STAGE_DEPTHS: [usize; 4] = [3, 4, 6, 3];
const STAGE_WIDTHS: [usize; 4] = [64, 128, 256, 512];

fn prefixed<T>(prefix: &str, pairs: Vec<(String, T)>) -> Vec<(String, T)> {
    pairs
        .into_iter()
        .map(|(name, value)| (format!("{prefix}.{name}"), value))
        .collect()
}

/// Two 3x3 convolutions with a projection shortcut when the stage changes
/// channel count or stride; ResNet-34's block.
pub(crate) struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl BasicBlock {
    pub(crate) fn new(
        axes: ResNetAxes,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Self {
        let ResNetAxes {
            channel,
            height,
            width,
        } = axes;
        let conv1 = Conv2d::new(channel, channel.of(out_channels), [height, width], [3, 3])
            .stride([stride, stride])
            .padding([1, 1])
            .bias(false);
        let bn1 = BatchNorm::new(channel);
        let conv2 = Conv2d::new(channel, channel.of(out_channels), [height, width], [3, 3])
            .padding([1, 1])
            .bias(false);
        let bn2 = BatchNorm::new(channel);
        let downsample = (stride != 1 || in_channels != out_channels).then(|| {
            (
                Conv2d::new(channel, channel.of(out_channels), [height, width], [1, 1])
                    .stride([stride, stride])
                    .bias(false),
                BatchNorm::new(channel),
            )
        });
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        }
    }
}

impl Module for BasicBlock {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let shape = self.conv1.output_shape(input)?;
        let shape = self.bn1.output_shape(&shape)?;
        let shape = self.conv2.output_shape(&shape)?;
        self.bn2.output_shape(&shape)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let shape = self.conv1.build(input, device, seed)?;
        let shape = self.bn1.build(&shape, device, seed)?;
        let shape = self.conv2.build(&shape, device, seed.wrapping_add(1))?;
        let shape = self.bn2.build(&shape, device, seed)?;
        if let Some((conv, bn)) = &mut self.downsample {
            let downsampled = conv.build(input, device, seed.wrapping_add(2))?;
            bn.build(&downsampled, device, seed)?;
        }
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(input)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        let shortcut = match &self.downsample {
            Some((conv, bn)) => bn.forward(&conv.forward(input)?)?,
            None => input.clone(),
        };
        out.add(&shortcut)?.relu()
    }

    fn forward_training(&self, input: &Tensor, pass: &mut TrainingPass) -> Result<Tensor> {
        let out = self
            .bn1
            .forward_training(&self.conv1.forward_training(input, pass)?, pass)?
            .relu()?;
        let out = self
            .bn2
            .forward_training(&self.conv2.forward_training(&out, pass)?, pass)?;
        let shortcut = match &self.downsample {
            Some((conv, bn)) => bn.forward_training(&conv.forward_training(input, pass)?, pass)?,
            None => input.clone(),
        };
        out.add(&shortcut)?.relu()
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut params = prefixed("conv1", self.conv1.named_parameters());
        params.extend(prefixed("bn1", self.bn1.named_parameters()));
        params.extend(prefixed("conv2", self.conv2.named_parameters()));
        params.extend(prefixed("bn2", self.bn2.named_parameters()));
        if let Some((conv, bn)) = &self.downsample {
            params.extend(prefixed("downsample.0", conv.named_parameters()));
            params.extend(prefixed("downsample.1", bn.named_parameters()));
        }
        params
    }

    fn named_states(&self) -> Vec<(String, State)> {
        let mut states = prefixed("bn1", self.bn1.named_states());
        states.extend(prefixed("bn2", self.bn2.named_states()));
        if let Some((_, bn)) = &self.downsample {
            states.extend(prefixed("downsample.1", bn.named_states()));
        }
        states
    }
}

/// 1x1 reduce, 3x3, 1x1 expand (factor [`BOTTLENECK_EXPANSION`]) with a
/// projection shortcut whenever the stage changes channel count or stride;
/// ResNet-50's block. The stride lives on the 3x3 conv (`conv2`), torchvision
/// v1.5's placement, not on the first 1x1 conv (the original paper's v1).
pub(crate) struct Bottleneck {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv3: Conv2d,
    bn3: BatchNorm,
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl Bottleneck {
    pub(crate) fn new(axes: ResNetAxes, in_channels: usize, width: usize, stride: usize) -> Self {
        let ResNetAxes {
            channel,
            height,
            width: width_axis,
        } = axes;
        let spatial = [height, width_axis];
        let out_channels = width * BOTTLENECK_EXPANSION;
        let conv1 = Conv2d::new(channel, channel.of(width), spatial, [1, 1]).bias(false);
        let bn1 = BatchNorm::new(channel);
        let conv2 = Conv2d::new(channel, channel.of(width), spatial, [3, 3])
            .stride([stride, stride])
            .padding([1, 1])
            .bias(false);
        let bn2 = BatchNorm::new(channel);
        let conv3 = Conv2d::new(channel, channel.of(out_channels), spatial, [1, 1]).bias(false);
        let bn3 = BatchNorm::new(channel);
        let downsample = (stride != 1 || in_channels != out_channels).then(|| {
            (
                Conv2d::new(channel, channel.of(out_channels), spatial, [1, 1])
                    .stride([stride, stride])
                    .bias(false),
                BatchNorm::new(channel),
            )
        });
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
        }
    }
}

impl Module for Bottleneck {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let shape = self.conv1.output_shape(input)?;
        let shape = self.bn1.output_shape(&shape)?;
        let shape = self.conv2.output_shape(&shape)?;
        let shape = self.bn2.output_shape(&shape)?;
        let shape = self.conv3.output_shape(&shape)?;
        self.bn3.output_shape(&shape)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let shape = self.conv1.build(input, device, seed)?;
        let shape = self.bn1.build(&shape, device, seed)?;
        let shape = self.conv2.build(&shape, device, seed.wrapping_add(1))?;
        let shape = self.bn2.build(&shape, device, seed)?;
        let shape = self.conv3.build(&shape, device, seed.wrapping_add(2))?;
        let shape = self.bn3.build(&shape, device, seed)?;
        if let Some((conv, bn)) = &mut self.downsample {
            let downsampled = conv.build(input, device, seed.wrapping_add(3))?;
            bn.build(&downsampled, device, seed)?;
        }
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(input)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?.relu()?;
        let out = self.bn3.forward(&self.conv3.forward(&out)?)?;
        let shortcut = match &self.downsample {
            Some((conv, bn)) => bn.forward(&conv.forward(input)?)?,
            None => input.clone(),
        };
        out.add(&shortcut)?.relu()
    }

    fn forward_training(&self, input: &Tensor, pass: &mut TrainingPass) -> Result<Tensor> {
        let out = self
            .bn1
            .forward_training(&self.conv1.forward_training(input, pass)?, pass)?
            .relu()?;
        let out = self
            .bn2
            .forward_training(&self.conv2.forward_training(&out, pass)?, pass)?
            .relu()?;
        let out = self
            .bn3
            .forward_training(&self.conv3.forward_training(&out, pass)?, pass)?;
        let shortcut = match &self.downsample {
            Some((conv, bn)) => bn.forward_training(&conv.forward_training(input, pass)?, pass)?,
            None => input.clone(),
        };
        out.add(&shortcut)?.relu()
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut params = prefixed("conv1", self.conv1.named_parameters());
        params.extend(prefixed("bn1", self.bn1.named_parameters()));
        params.extend(prefixed("conv2", self.conv2.named_parameters()));
        params.extend(prefixed("bn2", self.bn2.named_parameters()));
        params.extend(prefixed("conv3", self.conv3.named_parameters()));
        params.extend(prefixed("bn3", self.bn3.named_parameters()));
        if let Some((conv, bn)) = &self.downsample {
            params.extend(prefixed("downsample.0", conv.named_parameters()));
            params.extend(prefixed("downsample.1", bn.named_parameters()));
        }
        params
    }

    fn named_states(&self) -> Vec<(String, State)> {
        let mut states = prefixed("bn1", self.bn1.named_states());
        states.extend(prefixed("bn2", self.bn2.named_states()));
        states.extend(prefixed("bn3", self.bn3.named_states()));
        if let Some((_, bn)) = &self.downsample {
            states.extend(prefixed("downsample.1", bn.named_states()));
        }
        states
    }
}

fn stem_layers(axes: ResNetAxes, stem: Stem) -> Vec<Box<dyn Module>> {
    let ResNetAxes {
        channel,
        height,
        width,
    } = axes;
    let spatial = [height, width];
    match stem {
        Stem::Standard => vec![
            Box::new(
                Conv2d::new(channel, channel.of(64), spatial, [7, 7])
                    .stride([2, 2])
                    .padding([3, 3])
                    .bias(false),
            ),
            Box::new(BatchNorm::new(channel)),
            Box::new(ReLU),
            Box::new(
                MaxPool2d::new(channel, spatial, [3, 3])
                    .stride([2, 2])
                    .padding([1, 1]),
            ),
        ],
        Stem::SmallInput => vec![
            Box::new(
                Conv2d::new(channel, channel.of(64), spatial, [3, 3])
                    .padding([1, 1])
                    .bias(false),
            ),
            Box::new(BatchNorm::new(channel)),
            Box::new(ReLU),
        ],
    }
}

fn basic_stages(axes: ResNetAxes) -> Vec<Box<dyn Module>> {
    let mut layers: Vec<Box<dyn Module>> = Vec::new();
    let mut in_channels = 64;
    for (stage, &depth) in STAGE_DEPTHS.iter().enumerate() {
        let out_channels = STAGE_WIDTHS[stage];
        for block in 0..depth {
            let stride = if stage != 0 && block == 0 { 2 } else { 1 };
            layers.push(Box::new(BasicBlock::new(
                axes,
                in_channels,
                out_channels,
                stride,
            )));
            in_channels = out_channels;
        }
    }
    layers
}

fn bottleneck_stages(axes: ResNetAxes) -> Vec<Box<dyn Module>> {
    let mut layers: Vec<Box<dyn Module>> = Vec::new();
    let mut in_channels = 64;
    for (stage, &depth) in STAGE_DEPTHS.iter().enumerate() {
        let width = STAGE_WIDTHS[stage];
        for block in 0..depth {
            let stride = if stage != 0 && block == 0 { 2 } else { 1 };
            layers.push(Box::new(Bottleneck::new(axes, in_channels, width, stride)));
            in_channels = width * BOTTLENECK_EXPANSION;
        }
    }
    layers
}

fn head_layers(axes: ResNetAxes, num_classes: usize) -> Vec<Box<dyn Module>> {
    let ResNetAxes {
        channel,
        height,
        width,
    } = axes;
    vec![
        Box::new(AdaptiveAvgPool2d::new([height, width], [1, 1])),
        Box::new(Flatten::new([channel, height, width], channel)),
        Box::new(Linear::new(channel, channel.of(num_classes))),
    ]
}

fn resnet(
    axes: ResNetAxes,
    num_classes: usize,
    stem: Stem,
    stages: fn(ResNetAxes) -> Vec<Box<dyn Module>>,
) -> Sequential {
    let mut layers = stem_layers(axes, stem);
    layers.extend(stages(axes));
    layers.extend(head_layers(axes, num_classes));
    Sequential::new(layers)
}

/// torchvision 0.26's `resnet34`: `BasicBlock` stages `[3, 4, 6, 3]` behind
/// the standard 7x7 stride-2 stem. `num_classes` is the final `Linear`'s
/// output extent; torchvision's own default is `1000`.
pub fn resnet34(axes: ResNetAxes, num_classes: usize) -> Sequential {
    resnet(axes, num_classes, Stem::Standard, basic_stages)
}

/// [`resnet34`] with the small-input stem: see the module doc. Not a
/// torchvision configuration.
pub fn resnet34_small_input(axes: ResNetAxes, num_classes: usize) -> Sequential {
    resnet(axes, num_classes, Stem::SmallInput, basic_stages)
}

/// torchvision 0.26's `resnet50`: `Bottleneck` stages `[3, 4, 6, 3]` behind
/// the standard 7x7 stride-2 stem. `num_classes` is the final `Linear`'s
/// output extent; torchvision's own default is `1000`.
pub fn resnet50(axes: ResNetAxes, num_classes: usize) -> Sequential {
    resnet(axes, num_classes, Stem::Standard, bottleneck_stages)
}

/// [`resnet50`] with the small-input stem: see the module doc. Not a
/// torchvision configuration.
pub fn resnet50_small_input(axes: ResNetAxes, num_classes: usize) -> Sequential {
    resnet(axes, num_classes, Stem::SmallInput, bottleneck_stages)
}
