//! VGG16 (Simonyan and Zisserman, "Very Deep Convolutional Networks for
//! Large-Scale Image Recognition", 2015, configuration D), matching
//! torchvision 0.26's `vgg16`/`vgg16_bn` exactly: thirteen 3x3 stride-1
//! padding-1 convolutions with a learned bias (torchvision keeps the bias on
//! every conv here, `vgg16_bn` included, unlike ResNet's bias-free
//! pre-BatchNorm convs), a 2x2 stride-2 max pool after each of the five
//! blocks, an adaptive average pool to 7x7, and a
//! 25088-4096-4096-`num_classes` classifier with ReLU and `Dropout(0.5)`
//! between its Linear layers. `vgg16_bn` inserts a `BatchNorm` after every
//! conv and before its ReLU; nothing else differs.
//!
//! Axis modules never take a batch axis (every layer here passes through
//! whatever axes it does not itself name), so these constructors take only
//! the channel and two spatial axes; a caller's input tensor carries a batch
//! axis of its own choosing alongside them.

use crate::{
    AdaptiveAvgPool2d, Axis, BatchNorm, Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module, ReLU,
    Sequential,
};

/// The channel and two spatial axes every VGG layer reads and writes.
#[derive(Clone, Copy)]
pub struct VggAxes {
    pub channel: Axis,
    pub height: Axis,
    pub width: Axis,
}

/// Configuration D (`torchvision.models.vgg16`'s `cfgs["D"]`): output channel
/// count for each conv, with a max pool after each block.
const CFG_D: [Block; 18] = [
    Block::Conv(64),
    Block::Conv(64),
    Block::Pool,
    Block::Conv(128),
    Block::Conv(128),
    Block::Pool,
    Block::Conv(256),
    Block::Conv(256),
    Block::Conv(256),
    Block::Pool,
    Block::Conv(512),
    Block::Conv(512),
    Block::Conv(512),
    Block::Pool,
    Block::Conv(512),
    Block::Conv(512),
    Block::Conv(512),
    Block::Pool,
];

#[derive(Clone, Copy)]
enum Block {
    Conv(usize),
    Pool,
}

fn feature_layers(axes: VggAxes, batch_norm: bool) -> Vec<Box<dyn Module>> {
    let VggAxes {
        channel,
        height,
        width,
    } = axes;
    let spatial = [height, width];
    let mut layers: Vec<Box<dyn Module>> = Vec::new();
    for block in CFG_D {
        match block {
            Block::Conv(out_channels) => {
                layers.push(Box::new(
                    Conv2d::new(channel, channel.of(out_channels), spatial, [3, 3]).padding([1, 1]),
                ));
                if batch_norm {
                    layers.push(Box::new(BatchNorm::new(channel)));
                }
                layers.push(Box::new(ReLU));
            }
            Block::Pool => {
                layers.push(Box::new(MaxPool2d::new(channel, spatial, [2, 2])));
            }
        }
    }
    layers
}

fn classifier_layers(axes: VggAxes, num_classes: usize) -> Vec<Box<dyn Module>> {
    let VggAxes { channel, .. } = axes;
    vec![
        Box::new(Linear::new(channel, channel.of(4096))),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5).expect("0.5 is a valid dropout probability")),
        Box::new(Linear::new(channel, channel.of(4096))),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5).expect("0.5 is a valid dropout probability")),
        Box::new(Linear::new(channel, channel.of(num_classes))),
    ]
}

fn vgg(axes: VggAxes, num_classes: usize, batch_norm: bool) -> Sequential {
    let VggAxes {
        channel,
        height,
        width,
    } = axes;
    let mut layers = feature_layers(axes, batch_norm);
    layers.push(Box::new(AdaptiveAvgPool2d::new([height, width], [7, 7])));
    layers.push(Box::new(Flatten::new([channel, height, width], channel)));
    layers.extend(classifier_layers(axes, num_classes));
    Sequential::new(layers)
}

/// torchvision 0.26's `vgg16`: configuration D with no BatchNorm.
/// `num_classes` is the classifier's final `Linear` output extent;
/// torchvision's own default is `1000`.
pub fn vgg16(axes: VggAxes, num_classes: usize) -> Sequential {
    vgg(axes, num_classes, false)
}

/// torchvision 0.26's `vgg16_bn`: configuration D with a `BatchNorm` after
/// every conv, before its ReLU. `num_classes` is the classifier's final
/// `Linear` output extent; torchvision's own default is `1000`.
pub fn vgg16_bn(axes: VggAxes, num_classes: usize) -> Sequential {
    vgg(axes, num_classes, true)
}
