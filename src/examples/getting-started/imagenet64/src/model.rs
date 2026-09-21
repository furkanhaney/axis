use crate::data::{CHANNELS, CLASSES, SIDE, Sample};
use axis::prelude::*;
use axis_vision_data::{VisionAxes, byte_image_tensors};

pub struct Axes {
    pub vision: VisionAxes,
    pub feature_1: Axis,
    pub feature_2: Axis,
    pub feature_3: Axis,
}

impl Axes {
    pub fn new() -> Self {
        Self {
            vision: VisionAxes::new(),
            feature_1: Axis::new("feature_1"),
            feature_2: Axis::new("feature_2"),
            feature_3: Axis::new("feature_3"),
        }
    }

    pub fn image_shape(&self, batch: usize) -> Result<Shape> {
        Shape::new([
            self.vision.batch.of(batch),
            self.vision.channel.of(CHANNELS),
            self.vision.height.of(SIDE),
            self.vision.width.of(SIDE),
        ])
    }
}

impl Default for Axes {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Classifier {
    conv_1: Conv2d,
    conv_2: Conv2d,
    conv_3: Conv2d,
    head: Linear,
    spatial: [Axis; 2],
}

impl Classifier {
    pub fn new(axes: &Axes) -> Self {
        Self {
            conv_1: Conv2d::new(
                axes.vision.channel,
                axes.feature_1.of(16),
                [axes.vision.height, axes.vision.width],
                [3, 3],
            )
            .stride([2, 2])
            .padding([1, 1]),
            conv_2: Conv2d::new(
                axes.feature_1,
                axes.feature_2.of(32),
                [axes.vision.height, axes.vision.width],
                [3, 3],
            )
            .stride([2, 2])
            .padding([1, 1]),
            conv_3: Conv2d::new(
                axes.feature_2,
                axes.feature_3.of(64),
                [axes.vision.height, axes.vision.width],
                [3, 3],
            )
            .stride([2, 2])
            .padding([1, 1]),
            head: Linear::new(axes.feature_3, axes.vision.class.of(CLASSES)),
            spatial: [axes.vision.height, axes.vision.width],
        }
    }

    fn prefix(prefix: &str, module: &dyn Module) -> Vec<(String, Parameter)> {
        module
            .named_parameters()
            .into_iter()
            .map(|(name, parameter)| (format!("{prefix}.{name}"), parameter))
            .collect()
    }
}

impl Module for Classifier {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let first = self.conv_1.output_shape(input)?;
        let second = self.conv_2.output_shape(&first)?;
        let third = self.conv_3.output_shape(&second)?;
        let pooled = Shape::new(
            third
                .dims()
                .iter()
                .copied()
                .filter(|dim| !self.spatial.contains(&dim.axis)),
        )?;
        self.head.output_shape(&pooled)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let first = self.conv_1.build(input, device, seed)?;
        let second = self.conv_2.build(&first, device, seed.wrapping_add(1))?;
        let third = self.conv_3.build(&second, device, seed.wrapping_add(2))?;
        let pooled = Shape::new(
            third
                .dims()
                .iter()
                .copied()
                .filter(|dim| !self.spatial.contains(&dim.axis)),
        )?;
        self.head.build(&pooled, device, seed.wrapping_add(3))
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let first = self.conv_1.forward(input)?.relu()?;
        let second = self.conv_2.forward(&first)?.relu()?;
        let third = self.conv_3.forward(&second)?.relu()?;
        self.head.forward(&third.mean(self.spatial)?)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [
            ("conv_1", &self.conv_1 as &dyn Module),
            ("conv_2", &self.conv_2 as &dyn Module),
            ("conv_3", &self.conv_3 as &dyn Module),
            ("head", &self.head as &dyn Module),
        ]
        .into_iter()
        .flat_map(|(prefix, module)| Self::prefix(prefix, module))
        .collect()
    }
}

pub fn tensors(samples: &[Sample], axes: &Axes, device: &Device) -> Result<(Tensor, Tensor)> {
    byte_image_tensors(
        samples,
        (CHANNELS, SIDE, SIDE),
        CLASSES,
        axes.vision,
        device,
        |sample| &sample.pixels,
        |sample| usize::from(sample.label),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_reduces_spatial_axes_and_preserves_batch() -> Result<()> {
        let axes = Axes::new();
        let model = Classifier::new(&axes);
        let output = model.output_shape(&axes.image_shape(7)?)?;
        assert_eq!(output.extent(axes.vision.batch)?, 7);
        assert_eq!(output.extent(axes.vision.class)?, CLASSES);
        assert_eq!(output.rank(), 2);
        Ok(())
    }
}
