use axis::prelude::*;
use axis_vision_data::{
    ChannelStandardizers, FiniteClassificationConfig, ImageSample, VisionAxes, image_tensors,
    load_cifar100, train_finite_classifier,
};
use std::{env, path::PathBuf};

const CLASSES: usize = 100;
const FEATURES: usize = 32;

struct Classifier {
    conv: Conv2d,
    head: Linear,
    spatial: [Axis; 2],
}

impl Classifier {
    fn new(axes: VisionAxes, feature: Axis) -> Self {
        Self {
            conv: Conv2d::new(
                axes.channel,
                feature.of(FEATURES),
                [axes.height, axes.width],
                [4, 4],
            )
            .stride([4, 4]),
            head: Linear::new(feature, axes.class.of(CLASSES)),
            spatial: [axes.height, axes.width],
        }
    }
}

impl Module for Classifier {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let convolution = self.conv.output_shape(input)?;
        let pooled = Shape::new(
            convolution
                .dims()
                .iter()
                .copied()
                .filter(|dim| !self.spatial.contains(&dim.axis)),
        )?;
        self.head.output_shape(&pooled)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let convolution = self.conv.build(input, device, seed)?;
        let pooled = Shape::new(
            convolution
                .dims()
                .iter()
                .copied()
                .filter(|dim| !self.spatial.contains(&dim.axis)),
        )?;
        self.head.build(&pooled, device, seed.wrapping_add(1))
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.head
            .forward(&self.conv.forward(input)?.relu()?.mean(self.spatial)?)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [("conv", &self.conv as &dyn Module), ("head", &self.head)]
            .into_iter()
            .flat_map(|(prefix, module)| {
                module
                    .named_parameters()
                    .into_iter()
                    .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
            })
            .collect()
    }
}

struct Config {
    data: PathBuf,
    epochs: usize,
    batch: usize,
    learning_rate: f32,
    seed: u64,
    train_limit: Option<usize>,
    test_limit: Option<usize>,
    smoke: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config> {
    let mut config = Config {
        data: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/cifar-100-binary"),
        epochs: 20,
        batch: 128,
        learning_rate: 1e-3,
        seed: 0,
        train_limit: None,
        test_limit: None,
        smoke: false,
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                config.epochs = 5;
                config.batch = 64;
                config.train_limit = Some(2_048);
                config.test_limit = Some(512);
                config.smoke = true;
            }
            "--epochs" => config.epochs = args.next().ok_or("--epochs needs a value")?.parse()?,
            "--batch" => config.batch = args.next().ok_or("--batch needs a value")?.parse()?,
            "--data" => config.data = args.next().ok_or("--data needs a path")?.into(),
            "--help" | "-h" => {
                println!("cifar100 [--smoke] [--epochs N] [--batch N] [--data BINARY_DIR]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if config.epochs == 0 || config.batch == 0 {
        return Err("epochs and batch size must be positive".into());
    }
    Ok(config)
}

fn main() -> Result<()> {
    let config = parse_args(env::args().skip(1))?;
    let mut training = load_cifar100(&config.data.join("train.bin"))?;
    let mut held_out = load_cifar100(&config.data.join("test.bin"))?;
    training.truncate(config.train_limit);
    held_out.truncate(config.test_limit);
    training.validate_labels(CLASSES)?;
    held_out.validate_labels(CLASSES)?;
    let normalization = ChannelStandardizers::fit(&training)?;
    normalization.transform(&mut training)?;
    normalization.transform(&mut held_out)?;

    let device = Device::cuda(0)?;
    let axes = VisionAxes::new();
    let feature = Axis::new("feature");
    let mut model = Classifier::new(axes, feature);
    model.build(
        &Shape::new([
            axes.batch.of(config.batch),
            axes.channel.of(3),
            axes.height.of(32),
            axes.width.of(32),
        ])?,
        &device,
        config.seed,
    )?;
    println!(
        "CIFAR-100 | RGB 32x32 -> conv4/stride4/{FEATURES} -> mean -> {CLASSES} | params={}",
        model
            .parameters()
            .iter()
            .map(|parameter| parameter.tensor().shape().len())
            .sum::<usize>()
    );
    let tensors =
        |samples: &[ImageSample]| image_tensors(samples, (3, 32, 32), CLASSES, axes, &device);
    let run = train_finite_classifier(
        &mut model,
        AdamW::new(config.learning_rate, 1e-4)?,
        training.samples,
        &held_out.samples,
        FiniteClassificationConfig {
            epochs: config.epochs,
            batch: config.batch,
            seed: config.seed,
            smoke: config.smoke,
            batch_axis: axes.batch,
            class_axis: axes.class,
        },
        tensors,
    )?;
    println!(
        "final held_out_accuracy={:.4} observations={}",
        run.final_accuracy, run.observations
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_defaults_are_deterministic_and_bounded() -> Result<()> {
        let config = parse_args(["--smoke".to_owned()])?;
        assert_eq!((config.epochs, config.batch), (5, 64));
        assert_eq!(
            (config.train_limit, config.test_limit),
            (Some(2_048), Some(512))
        );
        assert!(config.smoke);
        Ok(())
    }
}
