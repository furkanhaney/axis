use axis::prelude::*;
use axis_vision_data::{
    ChannelStandardizers, FiniteClassificationConfig, ImageDataset, average_pool, flat_tensors,
    load_idx, train_finite_classifier,
};
use std::{env, path::PathBuf};

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";
const FEATURES: usize = 49;
const HIDDEN: usize = 32;
const CLASSES: usize = 10;

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
        data: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/raw"),
        epochs: 20,
        batch: 512,
        learning_rate: 3e-3,
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
                config.train_limit = Some(2_048);
                config.test_limit = Some(512);
                config.smoke = true;
            }
            "--epochs" => config.epochs = args.next().ok_or("--epochs needs a value")?.parse()?,
            "--batch" => config.batch = args.next().ok_or("--batch needs a value")?.parse()?,
            "--data" => config.data = args.next().ok_or("--data needs a path")?.into(),
            "--help" | "-h" => {
                println!("fashion-mnist [--smoke] [--epochs N] [--batch N] [--data RAW_DIR]");
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

fn pooled(mut dataset: ImageDataset) -> Result<ImageDataset> {
    if (dataset.channels, dataset.height, dataset.width) != (1, 28, 28) {
        return Err("Fashion-MNIST must contain 28x28 single-channel images".into());
    }
    dataset.samples = dataset
        .samples
        .iter()
        .map(|sample| average_pool(sample, 28, 28, 4))
        .collect::<Result<_>>()?;
    dataset.height = 7;
    dataset.width = 7;
    Ok(dataset)
}

fn main() -> Result<()> {
    let config = parse_args(env::args().skip(1))?;
    let mut training = pooled(load_idx(&config.data, TRAIN_IMAGES, TRAIN_LABELS)?)?;
    let mut held_out = pooled(load_idx(&config.data, TEST_IMAGES, TEST_LABELS)?)?;
    training.truncate(config.train_limit);
    held_out.truncate(config.test_limit);
    training.validate_labels(CLASSES)?;
    held_out.validate_labels(CLASSES)?;
    let normalization = ChannelStandardizers::fit(&training)?;
    normalization.transform(&mut training)?;
    normalization.transform(&mut held_out)?;

    let device = Device::cuda(0)?;
    let (batch, pixel, hidden, class) = (
        Axis::new("batch"),
        Axis::new("pixel"),
        Axis::new("hidden"),
        Axis::new("class"),
    );
    let mut model = Sequential::new((
        Linear::new(pixel, hidden.of(HIDDEN)),
        ReLU,
        Linear::new(hidden, class.of(CLASSES)),
    ));
    model.build(
        &Shape::new([batch.of(config.batch), pixel.of(FEATURES)])?,
        &device,
        config.seed,
    )?;
    println!(
        "Fashion-MNIST | 7x7 -> {HIDDEN} -> {CLASSES} | params={}",
        model
            .parameters()
            .iter()
            .map(|parameter| parameter.tensor().shape().len())
            .sum::<usize>()
    );
    let run = train_finite_classifier(
        &mut model,
        Adam::new(config.learning_rate)?,
        training.samples,
        &held_out.samples,
        FiniteClassificationConfig {
            epochs: config.epochs,
            batch: config.batch,
            seed: config.seed,
            smoke: config.smoke,
            batch_axis: batch,
            class_axis: class,
        },
        |samples| flat_tensors(samples, FEATURES, CLASSES, batch, pixel, class, &device),
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
        assert_eq!((config.epochs, config.batch), (5, 512));
        assert_eq!(
            (config.train_limit, config.test_limit),
            (Some(2_048), Some(512))
        );
        assert!(config.smoke);
        Ok(())
    }
}
