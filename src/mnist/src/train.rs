//! The training-dynamics 1K-parameter MNIST baseline on axis.
use axis::prelude::*;
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";
const INPUTS: usize = 49;
const HIDDEN: usize = 16;
const CLASSES: usize = 10;

#[derive(Clone)]
struct Digit {
    input: [f32; INPUTS],
    label: u8,
}

struct Config {
    data: PathBuf,
    epochs: usize,
    batch: usize,
    learning_rate: f32,
    seed: u64,
    train_limit: Option<usize>,
    test_limit: Option<usize>,
}

fn be_u32(bytes: &[u8], offset: usize) -> Result<usize> {
    let chunk: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or("truncated IDX header")?
        .try_into()?;
    Ok(usize::try_from(u32::from_be_bytes(chunk))?)
}

fn pool_4x4(image: &[u8]) -> Result<[f32; INPUTS]> {
    if image.len() != 28 * 28 {
        return Err("MNIST image must contain 28x28 pixels".into());
    }
    let mut pooled = [0.0; INPUTS];
    for py in 0..7 {
        for px in 0..7 {
            let mut sum = 0_u32;
            for dy in 0..4 {
                for dx in 0..4 {
                    sum += u32::from(image[(py * 4 + dy) * 28 + px * 4 + dx]);
                }
            }
            pooled[py * 7 + px] = sum as f32 / (16.0 * 255.0);
        }
    }
    Ok(pooled)
}

fn load_split(root: &Path, images_name: &str, labels_name: &str) -> Result<Vec<Digit>> {
    let images = fs::read(root.join(images_name))?;
    let labels = fs::read(root.join(labels_name))?;
    if be_u32(&images, 0)? != 2051 || be_u32(&labels, 0)? != 2049 {
        return Err("unexpected MNIST IDX magic".into());
    }
    let count = be_u32(&images, 4)?;
    if be_u32(&labels, 4)? != count
        || be_u32(&images, 8)? != 28
        || be_u32(&images, 12)? != 28
        || images.len() != 16 + count * 28 * 28
        || labels.len() != 8 + count
    {
        return Err("inconsistent MNIST IDX dimensions".into());
    }
    (0..count)
        .map(|index| {
            let start = 16 + index * 28 * 28;
            Ok(Digit {
                input: pool_4x4(&images[start..start + 28 * 28])?,
                label: labels[8 + index],
            })
        })
        .collect()
}

fn normalize(train: &mut [Digit], test: &mut [Digit]) -> Result<()> {
    let n = train
        .len()
        .checked_mul(INPUTS)
        .ok_or("pixel count overflow")?;
    if n < 2 {
        return Err("training split is too small to normalize".into());
    }
    let mean = train
        .iter()
        .flat_map(|digit| digit.input)
        .map(f64::from)
        .sum::<f64>()
        / n as f64;
    // torch.std() uses Bessel's correction by default.
    let variance = train
        .iter()
        .flat_map(|digit| digit.input)
        .map(|value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;
    let std = variance.sqrt();
    if !std.is_finite() || std == 0.0 {
        return Err("training normalization has invalid standard deviation".into());
    }
    for digit in train.iter_mut().chain(test) {
        for value in &mut digit.input {
            *value = ((f64::from(*value) - mean) / std) as f32;
        }
    }
    Ok(())
}

fn tensors(
    digits: &[Digit],
    batch_axis: Axis,
    input_axis: Axis,
    class_axis: Axis,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::with_capacity(digits.len() * INPUTS);
    let mut targets = vec![0.0; digits.len() * CLASSES];
    for (row, digit) in digits.iter().enumerate() {
        inputs.extend(digit.input);
        targets[row * CLASSES + usize::from(digit.label)] = 1.0;
    }
    Ok((
        Tensor::from_slice(
            &inputs,
            [batch_axis.of(digits.len()), input_axis.of(INPUTS)],
            device,
        )?,
        Tensor::from_slice(
            &targets,
            [batch_axis.of(digits.len()), class_axis.of(CLASSES)],
            device,
        )?,
    ))
}

fn accuracy(
    model: &Sequential,
    digits: &[Digit],
    batch_axis: Axis,
    input_axis: Axis,
    device: &Device,
) -> Result<f32> {
    let inputs: Vec<_> = digits.iter().flat_map(|digit| digit.input).collect();
    let tensor = Tensor::from_slice(
        &inputs,
        [batch_axis.of(digits.len()), input_axis.of(INPUTS)],
        device,
    )?;
    let logits = model.forward(&tensor)?.to_vec()?;
    let correct = logits
        .chunks_exact(CLASSES)
        .zip(digits)
        .filter(|(row, digit)| {
            let predicted = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap();
            predicted == usize::from(digit.label)
        })
        .count();
    Ok(correct as f32 / digits.len() as f32)
}

fn parse() -> Result<Config> {
    let mut config = Config {
        data: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../research/training-dynamics/data/MNIST/raw"),
        epochs: 20,
        batch: 512,
        learning_rate: 3e-3,
        seed: 0,
        train_limit: None,
        test_limit: None,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                config.epochs = 5;
                config.train_limit = Some(2_048);
                config.test_limit = Some(512);
            }
            "--epochs" => config.epochs = args.next().ok_or("--epochs needs a value")?.parse()?,
            "--data" => config.data = args.next().ok_or("--data needs a path")?.into(),
            "--help" | "-h" => {
                println!("train [--smoke] [--epochs N] [--data MNIST_RAW_DIR]");
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
    let config = parse()?;
    let mut train = load_split(&config.data, TRAIN_IMAGES, TRAIN_LABELS)?;
    let mut test = load_split(&config.data, TEST_IMAGES, TEST_LABELS)?;
    train.truncate(config.train_limit.unwrap_or(train.len()));
    test.truncate(config.test_limit.unwrap_or(test.len()));
    normalize(&mut train, &mut test)?;

    let device = Device::cuda(0)?;
    let (batch_axis, input_axis, hidden_axis, class_axis) = (
        Axis::new("batch"),
        Axis::new("pixel"),
        Axis::new("hidden"),
        Axis::new("class"),
    );
    let mut model = Sequential::new((
        Linear::new(input_axis, hidden_axis.of(HIDDEN)),
        ReLU,
        Linear::new(hidden_axis, class_axis.of(CLASSES)),
    ));
    model.build(
        &Shape::new([batch_axis.of(config.batch), input_axis.of(INPUTS)])?,
        &device,
        config.seed,
    )?;
    let parameters: usize = model
        .parameters()
        .iter()
        .map(|parameter| parameter.tensor().shape().len())
        .sum();
    if parameters != 970 {
        return Err(format!("parameter budget changed: expected 970, found {parameters}").into());
    }
    println!("params: {parameters}");

    let initial_accuracy = accuracy(&model, &test, batch_axis, input_axis, &device)?;
    println!("initial test_acc {initial_accuracy:.4}");
    let mut trainer = Trainer::new(Adam::new(config.learning_rate)?);
    let train_len = train.len();
    let mut loader = FinitePassesLoader::new(train, config.batch, config.epochs, config.seed)?;
    let started = Instant::now();
    let mut weighted_loss = 0.0_f64;
    let mut reported_passes = 0;
    while let Some(batch) = loader.next_batch()? {
        let count = batch.samples.len();
        let DataRegimeReceipt::FinitePasses(receipt) = batch.regime else {
            return Err("finite-pass loader returned the wrong regime receipt".into());
        };
        let (input, target) = tensors(&batch.samples, batch_axis, input_axis, class_axis, &device)?;
        let report = trainer.step(&mut model, |model| {
            model
                .forward(&input)?
                .categorical_cross_entropy_with_logits(&target, class_axis)?
                .mean(batch_axis)
        })?;
        weighted_loss += f64::from(report.pre_update_loss()?) * count as f64;
        if receipt.completed_passes > reported_passes {
            let test_accuracy = accuracy(&model, &test, batch_axis, input_axis, &device)?;
            println!(
                "epoch {:2} train_loss {:.4} test_acc {test_accuracy:.4}",
                receipt.completed_passes,
                weighted_loss / train_len as f64
            );
            weighted_loss = 0.0;
            reported_passes = receipt.completed_passes;
        }
    }
    let receipt = loader.receipt();
    if receipt.completed_passes != config.epochs {
        return Err("finite-pass training ended before every declared pass completed".into());
    }
    println!(
        "train time: {:.2}s\n{}",
        started.elapsed().as_secs_f64(),
        receipt,
    );
    if config.train_limit.is_some() {
        let final_accuracy = accuracy(&model, &test, batch_axis, input_axis, &device)?;
        if final_accuracy <= initial_accuracy {
            return Err("smoke run did not improve held-out accuracy".into());
        }
        println!("PASS: held-out accuracy improved in the bounded migration smoke");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_matches_non_overlapping_four_by_four_means() -> Result<()> {
        let mut image = vec![0; 28 * 28];
        for row in 0..4 {
            for column in 0..4 {
                image[row * 28 + column] = 255;
            }
        }
        let pooled = pool_4x4(&image)?;
        assert_eq!(pooled[0], 1.0);
        assert!(pooled[1..].iter().all(|&value| value == 0.0));
        Ok(())
    }
}
