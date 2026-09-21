use axis::prelude::*;
use axis_imagenet64::{Axes, Classifier, ImageNet64, Sample, tensors};
use std::{env, path::PathBuf, time::Instant};

struct Config {
    data: Option<PathBuf>,
    steps: usize,
    batch: usize,
    learning_rate: f32,
    seed: u64,
}

fn parse() -> Result<Config> {
    let mut config = Config {
        data: None,
        steps: 100,
        batch: 8,
        learning_rate: 1e-3,
        seed: 42,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => config.data = Some(args.next().ok_or("--data needs a path")?.into()),
            "--steps" => config.steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--batch" => config.batch = args.next().ok_or("--batch needs a value")?.parse()?,
            "--learning-rate" => {
                config.learning_rate = args
                    .next()
                    .ok_or("--learning-rate needs a value")?
                    .parse()?
            }
            "--seed" => config.seed = args.next().ok_or("--seed needs a value")?.parse()?,
            "--smoke" => {
                config.steps = 1;
                config.batch = 2;
            }
            "--help" | "-h" => {
                println!(
                    "train [--data PREPARED.bin] [--steps N] [--batch N] [--learning-rate F] [--seed N] [--smoke]\n\
                     Without --data, a deterministic synthetic mechanics batch is used."
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if config.steps == 0 || config.batch == 0 {
        return Err("steps and batch size must be positive".into());
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive".into());
    }
    Ok(config)
}

fn synthetic(index: usize) -> Sample {
    let label = u16::try_from(index % 4).unwrap();
    let pixels = (0..64 * 64 * 3)
        .map(|coordinate| {
            let channel = coordinate / (64 * 64);
            if channel == usize::from(label) % 3 {
                192 + ((coordinate % (64 * 64) + index) % 64) as u8
            } else {
                ((coordinate % (64 * 64) * 17 + index * 29 + channel * 11) % 64) as u8
            }
        })
        .collect();
    Sample { pixels, label }
}

fn coprime_stride(len: usize, seed: u64) -> usize {
    let mut stride = usize::try_from(seed).unwrap_or(1) % len.max(1);
    stride = stride.max(1);
    while gcd(stride, len) != 1 {
        stride += 1;
    }
    stride
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn main() -> Result<()> {
    let config = parse()?;
    let mut dataset = config.data.as_ref().map(ImageNet64::open).transpose()?;
    let available = dataset
        .as_ref()
        .map_or(config.steps * config.batch, ImageNet64::len);
    let requested = config
        .steps
        .checked_mul(config.batch)
        .ok_or("sample budget overflow")?;
    if requested > available {
        return Err(format!(
            "requested {requested} fresh samples, but the prepared file contains {available}"
        )
        .into());
    }
    let stride = coprime_stride(available, config.seed | 1);
    let offset = usize::try_from(config.seed).unwrap_or(0) % available;

    let device = Device::cuda_bf16(0)?;
    let axes = Axes::new();
    let mut model = Classifier::new(&axes);
    model.build(&axes.image_shape(config.batch)?, &device, config.seed)?;
    let parameters: usize = model
        .parameters()
        .iter()
        .map(|parameter| parameter.tensor().shape().len())
        .sum();
    println!(
        "source={} samples={} batch={} steps={} params={parameters}",
        if dataset.is_some() {
            "prepared ImageNet64"
        } else {
            "synthetic mechanics"
        },
        requested,
        config.batch,
        config.steps
    );

    let mut trainer = Trainer::new(AdamW::new(config.learning_rate, 1e-4)?);
    let mut single_pass = SinglePass::new(available)?;
    let started = Instant::now();
    let mut first_loss = None;
    let mut last_loss = 0.0;
    let mut correct = 0_usize;
    for step in 0..config.steps {
        let samples = (0..config.batch)
            .map(|row| {
                let ordinal = step * config.batch + row;
                let index = (offset + ordinal * stride) % available;
                dataset
                    .as_mut()
                    .map_or_else(|| Ok(synthetic(index)), |data| data.read(index))
            })
            .collect::<Result<Vec<_>>>()?;
        single_pass.consume(samples.len())?;
        let (images, targets) = tensors(&samples, &axes, &device)?;
        let logits = model.forward(&images)?;
        correct += logits
            .categorical_accuracy(&targets, axes.vision.class)?
            .correct();
        let report = trainer.step(&mut model, |model| {
            model
                .forward(&images)?
                .categorical_cross_entropy_with_logits(&targets, axes.vision.class)?
                .mean(axes.vision.batch)
        })?;
        last_loss = report.pre_update_loss()?;
        first_loss.get_or_insert(last_loss);
        if step == 0 || step + 1 == config.steps || (step + 1) % 10 == 0 {
            println!("step={} pre_update_loss={last_loss:.4}", step + 1);
        }
    }
    println!(
        "PASS samples={} unique_sample_positions={} coverage={:.4} first_loss={:.4} last_loss={last_loss:.4} online_top1={:.4} elapsed_seconds={:.2}",
        requested,
        requested,
        single_pass.snapshot().coverage().expect("fixed corpus"),
        first_loss.unwrap(),
        correct as f32 / requested as f32,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
