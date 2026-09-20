//! Sequential axis migration of the population MNIST probe.
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
const LR_MIN: f32 = 1e-4;
const LR_MAX: f32 = 3e-2;

#[derive(Clone)]
struct Digit {
    input: [f32; INPUTS],
    label: u8,
}

struct Config {
    data: PathBuf,
    population: usize,
    passes: usize,
    batch: usize,
    seed: u64,
    train_limit: Option<usize>,
    test_limit: Option<usize>,
    smoke: bool,
}

#[derive(Clone, Copy)]
struct ResultRow {
    learning_rate: f32,
    initial_accuracy: f32,
    final_accuracy: f32,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        ((self.u64() >> 40) as f32 + 0.5) / (1_u32 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        let radius = (-2.0 * self.unit().ln()).sqrt();
        radius * (std::f32::consts::TAU * self.unit()).cos()
    }

    fn log_uniform(&mut self, low: f32, high: f32) -> f32 {
        (low.ln() + self.unit() * (high.ln() - low.ln()))
            .exp()
            .clamp(low, high)
    }
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
    population_axis: Axis,
    input_axis: Axis,
    device: &Device,
) -> Result<Vec<f32>> {
    let inputs: Vec<_> = digits.iter().flat_map(|digit| digit.input).collect();
    let input = Tensor::from_slice(
        &inputs,
        [batch_axis.of(digits.len()), input_axis.of(INPUTS)],
        device,
    )?;
    let logits = model.forward(&input)?.to_vec()?;
    let population = model
        .output_shape(&Shape::new([
            batch_axis.of(digits.len()),
            input_axis.of(INPUTS),
        ])?)?
        .extent(population_axis)?;
    let mut correct = vec![0; population];
    for (batch, digit) in digits.iter().enumerate() {
        for (member, member_correct) in correct.iter_mut().enumerate() {
            let start = (batch * population + member) * CLASSES;
            let prediction = logits[start..start + CLASSES]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap();
            *member_correct += usize::from(prediction == usize::from(digit.label));
        }
    }
    Ok(correct
        .into_iter()
        .map(|count| count as f32 / digits.len() as f32)
        .collect())
}

fn kaiming_initialize(model: &Sequential, rng: &mut Rng) -> Result<()> {
    for (name, parameter) in model.named_parameters() {
        let shape = parameter.tensor().shape().clone();
        let values = if name.ends_with(".weight") {
            let fan_in = if name == "0.weight" { INPUTS } else { HIDDEN };
            let scale = (2.0 / fan_in as f32).sqrt();
            (0..shape.len())
                .map(|_| rng.normal() * scale)
                .collect::<Vec<_>>()
        } else {
            vec![0.0; shape.len()]
        };
        parameter.set_values(&values)?;
    }
    Ok(())
}

fn train_population(
    train: &[Digit],
    test: &[Digit],
    config: &Config,
    rates: Vec<f32>,
    initialization_seed: u64,
    device: &Device,
) -> Result<Vec<ResultRow>> {
    let (batch_axis, population_axis, input_axis, hidden_axis, class_axis) = (
        Axis::new("batch"),
        Axis::new("population"),
        Axis::new("pixel"),
        Axis::new("hidden"),
        Axis::new("class"),
    );
    let mut model = Sequential::new((
        PopulationLinear::new(
            population_axis.of(config.population),
            input_axis,
            hidden_axis.of(HIDDEN),
        ),
        ReLU,
        PopulationLinear::new(
            population_axis.of(config.population),
            hidden_axis,
            class_axis.of(CLASSES),
        ),
    ));
    model.build(
        &Shape::new([batch_axis.of(config.batch), input_axis.of(INPUTS)])?,
        device,
        initialization_seed,
    )?;
    let parameters: usize = model
        .parameters()
        .iter()
        .map(|parameter| parameter.tensor().shape().len())
        .sum();
    if parameters != 970 * config.population {
        return Err(format!(
            "parameter budget changed: expected {}, found {parameters}",
            970 * config.population
        )
        .into());
    }
    kaiming_initialize(&model, &mut Rng::new(initialization_seed))?;
    let initial_accuracy = accuracy(
        &model,
        test,
        batch_axis,
        population_axis,
        input_axis,
        device,
    )?;
    let mut trainer = Trainer::new(Adam::with_axis_learning_rates(
        population_axis,
        rates.clone(),
    )?);
    let mut loader =
        FinitePassesLoader::new(train.to_vec(), config.batch, config.passes, config.seed)?;
    while let Some(batch) = loader.next_batch()? {
        let (input, target) = tensors(&batch.samples, batch_axis, input_axis, class_axis, device)?;
        trainer.step(&mut model, |model| {
            model
                .forward(&input)?
                .categorical_cross_entropy_with_logits(&target, class_axis)?
                .mean(batch_axis)?
                .mean(population_axis)?
                .scale(config.population as f32)
        })?;
    }
    let receipt = loader.receipt();
    if receipt.completed_passes != config.passes
        || receipt.observations != train.len() * config.passes
    {
        return Err("population did not complete its declared finite passes".into());
    }
    let final_accuracy = accuracy(
        &model,
        test,
        batch_axis,
        population_axis,
        input_axis,
        device,
    )?;
    Ok(rates
        .into_iter()
        .zip(initial_accuracy)
        .zip(final_accuracy)
        .map(
            |((learning_rate, initial_accuracy), final_accuracy)| ResultRow {
                learning_rate,
                initial_accuracy,
                final_accuracy,
            },
        )
        .collect())
}

fn parse() -> Result<Config> {
    let mut config = Config {
        data: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../../research/training-dynamics/data/MNIST/raw"),
        population: 512,
        passes: 20,
        batch: 512,
        seed: 0,
        train_limit: None,
        test_limit: None,
        smoke: false,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                config.smoke = true;
                config.population = 4;
                config.passes = 5;
                config.train_limit = Some(2_048);
                config.test_limit = Some(512);
            }
            "--pop" => {
                config.population = args.next().ok_or("--pop needs a value")?.parse()?;
            }
            "--passes" => {
                config.passes = args.next().ok_or("--passes needs a value")?.parse()?;
            }
            "--data" => config.data = args.next().ok_or("--data needs a path")?.into(),
            "--help" | "-h" => {
                println!("train [--smoke] [--pop N] [--passes N] [--data MNIST_RAW_DIR]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if config.population == 0 || config.passes == 0 || config.batch == 0 {
        return Err("population, passes, and batch size must be positive".into());
    }
    Ok(config)
}

fn mean(rows: &[ResultRow]) -> f32 {
    rows.iter().map(|row| row.final_accuracy).sum::<f32>() / rows.len() as f32
}

fn main() -> Result<()> {
    let config = parse()?;
    let mut train = load_split(&config.data, TRAIN_IMAGES, TRAIN_LABELS)?;
    let mut test = load_split(&config.data, TEST_IMAGES, TEST_LABELS)?;
    train.truncate(config.train_limit.unwrap_or(train.len()));
    test.truncate(config.test_limit.unwrap_or(test.len()));
    normalize(&mut train, &mut test)?;

    let device = Device::cuda(0)?;
    let mut rng = Rng::new(config.seed);
    let rates = (0..config.population)
        .map(|_| rng.log_uniform(LR_MIN, LR_MAX))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let mut rows = train_population(&train, &test, &config, rates, rng.u64(), &device)?;
    for (member, row) in rows.iter().enumerate() {
        println!(
            "member {:4}/{:4} lr={:.3e} initial_acc={:.4} final_acc={:.4}",
            member + 1,
            config.population,
            row.learning_rate,
            row.initial_accuracy,
            row.final_accuracy,
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    rows.sort_by(|a, b| a.learning_rate.total_cmp(&b.learning_rate));
    let best = rows
        .iter()
        .map(|row| row.final_accuracy)
        .reduce(f32::max)
        .ok_or("empty population")?;
    let worst = rows
        .iter()
        .map(|row| row.final_accuracy)
        .reduce(f32::min)
        .unwrap();
    let mut sorted_accuracy = rows
        .iter()
        .map(|row| row.final_accuracy)
        .collect::<Vec<_>>();
    sorted_accuracy.sort_by(f32::total_cmp);
    println!("population {} device cuda:0", config.population);
    println!(
        "fused population time {elapsed:.2}s -> {:.2} runs/s",
        config.population as f64 / elapsed
    );
    println!(
        "accuracy: best {best:.4} median {:.4} worst {worst:.4}",
        sorted_accuracy[sorted_accuracy.len() / 2]
    );
    let quartiles = rows.len().min(4);
    print!("accuracy by learning-rate quantile (low->high):");
    for group in 0..quartiles {
        let start = group * rows.len() / quartiles;
        let end = (group + 1) * rows.len() / quartiles;
        print!(" {:.4}", mean(&rows[start..end]));
    }
    println!();
    println!(
        "learning-rate range sampled: [{:.2e}, {:.2e}]",
        rows.first().unwrap().learning_rate,
        rows.last().unwrap().learning_rate,
    );
    if config.smoke {
        let best_initial = rows
            .iter()
            .map(|row| row.initial_accuracy)
            .reduce(f32::max)
            .unwrap();
        if !rows.iter().all(|row| row.final_accuracy.is_finite())
            || rows.first().unwrap().learning_rate == rows.last().unwrap().learning_rate
            || best <= best_initial
        {
            return Err("population smoke invariant failed".into());
        }
        println!("PASS: bounded population improved held-out accuracy");
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

    #[test]
    fn sampled_rates_stay_inside_the_declared_log_interval() {
        let mut rng = Rng::new(0);
        let rates = (0..1_000)
            .map(|_| rng.log_uniform(LR_MIN, LR_MAX))
            .collect::<Vec<_>>();
        assert!(rates.iter().all(|&rate| (LR_MIN..=LR_MAX).contains(&rate)));
        assert!(rates.iter().any(|&rate| rate < 1e-3));
        assert!(rates.iter().any(|&rate| rate > 1e-2));
    }
}
