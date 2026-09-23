//! Bounded DCGAN training witness on real 64x64 data: Fashion-MNIST, resized to 64x64 and
//! replicated to 3 channels (the "CelebA-free" choice for a desk-GPU smoke run). This is a
//! MECHANICS witness -- both networks train, losses move, and a generated-sample grid is saved
//! -- not an image-quality or convergence claim. See `axis::architectures::gan` for the exact
//! alternating recipe this program runs, and this crate's README for the measured run.
use axis::architectures::{
    DCGAN_DISCRIMINATOR_FEATURES, DCGAN_GENERATOR_FEATURES, DCGAN_IMAGE_CHANNELS, DCGAN_LATENT,
    dcgan_tutorial_init,
};
use axis::prelude::*;
use axis_vision_data::{ImageDataset, load_idx};
use std::{env, fs, io::Write, path::PathBuf, time::Instant};

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const IMAGE_SIDE: usize = 64;
const UPSCALE: usize = 2; // 28 -> 56 by nearest-neighbor duplication
const PAD: usize = (IMAGE_SIDE - 28 * UPSCALE) / 2; // 56 -> 64, centered
const GRID: usize = 8; // 8x8 sample grid

struct Config {
    data: PathBuf,
    out_dir: PathBuf,
    steps: usize,
    batch: usize,
    seed: u64,
    ngf: usize,
    ndf: usize,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config> {
    let mut config = Config {
        data: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/raw"),
        out_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/runs/dcgan-witness"),
        steps: 400,
        batch: 32,
        seed: 0,
        ngf: DCGAN_GENERATOR_FEATURES / 2,
        ndf: DCGAN_DISCRIMINATOR_FEATURES / 2,
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                config.steps = 40;
                config.batch = 16;
            }
            "--steps" => config.steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--batch" => config.batch = args.next().ok_or("--batch needs a value")?.parse()?,
            "--data" => config.data = args.next().ok_or("--data needs a path")?.into(),
            "--out" => config.out_dir = args.next().ok_or("--out needs a path")?.into(),
            "--help" | "-h" => {
                println!(
                    "dcgan-witness [--smoke] [--steps N] [--batch N] [--data RAW_DIR] [--out DIR]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if config.steps == 0 || config.batch == 0 {
        return Err("steps and batch size must be positive".into());
    }
    Ok(config)
}

/// Nearest-neighbor upscale 28x28 -> 56x56, zero-pad (background) to 64x64, replicate the one
/// grayscale channel to `DCGAN_IMAGE_CHANNELS`, and rescale `[0, 1]` to `[-1, 1]` to match
/// `Tanh`'s own output range. Done host-side on plain `f32`, the same way the getting-started
/// examples prepare their own batches, rather than adding a new Axis resize op for one example.
fn prepare_batch(samples: &[&axis_vision_data::ImageSample]) -> Vec<f32> {
    let mut values = vec![0.0_f32; samples.len() * DCGAN_IMAGE_CHANNELS * IMAGE_SIDE * IMAGE_SIDE];
    for (sample_index, sample) in samples.iter().enumerate() {
        for y in 0..IMAGE_SIDE {
            for x in 0..IMAGE_SIDE {
                let inner = PAD..PAD + 28 * UPSCALE;
                let pixel = if inner.contains(&y) && inner.contains(&x) {
                    let sy = (y - PAD) / UPSCALE;
                    let sx = (x - PAD) / UPSCALE;
                    sample.pixels[sy * 28 + sx]
                } else {
                    0.0
                };
                let scaled = pixel * 2.0 - 1.0;
                for channel in 0..DCGAN_IMAGE_CHANNELS {
                    let index = ((sample_index * DCGAN_IMAGE_CHANNELS + channel) * IMAGE_SIDE + y)
                        * IMAGE_SIDE
                        + x;
                    values[index] = scaled;
                }
            }
        }
    }
    values
}

/// Write a `[batch, height, width, channel]`-ordered tensor (in `[-1, 1]`) as an `n * n` grid
/// PPM (P6): no image crate dependency for one witness artifact. `n * n` must equal `batch`.
fn write_grid_ppm(
    path: &PathBuf,
    values: &[f32],
    batch: usize,
    side: usize,
    n: usize,
) -> Result<()> {
    if n * n != batch {
        return Err("grid size must match batch".into());
    }
    let grid_side = side * n;
    let mut bytes = Vec::with_capacity(grid_side * grid_side * 3);
    for gy in 0..grid_side {
        let tile_y = gy / side;
        let y = gy % side;
        for gx in 0..grid_side {
            let tile_x = gx / side;
            let x = gx % side;
            let sample = tile_y * n + tile_x;
            for channel in 0..3 {
                let index = ((sample * 3 + channel) * side + y) * side + x;
                let value = ((values[index] + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0);
                bytes.push(value as u8);
            }
        }
    }
    let mut file = fs::File::create(path)?;
    write!(file, "P6\n{grid_side} {grid_side}\n255\n")?;
    file.write_all(&bytes)?;
    Ok(())
}

fn save_samples(
    generator: &DcganGenerator,
    noise: &Tensor,
    device: &Device,
    path: &PathBuf,
) -> Result<()> {
    let _ = device;
    let fake = generator.forward(noise)?;
    write_grid_ppm(path, &fake.to_vec()?, GRID * GRID, IMAGE_SIDE, GRID)
}

fn main() -> Result<()> {
    let config = parse_args(env::args().skip(1))?;
    fs::create_dir_all(&config.out_dir)?;

    let device = Device::cuda(0)?;
    let dataset: ImageDataset = load_idx(&config.data, TRAIN_IMAGES, TRAIN_LABELS)?;
    println!(
        "Fashion-MNIST | {} samples | resized 28x28 -> 64x64, replicated to {} channels",
        dataset.samples.len(),
        DCGAN_IMAGE_CHANNELS
    );

    let (batch, channel, height, width) = (
        Axis::new("dcgan_batch"),
        Axis::new("dcgan_channel"),
        Axis::new("dcgan_height"),
        Axis::new("dcgan_width"),
    );

    let mut generator =
        DcganGenerator::with_features(channel, [height, width], DCGAN_IMAGE_CHANNELS, config.ngf);
    let mut discriminator = DcganDiscriminator::with_features(channel, [height, width], config.ndf);

    let latent_shape = Shape::new([
        batch.of(config.batch),
        channel.of(DCGAN_LATENT),
        height.of(1),
        width.of(1),
    ])?;
    let image_shape = generator.build(&latent_shape, &device, config.seed)?;
    discriminator.build(&image_shape, &device, config.seed.wrapping_add(1))?;
    dcgan_tutorial_init(&generator, &device, config.seed.wrapping_add(2))?;
    dcgan_tutorial_init(&discriminator, &device, config.seed.wrapping_add(3))?;

    let generator_params: usize = generator
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    let discriminator_params: usize = discriminator
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    println!(
        "DCGAN | ngf={} ndf={} | generator_params={generator_params} discriminator_params={discriminator_params}",
        config.ngf, config.ndf
    );

    let mut g_trainer =
        Trainer::new(Adam::with_hyperparameters(2e-4, 0.5, 0.999, 1e-8)?).with_seed(1001);
    let mut d_trainer =
        Trainer::new(Adam::with_hyperparameters(2e-4, 0.5, 0.999, 1e-8)?).with_seed(2002);

    let logits_shape = discriminator.output_shape(&image_shape)?;
    let real_labels = Tensor::from_slice(
        &vec![1.0; logits_shape.len()],
        logits_shape.dims().iter().copied(),
        &device,
    )?;
    let fake_labels = Tensor::from_slice(
        &vec![0.0; logits_shape.len()],
        logits_shape.dims().iter().copied(),
        &device,
    )?;

    let fixed_noise = Tensor::uniform(
        [
            batch.of(GRID * GRID),
            channel.of(DCGAN_LATENT),
            height.of(1),
            width.of(1),
        ],
        999,
        -1.0,
        1.0,
        &device,
    )?;
    save_samples(
        &generator,
        &fixed_noise,
        &device,
        &config.out_dir.join("before.ppm"),
    )?;

    let mut log = fs::File::create(config.out_dir.join("loss.csv"))?;
    writeln!(log, "step,d_loss,g_loss,elapsed_seconds")?;

    let started = Instant::now();
    let mut order: Vec<usize> = (0..dataset.samples.len()).collect();
    for step in 0..config.steps {
        // Deterministic pseudo-shuffle: rotate a fixed-size cursor over the dataset order,
        // reshuffled every full epoch with a step-derived seed (no external RNG dependency).
        let epoch = step * config.batch / dataset.samples.len().max(1);
        if step * config.batch % dataset.samples.len().max(1) < config.batch {
            let mut seed = config
                .seed
                .wrapping_add(epoch as u64)
                .wrapping_add(0x9E37_79B9);
            for i in (1..order.len()).rev() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let j = (seed as usize) % (i + 1);
                order.swap(i, j);
            }
        }
        let start = (step * config.batch) % dataset.samples.len().max(1);
        let indices: Vec<usize> = (0..config.batch)
            .map(|i| order[(start + i) % order.len()])
            .collect();
        let samples: Vec<&axis_vision_data::ImageSample> =
            indices.iter().map(|&i| &dataset.samples[i]).collect();
        let real_values = prepare_batch(&samples);
        let real = Tensor::from_slice(&real_values, image_shape.dims().iter().copied(), &device)?;

        let noise_seed = 3000 + step as u64;
        let d_step = d_trainer.step_training(&mut discriminator, |discriminator, pass| {
            let real_logits = discriminator.forward_training(&real, pass)?;
            let real_loss = real_logits.binary_cross_entropy_with_logits(&real_labels)?;
            let fake = generator.forward(&Tensor::uniform(
                latent_shape.dims().iter().copied(),
                noise_seed,
                -1.0,
                1.0,
                &device,
            )?)?;
            let fake_logits = discriminator.forward_training(&fake.detach(), pass)?;
            let fake_loss = fake_logits.binary_cross_entropy_with_logits(&fake_labels)?;
            real_loss
                .add(&fake_loss)?
                .mean([batch, height, width, channel])
        })?;

        let g_step = g_trainer.step_training(&mut generator, |generator, pass| {
            let noise = Tensor::uniform(
                latent_shape.dims().iter().copied(),
                noise_seed + 1,
                -1.0,
                1.0,
                &device,
            )?;
            let fake = generator.forward_training(&noise, pass)?;
            let logits = discriminator.forward(&fake)?;
            logits
                .binary_cross_entropy_with_logits(&real_labels)?
                .mean([batch, height, width, channel])
        })?;

        let d_loss = d_step.pre_update_loss()?;
        let g_loss = g_step.pre_update_loss()?;
        let elapsed = started.elapsed().as_secs_f64();
        writeln!(log, "{step},{d_loss:.6},{g_loss:.6},{elapsed:.3}")?;
        if step % 20 == 0 || step + 1 == config.steps {
            println!("step {step:4} d_loss={d_loss:.4} g_loss={g_loss:.4} elapsed={elapsed:.1}s");
        }
    }

    save_samples(
        &generator,
        &fixed_noise,
        &device,
        &config.out_dir.join("after.ppm"),
    )?;
    println!(
        "DCGAN witness complete: {} steps in {:.1}s. Samples: {}",
        config.steps,
        started.elapsed().as_secs_f64(),
        config.out_dir.display()
    );
    Ok(())
}
