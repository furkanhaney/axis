//! Small, strict dataset adapters shared by Axis's introductory vision programs.

use axis::prelude::*;
use std::{fs, path::Path, time::Instant};

#[derive(Clone, Debug, PartialEq)]
pub struct ImageSample {
    pub pixels: Vec<f32>,
    pub label: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageDataset {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub samples: Vec<ImageSample>,
}

impl ImageDataset {
    pub fn truncate(&mut self, limit: Option<usize>) {
        self.samples.truncate(limit.unwrap_or(self.samples.len()));
    }

    pub fn pixels_per_image(&self) -> usize {
        self.channels * self.height * self.width
    }

    pub fn validate_labels(&self, classes: usize) -> Result<()> {
        if classes == 0 {
            return Err("class count must be positive".into());
        }
        if self.samples.iter().any(|sample| sample.label >= classes) {
            return Err("dataset label is outside the declared class axis".into());
        }
        Ok(())
    }
}

fn be_u32(bytes: &[u8], offset: usize) -> Result<usize> {
    let chunk: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or("truncated IDX header")?
        .try_into()?;
    Ok(usize::try_from(u32::from_be_bytes(chunk))?)
}

pub fn parse_idx(images: &[u8], labels: &[u8]) -> Result<ImageDataset> {
    if be_u32(images, 0)? != 2051 || be_u32(labels, 0)? != 2049 {
        return Err("unexpected IDX magic".into());
    }
    let count = be_u32(images, 4)?;
    let height = be_u32(images, 8)?;
    let width = be_u32(images, 12)?;
    let image_len = height
        .checked_mul(width)
        .ok_or("IDX image dimensions overflow")?;
    let payload_len = count
        .checked_mul(image_len)
        .and_then(|length| length.checked_add(16))
        .ok_or("IDX image payload length overflow")?;
    let label_len = count.checked_add(8).ok_or("IDX label length overflow")?;
    if height == 0
        || width == 0
        || be_u32(labels, 4)? != count
        || images.len() != payload_len
        || labels.len() != label_len
    {
        return Err("inconsistent IDX dimensions".into());
    }
    let samples = (0..count)
        .map(|index| {
            let start = 16 + index * image_len;
            ImageSample {
                pixels: images[start..start + image_len]
                    .iter()
                    .map(|&value| f32::from(value) / 255.0)
                    .collect(),
                label: usize::from(labels[8 + index]),
            }
        })
        .collect();
    Ok(ImageDataset {
        channels: 1,
        height,
        width,
        samples,
    })
}

pub fn load_idx(root: &Path, images: &str, labels: &str) -> Result<ImageDataset> {
    parse_idx(&fs::read(root.join(images))?, &fs::read(root.join(labels))?)
}

const CIFAR100_SIDE: usize = 32;
const CIFAR100_CHANNELS: usize = 3;
const CIFAR100_PIXELS: usize = CIFAR100_CHANNELS * CIFAR100_SIDE * CIFAR100_SIDE;
const CIFAR100_RECORD: usize = 2 + CIFAR100_PIXELS;

/// Parse the canonical CIFAR-100 binary layout: coarse label, fine label, then planar RGB.
pub fn parse_cifar100(bytes: &[u8]) -> Result<ImageDataset> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(CIFAR100_RECORD) {
        return Err("CIFAR-100 binary length is not a whole number of records".into());
    }
    let samples = bytes
        .chunks_exact(CIFAR100_RECORD)
        .map(|record| ImageSample {
            label: usize::from(record[1]),
            pixels: record[2..]
                .iter()
                .map(|&value| f32::from(value) / 255.0)
                .collect(),
        })
        .collect();
    Ok(ImageDataset {
        channels: CIFAR100_CHANNELS,
        height: CIFAR100_SIDE,
        width: CIFAR100_SIDE,
        samples,
    })
}

pub fn load_cifar100(path: &Path) -> Result<ImageDataset> {
    parse_cifar100(&fs::read(path)?)
}

pub fn average_pool(
    sample: &ImageSample,
    height: usize,
    width: usize,
    factor: usize,
) -> Result<ImageSample> {
    if factor == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(factor)
        || !width.is_multiple_of(factor)
    {
        return Err("pooling factor must divide positive image dimensions".into());
    }
    if sample.pixels.len() != height * width {
        return Err("single-channel image length does not match pooling geometry".into());
    }
    let output_height = height / factor;
    let output_width = width / factor;
    let mut pixels = vec![0.0; output_height * output_width];
    for output_y in 0..output_height {
        for output_x in 0..output_width {
            let mut sum = 0.0;
            for y in 0..factor {
                for x in 0..factor {
                    sum += sample.pixels[(output_y * factor + y) * width + output_x * factor + x];
                }
            }
            pixels[output_y * output_width + output_x] = sum / (factor * factor) as f32;
        }
    }
    Ok(ImageSample {
        pixels,
        label: sample.label,
    })
}

#[derive(Clone, Debug)]
pub struct ChannelStandardizers(Vec<Standardizer>);

impl ChannelStandardizers {
    pub fn fit(dataset: &ImageDataset) -> Result<Self> {
        if dataset.samples.is_empty() {
            return Err("cannot fit normalization on an empty training split".into());
        }
        let plane = dataset.height * dataset.width;
        let expected = dataset.pixels_per_image();
        if dataset
            .samples
            .iter()
            .any(|sample| sample.pixels.len() != expected)
        {
            return Err("image length does not match dataset geometry".into());
        }
        let mut standardizers = Vec::with_capacity(dataset.channels);
        for channel in 0..dataset.channels {
            let values: Vec<_> = dataset
                .samples
                .iter()
                .flat_map(|sample| sample.pixels[channel * plane..(channel + 1) * plane].iter())
                .copied()
                .collect();
            standardizers.push(Standardizer::fit(&values)?);
        }
        Ok(Self(standardizers))
    }

    pub fn transform(&self, dataset: &mut ImageDataset) -> Result<()> {
        if self.0.len() != dataset.channels {
            return Err("normalizer channel count does not match dataset".into());
        }
        let plane = dataset.height * dataset.width;
        let expected = dataset.pixels_per_image();
        if dataset
            .samples
            .iter()
            .any(|sample| sample.pixels.len() != expected)
        {
            return Err("image length does not match dataset geometry".into());
        }
        if dataset
            .samples
            .iter()
            .flat_map(|sample| &sample.pixels)
            .any(|value| !value.is_finite())
        {
            return Err("image pixels must be finite".into());
        }
        for sample in &mut dataset.samples {
            for (channel, standardizer) in self.0.iter().enumerate() {
                standardizer.transform_in_place(
                    &mut sample.pixels[channel * plane..(channel + 1) * plane],
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct VisionAxes {
    pub batch: Axis,
    pub channel: Axis,
    pub height: Axis,
    pub width: Axis,
    pub class: Axis,
}

impl VisionAxes {
    pub fn new() -> Self {
        Self {
            batch: Axis::new("batch"),
            channel: Axis::new("channel"),
            height: Axis::new("height"),
            width: Axis::new("width"),
            class: Axis::new("class"),
        }
    }
}

impl Default for VisionAxes {
    fn default() -> Self {
        Self::new()
    }
}

pub fn image_tensors(
    samples: &[ImageSample],
    geometry: (usize, usize, usize),
    classes: usize,
    axes: VisionAxes,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (channels, height, width) = geometry;
    let pixels_per_image = channels
        .checked_mul(height)
        .and_then(|length| length.checked_mul(width))
        .ok_or("image geometry overflow")?;
    if samples.is_empty() {
        return Err("classification batch must not be empty".into());
    }
    if samples
        .iter()
        .any(|sample| sample.pixels.len() != pixels_per_image || sample.label >= classes)
    {
        return Err("classification sample does not match declared geometry or classes".into());
    }
    let mut pixels = Vec::with_capacity(samples.len() * pixels_per_image);
    let mut targets = vec![0.0; samples.len() * classes];
    for (row, sample) in samples.iter().enumerate() {
        pixels.extend_from_slice(&sample.pixels);
        targets[row * classes + sample.label] = 1.0;
    }
    Ok((
        Tensor::from_slice(
            &pixels,
            [
                axes.batch.of(samples.len()),
                axes.channel.of(channels),
                axes.height.of(height),
                axes.width.of(width),
            ],
            device,
        )?,
        Tensor::from_slice(
            &targets,
            [axes.batch.of(samples.len()), axes.class.of(classes)],
            device,
        )?,
    ))
}

pub fn flat_tensors(
    samples: &[ImageSample],
    features: usize,
    classes: usize,
    batch_axis: Axis,
    feature_axis: Axis,
    class_axis: Axis,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|sample| sample.pixels.len() != features || sample.label >= classes)
    {
        return Err("classification sample does not match declared features or classes".into());
    }
    let mut inputs = Vec::with_capacity(samples.len() * features);
    let mut targets = vec![0.0; samples.len() * classes];
    for (row, sample) in samples.iter().enumerate() {
        inputs.extend_from_slice(&sample.pixels);
        targets[row * classes + sample.label] = 1.0;
    }
    Ok((
        Tensor::from_slice(
            &inputs,
            [batch_axis.of(samples.len()), feature_axis.of(features)],
            device,
        )?,
        Tensor::from_slice(
            &targets,
            [batch_axis.of(samples.len()), class_axis.of(classes)],
            device,
        )?,
    ))
}

#[derive(Clone, Copy, Debug)]
pub struct FiniteClassificationConfig {
    pub epochs: usize,
    pub batch: usize,
    pub seed: u64,
    pub smoke: bool,
    pub batch_axis: Axis,
    pub class_axis: Axis,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassificationRun {
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub observations: usize,
}

fn evaluate<M, F>(
    model: &M,
    samples: &[ImageSample],
    batch: usize,
    class: Axis,
    tensorize: &F,
) -> Result<f32>
where
    M: Module,
    F: Fn(&[ImageSample]) -> Result<(Tensor, Tensor)>,
{
    if samples.is_empty() {
        return Err("evaluation split must not be empty".into());
    }
    let mut receipt: Option<CategoricalAccuracy> = None;
    for chunk in samples.chunks(batch) {
        let (inputs, targets) = tensorize(chunk)?;
        let observed = model
            .forward(&inputs)?
            .categorical_accuracy(&targets, class)?;
        receipt = Some(match receipt {
            Some(previous) => previous.merge(observed),
            None => observed,
        });
    }
    Ok(receipt.expect("nonempty evaluation split").fraction())
}

/// Train an already-built classifier for exact finite passes over one population.
///
/// Dataset parsing and model construction stay visible in each example; this
/// function owns the otherwise identical update order, receipts, and held-out
/// categorical evaluation.
pub fn train_finite_classifier<M, O, F>(
    model: &mut M,
    optimizer: O,
    training: Vec<ImageSample>,
    held_out: &[ImageSample],
    config: FiniteClassificationConfig,
    tensorize: F,
) -> Result<ClassificationRun>
where
    M: Module,
    O: Optimizer,
    F: Fn(&[ImageSample]) -> Result<(Tensor, Tensor)>,
{
    if config.epochs == 0 || config.batch == 0 || training.is_empty() {
        return Err("epochs, batch size, and training split must be nonzero".into());
    }
    let initial_accuracy = evaluate(model, held_out, config.batch, config.class_axis, &tensorize)?;
    println!("initial held_out_accuracy={initial_accuracy:.4}");
    let mut learning = config
        .smoke
        .then(|| {
            LearningProgress::new(
                LearningDirection::Increase,
                LearningLimits::any_improvement(),
                LearningObservation::new(
                    "categorical accuracy",
                    "bounded held-out split",
                    "training samples",
                    0,
                    f64::from(initial_accuracy),
                )?,
            )
        })
        .transpose()?;
    let training_len = training.len();
    let mut loader = FinitePassesLoader::new(training, config.batch, config.epochs, config.seed)?;
    let mut trainer = Trainer::new(optimizer);
    let mut weighted_loss = 0.0_f64;
    let mut completed_passes = 0;
    let started = Instant::now();
    while let Some(batch) = loader.next_batch()? {
        let count = batch.samples.len();
        let DataRegimeReceipt::FinitePasses(progress) = batch.regime else {
            return Err("finite-pass loader returned the wrong regime receipt".into());
        };
        let (inputs, targets) = tensorize(&batch.samples)?;
        let report = trainer.step(model, |model| {
            model
                .forward(&inputs)?
                .categorical_cross_entropy_with_logits(&targets, config.class_axis)?
                .mean(config.batch_axis)
        })?;
        weighted_loss += f64::from(report.pre_update_loss()?) * count as f64;
        if progress.completed_passes > completed_passes {
            let accuracy = evaluate(model, held_out, config.batch, config.class_axis, &tensorize)?;
            println!(
                "epoch {:2} train_loss {:.4} held_out_accuracy {accuracy:.4}",
                progress.completed_passes,
                weighted_loss / training_len as f64
            );
            weighted_loss = 0.0;
            completed_passes = progress.completed_passes;
        }
    }
    let receipt = loader.receipt();
    let final_accuracy = evaluate(model, held_out, config.batch, config.class_axis, &tensorize)?;
    println!(
        "train time: {:.2}s\n{}",
        started.elapsed().as_secs_f64(),
        receipt
    );
    if let Some(learning) = learning.as_mut() {
        learning.observe(LearningObservation::new(
            "categorical accuracy",
            "bounded held-out split",
            "training samples",
            u64::try_from(receipt.observations)?,
            f64::from(final_accuracy),
        )?)?;
        println!("{}", learning.assert_learning()?);
    }
    Ok(ClassificationRun {
        initial_accuracy,
        final_accuracy,
        observations: receipt.observations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_be(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_be_bytes());
    }

    #[test]
    fn idx_parser_checks_geometry_and_scales_pixels() -> Result<()> {
        let mut images = Vec::new();
        for value in [2051, 2, 2, 2] {
            push_be(&mut images, value);
        }
        images.extend([0, 64, 128, 255, 255, 128, 64, 0]);
        let mut labels = Vec::new();
        for value in [2049, 2] {
            push_be(&mut labels, value);
        }
        labels.extend([3, 7]);

        let parsed = parse_idx(&images, &labels)?;
        assert_eq!((parsed.channels, parsed.height, parsed.width), (1, 2, 2));
        assert_eq!(parsed.samples[1].label, 7);
        assert_eq!(
            parsed.samples[0].pixels,
            [0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0]
        );
        images.pop();
        assert!(parse_idx(&images, &labels).is_err());
        Ok(())
    }

    #[test]
    fn cifar100_parser_uses_fine_label_and_planar_rgb() -> Result<()> {
        let mut record = vec![0; CIFAR100_RECORD];
        record[0] = 4;
        record[1] = 93;
        record[2] = 255;
        record[2 + 1024] = 128;
        record[2 + 2048] = 64;
        let parsed = parse_cifar100(&record)?;
        assert_eq!((parsed.channels, parsed.height, parsed.width), (3, 32, 32));
        assert_eq!(parsed.samples[0].label, 93);
        assert_eq!(parsed.samples[0].pixels[0], 1.0);
        assert_eq!(parsed.samples[0].pixels[1024], 128.0 / 255.0);
        assert!(parse_cifar100(&record[..record.len() - 1]).is_err());
        Ok(())
    }

    #[test]
    fn channel_normalization_is_fitted_only_on_training_pixels() -> Result<()> {
        let mut training = ImageDataset {
            channels: 2,
            height: 1,
            width: 2,
            samples: vec![ImageSample {
                pixels: vec![1.0, 3.0, 10.0, 14.0],
                label: 0,
            }],
        };
        let mut held_out = ImageDataset {
            channels: 2,
            height: 1,
            width: 2,
            samples: vec![ImageSample {
                pixels: vec![2.0, 4.0, 12.0, 16.0],
                label: 1,
            }],
        };
        let fit = ChannelStandardizers::fit(&training)?;
        fit.transform(&mut training)?;
        fit.transform(&mut held_out)?;
        assert!((training.samples[0].pixels[0] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((held_out.samples[0].pixels[0] - 0.0).abs() < 1e-6);
        assert!(held_out.samples[0].pixels[3] > training.samples[0].pixels[3]);
        Ok(())
    }

    #[test]
    fn average_pool_uses_non_overlapping_windows() -> Result<()> {
        let sample = ImageSample {
            pixels: (0..16).map(|value| value as f32).collect(),
            label: 2,
        };
        let pooled = average_pool(&sample, 4, 4, 2)?;
        assert_eq!(pooled.label, 2);
        assert_eq!(pooled.pixels, [2.5, 4.5, 10.5, 12.5]);
        assert!(average_pool(&sample, 4, 4, 3).is_err());
        Ok(())
    }
}
