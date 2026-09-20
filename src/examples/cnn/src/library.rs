//! The retained CNN expressed through axis's named-axis algebra.
use axis::prelude::*;
use std::env;

const IMAGE: usize = 10;
const FILTER: usize = 3;
#[allow(dead_code)]
const SIDE: usize = IMAGE - FILTER + 1;
#[allow(dead_code)]
const SPATIAL: usize = SIDE * SIDE;
const CHANNELS: usize = 16;
const PATCH: usize = 16; // Retained baseline padding; the library uses the first nine rows.
const BATCH: usize = 16;

#[path = "reference.rs"]
#[cfg_attr(not(test), allow(dead_code))]
mod reference;

struct Classifier {
    conv: Conv2d,
    head: Linear,
    spatial: [Axis; 2],
    output: Axis,
}

impl Classifier {
    fn new(input: Axis, feature: Dim, spatial: [Axis; 2], output: Dim) -> Self {
        Self {
            conv: Conv2d::new(input, feature, spatial, [FILTER, FILTER]),
            head: Linear::new(feature.axis, output),
            spatial,
            output: output.axis,
        }
    }
    fn parts(&self, input: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let activation = self.conv.forward(input)?.relu()?;
        let pooled = activation.mean(self.spatial)?;
        let logits = self.head.forward(&pooled)?;
        Ok((activation, pooled, logits))
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
        let (_, _, logits) = self.parts(input)?;
        debug_assert!(logits.shape().axes().contains(&self.output));
        Ok(logits)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [("conv", &self.conv as &dyn Module), ("head", &self.head)]
            .into_iter()
            .flat_map(|(prefix, layer)| {
                layer
                    .named_parameters()
                    .into_iter()
                    .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
            })
            .collect()
    }
}

struct Axes {
    batch: Axis,
    input: Axis,
    height: Axis,
    width: Axis,
    feature: Axis,
    output: Axis,
}
impl Axes {
    fn new() -> Self {
        Self {
            batch: Axis::new("batch"),
            input: Axis::new("input_channel"),
            height: Axis::new("height"),
            width: Axis::new("width"),
            feature: Axis::new("feature"),
            output: Axis::new("output"),
        }
    }
    fn image_dims(&self, batch: usize) -> [Dim; 4] {
        [
            self.batch.of(batch),
            self.input.of(1),
            self.height.of(IMAGE),
            self.width.of(IMAGE),
        ]
    }
    fn label_dims(&self, batch: usize) -> [Dim; 2] {
        [self.batch.of(batch), self.output.of(1)]
    }
}

fn model(axes: &Axes) -> Classifier {
    Classifier::new(
        axes.input,
        axes.feature.of(CHANNELS),
        [axes.height, axes.width],
        axes.output.of(1),
    )
}

fn load(model: &Classifier, weights: &reference::Weights) -> Result<()> {
    let compact: Vec<_> = weights.conv_weight[..FILTER * FILTER * CHANNELS].to_vec();
    model.parameter("conv.weight")?.set_values(&compact)?;
    model
        .parameter("conv.bias")?
        .set_values(&weights.conv_bias)?;
    model
        .parameter("head.weight")?
        .set_values(&weights.head_weight)?;
    model.parameter("head.bias")?.set_values(&weights.head_bias)
}

fn metrics(logits: &Tensor, targets: &Tensor, batch: Axis, output: Axis) -> Result<(f32, f32)> {
    let values = logits.to_vec()?;
    let labels = targets.to_vec()?;
    let loss = logits
        .binary_cross_entropy_with_logits(targets)?
        .mean([batch, output])?
        .item()?;
    let correct = values
        .iter()
        .zip(labels)
        .filter(|(z, y)| (**z >= 0.0) == (*y == 1.0))
        .count();
    Ok((loss, 100.0 * correct as f32 / values.len() as f32))
}

fn main() -> Result<()> {
    let mut steps = 100;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => steps = args.next().ok_or("--steps requires a value")?.parse()?,
            "--smoke" => steps = 5,
            "--help" | "-h" => {
                println!(
                    "axis-cnn [--steps N | --smoke]\nNamed-axis valid CNN; deterministic stripes, seed 42, SGD 1.0."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if steps == 0 {
        return Err("steps must be positive".into());
    }
    let device = Device::cuda(0)?;
    let axes = Axes::new();
    let (train_x, train_y) = reference::dataset(BATCH, 1001);
    let (valid_x, valid_y) = reference::dataset(BATCH, 2002);
    let x = Tensor::from_slice(&train_x, axes.image_dims(BATCH), &device)?;
    let y = Tensor::from_slice(&train_y, axes.label_dims(BATCH), &device)?;
    let vx = Tensor::from_slice(&valid_x, axes.image_dims(BATCH), &device)?;
    let vy = Tensor::from_slice(&valid_y, axes.label_dims(BATCH), &device)?;
    let mut model = model(&axes);
    model.build(x.shape(), &device, 42)?;
    load(&model, &reference::Weights::new(42))?;
    let mut trainer = Trainer::new(SGD::new(1.0)?);
    let initial = metrics(&model.forward(&x)?, &y, axes.batch, axes.output)?;
    let initial_valid = metrics(&model.forward(&vx)?, &vy, axes.batch, axes.output)?;
    println!(
        "initial train_bce={:.2e} validation_bce={:.2e} validation_accuracy={:.2}",
        initial.0, initial_valid.0, initial_valid.1
    );
    for _ in 0..steps {
        let report = trainer.step(&mut model, |model| {
            model
                .forward(&x)?
                .binary_cross_entropy_with_logits(&y)?
                .mean([axes.batch, axes.output])
        })?;
        let step = report.step();
        if step == 1 || step % 20 == 0 || step == steps {
            println!(
                "step={step} pre_update_train_bce={:.2e}",
                report.pre_update_loss()?
            );
        }
    }
    let final_train = metrics(&model.forward(&x)?, &y, axes.batch, axes.output)?;
    let final_valid = metrics(&model.forward(&vx)?, &vy, axes.batch, axes.output)?;
    println!(
        "final train_bce={:.2e} validation_bce={:.2e} validation_accuracy={:.2}",
        final_train.0, final_valid.0, final_valid.1
    );
    if !final_train.0.is_finite()
        || !final_valid.0.is_finite()
        || final_train.0 >= initial.0
        || final_valid.0 >= initial_valid.0
    {
        return Err("CNN training did not improve both losses".into());
    }
    println!("PASS: named-axis CNN learned on cuTile");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(name: &str, actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        let max = actual
            .iter()
            .zip(expected)
            .enumerate()
            .fold(0.0_f64, |max, (i, (&a, &e))| {
                let error = (f64::from(a) - e).abs();
                assert!(
                    a.is_finite() && e.is_finite() && error < 3e-5 + 4e-4 * e.abs(),
                    "{name}[{i}]: {a} != {e}"
                );
                max.max(error)
            });
        println!(
            "check {name}: PASS {} elements max_abs_error={max:.2e}",
            actual.len()
        );
    }

    #[test]
    #[ignore = "requires CUDA; workspace check enables this"]
    fn cnn_matches_scalar_oracle_and_accumulates_overlapping_input_gradients() -> Result<()> {
        let device = Device::cuda(0)?;
        let axes = Axes::new();
        let weights = reference::Weights::new(7);
        let (images, labels) = reference::dataset(4, 11);
        let oracle = reference::forward_backward(&weights, &images, &labels);
        reference::finite_difference_check(&weights, &images, &labels, &oracle)?;
        let x = Tensor::from_slice(&images, axes.image_dims(4), &device)?
            .with_layout([axes.width, axes.batch, axes.input, axes.height])?
            .with_grad();
        let y = Tensor::from_slice(&labels, axes.label_dims(4), &device)?;
        let mut model = model(&axes);
        model.build(x.shape(), &device, 7)?;
        load(&model, &weights)?;
        let (activation, pooled, logits) = model.parts(&x)?;
        close("CNN activation", &activation.to_vec()?, &oracle.activation);
        close("CNN pooled", &pooled.to_vec()?, &oracle.pooled);
        close("CNN logits", &logits.to_vec()?, &oracle.logits);
        logits
            .binary_cross_entropy_with_logits(&y)?
            .mean([axes.batch, axes.output])?
            .backward()?;
        close(
            "CNN conv weight gradient",
            &model.parameter("conv.weight")?.grad().unwrap().to_vec()?,
            &oracle.dconv_weight[..FILTER * FILTER * CHANNELS],
        );
        close(
            "CNN conv bias gradient",
            &model.parameter("conv.bias")?.grad().unwrap().to_vec()?,
            &oracle.dconv_bias,
        );
        close(
            "CNN head weight gradient",
            &model.parameter("head.weight")?.grad().unwrap().to_vec()?,
            &oracle.dhead_weight,
        );
        close(
            "CNN head bias gradient",
            &model.parameter("head.bias")?.grad().unwrap().to_vec()?,
            &oracle.dhead_bias,
        );

        let mut dx = vec![0.0; images.len()];
        for n in 0..4 {
            let z = oracle.logits[n];
            let probability = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                z.exp() / (1.0 + z.exp())
            };
            let dz = (probability - f64::from(labels[n])) / 4.0;
            for oy in 0..SIDE {
                for ox in 0..SIDE {
                    for c in 0..CHANNELS {
                        if oracle.activation[((n * SIDE + oy) * SIDE + ox) * CHANNELS + c] <= 0.0 {
                            continue;
                        }
                        let da = dz * f64::from(weights.head_weight[c]) / SPATIAL as f64;
                        for ky in 0..FILTER {
                            for kx in 0..FILTER {
                                dx[(n * IMAGE + oy + ky) * IMAGE + ox + kx] += da
                                    * f64::from(
                                        weights.conv_weight[(ky * FILTER + kx) * CHANNELS + c],
                                    );
                            }
                        }
                    }
                }
            }
        }
        close(
            "CNN overlapping input gradient",
            &x.grad().unwrap().to_vec()?,
            &dx,
        );
        let sample = Axis::new("sample");
        let extreme = Tensor::from_slice(&[1000., -1000., 0., 2.], [sample.of(4)], &device)?
            .with_layout([sample])?
            .with_grad();
        let target = Tensor::from_slice(&[1., 0., 1., 0.], [sample.of(4)], &device)?;
        let stable = extreme.binary_cross_entropy_with_logits(&target)?;
        close(
            "stable BCE",
            &stable.to_vec()?,
            &[
                0.0,
                0.0,
                std::f64::consts::LN_2,
                2.0 + (-2.0_f64).exp().ln_1p(),
            ],
        );
        stable.mean(sample)?.backward()?;
        close(
            "stable BCE gradient",
            &extreme.grad().unwrap().to_vec()?,
            &[0.0, 0.0, -0.125, 0.25 / (1.0 + (-2.0_f64).exp())],
        );
        assert!(
            extreme
                .detach()
                .binary_cross_entropy_with_logits(&target.with_grad())
                .is_err()
        );
        let before = ["conv.weight", "conv.bias", "head.weight", "head.bias"]
            .map(|name| model.parameter(name).unwrap().tensor().to_vec().unwrap());
        let gradients = ["conv.weight", "conv.bias", "head.weight", "head.bias"].map(|name| {
            model
                .parameter(name)
                .unwrap()
                .grad()
                .unwrap()
                .to_vec()
                .unwrap()
        });
        SGD::new(0.1)?.step(&mut model)?;
        for (slot, (old, gradient)) in ["conv.weight", "conv.bias", "head.weight", "head.bias"]
            .into_iter()
            .zip(before.into_iter().zip(gradients))
        {
            let expected: Vec<_> = old
                .iter()
                .zip(gradient)
                .map(|(&v, g)| f64::from(v - 0.1 * g))
                .collect();
            close(
                &format!("CNN update {slot}"),
                &model.parameter(slot)?.tensor().to_vec()?,
                &expected,
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA; workspace check enables this"]
    fn mobilenet_style_depthwise_separable_block_composes() -> Result<()> {
        let device = Device::cuda(0)?;
        let (batch, channel, height, width, feature) = (
            Axis::new("batch"),
            Axis::new("channel"),
            Axis::new("height"),
            Axis::new("width"),
            Axis::new("feature"),
        );
        let values: Vec<_> = (1..=9)
            .map(|value| value as f32)
            .chain((1..=9).map(|value| -(value as f32)))
            .collect();
        let input = Tensor::from_slice(
            &values,
            [batch.of(1), channel.of(2), height.of(3), width.of(3)],
            &device,
        )?
        .with_grad();
        let mut block = Sequential::new((
            Conv2d::new(channel, channel.of(2), [height, width], [3, 3])
                .padding([1, 1])
                .groups(2),
            ReLU,
            Conv2d::new(channel, feature.of(3), [height, width], [1, 1]),
        ));
        assert_eq!(
            block.build(input.shape(), &device, 31)?,
            Shape::new([batch.of(1), height.of(3), width.of(3), feature.of(3),])?
        );
        let mut depthwise = vec![0.0; 18];
        depthwise[4] = 1.0;
        depthwise[9 + 4] = -1.0;
        block.parameter("0.weight")?.set_values(&depthwise)?;
        block.parameter("0.bias")?.set_values(&[0.0, 0.0])?;
        block
            .parameter("2.weight")?
            .set_values(&[1.0, 0.5, -1.0, 0.25, -0.5, 2.0])?;
        block.parameter("2.bias")?.set_values(&[0.1, -0.2, 0.3])?;

        let output = block.forward(&input)?;
        let expected: Vec<_> = (1..=9)
            .flat_map(|value| {
                let value = f64::from(value);
                [1.25 * value + 0.1, -0.2, value + 0.3]
            })
            .collect();
        close("depthwise-separable forward", &output.to_vec()?, &expected);
        output.mean([batch, height, width, feature])?.backward()?;
        assert!(input.grad().is_some());
        for name in ["0.weight", "0.bias", "2.weight", "2.bias"] {
            let gradient = block.parameter(name)?.grad().unwrap().to_vec()?;
            assert!(
                gradient.iter().all(|value| value.is_finite()),
                "non-finite gradient for {name}"
            );
        }
        println!("check MobileNet-style depthwise-separable block: PASS");
        Ok(())
    }
}
