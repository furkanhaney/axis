use crate::architectures::{DCGAN_IMAGE_CHANNELS, DCGAN_LATENT};
use crate::prelude::*;
use crate::resnet::{BasicBlock, Bottleneck};

fn close(name: &str, actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let mut max_error = 0.0_f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - e).abs();
        assert!(
            a.is_finite() && e.is_finite() && error < 3e-6 + 3e-5 * e.abs(),
            "{name}[{i}]: {a} != {e}"
        );
        max_error = max_error.max(error);
    }
    println!(
        "check {name}: PASS {} elements max_abs_error={max_error:.2e}",
        actual.len()
    );
}

fn central_difference(values: &[f64], epsilon: f64, evaluate: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            let mut plus = values.to_vec();
            let mut minus = values.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
        })
        .collect()
}

fn mean_squared(actual: &[f64], target: &[f64]) -> f64 {
    actual
        .iter()
        .zip(target)
        .map(|(actual, target)| (actual - target).powi(2))
        .sum::<f64>()
        / actual.len() as f64
}

#[test]
fn neural_module_shapes_reject_invalid_architectures_before_allocation() -> Result<()> {
    let (batch, channel, depth, height, width, output, population) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
        Axis::new("population"),
    );
    let image = Shape::new([batch.of(2), channel.of(3), height.of(5), width.of(7)])?;

    let conv = Conv2d::new(channel, output.of(4), [height, width], [3, 2]);
    assert_eq!(
        conv.output_shape(&image)?,
        Shape::new([batch.of(2), height.of(3), width.of(6), output.of(4)])?
    );
    let configured = Conv2d::new(channel, output.of(6), [height, width], [3, 2])
        .stride([2, 3])
        .padding([1, 0])
        .groups(3);
    assert_eq!(
        configured.output_shape(&image)?,
        Shape::new([batch.of(2), height.of(3), width.of(2), output.of(6)])?
    );
    let depthwise = Conv2d::new(channel, channel.of(3), [height, width], [3, 3])
        .padding([1, 1])
        .groups(3);
    assert_eq!(
        depthwise.output_shape(&image)?,
        Shape::new([batch.of(2), height.of(5), width.of(7), channel.of(3)])?
    );
    let wide_padding = Conv2d::new(channel, output.of(4), [height, width], [1, 1]).padding([2, 3]);
    assert_eq!(
        wide_padding.output_shape(&image)?,
        Shape::new([batch.of(2), height.of(9), width.of(13), output.of(4)])?
    );
    for invalid in [
        Conv2d::new(channel, output.of(4), [height, height], [3, 2]),
        Conv2d::new(channel, output.of(4), [channel, width], [3, 2]),
        Conv2d::new(channel, output.of(4), [height, width], [0, 2]),
        Conv2d::new(channel, output.of(4), [height, width], [6, 2]),
        Conv2d::new(channel, output.of(4), [height, width], [3, 2]).stride([0, 1]),
        Conv2d::new(channel, output.of(4), [height, width], [3, 2]).groups(0),
        Conv2d::new(channel, output.of(4), [height, width], [3, 2]).groups(2),
        Conv2d::new(channel, output.of(4), [height, width], [3, 2]).groups(3),
    ] {
        assert!(invalid.output_shape(&image).is_err());
    }

    let volume = Shape::new([
        batch.of(2),
        channel.of(4),
        depth.of(3),
        height.of(5),
        width.of(7),
    ])?;
    let conv3d = Conv3d::new(channel, output.of(6), [depth, height, width], [2, 3, 2])
        .stride([1, 2, 3])
        .padding([1, 1, 0])
        .groups(2);
    assert_eq!(
        conv3d.output_shape(&volume)?,
        Shape::new([
            batch.of(2),
            depth.of(4),
            height.of(3),
            width.of(2),
            output.of(6),
        ])?
    );
    assert!(
        Conv3d::new(channel, output.of(6), [depth, height, height], [2, 3, 2],)
            .output_shape(&volume)
            .is_err()
    );
    assert!(
        Conv3d::new(channel, output.of(6), [depth, height, width], [2, 3, 2],)
            .stride([1, 0, 1])
            .output_shape(&volume)
            .is_err()
    );
    assert!(
        Conv3d::new(channel, output.of(5), [depth, height, width], [2, 3, 2],)
            .groups(2)
            .output_shape(&volume)
            .is_err()
    );

    let linear = Linear::new(channel, output.of(4));
    assert!(linear.output_shape(&Shape::new([batch.of(2)])?).is_err());
    let population_linear = PopulationLinear::new(population.of(3), channel, output.of(4));
    assert!(
        population_linear
            .output_shape(&Shape::new(
                [batch.of(2), population.of(2), channel.of(3),]
            )?)
            .is_err()
    );
    assert!(LayerNorm::new(channel)?.epsilon(0.0).is_err());
    assert!(LayerNorm::new(channel)?.epsilon(f32::INFINITY).is_err());
    assert!(LayerNorm::new([channel, channel]).is_err());
    assert!(LayerNorm::new([]).is_err());
    assert!(RmsNorm::new(channel)?.epsilon(f32::NAN).is_err());
    assert!(GroupNorm::new(channel, 0, [height, width]).is_err());
    assert!(GroupNorm::new(channel, 2, [channel, width]).is_err());
    assert!(
        GroupNorm::new(channel, 1, [height, width])?
            .epsilon(-1.0)
            .is_err()
    );
    assert!(InstanceNorm::new(channel, []).is_err());
    assert!(InstanceNorm::new(channel, [channel]).is_err());
    assert!(
        InstanceNorm::new(channel, [height, width])?
            .epsilon(0.0)
            .is_err()
    );

    let layer = LayerNorm::new([height, width])?;
    assert_eq!(layer.output_shape(&image)?, image);
    let rms = RmsNorm::new(Vec::from([height, width]))?;
    assert_eq!(rms.output_shape(&image)?, image);
    let group = GroupNorm::new(channel, 3, [height, width])?;
    assert_eq!(group.output_shape(&image)?, image);
    assert!(
        GroupNorm::new(channel, 2, [height, width])?
            .output_shape(&image)
            .is_err()
    );
    let instance = InstanceNorm::new(channel, &[height, width][..])?;
    assert_eq!(instance.output_shape(&image)?, image);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn recurrent_tensor_primitives_match_scalar_values_and_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, feature) = (Axis::new("batch"), Axis::new("time"), Axis::new("feature"));
    let values: Vec<_> = (0..12).map(|value| value as f32 - 5.0).collect();
    let sequence = Tensor::from_slice(&values, [batch.of(2), time.of(3), feature.of(2)], &device)?
        .with_layout([feature, time, batch])?
        .with_grad();
    let middle = sequence.select(time, 1)?;
    close("named select", &middle.to_vec()?, &[-3.0, -2.0, 3.0, 4.0]);
    middle.mean([batch, feature])?.backward()?;
    close(
        "named select gradient",
        &sequence.grad().unwrap().to_vec()?,
        &[
            0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0,
        ],
    );

    const LARGE_BATCH: usize = 7;
    const LARGE_TIME: usize = 3;
    const LARGE_FEATURE: usize = 23;
    let large_values: Vec<_> = (0..LARGE_BATCH * LARGE_TIME * LARGE_FEATURE)
        .map(|value| (value as f32 - 200.0) / 17.0)
        .collect();
    let large = Tensor::from_slice(
        &large_values,
        [
            batch.of(LARGE_BATCH),
            time.of(LARGE_TIME),
            feature.of(LARGE_FEATURE),
        ],
        &device,
    )?
    .with_layout([feature, batch, time])?
    .with_grad();
    let large_middle = large.select(time, 1)?;
    let mut large_expected = Vec::with_capacity(LARGE_BATCH * LARGE_FEATURE);
    for batch_index in 0..LARGE_BATCH {
        let start = (batch_index * LARGE_TIME + 1) * LARGE_FEATURE;
        large_expected.extend(
            large_values[start..start + LARGE_FEATURE]
                .iter()
                .copied()
                .map(f64::from),
        );
    }
    close(
        "multi-tile named select",
        &large_middle.to_vec()?,
        &large_expected,
    );
    large_middle.mean([batch, feature])?.backward()?;
    let selected_gradient = 1.0 / (LARGE_BATCH * LARGE_FEATURE) as f64;
    let large_gradient: Vec<_> = (0..large_values.len())
        .map(|index| {
            let time_coordinate = (index / LARGE_FEATURE) % LARGE_TIME;
            if time_coordinate == 1 {
                selected_gradient
            } else {
                0.0
            }
        })
        .collect();
    close(
        "multi-tile named select gradient",
        &large.grad().unwrap().to_vec()?,
        &large_gradient,
    );

    let slices: Vec<_> = (0..3)
        .map(|step| {
            let tensor = Tensor::from_slice(
                &values[step * 4..(step + 1) * 4],
                [batch.of(2), feature.of(2)],
                &device,
            )?;
            let tensor = if step % 2 == 0 {
                tensor.with_layout([feature, batch])?
            } else {
                tensor
            };
            Ok(tensor.with_grad())
        })
        .collect::<Result<_>>()?;
    let stacked = Tensor::stack(&slices, time, 1)?;
    assert_eq!(
        stacked.shape(),
        &Shape::new([batch.of(2), time.of(3), feature.of(2)])?
    );
    close(
        "named stack",
        &stacked.to_vec()?,
        &[
            -5.0, -4.0, -1.0, 0.0, 3.0, 4.0, -3.0, -2.0, 1.0, 2.0, 5.0, 6.0,
        ],
    );
    stacked.mean([batch, time, feature])?.backward()?;
    for (index, slice) in slices.iter().enumerate() {
        close(
            &format!("stack gradient {index}"),
            &slice.grad().unwrap().to_vec()?,
            &[1.0 / 12.0; 4],
        );
    }

    let activation_input =
        Tensor::from_slice(&[-20.0, -1.0, 0.0, 2.0, 20.0], [feature.of(5)], &device)?.with_grad();
    let activation = activation_input.sigmoid()?.add(&activation_input.tanh()?)?;
    let expected: Vec<_> = [-20.0_f64, -1.0, 0.0, 2.0, 20.0]
        .into_iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()) + x.tanh())
        .collect();
    close("sigmoid plus tanh", &activation.to_vec()?, &expected);
    activation.mean(feature)?.backward()?;
    let expected_gradient: Vec<_> = [-20.0_f64, -1.0, 0.0, 2.0, 20.0]
        .into_iter()
        .map(|x| {
            let probability = 1.0 / (1.0 + (-x).exp());
            (probability * (1.0 - probability) + 1.0 - x.tanh().powi(2)) / 5.0
        })
        .collect();
    close(
        "sigmoid plus tanh gradient",
        &activation_input.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    assert!(sequence.select(time, 3).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_minimum_preserves_axes_across_layouts_and_routes_ties() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate, time) = (
        Axis::new("batch"),
        Axis::new("candidate"),
        Axis::new("time"),
    );
    let values = Tensor::from_slice(
        &[3.0, 9.0, 1.0, 8.0, 1.0, 7.0, 4.0, 0.0, -2.0, 5.0, 6.0, -1.0],
        [batch.of(2), candidate.of(3), time.of(2)],
        &device,
    )?
    .with_layout([candidate, time, batch])?
    .with_grad();

    let minimum = values.min(candidate)?;
    assert_eq!(minimum.shape(), &Shape::new([batch.of(2), time.of(2)])?);
    close(
        "named-axis minimum",
        &minimum.to_vec()?,
        &[1.0, 7.0, -2.0, -1.0],
    );
    minimum.mean([batch, time])?.backward()?;
    close(
        "named-axis minimum gradient",
        &values.grad().unwrap().to_vec()?,
        &[
            0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25,
        ],
    );
    let error = values.min(Axis::new("missing")).err().unwrap().to_string();
    assert!(error.contains("missing axis missing#"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_minimum_ignores_nonfinite_values_and_marks_empty_groups() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate) = (Axis::new("batch"), Axis::new("candidate"));
    let values = Tensor::from_slice(
        &[
            f32::NAN,
            f32::INFINITY,
            3.0,
            f32::NEG_INFINITY,
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::INFINITY,
        ],
        [batch.of(2), candidate.of(5)],
        &device,
    )?
    .with_grad();

    let minimum = values.min(candidate)?;
    let actual = minimum.to_vec()?;
    assert_eq!(actual[0], 1.0);
    assert!(actual[1].is_nan());

    minimum.mean(batch)?.backward()?;
    close(
        "non-finite minimum gradient",
        &values.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    );

    let all_nonfinite = Tensor::from_slice(
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        [batch.of(1), candidate.of(3)],
        &device,
    )?
    .with_grad();
    let zero = Tensor::from_slice(&[0.0], [batch.of(1)], &device)?;
    let loss = all_nonfinite
        .min(candidate)?
        .squared_error(&zero)?
        .mean(batch)?;
    assert!(loss.to_vec()?[0].is_nan());
    loss.backward()?;
    close(
        "all-nonfinite minimum with NaN cotangent",
        &all_nonfinite.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn odd_features_short_batch_unrelated_axes_and_storage_order() -> Result<()> {
    let device = Device::cuda(0)?;
    let (b, t, f, o) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("feature"),
        Axis::new("output"),
    );
    let values: Vec<_> = (0..3 * 2 * 17)
        .map(|i| ((i * 7 % 31) as f32 - 15.0) / 16.0)
        .collect();
    let weights: Vec<_> = (0..17 * 5)
        .map(|i| ((i * 3 % 19) as f32 - 9.0) / 20.0)
        .collect();
    let biases = [0.1, -0.2, 0.3, -0.4, 0.5];
    let x = Tensor::from_slice(&values, [b.of(3), t.of(2), f.of(17)], &device)?
        .with_layout([f, b, t])?
        .with_grad();
    let mut model = Linear::new(f, o.of(5));
    // Bind using a larger batch and without time. Neither is part of Linear's feature binding.
    model.build(&Shape::new([b.of(256), f.of(17)])?, &device, 5)?;
    let params = model.parameters();
    params[0].set_values(&weights)?;
    params[1].set_values(&biases)?;
    let prediction = model.forward(&x)?;
    assert_eq!(
        prediction.shape(),
        &Shape::new([b.of(3), t.of(2), o.of(5)])?
    );
    let mut expected = vec![0.0; 6 * 5];
    let mut dx = vec![0.0; values.len()];
    let mut dw = vec![0.0; weights.len()];
    let mut db = vec![0.0; 5];
    for r in 0..6 {
        for j in 0..5 {
            let p = f64::from(biases[j])
                + (0..17)
                    .map(|i| f64::from(values[r * 17 + i]) * f64::from(weights[i * 5 + j]))
                    .sum::<f64>();
            expected[r * 5 + j] = p;
            let dy = 2.0 * p / 30.0;
            db[j] += dy;
            for i in 0..17 {
                dx[r * 17 + i] += dy * f64::from(weights[i * 5 + j]);
                dw[i * 5 + j] += dy * f64::from(values[r * 17 + i]);
            }
        }
    }
    close("odd/layout predictions", &prediction.to_vec()?, &expected);
    let canonical = Tensor::from_slice(&values, [b.of(3), t.of(2), f.of(17)], &device)?;
    close(
        "canonical predictions",
        &model.forward(&canonical)?.to_vec()?,
        &expected,
    );
    let mut leading_values = vec![0.0; values.len()];
    for r in 0..6 {
        for i in 0..17 {
            leading_values[i * 6 + r] = values[r * 17 + i];
        }
    }
    let leading = Tensor::from_slice(&leading_values, [f.of(17), b.of(3), t.of(2)], &device)?;
    close(
        "leading logical feature axis",
        &model.forward(&leading)?.to_vec()?,
        &expected,
    );
    prediction.mul(&prediction)?.mean([t, o, b])?.backward()?;
    close(
        "odd/layout input gradients",
        &x.grad().unwrap().to_vec()?,
        &dx,
    );
    close(
        "odd/layout weight gradients",
        &params[0].grad().unwrap().to_vec()?,
        &dw,
    );
    close(
        "odd/layout bias gradients",
        &params[1].grad().unwrap().to_vec()?,
        &db,
    );
    assert!(
        model
            .forward(&Tensor::from_slice(&[0.0; 18], [f.of(18)], &device)?)
            .is_err()
    );
    // Public input/output identity may be the same while their extents differ.
    let mut same = Linear::new(f, f.of(5));
    same.build(x.shape(), &device, 1)?;
    same.parameters()[0].set_values(&weights)?;
    same.parameters()[1].set_values(&biases)?;
    let out = same.forward(&x)?;
    assert_eq!(out.extent(f)?, 5);
    close("same-axis Linear", &out.to_vec()?, &expected);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn bias_disabled_linear_has_only_a_weight_parameter_and_matches_a_hand_computed_oracle()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature, output) = (
        Axis::new("batch"),
        Axis::new("feature"),
        Axis::new("output"),
    );
    let values = [1.0, -2.0, 0.5, 3.0, -0.25, 1.5]; // [batch=3, feature=2]
    let weights = [0.5, -1.0, 2.0, 0.25]; // [feature=2, output=2]
    let x = Tensor::from_slice(&values, [batch.of(3), feature.of(2)], &device)?.with_grad();
    let mut model = Linear::new(feature, output.of(2)).bias(false);
    model.build(&Shape::new([batch.of(3), feature.of(2)])?, &device, 3)?;

    let named = model.named_parameters();
    assert_eq!(
        named.len(),
        1,
        "a bias-disabled Linear must expose only its weight"
    );
    assert_eq!(named[0].0, "weight");
    let weight = model.parameters()[0].clone();
    assert_eq!(
        weight.tensor().shape().len(),
        2 * 2,
        "parameter count must be exactly in * out, with no + out for a bias"
    );
    weight.set_values(&weights)?;

    let prediction = model.forward(&x)?;
    let mut expected = vec![0.0_f64; 3 * 2];
    for r in 0..3 {
        for j in 0..2 {
            expected[r * 2 + j] = (0..2)
                .map(|i| f64::from(values[r * 2 + i]) * f64::from(weights[i * 2 + j]))
                .sum();
        }
    }
    close("bias-disabled forward", &prediction.to_vec()?, &expected);

    prediction
        .mul(&prediction)?
        .mean([batch, output])?
        .backward()?;
    let mut dx = vec![0.0_f64; values.len()];
    let mut dw = vec![0.0_f64; weights.len()];
    for r in 0..3 {
        for j in 0..2 {
            let dy = 2.0 * expected[r * 2 + j] / 6.0;
            for i in 0..2 {
                dx[r * 2 + i] += dy * f64::from(weights[i * 2 + j]);
                dw[i * 2 + j] += dy * f64::from(values[r * 2 + i]);
            }
        }
    }
    close(
        "bias-disabled input gradients",
        &x.grad().unwrap().to_vec()?,
        &dx,
    );
    close(
        "bias-disabled weight gradients",
        &weight.grad().unwrap().to_vec()?,
        &dw,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn shared_axes_contract_and_both_input_derivatives() -> Result<()> {
    let device = Device::cuda(0)?;
    let (b, h, f, time) = (
        Axis::new("batch"),
        Axis::new("head"),
        Axis::new("feature"),
        Axis::new("time"),
    );
    let q = time.role("query");
    let k = time.role("key");
    let av: Vec<_> = (0..2 * 3 * 2 * 5)
        .map(|i| (i as f32 - 25.0) / 30.0)
        .collect();
    let bv: Vec<_> = (0..2 * 4 * 2 * 5)
        .map(|i| (i as f32 - 35.0) / 40.0)
        .collect();
    let a = Tensor::from_slice(&av, [b.of(2), q.of(3), h.of(2), f.of(5)], &device)?
        .with_layout([f, b, h, q])?
        .with_grad();
    let rhs = Tensor::from_slice(&bv, [b.of(2), k.of(4), h.of(2), f.of(5)], &device)?
        .with_layout([h, f, k, b])?
        .with_grad();
    let out = a.contract(&rhs, f)?;
    assert_eq!(
        out.shape(),
        &Shape::new([b.of(2), q.of(3), h.of(2), k.of(4)])?
    );
    let mut expected = vec![];
    let mut da = vec![0.0; av.len()];
    let mut db = vec![0.0; bv.len()];
    for batch in 0..2 {
        for query in 0..3 {
            for head in 0..2 {
                for key in 0..4 {
                    let mut value = 0.0;
                    for feature in 0..5 {
                        let ai = ((batch * 3 + query) * 2 + head) * 5 + feature;
                        let bi = ((batch * 4 + key) * 2 + head) * 5 + feature;
                        value += f64::from(av[ai]) * f64::from(bv[bi]);
                        da[ai] += f64::from(bv[bi]) / 48.0;
                        db[bi] += f64::from(av[ai]) / 48.0;
                    }
                    expected.push(value);
                }
            }
        }
    }
    close("shared-axis contraction", &out.to_vec()?, &expected);
    out.mean([q, k, b, h])?.backward()?;
    close(
        "contraction left gradient",
        &a.grad().unwrap().to_vec()?,
        &da,
    );
    close(
        "contraction right gradient",
        &rhs.grad().unwrap().to_vec()?,
        &db,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multiple_axis_contraction_exceeds_plan_limit_with_both_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, row, a, b, column) = (
        Axis::new("batch"),
        Axis::new("row"),
        Axis::new("a"),
        Axis::new("b"),
        Axis::new("column"),
    );
    const B: usize = 2;
    const M: usize = 129;
    const A: usize = 16;
    const K: usize = 16;
    const N: usize = 257;
    const { assert!(B * M * A * K * N > 16_777_216) };
    let lv: Vec<f32> = (0..B * M * A * K).map(|i| (i % 19) as f32 - 9.0).collect();
    let rv: Vec<f32> = (0..B * A * K * N).map(|i| (i % 17) as f32 - 8.0).collect();
    let weights: Vec<f32> = (0..B * M * N).map(|i| (i % 5) as f32 - 2.0).collect();
    let mut expected = vec![0.0_f64; B * M * N];
    let mut dl = vec![0.0_f64; lv.len()];
    let mut dr = vec![0.0_f64; rv.len()];
    // Independent scalar oracle, including a nonuniform upstream derivative.
    for batch_index in 0..B {
        for m in 0..M {
            for n in 0..N {
                let o = (batch_index * M + m) * N + n;
                for aa in 0..A {
                    for bb in 0..K {
                        let l = ((batch_index * M + m) * A + aa) * K + bb;
                        let r = ((batch_index * A + aa) * K + bb) * N + n;
                        expected[o] += f64::from(lv[l]) * f64::from(rv[r]);
                        dl[l] += f64::from(weights[o]) * f64::from(rv[r]);
                        dr[r] += f64::from(weights[o]) * f64::from(lv[l]);
                    }
                }
            }
        }
    }
    for reordered in [false, true] {
        let mut left =
            Tensor::from_slice(&lv, [batch.of(B), row.of(M), a.of(A), b.of(K)], &device)?;
        let mut right =
            Tensor::from_slice(&rv, [batch.of(B), a.of(A), b.of(K), column.of(N)], &device)?;
        if reordered {
            left = left.with_layout([b, row, batch, a])?;
            right = right.with_layout([column, b, a, batch])?;
        }
        let left = left.with_grad();
        let right = right.with_grad();
        let output = left.contract(&right, if reordered { [b, a] } else { [a, b] })?;
        assert_eq!(
            output.shape(),
            &Shape::new([batch.of(B), row.of(M), column.of(N)])?
        );
        close("large multi-axis contraction", &output.to_vec()?, &expected);
        let weight = Tensor::from_slice(&weights, output.shape().dims().iter().copied(), &device)?;
        output.mul(&weight)?.sum([batch, row, column])?.backward()?;
        close(
            "large multi-axis left gradient",
            &left.grad().unwrap().to_vec()?,
            &dl,
        );
        close(
            "large multi-axis right gradient",
            &right.grad().unwrap().to_vec()?,
            &dr,
        );
        device.synchronize()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multiple_axis_contraction_matches_dense_reference_and_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, inner_a, inner_b, column) = (
        Axis::new("row"),
        Axis::new("inner_a"),
        Axis::new("inner_b"),
        Axis::new("column"),
    );
    let left = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [row.of(2), inner_a.of(2), inner_b.of(2)],
        &device,
    )?
    .with_layout([inner_b, row, inner_a])?
    .with_grad();
    let right = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [inner_a.of(2), inner_b.of(2), column.of(2)],
        &device,
    )?
    .with_layout([column, inner_a, inner_b])?
    .with_grad();

    let output = left.contract(&right, [inner_a, inner_b])?;
    assert_eq!(output.shape(), &Shape::new([row.of(2), column.of(2)])?);
    close(
        "multiple-axis contraction",
        &output.to_vec()?,
        &[50.0, 60.0, 114.0, 140.0],
    );
    output.mean([row, column])?.backward()?;
    close(
        "multiple-axis left gradient",
        &left.grad().unwrap().to_vec()?,
        &[0.75, 1.75, 2.75, 3.75, 0.75, 1.75, 2.75, 3.75],
    );
    close(
        "multiple-axis right gradient",
        &right.grad().unwrap().to_vec()?,
        &[1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn bf16_matrix_products_keep_fp32_state_and_gradients_close() -> Result<()> {
    let m = Axis::new("row");
    let k = Axis::new("reduction");
    let n = Axis::new("column");
    let left_values: Vec<_> = (0..5 * 16).map(|i| (i as f32 - 31.0) / 41.0).collect();
    let right_values: Vec<_> = (0..16 * 7).map(|i| (i as f32 - 47.0) / 53.0).collect();
    let run = |device: &Device| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let left = Tensor::from_slice(&left_values, [m.of(5), k.of(16)], device)?.with_grad();
        let right = Tensor::from_slice(&right_values, [k.of(16), n.of(7)], device)?.with_grad();
        let output = left.contract(&right, k)?;
        let values = output.to_vec()?;
        output.mean([m, n])?.backward()?;
        Ok((
            values,
            left.grad().unwrap().to_vec()?,
            right.grad().unwrap().to_vec()?,
        ))
    };
    let reference = run(&Device::cuda(0)?)?;
    let mixed = run(&Device::cuda_bf16(0)?)?;
    for (name, actual, expected) in [
        ("BF16 matrix product", &mixed.0, &reference.0),
        ("BF16 left derivative", &mixed.1, &reference.1),
        ("BF16 right derivative", &mixed.2, &reference.2),
    ] {
        let maximum = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(maximum < 2e-2, "{name} max error {maximum}");
        println!("check {name}: PASS max_abs_error={maximum:.2e}");
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn compact_merge_preserves_selected_axis_order_across_repeated_calls() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, head, depth, feature) = (
        Axis::new("batch"),
        Axis::new("head"),
        Axis::new("depth"),
        Axis::new("feature"),
    );
    let input = Tensor::from_slice(
        &(0..24).map(|value| value as f32).collect::<Vec<_>>(),
        [batch.of(2), head.of(3), depth.of(4)],
        &device,
    )?
    .with_layout([depth, batch, head])?;

    let expected_forward: Vec<_> = (0..24).map(|value| value as f32).collect();
    for _ in 0..4 {
        assert_eq!(
            input.merge([head, depth], feature)?.to_vec()?,
            expected_forward
        );
    }

    let mut expected_reversed = vec![];
    for batch_index in 0..2 {
        for depth_index in 0..4 {
            for head_index in 0..3 {
                expected_reversed.push((batch_index * 12 + head_index * 4 + depth_index) as f32);
            }
        }
    }
    for _ in 0..2 {
        assert_eq!(
            input.merge([depth, head], feature)?.to_vec()?,
            expected_reversed
        );
    }
    assert!(Tensor::layout_metadata_max() <= 9);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn algebra_and_errors_are_explicit() -> Result<()> {
    let device = Device::cuda(0)?;
    let (b, t, f, h, d) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("feature"),
        Axis::new("head"),
        Axis::new("depth"),
    );
    let x = Tensor::from_slice(
        &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
        [b.of(2), f.of(12)],
        &device,
    )?
    .with_grad();
    let split = x.with_layout([f, b])?.split(f, [h.of(3), d.of(4)])?;
    let merged = split.with_layout([d, b, h])?.merge([h, d], f)?;
    close(
        "split/merge values",
        &merged.to_vec()?,
        &(0..24).map(f64::from).collect::<Vec<_>>(),
    );
    merged.mean([f, b])?.backward()?;
    close(
        "split/merge gradient",
        &x.grad().unwrap().to_vec()?,
        &[1.0 / 24.0; 24],
    );
    x.zero_grad();
    // Use a fresh graph and nonuniform adjoints to check the inverse mapping.
    let reversed = x
        .split(f, [h.of(3), d.of(4)])?
        .with_layout([d, b, h])?
        .merge([d, h], f)?;
    let mut expected = vec![];
    let mut gradient = vec![0.0; 24];
    for batch in 0..2 {
        for depth in 0..4 {
            for head in 0..3 {
                let source = batch * 12 + head * 4 + depth;
                let target = batch * 12 + depth * 3 + head;
                expected.push(source as f64);
                gradient[source] = target as f64 / 24.0;
            }
        }
    }
    close("reordered merge", &reversed.to_vec()?, &expected);
    reversed.mul(&x.detach())?.mean([b, f])?.backward()?;
    close(
        "reordered merge gradient",
        &x.grad().unwrap().to_vec()?,
        &gradient,
    );
    let a = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.], [b.of(2), t.of(3)], &device)?.with_grad();
    let bias = Tensor::from_slice(&[10., 20., 30.], [t.of(3)], &device)?.with_grad();
    let reordered = Tensor::from_slice(&[1., 4., 2., 5., 3., 6.], [t.of(3), b.of(2)], &device)?;
    close(
        "logical-axis alignment",
        &a.sub(&reordered)?.to_vec()?,
        &[0.0; 6],
    );
    a.add(&bias)?.mean([t, b])?.backward()?;
    close(
        "broadcast input gradient",
        &a.grad().unwrap().to_vec()?,
        &[1.0 / 6.0; 6],
    );
    close(
        "broadcast bias reduction",
        &bias.grad().unwrap().to_vec()?,
        &[1.0 / 3.0; 3],
    );
    let scalar = Tensor::from_slice(&[2.0], [], &device)?.with_grad();
    scalar.mul(&a.detach())?.mean([b, t])?.backward()?;
    close(
        "scalar broadcast gradient",
        &scalar.grad().unwrap().to_vec()?,
        &[3.5],
    );
    let other = Tensor::from_slice(&[2.0, 3.0], [f.of(2)], &device)?;
    assert!(a.add(&other).is_err());
    assert_eq!(
        bias.outer(&other)?.shape(),
        &Shape::new([t.of(3), f.of(2)])?
    );
    assert!(a.mean(f).is_err());
    assert!(a.mean([b, b]).is_err());
    assert!(a.backward().is_err());
    assert!(a.rename(t, b).is_err());
    assert!(a.with_layout([b, b]).is_err());
    assert!(a.split(t, [f.of(2)]).is_err());
    assert!(a.merge([b, t], f)?.extent(f)? == 6);
    assert!(
        a.add(&Tensor::from_slice(&[1., 2.], [t.of(2)], &device)?)
            .is_err()
    );
    assert!(
        a.add(&Tensor::from_slice(
            &[1.; 6],
            [b.of(2), Axis::new("time").of(3)],
            &device
        )?)
        .is_err()
    );
    assert!(a.squared_error(&bias).is_err());
    assert!(a.contract(&other, t).is_err());
    assert!(Tensor::from_slice(&[1.0], [b.of(2)], &device).is_err());
    assert!(SGD::new(f32::NAN).is_err());
    let relu_input = Tensor::from_slice(&[-1.0, 0.0, 1.0], [f.of(3)], &device)?.with_grad();
    relu_input.relu()?.mean(f)?.backward()?;
    close(
        "ReLU derivative including zero",
        &relu_input.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 1.0 / 3.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn binary_cross_entropy_is_stable_and_differentiable() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("feature");
    let values = [-1000.0_f32, -2.0, 0.0, 2.0, 1000.0];
    let target_values = [0.0_f32, 1.0, 0.0, 1.0, 1.0];
    let logits = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    let targets = Tensor::from_slice(&target_values, [feature.of(target_values.len())], &device)?;

    let losses = logits.binary_cross_entropy_with_logits(&targets)?;
    let expected = values
        .iter()
        .zip(target_values)
        .map(|(&logit, target)| {
            let logit = f64::from(logit);
            logit.max(0.0) - logit * f64::from(target) + (-logit.abs()).exp().ln_1p()
        })
        .collect::<Vec<_>>();
    close("stable binary cross-entropy", &losses.to_vec()?, &expected);

    losses.mean(feature)?.backward()?;
    let expected_gradient = values
        .iter()
        .zip(target_values)
        .map(|(&logit, target)| {
            let sigmoid = 1.0 / (1.0 + (-f64::from(logit)).exp());
            (sigmoid - f64::from(target)) / values.len() as f64
        })
        .collect::<Vec<_>>();
    close(
        "binary cross-entropy gradient",
        &logits.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    assert!(
        logits
            .detach()
            .binary_cross_entropy_with_logits(&targets.with_grad())
            .is_err()
    );
    let other = Axis::new("other");
    let wrong_axes = Tensor::from_slice(&target_values, [other.of(5)], &device)?;
    assert!(
        logits
            .detach()
            .binary_cross_entropy_with_logits(&wrong_axes)
            .is_err()
    );
    assert!(logits.item().is_err());
    assert!(logits.scale(f32::NAN).is_err());
    assert!(logits.inverse_sqrt(0.0).is_err());
    Ok(())
}

// Independent derivation for `binary_cross_entropy_with_logits_weighted`, matching
// `torch.nn.BCEWithLogitsLoss(pos_weight=p)`:
//   loss = -[p*y*log(sigma(x)) + (1-y)*log(1-sigma(x))]
// Using log(sigma(x)) = x - softplus(x) and log(1-sigma(x)) = -softplus(x):
//   loss = -p*y*x + softplus(x)*(p*y + 1 - y) = log_weight*softplus(x) - p*y*x
// where log_weight = 1 + (p-1)*y and softplus(x) is the same stable
// max(x,0) + log(1+exp(-|x|)) form the unweighted kernel already uses (log_weight == 1
// at p == 1, recovering the unweighted formula exactly — the shared oracle case).
//   d(loss)/dx = log_weight*sigma(x) - p*y
#[test]
#[ignore = "requires CUDA"]
fn binary_cross_entropy_weighted_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("feature");
    let values = [-1000.0_f32, -2.0, 0.0, 2.0, 1000.0];
    let target_values = [0.0_f32, 1.0, 0.0, 1.0, 1.0];
    let logits = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    let targets = Tensor::from_slice(&target_values, [feature.of(target_values.len())], &device)?;
    // gastric's AxialMIL sets one scalar `pos_weight` per fold from its class balance
    // (`train_axial_mil.py:336`); a nontrivial scalar (!= 1) is this test's scalar case.
    let pos_weight = Tensor::from_slice(&[3.0_f32], [], &device)?;

    let losses = logits.binary_cross_entropy_with_logits_weighted(&targets, &pos_weight)?;
    // Hand-computed at p = 3, from the derivation above:
    //   x=-1000, y=0: log_weight=1, softplus=0                 -> loss=0
    //   x=-2,    y=1: log_weight=3, softplus=0.12692801104297  -> loss=3*0.12692801104297 - 3*1*(-2) = 6.38078403312892
    //   x=0,     y=0: log_weight=1, softplus=ln(2)=0.69314718056 -> loss=0.69314718056
    //   x=2,     y=1: log_weight=3, softplus=2.12692801104297  -> loss=3*2.12692801104297 - 3*1*2 = 0.38078403312892
    //   x=1000,  y=1: log_weight=3, softplus=1000               -> loss=3*1000 - 3*1*1000 = 0
    let expected = [
        0.0,
        6.380_784_033_128_92,
        std::f64::consts::LN_2, // x=0, y=0: loss = softplus(0) - 0 = ln(1 + e^0) = ln(2) exactly
        0.380_784_033_128_92,
        0.0,
    ];
    close(
        "weighted binary cross-entropy",
        &losses.to_vec()?,
        &expected,
    );

    losses.mean(feature)?.backward()?;
    // d(loss)/dx above, divided by the 5-element mean:
    //   x=-1000: 1*0 - 3*0 = 0                     -> 0
    //   x=-2:    3*0.11920292202212 - 3*1 = -2.64239123393365 -> /5 = -0.52847824678673
    //   x=0:     1*0.5 - 3*0 = 0.5                  -> /5 = 0.1
    //   x=2:     3*0.88079707797788 - 3*1 = -0.35760876606635 -> /5 = -0.07152175321327
    //   x=1000:  3*1 - 3*1 = 0                      -> 0
    let expected_gradient = [0.0, -0.528_478_246_786_73, 0.1, -0.071_521_753_213_27, 0.0];
    close(
        "weighted binary cross-entropy gradient",
        &logits.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    assert!(
        logits
            .detach()
            .binary_cross_entropy_with_logits_weighted(&targets, &pos_weight.with_grad())
            .is_err()
    );
    let other = Axis::new("other");
    let wrong_axes = Tensor::from_slice(&target_values, [other.of(5)], &device)?;
    assert!(
        logits
            .detach()
            .binary_cross_entropy_with_logits_weighted(&targets, &wrong_axes)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn binary_cross_entropy_weighted_broadcasts_a_per_class_tensor_over_reordered_storage() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (batch, class) = (
        Axis::new("weighted_bce_batch"),
        Axis::new("weighted_bce_class"),
    );
    // morpheus's objectness head weights `F.binary_cross_entropy_with_logits` by a
    // per-class tensor broadcast over batch and spatial axes
    // (`train_sup.py:156-157`, `pos_weight=pw[None,:,None,None]`); batch/class stand in
    // for that broadcast shape here. `with_layout` keeps the tensor's own logical
    // [class, batch] axis order (so `to_vec()`'s coordinate walk stays class-major) but
    // requests batch-major physical strides, forcing the op to read a genuinely permuted,
    // non-contiguous buffer rather than one that already matches its own shape order --
    // the CUDA non-trivial-layout case.
    let logit_values = [0.5_f32, -0.25, -1.5, 1.0, 2.0, -2.0]; // [class, batch]: c0b0,c0b1,c1b0,c1b1,c2b0,c2b1
    let target_values = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0];
    let pos_weight_values = [2.0_f32, 0.5, 4.0]; // per class, matches pw[None,:,None,None]

    let logits = Tensor::from_slice(&logit_values, [class.of(3), batch.of(2)], &device)?
        .with_layout([batch, class])?
        .with_grad();
    let targets = Tensor::from_slice(&target_values, [class.of(3), batch.of(2)], &device)?
        .with_layout([batch, class])?;
    let pos_weight = Tensor::from_slice(&pos_weight_values, [class.of(3)], &device)?;

    let losses = logits.binary_cross_entropy_with_logits_weighted(&targets, &pos_weight)?;

    // (x, y, p) in [class, batch] row-major order -- the tensors' own logical shape,
    // unchanged by `with_layout` -- matching `to_vec()`'s coordinate walk.
    let logical = [
        (0.5_f64, 1.0_f64, 2.0_f64), // class0, batch0
        (-0.25, 0.0, 2.0),           // class0, batch1
        (-1.5, 0.0, 0.5),            // class1, batch0
        (1.0, 1.0, 0.5),             // class1, batch1
        (2.0, 1.0, 4.0),             // class2, batch0
        (-2.0, 0.0, 4.0),            // class2, batch1
    ];
    let softplus = |x: f64| x.max(0.0) + (-x.abs()).exp().ln_1p();
    let expected_loss: Vec<f64> = logical
        .iter()
        .map(|&(x, y, p)| {
            let log_weight = 1.0 + (p - 1.0) * y;
            log_weight * softplus(x) - p * y * x
        })
        .collect();
    close(
        "weighted binary cross-entropy (per-class broadcast, reordered storage)",
        &losses.to_vec()?,
        &expected_loss,
    );

    losses.mean([batch, class])?.backward()?;
    let expected_gradient: Vec<f64> = logical
        .iter()
        .map(|&(x, y, p)| {
            let log_weight = 1.0 + (p - 1.0) * y;
            let sigma = 1.0 / (1.0 + (-x).exp());
            (log_weight * sigma - p * y) / logical.len() as f64
        })
        .collect();
    close(
        "weighted binary cross-entropy gradient (per-class broadcast, reordered storage)",
        &logits.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    let wrong_extent = Tensor::from_slice(&[1.0_f32, 2.0], [class.of(2)], &device)?;
    assert!(
        logits
            .detach()
            .binary_cross_entropy_with_logits_weighted(&targets, &wrong_extent)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn causal_softmax_masks_values_and_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let (query, key) = (Axis::new("query"), Axis::new("key"));
    let logits = Tensor::from_slice(&[0.0; 9], [query.of(3), key.of(3)], &device)?.with_grad();
    let probability = logits.causal_mask(query, key)?.softmax(key)?;
    close(
        "causal softmax probabilities",
        &probability.to_vec()?,
        &[
            1.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    );

    let weights = Tensor::from_slice(&[1.0, 2.0, 4.0], [key.of(3)], &device)?;
    probability.mul(&weights)?.mean([query, key])?.backward()?;
    close(
        "causal softmax gradient",
        &logits.grad().unwrap().to_vec()?,
        &[
            0.0,
            0.0,
            0.0,
            -0.25 / 9.0,
            0.25 / 9.0,
            0.0,
            -4.0 / 81.0,
            -1.0 / 81.0,
            5.0 / 81.0,
        ],
    );

    assert!(logits.detach().causal_mask(query, query).is_err());
    let rectangular = Tensor::from_slice(&[0.0; 6], [query.of(2), key.of(3)], &device)?;
    assert!(rectangular.causal_mask(query, key).is_err());
    Ok(())
}

#[test]
fn embedding_shapes_reject_missing_axes_before_allocation() -> Result<()> {
    let (batch, position, vocabulary, feature) = (
        Axis::new("batch"),
        Axis::new("position"),
        Axis::new("vocabulary"),
        Axis::new("feature"),
    );
    let tokens = Shape::new([batch.of(2), position.of(5), vocabulary.of(7)])?;
    let embedding = Embedding::new(vocabulary, feature.of(3));
    assert_eq!(
        embedding.output_shape(&tokens)?,
        Shape::new([batch.of(2), position.of(5), feature.of(3)])?
    );
    assert!(
        embedding
            .output_shape(&Shape::new([batch.of(2), position.of(5)])?)
            .is_err()
    );
    let positions = PositionEmbedding::new(position, feature);
    let features = embedding.output_shape(&tokens)?;
    assert_eq!(positions.output_shape(&features)?, features);
    assert!(positions.output_shape(&tokens).is_err());
    assert!(
        PositionEmbedding::new(feature, feature)
            .output_shape(&features)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn embedding_lookup_matches_table_rows_and_accumulates_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, position, vocabulary, feature) = (
        Axis::new("batch"),
        Axis::new("position"),
        Axis::new("vocabulary"),
        Axis::new("feature"),
    );
    // Token 1 appears twice so its table row must receive both gradient contributions.
    let tokens = [[1_usize, 3], [0, 1]];
    let mut one_hot = vec![0.0_f32; 2 * 2 * 4];
    for (b, row) in tokens.iter().enumerate() {
        for (p, &token) in row.iter().enumerate() {
            one_hot[(b * 2 + p) * 4 + token] = 1.0;
        }
    }
    let input = Tensor::from_slice(
        &one_hot,
        [batch.of(2), position.of(2), vocabulary.of(4)],
        &device,
    )?
    .with_layout([vocabulary, batch, position])?;

    let mut embedding = Embedding::new(vocabulary, feature.of(3));
    let mut positions = PositionEmbedding::new(position, feature);
    let shape = embedding.build(input.shape(), &device, 7)?;
    positions.build(&shape, &device, 11)?;
    assert_eq!(
        shape,
        Shape::new([batch.of(2), position.of(2), feature.of(3)])?
    );
    let table = embedding.parameter("table")?.tensor().to_vec()?;
    let offsets = positions.parameter("table")?.tensor().to_vec()?;
    assert!(table.iter().chain(&offsets).all(|v| v.abs() <= 0.02));

    let output = positions.forward(&embedding.forward(&input)?)?;
    let mut expected = Vec::new();
    for row in &tokens {
        for (p, &token) in row.iter().enumerate() {
            for f in 0..3 {
                expected.push(f64::from(table[token * 3 + f]) + f64::from(offsets[p * 3 + f]));
            }
        }
    }
    close("embedding forward", &output.to_vec()?, &expected);

    output.mean([batch, position, feature])?.backward()?;
    let counts = [1.0, 2.0, 0.0, 1.0];
    let table_gradient: Vec<f64> = counts
        .iter()
        .flat_map(|&count| std::iter::repeat_n(count / 12.0, 3))
        .collect();
    close(
        "embedding table gradient",
        &embedding.parameter("table")?.grad().unwrap().to_vec()?,
        &table_gradient,
    );
    close(
        "position table gradient",
        &positions.parameter("table")?.grad().unwrap().to_vec()?,
        &[2.0 / 12.0; 6],
    );

    let wider = Shape::new([batch.of(2), position.of(2), vocabulary.of(5)])?;
    assert!(embedding.output_shape(&wider).is_err());
    assert!(embedding.build(&wider, &device, 7).is_err());
    let longer = Shape::new([batch.of(2), position.of(3), feature.of(3)])?;
    assert!(positions.build(&longer, &device, 11).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn prefix_causal_softmax_keeps_memory_keys_visible_to_every_query() -> Result<()> {
    let device = Device::cuda(0)?;
    let (query, key) = (Axis::new("query"), Axis::new("key"));
    let logits = Tensor::from_slice(&[0.0; 16], [query.of(4), key.of(4)], &device)?.with_grad();
    let probability = logits.prefix_causal_mask(query, key, 2)?.softmax(key)?;
    let kept = |q: usize, k: usize| k <= q || k < 2;
    let mut expected = Vec::new();
    for q in 0..4 {
        let visible = (0..4).filter(|&k| kept(q, k)).count() as f64;
        for k in 0..4 {
            expected.push(if kept(q, k) { 1.0 / visible } else { 0.0 });
        }
    }
    close(
        "prefix causal probabilities",
        &probability.to_vec()?,
        &expected,
    );

    let weights = [1.0, 2.0, 4.0, 8.0];
    let weight_tensor = Tensor::from_slice(&weights.map(|w| w as f32), [key.of(4)], &device)?;
    probability
        .mul(&weight_tensor)?
        .mean([query, key])?
        .backward()?;
    let mut gradient = Vec::new();
    for q in 0..4 {
        let row = &expected[q * 4..q * 4 + 4];
        let mean: f64 = row.iter().zip(&weights).map(|(p, w)| p * w).sum();
        for k in 0..4 {
            gradient.push(row[k] * (weights[k] - mean) / 16.0);
        }
    }
    close(
        "prefix causal gradient",
        &logits.grad().unwrap().to_vec()?,
        &gradient,
    );

    let plain = logits.detach().causal_mask(query, key)?.to_vec()?;
    let zero_prefix = logits
        .detach()
        .prefix_causal_mask(query, key, 0)?
        .to_vec()?;
    assert_eq!(plain, zero_prefix);
    assert!(logits.detach().prefix_causal_mask(query, key, 5).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn conv2d_lowers_valid_patches_and_sums_overlapping_gradients() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, output) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
    );
    let input = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [channel.of(1), height.of(3), width.of(3)],
        &device,
    )?
    .with_grad();
    let mut conv = Conv2d::new(channel, output.of(1), [height, width], [2, 2]);
    assert!(conv.forward(&input).is_err());
    assert_eq!(
        conv.build(input.shape(), &device, 7)?,
        Shape::new([height.of(2), width.of(2), output.of(1)])?
    );
    conv.parameter("weight")?
        .set_values(&[1.0, 0.0, 0.0, -1.0])?;
    conv.parameter("bias")?.set_values(&[0.5])?;

    let result = conv.forward(&input)?;
    close("Conv2d forward", &result.to_vec()?, &[-3.5; 4]);
    result.mean([height, width, output])?.backward()?;
    close(
        "Conv2d overlapping input gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.25, 0.25, 0.0, 0.25, 0.0, -0.25, 0.0, -0.25, -0.25],
    );
    close(
        "Conv2d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &[3.0, 4.0, 6.0, 7.0],
    );
    close(
        "Conv2d bias gradient",
        &conv.parameter("bias")?.grad().unwrap().to_vec()?,
        &[1.0],
    );

    let wrong_channels = Shape::new([channel.of(2), height.of(3), width.of(3)])?;
    assert!(conv.output_shape(&wrong_channels).is_err());
    let normalization = LayerNorm::new(channel)?;
    assert!(normalization.forward(&input).is_err());
    let population = Axis::new("population");
    let unbuilt = PopulationLinear::new(population.of(2), channel, output.of(1));
    assert!(unbuilt.forward(&input).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn unfold2d_plan_cache_distinguishes_equal_extent_spatial_axes() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, time, group, patch) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("time"),
        Axis::new("conv_group"),
        Axis::new("conv_patch"),
    );
    let values: Vec<_> = (0..3)
        .flat_map(|y| (0..3).flat_map(move |x| (0..3).map(move |t| (100 * y + 10 * x + t) as f32)))
        .collect();
    let input = Tensor::from_slice(
        &values,
        [channel.of(1), height.of(3), width.of(3), time.of(3)],
        &device,
    )?;
    let builds = Tensor::unfold_plan_build_count();
    let height_width = input.unfold_grouped(
        channel,
        [height, width],
        group.of(1),
        patch.of(9),
        [3, 3],
        [1, 1],
        [1, 1],
        0.0,
    )?;
    assert_eq!(Tensor::unfold_plan_build_count(), builds + 1);
    let height_time = input.unfold_grouped(
        channel,
        [height, time],
        group.of(1),
        patch.of(9),
        [3, 3],
        [1, 1],
        [1, 1],
        0.0,
    )?;
    assert_eq!(Tensor::unfold_plan_build_count(), builds + 2);

    // Logical output coordinate h=1,w=2,t=0, patch(ky=1,kx=2).
    // [height,width] reaches right padding; [height,time] retains width=2
    // and reads input h=1,w=2,t=1.
    let probe = ((3 + 2) * 3) * 9 + 5;
    assert_eq!(height_width.to_vec()?[probe], 0.0);
    assert_eq!(height_time.to_vec()?[probe], 121.0);
    let repeated = input.unfold_grouped(
        channel,
        [height, width],
        group.of(1),
        patch.of(9),
        [3, 3],
        [1, 1],
        [1, 1],
        0.0,
    )?;
    assert_eq!(repeated.to_vec()?, height_width.to_vec()?);
    assert_eq!(Tensor::unfold_plan_build_count(), builds + 2);

    // An all-padding grouped result keeps the group-major layout required by
    // the following contraction, including when its zero plan is cached.
    let batch = Axis::new("batch");
    let zero_input = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0],
        [batch.of(2), channel.of(2), height.of(1), width.of(1)],
        &device,
    )?;
    let zero_group = Axis::new("zero_group");
    let zero_patch = Axis::new("zero_patch");
    let zero_builds = Tensor::unfold_plan_build_count();
    for _ in 0..2 {
        let all_padding = zero_input.unfold_grouped(
            channel,
            [height, width],
            zero_group.of(2),
            zero_patch.of(1),
            [1, 1],
            [100, 100],
            [10, 10],
            0.0,
        )?;
        assert_eq!(all_padding.to_vec()?, vec![0.0; 4]);
        assert_eq!(all_padding.layout_strides(), &[1, 1, 1, 2, 1]);
    }
    assert_eq!(Tensor::unfold_plan_build_count(), zero_builds + 1);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn conv2d_rejects_padded_extent_outside_kernel_index_range_atomically() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, output) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
    );
    let input = Tensor::from_slice(
        &[1.0, 3.0],
        [channel.of(1), height.of(2), width.of(1)],
        &device,
    )?
    .with_grad();
    let mut overflowing = Conv2d::new(channel, output.of(1), [height, width], [1, 1])
        .stride([1 << 30, 1])
        .padding([i32::MAX as usize, 0]);
    assert_eq!(
        overflowing.build(input.shape(), &device, 31)?,
        Shape::new([height.of(4), width.of(1), output.of(1)])?
    );
    overflowing.parameter("weight")?.set_values(&[1.0])?;
    let builds = Tensor::unfold_plan_build_count();
    let error = overflowing.forward(&input).err().unwrap().to_string();
    assert_eq!(
        error,
        "unfold2d padded spatial extent exceeds the i32 kernel index range"
    );
    assert_eq!(Tensor::unfold_plan_build_count(), builds);

    // Rejection happens before cache mutation, output allocation, or launch;
    // the same input and device remain usable for a valid forward/backward.
    let mut valid = Conv2d::new(channel, output.of(1), [height, width], [1, 1]);
    valid.build(input.shape(), &device, 37)?;
    valid.parameter("weight")?.set_values(&[1.0])?;
    valid.parameter("bias")?.set_values(&[0.0])?;
    let result = valid.forward(&input)?;
    close(
        "post-rejection Conv2d forward",
        &result.to_vec()?,
        &[1.0, 3.0],
    );
    result.mean([height, width, output])?.backward()?;
    close(
        "post-rejection Conv2d input gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.5, 0.5],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn configured_grouped_conv2d_matches_scalar_forward_and_all_gradients() -> Result<()> {
    const BATCHES: usize = 1;
    const CHANNELS: usize = 4;
    const HEIGHT: usize = 4;
    const WIDTH: usize = 5;
    const OUTPUTS: usize = 6;
    const GROUPS: usize = 2;
    const KERNEL: [usize; 2] = [2, 3];
    const STRIDE: [usize; 2] = [2, 1];
    const PADDING: [usize; 2] = [2, 3];
    const OUTPUT_HEIGHT: usize = 4;
    const OUTPUT_WIDTH: usize = 9;
    const CHANNELS_PER_GROUP: usize = CHANNELS / GROUPS;
    const OUTPUTS_PER_GROUP: usize = OUTPUTS / GROUPS;
    const PATCH: usize = CHANNELS_PER_GROUP * KERNEL[0] * KERNEL[1];

    let device = Device::cuda(0)?;
    let (batch, channel, height, width, output) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
    );
    let inputs: Vec<_> = (0..BATCHES * CHANNELS * HEIGHT * WIDTH)
        .map(|i| (i as f32 - 31.0) / 17.0)
        .collect();
    let weights: Vec<_> = (0..GROUPS * PATCH * OUTPUTS_PER_GROUP)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) / 19.0)
        .collect();
    let biases: Vec<_> = (0..OUTPUTS).map(|i| (i as f32 - 2.0) / 11.0).collect();
    let input = Tensor::from_slice(
        &inputs,
        [
            batch.of(BATCHES),
            channel.of(CHANNELS),
            height.of(HEIGHT),
            width.of(WIDTH),
        ],
        &device,
    )?
    .with_layout([width, batch, channel, height])?
    .with_grad();
    let mut conv = Conv2d::new(channel, output.of(OUTPUTS), [height, width], KERNEL)
        .stride(STRIDE)
        .padding(PADDING)
        .groups(GROUPS);
    assert_eq!(
        conv.build(input.shape(), &device, 19)?,
        Shape::new([
            batch.of(BATCHES),
            height.of(OUTPUT_HEIGHT),
            width.of(OUTPUT_WIDTH),
            output.of(OUTPUTS),
        ])?
    );
    conv.parameter("weight")?.set_values(&weights)?;
    conv.parameter("bias")?.set_values(&biases)?;

    let mut expected = vec![0.0_f64; BATCHES * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUTS];
    let mut input_gradient = vec![0.0_f64; inputs.len()];
    let mut weight_gradient = vec![0.0_f64; weights.len()];
    let mut bias_gradient = vec![0.0_f64; biases.len()];
    let upstream = 1.0 / expected.len() as f64;
    for n in 0..BATCHES {
        for oy in 0..OUTPUT_HEIGHT {
            for ox in 0..OUTPUT_WIDTH {
                for oc in 0..OUTPUTS {
                    let group = oc / OUTPUTS_PER_GROUP;
                    let output_in_group = oc % OUTPUTS_PER_GROUP;
                    let mut value = f64::from(biases[oc]);
                    bias_gradient[oc] += upstream;
                    for channel_in_group in 0..CHANNELS_PER_GROUP {
                        let input_channel = group * CHANNELS_PER_GROUP + channel_in_group;
                        for ky in 0..KERNEL[0] {
                            for kx in 0..KERNEL[1] {
                                let padded_y = oy * STRIDE[0] + ky;
                                let padded_x = ox * STRIDE[1] + kx;
                                let Some(iy) = padded_y.checked_sub(PADDING[0]) else {
                                    continue;
                                };
                                let Some(ix) = padded_x.checked_sub(PADDING[1]) else {
                                    continue;
                                };
                                if iy >= HEIGHT || ix >= WIDTH {
                                    continue;
                                }
                                let input_index =
                                    ((n * CHANNELS + input_channel) * HEIGHT + iy) * WIDTH + ix;
                                let patch = (channel_in_group * KERNEL[0] + ky) * KERNEL[1] + kx;
                                let weight_index =
                                    (group * PATCH + patch) * OUTPUTS_PER_GROUP + output_in_group;
                                value += f64::from(inputs[input_index])
                                    * f64::from(weights[weight_index]);
                                input_gradient[input_index] +=
                                    upstream * f64::from(weights[weight_index]);
                                weight_gradient[weight_index] +=
                                    upstream * f64::from(inputs[input_index]);
                            }
                        }
                    }
                    let output_index =
                        ((n * OUTPUT_HEIGHT + oy) * OUTPUT_WIDTH + ox) * OUTPUTS + oc;
                    expected[output_index] = value;
                }
            }
        }
    }

    let plan_builds_before = Tensor::unfold_plan_build_count();
    let actual = conv.forward(&input)?;
    let plan_builds_after = Tensor::unfold_plan_build_count();
    assert_eq!(plan_builds_after, plan_builds_before + 1);
    close(
        "configured grouped Conv2d forward",
        &actual.to_vec()?,
        &expected,
    );
    close(
        "cached configured grouped Conv2d forward",
        &conv.forward(&input)?.to_vec()?,
        &expected,
    );
    assert_eq!(Tensor::unfold_plan_build_count(), plan_builds_after);
    actual.mean([batch, height, width, output])?.backward()?;
    close(
        "configured grouped Conv2d input gradient",
        &input.grad().unwrap().to_vec()?,
        &input_gradient,
    );
    close(
        "configured grouped Conv2d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &weight_gradient,
    );
    close(
        "configured grouped Conv2d bias gradient",
        &conv.parameter("bias")?.grad().unwrap().to_vec()?,
        &bias_gradient,
    );
    let regrouped = conv.groups(1);
    let error = regrouped
        .output_shape(input.shape())
        .unwrap_err()
        .to_string();
    assert!(error.contains("built parameters"), "{error}");

    // A legal strided window can lie wholly in padding. It contributes zero
    // before bias and therefore has zero input/weight derivatives.
    let isolated = Tensor::from_slice(&[3.0], [channel.of(1), height.of(1), width.of(1)], &device)?
        .with_grad();
    let mut padded = Conv2d::new(channel, output.of(1), [height, width], [1, 1])
        .stride([100, 100])
        .padding([10, 10]);
    padded.build(isolated.shape(), &device, 23)?;
    padded.parameter("weight")?.set_values(&[2.0])?;
    padded.parameter("bias")?.set_values(&[0.25])?;
    let padded_result = padded.forward(&isolated)?;
    close(
        "all-padding Conv2d forward",
        &padded_result.to_vec()?,
        &[0.25],
    );
    padded_result.mean([height, width, output])?.backward()?;
    close(
        "all-padding Conv2d input gradient",
        &isolated.grad().unwrap().to_vec()?,
        &[0.0],
    );
    close(
        "all-padding Conv2d weight gradient",
        &padded.parameter("weight")?.grad().unwrap().to_vec()?,
        &[0.0],
    );
    close(
        "all-padding Conv2d bias gradient",
        &padded.parameter("bias")?.grad().unwrap().to_vec()?,
        &[1.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn configured_grouped_conv3d_matches_scalar_forward_and_all_gradients() -> Result<()> {
    const BATCHES: usize = 1;
    const CHANNELS: usize = 4;
    const DEPTH: usize = 3;
    const HEIGHT: usize = 4;
    const WIDTH: usize = 5;
    const OUTPUTS: usize = 6;
    const GROUPS: usize = 2;
    const KERNEL: [usize; 3] = [2, 2, 3];
    const STRIDE: [usize; 3] = [1, 2, 1];
    const PADDING: [usize; 3] = [1, 1, 2];
    const OUTPUT_DEPTH: usize = 4;
    const OUTPUT_HEIGHT: usize = 3;
    const OUTPUT_WIDTH: usize = 7;
    const CHANNELS_PER_GROUP: usize = CHANNELS / GROUPS;
    const OUTPUTS_PER_GROUP: usize = OUTPUTS / GROUPS;
    const PATCH: usize = CHANNELS_PER_GROUP * KERNEL[0] * KERNEL[1] * KERNEL[2];

    let device = Device::cuda(0)?;
    let (batch, channel, depth, height, width, output) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
    );
    let inputs: Vec<_> = (0..BATCHES * CHANNELS * DEPTH * HEIGHT * WIDTH)
        .map(|i| ((i * 11 % 97) as f32 - 48.0) / 23.0)
        .collect();
    let weights: Vec<_> = (0..GROUPS * PATCH * OUTPUTS_PER_GROUP)
        .map(|i| ((i * 13 % 41) as f32 - 20.0) / 29.0)
        .collect();
    let biases: Vec<_> = (0..OUTPUTS).map(|i| (i as f32 - 2.5) / 13.0).collect();
    let input = Tensor::from_slice(
        &inputs,
        [
            batch.of(BATCHES),
            channel.of(CHANNELS),
            depth.of(DEPTH),
            height.of(HEIGHT),
            width.of(WIDTH),
        ],
        &device,
    )?
    .with_layout([width, batch, channel, depth, height])?
    .with_grad();
    let mut conv = Conv3d::new(channel, output.of(OUTPUTS), [depth, height, width], KERNEL)
        .stride(STRIDE)
        .padding(PADDING)
        .groups(GROUPS);
    assert_eq!(
        conv.build(input.shape(), &device, 41)?,
        Shape::new([
            batch.of(BATCHES),
            depth.of(OUTPUT_DEPTH),
            height.of(OUTPUT_HEIGHT),
            width.of(OUTPUT_WIDTH),
            output.of(OUTPUTS),
        ])?
    );
    conv.parameter("weight")?.set_values(&weights)?;
    conv.parameter("bias")?.set_values(&biases)?;

    let mut expected =
        vec![0.0_f64; BATCHES * OUTPUT_DEPTH * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUTS];
    let mut input_gradient = vec![0.0_f64; inputs.len()];
    let mut weight_gradient = vec![0.0_f64; weights.len()];
    let mut bias_gradient = vec![0.0_f64; biases.len()];
    let upstream = 1.0 / expected.len() as f64;
    for n in 0..BATCHES {
        for oz in 0..OUTPUT_DEPTH {
            for oy in 0..OUTPUT_HEIGHT {
                for ox in 0..OUTPUT_WIDTH {
                    for oc in 0..OUTPUTS {
                        let group = oc / OUTPUTS_PER_GROUP;
                        let output_in_group = oc % OUTPUTS_PER_GROUP;
                        let mut value = f64::from(biases[oc]);
                        bias_gradient[oc] += upstream;
                        for channel_in_group in 0..CHANNELS_PER_GROUP {
                            let input_channel = group * CHANNELS_PER_GROUP + channel_in_group;
                            for kz in 0..KERNEL[0] {
                                for ky in 0..KERNEL[1] {
                                    for kx in 0..KERNEL[2] {
                                        let padded_z = oz * STRIDE[0] + kz;
                                        let padded_y = oy * STRIDE[1] + ky;
                                        let padded_x = ox * STRIDE[2] + kx;
                                        let Some(iz) = padded_z.checked_sub(PADDING[0]) else {
                                            continue;
                                        };
                                        let Some(iy) = padded_y.checked_sub(PADDING[1]) else {
                                            continue;
                                        };
                                        let Some(ix) = padded_x.checked_sub(PADDING[2]) else {
                                            continue;
                                        };
                                        if iz >= DEPTH || iy >= HEIGHT || ix >= WIDTH {
                                            continue;
                                        }
                                        let input_index =
                                            (((n * CHANNELS + input_channel) * DEPTH + iz)
                                                * HEIGHT
                                                + iy)
                                                * WIDTH
                                                + ix;
                                        let patch =
                                            ((channel_in_group * KERNEL[0] + kz) * KERNEL[1] + ky)
                                                * KERNEL[2]
                                                + kx;
                                        let weight_index = (group * PATCH + patch)
                                            * OUTPUTS_PER_GROUP
                                            + output_in_group;
                                        value += f64::from(inputs[input_index])
                                            * f64::from(weights[weight_index]);
                                        input_gradient[input_index] +=
                                            upstream * f64::from(weights[weight_index]);
                                        weight_gradient[weight_index] +=
                                            upstream * f64::from(inputs[input_index]);
                                    }
                                }
                            }
                        }
                        let output_index =
                            ((((n * OUTPUT_DEPTH + oz) * OUTPUT_HEIGHT + oy) * OUTPUT_WIDTH + ox)
                                * OUTPUTS)
                                + oc;
                        expected[output_index] = value;
                    }
                }
            }
        }
    }

    let plan_builds_before = Tensor::unfold_plan_build_count();
    let actual = conv.forward(&input)?;
    let plan_builds_after = Tensor::unfold_plan_build_count();
    assert_eq!(plan_builds_after, plan_builds_before + 1);
    close(
        "configured grouped Conv3d forward",
        &actual.to_vec()?,
        &expected,
    );
    close(
        "cached configured grouped Conv3d forward",
        &conv.forward(&input)?.to_vec()?,
        &expected,
    );
    assert_eq!(Tensor::unfold_plan_build_count(), plan_builds_after);
    actual
        .mean([batch, depth, height, width, output])?
        .backward()?;
    close(
        "configured grouped Conv3d input gradient",
        &input.grad().unwrap().to_vec()?,
        &input_gradient,
    );
    close(
        "configured grouped Conv3d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &weight_gradient,
    );
    close(
        "configured grouped Conv3d bias gradient",
        &conv.parameter("bias")?.grad().unwrap().to_vec()?,
        &bias_gradient,
    );
    let regrouped = conv.groups(1);
    let error = regrouped
        .output_shape(input.shape())
        .unwrap_err()
        .to_string();
    assert!(error.contains("built parameters"), "{error}");

    // Every legal output window is wholly in padding, so the bias is the
    // output and neither input nor weight receives a derivative.
    let isolated = Tensor::from_slice(
        &[3.0],
        [channel.of(1), depth.of(1), height.of(1), width.of(1)],
        &device,
    )?
    .with_grad();
    let mut padded = Conv3d::new(channel, output.of(1), [depth, height, width], [1, 1, 1])
        .stride([100, 100, 100])
        .padding([10, 10, 10]);
    padded.build(isolated.shape(), &device, 43)?;
    padded.parameter("weight")?.set_values(&[2.0])?;
    padded.parameter("bias")?.set_values(&[0.25])?;
    let padded_result = padded.forward(&isolated)?;
    close(
        "all-padding Conv3d forward",
        &padded_result.to_vec()?,
        &[0.25],
    );
    padded_result
        .mean([depth, height, width, output])?
        .backward()?;
    close(
        "all-padding Conv3d input gradient",
        &isolated.grad().unwrap().to_vec()?,
        &[0.0],
    );
    close(
        "all-padding Conv3d weight gradient",
        &padded.parameter("weight")?.grad().unwrap().to_vec()?,
        &[0.0],
    );
    close(
        "all-padding Conv3d bias gradient",
        &padded.parameter("bias")?.grad().unwrap().to_vec()?,
        &[1.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn unfold3d_cache_distinguishes_equal_extent_spatial_axis_order() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, depth, height, width, time, group, patch) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("time"),
        Axis::new("conv_group"),
        Axis::new("conv_patch"),
    );
    let values: Vec<_> = (0..3)
        .flat_map(|z| {
            (0..3).flat_map(move |y| {
                (0..3).flat_map(move |x| {
                    (0..3).map(move |t| (1000 * z + 100 * y + 10 * x + t) as f32)
                })
            })
        })
        .collect();
    let input = Tensor::from_slice(
        &values,
        [
            channel.of(1),
            depth.of(3),
            height.of(3),
            width.of(3),
            time.of(3),
        ],
        &device,
    )?;
    let builds = Tensor::unfold_plan_build_count();
    let depth_height_width = input.unfold_grouped(
        channel,
        [depth, height, width],
        group.of(1),
        patch.of(27),
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
        0.0,
    )?;
    assert_eq!(Tensor::unfold_plan_build_count(), builds + 1);
    let depth_height_time = input.unfold_grouped(
        channel,
        [depth, height, time],
        group.of(1),
        patch.of(27),
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
        0.0,
    )?;
    assert_eq!(Tensor::unfold_plan_build_count(), builds + 2);

    // Logical coordinate d=1,h=1,w=2,t=0 and patch kz=1,ky=1,kx=2.
    // Selecting width reaches right padding; selecting time retains width=2
    // and reads d=1,h=1,w=2,t=1.
    let logical_index = |z: usize, y: usize, x: usize, t: usize, patch: usize| {
        ((((z * 3 + y) * 3 + x) * 3 + t) * 27) + patch
    };
    let patch_index = 14; // kz=1, ky=1, kx=2
    let probe = logical_index(1, 1, 2, 0, patch_index);
    assert_eq!(depth_height_width.to_vec()?[probe], 0.0);
    assert_eq!(depth_height_time.to_vec()?[probe], 1121.0);
    let repeated = input.unfold_grouped(
        channel,
        [depth, height, width],
        group.of(1),
        patch.of(27),
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
        0.0,
    )?;
    assert_eq!(repeated.to_vec()?, depth_height_width.to_vec()?);
    assert_eq!(Tensor::unfold_plan_build_count(), builds + 2);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn conv3d_rejects_padded_extent_outside_kernel_index_range_atomically() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, depth, height, width, output) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
    );
    let input = Tensor::from_slice(
        &[1.0, 3.0],
        [channel.of(1), depth.of(2), height.of(1), width.of(1)],
        &device,
    )?
    .with_grad();
    let mut overflowing = Conv3d::new(channel, output.of(1), [depth, height, width], [1, 1, 1])
        .stride([1 << 30, 1, 1])
        .padding([i32::MAX as usize, 0, 0]);
    assert_eq!(
        overflowing.build(input.shape(), &device, 47)?,
        Shape::new([depth.of(4), height.of(1), width.of(1), output.of(1),])?
    );
    overflowing.parameter("weight")?.set_values(&[1.0])?;
    let builds = Tensor::unfold_plan_build_count();
    let error = overflowing.forward(&input).err().unwrap().to_string();
    assert_eq!(
        error,
        "unfold3d padded spatial extent exceeds the i32 kernel index range"
    );
    assert_eq!(Tensor::unfold_plan_build_count(), builds);

    let mut valid = Conv3d::new(channel, output.of(1), [depth, height, width], [1, 1, 1]);
    valid.build(input.shape(), &device, 53)?;
    valid.parameter("weight")?.set_values(&[1.0])?;
    valid.parameter("bias")?.set_values(&[0.0])?;
    let result = valid.forward(&input)?;
    close(
        "post-rejection Conv3d forward",
        &result.to_vec()?,
        &[1.0, 3.0],
    );
    result.mean([depth, height, width, output])?.backward()?;
    close(
        "post-rejection Conv3d input gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.5, 0.5],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn depthwise_conv3d_large_implicit_plan_completes_forward_and_backward() -> Result<()> {
    const BATCHES: usize = 20;
    const DEPTH: usize = 32;
    const HEIGHT: usize = 32;
    const WIDTH: usize = 32;

    let device = Device::cuda(0)?;
    let (batch, channel, depth, height, width) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let input = Tensor::from_slice(
        &vec![1.0; BATCHES * DEPTH * HEIGHT * WIDTH],
        [
            batch.of(BATCHES),
            channel.of(1),
            depth.of(DEPTH),
            height.of(HEIGHT),
            width.of(WIDTH),
        ],
        &device,
    )?
    .with_grad();
    let mut conv =
        Conv3d::new(channel, channel.of(1), [depth, height, width], [3, 3, 3]).padding([1, 1, 1]);
    conv.build(input.shape(), &device, 59)?;
    conv.parameter("weight")?.set_values(&[1.0; 27])?;
    conv.parameter("bias")?.set_values(&[0.25])?;

    // The indexed lowering would need 20 * 32^3 * 27 = 17,694,720
    // contributions and fail at the generic 16,777,216-plan cap.
    let output = conv.forward(&input)?;
    assert!(
        Tensor::unfold_plan_metadata_max() <= 4 * (5 + 6),
        "implicit unfold metadata must stay O(rank)"
    );
    assert_eq!(
        output.shape(),
        &Shape::new([
            batch.of(BATCHES),
            depth.of(DEPTH),
            height.of(HEIGHT),
            width.of(WIDTH),
            channel.of(1),
        ])?
    );
    let values = output.to_vec()?;
    for n in 0..BATCHES {
        for z in 0..DEPTH {
            let z_uses = if z == 0 || z + 1 == DEPTH { 2 } else { 3 };
            for y in 0..HEIGHT {
                let y_uses = if y == 0 || y + 1 == HEIGHT { 2 } else { 3 };
                for x in 0..WIDTH {
                    let x_uses = if x == 0 || x + 1 == WIDTH { 2 } else { 3 };
                    let index = ((n * DEPTH + z) * HEIGHT + y) * WIDTH + x;
                    assert_eq!(
                        values[index],
                        (z_uses * y_uses * x_uses) as f32 + 0.25,
                        "output[{index}]"
                    );
                }
            }
        }
    }
    let loss = output.mean([batch, depth, height, width, channel])?;
    assert!(loss.item()?.is_finite());
    loss.backward()?;
    // This stress witness checks bounded plan-free execution. The smaller
    // scalar oracle above carries the precise all-gradient comparison; this
    // generic FP32 reduction sums 655,360 contributions in one group.
    let bias_gradient = conv.parameter("bias")?.grad().unwrap().to_vec()?[0];
    assert!(
        (bias_gradient - 1.0).abs() < 0.01,
        "large depthwise Conv3d bias gradient: {bias_gradient}"
    );
    let input_gradient = input.grad().unwrap().to_vec()?;
    let denominator = (BATCHES * DEPTH * HEIGHT * WIDTH) as f64;
    for (index, &gradient) in input_gradient.iter().enumerate() {
        assert!(gradient.is_finite(), "input gradient[{index}]");
        assert!(f64::from(gradient) >= 8.0 / denominator);
        assert!(f64::from(gradient) <= 27.0 / denominator);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn depthwise_conv2d_large_implicit_plan_completes_forward_and_backward() -> Result<()> {
    const BATCHES: usize = 128;
    const CHANNELS: usize = 32;
    const HEIGHT: usize = 32;
    const WIDTH: usize = 32;

    let device = Device::cuda(0)?;
    let (batch, channel, height, width) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let input = Tensor::from_slice(
        &vec![1.0; BATCHES * CHANNELS * HEIGHT * WIDTH],
        [
            batch.of(BATCHES),
            channel.of(CHANNELS),
            height.of(HEIGHT),
            width.of(WIDTH),
        ],
        &device,
    )?
    .with_grad();
    let mut conv = Conv2d::new(channel, channel.of(CHANNELS), [height, width], [3, 3])
        .padding([1, 1])
        .groups(CHANNELS);
    conv.build(input.shape(), &device, 29)?;
    conv.parameter("weight")?.set_values(&[1.0; CHANNELS * 9])?;
    conv.parameter("bias")?.set_values(&[0.25; CHANNELS])?;

    // The former indexed lowering needed 128 * 32 * 32 * 32 * 9 =
    // 37,748,736 contributions and failed at its 16,777,216-plan cap.
    let output = conv.forward(&input)?;
    assert!(
        Tensor::unfold_plan_metadata_max() <= 4 * (4 + 5),
        "implicit unfold metadata must stay O(rank)"
    );
    assert_eq!(
        output.shape(),
        &Shape::new([
            batch.of(BATCHES),
            height.of(HEIGHT),
            width.of(WIDTH),
            channel.of(CHANNELS),
        ])?
    );
    let values = output.to_vec()?;
    for batch_index in 0..BATCHES {
        for y in 0..HEIGHT {
            let y_uses = if y == 0 || y + 1 == HEIGHT { 2 } else { 3 };
            for x in 0..WIDTH {
                let x_uses = if x == 0 || x + 1 == WIDTH { 2 } else { 3 };
                let expected = (y_uses * x_uses) as f32 + 0.25;
                for channel_index in 0..CHANNELS {
                    let index = ((batch_index * HEIGHT + y) * WIDTH + x) * CHANNELS + channel_index;
                    assert_eq!(values[index], expected, "output[{index}]");
                }
            }
        }
    }
    let loss = output.mean([batch, height, width, channel])?;
    assert!(loss.item()?.is_finite());
    loss.backward()?;
    assert_eq!(
        conv.parameter("bias")?.grad().unwrap().to_vec()?,
        vec![1.0 / CHANNELS as f32; CHANNELS]
    );
    let mut expected_weight_gradient = Vec::with_capacity(CHANNELS * 9);
    for _ in 0..CHANNELS {
        for kernel_y in 0_usize..3 {
            for kernel_x in 0_usize..3 {
                let valid = (HEIGHT - kernel_y.abs_diff(1)) * (WIDTH - kernel_x.abs_diff(1));
                expected_weight_gradient.push(valid as f64 / (CHANNELS * HEIGHT * WIDTH) as f64);
            }
        }
    }
    close(
        "large depthwise Conv2d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &expected_weight_gradient,
    );
    let input_gradient = input.grad().unwrap().to_vec()?;
    let denominator = (BATCHES * CHANNELS * HEIGHT * WIDTH) as f32;
    for batch_index in 0..BATCHES {
        for channel_index in 0..CHANNELS {
            for y in 0..HEIGHT {
                let y_uses = if y == 0 || y + 1 == HEIGHT { 2 } else { 3 };
                for x in 0..WIDTH {
                    let x_uses = if x == 0 || x + 1 == WIDTH { 2 } else { 3 };
                    let index = ((batch_index * CHANNELS + channel_index) * HEIGHT + y) * WIDTH + x;
                    let expected = (y_uses * x_uses) as f32 / denominator;
                    assert!(
                        (input_gradient[index] - expected).abs() < 1e-10,
                        "input gradient[{index}]"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn parameter_versions_accumulation_and_shared_updates() -> Result<()> {
    let device = Device::cuda(0)?;
    let f = Axis::new("feature");
    let p = Parameter::new(Tensor::from_slice(&[1.0, 2.0, 3.0], [f.of(3)], &device)?);
    let stale = p.tensor().mul(&p.tensor())?.mean(f)?;
    p.tensor().mul(&p.tensor())?.mean(f)?.backward()?;
    p.tensor().mul(&p.tensor())?.mean(f)?.backward()?;
    close(
        "leaf gradient accumulation",
        &p.grad().unwrap().to_vec()?,
        &[4.0 / 3.0, 8.0 / 3.0, 4.0],
    );
    p.zero_grad();
    assert!(p.grad().is_none());
    p.tensor().mul(&p.tensor())?.mean(f)?.backward()?;
    let id = p.id();
    SGD::new(0.3)?.step_parameters([p.clone(), p.clone()])?;
    assert_eq!(p.id(), id);
    close(
        "shared parameter updated once",
        &p.tensor().to_vec()?,
        &[0.8, 1.6, 2.4],
    );
    let error = stale.backward().err().unwrap().to_string();
    assert!(error.contains("parameter changed"), "{error}");
    let stale = p.tensor().mean(f)?;
    p.set_values(&[1.0, 2.0, 3.0])?;
    assert!(stale.backward().is_err());
    // A missing gradient must fail before any of the parameters are replaced.
    p.tensor().mean(f)?.backward()?;
    let missing = Parameter::new(Tensor::from_slice(&[0.0], [], &device)?);
    assert!(
        SGD::new(1.0)?
            .step_parameters([p.clone(), missing])
            .is_err()
    );
    close(
        "failed step is atomic",
        &p.tensor().to_vec()?,
        &[1., 2., 3.],
    );
    // A built module clone ties actual parameters and its two uses both contribute.
    let mut layer = Linear::new(f, f.of(3));
    assert!(layer.parameter("weight").is_err());
    layer.build(&Shape::new([f.of(3)])?, &device, 1)?;
    let params = [layer.parameter("weight")?, layer.parameter("bias")?];
    params[0].set_values(&[1., 0., 0., 0., 1., 0., 0., 0., 1.])?;
    params[1].set_values(&[0.; 3])?;
    let b = Axis::new("batch");
    layer.build(&Shape::new([b.of(7), f.of(3)])?, &device, 999)?;
    assert_eq!(layer.parameter("weight")?.id(), params[0].id());
    assert_eq!(
        layer.parameter("weight")?.tensor().to_vec()?,
        vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]
    );
    assert!(layer.build(&Shape::new([f.of(4)])?, &device, 1).is_err());
    let mut tied = Sequential::new((layer.clone(), layer));
    assert_eq!(
        tied.parameter("0.weight")?.id(),
        tied.parameter("1.weight")?.id()
    );
    assert!(tied.parameter("2.weight").is_err());
    let input = Tensor::from_slice(&[1., 2., 3.], [f.of(3)], &device)?;
    tied.forward(&input)?.mean(f)?.backward()?;
    close(
        "tied layer bias contributions",
        &params[1].grad().unwrap().to_vec()?,
        &[2.0 / 3.0; 3],
    );
    SGD::new(0.3)?.step(&mut tied)?;
    close(
        "tied layer one update",
        &params[1].tensor().to_vec()?,
        &[-0.2; 3],
    );
    let nested = Sequential::new((ReLU, tied));
    assert_eq!(nested.parameter("1.0.weight")?.id(), params[0].id());
    assert_eq!(nested.parameter("1.1.weight")?.id(), params[0].id());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn categorical_loss_and_device_adam_match_references() -> Result<()> {
    let device = Device::cuda(0)?;
    let (class, batch) = (Axis::new("class"), Axis::new("batch"));
    // Class is deliberately the leading logical axis; the loss must align it
    // internally and return only the retained batch axis.
    let logits = Tensor::from_slice(
        &[1000.0, -1.0, 999.0, 0.0, 998.0, 1.0],
        [class.of(3), batch.of(2)],
        &device,
    )?
    .with_grad();
    let targets = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [class.of(3), batch.of(2)],
        &device,
    )?;
    let losses = logits.categorical_cross_entropy_with_logits(&targets, class)?;
    let accuracy = logits.categorical_accuracy(&targets, class)?;
    assert_eq!(accuracy.correct(), 2);
    assert_eq!(accuracy.total(), 2);
    assert_eq!(accuracy.fraction(), 1.0);
    assert_eq!(accuracy.merge(accuracy).total(), 4);
    let selected = Tensor::from_slice(&[1.0, 0.0], [batch.of(2)], &device)?;
    let masked = logits.masked_categorical_accuracy(&targets, &selected, class)?;
    assert_eq!(masked.correct(), 1);
    assert_eq!(masked.total(), 1);
    assert_eq!(masked.fraction(), 1.0);
    let wrong_shape = Tensor::from_slice(&[1.0], [batch.of(1)], &device)?;
    let error = logits
        .masked_categorical_accuracy(&targets, &wrong_shape, class)
        .unwrap_err()
        .to_string();
    assert!(error.contains("mask must match"), "{error}");
    let nonbinary = Tensor::from_slice(&[1.0, 0.5], [batch.of(2)], &device)?;
    let error = logits
        .masked_categorical_accuracy(&targets, &nonbinary, class)
        .unwrap_err()
        .to_string();
    assert!(error.contains("mask must be binary"), "{error}");
    let empty = Tensor::from_slice(&[0.0, 0.0], [batch.of(2)], &device)?;
    let error = logits
        .masked_categorical_accuracy(&targets, &empty, class)
        .unwrap_err()
        .to_string();
    assert!(error.contains("select at least one row"), "{error}");
    assert_eq!(losses.shape(), &Shape::new([batch.of(2)])?);
    let expected_loss = (1.0_f64 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
    close(
        "stable categorical cross-entropy",
        &losses.to_vec()?,
        &[expected_loss, expected_loss],
    );
    losses.mean(batch)?.backward()?;
    let p0 = 1.0 / (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp());
    let p1 = (-1.0_f64).exp() * p0;
    let p2 = (-2.0_f64).exp() * p0;
    close(
        "categorical cross-entropy gradient",
        &logits.grad().unwrap().to_vec()?,
        &[
            (p0 - 1.0) / 2.0,
            p2 / 2.0,
            p1 / 2.0,
            p1 / 2.0,
            p2 / 2.0,
            (p0 - 1.0) / 2.0,
        ],
    );
    assert!(
        logits
            .detach()
            .categorical_cross_entropy_with_logits(&targets.with_grad(), class)
            .is_err()
    );
    let invalid_targets = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        [class.of(3), batch.of(2)],
        &device,
    )?;
    let error = logits
        .detach()
        .categorical_cross_entropy_with_logits(&invalid_targets, class)
        .err()
        .expect("invalid categorical target")
        .to_string();
    assert!(error.contains("row 1 must sum to 1"), "{error}");

    let feature = Axis::new("feature");
    let parameter = Parameter::new(Tensor::from_slice(&[1.0, -2.0], [feature.of(2)], &device)?);
    let coefficient = Tensor::from_slice(&[0.2, -0.4], [feature.of(2)], &device)?;
    let mut adam = Adam::new(0.01)?;
    for expected in [[0.99, -1.99], [0.98, -1.98]] {
        parameter
            .tensor()
            .mul(&coefficient)?
            .mean(feature)?
            .backward()?;
        adam.step_parameters([parameter.clone(), parameter.clone()])?;
        close(
            "device Adam update",
            &parameter.tensor().to_vec()?,
            &expected,
        );
        parameter.zero_grad();
    }
    assert_eq!(adam.completed_steps(), 2);
    assert!(Adam::with_hyperparameters(0.1, 1.0, 0.999, 1e-8).is_err());
    parameter.zero_grad();
    parameter.tensor().scale(0.0)?.mean(feature)?.backward()?;
    let mut adamw = AdamW::new(0.1, 0.2)?;
    adamw.step_parameters([parameter.clone()])?;
    close(
        "device AdamW decay",
        &parameter.tensor().to_vec()?,
        &[0.9604, -1.9404],
    );
    assert_eq!(adamw.completed_steps(), 1);
    assert!(AdamW::new(0.1, -0.1).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn population_linear_and_axis_adam_keep_members_independent() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, population, input, output, class) = (
        Axis::new("batch"),
        Axis::new("population"),
        Axis::new("input"),
        Axis::new("output"),
        Axis::new("class"),
    );
    let mut layer = PopulationLinear::new(population.of(2), input, output.of(2));
    layer.build(&Shape::new([batch.of(2), input.of(2)])?, &device, 7)?;
    layer
        .parameter("weight")?
        .set_values(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0])?;
    layer.parameter("bias")?.set_values(&[0.0; 4])?;
    let x =
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [batch.of(2), input.of(2)], &device)?.with_grad();
    let prediction = layer.forward(&x)?;
    assert_eq!(
        prediction.shape(),
        &Shape::new([batch.of(2), population.of(2), output.of(2)])?
    );
    close(
        "population linear forward",
        &prediction.to_vec()?,
        &[1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0],
    );
    prediction.mean([batch, population, output])?.backward()?;
    close(
        "population linear input gradient",
        &x.grad().unwrap().to_vec()?,
        &[0.375, 0.375, 0.375, 0.375],
    );
    close(
        "population linear weight gradient",
        &layer.parameter("weight")?.grad().unwrap().to_vec()?,
        &[0.5, 0.5, 0.75, 0.75, 0.5, 0.5, 0.75, 0.75],
    );
    let mut adam = Adam::with_axis_learning_rates(population, vec![0.01, 0.1])?;
    adam.step(&mut layer)?;
    close(
        "population Adam rates",
        &layer.parameter("weight")?.tensor().to_vec()?,
        &[0.99, -0.01, -0.01, 0.99, 1.9, -0.1, -0.1, 1.9],
    );

    // One target batch is intentionally shared across the population axis.
    let logits = Tensor::from_slice(
        &[2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0],
        [batch.of(2), population.of(2), class.of(2)],
        &device,
    )?;
    let targets = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], [batch.of(2), class.of(2)], &device)?;
    let losses = logits.categorical_cross_entropy_with_logits(&targets, class)?;
    assert_eq!(
        losses.shape(),
        &Shape::new([batch.of(2), population.of(2)])?
    );
    close(
        "population shared targets",
        &losses.to_vec()?,
        &[
            (1.0_f64 + (-2.0_f64).exp()).ln(),
            (1.0_f64 + 2.0_f64.exp()).ln(),
            (1.0_f64 + (-2.0_f64).exp()).ln(),
            (1.0_f64 + 2.0_f64.exp()).ln(),
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn layer_norm_and_gelu_match_a_scalar_reference() -> Result<()> {
    fn scalar(values: &[f64]) -> f64 {
        let mut total = 0.0;
        for row in values.chunks_exact(3) {
            let mean = row.iter().sum::<f64>() / 3.0;
            let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / 3.0;
            for value in row {
                let x = (value - mean) / (variance + 1e-5).sqrt();
                total += 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh());
            }
        }
        total / values.len() as f64
    }

    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    let values = [-1.5_f64, 0.25, 2.0, 3.0, -2.0, 0.5];
    let input = Tensor::from_slice(
        &values.map(|value| value as f32),
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();
    let mut norm = LayerNorm::new(feature)?;
    norm.build(input.shape(), &device, 0)?;
    let output = norm.forward(&input)?.gelu()?;

    let mut expected = Vec::new();
    for row in values.chunks_exact(3) {
        let mean = row.iter().sum::<f64>() / 3.0;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / 3.0;
        for value in row {
            let x = (value - mean) / (variance + 1e-5).sqrt();
            expected.push(0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()));
        }
    }
    close("LayerNorm plus GELU forward", &output.to_vec()?, &expected);
    output.mean([batch, feature])?.backward()?;
    let epsilon = 1e-4;
    let finite_difference = (0..values.len())
        .map(|index| {
            let mut plus = values;
            let mut minus = values;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            (scalar(&plus) - scalar(&minus)) / (2.0 * epsilon)
        })
        .collect::<Vec<_>>();
    close(
        "LayerNorm plus GELU gradient",
        &input.grad().unwrap().to_vec()?,
        &finite_difference,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn exact_gelu_matches_independent_quadrature_and_gradient() -> Result<()> {
    // Simpson quadrature of the normal density: deliberately independent of
    // the backend's rational CDF evaluation and the existing tanh GELU oracle.
    fn cdf(x: f64) -> f64 {
        if x.abs() > 10.0 {
            return if x > 0.0 { 1.0 } else { 0.0 };
        }
        let step = x / 2048.0;
        let density = |z: f64| (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let mut sum = density(0.0) + density(x);
        for i in 1..2048 {
            sum += (if i % 2 == 0 { 2.0 } else { 4.0 }) * density(i as f64 * step);
        }
        0.5 + step * sum / 3.0
    }
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("row"), Axis::new("col"));
    // Odd extent, partial CUDA tiles, dense central/tail coverage, signed zero,
    // very large finite values, and physical order different from logical order.
    let mut values = (0..8193)
        .map(|i| (i as f32 - 4096.0) / 512.0)
        .collect::<Vec<_>>();
    values[0] = -1000.0;
    values[1] = 1000.0;
    values[2] = -0.0;
    let input = Tensor::from_slice(&values, [row.of(3), col.of(2731)], &device)?.with_grad();
    let output = ExactGELU.forward(&input.with_layout([col, row])?)?;
    assert_eq!(output.extent(row)?, 3);
    assert_eq!(output.extent(col)?, 2731);
    let observed = output.to_vec()?;
    output.mean([row, col])?.backward()?;
    let gradients = input.grad().unwrap().to_vec()?;
    let mut max_forward = 0.0_f64;
    let mut max_backward = 0.0_f64;
    let mut max_tanh_difference = 0.0_f64;
    for ((&x, &y), &g) in values.iter().zip(&observed).zip(&gradients) {
        let x = f64::from(x);
        let p = cdf(x);
        let expected = x * p;
        let derivative = p + x * (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let forward = (f64::from(y) - expected).abs();
        let backward = (f64::from(g) * values.len() as f64 - derivative).abs();
        assert!(
            y.is_finite() && forward < 2e-6,
            "exact GELU({x}): {y} != {expected}"
        );
        assert!(
            g.is_finite() && backward < 5e-7,
            "exact GELU derivative({x}): error {backward}"
        );
        max_forward = max_forward.max(forward);
        max_backward = max_backward.max(backward);
        let tanh = 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh());
        max_tanh_difference = max_tanh_difference.max((tanh - expected).abs());
    }
    assert!(
        max_tanh_difference > 4e-4,
        "oracle must distinguish tanh and erf GELU"
    );
    println!(
        "exact GELU PASS n={} forward={max_forward:.3e} derivative={max_backward:.3e} tanh_gap={max_tanh_difference:.3e}",
        values.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn silu_and_leaky_relu_modules_match_independent_oracles() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("activation_row"), Axis::new("activation_col"));
    let values = (0..2051)
        .map(|index| (index as f64 - 1025.0) / 137.0)
        .collect::<Vec<_>>();
    let leaf = Tensor::from_slice(
        &values.iter().map(|&value| value as f32).collect::<Vec<_>>(),
        [row.of(7), col.of(293)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([col, row])?;

    let silu = SiLU.forward(&input)?;
    let expected_silu = values
        .iter()
        .map(|&x| x / (1.0 + (-x).exp()))
        .collect::<Vec<_>>();
    close("SiLU module forward", &silu.to_vec()?, &expected_silu);
    silu.mean([row, col])?.backward()?;
    let expected_silu_gradient = values
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            (sigmoid + x * sigmoid * (1.0 - sigmoid)) / values.len() as f64
        })
        .collect::<Vec<_>>();
    close(
        "SiLU module derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_silu_gradient,
    );

    let leaky_leaf = input.detach().with_layout([row, col])?.with_grad();
    let leaky_input = leaky_leaf.with_layout([col, row])?;
    let leaky = LeakyReLU::new(0.125)?.forward(&leaky_input)?;
    let expected_leaky = values
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.125 * x })
        .collect::<Vec<_>>();
    close(
        "LeakyReLU module forward",
        &leaky.to_vec()?,
        &expected_leaky,
    );
    leaky.mean([row, col])?.backward()?;
    let expected_leaky_gradient = values
        .iter()
        .map(|&x| {
            let derivative = if x > 0.0 { 1.0 } else { 0.125 };
            derivative / values.len() as f64
        })
        .collect::<Vec<_>>();
    close(
        "LeakyReLU module derivative",
        &leaky_leaf.grad().unwrap().to_vec()?,
        &expected_leaky_gradient,
    );

    assert!(LeakyReLU::new(-0.1).is_err());
    assert!(LeakyReLU::new(f32::NAN).is_err());
    assert!(input.leaky_relu(f32::INFINITY).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn sign_straight_through_forward_and_gradient_match_independent_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("feature");
    let values = [-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    // Independent oracle: `+1` where `x > 0`, otherwise `-1`, so `x == 0` maps
    // to `-1` -- this is bae's `torch.where(z > 0, 1.0, -1.0)`, not `torch.sign`.
    let expected = [-1.0_f64, -1.0, -1.0, 1.0, 1.0];
    // Straight-through backward: the gradient is the identity, so every element
    // gets the same upstream gradient regardless of its own sign or magnitude.
    let expected_gradient = [0.2_f64; 5];

    let leaf = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    let output = leaf.sign_straight_through()?;
    close(
        "sign_straight_through forward",
        &output.to_vec()?,
        &expected,
    );
    output.mean(feature)?.backward()?;
    close(
        "sign_straight_through straight-through derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    let module_leaf = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    let module_output = SignStraightThrough.forward(&module_leaf)?;
    close(
        "SignStraightThrough module forward",
        &module_output.to_vec()?,
        &expected,
    );
    module_output.mean(feature)?.backward()?;
    close(
        "SignStraightThrough module straight-through derivative",
        &module_leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA and AXIS_PYTHON with PyTorch"]
fn pytorch_activation_forward_and_gradient_parity() -> Result<()> {
    let Ok(python) = std::env::var("AXIS_PYTHON") else {
        println!("SKIP: set AXIS_PYTHON to a Python interpreter with PyTorch");
        return Ok(());
    };
    let script = r#"
import torch
import torch.nn.functional as F

values = [-8.0, -3.0, -1.0, -0.0, 0.0, 0.5, 2.0, 8.0]
operations = (
    ("exact_gelu", lambda x: F.gelu(x, approximate="none")),
    ("silu", F.silu),
    ("leaky_relu", lambda x: F.leaky_relu(x, negative_slope=0.125)),
)
for name, operation in operations:
    x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    y = operation(x)
    y.mean().backward()
    print(name + ":forward " + " ".join(format(float(v), ".9g") for v in y))
    print(name + ":gradient " + " ".join(format(float(v), ".9g") for v in x.grad))
"#;
    let output = std::process::Command::new(&python)
        .args(["-c", script])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "PyTorch oracle failed via {python}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let oracle = String::from_utf8(output.stdout)?;
    let mut lines = oracle.lines();
    let mut expected = |label: &str| -> Result<Vec<f64>> {
        let line = lines
            .next()
            .ok_or_else(|| format!("PyTorch oracle omitted {label}"))?;
        let (observed_label, values) = line
            .split_once(' ')
            .ok_or_else(|| format!("malformed PyTorch oracle line: {line}"))?;
        if observed_label != label {
            return Err(
                format!("PyTorch oracle emitted {observed_label}, expected {label}").into(),
            );
        }
        values
            .split_whitespace()
            .map(|value| value.parse::<f64>().map_err(Into::into))
            .collect()
    };

    let device = Device::cuda(0)?;
    let feature = Axis::new("pytorch_parity_feature");
    let values = [-8.0, -3.0, -1.0, -0.0, 0.0, 0.5, 2.0, 8.0];
    for (name, module) in [
        ("exact_gelu", &ExactGELU as &dyn Module),
        ("silu", &SiLU as &dyn Module),
        ("leaky_relu", &LeakyReLU::new(0.125)? as &dyn Module),
    ] {
        let input = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
        let output = module.forward(&input)?;
        close(
            &format!("PyTorch {name} forward"),
            &output.to_vec()?,
            &expected(&format!("{name}:forward"))?,
        );
        output.mean(feature)?.backward()?;
        close(
            &format!("PyTorch {name} gradient"),
            &input.grad().unwrap().to_vec()?,
            &expected(&format!("{name}:gradient"))?,
        );
    }
    assert!(
        lines.next().is_none(),
        "PyTorch oracle emitted extra output"
    );
    println!("PyTorch activation parity PASS via {python}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA and AXIS_PYTHON with CUDA PyTorch"]
fn pytorch_activation_forward_performance() -> Result<()> {
    let Ok(python) = std::env::var("AXIS_PYTHON") else {
        println!("SKIP: set AXIS_PYTHON to a Python interpreter with CUDA PyTorch");
        return Ok(());
    };
    const ELEMENTS: usize = 1_048_576;
    const WARMUPS: usize = 8;
    const ITERATIONS: usize = 40;
    let script = format!(
        r#"
import time
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    raise RuntimeError("CUDA PyTorch is required")
x = torch.linspace(-8.0, 8.0, {ELEMENTS}, dtype=torch.float32, device="cuda")
operations = (
    ("exact_gelu", lambda value: F.gelu(value, approximate="none")),
    ("silu", F.silu),
    ("leaky_relu", lambda value: F.leaky_relu(value, negative_slope=0.125)),
)
with torch.inference_mode():
    for name, operation in operations:
        for _ in range({WARMUPS}):
            operation(x)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range({ITERATIONS}):
            operation(x)
        torch.cuda.synchronize()
        print(name + " " + format((time.perf_counter() - started) / {ITERATIONS}, ".12g"))
"#
    );
    let output = std::process::Command::new(&python)
        .args(["-c", &script])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "PyTorch benchmark failed via {python}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let pytorch = String::from_utf8(output.stdout)?;
    let mut pytorch_lines = pytorch.lines();

    let device = Device::cuda(0)?;
    let feature = Axis::new("pytorch_performance_feature");
    let values = (0..ELEMENTS)
        .map(|index| -8.0 + 16.0 * index as f32 / (ELEMENTS - 1) as f32)
        .collect::<Vec<_>>();
    let input = Tensor::from_slice(&values, [feature.of(ELEMENTS)], &device)?;
    for (name, module) in [
        ("exact_gelu", &ExactGELU as &dyn Module),
        ("silu", &SiLU as &dyn Module),
        ("leaky_relu", &LeakyReLU::new(0.125)? as &dyn Module),
    ] {
        for _ in 0..WARMUPS {
            std::hint::black_box(module.forward(&input)?);
        }
        device.synchronize()?;
        let started = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(module.forward(&input)?);
        }
        device.synchronize()?;
        let axis_seconds = started.elapsed().as_secs_f64() / ITERATIONS as f64;

        let line = pytorch_lines
            .next()
            .ok_or_else(|| format!("PyTorch benchmark omitted {name}"))?;
        let (observed_name, seconds) = line
            .split_once(' ')
            .ok_or_else(|| format!("malformed PyTorch benchmark line: {line}"))?;
        if observed_name != name {
            return Err(
                format!("PyTorch benchmark emitted {observed_name}, expected {name}").into(),
            );
        }
        let pytorch_seconds = seconds.parse::<f64>()?;
        let ratio = axis_seconds / pytorch_seconds;
        assert!(
            axis_seconds.is_finite() && pytorch_seconds.is_finite() && pytorch_seconds > 0.0,
            "invalid benchmark timing for {name}"
        );
        println!(
            "performance {name}: Axis={:.3} ms PyTorch={:.3} ms ratio={ratio:.2}x elements={ELEMENTS}",
            axis_seconds * 1e3,
            pytorch_seconds * 1e3,
        );
    }
    assert!(
        pytorch_lines.next().is_none(),
        "PyTorch benchmark emitted extra output"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multi_axis_layer_and_rms_norm_match_scalar_forward_and_gradient_oracles() -> Result<()> {
    fn layer_reference(
        values: &[f64],
        scale: &[f64],
        bias: &[f64],
        target: &[f64],
    ) -> (Vec<f64>, f64) {
        let mut output = vec![0.0; values.len()];
        for (batch, row) in values.chunks_exact(6).enumerate() {
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / row.len() as f64;
            let inverse_standard_deviation = (variance + 1e-5).sqrt().recip();
            for (coordinate, value) in row.iter().enumerate() {
                output[batch * 6 + coordinate] =
                    (value - mean) * inverse_standard_deviation * scale[coordinate]
                        + bias[coordinate];
            }
        }
        let loss = mean_squared(&output, target);
        (output, loss)
    }

    fn rms_reference(values: &[f64], scale: &[f64], target: &[f64]) -> (Vec<f64>, f64) {
        let mut output = vec![0.0; values.len()];
        for (batch, row) in values.chunks_exact(6).enumerate() {
            let mean_square = row.iter().map(|value| value * value).sum::<f64>() / row.len() as f64;
            let inverse_rms = (mean_square + 1e-6).sqrt().recip();
            for (coordinate, value) in row.iter().enumerate() {
                output[batch * 6 + coordinate] = value * inverse_rms * scale[coordinate];
            }
        }
        let loss = mean_squared(&output, target);
        (output, loss)
    }

    let device = Device::cuda(0)?;
    let (batch, row, feature) = (Axis::new("batch"), Axis::new("row"), Axis::new("feature"));
    let values = vec![
        -1.5_f64, 0.25, 2.0, 3.0, -2.0, 0.5, 1.25, -0.75, 0.1, 2.5, 1.0, -3.0,
    ];
    let scale = vec![0.5_f64, -1.0, 1.5, 0.75, -0.25, 2.0];
    let bias = vec![0.1_f64, -0.2, 0.3, -0.4, 0.5, -0.6];
    let target = vec![
        0.2_f64, -0.1, 0.4, 0.8, -0.3, 0.5, -0.7, 0.9, 0.6, -0.5, 0.25, -0.8,
    ];
    let dims = [batch.of(2), row.of(2), feature.of(3)];
    let input = Tensor::from_slice(
        &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?
    .with_layout([feature, batch, row])?
    .with_grad();
    let mut layer = LayerNorm::new([row, feature])?;
    layer.build(input.shape(), &device, 0)?;
    assert_eq!(
        layer
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["scale", "bias"]
    );
    layer
        .parameter("scale")?
        .set_values(&scale.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
    layer
        .parameter("bias")?
        .set_values(&bias.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
    let output = layer.forward(&input)?;
    let (expected, _) = layer_reference(&values, &scale, &bias, &target);
    close("multi-axis LayerNorm forward", &output.to_vec()?, &expected);
    let target_tensor = Tensor::from_slice(
        &target.iter().map(|value| *value as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?;
    output
        .squared_error(&target_tensor)?
        .mean([batch, row, feature])?
        .backward()?;
    close(
        "multi-axis LayerNorm input gradient",
        &input.grad().unwrap().to_vec()?,
        &central_difference(&values, 1e-4, |candidate| {
            layer_reference(candidate, &scale, &bias, &target).1
        }),
    );
    close(
        "multi-axis LayerNorm scale gradient",
        &layer.parameter("scale")?.grad().unwrap().to_vec()?,
        &central_difference(&scale, 1e-4, |candidate| {
            layer_reference(&values, candidate, &bias, &target).1
        }),
    );
    close(
        "multi-axis LayerNorm bias gradient",
        &layer.parameter("bias")?.grad().unwrap().to_vec()?,
        &central_difference(&bias, 1e-4, |candidate| {
            layer_reference(&values, &scale, candidate, &target).1
        }),
    );

    let rms_input = Tensor::from_slice(
        &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?
    .with_layout([row, feature, batch])?
    .with_grad();
    let mut rms = RmsNorm::new(vec![row, feature])?;
    rms.build(rms_input.shape(), &device, 0)?;
    assert_eq!(
        rms.named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["scale"]
    );
    rms.parameter("scale")?
        .set_values(&scale.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
    let rms_output = rms.forward(&rms_input)?;
    let (expected, _) = rms_reference(&values, &scale, &target);
    close(
        "multi-axis RmsNorm forward",
        &rms_output.to_vec()?,
        &expected,
    );
    rms_output
        .squared_error(&target_tensor)?
        .mean([batch, row, feature])?
        .backward()?;
    close(
        "multi-axis RmsNorm input gradient",
        &rms_input.grad().unwrap().to_vec()?,
        &central_difference(&values, 1e-4, |candidate| {
            rms_reference(candidate, &scale, &target).1
        }),
    );
    close(
        "multi-axis RmsNorm scale gradient",
        &rms.parameter("scale")?.grad().unwrap().to_vec()?,
        &central_difference(&scale, 1e-4, |candidate| {
            rms_reference(&values, candidate, &target).1
        }),
    );

    let constant = Tensor::from_slice(&[3.0; 12], dims, &device)?;
    let mut centered = LayerNorm::new([row, feature])?.affine(false);
    centered.build(constant.shape(), &device, 0)?;
    assert!(centered.parameters().is_empty());
    close(
        "constant LayerNorm",
        &centered.forward(&constant)?.to_vec()?,
        &[0.0; 12],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn group_and_instance_norm_match_scalar_oracles_and_each_other() -> Result<()> {
    fn group_reference(
        values: &[f64],
        groups: usize,
        scale: &[f64],
        bias: &[f64],
        target: &[f64],
    ) -> (Vec<f64>, f64) {
        const CHANNELS: usize = 4;
        const SAMPLE_SIZE: usize = 4;
        let channels_per_group = CHANNELS / groups;
        let mut output = vec![0.0; values.len()];
        for batch in 0..2 {
            for group in 0..groups {
                let channels = group * channels_per_group..(group + 1) * channels_per_group;
                let selected = channels
                    .clone()
                    .flat_map(|channel| {
                        (0..SAMPLE_SIZE)
                            .map(move |sample| (batch * CHANNELS + channel) * SAMPLE_SIZE + sample)
                    })
                    .collect::<Vec<_>>();
                let mean = selected.iter().map(|&index| values[index]).sum::<f64>()
                    / selected.len() as f64;
                let variance = selected
                    .iter()
                    .map(|&index| (values[index] - mean).powi(2))
                    .sum::<f64>()
                    / selected.len() as f64;
                let inverse_standard_deviation = (variance + 1e-5).sqrt().recip();
                for index in selected {
                    let channel = (index / SAMPLE_SIZE) % CHANNELS;
                    output[index] =
                        (values[index] - mean) * inverse_standard_deviation * scale[channel]
                            + bias[channel];
                }
            }
        }
        let loss = mean_squared(&output, target);
        (output, loss)
    }

    fn instance_reference(
        values: &[f64],
        scale: &[f64],
        bias: &[f64],
        target: &[f64],
    ) -> (Vec<f64>, f64) {
        const CHANNELS: usize = 4;
        const SAMPLE_SIZE: usize = 4;
        let mut output = vec![0.0; values.len()];
        for batch in 0..2 {
            for channel in 0..CHANNELS {
                let start = (batch * CHANNELS + channel) * SAMPLE_SIZE;
                let samples = &values[start..start + SAMPLE_SIZE];
                let mean = samples.iter().sum::<f64>() / SAMPLE_SIZE as f64;
                let variance = samples
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / SAMPLE_SIZE as f64;
                let inverse_standard_deviation = (variance + 1e-5).sqrt().recip();
                for sample in 0..SAMPLE_SIZE {
                    output[start + sample] =
                        (samples[sample] - mean) * inverse_standard_deviation * scale[channel]
                            + bias[channel];
                }
            }
        }
        let loss = mean_squared(&output, target);
        (output, loss)
    }

    let device = Device::cuda(0)?;
    let (batch, channel, height, width) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let values = vec![
        -2.0_f64, -1.5, -1.0, -0.5, 0.25, 0.75, 1.25, 1.75, 2.0, -0.25, 0.5, -1.0, 3.0, 2.0, 1.0,
        0.0, -0.2, 0.4, -0.6, 0.8, 1.1, -1.3, 1.5, -1.7, 2.2, 2.4, -2.6, -2.8, 0.3, 0.6, 0.9, 1.2,
    ];
    let scale = vec![0.5_f64, -1.25, 0.75, 1.5];
    let bias = vec![0.1_f64, -0.2, 0.3, -0.4];
    let target = (0..32)
        .map(|index| (index as f64 * 0.17).sin() * 0.5)
        .collect::<Vec<_>>();
    let dims = [batch.of(2), channel.of(4), height.of(2), width.of(2)];
    let input = Tensor::from_slice(
        &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?
    .with_layout([height, channel, width, batch])?
    .with_grad();
    let mut group = GroupNorm::new(channel, 2, [height, width])?;
    group.build(input.shape(), &device, 0)?;
    assert_eq!(
        group
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["scale", "bias"]
    );
    group
        .parameter("scale")?
        .set_values(&scale.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
    group
        .parameter("bias")?
        .set_values(&bias.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
    let output = group.forward(&input)?;
    let (expected, _) = group_reference(&values, 2, &scale, &bias, &target);
    close("GroupNorm forward", &output.to_vec()?, &expected);
    let target_tensor = Tensor::from_slice(
        &target.iter().map(|value| *value as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?;
    output
        .squared_error(&target_tensor)?
        .mean([batch, channel, height, width])?
        .backward()?;
    close(
        "GroupNorm input gradient",
        &input.grad().unwrap().to_vec()?,
        &central_difference(&values, 1e-4, |candidate| {
            group_reference(candidate, 2, &scale, &bias, &target).1
        }),
    );
    close(
        "GroupNorm scale gradient",
        &group.parameter("scale")?.grad().unwrap().to_vec()?,
        &central_difference(&scale, 1e-4, |candidate| {
            group_reference(&values, 2, candidate, &bias, &target).1
        }),
    );
    close(
        "GroupNorm bias gradient",
        &group.parameter("bias")?.grad().unwrap().to_vec()?,
        &central_difference(&bias, 1e-4, |candidate| {
            group_reference(&values, 2, &scale, candidate, &target).1
        }),
    );

    for groups in [1, 4] {
        let mut norm = GroupNorm::new(channel, groups, [height, width])?.affine(false);
        norm.build(input.shape(), &device, 0)?;
        let actual = norm.forward(&input)?.to_vec()?;
        let (expected, _) = group_reference(&values, groups, &[1.0; 4], &[0.0; 4], &target);
        close(&format!("GroupNorm groups={groups}"), &actual, &expected);
    }

    let mut instance = InstanceNorm::new(channel, [height, width])?;
    instance.build(input.shape(), &device, 0)?;
    assert!(instance.parameters().is_empty());
    let instance_output = instance.forward(&input)?;
    let (expected, _) = instance_reference(&values, &[1.0; 4], &[0.0; 4], &target);
    close(
        "InstanceNorm independent scalar forward",
        &instance_output.to_vec()?,
        &expected,
    );
    let mut equivalent = GroupNorm::new(channel, 4, [height, width])?.affine(false);
    equivalent.build(input.shape(), &device, 0)?;
    close(
        "InstanceNorm equals per-channel GroupNorm",
        &instance_output.to_vec()?,
        &equivalent
            .forward(&input)?
            .to_vec()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
    );

    let instance_input = Tensor::from_slice(
        &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?
    .with_layout([width, batch, height, channel])?
    .with_grad();
    let mut affine_instance = InstanceNorm::new(channel, vec![height, width])?.affine(true);
    affine_instance.build(instance_input.shape(), &device, 0)?;
    assert_eq!(
        affine_instance
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["scale", "bias"]
    );
    affine_instance
        .parameter("scale")?
        .set_values(&scale.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
    affine_instance
        .parameter("bias")?
        .set_values(&bias.iter().map(|value| *value as f32).collect::<Vec<_>>())?;
    let affine_output = affine_instance.forward(&instance_input)?;
    let (expected, _) = instance_reference(&values, &scale, &bias, &target);
    close(
        "affine InstanceNorm independent scalar forward",
        &affine_output.to_vec()?,
        &expected,
    );
    affine_output
        .squared_error(&target_tensor)?
        .mean([batch, channel, height, width])?
        .backward()?;
    close(
        "affine InstanceNorm input gradient",
        &instance_input.grad().unwrap().to_vec()?,
        &central_difference(&values, 1e-4, |candidate| {
            instance_reference(candidate, &scale, &bias, &target).1
        }),
    );
    close(
        "affine InstanceNorm scale gradient",
        &affine_instance
            .parameter("scale")?
            .grad()
            .unwrap()
            .to_vec()?,
        &central_difference(&scale, 1e-4, |candidate| {
            instance_reference(&values, candidate, &bias, &target).1
        }),
    );
    close(
        "affine InstanceNorm bias gradient",
        &affine_instance
            .parameter("bias")?
            .grad()
            .unwrap()
            .to_vec()?,
        &central_difference(&bias, 1e-4, |candidate| {
            instance_reference(&values, &scale, candidate, &target).1
        }),
    );

    let constant_values = [
        1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0, 3.0, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5, 0.5, 4.0,
        4.0, 4.0, 4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0,
    ];
    let constant = Tensor::from_slice(&constant_values, dims, &device)?;
    close(
        "constant InstanceNorm",
        &instance.forward(&constant)?.to_vec()?,
        &[0.0; 32],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn masked_mean_normalizes_only_selected_elements() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, position) = (Axis::new("batch"), Axis::new("position"));
    let values = Tensor::from_slice(
        &[1.0, 100.0, 3.0, 100.0],
        [batch.of(2), position.of(2)],
        &device,
    )?
    .with_grad();
    let mask = Tensor::from_slice(
        &[1.0, 1.0, 0.0, 0.0],
        [position.of(2), batch.of(2)],
        &device,
    )?;
    let mean = values.masked_mean(&mask)?;
    close("masked mean", &mean.to_vec()?, &[2.0]);
    mean.backward()?;
    close(
        "masked mean gradient",
        &values.grad().unwrap().to_vec()?,
        &[0.5, 0.0, 0.5, 0.0],
    );
    let empty = Tensor::from_slice(&[0.0; 4], [batch.of(2), position.of(2)], &device)?;
    assert!(values.detach().masked_mean(&empty).is_err());
    let fractional = Tensor::from_slice(
        &[1.0, 0.5, 1.0, 0.0],
        [batch.of(2), position.of(2)],
        &device,
    )?;
    assert!(values.detach().masked_mean(&fractional).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn trainer_enforces_scalar_losses_and_drives_each_optimizer() -> Result<()> {
    fn train_once<O: Optimizer>(device: &Device, optimizer: O) -> Result<(usize, f32)> {
        let (batch, input_axis, output) =
            (Axis::new("batch"), Axis::new("input"), Axis::new("output"));
        let input = Tensor::from_slice(
            &[1.0, -1.0, 0.5, 2.0],
            [batch.of(2), input_axis.of(2)],
            device,
        )?;
        let target = Tensor::from_slice(&[0.25, -0.5], [batch.of(2), output.of(1)], device)?;
        let mut model = Linear::new(input_axis, output.of(1));
        model.build(input.shape(), device, 17)?;
        let mut trainer = Trainer::new(optimizer);

        assert_eq!(trainer.completed_steps(), 0);
        let _ = trainer.optimizer();
        let _ = trainer.optimizer_mut();
        let error = trainer
            .step(&mut model, |model| model.forward(&input))
            .err()
            .expect("non-scalar loss must fail")
            .to_string();
        assert!(error.contains("must return a scalar"), "{error}");
        assert_eq!(trainer.completed_steps(), 0);

        let step = trainer.step(&mut model, |model| {
            model
                .forward(&input)?
                .squared_error(&target)?
                .mean([batch, output])
        })?;
        Ok((step.step(), step.pre_update_loss()?))
    }

    let device = Device::cuda(0)?;
    for result in [
        train_once(&device, SGD::new(0.01)?)?,
        train_once(&device, Adam::new(0.01)?)?,
        train_once(&device, AdamW::new(0.01, 0.001)?)?,
    ] {
        assert_eq!(result.0, 1);
        assert!(result.1.is_finite() && result.1 >= 0.0);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn sine_and_tanh_module_match_independent_forward_and_gradient_oracles() -> Result<()> {
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let values = [
        -std::f32::consts::FRAC_PI_2,
        -0.4,
        0.0,
        0.7,
        std::f32::consts::FRAC_PI_2,
    ];
    let input = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();
    let sine = input.sin()?;
    close(
        "sine forward",
        &sine.to_vec()?,
        &values
            .iter()
            .map(|&value| f64::from(value).sin())
            .collect::<Vec<_>>(),
    );
    sine.mean(sample)?.backward()?;
    close(
        "sine derivative",
        &input.grad().unwrap().to_vec()?,
        &values
            .iter()
            .map(|&value| f64::from(value).cos() / values.len() as f64)
            .collect::<Vec<_>>(),
    );

    let tanh_input = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();
    let mut tanh = Tanh;
    assert_eq!(
        tanh.build(tanh_input.shape(), &device, 0)?,
        tanh_input.shape().clone()
    );
    let output = tanh.forward(&tanh_input)?;
    close(
        "Tanh module forward",
        &output.to_vec()?,
        &values
            .iter()
            .map(|&value| f64::from(value).tanh())
            .collect::<Vec<_>>(),
    );
    output.mean(sample)?.backward()?;
    close(
        "Tanh module derivative",
        &tanh_input.grad().unwrap().to_vec()?,
        &values
            .iter()
            .map(|&value| {
                let y = f64::from(value).tanh();
                (1.0 - y * y) / values.len() as f64
            })
            .collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn binary_alignment_of_permuted_layouts_matches_values_and_gradients() -> Result<()> {
    // Operands that share axes in different orders and layouts are aligned through the
    // compact copier plus a metadata view; there is no element-sized index plan.
    let device = Device::cuda(0)?;
    let (batch, time, unit) = (Axis::new("batch"), Axis::new("time"), Axis::new("unit"));
    let lhs_values: Vec<_> = (0..12).map(|v| v as f32 * 0.5 - 2.0).collect();
    let rhs_values: Vec<_> = (0..12).map(|v| 1.0 - v as f32 * 0.25).collect();
    let lhs = Tensor::from_slice(&lhs_values, [batch.of(2), time.of(3), unit.of(2)], &device)?
        .with_layout([unit, batch, time])?
        .with_grad();
    let rhs = Tensor::from_slice(&rhs_values, [time.of(3), unit.of(2), batch.of(2)], &device)?
        .with_grad();
    let sum = lhs.add(&rhs)?;
    let out = sum.shape().clone();
    let actual = sum.to_vec()?;
    let mut expected = Vec::with_capacity(12);
    for index in 0..out.len() {
        let coords = out.coords(index);
        let at = |axis: Axis| coords[out.index(axis).expect("shared axis")];
        let (b, t, u) = (at(batch), at(time), at(unit));
        expected.push(f64::from(
            lhs_values[(b * 3 + t) * 2 + u] + rhs_values[(t * 2 + u) * 2 + b],
        ));
    }
    close("aligned binary values", &actual, &expected);
    sum.mean([batch, time, unit])?.backward()?;
    close(
        "aligned lhs gradient",
        &lhs.grad().expect("lhs gradient").to_vec()?,
        &[1.0 / 12.0; 12],
    );
    close(
        "aligned rhs gradient",
        &rhs.grad().expect("rhs gradient").to_vec()?,
        &[1.0 / 12.0; 12],
    );
    // A singleton extent keeps the zero-copy identity view and still aligns.
    let thin = Tensor::from_slice(&[1.0, 2.0, 3.0], [time.of(3), unit.of(1)], &device)?
        .with_layout([unit, time])?;
    let wide = Tensor::from_slice(&[10.0, 20.0, 30.0], [unit.of(1), time.of(3)], &device)?;
    close(
        "singleton alignment",
        &thin.add(&wide)?.to_vec()?,
        &[11.0, 22.0, 33.0],
    );
    // Broadcasting a bias over a larger, permuted operand: stride-0 forward, summed gradient.
    let bias = Tensor::from_slice(&[0.5, -1.5], [unit.of(2)], &device)?.with_grad();
    let field = Tensor::from_slice(
        &(0..6).map(|v| v as f32).collect::<Vec<_>>(),
        [batch.of(3), unit.of(2)],
        &device,
    )?
    .with_layout([unit, batch])?
    .with_grad();
    let biased = field.add(&bias)?;
    let out = biased.shape().clone();
    let mut expected = Vec::with_capacity(6);
    for index in 0..out.len() {
        let coords = out.coords(index);
        let (b, u) = (coords[out.index(batch)?], coords[out.index(unit)?]);
        expected.push(f64::from((b * 2 + u) as f32 + [0.5f32, -1.5][u]));
    }
    close("broadcast alignment values", &biased.to_vec()?, &expected);
    biased.mean([batch, unit])?.backward()?;
    close(
        "broadcast bias gradient",
        &bias.grad().expect("bias gradient").to_vec()?,
        &[0.5, 0.5],
    );
    close(
        "broadcast field gradient",
        &field.grad().expect("field gradient").to_vec()?,
        &[1.0 / 6.0; 6],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn elementwise_division_matches_hand_computed_forward_and_both_gradients() -> Result<()> {
    // world/energy-output and world/fluid both divide a per-feature sum by a
    // per-feature, data-dependent count; upscale_eval divides one MSE by
    // another. All three need forward plus the gradient with respect to BOTH
    // operands, matching hand-derived d/da = g/b and d/db = -g*a/b^2.
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let a_values = [2.0f32, -3.0, 4.5, 0.5];
    let b_values = [4.0f32, 1.5, -2.0, 0.25];
    let a = Tensor::from_slice(&a_values, [sample.of(4)], &device)?.with_grad();
    let b = Tensor::from_slice(&b_values, [sample.of(4)], &device)?.with_grad();
    let quotient = a.div(&b)?;
    close(
        "division forward",
        &quotient.to_vec()?,
        &[0.5, -2.0, -2.25, 2.0],
    );
    quotient.mean(sample)?.backward()?;
    close(
        "division numerator gradient",
        &a.grad().expect("numerator gradient").to_vec()?,
        &[0.0625, 1.0 / 6.0, -0.125, 1.0],
    );
    close(
        "division denominator gradient",
        &b.grad().expect("denominator gradient").to_vec()?,
        &[-0.03125, 1.0 / 3.0, -0.28125, -2.0],
    );

    // Masked-mean shape from energy-output's `losses / counts.clamp(min=1)`:
    // a per-column sum divided by a per-column count that the CONSUMER has
    // already clamped away from zero before calling div. Axis performs no
    // clamping or epsilon of its own; division by an actual zero yields IEEE
    // inf/nan, as in PyTorch.
    let feature = Axis::new("feature");
    let sums = Tensor::from_slice(&[10.0f32, 6.0, 0.0], [feature.of(3)], &device)?.with_grad();
    let counts = Tensor::from_slice(&[5.0f32, 1.0, 1.0], [feature.of(3)], &device)?.with_grad();
    let ratio = sums.div(&counts)?;
    close("masked-mean forward", &ratio.to_vec()?, &[2.0, 6.0, 0.0]);
    ratio.mean(feature)?.backward()?;
    close(
        "masked-mean sums gradient",
        &sums.grad().expect("sums gradient").to_vec()?,
        &[1.0 / 15.0, 1.0 / 3.0, 1.0 / 3.0],
    );
    close(
        "masked-mean counts gradient",
        &counts.grad().expect("counts gradient").to_vec()?,
        &[-2.0 / 15.0, -2.0, 0.0],
    );

    // Reordered-layout CUDA case: numerator and denominator share axes but
    // keep different storage orders, mirroring
    // binary_alignment_of_permuted_layouts for add.
    let (batch, time) = (Axis::new("batch"), Axis::new("time"));
    let num_values: Vec<_> = (0..6).map(|v| v as f32 + 1.0).collect();
    let den_values: Vec<_> = (0..6).map(|v| 2.0 + v as f32 * 0.5).collect();
    let numerator = Tensor::from_slice(&num_values, [batch.of(2), time.of(3)], &device)?
        .with_layout([time, batch])?
        .with_grad();
    let denominator =
        Tensor::from_slice(&den_values, [time.of(3), batch.of(2)], &device)?.with_grad();
    let reordered = numerator.div(&denominator)?;
    let shape = reordered.shape().clone();
    let actual = reordered.to_vec()?;
    let mut expected = Vec::with_capacity(6);
    for index in 0..shape.len() {
        let coords = shape.coords(index);
        let at = |axis: Axis| coords[shape.index(axis).expect("shared axis")];
        let (b, t) = (at(batch), at(time));
        let numerator_value = f64::from(num_values[b * 3 + t]);
        let denominator_value = f64::from(den_values[t * 2 + b]);
        expected.push(numerator_value / denominator_value);
    }
    close("reordered division values", &actual, &expected);
    Ok(())
}

#[test]
fn cosine_annealing_lr_matches_pytorch_closed_form_at_exact_angles() -> Result<()> {
    // torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=0.0) from a base
    // learning rate of 0.1: eta_t = eta_min + (eta_max - eta_min) * (1 + cos(pi * t / T_max)) / 2.
    // Steps 0..=4 land exactly on cos(0), cos(pi/4), cos(pi/2), cos(3pi/4), cos(pi)
    // = 1, sqrt(2)/2, 0, -sqrt(2)/2, -1 — well-known values, not approximated by the op
    // under test.
    let actual: Vec<f32> = (0u32..=4)
        .map(|step| cosine_annealing_lr(0.1, 0.0, 4, step))
        .collect::<Result<_>>()?;
    close(
        "cosine annealing at exact trig angles",
        &actual,
        &[0.1, 0.08535533905932738, 0.05, 0.014644660940672627, 0.0],
    );
    assert!(cosine_annealing_lr(0.1, 0.2, 4, 0).is_err());
    assert!(cosine_annealing_lr(0.1, 0.0, 0, 0).is_err());
    Ok(())
}

#[test]
fn one_cycle_lr_matches_pytorch_default_schedule() -> Result<()> {
    // torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.0, total_steps=10) at PyTorch's
    // own defaults (pct_start=0.3, div_factor=25, final_div_factor=1e4, anneal_strategy="cos"):
    // step_size_up = 0.3*10 - 1 = 2, step_size_down = (10-1) - 2 = 7; initial_lr = 1/25 = 0.04,
    // min_lr = 0.04/1e4 = 0.000004. Ten values, one per call to get_last_lr() after each
    // .step(), computed independently from the same public closed form (not the op under
    // test): warmup crosses exactly through max_lr at step 2 (the phase boundary), and the
    // final value at step 9 is the exact rational 1/25/10000.
    let actual: Vec<f32> = (0u32..10)
        .map(|step| one_cycle_lr(1.0, 10, step))
        .collect::<Result<_>>()?;
    close(
        "one-cycle default schedule",
        &actual,
        &[
            0.040000000000000036,
            0.52,
            1.0,
            0.9504846320134737,
            0.8117456539497631,
            0.6112620219362893,
            0.38874197806371075,
            0.18825834605023697,
            0.049519367986526286,
            0.000004,
        ],
    );
    assert!(one_cycle_lr(1.0, 10, 10).is_err());
    assert!(one_cycle_lr(1.0, 1, 0).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn trainer_driven_sgd_consumes_a_cosine_annealing_schedule_each_step() -> Result<()> {
    // The consumer's own training loop shape: call the pure schedule function, hand the
    // result to the optimizer's setter, then let Trainer::step order zero_grad/loss/
    // backward/optimizer.step as usual. This is the end-to-end witness that Trainer and SGD
    // actually consume the scheduled rate, not just that the schedule function is correct in
    // isolation (that is covered independently above).
    struct ConstantGradientParameter(Parameter);
    impl Module for ConstantGradientParameter {
        fn output_shape(&self, input: &Shape) -> Result<Shape> {
            Ok(input.clone())
        }
        fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
            Ok(input.clone())
        }
        fn forward(&self, _input: &Tensor) -> Result<Tensor> {
            Ok(self.0.tensor())
        }
        fn named_parameters(&self) -> Vec<(String, Parameter)> {
            vec![("weight".into(), self.0.clone())]
        }
    }

    let device = Device::cuda(0)?;
    let feature = Axis::new("feature");
    let parameter = Parameter::new(Tensor::from_slice(&[1.0, 2.0], [feature.of(2)], &device)?);
    let dummy = Tensor::from_slice(&[0.0, 0.0], [feature.of(2)], &device)?;
    let mut model = ConstantGradientParameter(parameter.clone());
    let mut trainer = Trainer::new(SGD::new(1.0)?);

    // eta_max=0.5, eta_min=0.1, t_max=2: step 0 -> 0.5 (cos(0)=1), step 1 -> 0.3 (cos(pi/2)=0).
    let mut expected = [1.0_f64, 2.0_f64];
    for step in 0..2u32 {
        let rate = cosine_annealing_lr(0.5, 0.1, 2, step)?;
        trainer.optimizer_mut().set_learning_rate(rate)?;
        trainer.step(&mut model, |model| model.forward(&dummy)?.mean(feature))?;
        // The identity forward's mean over 2 elements has a constant gradient of 1/2 per
        // element regardless of the parameter's value, so the expected update is exact
        // arithmetic independent of both the schedule and SGD implementations.
        for value in &mut expected {
            *value -= f64::from(rate) * 0.5;
        }
        close(
            &format!("trainer-driven SGD step {step} at scheduled rate {rate}"),
            &parameter.tensor().to_vec()?,
            &expected,
        );
    }
    assert_eq!(trainer.completed_steps(), 2);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_concat_matches_independent_values_gradients_and_composed_paths() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature, group, combined, missing) = (
        Axis::new("batch"),
        Axis::new("feature"),
        Axis::new("group"),
        Axis::new("combined"),
        Axis::new("missing"),
    );

    // Three unequal-width operands (mirrors the gastric PhaseSeparableFusion
    // 5*256+3+512 unequal-width cat): declared axis order [batch, feature] on
    // the first operand, a physically permuted layout on the second (same
    // declared order, transposed storage), and a swapped declared axis order
    // on the third, checking that the output follows the FIRST operand only.
    let a = Tensor::from_slice(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();
    let b = Tensor::from_slice(
        &[10.0, 11.0, 12.0, 13.0],
        [batch.of(2), feature.of(2)],
        &device,
    )?
    .with_layout([feature, batch])?
    .with_grad();
    let c = Tensor::from_slice(&[100.0, 101.0], [feature.of(1), batch.of(2)], &device)?.with_grad();

    let concatenated = Tensor::concat(&[a.clone(), b.clone(), c.clone()], feature)?;
    assert_eq!(
        concatenated.shape(),
        &Shape::new([batch.of(2), feature.of(6)])?
    );
    close(
        "unequal-width concat values",
        &concatenated.to_vec()?,
        &[
            0.0, 1.0, 2.0, 10.0, 11.0, 100.0, 3.0, 4.0, 5.0, 12.0, 13.0, 101.0,
        ],
    );

    let upstream = Tensor::from_slice(
        &(1..=12).map(|v| v as f32).collect::<Vec<_>>(),
        [batch.of(2), feature.of(6)],
        &device,
    )?;
    concatenated
        .mul(&upstream)?
        .mean([batch, feature])?
        .backward()?;
    close(
        "unequal-width concat gradient a",
        &a.grad().expect("a gradient").to_vec()?,
        &[
            1.0 / 12.0,
            2.0 / 12.0,
            3.0 / 12.0,
            7.0 / 12.0,
            8.0 / 12.0,
            9.0 / 12.0,
        ],
    );
    close(
        "unequal-width concat gradient b",
        &b.grad().expect("b gradient").to_vec()?,
        &[4.0 / 12.0, 5.0 / 12.0, 10.0 / 12.0, 11.0 / 12.0],
    );
    close(
        "unequal-width concat gradient c",
        &c.grad().expect("c gradient").to_vec()?,
        &[6.0 / 12.0, 12.0 / 12.0],
    );

    // Equal-width cross-check: concat must agree bit-for-bit with the
    // already-proven stack+merge composition (group outer, feature inner).
    let x = Tensor::from_slice(
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        [batch.of(2), feature.of(3)],
        &device,
    )?;
    let y = Tensor::from_slice(
        &[7.0, -8.0, 9.0, -10.0, 11.0, -12.0],
        [batch.of(2), feature.of(3)],
        &device,
    )?;
    let by_concat = Tensor::concat(&[x.clone(), y.clone()], feature)?;
    let stacked = Tensor::stack(&[x, y], group, 2)?;
    let by_stack_and_merge = stacked.merge([group, feature], combined)?;
    assert_eq!(by_concat.to_vec()?, by_stack_and_merge.to_vec()?);

    // Rejections: axis must already exist (unlike `stack`), axis sets must
    // match exactly, and with more than two operands a mismatched
    // non-concat-axis extent must be rejected before any device work.
    assert!(Tensor::concat(std::slice::from_ref(&a), missing).is_err());
    let wrong_axes = Tensor::from_slice(&[0.0, 1.0], [missing.of(2)], &device)?;
    assert!(Tensor::concat(&[a.clone(), wrong_axes], feature).is_err());
    let mismatched_batch =
        Tensor::from_slice(&[0.0, 1.0, 2.0], [batch.of(1), feature.of(3)], &device)?;
    assert!(Tensor::concat(&[a.clone(), b.clone(), mismatched_batch], feature).is_err());
    println!("concat values, gradients, and composed-path cross-check PASS");
    Ok(())
}

#[test]
fn max_pool_rejects_invalid_configuration_before_launch() {
    let (channel, height, width, depth) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("depth"),
    );
    let shape = Shape::new([channel.of(2), height.of(4), width.of(4)]).unwrap();

    // Padding more than half the kernel extent: PyTorch's own `MaxPool2d` constraint, which
    // also guarantees every window keeps at least one real, unpadded element.
    let error = MaxPool2d::new(channel, [height, width], [2, 2])
        .padding([2, 0])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("padding must be at most half"), "{error}");

    // The channel axis cannot also be a spatial axis.
    let error = MaxPool2d::new(channel, [channel, width], [2, 2])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("distinct"), "{error}");

    // Kernel and stride extents must be positive.
    let error = MaxPool2d::new(channel, [height, width], [0, 2])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("positive"), "{error}");
    let error = MaxPool2d::new(channel, [height, width], [2, 2])
        .stride([0, 1])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("positive"), "{error}");

    // A kernel that does not fit even the padded extent is rejected.
    let error = MaxPool2d::new(channel, [height, width], [9, 9])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("fit"), "{error}");

    // The 3D entry point follows the same contract with its own three spatial axes.
    let volume = Shape::new([channel.of(2), depth.of(3), height.of(4), width.of(4)]).unwrap();
    let error = MaxPool3d::new(channel, [depth, height, width], [3, 3, 3])
        .padding([0, 2, 0])
        .output_shape(&volume)
        .unwrap_err()
        .to_string();
    assert!(error.contains("padding must be at most half"), "{error}");

    // A well-formed configuration reports the expected output shape, purely from shape math
    // (no device is needed: pooling has no parameters to build). Spatial axes are preserved in
    // place at their pooled extent; the channel axis is appended, exactly as `Conv2d` appends
    // its output-channel axis.
    let pool = MaxPool2d::new(channel, [height, width], [2, 2]);
    assert_eq!(
        pool.output_shape(&shape).unwrap(),
        Shape::new([height.of(2), width.of(2), channel.of(2)]).unwrap()
    );
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool2d_matches_hand_computed_forward_and_gradient_with_remainder_and_ties() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    // Height 3 with kernel 2 / stride 2 leaves row 2 as an uncovered remainder (PyTorch's own
    // default `ceil_mode=False` behavior). Channel 0's first window ties at 5.0; channel 1 has
    // no ties, to check ordinary gradient routing too.
    #[rustfmt::skip]
    let inputs: [f32; 24] = [
        1.0, 5.0, 2.0, 6.0,
        5.0, 3.0, 6.0, 0.5,
        9.0, 9.0, 9.0, 9.0,

        -1.0, -2.0, -3.0, -4.0,
        -5.0, -0.5, -6.0, -7.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let input = Tensor::from_slice(&inputs, [channel.of(2), height.of(3), width.of(4)], &device)?
        .with_grad();

    let mut pool = MaxPool2d::new(channel, [height, width], [2, 2]);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    // Spatial axes are preserved in place at their pooled extent; the channel axis is appended,
    // exactly as `Conv2d` appends its output-channel axis.
    assert_eq!(
        output_shape,
        Shape::new([height.of(1), width.of(2), channel.of(2)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "MaxPool2d forward",
        &actual.to_vec()?,
        &[5.0, -0.5, 6.0, -3.0],
    );

    actual.mean([channel, height, width])?.backward()?;
    #[rustfmt::skip]
    let expected_gradient: [f64; 24] = [
        0.0, 0.25, 0.0, 0.25,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.25, 0.0,
        0.0, 0.25, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    close(
        "MaxPool2d gradient (ties route to the first logical coordinate, the remainder row gets none)",
        &input.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool2d_with_padding_matches_hand_computed_forward_and_sums_overlapping_gradients()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    // Kernel 3 / stride 1 / padding 1 ("same" padding), the exact convention `morpheus`'s
    // peak-NMS decode and `gastric`'s morphological dilation both use. Every value is negative,
    // so a stray zero-fill at a padded position (instead of negative infinity) would win the
    // corner windows incorrectly and this oracle would fail.
    #[rustfmt::skip]
    let inputs: [f32; 9] = [
        -1.0, -9.0, -2.0,
        -8.0, -7.0, -9.0,
        -3.0, -9.0, -4.0,
    ];
    let input = Tensor::from_slice(&inputs, [channel.of(1), height.of(3), width.of(3)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();

    let mut pool = MaxPool2d::new(channel, [height, width], [3, 3])
        .stride([1, 1])
        .padding([1, 1]);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        output_shape,
        Shape::new([height.of(3), width.of(3), channel.of(1)])?
    );
    let actual = pool.forward(&input)?;
    #[rustfmt::skip]
    let expected: [f64; 9] = [
        -1.0, -1.0, -2.0,
        -1.0, -1.0, -2.0,
        -3.0, -3.0, -4.0,
    ];
    close("padded MaxPool2d forward", &actual.to_vec()?, &expected);

    actual.mean([channel, height, width])?.backward()?;
    // Source (0, 0) wins four overlapping windows, (0, 2) and (2, 0) each win two, and (2, 2)
    // wins one; `unfold`'s col2im backward must sum every overlapping contribution exactly.
    #[rustfmt::skip]
    let expected_gradient: [f64; 9] = [
        4.0 / 9.0, 0.0, 2.0 / 9.0,
        0.0, 0.0, 0.0,
        2.0 / 9.0, 0.0, 1.0 / 9.0,
    ];
    close(
        "padded MaxPool2d gradient (overlapping windows sum onto their shared source)",
        &input.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool3d_matches_gastric_kernel_and_sums_overlapping_gradients_per_channel() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, depth, height, width) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
    );
    // `gastric`'s `interface_region_masks_3d` dilation: kernel (1, 3, 3), stride 1,
    // padding (0, 1, 1) -- depth is a pure pass-through (no depth reduction, no depth padding),
    // and each channel pools independently. Channel 1 repeats channel 0's grid shifted by -100
    // so it shares the same argmax pattern under a different absolute scale.
    #[rustfmt::skip]
    let inputs: [f32; 18] = [
        -1.0, -9.0, -2.0,
        -8.0, -7.0, -9.0,
        -3.0, -9.0, -4.0,

        -101.0, -109.0, -102.0,
        -108.0, -107.0, -109.0,
        -103.0, -109.0, -104.0,
    ];
    let input = Tensor::from_slice(
        &inputs,
        [channel.of(2), depth.of(1), height.of(3), width.of(3)],
        &device,
    )?
    .with_grad();

    let mut pool = MaxPool3d::new(channel, [depth, height, width], [1, 3, 3])
        .stride([1, 1, 1])
        .padding([0, 1, 1]);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    // Spatial axes stay in place at their pooled extent; the channel axis is appended.
    assert_eq!(
        output_shape,
        Shape::new([depth.of(1), height.of(3), width.of(3), channel.of(2)])?
    );
    let actual = pool.forward(&input)?;
    #[rustfmt::skip]
    let expected: [f64; 18] = [
        -1.0, -101.0,   -1.0, -101.0,   -2.0, -102.0,
        -1.0, -101.0,   -1.0, -101.0,   -2.0, -102.0,
        -3.0, -103.0,   -3.0, -103.0,   -4.0, -104.0,
    ];
    close("MaxPool3d forward", &actual.to_vec()?, &expected);

    actual.mean([channel, depth, height, width])?.backward()?;
    #[rustfmt::skip]
    let expected_gradient: [f64; 18] = [
        4.0 / 18.0, 0.0, 2.0 / 18.0,
        0.0, 0.0, 0.0,
        2.0 / 18.0, 0.0, 1.0 / 18.0,

        4.0 / 18.0, 0.0, 2.0 / 18.0,
        0.0, 0.0, 0.0,
        2.0 / 18.0, 0.0, 1.0 / 18.0,
    ];
    close(
        "MaxPool3d gradient (each channel pools independently)",
        &input.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_avg_pool3d_matches_hand_computed_uneven_bins_and_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let (depth, height, width) = (Axis::new("depth"), Axis::new("height"), Axis::new("width"));
    // depth 2 -> 1 and width 1 -> 1 are trivial single-bin axes; height 5 -> 2 is PyTorch's own
    // textbook uneven case: bin 0 = [0, 3), bin 1 = [2, 5), sharing input index 2.
    #[rustfmt::skip]
    let inputs: [f32; 10] = [
        1.0, 2.0, 3.0, 4.0, 5.0,
        10.0, 20.0, 30.0, 40.0, 50.0,
    ];
    let input = Tensor::from_slice(&inputs, [depth.of(2), height.of(5), width.of(1)], &device)?
        .with_layout([width, height, depth])?
        .with_grad();

    let actual = input.adaptive_avg_pool3d([depth, height, width], [1, 2, 1])?;
    assert_eq!(
        actual.shape(),
        &Shape::new([depth.of(1), height.of(2), width.of(1)])?
    );
    close(
        "AdaptiveAvgPool3d forward with an uneven bin",
        &actual.to_vec()?,
        &[11.0, 22.0],
    );

    actual.mean([depth, height, width])?.backward()?;
    // Height index 2 falls in both bins (depth-averaged over both bins' 1/6 weight each);
    // every other height index falls in exactly one bin.
    #[rustfmt::skip]
    let expected_gradient: [f64; 10] = [
        1.0 / 12.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 12.0, 1.0 / 12.0,
        1.0 / 12.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 12.0, 1.0 / 12.0,
    ];
    close(
        "AdaptiveAvgPool3d gradient (the shared boundary element is averaged into both bins)",
        &input.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn sum_matches_hand_computed_values_and_gradient_over_multiple_axes() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, feature) = (Axis::new("batch"), Axis::new("time"), Axis::new("feature"));
    // Canonical (batch, time, feature) order, feature fastest.
    let values: Vec<_> = (1..=12).map(|value| value as f32).collect();
    let x =
        Tensor::from_slice(&values, [batch.of(2), time.of(3), feature.of(2)], &device)?.with_grad();

    let summed = x.sum([batch, time])?;
    // feature 0: 1+3+5+7+9+11 = 36; feature 1: 2+4+6+8+10+12 = 42.
    close(
        "sum over two axes at once",
        &summed.to_vec()?,
        &[36.0, 42.0],
    );

    // Weight the two features differently before reducing to a scalar so the gradient
    // check distinguishes them; sum's own local derivative is exactly 1 per contributing
    // element, so d(scalar)/d(x[b, t, f]) = weight[f] / feature_count regardless of b, t.
    let weights = Tensor::from_slice(&[1.0, 2.0], [feature.of(2)], &device)?;
    summed.mul(&weights)?.mean(feature)?.backward()?;
    close(
        "sum gradient broadcasts unscaled across the reduced axes",
        &x.grad().unwrap().to_vec()?,
        &[0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
    );

    let missing = Axis::new("missing");
    let error = x.sum(missing).err().unwrap().to_string();
    assert!(error.contains("missing axis missing#"), "{error}");
    assert!(x.sum([batch, batch]).is_err());
    println!("sum forward, multi-axis gradient, and axis rejection PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn sum_over_reordered_asymmetric_storage_matches_hand_computed_values_and_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, feature) = (Axis::new("batch"), Axis::new("time"), Axis::new("feature"));
    const BATCH: usize = 3;
    const TIME: usize = 2;
    const FEATURE: usize = 4;
    // value(b, t, f) = b*100 + t*10 + f, written in canonical (batch, time, feature) order;
    // `with_layout` below then forces a physically reordered, non-contiguous storage while
    // `to_vec()` keeps following this canonical order (Shape.dims(), not physical strides).
    let mut values = vec![0.0f32; BATCH * TIME * FEATURE];
    for b in 0..BATCH {
        for t in 0..TIME {
            for f in 0..FEATURE {
                values[(b * TIME + t) * FEATURE + f] = (b * 100 + t * 10 + f) as f32;
            }
        }
    }
    let x = Tensor::from_slice(
        &values,
        [batch.of(BATCH), time.of(TIME), feature.of(FEATURE)],
        &device,
    )?
    .with_layout([feature, time, batch])?
    .with_grad();

    let summed = x.sum([batch, time])?;
    // sum_f = sum_b sum_t (b*100 + t*10 + f) = (0+100+200)*TIME + (0+10)*BATCH + f*BATCH*TIME
    //       = 300*2 + 10*3 + 6f = 630 + 6f
    close(
        "sum over reordered and asymmetric storage",
        &summed.to_vec()?,
        &[630.0, 636.0, 642.0, 648.0],
    );

    let weights = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [feature.of(FEATURE)], &device)?;
    summed.mul(&weights)?.mean(feature)?.backward()?;
    let mut expected_gradient = vec![0.0f64; BATCH * TIME * FEATURE];
    for b in 0..BATCH {
        for t in 0..TIME {
            for f in 0..FEATURE {
                expected_gradient[(b * TIME + t) * FEATURE + f] = (f + 1) as f64 / FEATURE as f64;
            }
        }
    }
    close(
        "sum gradient under reordered storage lands on the original logical positions",
        &x.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn sum_composes_with_divide_for_masked_counts_including_a_zero_count() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, feature) = (Axis::new("row"), Axis::new("feature"));
    // Per-feature masked squared error, already zeroed where the mask is zero, mirroring
    // energy-output's fit_reconstruction.py loss_fn (`losses.sum(dim=0) /
    // counts.clamp(min=1)`) and fluid's train.py batch_loss (`num = (... * mb).sum()`).
    #[rustfmt::skip]
    let masked_losses = Tensor::from_slice(
        &[
            2.0, 0.0, 0.0,
            4.0, 0.0, 6.0,
            0.0, 0.0, 3.0,
            8.0, 0.0, 0.0,
        ],
        [row.of(4), feature.of(3)],
        &device,
    )?;
    #[rustfmt::skip]
    let mask = Tensor::from_slice(
        &[
            1.0, 0.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
        ],
        [row.of(4), feature.of(3)],
        &device,
    )?;

    let counts = mask.sum(row)?;
    close("mask.sum(row) counts", &counts.to_vec()?, &[3.0, 0.0, 2.0]);

    let losses = masked_losses.sum(row)?.div(&counts)?;
    let actual = losses.to_vec()?;
    // feature 0: 14/3; feature 2: 9/2 -- ordinary division, matching PyTorch's
    // `(...).sum(dim=0) / counts.clamp(min=1)` at those columns exactly, since the clamp
    // is a no-op once the count is already positive.
    assert!(
        (f64::from(actual[0]) - 14.0 / 3.0).abs() < 1e-6,
        "{actual:?}"
    );
    assert!((f64::from(actual[2]) - 4.5).abs() < 1e-6, "{actual:?}");
    // feature 1 has zero mask everywhere, so both the summed numerator and the summed
    // count are exactly zero. `div` applies no clamp (documented on `Tensor::div`), so
    // this is IEEE `0.0 / 0.0 = NaN` -- exactly the case the consumer studies' own
    // `counts.clamp(min=1)` exists to avoid; `sum` itself makes no such policy choice.
    assert!(actual[1].is_nan(), "{actual:?}");
    println!("sum composed with divide over masked counts, including a zero count, PASS");
    Ok(())
}

#[test]
fn clip_grad_norm_rejects_invalid_max_norm() {
    assert!(clip_grad_norm(std::iter::empty::<Parameter>(), 0.0).is_err());
    assert!(clip_grad_norm(std::iter::empty::<Parameter>(), -1.0).is_err());
    assert!(clip_grad_norm(std::iter::empty::<Parameter>(), f32::NAN).is_err());
    assert!(clip_grad_norm(std::iter::empty::<Parameter>(), f32::INFINITY).is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn clip_grad_norm_scales_every_gradient_by_one_global_factor_when_the_norm_exceeds_max_norm()
-> Result<()> {
    let device = Device::cuda(0)?;
    let f = Axis::new("feature");
    // mean(x*x) over a single-element axis has derivative 2*x/1 = 2*x, so
    // x = 1.5 and x = 2.0 give clean gradients 3.0 and 4.0 by construction —
    // a closed form independent of clip_grad_norm itself, the same trick
    // `parameter_versions_accumulation_and_shared_updates` above uses.
    let a = Parameter::new(Tensor::from_slice(&[1.5], [f.of(1)], &device)?);
    a.tensor().mul(&a.tensor())?.mean(f)?.backward()?;
    let b = Parameter::new(Tensor::from_slice(&[2.0], [f.of(1)], &device)?);
    b.tensor().mul(&b.tensor())?.mean(f)?.backward()?;
    // A third parameter with no gradient yet must be skipped, not erred on.
    let untouched = Parameter::new(Tensor::from_slice(&[9.0], [f.of(1)], &device)?);
    close(
        "gradient a before clipping",
        &a.grad().unwrap().to_vec()?,
        &[3.0],
    );
    close(
        "gradient b before clipping",
        &b.grad().unwrap().to_vec()?,
        &[4.0],
    );

    // total_norm = sqrt(3.0^2 + 4.0^2) = sqrt(25) = 5.0, hand-computed and
    // independent of the op under test. `a` is listed twice below (as a
    // stand-in for a tied/shared parameter reachable through two paths): if
    // ParamId deduplication failed, the sum of squares would double-count it
    // to 2*9 + 16 = 34 and total_norm would be sqrt(34) =/= 5.0, failing the
    // very next assertion.
    let total_norm = clip_grad_norm([a.clone(), b.clone(), untouched.clone(), a.clone()], 4.0)?;
    close("pre-clip total norm", &[total_norm], &[5.0]);

    // 5.0 > max_norm (4.0), so every gradient is scaled by the SAME factor
    // max_norm / (total_norm + 1e-6) = 4.0 / 5.000001 = 0.79999984 (by hand,
    // long division to 8 significant figures): 3.0*0.79999984 = 2.39999952,
    // 4.0*0.79999984 = 3.19999936.
    close(
        "gradient a scaled by the global factor",
        &a.grad().unwrap().to_vec()?,
        &[2.39999952],
    );
    close(
        "gradient b scaled by the SAME global factor",
        &b.grad().unwrap().to_vec()?,
        &[3.19999936],
    );
    assert!(
        untouched.grad().is_none(),
        "a parameter with no gradient must not gain one"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn clip_grad_norm_leaves_gradients_bit_exact_below_max_norm() -> Result<()> {
    let device = Device::cuda(0)?;
    let f = Axis::new("feature");
    let a = Parameter::new(Tensor::from_slice(&[1.5], [f.of(1)], &device)?);
    a.tensor().mul(&a.tensor())?.mean(f)?.backward()?;
    let b = Parameter::new(Tensor::from_slice(&[2.0], [f.of(1)], &device)?);
    b.tensor().mul(&b.tensor())?.mean(f)?.backward()?;

    // Same hand-computed total_norm = 5.0 as above, but max_norm = 10.0 is
    // above it: a no-op, never a silent renormalization to exactly max_norm.
    let total_norm = clip_grad_norm([a.clone(), b.clone()], 10.0)?;
    close("pre-clip total norm below threshold", &[total_norm], &[5.0]);
    assert_eq!(
        a.grad().unwrap().to_vec()?,
        vec![3.0_f32],
        "untouched gradient a must be bit-exact"
    );
    assert_eq!(
        b.grad().unwrap().to_vec()?,
        vec![4.0_f32],
        "untouched gradient b must be bit-exact"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_nearest_upsample_matches_hand_computed_values_and_gradient() -> Result<()> {
    // Mirrors morpheus's `upsample2x` helper (proven bit-exact against a real
    // `F.interpolate(mode="nearest")` oracle in
    // `research/src/vision/morpheus/axis/tests/partitioner_trunk_fpn_obj.rs`),
    // now as one library primitive called once per spatial axis.
    let device = Device::cuda(0)?;
    let (height, width) = (Axis::new("height"), Axis::new("width"));
    let x = Tensor::from_slice(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [height.of(2), width.of(3)],
        &device,
    )?
    .with_grad();

    let doubled = x.upsample_nearest(height, 2)?.upsample_nearest(width, 2)?;
    assert_eq!(doubled.shape(), &Shape::new([height.of(4), width.of(6)])?);
    // Source row/column (h, w) of the 2x3 input becomes the 2x2 output block
    // at rows [2h, 2h+1], columns [2w, 2w+1].
    close(
        "nearest upsample values",
        &doubled.to_vec()?,
        &[
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, //
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, //
            3.0, 3.0, 4.0, 4.0, 5.0, 5.0, //
            3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
        ],
    );

    // Weighted upstream (distinct values 1..=24) so the gradient check is
    // not a uniform constant: each source element's gradient is the SUM of
    // the upstream weights over its 2x2 output block, divided by `mean`'s
    // 24-element denominator.
    let upstream: Vec<f32> = (1..=24).map(|v| v as f32).collect();
    let scaled = doubled.mul(&Tensor::from_slice(
        &upstream,
        [height.of(4), width.of(6)],
        &device,
    )?)?;
    scaled.mean([height, width])?.backward()?;
    close(
        "nearest upsample gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[
            (1.0 + 2.0 + 7.0 + 8.0) / 24.0,
            (3.0 + 4.0 + 9.0 + 10.0) / 24.0,
            (5.0 + 6.0 + 11.0 + 12.0) / 24.0,
            (13.0 + 14.0 + 19.0 + 20.0) / 24.0,
            (15.0 + 16.0 + 21.0 + 22.0) / 24.0,
            (17.0 + 18.0 + 23.0 + 24.0) / 24.0,
        ],
    );

    assert!(x.upsample_nearest(height, 0).is_err());
    let missing = Axis::new("missing");
    assert!(x.upsample_nearest(missing, 2).is_err());
    println!("nearest upsample values, gradient, and rejections PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_nearest_upsample_matches_a_non_power_of_two_reordered_storage_case() -> Result<()> {
    // The proven composition this wraps was only checked at extents that
    // stay whole under repeated halving (2x3); this exercises extent 5
    // (5 -> 10, not a power of two) on a tensor whose physical storage is
    // permuted relative to its declared axis order, so the internal
    // `stack`+`merge` reorder is exercised for real rather than skipped as
    // a no-op. An unrelated `batch` axis is preserved through the resample.
    let device = Device::cuda(0)?;
    let (batch, length) = (Axis::new("batch"), Axis::new("length"));
    let x = Tensor::from_slice(
        &[10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        [batch.of(2), length.of(5)],
        &device,
    )?
    .with_layout([length, batch])?
    .with_grad();

    let doubled = x.upsample_nearest(length, 2)?;
    assert_eq!(doubled.shape(), &Shape::new([batch.of(2), length.of(10)])?);
    close(
        "nearest upsample non-power-of-two values",
        &doubled.to_vec()?,
        &[
            10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, //
            20.0, 20.0, 21.0, 21.0, 22.0, 22.0, 23.0, 23.0, 24.0, 24.0,
        ],
    );

    doubled.mean([batch, length])?.backward()?;
    // Each source element is exactly two of the 20 mean-reduced output
    // elements, so every gradient is 2 / 20 regardless of its value.
    close(
        "nearest upsample non-power-of-two gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[2.0 / 20.0; 10],
    );
    println!("nearest upsample non-power-of-two, reordered-storage case PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_bilinear_upsample_matches_hand_computed_values_and_mass_conserving_gradient()
-> Result<()> {
    // align_corners=False half-pixel weight rows for 2 -> 4 are the standard
    // [1,0], [0.75,0.25], [0.25,0.75], [0,1] (matching a real
    // `F.interpolate(mode="bilinear", align_corners=False)` at that exact
    // factor), and for 3 -> 6 the standard [1,0,0], [0.75,0.25,0],
    // [0.25,0.75,0], [0,0.75,0.25], [0,0.25,0.75], [0,0,1]. Applying height
    // then width to a hand-picked 2x3 grid gives this 4x6 grid by hand
    // matrix multiplication.
    let device = Device::cuda(0)?;
    let (height, width) = (Axis::new("height"), Axis::new("width"));
    let x = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [height.of(2), width.of(3)],
        &device,
    )?
    .with_grad();

    let resized = x
        .resample_bilinear(height, 4)?
        .resample_bilinear(width, 6)?;
    assert_eq!(resized.shape(), &Shape::new([height.of(4), width.of(6)])?);
    close(
        "bilinear upsample values",
        &resized.to_vec()?,
        &[
            1.0, 1.25, 1.75, 2.25, 2.75, 3.0, //
            1.75, 2.0, 2.5, 3.0, 3.5, 3.75, //
            3.25, 3.5, 4.0, 4.5, 5.0, 5.25, //
            4.0, 4.25, 4.75, 5.25, 5.75, 6.0,
        ],
    );

    resized.mean([height, width])?.backward()?;
    // Every interpolation weight row sums to 1 (a proper convex
    // combination), so each source element's total downstream weight over
    // all 24 mean-reduced outputs is exactly (4/2) * (6/3) = 4, giving a
    // uniform gradient of 4 / 24 independent of the hand-picked input
    // values -- an invariant check independent of the per-element weighted
    // check below.
    close(
        "bilinear upsample gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[4.0 / 24.0; 6],
    );

    assert!(x.resample_bilinear(height, 0).is_err());
    let missing = Axis::new("missing");
    assert!(x.resample_bilinear(missing, 4).is_err());
    println!("bilinear upsample values, mass-conserving gradient, and rejections PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_bilinear_upsample_matches_a_reordered_storage_weighted_gradient_case() -> Result<()> {
    // A second, independently-derived gradient check (the raw per-element
    // transpose-weight sum, unlike the mass-conservation shortcut above) on
    // a tensor whose physical storage is permuted relative to its declared
    // axis order, with an unrelated `batch` axis preserved through the
    // resample -- this is `world/fluid`'s actual shape (`scripts/train.py:101-102`
    // resamples `height`/`width` while preserving `batch` and `channel`).
    let device = Device::cuda(0)?;
    let (batch, length) = (Axis::new("batch"), Axis::new("length"));
    let x = Tensor::from_slice(
        &[2.0, 5.0, 20.0, 50.0],
        [batch.of(2), length.of(2)],
        &device,
    )?
    .with_layout([length, batch])?
    .with_grad();

    let resized = x.resample_bilinear(length, 4)?;
    assert_eq!(resized.shape(), &Shape::new([batch.of(2), length.of(4)])?);
    // Weight rows for 2 -> 4, align_corners=False: [1,0], [0.75,0.25],
    // [0.25,0.75], [0,1].
    close(
        "bilinear upsample reordered-storage values",
        &resized.to_vec()?,
        &[
            2.0, 2.75, 4.25, 5.0, //
            20.0, 27.5, 42.5, 50.0,
        ],
    );

    let upstream = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        [batch.of(2), length.of(4)],
        &device,
    )?;
    resized.mul(&upstream)?.mean([batch, length])?.backward()?;
    // grad[batch, source] = (1/8) * sum_j upstream[batch, j] * weight[j][source].
    close(
        "bilinear upsample reordered-storage gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[
            (1.0 * 1.0 + 2.0 * 0.75 + 3.0 * 0.25 + 4.0 * 0.0) / 8.0,
            (1.0 * 0.0 + 2.0 * 0.25 + 3.0 * 0.75 + 4.0 * 1.0) / 8.0,
            (10.0 * 1.0 + 20.0 * 0.75 + 30.0 * 0.25 + 40.0 * 0.0) / 8.0,
            (10.0 * 0.0 + 20.0 * 0.25 + 30.0 * 0.75 + 40.0 * 1.0) / 8.0,
        ],
    );
    println!("bilinear upsample reordered-storage forward and gradient PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_maximum_preserves_axes_across_layouts_and_routes_ties() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate, time) = (
        Axis::new("batch"),
        Axis::new("candidate"),
        Axis::new("time"),
    );
    let values = Tensor::from_slice(
        &[
            2.0, -5.0, 9.0, 9.0, 4.0, 9.0, 4.0, 0.0, -2.0, 5.0, 6.0, -1.0,
        ],
        [batch.of(2), candidate.of(3), time.of(2)],
        &device,
    )?
    .with_layout([candidate, time, batch])?
    .with_grad();

    let maximum = values.max(candidate)?;
    assert_eq!(maximum.shape(), &Shape::new([batch.of(2), time.of(2)])?);
    close(
        "named-axis maximum",
        &maximum.to_vec()?,
        &[9.0, 9.0, 6.0, 5.0],
    );
    maximum.mean([batch, time])?.backward()?;
    close(
        "named-axis maximum gradient",
        &values.grad().unwrap().to_vec()?,
        &[
            0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0,
        ],
    );
    let error = values.max(Axis::new("missing")).err().unwrap().to_string();
    assert!(error.contains("missing axis missing#"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_maximum_ignores_nonfinite_values_and_marks_empty_groups() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate) = (Axis::new("batch"), Axis::new("candidate"));
    let values = Tensor::from_slice(
        &[
            f32::NAN,
            f32::INFINITY,
            3.0,
            f32::NEG_INFINITY,
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::INFINITY,
        ],
        [batch.of(2), candidate.of(5)],
        &device,
    )?
    .with_grad();

    let maximum = values.max(candidate)?;
    let actual = maximum.to_vec()?;
    assert_eq!(actual[0], 3.0);
    assert!(actual[1].is_nan());

    maximum.mean(batch)?.backward()?;
    close(
        "non-finite maximum gradient",
        &values.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );

    let all_nonfinite = Tensor::from_slice(
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        [batch.of(1), candidate.of(3)],
        &device,
    )?
    .with_grad();
    let zero = Tensor::from_slice(&[0.0], [batch.of(1)], &device)?;
    let loss = all_nonfinite
        .max(candidate)?
        .squared_error(&zero)?
        .mean(batch)?;
    assert!(loss.to_vec()?[0].is_nan());
    loss.backward()?;
    close(
        "all-nonfinite maximum with NaN cotangent",
        &all_nonfinite.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 0.0],
    );
    Ok(())
}

/// Issue #69's exact evidence bar: `gastric`'s and `energy-output`'s production code
/// already computes max-pooling and stability shifts as
/// `x.scale(-1.0)?.min(axis)?.scale(-1.0)?`. That forward composition is exact, but
/// its backward path was never checked in either study. This proves the native
/// `Tensor::max` kernel and the negate-min-negate composition agree bit-for-bit on
/// both the forward value and the gradient, on a group with a tie, before the native
/// kernel replaces the composition in either consumer.
#[test]
#[ignore = "requires CUDA"]
fn named_axis_maximum_matches_negate_min_negate_composition_bit_exact() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate) = (Axis::new("batch"), Axis::new("candidate"));
    let values = [2.0, 9.0, 9.0, -5.0, 4.0, -2.0, 6.0, 6.0];
    let native = Tensor::from_slice(&values, [batch.of(2), candidate.of(4)], &device)?.with_grad();
    let composed =
        Tensor::from_slice(&values, [batch.of(2), candidate.of(4)], &device)?.with_grad();

    let native_max = native.max(candidate)?;
    let composed_max = composed.scale(-1.0)?.min(candidate)?.scale(-1.0)?;
    assert_eq!(native_max.to_vec()?, composed_max.to_vec()?);

    native_max.mean(batch)?.backward()?;
    composed_max.mean(batch)?.backward()?;
    let native_gradient = native.grad().unwrap().to_vec()?;
    assert_eq!(native_gradient, composed.grad().unwrap().to_vec()?);
    close(
        "maximum gradient (native, cross-checked bit-exact against negate-min-negate)",
        &native_gradient,
        &[0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gather_scatter_add_matches_hand_computed_oracle_on_repeated_index() -> Result<()> {
    // render's hash_grid.py picks HashGrid table rows by a computed integer
    // hash; upscale_eval's SwinIR picks relative_position_bias_table rows by
    // a precomputed geometry index. Both need forward row copies AND a
    // gradient that ACCUMULATES into a row picked more than once — the case
    // a one-hot-matmul implementation gets wrong silently (correct forward,
    // dropped accumulated gradient) unless backward is a real scatter-add.
    let device = Device::cuda(0)?;
    let (row, feature, pick) = (Axis::new("row"), Axis::new("feature"), Axis::new("pick"));
    let table_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let table = Tensor::from_slice(&table_values, [row.of(5), feature.of(2)], &device)?.with_grad();

    // Non-monotonic order; row 2 is picked twice, at positions 1 and 3.
    let index = [0usize, 2, 4, 2, 1];
    let gathered = table.gather(row, &index, pick)?;
    assert_eq!(gathered.shape(), &Shape::new([pick.of(5), feature.of(2)])?);
    close(
        "gather forward",
        &gathered.to_vec()?,
        &[1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 5.0, 6.0, 3.0, 4.0],
    );

    gathered.mean([pick, feature])?.backward()?;
    close(
        "gather scatter-add gradient",
        &table.grad().expect("table gradient").to_vec()?,
        &[0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.0, 0.0, 0.1, 0.1],
    );

    // Rejected before any device work: empty index, an index outside the
    // row extent, and an axis the table does not have.
    assert!(table.gather(row, &[], pick).is_err());
    assert!(table.gather(row, &[0, 5], pick).is_err());
    assert!(table.gather(Axis::new("missing"), &[0], pick).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gather_matches_hand_computed_oracle_under_reordered_table_storage() -> Result<()> {
    // Same accumulation contract as the test above, this time with the
    // table's PHYSICAL storage transposed relative to its logical
    // [row, feature] order (mirroring a table copied through `with_layout`,
    // or one whose default construction stride order differs from a
    // gather's own row-major assumption): gather must read through the
    // permuted strides in `self.0.layout`, not assume row-major storage.
    let device = Device::cuda(0)?;
    let (row, feature, pick) = (Axis::new("row"), Axis::new("feature"), Axis::new("pick"));
    let table_values = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let table = Tensor::from_slice(&table_values, [row.of(4), feature.of(3)], &device)?
        .with_layout([feature, row])?
        .with_grad();

    // Non-monotonic order; row 3 is picked twice, at positions 0 and 2.
    let index = [3usize, 0, 3, 1];
    let gathered = table.gather(row, &index, pick)?;
    close(
        "reordered gather forward",
        &gathered.to_vec()?,
        &[
            10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 4.0, 5.0, 6.0,
        ],
    );

    gathered.mean([pick, feature])?.backward()?;
    close(
        "reordered gather scatter-add gradient",
        &table.grad().expect("table gradient").to_vec()?,
        &[
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 12.0,
            0.0,
            0.0,
            0.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn comparison_scalar_ops_match_hand_computed_edge_cases_including_nan() -> Result<()> {
    // energy-output/fit_reconstruction.py:108,127-130 and fluid/scripts/train.py:133
    // both compare a tensor against a plain scalar (`counts > 0`, `rand < MASK_RATE`,
    // `xb.abs() > 0`), never against another tensor, so only the scalar form is
    // implemented. Edge cases: exact equality at a representable value (0.0, and a
    // non-zero 1.5), and NaN -- Axis tensors are dense f32 (no separate NaN-free
    // invariant), so a real caller can hand one to a comparison, and every ordered
    // IEEE comparison against NaN must come back false (0.0), never true or NaN.
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let values = [-2.0f32, -0.5, 0.0, 0.5, 1.5, 2.0, f32::NAN];
    let x = Tensor::from_slice(&values, [sample.of(values.len())], &device)?;

    close(
        "gt(0.0)",
        &x.gt(0.0)?.to_vec()?,
        &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    );
    close(
        "ge(0.0)",
        &x.ge(0.0)?.to_vec()?,
        &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    );
    close(
        "lt(0.0)",
        &x.lt(0.0)?.to_vec()?,
        &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    close(
        "le(0.0)",
        &x.le(0.0)?.to_vec()?,
        &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    );
    close(
        "eq(0.0)",
        &x.eq(0.0)?.to_vec()?,
        &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    );
    // Exact equality at a representable non-zero value: only index 4 (1.5) matches.
    close(
        "eq(1.5)",
        &x.eq(1.5)?.to_vec()?,
        &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );

    // Comparisons carry no autograd edge at all, matching PyTorch (`>` etc. are
    // non-differentiable): even when the input requires grad, the mask does not.
    assert!(
        !x.with_grad().gt(0.0)?.requires_grad(),
        "a comparison output must not require grad even when its input does"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn comparison_masks_gate_multiplication_and_compose_with_logical_and_not() -> Result<()> {
    // fluid/scripts/train.py:133: `xb + noise * (xb.abs() > 0).float()` uses a
    // comparison-derived mask as a multiplicative gate. The mask must contribute
    // its VALUE to the forward pass but none of its own gradient: `d(x*mask)/dx`
    // is `mask` treated as a constant, never `mask`'s own (nonexistent) derivative.
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let x_values = [-3.0f32, 0.0, 2.0, -1.0, 4.0];
    let x = Tensor::from_slice(&x_values, [sample.of(5)], &device)?.with_grad();

    let gate = x.gt(0.0)?;
    assert!(
        !gate.requires_grad(),
        "the gate must carry no autograd edge of its own"
    );
    let gated = x.mul(&gate)?;
    close(
        "multiplicative gate forward",
        &gated.to_vec()?,
        &[0.0, 0.0, 2.0, 0.0, 4.0],
    );
    gated.mean(sample)?.backward()?;
    // d(mean(x*mask))/dx_i = mask_i / n, treating mask as a constant.
    close(
        "gate gradient treats the mask as a constant",
        &x.grad().expect("x gradient").to_vec()?,
        &[0.0, 0.0, 0.2, 0.0, 0.2],
    );

    // energy-output/fit_reconstruction.py:127-129:
    //   hidden = (torch.rand(...) < MASK_RATE) & observed
    //   visible = observed & ~hidden
    let feature = Axis::new("feature");
    let observed = Tensor::from_slice(&[1.0f32, 1.0, 0.0, 1.0], [feature.of(4)], &device)?;
    let draws = Tensor::from_slice(&[0.1f32, 0.9, 0.2, 0.4], [feature.of(4)], &device)?;
    let hidden = draws.lt(0.5)?.logical_and(&observed)?;
    close(
        "hidden mask (rand < rate) & observed",
        &hidden.to_vec()?,
        &[1.0, 0.0, 0.0, 1.0],
    );
    let visible = observed.logical_and(&hidden.logical_not()?)?;
    close(
        "visible mask observed & ~hidden",
        &visible.to_vec()?,
        &[0.0, 1.0, 0.0, 0.0],
    );
    assert!(!hidden.requires_grad());
    assert!(!visible.requires_grad());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn comparison_matches_logical_order_under_reordered_cuda_storage() -> Result<()> {
    // Elementwise ops read the raw physical buffer and keep the tensor's existing
    // Layout unchanged (`with_layout` only ever changes physical strides), so a
    // comparison on a reordered-storage tensor must still read out in the
    // tensor's own logical Shape order via `to_vec()`, exactly like `sign` or
    // `sin` on a permuted tensor.
    let device = Device::cuda(0)?;
    let (batch, time) = (Axis::new("batch"), Axis::new("time"));
    let values: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(&values, [batch.of(2), time.of(3)], &device)?
        .with_layout([time, batch])?;
    let mask = x.gt(0.5)?;
    assert_eq!(mask.shape(), x.shape());
    let expected: Vec<f64> = values
        .iter()
        .map(|&v| f64::from(u8::from(v > 0.5)))
        .collect();
    close("reordered-storage comparison", &mask.to_vec()?, &expected);
    Ok(())
}

#[test]
fn seeded_uniform_and_normal_host_values_match_an_independent_xorshift_oracle() {
    // Independent Rust reimplementation of the xorshift64 stream (mirrors, but does not call,
    // `crate::tensor::xorshift_unit_stream`), used only to hand-derive the literals below via a
    // one-off script; not part of the crate under test. The generator: rng = seed.max(1); each
    // draw does rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; raw = (rng>>40) as f64 / 2f64^24.
    // uniform(seed, count, low, high) maps raw -> low + raw*(high-low). normal(seed, count, mean,
    // std) consumes raw pairs (u1, u2) via Box-Muller: z0 = sqrt(-2*ln(1-u1))*cos(2*pi*u2), z1 =
    // sqrt(-2*ln(1-u1))*sin(2*pi*u2), each scaled to mean + std*z; an odd count drops the final
    // z1. energy-output (world/energy-output) needs the uniform sequence to threshold into a
    // reproducible Bernoulli mask; fluid (world/fluid) needs the normal sequence as additive
    // noise — both need the WHOLE multi-draw sequence for one seed to match, not just one value.
    //
    // world/energy-output shape: a fresh 30% mask drawn every epoch from one seed (`seed+37`).
    // seed=7 is far below 2^40, so the shared stream's documented quirk (first raw sample
    // exactly 0) makes the very first uniform draw land exactly on `low`.
    close(
        "uniform seed=7 low=-1 high=1 (six successive draws)",
        &crate::tensor::uniform_host_values(7, 6, -1.0, 1.0),
        &[
            -1.0,
            -0.1249457597732544,
            0.5105017423629761,
            -0.056897759437561035,
            -0.08123886585235596,
            0.4238666296005249,
        ],
    );
    // Same seed, different range: a second, independent call reproduces the SAME raw stream
    // (determinism), rescaled to [0, 1) instead of [-1, 1).
    close(
        "uniform seed=7 low=0 high=1, second call (determinism across calls)",
        &crate::tensor::uniform_host_values(7, 6, 0.0, 1.0),
        &[
            0.0,
            0.4375271201133728,
            0.755250871181488,
            0.4715511202812195,
            0.459380567073822,
            0.7119333148002625,
        ],
    );
    // A distinct seed diverges from the first draw on.
    close(
        "uniform seed=12345 low=-3 high=5 (distinct seed diverges)",
        &crate::tensor::uniform_host_values(12345, 4, -3.0, 5.0),
        &[
            -2.9999942779541016,
            1.8767971992492676,
            2.067387104034424,
            -1.9700355529785156,
        ],
    );
    for &(seed, low, high) in &[(7u64, -1.0f32, 1.0f32), (12345, -3.0, 5.0), (1, -1.0, 1.0)] {
        for &value in &crate::tensor::uniform_host_values(seed, 64, low, high) {
            assert!(
                value >= low && value < high,
                "uniform seed={seed}: {value} outside [{low}, {high})"
            );
        }
    }

    // world/fluid shape: `torch.randn_like(xb)` additive noise, needed as a normal draw, not
    // just uniform. seed=99 is also below 2^40, so the shared stream's quirk collapses the
    // FIRST PAIR of normal draws (both z0 and z1) to exactly `mean`, since raw=0 forces
    // radius=0 for the whole pair, not just one value.
    close(
        "normal seed=99 mean=0 std=1 (six successive draws, two full pairs plus one)",
        &crate::tensor::normal_host_values(99, 6, 0.0, 1.0),
        &[
            0.0,
            0.0,
            -0.04610706669773621,
            0.15413647086258625,
            1.160083365411145,
            -1.2878684354641539,
        ],
    );
    // Odd count: the trailing unpaired z1 is dropped, not returned.
    close(
        "normal seed=99 mean=2 std=0.5, odd count drops the final pair's second value",
        &crate::tensor::normal_host_values(99, 5, 2.0, 0.5),
        &[
            2.0,
            2.0,
            1.9769464666511318,
            2.0770682354312933,
            2.5800416827055725,
        ],
    );
    // Mean/std sanity over a modest sample: 2,000 draws from one seed land within a generous
    // tolerance of the requested mean and standard deviation (this is a statistical sanity
    // check, not a bit-exact oracle).
    let sample = crate::tensor::normal_host_values(4242, 2000, 1.0, 2.0);
    let sample_mean = sample.iter().map(|&v| f64::from(v)).sum::<f64>() / sample.len() as f64;
    let sample_variance = sample
        .iter()
        .map(|&v| (f64::from(v) - sample_mean).powi(2))
        .sum::<f64>()
        / sample.len() as f64;
    assert!(
        (sample_mean - 1.0).abs() < 0.1,
        "normal sample mean {sample_mean} too far from 1.0"
    );
    assert!(
        (sample_variance.sqrt() - 2.0).abs() < 0.1,
        "normal sample std {} too far from 2.0",
        sample_variance.sqrt()
    );
    println!("seeded uniform/normal host values PASS");
}

#[test]
#[ignore = "requires CUDA"]
fn seeded_uniform_and_normal_tensors_build_named_axis_tensors_with_no_gradient_edge() -> Result<()>
{
    // world/energy-output composes a fresh seeded uniform draw into a Bernoulli mask every
    // epoch (`torch.rand(z.shape, generator=rng) < MASK_RATE`); world/fluid adds a fresh
    // seeded normal draw as training noise (`xb + torch.randn_like(xb) * 0.5 * mask`). Both
    // need `Tensor::uniform`/`Tensor::normal` to build an ordinary named-axis device tensor
    // whose values are usable directly in later elementwise composition.
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let feature = Axis::new("feature");

    let mask_source = Tensor::uniform([sample.of(2), feature.of(3)], 7, 0.0, 1.0, &device)?;
    close(
        "uniform tensor values match the host oracle",
        &mask_source.to_vec()?,
        &[
            0.0,
            0.4375271201133728,
            0.755250871181488,
            0.4715511202812195,
            0.459380567073822,
            0.7119333148002625,
        ],
    );
    assert!(
        !mask_source.requires_grad(),
        "a random draw is a constant, not a differentiable leaf"
    );

    let noise = Tensor::normal([sample.of(2), feature.of(3)], 99, 0.0, 1.0, &device)?;
    close(
        "normal tensor values match the host oracle",
        &noise.to_vec()?,
        &[
            0.0,
            0.0,
            -0.04610706669773621,
            0.15413647086258625,
            1.160083365411145,
            -1.2878684354641539,
        ],
    );
    assert!(!noise.requires_grad());

    // fluid's actual composition: additive noise into an existing input tensor, exercising the
    // random tensor as an ordinary operand of `mul`/`add` alongside a tensor with reordered
    // physical storage (the input built with a permuted layout).
    let input_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_slice(&input_values, [sample.of(2), feature.of(3)], &device)?
        .with_layout([feature, sample])?
        .with_grad();
    let scaled_noise = noise.scale(0.5)?;
    let noisy = input.add(&scaled_noise)?;
    let noise_values = noise.to_vec()?;
    let out_shape = noisy.shape().clone();
    let mut expected = Vec::with_capacity(6);
    for index in 0..out_shape.len() {
        let coords = out_shape.coords(index);
        let (s, f) = (
            coords[out_shape.index(sample)?],
            coords[out_shape.index(feature)?],
        );
        expected
            .push(f64::from(input_values[s * 3 + f]) + f64::from(noise_values[s * 3 + f]) * 0.5);
    }
    close(
        "input plus scaled seeded noise",
        &noisy.to_vec()?,
        &expected,
    );
    noisy.mean([sample, feature])?.backward()?;
    close(
        "gradient flows through the non-random operand only",
        &input.grad().expect("input gradient").to_vec()?,
        &[1.0 / 6.0; 6],
    );

    // A second, independent seed produces different values from both constructors (distinct
    // seeds diverge, not just distinct calls).
    let other_seed_uniform = Tensor::uniform([sample.of(2), feature.of(3)], 8, 0.0, 1.0, &device)?;
    assert_ne!(mask_source.to_vec()?, other_seed_uniform.to_vec()?);
    let other_seed_normal = Tensor::normal([sample.of(2), feature.of(3)], 100, 0.0, 1.0, &device)?;
    assert_ne!(noise.to_vec()?, other_seed_normal.to_vec()?);

    // Rejections: low must be strictly less than high, and std must be non-negative.
    assert!(Tensor::uniform([sample.of(2)], 1, 1.0, 1.0, &device).is_err());
    assert!(Tensor::uniform([sample.of(2)], 1, 1.0, -1.0, &device).is_err());
    assert!(Tensor::normal([sample.of(2)], 1, 0.0, -1.0, &device).is_err());
    println!("seeded uniform/normal tensor construction PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adamw_with_hyperparameters_matches_hand_computed_two_step_update() -> Result<()> {
    // Independent scalar oracle for issue #82 (AdamW's public constructor could not express a
    // non-default beta2, blocking upscale_eval's SwinIR trainer, which uses betas=(0.9, 0.99)).
    // lr=0.1, weight_decay=0.1, beta1=0.8, beta2=0.9, epsilon=1e-8, theta0=1.0, constant
    // gradient g=0.4 every step. epsilon's ~1e-8 contribution to each denominator is far below
    // this test's tolerance and is omitted from the arithmetic below.
    //
    // step 1:
    //   m1 = (1 - beta1) * g              = 0.2 * 0.4   = 0.08
    //   v1 = (1 - beta2) * g^2            = 0.1 * 0.16  = 0.016
    //   mhat1 = m1 / (1 - beta1^1)        = 0.08 / 0.2  = 0.4
    //   vhat1 = v1 / (1 - beta2^1)        = 0.016 / 0.1 = 0.16
    //   decayed0 = theta0 * (1 - lr*wd)   = 1.0 * 0.99  = 0.99
    //   theta1 = decayed0 - lr*mhat1/sqrt(vhat1) = 0.99 - 0.1*(0.4/0.4) = 0.89
    // step 2 (g=0.4 again):
    //   m2 = beta1*m1 + (1-beta1)*g       = 0.8*0.08 + 0.2*0.4   = 0.144
    //   v2 = beta2*v1 + (1-beta2)*g^2     = 0.9*0.016 + 0.1*0.16 = 0.0304
    //   mhat2 = m2 / (1 - beta1^2)        = 0.144 / 0.36 = 0.4
    //   vhat2 = v2 / (1 - beta2^2)        = 0.0304 / 0.19 = 0.16
    //   decayed1 = theta1 * (1 - lr*wd)   = 0.89 * 0.99  = 0.8811
    //   theta2 = decayed1 - lr*mhat2/sqrt(vhat2) = 0.8811 - 0.1*(0.4/0.4) = 0.7811
    let device = Device::cuda(0)?;
    let unit = Axis::new("unit");
    let parameter = Parameter::new(Tensor::from_slice(&[1.0], [unit.of(1)], &device)?);
    let coefficient = Tensor::from_slice(&[0.4], [unit.of(1)], &device)?;
    let mut adamw = AdamW::with_hyperparameters(0.1, 0.1, 0.8, 0.9, 1e-8)?;
    for expected in [0.89_f64, 0.7811_f64] {
        parameter
            .tensor()
            .mul(&coefficient)?
            .mean(unit)?
            .backward()?;
        adamw.step_parameters([parameter.clone()])?;
        close(
            "AdamW hand-computed non-default-beta update",
            &parameter.tensor().to_vec()?,
            &[expected],
        );
        parameter.zero_grad();
    }
    assert_eq!(adamw.completed_steps(), 2);

    // Validation mirrors Adam::with_hyperparameters exactly (same finite/range checks, in the
    // same argument order), plus AdamW's own weight-decay check.
    assert!(AdamW::with_hyperparameters(0.0, 0.0, 0.9, 0.999, 1e-8).is_err());
    assert!(AdamW::with_hyperparameters(f32::NAN, 0.0, 0.9, 0.999, 1e-8).is_err());
    assert!(AdamW::with_hyperparameters(0.1, 0.0, 1.0, 0.999, 1e-8).is_err());
    assert!(AdamW::with_hyperparameters(0.1, 0.0, 0.9, 1.0, 1e-8).is_err());
    assert!(AdamW::with_hyperparameters(0.1, 0.0, 0.9, 0.999, 0.0).is_err());
    assert!(AdamW::with_hyperparameters(0.1, -0.1, 0.9, 0.999, 1e-8).is_err());
    assert!(AdamW::with_hyperparameters(0.1, 0.0, 0.9, 0.99, 1e-8).is_ok());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adamw_new_matches_with_hyperparameters_at_defaults_bit_exactly() -> Result<()> {
    // AdamW::new(lr, weight_decay) must stay byte-identical to before #82: it now forwards to
    // with_hyperparameters at Adam's own defaults (beta1=0.9, beta2=0.999, epsilon=1e-8), so the
    // two must drive an identical parameter to the same bits, not merely within tolerance.
    let device = Device::cuda(0)?;
    let unit = Axis::new("unit");
    let make_parameter = |value: f32| -> Result<Parameter> {
        Ok(Parameter::new(Tensor::from_slice(
            &[value],
            [unit.of(1)],
            &device,
        )?))
    };
    let coefficient = Tensor::from_slice(&[0.3], [unit.of(1)], &device)?;
    let default_parameter = make_parameter(1.0)?;
    let explicit_parameter = make_parameter(1.0)?;
    let mut default_adamw = AdamW::new(0.05, 0.02)?;
    let mut explicit_adamw = AdamW::with_hyperparameters(0.05, 0.02, 0.9, 0.999, 1e-8)?;

    for _ in 0..3 {
        for parameter in [&default_parameter, &explicit_parameter] {
            parameter
                .tensor()
                .mul(&coefficient)?
                .mean(unit)?
                .backward()?;
        }
        default_adamw.step_parameters([default_parameter.clone()])?;
        explicit_adamw.step_parameters([explicit_parameter.clone()])?;
        assert_eq!(
            default_parameter.tensor().to_vec()?,
            explicit_parameter.tensor().to_vec()?,
            "AdamW::new must stay bit-exact with AdamW::with_hyperparameters at its own defaults"
        );
        default_parameter.zero_grad();
        explicit_parameter.zero_grad();
    }
    assert_eq!(
        default_adamw.completed_steps(),
        explicit_adamw.completed_steps()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn abs_and_absolute_error_match_hand_computed_forward_and_gradient_oracles() -> Result<()> {
    // morpheus's ctr/scl/rad heads train with masked `F.l1_loss` (mean absolute error over
    // positive-class slots only); it composes as `(pred - target).abs()` then a caller-side
    // masked mean, exactly like `squared_error`'s own unreduced-then-`.mean()` convention
    // (research/src/vision/morpheus/docs/axis-issues.md "Elementwise L1 loss (abs)",
    // axis#89). `sign(0) == 0` below matches PyTorch's own `abs` backward subgradient choice
    // at exactly zero, not `NaN` or either one-sided slope.
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let values = [-3.0f32, -1.0, 0.0, 2.0, 5.0];
    let x = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();
    let absolute = x.abs()?;
    close(
        "abs forward",
        &absolute.to_vec()?,
        &values
            .iter()
            .map(|&v| f64::from(v).abs())
            .collect::<Vec<_>>(),
    );
    absolute.mean(sample)?.backward()?;
    close(
        "abs gradient, zero exactly at x == 0",
        &x.grad().expect("x gradient").to_vec()?,
        &[-0.2, -0.2, 0.0, 0.2, 0.2],
    );

    let target_values = [-1.0f32, 2.0, 0.0, 2.0, 1.0];
    let pred = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();
    let target =
        Tensor::from_slice(&target_values, [sample.of(values.len())], &device)?.with_grad();
    let loss = pred.absolute_error(&target)?;
    close(
        "absolute_error forward (unreduced |pred - target|)",
        &loss.to_vec()?,
        &[2.0, 3.0, 0.0, 0.0, 4.0],
    );
    loss.mean(sample)?.backward()?;
    close(
        "absolute_error gradient wrt prediction, zero where pred == target",
        &pred.grad().expect("prediction gradient").to_vec()?,
        &[-0.2, -0.2, 0.0, 0.0, 0.2],
    );
    close(
        "absolute_error gradient wrt target",
        &target.grad().expect("target gradient").to_vec()?,
        &[0.2, 0.2, 0.0, 0.0, -0.2],
    );

    let mismatched = Tensor::from_slice(&[1.0, 2.0], [Axis::new("other").of(2)], &device)?;
    assert!(x.absolute_error(&mismatched).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn abs_matches_hand_computed_oracle_under_reordered_asymmetric_cuda_storage() -> Result<()> {
    // Elementwise ops read the raw physical buffer and keep the tensor's existing Layout
    // unchanged (`with_layout` only ever changes physical strides), so `abs` on a permuted,
    // asymmetric-extent tensor must still read out in the tensor's own logical Shape order via
    // `to_vec()`, exactly like `sin` or a comparison on a reordered tensor.
    let device = Device::cuda(0)?;
    let (batch, time) = (Axis::new("batch"), Axis::new("time"));
    // value(b, t) written in canonical (batch, time) order; asymmetric extents (3, 2).
    let values: Vec<f32> = vec![-3.0, 4.0, 0.0, -5.0, 2.0, -1.0];
    let x = Tensor::from_slice(&values, [batch.of(3), time.of(2)], &device)?
        .with_layout([time, batch])?
        .with_grad();
    let absolute = x.abs()?;
    assert_eq!(absolute.shape(), x.shape());
    close(
        "abs forward under reordered, asymmetric storage",
        &absolute.to_vec()?,
        &values
            .iter()
            .map(|&v| f64::from(v).abs())
            .collect::<Vec<_>>(),
    );
    absolute.mean([batch, time])?.backward()?;
    close(
        "abs gradient under reordered storage lands on the original logical positions",
        &x.grad().expect("x gradient").to_vec()?,
        &[
            -1.0 / 6.0,
            1.0 / 6.0,
            0.0,
            -1.0 / 6.0,
            1.0 / 6.0,
            -1.0 / 6.0,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn exp_forward_and_gradient_match_independent_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    // Hand-computed literals: exp(0) = 1, exp(1) = e, plus exp(-2)/exp(-1)/exp(2).
    let values = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let expected = [
        0.1353352832366127,
        0.36787944117144233,
        1.0,
        std::f64::consts::E,
        7.38905609893065,
    ];
    let leaf = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();
    let output = leaf.exp()?;
    close("exp forward", &output.to_vec()?, &expected);
    output.mean(sample)?.backward()?;
    // Backward is `g * exp(x)`; mean seeds `g = 1 / len`, so the expected gradient is
    // just the forward oracle scaled the same way.
    let expected_gradient: Vec<f64> = expected.iter().map(|&y| y / values.len() as f64).collect();
    close(
        "exp derivative (g * exp(x))",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    // Reordered-storage CUDA case: physical order different from logical order.
    let (row, col) = (Axis::new("exp_row"), Axis::new("exp_col"));
    let wide_values: Vec<f32> = (0..201).map(|i| (i as f32 - 100.0) / 25.0).collect();
    let wide_leaf = Tensor::from_slice(&wide_values, [row.of(3), col.of(67)], &device)?.with_grad();
    let wide_output = wide_leaf.with_layout([col, row])?.exp()?;
    let wide_expected: Vec<f64> = wide_values.iter().map(|&x| f64::from(x).exp()).collect();
    close(
        "exp forward (reordered storage)",
        &wide_output.to_vec()?,
        &wide_expected,
    );
    wide_output.mean([row, col])?.backward()?;
    let wide_expected_gradient: Vec<f64> = wide_expected
        .iter()
        .map(|&y| y / wide_values.len() as f64)
        .collect();
    close(
        "exp derivative (reordered storage)",
        &wide_leaf.grad().unwrap().to_vec()?,
        &wide_expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn ln_forward_and_gradient_match_independent_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    // Hand-computed literals: ln(1) = 0, ln(e) = 1, plus ln(0.25) = -ln(4).
    let values = [0.25_f32, 1.0, std::f32::consts::E, 4.0];
    let expected = [-1.3862943611198906, 0.0, 1.0, 1.3862943611198906];
    let leaf = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();
    let output = leaf.ln()?;
    close("ln forward", &output.to_vec()?, &expected);
    output.mean(sample)?.backward()?;
    let expected_gradient: Vec<f64> = values
        .iter()
        .map(|&x| 1.0 / f64::from(x) / values.len() as f64)
        .collect();
    close(
        "ln derivative (g / x)",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    // IEEE behaviour at and below zero: no clamping, matching f32::ln/torch.log.
    let boundary = Tensor::from_slice(&[0.0_f32, -1.0], [sample.of(2)], &device)?;
    let boundary_output = boundary.ln()?.to_vec()?;
    assert!(
        boundary_output[0].is_infinite() && boundary_output[0].is_sign_negative(),
        "ln(0) must be -inf, got {}",
        boundary_output[0]
    );
    assert!(
        boundary_output[1].is_nan(),
        "ln(-1) must be NaN, got {}",
        boundary_output[1]
    );

    // Reordered-storage CUDA case: physical order different from logical order.
    let (row, col) = (Axis::new("ln_row"), Axis::new("ln_col"));
    let wide_values: Vec<f32> = (0..201).map(|i| (i as f32 + 1.0) / 25.0).collect();
    let wide_leaf = Tensor::from_slice(&wide_values, [row.of(3), col.of(67)], &device)?.with_grad();
    let wide_output = wide_leaf.with_layout([col, row])?.ln()?;
    let wide_expected: Vec<f64> = wide_values.iter().map(|&x| f64::from(x).ln()).collect();
    close(
        "ln forward (reordered storage)",
        &wide_output.to_vec()?,
        &wide_expected,
    );
    wide_output.mean([row, col])?.backward()?;
    let wide_expected_gradient: Vec<f64> = wide_values
        .iter()
        .map(|&x| 1.0 / f64::from(x) / wide_values.len() as f64)
        .collect();
    close(
        "ln derivative (reordered storage)",
        &wide_leaf.grad().unwrap().to_vec()?,
        &wide_expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn softplus_forward_and_gradient_match_pytorch_beta1_threshold40_oracle() -> Result<()> {
    // `world/energy-output`'s exact consumer configuration (every `fit_*.py`'s
    // nonlinearity): `torch.nn.Softplus(beta=1, threshold=40)`.
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let (beta, threshold) = (1.0_f32, 40.0_f32);
    // Hand-computed literals: softplus(0) = ln(2); x = 41 clears beta*x > 40, so the
    // linear branch returns x itself with gradient exactly 1; x = -41 is deep in the
    // logarithmic branch, where softplus(x) is negligible.
    let values = [0.0_f32, 41.0, -41.0, -2.0, 2.0];
    let expected = [
        std::f64::consts::LN_2,
        41.0,
        1.5628821893349888e-18,
        0.1269280110429725,
        2.1269280110429727,
    ];
    let leaf = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();
    let output = leaf.softplus(beta, threshold)?;
    close("softplus forward", &output.to_vec()?, &expected);
    output.mean(sample)?.backward()?;
    let n = values.len() as f64;
    let expected_gradient = [
        0.5 / n,
        1.0 / n,
        1.5628821893349888e-18 / n,
        0.11920292202211755 / n,
        0.8807970779778823 / n,
    ];
    close(
        "softplus derivative (sigmoid(beta*x), 1 past the threshold)",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    assert!(leaf.softplus(0.0, threshold).is_err());
    assert!(leaf.softplus(-1.0, threshold).is_err());
    assert!(leaf.softplus(f32::NAN, threshold).is_err());
    assert!(leaf.softplus(beta, f32::NAN).is_err());
    assert!(leaf.softplus(beta, f32::INFINITY).is_err());

    // Reordered-storage CUDA case spanning both branches (|x| up to 50 crosses
    // beta * x > 40 on both signs of the permuted tensor).
    let (row, col) = (Axis::new("softplus_row"), Axis::new("softplus_col"));
    let wide_values: Vec<f32> = (0..201).map(|i| (i as f32 - 100.0) / 2.0).collect();
    let wide_leaf = Tensor::from_slice(&wide_values, [row.of(3), col.of(67)], &device)?.with_grad();
    let wide_output = wide_leaf
        .with_layout([col, row])?
        .softplus(beta, threshold)?;
    let wide_expected: Vec<f64> = wide_values
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            let scaled = f64::from(beta) * x;
            if scaled > f64::from(threshold) {
                x
            } else {
                (1.0 + scaled.exp()).ln() / f64::from(beta)
            }
        })
        .collect();
    close(
        "softplus forward (reordered storage)",
        &wide_output.to_vec()?,
        &wide_expected,
    );
    wide_output.mean([row, col])?.backward()?;
    let wide_expected_gradient: Vec<f64> = wide_values
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            let scaled = f64::from(beta) * x;
            let derivative = if scaled > f64::from(threshold) {
                1.0
            } else {
                1.0 / (1.0 + (-scaled).exp())
            };
            derivative / wide_values.len() as f64
        })
        .collect();
    close(
        "softplus derivative (reordered storage)",
        &wide_leaf.grad().unwrap().to_vec()?,
        &wide_expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn clamp_matches_hand_computed_values_below_inside_above_at_bound_and_nan() -> Result<()> {
    // vision/image-encode's stage_a.py:72, `sites.clamp_(0, 1)`, keeps site positions inside
    // the unit square after each Adam step -- both bounds given, matching torch.clamp(x, 0, 1).
    // Cover below-both-bounds, exactly-at-min, strictly-inside, exactly-at-max,
    // above-both-bounds, and a NaN element (Axis tensors are dense f32, so a real caller can
    // hand clamp() one).
    let device = Device::cuda(0)?;
    let sample = Axis::new("sample");
    let values = [-2.0f32, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5, 2.0, f32::NAN];
    let x = Tensor::from_slice(&values, [sample.of(values.len())], &device)?.with_grad();

    let clamped = x.clamp(Some(0.0), Some(1.0))?;
    assert_eq!(clamped.shape(), x.shape());
    let actual = clamped.to_vec()?;
    assert!(
        actual[8].is_nan(),
        "a NaN input must propagate unclamped, got {}",
        actual[8]
    );
    close(
        "clamp(0, 1) forward, below/at-min/inside/at-max/above",
        &actual[..8],
        &[0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0],
    );

    clamped.mean(sample)?.backward()?;
    // PyTorch's clamp gradient: 1 where min <= x <= max (inclusive of both bounds), 0
    // elsewhere -- including NaN, since `x >= min` is itself false for NaN.
    let n = values.len() as f64;
    close(
        "clamp(0, 1) gradient, below/at-min/inside/at-max/above/nan",
        &x.grad().expect("x gradient").to_vec()?,
        &[0.0, 0.0, 1.0 / n, 1.0 / n, 1.0 / n, 1.0 / n, 0.0, 0.0, 0.0],
    );

    // Rejections before any device launch.
    assert!(
        x.clamp(Some(2.0), Some(1.0)).is_err(),
        "min > max must be rejected"
    );
    assert!(
        x.clamp(Some(f32::NAN), None).is_err(),
        "a NaN min must be rejected"
    );
    assert!(
        x.clamp(None, Some(f32::NAN)).is_err(),
        "a NaN max must be rejected"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn clamp_with_only_a_min_bound_matches_energy_output_denominator_floor() -> Result<()> {
    // world/energy-output's fit_reconstruction.py:100-101, `counts.clamp(min=1)`, floors a
    // per-feature observed count before it becomes a division denominator -- only `min` is
    // given, so `max` stays unbounded (`None`), matching torch.clamp(counts, min=1).
    let device = Device::cuda(0)?;
    let feature = Axis::new("feature");
    let values = [0.0f32, 0.5, 1.0, 3.0];
    let counts = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();

    let floored = counts.clamp(Some(1.0), None)?;
    close(
        "clamp(min=1) forward",
        &floored.to_vec()?,
        &[1.0, 1.0, 1.0, 3.0],
    );

    floored.mean(feature)?.backward()?;
    close(
        "clamp(min=1) gradient: zero below the floor, one at and above it",
        &counts.grad().expect("counts gradient").to_vec()?,
        &[0.0, 0.0, 0.25, 0.25],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn clamp_matches_logical_order_under_reordered_cuda_storage() -> Result<()> {
    // Elementwise ops read the raw physical buffer and keep the tensor's existing Layout
    // unchanged (`with_layout` only ever changes physical strides), so clamp on a
    // reordered-storage tensor must still read out in the tensor's own logical Shape order
    // via `to_vec()`, exactly like `sign`, `sin`, or the scalar comparisons.
    let device = Device::cuda(0)?;
    let (batch, time) = (Axis::new("batch"), Axis::new("time"));
    let values: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(&values, [batch.of(2), time.of(3)], &device)?
        .with_layout([time, batch])?;
    let clamped = x.clamp(Some(-1.0), Some(1.0))?;
    assert_eq!(clamped.shape(), x.shape());
    let expected: Vec<f64> = values
        .iter()
        .map(|&v| f64::from(v.clamp(-1.0, 1.0)))
        .collect();
    close(
        "reordered-storage clamp(-1, 1)",
        &clamped.to_vec()?,
        &expected,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_axis_roll_matches_hand_computed_values_gradients_and_wrap_around() -> Result<()> {
    // SwinIR's shifted-window attention (network_swinir.py:251,271): a
    // [height, width] feature map, `torch.roll(x, shifts=(-s, -s), dims=(1, 2))`
    // done as two independent single-axis calls. Values 0..12 in
    // [height(4), width(3)] row-major order:
    //   row0 [0,1,2]  row1 [3,4,5]  row2 [6,7,8]  row3 [9,10,11]
    let device = Device::cuda(0)?;
    let (height, width, missing) = (
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("missing"),
    );
    let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let a = Tensor::from_slice(&values, [height.of(4), width.of(3)], &device)?;

    // Positive shift: torch.roll's convention moves element i to i+shift, so
    // output row j holds input row (j - shift) mod 4. shift=1 -> row0 <- row3,
    // row1 <- row0, row2 <- row1, row3 <- row2.
    close(
        "roll height by +1",
        &a.roll(height, 1)?.to_vec()?,
        &[9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    // Negative shift is the exact inverse: row j holds input row (j + 1) mod 4.
    close(
        "roll height by -1",
        &a.roll(height, -1)?.to_vec()?,
        &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0],
    );
    // Wrap-around: a shift outside [-extent, extent] reduces mod the axis
    // extent (4), so +5 and -5 reproduce the +1 and -1 results above exactly.
    close(
        "roll height by +5 (wraps to +1)",
        &a.roll(height, 5)?.to_vec()?,
        &[9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    close(
        "roll height by -5 (wraps to -1)",
        &a.roll(height, -5)?.to_vec()?,
        &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0],
    );
    // Shift 0 and shift = a multiple of the extent are the identity.
    let identity: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    close(
        "roll height by 0 is identity",
        &a.roll(height, 0)?.to_vec()?,
        &identity,
    );
    close(
        "roll height by extent (4) is identity",
        &a.roll(height, 4)?.to_vec()?,
        &identity,
    );

    // Rolling the OTHER named axis leaves height untouched: shift width by +1
    // rotates each row's 3 columns independently (col0 <- col2 within the row).
    close(
        "roll width by +1 (independent axis)",
        &a.roll(width, 1)?.to_vec()?,
        &[2.0, 0.0, 1.0, 5.0, 3.0, 4.0, 8.0, 6.0, 7.0, 11.0, 9.0, 10.0],
    );

    // Gradient: y = roll(x, height, +1), loss = mean(y * w) over w = 1..12
    // (same [height, width] layout). dLoss/dy = w / 12. The chosen composition
    // (concat of two narrow slices) makes backward exactly the inverse roll,
    // dLoss/dx = roll(dLoss/dy, -1): row j of dx holds row (j + 1) mod 4 of
    // w/12 -- computed here from the mathematical definition of roll, not by
    // calling the op under test.
    let x = Tensor::from_slice(&values, [height.of(4), width.of(3)], &device)?.with_grad();
    let weights: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let w = Tensor::from_slice(&weights, [height.of(4), width.of(3)], &device)?;
    x.roll(height, 1)?
        .mul(&w)?
        .mean([height, width])?
        .backward()?;
    close(
        "roll height by +1 gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[
            4.0 / 12.0,
            5.0 / 12.0,
            6.0 / 12.0,
            7.0 / 12.0,
            8.0 / 12.0,
            9.0 / 12.0,
            10.0 / 12.0,
            11.0 / 12.0,
            12.0 / 12.0,
            1.0 / 12.0,
            2.0 / 12.0,
            3.0 / 12.0,
        ],
    );

    // Reordered-storage CUDA case: the same logical tensor built with a
    // transposed physical layout must roll to the identical logical values
    // (`to_vec` follows `Shape.dims()` order, not physical strides).
    let permuted = Tensor::from_slice(&values, [height.of(4), width.of(3)], &device)?
        .with_layout([width, height])?;
    close(
        "roll height by +1 over reordered storage",
        &permuted.roll(height, 1)?.to_vec()?,
        &[9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );

    // A missing axis is rejected before any device work.
    assert!(a.roll(missing, 1).is_err());
    println!("named-axis roll values, gradients, and reordered-storage PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn broadcast_to_composes_an_outer_pairwise_squared_distance_matching_torch_cdist() -> Result<()> {
    // vision/image-encode's Stage A `soft_recon` computes `torch.cdist(P, sites) ** 2`, a
    // (pixel, site) squared-distance matrix from `P: (pixel, coord)` and `sites: (site,
    // coord)` (`stage_a.py:42`). Neither operand's axis set is a subset of the other's --
    // `pixel` and `site` are each missing from the other operand -- so plain elementwise
    // `sub` refuses them ("incomparable axis sets: elementwise broadcasting cannot introduce
    // an implicit outer product", `binary` in `algebra/tensor.rs`). `broadcast_to` gives each
    // operand the axis it lacks, onto one explicit shared `[pixel, site, coord]` shape, so an
    // ordinary `sub`/`mul`/`sum` composes the outer pairwise squared distance with no
    // dedicated outer-product op.
    let device = Device::cuda(0)?;
    let (pixel, site, coord) = (Axis::new("pixel"), Axis::new("site"), Axis::new("coord"));
    // [pixel, coord]: p0=(0,0) p1=(1,2) p2=(3,-1)
    let p_values = [0.0f32, 0.0, 1.0, 2.0, 3.0, -1.0];
    // [site, coord]: s0=(1,0) s1=(-2,3)
    let s_values = [1.0f32, 0.0, -2.0, 3.0];
    let p = Tensor::from_slice(&p_values, [pixel.of(3), coord.of(2)], &device)?.with_grad();
    let sites = Tensor::from_slice(&s_values, [site.of(2), coord.of(2)], &device)?.with_grad();
    let shape = Shape::new([pixel.of(3), site.of(2), coord.of(2)])?;

    // Rejections before any device work: the target must carry every axis `self` has, at
    // `self`'s own extent -- it can add axes, never drop or resize one.
    let missing_coord = Shape::new([pixel.of(3), site.of(2)])?; // drops `coord`, which p has
    assert!(p.broadcast_to(&missing_coord).is_err());
    let wrong_extent = Shape::new([pixel.of(3), site.of(2), coord.of(5)])?; // coord resized
    assert!(p.broadcast_to(&wrong_extent).is_err());

    let p_outer = p.broadcast_to(&shape)?;
    let sites_outer = sites.broadcast_to(&shape)?;
    let delta = p_outer.sub(&sites_outer)?;
    let squared_distance = delta.mul(&delta)?.sum(coord)?;

    // Hand-computed: d2[p, s] = (P[p,0]-S[s,0])^2 + (P[p,1]-S[s,1])^2, pixel-major.
    close(
        "outer-broadcast pairwise squared distance matches torch.cdist(P, sites) ** 2",
        &squared_distance.to_vec()?,
        &[1.0, 13.0, 4.0, 10.0, 5.0, 41.0],
    );

    squared_distance.mean([pixel, site])?.backward()?;
    // Broadcast backward sums the upstream gradient over the axis each side added: `p`'s
    // gradient sums over every `site`, `sites`'s gradient sums over every `pixel`.
    close(
        "broadcast backward sums P's gradient over every site",
        &p.grad().expect("P gradient").to_vec()?,
        &[1.0 / 3.0, -1.0, 1.0, 1.0 / 3.0, 7.0 / 3.0, -5.0 / 3.0],
    );
    close(
        "broadcast backward sums sites's gradient over every pixel",
        &sites.grad().expect("sites gradient").to_vec()?,
        &[-1.0 / 3.0, -1.0 / 3.0, -10.0 / 3.0, 8.0 / 3.0],
    );
    println!("outer-broadcast pairwise squared distance PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn broadcast_to_matches_hand_computed_pairwise_squared_distance_under_reordered_storage()
-> Result<()> {
    // Same outer-broadcast composition as the test above, this time with both operands'
    // PHYSICAL storage transposed relative to their logical [pixel, coord] / [site, coord]
    // order (mirroring `stage_a.py`'s `P`/`sites` tensors, which are ordinary row-major
    // torch tensors but could equally arrive permuted): `broadcast_to` must read through
    // `self.0.layout`'s permuted strides, not assume either operand is already contiguous
    // in its declared axis order.
    let device = Device::cuda(0)?;
    let (pixel, site, coord) = (Axis::new("pixel"), Axis::new("site"), Axis::new("coord"));
    // [pixel, coord]: p0=(0,1) p1=(2,-1) p2=(-3,0) p3=(1,1)
    let p_values = [0.0f32, 1.0, 2.0, -1.0, -3.0, 0.0, 1.0, 1.0];
    // [site, coord]: s0=(0,0) s1=(2,2) s2=(-1,-1)
    let s_values = [0.0f32, 0.0, 2.0, 2.0, -1.0, -1.0];
    let p = Tensor::from_slice(&p_values, [pixel.of(4), coord.of(2)], &device)?
        .with_layout([coord, pixel])?
        .with_grad();
    let sites = Tensor::from_slice(&s_values, [site.of(3), coord.of(2)], &device)?
        .with_layout([coord, site])?
        .with_grad();
    let shape = Shape::new([pixel.of(4), site.of(3), coord.of(2)])?;

    let delta = p.broadcast_to(&shape)?.sub(&sites.broadcast_to(&shape)?)?;
    let squared_distance = delta.mul(&delta)?.sum(coord)?;

    close(
        "reordered-storage outer-broadcast forward",
        &squared_distance.to_vec()?,
        &[1.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0, 29.0, 5.0, 2.0, 2.0, 8.0],
    );

    squared_distance.mean([pixel, site])?.backward()?;
    close(
        "reordered-storage outer-broadcast P gradient",
        &p.grad().expect("P gradient").to_vec()?,
        &[
            -1.0 / 6.0,
            1.0 / 3.0,
            5.0 / 6.0,
            -2.0 / 3.0,
            -5.0 / 3.0,
            -1.0 / 6.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
    );
    close(
        "reordered-storage outer-broadcast sites gradient",
        &sites.grad().expect("sites gradient").to_vec()?,
        &[
            0.0,
            -1.0 / 6.0,
            4.0 / 3.0,
            7.0 / 6.0,
            -2.0 / 3.0,
            -5.0 / 6.0,
        ],
    );
    println!("reordered-storage outer-broadcast pairwise squared distance PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn argmin_matches_hand_computed_oracle_with_a_tie() -> Result<()> {
    // image-encode's `stage_a.py` `hard_eval`: `torch.cdist(P, sites).argmin(1)` picks each
    // pixel's nearest site. Eight pixels, three sites (`site` extent 3); pixel 3 is exactly
    // equidistant from site 0 and site 1 (squared distances 25.0 and 25.0) and must resolve to
    // site 0, the first logical coordinate, exactly matching `Tensor::min`'s own tie rule. Site
    // 2 (a distant, never-nearest site) is never a winner, setting up the empty-bucket case the
    // scatter-add/bincount tests below exercise.
    let device = Device::cuda(0)?;
    let (pixel, site) = (Axis::new("pixel"), Axis::new("site"));
    #[rustfmt::skip]
    let distances = Tensor::from_slice(
        &[
            0.0, 100.0, 10000.0,
            1.0, 81.0, 10000.0,
            81.0, 1.0, 10000.0,
            25.0, 25.0, 10000.0, // tie: site 0 and site 1
            4.0, 64.0, 10000.0,
            64.0, 4.0, 10000.0,
            9.0, 49.0, 10000.0,
            49.0, 9.0, 10000.0,
        ],
        [pixel.of(8), site.of(3)],
        &device,
    )?;
    assert_eq!(distances.argmin(site)?, vec![0, 0, 1, 0, 0, 1, 0, 1]);

    // Reordered physical storage (site-major rather than pixel-major) must read the same
    // logical values through the permuted strides, mirroring `gather`'s own reordered-storage
    // coverage.
    let reordered = distances.with_layout([site, pixel])?;
    assert_eq!(reordered.argmin(site)?, vec![0, 0, 1, 0, 0, 1, 0, 1]);

    let error = distances
        .argmin(Axis::new("missing"))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("missing axis missing#"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn argmin_errors_on_a_group_with_no_finite_candidate() -> Result<()> {
    // usize cannot name "no winner", so unlike `min` (which returns NaN), a group with no
    // finite candidate at all is an error rather than a silently meaningless index.
    let device = Device::cuda(0)?;
    let (row, candidate) = (Axis::new("row"), Axis::new("candidate"));
    let values = Tensor::from_slice(
        &[
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            3.0,
            f32::NAN,
            2.0,
        ],
        [row.of(2), candidate.of(3)],
        &device,
    )?;
    let error = values.argmin(candidate).err().unwrap().to_string();
    assert!(error.contains("no finite candidate"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn scatter_add_and_bincount_match_hand_computed_oracle_with_an_empty_bucket_and_gradient()
-> Result<()> {
    // image-encode's `stage_a.py` `hard_eval`: `torch.bincount(lab, minlength=k)` and
    // `torch.zeros(k, 3).index_add_(0, lab, I)`, given the labels the argmin test above computed
    // for the same eight pixels and three sites. Site 2 is picked by no pixel: its count and sum
    // must come out exact zero, not an error.
    let device = Device::cuda(0)?;
    let (pixel, site, channel) = (Axis::new("pixel"), Axis::new("site"), Axis::new("channel"));
    let labels = [0usize, 0, 1, 0, 0, 1, 0, 1];
    #[rustfmt::skip]
    let colours = Tensor::from_slice(
        &[
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 0.0,
            5.0, 0.0, 0.0,
            0.0, 6.0, 0.0,
            7.0, 0.0, 0.0,
            0.0, 8.0, 0.0,
        ],
        [pixel.of(8), channel.of(3)],
        &device,
    )?
    .with_grad();

    let sums = colours.scatter_add(pixel, &labels, site, 3)?;
    assert_eq!(sums.shape(), &Shape::new([site.of(3), channel.of(3)])?);
    close(
        "scatter-add per-site colour sums, including the empty site",
        &sums.to_vec()?,
        &[19.0, 0.0, 0.0, 0.0, 17.0, 0.0, 0.0, 0.0, 0.0],
    );

    let counts = Tensor::bincount(&labels, site, 3, &device)?;
    assert_eq!(counts.shape(), &Shape::new([site.of(3)])?);
    close(
        "bincount, including the empty site",
        &counts.to_vec()?,
        &[5.0, 3.0, 0.0],
    );
    // "Mean colour per site" (`hard_eval`'s actual metric) is the caller's own division of
    // `scatter_add`'s sum by `bincount`'s count, unclamped: site 2 divides 0.0 by 0.0, IEEE NaN,
    // exactly as `x.div(y)` is documented to do with no built-in safe-denominator behaviour.
    let sums_flat = sums.to_vec()?;
    let counts_flat = counts.to_vec()?;
    for (site_index, &count) in counts_flat.iter().enumerate() {
        for channel_index in 0..3 {
            let mean = sums_flat[site_index * 3 + channel_index] / count;
            match site_index {
                0 => {
                    assert!((f64::from(mean) - [19.0 / 5.0, 0.0, 0.0][channel_index]).abs() < 1e-5)
                }
                1 => {
                    assert!((f64::from(mean) - [0.0, 17.0 / 3.0, 0.0][channel_index]).abs() < 1e-5)
                }
                _ => assert!(
                    mean.is_nan(),
                    "empty site's unclamped mean is NaN, not zero"
                ),
            }
        }
    }

    // A weighted reduction downstream makes the gradient test discriminate bucket routing: a
    // transposed or misrouted backward would not reproduce this pattern the way a uniform
    // upstream gradient could not catch.
    #[rustfmt::skip]
    let weights = Tensor::from_slice(
        &[
            1.0, 11.0, 21.0,
            101.0, 111.0, 121.0,
            201.0, 211.0, 221.0,
        ],
        [site.of(3), channel.of(3)],
        &device,
    )?;
    sums.mul(&weights)?.sum([site, channel])?.backward()?;
    #[rustfmt::skip]
    close(
        "scatter-add gradient routes each pixel's own weighted bucket back to it",
        &colours.grad().expect("colours gradient").to_vec()?,
        &[
            1.0, 11.0, 21.0,
            1.0, 11.0, 21.0,
            101.0, 111.0, 121.0,
            1.0, 11.0, 21.0,
            1.0, 11.0, 21.0,
            101.0, 111.0, 121.0,
            1.0, 11.0, 21.0,
            101.0, 111.0, 121.0,
        ],
    );

    // Rejected before any device work: a mismatched index length, an out-of-range bucket, a
    // zero bucket count, and an axis this tensor does not have.
    assert!(colours.scatter_add(pixel, &labels[..7], site, 3).is_err());
    assert!(
        colours
            .scatter_add(pixel, &[0, 1, 2, 0, 0, 1, 0, 3], site, 3)
            .is_err()
    );
    assert!(colours.scatter_add(pixel, &labels, site, 0).is_err());
    assert!(
        colours
            .scatter_add(Axis::new("missing"), &labels, site, 3)
            .is_err()
    );
    assert!(Tensor::bincount(&[], site, 3, &device).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn scatter_add_matches_hand_computed_oracle_under_reordered_pixel_storage() -> Result<()> {
    // Same routing contract as the test above, this time with the source tensor's PHYSICAL
    // storage transposed relative to its logical [pixel, channel] order (mirroring a value
    // tensor copied through `with_layout`), so `scatter_add` must read through the permuted
    // strides in `self.0.layout` rather than assume row-major storage -- exactly the case
    // `gather_matches_hand_computed_oracle_under_reordered_table_storage` covers for `gather`.
    let device = Device::cuda(0)?;
    let (pixel, site, channel) = (Axis::new("pixel"), Axis::new("site"), Axis::new("channel"));
    let values = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [pixel.of(4), channel.of(2)],
        &device,
    )?
    .with_layout([channel, pixel])?
    .with_grad();

    let labels = [1usize, 0, 1, 0];
    let sums = values.scatter_add(pixel, &labels, site, 2)?;
    close(
        "reordered scatter-add forward",
        &sums.to_vec()?,
        &[10.0, 12.0, 6.0, 8.0],
    );

    let weights = Tensor::from_slice(
        &[1000.0, 2000.0, 3000.0, 4000.0],
        [site.of(2), channel.of(2)],
        &device,
    )?;
    sums.mul(&weights)?.sum([site, channel])?.backward()?;
    close(
        "reordered scatter-add gradient",
        &values.grad().expect("values gradient").to_vec()?,
        &[
            3000.0, 4000.0, 1000.0, 2000.0, 3000.0, 4000.0, 1000.0, 2000.0,
        ],
    );
    Ok(())
}

/// Issue #80: `energy-output/fit_readout_sweep.py:46` computes `q = a + torch.logsumexp(z,
/// dim=1)` as one of its four swept readouts. Hand-computed literals: row 0's inputs are
/// `[ln(1), ln(2), ln(3)]`, whose exponentials sum to `6`, so `logsumexp = ln(6)`; row 1's
/// are `[ln(4), ln(4), ln(8)]`, summing to `16`, so `logsumexp = ln(16)`. The gradient of
/// `logsumexp` is exactly `softmax` along the reduced axis (`exp(x - m) / sum(exp(x - m))`,
/// independent of the detached shift `m`), so `mean(batch)`'s backward should land precisely
/// on each row's softmax scaled by `1 / batch_extent`.
#[test]
#[ignore = "requires CUDA"]
fn logsumexp_matches_hand_computed_literals_and_gradient_equals_softmax() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate) = (Axis::new("batch"), Axis::new("candidate"));
    let values = Tensor::from_slice(
        &[
            0.0_f32,
            2.0_f32.ln(),
            3.0_f32.ln(),
            4.0_f32.ln(),
            4.0_f32.ln(),
            8.0_f32.ln(),
        ],
        [batch.of(2), candidate.of(3)],
        &device,
    )?
    .with_grad();

    let reduced = values.logsumexp(candidate)?;
    assert_eq!(reduced.shape(), &Shape::new([batch.of(2)])?);
    close(
        "logsumexp hand-computed forward",
        &reduced.to_vec()?,
        &[6.0_f64.ln(), 16.0_f64.ln()],
    );

    reduced.mean(batch)?.backward()?;
    close(
        "logsumexp gradient equals softmax/batch_extent",
        &values.grad().expect("values gradient").to_vec()?,
        &[
            0.5 / 6.0,
            1.0 / 6.0,
            1.5 / 6.0,
            0.5 / 4.0,
            0.5 / 4.0,
            1.0 / 4.0,
        ],
    );

    let error = values
        .logsumexp(Axis::new("missing"))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("missing axis missing#"), "{error}");
    println!("logsumexp hand-computed forward and gradient PASS");
    Ok(())
}

/// Same evidence bar, at a magnitude where naively exponentiating `x` directly (without
/// subtracting the per-row maximum first) would overflow `f32`. Row 0 has one dominant
/// entry (`1000` against two `0`s, a large spread); row 1 is a near-tie shifted onto the
/// same large magnitude (`1000 + ln(k)`, a small spread). Both rows stay exactly the shape
/// of the small-magnitude case above -- `logsumexp` is shift-invariant, so row 1's softmax
/// gradient reproduces row 0's from the previous test bit for bit.
#[test]
#[ignore = "requires CUDA"]
fn logsumexp_large_magnitude_input_does_not_overflow() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate) = (Axis::new("batch"), Axis::new("candidate"));
    let values = Tensor::from_slice(
        &[
            1000.0_f32,
            0.0_f32,
            0.0_f32,
            1000.0_f32,
            1000.0_f32 + 2.0_f32.ln(),
            1000.0_f32 + 3.0_f32.ln(),
        ],
        [batch.of(2), candidate.of(3)],
        &device,
    )?
    .with_grad();

    let reduced = values.logsumexp(candidate)?;
    close(
        "logsumexp large-magnitude forward",
        &reduced.to_vec()?,
        &[1000.0, 1000.0 + 6.0_f64.ln()],
    );

    reduced.mean(batch)?.backward()?;
    close(
        "logsumexp large-magnitude gradient equals softmax/batch_extent",
        &values.grad().expect("values gradient").to_vec()?,
        &[0.5, 0.0, 0.0, 0.5 / 6.0, 1.0 / 6.0, 1.5 / 6.0],
    );
    println!("logsumexp large-magnitude forward and gradient PASS");
    Ok(())
}

/// Documents the one place `logsumexp` diverges from `torch.logsumexp`: a group with no
/// finite candidate at all. `Tensor::max`'s own convention (see its doc comment) ignores
/// non-finite candidates and returns `NaN`, with zero derivative, for a group that has no
/// finite one -- rather than PyTorch's `-infinity` for an all-`-infinity` `max`. `logsumexp`
/// reuses `max` verbatim for its shift `m`, so an all-`-infinity` (or otherwise all
/// non-finite) group produces `m = NaN` and therefore `logsumexp = NaN` too, where
/// `torch.logsumexp` returns `-infinity`. Unlike `max`, `logsumexp` has no dedicated
/// backward rule to zero that group's gradient: `NaN` propagates through the ordinary
/// `sub`/`exp`/`sum`/`ln` composition, so the gradient is `NaN`, not zero.
#[test]
#[ignore = "requires CUDA"]
fn logsumexp_all_nonfinite_group_returns_nan_unlike_pytorchs_negative_infinity() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, candidate) = (Axis::new("batch"), Axis::new("candidate"));
    let values = Tensor::from_slice(
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        [batch.of(1), candidate.of(3)],
        &device,
    )?
    .with_grad();

    let reduced = values.logsumexp(candidate)?;
    assert!(
        reduced.to_vec()?[0].is_nan(),
        "expected NaN, matching max's empty-group convention, not PyTorch's -infinity"
    );

    reduced.mean(batch)?.backward()?;
    let gradient = values.grad().expect("values gradient").to_vec()?;
    assert!(
        gradient.iter().all(|g| g.is_nan()),
        "logsumexp has no dedicated backward rule to zero an empty group's gradient the way max does: {gradient:?}"
    );
    println!("logsumexp all-nonfinite group PASS");
    Ok(())
}

/// Reordered-storage CUDA case: physical storage is transposed relative to the declared
/// `[batch, candidate]` logical order, so `logsumexp`'s internal `max`/`sub`/`sum` calls
/// must all read through `self.0.layout`'s permuted strides rather than assume contiguous
/// storage in axis order. Row 0's inputs are `[ln(1), ln(1), ln(2), ln(4)]` (exponentials
/// sum to `8`); row 1's are `[ln(3), ln(5), ln(5), ln(3)]` (sum to `16`).
#[test]
#[ignore = "requires CUDA"]
fn logsumexp_reordered_storage_matches_hand_computed_forward_and_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, group) = (Axis::new("row"), Axis::new("group"));
    let values = Tensor::from_slice(
        &[
            0.0_f32,
            0.0_f32,
            2.0_f32.ln(),
            4.0_f32.ln(),
            3.0_f32.ln(),
            5.0_f32.ln(),
            5.0_f32.ln(),
            3.0_f32.ln(),
        ],
        [row.of(2), group.of(4)],
        &device,
    )?
    .with_layout([group, row])?
    .with_grad();

    let reduced = values.logsumexp(group)?;
    assert_eq!(reduced.shape(), &Shape::new([row.of(2)])?);
    close(
        "logsumexp reordered-storage forward",
        &reduced.to_vec()?,
        &[8.0_f64.ln(), 16.0_f64.ln()],
    );

    reduced.mean(row)?.backward()?;
    close(
        "logsumexp reordered-storage gradient equals softmax/row_extent",
        &values.grad().expect("values gradient").to_vec()?,
        &[
            0.5 / 8.0,
            0.5 / 8.0,
            1.0 / 8.0,
            2.0 / 8.0,
            1.5 / 16.0,
            2.5 / 16.0,
            2.5 / 16.0,
            1.5 / 16.0,
        ],
    );
    println!("logsumexp reordered-storage forward and gradient PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn masked_softmax_zeroes_masked_positions_and_a_fully_masked_bag_returns_zero() -> Result<()> {
    // gastric `InterfaceMIL`/`PhaseSeparableFusion` pattern: `scores.masked_fill(~valid,
    // -inf).softmax(dim)` over a variable-length token bank, but with one bag whose validity
    // mask is entirely zero -- PyTorch's composition leaves that whole row `NaN` and every
    // consumer calls `nan_to_num(neginf=0.0)` on it by hand; `masked_softmax` folds that
    // cleanup into the op itself.
    let device = Device::cuda(0)?;
    let (bag, token) = (Axis::new("bag"), Axis::new("token"));
    // Bag 0: token 1 is masked out; its huge score (5.0) must never influence the result.
    // Bag 1: every token is masked out.
    let scores = Tensor::from_slice(
        &[1.0, 5.0, 2.0, 3.0, -1.0, 0.5],
        [bag.of(2), token.of(3)],
        &device,
    )?
    .with_grad();
    // Built in [token, bag] order -- the transpose of `scores`'s own axis order -- so the
    // mask must be realigned before use, exactly like `masked_mean`'s own test.
    let mask = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [token.of(3), bag.of(2)],
        &device,
    )?;

    let probability = scores.masked_softmax(token, &mask)?;
    close(
        "masked softmax forward",
        &probability.to_vec()?,
        &[0.2689414213699951, 0.0, 0.7310585786300049, 0.0, 0.0, 0.0],
    );

    let weights = Tensor::from_slice(&[1.0, 2.0, 4.0], [token.of(3)], &device)?;
    probability.mul(&weights)?.mean([bag, token])?.backward()?;
    close(
        "masked softmax gradient",
        &scores.grad().unwrap().to_vec()?,
        &[
            -0.09830596662074094,
            0.0,
            0.09830596662074088,
            0.0,
            0.0,
            0.0,
        ],
    );

    let grad_mask = Tensor::from_slice(&[1.0; 6], [bag.of(2), token.of(3)], &device)?.with_grad();
    assert!(scores.detach().masked_softmax(token, &grad_mask).is_err());
    let fractional = Tensor::from_slice(
        &[1.0, 0.5, 1.0, 0.0, 1.0, 0.0],
        [bag.of(2), token.of(3)],
        &device,
    )?;
    assert!(scores.detach().masked_softmax(token, &fractional).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn masked_softmax_matches_hand_computed_oracle_under_reordered_storage() -> Result<()> {
    // Same capability, asymmetric extents (2 bags x 4 tokens) with both `scores` and `mask`
    // stored transposed relative to their logical [bag, token] axis order, so the composition
    // must read through `with_layout`'s permuted strides rather than assume either operand is
    // already contiguous in its declared axis order. Bag 0 keeps 3 of 4 tokens valid (an
    // ordinary partial mask); bag 1 is fully masked.
    let device = Device::cuda(0)?;
    let (bag, token) = (Axis::new("bag"), Axis::new("token"));
    let scores = Tensor::from_slice(
        &[0.5, -0.5, 2.0, 7.0, 1.0, 1.0, 1.0, 1.0],
        [bag.of(2), token.of(4)],
        &device,
    )?
    .with_layout([token, bag])?
    .with_grad();
    let mask = Tensor::from_slice(
        &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [bag.of(2), token.of(4)],
        &device,
    )?
    .with_layout([token, bag])?;

    let probability = scores.masked_softmax(token, &mask)?;
    close(
        "masked softmax forward under reordered storage",
        &probability.to_vec()?,
        &[
            0.17095278019779028,
            0.0628900132458675,
            0.7661572065563422,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    );

    let weights = Tensor::from_slice(&[2.0, 0.5, 1.0, 100.0], [token.of(4)], &device)?;
    probability.mul(&weights)?.mean([bag, token])?.backward()?;
    close(
        "masked softmax gradient under reordered storage",
        &scores.grad().unwrap().to_vec()?,
        &[
            0.018387942305745593,
            -0.005027331543869745,
            -0.013360610761875839,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    );
    Ok(())
}

// -- nn-act-smooth: ELU, CELU, SELU, Softplus (module), LogSigmoid, Mish,
// GLU, PReLU, LogSoftmax, Softmin, Softmax2d. --

#[test]
fn smooth_activation_modules_reject_invalid_configuration() -> Result<()> {
    let (row, feature, channel, height, width) = (
        Axis::new("smooth_row"),
        Axis::new("smooth_feature"),
        Axis::new("smooth_channel"),
        Axis::new("smooth_height"),
        Axis::new("smooth_width"),
    );

    // Negative alpha is PyTorch's own documented domain (any nonzero alpha) -- see
    // `Tensor::elu`/`Tensor::celu` for the sign-algebra argument.
    assert!(ELU::new(0.0).is_err());
    assert!(ELU::new(f32::NAN).is_err());
    assert!(ELU::new(f32::INFINITY).is_err());
    assert!(ELU::new(1.0).is_ok());
    assert!(ELU::new(-1.0).is_ok());

    assert!(CELU::new(0.0).is_err());
    assert!(CELU::new(f32::NAN).is_err());
    assert!(CELU::new(1.0).is_ok());
    assert!(CELU::new(-0.5).is_ok());

    assert!(Softplus::new().beta(0.0).is_err());
    assert!(Softplus::new().beta(-1.0).is_err());
    assert!(Softplus::new().beta(f32::NAN).is_err());
    assert!(Softplus::new().threshold(f32::NAN).is_err());
    assert!(Softplus::new().threshold(f32::INFINITY).is_err());
    assert!(Softplus::new().beta(2.0).is_ok());

    let full = Shape::new([row.of(2), feature.of(6)])?;
    assert_eq!(
        GLU::new(feature).output_shape(&full)?,
        Shape::new([row.of(2), feature.of(3)])?
    );
    let odd = Shape::new([row.of(2), feature.of(5)])?;
    assert!(GLU::new(feature).output_shape(&odd).is_err());
    assert!(GLU::new(channel).output_shape(&full).is_err());

    assert!(
        PReLU::channel(channel)
            .output_shape(&Shape::new([row.of(2), feature.of(5)])?)
            .is_err()
    );
    assert!(
        PReLU::channel(channel)
            .output_shape(&Shape::new([row.of(2), channel.of(3)])?)
            .is_ok()
    );
    assert!(PReLU::shared().output_shape(&full).is_ok());

    assert!(LogSoftmax::new(channel).output_shape(&full).is_err());
    assert!(LogSoftmax::new(feature).output_shape(&full).is_ok());
    assert!(Softmin::new(channel).output_shape(&full).is_err());
    assert!(Softmin::new(feature).output_shape(&full).is_ok());

    assert!(Softmax2d::new(channel, channel, width).is_err());
    assert!(Softmax2d::new(channel, height, channel).is_err());
    let softmax2d = Softmax2d::new(channel, height, width)?;
    assert!(
        softmax2d
            .output_shape(&Shape::new([channel.of(3), height.of(2)])?)
            .is_err()
    );
    assert!(
        softmax2d
            .output_shape(&Shape::new([
                row.of(1),
                channel.of(3),
                height.of(2),
                width.of(4),
            ])?)
            .is_err()
    );
    assert!(
        softmax2d
            .output_shape(&Shape::new([channel.of(3), height.of(2), width.of(4)])?)
            .is_ok()
    );

    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn elu_and_celu_match_hand_computed_oracle_under_reordered_asymmetric_storage() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("elu_row"), Axis::new("elu_col"));
    // Asymmetric extents (3 x 67 = 201), spanning the x == 0 boundary at index 100.
    let values: Vec<f64> = (0..201).map(|i| (i as f64 - 100.0) / 20.0).collect();
    let n = values.len() as f64;

    let alpha = 1.5_f64;
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(3), col.of(67)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([col, row])?;
    let output = ELU::new(alpha as f32)?.forward(&input)?;
    let expected: Vec<f64> = values
        .iter()
        .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
        .collect();
    close("ELU module forward", &output.to_vec()?, &expected);
    output.mean([row, col])?.backward()?;
    let expected_gradient: Vec<f64> = values
        .iter()
        .map(|&x| (if x > 0.0 { 1.0 } else { alpha * x.exp() }) / n)
        .collect();
    close(
        "ELU module derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    assert!(input.elu(f32::INFINITY).is_err());

    let celu_alpha = 0.8_f64;
    let celu_leaf = input.detach().with_layout([row, col])?.with_grad();
    let celu_input = celu_leaf.with_layout([col, row])?;
    let celu_output = CELU::new(celu_alpha as f32)?.forward(&celu_input)?;
    let expected_celu: Vec<f64> = values
        .iter()
        .map(|&x| {
            if x > 0.0 {
                x
            } else {
                celu_alpha * ((x / celu_alpha).exp() - 1.0)
            }
        })
        .collect();
    close(
        "CELU module forward",
        &celu_output.to_vec()?,
        &expected_celu,
    );
    celu_output.mean([row, col])?.backward()?;
    let expected_celu_gradient: Vec<f64> = values
        .iter()
        .map(|&x| (if x > 0.0 { 1.0 } else { (x / celu_alpha).exp() }) / n)
        .collect();
    close(
        "CELU module derivative",
        &celu_leaf.grad().unwrap().to_vec()?,
        &expected_celu_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn selu_matches_hand_computed_oracle_under_reordered_asymmetric_storage() -> Result<()> {
    const SELU_ALPHA: f64 = 1.6732632423543772;
    const SELU_SCALE: f64 = 1.0507009873554805;
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("selu_row"), Axis::new("selu_col"));
    // Asymmetric extents (7 x 25 = 175).
    let values: Vec<f64> = (0..175).map(|i| (i as f64 - 87.0) / 25.0).collect();
    let n = values.len() as f64;
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(7), col.of(25)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([col, row])?;
    let output = SELU.forward(&input)?;
    let expected: Vec<f64> = values
        .iter()
        .map(|&x| {
            SELU_SCALE
                * if x > 0.0 {
                    x
                } else {
                    SELU_ALPHA * (x.exp() - 1.0)
                }
        })
        .collect();
    close("SELU forward", &output.to_vec()?, &expected);
    output.mean([row, col])?.backward()?;
    let expected_gradient: Vec<f64> = values
        .iter()
        .map(|&x| SELU_SCALE * (if x > 0.0 { 1.0 } else { SELU_ALPHA * x.exp() }) / n)
        .collect();
    close(
        "SELU derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn softplus_module_matches_tensor_op_with_default_and_explicit_parameters() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (
        Axis::new("softplus_module_row"),
        Axis::new("softplus_module_col"),
    );
    // Asymmetric extents (7 x 25 = 175), within [-20, 20] so the default
    // threshold never engages.
    let values: Vec<f64> = (0..175).map(|i| (i as f64 - 87.0) / 8.0).collect();
    let n = values.len() as f64;

    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(7), col.of(25)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([col, row])?;
    let output = Softplus::new().forward(&input)?;
    let expected: Vec<f64> = values
        .iter()
        .map(|&x| if x > 20.0 { x } else { (1.0 + x.exp()).ln() })
        .collect();
    close(
        "Softplus module default forward",
        &output.to_vec()?,
        &expected,
    );
    output.mean([row, col])?.backward()?;
    let expected_gradient: Vec<f64> = values
        .iter()
        .map(|&x| {
            (if x > 20.0 {
                1.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }) / n
        })
        .collect();
    close(
        "Softplus module default derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    // Explicit override, beta = 2, threshold = 3 -- this range's positive tail
    // (up to 10.875) crosses beta * x > threshold.
    let explicit_leaf = input.detach().with_layout([row, col])?.with_grad();
    let explicit_input = explicit_leaf.with_layout([col, row])?;
    let explicit = Softplus::new().beta(2.0)?.threshold(3.0)?;
    let explicit_output = explicit.forward(&explicit_input)?;
    let expected_explicit: Vec<f64> = values
        .iter()
        .map(|&x| {
            let scaled = 2.0 * x;
            if scaled > 3.0 {
                x
            } else {
                (1.0 + scaled.exp()).ln() / 2.0
            }
        })
        .collect();
    close(
        "Softplus module explicit forward",
        &explicit_output.to_vec()?,
        &expected_explicit,
    );
    explicit_output.mean([row, col])?.backward()?;
    let expected_explicit_gradient: Vec<f64> = values
        .iter()
        .map(|&x| {
            let scaled = 2.0 * x;
            let derivative = if scaled > 3.0 {
                1.0
            } else {
                1.0 / (1.0 + (-scaled).exp())
            };
            derivative / n
        })
        .collect();
    close(
        "Softplus module explicit derivative",
        &explicit_leaf.grad().unwrap().to_vec()?,
        &expected_explicit_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn log_sigmoid_matches_hand_computed_oracle_under_reordered_asymmetric_storage() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("log_sigmoid_row"), Axis::new("log_sigmoid_col"));
    // Asymmetric extents (7 x 25 = 175), including large-magnitude values that
    // cross softplus's internal linear seam.
    let values: Vec<f64> = (0..175).map(|i| (i as f64 - 87.0) / 4.0).collect();
    let n = values.len() as f64;
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(7), col.of(25)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([col, row])?;
    let output = LogSigmoid.forward(&input)?;
    let expected: Vec<f64> = values.iter().map(|&x| -(1.0 + (-x).exp()).ln()).collect();
    close("LogSigmoid forward", &output.to_vec()?, &expected);
    output.mean([row, col])?.backward()?;
    let expected_gradient: Vec<f64> = values
        .iter()
        .map(|&x| (1.0 / (1.0 + x.exp())) / n)
        .collect();
    close(
        "LogSigmoid derivative (sigmoid(-x))",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn mish_matches_hand_computed_oracle_under_reordered_asymmetric_storage() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("mish_row"), Axis::new("mish_col"));
    // Asymmetric extents (7 x 25 = 175).
    let values: Vec<f64> = (0..175).map(|i| (i as f64 - 87.0) / 20.0).collect();
    let n = values.len() as f64;
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(7), col.of(25)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([col, row])?;
    let output = Mish.forward(&input)?;
    let expected: Vec<f64> = values
        .iter()
        .map(|&x| {
            let softplus = (1.0 + x.exp()).ln();
            x * softplus.tanh()
        })
        .collect();
    close("Mish forward", &output.to_vec()?, &expected);
    output.mean([row, col])?.backward()?;
    // d/dx [x * tanh(softplus(x))] = tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2).
    let expected_gradient: Vec<f64> = values
        .iter()
        .map(|&x| {
            let softplus = (1.0 + x.exp()).ln();
            let tanh_sp = softplus.tanh();
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            (tanh_sp + x * sigmoid * (1.0 - tanh_sp * tanh_sp)) / n
        })
        .collect();
    close(
        "Mish derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn glu_splits_named_axis_and_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, feature) = (Axis::new("glu_row"), Axis::new("glu_feature"));
    // Asymmetric extents (3 rows x 8 features, split into two 4-wide halves).
    let values: Vec<f64> = (0..24).map(|i| (i as f64 - 12.0) / 3.0).collect();
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(3), feature.of(8)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([feature, row])?;
    let output = GLU::new(feature).forward(&input)?;
    assert_eq!(output.extent(feature)?, 4);
    let n_out = 12.0_f64; // 3 rows x 4 output features

    let mut expected = Vec::with_capacity(12);
    let mut expected_gradient = vec![0.0_f64; 24];
    for r in 0..3usize {
        for c in 0..4usize {
            let a = values[r * 8 + c];
            let b = values[r * 8 + 4 + c];
            let sigmoid_b = 1.0 / (1.0 + (-b).exp());
            expected.push(a * sigmoid_b);
            expected_gradient[r * 8 + c] = sigmoid_b / n_out;
            expected_gradient[r * 8 + 4 + c] = a * sigmoid_b * (1.0 - sigmoid_b) / n_out;
        }
    }
    close("GLU forward", &output.to_vec()?, &expected);
    output.mean([row, feature])?.backward()?;
    close(
        "GLU derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    assert!(
        GLU::new(feature)
            .output_shape(&Shape::new([row.of(3), feature.of(7)])?)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn prelu_shared_and_per_channel_match_hand_computed_oracle_including_weight_gradient() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (row, channel) = (Axis::new("prelu_row"), Axis::new("prelu_channel"));
    // Asymmetric extents (5 rows x 3 channels).
    let values: Vec<f64> = (0..15).map(|i| (i as f64 - 7.0) / 2.0).collect();
    let n = values.len() as f64;

    // Shared: one weight for every element, default init 0.25 (PyTorch's
    // num_parameters=1 default).
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(5), channel.of(3)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([channel, row])?;
    assert!(PReLU::shared().forward(&input).is_err());
    let mut shared = PReLU::shared();
    shared.build(input.shape(), &device, 0)?;
    assert_eq!(shared.named_parameters().len(), 1);
    let weight = shared.parameter("weight")?;
    close(
        "PReLU shared weight init",
        &weight.tensor().to_vec()?,
        &[0.25],
    );
    let output = shared.forward(&input)?;
    let w = 0.25_f64;
    let expected: Vec<f64> = values
        .iter()
        .map(|&x| if x > 0.0 { x } else { w * x })
        .collect();
    close("PReLU shared forward", &output.to_vec()?, &expected);
    output.mean([row, channel])?.backward()?;
    let expected_gradient: Vec<f64> = values
        .iter()
        .map(|&x| (if x > 0.0 { 1.0 } else { w }) / n)
        .collect();
    close(
        "PReLU shared input derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    let expected_weight_gradient: f64 = values
        .iter()
        .map(|&x| if x > 0.0 { 0.0 } else { x / n })
        .sum();
    close(
        "PReLU shared weight derivative",
        &weight.grad().unwrap().to_vec()?,
        &[expected_weight_gradient],
    );

    // Per-channel: one weight per entry of `channel` (PyTorch's
    // num_parameters=C), perturbed away from the shared init so the oracle
    // exercises three distinct slopes.
    let channel_leaf = input.detach().with_layout([row, channel])?.with_grad();
    let channel_input = channel_leaf.with_layout([channel, row])?;
    let mut per_channel = PReLU::channel(channel);
    per_channel.build(channel_input.shape(), &device, 0)?;
    let channel_weight = per_channel.parameter("weight")?;
    close(
        "PReLU per-channel weight init",
        &channel_weight.tensor().to_vec()?,
        &[0.25, 0.25, 0.25],
    );
    channel_weight.set_values(&[0.1, 0.25, 0.6])?;
    let channel_output = per_channel.forward(&channel_input)?;
    let weights = [0.1_f64, 0.25, 0.6];
    let mut expected_channel = Vec::with_capacity(15);
    let mut expected_channel_gradient = vec![0.0_f64; 15];
    let mut expected_channel_weight_gradient = [0.0_f64; 3];
    for r in 0..5usize {
        for c in 0..3usize {
            let x = values[r * 3 + c];
            let w = weights[c];
            expected_channel.push(if x > 0.0 { x } else { w * x });
            expected_channel_gradient[r * 3 + c] = (if x > 0.0 { 1.0 } else { w }) / n;
            if x <= 0.0 {
                expected_channel_weight_gradient[c] += x / n;
            }
        }
    }
    close(
        "PReLU per-channel forward",
        &channel_output.to_vec()?,
        &expected_channel,
    );
    channel_output.mean([row, channel])?.backward()?;
    close(
        "PReLU per-channel input derivative",
        &channel_leaf.grad().unwrap().to_vec()?,
        &expected_channel_gradient,
    );
    close(
        "PReLU per-channel weight derivative",
        &channel_weight.grad().unwrap().to_vec()?,
        &expected_channel_weight_gradient,
    );

    assert!(
        PReLU::channel(channel)
            .output_shape(&Shape::new([row.of(5)])?)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn log_softmax_matches_hand_computed_oracle_and_stays_finite_for_large_logits() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, class) = (Axis::new("log_softmax_row"), Axis::new("log_softmax_class"));
    // Asymmetric extents (3 rows x 5 classes); the middle row's raw logits are
    // large enough (~1000) that a literal exp/sum/ln composition without the
    // logsumexp shift would overflow toward `inf`.
    let values: Vec<f64> = vec![
        0.5, -1.0, 2.0, 0.0, 3.0, 1000.0, 1001.0, 999.0, 1000.5, 998.0, -2.0, -1.0, 0.0, 1.0, 2.0,
    ];
    let n = values.len() as f64;
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(3), class.of(5)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([class, row])?;
    let output = LogSoftmax::new(class).forward(&input)?;

    let mut expected = Vec::with_capacity(15);
    let mut softmax_rows = Vec::with_capacity(15);
    for row_values in values.chunks(5) {
        let max = row_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_values.iter().map(|&x| (x - max).exp()).sum();
        let logsumexp = max + sum_exp.ln();
        for &x in row_values {
            expected.push(x - logsumexp);
            softmax_rows.push((x - max).exp() / sum_exp);
        }
    }
    let output_values = output.to_vec()?;
    close("LogSoftmax forward", &output_values, &expected);
    assert!(output_values.iter().all(|v| v.is_finite()));

    output.mean([row, class])?.backward()?;
    // d(log_softmax_i)/dx_j = delta_ij - softmax_j; the uniform 1/n upstream
    // gradient from `mean` over each 5-wide row gives grad_j = (1 - 5*p_j)/n.
    let mut expected_gradient = Vec::with_capacity(15);
    for chunk in softmax_rows.chunks(5) {
        for &p in chunk {
            expected_gradient.push((1.0 - 5.0 * p) / n);
        }
    }
    close(
        "LogSoftmax derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn softmin_matches_hand_computed_oracle_under_reordered_storage() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, class) = (Axis::new("softmin_row"), Axis::new("softmin_class"));
    // Asymmetric extents (2 rows x 6 classes).
    let values: Vec<f64> = vec![
        0.5, -1.0, 2.0, 0.0, 3.0, -0.5, 4.0, -2.0, 1.0, 0.0, 2.5, -1.5,
    ];
    let weights = [2.0_f64, 0.5, 1.0, 3.0, 0.25, 1.5];
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [row.of(2), class.of(6)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([class, row])?;
    let output = Softmin::new(class).forward(&input)?;

    let mut expected = Vec::with_capacity(12);
    let mut prob_rows: Vec<Vec<f64>> = Vec::with_capacity(2);
    for row_values in values.chunks(6) {
        let negated: Vec<f64> = row_values.iter().map(|&x| -x).collect();
        let max = negated.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = negated.iter().map(|&x| (x - max).exp()).sum();
        let probabilities: Vec<f64> = negated.iter().map(|&x| (x - max).exp() / sum_exp).collect();
        expected.extend(&probabilities);
        prob_rows.push(probabilities);
    }
    close("Softmin forward", &output.to_vec()?, &expected);

    let weight_tensor = Tensor::from_slice(
        &weights.iter().map(|&w| w as f32).collect::<Vec<_>>(),
        [class.of(6)],
        &device,
    )?;
    output.mul(&weight_tensor)?.mean([row, class])?.backward()?;

    // Softmin(x) = softmax(-x), so d(softmin_i)/dx_j = p_i * (p_j - delta_ij)
    // (the sign-flipped softmax Jacobian). Weighted and reduced uniformly
    // over the 12 outputs: grad_j = p_j * (sum_i w_i*p_i - w_j) / n.
    let n = values.len() as f64;
    let mut expected_gradient = Vec::with_capacity(12);
    for probabilities in &prob_rows {
        let weighted_sum: f64 = probabilities
            .iter()
            .zip(&weights)
            .map(|(&p, &w)| p * w)
            .sum();
        for (j, &p_j) in probabilities.iter().enumerate() {
            expected_gradient.push(p_j * (weighted_sum - weights[j]) / n);
        }
    }
    close(
        "Softmin weighted derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn softmax2d_normalizes_channel_axis_and_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("softmax2d_channel"),
        Axis::new("softmax2d_height"),
        Axis::new("softmax2d_width"),
    );
    // Asymmetric extents: 3 channels x 2 height x 5 width = 30 elements.
    let values: Vec<f64> = (0..30).map(|i| ((i as f64) * 7.0 % 23.0) - 11.0).collect();
    let weights = [2.0_f64, 0.5, 1.5];
    let leaf = Tensor::from_slice(
        &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        [channel.of(3), height.of(2), width.of(5)],
        &device,
    )?
    .with_grad();
    let input = leaf.with_layout([width, height, channel])?;
    let module = Softmax2d::new(channel, height, width)?;
    let output = module.forward(&input)?;

    let index = |c: usize, h: usize, w: usize| (c * 2 + h) * 5 + w;
    let mut expected = vec![0.0_f64; 30];
    let mut probabilities = vec![0.0_f64; 30];
    for h in 0..2usize {
        for w in 0..5usize {
            let column = [
                values[index(0, h, w)],
                values[index(1, h, w)],
                values[index(2, h, w)],
            ];
            let max = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = column.iter().map(|&x| (x - max).exp()).sum();
            for (c, &value) in column.iter().enumerate() {
                let p = (value - max).exp() / sum_exp;
                expected[index(c, h, w)] = p;
                probabilities[index(c, h, w)] = p;
            }
        }
    }
    close("Softmax2d forward", &output.to_vec()?, &expected);

    let weight_tensor = Tensor::from_slice(
        &weights.iter().map(|&w| w as f32).collect::<Vec<_>>(),
        [channel.of(3)],
        &device,
    )?;
    output
        .mul(&weight_tensor)?
        .mean([channel, height, width])?
        .backward()?;

    // Weighted, uniformly-reduced softmax gradient: grad_j = p_j * (w_j - S)/n
    // where S = sum_i w_i*p_i over the 3-entry channel axis at that location.
    let n = values.len() as f64;
    let mut expected_gradient = vec![0.0_f64; 30];
    for h in 0..2usize {
        for w in 0..5usize {
            let weighted_sum: f64 = (0..3)
                .map(|i| weights[i] * probabilities[index(i, h, w)])
                .sum();
            for j in 0..3usize {
                let p_j = probabilities[index(j, h, w)];
                expected_gradient[index(j, h, w)] = p_j * (weights[j] - weighted_sum) / n;
            }
        }
    }
    close(
        "Softmax2d weighted derivative",
        &leaf.grad().unwrap().to_vec()?,
        &expected_gradient,
    );

    assert!(
        Softmax2d::new(channel, channel, width).is_err(),
        "Softmax2d must reject reused axis identities"
    );
    assert!(
        module
            .output_shape(&Shape::new([channel.of(3), height.of(2)])?)
            .is_err(),
        "Softmax2d must reject a non-3D input"
    );
    Ok(())
}

#[test]
fn piecewise_activation_modules_reject_invalid_configuration_before_launch() -> Result<()> {
    assert!(
        Hardtanh::new(1.0, -1.0).is_err(),
        "min_val > max_val must be rejected"
    );
    assert!(
        Hardtanh::new(f32::NAN, 1.0).is_err(),
        "a NaN bound must be rejected"
    );
    assert!(
        Hardtanh::new(0.0, f32::INFINITY).is_err(),
        "a non-finite bound must be rejected"
    );
    assert!(
        Hardshrink::new(-0.1).is_err(),
        "a negative lambd must be rejected"
    );
    assert!(
        Hardshrink::new(f32::NAN).is_err(),
        "a NaN lambd must be rejected"
    );
    assert!(
        Softshrink::new(-0.1).is_err(),
        "a negative lambd must be rejected"
    );
    assert!(
        Softshrink::new(f32::NAN).is_err(),
        "a NaN lambd must be rejected"
    );
    assert!(
        Threshold::new(f32::NAN, 0.0).is_err(),
        "a non-finite threshold must be rejected"
    );
    assert!(
        Threshold::new(0.0, f32::INFINITY).is_err(),
        "a non-finite value must be rejected"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn relu6_and_hardtanh_modules_match_independent_oracles_at_the_kink() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("piecewise_row"), Axis::new("piecewise_col"));

    // PyTorch's hardtanh_backward: zero AT as well as outside either bound
    // (`self <= min_val || self >= max_val`), unlike clamp's inclusive-boundary rule.
    let hardtanh_oracle = |x: f64, min_val: f64, max_val: f64| -> (f64, f64) {
        let value = x.clamp(min_val, max_val);
        let grad = if x > min_val && x < max_val { 1.0 } else { 0.0 };
        (value, grad)
    };

    let relu6_values: Vec<f32> = vec![-3.0, -0.001, 0.0, 0.5, 3.0, 5.999, 6.0, 6.001, 9.0];
    let n = relu6_values.len() as f64;
    let relu6_leaf =
        Tensor::from_slice(&relu6_values, [row.of(3), col.of(3)], &device)?.with_grad();
    let relu6_input = relu6_leaf.with_layout([col, row])?;
    let relu6_out = ReLU6.forward(&relu6_input)?;
    let (expected_relu6, expected_relu6_grad): (Vec<f64>, Vec<f64>) = relu6_values
        .iter()
        .map(|&x| {
            let (v, g) = hardtanh_oracle(f64::from(x), 0.0, 6.0);
            (v, g / n)
        })
        .unzip();
    close(
        "ReLU6 module forward, below/at-min/interior/at-max/above",
        &relu6_out.to_vec()?,
        &expected_relu6,
    );
    relu6_out.mean([row, col])?.backward()?;
    close(
        "ReLU6 module derivative, strictly interior only",
        &relu6_leaf.grad().unwrap().to_vec()?,
        &expected_relu6_grad,
    );

    let hardtanh_values: Vec<f32> = vec![-5.0, -2.0, -1.5, 0.0, 1.0, 2.999, 3.0, 3.2, 6.0];
    let hardtanh_leaf =
        Tensor::from_slice(&hardtanh_values, [row.of(3), col.of(3)], &device)?.with_grad();
    let hardtanh_input = hardtanh_leaf.with_layout([col, row])?;
    let hardtanh_out = Hardtanh::new(-2.0, 3.0)?.forward(&hardtanh_input)?;
    let (expected_hardtanh, expected_hardtanh_grad): (Vec<f64>, Vec<f64>) = hardtanh_values
        .iter()
        .map(|&x| {
            let (v, g) = hardtanh_oracle(f64::from(x), -2.0, 3.0);
            (v, g / n)
        })
        .unzip();
    close(
        "Hardtanh(-2, 3) module forward, below/at-min/interior/at-max/above",
        &hardtanh_out.to_vec()?,
        &expected_hardtanh,
    );
    hardtanh_out.mean([row, col])?.backward()?;
    close(
        "Hardtanh(-2, 3) module derivative, strictly interior only",
        &hardtanh_leaf.grad().unwrap().to_vec()?,
        &expected_hardtanh_grad,
    );

    assert!(hardtanh_input.hardtanh(1.0, -1.0).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn hardsigmoid_and_hardswish_modules_match_independent_oracles_at_the_kink() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("piecewise_row2"), Axis::new("piecewise_col2"));
    let values: Vec<f32> = vec![-5.0, -3.0, -1.5, 0.0, 1.0, 2.0, 3.0, 3.5, 6.0];
    let n = values.len() as f64;

    let hardsigmoid_leaf =
        Tensor::from_slice(&values, [row.of(3), col.of(3)], &device)?.with_grad();
    let hardsigmoid_input = hardsigmoid_leaf.with_layout([col, row])?;
    let hardsigmoid_out = Hardsigmoid.forward(&hardsigmoid_input)?;
    let expected_hardsigmoid: Vec<f64> = values
        .iter()
        .map(|&x| (f64::from(x) / 6.0 + 0.5).clamp(0.0, 1.0))
        .collect();
    close(
        "Hardsigmoid module forward",
        &hardsigmoid_out.to_vec()?,
        &expected_hardsigmoid,
    );
    hardsigmoid_out.mean([row, col])?.backward()?;
    // PyTorch's hardsigmoid_backward: grad/6 strictly inside (-3, 3), zero at and
    // outside either bound (`self > -3 && self < 3`, both strict).
    let expected_hardsigmoid_grad: Vec<f64> = values
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            if x > -3.0 && x < 3.0 {
                (1.0 / 6.0) / n
            } else {
                0.0
            }
        })
        .collect();
    close(
        "Hardsigmoid module derivative, strictly interior only",
        &hardsigmoid_leaf.grad().unwrap().to_vec()?,
        &expected_hardsigmoid_grad,
    );

    let hardswish_leaf = Tensor::from_slice(&values, [row.of(3), col.of(3)], &device)?.with_grad();
    let hardswish_input = hardswish_leaf.with_layout([col, row])?;
    let hardswish_out = Hardswish.forward(&hardswish_input)?;
    // PyTorch's Hardswish: x * clamp(x + 3, 0, 6) / 6.
    let expected_hardswish: Vec<f64> = values
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            x * (x + 3.0).clamp(0.0, 6.0) / 6.0
        })
        .collect();
    close(
        "Hardswish module forward",
        &hardswish_out.to_vec()?,
        &expected_hardswish,
    );
    hardswish_out.mean([row, col])?.backward()?;
    // PyTorch's hardswish_backward: zero for x <= -3, `x / 3 + 0.5` strictly inside
    // (-3, 3), and exactly `1` (pass-through) for x >= 3 -- an ASYMMETRIC kink: x == -3
    // routes to the zero branch, but x == 3 routes to the pass-through branch, not the
    // interior formula's limit there (1.5).
    let expected_hardswish_grad: Vec<f64> = values
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            let g = if x <= -3.0 {
                0.0
            } else if x < 3.0 {
                x / 3.0 + 0.5
            } else {
                1.0
            };
            g / n
        })
        .collect();
    close(
        "Hardswish module derivative, asymmetric kinks",
        &hardswish_leaf.grad().unwrap().to_vec()?,
        &expected_hardswish_grad,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn hardshrink_and_softshrink_modules_match_independent_oracle_on_the_band() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("piecewise_row3"), Axis::new("piecewise_col3"));
    let values: Vec<f32> = vec![-2.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0];
    let n = values.len() as f64;
    let lambd = 0.5f32;
    // PyTorch's shared shrink_backward_kernel, behind both Hardshrink and Softshrink:
    // zero on the CLOSED band [-lambd, lambd], grad outside it.
    let shrink_grad = |x: f64, lambd: f64, n: f64| -> f64 {
        if (-lambd..=lambd).contains(&x) {
            0.0
        } else {
            1.0 / n
        }
    };

    let hardshrink_leaf = Tensor::from_slice(&values, [row.of(2), col.of(4)], &device)?.with_grad();
    let hardshrink_input = hardshrink_leaf.with_layout([col, row])?;
    let hardshrink_out = Hardshrink::new(lambd)?.forward(&hardshrink_input)?;
    let expected_hardshrink: Vec<f64> = values
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            if x.abs() <= f64::from(lambd) { 0.0 } else { x }
        })
        .collect();
    close(
        "Hardshrink(0.5) module forward",
        &hardshrink_out.to_vec()?,
        &expected_hardshrink,
    );
    hardshrink_out.mean([row, col])?.backward()?;
    let expected_hardshrink_grad: Vec<f64> = values
        .iter()
        .map(|&x| shrink_grad(f64::from(x), f64::from(lambd), n))
        .collect();
    close(
        "Hardshrink(0.5) module derivative, zero on the closed band",
        &hardshrink_leaf.grad().unwrap().to_vec()?,
        &expected_hardshrink_grad,
    );
    assert!(hardshrink_input.hardshrink(-0.1).is_err());

    let softshrink_leaf = Tensor::from_slice(&values, [row.of(2), col.of(4)], &device)?.with_grad();
    let softshrink_input = softshrink_leaf.with_layout([col, row])?;
    let softshrink_out = Softshrink::new(lambd)?.forward(&softshrink_input)?;
    let expected_softshrink: Vec<f64> = values
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            let lambd = f64::from(lambd);
            if x > lambd {
                x - lambd
            } else if x < -lambd {
                x + lambd
            } else {
                0.0
            }
        })
        .collect();
    close(
        "Softshrink(0.5) module forward",
        &softshrink_out.to_vec()?,
        &expected_softshrink,
    );
    softshrink_out.mean([row, col])?.backward()?;
    let expected_softshrink_grad: Vec<f64> = values
        .iter()
        .map(|&x| shrink_grad(f64::from(x), f64::from(lambd), n))
        .collect();
    close(
        "Softshrink(0.5) module derivative, zero on the closed band",
        &softshrink_leaf.grad().unwrap().to_vec()?,
        &expected_softshrink_grad,
    );
    assert!(softshrink_input.softshrink(-0.1).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn threshold_module_matches_independent_oracle_at_the_kink() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("piecewise_row4"), Axis::new("piecewise_col4"));
    let values: Vec<f32> = vec![-2.0, 0.0, 0.999, 1.0, 1.001, 3.0];
    let n = values.len() as f64;
    let (threshold_value, replacement) = (1.0f32, -5.0f32);

    let leaf = Tensor::from_slice(&values, [row.of(2), col.of(3)], &device)?.with_grad();
    let input = leaf.with_layout([col, row])?;
    let out = Threshold::new(threshold_value, replacement)?.forward(&input)?;
    // PyTorch's Threshold: x where x > threshold, else the constant `value` -- the SAME
    // `<=`/`>` split governs threshold_backward (grad where x > threshold, 0 at and
    // below it, exactly matching the forward's own boundary).
    let expected: Vec<f64> = values
        .iter()
        .map(|&x| {
            if f64::from(x) > f64::from(threshold_value) {
                f64::from(x)
            } else {
                f64::from(replacement)
            }
        })
        .collect();
    close("Threshold(1, -5) module forward", &out.to_vec()?, &expected);
    out.mean([row, col])?.backward()?;
    let expected_grad: Vec<f64> = values
        .iter()
        .map(|&x| {
            if f64::from(x) > f64::from(threshold_value) {
                1.0 / n
            } else {
                0.0
            }
        })
        .collect();
    close(
        "Threshold(1, -5) module derivative, zero at and below the threshold",
        &leaf.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(input.threshold(f32::NAN, 0.0).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn softsign_and_tanhshrink_modules_match_independent_oracles() -> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("piecewise_row5"), Axis::new("piecewise_col5"));
    let values: Vec<f32> = vec![-4.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 4.0];
    let n = values.len() as f64;

    let softsign_leaf = Tensor::from_slice(&values, [row.of(2), col.of(4)], &device)?.with_grad();
    let softsign_input = softsign_leaf.with_layout([col, row])?;
    let softsign_out = Softsign.forward(&softsign_input)?;
    let expected_softsign: Vec<f64> = values
        .iter()
        .map(|&x| f64::from(x) / (1.0 + f64::from(x).abs()))
        .collect();
    close(
        "Softsign module forward",
        &softsign_out.to_vec()?,
        &expected_softsign,
    );
    softsign_out.mean([row, col])?.backward()?;
    let expected_softsign_grad: Vec<f64> = values
        .iter()
        .map(|&x| (1.0 / (1.0 + f64::from(x).abs()).powi(2)) / n)
        .collect();
    close(
        "Softsign module derivative, smooth through x == 0",
        &softsign_leaf.grad().unwrap().to_vec()?,
        &expected_softsign_grad,
    );

    let tanhshrink_leaf = Tensor::from_slice(&values, [row.of(2), col.of(4)], &device)?.with_grad();
    let tanhshrink_input = tanhshrink_leaf.with_layout([col, row])?;
    let tanhshrink_out = Tanhshrink.forward(&tanhshrink_input)?;
    let expected_tanhshrink: Vec<f64> = values
        .iter()
        .map(|&x| f64::from(x) - f64::from(x).tanh())
        .collect();
    close(
        "Tanhshrink module forward",
        &tanhshrink_out.to_vec()?,
        &expected_tanhshrink,
    );
    tanhshrink_out.mean([row, col])?.backward()?;
    let expected_tanhshrink_grad: Vec<f64> = values
        .iter()
        .map(|&x| f64::from(x).tanh().powi(2) / n)
        .collect();
    close(
        "Tanhshrink module derivative",
        &tanhshrink_leaf.grad().unwrap().to_vec()?,
        &expected_tanhshrink_grad,
    );
    Ok(())
}

fn to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

#[test]
#[ignore = "requires CUDA"]
fn cosine_similarity_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("cos_sim_batch"), Axis::new("cos_sim_feature"));
    let x1_values = [1.0_f64, 2.0, 2.0, 3.0, 4.0, 0.0];
    let x2_values = [2.0_f64, 0.0, 0.0, 0.0, 3.0, 4.0];
    let x1 =
        Tensor::from_slice(&to_f32(&x1_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let x2 =
        Tensor::from_slice(&to_f32(&x2_values), [batch.of(2), feature.of(3)], &device)?.with_grad();

    let similarity = x1.cosine_similarity(&x2, feature, 1e-8)?;
    // x1=[1,2,2], x2=[2,0,0]: dot=2, |x1|=3, |x2|=2 -> 2/6 = 1/3.
    // x1=[3,4,0], x2=[0,3,4]: dot=12, |x1|=5, |x2|=5 -> 12/25 = 0.48.
    close(
        "cosine similarity forward",
        &similarity.to_vec()?,
        &[1.0 / 3.0, 0.48],
    );

    fn cos_scalar_loss(a: &[f64], b: &[f64]) -> f64 {
        let row = |a: &[f64], b: &[f64]| {
            let dot: f64 = a.iter().zip(b).map(|(p, q)| p * q).sum();
            let na = a.iter().map(|p| p * p).sum::<f64>().sqrt().max(1e-8);
            let nb = b.iter().map(|q| q * q).sum::<f64>().sqrt().max(1e-8);
            dot / (na * nb)
        };
        (row(&a[0..3], &b[0..3]) + row(&a[3..6], &b[3..6])) / 2.0
    }

    similarity.mean(batch)?.backward()?;
    close(
        "cosine similarity gradient wrt x1",
        &x1.grad().unwrap().to_vec()?,
        &central_difference(&x1_values, 1e-4, |candidate| {
            cos_scalar_loss(candidate, &x2_values)
        }),
    );
    close(
        "cosine similarity gradient wrt x2",
        &x2.grad().unwrap().to_vec()?,
        &central_difference(&x2_values, 1e-4, |candidate| {
            cos_scalar_loss(&x1_values, candidate)
        }),
    );

    let other = Axis::new("cos_sim_other");
    let wrong = Tensor::from_slice(&[1.0_f32, 2.0], [other.of(2)], &device)?;
    assert!(
        x1.detach()
            .cosine_similarity(&wrong, feature, 1e-8)
            .is_err()
    );
    assert!(
        x1.detach()
            .cosine_similarity(&x2.detach(), feature, 0.0)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn pairwise_distance_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("pdist_batch"), Axis::new("pdist_feature"));
    let x1_values = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let x2_values = [3.0_f64, 4.0, 0.0, 0.0, 0.0, 0.0];
    let eps = 1e-6_f64;
    let x1 =
        Tensor::from_slice(&to_f32(&x1_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let x2 =
        Tensor::from_slice(&to_f32(&x2_values), [batch.of(2), feature.of(3)], &device)?.with_grad();

    let distance = x1.pairwise_distance(&x2, feature, 2.0, eps as f32)?;
    close(
        "pairwise distance forward",
        &distance.to_vec()?,
        &[4.9999986000001035, 1.7320525396196849],
    );

    fn pdist_row(a: &[f64], b: &[f64], eps: f64) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q + eps).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    fn pdist_scalar_loss(a: &[f64], b: &[f64], eps: f64) -> f64 {
        (pdist_row(&a[0..3], &b[0..3], eps) + pdist_row(&a[3..6], &b[3..6], eps)) / 2.0
    }

    distance.mean(batch)?.backward()?;
    close(
        "pairwise distance gradient wrt x1",
        &x1.grad().unwrap().to_vec()?,
        &central_difference(&x1_values, 1e-4, |candidate| {
            pdist_scalar_loss(candidate, &x2_values, eps)
        }),
    );
    close(
        "pairwise distance gradient wrt x2",
        &x2.grad().unwrap().to_vec()?,
        &central_difference(&x2_values, 1e-4, |candidate| {
            pdist_scalar_loss(&x1_values, candidate, eps)
        }),
    );

    let other = Axis::new("pdist_other");
    let wrong = Tensor::from_slice(&[1.0_f32, 2.0], [other.of(2)], &device)?;
    assert!(
        x1.detach()
            .pairwise_distance(&wrong, feature, 2.0, eps as f32)
            .is_err()
    );
    assert!(
        x1.detach()
            .pairwise_distance(&x2.detach(), feature, 2.0, f32::NAN)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn pairwise_distance_broadcasts_over_reordered_storage_and_asymmetric_extents() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (
        Axis::new("pdist_layout_batch"),
        Axis::new("pdist_layout_feature"),
    );
    // 3 batch rows, 4 features: asymmetric, non-square extents.
    let x1_values = [
        0.0_f64, 1.0, 2.0, 3.0, // batch0
        1.0, 0.0, -1.0, 2.0, // batch1
        -2.0, 2.0, 0.0, 1.0, // batch2
    ];
    let x2_values = [
        3.0_f64, 1.0, 2.0, 0.0, // batch0
        1.0, 3.0, -1.0, -1.0, // batch1
        0.0, 0.0, 0.0, 0.0, // batch2
    ];
    let eps = 1e-6_f64;
    // Logical [batch, feature] order stays what `to_vec()`'s coordinate walk uses; physical
    // storage is requested feature-major, forcing the op to read a genuinely permuted,
    // non-contiguous buffer rather than one that already matches its own shape order.
    let x1 = Tensor::from_slice(&to_f32(&x1_values), [batch.of(3), feature.of(4)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let x2 = Tensor::from_slice(&to_f32(&x2_values), [batch.of(3), feature.of(4)], &device)?
        .with_layout([feature, batch])?;

    let distance = x1.pairwise_distance(&x2, feature, 2.0, eps as f32)?;
    fn pdist_row(a: &[f64], b: &[f64], eps: f64) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q + eps).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    let expected: Vec<f64> = (0..3)
        .map(|row| {
            pdist_row(
                &x1_values[row * 4..row * 4 + 4],
                &x2_values[row * 4..row * 4 + 4],
                eps,
            )
        })
        .collect();
    close(
        "pairwise distance forward (reordered storage, asymmetric extents)",
        &distance.to_vec()?,
        &expected,
    );

    distance.mean(batch)?.backward()?;
    fn pdist_scalar_loss(a: &[f64], b: &[f64], eps: f64) -> f64 {
        (0..3)
            .map(|row| pdist_row(&a[row * 4..row * 4 + 4], &b[row * 4..row * 4 + 4], eps))
            .sum::<f64>()
            / 3.0
    }
    close(
        "pairwise distance gradient (reordered storage, asymmetric extents)",
        &x1.grad().unwrap().to_vec()?,
        &central_difference(&x1_values, 1e-4, |candidate| {
            pdist_scalar_loss(candidate, &x2_values, eps)
        }),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn margin_ranking_loss_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("mrl_batch"), Axis::new("mrl_feature"));
    let x1_values = [2.0_f64, 0.0, 1.2, -1.0, 3.0, 0.0];
    let x2_values = [0.0_f64, 1.3, 1.0, 2.0, 1.0, 0.3];
    let target_values = [1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0];
    let margin = 1.0_f32;
    let x1 =
        Tensor::from_slice(&to_f32(&x1_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let x2 =
        Tensor::from_slice(&to_f32(&x2_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?;

    let loss = x1.margin_ranking_loss(&x2, &target, margin)?;
    close(
        "margin ranking loss forward",
        &loss.to_vec()?,
        &[0.0, 0.0, 0.8, 0.0, 0.0, 0.7],
    );

    fn mrl_scalar_loss(a: &[f64], b: &[f64], t: &[f32], margin: f64) -> f64 {
        a.iter()
            .zip(b)
            .zip(t)
            .map(|((p, q), &y)| (-f64::from(y) * (p - q) + margin).max(0.0))
            .sum::<f64>()
            / a.len() as f64
    }

    loss.mean([batch, feature])?.backward()?;
    close(
        "margin ranking loss gradient wrt x1",
        &x1.grad().unwrap().to_vec()?,
        &central_difference(&x1_values, 1e-4, |candidate| {
            mrl_scalar_loss(candidate, &x2_values, &target_values, f64::from(margin))
        }),
    );
    close(
        "margin ranking loss gradient wrt x2",
        &x2.grad().unwrap().to_vec()?,
        &central_difference(&x2_values, 1e-4, |candidate| {
            mrl_scalar_loss(&x1_values, candidate, &target_values, f64::from(margin))
        }),
    );

    assert!(
        x1.detach()
            .margin_ranking_loss(&x2.detach(), &target.with_grad(), margin)
            .is_err()
    );
    let bad_target = Tensor::from_slice(&[0.5_f32; 6], [batch.of(2), feature.of(3)], &device)?;
    assert!(
        x1.detach()
            .margin_ranking_loss(&x2.detach(), &bad_target, margin)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn hinge_embedding_loss_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("hinge_batch"), Axis::new("hinge_feature"));
    let x_values = [0.5_f64, -2.0, 3.0, -0.5, 2.0, -3.0];
    let target_values = [1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0];
    let margin = 1.0_f32;
    let x =
        Tensor::from_slice(&to_f32(&x_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?;

    let loss = x.hinge_embedding_loss(&target, margin)?;
    close(
        "hinge embedding loss forward",
        &loss.to_vec()?,
        &[0.5, 3.0, 3.0, 1.5, 2.0, 4.0],
    );

    fn hinge_scalar_loss(x: &[f64], t: &[f32], margin: f64) -> f64 {
        x.iter()
            .zip(t)
            .map(|(&v, &y)| if y > 0.0 { v } else { (margin - v).max(0.0) })
            .sum::<f64>()
            / x.len() as f64
    }

    loss.mean([batch, feature])?.backward()?;
    close(
        "hinge embedding loss gradient",
        &x.grad().unwrap().to_vec()?,
        &central_difference(&x_values, 1e-4, |candidate| {
            hinge_scalar_loss(candidate, &target_values, f64::from(margin))
        }),
    );

    assert!(
        x.detach()
            .hinge_embedding_loss(&target.with_grad(), margin)
            .is_err()
    );
    let bad_target = Tensor::from_slice(&[0.5_f32; 6], [batch.of(2), feature.of(3)], &device)?;
    assert!(
        x.detach()
            .hinge_embedding_loss(&bad_target, margin)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn cosine_embedding_loss_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("cel_batch"), Axis::new("cel_feature"));
    let x1_values = [1.0_f64, 2.0, 2.0, 3.0, 4.0, 0.0];
    let x2_values = [2.0_f64, 0.0, 0.0, 0.0, 3.0, 4.0];
    let target_values = [1.0_f32, -1.0];
    let margin = 0.5_f32;
    let x1 =
        Tensor::from_slice(&to_f32(&x1_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let x2 =
        Tensor::from_slice(&to_f32(&x2_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2)], &device)?;

    let loss = x1.cosine_embedding_loss(&x2, &target, feature, margin)?;
    close(
        "cosine embedding loss forward",
        &loss.to_vec()?,
        &[0.6666666666666667, 0.0],
    );

    fn cel_scalar_loss(a: &[f64], b: &[f64], t: &[f32], margin: f64) -> f64 {
        let row = |a: &[f64], b: &[f64]| {
            let dot: f64 = a.iter().zip(b).map(|(p, q)| p * q).sum();
            let na = a.iter().map(|p| p * p).sum::<f64>().sqrt().max(1e-8);
            let nb = b.iter().map(|q| q * q).sum::<f64>().sqrt().max(1e-8);
            dot / (na * nb)
        };
        let c0 = row(&a[0..3], &b[0..3]);
        let c1 = row(&a[3..6], &b[3..6]);
        let per = |c: f64, y: f32| {
            if y > 0.0 {
                1.0 - c
            } else {
                (c - margin).max(0.0)
            }
        };
        (per(c0, t[0]) + per(c1, t[1])) / 2.0
    }

    loss.mean(batch)?.backward()?;
    close(
        "cosine embedding loss gradient wrt x1",
        &x1.grad().unwrap().to_vec()?,
        &central_difference(&x1_values, 1e-4, |candidate| {
            cel_scalar_loss(candidate, &x2_values, &target_values, f64::from(margin))
        }),
    );
    close(
        "cosine embedding loss gradient wrt x2",
        &x2.grad().unwrap().to_vec()?,
        &central_difference(&x2_values, 1e-4, |candidate| {
            cel_scalar_loss(&x1_values, candidate, &target_values, f64::from(margin))
        }),
    );

    assert!(
        x1.detach()
            .cosine_embedding_loss(&x2.detach(), &target.with_grad(), feature, margin)
            .is_err()
    );
    let contains_feature =
        Tensor::from_slice(&[0.0_f32; 6], [batch.of(2), feature.of(3)], &device)?;
    assert!(
        x1.detach()
            .cosine_embedding_loss(&x2.detach(), &contains_feature, feature, margin)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn triplet_margin_loss_matches_hand_computed_oracle_with_and_without_swap() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("triplet_batch"), Axis::new("triplet_feature"));
    let anchor_values = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let positive_values = [0.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0];
    let negative_values = [0.0_f64, 0.0, 2.5, -1.0, -1.0, -1.0];
    let margin = 1.0_f32;
    let eps = 1e-6_f64;

    fn pdist_row(a: &[f64], b: &[f64], eps: f64) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q + eps).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    fn triplet_scalar_loss(
        anchor: &[f64],
        positive: &[f64],
        negative: &[f64],
        margin: f64,
        eps: f64,
        swap: bool,
    ) -> f64 {
        (0..2)
            .map(|row| {
                let a = &anchor[row * 3..row * 3 + 3];
                let p = &positive[row * 3..row * 3 + 3];
                let n = &negative[row * 3..row * 3 + 3];
                let d_pos = pdist_row(a, p, eps);
                let mut d_neg = pdist_row(a, n, eps);
                if swap {
                    d_neg = d_neg.min(pdist_row(p, n, eps));
                }
                (margin + d_pos - d_neg).max(0.0)
            })
            .sum::<f64>()
            / 2.0
    }

    // swap = false.
    {
        let anchor = Tensor::from_slice(
            &to_f32(&anchor_values),
            [batch.of(2), feature.of(3)],
            &device,
        )?
        .with_grad();
        let positive = Tensor::from_slice(
            &to_f32(&positive_values),
            [batch.of(2), feature.of(3)],
            &device,
        )?
        .with_grad();
        let negative = Tensor::from_slice(
            &to_f32(&negative_values),
            [batch.of(2), feature.of(3)],
            &device,
        )?
        .with_grad();
        let loss = anchor.triplet_margin_loss(
            &positive, &negative, feature, margin, 2.0, eps as f32, false,
        )?;
        close(
            "triplet margin loss forward (no swap)",
            &loss.to_vec()?,
            &[0.0, 0.0],
        );
        loss.mean(batch)?.backward()?;
        close(
            "triplet margin loss gradient wrt anchor (no swap)",
            &anchor.grad().unwrap().to_vec()?,
            &central_difference(&anchor_values, 1e-4, |candidate| {
                triplet_scalar_loss(
                    candidate,
                    &positive_values,
                    &negative_values,
                    f64::from(margin),
                    eps,
                    false,
                )
            }),
        );
        close(
            "triplet margin loss gradient wrt positive (no swap)",
            &positive.grad().unwrap().to_vec()?,
            &central_difference(&positive_values, 1e-4, |candidate| {
                triplet_scalar_loss(
                    &anchor_values,
                    candidate,
                    &negative_values,
                    f64::from(margin),
                    eps,
                    false,
                )
            }),
        );
        close(
            "triplet margin loss gradient wrt negative (no swap)",
            &negative.grad().unwrap().to_vec()?,
            &central_difference(&negative_values, 1e-4, |candidate| {
                triplet_scalar_loss(
                    &anchor_values,
                    &positive_values,
                    candidate,
                    f64::from(margin),
                    eps,
                    false,
                )
            }),
        );
    }

    // swap = true: sample0's positive sits closer to its negative than the anchor does, so the
    // swap term becomes the binding one and the loss goes from clamped-zero to active.
    {
        let anchor = Tensor::from_slice(
            &to_f32(&anchor_values),
            [batch.of(2), feature.of(3)],
            &device,
        )?
        .with_grad();
        let positive = Tensor::from_slice(
            &to_f32(&positive_values),
            [batch.of(2), feature.of(3)],
            &device,
        )?
        .with_grad();
        let negative = Tensor::from_slice(
            &to_f32(&negative_values),
            [batch.of(2), feature.of(3)],
            &device,
        )?
        .with_grad();
        let loss = anchor
            .triplet_margin_loss(&positive, &negative, feature, margin, 2.0, eps as f32, true)?;
        close(
            "triplet margin loss forward (swap)",
            &loss.to_vec()?,
            &[0.5000000000003333, 0.0],
        );
        loss.mean(batch)?.backward()?;
        close(
            "triplet margin loss gradient wrt anchor (swap)",
            &anchor.grad().unwrap().to_vec()?,
            &central_difference(&anchor_values, 1e-4, |candidate| {
                triplet_scalar_loss(
                    candidate,
                    &positive_values,
                    &negative_values,
                    f64::from(margin),
                    eps,
                    true,
                )
            }),
        );
        close(
            "triplet margin loss gradient wrt positive (swap)",
            &positive.grad().unwrap().to_vec()?,
            &central_difference(&positive_values, 1e-4, |candidate| {
                triplet_scalar_loss(
                    &anchor_values,
                    candidate,
                    &negative_values,
                    f64::from(margin),
                    eps,
                    true,
                )
            }),
        );
        close(
            "triplet margin loss gradient wrt negative (swap)",
            &negative.grad().unwrap().to_vec()?,
            &central_difference(&negative_values, 1e-4, |candidate| {
                triplet_scalar_loss(
                    &anchor_values,
                    &positive_values,
                    candidate,
                    f64::from(margin),
                    eps,
                    true,
                )
            }),
        );
    }

    let a_err = Tensor::from_slice(
        &to_f32(&anchor_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?;
    let p_err = Tensor::from_slice(
        &to_f32(&positive_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?;
    let n_err = Tensor::from_slice(&[1.0_f32, 2.0], [Axis::new("triplet_bad").of(2)], &device)?;
    assert!(
        a_err
            .triplet_margin_loss(&p_err, &n_err, feature, margin, 2.0, eps as f32, false)
            .is_err()
    );
    assert!(
        a_err
            .triplet_margin_loss(&p_err, &n_err, feature, f32::NAN, 2.0, eps as f32, false)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn triplet_margin_with_distance_loss_uses_the_supplied_distance_closure() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("tmwd_batch"), Axis::new("tmwd_feature"));
    let anchor_values = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let positive_values = [0.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0];
    let negative_values = [0.0_f64, 0.0, 2.5, -1.0, -1.0, -1.0];
    let margin = 4.0_f32;
    let anchor = Tensor::from_slice(
        &to_f32(&anchor_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();
    let positive = Tensor::from_slice(
        &to_f32(&positive_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();
    let negative = Tensor::from_slice(
        &to_f32(&negative_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();

    // Manhattan (L1) distance closure, in place of the Euclidean `pairwise_distance` default,
    // proving the caller's own distance function drives the computation.
    let l1_distance =
        |a: &Tensor, b: &Tensor| -> Result<Tensor> { a.absolute_error(b)?.sum(feature) };

    let loss = anchor.triplet_margin_with_distance_loss(
        &positive,
        &negative,
        margin,
        false,
        l1_distance,
    )?;
    close(
        "triplet margin with distance loss forward (L1)",
        &loss.to_vec()?,
        &[2.5, 0.0],
    );

    fn l1_row(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum()
    }
    fn triplet_l1_scalar_loss(
        anchor: &[f64],
        positive: &[f64],
        negative: &[f64],
        margin: f64,
    ) -> f64 {
        (0..2)
            .map(|row| {
                let a = &anchor[row * 3..row * 3 + 3];
                let p = &positive[row * 3..row * 3 + 3];
                let n = &negative[row * 3..row * 3 + 3];
                (margin + l1_row(a, p) - l1_row(a, n)).max(0.0)
            })
            .sum::<f64>()
            / 2.0
    }

    loss.mean(batch)?.backward()?;
    close(
        "triplet margin with distance loss gradient wrt anchor (L1)",
        &anchor.grad().unwrap().to_vec()?,
        &central_difference(&anchor_values, 1e-4, |candidate| {
            triplet_l1_scalar_loss(
                candidate,
                &positive_values,
                &negative_values,
                f64::from(margin),
            )
        }),
    );
    close(
        "triplet margin with distance loss gradient wrt positive (L1)",
        &positive.grad().unwrap().to_vec()?,
        &central_difference(&positive_values, 1e-4, |candidate| {
            triplet_l1_scalar_loss(
                &anchor_values,
                candidate,
                &negative_values,
                f64::from(margin),
            )
        }),
    );
    close(
        "triplet margin with distance loss gradient wrt negative (L1)",
        &negative.grad().unwrap().to_vec()?,
        &central_difference(&negative_values, 1e-4, |candidate| {
            triplet_l1_scalar_loss(
                &anchor_values,
                &positive_values,
                candidate,
                f64::from(margin),
            )
        }),
    );

    // The default distance -- a closure composed from `pairwise_distance` -- reproduces
    // `triplet_margin_loss` exactly, witnessing the "PairwiseDistance is the default" contract.
    let eps = 1e-6_f32;
    let anchor2 = Tensor::from_slice(
        &to_f32(&anchor_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?;
    let positive2 = Tensor::from_slice(
        &to_f32(&positive_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?;
    let negative2 = Tensor::from_slice(
        &to_f32(&negative_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?;
    let via_closure = anchor2.triplet_margin_with_distance_loss(
        &positive2,
        &negative2,
        1.0,
        false,
        |a: &Tensor, b: &Tensor| a.pairwise_distance(b, feature, 2.0, eps),
    )?;
    let via_direct =
        anchor2.triplet_margin_loss(&positive2, &negative2, feature, 1.0, 2.0, eps, false)?;
    close(
        "triplet margin with distance loss matches triplet_margin_loss under the default closure",
        &via_closure.to_vec()?,
        &via_direct
            .to_vec()?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );

    assert!(
        anchor2
            .triplet_margin_with_distance_loss(
                &positive2,
                &negative2,
                f32::NAN,
                false,
                |a: &Tensor, b: &Tensor| { a.pairwise_distance(b, feature, 2.0, eps) }
            )
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multi_margin_loss_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("mml_batch"), Axis::new("mml_class"));
    let x_values = [2.0_f64, 1.5, -1.0, 0.0, 2.5, 3.0];
    let target_values = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0];
    let margin = 1.0_f32;
    let x =
        Tensor::from_slice(&to_f32(&x_values), [batch.of(2), class.of(3)], &device)?.with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), class.of(3)], &device)?;

    let loss = x.multi_margin_loss(&target, class, 1.0, margin, None)?;
    close(
        "multi margin loss forward",
        &loss.to_vec()?,
        &[1.0 / 6.0, 1.0 / 6.0],
    );

    fn multi_margin_scalar_loss(x: &[f64], t: &[f32], margin: f64) -> f64 {
        (0..2)
            .map(|row| {
                let xs = &x[row * 3..row * 3 + 3];
                let ts = &t[row * 3..row * 3 + 3];
                let y = ts.iter().position(|&v| v == 1.0).unwrap();
                let xy = xs[y];
                let total: f64 = (0..3)
                    .filter(|&i| i != y)
                    .map(|i| (margin - xy + xs[i]).max(0.0))
                    .sum();
                total / 3.0
            })
            .sum::<f64>()
            / 2.0
    }

    loss.mean(batch)?.backward()?;
    close(
        "multi margin loss gradient",
        &x.grad().unwrap().to_vec()?,
        &central_difference(&x_values, 1e-4, |candidate| {
            multi_margin_scalar_loss(candidate, &target_values, f64::from(margin))
        }),
    );

    assert!(
        x.detach()
            .multi_margin_loss(&target.with_grad(), class, 1.0, margin, None)
            .is_err()
    );
    let bad_target = Tensor::from_slice(&[0.5_f32; 6], [batch.of(2), class.of(3)], &device)?;
    assert!(
        x.detach()
            .multi_margin_loss(&bad_target, class, 1.0, margin, None)
            .is_err()
    );
    let solo_class = Axis::new("mml_solo_class");
    let solo_x = Tensor::from_slice(&[1.0_f32, 2.0], [batch.of(2), solo_class.of(1)], &device)?;
    let solo_t = Tensor::from_slice(&[1.0_f32, 1.0], [batch.of(2), solo_class.of(1)], &device)?;
    assert!(
        solo_x
            .multi_margin_loss(&solo_t, solo_class, 1.0, margin, None)
            .is_err()
    );
    Ok(())
}

fn mlml_scalar_loss(x: &[f64], t: &[f32]) -> f64 {
    (0..2)
        .map(|row| {
            let xs = &x[row * 3..row * 3 + 3];
            let ts = &t[row * 3..row * 3 + 3];
            let mut total = 0.0;
            for i in 0..3 {
                if ts[i] == 1.0 {
                    continue;
                }
                for j in 0..3 {
                    if ts[j] == 1.0 {
                        total += (1.0 - (xs[j] - xs[i])).max(0.0);
                    }
                }
            }
            total / 3.0
        })
        .sum::<f64>()
        / 2.0
}

#[test]
#[ignore = "requires CUDA"]
fn multi_label_margin_loss_matches_hand_computed_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("mlml_batch"), Axis::new("mlml_class"));
    let x_values = [1.0_f64, 0.5, -1.0, 0.7, 2.0, 1.0];
    let target_values = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0];
    let x =
        Tensor::from_slice(&to_f32(&x_values), [batch.of(2), class.of(3)], &device)?.with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), class.of(3)], &device)?;

    let loss = x.multi_label_margin_loss(&target, class)?;
    close(
        "multi label margin loss forward",
        &loss.to_vec()?,
        &[1.0 / 6.0, 0.7 / 3.0],
    );

    loss.mean(batch)?.backward()?;
    close(
        "multi label margin loss gradient",
        &x.grad().unwrap().to_vec()?,
        &central_difference(&x_values, 1e-4, |candidate| {
            mlml_scalar_loss(candidate, &target_values)
        }),
    );

    assert!(
        x.detach()
            .multi_label_margin_loss(&target.with_grad(), class)
            .is_err()
    );
    let bad_target = Tensor::from_slice(&[0.3_f32; 6], [batch.of(2), class.of(3)], &device)?;
    assert!(
        x.detach()
            .multi_label_margin_loss(&bad_target, class)
            .is_err()
    );
    let solo_class = Axis::new("mlml_solo_class");
    let solo_x = Tensor::from_slice(&[1.0_f32, 2.0], [batch.of(2), solo_class.of(1)], &device)?;
    let solo_t = Tensor::from_slice(&[1.0_f32, 1.0], [batch.of(2), solo_class.of(1)], &device)?;
    assert!(solo_x.multi_label_margin_loss(&solo_t, solo_class).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multi_label_margin_loss_handles_reordered_storage() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, class) = (
        Axis::new("mlml_layout_batch"),
        Axis::new("mlml_layout_class"),
    );
    let x_values = [1.0_f64, 0.5, -1.0, 0.7, 2.0, 1.0];
    let target_values = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0];
    // Logical [batch, class] order stays what `to_vec()`'s coordinate walk uses; physical
    // storage is requested class-major, so the op reads a permuted, non-contiguous buffer.
    let x = Tensor::from_slice(&to_f32(&x_values), [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?
        .with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?;

    let loss = x.multi_label_margin_loss(&target, class)?;
    close(
        "multi label margin loss forward (reordered storage)",
        &loss.to_vec()?,
        &[1.0 / 6.0, 0.7 / 3.0],
    );

    loss.mean(batch)?.backward()?;
    close(
        "multi label margin loss gradient (reordered storage)",
        &x.grad().unwrap().to_vec()?,
        &central_difference(&x_values, 1e-4, |candidate| {
            mlml_scalar_loss(candidate, &target_values)
        }),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn l1_and_mse_losses_match_hand_computed_oracles_under_reordered_asymmetric_storage() -> Result<()>
{
    // L1Loss (`absolute_error`) and MSELoss (`squared_error`) already existed; this closes the
    // remaining CUDA-coverage gap those catalog rows were waiting on: a reordered,
    // asymmetric-extent case for both, exercised together the way `abs`'s own reordered test
    // does (`abs_matches_hand_computed_oracle_under_reordered_asymmetric_cuda_storage`).
    // `f64::signum` returns `1.0` at exactly `0.0` (not `0.0`), unlike `abs`'s documented
    // zero-at-`x == 0` backward convention that `absolute_error` inherits, so the oracle below
    // uses this helper instead of `.signum()`.
    fn sign_or_zero(value: f64) -> f64 {
        if value == 0.0 { 0.0 } else { value.signum() }
    }
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    // value(b, f) written in canonical (batch, feature) order; asymmetric extents (3, 2).
    let pred_values = [1.0f32, -2.0, 0.5, 3.0, -1.5, 0.0];
    let target_values = [0.0f32, -1.0, 0.5, 1.0, -3.0, 0.0];
    let expected_l1: Vec<f64> = pred_values
        .iter()
        .zip(target_values)
        .map(|(&p, t)| (f64::from(p) - f64::from(t)).abs())
        .collect();
    let expected_mse: Vec<f64> = pred_values
        .iter()
        .zip(target_values)
        .map(|(&p, t)| (f64::from(p) - f64::from(t)).powi(2))
        .collect();
    let n = pred_values.len() as f64;
    let expected_l1_grad_pred: Vec<f64> = pred_values
        .iter()
        .zip(target_values)
        .map(|(&p, t)| sign_or_zero(f64::from(p) - f64::from(t)) / n)
        .collect();
    let expected_mse_grad_pred: Vec<f64> = pred_values
        .iter()
        .zip(target_values)
        .map(|(&p, t)| 2.0 * (f64::from(p) - f64::from(t)) / n)
        .collect();

    let pred = Tensor::from_slice(&pred_values, [batch.of(3), feature.of(2)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(3), feature.of(2)], &device)?
        .with_layout([feature, batch])?
        .with_grad();

    let l1 = pred.absolute_error(&target)?;
    close(
        "L1Loss forward under reordered, asymmetric storage",
        &l1.to_vec()?,
        &expected_l1,
    );
    l1.mean([batch, feature])?.backward()?;
    close(
        "L1Loss gradient wrt prediction under reordered storage",
        &pred.grad().unwrap().to_vec()?,
        &expected_l1_grad_pred,
    );
    close(
        "L1Loss gradient wrt target under reordered storage",
        &target.grad().unwrap().to_vec()?,
        &expected_l1_grad_pred
            .iter()
            .map(|&g| -g)
            .collect::<Vec<_>>(),
    );

    let pred2 = Tensor::from_slice(&pred_values, [batch.of(3), feature.of(2)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let target2 = Tensor::from_slice(&target_values, [batch.of(3), feature.of(2)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let mse = pred2.squared_error(&target2)?;
    close(
        "MSELoss forward under reordered, asymmetric storage",
        &mse.to_vec()?,
        &expected_mse,
    );
    mse.mean([batch, feature])?.backward()?;
    close(
        "MSELoss gradient wrt prediction under reordered storage",
        &pred2.grad().unwrap().to_vec()?,
        &expected_mse_grad_pred,
    );
    close(
        "MSELoss gradient wrt target under reordered storage",
        &target2.grad().unwrap().to_vec()?,
        &expected_mse_grad_pred
            .iter()
            .map(|&g| -g)
            .collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn huber_and_smooth_l1_loss_match_hand_computed_oracle_and_pytorch_consumer_default() -> Result<()>
{
    // Hand oracle: HuberLoss(delta) = 0.5*min(|d|,delta)^2 + delta*relu(|d|-delta), an exact
    // algebraic identity with PyTorch's two-branch definition; its derivative wrt d is
    // clamp(d, -delta, delta). SmoothL1Loss(beta) == HuberLoss(delta=beta) / beta, another
    // exact identity, checked directly below as well as against its own closed form.
    fn huber(diff: f64, delta: f64) -> f64 {
        if diff.abs() < delta {
            0.5 * diff * diff
        } else {
            delta * (diff.abs() - 0.5 * delta)
        }
    }
    fn huber_grad(diff: f64, delta: f64) -> f64 {
        diff.clamp(-delta, delta)
    }

    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    // value(b, f) written in canonical (batch, feature) order; asymmetric extents (2, 3).
    // target is zero everywhere, so diff == pred.
    let pred_values = [0.0f32, 2.0, -3.0, 0.3, 5.0, -0.5];
    let target_values = [0.0f32; 6];
    let n = pred_values.len() as f64;
    let expected_huber: Vec<f64> = pred_values
        .iter()
        .map(|&p| huber(f64::from(p), 1.0))
        .collect();
    let expected_huber_grad: Vec<f64> = pred_values
        .iter()
        .map(|&p| huber_grad(f64::from(p), 1.0) / n)
        .collect();

    let pred = Tensor::from_slice(&pred_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;

    let huber_loss = pred.huber_loss(&target, 1.0)?;
    close(
        "HuberLoss(delta=1) forward under reordered, asymmetric storage",
        &huber_loss.to_vec()?,
        &expected_huber,
    );
    // SmoothL1Loss(beta=1) must equal HuberLoss(delta=1) exactly (the algebraic identity at
    // beta == delta == 1, where dividing by beta changes nothing).
    let smooth_at_one = pred.smooth_l1_loss(&target, 1.0)?;
    close(
        "SmoothL1Loss(beta=1) matches HuberLoss(delta=1) exactly",
        &smooth_at_one.to_vec()?,
        &expected_huber,
    );
    huber_loss.mean([batch, feature])?.backward()?;
    close(
        "HuberLoss gradient under reordered storage",
        &pred.grad().unwrap().to_vec()?,
        &expected_huber_grad,
    );

    // morpheus's RBC/WBC radius and offset regression heads call
    // `F.smooth_l1_loss(..., beta=.02)` throughout
    // `research/src/vision/morpheus/mobilesam/scale10` (for example `train_click_rbc.py:558`);
    // this is that non-default beta.
    let beta = 0.02f32;
    let small_pred_values = [0.0f32, 0.05, -0.1, 0.01, 0.03, -0.019];
    let small_target_values = [0.0f32; 6];
    let expected_smooth: Vec<f64> = small_pred_values
        .iter()
        .map(|&p| huber(f64::from(p), f64::from(beta)) / f64::from(beta))
        .collect();
    let expected_smooth_grad: Vec<f64> = small_pred_values
        .iter()
        .map(|&p| huber_grad(f64::from(p), f64::from(beta)) / f64::from(beta) / n)
        .collect();
    let small_pred =
        Tensor::from_slice(&small_pred_values, [batch.of(2), feature.of(3)], &device)?.with_grad();
    let small_target =
        Tensor::from_slice(&small_target_values, [batch.of(2), feature.of(3)], &device)?;
    let smooth = small_pred.smooth_l1_loss(&small_target, beta)?;
    close(
        "SmoothL1Loss(beta=.02) matches morpheus's own default",
        &smooth.to_vec()?,
        &expected_smooth,
    );
    smooth.mean([batch, feature])?.backward()?;
    close(
        "SmoothL1Loss(beta=.02) gradient",
        &small_pred.grad().unwrap().to_vec()?,
        &expected_smooth_grad,
    );

    assert!(pred.huber_loss(&target, 0.0).is_err());
    assert!(pred.huber_loss(&target, f32::NAN).is_err());
    assert!(pred.smooth_l1_loss(&target, -1.0).is_err());
    let other = Axis::new("other");
    let wrong_axes = Tensor::from_slice(&[1.0f32], [other.of(1)], &device)?;
    assert!(pred.huber_loss(&wrong_axes, 1.0).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn nll_loss_matches_hand_computed_oracle_mirroring_cross_entropy_targets() -> Result<()> {
    // `self` holds log-probabilities; targets follow the same constant one-hot/probability
    // convention `categorical_cross_entropy_with_logits` uses (row 1 below is a genuinely soft
    // target, demonstrating the generalization). Loss is -sum(class, target * log_prob);
    // gradient wrt `self` is exactly -target (the expression is linear in `self`).
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("batch"), Axis::new("class"));
    let probabilities = [[0.7f64, 0.2, 0.1], [0.2, 0.3, 0.5]];
    let log_prob_values: Vec<f32> = probabilities
        .iter()
        .flat_map(|row| row.iter().map(|p| p.ln() as f32))
        .collect();
    let target_values = [1.0f32, 0.0, 0.0, 0.5, 0.5, 0.0];
    let target_rows = [[1.0f64, 0.0, 0.0], [0.5, 0.5, 0.0]];
    let expected_loss: Vec<f64> = probabilities
        .iter()
        .zip(target_rows)
        .map(|(probs, targets)| {
            -probs
                .iter()
                .zip(targets)
                .map(|(&p, t)| t * p.ln())
                .sum::<f64>()
        })
        .collect();
    let n_rows = probabilities.len() as f64;
    let expected_grad: Vec<f64> = target_values
        .iter()
        .map(|&t| -f64::from(t) / n_rows)
        .collect();

    // value(b, c) written in canonical (batch, class) order; asymmetric extents (2, 3).
    let log_prob = Tensor::from_slice(&log_prob_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?
        .with_grad();
    let targets = Tensor::from_slice(&target_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?;

    let loss = log_prob.nll_loss(&targets, class)?;
    assert_eq!(loss.shape(), &Shape::new([batch.of(2)])?);
    close(
        "NLLLoss forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean(batch)?.backward()?;
    close(
        "NLLLoss gradient (exactly -target)",
        &log_prob.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(
        log_prob
            .detach()
            .nll_loss(&targets.with_grad(), class)
            .is_err()
    );
    let invalid_targets = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 0.5, 0.6, 0.0],
        [batch.of(2), class.of(3)],
        &device,
    )?;
    let error = log_prob
        .detach()
        .nll_loss(&invalid_targets, class)
        .err()
        .expect("invalid NLL target")
        .to_string();
    assert!(error.contains("row 1 must sum to 1"), "{error}");
    let missing_class = Tensor::from_slice(&[1.0, 0.0], [batch.of(2)], &device)?;
    assert!(log_prob.detach().nll_loss(&missing_class, class).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn binary_cross_entropy_matches_hand_computed_oracle_with_pytorch_log_clamp() -> Result<()> {
    // `binary_cross_entropy` takes probabilities (unlike the existing logits-based
    // `binary_cross_entropy_with_logits`). Forward matches PyTorch's documented
    // -[y*log(x) + (1-y)*log(1-x)] with each logarithm floored at PyTorch's own exact literal
    // `-100`, computed by a dedicated kernel pair (`bce_loss`/`bce_loss_backward`) rather than
    // composed from `Tensor::clamp`/`Tensor::ln`, so backward stays finite at x == 0 and x == 1
    // without needing an input-side approximation of `-100` (see the doc comment on
    // `binary_cross_entropy` for why the earlier composed floor could only reach `~-87.3`).
    // Backward is PyTorch's own explicit formula, `(x - y) / max((1 - x) * x, eps)` with
    // `eps = 1e-12`, not a derivative of the floored forward expression.
    fn bce(x: f32, y: f32) -> f64 {
        let log_x = f64::from(x).ln().max(-100.0);
        let log_1mx = f64::from(1.0 - x).ln().max(-100.0);
        -(f64::from(y) * log_x + f64::from(1.0 - y) * log_1mx)
    }
    fn bce_grad(x: f32, y: f32) -> f64 {
        let denominator = (f64::from(1.0 - x) * f64::from(x)).max(1e-12);
        (f64::from(x) - f64::from(y)) / denominator
    }

    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    // value(b, f) written in canonical (batch, feature) order; asymmetric extents (2, 3).
    let x_values = [0.5f32, 0.1, 0.0, 0.0, 1.0, 1.0];
    let y_values = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0];
    let n = x_values.len() as f64;
    let expected_loss: Vec<f64> = x_values
        .iter()
        .zip(y_values)
        .map(|(&x, y)| bce(x, y))
        .collect();
    let expected_grad: Vec<f64> = x_values
        .iter()
        .zip(y_values)
        .map(|(&x, y)| bce_grad(x, y) / n)
        .collect();

    let x = Tensor::from_slice(&x_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let y = Tensor::from_slice(&y_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;

    let loss = x.binary_cross_entropy(&y)?;
    close(
        "BCELoss forward with PyTorch's exact -100 log floor",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, feature])?.backward()?;
    close(
        "BCELoss gradient (PyTorch's explicit clamped-denominator form, finite at 0 and 1)",
        &x.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(x.detach().binary_cross_entropy(&y.with_grad()).is_err());
    let other = Axis::new("other");
    let wrong_axes = Tensor::from_slice(&y_values, [other.of(6)], &device)?;
    assert!(x.detach().binary_cross_entropy(&wrong_axes).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn kl_div_loss_matches_hand_computed_oracle_including_zero_target_convention() -> Result<()> {
    // `self` holds log-probabilities (a student's `log_softmax`), `target` holds probabilities
    // (a teacher's `softmax`) -- matching morpheus's distillation heads,
    // `F.kl_div(F.log_softmax(logits / t, -1), F.softmax(teacher_logits / t, -1), ...)`
    // (`research/src/vision/morpheus/mobilesam/scale10/microtier/runs/wbc-edgepath-slice4m/train.py:124`).
    // Row 1 includes a zero target element to exercise the `xlogy` convention: `0 * log(0)`
    // must contribute exactly `0`, not `NaN`.
    fn kl_term(log_q: f64, p: f64) -> f64 {
        if p == 0.0 { 0.0 } else { p * (p.ln() - log_q) }
    }
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("batch"), Axis::new("class"));
    // value(b, c) written in canonical (batch, class) order; asymmetric extents (2, 3).
    let log_q_values = [
        0.3f64.ln() as f32,
        0.2f64.ln() as f32,
        0.5f64.ln() as f32,
        -0.5,
        -1.0,
        -2.0,
    ];
    let p_values = [0.6f32, 0.4, 0.0, 0.0, 0.3, 0.7];
    let n = log_q_values.len() as f64;
    let expected_loss: Vec<f64> = log_q_values
        .iter()
        .zip(p_values)
        .map(|(&lq, p)| kl_term(f64::from(lq), f64::from(p)))
        .collect();
    let expected_grad: Vec<f64> = p_values.iter().map(|&p| -f64::from(p) / n).collect();

    let log_q = Tensor::from_slice(&log_q_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?
        .with_grad();
    let p = Tensor::from_slice(&p_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?;

    let loss = log_q.kl_div_loss(&p)?;
    close(
        "KLDivLoss forward under reordered, asymmetric storage, zero-target convention",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, class])?.backward()?;
    close(
        "KLDivLoss gradient (exactly -target)",
        &log_q.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(log_q.detach().kl_div_loss(&p.with_grad()).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn poisson_nll_loss_matches_hand_computed_oracle_at_default_log_input() -> Result<()> {
    // PyTorch's default `PoissonNLLLoss(log_input=True, full=False)`: loss = exp(self) -
    // target * self. The Stirling `full=True` term and the `log_input=False`/`eps` branch are
    // not implemented.
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    // value(b, f) written in canonical (batch, feature) order; asymmetric extents (2, 3).
    let log_rate_values = [0.0f32, 1.0, -0.5, 2.0, -1.0, 0.5];
    let target_values = [1.0f32, 3.0, 0.5, 8.0, 0.2, 1.5];
    let n = log_rate_values.len() as f64;
    let expected_loss: Vec<f64> = log_rate_values
        .iter()
        .zip(target_values)
        .map(|(&x, t)| f64::from(x).exp() - f64::from(t) * f64::from(x))
        .collect();
    let expected_grad: Vec<f64> = log_rate_values
        .iter()
        .zip(target_values)
        .map(|(&x, t)| (f64::from(x).exp() - f64::from(t)) / n)
        .collect();

    let log_rate = Tensor::from_slice(&log_rate_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;

    let loss = log_rate.poisson_nll_loss(&target)?;
    close(
        "PoissonNLLLoss forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, feature])?.backward()?;
    close(
        "PoissonNLLLoss gradient (exp(input) - target)",
        &log_rate.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(
        log_rate
            .detach()
            .poisson_nll_loss(&target.with_grad())
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gaussian_nll_loss_matches_hand_computed_oracle_and_rejects_negative_variance() -> Result<()> {
    // PyTorch's default `GaussianNLLLoss(full=False)`: loss = 0.5*(ln(max(var,eps)) +
    // (mean-target)^2/max(var,eps)). Element (0, 1) below has var below eps and exercises the
    // documented deviation from PyTorch: Axis's ordinary `clamp` zeroes *var*'s own gradient
    // there (PyTorch's no_grad-based clamp would instead pass it straight through).
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    let eps = 1e-3f32;
    // value(b, f) written in canonical (batch, feature) order; asymmetric extents (2, 3).
    let mean_values = [0.0f32, 2.0, -1.0, 0.5, 1.0, -2.0];
    let target_values = [1.0f32, 2.0, 0.0, 0.5, 1.5, -2.0];
    let var_values = [0.5f32, 1e-8, 2.0, 1.0, 3.0, 0.25];
    let n = mean_values.len() as f64;
    let clamped_var: Vec<f64> = var_values
        .iter()
        .map(|&v| f64::from(v).max(f64::from(eps)))
        .collect();
    let expected_loss: Vec<f64> = mean_values
        .iter()
        .zip(target_values)
        .zip(clamped_var.iter())
        .map(|((&m, t), &v)| {
            let diff = f64::from(m) - f64::from(t);
            0.5 * (v.ln() + diff * diff / v)
        })
        .collect();
    let expected_mean_grad: Vec<f64> = mean_values
        .iter()
        .zip(target_values)
        .zip(clamped_var.iter())
        .map(|((&m, t), &v)| (f64::from(m) - f64::from(t)) / v / n)
        .collect();
    let expected_var_grad: Vec<f64> = mean_values
        .iter()
        .zip(target_values)
        .zip(var_values.iter())
        .map(|((&m, t), &raw_var)| {
            if f64::from(raw_var) < f64::from(eps) {
                0.0 // clamped: Axis's ordinary `clamp` boundary rule, not PyTorch's no_grad passthrough
            } else {
                let diff = f64::from(m) - f64::from(t);
                let v = f64::from(raw_var);
                0.5 * (1.0 / v - diff * diff / (v * v)) / n
            }
        })
        .collect();

    let mean = Tensor::from_slice(&mean_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;
    let var = Tensor::from_slice(&var_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();

    let loss = mean.gaussian_nll_loss(&target, &var, eps)?;
    close(
        "GaussianNLLLoss forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, feature])?.backward()?;
    close(
        "GaussianNLLLoss gradient wrt mean",
        &mean.grad().unwrap().to_vec()?,
        &expected_mean_grad,
    );
    close(
        "GaussianNLLLoss gradient wrt var (clamp-zeroed below eps)",
        &var.grad().unwrap().to_vec()?,
        &expected_var_grad,
    );

    assert!(
        mean.detach()
            .gaussian_nll_loss(&target.with_grad(), &var.detach(), eps)
            .is_err()
    );
    assert!(
        mean.detach()
            .gaussian_nll_loss(&target, &var.detach(), 0.0)
            .is_err()
    );
    assert!(
        mean.detach()
            .gaussian_nll_loss(&target, &var.detach(), f32::NAN)
            .is_err()
    );
    let negative_var = Tensor::from_slice(&[-1.0f32; 6], [batch.of(2), feature.of(3)], &device)?;
    let error = mean
        .detach()
        .gaussian_nll_loss(&target, &negative_var, eps)
        .err()
        .expect("negative var")
        .to_string();
    assert!(error.contains("nonnegative"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn soft_margin_loss_matches_hand_computed_oracle_with_pm_one_targets() -> Result<()> {
    // Unreduced `SoftMarginLoss`: log(1 + exp(-target * self)) for target in {-1, +1}.
    // Gradient is -target * sigmoid(-target * self).
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    // value(b, f) written in canonical (batch, feature) order; asymmetric extents (2, 3).
    let logit_values = [0.0f32, 2.0, -1.0, 5.0, -3.0, 0.5];
    let target_values = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0];
    let n = logit_values.len() as f64;
    let expected_loss: Vec<f64> = logit_values
        .iter()
        .zip(target_values)
        .map(|(&x, y)| {
            let z = -f64::from(y) * f64::from(x);
            z.max(0.0) + (-z.abs()).exp().ln_1p()
        })
        .collect();
    let expected_grad: Vec<f64> = logit_values
        .iter()
        .zip(target_values)
        .map(|(&x, y)| {
            let z = -f64::from(y) * f64::from(x);
            let sigmoid = 1.0 / (1.0 + (-z).exp());
            -f64::from(y) * sigmoid / n
        })
        .collect();

    let logits = Tensor::from_slice(&logit_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let targets = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;

    let loss = logits.soft_margin_loss(&targets)?;
    close(
        "SoftMarginLoss forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, feature])?.backward()?;
    close(
        "SoftMarginLoss gradient",
        &logits.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(
        logits
            .detach()
            .soft_margin_loss(&targets.with_grad())
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multilabel_soft_margin_loss_matches_composition_of_bce_with_logits_and_mean() -> Result<()> {
    // MultiLabelSoftMarginLoss's own formula, -1/C * sum_c [y_c*log(sigmoid(x_c)) +
    // (1-y_c)*log(1-sigmoid(x_c))], is exactly the mean over `class` of
    // `binary_cross_entropy_with_logits`'s own stable elementwise output -- reusing that
    // already-tested kernel's stable form as the independent oracle here.
    fn stable_bce_with_logits(x: f64, y: f64) -> f64 {
        x.max(0.0) - x * y + (-x.abs()).exp().ln_1p()
    }
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("batch"), Axis::new("class"));
    // value(b, c) written in canonical (batch, class) order; asymmetric extents (2, 3).
    let logit_values = [2.0f32, -1.0, 0.0, -3.0, 1.0, 4.0];
    let target_values = [1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0];
    let width = 3usize;
    let expected_loss: Vec<f64> = logit_values
        .chunks_exact(width)
        .zip(target_values.chunks_exact(width))
        .map(|(logits, targets)| {
            logits
                .iter()
                .zip(targets)
                .map(|(&x, &y)| stable_bce_with_logits(f64::from(x), f64::from(y)))
                .sum::<f64>()
                / width as f64
        })
        .collect();
    let batches = logit_values.len() / width;
    let expected_grad: Vec<f64> = logit_values
        .iter()
        .zip(target_values)
        .map(|(&x, y)| {
            let sigmoid = 1.0 / (1.0 + (-f64::from(x)).exp());
            (sigmoid - f64::from(y)) / width as f64 / batches as f64
        })
        .collect();

    let logits = Tensor::from_slice(&logit_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?
        .with_grad();
    let targets = Tensor::from_slice(&target_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?;

    let loss = logits.multilabel_soft_margin_loss(&targets, class)?;
    assert_eq!(loss.shape(), &Shape::new([batch.of(2)])?);
    close(
        "MultiLabelSoftMarginLoss forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean(batch)?.backward()?;
    close(
        "MultiLabelSoftMarginLoss gradient",
        &logits.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(
        logits
            .detach()
            .multilabel_soft_margin_loss(&targets.with_grad(), class)
            .is_err()
    );
    Ok(())
}

#[test]
fn padding_rejects_invalid_configuration_before_launch() {
    let (height, width) = (Axis::new("height"), Axis::new("width"));
    let shape = Shape::new([height.of(3), width.of(4)]).unwrap();

    // Every padding mode requires at least one entry.
    assert!(ZeroPad::new(Vec::<(Axis, usize, usize)>::new()).is_err());
    assert!(ConstantPad::new(Vec::<(Axis, usize, usize)>::new(), 0.0).is_err());
    assert!(ReflectionPad::new(Vec::<(Axis, usize, usize)>::new()).is_err());
    assert!(ReplicationPad::new(Vec::<(Axis, usize, usize)>::new()).is_err());
    assert!(CircularPad::new(Vec::<(Axis, usize, usize)>::new()).is_err());

    // The same axis cannot be listed twice.
    let error = ZeroPad::new([(height, 1, 1), (height, 0, 1)])
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("more than once"), "{error}");

    // An axis the input does not have is rejected before any device work.
    let missing = Axis::new("missing");
    let error = ZeroPad::new([(missing, 1, 1)])
        .unwrap()
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not in the input shape"), "{error}");

    // `ConstantPad` rejects a non-finite value.
    assert!(ConstantPad::new([(height, 1, 1)], f32::NAN).is_err());
    assert!(ConstantPad::new([(height, 1, 1)], f32::INFINITY).is_err());

    // Reflection padding must stay strictly less than the axis extent
    // (PyTorch's own `ReflectionPad*` constraint): height's extent is 3, so
    // a pad of 3 on either side is rejected, but 2 is fine.
    let error = ReflectionPad::new([(height, 3, 0)])
        .unwrap()
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("less than"), "{error}");
    assert!(
        ReflectionPad::new([(height, 2, 2)])
            .unwrap()
            .output_shape(&shape)
            .is_ok()
    );

    // Circular padding allows padding equal to the extent (a full wrap) but
    // not more.
    assert!(
        CircularPad::new([(height, 3, 3)])
            .unwrap()
            .output_shape(&shape)
            .is_ok()
    );
    let error = CircularPad::new([(height, 4, 0)])
        .unwrap()
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("at most"), "{error}");

    // Replication padding has no upper bound: a pad larger than the extent
    // is still accepted (it just repeats the edge value further).
    assert!(
        ReplicationPad::new([(height, 10, 10)])
            .unwrap()
            .output_shape(&shape)
            .is_ok()
    );

    // A well-formed `ZeroPad` reports the expected shape purely from shape
    // math; every axis not listed (`width`) is preserved unchanged.
    let pad = ZeroPad::new([(height, 1, 2)]).unwrap();
    assert_eq!(
        pad.output_shape(&shape).unwrap(),
        Shape::new([height.of(6), width.of(4)]).unwrap()
    );
}

#[test]
fn pixel_and_channel_shuffle_reject_invalid_configuration_before_launch() {
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let shape = Shape::new([channel.of(9), height.of(2), width.of(3)]).unwrap();

    // PixelShuffle requires distinct channel/spatial axes and factor >= 1.
    assert!(PixelShuffle::new(channel, [channel, width], 3).is_err());
    assert!(PixelShuffle::new(channel, [height, height], 3).is_err());
    assert!(PixelShuffle::new(channel, [height, width], 0).is_err());

    // Channel extent 9 is not divisible by upscale_factor^2 (4).
    let error = PixelShuffle::new(channel, [height, width], 2)
        .unwrap()
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not divisible"), "{error}");
    // ...but it is divisible by 3^2 = 9.
    assert_eq!(
        PixelShuffle::new(channel, [height, width], 3)
            .unwrap()
            .output_shape(&shape)
            .unwrap(),
        Shape::new([channel.of(1), height.of(6), width.of(9)]).unwrap()
    );

    // PixelUnshuffle requires height/width divisible by the downscale factor.
    let error = PixelUnshuffle::new(channel, [height, width], 2)
        .unwrap()
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("must both be divisible"), "{error}");
    assert!(PixelUnshuffle::new(channel, [height, width], 0).is_err());

    // ChannelShuffle requires the channel extent divisible by groups, and
    // at least one group.
    assert!(ChannelShuffle::new(channel, 0).is_err());
    let error = ChannelShuffle::new(channel, 2)
        .unwrap()
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("not divisible"), "{error}");
    assert!(
        ChannelShuffle::new(channel, 3)
            .unwrap()
            .output_shape(&shape)
            .is_ok()
    );
}

#[test]
#[ignore = "requires CUDA"]
fn zero_pad_matches_hand_computed_forward_and_gradient_across_1d_2d_3d() -> Result<()> {
    // Independent oracle: real PyTorch 2.14 `F.pad(mode="constant", value=0.0)`,
    // run offline and transcribed as literals -- never computed by the op
    // under test.
    let device = Device::cuda(0)?;
    let (channel, height, width, depth) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("depth"),
    );

    // "1d": pad only `height`, asymmetric (before=1, after=2).
    let x1: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let input1 = Tensor::from_slice(&x1, [channel.of(2), height.of(3)], &device)?.with_grad();
    let mut pad1 = ZeroPad::new([(height, 1, 2)])?;
    assert_eq!(
        pad1.build(input1.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6)])?
    );
    let out1 = pad1.forward(&input1)?;
    close(
        "ZeroPad 1-axis forward",
        &out1.to_vec()?,
        &[0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0],
    );
    let w1: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let weight1 = Tensor::from_slice(&w1, [channel.of(2), height.of(6)], &device)?;
    out1.mul(&weight1)?.mean([channel, height])?.backward()?;
    close(
        "ZeroPad 1-axis gradient",
        &input1.grad().unwrap().to_vec()?,
        &[0.166667, 0.250000, 0.333333, 0.666667, 0.750000, 0.833333],
    );

    // "2d": pad `height` and `width`, asymmetric on both, under physical
    // storage transposed relative to the logical [channel, height, width]
    // order.
    let x2: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input2 = Tensor::from_slice(&x2, [channel.of(2), height.of(3), width.of(4)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();
    let mut pad2 = ZeroPad::new([(height, 1, 2), (width, 2, 1)])?;
    assert_eq!(
        pad2.build(input2.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6), width.of(7)])?
    );
    let out2 = pad2.forward(&input2)?;
    close(
        "ZeroPad 2-axis forward under reordered storage",
        &out2.to_vec()?,
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0,
            5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 16.0, 17.0, 18.0, 19.0, 0.0, 0.0, 0.0, 20.0,
            21.0, 22.0, 23.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ],
    );
    let w2: Vec<f32> = (1..=84).map(|v| v as f32).collect();
    let weight2 = Tensor::from_slice(&w2, [channel.of(2), height.of(6), width.of(7)], &device)?;
    out2.mul(&weight2)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "ZeroPad 2-axis gradient under reordered storage",
        &input2.grad().unwrap().to_vec()?,
        &[
            0.119048, 0.130952, 0.142857, 0.154762, 0.202381, 0.214286, 0.226190, 0.238095,
            0.285714, 0.297619, 0.309524, 0.321429, 0.619048, 0.630952, 0.642857, 0.654762,
            0.702381, 0.714286, 0.726191, 0.738095, 0.785714, 0.797619, 0.809524, 0.821429,
        ],
    );

    // "3d": pad `depth`, `height`, `width` simultaneously, symmetric.
    let x3: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let input3 =
        Tensor::from_slice(&x3, [depth.of(2), height.of(3), width.of(3)], &device)?.with_grad();
    let mut pad3 = ZeroPad::new([(depth, 1, 1), (height, 1, 1), (width, 1, 1)])?;
    assert_eq!(
        pad3.build(input3.shape(), &device, 0)?,
        Shape::new([depth.of(4), height.of(5), width.of(5)])?
    );
    let out3 = pad3.forward(&input3)?;
    close(
        "ZeroPad 3-axis forward",
        &out3.to_vec()?,
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0,
            0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0, 11.0, 0.0, 0.0, 12.0, 13.0, 14.0, 0.0, 0.0, 15.0,
            16.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    let w3: Vec<f32> = (1..=100).map(|v| v as f32).collect();
    let weight3 = Tensor::from_slice(&w3, [depth.of(4), height.of(5), width.of(5)], &device)?;
    out3.mul(&weight3)?
        .mean([depth, height, width])?
        .backward()?;
    close(
        "ZeroPad 3-axis gradient",
        &input3.grad().unwrap().to_vec()?,
        &[
            0.32, 0.33, 0.34, 0.37, 0.38, 0.39, 0.42, 0.43, 0.44, 0.57, 0.58, 0.59, 0.62, 0.63,
            0.64, 0.67, 0.68, 0.69,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn constant_pad_matches_hand_computed_forward_and_gradient_across_1d_2d_3d() -> Result<()> {
    // Independent oracle: real PyTorch 2.14 `F.pad(mode="constant", value=...)`.
    let device = Device::cuda(0)?;
    let (channel, height, width, depth) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("depth"),
    );

    // "1d": value=7.5.
    let x1: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let input1 = Tensor::from_slice(&x1, [channel.of(2), height.of(3)], &device)?.with_grad();
    let mut pad1 = ConstantPad::new([(height, 1, 2)], 7.5)?;
    assert_eq!(
        pad1.build(input1.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6)])?
    );
    let out1 = pad1.forward(&input1)?;
    close(
        "ConstantPad 1-axis forward (value=7.5)",
        &out1.to_vec()?,
        &[7.5, 0.0, 1.0, 2.0, 7.5, 7.5, 7.5, 3.0, 4.0, 5.0, 7.5, 7.5],
    );
    let w1: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let weight1 = Tensor::from_slice(&w1, [channel.of(2), height.of(6)], &device)?;
    out1.mul(&weight1)?.mean([channel, height])?.backward()?;
    close(
        // Identical to ZeroPad's own gradient: the constant border term is
        // detached and contributes zero derivative.
        "ConstantPad 1-axis gradient",
        &input1.grad().unwrap().to_vec()?,
        &[0.166667, 0.250000, 0.333333, 0.666667, 0.750000, 0.833333],
    );

    // "2d": value=-3.25, asymmetric on both axes.
    let x2: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input2 =
        Tensor::from_slice(&x2, [channel.of(2), height.of(3), width.of(4)], &device)?.with_grad();
    let mut pad2 = ConstantPad::new([(height, 1, 2), (width, 2, 1)], -3.25)?;
    assert_eq!(
        pad2.build(input2.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6), width.of(7)])?
    );
    let out2 = pad2.forward(&input2)?;
    close(
        "ConstantPad 2-axis forward (value=-3.25)",
        &out2.to_vec()?,
        &[
            -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, 0.0, 1.0, 2.0, 3.0,
            -3.25, -3.25, -3.25, 4.0, 5.0, 6.0, 7.0, -3.25, -3.25, -3.25, 8.0, 9.0, 10.0, 11.0,
            -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25,
            -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25,
            12.0, 13.0, 14.0, 15.0, -3.25, -3.25, -3.25, 16.0, 17.0, 18.0, 19.0, -3.25, -3.25,
            -3.25, 20.0, 21.0, 22.0, 23.0, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25,
            -3.25, -3.25, -3.25, -3.25, -3.25, -3.25, -3.25,
        ],
    );
    let w2: Vec<f32> = (1..=84).map(|v| v as f32).collect();
    let weight2 = Tensor::from_slice(&w2, [channel.of(2), height.of(6), width.of(7)], &device)?;
    out2.mul(&weight2)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "ConstantPad 2-axis gradient",
        &input2.grad().unwrap().to_vec()?,
        &[
            0.119048, 0.130952, 0.142857, 0.154762, 0.202381, 0.214286, 0.226190, 0.238095,
            0.285714, 0.297619, 0.309524, 0.321429, 0.619048, 0.630952, 0.642857, 0.654762,
            0.702381, 0.714286, 0.726191, 0.738095, 0.785714, 0.797619, 0.809524, 0.821429,
        ],
    );

    // "3d": value=2.0, symmetric.
    let x3: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let input3 =
        Tensor::from_slice(&x3, [depth.of(2), height.of(3), width.of(3)], &device)?.with_grad();
    let mut pad3 = ConstantPad::new([(depth, 1, 1), (height, 1, 1), (width, 1, 1)], 2.0)?;
    assert_eq!(
        pad3.build(input3.shape(), &device, 0)?,
        Shape::new([depth.of(4), height.of(5), width.of(5)])?
    );
    let out3 = pad3.forward(&input3)?;
    close(
        "ConstantPad 3-axis forward (value=2.0)",
        &out3.to_vec()?,
        &[
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0,
            2.0, 2.0, 3.0, 4.0, 5.0, 2.0, 2.0, 6.0, 7.0, 8.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 9.0, 10.0, 11.0, 2.0, 2.0, 12.0, 13.0, 14.0, 2.0, 2.0, 15.0,
            16.0, 17.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ],
    );
    let w3: Vec<f32> = (1..=100).map(|v| v as f32).collect();
    let weight3 = Tensor::from_slice(&w3, [depth.of(4), height.of(5), width.of(5)], &device)?;
    out3.mul(&weight3)?
        .mean([depth, height, width])?
        .backward()?;
    close(
        "ConstantPad 3-axis gradient",
        &input3.grad().unwrap().to_vec()?,
        &[
            0.32, 0.33, 0.34, 0.37, 0.38, 0.39, 0.42, 0.43, 0.44, 0.57, 0.58, 0.59, 0.62, 0.63,
            0.64, 0.67, 0.68, 0.69,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn reflection_pad_matches_hand_computed_forward_and_gradient_across_1d_2d_3d() -> Result<()> {
    // Independent oracle: real PyTorch 2.14 `F.pad(mode="reflect")`. PyTorch's
    // whole-sample reflection never repeats the edge element (`-1` reflects
    // to `1`, not `0`), and requires each side's pad strictly less than the
    // axis's extent; both properties are exercised here.
    let device = Device::cuda(0)?;
    let (channel, height, width, depth) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("depth"),
    );

    // "1d": pad only `height` (extent 3), asymmetric (before=1, after=2),
    // both strictly less than the extent.
    let x1: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let input1 = Tensor::from_slice(&x1, [channel.of(2), height.of(3)], &device)?.with_grad();
    let mut pad1 = ReflectionPad::new([(height, 1, 2)])?;
    assert_eq!(
        pad1.build(input1.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6)])?
    );
    let out1 = pad1.forward(&input1)?;
    close(
        "ReflectionPad 1-axis forward",
        &out1.to_vec()?,
        &[1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0],
    );
    let w1: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let weight1 = Tensor::from_slice(&w1, [channel.of(2), height.of(6)], &device)?;
    out1.mul(&weight1)?.mean([channel, height])?.backward()?;
    close(
        "ReflectionPad 1-axis gradient",
        &input1.grad().unwrap().to_vec()?,
        &[0.666667, 0.750000, 0.333333, 1.666667, 2.250000, 0.833333],
    );

    // "2d": under reordered physical storage.
    let x2: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input2 = Tensor::from_slice(&x2, [channel.of(2), height.of(3), width.of(4)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();
    let mut pad2 = ReflectionPad::new([(height, 1, 2), (width, 2, 1)])?;
    assert_eq!(
        pad2.build(input2.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6), width.of(7)])?
    );
    let out2 = pad2.forward(&input2)?;
    close(
        "ReflectionPad 2-axis forward under reordered storage",
        &out2.to_vec()?,
        &[
            6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 6.0, 5.0, 4.0,
            5.0, 6.0, 7.0, 6.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 6.0, 5.0, 4.0, 5.0, 6.0,
            7.0, 6.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 18.0,
            14.0, 13.0, 12.0, 13.0, 14.0, 15.0, 14.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 18.0,
            22.0, 21.0, 20.0, 21.0, 22.0, 23.0, 22.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 18.0,
            14.0, 13.0, 12.0, 13.0, 14.0, 15.0, 14.0,
        ],
    );
    let w2: Vec<f32> = (1..=84).map(|v| v as f32).collect();
    let weight2 = Tensor::from_slice(&w2, [channel.of(2), height.of(6), width.of(7)], &device)?;
    out2.mul(&weight2)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "ReflectionPad 2-axis gradient under reordered storage",
        &input2.grad().unwrap().to_vec()?,
        &[
            0.571429, 1.142857, 1.809524, 0.642857, 0.607143, 1.214286, 1.964286, 0.714286,
            0.285714, 0.571429, 0.904762, 0.321429, 1.571429, 3.142857, 4.809524, 1.642857,
            2.107143, 4.214286, 6.464286, 2.214286, 0.785714, 1.571429, 2.404762, 0.821429,
        ],
    );

    // "3d": pad `depth`, `height`, `width` simultaneously, symmetric by 1
    // (each axis's extent is at least 2, so before=after=1 stays strictly
    // less than every extent).
    let x3: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let input3 =
        Tensor::from_slice(&x3, [depth.of(2), height.of(3), width.of(3)], &device)?.with_grad();
    let mut pad3 = ReflectionPad::new([(depth, 1, 1), (height, 1, 1), (width, 1, 1)])?;
    assert_eq!(
        pad3.build(input3.shape(), &device, 0)?,
        Shape::new([depth.of(4), height.of(5), width.of(5)])?
    );
    let out3 = pad3.forward(&input3)?;
    close(
        "ReflectionPad 3-axis forward",
        &out3.to_vec()?,
        &[
            13.0, 12.0, 13.0, 14.0, 13.0, 10.0, 9.0, 10.0, 11.0, 10.0, 13.0, 12.0, 13.0, 14.0,
            13.0, 16.0, 15.0, 16.0, 17.0, 16.0, 13.0, 12.0, 13.0, 14.0, 13.0, 4.0, 3.0, 4.0, 5.0,
            4.0, 1.0, 0.0, 1.0, 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 4.0, 7.0, 6.0, 7.0, 8.0, 7.0, 4.0,
            3.0, 4.0, 5.0, 4.0, 13.0, 12.0, 13.0, 14.0, 13.0, 10.0, 9.0, 10.0, 11.0, 10.0, 13.0,
            12.0, 13.0, 14.0, 13.0, 16.0, 15.0, 16.0, 17.0, 16.0, 13.0, 12.0, 13.0, 14.0, 13.0,
            4.0, 3.0, 4.0, 5.0, 4.0, 1.0, 0.0, 1.0, 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 4.0, 7.0, 6.0,
            7.0, 8.0, 7.0, 4.0, 3.0, 4.0, 5.0, 4.0,
        ],
    );
    let w3: Vec<f32> = (1..=100).map(|v| v as f32).collect();
    let weight3 = Tensor::from_slice(&w3, [depth.of(4), height.of(5), width.of(5)], &device)?;
    out3.mul(&weight3)?
        .mean([depth, height, width])?
        .backward()?;
    close(
        "ReflectionPad 3-axis gradient",
        &input3.grad().unwrap().to_vec()?,
        &[
            1.14, 3.48, 1.18, 3.72, 11.34, 3.84, 1.34, 4.08, 1.38, 0.64, 1.98, 0.68, 2.22, 6.84,
            2.34, 0.84, 2.58, 0.88,
        ],
    );

    // Rejected before any device work: padding at or beyond the axis extent.
    let shape = Shape::new([height.of(3)]).unwrap();
    assert!(
        ReflectionPad::new([(height, 3, 0)])
            .unwrap()
            .output_shape(&shape)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn replication_pad_matches_hand_computed_forward_and_gradient_across_1d_2d_3d() -> Result<()> {
    // Independent oracle: real PyTorch 2.14 `F.pad(mode="replicate")`.
    let device = Device::cuda(0)?;
    let (channel, height, width, depth) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("depth"),
    );

    // "1d".
    let x1: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let input1 = Tensor::from_slice(&x1, [channel.of(2), height.of(3)], &device)?.with_grad();
    let mut pad1 = ReplicationPad::new([(height, 1, 2)])?;
    assert_eq!(
        pad1.build(input1.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6)])?
    );
    let out1 = pad1.forward(&input1)?;
    close(
        "ReplicationPad 1-axis forward",
        &out1.to_vec()?,
        &[0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0],
    );
    let w1: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let weight1 = Tensor::from_slice(&w1, [channel.of(2), height.of(6)], &device)?;
    out1.mul(&weight1)?.mean([channel, height])?.backward()?;
    close(
        "ReplicationPad 1-axis gradient",
        &input1.grad().unwrap().to_vec()?,
        &[0.25, 0.25, 1.25, 1.25, 0.75, 2.75],
    );

    // "2d": under reordered physical storage.
    let x2: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input2 = Tensor::from_slice(&x2, [channel.of(2), height.of(3), width.of(4)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();
    let mut pad2 = ReplicationPad::new([(height, 1, 2), (width, 2, 1)])?;
    assert_eq!(
        pad2.build(input2.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6), width.of(7)])?
    );
    let out2 = pad2.forward(&input2)?;
    close(
        "ReplicationPad 2-axis forward under reordered storage",
        &out2.to_vec()?,
        &[
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0,
            5.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 10.0, 11.0, 11.0, 8.0, 8.0, 8.0, 9.0, 10.0,
            11.0, 11.0, 8.0, 8.0, 8.0, 9.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0, 13.0, 14.0, 15.0,
            15.0, 12.0, 12.0, 12.0, 13.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0, 17.0, 18.0, 19.0,
            19.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 23.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0,
            23.0, 20.0, 20.0, 20.0, 21.0, 22.0, 23.0, 23.0,
        ],
    );
    let w2: Vec<f32> = (1..=84).map(|v| v as f32).collect();
    let weight2 = Tensor::from_slice(&w2, [channel.of(2), height.of(6), width.of(7)], &device)?;
    out2.mul(&weight2)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "ReplicationPad 2-axis gradient under reordered storage",
        &input2.grad().unwrap().to_vec()?,
        &[
            0.392857, 0.178571, 0.202381, 0.476191, 0.571429, 0.214286, 0.226190, 0.488095,
            3.214286, 1.142857, 1.178571, 2.464286, 3.392857, 1.178571, 1.202381, 2.476191,
            2.071429, 0.714286, 0.726191, 1.488095, 7.714287, 2.642857, 2.678571, 5.464286,
        ],
    );

    // "3d", symmetric by 1. Also demonstrates no upper bound on padding
    // size: replicate does not share reflect's `pad < extent` constraint.
    let x3: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let input3 =
        Tensor::from_slice(&x3, [depth.of(2), height.of(3), width.of(3)], &device)?.with_grad();
    let mut pad3 = ReplicationPad::new([(depth, 1, 1), (height, 1, 1), (width, 1, 1)])?;
    assert_eq!(
        pad3.build(input3.shape(), &device, 0)?,
        Shape::new([depth.of(4), height.of(5), width.of(5)])?
    );
    let out3 = pad3.forward(&input3)?;
    close(
        "ReplicationPad 3-axis forward",
        &out3.to_vec()?,
        &[
            0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0,
            7.0, 8.0, 8.0, 6.0, 6.0, 7.0, 8.0, 8.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.0,
            2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 8.0, 8.0, 6.0, 6.0, 7.0, 8.0, 8.0, 9.0,
            9.0, 10.0, 11.0, 11.0, 9.0, 9.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 14.0, 14.0, 15.0,
            15.0, 16.0, 17.0, 17.0, 15.0, 15.0, 16.0, 17.0, 17.0, 9.0, 9.0, 10.0, 11.0, 11.0, 9.0,
            9.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 17.0, 17.0,
            15.0, 15.0, 16.0, 17.0, 17.0,
        ],
    );
    let w3: Vec<f32> = (1..=100).map(|v| v as f32).collect();
    let weight3 = Tensor::from_slice(&w3, [depth.of(4), height.of(5), width.of(5)], &device)?;
    out3.mul(&weight3)?
        .mean([depth, height, width])?
        .backward()?;
    close(
        "ReplicationPad 3-axis gradient",
        &input3.grad().unwrap().to_vec()?,
        &[
            1.32, 0.72, 1.56, 0.96, 0.51, 1.08, 2.52, 1.32, 2.76, 5.32, 2.72, 5.559999, 2.96, 1.51,
            3.08, 6.52, 3.32, 6.76,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn circular_pad_matches_hand_computed_forward_and_gradient_and_allows_full_wrap() -> Result<()> {
    // Independent oracle: real PyTorch 2.14 `F.pad(mode="circular")`.
    let device = Device::cuda(0)?;
    let (channel, height, width, depth) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("depth"),
    );

    // "1d".
    let x1: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let input1 = Tensor::from_slice(&x1, [channel.of(2), height.of(3)], &device)?.with_grad();
    let mut pad1 = CircularPad::new([(height, 1, 2)])?;
    assert_eq!(
        pad1.build(input1.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6)])?
    );
    let out1 = pad1.forward(&input1)?;
    close(
        "CircularPad 1-axis forward",
        &out1.to_vec()?,
        &[2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0],
    );
    let w1: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let weight1 = Tensor::from_slice(&w1, [channel.of(2), height.of(6)], &device)?;
    out1.mul(&weight1)?.mean([channel, height])?.backward()?;
    close(
        "CircularPad 1-axis gradient",
        &input1.grad().unwrap().to_vec()?,
        &[0.583333, 0.750000, 0.416667, 1.583333, 1.750000, 1.416667],
    );

    // "2d": under reordered physical storage.
    let x2: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input2 = Tensor::from_slice(&x2, [channel.of(2), height.of(3), width.of(4)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();
    let mut pad2 = CircularPad::new([(height, 1, 2), (width, 2, 1)])?;
    assert_eq!(
        pad2.build(input2.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(6), width.of(7)])?
    );
    let out2 = pad2.forward(&input2)?;
    close(
        "CircularPad 2-axis forward under reordered storage",
        &out2.to_vec()?,
        &[
            10.0, 11.0, 8.0, 9.0, 10.0, 11.0, 8.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 6.0, 7.0,
            4.0, 5.0, 6.0, 7.0, 4.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0, 8.0, 2.0, 3.0, 0.0, 1.0,
            2.0, 3.0, 0.0, 6.0, 7.0, 4.0, 5.0, 6.0, 7.0, 4.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0,
            20.0, 14.0, 15.0, 12.0, 13.0, 14.0, 15.0, 12.0, 18.0, 19.0, 16.0, 17.0, 18.0, 19.0,
            16.0, 22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0, 14.0, 15.0, 12.0, 13.0, 14.0, 15.0,
            12.0, 18.0, 19.0, 16.0, 17.0, 18.0, 19.0, 16.0,
        ],
    );
    let w2: Vec<f32> = (1..=84).map(|v| v as f32).collect();
    let weight2 = Tensor::from_slice(&w2, [channel.of(2), height.of(6), width.of(7)], &device)?;
    out2.mul(&weight2)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "CircularPad 2-axis gradient under reordered storage",
        &input2.grad().unwrap().to_vec()?,
        &[
            1.071429, 0.511905, 0.976191, 1.023810, 1.404762, 0.678571, 1.309524, 1.357143,
            0.738095, 0.345238, 0.642857, 0.690476, 3.071429, 1.511905, 2.976191, 3.023809,
            3.404762, 1.678571, 3.309524, 3.357143, 2.738095, 1.345238, 2.642857, 2.690476,
        ],
    );

    // "3d", symmetric by 1.
    let x3: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let input3 =
        Tensor::from_slice(&x3, [depth.of(2), height.of(3), width.of(3)], &device)?.with_grad();
    let mut pad3 = CircularPad::new([(depth, 1, 1), (height, 1, 1), (width, 1, 1)])?;
    assert_eq!(
        pad3.build(input3.shape(), &device, 0)?,
        Shape::new([depth.of(4), height.of(5), width.of(5)])?
    );
    let out3 = pad3.forward(&input3)?;
    close(
        "CircularPad 3-axis forward",
        &out3.to_vec()?,
        &[
            17.0, 15.0, 16.0, 17.0, 15.0, 11.0, 9.0, 10.0, 11.0, 9.0, 14.0, 12.0, 13.0, 14.0, 12.0,
            17.0, 15.0, 16.0, 17.0, 15.0, 11.0, 9.0, 10.0, 11.0, 9.0, 8.0, 6.0, 7.0, 8.0, 6.0, 2.0,
            0.0, 1.0, 2.0, 0.0, 5.0, 3.0, 4.0, 5.0, 3.0, 8.0, 6.0, 7.0, 8.0, 6.0, 2.0, 0.0, 1.0,
            2.0, 0.0, 17.0, 15.0, 16.0, 17.0, 15.0, 11.0, 9.0, 10.0, 11.0, 9.0, 14.0, 12.0, 13.0,
            14.0, 12.0, 17.0, 15.0, 16.0, 17.0, 15.0, 11.0, 9.0, 10.0, 11.0, 9.0, 8.0, 6.0, 7.0,
            8.0, 6.0, 2.0, 0.0, 1.0, 2.0, 0.0, 5.0, 3.0, 4.0, 5.0, 3.0, 8.0, 6.0, 7.0, 8.0, 6.0,
            2.0, 0.0, 1.0, 2.0, 0.0,
        ],
    );
    let w3: Vec<f32> = (1..=100).map(|v| v as f32).collect();
    let weight3 = Tensor::from_slice(&w3, [depth.of(4), height.of(5), width.of(5)], &device)?;
    out3.mul(&weight3)?
        .mean([depth, height, width])?
        .backward()?;
    close(
        "CircularPad 3-axis gradient",
        &input3.grad().unwrap().to_vec()?,
        &[
            5.28, 2.62, 5.2, 2.54, 1.26, 2.5, 4.88, 2.42, 4.8, 3.28, 1.62, 3.2, 1.54, 0.76, 1.5,
            2.88, 1.42, 2.8,
        ],
    );

    // A full wrap (padding equal to the extent) is allowed, unlike reflect.
    let full: Vec<f32> = (0..4).map(|v| v as f32).collect();
    let input4 = Tensor::from_slice(&full, [height.of(4)], &device)?;
    let out4 = CircularPad::new([(height, 4, 4)])?.forward(&input4)?;
    close(
        "CircularPad forward, padding == extent (full wrap)",
        &out4.to_vec()?,
        &[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn pixel_shuffle_matches_hand_computed_forward_and_gradient_under_reordered_storage() -> Result<()>
{
    // Independent oracle: real PyTorch 2.14 `F.pixel_shuffle`.
    // C*r^2=8, r=2 -> out_channel=2; height=2, width=3 -> new height=4, width=6.
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let values: Vec<f32> = (0..48).map(|v| v as f32).collect();
    let input = Tensor::from_slice(&values, [channel.of(8), height.of(2), width.of(3)], &device)?
        // Reordered physical storage: PixelShuffle's `split`/`merge`
        // composition must read through the permuted strides, not assume
        // row-major.
        .with_layout([width, channel, height])?
        .with_grad();

    let mut shuffle = PixelShuffle::new(channel, [height, width], 2)?;
    assert_eq!(
        shuffle.build(input.shape(), &device, 0)?,
        Shape::new([channel.of(2), height.of(4), width.of(6)])?
    );
    let out = shuffle.forward(&input)?;
    close(
        "PixelShuffle forward under reordered storage",
        &out.to_vec()?,
        &[
            0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 12.0, 18.0, 13.0, 19.0, 14.0, 20.0, 3.0, 9.0, 4.0, 10.0,
            5.0, 11.0, 15.0, 21.0, 16.0, 22.0, 17.0, 23.0, 24.0, 30.0, 25.0, 31.0, 26.0, 32.0,
            36.0, 42.0, 37.0, 43.0, 38.0, 44.0, 27.0, 33.0, 28.0, 34.0, 29.0, 35.0, 39.0, 45.0,
            40.0, 46.0, 41.0, 47.0,
        ],
    );
    let w: Vec<f32> = (1..=48).map(|v| v as f32).collect();
    let weight = Tensor::from_slice(&w, [channel.of(2), height.of(4), width.of(6)], &device)?;
    out.mul(&weight)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "PixelShuffle gradient under reordered storage",
        &input.grad().unwrap().to_vec()?,
        &[
            0.020833, 0.062500, 0.104167, 0.270833, 0.312500, 0.354167, 0.041667, 0.083333,
            0.125000, 0.291667, 0.333333, 0.375000, 0.145833, 0.187500, 0.229167, 0.395833,
            0.437500, 0.479167, 0.166667, 0.208333, 0.250000, 0.416667, 0.458333, 0.500000,
            0.520833, 0.562500, 0.604167, 0.770833, 0.812500, 0.854167, 0.541667, 0.583333,
            0.625000, 0.791667, 0.833333, 0.875000, 0.645833, 0.687500, 0.729167, 0.895833,
            0.937500, 0.979167, 0.666667, 0.708333, 0.750000, 0.916667, 0.958333, 1.000000,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn pixel_unshuffle_matches_hand_computed_forward_and_gradient_and_is_pixel_shuffles_inverse()
-> Result<()> {
    // Independent oracle: real PyTorch 2.14 `F.pixel_unshuffle`.
    // channel=2, height=4, width=6, r=2 -> new_channel=8, height=2, width=3.
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let values: Vec<f32> = (0..48).map(|v| v as f32).collect();
    let input = Tensor::from_slice(&values, [channel.of(2), height.of(4), width.of(6)], &device)?
        .with_grad();

    let mut unshuffle = PixelUnshuffle::new(channel, [height, width], 2)?;
    assert_eq!(
        unshuffle.build(input.shape(), &device, 0)?,
        Shape::new([channel.of(8), height.of(2), width.of(3)])?
    );
    let out = unshuffle.forward(&input)?;
    close(
        "PixelUnshuffle forward",
        &out.to_vec()?,
        &[
            0.0, 2.0, 4.0, 12.0, 14.0, 16.0, 1.0, 3.0, 5.0, 13.0, 15.0, 17.0, 6.0, 8.0, 10.0, 18.0,
            20.0, 22.0, 7.0, 9.0, 11.0, 19.0, 21.0, 23.0, 24.0, 26.0, 28.0, 36.0, 38.0, 40.0, 25.0,
            27.0, 29.0, 37.0, 39.0, 41.0, 30.0, 32.0, 34.0, 42.0, 44.0, 46.0, 31.0, 33.0, 35.0,
            43.0, 45.0, 47.0,
        ],
    );
    let w: Vec<f32> = (1..=48).map(|v| v as f32).collect();
    let weight = Tensor::from_slice(&w, [channel.of(8), height.of(2), width.of(3)], &device)?;
    out.mul(&weight)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "PixelUnshuffle gradient",
        &input.grad().unwrap().to_vec()?,
        &[
            0.020833, 0.145833, 0.041667, 0.166667, 0.062500, 0.187500, 0.270833, 0.395833,
            0.291667, 0.416667, 0.312500, 0.437500, 0.083333, 0.208333, 0.104167, 0.229167,
            0.125000, 0.250000, 0.333333, 0.458333, 0.354167, 0.479167, 0.375000, 0.500000,
            0.520833, 0.645833, 0.541667, 0.666667, 0.562500, 0.687500, 0.770833, 0.895833,
            0.791667, 0.916667, 0.812500, 0.937500, 0.583333, 0.708333, 0.604167, 0.729167,
            0.625000, 0.750000, 0.833333, 0.958333, 0.854167, 0.979167, 0.875000, 1.000000,
        ],
    );

    // Exact round trip: PixelUnshuffle(PixelShuffle(x)) == x, for a fresh
    // tensor unrelated to the oracle-derived values above.
    let round_trip_values: Vec<f32> = (0..48).map(|v| (v as f32) * 0.5 - 3.0).collect();
    let round_trip_input = Tensor::from_slice(
        &round_trip_values,
        [channel.of(2), height.of(4), width.of(6)],
        &device,
    )?;
    let shuffled = PixelShuffle::new(channel, [height, width], 2)?
        .forward(&PixelUnshuffle::new(channel, [height, width], 2)?.forward(&round_trip_input)?)?;
    close(
        "PixelShuffle(PixelUnshuffle(x)) round trip",
        &shuffled.to_vec()?,
        &round_trip_values
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn channel_shuffle_matches_hand_computed_forward_and_gradient_under_reordered_storage() -> Result<()>
{
    // Independent oracle: real PyTorch 2.14 `F.channel_shuffle`. The
    // small illustrative case first reproduces the PyTorch docs' own
    // example ([ch0, ch1, ch2, ch3] at groups=2 becomes [ch0, ch2, ch1,
    // ch3]) with real values, then a larger case checks the gradient
    // under reordered storage.
    let device = Device::cuda(0)?;
    let channel = Axis::new("channel");
    let doc_values: Vec<f32> = (0..4).map(|v| v as f32).collect();
    let doc_input = Tensor::from_slice(&doc_values, [channel.of(4)], &device)?;
    let doc_out = ChannelShuffle::new(channel, 2)?.forward(&doc_input)?;
    close(
        "ChannelShuffle doc example ([ch0,ch1,ch2,ch3] groups=2 -> [ch0,ch2,ch1,ch3])",
        &doc_out.to_vec()?,
        &[0.0, 2.0, 1.0, 3.0],
    );

    // C=6, groups=3, plus two trailing spatial axes (2x2) left untouched,
    // under physical storage transposed relative to the logical
    // [channel, height, width] order.
    let (height, width) = (Axis::new("height"), Axis::new("width"));
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let input = Tensor::from_slice(&values, [channel.of(6), height.of(2), width.of(2)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();

    let mut shuffle = ChannelShuffle::new(channel, 3)?;
    assert_eq!(
        shuffle.build(input.shape(), &device, 0)?,
        Shape::new([channel.of(6), height.of(2), width.of(2)])?
    );
    let out = shuffle.forward(&input)?;
    close(
        "ChannelShuffle forward under reordered storage",
        &out.to_vec()?,
        &[
            0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 4.0, 5.0, 6.0, 7.0,
            12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0,
        ],
    );
    let w: Vec<f32> = (1..=24).map(|v| v as f32).collect();
    let weight = Tensor::from_slice(&w, [channel.of(6), height.of(2), width.of(2)], &device)?;
    out.mul(&weight)?
        .mean([channel, height, width])?
        .backward()?;
    close(
        "ChannelShuffle gradient under reordered storage",
        &input.grad().unwrap().to_vec()?,
        &[
            0.041667, 0.083333, 0.125000, 0.166667, 0.541667, 0.583333, 0.625000, 0.666667,
            0.208333, 0.250000, 0.291667, 0.333333, 0.708333, 0.750000, 0.791667, 0.833333,
            0.375000, 0.416667, 0.458333, 0.500000, 0.875000, 0.916667, 0.958333, 1.000000,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn trilinear_upsample_matches_composed_bilinear_against_independent_oracle() -> Result<()> {
    // Closes part of `Upsample`'s "partial" gap (docs/nn/catalog.md: "no
    // trilinear or bicubic"). `Tensor::resample_bilinear` already operates
    // on ONE named axis at a time via a fixed per-axis weight matrix
    // (`docs/design/library.md`); since 2D/3D linear interpolation is
    // separable, applying it once per spatial axis -- already the documented
    // pattern for 2D bilinear (`world/fluid`'s U-Net calls it on `height`
    // then `width`) -- composes exact trilinear interpolation for free when
    // applied to three axes, with no new code. Independent oracle: real
    // PyTorch 2.14 `F.interpolate(mode="trilinear", align_corners=False)`.
    // What remains open (not attempted here; a genuinely new interpolation
    // kernel, not a composition): bicubic, and a literal `scale_factor=`
    // spelling (`resample_bilinear` already takes an explicit output
    // extent, which every real consumer computes itself).
    let device = Device::cuda(0)?;
    let (depth, height, width) = (Axis::new("depth"), Axis::new("height"), Axis::new("width"));
    let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let input =
        Tensor::from_slice(&values, [depth.of(2), height.of(2), width.of(3)], &device)?.with_grad();

    let out = input
        .resample_bilinear(depth, 4)?
        .resample_bilinear(height, 3)?
        .resample_bilinear(width, 5)?;
    assert_eq!(
        out.shape(),
        &Shape::new([depth.of(4), height.of(3), width.of(5)])?
    );
    close(
        "trilinear-by-composed-bilinear forward (depth2 height2 width3 -> depth4 height3 width5)",
        &out.to_vec()?,
        &[
            0.0, 0.4, 1.0, 1.6, 2.0, 1.5, 1.9, 2.5, 3.1, 3.5, 3.0, 3.4, 4.0, 4.6, 5.0, 1.5, 1.9,
            2.5, 3.1, 3.5, 3.0, 3.4, 4.0, 4.6, 5.0, 4.5, 4.9, 5.5, 6.1, 6.5, 4.5, 4.9, 5.5, 6.1,
            6.5, 6.0, 6.4, 7.0, 7.6, 8.0, 7.5, 7.9, 8.5, 9.1, 9.5, 6.0, 6.4, 7.0, 7.6, 8.0, 7.5,
            7.900001, 8.5, 9.1, 9.5, 9.0, 9.400001, 10.0, 10.6, 11.0,
        ],
    );
    let w: Vec<f32> = (1..=60).map(|v| v as f32).collect();
    let weight = Tensor::from_slice(&w, [depth.of(4), height.of(3), width.of(5)], &device)?;
    out.mul(&weight)?.mean([depth, height, width])?.backward()?;
    close(
        "trilinear-by-composed-bilinear gradient",
        &input.grad().unwrap().to_vec()?,
        &[
            0.993333, 1.263750, 1.253334, 1.526667, 1.863750, 1.786667, 3.093333, 3.626250,
            3.353334, 3.626667, 4.226250, 3.886667,
        ],
    );
    Ok(())
}

#[test]
fn containers_have_no_single_forward_and_reject_duplicate_module_keys() -> Result<()> {
    let shape = Shape::new([Axis::new("feature").of(3)])?;

    let mut list = ModuleList::new(vec![Box::new(ReLU), Box::new(Tanh)]);
    assert_eq!(list.len(), 2);
    assert!(!list.is_empty());
    assert!(list.output_shape(&shape).is_err());
    assert_eq!(list.get(0).unwrap().output_shape(&shape)?, shape);
    assert!(list.get_mut(1).is_some());
    list.push(Box::new(SiLU));
    assert_eq!(list.len(), 3);
    assert_eq!(list.iter().count(), 3);
    assert!(list.get(5).is_none());

    let entries: Vec<(String, Box<dyn Module>)> =
        vec![("a".into(), Box::new(ReLU)), ("b".into(), Box::new(Tanh))];
    let mut dict = ModuleDict::new(entries)?;
    assert_eq!(dict.len(), 2);
    assert!(dict.output_shape(&shape).is_err());
    assert_eq!(dict.get("a").unwrap().output_shape(&shape)?, shape);
    assert!(
        ModuleDict::new(vec![
            ("dup".into(), Box::new(ReLU)),
            ("dup".into(), Box::new(Tanh))
        ])
        .is_err()
    );
    assert!(dict.insert("a", Box::new(SiLU)).is_err());
    dict.insert("c", Box::new(SiLU))?;
    assert_eq!(dict.len(), 3);
    assert!(dict.get("c").is_some());
    assert!(dict.get("missing").is_none());
    Ok(())
}

#[test]
fn flatten_and_unflatten_shapes_invert_and_reject_bad_configuration() -> Result<()> {
    let (sample, row, col, merged, bogus) = (
        Axis::new("sample"),
        Axis::new("row"),
        Axis::new("col"),
        Axis::new("merged"),
        Axis::new("bogus"),
    );
    let input = Shape::new([sample.of(2), row.of(2), col.of(3)])?;

    let flatten = Flatten::new([col, row], merged);
    assert_eq!(
        flatten.output_shape(&input)?,
        Shape::new([sample.of(2), merged.of(6)])?
    );
    assert!(
        Flatten::new(Vec::<Axis>::new(), merged)
            .output_shape(&input)
            .is_err()
    );
    assert!(
        Flatten::new([col, bogus], merged)
            .output_shape(&input)
            .is_err()
    );
    assert!(
        Flatten::new([col, col], merged)
            .output_shape(&input)
            .is_err()
    );

    let packed = Shape::new([sample.of(2), merged.of(6)])?;
    let unflatten = Unflatten::new(merged, [col.of(3), row.of(2)]);
    assert_eq!(
        unflatten.output_shape(&packed)?,
        Shape::new([sample.of(2), col.of(3), row.of(2)])?
    );
    assert!(
        Unflatten::new(merged, [col.of(4), row.of(2)])
            .output_shape(&packed)
            .is_err()
    );
    assert!(
        Unflatten::new(bogus, [col.of(3), row.of(2)])
            .output_shape(&packed)
            .is_err()
    );

    assert_eq!(Identity.output_shape(&input)?, input);
    Ok(())
}

#[test]
fn bilinear_and_local_response_norm_reject_invalid_configuration_before_allocation() -> Result<()> {
    let (batch, in1, in2, output, bogus) = (
        Axis::new("batch"),
        Axis::new("in1"),
        Axis::new("in2"),
        Axis::new("output"),
        Axis::new("bogus"),
    );
    let x1 = Shape::new([batch.of(2), in1.of(2)])?;
    let x2 = Shape::new([batch.of(2), in2.of(2)])?;
    let bilinear = Bilinear::new(in1, in2, output.of(2));
    assert_eq!(
        bilinear.output_shape(&x1, &x2)?,
        Shape::new([batch.of(2), output.of(2)])?
    );
    assert!(
        bilinear
            .output_shape(&Shape::new([batch.of(2), bogus.of(2)])?, &x2)
            .is_err()
    );
    let mismatched_batch = Shape::new([batch.of(3), in2.of(2)])?;
    assert!(bilinear.output_shape(&x1, &mismatched_batch).is_err());

    let channel = Axis::new("channel");
    assert!(LocalResponseNorm::new(channel, 0).is_err());
    let mut norm = LocalResponseNorm::new(channel, 4)?;
    assert!(norm.output_shape(&x1).is_err());
    let signal = Shape::new([channel.of(4)])?;
    assert_eq!(norm.output_shape(&signal)?, signal);
    assert!(LocalResponseNorm::new(channel, 4)?.alpha(f32::NAN).is_err());
    assert!(
        LocalResponseNorm::new(channel, 4)?
            .beta(f32::INFINITY)
            .is_err()
    );
    assert!(
        LocalResponseNorm::new(channel, 4)?
            .k(f32::NEG_INFINITY)
            .is_err()
    );
    norm = norm.alpha(0.5)?.beta(0.5)?.k(1.0)?;
    assert_eq!(norm.output_shape(&signal)?, signal);
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn flatten_and_unflatten_match_tensor_merge_and_split_forward_and_gradient() -> Result<()> {
    // Merge order [col, row] is the reverse of the input's own physical storage order
    // ([sample, row, col], `col` fastest), so `Flatten` must force a real permutation
    // through `Tensor::merge`'s own `with_layout` rather than reinterpret storage in place.
    let device = Device::cuda(0)?;
    let (sample, row, col, merged) = (
        Axis::new("sample"),
        Axis::new("row"),
        Axis::new("col"),
        Axis::new("merged"),
    );
    let values = [
        0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 100.0, 101.0, 102.0, 110.0, 111.0, 112.0,
    ];
    let input =
        Tensor::from_slice(&values, [sample.of(2), row.of(2), col.of(3)], &device)?.with_grad();
    let mut flatten = Flatten::new([col, row], merged);
    let expected_shape = Shape::new([sample.of(2), merged.of(6)])?;
    assert_eq!(flatten.build(input.shape(), &device, 0)?, expected_shape);
    let flattened = flatten.forward(&input)?;
    close(
        "Flatten forward under a reversed merge order",
        &flattened.to_vec()?,
        &[
            0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 100.0, 110.0, 101.0, 111.0, 102.0, 112.0,
        ],
    );

    let weights = Tensor::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        [sample.of(2), merged.of(6)],
        &device,
    )?;
    flattened
        .mul(&weights)?
        .mean([sample, merged])?
        .backward()?;
    close(
        "Flatten gradient under a reversed merge order",
        &input.grad().unwrap().to_vec()?,
        &[
            1.0 / 12.0,
            3.0 / 12.0,
            5.0 / 12.0,
            2.0 / 12.0,
            4.0 / 12.0,
            6.0 / 12.0,
            7.0 / 12.0,
            9.0 / 12.0,
            11.0 / 12.0,
            8.0 / 12.0,
            10.0 / 12.0,
            12.0 / 12.0,
        ],
    );

    let merged_values = [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    let packed =
        Tensor::from_slice(&merged_values, [sample.of(2), merged.of(6)], &device)?.with_grad();
    let mut unflatten = Unflatten::new(merged, [col.of(3), row.of(2)]);
    let split_shape = Shape::new([sample.of(2), col.of(3), row.of(2)])?;
    assert_eq!(unflatten.build(packed.shape(), &device, 0)?, split_shape);
    let split = unflatten.forward(&packed)?;
    close(
        "Unflatten forward is a pure reshape (identical physical values)",
        &split.to_vec()?,
        &merged_values
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );

    let split_weights = Tensor::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        [sample.of(2), col.of(3), row.of(2)],
        &device,
    )?;
    split
        .mul(&split_weights)?
        .mean([sample, col, row])?
        .backward()?;
    close(
        "Unflatten gradient reshapes back onto the source axis",
        &packed.grad().unwrap().to_vec()?,
        &[
            1.0 / 12.0,
            2.0 / 12.0,
            3.0 / 12.0,
            4.0 / 12.0,
            5.0 / 12.0,
            6.0 / 12.0,
            7.0 / 12.0,
            8.0 / 12.0,
            9.0 / 12.0,
            10.0 / 12.0,
            11.0 / 12.0,
            12.0 / 12.0,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn identity_module_passes_through_values_and_gradients_unchanged() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("feature");
    let input = Tensor::from_slice(&[1.0, -2.0, 3.5], [feature.of(3)], &device)?.with_grad();
    let mut identity = Identity;
    assert_eq!(
        identity.build(input.shape(), &device, 0)?,
        input.shape().clone()
    );
    let output = identity.forward(&input)?;
    close("Identity forward", &output.to_vec()?, &[1.0, -2.0, 3.5]);
    let weights = Tensor::from_slice(&[2.0, 3.0, 4.0], [feature.of(3)], &device)?;
    output.mul(&weights)?.mean([feature])?.backward()?;
    close(
        "Identity gradient",
        &input.grad().unwrap().to_vec()?,
        &[2.0 / 3.0, 3.0 / 3.0, 4.0 / 3.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn bilinear_matches_hand_computed_quadratic_form_forward_and_every_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, in1, in2, output) = (
        Axis::new("batch"),
        Axis::new("in1"),
        Axis::new("in2"),
        Axis::new("output"),
    );
    let x1 =
        Tensor::from_slice(&[1.0, 2.0, 3.0, -1.0], [batch.of(2), in1.of(2)], &device)?.with_grad();
    let x2 =
        Tensor::from_slice(&[0.5, -2.0, 1.0, 2.0], [batch.of(2), in2.of(2)], &device)?.with_grad();

    let mut bilinear = Bilinear::new(in1, in2, output.of(2));
    let expected_shape = Shape::new([batch.of(2), output.of(2)])?;
    assert_eq!(
        bilinear.build(x1.shape(), x2.shape(), &device, 0)?,
        expected_shape
    );
    bilinear
        .named_parameters()
        .iter()
        .find(|(name, _)| name == "weight")
        .unwrap()
        .1
        .set_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    bilinear
        .named_parameters()
        .iter()
        .find(|(name, _)| name == "bias")
        .unwrap()
        .1
        .set_values(&[0.1, -0.2])?;

    let y = bilinear.forward(&x1, &x2)?;
    close("Bilinear forward", &y.to_vec()?, &[-28.4, -33.2, 2.1, 7.8]);

    // Reordered physical storage for both operands must not change the forward value: same
    // logical [batch, in*] shapes and values as `x1`/`x2`, transposed in physical storage.
    let x1_reordered =
        Tensor::from_slice(&[1.0, 2.0, 3.0, -1.0], [batch.of(2), in1.of(2)], &device)?
            .with_layout([in1, batch])?;
    let x2_reordered =
        Tensor::from_slice(&[0.5, -2.0, 1.0, 2.0], [batch.of(2), in2.of(2)], &device)?
            .with_layout([in2, batch])?;
    close(
        "Bilinear forward is unaffected by reordered operand storage",
        &bilinear.forward(&x1_reordered, &x2_reordered)?.to_vec()?,
        &[-28.4, -33.2, 2.1, 7.8],
    );

    let weight_tensor =
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [batch.of(2), output.of(2)], &device)?;
    y.mul(&weight_tensor)?.mean([batch, output])?.backward()?;
    close(
        "Bilinear bias gradient",
        &bilinear
            .named_parameters()
            .iter()
            .find(|(name, _)| name == "bias")
            .unwrap()
            .1
            .grad()
            .unwrap()
            .to_vec()?,
        &[1.0, 1.5],
    );
    close(
        "Bilinear weight gradient",
        &bilinear
            .named_parameters()
            .iter()
            .find(|(name, _)| name == "weight")
            .unwrap()
            .1
            .grad()
            .unwrap()
            .to_vec()?,
        &[2.375, 3.25, 4.0, 5.0, -0.5, -0.5, -2.5, -4.0],
    );
    close(
        "Bilinear x1 gradient",
        &x1.grad().unwrap().to_vec()?,
        &[-4.875, -9.375, 15.25, 36.25],
    );
    close(
        "Bilinear x2 gradient",
        &x2.grad().unwrap().to_vec()?,
        &[9.75, 14.25, -1.5, 5.5],
    );

    assert!(
        Bilinear::new(in1, in2, output.of(2))
            .forward(&x1, &x2)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn embedding_bag_sum_mean_max_match_hand_computed_pooling_oracle() -> Result<()> {
    // Asymmetric extents throughout (vocabulary 5, feature 3, 3 bags of sizes 3/0/2) so no
    // dimension coincidentally hides a transposition bug. Bag 1 is deliberately empty: `Sum`
    // and `Mean` must read it back as an exact zero row, and `Max` as `NaN` (documented, a
    // deliberate divergence from PyTorch's zero-filled empty bag for that one mode).
    let device = Device::cuda(0)?;
    let (vocabulary, feature, bag) = (
        Axis::new("vocabulary"),
        Axis::new("feature"),
        Axis::new("bag"),
    );
    let table_values: Vec<f32> = (1..=15).map(|v| v as f32).collect();
    let index = [0usize, 2, 4, 1, 3];
    let offsets = [0usize, 3, 3];

    let mut sum_bag = EmbeddingBag::new(vocabulary.of(5), feature.of(3), EmbeddingBagMode::Sum);
    sum_bag.build(&device, 0)?;
    sum_bag.named_parameters()[0].1.set_values(&table_values)?;
    let sum_output = sum_bag.forward(&index, &offsets, bag)?;
    close(
        "EmbeddingBag Sum forward",
        &sum_output.to_vec()?,
        &[21.0, 24.0, 27.0, 0.0, 0.0, 0.0, 14.0, 16.0, 18.0],
    );
    let sum_weight = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0],
        [bag.of(3), feature.of(3)],
        &device,
    )?;
    sum_output
        .mul(&sum_weight)?
        .mean([bag, feature])?
        .backward()?;
    close(
        "EmbeddingBag Sum table gradient",
        &sum_bag.named_parameters()[0].1.grad().unwrap().to_vec()?,
        &[
            1.0 / 9.0,
            2.0 / 9.0,
            3.0 / 9.0,
            4.0 / 9.0,
            5.0 / 9.0,
            6.0 / 9.0,
            1.0 / 9.0,
            2.0 / 9.0,
            3.0 / 9.0,
            4.0 / 9.0,
            5.0 / 9.0,
            6.0 / 9.0,
            1.0 / 9.0,
            2.0 / 9.0,
            3.0 / 9.0,
        ],
    );

    let mut mean_bag = EmbeddingBag::new(vocabulary.of(5), feature.of(3), EmbeddingBagMode::Mean);
    mean_bag.build(&device, 0)?;
    mean_bag.named_parameters()[0].1.set_values(&table_values)?;
    let mean_output = mean_bag.forward(&index, &offsets, bag)?;
    close(
        "EmbeddingBag Mean forward (empty bag reads back as an exact zero row)",
        &mean_output.to_vec()?,
        &[7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0],
    );
    mean_output
        .mul(&sum_weight)?
        .mean([bag, feature])?
        .backward()?;
    close(
        "EmbeddingBag Mean table gradient",
        &mean_bag.named_parameters()[0].1.grad().unwrap().to_vec()?,
        &[
            1.0 / 27.0,
            2.0 / 27.0,
            3.0 / 27.0,
            2.0 / 9.0,
            2.5 / 9.0,
            3.0 / 9.0,
            1.0 / 27.0,
            2.0 / 27.0,
            3.0 / 27.0,
            2.0 / 9.0,
            2.5 / 9.0,
            3.0 / 9.0,
            1.0 / 27.0,
            2.0 / 27.0,
            3.0 / 27.0,
        ],
    );

    let mut max_bag = EmbeddingBag::new(vocabulary.of(5), feature.of(3), EmbeddingBagMode::Max);
    max_bag.build(&device, 0)?;
    max_bag.named_parameters()[0].1.set_values(&table_values)?;
    let max_output = max_bag.forward(&index, &offsets, bag)?.detach();
    let max_values = max_output.to_vec()?;
    close(
        "EmbeddingBag Max forward, bag 0",
        &max_values[0..3],
        &[13.0, 14.0, 15.0],
    );
    assert!(max_values[3..6].iter().all(|v| v.is_nan()));
    close(
        "EmbeddingBag Max forward, bag 2",
        &max_values[6..9],
        &[10.0, 11.0, 12.0],
    );

    // A separate, entirely nonempty two-bag configuration isolates the gradient check from the
    // empty bag's `NaN`: `max`'s own "no finite candidate" rule would otherwise poison a
    // reduction that touches it at all, even multiplied by a zero weight (`NaN * 0.0 = NaN`).
    let two_bag_offsets = [0usize, 3];
    let max_output_two = max_bag.forward(&index, &two_bag_offsets, bag)?;
    let max_weight = Tensor::from_slice(
        &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        [bag.of(2), feature.of(3)],
        &device,
    )?;
    max_output_two
        .mul(&max_weight)?
        .mean([bag, feature])?
        .backward()?;
    close(
        "EmbeddingBag Max table gradient (nonempty bags only)",
        &max_bag.named_parameters()[0].1.grad().unwrap().to_vec()?,
        &[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
        ],
    );

    assert!(sum_bag.forward(&[], &offsets, bag).is_err());
    assert!(sum_bag.forward(&index, &[], bag).is_err());
    assert!(sum_bag.forward(&index, &[1, 0], bag).is_err());
    assert!(sum_bag.forward(&index, &[0, 2, 10], bag).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn local_response_norm_matches_hand_computed_oracle_under_asymmetric_window() -> Result<()> {
    // size = 4 is even, so PyTorch's own `size // 2` / `(size - 1) // 2` split is asymmetric
    // (2 channels before, 1 after); oracle values are independent central differences.
    let device = Device::cuda(0)?;
    let (batch, channel) = (Axis::new("batch"), Axis::new("channel"));
    let input = Tensor::from_slice(
        &[1.0, -2.0, 0.5, 3.0, 2.0, 1.0, -1.5, 0.5],
        [batch.of(2), channel.of(4)],
        &device,
    )?
    .with_grad();
    let mut norm = LocalResponseNorm::new(channel, 4)?
        .alpha(0.5)?
        .beta(0.5)?
        .k(1.0)?;
    assert_eq!(
        norm.build(input.shape(), &device, 0)?,
        input.shape().clone()
    );
    let output = norm.forward(&input)?;
    close(
        "LocalResponseNorm forward under an asymmetric window",
        &output.to_vec()?,
        &[
            0.7844645405527362,
            -1.5540573797716226,
            0.29981267559834457,
            1.8407159732336889,
            1.5689290811054724,
            0.7242859683401482,
            -1.0776318121606494,
            0.41702882811414954,
        ],
    );

    // Reordered physical storage (channel-major instead of batch-major) must not change it.
    let reordered = Tensor::from_slice(
        &[1.0, -2.0, 0.5, 3.0, 2.0, 1.0, -1.5, 0.5],
        [batch.of(2), channel.of(4)],
        &device,
    )?
    .with_layout([channel, batch])?;
    close(
        "LocalResponseNorm forward is unaffected by reordered channel storage",
        &norm.forward(&reordered)?.to_vec()?,
        &[
            0.7844645405527362,
            -1.5540573797716226,
            0.29981267559834457,
            1.8407159732336889,
            1.5689290811054724,
            0.7242859683401482,
            -1.0776318121606494,
            0.41702882811414954,
        ],
    );

    let weights = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5],
        [batch.of(2), channel.of(4)],
        &device,
    )?;
    output.mul(&weights)?.mean([batch, channel])?.backward()?;
    close(
        "LocalResponseNorm gradient under an asymmetric window",
        &input.grad().unwrap().to_vec()?,
        &[
            0.11478395397890306,
            0.24742732767868425,
            0.21533843197807379,
            0.1616940354942642,
            0.05958576218323408,
            0.12521675608834215,
            0.22907252950454815,
            0.3678308347854209,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn module_list_and_module_dict_named_parameters_stay_addressable_like_sequential() -> Result<()> {
    // A representative composition: two parallel branches held in a container instead of a
    // chain, each built and run directly (`ModuleList`/`ModuleDict` define no forward order of
    // their own), summed by the caller -- exactly the multi-branch pattern PyTorch users reach
    // for a `ModuleList`/`ModuleDict` over a plain `Vec`/`HashMap` for: parameter registration.
    let device = Device::cuda(0)?;
    let (input, output) = (Axis::new("input"), Axis::new("output"));
    let x = Tensor::from_slice(&[3.0, 5.0], [input.of(2)], &device)?.with_grad();
    let input_shape = Shape::new([input.of(2)])?;

    let mut list = ModuleList::new(vec![
        Box::new(Linear::new(input, output.of(2)).bias(false)) as Box<dyn Module>,
        Box::new(Linear::new(input, output.of(2)).bias(false)) as Box<dyn Module>,
    ]);
    list.get_mut(0).unwrap().build(&input_shape, &device, 0)?;
    list.get_mut(1).unwrap().build(&input_shape, &device, 0)?;
    list.get_mut(0)
        .unwrap()
        .named_parameters()
        .iter()
        .find(|(name, _)| name == "weight")
        .unwrap()
        .1
        .set_values(&[1.0, 0.0, 0.0, 1.0])?;
    list.get_mut(1)
        .unwrap()
        .named_parameters()
        .iter()
        .find(|(name, _)| name == "weight")
        .unwrap()
        .1
        .set_values(&[0.0, 1.0, 1.0, 0.0])?;
    let paths: Vec<_> = list
        .named_parameters()
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    assert_eq!(paths, vec!["0.weight".to_string(), "1.weight".to_string()]);

    let branch0 = list.get(0).unwrap().forward(&x)?;
    let branch1 = list.get(1).unwrap().forward(&x)?;
    let sum = branch0.add(&branch1)?;
    close("ModuleList branch sum forward", &sum.to_vec()?, &[8.0, 8.0]);
    sum.mean([output])?.backward()?;
    close(
        "ModuleList branch 0 weight gradient",
        &list
            .get(0)
            .unwrap()
            .parameter("weight")?
            .grad()
            .unwrap()
            .to_vec()?,
        &[1.5, 1.5, 2.5, 2.5],
    );
    close(
        "ModuleList branch 1 weight gradient",
        &list
            .get(1)
            .unwrap()
            .parameter("weight")?
            .grad()
            .unwrap()
            .to_vec()?,
        &[1.5, 1.5, 2.5, 2.5],
    );

    let mut dict = ModuleDict::new(vec![
        (
            "even".to_string(),
            Box::new(Linear::new(input, output.of(2)).bias(false)) as Box<dyn Module>,
        ),
        (
            "odd".to_string(),
            Box::new(Linear::new(input, output.of(2)).bias(false)) as Box<dyn Module>,
        ),
    ])?;
    dict.get_mut("even")
        .unwrap()
        .build(&input_shape, &device, 0)?;
    dict.get_mut("odd")
        .unwrap()
        .build(&input_shape, &device, 0)?;
    let dict_paths: Vec<_> = dict
        .named_parameters()
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    assert_eq!(
        dict_paths,
        vec!["even.weight".to_string(), "odd.weight".to_string()]
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn parameter_list_and_parameter_dict_named_parameters_stay_addressable() -> Result<()> {
    // A representative composition: standalone parameters combined directly by the caller
    // (PyTorch's own use case for `ParameterList`/`ParameterDict` over a bare `Vec`/`HashMap`
    // of tensors -- automatic registration for the optimizer to find).
    let device = Device::cuda(0)?;
    let feature = Axis::new("feature");
    let p0 = Parameter::new(Tensor::from_slice(&[1.0, 2.0], [feature.of(2)], &device)?);
    let p1 = Parameter::new(Tensor::from_slice(&[3.0, 4.0], [feature.of(2)], &device)?);

    let list = ParameterList::new(vec![p0.clone(), p1.clone()]);
    assert_eq!(list.len(), 2);
    assert_eq!(list.get(0).unwrap().id(), p0.id());
    let list_paths: Vec<_> = list
        .named_parameters()
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    assert_eq!(list_paths, vec!["0".to_string(), "1".to_string()]);

    let residual = p0.tensor().add(&p1.tensor())?;
    residual.mean([feature])?.backward()?;
    close(
        "ParameterList p0 gradient",
        &p0.grad().unwrap().to_vec()?,
        &[0.5, 0.5],
    );
    close(
        "ParameterList p1 gradient",
        &p1.grad().unwrap().to_vec()?,
        &[0.5, 0.5],
    );

    let mut dict = ParameterDict::new(vec![("a".to_string(), p0.clone())])?;
    assert!(dict.insert("a", p1.clone()).is_err());
    dict.insert("b", p1.clone())?;
    assert_eq!(dict.get("a").unwrap().id(), p0.id());
    assert_eq!(dict.get("b").unwrap().id(), p1.id());
    assert!(dict.get("missing").is_none());
    assert!(
        ParameterDict::new(vec![
            ("dup".to_string(), p0.clone()),
            ("dup".to_string(), p1.clone())
        ])
        .is_err()
    );
    Ok(())
}

#[test]
fn pooling_family_rejects_invalid_configuration_before_launch() {
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let shape = Shape::new([channel.of(2), height.of(4), width.of(4)]).unwrap();
    let line = Shape::new([channel.of(2), height.of(5)]).unwrap();

    // MaxPool1d/AvgPool1d reuse `Pooling<2>`'s own geometry through the lift, so the same
    // padding/positive/fit checks apply through one spatial axis.
    let error = MaxPool1d::new(channel, height, 2)
        .padding(3)
        .output_shape(&line)
        .unwrap_err()
        .to_string();
    assert!(error.contains("padding must be at most half"), "{error}");
    let error = AvgPool1d::new(channel, height, 0)
        .output_shape(&line)
        .unwrap_err()
        .to_string();
    assert!(error.contains("positive"), "{error}");

    // AvgPool2d/AvgPool3d share `MaxPool2d`/`MaxPool3d`'s own padding and distinct-axis checks.
    let error = AvgPool2d::new(channel, [height, width], [2, 2])
        .padding([3, 0])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("padding must be at most half"), "{error}");
    let error = AvgPool2d::new(channel, [channel, width], [2, 2])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("distinct"), "{error}");

    // LPPool rejects a non-positive `p` before any shape is even consulted.
    assert!(LPPool1d::new(channel, height, 0.0, 2).is_err());
    assert!(LPPool2d::new(channel, [height, width], 0.0, [2, 2]).is_err());
    assert!(LPPool3d::new(channel, [height, width, channel.role("d")], 0.0, [2, 2, 2]).is_err());
    let pool = LPPool2d::new(channel, [height, width], 2.0, [2, 2]).unwrap();
    assert_eq!(
        pool.output_shape(&shape).unwrap(),
        Shape::new([height.of(2), width.of(2), channel.of(2)]).unwrap()
    );

    // Adaptive pooling rejects a zero target and a repeated spatial axis before launch, and
    // otherwise reports the expected pooled shape purely from shape math (no device needed).
    let error = AdaptiveAvgPool1d::new(height, 0)
        .output_shape(&line)
        .unwrap_err()
        .to_string();
    assert!(error.contains("positive"), "{error}");
    let error = AdaptiveMaxPool2d::new([height, height], [2, 2])
        .output_shape(&shape)
        .unwrap_err()
        .to_string();
    assert!(error.contains("distinct"), "{error}");
    let adaptive = AdaptiveAvgPool2d::new([height, width], [1, 1]);
    assert_eq!(
        adaptive.output_shape(&shape).unwrap(),
        Shape::new([channel.of(2), height.of(1), width.of(1)]).unwrap()
    );
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool2d_asymmetric_kernel_matches_hand_computed_forward_and_gradient() -> Result<()> {
    // The existing MaxPool2d oracles both use square kernels; this closes that gap with a
    // genuinely non-square kernel/stride (height 2, width 3), auditing MaxPool2d's own CUDA
    // coverage against the module-backlog's "asymmetric geometry" bar.
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    #[rustfmt::skip]
    let inputs: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];
    let input = Tensor::from_slice(&inputs, [channel.of(1), height.of(3), width.of(4)], &device)?
        .with_grad();

    let mut pool = MaxPool2d::new(channel, [height, width], [2, 3]).stride([1, 1]);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        output_shape,
        Shape::new([height.of(2), width.of(2), channel.of(1)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "asymmetric-kernel MaxPool2d forward",
        &actual.to_vec()?,
        &[7.0, 8.0, 11.0, 12.0],
    );

    actual.mean([channel, height, width])?.backward()?;
    #[rustfmt::skip]
    let expected_gradient: [f64; 12] = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.25, 0.25,
        0.0, 0.0, 0.25, 0.25,
    ];
    close(
        "asymmetric-kernel MaxPool2d gradient",
        &input.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool3d_matches_hand_computed_forward_and_gradient_under_reordered_storage() -> Result<()> {
    // The existing MaxPool3d oracle never reorders physical storage; this closes that gap
    // (`max_pool2d_with_padding...` already covers reordering for the 2D case). Two channels
    // pool independently and pick different winning positions, so independence survives the
    // reorder too.
    let device = Device::cuda(0)?;
    let (channel, depth, height, width) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
    );
    #[rustfmt::skip]
    let inputs: [f32; 8] = [
        1.0, 2.0, 3.0, 4.0,
        -1.0, -2.0, -3.0, -4.0,
    ];
    let input = Tensor::from_slice(
        &inputs,
        [channel.of(2), depth.of(1), height.of(2), width.of(2)],
        &device,
    )?
    .with_layout([width, height, depth, channel])?
    .with_grad();

    let mut pool = MaxPool3d::new(channel, [depth, height, width], [1, 2, 2]);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        output_shape,
        Shape::new([depth.of(1), height.of(1), width.of(1), channel.of(2)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "reordered-storage MaxPool3d forward",
        &actual.to_vec()?,
        &[4.0, -1.0],
    );

    actual.mean([channel, depth, height, width])?.backward()?;
    close(
        "reordered-storage MaxPool3d gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool1d_matches_hand_computed_forward_and_gradient_with_padding() -> Result<()> {
    // `MaxPool1d` has no dedicated unfold kernel; it lifts through `MaxPool2d`'s own two-axis
    // machinery via a broadcast unit axis. Padding, overlap (stride < kernel), and negative
    // infinity fill all round-trip through the lift correctly.
    let device = Device::cuda(0)?;
    let (channel, length) = (Axis::new("channel"), Axis::new("length"));
    let inputs: [f32; 5] = [3.0, -1.0, 5.0, 2.0, -4.0];
    let input = Tensor::from_slice(&inputs, [channel.of(1), length.of(5)], &device)?.with_grad();

    let mut pool = MaxPool1d::new(channel, length, 3).stride(1).padding(1);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(output_shape, Shape::new([length.of(5), channel.of(1)])?);
    let actual = pool.forward(&input)?;
    close(
        "MaxPool1d forward",
        &actual.to_vec()?,
        &[3.0, 5.0, 5.0, 5.0, 2.0],
    );

    actual.mean([channel, length])?.backward()?;
    close(
        "MaxPool1d gradient (overlapping windows sum onto their shared source)",
        &input.grad().unwrap().to_vec()?,
        &[0.2, 0.0, 0.6, 0.2, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn avg_pool1d_matches_hand_computed_forward_and_gradient_with_padding() -> Result<()> {
    // `count_include_pad=True` (PyTorch's default): every window divides by the full kernel
    // extent (3), never by the count of real positions, so the edge windows here divide by 3
    // even though one of their three positions is padding.
    let device = Device::cuda(0)?;
    let (channel, length) = (Axis::new("channel"), Axis::new("length"));
    let inputs: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_slice(&inputs, [channel.of(1), length.of(4)], &device)?.with_grad();

    let mut pool = AvgPool1d::new(channel, length, 3).stride(1).padding(1);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(output_shape, Shape::new([length.of(4), channel.of(1)])?);
    let actual = pool.forward(&input)?;
    close(
        "AvgPool1d forward with count_include_pad=True",
        &actual.to_vec()?,
        &[1.0, 2.0, 3.0, 2.3333333333333335],
    );

    actual.mean([channel, length])?.backward()?;
    close(
        "AvgPool1d gradient",
        &input.grad().unwrap().to_vec()?,
        &[1.0 / 6.0, 0.25, 0.25, 1.0 / 6.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn avg_pool2d_matches_hand_computed_forward_and_gradient_with_asymmetric_kernel_and_reordered_storage()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let inputs: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_slice(&inputs, [channel.of(1), height.of(2), width.of(3)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();

    let mut pool = AvgPool2d::new(channel, [height, width], [1, 2]).stride([1, 1]);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        output_shape,
        Shape::new([height.of(2), width.of(2), channel.of(1)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "reordered-storage asymmetric-kernel AvgPool2d forward",
        &actual.to_vec()?,
        &[1.5, 2.5, 4.5, 5.5],
    );

    actual.mean([channel, height, width])?.backward()?;
    close(
        "reordered-storage asymmetric-kernel AvgPool2d gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.125, 0.25, 0.125, 0.125, 0.25, 0.125],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn avg_pool3d_matches_hand_computed_forward_and_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, depth, height, width) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
    );
    #[rustfmt::skip]
    let inputs: [f32; 8] = [
        1.0, 2.0, 3.0, 4.0,
        10.0, 20.0, 30.0, 40.0,
    ];
    let input = Tensor::from_slice(
        &inputs,
        [channel.of(1), depth.of(2), height.of(2), width.of(2)],
        &device,
    )?
    .with_grad();

    let mut pool = AvgPool3d::new(channel, [depth, height, width], [2, 1, 1]);
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        output_shape,
        Shape::new([depth.of(1), height.of(2), width.of(2), channel.of(1)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "AvgPool3d forward",
        &actual.to_vec()?,
        &[5.5, 11.0, 16.5, 22.0],
    );

    actual.mean([channel, depth, height, width])?.backward()?;
    close(
        "AvgPool3d gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.125; 8],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lp_pool1d_matches_sum_pooling_at_p_one() -> Result<()> {
    // At `p = 1`, `(sum(x^1))^(1/1)` is exactly sum pooling; negative values exercise the
    // sign-guarded root (`sign(sum) * |sum|^(1/p)`) at its simplest case, where it must reduce
    // to the identity.
    let device = Device::cuda(0)?;
    let (channel, length) = (Axis::new("channel"), Axis::new("length"));
    let inputs: [f32; 4] = [3.0, -5.0, 2.0, -1.0];
    let input = Tensor::from_slice(&inputs, [channel.of(1), length.of(4)], &device)?.with_grad();

    let mut pool = LPPool1d::new(channel, length, 1.0, 2)?;
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(output_shape, Shape::new([length.of(2), channel.of(1)])?);
    let actual = pool.forward(&input)?;
    close("LPPool1d(p=1) forward", &actual.to_vec()?, &[-2.0, 1.0]);

    actual.mean([channel, length])?.backward()?;
    close(
        "LPPool1d(p=1) gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.5, 0.5, 0.5, 0.5],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lp_pool2d_matches_hand_computed_l2_forward_and_gradient_under_reordered_storage() -> Result<()> {
    // `p = 2`: `sqrt(sum(x^2))`, the ordinary L2 norm, whose gradient `x_i / y` is an
    // independent closed form distinct from the op's own composition.
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let inputs: [f32; 4] = [3.0, -4.0, 0.0, 0.0];
    let input = Tensor::from_slice(&inputs, [channel.of(1), height.of(2), width.of(2)], &device)?
        .with_layout([width, height, channel])?
        .with_grad();

    let mut pool = LPPool2d::new(channel, [height, width], 2.0, [2, 2])?;
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        output_shape,
        Shape::new([height.of(1), width.of(1), channel.of(1)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "reordered-storage LPPool2d(p=2) forward",
        &actual.to_vec()?,
        &[5.0],
    );

    actual.mean([channel, height, width])?.backward()?;
    close(
        "reordered-storage LPPool2d(p=2) gradient (x_i / ||x||_2)",
        &input.grad().unwrap().to_vec()?,
        &[0.6, -0.8, 0.0, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lp_pool3d_matches_hand_computed_forward_and_gradient_with_negative_sum() -> Result<()> {
    // `p = 3` (odd) over an all-negative window leaves `sum(x^3)` negative, exercising the
    // `sign(sum)` branch for real: a literal `(-10.0).powf(1.0 / 3.0)` would be `NaN`.
    let device = Device::cuda(0)?;
    let (channel, depth, height, width) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let inputs: [f32; 3] = [-1.0, -2.0, -1.0];
    let input = Tensor::from_slice(
        &inputs,
        [channel.of(1), depth.of(1), height.of(1), width.of(3)],
        &device,
    )?
    .with_grad();

    let mut pool = LPPool3d::new(channel, [depth, height, width], 3.0, [1, 1, 3])?;
    let output_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        output_shape,
        Shape::new([depth.of(1), height.of(1), width.of(1), channel.of(1)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "LPPool3d(p=3) forward",
        &actual.to_vec()?,
        &[-2.154434690031884],
    );

    actual.mean([channel, depth, height, width])?.backward()?;
    close(
        "LPPool3d(p=3) gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.2154434690031884, 0.8617738760127536, 0.2154434690031884],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_avg_pool1d_and_2d_match_hand_computed_uneven_bins() -> Result<()> {
    // Generalizes `adaptive_avg_pool3d`'s own uneven-bin oracle down to one and two spatial
    // axes, over the same private bin/weighted-sum machinery.
    let device = Device::cuda(0)?;
    let length = Axis::new("length");
    let input =
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], [length.of(5)], &device)?.with_grad();
    let actual = input.adaptive_avg_pool1d(length, 2)?;
    assert_eq!(actual.shape(), &Shape::new([length.of(2)])?);
    close(
        "AdaptiveAvgPool1d forward with an uneven bin",
        &actual.to_vec()?,
        &[2.0, 4.0],
    );
    actual.mean(length)?.backward()?;
    close(
        "AdaptiveAvgPool1d gradient (the shared boundary element is averaged into both bins)",
        &input.grad().unwrap().to_vec()?,
        &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    );

    let (height, width) = (Axis::new("height"), Axis::new("width"));
    #[rustfmt::skip]
    let grid: [f32; 15] = [
        1.0, 1.0, 5.0,
        2.0, 2.0, 1.0,
        9.0, 3.0, 2.0,
        3.0, 4.0, 3.0,
        5.0, 9.0, 4.0,
    ];
    let input2d = Tensor::from_slice(&grid, [height.of(5), width.of(3)], &device)?.with_grad();
    let actual2d = input2d.adaptive_avg_pool2d([height, width], [2, 3])?;
    assert_eq!(actual2d.shape(), &Shape::new([height.of(2), width.of(3)])?);
    close(
        "AdaptiveAvgPool2d forward with an uneven height bin, identity width",
        &actual2d.to_vec()?,
        &[
            4.0,
            2.0,
            2.6666666666666665,
            5.666666666666667,
            5.333333333333333,
            3.0,
        ],
    );
    actual2d.mean([height, width])?.backward()?;
    #[rustfmt::skip]
    let expected_gradient2d: [f64; 15] = [
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    ];
    close(
        "AdaptiveAvgPool2d gradient",
        &input2d.grad().unwrap().to_vec()?,
        &expected_gradient2d,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_max_pool1d_matches_hand_computed_forward_and_gradient_with_overlapping_bins()
-> Result<()> {
    // `in = 7, out = 3` gives windows [0,3), [2,5), [4,7): index 2 is the unique maximum of
    // both the first two bins, so `gather`'s scatter-add backward must accumulate its
    // gradient from both, exactly like PyTorch's autograd summing a value's two downstream
    // uses.
    let device = Device::cuda(0)?;
    let length = Axis::new("length");
    let inputs: [f32; 7] = [1.0, 2.0, 9.0, 3.0, 5.0, 1.0, 2.0];
    let input = Tensor::from_slice(&inputs, [length.of(7)], &device)?.with_grad();

    let actual = input.adaptive_max_pool1d(length, 3)?;
    assert_eq!(actual.shape(), &Shape::new([length.of(3)])?);
    close(
        "AdaptiveMaxPool1d forward with overlapping bins",
        &actual.to_vec()?,
        &[9.0, 9.0, 5.0],
    );

    actual.mean(length)?.backward()?;
    close(
        "AdaptiveMaxPool1d gradient (the shared winner accumulates both bins' contributions)",
        &input.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 2.0 / 3.0, 0.0, 1.0 / 3.0, 0.0, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_max_pool2d_matches_hand_computed_forward_and_gradient_over_both_axes() -> Result<()> {
    // A genuine 5x5 -> 2x2 reduction over both spatial axes at once (not one axis held
    // trivial), checking that reducing height then width separably reproduces the joint
    // maximum, including a source shared between two output cells across BOTH axes.
    let device = Device::cuda(0)?;
    let (height, width) = (Axis::new("height"), Axis::new("width"));
    #[rustfmt::skip]
    let inputs: [f32; 25] = [
        -3.0, -1.99, -0.98, 0.03, 1.04,
        2.05, 3.06, -2.93, -1.92, -0.91,
        0.1, 1.11, 2.12, 3.13, -2.86,
        -1.85, -0.84, 0.17, 1.18, 2.19,
        3.2, -2.79, -1.78, -0.77, 0.24,
    ];
    let input = Tensor::from_slice(&inputs, [height.of(5), width.of(5)], &device)?.with_grad();

    let actual = input.adaptive_max_pool2d([height, width], [2, 2])?;
    assert_eq!(actual.shape(), &Shape::new([height.of(2), width.of(2)])?);
    close(
        "AdaptiveMaxPool2d forward over both axes",
        &actual.to_vec()?,
        &[3.06, 3.13, 3.2, 3.13],
    );

    actual.mean([height, width])?.backward()?;
    #[rustfmt::skip]
    let expected_gradient: [f64; 25] = [
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.25, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.25, 0.0, 0.0, 0.0, 0.0,
    ];
    close(
        "AdaptiveMaxPool2d gradient (a source shared across both axes accumulates twice)",
        &input.grad().unwrap().to_vec()?,
        &expected_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_max_pool3d_matches_hand_computed_forward_and_gradient_under_reordered_storage()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (depth, height, width) = (Axis::new("depth"), Axis::new("height"), Axis::new("width"));
    #[rustfmt::skip]
    let inputs: [f32; 10] = [
        1.0, 2.0, 9.0, 3.0, 5.0,
        10.0, 20.0, 90.0, 30.0, 50.0,
    ];
    let input = Tensor::from_slice(&inputs, [depth.of(2), height.of(5), width.of(1)], &device)?
        .with_layout([width, height, depth])?
        .with_grad();

    let actual = input.adaptive_max_pool3d([depth, height, width], [1, 2, 1])?;
    assert_eq!(
        actual.shape(),
        &Shape::new([depth.of(1), height.of(2), width.of(1)])?
    );
    close(
        "reordered-storage AdaptiveMaxPool3d forward",
        &actual.to_vec()?,
        &[90.0, 90.0],
    );

    actual.mean([depth, height, width])?.backward()?;
    // Depth 1 wins the depth reduction at every height, then also wins both height bins at
    // height 2, so it alone carries the whole gradient; depth 0's own height-2 value (9) never
    // wins the depth step and gets none.
    close(
        "reordered-storage AdaptiveMaxPool3d gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_max_pool_ties_route_to_first_maximum() -> Result<()> {
    // `in = 4, out = 2`: bin 0 is an exact tie between indices 0 and 1. Ties must break to the
    // first (lowest-coordinate) maximum, matching `Tensor::max`'s own documented rule, since
    // `adaptive_max_pool` composes it directly.
    let device = Device::cuda(0)?;
    let length = Axis::new("length");
    let input = Tensor::from_slice(&[5.0, 5.0, 1.0, 2.0], [length.of(4)], &device)?.with_grad();

    let actual = input.adaptive_max_pool1d(length, 2)?;
    close(
        "AdaptiveMaxPool1d tie forward",
        &actual.to_vec()?,
        &[5.0, 2.0],
    );

    actual.mean(length)?.backward()?;
    close(
        "AdaptiveMaxPool1d tie gradient (routes to the first logical coordinate)",
        &input.grad().unwrap().to_vec()?,
        &[0.5, 0.0, 0.0, 0.5],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_avg_pool2d_global_head_composes_with_linear() -> Result<()> {
    // The consumer for a common primitive can be a representative composition
    // (`docs/direction/module-backlog.md`'s admission rule): `AdaptiveAvgPool2d(1)` is exactly
    // the global-pool head every sampled vision port uses ahead of a classifier `Linear`
    // (`morpheus/mobilesam`'s `bootstrap2_common.py`, `gastric`'s `train_serosal_3d.py`), here
    // as a `Module` end to end: build, forward, and backward through both layers together.
    let device = Device::cuda(0)?;
    let (channel, height, width, feature) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("feature"),
    );
    let inputs: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let input = Tensor::from_slice(&inputs, [channel.of(3), height.of(2), width.of(2)], &device)?
        .with_grad();

    let mut pool = AdaptiveAvgPool2d::new([height, width], [1, 1]);
    let pooled_shape = pool.build(input.shape(), &device, 0)?;
    assert_eq!(
        pooled_shape,
        Shape::new([channel.of(3), height.of(1), width.of(1)])?
    );
    let pooled = pool
        .forward(&input)?
        .merge([channel, height, width], feature)?;

    let mut head = Linear::new(feature, feature.role("out").of(1));
    head.build(pooled.shape(), &device, 1)?;
    let logit = head.forward(&pooled)?;
    assert_eq!(logit.shape().rank(), 1);
    logit.mean(logit.shape().axes())?.backward()?;
    assert!(input.grad().is_some());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn conv1d_matches_scalar_oracle_and_reuses_conv2d_exactly() -> Result<()> {
    // groups=1, asymmetric stride/padding relative to kernel, under reordered storage: proves
    // the unit-spatial-axis composition over Conv2d is exact, not merely close.
    const CHANNELS: usize = 3;
    const LEN: usize = 5;
    const OUT_CHANNELS: usize = 4;
    const KERNEL: usize = 3;
    const STRIDE: usize = 2;
    const PADDING: usize = 1;
    const OUT_LEN: usize = 3;

    let device = Device::cuda(0)?;
    let (batch, channel, length, output) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("length"),
        Axis::new("output"),
    );
    let inputs: Vec<_> = (0..CHANNELS * LEN)
        .map(|i| (i as f32 - 7.0) / 5.0)
        .collect();
    let weights: Vec<_> = (0..CHANNELS * KERNEL * OUT_CHANNELS)
        .map(|i| ((i * 5 % 23) as f32 - 11.0) / 13.0)
        .collect();
    let biases: Vec<_> = (0..OUT_CHANNELS).map(|i| (i as f32 - 1.5) / 7.0).collect();
    let input = Tensor::from_slice(
        &inputs,
        [batch.of(1), channel.of(CHANNELS), length.of(LEN)],
        &device,
    )?
    .with_layout([length, channel, batch])?
    .with_grad();
    let mut conv = Conv1d::new(channel, output.of(OUT_CHANNELS), length, KERNEL)
        .stride(STRIDE)
        .padding(PADDING);
    assert_eq!(
        conv.build(input.shape(), &device, 11)?,
        Shape::new([batch.of(1), length.of(OUT_LEN), output.of(OUT_CHANNELS)])?
    );
    conv.parameter("weight")?.set_values(&weights)?;
    conv.parameter("bias")?.set_values(&biases)?;

    let mut expected = vec![0.0_f64; OUT_LEN * OUT_CHANNELS];
    let mut input_gradient = vec![0.0_f64; inputs.len()];
    let mut weight_gradient = vec![0.0_f64; weights.len()];
    let mut bias_gradient = vec![0.0_f64; biases.len()];
    let upstream = 1.0 / expected.len() as f64;
    for o in 0..OUT_LEN {
        for oc in 0..OUT_CHANNELS {
            let mut value = f64::from(biases[oc]);
            bias_gradient[oc] += upstream;
            for ic in 0..CHANNELS {
                for k in 0..KERNEL {
                    let padded = o * STRIDE + k;
                    let Some(i) = padded.checked_sub(PADDING) else {
                        continue;
                    };
                    if i >= LEN {
                        continue;
                    }
                    let input_index = ic * LEN + i;
                    let patch = ic * KERNEL + k;
                    let weight_index = patch * OUT_CHANNELS + oc;
                    value += f64::from(inputs[input_index]) * f64::from(weights[weight_index]);
                    input_gradient[input_index] += upstream * f64::from(weights[weight_index]);
                    weight_gradient[weight_index] += upstream * f64::from(inputs[input_index]);
                }
            }
            expected[o * OUT_CHANNELS + oc] = value;
        }
    }

    let actual = conv.forward(&input)?;
    close("Conv1d forward", &actual.to_vec()?, &expected);
    actual.mean([batch, length, output])?.backward()?;
    close(
        "Conv1d input gradient",
        &input.grad().unwrap().to_vec()?,
        &input_gradient,
    );
    close(
        "Conv1d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &weight_gradient,
    );
    close(
        "Conv1d bias gradient",
        &conv.parameter("bias")?.grad().unwrap().to_vec()?,
        &bias_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn conv_transpose2d_matches_scalar_forward_and_all_gradients_under_reordered_storage() -> Result<()>
{
    // Groups, asymmetric stride/padding/output_padding, and storage reordered relative to its
    // declared axis order: exercises the split -> contract -> fold_grouped -> pad_zeros -> add
    // composition end to end, including Rule::Fold's own backward (a forward unfold gather).
    const IN_CHANNELS: usize = 4;
    const IN_H: usize = 3;
    const IN_W: usize = 3;
    const OUT_CHANNELS: usize = 6;
    const GROUPS: usize = 2;
    const KERNEL: [usize; 2] = [2, 2];
    const STRIDE: [usize; 2] = [2, 2];
    const PADDING: [usize; 2] = [1, 1];
    const OUTPUT_PADDING: [usize; 2] = [1, 0];
    const IN_PER_GROUP: usize = IN_CHANNELS / GROUPS;
    const OUT_PER_GROUP: usize = OUT_CHANNELS / GROUPS;
    const PATCH: usize = OUT_PER_GROUP * KERNEL[0] * KERNEL[1];
    const CORE_H: usize = (IN_H - 1) * STRIDE[0] + KERNEL[0] - 2 * PADDING[0];
    const CORE_W: usize = (IN_W - 1) * STRIDE[1] + KERNEL[1] - 2 * PADDING[1];
    const OUT_H: usize = CORE_H + OUTPUT_PADDING[0];
    const OUT_W: usize = CORE_W + OUTPUT_PADDING[1];

    let device = Device::cuda(0)?;
    let (batch, channel, height, width, output) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
    );
    let inputs: Vec<_> = (0..IN_CHANNELS * IN_H * IN_W)
        .map(|i| (i as f32 - 17.0) / 11.0)
        .collect();
    let weights: Vec<_> = (0..GROUPS * PATCH * IN_PER_GROUP)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 13.0)
        .collect();
    let biases: Vec<_> = (0..OUT_CHANNELS).map(|i| (i as f32 - 2.5) / 9.0).collect();
    let input = Tensor::from_slice(
        &inputs,
        [
            batch.of(1),
            channel.of(IN_CHANNELS),
            height.of(IN_H),
            width.of(IN_W),
        ],
        &device,
    )?
    .with_layout([width, batch, channel, height])?
    .with_grad();
    let mut conv = ConvTranspose2d::new(channel, output.of(OUT_CHANNELS), [height, width], KERNEL)
        .stride(STRIDE)
        .padding(PADDING)
        .output_padding(OUTPUT_PADDING)
        .groups(GROUPS);
    assert_eq!(
        conv.build(input.shape(), &device, 29)?,
        Shape::new([
            batch.of(1),
            height.of(OUT_H),
            width.of(OUT_W),
            output.of(OUT_CHANNELS),
        ])?
    );
    conv.parameter("weight")?.set_values(&weights)?;
    conv.parameter("bias")?.set_values(&biases)?;

    let mut expected = vec![0.0_f64; OUT_H * OUT_W * OUT_CHANNELS];
    let mut input_gradient = vec![0.0_f64; inputs.len()];
    let mut weight_gradient = vec![0.0_f64; weights.len()];
    let mut bias_gradient = vec![0.0_f64; biases.len()];
    // Bias applies over the whole (including output-padding) output, matching PyTorch's own
    // `output = conv_transpose_no_bias + bias`.
    let element_count = expected.len() as f64;
    let upstream = 1.0 / element_count;
    for gradient in bias_gradient.iter_mut() {
        *gradient += upstream * (OUT_H * OUT_W) as f64;
    }
    for group in 0..GROUPS {
        for ic_in_group in 0..IN_PER_GROUP {
            let ic = group * IN_PER_GROUP + ic_in_group;
            for iy in 0..IN_H {
                for ix in 0..IN_W {
                    let input_index = (ic * IN_H + iy) * IN_W + ix;
                    for ky in 0..KERNEL[0] {
                        for kx in 0..KERNEL[1] {
                            let Some(oy) = (iy * STRIDE[0] + ky).checked_sub(PADDING[0]) else {
                                continue;
                            };
                            let Some(ox) = (ix * STRIDE[1] + kx).checked_sub(PADDING[1]) else {
                                continue;
                            };
                            if oy >= CORE_H || ox >= CORE_W {
                                continue;
                            }
                            for oc_in_group in 0..OUT_PER_GROUP {
                                let oc = group * OUT_PER_GROUP + oc_in_group;
                                let patch_index =
                                    oc_in_group * (KERNEL[0] * KERNEL[1]) + ky * KERNEL[1] + kx;
                                let weight_index =
                                    (group * PATCH + patch_index) * IN_PER_GROUP + ic_in_group;
                                let output_index = (oy * OUT_W + ox) * OUT_CHANNELS + oc;
                                expected[output_index] += f64::from(inputs[input_index])
                                    * f64::from(weights[weight_index]);
                                input_gradient[input_index] +=
                                    upstream * f64::from(weights[weight_index]);
                                weight_gradient[weight_index] +=
                                    upstream * f64::from(inputs[input_index]);
                            }
                        }
                    }
                }
            }
        }
    }
    for oy in 0..OUT_H {
        for ox in 0..OUT_W {
            for oc in 0..OUT_CHANNELS {
                expected[(oy * OUT_W + ox) * OUT_CHANNELS + oc] += f64::from(biases[oc]);
            }
        }
    }

    let actual = conv.forward(&input)?;
    close(
        "ConvTranspose2d forward under reordered storage",
        &actual.to_vec()?,
        &expected,
    );
    actual.mean([batch, height, width, output])?.backward()?;
    close(
        "ConvTranspose2d input gradient",
        &input.grad().unwrap().to_vec()?,
        &input_gradient,
    );
    close(
        "ConvTranspose2d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &weight_gradient,
    );
    close(
        "ConvTranspose2d bias gradient",
        &conv.parameter("bias")?.grad().unwrap().to_vec()?,
        &bias_gradient,
    );

    let error = ConvTranspose2d::new(channel, output.of(OUT_CHANNELS), [height, width], KERNEL)
        .output_padding([STRIDE[0], 0])
        .build(input.shape(), &device, 5)
        .unwrap_err()
        .to_string();
    assert!(error.contains("output_padding"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn conv_transpose1d_matches_scalar_oracle_and_reuses_conv_transpose2d_exactly() -> Result<()> {
    const IN_CHANNELS: usize = 2;
    const LEN: usize = 3;
    const OUT_CHANNELS: usize = 3;
    const KERNEL: usize = 2;
    const STRIDE: usize = 2;
    const PADDING: usize = 0;
    const OUTPUT_PADDING: usize = 1;
    const PATCH: usize = OUT_CHANNELS * KERNEL;
    const CORE_LEN: usize = (LEN - 1) * STRIDE + KERNEL - 2 * PADDING;
    const OUT_LEN: usize = CORE_LEN + OUTPUT_PADDING;

    let device = Device::cuda(0)?;
    let (batch, channel, length, output) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("length"),
        Axis::new("output"),
    );
    let inputs: Vec<_> = (0..IN_CHANNELS * LEN)
        .map(|i| (i as f32 - 2.5) / 3.0)
        .collect();
    let weights: Vec<_> = (0..PATCH * IN_CHANNELS)
        .map(|i| ((i * 3 % 17) as f32 - 8.0) / 9.0)
        .collect();
    let biases: Vec<_> = (0..OUT_CHANNELS).map(|i| (i as f32 - 1.0) / 5.0).collect();
    let input = Tensor::from_slice(
        &inputs,
        [batch.of(1), channel.of(IN_CHANNELS), length.of(LEN)],
        &device,
    )?
    .with_grad();
    let mut conv = ConvTranspose1d::new(channel, output.of(OUT_CHANNELS), length, KERNEL)
        .stride(STRIDE)
        .padding(PADDING)
        .output_padding(OUTPUT_PADDING);
    assert_eq!(
        conv.build(input.shape(), &device, 41)?,
        Shape::new([batch.of(1), length.of(OUT_LEN), output.of(OUT_CHANNELS)])?
    );
    conv.parameter("weight")?.set_values(&weights)?;
    conv.parameter("bias")?.set_values(&biases)?;

    let mut expected = vec![0.0_f64; OUT_LEN * OUT_CHANNELS];
    for o in 0..OUT_LEN {
        for oc in 0..OUT_CHANNELS {
            expected[o * OUT_CHANNELS + oc] = f64::from(biases[oc]);
        }
    }
    let mut input_gradient = vec![0.0_f64; inputs.len()];
    let mut weight_gradient = vec![0.0_f64; weights.len()];
    let mut bias_gradient = vec![0.0_f64; biases.len()];
    let upstream = 1.0 / expected.len() as f64;
    for gradient in bias_gradient.iter_mut() {
        *gradient += upstream * OUT_LEN as f64;
    }
    for ic in 0..IN_CHANNELS {
        for i in 0..LEN {
            let input_index = ic * LEN + i;
            for k in 0..KERNEL {
                let Some(o) = (i * STRIDE + k).checked_sub(PADDING) else {
                    continue;
                };
                if o >= CORE_LEN {
                    continue;
                }
                for oc in 0..OUT_CHANNELS {
                    let patch_index = oc * KERNEL + k;
                    let weight_index = patch_index * IN_CHANNELS + ic;
                    let output_index = o * OUT_CHANNELS + oc;
                    expected[output_index] +=
                        f64::from(inputs[input_index]) * f64::from(weights[weight_index]);
                    input_gradient[input_index] += upstream * f64::from(weights[weight_index]);
                    weight_gradient[weight_index] += upstream * f64::from(inputs[input_index]);
                }
            }
        }
    }

    let actual = conv.forward(&input)?;
    close("ConvTranspose1d forward", &actual.to_vec()?, &expected);
    actual.mean([batch, length, output])?.backward()?;
    close(
        "ConvTranspose1d input gradient",
        &input.grad().unwrap().to_vec()?,
        &input_gradient,
    );
    close(
        "ConvTranspose1d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &weight_gradient,
    );
    close(
        "ConvTranspose1d bias gradient",
        &conv.parameter("bias")?.grad().unwrap().to_vec()?,
        &bias_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn conv_transpose3d_matches_scalar_forward_and_all_gradients() -> Result<()> {
    const IN_CHANNELS: usize = 2;
    const DEPTH: usize = 2;
    const HEIGHT: usize = 2;
    const WIDTH: usize = 2;
    const OUT_CHANNELS: usize = 3;
    const KERNEL: [usize; 3] = [2, 2, 2];
    const STRIDE: [usize; 3] = [2, 2, 2];
    const PADDING: [usize; 3] = [0, 0, 0];
    const PATCH: usize = OUT_CHANNELS * KERNEL[0] * KERNEL[1] * KERNEL[2];
    const OUT_D: usize = (DEPTH - 1) * STRIDE[0] + KERNEL[0];
    const OUT_H: usize = (HEIGHT - 1) * STRIDE[1] + KERNEL[1];
    const OUT_W: usize = (WIDTH - 1) * STRIDE[2] + KERNEL[2];

    let device = Device::cuda(0)?;
    let (batch, channel, depth, height, width, output) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("output"),
    );
    let inputs: Vec<_> = (0..IN_CHANNELS * DEPTH * HEIGHT * WIDTH)
        .map(|i| (i as f32 - 7.0) / 6.0)
        .collect();
    let weights: Vec<_> = (0..PATCH * IN_CHANNELS)
        .map(|i| ((i * 5 % 19) as f32 - 9.0) / 11.0)
        .collect();
    let biases: Vec<_> = (0..OUT_CHANNELS).map(|i| (i as f32 - 1.0) / 4.0).collect();
    let input = Tensor::from_slice(
        &inputs,
        [
            batch.of(1),
            channel.of(IN_CHANNELS),
            depth.of(DEPTH),
            height.of(HEIGHT),
            width.of(WIDTH),
        ],
        &device,
    )?
    .with_grad();
    let mut conv = ConvTranspose3d::new(
        channel,
        output.of(OUT_CHANNELS),
        [depth, height, width],
        KERNEL,
    )
    .stride(STRIDE)
    .padding(PADDING);
    assert_eq!(
        conv.build(input.shape(), &device, 53)?,
        Shape::new([
            batch.of(1),
            depth.of(OUT_D),
            height.of(OUT_H),
            width.of(OUT_W),
            output.of(OUT_CHANNELS),
        ])?
    );
    conv.parameter("weight")?.set_values(&weights)?;
    conv.parameter("bias")?.set_values(&biases)?;

    let mut expected = vec![0.0_f64; OUT_D * OUT_H * OUT_W * OUT_CHANNELS];
    let mut input_gradient = vec![0.0_f64; inputs.len()];
    let mut weight_gradient = vec![0.0_f64; weights.len()];
    let mut bias_gradient = vec![0.0_f64; biases.len()];
    let upstream = 1.0 / (OUT_D * OUT_H * OUT_W * OUT_CHANNELS) as f64;
    for od in 0..OUT_D {
        for oh in 0..OUT_H {
            for ow in 0..OUT_W {
                for oc in 0..OUT_CHANNELS {
                    bias_gradient[oc] += upstream;
                    expected[((od * OUT_H + oh) * OUT_W + ow) * OUT_CHANNELS + oc] +=
                        f64::from(biases[oc]);
                }
            }
        }
    }
    for ic in 0..IN_CHANNELS {
        for id in 0..DEPTH {
            for ih in 0..HEIGHT {
                for iw in 0..WIDTH {
                    let input_index = ((ic * DEPTH + id) * HEIGHT + ih) * WIDTH + iw;
                    for kd in 0..KERNEL[0] {
                        for kh in 0..KERNEL[1] {
                            for kw in 0..KERNEL[2] {
                                let od = id * STRIDE[0] + kd;
                                let oh = ih * STRIDE[1] + kh;
                                let ow = iw * STRIDE[2] + kw;
                                for oc in 0..OUT_CHANNELS {
                                    let patch_index = oc * (KERNEL[0] * KERNEL[1] * KERNEL[2])
                                        + (kd * KERNEL[1] + kh) * KERNEL[2]
                                        + kw;
                                    let weight_index = patch_index * IN_CHANNELS + ic;
                                    let output_index =
                                        ((od * OUT_H + oh) * OUT_W + ow) * OUT_CHANNELS + oc;
                                    expected[output_index] += f64::from(inputs[input_index])
                                        * f64::from(weights[weight_index]);
                                    input_gradient[input_index] +=
                                        upstream * f64::from(weights[weight_index]);
                                    weight_gradient[weight_index] +=
                                        upstream * f64::from(inputs[input_index]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let actual = conv.forward(&input)?;
    close("ConvTranspose3d forward", &actual.to_vec()?, &expected);
    actual
        .mean([batch, depth, height, width, output])?
        .backward()?;
    close(
        "ConvTranspose3d input gradient",
        &input.grad().unwrap().to_vec()?,
        &input_gradient,
    );
    close(
        "ConvTranspose3d weight gradient",
        &conv.parameter("weight")?.grad().unwrap().to_vec()?,
        &weight_gradient,
    );
    close(
        "ConvTranspose3d bias gradient",
        &conv.parameter("bias")?.grad().unwrap().to_vec()?,
        &bias_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn unfold_module_matches_scalar_patches_and_gradient_under_reordered_storage() -> Result<()> {
    const CHANNELS: usize = 2;
    const HEIGHT: usize = 3;
    const WIDTH: usize = 4;
    const KERNEL: [usize; 2] = [2, 2];
    const STRIDE: [usize; 2] = [1, 2];
    const PADDING: [usize; 2] = [1, 0];
    const OUT_H: usize = (HEIGHT + 2 * PADDING[0] - KERNEL[0]) / STRIDE[0] + 1;
    const OUT_W: usize = (WIDTH + 2 * PADDING[1] - KERNEL[1]) / STRIDE[1] + 1;
    const PATCH: usize = CHANNELS * KERNEL[0] * KERNEL[1];

    let device = Device::cuda(0)?;
    let (batch, channel, height, width, patch) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("patch"),
    );
    let inputs: Vec<_> = (0..CHANNELS * HEIGHT * WIDTH)
        .map(|i| (i as f32 - 11.0) / 7.0)
        .collect();
    let input = Tensor::from_slice(
        &inputs,
        [
            batch.of(1),
            channel.of(CHANNELS),
            height.of(HEIGHT),
            width.of(WIDTH),
        ],
        &device,
    )?
    .with_layout([width, batch, channel, height])?
    .with_grad();
    let unfold = Unfold::new(channel, [height, width], patch, KERNEL)
        .stride(STRIDE)
        .padding(PADDING);
    assert_eq!(
        unfold.output_shape(input.shape())?,
        Shape::new([
            batch.of(1),
            patch.of(PATCH),
            height.of(OUT_H),
            width.of(OUT_W)
        ])?
    );

    let mut expected = vec![0.0_f64; OUT_H * OUT_W * PATCH];
    let mut input_gradient = vec![0.0_f64; inputs.len()];
    let upstream = 1.0 / expected.len() as f64;
    for oy in 0..OUT_H {
        for ox in 0..OUT_W {
            for c in 0..CHANNELS {
                for ky in 0..KERNEL[0] {
                    for kx in 0..KERNEL[1] {
                        let patch_index = (c * KERNEL[0] + ky) * KERNEL[1] + kx;
                        let output_index = (patch_index * OUT_H + oy) * OUT_W + ox;
                        let padded_y = oy * STRIDE[0] + ky;
                        let padded_x = ox * STRIDE[1] + kx;
                        let (Some(iy), Some(ix)) = (
                            padded_y.checked_sub(PADDING[0]),
                            padded_x.checked_sub(PADDING[1]),
                        ) else {
                            continue;
                        };
                        if iy >= HEIGHT || ix >= WIDTH {
                            continue;
                        }
                        let input_index = (c * HEIGHT + iy) * WIDTH + ix;
                        expected[output_index] = f64::from(inputs[input_index]);
                        input_gradient[input_index] += upstream;
                    }
                }
            }
        }
    }

    let actual = unfold.forward(&input)?;
    close(
        "Unfold forward under reordered storage",
        &actual.to_vec()?,
        &expected,
    );
    actual.mean([batch, height, width, patch])?.backward()?;
    close(
        "Unfold input gradient under reordered storage",
        &input.grad().unwrap().to_vec()?,
        &input_gradient,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn fold_module_matches_scalar_col2im_and_is_unfolds_adjoint() -> Result<()> {
    const CHANNELS: usize = 2;
    const OUT_H: usize = 4;
    const OUT_W: usize = 4;
    const KERNEL: [usize; 2] = [2, 2];
    const STRIDE: [usize; 2] = [2, 2];
    const PADDING: [usize; 2] = [0, 0];
    const IN_H: usize = (OUT_H - KERNEL[0]) / STRIDE[0] + 1;
    const IN_W: usize = (OUT_W - KERNEL[1]) / STRIDE[1] + 1;
    const PATCH: usize = CHANNELS * KERNEL[0] * KERNEL[1];

    let device = Device::cuda(0)?;
    let (batch, channel, height, width, patch) = (
        Axis::new("batch"),
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("patch"),
    );

    // Non-overlapping windows (stride == kernel): a real image folds and unfolds back to
    // itself exactly, the representative composition that motivates `Fold`/`Unfold` as a pair.
    let image_values: Vec<_> = (0..CHANNELS * OUT_H * OUT_W)
        .map(|i| (i as f32 - 15.0) / 9.0)
        .collect();
    let image = Tensor::from_slice(
        &image_values,
        [
            batch.of(1),
            channel.of(CHANNELS),
            height.of(OUT_H),
            width.of(OUT_W),
        ],
        &device,
    )?;
    let unfold = Unfold::new(channel, [height, width], patch, KERNEL)
        .stride(STRIDE)
        .padding(PADDING);
    let patches = unfold.forward(&image)?.with_grad();
    let fold = Fold::new(channel, [height, width], [OUT_H, OUT_W], patch, KERNEL)
        .stride(STRIDE)
        .padding(PADDING);
    assert_eq!(
        fold.output_shape(patches.shape())?,
        Shape::new([
            batch.of(1),
            channel.of(CHANNELS),
            height.of(OUT_H),
            width.of(OUT_W),
        ])?
    );
    let folded = fold.forward(&patches)?;
    close(
        "Fold(Unfold(x)) reconstructs x exactly under non-overlapping windows",
        &folded.to_vec()?,
        &image_values
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );

    // An independent scalar oracle for Fold's own forward (col2im: sum overlapping patch
    // contributions into the image) and its gradient (the adjoint: a plain `unfold` gather of
    // the incoming image gradient, `Rule::Fold`'s own backward path).
    let patch_values: Vec<_> = (0..IN_H * IN_W * PATCH)
        .map(|i| (i as f32 - 20.0) / 13.0)
        .collect();
    let patches_input = Tensor::from_slice(
        &patch_values,
        [
            batch.of(1),
            height.of(IN_H),
            width.of(IN_W),
            patch.of(PATCH),
        ],
        &device,
    )?
    .with_grad();
    let mut expected = vec![0.0_f64; CHANNELS * OUT_H * OUT_W];
    let mut patches_gradient = vec![0.0_f64; patch_values.len()];
    let element_count = (CHANNELS * OUT_H * OUT_W) as f64;
    for iy in 0..IN_H {
        for ix in 0..IN_W {
            for c in 0..CHANNELS {
                for ky in 0..KERNEL[0] {
                    for kx in 0..KERNEL[1] {
                        let oy = iy * STRIDE[0] + ky;
                        let ox = ix * STRIDE[1] + kx;
                        let patch_index = (c * KERNEL[0] + ky) * KERNEL[1] + kx;
                        let patches_index = (iy * IN_W + ix) * PATCH + patch_index;
                        let output_index = (c * OUT_H + oy) * OUT_W + ox;
                        expected[output_index] += f64::from(patch_values[patches_index]);
                        // upstream = mean over the image, so d(mean)/d(this image element) is
                        // 1/element_count; col2im sums exactly one contribution per (iy,ix,ky,kx)
                        // into a non-overlapping window here, so the adjoint gather is exact.
                        patches_gradient[patches_index] = 1.0 / element_count;
                    }
                }
            }
        }
    }
    let folded = fold.forward(&patches_input)?;
    close("Fold forward scalar oracle", &folded.to_vec()?, &expected);
    folded.mean([batch, channel, height, width])?.backward()?;
    close(
        "Fold input gradient scalar oracle",
        &patches_input.grad().unwrap().to_vec()?,
        &patches_gradient,
    );

    let mismatched = Tensor::from_slice(
        &[0.0; PATCH],
        [batch.of(1), height.of(1), width.of(1), patch.of(PATCH)],
        &device,
    )?;
    let error = Fold::new(channel, [height, width], [OUT_H, OUT_W], patch, KERNEL)
        .stride(STRIDE)
        .padding(PADDING)
        .output_shape(mismatched.shape())
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not match"), "{error}");
    Ok(())
}

// -- nn-climb-norms: PairwiseDistance/TripletMarginLoss general p-norm and eps placement,
// MultiMarginLoss p=2 and per-class weight, CosineSimilarity's joint squared-norm clamp fix,
// and ELU/CELU signed alpha. See `docs/design/library.md`'s "Norm orders and signed alpha". --

#[test]
#[ignore = "requires CUDA"]
fn pairwise_distance_supports_general_p_and_infinity_norm_matching_hand_computed_oracle()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("pdist_p_batch"), Axis::new("pdist_p_feature"));
    // Chosen so no diff-component lands near a norm's non-smooth point: p=1's |.| kink at 0,
    // and p=inf's max() tie when two components share a magnitude, both make central-difference
    // gradient checks spurious near those exact points, so every |diff| here is both bounded
    // away from zero and pairwise distinct within its row.
    let x1_values = [0.0_f64, 0.0, 0.0, 1.0, 2.0, 3.0];
    let x2_values = [3.0_f64, 4.0, 2.0, 0.0, 0.0, 0.0];
    let eps = 1e-6_f64;

    fn lp_row(a: &[f64], b: &[f64], eps: f64, p: f64) -> f64 {
        if p.is_infinite() {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y + eps).abs())
                .fold(0.0_f64, f64::max)
        } else {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y + eps).abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p)
        }
    }
    fn lp_scalar_loss(a: &[f64], b: &[f64], eps: f64, p: f64) -> f64 {
        (lp_row(&a[0..3], &b[0..3], eps, p) + lp_row(&a[3..6], &b[3..6], eps, p)) / 2.0
    }

    // p=1, p=3 (a non-special general exponent), and p -> the infinity-norm sentinel.
    for &p in &[1.0_f64, 3.0, f64::INFINITY] {
        let x1 = Tensor::from_slice(&to_f32(&x1_values), [batch.of(2), feature.of(3)], &device)?
            .with_grad();
        let x2 = Tensor::from_slice(&to_f32(&x2_values), [batch.of(2), feature.of(3)], &device)?
            .with_grad();
        let p32 = if p.is_infinite() {
            f32::INFINITY
        } else {
            p as f32
        };
        let distance = x1.pairwise_distance(&x2, feature, p32, eps as f32)?;
        let expected: Vec<f64> = (0..2)
            .map(|row| {
                lp_row(
                    &x1_values[row * 3..row * 3 + 3],
                    &x2_values[row * 3..row * 3 + 3],
                    eps,
                    p,
                )
            })
            .collect();
        close(
            &format!("pairwise distance forward (p={p})"),
            &distance.to_vec()?,
            &expected,
        );

        distance.mean(batch)?.backward()?;
        close(
            &format!("pairwise distance gradient wrt x1 (p={p})"),
            &x1.grad().unwrap().to_vec()?,
            &central_difference(&x1_values, 1e-4, |candidate| {
                lp_scalar_loss(candidate, &x2_values, eps, p)
            }),
        );
        close(
            &format!("pairwise distance gradient wrt x2 (p={p})"),
            &x2.grad().unwrap().to_vec()?,
            &central_difference(&x2_values, 1e-4, |candidate| {
                lp_scalar_loss(&x1_values, candidate, eps, p)
            }),
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn pairwise_distance_p_must_be_a_positive_real_or_infinite() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (
        Axis::new("pdist_p_err_batch"),
        Axis::new("pdist_p_err_feature"),
    );
    let x1 = Tensor::from_slice(&[1.0_f32, 2.0], [batch.of(1), feature.of(2)], &device)?;
    let x2 = Tensor::from_slice(&[0.0_f32, 0.0], [batch.of(1), feature.of(2)], &device)?;
    assert!(x1.pairwise_distance(&x2, feature, 0.0, 1e-6).is_err());
    assert!(x1.pairwise_distance(&x2, feature, -1.0, 1e-6).is_err());
    assert!(x1.pairwise_distance(&x2, feature, f32::NAN, 1e-6).is_err());
    assert!(
        x1.pairwise_distance(&x2, feature, f32::NEG_INFINITY, 1e-6)
            .is_err()
    );
    assert!(x1.pairwise_distance(&x2, feature, 1.0, 1e-6).is_ok());
    assert!(
        x1.pairwise_distance(&x2, feature, f32::INFINITY, 1e-6)
            .is_ok()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn triplet_margin_loss_p_generalizes_the_underlying_pairwise_distance() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("triplet_p_batch"), Axis::new("triplet_p_feature"));
    let anchor_values = [0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let positive_values = [0.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0];
    let negative_values = [0.0_f64, 0.0, 2.5, -1.0, -1.0, -1.0];
    let margin = 1.0_f32;
    let eps = 1e-6_f64;
    let p = 1.0_f64;

    fn l1_row(a: &[f64], b: &[f64], eps: f64) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y + eps).abs()).sum()
    }
    fn triplet_scalar_loss_p1(
        anchor: &[f64],
        positive: &[f64],
        negative: &[f64],
        margin: f64,
        eps: f64,
    ) -> f64 {
        (0..2)
            .map(|row| {
                let a = &anchor[row * 3..row * 3 + 3];
                let p = &positive[row * 3..row * 3 + 3];
                let n = &negative[row * 3..row * 3 + 3];
                (margin + l1_row(a, p, eps) - l1_row(a, n, eps)).max(0.0)
            })
            .sum::<f64>()
            / 2.0
    }

    let anchor = Tensor::from_slice(
        &to_f32(&anchor_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();
    let positive = Tensor::from_slice(
        &to_f32(&positive_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();
    let negative = Tensor::from_slice(
        &to_f32(&negative_values),
        [batch.of(2), feature.of(3)],
        &device,
    )?
    .with_grad();

    let loss = anchor.triplet_margin_loss(
        &positive, &negative, feature, margin, p as f32, eps as f32, false,
    )?;
    let expected: Vec<f64> = (0..2)
        .map(|row| {
            let a = &anchor_values[row * 3..row * 3 + 3];
            let pos = &positive_values[row * 3..row * 3 + 3];
            let neg = &negative_values[row * 3..row * 3 + 3];
            (f64::from(margin) + l1_row(a, pos, eps) - l1_row(a, neg, eps)).max(0.0)
        })
        .collect();
    close(
        "triplet margin loss forward (p=1)",
        &loss.to_vec()?,
        &expected,
    );

    loss.mean(batch)?.backward()?;
    close(
        "triplet margin loss gradient wrt anchor (p=1)",
        &anchor.grad().unwrap().to_vec()?,
        &central_difference(&anchor_values, 1e-4, |candidate| {
            triplet_scalar_loss_p1(
                candidate,
                &positive_values,
                &negative_values,
                f64::from(margin),
                eps,
            )
        }),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multi_margin_loss_supports_p2_and_per_class_weight_matching_hand_computed_oracle() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("mml_w_batch"), Axis::new("mml_w_class"));
    let x_values = [2.0_f64, 1.5, -1.0, 0.0, 2.5, 3.0];
    let target_values = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0];
    let weight_values = [2.0_f32, 1.0, 0.5];
    let margin = 1.0_f32;
    let p = 2.0_f32;
    let x =
        Tensor::from_slice(&to_f32(&x_values), [batch.of(2), class.of(3)], &device)?.with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), class.of(3)], &device)?;
    let weight = Tensor::from_slice(&weight_values, [class.of(3)], &device)?;

    let loss = x.multi_margin_loss(&target, class, p, margin, Some(&weight))?;
    close(
        "multi margin loss forward (p=2, weighted)",
        &loss.to_vec()?,
        &[1.0 / 6.0, 1.0 / 24.0],
    );

    fn weighted_multi_margin_scalar_loss(
        x: &[f64],
        t: &[f32],
        w: &[f64],
        margin: f64,
        p: f64,
    ) -> f64 {
        (0..2)
            .map(|row| {
                let xs = &x[row * 3..row * 3 + 3];
                let ts = &t[row * 3..row * 3 + 3];
                let y = ts.iter().position(|&v| v == 1.0).unwrap();
                let xy = xs[y];
                let total: f64 = (0..3)
                    .filter(|&i| i != y)
                    .map(|i| (margin - xy + xs[i]).max(0.0).powf(p))
                    .sum();
                w[y] * total / 3.0
            })
            .sum::<f64>()
            / 2.0
    }
    let weight_f64: Vec<f64> = weight_values.iter().map(|&v| f64::from(v)).collect();

    loss.mean(batch)?.backward()?;
    close(
        "multi margin loss gradient (p=2, weighted)",
        &x.grad().unwrap().to_vec()?,
        &central_difference(&x_values, 1e-4, |candidate| {
            weighted_multi_margin_scalar_loss(
                candidate,
                &target_values,
                &weight_f64,
                f64::from(margin),
                f64::from(p),
            )
        }),
    );

    assert!(
        x.detach()
            .multi_margin_loss(&target, class, 1.5, margin, None)
            .is_err()
    );
    let wrong_extent = Tensor::from_slice(&[1.0_f32, 1.0], [class.of(2)], &device)?;
    assert!(
        x.detach()
            .multi_margin_loss(&target, class, p, margin, Some(&wrong_extent))
            .is_err()
    );
    let wrong_axes = Tensor::from_slice(&[1.0_f32, 1.0, 1.0], [batch.of(3)], &device)?;
    assert!(
        x.detach()
            .multi_margin_loss(&target, class, p, margin, Some(&wrong_axes))
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn cosine_similarity_joint_clamp_matches_pytorch_and_diverges_from_the_old_per_vector_clamp()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (
        Axis::new("cos_sim_joint_batch"),
        Axis::new("cos_sim_joint_feature"),
    );
    // eps deliberately large (1e-3) to make the divergence visible with ordinary-magnitude
    // inputs. Row 0: x1's true norm (1e-5) sits far below eps while x2's (5.0) sits far above
    // it -- exactly the regime where the old per-vector clamp and the new joint-product clamp
    // disagree. Row 1: both norms sit comfortably above eps, where the two formulas already
    // agreed (recorded here so the fix is shown NOT to move that case).
    let eps = 1e-3_f32;
    let x1_values = [1e-5_f64, 0.0, 0.0, 3.0, 4.0, 0.0];
    let x2_values = [3.0_f64, 4.0, 0.0, 0.0, 3.0, 4.0];
    let x1 =
        Tensor::from_slice(&to_f32(&x1_values), [batch.of(2), feature.of(3)], &device)?.with_grad();
    let x2 =
        Tensor::from_slice(&to_f32(&x2_values), [batch.of(2), feature.of(3)], &device)?.with_grad();

    let similarity = x1.cosine_similarity(&x2, feature, eps)?;
    // Hand-computed (not from the op under test): row 0's joint-clamp value is 0.03, five
    // times the OLD per-vector-clamp formula's 0.006 -- the semantic change this PR makes.
    close(
        "cosine similarity forward (joint clamp)",
        &similarity.to_vec()?,
        &[0.03, 0.48],
    );

    fn cos_scalar_loss_joint_clamp(a: &[f64], b: &[f64], eps: f64) -> f64 {
        let row = |a: &[f64], b: &[f64]| {
            let dot: f64 = a.iter().zip(b).map(|(p, q)| p * q).sum();
            let na2: f64 = a.iter().map(|p| p * p).sum();
            let nb2: f64 = b.iter().map(|q| q * q).sum();
            let denom = (na2 * nb2).max(eps * eps).sqrt();
            dot / denom
        };
        (row(&a[0..3], &b[0..3]) + row(&a[3..6], &b[3..6])) / 2.0
    }

    similarity.mean(batch)?.backward()?;
    close(
        "cosine similarity gradient wrt x1 (joint clamp)",
        &x1.grad().unwrap().to_vec()?,
        &central_difference(&x1_values, 1e-5, |candidate| {
            cos_scalar_loss_joint_clamp(candidate, &x2_values, f64::from(eps))
        }),
    );
    close(
        "cosine similarity gradient wrt x2 (joint clamp)",
        &x2.grad().unwrap().to_vec()?,
        &central_difference(&x2_values, 1e-5, |candidate| {
            cos_scalar_loss_joint_clamp(&x1_values, candidate, f64::from(eps))
        }),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn elu_and_celu_support_negative_alpha_matching_hand_computed_oracle_under_reordered_storage()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (row, col) = (Axis::new("elu_neg_row"), Axis::new("elu_neg_col"));
    // 5 rows, 2 cols: asymmetric extents, feature-major physical storage forces a genuinely
    // permuted, non-contiguous read.
    let values = [-2.0_f64, 1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0, 2.0];
    let alpha = -0.5_f64;
    let n = values.len() as f64;

    fn elu(x: f64, alpha: f64) -> f64 {
        if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    }
    fn elu_grad(x: f64, alpha: f64) -> f64 {
        if x > 0.0 { 1.0 } else { alpha * x.exp() }
    }
    fn celu(x: f64, alpha: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            alpha * ((x / alpha).exp() - 1.0)
        }
    }
    fn celu_grad(x: f64, alpha: f64) -> f64 {
        if x > 0.0 { 1.0 } else { (x / alpha).exp() }
    }

    let elu_leaf = Tensor::from_slice(&to_f32(&values), [row.of(5), col.of(2)], &device)?
        .with_layout([col, row])?
        .with_grad();
    let elu_output = elu_leaf.elu(alpha as f32)?;
    let expected_elu: Vec<f64> = values.iter().map(|&x| elu(x, alpha)).collect();
    close(
        "elu forward (negative alpha, reordered storage)",
        &elu_output.to_vec()?,
        &expected_elu,
    );
    elu_output.mean([row, col])?.backward()?;
    let expected_elu_gradient: Vec<f64> = values.iter().map(|&x| elu_grad(x, alpha) / n).collect();
    close(
        "elu gradient (negative alpha, reordered storage)",
        &elu_leaf.grad().unwrap().to_vec()?,
        &expected_elu_gradient,
    );

    let celu_leaf = Tensor::from_slice(&to_f32(&values), [row.of(5), col.of(2)], &device)?
        .with_layout([col, row])?
        .with_grad();
    let celu_output = celu_leaf.celu(alpha as f32)?;
    let expected_celu: Vec<f64> = values.iter().map(|&x| celu(x, alpha)).collect();
    close(
        "celu forward (negative alpha, reordered storage)",
        &celu_output.to_vec()?,
        &expected_celu,
    );
    celu_output.mean([row, col])?.backward()?;
    let expected_celu_gradient: Vec<f64> =
        values.iter().map(|&x| celu_grad(x, alpha) / n).collect();
    close(
        "celu gradient (negative alpha, reordered storage)",
        &celu_leaf.grad().unwrap().to_vec()?,
        &expected_celu_gradient,
    );

    assert!(elu_leaf.detach().elu(0.0).is_err());
    assert!(celu_leaf.detach().celu(0.0).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn nll_loss_weighted_matches_hand_computed_oracle_with_weight_and_ignore_index() -> Result<()> {
    // `nll_loss_weighted` generalizes `nll_loss` with a per-class `weight` (multiplying each
    // class's term before the class axis is summed) and an `ignore_index` (zeroing a row in
    // proportion to its target mass on that class). Row 0 is one-hot at the ignored class and
    // is fully zeroed; row 1 has no mass there and is unaffected by `ignore_index`, only by
    // `weight`.
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("batch"), Axis::new("class"));
    let probabilities = [[0.7f64, 0.2, 0.1], [0.2, 0.3, 0.5]];
    let log_prob_values: Vec<f32> = probabilities
        .iter()
        .flat_map(|row| row.iter().map(|p| p.ln() as f32))
        .collect();
    let target_rows = [[1.0f64, 0.0, 0.0], [0.0, 0.5, 0.5]];
    let target_values: Vec<f32> = target_rows
        .iter()
        .flat_map(|row| row.iter().map(|&t| t as f32))
        .collect();
    let weight_values = [2.0f32, 0.5, 1.0];
    let ignore_index = 0usize;

    let expected_loss: Vec<f64> = probabilities
        .iter()
        .zip(target_rows)
        .map(|(probs, targets)| {
            let keep = 1.0 - targets[ignore_index];
            let raw: f64 = -probs
                .iter()
                .zip(targets)
                .zip(weight_values)
                .map(|((&p, t), w)| f64::from(w) * t * p.ln())
                .sum::<f64>();
            keep * raw
        })
        .collect();
    let n_rows = probabilities.len() as f64;
    let expected_grad: Vec<f64> = target_rows
        .iter()
        .flat_map(|row| {
            let keep = 1.0 - row[ignore_index];
            row.iter()
                .zip(weight_values)
                .map(move |(&t, w)| -keep * f64::from(w) * t / n_rows)
        })
        .collect();

    // value(b, c) written in canonical (batch, class) order; asymmetric extents (2, 3).
    let log_prob = Tensor::from_slice(&log_prob_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?
        .with_grad();
    let targets = Tensor::from_slice(&target_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?;
    let weight = Tensor::from_slice(&weight_values, [class.of(3)], &device)?;

    let loss = log_prob.nll_loss_weighted(&targets, class, &weight, Some(ignore_index))?;
    assert_eq!(loss.shape(), &Shape::new([batch.of(2)])?);
    close(
        "NLLLoss(weight, ignore_index) forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean(batch)?.backward()?;
    close(
        "NLLLoss(weight, ignore_index) gradient",
        &log_prob.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(
        log_prob
            .detach()
            .nll_loss_weighted(&targets.with_grad(), class, &weight, None)
            .is_err()
    );
    assert!(
        log_prob
            .detach()
            .nll_loss_weighted(&targets, class, &weight.with_grad(), None)
            .is_err()
    );
    let error = log_prob
        .detach()
        .nll_loss_weighted(&targets, class, &weight, Some(3))
        .err()
        .expect("out of range ignore_index")
        .to_string();
    assert!(error.contains("out of range"), "{error}");
    let other = Axis::new("other");
    let wrong_weight = Tensor::from_slice(&[1.0f32], [other.of(1)], &device)?;
    assert!(
        log_prob
            .detach()
            .nll_loss_weighted(&targets, class, &wrong_weight, None)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn kl_div_loss_log_target_matches_hand_computed_oracle() -> Result<()> {
    // `kl_div_loss_log_target` is `kl_div_loss` at PyTorch's `log_target = true`: `targets`
    // also holds log-probabilities, and the pointwise loss is `exp(target) * (target - self)`.
    // Unlike `log_target = false`, no `xlogy` zero-target substitution is needed, so this
    // oracle needs no zero-probability row.
    fn kl_term(log_q: f64, log_p: f64) -> f64 {
        log_p.exp() * (log_p - log_q)
    }
    let device = Device::cuda(0)?;
    let (batch, class) = (Axis::new("batch"), Axis::new("class"));
    // value(b, c) written in canonical (batch, class) order; asymmetric extents (2, 3).
    let log_q_values = [-0.5f32, -1.0, -2.0, -0.3, -0.9, -1.6];
    let p_values = [0.5f64, 0.3, 0.2, 0.1, 0.4, 0.5];
    let log_p_values: Vec<f32> = p_values.iter().map(|p| p.ln() as f32).collect();
    let n = log_q_values.len() as f64;
    let expected_loss: Vec<f64> = log_q_values
        .iter()
        .zip(log_p_values.iter())
        .map(|(&lq, &lp)| kl_term(f64::from(lq), f64::from(lp)))
        .collect();
    let expected_grad: Vec<f64> = log_p_values
        .iter()
        .map(|&lp| -f64::from(lp).exp() / n)
        .collect();

    let log_q = Tensor::from_slice(&log_q_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?
        .with_grad();
    let log_p = Tensor::from_slice(&log_p_values, [batch.of(2), class.of(3)], &device)?
        .with_layout([class, batch])?;

    let loss = log_q.kl_div_loss_log_target(&log_p)?;
    close(
        "KLDivLoss(log_target=true) forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, class])?.backward()?;
    close(
        "KLDivLoss(log_target=true) gradient (exactly -exp(target))",
        &log_q.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    assert!(
        log_q
            .detach()
            .kl_div_loss_log_target(&log_p.with_grad())
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn poisson_nll_loss_full_matches_hand_computed_oracle_at_log_input_false_and_full_true()
-> Result<()> {
    // `poisson_nll_loss_full` generalizes `poisson_nll_loss` with PyTorch's remaining
    // `PoissonNLLLoss` options. This oracle exercises `log_input = false` (`self` holds the
    // rate itself, base loss `self - target*log(self+eps)`) together with `full = true` (adds
    // the Stirling term `target*log(target) - target + 0.5*log(2*pi*target)` where
    // `target > 1`, else `0`); target `1.0` sits exactly at the `> 1` boundary (excluded) to
    // exercise the mask's strictness.
    let eps = 1e-3f64;
    fn base(rate: f64, target: f64, eps: f64) -> f64 {
        rate - target * (rate + eps).ln()
    }
    fn base_grad(rate: f64, target: f64, eps: f64) -> f64 {
        1.0 - target / (rate + eps)
    }
    fn stirling(target: f64) -> f64 {
        if target > 1.0 {
            target * target.ln() - target + 0.5 * (2.0 * std::f64::consts::PI * target).ln()
        } else {
            0.0
        }
    }

    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    // value(b, f) written in canonical (batch, feature) order; asymmetric extents (2, 3).
    let rate_values = [0.5f32, 2.0, 1.0, 3.0, 0.1, 5.0];
    let target_values = [1.0f32, 3.0, 0.5, 8.0, 0.2, 1.5];
    let n = rate_values.len() as f64;
    let expected_loss: Vec<f64> = rate_values
        .iter()
        .zip(target_values)
        .map(|(&r, t)| base(f64::from(r), f64::from(t), eps) + stirling(f64::from(t)))
        .collect();
    let expected_grad: Vec<f64> = rate_values
        .iter()
        .zip(target_values)
        .map(|(&r, t)| base_grad(f64::from(r), f64::from(t), eps) / n)
        .collect();

    let rate = Tensor::from_slice(&rate_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;

    let loss = rate.poisson_nll_loss_full(&target, false, true, eps as f32)?;
    close(
        "PoissonNLLLoss(log_input=false, full=true) forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, feature])?.backward()?;
    close(
        "PoissonNLLLoss(log_input=false, full=true) gradient (Stirling term contributes none)",
        &rate.grad().unwrap().to_vec()?,
        &expected_grad,
    );

    // `poisson_nll_loss` delegates to `poisson_nll_loss_full` with `full = false`; confirm the
    // two remain bit-exact for the default `log_input = true` case.
    let log_rate = Tensor::from_slice(&rate_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;
    let default_loss = log_rate.poisson_nll_loss(&target)?;
    let explicit_loss = log_rate.poisson_nll_loss_full(&target, true, false, 0.0)?;
    let explicit_loss_f64: Vec<f64> = explicit_loss
        .to_vec()?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    close(
        "poisson_nll_loss / poisson_nll_loss_full(true, false) bit-exact delegation",
        &default_loss.to_vec()?,
        &explicit_loss_f64,
    );

    assert!(
        rate.detach()
            .poisson_nll_loss_full(&target.with_grad(), false, true, eps as f32)
            .is_err()
    );
    assert!(
        rate.detach()
            .poisson_nll_loss_full(&target, false, true, 0.0)
            .is_err()
    );
    assert!(
        rate.detach()
            .poisson_nll_loss_full(&target, false, true, f32::NAN)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gaussian_nll_loss_full_matches_hand_computed_oracle_with_full_true_constant() -> Result<()> {
    // `gaussian_nll_loss_full` adds PyTorch's `full = true` constant term `0.5*log(2*pi)` to
    // `gaussian_nll_loss`'s own `full = false` value; that constant depends on none of `self`,
    // `targets`, or `var`, so no operand's gradient changes.
    let device = Device::cuda(0)?;
    let (batch, feature) = (Axis::new("batch"), Axis::new("feature"));
    let eps = 1e-3f32;
    let mean_values = [0.0f32, 2.0, -1.0, 0.5, 1.0, -2.0];
    let target_values = [1.0f32, 2.0, 0.0, 0.5, 1.5, -2.0];
    let var_values = [0.5f32, 1e-8, 2.0, 1.0, 3.0, 0.25];
    let half_ln_two_pi = 0.5 * (2.0 * std::f64::consts::PI).ln();

    let mean = Tensor::from_slice(&mean_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let target = Tensor::from_slice(&target_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;
    let var = Tensor::from_slice(&var_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();

    let base = mean.gaussian_nll_loss(&target, &var, eps)?;
    let expected_loss: Vec<f64> = base
        .to_vec()?
        .iter()
        .map(|&v| f64::from(v) + half_ln_two_pi)
        .collect();

    let mean_full = mean.gaussian_nll_loss_full(&target, &var, eps, true)?;
    close(
        "GaussianNLLLoss(full=true) forward adds 0.5*ln(2*pi) to the full=false value",
        &mean_full.to_vec()?,
        &expected_loss,
    );
    mean_full.mean([batch, feature])?.backward()?;
    let mean_grad = mean.grad().unwrap().to_vec()?;
    let var_grad = var.grad().unwrap().to_vec()?;

    // Independent second graph at full=false, to compare gradients against (the constant term
    // contributes none).
    let mean2 = Tensor::from_slice(&mean_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let var2 = Tensor::from_slice(&var_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    mean2
        .gaussian_nll_loss(&target, &var2, eps)?
        .mean([batch, feature])?
        .backward()?;
    let mean2_grad_f64: Vec<f64> = mean2
        .grad()
        .unwrap()
        .to_vec()?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let var2_grad_f64: Vec<f64> = var2
        .grad()
        .unwrap()
        .to_vec()?
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    close(
        "GaussianNLLLoss(full=true) mean gradient matches full=false (constant term, no gradient)",
        &mean_grad,
        &mean2_grad_f64,
    );
    close(
        "GaussianNLLLoss(full=true) var gradient matches full=false (constant term, no gradient)",
        &var_grad,
        &var2_grad_f64,
    );

    // `gaussian_nll_loss_full(..., false)` is bit-exact with `gaussian_nll_loss`.
    let mean3 = Tensor::from_slice(&mean_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;
    let var3 = Tensor::from_slice(&var_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;
    let delegated = mean3.gaussian_nll_loss_full(&target, &var3, eps, false)?;
    let delegated_f64: Vec<f64> = delegated.to_vec()?.iter().map(|&v| f64::from(v)).collect();
    close(
        "gaussian_nll_loss / gaussian_nll_loss_full(eps, false) bit-exact delegation",
        &base.to_vec()?,
        &delegated_f64,
    );
    Ok(())
}

#[test]
fn pooling_options_and_unpooling_reject_invalid_configurations_before_launch() {
    let (channel, length, height, width) = (
        Axis::new("channel"),
        Axis::new("length"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let line = Shape::new([channel.of(1), length.of(5)]).unwrap();

    // ceil_mode drops a last window that would start in the right padding: L=5, k=2, s=2,
    // p=1 rounds (5 + 2 - 2) / 2 up to 3, +1 = 4 windows, but window 3 would start at
    // padded 6 >= L + p = 6, so PyTorch keeps 3.
    let pool = AvgPool1d::new(channel, length, 2)
        .padding(1)
        .ceil_mode(true);
    assert_eq!(
        pool.output_shape(&line).unwrap(),
        Shape::new([length.of(3), channel.of(1)]).unwrap()
    );
    // ceil_mode keeps an overhanging window that still starts inside the input: L=5, k=2,
    // s=2, no padding gives 3 windows (floor mode gives 2).
    let pool = AvgPool1d::new(channel, length, 2).ceil_mode(true);
    assert_eq!(
        pool.output_shape(&line).unwrap(),
        Shape::new([length.of(3), channel.of(1)]).unwrap()
    );
    // ceil_mode admits a kernel wider than the input when the stride rounds it in, as
    // PyTorch does (L=1, k=2, s=2 -> 1 window).
    let unit = Shape::new([channel.of(1), length.of(1)]).unwrap();
    assert_eq!(
        LPPool1d::new(channel, length, 2.0, 2)
            .unwrap()
            .ceil_mode(true)
            .output_shape(&unit)
            .unwrap(),
        Shape::new([length.of(1), channel.of(1)]).unwrap()
    );
    assert!(
        LPPool1d::new(channel, length, 2.0, 2)
            .unwrap()
            .output_shape(&unit)
            .is_err()
    );

    let error = AvgPool2d::new(channel, [height, width], [2, 2])
        .divisor_override(0)
        .output_shape(&Shape::new([channel.of(1), height.of(4), width.of(4)]).unwrap())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("divisor_override must be positive"),
        "{error}"
    );

    // Any finite positive real `p` is accepted; zero, negative and non-finite are not.
    assert!(LPPool1d::new(channel, length, 1.5, 2).is_ok());
    for p in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(LPPool1d::new(channel, length, p, 2).is_err(), "{p}");
    }

    // MaxUnpool: default size `(in - 1) * s - 2p + k`, explicit sizes strictly within one
    // stride of it, distinct axes.
    let pooled = Shape::new([height.of(2), width.of(3), channel.of(1)]).unwrap();
    let unpool = MaxUnpool2d::new([height, width], [2, 2]);
    assert_eq!(
        unpool.output_shape(&pooled).unwrap(),
        Shape::new([height.of(4), width.of(6), channel.of(1)]).unwrap()
    );
    let unpool = MaxUnpool2d::new([height, width], [3, 3])
        .stride([2, 2])
        .padding([1, 1]);
    assert_eq!(
        unpool.output_shape(&pooled).unwrap(),
        Shape::new([height.of(3), width.of(5), channel.of(1)]).unwrap()
    );
    let error = MaxUnpool2d::new([height, height], [2, 2])
        .output_shape(&pooled)
        .unwrap_err()
        .to_string();
    assert!(error.contains("distinct"), "{error}");
    let error = MaxUnpool1d::new(length, 0)
        .output_shape(&line)
        .unwrap_err()
        .to_string();
    assert!(error.contains("positive"), "{error}");
}

#[test]
#[ignore = "requires CUDA"]
fn avg_pool1d_ceil_mode_count_include_pad_and_divisor_override_match_hand_computed() -> Result<()> {
    // L=6, k=3, s=2, p=1, ceil_mode: 4 windows over padded [pad, 1..6, pad] plus one
    // overhang slot; padded starts 0, 2, 4, 6. Window 3 holds x5, the right padding and the
    // overhang: PyTorch divides it by 2 (clipped to the padded input) with
    // count_include_pad=True, by 1 (real elements) without it, by 4 under divisor_override=4.
    let device = Device::cuda(0)?;
    let (channel, length) = (Axis::new("channel"), Axis::new("length"));
    let inputs: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    #[allow(clippy::type_complexity)]
    let cases: [(&str, AvgPool1d, [f64; 4], [f64; 6]); 3] = [
        (
            "count_include_pad=True",
            AvgPool1d::new(channel, length, 3)
                .stride(2)
                .padding(1)
                .ceil_mode(true),
            [1.0, 3.0, 5.0, 3.0],
            [
                1.0 / 3.0,
                2.0 / 3.0,
                1.0 / 3.0,
                2.0 / 3.0,
                1.0 / 3.0,
                5.0 / 6.0,
            ],
        ),
        (
            "count_include_pad=False",
            AvgPool1d::new(channel, length, 3)
                .stride(2)
                .padding(1)
                .ceil_mode(true)
                .count_include_pad(false),
            [1.5, 3.0, 5.0, 6.0],
            [0.5, 5.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 4.0 / 3.0],
        ),
        (
            "divisor_override=4",
            AvgPool1d::new(channel, length, 3)
                .stride(2)
                .padding(1)
                .ceil_mode(true)
                .divisor_override(4),
            [0.75, 2.25, 3.75, 1.5],
            [0.25, 0.5, 0.25, 0.5, 0.25, 0.5],
        ),
    ];
    for (name, mut pool, forward, gradient) in cases {
        let input =
            Tensor::from_slice(&inputs, [channel.of(1), length.of(6)], &device)?.with_grad();
        assert_eq!(
            pool.build(input.shape(), &device, 0)?,
            Shape::new([length.of(4), channel.of(1)])?
        );
        let actual = pool.forward(&input)?;
        close(
            &format!("AvgPool1d ceil_mode {name} forward"),
            &actual.to_vec()?,
            &forward,
        );
        actual.sum([channel, length])?.backward()?;
        close(
            &format!("AvgPool1d ceil_mode {name} gradient"),
            &input.grad().unwrap().to_vec()?,
            &gradient,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn avg_pool2d_ceil_mode_matches_hand_computed_under_reordered_storage() -> Result<()> {
    // 2x3 input, 2x2 kernel, stride 2, ceil_mode: height keeps 1 window, width rounds up to
    // 2, the second clipped to column 2 alone. Divisors: 4 and 2 (no padding, so both
    // count_include_pad settings agree).
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let inputs: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for include in [true, false] {
        let input =
            Tensor::from_slice(&inputs, [channel.of(1), height.of(2), width.of(3)], &device)?
                .with_layout([width, height, channel])?
                .with_grad();
        let mut pool = AvgPool2d::new(channel, [height, width], [2, 2])
            .ceil_mode(true)
            .count_include_pad(include);
        assert_eq!(
            pool.build(input.shape(), &device, 0)?,
            Shape::new([height.of(1), width.of(2), channel.of(1)])?
        );
        let actual = pool.forward(&input)?;
        close(
            "reordered-storage ceil_mode AvgPool2d forward",
            &actual.to_vec()?,
            &[3.0, 4.5],
        );
        actual.sum([channel, height, width])?.backward()?;
        close(
            "reordered-storage ceil_mode AvgPool2d gradient",
            &input.grad().unwrap().to_vec()?,
            &[0.25, 0.25, 0.5, 0.25, 0.25, 0.5],
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn avg_pool3d_count_exclude_pad_matches_hand_computed() -> Result<()> {
    // Width [2, 4], kernel 2, stride 2, padding 1: windows [pad, 2] and [4, pad]. Without
    // count_include_pad each divides by its one real element.
    let device = Device::cuda(0)?;
    let (channel, depth, height, width) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let input = Tensor::from_slice(
        &[2.0, 4.0],
        [channel.of(1), depth.of(1), height.of(1), width.of(2)],
        &device,
    )?
    .with_grad();
    let mut pool = AvgPool3d::new(channel, [depth, height, width], [1, 1, 2])
        .padding([0, 0, 1])
        .count_include_pad(false);
    pool.build(input.shape(), &device, 0)?;
    let actual = pool.forward(&input)?;
    close(
        "AvgPool3d count_include_pad=False forward",
        &actual.to_vec()?,
        &[2.0, 4.0],
    );
    actual.sum([channel, depth, height, width])?.backward()?;
    close(
        "AvgPool3d count_include_pad=False gradient",
        &input.grad().unwrap().to_vec()?,
        &[1.0, 1.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lp_pool_real_p_matches_hand_computed_forward_and_gradient() -> Result<()> {
    // p = 1.5 over [1, 4] and [0, 9]: sums 1 + 8 = 9 and 0 + 27 = 27, roots 9^(2/3) and 9.
    // Gradient x_i^(p-1) * S^(1/p-1): sqrt(x_i) * S^(-1/3); zero input has zero gradient.
    let device = Device::cuda(0)?;
    let (channel, length) = (Axis::new("channel"), Axis::new("length"));
    let input = Tensor::from_slice(
        &[1.0, 4.0, 0.0, 9.0],
        [channel.of(1), length.of(4)],
        &device,
    )?
    .with_grad();
    let mut pool = LPPool1d::new(channel, length, 1.5, 2)?;
    pool.build(input.shape(), &device, 0)?;
    let actual = pool.forward(&input)?;
    close(
        "LPPool1d(p=1.5) forward",
        &actual.to_vec()?,
        &[4.3267487109222245, 9.0],
    );
    actual.sum([channel, length])?.backward()?;
    close(
        "LPPool1d(p=1.5) gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.4807498567691362, 0.9614997135382723, 0.0, 1.0],
    );

    // p = 0.5 over [4, 9]: (2 + 3)^2 = 25; gradient S / sqrt(x_i) = 2.5, 5/3.
    let (depth, height, width) = (Axis::new("depth"), Axis::new("height"), Axis::new("width"));
    let input = Tensor::from_slice(
        &[4.0, 9.0],
        [channel.of(1), depth.of(1), height.of(1), width.of(2)],
        &device,
    )?
    .with_grad();
    let mut pool = LPPool3d::new(channel, [depth, height, width], 0.5, [1, 1, 2])?;
    pool.build(input.shape(), &device, 0)?;
    let actual = pool.forward(&input)?;
    close("LPPool3d(p=0.5) forward", &actual.to_vec()?, &[25.0]);
    actual.sum([channel, depth, height, width])?.backward()?;
    close(
        "LPPool3d(p=0.5) gradient",
        &input.grad().unwrap().to_vec()?,
        &[2.5, 5.0 / 3.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lp_pool2d_real_p_ceil_mode_matches_hand_computed_under_reordered_storage() -> Result<()> {
    // p = 2.5, kernel [1, 2], stride [1, 2], ceil_mode over rows [1, 2, 4] and [3, 1, 1]: the
    // second window of each row is clipped to one element, which PyTorch's
    // `avg_pool(x^p) * k` composition scales by k / 1 = 2: `(2 x^p)^(1/p) = 2^0.4 x`.
    // Full windows: `(a^p + b^p)^(1/p)`, gradient `x_i^(p-1) S^(1/p-1)`.
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let input = Tensor::from_slice(
        &[1.0, 2.0, 4.0, 3.0, 1.0, 1.0],
        [channel.of(1), height.of(2), width.of(3)],
        &device,
    )?
    .with_layout([width, channel, height])?
    .with_grad();
    let mut pool = LPPool2d::new(channel, [height, width], 2.5, [1, 2])?.ceil_mode(true);
    assert_eq!(
        pool.build(input.shape(), &device, 0)?,
        Shape::new([height.of(2), width.of(2), channel.of(1)])?
    );
    let actual = pool.forward(&input)?;
    close(
        "reordered-storage ceil_mode LPPool2d(p=2.5) forward",
        &actual.to_vec()?,
        &[
            2.1345563259430547,
            5.278031643091578,
            3.0755472204062158,
            1.3195079107728942,
        ],
    );
    actual.sum([channel, height, width])?.backward()?;
    close(
        "reordered-storage ceil_mode LPPool2d(p=2.5) gradient",
        &input.grad().unwrap().to_vec()?,
        &[
            0.3206554095886695,
            0.9069504581771924,
            1.3195079107728942,
            0.9633814574894259,
            0.18540284793793801,
            1.3195079107728942,
        ],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_max_pool_cross_axis_ties_route_to_pytorch_scan_order() -> Result<()> {
    // PyTorch's adaptive max kernel scans a window with the last spatial axis innermost and
    // keeps the first strict maximum. [[1, 5], [5, 2]] ties at (0, 1) and (1, 0): PyTorch
    // keeps (0, 1), flat index 1. (Reducing height first, then width, would pick (1, 0).)
    let device = Device::cuda(0)?;
    let (height, width) = (Axis::new("height"), Axis::new("width"));
    let input = Tensor::from_slice(&[1.0, 5.0, 5.0, 2.0], [height.of(2), width.of(2)], &device)?
        .with_layout([width, height])?
        .with_grad();
    let actual = AdaptiveMaxPool2d::new([height, width], [1, 1]).forward(&input)?;
    close(
        "AdaptiveMaxPool2d cross-axis tie forward",
        &actual.to_vec()?,
        &[5.0],
    );
    actual.sum([height, width])?.backward()?;
    close(
        "reordered-storage AdaptiveMaxPool2d cross-axis tie gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.0, 1.0, 0.0, 0.0],
    );

    // Depth x width tie in 3D: depth 0 = [1, 7], depth 1 = [7, 0]. Scan order (d, h, w)
    // reaches (0, 0, 1) first.
    let depth = Axis::new("depth");
    let input = Tensor::from_slice(
        &[1.0, 7.0, 7.0, 0.0],
        [depth.of(2), height.of(1), width.of(2)],
        &device,
    )?
    .with_grad();
    let actual = AdaptiveMaxPool3d::new([depth, height, width], [1, 1, 1]).forward(&input)?;
    close(
        "AdaptiveMaxPool3d cross-axis tie forward",
        &actual.to_vec()?,
        &[7.0],
    );
    actual.sum([depth, height, width])?.backward()?;
    close(
        "AdaptiveMaxPool3d cross-axis tie gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.0, 1.0, 0.0, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool2d_indices_and_max_unpool2d_match_hand_computed() -> Result<()> {
    // [[1, 6, 3, 2], [5, 4, 8, 8]], 2x2 windows: maxima 6 at flat 1 and 8 at flat 6 (the
    // first of the tied pair in row-major scan). Unpooling puts them back in a zero 2x4
    // plane; with weights w = 1..8 on the unpooled output, each pooled value's gradient is
    // w[index] (2 and 7), which max pooling routes to its winner.
    let device = Device::cuda(0)?;
    let (channel, height, width) = (
        Axis::new("channel"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let input = Tensor::from_slice(
        &[1.0, 6.0, 3.0, 2.0, 5.0, 4.0, 8.0, 8.0],
        [channel.of(1), height.of(2), width.of(4)],
        &device,
    )?
    .with_grad();
    let pool = MaxPool2d::new(channel, [height, width], [2, 2]);
    let (pooled, indices) = pool.forward_with_indices(&input)?;
    assert_eq!(indices, vec![1, 6]);
    close(
        "MaxPool2d return_indices forward",
        &pooled.to_vec()?,
        &[6.0, 8.0],
    );
    let unpool = MaxUnpool2d::new([height, width], [2, 2]);
    let unpooled = unpool.forward(&pooled, &indices)?;
    assert_eq!(
        unpooled.shape(),
        &Shape::new([height.of(2), width.of(4), channel.of(1)])?
    );
    close(
        "MaxUnpool2d forward",
        &unpooled.to_vec()?,
        &[0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0],
    );
    let weights = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [height.of(2), width.of(4), channel.of(1)],
        &device,
    )?;
    unpooled
        .mul(&weights)?
        .sum([height, width, channel])?
        .backward()?;
    close(
        "MaxUnpool2d then MaxPool2d gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn max_unpool1d_duplicates_output_size_and_bounds_match_pytorch() -> Result<()> {
    // Overlapping MaxPool1d(k=3, s=1) over [1, 5, 2, 0] picks index 1 twice.
    let device = Device::cuda(0)?;
    let (channel, length) = (Axis::new("channel"), Axis::new("length"));
    let input = Tensor::from_slice(
        &[1.0, 5.0, 2.0, 0.0],
        [channel.of(1), length.of(4)],
        &device,
    )?;
    let (pooled, indices) = MaxPool1d::new(channel, length, 3)
        .stride(1)
        .forward_with_indices(&input)?;
    assert_eq!(indices, vec![1, 1]);
    close(
        "overlapping MaxPool1d forward",
        &pooled.to_vec()?,
        &[5.0, 5.0],
    );

    // Unpool distinct values at the duplicate index: the last write (20) wins, as in
    // PyTorch's CPU kernel, and both inputs receive the upstream gradient at index 1 (2.0).
    let values =
        Tensor::from_slice(&[10.0, 20.0], [length.of(2), channel.of(1)], &device)?.with_grad();
    let unpool = MaxUnpool1d::new(length, 3).stride(1);
    let unpooled = unpool.forward(&values, &indices)?;
    close(
        "MaxUnpool1d duplicate index forward (last write wins)",
        &unpooled.to_vec()?,
        &[0.0, 20.0, 0.0, 0.0],
    );
    let weights = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0],
        [length.of(4), channel.of(1)],
        &device,
    )?;
    unpooled.mul(&weights)?.sum([length, channel])?.backward()?;
    close(
        "MaxUnpool1d duplicate index gradient (gather by index)",
        &values.grad().unwrap().to_vec()?,
        &[2.0, 2.0],
    );

    // MaxPool1d(k=2) over odd length 5 drops the tail; output_size=5 restores it, within
    // PyTorch's (default - stride, default + stride) = (2, 6) window.
    let input = Tensor::from_slice(
        &[3.0, 1.0, 4.0, 1.0, 5.0],
        [channel.of(1), length.of(5)],
        &device,
    )?;
    let (pooled, indices) = MaxPool1d::new(channel, length, 2).forward_with_indices(&input)?;
    assert_eq!(indices, vec![0, 2]);
    let unpool = MaxUnpool1d::new(length, 2);
    close(
        "MaxUnpool1d output_size forward",
        &unpool.forward_sized(&pooled, &indices, 5)?.to_vec()?,
        &[3.0, 0.0, 4.0, 0.0, 0.0],
    );
    let error = unpool
        .forward_sized(&pooled, &indices, 6)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("outside"), "{error}");
    let error = pooled
        .max_unpool1d(length, &[0, 4], 4)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("outside the output volume"), "{error}");
    let error = pooled
        .max_unpool1d(length, &[0], 4)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("indices"), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn max_pool3d_indices_and_max_unpool3d_match_hand_computed_under_reordered_storage() -> Result<()> {
    // Two channels, kernel [1, 2, 1] over height: channel 0 [3, 7] -> 7 at flat 1, channel 1
    // [9, -1] -> 9 at flat 0. Indices follow the pooled output's logical order (channel
    // last). Weighted unpooled gradients: w(h1, c0) = 3, w(h0, c1) = 2.
    let device = Device::cuda(0)?;
    let (channel, depth, height, width) = (
        Axis::new("channel"),
        Axis::new("depth"),
        Axis::new("height"),
        Axis::new("width"),
    );
    let input = Tensor::from_slice(
        &[3.0, 7.0, 9.0, -1.0],
        [channel.of(2), depth.of(1), height.of(2), width.of(1)],
        &device,
    )?
    .with_layout([height, width, channel, depth])?
    .with_grad();
    let (pooled, indices) =
        MaxPool3d::new(channel, [depth, height, width], [1, 2, 1]).forward_with_indices(&input)?;
    assert_eq!(indices, vec![1, 0]);
    close(
        "MaxPool3d return_indices forward",
        &pooled.to_vec()?,
        &[7.0, 9.0],
    );
    let unpooled =
        MaxUnpool3d::new([depth, height, width], [1, 2, 1]).forward(&pooled, &indices)?;
    assert_eq!(
        unpooled.shape(),
        &Shape::new([depth.of(1), height.of(2), width.of(1), channel.of(2)])?
    );
    close(
        "MaxUnpool3d forward",
        &unpooled.to_vec()?,
        &[0.0, 9.0, 7.0, 0.0],
    );
    let weights = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0],
        [depth.of(1), height.of(2), width.of(1), channel.of(2)],
        &device,
    )?;
    unpooled
        .mul(&weights)?
        .sum([depth, height, width, channel])?
        .backward()?;
    close(
        "reordered-storage MaxUnpool3d then MaxPool3d gradient",
        &input.grad().unwrap().to_vec()?,
        &[0.0, 3.0, 2.0, 0.0],
    );
    Ok(())
}

#[test]
fn multihead_attention_rejects_invalid_configuration_before_allocation() {
    let (feature, time) = (Axis::new("feature"), Axis::new("time"));
    assert!(MultiheadAttention::new(feature, feature, 4, 2, 0.0).is_err());
    assert!(MultiheadAttention::new(feature, time, 4, 2, -0.1).is_err());
    assert!(MultiheadAttention::new(feature, time, 4, 2, 1.1).is_err());
    assert!(MultiheadAttention::new(feature, time, 4, 2, f32::NAN).is_err());
    assert!(MultiheadAttention::new(feature, time, 0, 2, 0.0).is_err());
    assert!(MultiheadAttention::new(feature, time, 4, 0, 0.0).is_err());
    assert!(MultiheadAttention::new(feature, time, 5, 2, 0.0).is_err());
    // dropout=0.1 (PyTorch's own default) is a valid configuration now that the
    // seeded training-pass contract (`TrainingPass`) has landed; only `forward`/
    // `forward_with_weights` -- evaluation semantics -- ignore it.
    assert!(MultiheadAttention::new(feature, time, 4, 2, 0.1).is_ok());
    assert!(
        MultiheadAttention::new(feature, time, 4, 2, 0.0)
            .unwrap()
            .dropout(1.5)
            .is_err()
    );

    let mha = MultiheadAttention::new(feature, time, 4, 2, 0.0).unwrap();
    let q = Shape::new([time.of(2), feature.of(4)]).unwrap();
    let k = Shape::new([time.of(3), feature.of(4)]).unwrap();
    let v_mismatched = Shape::new([time.of(4), feature.of(4)]).unwrap();
    assert!(mha.output_shape(&q, &k, &k).is_ok());
    assert!(mha.output_shape(&q, &k, &v_mismatched).is_err());
}

#[test]
#[ignore = "requires CUDA"]
fn multihead_attention_two_head_self_attention_matches_hand_computed_oracle_with_key_padding_mask()
-> Result<()> {
    // Independent f64 oracle (finite differences of a from-scratch numpy forward, Richardson
    // checked at eps in {1e-4, 5e-5} to under 1e-10 before being frozen as literals here) for
    // embed_dim=4, num_heads=2 (head_feature=2), 3 self-attended time steps, and a key padding
    // mask that excludes the last key from every query. `bk`'s gradient is exactly zero: adding
    // a bias to every key shifts every score for a fixed (head, query) by the SAME amount
    // (independent of the key position being scored), and softmax is invariant to a constant
    // shift along its own reduced axis -- a property of the mechanism, not a coincidence of
    // these particular weights.
    let device = Device::cuda(0)?;
    let (time, feature) = (Axis::new("time"), Axis::new("feature"));
    let x_values: [f32; 12] = [
        0.6, -0.3, 0.9, 0.15, -1.2, 0.45, -0.6, 0.3, 0.15, 0.75, -1.05, 0.6,
    ];
    let x = Tensor::from_slice(&x_values, [time.of(3), feature.of(4)], &device)?.with_grad();

    let mut mha = MultiheadAttention::new(feature, time, 4, 2, 0.0)?;
    let expected_shape = Shape::new([time.of(3), feature.of(4)])?;
    assert_eq!(
        mha.build(x.shape(), x.shape(), x.shape(), &device, 0)?,
        expected_shape
    );

    let set = |mha: &MultiheadAttention, name: &str, values: &[f32]| -> Result<()> {
        mha.named_parameters()
            .iter()
            .find(|(n, _)| n == name)
            .ok_or("missing parameter")?
            .1
            .set_values(values)
    };
    set(
        &mha,
        "query.weight",
        &[
            0.4, 0.8, -0.4, 0.2, 0.2, -0.8, 1.2, 0.4, -0.6, 0.4, 0.8, -0.2, 0.8, 0.2, -0.4, 0.6,
        ],
    )?;
    set(&mha, "query.bias", &[0.05, -0.1, 0.15, 0.0])?;
    set(
        &mha,
        "key.weight",
        &[
            0.8, -0.4, 0.2, 0.4, -0.2, 0.6, 0.8, -0.4, 0.4, 0.2, -0.8, 0.6, -0.4, 0.8, 0.4, -0.2,
        ],
    )?;
    set(&mha, "key.bias", &[-0.05, 0.1, 0.0, 0.05])?;
    set(
        &mha,
        "value.weight",
        &[
            0.2, 0.4, -0.6, 0.8, 0.8, -0.2, 0.4, -0.4, -0.4, 0.6, 0.2, 0.4, 0.4, -0.4, 0.8, 0.2,
        ],
    )?;
    set(&mha, "value.bias", &[0.0, 0.05, -0.05, 0.1])?;
    set(
        &mha,
        "output.weight",
        &[
            0.6, -0.4, 0.2, 0.4, 0.4, 0.8, -0.4, -0.2, -0.2, 0.4, 0.6, 0.8, 0.8, -0.6, 0.4, 0.2,
        ],
    )?;
    set(&mha, "output.bias", &[0.1, 0.0, -0.1, 0.05])?;

    // PyTorch's own key_padding_mask convention: 1.0 = ignore that key for every query. The last
    // key (index 2) is padding.
    let key_padding = Tensor::from_slice(&[0.0, 0.0, 1.0], [mha.key_time().of(3)], &device)?;

    let (output, weights) = mha.forward_with_weights(&x, &x, &x, None, Some(&key_padding), true)?;
    close(
        "MultiheadAttention forward",
        &output.to_vec()?,
        &[
            -0.2659694341,
            -0.1575912433,
            0.3308803281,
            0.5738299373,
            -0.1220312472,
            0.8221968242,
            -0.1720345522,
            0.1616877521,
            0.6524414440,
            0.3693247401,
            -0.2079593236,
            -0.1045806274,
        ],
    );
    close(
        "MultiheadAttention head-averaged weights (masked key gets zero weight)",
        &weights.unwrap().to_vec()?,
        &[
            0.3406310228,
            0.6593689772,
            0.0,
            0.6272801610,
            0.3727198390,
            0.0,
            0.8671724567,
            0.1328275433,
            0.0,
        ],
    );

    // Reordered physical storage for the input must not change the forward value.
    let x_reordered = Tensor::from_slice(&x_values, [time.of(3), feature.of(4)], &device)?
        .with_layout([feature, time])?;
    close(
        "MultiheadAttention forward is unaffected by reordered input storage",
        &mha.forward(
            &x_reordered,
            &x_reordered,
            &x_reordered,
            None,
            Some(&key_padding),
        )?
        .to_vec()?,
        &[
            -0.2659694341,
            -0.1575912433,
            0.3308803281,
            0.5738299373,
            -0.1220312472,
            0.8221968242,
            -0.1720345522,
            0.1616877521,
            0.6524414440,
            0.3693247401,
            -0.2079593236,
            -0.1045806274,
        ],
    );

    output.mean([time, feature])?.backward()?;
    let grad = |name: &str| -> Vec<f32> {
        mha.named_parameters()
            .iter()
            .find(|(n, _)| n == name)
            .unwrap()
            .1
            .grad()
            .unwrap()
            .to_vec()
            .unwrap()
    };
    close(
        "MultiheadAttention x gradient",
        &x.grad().unwrap().to_vec()?,
        &[
            0.0251251226,
            0.1364713234,
            0.0885826173,
            0.2129792737,
            -0.0063528112,
            0.0850940605,
            0.0714196287,
            0.1755497389,
            -0.0005616756,
            0.0022137745,
            -0.0009979275,
            0.0006980926,
        ],
    );
    close(
        "MultiheadAttention query.weight gradient",
        &grad("query.weight"),
        &[
            -0.0024837483,
            0.0010928493,
            -0.0007901254,
            0.0010271630,
            0.0023874106,
            -0.0010504607,
            0.0010260336,
            -0.0013338437,
            0.0015863129,
            -0.0006979777,
            -0.0006484385,
            0.0008429700,
            0.0052972790,
            -0.0023308027,
            0.0013293262,
            -0.0017281241,
        ],
    );
    close(
        "MultiheadAttention query.bias gradient",
        &grad("query.bias"),
        &[0.0198445104, -0.0087315846, 0.0041543950, -0.0054007134],
    );
    close(
        "MultiheadAttention key.weight gradient",
        &grad("key.weight"),
        &[
            0.0030097950,
            -0.0033499178,
            -0.0013439622,
            -0.0014156062,
            -0.0012540812,
            0.0013957991,
            0.0005599843,
            0.0005898359,
            0.0025081625,
            -0.0027915982,
            -0.0011199685,
            -0.0011796718,
            -0.0002508162,
            0.0002791598,
            0.0001119969,
            0.0001179672,
        ],
    );
    close(
        "MultiheadAttention key.bias gradient is exactly zero (uniform shift along the softmax's own reduced axis)",
        &grad("key.bias"),
        &[0.0, 0.0, 0.0, 0.0],
    );
    close(
        "MultiheadAttention value.weight gradient",
        &grad("value.weight"),
        &[
            0.0038524380,
            0.0028893285,
            -0.0868647285,
            -0.0434323642,
            -0.0116051825,
            -0.0087038869,
            0.0161936369,
            0.0080968184,
            0.0832103650,
            0.0624077737,
            0.0876127263,
            0.0438063631,
            0.0396789635,
            0.0297592226,
            0.0872387274,
            0.0436193637,
        ],
    );
    close(
        "MultiheadAttention value.bias gradient",
        &grad("value.bias"),
        &[0.2, 0.15, 0.4, 0.2],
    );
    close(
        "MultiheadAttention output.weight gradient",
        &grad("output.weight"),
        &[
            -0.0324077737,
            -0.0324077737,
            -0.0324077737,
            -0.0324077737,
            0.0598958066,
            0.0598958066,
            0.0598958066,
            0.0598958066,
            0.0786936369,
            0.0786936369,
            0.0786936369,
            0.0786936369,
            0.0103272490,
            0.0103272490,
            0.0103272490,
            0.0103272490,
        ],
    );
    close(
        "MultiheadAttention output.bias gradient",
        &grad("output.bias"),
        &[0.25, 0.25, 0.25, 0.25],
    );

    Ok(())
}

// --- nn-climb-misc: Upsample bicubic/scale_factor, LinearCrossEntropyLoss,
// AdaptiveLogSoftmaxWithLoss, CTCLoss -------------------------------------

#[test]
#[ignore = "requires CUDA"]
fn resample_bicubic_matches_independent_oracle_forward_and_gradient() -> Result<()> {
    // Independent oracle: a from-scratch Python reimplementation of PyTorch's
    // documented `a = -0.75` separable bicubic kernel and
    // `align_corners=False` half-pixel source formula (never calling this
    // crate), forward-mode for the values and central differences for the
    // gradient. Storage is reordered (`with_layout([length, batch])`, an
    // unrelated `batch` axis preserved through the resample) for the CUDA
    // non-trivial-layout coverage the base order requires.
    let device = Device::cuda(0)?;
    let (batch, length) = (Axis::new("batch"), Axis::new("length"));
    let x = Tensor::from_slice(
        &[0.0, 1.0, 2.0, 3.0, 10.0, 7.0, 3.0, -2.0],
        [batch.of(2), length.of(4)],
        &device,
    )?
    .with_layout([length, batch])?
    .with_grad();

    let out = x.resample_bicubic(length, 7)?;
    assert_eq!(out.shape(), &Shape::new([batch.of(2), length.of(7)])?);
    close(
        "bicubic upsample values",
        &out.to_vec()?,
        &[
            -0.099216, 0.279246, 0.896593, 1.5, 2.103407, 2.720754, 3.099216, 10.29765, 9.223761,
            7.356414, 5.1875, 2.529155, -0.542274, -2.496082,
        ],
    );
    let weight: Vec<f32> = (1..=14).map(|v| v as f32).collect();
    let weight = Tensor::from_slice(&weight, [batch.of(2), length.of(7)], &device)?;
    out.mul(&weight)?.mean([batch, length])?.backward()?;
    close(
        "bicubic upsample gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[
            0.15817, 0.389089, 0.626946, 0.825795, 1.019139, 1.27812, 1.515976, 1.686765,
        ],
    );

    assert!(x.resample_bicubic(length, 0).is_err());
    assert!(
        Tensor::from_slice(&[0.0_f32], [length.of(1)], &device)?
            .resample_bicubic(length, 2)
            .is_err()
    );
    println!("bicubic upsample values, gradient, and rejections PASS");
    Ok(())
}

#[test]
fn upsample_module_rejects_invalid_configurations_before_launch() -> Result<()> {
    let (height, width, depth) = (Axis::new("height"), Axis::new("width"), Axis::new("depth"));
    assert!(Upsample::size(vec![height], UpsampleMode::Bilinear, vec![8]).is_err());
    assert!(Upsample::size(vec![height, width], UpsampleMode::Trilinear, vec![8, 8]).is_err());
    assert!(Upsample::size(vec![height, height], UpsampleMode::Bicubic, vec![8, 8]).is_err());
    assert!(Upsample::size(vec![height], UpsampleMode::Nearest, vec![]).is_err());
    assert!(Upsample::scale_factor(vec![height], UpsampleMode::Nearest, vec![-1.0]).is_err());

    let shape = Shape::new([height.of(3), width.of(3)])?;
    let odd_ratio = Upsample::size(vec![height, width], UpsampleMode::Bilinear, vec![7, 5])?
        .output_shape(&shape);
    assert!(odd_ratio.is_ok()); // bilinear tolerates a non-integer ratio
    let odd_nearest = Upsample::size(vec![height, width], UpsampleMode::Nearest, vec![7, 6])?
        .output_shape(&shape);
    assert!(
        odd_nearest.is_err(),
        "nearest must reject a non-integer output ratio before forward runs a kernel"
    );

    let _ = depth; // exercised in the CUDA trilinear/bicubic module tests below
    println!("Upsample rejects bad configurations before launch PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn upsample_module_nearest_scale_factor_and_bilinear_size_match_the_composed_primitives()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, height, width) = (Axis::new("batch"), Axis::new("height"), Axis::new("width"));
    let values: Vec<f32> = (0..2 * 3 * 4).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&values, [batch.of(2), height.of(3), width.of(4)], &device)?;

    let nearest =
        Upsample::scale_factor(vec![height, width], UpsampleMode::Nearest, vec![2.0, 2.0])?;
    let via_module = nearest.forward(&x)?;
    let via_primitive = x.upsample_nearest(height, 2)?.upsample_nearest(width, 2)?;
    assert_eq!(
        via_module.shape(),
        &Shape::new([batch.of(2), height.of(6), width.of(8)])?
    );
    close(
        "Upsample nearest module matches composed upsample_nearest",
        &via_module.to_vec()?,
        &via_primitive
            .to_vec()?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );

    let bilinear = Upsample::size(vec![height, width], UpsampleMode::Bilinear, vec![5, 6])?;
    let via_module = bilinear.forward(&x)?;
    let via_primitive = x
        .resample_bilinear(height, 5)?
        .resample_bilinear(width, 6)?;
    assert_eq!(
        via_module.shape(),
        &Shape::new([batch.of(2), height.of(5), width.of(6)])?
    );
    close(
        "Upsample bilinear module matches composed resample_bilinear",
        &via_module.to_vec()?,
        &via_primitive
            .to_vec()?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );
    println!("Upsample nearest and bilinear module composition PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn upsample_module_bicubic_and_trilinear_match_the_composed_primitives() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, height, width, depth) = (
        Axis::new("batch"),
        Axis::new("height"),
        Axis::new("width"),
        Axis::new("depth"),
    );
    let values: Vec<f32> = (0..2 * 4 * 5).map(|v| v as f32 / 3.0).collect();
    let x = Tensor::from_slice(&values, [batch.of(2), height.of(4), width.of(5)], &device)?;
    let bicubic =
        Upsample::scale_factor(vec![height, width], UpsampleMode::Bicubic, vec![1.75, 1.4])?;
    let via_module = bicubic.forward(&x)?;
    let via_primitive = x.resample_bicubic(height, 7)?.resample_bicubic(width, 7)?;
    assert_eq!(via_module.shape(), via_primitive.shape());
    close(
        "Upsample bicubic module matches composed resample_bicubic",
        &via_module.to_vec()?,
        &via_primitive
            .to_vec()?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );

    let cube_values: Vec<f32> = (0..2 * 2 * 3).map(|v| v as f32).collect();
    let cube = Tensor::from_slice(
        &cube_values,
        [depth.of(2), height.of(2), width.of(3)],
        &device,
    )?;
    let trilinear = Upsample::size(
        vec![depth, height, width],
        UpsampleMode::Trilinear,
        vec![4, 3, 5],
    )?;
    let via_module = trilinear.forward(&cube)?;
    let via_primitive = cube
        .resample_bilinear(depth, 4)?
        .resample_bilinear(height, 3)?
        .resample_bilinear(width, 5)?;
    assert_eq!(
        via_module.shape(),
        &Shape::new([depth.of(4), height.of(3), width.of(5)])?
    );
    close(
        "Upsample trilinear module matches composed resample_bilinear",
        &via_module.to_vec()?,
        &via_primitive
            .to_vec()?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );
    println!("Upsample bicubic and trilinear module composition PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn linear_cross_entropy_with_logits_matches_independent_oracle_and_label_smoothing() -> Result<()> {
    // Independent oracle: a from-scratch Python reimplementation (linear
    // projection + log-softmax + one-hot NLL, never calling this crate),
    // forward-mode for the values, central differences for the gradient.
    let device = Device::cuda(0)?;
    let (batch, feature, class) = (Axis::new("batch"), Axis::new("feature"), Axis::new("class"));
    let x = Tensor::from_slice(
        &[1.0, -1.0, 0.5, 2.0],
        [batch.of(2), feature.of(2)],
        &device,
    )?
    .with_grad();
    let weight = Tensor::from_slice(
        &[0.3, -0.2, 0.1, 0.5, 0.4, -0.3],
        [feature.of(2), class.of(3)],
        &device,
    )?;
    let bias = Tensor::from_slice(&[0.1, -0.1, 0.05], [class.of(3)], &device)?;
    let targets = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [batch.of(2), class.of(3)],
        &device,
    )?;

    let loss = x.linear_cross_entropy_with_logits(
        &weight,
        Some(&bias),
        feature,
        &targets,
        class,
        LinearCrossEntropyOptions::default(),
    )?;
    assert_eq!(loss.shape(), &Shape::new([batch.of(2)])?);
    close(
        "linear_cross_entropy forward",
        &loss.to_vec()?,
        &[1.188473, 2.278166],
    );
    loss.sum(batch)?.backward()?;
    close(
        "linear_cross_entropy gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[-0.1892274, -0.4392002, 0.02558424, 0.6872382],
    );

    let smoothed = x.linear_cross_entropy_with_logits(
        &weight,
        Some(&bias),
        feature,
        &targets,
        class,
        LinearCrossEntropyOptions {
            label_smoothing: 0.1,
        },
    )?;
    close(
        "linear_cross_entropy label_smoothing=0.1 forward",
        &smoothed.to_vec()?,
        &[1.190139, 2.183166],
    );

    let bad_options = x.linear_cross_entropy_with_logits(
        &weight,
        Some(&bias),
        feature,
        &targets,
        class,
        LinearCrossEntropyOptions {
            label_smoothing: 1.0,
        },
    );
    assert!(bad_options.is_err());
    println!("linear_cross_entropy forward, gradient, smoothing, and rejection PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn adaptive_log_softmax_with_loss_matches_independent_oracle_and_gradient() -> Result<()> {
    // Independent oracle: a from-scratch Python reimplementation of the
    // documented head/tail-cluster algorithm (Grave et al., never calling
    // this crate), forward-mode for the loss, central differences on the
    // input for the gradient.
    let device = Device::cuda(0)?;
    let (batch, feature, head, hidden, tail) = (
        Axis::new("batch"),
        Axis::new("feature"),
        Axis::new("head"),
        Axis::new("hidden"),
        Axis::new("tail"),
    );
    let x = Tensor::from_slice(
        &[1.0, -1.0, 0.5, 2.0],
        [batch.of(2), feature.of(2)],
        &device,
    )?
    .with_grad();
    let head_weight = Tensor::from_slice(
        &[0.3, -0.2, 0.1, 0.5, 0.4, -0.3],
        [feature.of(2), head.of(3)],
        &device,
    )?;
    let down = Tensor::from_slice(&[0.7, -0.6], [feature.of(2), hidden.of(1)], &device)?;
    let up = Tensor::from_slice(&[0.2, -0.4], [hidden.of(1), tail.of(2)], &device)?;

    let loss = x.adaptive_log_softmax_with_loss(
        feature,
        &[0, 3],
        &head_weight,
        None,
        &[(down, up)],
        &[2],
        4,
        4.0,
    )?;
    assert_eq!(loss.shape(), &Shape::new([])?);
    close(
        "adaptive_log_softmax_with_loss forward",
        &loss.to_vec()?,
        &[2.009961],
    );
    loss.backward()?;
    close(
        "adaptive_log_softmax_with_loss gradient",
        &x.grad().expect("x gradient").to_vec()?,
        &[-0.1001569, -0.2182897, 0.08118351, 0.2748076],
    );

    let bad_cutoffs =
        x.adaptive_log_softmax_with_loss(feature, &[0, 3], &head_weight, None, &[], &[2], 4, 4.0);
    assert!(bad_cutoffs.is_err());
    let bad_target = x.adaptive_log_softmax_with_loss(
        feature,
        &[0, 9],
        &head_weight,
        None,
        &[(
            Tensor::from_slice(&[0.7, -0.6], [feature.of(2), hidden.of(1)], &device)?,
            Tensor::from_slice(&[0.2, -0.4], [hidden.of(1), tail.of(2)], &device)?,
        )],
        &[2],
        4,
        4.0,
    );
    assert!(bad_target.is_err());
    println!("adaptive_log_softmax_with_loss forward, gradient, and rejections PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn ctc_loss_matches_a_brute_force_alignment_enumeration_oracle_and_gradient() -> Result<()> {
    // Independent oracle: a from-scratch Python brute-force enumeration of
    // every length-5 alignment over {blank, 1, 2} for each row, summing the
    // probability of every alignment that collapses (remove repeats, then
    // blanks) to that row's target, never calling this crate's recursion;
    // central differences on `log_probs` for the gradient.
    let device = Device::cuda(0)?;
    let (time, batch, class) = (Axis::new("time"), Axis::new("batch"), Axis::new("class"));
    let values: Vec<f32> = vec![
        -1.356791, -0.479507, -2.092115, -0.488288, -1.761924, -1.538934, -0.6963312, -1.533338,
        -1.252549, -2.085385, -0.6334775, -1.064217, -1.493144, -0.5757505, -1.546218, -0.888717,
        -1.527629, -0.9894868, -1.631313, -1.513596, -0.5374938, -1.881604, -1.47204, -0.4809473,
        -0.7197948, -1.193529, -1.560655, -1.75642, -1.988898, -0.3703511,
    ];
    let log_probs =
        Tensor::from_slice(&values, [time.of(5), batch.of(2), class.of(3)], &device)?.with_grad();
    let loss = Tensor::ctc_loss(&log_probs, time, class, &[vec![1, 2], vec![2, 1]], 0)?;
    assert_eq!(loss.shape(), &Shape::new([])?);
    close("ctc_loss forward", &loss.to_vec()?, &[1.140298]);
    loss.backward()?;
    close(
        "ctc_loss gradient",
        &log_probs.grad().expect("log_probs gradient").to_vec()?,
        &[
            -0.07417303,
            -0.175827,
            0.0,
            -0.1755447,
            0.0,
            -0.07445528,
            -0.117174,
            -0.1064924,
            -0.02633362,
            -0.04761355,
            -0.01877553,
            -0.1836109,
            -0.06401731,
            -0.1022976,
            -0.08368509,
            -0.08095412,
            -0.0464776,
            -0.1225683,
            -0.03408435,
            -0.01110847,
            -0.2048072,
            -0.04722227,
            -0.1303284,
            -0.07244934,
            -0.1633451,
            0.0,
            -0.08665488,
            -0.09162478,
            -0.1583752,
            0.0,
        ],
    );

    let mismatched_lengths = Tensor::ctc_loss(&log_probs, time, class, &[vec![1, 2], vec![2]], 0);
    assert!(mismatched_lengths.is_err());
    let blank_in_target = Tensor::ctc_loss(&log_probs, time, class, &[vec![0, 1], vec![1, 2]], 0);
    assert!(blank_in_target.is_err());
    println!("ctc_loss forward, gradient, and rejections PASS");
    Ok(())
}

#[test]
fn training_pass_seed_and_next_seed_match_the_documented_splitmix64_mix() {
    // Independent reimplementation of TrainingPass's doc-commented formula: `new` stores its
    // argument as `seed` unmixed (only Trainer::step_training's private `from_run` mixes a run
    // seed and step index into one), and `next_seed` returns
    // splitmix64(seed ^ counter.wrapping_mul(GOLDEN)) for the current draw counter, then
    // increments it. Mirrors, but does not call, `runtime::train`'s private mixer.
    const GOLDEN: u64 = 0x9E3779B97F4A7C15;
    fn splitmix64(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn expected_next_seed(seed: u64, counter: u64) -> u64 {
        splitmix64(seed ^ counter.wrapping_mul(GOLDEN))
    }

    let mut pass = TrainingPass::new(2026);
    assert_eq!(
        pass.seed(),
        2026,
        "TrainingPass::new stores its seed unmixed"
    );
    for counter in 0..4u64 {
        assert_eq!(pass.next_seed(), expected_next_seed(2026, counter));
    }

    // A distinct pass seed diverges from the first draw on; repeated calls on one pass never
    // repeat a seed (checked over the four draws just taken).
    let mut other = TrainingPass::new(7);
    assert_eq!(other.seed(), 7);
    assert_eq!(other.next_seed(), expected_next_seed(7, 0));
    assert_ne!(
        expected_next_seed(2026, 0),
        expected_next_seed(7, 0),
        "distinct pass seeds must not collide on their first draw"
    );
    println!("TrainingPass seed/next_seed formula PASS");
}

#[test]
fn dropout_constructor_rejects_invalid_probabilities() {
    assert!(Dropout::new(0.0).is_ok());
    assert!(Dropout::new(1.0).is_ok());
    assert!(Dropout::new(0.5).is_ok());
    assert!(Dropout::new(-0.01).is_err());
    assert!(Dropout::new(1.01).is_err());
    assert!(Dropout::new(f32::NAN).is_err());
    assert!(Dropout::new(f32::INFINITY).is_err());
    assert!(Dropout::new(f32::NEG_INFINITY).is_err());
    println!("Dropout constructor validation PASS");
}

#[test]
fn trainer_step_training_without_a_seed_errors_before_any_device_work() -> Result<()> {
    // No Device is ever constructed in this test: `step_training` must fail on the missing-seed
    // check before touching the model or the loss closure at all.
    struct NeverTouched;
    impl Module for NeverTouched {
        fn output_shape(&self, _input: &Shape) -> Result<Shape> {
            unreachable!("output_shape must not run before the seed check")
        }
        fn build(&mut self, _input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
            unreachable!("build must not run before the seed check")
        }
        fn forward(&self, _input: &Tensor) -> Result<Tensor> {
            unreachable!("forward must not run before the seed check")
        }
    }

    let mut model = NeverTouched;
    let mut trainer = Trainer::new(SGD::new(0.1)?);
    let error = trainer
        .step_training(&mut model, |_model, _pass| -> Result<Tensor> {
            unreachable!("the loss closure must not run before the seed check")
        })
        .err()
        .expect("step_training without with_seed must error");
    assert!(error.to_string().contains("with_seed"), "{error}");
    assert_eq!(trainer.completed_steps(), 0);
    println!("Trainer::step_training missing-seed guard PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dropout_forward_is_the_identity_regardless_of_probability() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("dropout_eval_feature");
    let values = [3.0_f32, -1.5, 0.0, 42.25, -7.0];
    let input = Tensor::from_slice(&values, [feature.of(values.len())], &device)?;
    let expected = values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    for p in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let output = Dropout::new(p)?.forward(&input)?;
        close(
            &format!("Dropout::forward is the identity at p={p}"),
            &output.to_vec()?,
            &expected,
        );
    }
    println!("Dropout eval-mode identity PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dropout_forward_training_matches_a_hand_derived_mask_oracle_forward_and_gradient() -> Result<()>
{
    // Independent oracle, hand-derived via a one-off Python reimplementation of the device
    // generator's formula (`backend::kernels::uniform_device`'s doc comment): SplitMix64 (the
    // same mixer `TrainingPass` uses) applied to `seed ^ i.wrapping_mul(GOLDEN)` for each flat
    // element index `i`, keeping the top 24 bits of the 64-bit hash divided by 2^24.
    // `TrainingPass::new(2026)`'s first `next_seed()` (draw counter 0) is
    // splitmix64(2026 ^ (0 * GOLDEN)) = 802045514593000271. The first eight
    // `Tensor::uniform_device` draws from that seed (i = 0..8) are
    // [0.8668110966682434, 0.1507960557937622, 0.4262182116508484, 0.3728508949279785,
    //  0.9675002098083496, 0.860875129699707, 0.7212669849395752, 0.5788084268569946],
    // so keeping where the draw is >= 0.5 gives the mask [1, 0, 0, 0, 1, 1, 1, 1].
    let device = Device::cuda(0)?;
    let feature = Axis::new("dropout_oracle_feature");
    let input_values = [1.0_f32, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5];
    let leaf = Tensor::from_slice(&input_values, [feature.of(8)], &device)?.with_grad();

    let dropout = Dropout::new(0.5)?;
    let mut pass = TrainingPass::new(2026);
    let output = dropout.forward_training(&leaf, &mut pass)?;
    // Kept elements are scaled by 1 / (1 - 0.5) = 2; dropped elements are exact zeros.
    let expected = [2.0, -0.0, 0.0, -0.0, 1.0, -1.0, 5.0, -3.0];
    close(
        "Dropout forward_training p=0.5 mask oracle",
        &output.to_vec()?,
        &expected,
    );

    output.mean(feature)?.backward()?;
    // Gradient is mask / (1 - p): the mean's 1/8 upstream, times the mask, times the same 1/(1-p)
    // scale -- 0.25 where kept, exactly 0 where dropped.
    let expected_gradient = [0.25, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25];
    close(
        "Dropout forward_training p=0.5 gradient (mask / (1 - p))",
        &leaf.grad().expect("Dropout input gradient").to_vec()?,
        &expected_gradient,
    );

    // Same pass seed, same forward order: a fresh pass built from the identical seed reproduces
    // the identical draw and therefore a bit-identical masked output.
    let mut repeat_pass = TrainingPass::new(2026);
    let repeat = dropout.forward_training(&leaf, &mut repeat_pass)?;
    close(
        "Dropout forward_training is bit-identical across two runs of the same pass seed",
        &repeat.to_vec()?,
        &expected,
    );
    println!("Dropout forward_training mask oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dropout_forward_training_handles_p_zero_identity_and_p_one_zero_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("dropout_edge_feature");
    let values = [2.0_f32, -3.0, 5.5, -0.25];
    let mut pass = TrainingPass::new(4242);

    let identity_leaf =
        Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    let identity = Dropout::new(0.0)?.forward_training(&identity_leaf, &mut pass)?;
    close(
        "Dropout forward_training p=0 is the identity",
        &identity.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );

    let zero_leaf = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    // Reuses `pass` (now on its second draw) to also show p=1 consumes exactly one draw, the
    // same as any other probability, even though the drawn seed feeds no actual random tensor.
    let zeros = Dropout::new(1.0)?.forward_training(&zero_leaf, &mut pass)?;
    close(
        "Dropout forward_training p=1 is exact zeros",
        &zeros.to_vec()?,
        &[0.0; 4],
    );
    zeros.mean(feature)?.backward()?;
    close(
        "Dropout forward_training p=1 gradient is exactly zero",
        &zero_leaf.grad().expect("p=1 input gradient").to_vec()?,
        &[0.0; 4],
    );
    println!("Dropout forward_training p=0/p=1 edge cases PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dropout_forward_training_statistical_sanity_at_p_quarter() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("dropout_statistical_feature");
    let count = 4096;
    let values: Vec<f32> = (0..count).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let input_mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
    let leaf = Tensor::from_slice(&values, [feature.of(count)], &device)?;
    let mut pass = TrainingPass::new(777);
    let output = Dropout::new(0.25)?
        .forward_training(&leaf, &mut pass)?
        .to_vec()?;

    // Input values never hit exactly zero, so a dropped (zeroed) element is distinguishable from
    // a kept one by value alone.
    let kept = output.iter().filter(|&&v| v != 0.0).count();
    let kept_fraction = kept as f64 / count as f64;
    assert!(
        (kept_fraction - 0.75).abs() < 0.03,
        "kept fraction {kept_fraction} too far from 0.75"
    );

    let output_mean = output.iter().map(|&v| f64::from(v)).sum::<f64>() / output.len() as f64;
    assert!(
        (output_mean - input_mean).abs() < 0.05 * input_mean.abs(),
        "output mean {output_mean} too far from input mean {input_mean}"
    );
    println!(
        "Dropout statistical sanity PASS kept_fraction={kept_fraction:.4} output_mean={output_mean:.4} input_mean={input_mean:.4}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn sequential_threads_one_training_pass_through_two_dropout_layers_with_distinct_masks()
-> Result<()> {
    // Continues the mask oracle above for a second draw: TrainingPass::new(2026)'s SECOND
    // next_seed() (draw counter 1) is splitmix64(2026 ^ (1 * GOLDEN)) = 17653457804415869398,
    // whose first eight Tensor::uniform_device draws are [0.4116435647010803, 0.811515212059021,
    // 0.06670206785202026, 0.7735949158668518, 0.19689422845840454, 0.22576063871383667,
    // 0.5011497735977173, 0.6153182983398438], giving keep mask [0, 1, 0, 1, 0, 0, 1, 1] --
    // distinct from the first layer's [1, 0, 0, 0, 1, 1, 1, 1], so composing both (each scaling by
    // 2) keeps only where BOTH masks are 1: indices 6 and 7 (2.5 -> 5.0 -> 10.0, and
    // -1.5 -> -3.0 -> -6.0) survive at input scale 4.
    let device = Device::cuda(0)?;
    let feature = Axis::new("sequential_dropout_feature");
    let input_values = [1.0_f32, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5];
    let input = Tensor::from_slice(&input_values, [feature.of(8)], &device)?;

    let mut seq = Sequential::new((Dropout::new(0.5)?, Dropout::new(0.5)?));
    seq.build(input.shape(), &device, 0)?;
    let mut pass = TrainingPass::new(2026);
    let output = seq.forward_training(&input, &mut pass)?;

    let expected = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 10.0, -6.0];
    close(
        "Sequential(Dropout, Dropout) threads one pass through both layers with distinct masks",
        &output.to_vec()?,
        &expected,
    );
    println!("Sequential two-Dropout pass-threading PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn trainer_records_pass_seed_only_for_step_training_and_it_follows_the_documented_formula()
-> Result<()> {
    // Independent reimplementation of TrainingPass's doc-commented formula, mirroring
    // training_pass_seed_and_next_seed_match_the_documented_splitmix64_mix above, to recompute the
    // pass seed Trainer::step_training should have recorded for a given (run_seed,
    // completed_steps) pair, without calling the crate's private mixer.
    const GOLDEN: u64 = 0x9E3779B97F4A7C15;
    fn splitmix64(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn expected_pass_seed(run_seed: u64, step_index: u64) -> u64 {
        splitmix64(run_seed ^ step_index.wrapping_mul(GOLDEN))
    }

    // A model whose forward ignores its `input` argument and instead reads a real `Parameter`
    // through a real `Dropout`, so `backward` has a tracked leaf to accumulate into -- the same
    // shape as `trainer_driven_sgd_consumes_a_cosine_annealing_schedule_each_step`'s
    // `ConstantGradientParameter`, above.
    struct DropoutOverParameter(Parameter, Dropout);
    impl Module for DropoutOverParameter {
        fn output_shape(&self, input: &Shape) -> Result<Shape> {
            Ok(input.clone())
        }
        fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
            Ok(input.clone())
        }
        fn forward(&self, _input: &Tensor) -> Result<Tensor> {
            self.1.forward(&self.0.tensor())
        }
        fn forward_training(&self, _input: &Tensor, pass: &mut TrainingPass) -> Result<Tensor> {
            self.1.forward_training(&self.0.tensor(), pass)
        }
        fn named_parameters(&self) -> Vec<(String, Parameter)> {
            vec![("weight".into(), self.0.clone())]
        }
    }

    let device = Device::cuda(0)?;
    let feature = Axis::new("pass_seed_feature");
    let weight = Parameter::new(Tensor::from_slice(
        &[1.0, 1.0, 1.0, 1.0],
        [feature.of(4)],
        &device,
    )?);
    let dummy = Tensor::from_slice(&[0.0; 4], [feature.of(4)], &device)?;
    let mut model = DropoutOverParameter(weight, Dropout::new(0.5)?);

    let plain_step = Trainer::new(SGD::new(0.001)?)
        .step(&mut model, |model| model.forward(&dummy)?.mean(feature))?;
    assert_eq!(
        plain_step.pass_seed(),
        None,
        "step must never record a pass seed"
    );

    const RUN_SEED: u64 = 20260922;
    let mut trainer = Trainer::new(SGD::new(0.001)?).with_seed(RUN_SEED);
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for step_index in 0..2u64 {
        let recorded = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let capture = recorded.clone();
        let train_step = trainer.step_training(&mut model, |model, pass| {
            let out = model.forward_training(&dummy, pass)?;
            *capture.borrow_mut() = out.to_vec()?;
            out.mean(feature)
        })?;
        assert_eq!(
            train_step.pass_seed(),
            Some(expected_pass_seed(RUN_SEED, step_index)),
            "step_training must record splitmix64(run_seed ^ step_index * GOLDEN)"
        );
        outputs.push(recorded.borrow().clone());
    }
    assert_ne!(
        outputs[0], outputs[1],
        "a different step index must draw a different mask"
    );
    println!("Trainer pass_seed formula and step-to-step mask change PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn uniform_device_matches_an_independent_host_splitmix64_oracle_bit_exact() -> Result<()> {
    // Independent reimplementation of `backend::kernels::uniform_device`'s doc-commented formula,
    // written separately from the kernel: plain host `u64` arithmetic, never calling into the
    // crate's own mixer (`runtime::train`'s private `splitmix64`/`GOLDEN`, which the kernel and
    // this oracle each independently mirror -- see
    // `training_pass_seed_and_next_seed_match_the_documented_splitmix64_mix`, above, for the same
    // pattern applied to `TrainingPass`).
    const GOLDEN: u64 = 0x9E3779B97F4A7C15;
    fn splitmix64(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn expected_value(seed: u64, index: u64) -> f32 {
        let hash = splitmix64(seed ^ index.wrapping_mul(GOLDEN));
        let top24 = (hash >> 40) as u32;
        top24 as f32 / (1u32 << 24) as f32
    }

    let device = Device::cuda(0)?;
    let feature = Axis::new("uniform_device_oracle_feature");
    // 300 is not a multiple of the kernel's 128-wide tile, exercising the padded final tile.
    let count = 300usize;
    for &seed in &[0u64, 1, 2026, u64::MAX, 0x1234_5678_9abc_def0] {
        let draw = Tensor::uniform_device([feature.of(count)], seed, &device)?;
        let actual = draw.to_vec()?;
        let expected: Vec<f32> = (0..count as u64).map(|i| expected_value(seed, i)).collect();
        for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "seed={seed} index={index}: device {a} (bits {:x}) != oracle {e} (bits {:x})",
                a.to_bits(),
                e.to_bits()
            );
        }
    }
    println!("Tensor::uniform_device bit-exact host oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn uniform_device_is_deterministic_across_two_launches() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("uniform_device_determinism_feature");
    let count = 777usize;
    let first = Tensor::uniform_device([feature.of(count)], 999_u64, &device)?.to_vec()?;
    let second = Tensor::uniform_device([feature.of(count)], 999_u64, &device)?.to_vec()?;
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "two launches with the same seed must be bit-identical"
    );
    let different = Tensor::uniform_device([feature.of(count)], 1000_u64, &device)?.to_vec()?;
    assert_ne!(
        first, different,
        "a different seed must draw different values"
    );
    println!("Tensor::uniform_device determinism across launches PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn uniform_device_statistical_sanity_over_a_large_draw() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("uniform_device_statistics_feature");
    let count = 200_000usize;
    let values = Tensor::uniform_device([feature.of(count)], 424242_u64, &device)?.to_vec()?;
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    // Uniform[0, 1) has mean 0.5 and variance 1/12 ~= 0.08333.
    assert!((mean - 0.5).abs() < 0.01, "mean {mean} too far from 0.5");
    assert!(
        (variance - 1.0 / 12.0).abs() < 0.005,
        "variance {variance} too far from 1/12"
    );
    println!(
        "Tensor::uniform_device statistical sanity PASS mean={mean:.6} variance={variance:.6}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn batch_norm_forward_training_matches_a_scalar_oracle_forward_and_every_gradient() -> Result<()> {
    // Independent host oracle: BatchNorm's own training-mode normalization,
    // reduced over every axis but the declared feature axis (here batch and
    // spatial together), using the BIASED variance -- PyTorch's own
    // normalization convention -- entirely separate from the crate's own
    // `Tensor::moments`.
    fn batch_norm_reference(
        values: &[f64],
        scale: &[f64],
        bias: &[f64],
        target: &[f64],
    ) -> (Vec<f64>, f64) {
        const BATCH: usize = 3;
        const FEATURE: usize = 2;
        const SPATIAL: usize = 2;
        let mut output = vec![0.0; values.len()];
        for f in 0..FEATURE {
            let selected: Vec<usize> = (0..BATCH)
                .flat_map(|b| (0..SPATIAL).map(move |s| (b * FEATURE + f) * SPATIAL + s))
                .collect();
            let mean = selected.iter().map(|&i| values[i]).sum::<f64>() / selected.len() as f64;
            let variance = selected
                .iter()
                .map(|&i| (values[i] - mean).powi(2))
                .sum::<f64>()
                / selected.len() as f64;
            let inverse_standard_deviation = (variance + 1e-5).sqrt().recip();
            for &i in &selected {
                output[i] = (values[i] - mean) * inverse_standard_deviation * scale[f] + bias[f];
            }
        }
        let loss = mean_squared(&output, target);
        (output, loss)
    }

    let device = Device::cuda(0)?;
    let (batch, feature, spatial) = (
        Axis::new("bn_oracle_batch"),
        Axis::new("bn_oracle_feature"),
        Axis::new("bn_oracle_spatial"),
    );
    let dims = [batch.of(3), feature.of(2), spatial.of(2)];
    let values = vec![
        -1.5_f64, 0.5, 2.0, -0.25, 1.25, -2.0, 0.75, 1.75, -1.0, 0.0, 2.5, -0.5,
    ];
    let scale = vec![1.5_f64, -0.75];
    let bias = vec![0.2_f64, -0.4];
    let target: Vec<f64> = (0..12).map(|i| (i as f64 * 0.23).cos() * 0.5).collect();

    let input = Tensor::from_slice(
        &values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?
    .with_layout([spatial, feature, batch])?
    .with_grad();
    let mut bn = BatchNorm::new(feature);
    bn.build(input.shape(), &device, 0)?;
    bn.parameter("scale")?
        .set_values(&scale.iter().map(|v| *v as f32).collect::<Vec<_>>())?;
    bn.parameter("bias")?
        .set_values(&bias.iter().map(|v| *v as f32).collect::<Vec<_>>())?;

    let mut pass = TrainingPass::new(1234);
    let output = bn.forward_training(&input, &mut pass)?;
    let (expected_output, _) = batch_norm_reference(&values, &scale, &bias, &target);
    close(
        "BatchNorm training forward",
        &output.to_vec()?,
        &expected_output,
    );

    let target_tensor = Tensor::from_slice(
        &target.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?;
    output
        .squared_error(&target_tensor)?
        .mean([batch, feature, spatial])?
        .backward()?;
    close(
        "BatchNorm input gradient",
        &input.grad().unwrap().to_vec()?,
        &central_difference(&values, 1e-4, |candidate| {
            batch_norm_reference(candidate, &scale, &bias, &target).1
        }),
    );
    close(
        "BatchNorm scale gradient",
        &bn.parameter("scale")?.grad().unwrap().to_vec()?,
        &central_difference(&scale, 1e-4, |candidate| {
            batch_norm_reference(&values, candidate, &bias, &target).1
        }),
    );
    close(
        "BatchNorm bias gradient",
        &bn.parameter("bias")?.grad().unwrap().to_vec()?,
        &central_difference(&bias, 1e-4, |candidate| {
            batch_norm_reference(&values, &scale, candidate, &target).1
        }),
    );
    println!("BatchNorm forward_training scalar oracle forward+gradient PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn trainer_step_training_commits_batch_norm_running_statistics_and_leaves_them_untouched_on_failure()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (
        Axis::new("bn_trainer_batch"),
        Axis::new("bn_trainer_feature"),
    );
    let dims = [batch.of(4), feature.of(2)];
    // `affine(false)` keeps the post-normalization scale/bias fixed at
    // identity, since the fixed-input eval oracle below assumes them
    // untouched by the three SGD steps -- the running-statistics EMA under
    // test is independent of the affine transform either way.
    let mut bn = BatchNorm::new(feature).momentum(Some(0.5))?.affine(false);
    bn.build(&Shape::new(dims)?, &device, 0)?;

    // Three fixed steps of input, one 4x2 batch each, chosen so a plain
    // f64 replay of BatchNorm's documented EMA formula is trivial to
    // hand-check: momentum=0.5, biased variance for normalization, UNBIASED
    // running_var (n=4 samples per feature per step -> factor 4/3).
    let steps: [[f64; 8]; 3] = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        [0.5, 1.5, -0.5, 2.5, 3.5, -1.5, 4.5, -2.5],
    ];
    fn expected_after(steps: &[[f64; 8]], momentum: f64) -> (Vec<f64>, Vec<f64>, f64) {
        let mut running_mean = vec![0.0; 2];
        let mut running_var = vec![1.0; 2];
        let mut count = 0.0;
        for step in steps {
            count += 1.0;
            for f in 0..2 {
                let samples: Vec<f64> = (0..4).map(|b| step[b * 2 + f]).collect();
                let mean = samples.iter().sum::<f64>() / 4.0;
                let biased_var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
                let unbiased_var = biased_var * 4.0 / 3.0;
                running_mean[f] = (1.0 - momentum) * running_mean[f] + momentum * mean;
                running_var[f] = (1.0 - momentum) * running_var[f] + momentum * unbiased_var;
            }
        }
        (running_mean, running_var, count)
    }

    let mut trainer = Trainer::new(SGD::new(0.01)?).with_seed(2026);
    let mut final_mean = vec![0.0; 2];
    let mut final_var = vec![0.0; 2];
    for (index, step_values) in steps.iter().enumerate() {
        let values: Vec<f32> = step_values.iter().map(|v| *v as f32).collect();
        // `with_grad`: with `affine(false)` the model itself has no
        // parameters, so `backward` needs a tracked leaf somewhere in the
        // loss graph to run at all; the running-statistics EMA under test
        // never depends on it.
        let input = Tensor::from_slice(&values, dims, &device)?.with_grad();
        let step = trainer.step_training(&mut bn, |model, pass| {
            model.forward_training(&input, pass)?.mean([batch, feature])
        })?;
        assert_eq!(
            step.committed_states(),
            3,
            "step {index}: running_mean, running_var, num_batches_tracked"
        );
        let (expected_mean, expected_var, expected_count) = expected_after(&steps[..=index], 0.5);
        close(
            &format!("BatchNorm running_mean after step {index}"),
            &bn.state("running_mean")?.tensor().to_vec()?,
            &expected_mean,
        );
        close(
            &format!("BatchNorm running_var after step {index}"),
            &bn.state("running_var")?.tensor().to_vec()?,
            &expected_var,
        );
        assert_eq!(
            bn.state("num_batches_tracked")?.tensor().item()?,
            expected_count as f32,
            "step {index}: num_batches_tracked"
        );
        final_mean = expected_mean;
        final_var = expected_var;
    }

    // Evaluation with the committed running statistics: scale=1, bias=0
    // (never set, so still their identity default), so the expected output
    // is the plain per-feature normalization against the last step's
    // committed running_mean/running_var.
    let eval_input_values: Vec<f64> = vec![10.0, -5.0, 3.0, 2.0, -1.0, 6.0, 0.0, -2.0];
    let eval_input = Tensor::from_slice(
        &eval_input_values
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<_>>(),
        dims,
        &device,
    )?;
    let eval_output = bn.forward(&eval_input)?.to_vec()?;
    let mut expected_eval = vec![0.0; 8];
    for b in 0..4 {
        for f in 0..2 {
            let idx = b * 2 + f;
            expected_eval[idx] =
                (eval_input_values[idx] - final_mean[f]) / (final_var[f] + 1e-5).sqrt();
        }
    }
    close(
        "BatchNorm eval uses committed running statistics",
        &eval_output,
        &expected_eval,
    );

    // A step whose loss errors AFTER forward_training has already staged
    // updates must leave every State untouched: `step_training` only ever
    // commits once the whole step -- loss, backward, optimizer step -- has
    // succeeded, so an error from the loss closure itself never reaches
    // `TrainingPass::commit`.
    let before_mean = bn
        .state("running_mean")?
        .tensor()
        .to_vec()?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let before_var = bn
        .state("running_var")?
        .tensor()
        .to_vec()?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let before_count = bn.state("num_batches_tracked")?.tensor().item()?;
    let failing_input = Tensor::from_slice(&[9.0; 8], dims, &device)?;
    let failed = trainer.step_training(&mut bn, |model, pass| {
        let _ = model.forward_training(&failing_input, pass)?;
        Err::<Tensor, _>("injected failure after forward".into())
    });
    assert!(failed.is_err(), "the injected failure must propagate");
    close(
        "BatchNorm running_mean unchanged after a failed step",
        &bn.state("running_mean")?.tensor().to_vec()?,
        &before_mean,
    );
    close(
        "BatchNorm running_var unchanged after a failed step",
        &bn.state("running_var")?.tensor().to_vec()?,
        &before_var,
    );
    assert_eq!(
        bn.state("num_batches_tracked")?.tensor().item()?,
        before_count,
        "num_batches_tracked unchanged after a failed step"
    );
    println!("Trainer::step_training BatchNorm running-statistics commit-on-success PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn batch_norm_without_tracking_uses_batch_statistics_in_eval_too() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (
        Axis::new("bn_no_track_batch"),
        Axis::new("bn_no_track_feature"),
    );
    let dims = [batch.of(4), feature.of(2)];
    let values: Vec<f64> = vec![1.0, 5.0, 2.0, -3.0, 0.0, 4.0, -1.0, 2.0];
    let input = Tensor::from_slice(
        &values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
        dims,
        &device,
    )?;
    let mut bn = BatchNorm::new(feature)
        .track_running_stats(false)
        .affine(false);
    bn.build(input.shape(), &device, 0)?;
    assert!(
        bn.states().is_empty(),
        "track_running_stats(false) must allocate no State"
    );

    fn batch_stats_reference(values: &[f64]) -> Vec<f64> {
        const BATCH: usize = 4;
        const FEATURE: usize = 2;
        let mut output = vec![0.0; values.len()];
        for f in 0..FEATURE {
            let samples: Vec<f64> = (0..BATCH).map(|b| values[b * FEATURE + f]).collect();
            let mean = samples.iter().sum::<f64>() / BATCH as f64;
            let variance = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / BATCH as f64;
            let inverse_standard_deviation = (variance + 1e-5).sqrt().recip();
            for b in 0..BATCH {
                output[b * FEATURE + f] =
                    (values[b * FEATURE + f] - mean) * inverse_standard_deviation;
            }
        }
        output
    }

    let expected = batch_stats_reference(&values);
    close(
        "BatchNorm eval batch statistics (forward)",
        &bn.forward(&input)?.to_vec()?,
        &expected,
    );
    let mut pass = TrainingPass::new(1);
    close(
        "BatchNorm eval batch statistics (forward_training)",
        &bn.forward_training(&input, &mut pass)?.to_vec()?,
        &expected,
    );
    assert_eq!(
        pass.commit(),
        0,
        "no state to commit when track_running_stats is false"
    );
    println!("BatchNorm track_running_stats=false eval-uses-batch-statistics PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn batch_norm_momentum_none_uses_the_cumulative_moving_average() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, feature) = (
        Axis::new("bn_cumulative_batch"),
        Axis::new("bn_cumulative_feature"),
    );
    let dims = [batch.of(2), feature.of(1)];
    let mut bn = BatchNorm::new(feature).momentum(None)?.affine(false);
    bn.build(&Shape::new(dims)?, &device, 0)?;

    let steps: [[f32; 2]; 3] = [[1.0, 3.0], [10.0, 20.0], [-4.0, 4.0]];
    // Cumulative average with running_mean starting at 0.0: after step k
    // (1-indexed), factor_k = 1/k, so this is an ordinary online mean/EMA
    // update against that factor -- independently replayed here in f64.
    let mut expected_mean = 0.0_f64;
    let mut expected_var = 1.0_f64;
    for (index, step) in steps.iter().enumerate() {
        let batch_mean = (f64::from(step[0]) + f64::from(step[1])) / 2.0;
        let biased_var = ((f64::from(step[0]) - batch_mean).powi(2)
            + (f64::from(step[1]) - batch_mean).powi(2))
            / 2.0;
        let unbiased_var = biased_var * 2.0; // n=2 -> n / (n - 1) = 2
        let factor = 1.0 / (index as f64 + 1.0);
        expected_mean = (1.0 - factor) * expected_mean + factor * batch_mean;
        expected_var = (1.0 - factor) * expected_var + factor * unbiased_var;

        let input = Tensor::from_slice(step, dims, &device)?;
        let mut pass = TrainingPass::new(index as u64);
        let _ = bn.forward_training(&input, &mut pass)?;
        assert_eq!(pass.commit(), 3);
        close(
            &format!("cumulative running_mean after step {index}"),
            &bn.state("running_mean")?.tensor().to_vec()?,
            &[expected_mean],
        );
        close(
            &format!("cumulative running_var after step {index}"),
            &bn.state("running_var")?.tensor().to_vec()?,
            &[expected_var],
        );
        assert_eq!(
            bn.state("num_batches_tracked")?.tensor().item()?,
            (index + 1) as f32
        );
    }
    println!("BatchNorm momentum=None cumulative moving average PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn instance_norm_track_running_stats_matches_a_hand_derived_running_statistics_oracle() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (batch, channel, spatial) = (
        Axis::new("in_running_batch"),
        Axis::new("in_running_channel"),
        Axis::new("in_running_spatial"),
    );
    let dims = [batch.of(2), channel.of(2), spatial.of(2)];
    let values: Vec<f32> = vec![
        1.0, 3.0, 5.0, 7.0, // batch 0: channel 0 = [1, 3], channel 1 = [5, 7]
        2.0, 0.0, -1.0, 3.0, // batch 1: channel 0 = [2, 0], channel 1 = [-1, 3]
    ];
    let input = Tensor::from_slice(&values, dims, &device)?;
    let mut instance = InstanceNorm::new(channel, [spatial])?
        .affine(false)
        .track_running_stats(true)
        .momentum(Some(0.5))?;
    instance.build(input.shape(), &device, 0)?;

    // Independent host replay of InstanceNorm's documented running-statistics
    // formula (see its own doc comment): per-instance (batch, channel)
    // mean/var over the spatial axis only, the variance unbiased with the
    // spatial sample size ALONE (n=2 -> n / (n - 1) = 2, not the whole
    // batch's count), then averaged over batch before the momentum EMA.
    fn instance_stats(pair: [f64; 2]) -> (f64, f64) {
        let mean = (pair[0] + pair[1]) / 2.0;
        let biased = ((pair[0] - mean).powi(2) + (pair[1] - mean).powi(2)) / 2.0;
        (mean, biased * 2.0)
    }
    let (mean_b0_c0, var_b0_c0) = instance_stats([1.0, 3.0]);
    let (mean_b0_c1, var_b0_c1) = instance_stats([5.0, 7.0]);
    let (mean_b1_c0, var_b1_c0) = instance_stats([2.0, 0.0]);
    let (mean_b1_c1, var_b1_c1) = instance_stats([-1.0, 3.0]);
    let pooled_mean = [
        (mean_b0_c0 + mean_b1_c0) / 2.0,
        (mean_b0_c1 + mean_b1_c1) / 2.0,
    ];
    let pooled_var = [(var_b0_c0 + var_b1_c0) / 2.0, (var_b0_c1 + var_b1_c1) / 2.0];
    let momentum = 0.5_f64;
    let expected_mean = [momentum * pooled_mean[0], momentum * pooled_mean[1]];
    let expected_var = [
        0.5 * 1.0 + momentum * pooled_var[0],
        0.5 * 1.0 + momentum * pooled_var[1],
    ];

    let mut pass = TrainingPass::new(99);
    let _ = instance.forward_training(&input, &mut pass)?;
    assert_eq!(pass.commit(), 3);
    close(
        "InstanceNorm running_mean",
        &instance.state("running_mean")?.tensor().to_vec()?,
        &expected_mean,
    );
    close(
        "InstanceNorm running_var",
        &instance.state("running_var")?.tensor().to_vec()?,
        &expected_var,
    );
    assert_eq!(instance.state("num_batches_tracked")?.tensor().item()?, 1.0);
    println!("InstanceNorm track_running_stats running-statistics oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn named_states_thread_batch_norm_through_sequential_slot_paths() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("seq_states_feature");
    let dims = [feature.of(3)];
    let mut sequential = Sequential::new((
        BatchNorm::new(feature),
        BatchNorm::new(feature).track_running_stats(false),
    ));
    sequential.build(&Shape::new(dims)?, &device, 0)?;
    let mut paths: Vec<String> = sequential
        .named_states()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    paths.sort();
    assert_eq!(
        paths,
        ["0.num_batches_tracked", "0.running_mean", "0.running_var"]
    );
    println!("named_states thread through Sequential slot paths PASS");
    Ok(())
}

// --- ChannelDropout ------------------------------------------------------------------------

#[test]
fn channel_dropout_constructor_rejects_invalid_configurations() {
    let channel = Axis::new("cd_ctor_channel");
    let spatial = Axis::new("cd_ctor_spatial");
    let other_spatial = Axis::new("cd_ctor_spatial_2");
    assert!(ChannelDropout::new(0.5, channel, [spatial]).is_ok());
    assert!(ChannelDropout::new(0.5, channel, [spatial, other_spatial]).is_ok());
    assert!(ChannelDropout::new(-0.01, channel, [spatial]).is_err());
    assert!(ChannelDropout::new(1.01, channel, [spatial]).is_err());
    assert!(ChannelDropout::new(f32::NAN, channel, [spatial]).is_err());
    // Empty spatial list.
    assert!(ChannelDropout::new(0.5, channel, Vec::<Axis>::new()).is_err());
    // Channel duplicated into the spatial list.
    assert!(ChannelDropout::new(0.5, channel, [channel]).is_err());
    // Spatial axis listed twice.
    assert!(ChannelDropout::new(0.5, channel, [spatial, spatial]).is_err());
    println!("ChannelDropout constructor validation PASS");
}

#[test]
#[ignore = "requires CUDA"]
fn channel_dropout_rejects_missing_axes_before_any_device_work() -> Result<()> {
    let channel = Axis::new("cd_missing_channel");
    let spatial = Axis::new("cd_missing_spatial");
    let unrelated = Axis::new("cd_missing_unrelated");
    let device = Device::cuda(0)?;
    let input = Tensor::from_slice(&[1.0, 2.0], [unrelated.of(2)], &device)?;
    let dropout = ChannelDropout::new(0.5, channel, [spatial])?;
    assert!(dropout.output_shape(input.shape()).is_err());
    assert!(dropout.forward(&input).is_err());
    println!("ChannelDropout missing-axis rejection PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn channel_dropout_forward_is_the_identity() -> Result<()> {
    let device = Device::cuda(0)?;
    let batch = Axis::new("cd_eval_batch");
    let channel = Axis::new("cd_eval_channel");
    let spatial = Axis::new("cd_eval_spatial");
    let values: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let input = Tensor::from_slice(
        &values,
        [batch.of(2), channel.of(2), spatial.of(2)],
        &device,
    )?;
    let output = ChannelDropout::new(0.5, channel, [spatial])?.forward(&input)?;
    close(
        "ChannelDropout::forward is the identity",
        &output.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );
    println!("ChannelDropout eval-mode identity PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn channel_dropout_forward_training_matches_a_broadcast_mask_oracle_forward_and_gradient()
-> Result<()> {
    // Independent oracle: `TrainingPass::new(5050)`'s first `next_seed()` is
    // 13144240536140775429. Its first four `Tensor::uniform_device` draws (one per (batch,
    // channel) combination, since `spatial` is excluded from the mask draw) are
    // [0.0789562463760376, 0.3713812828063965, 0.6667241454124451, 0.11031758785247803], so
    // keeping where the draw is >= 0.5 gives keep_small = [0, 0, 1, 0] -- only (batch=1,
    // channel=0) survives -- broadcast over the two `spatial` positions of that combination.
    let device = Device::cuda(0)?;
    let batch = Axis::new("cd_oracle_batch");
    let channel = Axis::new("cd_oracle_channel");
    let spatial = Axis::new("cd_oracle_spatial");
    let input_values: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let leaf = Tensor::from_slice(
        &input_values,
        [batch.of(2), channel.of(2), spatial.of(2)],
        &device,
    )?
    .with_grad();

    let dropout = ChannelDropout::new(0.5, channel, [spatial])?;
    let mut pass = TrainingPass::new(5050);
    let output = dropout.forward_training(&leaf, &mut pass)?;
    let expected = [0.0, 0.0, 0.0, 0.0, 10.0, 12.0, 0.0, 0.0];
    close(
        "ChannelDropout forward_training p=0.5 broadcast mask oracle",
        &output.to_vec()?,
        &expected,
    );

    output.mean([batch, channel, spatial])?.backward()?;
    let expected_gradient = [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0];
    close(
        "ChannelDropout forward_training gradient (mask / (1 - p), broadcast over spatial)",
        &leaf
            .grad()
            .expect("ChannelDropout input gradient")
            .to_vec()?,
        &expected_gradient,
    );

    let mut repeat_pass = TrainingPass::new(5050);
    let repeat = dropout.forward_training(&leaf, &mut repeat_pass)?;
    close(
        "ChannelDropout forward_training is bit-identical across two runs of the same pass seed",
        &repeat.to_vec()?,
        &expected,
    );
    println!("ChannelDropout forward_training broadcast mask oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn channel_dropout_forward_training_handles_p_zero_identity_and_p_one_zero_gradient() -> Result<()>
{
    let device = Device::cuda(0)?;
    let batch = Axis::new("cd_edge_batch");
    let channel = Axis::new("cd_edge_channel");
    let spatial = Axis::new("cd_edge_spatial");
    let values: Vec<f32> = vec![2.0, -3.0, 5.5, -0.25];
    let mut pass = TrainingPass::new(1234);

    let identity_leaf = Tensor::from_slice(
        &values,
        [batch.of(2), channel.of(1), spatial.of(2)],
        &device,
    )?
    .with_grad();
    let identity = ChannelDropout::new(0.0, channel, [spatial])?
        .forward_training(&identity_leaf, &mut pass)?;
    close(
        "ChannelDropout forward_training p=0 is the identity",
        &identity.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );

    let zero_leaf = Tensor::from_slice(
        &values,
        [batch.of(2), channel.of(1), spatial.of(2)],
        &device,
    )?
    .with_grad();
    let zeros =
        ChannelDropout::new(1.0, channel, [spatial])?.forward_training(&zero_leaf, &mut pass)?;
    close(
        "ChannelDropout forward_training p=1 is exact zeros",
        &zeros.to_vec()?,
        &[0.0; 4],
    );
    zeros.mean([batch, channel, spatial])?.backward()?;
    close(
        "ChannelDropout forward_training p=1 gradient is exactly zero",
        &zero_leaf.grad().expect("p=1 input gradient").to_vec()?,
        &[0.0; 4],
    );
    println!("ChannelDropout forward_training p=0/p=1 edge cases PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn channel_dropout_statistical_sanity_kept_fraction_is_per_channel_not_per_element() -> Result<()> {
    // A whole (batch, channel) combination is kept or dropped together: the fraction of kept
    // COMBINATIONS should track (1 - p), and every combination's `spatial` positions must agree.
    let device = Device::cuda(0)?;
    let batch = Axis::new("cd_stat_batch");
    let channel = Axis::new("cd_stat_channel");
    let spatial = Axis::new("cd_stat_spatial");
    let combinations = 2048;
    let spatial_extent = 4;
    let values: Vec<f32> = (0..combinations * spatial_extent)
        .map(|i| 1.0 + (i as f32) * 0.0001)
        .collect();
    let leaf = Tensor::from_slice(
        &values,
        [
            batch.of(combinations),
            channel.of(1),
            spatial.of(spatial_extent),
        ],
        &device,
    )?;
    let mut pass = TrainingPass::new(999);
    let output = ChannelDropout::new(0.25, channel, [spatial])?
        .forward_training(&leaf, &mut pass)?
        .to_vec()?;

    let mut agreeing = 0;
    let mut kept_combinations = 0;
    for combination in 0..combinations {
        let row = &output[combination * spatial_extent..(combination + 1) * spatial_extent];
        let all_kept = row.iter().all(|&v| v != 0.0);
        let all_dropped = row.iter().all(|&v| v == 0.0);
        if all_kept || all_dropped {
            agreeing += 1;
        }
        if all_kept {
            kept_combinations += 1;
        }
    }
    assert_eq!(
        agreeing, combinations,
        "every spatial position of one (batch, channel) combination must share one decision"
    );
    let kept_fraction = kept_combinations as f64 / combinations as f64;
    assert!(
        (kept_fraction - 0.75).abs() < 0.03,
        "kept combination fraction {kept_fraction} too far from 0.75"
    );
    println!(
        "ChannelDropout statistical sanity PASS kept_fraction={kept_fraction:.4} (broadcast agreement 100%)"
    );
    Ok(())
}

// --- AlphaDropout ----------------------------------------------------------------------------

#[test]
fn alpha_dropout_constructor_rejects_invalid_probabilities() {
    assert!(AlphaDropout::new(0.0).is_ok());
    assert!(AlphaDropout::new(1.0).is_ok());
    assert!(AlphaDropout::new(0.3).is_ok());
    assert!(AlphaDropout::new(-0.01).is_err());
    assert!(AlphaDropout::new(1.01).is_err());
    assert!(AlphaDropout::new(f32::NAN).is_err());
    assert!(AlphaDropout::new(f32::INFINITY).is_err());
    println!("AlphaDropout constructor validation PASS");
}

#[test]
#[ignore = "requires CUDA"]
fn alpha_dropout_forward_is_the_identity() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("ad_eval_feature");
    let values = [3.0_f32, -1.5, 0.0, 42.25];
    let input = Tensor::from_slice(&values, [feature.of(values.len())], &device)?;
    let output = AlphaDropout::new(0.4)?.forward(&input)?;
    close(
        "AlphaDropout::forward is the identity",
        &output.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );
    println!("AlphaDropout eval-mode identity PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn alpha_dropout_forward_training_matches_pytorch_affine_formula_forward_and_gradient() -> Result<()>
{
    // Independent oracle, bit-for-bit PyTorch's `_alpha_dropout_impl` (`aten/src/ATen/native/
    // Dropout.cpp`): `alpha = 1.7580993408473766` (its f32 rounding, matching
    // `ALPHA_DROPOUT_ALPHA`, is 1.7580993175506592), `a = 1 / sqrt((alpha^2 * p + 1) * (1 -
    // p))`. `TrainingPass::new(90210)`'s first `next_seed()` is 8618671413115748481; its first
    // four `Tensor::uniform_device` draws feed `keep = draw >= p` at `p = 0.3`, giving `keep =
    // [0, 1, 0, 1]`. At `p = 0.3`, `a = 0.8609525561332703` and the dropped-element constant
    // `alpha * a * (p - 1) = -1.059548020362854`.
    let device = Device::cuda(0)?;
    let feature = Axis::new("ad_oracle_feature");
    let input_values = [1.0_f32, -2.0, 0.5, -0.5];
    let leaf = Tensor::from_slice(&input_values, [feature.of(4)], &device)?.with_grad();

    let dropout = AlphaDropout::new(0.3)?;
    let mut pass = TrainingPass::new(90210);
    let output = dropout.forward_training(&leaf, &mut pass)?;
    let expected = [
        -1.059548020362854,
        -1.2678130865097046,
        -1.059548020362854,
        0.023615717887878418,
    ];
    close(
        "AlphaDropout forward_training p=0.3 affine oracle",
        &output.to_vec()?,
        &expected,
    );

    output.mean(feature)?.backward()?;
    let expected_gradient = [0.0, 0.21523813903331757, 0.0, 0.21523813903331757];
    close(
        "AlphaDropout forward_training gradient (a * keep / count)",
        &leaf.grad().expect("AlphaDropout input gradient").to_vec()?,
        &expected_gradient,
    );

    let mut repeat_pass = TrainingPass::new(90210);
    let repeat = dropout.forward_training(&leaf, &mut repeat_pass)?;
    close(
        "AlphaDropout forward_training is bit-identical across two runs of the same pass seed",
        &repeat.to_vec()?,
        &expected,
    );
    println!("AlphaDropout forward_training affine oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn alpha_dropout_forward_training_handles_p_zero_identity_and_p_one_zero_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("ad_edge_feature");
    let values = [2.0_f32, -3.0, 5.5, -0.25];
    let mut pass = TrainingPass::new(4321);

    let identity_leaf =
        Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    let identity = AlphaDropout::new(0.0)?.forward_training(&identity_leaf, &mut pass)?;
    close(
        "AlphaDropout forward_training p=0 is the identity",
        &identity.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );

    let zero_leaf = Tensor::from_slice(&values, [feature.of(values.len())], &device)?.with_grad();
    let zeros = AlphaDropout::new(1.0)?.forward_training(&zero_leaf, &mut pass)?;
    close(
        "AlphaDropout forward_training p=1 is exact zeros",
        &zeros.to_vec()?,
        &[0.0; 4],
    );
    zeros.mean(feature)?.backward()?;
    close(
        "AlphaDropout forward_training p=1 gradient is exactly zero",
        &zero_leaf.grad().expect("p=1 input gradient").to_vec()?,
        &[0.0; 4],
    );
    println!("AlphaDropout forward_training p=0/p=1 edge cases PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn alpha_dropout_statistical_sanity_preserves_mean_and_variance() -> Result<()> {
    // Self-normalizing networks assume zero mean, unit variance activations; alpha dropout's
    // whole point is preserving both through training. Starting from `Tensor::normal(..., 0.0,
    // 1.0, ...)` (its own oracle: `docs/design/library.md`'s Box-Muller construction over the
    // host xorshift stream), the output mean and variance should stay close to 0 and 1.
    let device = Device::cuda(0)?;
    let feature = Axis::new("ad_stat_feature");
    let count = 200_000;
    let leaf = Tensor::normal([feature.of(count)], 13, 0.0, 1.0, &device)?;
    let mut pass = TrainingPass::new(55);
    let output = AlphaDropout::new(0.2)?
        .forward_training(&leaf, &mut pass)?
        .to_vec()?;
    let mean = output.iter().map(|&v| f64::from(v)).sum::<f64>() / output.len() as f64;
    let variance = output
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / output.len() as f64;
    assert!(mean.abs() < 0.05, "output mean {mean} too far from 0");
    assert!(
        (variance - 1.0).abs() < 0.1,
        "output variance {variance} too far from 1"
    );
    println!("AlphaDropout statistical sanity PASS mean={mean:.4} variance={variance:.4}");
    Ok(())
}

// --- FeatureAlphaDropout ----------------------------------------------------------------------

#[test]
fn feature_alpha_dropout_constructor_rejects_invalid_configurations() {
    let channel = Axis::new("fad_ctor_channel");
    let spatial = Axis::new("fad_ctor_spatial");
    assert!(FeatureAlphaDropout::new(0.4, channel, [spatial]).is_ok());
    assert!(FeatureAlphaDropout::new(-0.01, channel, [spatial]).is_err());
    assert!(FeatureAlphaDropout::new(1.01, channel, [spatial]).is_err());
    assert!(FeatureAlphaDropout::new(0.4, channel, Vec::<Axis>::new()).is_err());
    assert!(FeatureAlphaDropout::new(0.4, channel, [channel]).is_err());
    println!("FeatureAlphaDropout constructor validation PASS");
}

#[test]
#[ignore = "requires CUDA"]
fn feature_alpha_dropout_forward_training_matches_a_broadcast_affine_oracle_forward_and_gradient()
-> Result<()> {
    // Same construction as `channel_dropout_forward_training_matches_a_broadcast_mask_oracle_
    // forward_and_gradient`, with `AlphaDropout`'s affine formula in place of the inverted-
    // dropout scale. `TrainingPass::new(31415)`'s first `next_seed()` is 1944086767984051203;
    // its first four `Tensor::uniform_device` draws feed `keep = draw >= p` at `p = 0.4`, giving
    // keep_small = [1, 1, 0, 1] -- every (batch, channel) combination except (batch=1,
    // channel=0) survives, broadcast over `spatial`. `a = 0.8632826209068298`, constant term
    // `alpha * a * (p - 1) = -0.9106419682502747`.
    let device = Device::cuda(0)?;
    let batch = Axis::new("fad_oracle_batch");
    let channel = Axis::new("fad_oracle_channel");
    let spatial = Axis::new("fad_oracle_spatial");
    let input_values: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let leaf = Tensor::from_slice(
        &input_values,
        [batch.of(2), channel.of(2), spatial.of(2)],
        &device,
    )?
    .with_grad();

    let dropout = FeatureAlphaDropout::new(0.4, channel, [spatial])?;
    let mut pass = TrainingPass::new(31415);
    let output = dropout.forward_training(&leaf, &mut pass)?;
    let expected = [
        1.4703772068023682,
        2.3336598873138428,
        3.1969425678253174,
        4.060225009918213,
        -0.9106419682502747,
        -0.9106419682502747,
        6.6500725746154785,
        7.513355731964111,
    ];
    close(
        "FeatureAlphaDropout forward_training p=0.4 broadcast affine oracle",
        &output.to_vec()?,
        &expected,
    );

    output.mean([batch, channel, spatial])?.backward()?;
    let expected_gradient = [
        0.10791032761335373,
        0.10791032761335373,
        0.10791032761335373,
        0.10791032761335373,
        0.0,
        0.0,
        0.10791032761335373,
        0.10791032761335373,
    ];
    close(
        "FeatureAlphaDropout forward_training gradient (a * keep / count, broadcast over spatial)",
        &leaf
            .grad()
            .expect("FeatureAlphaDropout input gradient")
            .to_vec()?,
        &expected_gradient,
    );
    println!("FeatureAlphaDropout forward_training broadcast affine oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn feature_alpha_dropout_forward_is_the_identity() -> Result<()> {
    let device = Device::cuda(0)?;
    let batch = Axis::new("fad_eval_batch");
    let channel = Axis::new("fad_eval_channel");
    let spatial = Axis::new("fad_eval_spatial");
    let values: Vec<f32> = (1..=4).map(|v| v as f32).collect();
    let input = Tensor::from_slice(
        &values,
        [batch.of(2), channel.of(1), spatial.of(2)],
        &device,
    )?;
    let output = FeatureAlphaDropout::new(0.4, channel, [spatial])?.forward(&input)?;
    close(
        "FeatureAlphaDropout::forward is the identity",
        &output.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );
    println!("FeatureAlphaDropout eval-mode identity PASS");
    Ok(())
}

// --- RReLU -------------------------------------------------------------------------------------

#[test]
fn rrelu_builder_rejects_invalid_bounds() -> Result<()> {
    assert!(RReLU::new().lower(-0.01).is_err());
    assert!(RReLU::new().lower(f32::NAN).is_err());
    assert!(RReLU::new().upper(f32::NAN).is_err());
    // Default upper is 1/3: raising lower past it is rejected immediately, before `upper` is
    // ever touched.
    assert!(RReLU::new().lower(0.5).is_err());
    // Default lower is 1/8: lowering upper below it is rejected too.
    assert!(RReLU::new().upper(0.05).is_err());
    // Raise lower first (still under the default upper), then raise upper past the new lower.
    assert!(RReLU::new().lower(0.1)?.upper(0.4).is_ok());
    println!("RReLU builder validation PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn rrelu_forward_is_a_fixed_leaky_relu_at_the_mean_slope() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("rrelu_eval_feature");
    let values = [2.0_f32, -3.0, 0.0, 5.0, -1.5, -0.25];
    let input = Tensor::from_slice(&values, [feature.of(values.len())], &device)?;
    // Default lower = 1/8, upper = 1/3: fixed eval slope (1/8 + 1/3) / 2 = 0.2291666...
    let output = RReLU::new().forward(&input)?;
    let expected: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v > 0.0 {
                v as f64
            } else {
                v as f64 * (1.0 / 8.0 + 1.0 / 3.0) / 2.0
            }
        })
        .collect();
    close(
        "RReLU::forward at PyTorch defaults",
        &output.to_vec()?,
        &expected,
    );
    close(
        "RReLU::default() matches RReLU::new()",
        &RReLU::default().forward(&input)?.to_vec()?,
        &expected,
    );

    let custom = RReLU::new().lower(0.1)?.upper(0.4)?;
    let custom_output = custom.forward(&input)?;
    let custom_expected = [2.0, -0.75, 0.0, 5.0, -0.375, -0.0625];
    close(
        "RReLU::forward at custom lower/upper is leaky_relu((lower + upper) / 2)",
        &custom_output.to_vec()?,
        &custom_expected,
    );
    println!("RReLU eval-mode fixed-slope PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn rrelu_forward_training_matches_a_per_element_slope_oracle_forward_and_gradient() -> Result<()> {
    // Independent oracle: `TrainingPass::new(2718)`'s first `next_seed()` is
    // 11642600825927962904. Its first six `Tensor::uniform_device` draws feed `slope = lower +
    // (upper - lower) * draw` at `lower = 0.1, upper = 0.4`: slope = [0.13487684726715088,
    // 0.24346089363098145, 0.12079253792762756, 0.2371707558631897, 0.19090664386749268,
    // 0.2531281113624573]. Positive elements (including none here at exactly zero) pass
    // through; non-positive elements (`x <= 0`, so the zero element too) scale by their slope.
    let device = Device::cuda(0)?;
    let feature = Axis::new("rrelu_oracle_feature");
    let input_values = [2.0_f32, -3.0, 0.0, 5.0, -1.5, -0.25];
    let leaf = Tensor::from_slice(&input_values, [feature.of(6)], &device)?.with_grad();

    let rrelu = RReLU::new().lower(0.1)?.upper(0.4)?;
    let mut pass = TrainingPass::new(2718);
    let output = rrelu.forward_training(&leaf, &mut pass)?;
    let expected = [
        2.0,
        -0.7303826808929443,
        0.0,
        5.0,
        -0.286359965801239,
        -0.06328202784061432,
    ];
    close(
        "RReLU forward_training per-element slope oracle",
        &output.to_vec()?,
        &expected,
    );

    output.mean(feature)?.backward()?;
    let expected_gradient = [
        0.1666666716337204,
        0.040576815605163574,
        0.020132089033722878,
        0.1666666716337204,
        0.03181777521967888,
        0.042188018560409546,
    ];
    close(
        "RReLU forward_training gradient (1 where kept, slope where scaled)",
        &leaf.grad().expect("RReLU input gradient").to_vec()?,
        &expected_gradient,
    );

    let mut repeat_pass = TrainingPass::new(2718);
    let repeat = rrelu.forward_training(&leaf, &mut repeat_pass)?;
    close(
        "RReLU forward_training is bit-identical across two runs of the same pass seed",
        &repeat.to_vec()?,
        &expected,
    );
    println!("RReLU forward_training per-element slope oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn rrelu_forward_training_statistical_sanity_slopes_stay_in_bounds() -> Result<()> {
    let device = Device::cuda(0)?;
    let feature = Axis::new("rrelu_stat_feature");
    let count = 100_000;
    // All-negative input isolates the scaled branch: output / input is exactly the drawn slope.
    let values: Vec<f32> = (0..count).map(|_| -1.0).collect();
    let leaf = Tensor::from_slice(&values, [feature.of(count)], &device)?;
    let mut pass = TrainingPass::new(31);
    let rrelu = RReLU::new().lower(0.1)?.upper(0.4)?;
    let output = rrelu.forward_training(&leaf, &mut pass)?.to_vec()?;
    let mut min_slope = f64::INFINITY;
    let mut max_slope: f64 = 0.0;
    let mut sum = 0.0_f64;
    for &v in &output {
        let slope = f64::from(-v); // output = -1 * slope, so slope = -output
        assert!(
            (0.1..0.4).contains(&slope),
            "slope {slope} outside [lower, upper)"
        );
        min_slope = min_slope.min(slope);
        max_slope = max_slope.max(slope);
        sum += slope;
    }
    let mean_slope = sum / output.len() as f64;
    assert!(
        (mean_slope - 0.25).abs() < 0.01,
        "mean slope {mean_slope} too far from the midpoint 0.25"
    );
    println!(
        "RReLU statistical sanity PASS mean_slope={mean_slope:.4} range=[{min_slope:.4}, {max_slope:.4}]"
    );
    Ok(())
}

// --- PrefixDropout -------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA"]
fn prefix_dropout_rejects_a_missing_axis_before_any_device_work() -> Result<()> {
    let device = Device::cuda(0)?;
    let axis = Axis::new("pd_missing_axis");
    let unrelated = Axis::new("pd_missing_unrelated");
    let input = Tensor::from_slice(&[1.0, 2.0], [unrelated.of(2)], &device)?;
    let dropout = PrefixDropout::new(axis);
    assert!(dropout.output_shape(input.shape()).is_err());
    assert!(dropout.forward(&input).is_err());
    println!("PrefixDropout missing-axis rejection PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn prefix_dropout_forward_is_the_identity() -> Result<()> {
    let device = Device::cuda(0)?;
    let code = Axis::new("pd_eval_code");
    let values: Vec<f32> = (1..=6).map(|v| v as f32).collect();
    let input = Tensor::from_slice(&values, [code.of(6)], &device)?;
    let output = PrefixDropout::new(code).forward(&input)?;
    close(
        "PrefixDropout::forward is the identity",
        &output.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );
    println!("PrefixDropout eval-mode identity PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn prefix_dropout_forward_training_matches_a_per_example_width_oracle_forward_and_gradient()
-> Result<()> {
    // Independent oracle: `TrainingPass::new(6060)`'s first `next_seed()` is
    // 5621291624497032630. Its first two `Tensor::uniform_device` draws (one per `batch`
    // example, `code` excluded from the draw) are [0.068539559841156, 0.5928589701652527],
    // scaled by `extent(code) + 1 = 5` into `w = [0.3426978, 2.964295]`. Position `i` (0..3)
    // survives when `i < w`: batch 0 keeps only position 0 (`floor(w) + 1 = 1`); batch 1 keeps
    // positions 0..2 (`floor(w) + 1 = 3`).
    let device = Device::cuda(0)?;
    let batch = Axis::new("pd_oracle_batch");
    let code = Axis::new("pd_oracle_code");
    let input_values: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let leaf = Tensor::from_slice(&input_values, [batch.of(2), code.of(4)], &device)?.with_grad();

    let dropout = PrefixDropout::new(code);
    let mut pass = TrainingPass::new(6060);
    let output = dropout.forward_training(&leaf, &mut pass)?;
    let expected = [1.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0];
    close(
        "PrefixDropout forward_training per-example width oracle",
        &output.to_vec()?,
        &expected,
    );

    output.mean([batch, code])?.backward()?;
    let expected_gradient = [0.125, 0.0, 0.0, 0.0, 0.125, 0.125, 0.125, 0.0];
    close(
        "PrefixDropout forward_training gradient (1 kept, 0 dropped, no rescale)",
        &leaf
            .grad()
            .expect("PrefixDropout input gradient")
            .to_vec()?,
        &expected_gradient,
    );

    // Same pass seed, same forward order: bit-identical.
    let mut repeat_pass = TrainingPass::new(6060);
    let repeat = dropout.forward_training(&leaf, &mut repeat_pass)?;
    close(
        "PrefixDropout forward_training is bit-identical across two runs of the same pass seed",
        &repeat.to_vec()?,
        &expected,
    );

    // A second call on the SAME pass (draw counter 1) draws a different per-example width,
    // mirroring `sequential_threads_one_training_pass_through_two_dropout_layers_with_distinct_
    // masks`'s use of `next_seed`'s second draw. `next_seed(6060, 1)`'s first two
    // `Tensor::uniform_device` draws are [0.6250380873680115, 0.228529155254364], scaled by
    // `extent + 1 = 5` into `w = [3.125190496444702, 1.1426458358764648]`, so batch 0 now keeps
    // all 4 positions (`floor(3.125) + 1 = 4`) and batch 1 keeps 2 (`floor(1.1426) + 1 = 2`).
    let mut second_pass = TrainingPass::new(6060);
    second_pass.next_seed(); // consume the same first draw the oracle above used
    let second_output = dropout.forward_training(&leaf, &mut second_pass)?;
    let second_expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0];
    close(
        "PrefixDropout second draw on one pass keeps a different per-example width",
        &second_output.to_vec()?,
        &second_expected,
    );
    println!("PrefixDropout forward_training per-example width oracle PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn prefix_dropout_statistical_sanity_kept_fraction_matches_the_closed_form_mean() -> Result<()> {
    // Kept count is `floor(w) + 1` for `w = draw * (extent + 1)` uniform on `[0, extent + 1)`,
    // so `floor(w)` is discrete-uniform over `{0, ..., extent}` with mean `extent / 2`: the
    // expected kept fraction is `(extent / 2 + 1) / extent = 0.5 + 1 / extent`.
    let device = Device::cuda(0)?;
    let batch = Axis::new("pd_stat_batch");
    let code = Axis::new("pd_stat_code");
    let extent = 64;
    let examples = 4096;
    let values: Vec<f32> = vec![1.0; examples * extent];
    let leaf = Tensor::from_slice(&values, [batch.of(examples), code.of(extent)], &device)?;
    let mut pass = TrainingPass::new(20260922);
    let output = PrefixDropout::new(code)
        .forward_training(&leaf, &mut pass)?
        .to_vec()?;
    let kept = output.iter().filter(|&&v| v != 0.0).count();
    let kept_fraction = kept as f64 / output.len() as f64;
    let expected_fraction = 0.5 + 1.0 / extent as f64;
    assert!(
        (kept_fraction - expected_fraction).abs() < 0.02,
        "kept fraction {kept_fraction} too far from the closed-form mean {expected_fraction}"
    );
    println!(
        "PrefixDropout statistical sanity PASS kept_fraction={kept_fraction:.4} expected={expected_fraction:.4}"
    );
    Ok(())
}

// --- Dropout family composition (shared consumer evidence) --------------------------------

#[test]
#[ignore = "requires CUDA"]
fn dropout_family_composes_in_one_training_pass_with_distinct_draws_per_layer() -> Result<()> {
    // The six-point module definition's "a consumer" bar, for one composed pipeline shared by
    // every row landed in this PR: a `Sequential` threading one `TrainingPass` through
    // `ChannelDropout`, `AlphaDropout`, `RReLU`, and `PrefixDropout` in order, the way a real
    // convolutional-then-dense model would. Each layer must consume its own draw (so the whole
    // pipeline is reproducible from one pass seed) and the composition must run end to end
    // without error on both `forward` and `forward_training`.
    let device = Device::cuda(0)?;
    let batch = Axis::new("family_batch");
    let channel = Axis::new("family_channel");
    let spatial = Axis::new("family_spatial");
    let values: Vec<f32> = (0..16).map(|i| 1.0 + i as f32 * 0.25).collect();
    let input = Tensor::from_slice(
        &values,
        [batch.of(2), channel.of(2), spatial.of(4)],
        &device,
    )?;

    let mut seq = Sequential::new((
        ChannelDropout::new(0.3, channel, [spatial])?,
        AlphaDropout::new(0.2)?,
        RReLU::new(),
    ));
    seq.build(input.shape(), &device, 0)?;

    // All-positive input: ChannelDropout/AlphaDropout's eval is the identity, and RReLU's eval
    // (leaky_relu at the fixed midpoint slope) is also the identity on strictly positive values,
    // so the whole pipeline reduces to the identity here without special-casing RReLU.
    let eval_output = seq.forward(&input)?;
    close(
        "Dropout-family Sequential eval forward is the identity on all-positive input",
        &eval_output.to_vec()?,
        &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
    );

    let mut pass = TrainingPass::new(2468);
    let train_output = seq.forward_training(&input, &mut pass)?;
    assert_eq!(train_output.shape().len(), input.shape().len());
    // Three draws consumed: one per layer, in call order.
    assert_eq!(pass.next_seed(), {
        let mut fresh = TrainingPass::new(2468);
        fresh.next_seed();
        fresh.next_seed();
        fresh.next_seed();
        fresh.next_seed()
    });

    // PrefixDropout composed separately over a fourth axis, sharing one pass with a plain
    // Dropout layer, to also exercise the new module inside the existing Dropout consumer test
    // shape (two draws, distinct masks/widths).
    let code = Axis::new("family_code");
    let code_input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0], [code.of(5)], &device)?;
    let mut mixed = Sequential::new((Dropout::new(0.5)?, PrefixDropout::new(code)));
    mixed.build(code_input.shape(), &device, 0)?;
    let mut mixed_pass = TrainingPass::new(13579);
    let mixed_output = mixed
        .forward_training(&code_input, &mut mixed_pass)?
        .to_vec()?;
    assert_eq!(mixed_output.len(), 5);
    println!("Dropout-family Sequential composition PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multihead_attention_forward_training_dropout_zero_is_bit_exact_and_consumes_one_draw()
-> Result<()> {
    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("mha_dropout_zero_feature"),
        Axis::new("mha_dropout_zero_time"),
    );
    let x_values = [0.3_f32, -0.6, 0.9, 0.2, -0.4, 0.7, -0.1, 0.5];
    let x = Tensor::from_slice(&x_values, [time.of(2), feature.of(4)], &device)?.with_grad();

    let mut mha = MultiheadAttention::new(feature, time, 4, 2, 0.0)?;
    mha.build(x.shape(), x.shape(), x.shape(), &device, 7)?;
    let eval_output = mha.forward(&x, &x, &x, None, None)?.to_vec()?;
    let mut pass = TrainingPass::new(11);
    let train_output = mha.forward_training(&x, &x, &x, None, None, &mut pass)?;
    close(
        "MultiheadAttention forward_training p=0 is bit-exact with forward",
        &train_output.to_vec()?,
        &eval_output
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );

    // forward_training must consume exactly one pass.next_seed() draw even when p=0.0 does not
    // feed it into an actual random tensor -- Dropout's own draw-count convention, so a
    // downstream random consumer's position in the pass does not shift when p changes.
    let mut reference_pass = TrainingPass::new(11);
    let consumed = reference_pass.next_seed();
    let next_expected = reference_pass.next_seed();
    let mut fresh_pass = TrainingPass::new(11);
    let _ = mha.forward_training(&x, &x, &x, None, None, &mut fresh_pass)?;
    let _ = consumed;
    assert_eq!(
        fresh_pass.next_seed(),
        next_expected,
        "forward_training must consume exactly one draw at p=0"
    );
    println!("MultiheadAttention forward_training p=0 edge case PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multihead_attention_forward_training_dropout_one_is_exact_zero_with_zero_gradient() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("mha_dropout_one_feature"),
        Axis::new("mha_dropout_one_time"),
    );
    let x_values = [0.3_f32, -0.6, 0.9, 0.2, -0.4, 0.7, -0.1, 0.5];
    let x = Tensor::from_slice(&x_values, [time.of(2), feature.of(4)], &device)?.with_grad();

    // No output bias: with the attention weights forced to exact zero, `attended` is exact
    // zero, so the final output (and every gradient) is exact zero too.
    let mut mha = MultiheadAttention::new(feature, time, 4, 2, 1.0)?.bias(false);
    mha.build(x.shape(), x.shape(), x.shape(), &device, 13)?;
    let mut pass = TrainingPass::new(23);
    let output = mha.forward_training(&x, &x, &x, None, None, &mut pass)?;
    close(
        "MultiheadAttention forward_training p=1 (no output bias) is exact zero",
        &output.to_vec()?,
        &[0.0; 8],
    );
    output.mean([time, feature])?.backward()?;
    close(
        "MultiheadAttention forward_training p=1 input gradient is exact zero",
        &x.grad().unwrap().to_vec()?,
        &[0.0; 8],
    );
    for (name, parameter) in mha.named_parameters() {
        let grad = parameter
            .grad()
            .unwrap_or_else(|| panic!("missing gradient for {name}"))
            .to_vec()
            .unwrap();
        close(
            &format!("MultiheadAttention forward_training p=1 {name} gradient is exact zero"),
            &grad,
            &vec![0.0; grad.len()],
        );
    }
    println!("MultiheadAttention forward_training p=1 edge case PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multihead_attention_forward_training_dropout_preserves_expectation_statistically() -> Result<()>
{
    // Inverted dropout scales survivors by 1/(1-p) so its expectation matches the undropped
    // value; average many independent pass seeds and compare against the p=0 forward, mirroring
    // Dropout's own statistical-sanity test but through the whole attention module (query/key/
    // value/output projections included), where the exact physical order of the dropped
    // probability elements is an internal detail this test does not need to know.
    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("mha_dropout_stat_feature"),
        Axis::new("mha_dropout_stat_time"),
    );
    let extent = 6usize;
    let x_values: Vec<f32> = (0..extent * 4).map(|i| ((i as f32) * 0.37).sin()).collect();
    let x = Tensor::from_slice(&x_values, [time.of(extent), feature.of(4)], &device)?;

    let mut eval_mha = MultiheadAttention::new(feature, time, 4, 2, 0.0)?;
    eval_mha.build(x.shape(), x.shape(), x.shape(), &device, 41)?;
    let expected = eval_mha.forward(&x, &x, &x, None, None)?.to_vec()?;

    // Cloning an already-built MultiheadAttention shares its Parameters (see the struct doc
    // comment), so train_mha scores identically to eval_mha; only its own dropout differs.
    let train_mha = eval_mha.clone().dropout(0.3)?;
    let trials = 96;
    let mut sum = vec![0.0_f64; expected.len()];
    for trial in 0..trials {
        let mut pass = TrainingPass::new(9000 + trial as u64);
        let output = train_mha
            .forward_training(&x, &x, &x, None, None, &mut pass)?
            .to_vec()?;
        for (s, &v) in sum.iter_mut().zip(&output) {
            *s += f64::from(v);
        }
    }
    for (index, (&s, &e)) in sum.iter().zip(&expected).enumerate() {
        let mean = s / trials as f64;
        let error = (mean - f64::from(e)).abs();
        assert!(
            error < 0.15,
            "MultiheadAttention dropout=0.3 trial-mean[{index}] {mean} too far from p=0 forward {e}"
        );
    }
    println!(
        "MultiheadAttention forward_training dropout expectation-preservation PASS trials={trials}"
    );
    Ok(())
}

#[test]
fn transformer_encoder_decoder_layer_constructors_reject_invalid_configuration() -> Result<()> {
    let (feature, time) = (
        Axis::new("transformer_layer_reject_feature"),
        Axis::new("transformer_layer_reject_time"),
    );
    assert!(TransformerEncoderLayer::new(feature, time, 4, 2, 0).is_err());
    assert!(TransformerEncoderLayer::new(feature, time, 5, 2, 8).is_err());
    assert!(TransformerDecoderLayer::new(feature, time, 4, 2, 0).is_err());
    assert!(TransformerDecoderLayer::new(feature, time, 4, 0, 8).is_err());

    let layer = TransformerEncoderLayer::new(feature, time, 4, 2, 8)?;
    assert!(layer.dropout(-0.1).is_err());
    assert!(
        TransformerEncoderLayer::new(feature, time, 4, 2, 8)?
            .dropout(1.0)
            .is_ok()
    );

    assert!(
        TransformerEncoder::new(TransformerEncoderLayer::new(feature, time, 4, 2, 8)?, 0).is_err()
    );
    assert!(
        TransformerDecoder::new(TransformerDecoderLayer::new(feature, time, 4, 2, 8)?, 0).is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn transformer_encoder_layer_matches_independent_f64_oracle_post_and_pre_norm() -> Result<()> {
    // Independent from-scratch f64 reference for one TransformerEncoderLayer, following PyTorch
    // 2.14's own _sa_block/_ff_block composition and norm_first branch (transformer.py). Every
    // gradient is checked by central difference against this reference, never against Axis's own
    // analytic backward.
    fn linear(x: &[f64], w: &[f64], b: &[f64], t: usize, din: usize, dout: usize) -> Vec<f64> {
        let mut out = vec![0.0; t * dout];
        for i in 0..t {
            for o in 0..dout {
                let mut sum = b[o];
                for k in 0..din {
                    sum += x[i * din + k] * w[k * dout + o];
                }
                out[i * dout + o] = sum;
            }
        }
        out
    }
    fn layer_norm(x: &[f64], scale: &[f64], bias: &[f64], t: usize, d: usize) -> Vec<f64> {
        let eps = 1e-5_f64;
        let mut out = vec![0.0; t * d];
        for i in 0..t {
            let row = &x[i * d..(i + 1) * d];
            let mean = row.iter().sum::<f64>() / d as f64;
            let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = (variance + eps).sqrt().recip();
            for k in 0..d {
                out[i * d + k] = (row[k] - mean) * inv * scale[k] + bias[k];
            }
        }
        out
    }
    fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }
    fn relu(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }
    #[allow(clippy::too_many_arguments)]
    fn sa_block(
        x: &[f64],
        wq: &[f64],
        bq: &[f64],
        wk: &[f64],
        bk: &[f64],
        wv: &[f64],
        bv: &[f64],
        wo: &[f64],
        bo: &[f64],
        t: usize,
        d: usize,
        h: usize,
    ) -> Vec<f64> {
        let dh = d / h;
        let q = linear(x, wq, bq, t, d, d);
        let k = linear(x, wk, bk, t, d, d);
        let v = linear(x, wv, bv, t, d, d);
        let mut attn = vec![0.0; t * d];
        for head in 0..h {
            for i in 0..t {
                let mut scores = vec![0.0; t];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for dd in 0..dh {
                        s += q[i * d + head * dh + dd] * k[j * d + head * dh + dd];
                    }
                    *score = s / (dh as f64).sqrt();
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
                for dd in 0..dh {
                    let mut acc = 0.0;
                    for j in 0..t {
                        acc += probs[j] * v[j * d + head * dh + dd];
                    }
                    attn[i * d + head * dh + dd] = acc;
                }
            }
        }
        linear(&attn, wo, bo, t, d, d)
    }
    #[allow(clippy::too_many_arguments)]
    fn ff_block(
        x: &[f64],
        w1: &[f64],
        b1: &[f64],
        w2: &[f64],
        b2: &[f64],
        t: usize,
        d: usize,
        ff: usize,
    ) -> Vec<f64> {
        let hidden = relu(&linear(x, w1, b1, t, d, ff));
        linear(&hidden, w2, b2, t, ff, d)
    }
    #[allow(clippy::too_many_arguments, unused_assignments)]
    fn encoder_layer_forward(
        x: &[f64],
        p: &[f64],
        t: usize,
        d: usize,
        h: usize,
        ff: usize,
        norm_first: bool,
    ) -> Vec<f64> {
        let mut o = 0usize;
        macro_rules! take {
            ($n:expr) => {{
                let s = &p[o..o + $n];
                o += $n;
                s
            }};
        }
        let wq = take!(d * d);
        let bq = take!(d);
        let wk = take!(d * d);
        let bk = take!(d);
        let wv = take!(d * d);
        let bv = take!(d);
        let wo = take!(d * d);
        let bo = take!(d);
        let w1 = take!(d * ff);
        let b1 = take!(ff);
        let w2 = take!(ff * d);
        let b2 = take!(d);
        let n1s = take!(d);
        let n1b = take!(d);
        let n2s = take!(d);
        let n2b = take!(d);
        if norm_first {
            let n1 = layer_norm(x, n1s, n1b, t, d);
            let sa = sa_block(&n1, wq, bq, wk, bk, wv, bv, wo, bo, t, d, h);
            let x1 = add(x, &sa);
            let n2 = layer_norm(&x1, n2s, n2b, t, d);
            let ff_out = ff_block(&n2, w1, b1, w2, b2, t, d, ff);
            add(&x1, &ff_out)
        } else {
            let sa = sa_block(x, wq, bq, wk, bk, wv, bv, wo, bo, t, d, h);
            let x1 = layer_norm(&add(x, &sa), n1s, n1b, t, d);
            let ff_out = ff_block(&x1, w1, b1, w2, b2, t, d, ff);
            layer_norm(&add(&x1, &ff_out), n2s, n2b, t, d)
        }
    }
    fn encoder_layer_loss(
        x: &[f64],
        p: &[f64],
        t: usize,
        d: usize,
        h: usize,
        ff: usize,
        norm_first: bool,
    ) -> f64 {
        let output = encoder_layer_forward(x, p, t, d, h, ff, norm_first);
        output.iter().sum::<f64>() / output.len() as f64
    }

    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("transformer_encoder_layer_feature"),
        Axis::new("transformer_encoder_layer_time"),
    );
    let (t, d, h, ff) = (2usize, 4usize, 2usize, 3usize);
    let x_values: Vec<f64> = (0..t * d)
        .map(|i| ((i as f64 + 1.0) * 0.41).sin() * 0.7)
        .collect();
    let param_count = 4 * (d * d + d) + (d * ff + ff) + (ff * d + d) + 4 * d;
    let p_values: Vec<f64> = (0..param_count)
        .map(|i| ((i as f64 + 3.0) * 0.27).sin() * 0.5)
        .collect();
    let expected_names = [
        "self_attn.query.weight",
        "self_attn.query.bias",
        "self_attn.key.weight",
        "self_attn.key.bias",
        "self_attn.value.weight",
        "self_attn.value.bias",
        "self_attn.output.weight",
        "self_attn.output.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
        "norm1.scale",
        "norm1.bias",
        "norm2.scale",
        "norm2.bias",
    ];

    for &norm_first in &[false, true] {
        let mut layer = TransformerEncoderLayer::new(feature, time, d, h, ff)?
            .dropout(0.0)?
            .norm_first(norm_first);
        let x_tensor = Tensor::from_slice(
            &x_values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            [time.of(t), feature.of(d)],
            &device,
        )?
        .with_grad();
        let shape = layer.build(x_tensor.shape(), &device, 0)?;
        assert_eq!(shape, *x_tensor.shape());
        let params = layer.named_parameters();
        assert_eq!(
            params.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            expected_names
        );
        let mut offset = 0;
        for (_, parameter) in &params {
            let len = parameter.tensor().shape().len();
            let slice: Vec<f32> = p_values[offset..offset + len]
                .iter()
                .map(|&v| v as f32)
                .collect();
            parameter.set_values(&slice)?;
            offset += len;
        }
        assert_eq!(offset, p_values.len());

        let output = layer.forward(&x_tensor, None, None)?;
        close(
            &format!("TransformerEncoderLayer forward (norm_first={norm_first})"),
            &output.to_vec()?,
            &encoder_layer_forward(&x_values, &p_values, t, d, h, ff, norm_first),
        );

        output.mean([time, feature])?.backward()?;
        close(
            &format!("TransformerEncoderLayer input gradient (norm_first={norm_first})"),
            &x_tensor.grad().unwrap().to_vec()?,
            &central_difference(&x_values, 1e-4, |candidate| {
                encoder_layer_loss(candidate, &p_values, t, d, h, ff, norm_first)
            }),
        );
        let expected_grad = central_difference(&p_values, 1e-4, |candidate| {
            encoder_layer_loss(&x_values, candidate, t, d, h, ff, norm_first)
        });
        let mut offset = 0;
        for (name, parameter) in &params {
            let len = parameter.tensor().shape().len();
            close(
                &format!("TransformerEncoderLayer {name} gradient (norm_first={norm_first})"),
                &parameter.grad().unwrap().to_vec()?,
                &expected_grad[offset..offset + len],
            );
            offset += len;
        }

        // Reordered physical input storage must not change the forward value (post-norm case
        // only, matching MultiheadAttention's own reordered-storage witness).
        if !norm_first {
            let x_reordered = Tensor::from_slice(
                &x_values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                [time.of(t), feature.of(d)],
                &device,
            )?
            .with_layout([feature, time])?;
            close(
                "TransformerEncoderLayer forward is unaffected by reordered input storage",
                &layer.forward(&x_reordered, None, None)?.to_vec()?,
                &encoder_layer_forward(&x_values, &p_values, t, d, h, ff, norm_first),
            );
        }
    }
    println!("TransformerEncoderLayer forward/gradient oracle PASS (post-norm and pre-norm)");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn transformer_decoder_layer_matches_independent_f64_oracle_post_and_pre_norm_with_asymmetric_extents()
-> Result<()> {
    // Same independent from-scratch f64 method as the encoder-layer oracle above, extended with
    // PyTorch's _mha_block cross-attention. tgt_len != mem_len exercises the asymmetric-extent
    // cross-attention contract MultiheadAttention's own doc comment describes.
    fn linear(x: &[f64], w: &[f64], b: &[f64], t: usize, din: usize, dout: usize) -> Vec<f64> {
        let mut out = vec![0.0; t * dout];
        for i in 0..t {
            for o in 0..dout {
                let mut sum = b[o];
                for k in 0..din {
                    sum += x[i * din + k] * w[k * dout + o];
                }
                out[i * dout + o] = sum;
            }
        }
        out
    }
    fn layer_norm(x: &[f64], scale: &[f64], bias: &[f64], t: usize, d: usize) -> Vec<f64> {
        let eps = 1e-5_f64;
        let mut out = vec![0.0; t * d];
        for i in 0..t {
            let row = &x[i * d..(i + 1) * d];
            let mean = row.iter().sum::<f64>() / d as f64;
            let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = (variance + eps).sqrt().recip();
            for k in 0..d {
                out[i * d + k] = (row[k] - mean) * inv * scale[k] + bias[k];
            }
        }
        out
    }
    fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }
    fn relu(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }
    #[allow(clippy::too_many_arguments)]
    fn cross_attention(
        query_x: &[f64],
        key_value_x: &[f64],
        wq: &[f64],
        bq: &[f64],
        wk: &[f64],
        bk: &[f64],
        wv: &[f64],
        bv: &[f64],
        wo: &[f64],
        bo: &[f64],
        tq: usize,
        tk: usize,
        d: usize,
        h: usize,
    ) -> Vec<f64> {
        let dh = d / h;
        let q = linear(query_x, wq, bq, tq, d, d);
        let k = linear(key_value_x, wk, bk, tk, d, d);
        let v = linear(key_value_x, wv, bv, tk, d, d);
        let mut attn = vec![0.0; tq * d];
        for head in 0..h {
            for i in 0..tq {
                let mut scores = vec![0.0; tk];
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for dd in 0..dh {
                        s += q[i * d + head * dh + dd] * k[j * d + head * dh + dd];
                    }
                    *score = s / (dh as f64).sqrt();
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
                for dd in 0..dh {
                    let mut acc = 0.0;
                    for j in 0..tk {
                        acc += probs[j] * v[j * d + head * dh + dd];
                    }
                    attn[i * d + head * dh + dd] = acc;
                }
            }
        }
        linear(&attn, wo, bo, tq, d, d)
    }
    #[allow(clippy::too_many_arguments)]
    fn ff_block(
        x: &[f64],
        w1: &[f64],
        b1: &[f64],
        w2: &[f64],
        b2: &[f64],
        t: usize,
        d: usize,
        ff: usize,
    ) -> Vec<f64> {
        let hidden = relu(&linear(x, w1, b1, t, d, ff));
        linear(&hidden, w2, b2, t, ff, d)
    }
    #[allow(clippy::too_many_arguments, unused_assignments)]
    fn decoder_layer_forward(
        tgt: &[f64],
        memory: &[f64],
        p: &[f64],
        tq: usize,
        tk: usize,
        d: usize,
        h: usize,
        ff: usize,
        norm_first: bool,
    ) -> Vec<f64> {
        let mut o = 0usize;
        macro_rules! take {
            ($n:expr) => {{
                let s = &p[o..o + $n];
                o += $n;
                s
            }};
        }
        let sa_wq = take!(d * d);
        let sa_bq = take!(d);
        let sa_wk = take!(d * d);
        let sa_bk = take!(d);
        let sa_wv = take!(d * d);
        let sa_bv = take!(d);
        let sa_wo = take!(d * d);
        let sa_bo = take!(d);
        let ca_wq = take!(d * d);
        let ca_bq = take!(d);
        let ca_wk = take!(d * d);
        let ca_bk = take!(d);
        let ca_wv = take!(d * d);
        let ca_bv = take!(d);
        let ca_wo = take!(d * d);
        let ca_bo = take!(d);
        let w1 = take!(d * ff);
        let b1 = take!(ff);
        let w2 = take!(ff * d);
        let b2 = take!(d);
        let n1s = take!(d);
        let n1b = take!(d);
        let n2s = take!(d);
        let n2b = take!(d);
        let n3s = take!(d);
        let n3b = take!(d);
        if norm_first {
            let n1 = layer_norm(tgt, n1s, n1b, tq, d);
            let sa = cross_attention(
                &n1, &n1, sa_wq, sa_bq, sa_wk, sa_bk, sa_wv, sa_bv, sa_wo, sa_bo, tq, tq, d, h,
            );
            let x1 = add(tgt, &sa);
            let n2 = layer_norm(&x1, n2s, n2b, tq, d);
            let ca = cross_attention(
                &n2, memory, ca_wq, ca_bq, ca_wk, ca_bk, ca_wv, ca_bv, ca_wo, ca_bo, tq, tk, d, h,
            );
            let x2 = add(&x1, &ca);
            let n3 = layer_norm(&x2, n3s, n3b, tq, d);
            let ff_out = ff_block(&n3, w1, b1, w2, b2, tq, d, ff);
            add(&x2, &ff_out)
        } else {
            let sa = cross_attention(
                tgt, tgt, sa_wq, sa_bq, sa_wk, sa_bk, sa_wv, sa_bv, sa_wo, sa_bo, tq, tq, d, h,
            );
            let x1 = layer_norm(&add(tgt, &sa), n1s, n1b, tq, d);
            let ca = cross_attention(
                &x1, memory, ca_wq, ca_bq, ca_wk, ca_bk, ca_wv, ca_bv, ca_wo, ca_bo, tq, tk, d, h,
            );
            let x2 = layer_norm(&add(&x1, &ca), n2s, n2b, tq, d);
            let ff_out = ff_block(&x2, w1, b1, w2, b2, tq, d, ff);
            layer_norm(&add(&x2, &ff_out), n3s, n3b, tq, d)
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn decoder_layer_loss(
        tgt: &[f64],
        memory: &[f64],
        p: &[f64],
        tq: usize,
        tk: usize,
        d: usize,
        h: usize,
        ff: usize,
        norm_first: bool,
    ) -> f64 {
        let output = decoder_layer_forward(tgt, memory, p, tq, tk, d, h, ff, norm_first);
        output.iter().sum::<f64>() / output.len() as f64
    }

    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("transformer_decoder_layer_feature"),
        Axis::new("transformer_decoder_layer_time"),
    );
    let (tq, tk, d, h, ff) = (2usize, 3usize, 4usize, 2usize, 3usize);
    let tgt_values: Vec<f64> = (0..tq * d)
        .map(|i| ((i as f64 + 1.0) * 0.41).sin() * 0.7)
        .collect();
    let memory_values: Vec<f64> = (0..tk * d)
        .map(|i| ((i as f64 + 1.0) * 0.53).cos() * 0.6)
        .collect();
    let param_count = 8 * (d * d + d) + (d * ff + ff) + (ff * d + d) + 6 * d;
    let p_values: Vec<f64> = (0..param_count)
        .map(|i| ((i as f64 + 5.0) * 0.19).sin() * 0.5)
        .collect();
    let expected_names = [
        "self_attn.query.weight",
        "self_attn.query.bias",
        "self_attn.key.weight",
        "self_attn.key.bias",
        "self_attn.value.weight",
        "self_attn.value.bias",
        "self_attn.output.weight",
        "self_attn.output.bias",
        "multihead_attn.query.weight",
        "multihead_attn.query.bias",
        "multihead_attn.key.weight",
        "multihead_attn.key.bias",
        "multihead_attn.value.weight",
        "multihead_attn.value.bias",
        "multihead_attn.output.weight",
        "multihead_attn.output.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
        "norm1.scale",
        "norm1.bias",
        "norm2.scale",
        "norm2.bias",
        "norm3.scale",
        "norm3.bias",
    ];

    for &norm_first in &[false, true] {
        let mut layer = TransformerDecoderLayer::new(feature, time, d, h, ff)?
            .dropout(0.0)?
            .norm_first(norm_first);
        let tgt_tensor = Tensor::from_slice(
            &tgt_values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            [time.of(tq), feature.of(d)],
            &device,
        )?
        .with_grad();
        let memory_tensor = Tensor::from_slice(
            &memory_values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            [time.of(tk), feature.of(d)],
            &device,
        )?
        .with_grad();
        let shape = layer.build(tgt_tensor.shape(), memory_tensor.shape(), &device, 0)?;
        assert_eq!(shape, *tgt_tensor.shape());
        let params = layer.named_parameters();
        assert_eq!(
            params.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            expected_names
        );
        let mut offset = 0;
        for (_, parameter) in &params {
            let len = parameter.tensor().shape().len();
            let slice: Vec<f32> = p_values[offset..offset + len]
                .iter()
                .map(|&v| v as f32)
                .collect();
            parameter.set_values(&slice)?;
            offset += len;
        }
        assert_eq!(offset, p_values.len());

        let output = layer.forward(&tgt_tensor, &memory_tensor, None, None, None, None)?;
        close(
            &format!("TransformerDecoderLayer forward (norm_first={norm_first})"),
            &output.to_vec()?,
            &decoder_layer_forward(
                &tgt_values,
                &memory_values,
                &p_values,
                tq,
                tk,
                d,
                h,
                ff,
                norm_first,
            ),
        );

        output.mean([time, feature])?.backward()?;
        close(
            &format!("TransformerDecoderLayer tgt gradient (norm_first={norm_first})"),
            &tgt_tensor.grad().unwrap().to_vec()?,
            &central_difference(&tgt_values, 1e-4, |candidate| {
                decoder_layer_loss(
                    candidate,
                    &memory_values,
                    &p_values,
                    tq,
                    tk,
                    d,
                    h,
                    ff,
                    norm_first,
                )
            }),
        );
        close(
            &format!("TransformerDecoderLayer memory gradient (norm_first={norm_first})"),
            &memory_tensor.grad().unwrap().to_vec()?,
            &central_difference(&memory_values, 1e-4, |candidate| {
                decoder_layer_loss(
                    &tgt_values,
                    candidate,
                    &p_values,
                    tq,
                    tk,
                    d,
                    h,
                    ff,
                    norm_first,
                )
            }),
        );
        let expected_grad = central_difference(&p_values, 1e-4, |candidate| {
            decoder_layer_loss(
                &tgt_values,
                &memory_values,
                candidate,
                tq,
                tk,
                d,
                h,
                ff,
                norm_first,
            )
        });
        let mut offset = 0;
        for (name, parameter) in &params {
            let len = parameter.tensor().shape().len();
            close(
                &format!("TransformerDecoderLayer {name} gradient (norm_first={norm_first})"),
                &parameter.grad().unwrap().to_vec()?,
                &expected_grad[offset..offset + len],
            );
            offset += len;
        }
    }
    println!(
        "TransformerDecoderLayer forward/gradient oracle PASS (post-norm, pre-norm, asymmetric tgt/memory extents)"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn transformer_encoder_stacks_independently_built_layers_matching_manual_composition() -> Result<()>
{
    // TransformerEncoder must be exactly the composition of its own stacked layers plus the
    // final norm: build two standalone layers with the SAME per-layer seeds TransformerEncoder
    // itself would use, chain them by hand, and compare against the stack's own forward.
    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("transformer_encoder_stack_feature"),
        Axis::new("transformer_encoder_stack_time"),
    );
    let x_values: [f32; 8] = [0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.7, 0.4];
    let x = Tensor::from_slice(&x_values, [time.of(2), feature.of(4)], &device)?;

    let mut stack = TransformerEncoder::new(
        TransformerEncoderLayer::new(feature, time, 4, 2, 6)?.dropout(0.0)?,
        2,
    )?
    .norm(LayerNorm::new(feature)?);
    stack.build(x.shape(), &device, 100)?;
    let stacked_output = stack.forward(&x, None, None)?.to_vec()?;

    let mut manual_first = TransformerEncoderLayer::new(feature, time, 4, 2, 6)?.dropout(0.0)?;
    manual_first.build(x.shape(), &device, 100)?;
    let mut manual_second = TransformerEncoderLayer::new(feature, time, 4, 2, 6)?.dropout(0.0)?;
    let after_first = manual_first.output_shape(x.shape())?;
    manual_second.build(&after_first, &device, 200)?;
    let mut manual_norm = LayerNorm::new(feature)?;
    manual_norm.build(&after_first, &device, 100)?;
    let manual_output = manual_norm
        .forward(&manual_second.forward(&manual_first.forward(&x, None, None)?, None, None)?)?
        .to_vec()?;

    close(
        "TransformerEncoder stacked forward equals manual layer-by-layer composition",
        &stacked_output,
        &manual_output
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );
    println!("TransformerEncoder stacking composition PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn transformer_end_to_end_encode_then_decode_matches_manual_encoder_decoder_composition()
-> Result<()> {
    // Transformer::forward must equal calling its own encoder then its own decoder by hand.
    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("transformer_e2e_feature"),
        Axis::new("transformer_e2e_time"),
    );
    let src_values: [f32; 8] = [0.1, 0.4, -0.2, 0.6, -0.5, 0.3, 0.2, -0.1];
    let tgt_values: [f32; 12] = [
        0.05, -0.2, 0.3, 0.1, -0.4, 0.25, 0.15, -0.05, 0.2, -0.3, 0.1, 0.4,
    ];
    let src = Tensor::from_slice(&src_values, [time.of(2), feature.of(4)], &device)?;
    let tgt = Tensor::from_slice(&tgt_values, [time.of(3), feature.of(4)], &device)?.with_grad();

    let mut model = Transformer::new(feature, time, 4, 2, 2, 2, 6)?.dropout(0.0)?;
    model.build(src.shape(), tgt.shape(), &device, 500)?;
    let output = model.forward(&src, &tgt, None, None, None, None, None, None)?;

    let manual_memory = {
        let mut encoder = TransformerEncoder::new(
            TransformerEncoderLayer::new(feature, time, 4, 2, 6)?.dropout(0.0)?,
            2,
        )?
        .norm(LayerNorm::new(feature)?);
        encoder.build(src.shape(), &device, 500)?;
        encoder.forward(&src, None, None)?
    };
    let manual_output = {
        let mut decoder = TransformerDecoder::new(
            TransformerDecoderLayer::new(feature, time, 4, 2, 6)?.dropout(0.0)?,
            2,
        )?
        .norm(LayerNorm::new(feature)?);
        // Must match Transformer::build's own decoder seed exactly:
        // `seed.wrapping_add(1_000_000)`.
        decoder.build(tgt.shape(), manual_memory.shape(), &device, 500 + 1_000_000)?;
        decoder.forward(&tgt, &manual_memory, None, None, None, None)?
    };

    // Transformer's own encoder/decoder seeds (`build`'s `seed`/`seed.wrapping_add(1_000_000)`)
    // must line up with these manually constructed stacks for the comparison to be meaningful;
    // assert equal shapes first, then bit-exact values.
    assert_eq!(output.shape(), manual_output.shape());
    close(
        "Transformer forward equals manual encoder-then-decoder composition",
        &output.to_vec()?,
        &manual_output
            .to_vec()?
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );

    output.mean([time, feature])?.backward()?;
    assert!(
        tgt.grad().is_some(),
        "Transformer end-to-end backward must reach the tgt input"
    );

    // Training-mode determinism: the same pass seed reproduces the identical output; a different
    // step index changes it (dropout is active here).
    let mut trained_model = Transformer::new(feature, time, 4, 2, 1, 1, 6)?.dropout(0.2)?;
    trained_model.build(src.shape(), tgt.shape(), &device, 900)?;
    let mut pass_a = TrainingPass::new(4242);
    let out_a = trained_model
        .forward_training(&src, &tgt, None, None, None, None, None, None, &mut pass_a)?
        .to_vec()?;
    let mut pass_b = TrainingPass::new(4242);
    let out_b = trained_model
        .forward_training(&src, &tgt, None, None, None, None, None, None, &mut pass_b)?
        .to_vec()?;
    assert_eq!(
        out_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "same pass seed must give a bit-identical Transformer training forward"
    );
    let mut pass_c = TrainingPass::new(777);
    let out_c = trained_model
        .forward_training(&src, &tgt, None, None, None, None, None, None, &mut pass_c)?
        .to_vec()?;
    assert_ne!(
        out_a, out_c,
        "a different pass seed must change the dropout-active Transformer training forward"
    );
    println!("Transformer end-to-end composition and training-mode determinism PASS");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn multihead_attention_causal_mask_matches_hand_computed_zero_and_neg_infinity_pattern()
-> Result<()> {
    // `MultiheadAttention::causal_mask` builds a reusable additive mask against the attention's
    // own private query_time/key_time roles: zero where a key is visible, -infinity where the key
    // position is strictly greater than the query position (PyTorch's
    // `generate_square_subsequent_mask`). `close` requires finite values, so compare the pattern
    // directly rather than through it.
    let device = Device::cuda(0)?;
    let (feature, time) = (
        Axis::new("mha_causal_mask_feature"),
        Axis::new("mha_causal_mask_time"),
    );
    let probe = MultiheadAttention::new(feature, time, 4, 1, 0.0)?;
    let causal = probe.causal_mask(3, &device)?;
    let AttentionMask::Additive(mask_tensor) = &causal else {
        panic!("causal_mask must return an Additive mask");
    };
    let values = mask_tensor.to_vec()?;
    // Row-major [query, key]: query i sees key j exactly when j <= i.
    let expected_visible = [
        true, false, false, // query 0
        true, true, false, // query 1
        true, true, true, // query 2
    ];
    assert_eq!(values.len(), expected_visible.len());
    for (index, (&value, &visible)) in values.iter().zip(&expected_visible).enumerate() {
        if visible {
            assert_eq!(value, 0.0, "mask[{index}] should be exactly 0.0 (visible)");
        } else {
            assert!(
                value.is_infinite() && value.is_sign_negative(),
                "mask[{index}] should be -infinity (forbidden), got {value}"
            );
        }
    }

    // Applied to real, distinguishable scores, the causal mask actually blocks future keys: a
    // uniform-logit self-attention layer restricted causally must only look at query <= key
    // positions, so query 0's attended value equals value[0] alone (no key 1/2 contribution).
    let mut layer = MultiheadAttention::new(feature, time, 4, 1, 0.0)?;
    layer.build(
        &Shape::new([time.of(3), feature.of(4)])?,
        &Shape::new([time.of(3), feature.of(4)])?,
        &Shape::new([time.of(3), feature.of(4)])?,
        &device,
        5,
    )?;
    // Zero query/key weights and bias: every score is identical, so softmax without a mask would
    // be exactly uniform; causal masking must instead give query 0 all its weight on key 0.
    for name in ["query.weight", "query.bias", "key.weight", "key.bias"] {
        let parameter = layer
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == name)
            .unwrap()
            .1;
        let len = parameter.tensor().shape().len();
        parameter.set_values(&vec![0.0; len])?;
    }
    for name in ["value.weight", "output.weight"] {
        let parameter = layer
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == name)
            .unwrap()
            .1;
        let len = parameter.tensor().shape().len();
        let identity_like: Vec<f32> = (0..len)
            .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
            .collect();
        parameter.set_values(&identity_like)?;
    }
    for name in ["value.bias", "output.bias"] {
        let parameter = layer
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == name)
            .unwrap()
            .1;
        let len = parameter.tensor().shape().len();
        parameter.set_values(&vec![0.0; len])?;
    }
    let x_values: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let x = Tensor::from_slice(&x_values, [time.of(3), feature.of(4)], &device)?;
    let mask = layer.causal_mask(3, &device)?;
    let output = layer.forward(&x, &x, &x, Some(&mask), None)?.to_vec()?;
    // value.weight/output.weight are both the "identity-like" pattern (1.0 on the stride-5
    // diagonal of a row-major [4,4] matrix, i.e. the true 4x4 identity), so value == x and
    // output projection == identity: query 0's causally-masked output must equal x's own row 0.
    close(
        "MultiheadAttention causal mask restricts query 0 to key 0 alone",
        &output[0..4],
        &x_values[0..4]
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>(),
    );
    println!("MultiheadAttention::causal_mask PASS");
    Ok(())
}

// --- architectures: ResNet-34/50 and VGG16/VGG16-bn (architectures-resnet-vgg) ---
const RESNET_TINY_INPUT: [f32; 384] = [
    -1.5255959e+00,
    -7.502318e-01,
    -6.539809e-01,
    -1.6094848e+00,
    -1.0016718e-01,
    -6.091889e-01,
    -9.7977227e-01,
    -1.6090963e+00,
    -7.121446e-01,
    3.03722e-01,
    -7.773143e-01,
    -2.5145525e-01,
    -2.2227049e-01,
    1.6871134e+00,
    2.2842517e-01,
    4.676355e-01,
    -6.969724e-01,
    -1.1607615e+00,
    6.995424e-01,
    1.9908163e-01,
    8.656924e-01,
    2.444039e-01,
    -6.6291136e-01,
    8.0730826e-01,
    1.1016806e+00,
    -1.7593604e-01,
    -2.2455578e+00,
    -1.446458e+00,
    6.1155282e-02,
    -6.1774445e-01,
    -7.9806983e-01,
    -1.3162321e-01,
    1.8793458e+00,
    -7.213178e-02,
    1.577706e-01,
    -7.734549e-01,
    1.990565e-01,
    4.5702778e-02,
    1.5295692e-01,
    -4.756788e-01,
    -1.1101983e-01,
    2.9273525e-01,
    -1.5784515e-01,
    -2.878714e-02,
    2.357111e+00,
    -1.0373387e+00,
    1.5747981e+00,
    -6.298472e-01,
    -9.273917e-01,
    5.451415e-01,
    6.628026e-02,
    -4.3704012e-01,
    7.626006e-01,
    4.4151092e-01,
    1.1651385e+00,
    2.0153918e+00,
    1.3741246e-01,
    9.386447e-01,
    -1.8600109e-01,
    -6.446393e-01,
    1.5392458e+00,
    -8.695876e-01,
    -3.3311536e+00,
    -7.4787223e-01,
    -2.5502462e-02,
    -1.0233306e+00,
    -5.9618515e-01,
    -1.0055307e+00,
    -2.106063e-01,
    -7.5475276e-03,
    1.6734272e+00,
    1.0342831e-02,
    -7.0395666e-01,
    -1.852658e-01,
    -9.962351e-01,
    -8.3125526e-01,
    -4.6102202e-01,
    -5.6008244e-01,
    3.9557618e-01,
    -9.822771e-01,
    -5.064866e-01,
    9.9775404e-02,
    -6.539734e-01,
    7.316937e-01,
    -1.4343859e+00,
    -5.0081307e-01,
    1.7163314e-01,
    -1.5999313e-01,
    2.546335e-01,
    -5.019573e-01,
    -1.0412e+00,
    7.322672e-01,
    -1.0483401e+00,
    -4.708772e-01,
    2.9113635e-01,
    1.9907043e+00,
    6.614453e-01,
    1.1899205e+00,
    8.165339e-01,
    -9.135236e-01,
    1.3851457e+00,
    -8.1384623e-01,
    -9.2757654e-01,
    1.1119633e+00,
    1.3352057e+00,
    6.042736e-01,
    -1.0344208e-01,
    -1.5121692e-01,
    -2.102083e+00,
    -6.2002194e-01,
    -1.478231e+00,
    -1.1334175e+00,
    8.737963e-01,
    -5.602594e-01,
    1.2857845e+00,
    8.1682384e-01,
    2.053041e-01,
    3.0510718e-01,
    5.35687e-01,
    -4.31185e-01,
    2.5581384e+00,
    -2.3336388e-01,
    -1.347213e-02,
    1.8606348e+00,
    -1.9804063e+00,
    1.7985829e+00,
    1.0181159e-01,
    3.40006e-01,
    7.1236455e-01,
    -1.7765073e+00,
    3.5386458e-01,
    1.1996132e+00,
    -3.029974e-01,
    -1.7618417e+00,
    6.348446e-01,
    -8.043591e-01,
    -1.6111118e+00,
    -1.8716129e+00,
    5.430836e-01,
    6.606786e-01,
    2.2952116e+00,
    6.7490596e-01,
    1.7133216e+00,
    -1.7942734e+00,
    -1.3632672e+00,
    -9.832196e-01,
    1.5112667e+00,
    6.4187074e-01,
    4.729638e-01,
    -4.285901e-01,
    5.5137074e-01,
    -1.5473709e+00,
    5.181121e-01,
    1.0653535e-01,
    2.6924077e-01,
    1.324768e+00,
    1.7460191e+00,
    1.854969e+00,
    -7.063691e-01,
    2.5570862e+00,
    4.175343e-01,
    -2.127186e-01,
    -8.39958e-01,
    -4.2001787e-01,
    -6.240363e-01,
    -9.772961e-01,
    8.748428e-01,
    9.872814e-01,
    3.0957633e-01,
    1.52069e+00,
    1.2052339e+00,
    -1.815591e+00,
    -4.0346155e-01,
    -9.591452e-01,
    -5.207704e-03,
    -7.886314e-02,
    8.436543e-01,
    1.1657013e+00,
    5.269322e-01,
    1.6192533e+00,
    -9.6397626e-01,
    1.4152038e-01,
    -1.6366096e-01,
    -3.5822257e-01,
    1.7222793e+00,
    -3.035756e-01,
    2.388742e-01,
    1.3440012e+00,
    1.0322569e-01,
    1.1003542e+00,
    -3.416802e-01,
    9.473389e-01,
    -5.6851596e-01,
    8.375962e-01,
    1.7836607e+00,
    -1.9542466e-01,
    5.149161e-01,
    -1.8474776e+00,
    -2.9167426e+00,
    -5.673299e-01,
    -5.412802e-01,
    8.95174e-01,
    -8.82507e-01,
    5.3181124e-01,
    -1.5457772e+00,
    -1.7329982e-01,
    7.2824633e-01,
    5.706102e-02,
    9.055172e-01,
    1.0462948e+00,
    -5.20597e-01,
    1.3547838e+00,
    2.3519313e-01,
    1.9142433e+00,
    1.8364111e+00,
    1.3245324e+00,
    -9.6900916e-01,
    1.2516364e+00,
    1.2103242e+00,
    -5.279206e-01,
    2.1856615e-01,
    -5.7430726e-01,
    1.4571251e+00,
    1.7709557e+00,
    1.6499138e+00,
    -4.3200457e-01,
    -2.710269e-01,
    -1.4391626e+00,
    1.2470404e+00,
    1.2738512e+00,
    3.9094925e-01,
    3.872105e-01,
    -7.982872e-02,
    3.417243e-01,
    9.488273e-01,
    -1.3839359e+00,
    1.7240863e+00,
    -2.3647652e+00,
    -9.294909e-01,
    2.936253e-01,
    2.1513203e-01,
    9.384637e-01,
    1.4657077e+00,
    -5.564744e-01,
    -7.448408e-01,
    -2.0215721e-01,
    -2.296679e-01,
    1.3313366e-03,
    3.752759e-01,
    -5.810679e-01,
    -5.7230884e-01,
    1.0097175e+00,
    -1.0564939e-01,
    -1.179696e+00,
    -9.0779595e-02,
    5.6311435e-01,
    -1.2560141e+00,
    8.95555e-01,
    1.6747737e-01,
    7.5142086e-01,
    2.4142299e+00,
    1.020584e+00,
    -4.404838e-01,
    -1.7341677e+00,
    -1.236225e+00,
    1.5785813e+00,
    -1.1160507e+00,
    7.6777023e-01,
    -5.8820677e-01,
    2.1188903e+00,
    -5.4219025e-01,
    -2.4592547e+00,
    -1.1108288e+00,
    -1.1187209e+00,
    7.5799555e-01,
    -4.9565765e-01,
    -1.9700006e-01,
    -3.3396218e-02,
    7.192915e-01,
    1.0644146e+00,
    8.3402544e-01,
    -1.9162164e+00,
    -3.4202927e-01,
    -6.6049206e-01,
    3.1508535e-01,
    1.1422518e+00,
    3.0550566e-01,
    -5.7888174e-01,
    -2.3828252e-01,
    -1.3541743e+00,
    2.686894e-01,
    1.1455697e-01,
    -1.5562972e+00,
    -1.0757437e+00,
    -8.751946e-01,
    -4.7281876e-01,
    9.912368e-01,
    -5.8622282e-02,
    1.1787646e+00,
    6.22185e-01,
    7.878501e-01,
    1.3685523e+00,
    -8.506898e-01,
    5.1260746e-01,
    1.0476325e+00,
    -3.1758463e-01,
    1.3948506e-01,
    2.3402624e+00,
    -6.1160916e-01,
    8.160271e-01,
    2.47723e-01,
    -3.867267e-01,
    1.9948451e-01,
    7.9926956e-01,
    -2.619034e-01,
    1.5132962e-01,
    1.1981666e+00,
    -2.2832582e+00,
    -1.0129594e+00,
    -8.878909e-01,
    6.5221924e-01,
    -8.726203e-01,
    3.5253752e-02,
    -3.365303e-01,
    1.402332e+00,
    4.841214e-01,
    -7.0304507e-01,
    -8.267661e-01,
    7.74396e-01,
    6.9199395e-01,
    -1.01848e+00,
    -8.0337167e-01,
    -7.071132e-01,
    7.521183e-01,
    -1.9208279e-02,
    1.103333e+00,
    -6.0679215e-01,
    -5.252235e-01,
    -5.6618774e-01,
    6.6039857e-04,
    7.224588e-01,
    1.526352e-01,
    1.4495978e-01,
    -2.3442194e+00,
    3.60003e-01,
    4.6668175e-01,
    1.2830665e+00,
    1.2678007e+00,
    1.9883296e-01,
    5.440877e-01,
    -3.9781693e-01,
    -1.9291055e+00,
    2.323687e-01,
    8.6146563e-01,
    6.217573e-01,
    -1.7811896e+00,
    -7.820609e-01,
    -1.42367e+00,
    1.6090765e+00,
    -3.2787595e-02,
    8.532334e-01,
    5.506365e-02,
    -1.7425371e+00,
    8.7500376e-01,
    -2.7188172e+00,
    -2.2192061e-01,
    3.4208494e-01,
    1.1093477e+00,
    -5.731476e-01,
    9.577846e-01,
    9.820232e-04,
    -1.3847686e+00,
    -9.965023e-01,
    8.0734813e-01,
    1.1738863e+00,
    -9.3984646e-01,
    1.3109189e+00,
    -3.1670693e-01,
    -1.861041e-01,
    -5.764602e-01,
    6.866518e-01,
    4.2086706e-01,
    -1.0213808e+00,
    9.8856664e-01,
];
const RESNET_TINY_PARAMS: &[(&str, &[f32])] = &[
    (
        "0.weight",
        &[
            -6.1750255e-04,
            -5.346661e-04,
            -1.8549617e-02,
            -8.014122e-02,
            4.4245124e-02,
            -4.0995695e-02,
            -5.0842274e-02,
            -1.6731886e-02,
            -6.788362e-02,
            -6.320433e-02,
            5.703765e-02,
            5.5461332e-02,
            -6.069923e-02,
            -7.718778e-02,
            -6.139907e-02,
            -7.806427e-02,
            -3.1767e-02,
            -6.961263e-02,
            3.378864e-02,
            6.855516e-02,
            2.2117248e-02,
            -1.6729694e-02,
            -2.7733725e-02,
            -3.299333e-02,
            -1.6341635e-03,
            4.5231767e-02,
            -3.979329e-02,
            2.4156675e-02,
            6.539642e-02,
            4.4591326e-02,
            1.4816876e-02,
            3.7612654e-03,
            -7.319486e-03,
            -7.9545006e-02,
            -4.284409e-02,
            -7.437254e-02,
            2.1824876e-02,
            5.1448677e-02,
            1.9003473e-02,
            6.84019e-02,
            -2.4926115e-02,
            -6.45403e-02,
            1.619722e-02,
            4.441007e-02,
            -1.6212441e-02,
            -1.7436827e-02,
            -6.12395e-02,
            8.198338e-02,
            -7.879582e-02,
            -3.344282e-02,
            1.3732568e-02,
            4.16692e-02,
            -5.462411e-02,
            -1.588664e-02,
            3.513005e-02,
            -5.4441575e-02,
            -3.399959e-02,
            -1.6194073e-02,
            3.264759e-02,
            6.883531e-02,
            3.0553024e-03,
            -7.401218e-02,
            -1.03823e-02,
            4.432766e-03,
            3.26067e-02,
            -7.121516e-02,
            -6.761494e-02,
            3.9112717e-02,
            4.9489047e-02,
            -1.29062105e-02,
            -1.2714413e-02,
            -6.613375e-02,
            -5.591564e-02,
            1.0666284e-03,
            2.8645456e-02,
            -2.3723034e-02,
            -3.591637e-02,
            -3.7467964e-02,
            -3.0094557e-02,
            -8.09839e-02,
            2.9957645e-02,
            3.1069629e-02,
            3.130732e-02,
            -3.2124802e-02,
            6.8489246e-02,
            -7.4235566e-02,
            5.492767e-02,
            1.7793205e-02,
            -1.6974114e-02,
            -5.566254e-03,
            -4.3066338e-02,
            -6.475905e-02,
            6.1719716e-02,
            7.2533295e-02,
            8.133898e-04,
            2.629123e-02,
            -1.329418e-02,
            -3.3642367e-02,
            3.4104824e-02,
            4.4275083e-02,
            8.727401e-03,
            7.447844e-02,
            6.464496e-03,
            1.1490037e-02,
            7.468242e-02,
            2.986994e-02,
            6.889562e-03,
            -5.518507e-02,
            -7.651296e-02,
            -7.443373e-02,
            1.0298726e-02,
            -6.394729e-02,
            -5.1923417e-02,
            5.2184e-02,
            -6.484038e-02,
            -2.5446024e-02,
            -2.0880718e-02,
            -9.517576e-03,
            6.4818496e-03,
            3.6204662e-02,
            -3.215016e-02,
            -3.681902e-02,
            5.7113953e-02,
            8.135661e-02,
            7.1261585e-02,
            6.59543e-02,
            7.432261e-02,
            4.7427595e-02,
            -5.346096e-02,
            -6.665057e-02,
            4.847688e-02,
            -9.2878565e-03,
            -3.7967615e-02,
            8.850362e-03,
            1.1052385e-02,
            2.891836e-02,
            -5.7622895e-02,
            -1.726844e-02,
            3.8518682e-02,
            -8.09167e-02,
            -7.724625e-02,
            5.889902e-02,
            -4.0124167e-02,
            -7.044509e-02,
            -4.8146103e-02,
            2.3023453e-02,
            -6.834907e-02,
            3.8485173e-02,
            7.089845e-02,
            3.9631415e-02,
            -7.09251e-02,
            -4.671714e-02,
            3.6803473e-02,
            2.912806e-02,
            8.228147e-02,
            3.9680105e-02,
            3.9975118e-02,
            -1.983402e-02,
            5.2356545e-02,
            -5.82243e-02,
            4.337679e-03,
            -1.7345702e-02,
            -5.7012003e-02,
            -4.0852558e-02,
            -4.2285424e-02,
            -6.796909e-02,
            3.226915e-02,
            -6.7936726e-02,
            1.3954118e-02,
            4.4690553e-02,
            6.2285177e-02,
            4.3040697e-02,
            -7.700984e-02,
            6.5486215e-02,
            8.244684e-02,
            -8.404166e-03,
            -5.9596263e-02,
            5.6433916e-02,
            7.211888e-02,
            6.347563e-02,
            -4.25202e-02,
            -5.8178622e-02,
            6.390062e-02,
            5.104372e-02,
            5.2038886e-02,
            3.6785372e-03,
            -1.8908985e-02,
            4.563893e-02,
            4.8358962e-02,
            -5.8141988e-02,
            -2.8945182e-02,
            2.6521913e-03,
            -3.6578856e-02,
            -4.540317e-02,
            6.771897e-02,
            -2.5500571e-02,
            -2.9760252e-03,
            -4.806074e-02,
            4.6219334e-02,
            -1.7933462e-02,
            5.275008e-02,
            2.8186655e-02,
            -4.9633894e-02,
            1.0961457e-02,
            8.199472e-02,
            -4.915009e-02,
            7.414969e-02,
            4.0885434e-02,
            3.273429e-02,
            -1.799453e-03,
            3.985025e-02,
            -5.7783682e-02,
            1.1142272e-02,
            3.4697205e-03,
            4.496202e-02,
            6.922247e-02,
            5.5300783e-02,
            5.3167615e-02,
            -5.169564e-02,
            -8.967906e-03,
            -4.85636e-02,
            -6.2347237e-02,
            2.3663098e-02,
            -6.911259e-02,
            1.5369397e-02,
            -5.6622576e-02,
            -2.8913904e-02,
            -4.46257e-02,
            -6.394612e-02,
            -4.7892157e-02,
            6.4445205e-02,
            7.297854e-02,
            -5.7164785e-02,
            5.7729535e-02,
            -1.4841249e-02,
            7.542893e-02,
            -4.2607095e-02,
            -2.9648175e-02,
            3.2109957e-02,
            -7.639826e-02,
            3.7319347e-02,
            6.9569774e-02,
            1.4650003e-02,
            5.8172025e-02,
            3.316963e-02,
            2.9824888e-02,
            3.5092026e-02,
            4.1334916e-02,
            -4.8856415e-02,
            1.0443938e-02,
            -2.8028503e-02,
            4.8820615e-02,
            2.4917373e-02,
            -6.139728e-04,
            4.021421e-02,
            6.982056e-02,
            4.527845e-02,
            -1.63045e-02,
            -5.7608914e-02,
            -4.4451874e-02,
            -1.0410233e-02,
            1.034829e-02,
            1.8628206e-02,
            2.6044747e-02,
            3.1491606e-03,
            -1.8833552e-02,
            -5.580436e-02,
            3.375299e-02,
            1.9110683e-02,
            -5.7953055e-04,
            -8.136825e-02,
            -2.4371883e-02,
            5.11678e-02,
            1.0523697e-02,
            -6.6234924e-02,
            2.7601726e-02,
            7.919548e-02,
            -6.451519e-02,
            6.5114394e-02,
            -2.3730204e-02,
            -6.3559964e-02,
            -4.3229632e-02,
            4.462387e-02,
            5.099334e-02,
            -3.0225916e-02,
            6.660088e-02,
            7.738273e-02,
            -2.288407e-02,
            3.2414913e-02,
            -6.693524e-02,
            6.607537e-02,
            -3.0747673e-02,
            6.8337604e-02,
            -5.922475e-03,
            -7.365715e-02,
            2.0763911e-02,
            7.177349e-02,
            8.159104e-02,
            -5.628597e-02,
            2.925464e-02,
            7.2775565e-02,
            2.979433e-02,
            -1.3327108e-02,
            -4.029671e-02,
            1.6414443e-02,
            2.3352203e-03,
            -5.3564277e-02,
            7.2909724e-03,
            -7.172197e-02,
            -7.147679e-02,
            5.727286e-02,
            4.7799304e-02,
            7.587414e-03,
            4.0858082e-02,
            -6.2354475e-02,
            -8.206352e-03,
            -5.159906e-02,
            -5.874793e-02,
            -4.0241435e-02,
            2.510104e-02,
            -7.6866284e-02,
            -2.341273e-02,
            -7.968188e-02,
            -1.9892767e-02,
            7.328162e-02,
            -2.7672922e-02,
            -4.6826866e-02,
            2.8908724e-02,
            6.271342e-02,
            -1.2214042e-02,
            6.78368e-02,
            -5.974703e-02,
            -8.227472e-02,
            9.0217474e-04,
            6.753019e-02,
            -4.84998e-02,
            1.54376915e-02,
            6.802898e-02,
            5.9041303e-02,
            -4.1865364e-02,
            -1.3894348e-02,
            1.0296534e-02,
            6.368399e-02,
            7.579874e-02,
            -1.3572776e-02,
            7.38755e-02,
            7.3338866e-02,
            -2.2193683e-02,
            -3.775515e-02,
            5.0453193e-02,
            -2.111972e-02,
            -2.2519709e-04,
            3.1717658e-02,
            -5.214413e-02,
            3.629076e-02,
            -3.996052e-02,
            -4.8852373e-02,
            3.6991995e-02,
            7.348271e-02,
            8.233758e-02,
            3.0235944e-02,
            -5.8303807e-02,
            2.7285473e-02,
            8.0554396e-02,
            4.17101e-02,
            -3.495651e-02,
            8.245192e-02,
            -6.2204268e-02,
            5.9044093e-02,
            2.4258832e-02,
            4.27786e-02,
            -6.6862926e-02,
            3.0839672e-02,
            2.7233785e-02,
            5.127269e-02,
            -6.251821e-02,
            -8.163199e-02,
            6.187777e-02,
            -2.8866867e-02,
            -3.9777215e-04,
            -5.350362e-02,
            -2.6551174e-02,
            3.957551e-02,
            -2.1024121e-02,
            4.118281e-02,
            1.3203685e-04,
            9.477333e-03,
            -5.3986017e-02,
            1.7262885e-02,
            4.246193e-02,
            -1.9697322e-02,
            -2.958278e-02,
            -6.434025e-02,
            -7.976442e-02,
            -4.6494007e-02,
            1.5582756e-02,
            -4.7492795e-02,
            5.963044e-02,
            -4.6280116e-02,
            -4.309431e-02,
            7.7591695e-02,
            -6.820341e-02,
            -6.3465714e-02,
            1.827582e-02,
            5.5575565e-02,
            1.1367616e-03,
            5.5370394e-02,
            -1.8913655e-02,
            -3.596275e-02,
            -1.4022747e-02,
            5.8636617e-02,
            -3.9966114e-02,
            -2.0758612e-02,
            -4.3439254e-02,
            -9.387112e-03,
            1.13313645e-02,
            -7.856897e-02,
            1.09012835e-02,
            -4.7729217e-02,
            6.7818716e-02,
            -1.482482e-03,
            6.820311e-02,
            6.374817e-02,
            -5.5762082e-02,
            -6.2111255e-02,
            -2.411011e-02,
            5.274416e-02,
            3.829845e-03,
            -6.362043e-02,
            -4.8967063e-02,
            6.1309673e-03,
            -3.0415451e-02,
            -4.544539e-03,
            -3.0503795e-02,
            -3.8941067e-02,
            8.093678e-02,
            1.2383755e-02,
            -8.174854e-02,
            7.580179e-02,
            -7.825267e-02,
            -3.377749e-02,
            3.7230346e-02,
            3.372886e-02,
            -7.906905e-02,
            4.894096e-02,
            -3.9611485e-02,
            -6.261326e-02,
            8.127318e-02,
            -5.0191462e-02,
            -5.5041302e-02,
            7.8938864e-02,
            -5.218339e-02,
            7.483862e-02,
            -4.7516044e-02,
            6.26321e-02,
            1.5813075e-02,
            5.652258e-02,
            4.742158e-02,
            -3.0062702e-02,
            -7.118407e-03,
            -6.955281e-02,
            4.3678503e-02,
            4.6365727e-02,
            -1.7374698e-02,
            -2.0527633e-02,
            6.330414e-02,
            -4.6863444e-02,
            -1.8421415e-02,
            3.7216514e-03,
            2.99169e-02,
            -1.292558e-02,
            5.241026e-02,
            1.2033719e-02,
            -2.7544532e-02,
            7.0032686e-02,
            3.9422275e-03,
            1.9561805e-02,
            -2.3048496e-02,
            3.4079447e-03,
            -8.030238e-02,
            3.236694e-02,
            2.4366653e-02,
            -5.8330372e-02,
            -4.869209e-02,
            4.9404823e-03,
            6.780028e-02,
            -2.7567068e-02,
            -2.8116984e-02,
            -4.0243685e-02,
            2.242529e-02,
            -2.238842e-02,
            4.1504588e-02,
            3.902797e-02,
            -3.902437e-02,
            -1.59125e-02,
            -5.337469e-02,
            -7.911752e-02,
            -3.877133e-02,
            7.893254e-03,
            7.77707e-02,
            -4.8885625e-02,
            -7.798097e-02,
            7.6278664e-02,
            -1.8370405e-02,
            -2.0646859e-02,
            1.781934e-02,
            4.4159335e-03,
            -1.4810022e-02,
            -4.0176433e-02,
            -4.6285555e-02,
            -5.0925594e-02,
            6.4626776e-02,
            -2.885377e-02,
            -7.3535934e-02,
            4.227381e-03,
            4.1454386e-02,
            -6.7601256e-02,
            7.232355e-02,
            3.9547436e-02,
            6.995179e-02,
            -1.754445e-02,
            -5.3562533e-02,
            4.091271e-02,
            4.770929e-02,
            1.7630335e-02,
            -9.382894e-03,
            -7.5380385e-02,
            -2.502021e-02,
            -5.3731993e-02,
            2.3629589e-02,
            -1.4758924e-02,
            -5.4722283e-02,
            -4.23275e-03,
            2.6276305e-03,
            -6.1293527e-02,
            -6.13666e-03,
            5.9042383e-02,
            -5.549835e-02,
            -3.5192434e-02,
            6.82624e-02,
            -8.478822e-03,
            -6.6669255e-02,
            2.97165e-02,
            -2.7681457e-02,
            2.2922633e-03,
            6.5742254e-02,
            -5.8570556e-02,
            -7.648819e-02,
            -7.1153394e-03,
            1.343071e-02,
            3.0659517e-02,
            3.381033e-02,
            1.6692135e-02,
            6.842628e-02,
            7.000604e-02,
            8.029057e-02,
            5.2443158e-02,
            -2.765018e-02,
            5.4096365e-03,
            -2.3480278e-02,
            7.8127556e-02,
            2.4294524e-02,
            -5.497098e-02,
            -6.8294235e-02,
            5.2378528e-02,
            -1.8860895e-02,
            -2.9550845e-02,
            -7.481302e-02,
            7.830631e-02,
            -3.6679183e-03,
            1.801037e-02,
            2.066909e-02,
            -5.96499e-03,
            -5.0233122e-02,
            -6.2874936e-02,
            -6.245257e-03,
            -7.409231e-02,
            2.7894342e-02,
            4.0976226e-02,
            -4.1650645e-02,
            -3.910123e-02,
            2.6076926e-02,
            -7.4879825e-02,
            1.6671555e-02,
            5.616012e-02,
            -1.701416e-03,
            -7.928614e-02,
            3.132198e-02,
        ],
    ),
    ("1.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("1.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "4.conv1.weight",
        &[
            1.3255316e-01,
            -4.7448598e-02,
            -4.1784506e-02,
            1.16998315e-01,
            -3.769505e-02,
            -1.2787245e-01,
            1.2939703e-01,
            -1.1352792e-01,
            -3.614372e-02,
            -1.2357521e-01,
            1.5175959e-02,
            -4.8712872e-02,
            -2.4947286e-02,
            1.461255e-01,
            1.2255057e-01,
            3.8312018e-02,
            -1.1650543e-01,
            5.4663382e-02,
            -1.469612e-01,
            -2.7524313e-02,
            -1.2916432e-01,
            2.4219632e-02,
            1.4187348e-01,
            -1.4601418e-01,
            -1.5060386e-01,
            -1.5192083e-01,
            1.7702222e-02,
            -7.800609e-02,
            -2.1544974e-02,
            9.136637e-02,
            1.5561494e-01,
            -1.0477225e-02,
            1.5673487e-01,
            5.8032595e-02,
            -1.2189841e-01,
            8.675794e-02,
            7.367587e-02,
            1.0466882e-01,
            -2.2902032e-02,
            -1.4907211e-01,
            4.880172e-02,
            1.0336759e-01,
            7.265123e-02,
            4.3300074e-02,
            1.2940636e-01,
            3.7959915e-02,
            -1.185029e-01,
            -1.06257856e-01,
            -1.4420351e-01,
            5.270074e-02,
            -5.1326137e-02,
            -1.05658494e-01,
            -1.3225123e-01,
            1.6600017e-01,
            1.5433596e-01,
            1.547887e-02,
            1.3416037e-01,
            -1.802228e-02,
            1.0153195e-02,
            -9.879359e-02,
            1.5789053e-01,
            6.2136453e-02,
            -1.6122173e-01,
            2.1438142e-02,
            1.321406e-01,
            1.6639009e-01,
            1.5047672e-01,
            -4.0608667e-02,
            -2.4020612e-02,
            1.4199075e-01,
            -4.7164004e-02,
            -1.5994841e-01,
            -1.405975e-01,
            -6.6300035e-02,
            -2.9274404e-02,
            -7.952005e-02,
            7.844448e-02,
            -1.48495e-01,
            -6.288724e-02,
            -1.5578476e-01,
            5.401417e-02,
            1.0677268e-01,
            1.4322022e-01,
            1.0236605e-01,
            -1.146273e-01,
            -1.2555616e-01,
            6.5387905e-02,
            -2.1172306e-02,
            1.1054349e-01,
            1.7418822e-02,
            1.578276e-01,
            7.223376e-02,
            1.2796949e-01,
            -7.915463e-02,
            -8.742684e-02,
            9.614885e-03,
            -7.1613334e-02,
            -9.868556e-02,
            -2.4830997e-02,
            -1.4513329e-01,
            -1.8269977e-02,
            -9.229307e-02,
            -7.609242e-02,
            2.391108e-02,
            -6.609182e-03,
            -1.5291974e-01,
            -5.244748e-02,
            -6.9912575e-02,
            8.731882e-02,
            5.3198576e-02,
            1.14135206e-01,
            1.6276845e-01,
            -1.3402049e-01,
            -4.873125e-02,
            -7.709873e-02,
            1.1800647e-02,
            -4.5095526e-02,
            -4.156659e-02,
            6.8863235e-05,
            -1.6235924e-01,
            -8.2087025e-02,
            -1.08058296e-01,
            1.4610562e-01,
            8.324802e-03,
            1.2540585e-01,
            8.661449e-03,
            -1.4578581e-02,
            1.5938807e-01,
            -1.10968694e-01,
            4.5185387e-02,
            1.4041927e-01,
            2.9476484e-02,
            -1.6019922e-02,
            -9.691658e-02,
            -1.8034976e-02,
            1.132887e-01,
            1.5550117e-02,
            -1.4110407e-04,
            -1.2982975e-01,
            1.3708344e-01,
            -8.951823e-03,
            1.422457e-01,
            3.7853837e-02,
            5.3821724e-02,
            1.3894469e-01,
            -1.3258472e-01,
            7.436832e-02,
            1.3516954e-01,
            -7.215361e-02,
            1.5814184e-01,
            -7.352029e-02,
            -4.0091418e-02,
            1.1394602e-01,
            -1.2348052e-01,
            1.2580708e-01,
            4.431401e-02,
            5.9119146e-02,
            9.0661466e-02,
            -2.6417296e-02,
            -2.6702803e-02,
            -6.935142e-02,
            -6.101207e-02,
            1.4496395e-01,
            -6.809678e-02,
            -1.380801e-01,
            -9.862294e-02,
            -1.15775466e-01,
            -6.859112e-02,
            8.405274e-02,
            1.4000717e-01,
            8.258204e-02,
            -9.52253e-02,
            2.5665581e-02,
            -1.0663426e-01,
            2.3596644e-02,
            -1.1469275e-01,
            4.985529e-02,
            3.953451e-02,
            9.989756e-02,
            -1.1553647e-01,
            1.4180951e-01,
            -1.3996193e-01,
            6.694989e-02,
            1.5644133e-01,
            -1.5026526e-01,
            -2.684164e-02,
            2.2404373e-02,
            -7.5153194e-02,
            -1.0281128e-01,
            -1.3351732e-01,
            1.5066126e-01,
            -2.9477596e-02,
            -7.710944e-02,
            2.6941737e-02,
            1.07265934e-01,
            1.0086741e-01,
            5.995518e-02,
            7.414375e-02,
            1.576666e-01,
            1.5345931e-01,
            1.5785918e-01,
            -8.641112e-02,
            -1.167688e-01,
            -7.123947e-02,
            3.9446753e-02,
            -7.956682e-02,
            1.4461895e-02,
            -3.246238e-02,
            -3.590125e-02,
            1.3286728e-01,
            -1.6261348e-01,
            5.9604704e-02,
            -1.5567474e-01,
            1.3229516e-01,
            1.4459667e-01,
            -1.169467e-01,
        ],
    ),
    ("4.bn1.scale", &[1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00]),
    ("4.bn1.bias", &[0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "4.conv2.weight",
        &[
            3.867728e-04,
            8.98192e-02,
            2.1957217e-02,
            1.0247336e-01,
            7.749605e-02,
            1.3312203e-01,
            1.22327514e-01,
            -1.2669837e-01,
            3.287225e-02,
            -8.280497e-02,
            -8.5017696e-02,
            -1.3531303e-01,
            1.3531868e-01,
            8.251141e-02,
            1.1954222e-01,
            2.9956868e-02,
            -7.896794e-02,
            -1.3035984e-01,
            -7.882761e-02,
            1.346344e-01,
            4.956561e-02,
            9.566418e-02,
            6.0148343e-02,
            -3.2156583e-02,
            2.4240281e-02,
            2.7535005e-02,
            -1.19336344e-01,
            4.1956544e-02,
            -9.534772e-03,
            1.1115845e-01,
            1.6074464e-02,
            -9.066741e-03,
            -9.8779865e-02,
            2.9449774e-02,
            -1.2853e-01,
            1.3625395e-02,
            -6.380264e-02,
            1.3395305e-01,
            5.9978433e-02,
            1.3577332e-01,
            -7.846329e-02,
            5.2262004e-02,
            -4.7015898e-02,
            -5.8155287e-02,
            1.8545255e-02,
            8.8654205e-02,
            5.518603e-02,
            -9.97496e-02,
            3.6858562e-02,
            -1.0904796e-02,
            6.636643e-02,
            -7.6536626e-02,
            -4.6155967e-02,
            4.962745e-02,
            -9.4627224e-02,
            -5.959132e-02,
            -1.3590857e-01,
            1.1762519e-02,
            9.392828e-02,
            -1.5206358e-02,
            2.2453068e-02,
            -1.8785037e-02,
            -1.2558661e-01,
            6.952093e-02,
            1.0757885e-01,
            5.4545183e-02,
            5.8886375e-02,
            2.9576082e-02,
            8.344955e-02,
            1.20012686e-01,
            1.9795947e-02,
            9.6107446e-02,
            -5.3626157e-02,
            1.81754e-02,
            8.704476e-02,
            -1.3539322e-01,
            -1.076071e-02,
            5.9152488e-02,
            1.1303595e-01,
            -2.5702627e-02,
            -1.2311558e-01,
            7.368954e-02,
            -4.3196365e-02,
            -1.1575768e-02,
            -8.955667e-03,
            -1.3606107e-01,
            5.1655013e-02,
            -3.9509162e-02,
            -6.8811392e-03,
            -8.382581e-03,
            6.174248e-02,
            6.967994e-03,
            -1.0625944e-01,
            -1.3325514e-01,
            2.4769908e-02,
            -8.534842e-02,
            1.3476363e-01,
            -4.0310384e-03,
            1.03056535e-01,
            3.0206839e-02,
            -1.0394696e-01,
            -4.9221423e-02,
            -4.1596197e-02,
            2.047185e-02,
            4.270147e-02,
            -7.4085325e-02,
            -3.2523192e-02,
            8.843177e-02,
            7.394849e-02,
            1.0427345e-01,
            1.3455686e-01,
            1.9284587e-02,
            -1.1319377e-01,
            -5.4568723e-02,
            -3.8923647e-02,
            1.3227373e-01,
            -1.3417651e-01,
            -8.835132e-02,
            8.352146e-02,
            8.451281e-02,
            -1.9878017e-02,
            -8.073552e-02,
            -1.3100363e-01,
            -1.0240623e-01,
            -8.666245e-02,
            -5.3961165e-02,
            -2.3062102e-02,
            -8.485366e-03,
            9.2005834e-02,
            -1.1546763e-02,
            1.2431757e-01,
            -3.169352e-02,
            -8.4542675e-04,
            -5.4524776e-02,
            1.2081742e-01,
            -4.57137e-02,
            -3.5079524e-02,
            2.89614e-03,
            -5.140777e-02,
            -1.254201e-01,
            8.3375186e-02,
            7.9590015e-02,
            -7.795845e-02,
            -1.2487846e-01,
            3.189602e-02,
            -9.902187e-02,
            9.0471886e-02,
            1.17328e-02,
            6.538139e-02,
            -1.3587824e-03,
            5.1276516e-03,
            1.27722e-01,
            -1.14618614e-01,
            1.0622057e-01,
            2.0287095e-02,
            -1.37165645e-02,
            8.626148e-02,
            7.934457e-02,
            9.849857e-02,
            2.5502315e-02,
            9.423133e-02,
            -1.2225181e-02,
            -2.7543878e-02,
            7.3097005e-02,
            -1.2848215e-01,
            -4.375896e-02,
            5.681044e-02,
            -2.7203517e-02,
            1.3646402e-02,
            7.562707e-02,
            3.0401116e-02,
            9.217343e-02,
            -1.310939e-01,
            1.0728456e-01,
            -5.0621472e-02,
            -1.08114906e-01,
            -8.913895e-02,
            7.4362513e-03,
            8.6071044e-02,
            1.0042734e-01,
            -1.1396302e-01,
            1.4291305e-02,
            -5.405146e-02,
            2.5541605e-02,
            -2.5626268e-02,
            -9.223109e-02,
            5.506824e-02,
            1.2665422e-01,
            2.4096016e-02,
            -4.3767296e-02,
            -6.0144644e-02,
            6.3214086e-02,
            1.7412871e-02,
            -7.594588e-02,
            -5.3505026e-02,
            4.883733e-04,
            8.641989e-02,
            -1.0674095e-01,
            -5.4498933e-02,
            1.2105846e-01,
            -2.1531412e-02,
            -2.616506e-02,
            9.924929e-02,
            -1.15850165e-01,
            -4.601026e-02,
            3.576779e-02,
            -1.2898256e-01,
            -1.1569393e-02,
            -1.1960732e-01,
            4.1598044e-02,
            3.5588548e-02,
            9.5740154e-02,
            -1.283088e-01,
            -1.0612814e-01,
            -1.23050595e-02,
            1.9999375e-03,
            -2.4606062e-02,
            -5.8296178e-02,
            5.271344e-02,
            -8.368938e-04,
            1.1175206e-01,
            -6.348348e-02,
            8.892681e-02,
            6.273657e-02,
            -5.7600904e-02,
            -1.1295472e-01,
            5.2703056e-02,
            -1.3125235e-01,
            7.7081537e-03,
            -1.206357e-01,
            -8.279134e-02,
            -2.9275596e-02,
            1.1464556e-01,
            -5.2682552e-02,
            5.1403455e-02,
            -8.640646e-03,
            -5.3695846e-02,
            7.886625e-02,
            -4.6635386e-02,
            4.5451578e-02,
            5.929075e-02,
            4.5390565e-02,
            -1.1125699e-01,
            -9.462519e-02,
            -7.5069025e-02,
            -1.2595841e-01,
            -3.339013e-02,
            4.0810466e-02,
            -1.0878459e-01,
            -9.172683e-02,
            1.17018886e-01,
            -1.321997e-01,
            5.887823e-02,
            1.1387516e-01,
            8.142816e-02,
            -3.3228215e-02,
            5.6722414e-02,
            1.7265167e-02,
            1.00420006e-01,
            1.337227e-01,
            6.9162995e-02,
            -3.730492e-02,
            1.3063708e-01,
            -9.803459e-02,
            6.262306e-03,
            1.2380225e-01,
            1.8126702e-02,
            2.3037104e-02,
            -5.684131e-02,
            -1.1921943e-01,
            2.6630431e-02,
            9.136491e-02,
            -7.433652e-02,
            2.2591867e-02,
            -8.737423e-02,
            -5.2430816e-02,
            4.944647e-03,
            9.5674634e-02,
            4.527487e-02,
            6.90071e-02,
            -1.5950656e-02,
            -2.1909442e-02,
            9.4219275e-02,
            -1.7558726e-02,
            6.1135474e-03,
            -9.374478e-02,
            -1.281328e-01,
            -1.2365913e-01,
            -5.7453506e-02,
            -1.0560661e-01,
            -9.7254656e-02,
            -9.603522e-03,
            5.3341065e-02,
            -9.086624e-02,
            -7.233749e-02,
            -5.051762e-02,
            2.9283626e-02,
            7.61099e-02,
            1.00373074e-01,
            8.922655e-03,
            5.9298553e-02,
            -1.5277266e-02,
            1.2390934e-01,
            1.1770772e-01,
            3.2667603e-02,
            -1.283222e-01,
            -1.1843476e-01,
            -7.35373e-02,
            -1.3385187e-01,
            -9.926622e-03,
            -1.3445911e-02,
            -1.9127036e-02,
            2.1817412e-03,
            6.956155e-02,
            -1.3531964e-01,
            -7.874024e-02,
            6.74707e-02,
            -8.786901e-02,
            -6.258325e-02,
            4.776876e-02,
            1.74356e-02,
            3.9118707e-02,
            -8.6378716e-02,
            -9.433765e-02,
        ],
    ),
    ("4.bn2.scale", &[1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00]),
    ("4.bn2.bias", &[0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "4.downsample.0.weight",
        &[
            1.965155e-01,
            9.348315e-02,
            -4.627323e-02,
            -3.5600364e-02,
            2.4163389e-01,
            2.4853092e-01,
            -3.8484484e-01,
            4.5535642e-01,
            2.5296807e-03,
            2.2117609e-01,
            -4.790926e-01,
            -4.2752022e-01,
            -3.789171e-01,
            -1.3364774e-01,
            3.8132215e-01,
            -2.753415e-01,
            1.1847788e-01,
            3.0740237e-01,
            2.8514075e-01,
            -2.4150753e-01,
            -4.9803996e-01,
            -2.540704e-01,
            -3.8709325e-01,
            -4.733163e-01,
        ],
    ),
    (
        "4.downsample.1.scale",
        &[1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00],
    ),
    (
        "4.downsample.1.bias",
        &[0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00],
    ),
    (
        "5.conv1.weight",
        &[
            -1.1438665e-01,
            -8.678012e-02,
            -1.2660661e-01,
            4.177873e-02,
            -1.0755384e-01,
            -6.023181e-02,
            2.005437e-03,
            1.3057142e-01,
            1.4237836e-02,
            8.6274214e-02,
            1.0891858e-01,
            -6.355802e-02,
            1.0723165e-01,
            -1.201867e-02,
            1.161702e-01,
            -1.3416617e-01,
            -7.759924e-02,
            -1.3482565e-01,
            -1.16540685e-01,
            -9.026922e-02,
            -9.897411e-02,
            -9.58189e-02,
            8.700141e-02,
            1.2978318e-02,
            -6.070104e-02,
            -6.0555395e-02,
            4.4651378e-02,
            1.00797564e-01,
            -1.2963837e-01,
            9.2766814e-02,
            1.1951912e-01,
            -7.5167306e-02,
            6.0131896e-02,
            3.2219574e-02,
            -2.354264e-02,
            5.5231016e-02,
            -2.282073e-02,
            5.5520326e-02,
            3.5665944e-02,
            -8.48423e-03,
            1.4423274e-03,
            -6.4657405e-02,
            5.6376994e-02,
            -1.1869835e-01,
            1.3530381e-01,
            2.432592e-02,
            2.3545528e-02,
            -1.2496007e-01,
            -1.3091138e-01,
            -1.26884e-01,
            1.1142179e-01,
            -2.0690285e-02,
            1.0999843e-02,
            -1.0978845e-01,
            -4.279543e-02,
            -1.3414511e-01,
            1.16752855e-01,
            -9.5117085e-02,
            4.2880535e-02,
            2.524227e-03,
            -6.9107205e-02,
            5.8586914e-02,
            -1.6962279e-02,
            -1.271598e-01,
            -1.5152078e-02,
            9.361315e-02,
            1.14726104e-01,
            6.728675e-02,
            1.1413433e-01,
            8.9502715e-02,
            3.332727e-02,
            -5.5109788e-02,
            5.647978e-02,
            -8.4518164e-02,
            -1.2681726e-01,
            -1.618523e-02,
            1.3466537e-01,
            -7.404876e-02,
            -9.929628e-02,
            3.374561e-02,
            -7.3434256e-02,
            -5.0539404e-02,
            -8.805264e-02,
            -7.259423e-02,
            2.1940215e-02,
            1.0045422e-01,
            -2.9105537e-02,
            -3.512503e-02,
            -7.4296813e-03,
            -1.178051e-02,
            -1.5640598e-03,
            1.9773008e-02,
            -4.042431e-02,
            -7.422268e-02,
            -5.194593e-02,
            -1.2923165e-01,
            -5.5912305e-02,
            -2.8792609e-02,
            1.2569894e-01,
            -4.030545e-02,
            1.1940305e-01,
            -1.2008549e-01,
            -7.771369e-02,
            1.0054113e-02,
            6.4185515e-02,
            -1.1681541e-01,
            1.1649228e-01,
            1.339586e-01,
            -4.515752e-02,
            -1.459184e-02,
            8.7611854e-02,
            1.0183667e-01,
            -3.7108744e-03,
            -6.490006e-02,
            1.0726404e-01,
            6.851756e-02,
            9.256478e-02,
            5.5568375e-02,
            -4.7170725e-02,
            -2.4544386e-02,
            -9.755125e-02,
            -9.240174e-02,
            -1.0165431e-01,
            -6.862946e-02,
            -8.4080935e-02,
            -3.640932e-02,
            -9.693766e-02,
            6.900791e-02,
            -1.5761243e-02,
            -4.6446852e-02,
            9.737261e-02,
            -1.1954955e-01,
            -4.1984137e-02,
            7.669945e-02,
            1.987143e-02,
            6.105183e-02,
            -9.371153e-03,
            -2.3990458e-02,
            2.367341e-03,
            -8.24491e-02,
            3.5394106e-02,
            -5.0392188e-02,
            1.2816405e-01,
            4.165721e-02,
            3.704335e-02,
            1.1112024e-01,
            6.99782e-02,
            3.5884168e-02,
            7.7602826e-02,
            -5.5257443e-02,
            4.083582e-03,
            1.3365842e-01,
            -5.4148257e-02,
            3.7469234e-02,
            1.1911387e-01,
            -5.799824e-03,
            -3.2758612e-02,
            -5.3647734e-02,
            -4.1033637e-02,
            -8.5663185e-02,
            1.0763107e-01,
            -6.5095344e-04,
            -8.251381e-02,
            -3.811602e-02,
            1.13866076e-01,
            3.408894e-02,
            -8.8432334e-02,
            -1.3547106e-01,
            -2.441769e-02,
            6.78176e-02,
            -3.6522485e-02,
            8.326111e-03,
            3.9831724e-02,
            9.889358e-02,
            -5.478057e-02,
            -1.0661002e-01,
            -9.452872e-02,
            1.18852496e-01,
            -5.6427933e-02,
            1.0992395e-01,
            1.09945185e-01,
            5.381502e-02,
            9.226953e-02,
            -4.9237464e-02,
            6.827997e-02,
            9.2781976e-02,
            -1.3246426e-01,
            3.1748753e-02,
            9.21492e-02,
            -3.4982596e-02,
            -1.2235173e-02,
            -4.4689357e-02,
            -8.680162e-02,
            8.5777014e-02,
            -4.2671055e-02,
            1.17249e-01,
            1.1780673e-02,
            7.111185e-02,
            -1.10715605e-01,
            -1.1820242e-01,
            -2.7205676e-02,
            -1.1112551e-01,
            -9.411604e-02,
            -1.27499e-01,
            -2.487005e-02,
            -1.130555e-01,
            -1.0041275e-01,
            9.83177e-03,
            -1.21620655e-01,
            -3.3665065e-02,
            1.0604568e-01,
            6.1118733e-02,
            -1.13641866e-01,
            1.3111606e-01,
            5.1245093e-02,
            -2.7677096e-02,
            1.1928189e-01,
            4.901848e-02,
            5.117386e-02,
            -6.101515e-02,
            3.1623647e-02,
            -4.8057582e-02,
            -1.1846757e-01,
            -1.2666124e-01,
            -5.8133546e-02,
            1.3046013e-01,
            1.18205305e-02,
            5.7154775e-02,
            -5.8960384e-03,
            -7.588785e-02,
            1.0833488e-01,
            1.20751776e-01,
            9.242668e-02,
            -6.200718e-02,
            7.987598e-02,
            -1.2451281e-01,
            -1.08963266e-01,
            1.266032e-01,
            -5.682468e-02,
            -1.5772518e-02,
            5.9984937e-02,
            -9.7492814e-02,
            -1.881005e-02,
            -1.298544e-02,
            4.8306774e-02,
            -3.9294362e-02,
            1.11465916e-01,
            -6.14005e-02,
            2.9382696e-03,
            1.23105004e-01,
            -7.494916e-02,
            -1.2779301e-01,
            -3.774693e-03,
            1.1261373e-01,
            6.9939345e-02,
            -5.92939e-02,
            2.819634e-02,
            -2.2472793e-02,
            -9.1433026e-02,
            -2.7087398e-02,
            1.10135965e-01,
            -4.658184e-02,
            -3.1067498e-02,
            -1.1532691e-01,
            9.00229e-02,
            7.81593e-02,
            9.215586e-02,
            5.147094e-02,
            9.5302396e-02,
            7.935043e-02,
            -4.6664767e-02,
            1.2923521e-01,
            5.9903096e-02,
            2.8747104e-02,
            1.19002e-01,
            7.27312e-02,
            7.046177e-02,
            6.2017303e-02,
            -6.2461145e-02,
            7.501911e-02,
            1.18191116e-01,
            1.2748125e-01,
            -3.3535756e-02,
            -1.2182271e-01,
            -9.71832e-02,
            1.2932208e-01,
            6.121577e-03,
            -4.5696974e-02,
            5.7847984e-02,
            6.847994e-02,
            -1.21741384e-01,
            -8.5186325e-02,
            8.024707e-02,
            6.9201053e-03,
            1.2239638e-01,
            5.190667e-02,
            6.7503387e-03,
            7.2222754e-02,
            -1.3596551e-01,
            2.6657345e-02,
            1.2653731e-01,
            1.05473064e-01,
            1.3198398e-01,
            4.552847e-02,
            -7.9825364e-02,
            1.3584034e-01,
            -1.2384417e-01,
            -1.342367e-01,
            -2.5847899e-02,
            7.179293e-02,
            2.432115e-02,
            1.24735296e-01,
            -9.0016074e-02,
            8.369308e-02,
            -5.7288837e-03,
            6.4526685e-02,
            7.589293e-04,
            -7.042821e-02,
            -5.6265708e-02,
            1.1138938e-01,
            -9.889053e-02,
            -5.090447e-02,
            -1.06834896e-01,
            -1.1579319e-01,
        ],
    ),
    ("5.bn1.scale", &[1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00]),
    ("5.bn1.bias", &[0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "5.conv2.weight",
        &[
            -1.22649804e-01,
            1.161751e-01,
            5.085814e-02,
            1.0568355e-01,
            6.042573e-02,
            6.810994e-03,
            -6.5531984e-02,
            -2.0100975e-02,
            -1.7069152e-02,
            2.2646762e-02,
            -7.622793e-02,
            -6.643419e-02,
            1.1441656e-01,
            1.15377575e-01,
            -1.125228e-01,
            9.393529e-02,
            1.06119625e-01,
            -1.2658513e-01,
            4.5703348e-02,
            6.7274005e-04,
            5.4429777e-02,
            -1.2494212e-01,
            1.3219646e-01,
            -7.0097044e-02,
            1.3129988e-01,
            -1.4761072e-02,
            8.8799655e-02,
            -9.8042585e-02,
            3.2338258e-02,
            -2.7080098e-02,
            1.0086738e-01,
            -7.937731e-02,
            -2.3801954e-02,
            -1.3094299e-01,
            1.2400894e-01,
            1.0276215e-01,
            1.9728786e-03,
            -4.5248393e-02,
            6.414094e-02,
            1.0964001e-01,
            -1.3515276e-01,
            1.0011001e-01,
            -8.434075e-02,
            4.38499e-02,
            5.044032e-02,
            -1.231356e-03,
            -1.4941999e-02,
            -8.15343e-02,
            1.328849e-01,
            6.0194515e-02,
            -5.4428693e-02,
            -6.266131e-02,
            2.7038699e-02,
            -5.2252123e-03,
            1.0916168e-01,
            -4.8418464e-03,
            -9.4469115e-03,
            4.6046775e-02,
            -1.1828136e-01,
            -3.927572e-02,
            5.891608e-02,
            1.296655e-01,
            1.3546202e-01,
            -5.0940257e-02,
            6.62222e-02,
            -7.2799996e-02,
            -7.433704e-02,
            -9.418735e-02,
            6.424529e-04,
            -8.727589e-02,
            -5.1863115e-02,
            -2.3929803e-02,
            1.0720319e-01,
            5.217596e-02,
            -1.0833733e-01,
            5.2486897e-02,
            -4.921876e-02,
            -4.721741e-02,
            1.2385701e-02,
            1.2256644e-01,
            -9.924286e-03,
            2.4916591e-02,
            -1.0797002e-01,
            7.6063536e-02,
            8.998254e-02,
            -2.3662394e-02,
            6.770018e-02,
            -9.745138e-02,
            5.5633053e-02,
            -6.7488626e-02,
            5.679642e-02,
            1.08037345e-01,
            -4.4420633e-02,
            -8.963254e-02,
            -9.90175e-02,
            2.0936718e-02,
            7.226109e-02,
            2.2432401e-03,
            -7.1264304e-02,
            6.380509e-02,
            -2.8877532e-02,
            -7.679346e-02,
            -1.1940936e-01,
            -1.01904884e-01,
            1.12102024e-01,
            1.03474185e-01,
            6.0452238e-02,
            -4.247212e-02,
            -7.545533e-02,
            -7.465724e-02,
            7.198938e-02,
            1.10781e-02,
            1.0251743e-01,
            1.071276e-01,
            8.161857e-02,
            5.0652117e-02,
            -2.3119498e-02,
            -6.8352334e-02,
            3.077175e-02,
            -6.6807956e-02,
            -2.8350923e-02,
            -5.888967e-02,
            6.952323e-03,
            6.571089e-02,
            -5.0764684e-02,
            -8.182738e-03,
            -1.1838925e-01,
            3.272941e-02,
            -1.4392484e-02,
            2.3733577e-02,
            -5.919129e-02,
            1.2996568e-01,
            1.2375357e-01,
            1.3394019e-01,
            3.1593733e-02,
            -1.1913765e-01,
            7.638091e-02,
            4.657668e-02,
            -2.9428604e-03,
            4.6224654e-02,
            4.836315e-03,
            -8.818775e-02,
            2.5248483e-02,
            -6.926598e-02,
            1.0977578e-01,
            -1.2735956e-02,
            -8.234844e-02,
            2.9785367e-02,
            1.1455502e-01,
            -8.170451e-02,
            -1.0636458e-01,
            3.401456e-02,
            2.5212632e-02,
            8.420453e-02,
            5.124686e-02,
            -4.7720708e-02,
            4.685246e-02,
            -7.8403234e-02,
            3.5592117e-03,
            -4.3718465e-02,
            1.0962585e-01,
            -4.261421e-02,
            -5.1258117e-02,
            -1.1434397e-01,
            8.7982295e-03,
            6.580058e-02,
            -1.0965668e-01,
            5.9248652e-02,
            -5.7265718e-02,
            -1.0229336e-01,
            -5.1859498e-02,
            -8.4277794e-02,
            1.0940075e-01,
            -3.025458e-02,
            -2.838275e-02,
            -1.1418739e-01,
            3.486748e-02,
            1.3022217e-01,
            -6.25238e-02,
            1.1477105e-01,
            -5.037048e-02,
            2.7926838e-02,
            -1.10042796e-01,
            9.696497e-02,
            -3.642726e-02,
            -7.466611e-02,
            7.456794e-02,
            2.6506638e-02,
            4.067829e-02,
            2.5489287e-02,
            8.1450696e-04,
            -4.255359e-02,
            -2.5855605e-02,
            -5.783082e-02,
            7.374235e-02,
            6.0542207e-02,
            -2.551985e-02,
            4.9545057e-02,
            1.03180364e-01,
            -1.3310944e-01,
            -1.4976341e-02,
            4.737814e-02,
            2.6917955e-02,
            1.0868314e-01,
            3.1051794e-02,
            7.3716685e-02,
            -1.0570911e-01,
            -8.030206e-02,
            1.2126292e-01,
            1.3017583e-01,
            -2.1026865e-02,
            -7.721463e-02,
            -4.9468374e-04,
            -1.186631e-03,
            -8.130744e-02,
            2.1781122e-02,
            -7.860132e-02,
            -2.2718709e-02,
            1.9317551e-04,
            3.6287893e-02,
            4.2047825e-02,
            5.7360376e-03,
            -1.2460732e-01,
            -1.1410294e-01,
            -1.1760338e-01,
            -3.0715814e-02,
            -1.093703e-01,
            -6.0969908e-02,
            -3.9698377e-02,
            -7.677221e-02,
            -9.625068e-02,
            1.3049595e-01,
            2.1367095e-02,
            7.688932e-02,
            -1.2601694e-01,
            5.6768276e-02,
            1.2586822e-01,
            -5.2478477e-02,
            1.1210654e-01,
            -7.888025e-02,
            -2.5212144e-02,
            -1.2679198e-01,
            1.0176524e-01,
            -1.2153139e-01,
            9.175833e-02,
            -3.5768777e-02,
            2.2241788e-03,
            -7.759979e-02,
            7.541748e-02,
            -6.4785756e-02,
            1.1653619e-01,
            -1.1373753e-01,
            -8.769695e-03,
            1.0202537e-01,
            9.427252e-02,
            -1.2629636e-01,
            7.36649e-02,
            -1.3082224e-01,
            -6.6846155e-02,
            -6.978988e-02,
            5.238682e-02,
            1.12138756e-01,
            -1.12588e-02,
            7.6063715e-02,
            9.44501e-02,
            -5.663165e-02,
            7.6585844e-02,
            -6.108882e-02,
            3.5530746e-02,
            9.245073e-03,
            7.663907e-02,
            -3.820847e-02,
            9.013057e-02,
            -9.574419e-02,
            -5.686282e-02,
            1.0848174e-01,
            -3.352768e-02,
            5.816982e-02,
            -3.1789977e-02,
            -1.4710377e-02,
            1.1551016e-01,
            1.08285256e-01,
            -5.3767096e-02,
            -8.926616e-02,
            -1.729912e-02,
            -9.169372e-02,
            6.813729e-02,
            7.140987e-02,
            8.555389e-02,
            1.3644391e-02,
            9.703333e-02,
            -7.108603e-02,
            -9.96017e-02,
            5.4459546e-02,
            1.1839532e-01,
            5.5576567e-02,
            -4.900766e-05,
            6.937637e-02,
            1.17664844e-01,
            -3.5315834e-02,
            -1.2460263e-01,
            -9.878686e-03,
            -8.854722e-02,
            -1.0217034e-01,
            -2.0193558e-03,
            7.2275154e-02,
            -2.9401936e-02,
            2.4606306e-02,
            -1.0551842e-01,
            6.417558e-02,
            -6.049177e-03,
            6.5437816e-02,
            -1.2628776e-02,
            -1.0381041e-01,
            7.610774e-02,
            9.9236794e-02,
            -3.7243843e-02,
            4.7108866e-02,
            -4.927025e-02,
            6.1522942e-02,
            1.1272e-01,
            -3.0523937e-04,
            -3.590412e-02,
            -3.928286e-03,
        ],
    ),
    ("5.bn2.scale", &[1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00]),
    ("5.bn2.bias", &[0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "8.weight",
        &[
            6.0979933e-02,
            -2.4323826e-01,
            2.0474888e-01,
            1.3881163e-01,
            1.8113367e-03,
            -1.3335451e-01,
            2.8866193e-01,
            -3.1922323e-01,
            -2.962762e-01,
            -1.9361305e-01,
            1.0228828e-01,
            -1.0950438e-01,
            9.47702e-02,
            3.2223496e-01,
            2.2255604e-01,
            3.701665e-02,
            -3.0012184e-01,
            -2.8749606e-01,
            3.227203e-01,
            4.9462695e-02,
            1.9808263e-01,
            2.0845717e-01,
            8.04152e-02,
            -1.442346e-01,
        ],
    ),
    (
        "8.bias",
        &[-4.0471196e-01, -1.0616918e-01, 1.5629418e-01, 3.5153058e-01],
    ),
];
const RESNET_TINY_LOGITS: [f64; 8] = [
    -4.04659033e-01,
    -1.09068282e-01,
    1.62613243e-01,
    3.44547927e-01,
    -4.04548883e-01,
    -1.15099147e-01,
    1.75758511e-01,
    3.30022186e-01,
];
const RESNET_TINY_CONV1_GRAD: [f64; 588] = [
    -3.94964794e-04,
    1.18290547e-04,
    -1.54718189e-04,
    -3.02553410e-03,
    -1.97761855e-03,
    1.04561346e-04,
    4.36916511e-04,
    2.98272190e-03,
    5.34319505e-03,
    -7.93658430e-04,
    6.89070075e-05,
    -1.79743045e-03,
    -1.80768617e-03,
    7.19250878e-04,
    -3.73737101e-04,
    3.89157166e-03,
    2.12613260e-03,
    -5.72568737e-04,
    1.78457834e-04,
    2.13227840e-03,
    -4.11727512e-03,
    9.02361120e-04,
    -4.17779054e-04,
    -4.34567546e-03,
    -3.48855683e-04,
    2.91906937e-04,
    1.78364600e-04,
    4.00891993e-04,
    2.80629308e-03,
    2.55980121e-04,
    -1.77971297e-03,
    -3.55243147e-03,
    9.39419866e-03,
    -3.32753116e-04,
    -1.23817492e-02,
    -5.74441217e-02,
    2.94997450e-03,
    1.11506251e-03,
    1.68300252e-02,
    -3.24605741e-02,
    -5.96343598e-04,
    4.42123413e-02,
    4.44055349e-03,
    -4.39911000e-02,
    6.40406739e-04,
    -6.05977736e-02,
    -2.16676481e-02,
    -3.70908715e-02,
    -8.45576543e-03,
    -1.32441923e-01,
    -3.10905557e-03,
    5.57349436e-02,
    3.93360341e-03,
    1.48404259e-02,
    -7.33149722e-02,
    -3.49401608e-02,
    3.93031631e-04,
    3.86773434e-04,
    -1.14758928e-04,
    -4.23283596e-03,
    -5.44659747e-03,
    7.84906326e-04,
    -1.15620857e-02,
    -3.20343450e-02,
    4.37861541e-03,
    4.44014615e-04,
    2.11226214e-02,
    1.44541739e-02,
    -2.96997582e-03,
    3.89936306e-02,
    -3.45796421e-02,
    -5.20195737e-02,
    -2.19025463e-03,
    -6.66393787e-02,
    1.85487568e-02,
    2.74424516e-02,
    4.36639693e-03,
    6.78307116e-02,
    -4.89780307e-02,
    -4.74659353e-02,
    -5.65509894e-04,
    -3.78547981e-02,
    -9.28618945e-03,
    7.62467012e-02,
    1.40237540e-03,
    3.47914145e-04,
    5.28174220e-04,
    1.46209693e-03,
    3.29292356e-03,
    1.11299254e-01,
    1.99184306e-02,
    -2.48307511e-02,
    1.73773919e-03,
    5.42078465e-02,
    2.24071257e-02,
    -3.85392234e-02,
    6.73809182e-03,
    -2.23132297e-02,
    -2.60451324e-02,
    5.91335371e-02,
    -1.14735151e-02,
    4.02191579e-02,
    4.69469093e-03,
    4.25541662e-02,
    -1.65411066e-02,
    4.59311716e-02,
    1.98163688e-02,
    1.27908867e-02,
    9.97839123e-03,
    -5.50513081e-02,
    4.45604622e-02,
    5.72131351e-02,
    3.88247566e-03,
    2.57543172e-04,
    -1.18474313e-03,
    -1.16947002e-03,
    6.54979702e-03,
    5.18260710e-02,
    -2.10003052e-02,
    3.80168855e-02,
    -2.33971421e-03,
    -2.10165419e-02,
    2.66418308e-02,
    -6.96309842e-04,
    1.34948613e-02,
    1.27518788e-01,
    4.84489948e-02,
    -1.26469731e-01,
    -1.84923559e-02,
    -7.46159032e-02,
    -1.49526829e-02,
    -1.47212669e-03,
    2.21551694e-02,
    -7.28059411e-02,
    -4.30312455e-02,
    5.14263436e-02,
    -8.96690506e-03,
    -8.44141990e-02,
    -4.36757281e-02,
    -4.67789844e-02,
    -2.74254754e-03,
    7.16532406e-04,
    -7.21319520e-04,
    -3.17959976e-03,
    1.50223356e-03,
    5.02096638e-02,
    3.56009379e-02,
    6.90308958e-02,
    -6.35499426e-04,
    8.46726969e-02,
    -9.08415299e-03,
    -1.36691169e-03,
    -1.52988685e-03,
    -1.73016369e-01,
    3.45302783e-02,
    6.84432387e-02,
    -9.59428865e-03,
    1.56613532e-02,
    -3.23990732e-02,
    -4.46736738e-02,
    5.73506719e-03,
    -4.34818342e-02,
    3.05117723e-02,
    2.06245668e-03,
    -1.76103469e-02,
    8.92928541e-02,
    1.02117751e-02,
    -5.08006290e-02,
    -1.11907953e-04,
    6.57532248e-04,
    -3.66624794e-04,
    1.96362101e-03,
    4.88196127e-03,
    -8.19323137e-02,
    7.13482383e-04,
    -2.39294837e-03,
    -8.19364563e-03,
    1.41004520e-02,
    6.59497827e-03,
    8.19471199e-03,
    1.17076794e-02,
    1.71203166e-01,
    1.94827262e-02,
    -6.90936437e-03,
    -2.37752125e-02,
    8.02071765e-02,
    -2.42241845e-02,
    1.37333274e-02,
    -2.04521455e-02,
    -7.50776604e-02,
    3.36098298e-02,
    1.26552194e-01,
    5.07988920e-03,
    1.48650035e-01,
    -5.05907200e-02,
    -9.41370651e-02,
    -1.30566594e-03,
    3.91042326e-04,
    -1.47731858e-04,
    -5.33533329e-03,
    -4.44343919e-03,
    2.16875764e-04,
    2.51191319e-04,
    3.77205922e-03,
    3.86958662e-03,
    2.63476133e-04,
    -3.68912937e-04,
    -2.59493128e-03,
    -2.25339667e-03,
    6.31113886e-04,
    -3.70337017e-04,
    -3.61865154e-04,
    8.94152210e-04,
    6.77156146e-04,
    1.31082826e-03,
    -4.97675035e-03,
    -1.08611980e-03,
    1.15649973e-03,
    -5.41276881e-04,
    -5.60425979e-04,
    6.94524823e-03,
    9.64979525e-04,
    4.58472990e-04,
    8.81665666e-03,
    3.81542346e-03,
    -2.09398044e-04,
    -9.14409757e-04,
    3.77100357e-03,
    -5.09645650e-03,
    8.91107484e-05,
    -2.69185062e-02,
    -4.93515097e-03,
    -1.91537559e-03,
    2.73211976e-04,
    1.90408602e-02,
    -3.66212763e-02,
    1.00849727e-02,
    9.36869085e-02,
    4.24111634e-03,
    -7.01311305e-02,
    1.22736359e-03,
    -6.57309219e-02,
    -6.44997787e-03,
    -6.36675581e-03,
    -1.45533215e-03,
    -1.22684520e-02,
    3.91593501e-02,
    -1.07136928e-03,
    -5.76528022e-03,
    -5.64625859e-02,
    1.39096752e-04,
    2.46403310e-02,
    5.84974978e-03,
    -5.32366103e-04,
    -6.29476912e-04,
    6.46814797e-03,
    -2.45997147e-03,
    1.06790441e-03,
    -2.68144310e-02,
    -2.72458885e-02,
    -4.21902491e-03,
    -1.75923575e-04,
    3.28139737e-02,
    -3.13669862e-03,
    -2.49759038e-03,
    9.14115831e-02,
    -3.80385183e-02,
    -8.44939575e-02,
    7.79219903e-03,
    -1.16139024e-01,
    1.07955839e-02,
    1.97975971e-02,
    4.82309260e-05,
    8.36528540e-02,
    -3.40143628e-02,
    -6.16285875e-02,
    -4.38726461e-03,
    -5.72974943e-02,
    2.69858371e-02,
    1.05557591e-02,
    -5.44421654e-03,
    5.58546628e-04,
    5.66857343e-04,
    4.58421092e-03,
    3.31992097e-03,
    1.52995787e-03,
    -2.49260943e-02,
    -2.15992797e-02,
    -7.49510061e-03,
    7.50672370e-02,
    -2.40516700e-02,
    2.19863374e-03,
    1.57771911e-02,
    1.25842512e-01,
    5.47461119e-03,
    -5.96245565e-02,
    -1.13828257e-02,
    1.54365331e-01,
    -9.01419390e-03,
    -1.09760743e-02,
    -4.33866959e-03,
    -4.16835994e-02,
    -1.79350302e-02,
    -2.35040113e-02,
    -9.96678136e-03,
    3.83552164e-02,
    1.48194581e-02,
    -3.63752916e-02,
    -6.06514420e-03,
    4.83085925e-04,
    5.25918906e-04,
    2.05474193e-04,
    -1.99862430e-03,
    5.28358109e-02,
    1.65980719e-02,
    6.70601148e-03,
    -3.66798509e-03,
    1.31997708e-02,
    -4.16909605e-02,
    -2.35030800e-02,
    1.82690490e-02,
    1.06826536e-02,
    -6.96535222e-04,
    -9.94264893e-03,
    -1.94212254e-02,
    2.02897251e-01,
    -2.28827316e-02,
    -4.71762493e-02,
    1.92571208e-02,
    5.90878054e-02,
    -1.47922728e-02,
    -4.79189083e-02,
    -8.91576335e-03,
    8.93232971e-02,
    4.04411331e-02,
    -4.19815704e-02,
    -2.68230541e-03,
    -8.73130746e-04,
    -1.37041672e-03,
    1.06422335e-03,
    3.84048931e-03,
    3.72131281e-02,
    -3.71592212e-03,
    2.46733911e-02,
    3.50152550e-04,
    -6.74394192e-03,
    -3.00254263e-02,
    3.62662449e-02,
    1.06475353e-02,
    6.40355796e-02,
    1.93066336e-02,
    2.71605663e-02,
    1.88946128e-02,
    4.79462519e-02,
    2.88719740e-02,
    -8.82777572e-02,
    -1.58389695e-02,
    8.40303227e-02,
    -1.58192366e-02,
    6.16515428e-02,
    7.32862670e-03,
    2.68332772e-02,
    -4.13899794e-02,
    -2.28670090e-02,
    2.41504190e-03,
    -1.31308986e-03,
    -1.57467279e-04,
    -2.70290952e-03,
    3.08131566e-04,
    -1.79436766e-02,
    1.87936462e-02,
    5.32255545e-02,
    7.56777544e-03,
    3.55152600e-02,
    -1.59581483e-03,
    2.26173662e-02,
    -1.24238953e-02,
    9.65870917e-04,
    5.27423844e-02,
    3.18658613e-02,
    3.17713991e-02,
    -4.91204895e-02,
    2.53788903e-02,
    1.43933482e-03,
    6.06888579e-03,
    -9.87807661e-03,
    1.44931693e-02,
    -3.35464254e-02,
    3.95875564e-03,
    -1.32883899e-02,
    2.60315184e-02,
    5.16906567e-03,
    1.03773852e-03,
    -3.10798991e-04,
    4.05033381e-04,
    -2.33882037e-03,
    5.93497232e-03,
    -1.07972231e-03,
    9.08979637e-05,
    3.44228675e-03,
    3.14201741e-03,
    -3.17491882e-04,
    -2.46838288e-04,
    2.71525793e-03,
    7.32395099e-04,
    1.06430438e-03,
    6.82435057e-05,
    2.31756037e-03,
    -3.95112578e-03,
    3.01676895e-03,
    -2.05479591e-05,
    -2.93176435e-03,
    -2.50005163e-03,
    -6.30449620e-04,
    -6.57934113e-04,
    -1.02587824e-03,
    1.80950272e-03,
    -7.66962185e-04,
    -1.37535098e-04,
    -4.41466644e-03,
    -1.58966531e-03,
    4.83541633e-04,
    1.32662943e-03,
    1.78719114e-03,
    -9.77748539e-04,
    -8.27689306e-04,
    1.32835228e-02,
    2.99834050e-02,
    -3.25525785e-03,
    -4.15120594e-04,
    -2.05250420e-02,
    -6.91928715e-02,
    4.27286653e-03,
    -4.65936624e-02,
    1.62388720e-02,
    3.32054198e-02,
    -3.12596676e-04,
    6.73926994e-02,
    -4.55759466e-02,
    1.16234422e-02,
    -9.56377073e-04,
    -4.60364204e-03,
    3.71905863e-02,
    -8.97129811e-03,
    -2.44254642e-03,
    2.44644098e-02,
    3.61332856e-02,
    -7.10491464e-02,
    4.14095679e-03,
    3.55171622e-04,
    3.71092581e-04,
    -1.57068064e-03,
    7.93506624e-04,
    -7.89295475e-04,
    1.87754035e-02,
    -6.53110743e-02,
    1.92598428e-03,
    -1.37543830e-03,
    1.51796741e-02,
    -7.53685385e-02,
    2.90094293e-04,
    -5.84409982e-02,
    -5.43823205e-02,
    4.84058671e-02,
    9.99677926e-03,
    -5.31641692e-02,
    -5.85049018e-02,
    4.87803817e-02,
    3.56286619e-04,
    7.44120702e-02,
    -2.86032632e-03,
    5.33969998e-02,
    4.01267549e-03,
    5.85951582e-02,
    3.04174796e-02,
    6.64691254e-03,
    -3.40177398e-03,
    5.69508993e-04,
    1.06676074e-03,
    -2.90517067e-03,
    -1.73193775e-03,
    -5.26957698e-02,
    -1.22102592e-02,
    -5.51094301e-02,
    1.85445556e-03,
    1.30279288e-01,
    -1.12841008e-02,
    -3.23152654e-02,
    -4.23419755e-03,
    1.78644527e-02,
    -4.08146381e-02,
    3.84319797e-02,
    7.85522815e-03,
    -4.82593551e-02,
    -2.20224671e-02,
    1.12740826e-02,
    -1.41806703e-03,
    6.48593307e-02,
    4.78570163e-02,
    -6.97378255e-03,
    -6.79033529e-03,
    1.30064040e-01,
    1.60532594e-02,
    -1.90893169e-02,
    -2.70392653e-03,
    3.87724314e-04,
    9.55176598e-04,
    -1.72824861e-04,
    6.11335086e-03,
    1.18021064e-01,
    9.90408659e-03,
    2.61979662e-02,
    2.87171453e-03,
    1.36846393e-01,
    1.21052926e-02,
    1.66868814e-03,
    -1.42036267e-02,
    -6.73024729e-02,
    3.81049290e-02,
    3.56416851e-02,
    -6.16708491e-03,
    -8.47080499e-02,
    3.18238363e-02,
    7.13947192e-02,
    5.22161555e-03,
    -2.64190108e-01,
    9.22422111e-03,
    1.11526951e-01,
    4.90580313e-03,
    -1.42115414e-01,
    4.02938835e-02,
    1.14519484e-01,
    4.45584115e-03,
    -8.26459727e-05,
    -2.57176434e-04,
    -3.24135786e-03,
    -3.85011634e-04,
    1.00881740e-01,
    4.49242210e-03,
    1.62784476e-02,
    -1.77309732e-03,
    7.25831613e-02,
    1.70425195e-02,
    -2.54172320e-03,
    5.65802353e-03,
    -1.29090682e-01,
    2.26769112e-02,
    -1.98643003e-02,
    -2.68922979e-03,
    -1.11568451e-01,
    -4.09794860e-02,
    1.30828079e-02,
    9.73283593e-03,
    -8.15967917e-02,
    -3.46129537e-02,
    -5.57779707e-03,
    1.84662407e-04,
    1.60954893e-01,
    -3.99441123e-02,
    -1.02109134e-01,
    2.34564138e-03,
    -1.50187372e-03,
    -1.66301162e-03,
    1.05221360e-03,
    -9.02913511e-04,
    -3.75382975e-02,
    1.76384393e-02,
    1.34562198e-02,
    4.25921706e-03,
    -7.55865406e-03,
    -4.55552945e-05,
    6.03839569e-02,
    -1.42605091e-02,
    -8.46643746e-02,
    -2.98264008e-02,
    7.65589997e-02,
    -6.00983109e-03,
    -1.01205803e-01,
    5.05759418e-02,
    -6.88339472e-02,
    -1.55331865e-02,
    8.01190734e-04,
    -3.23928446e-02,
    -7.90931806e-02,
    -1.24281608e-02,
    -2.02145875e-01,
    -4.38614450e-02,
    -8.13818164e-03,
];

#[test]
#[ignore = "requires CUDA"]
fn resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new(
            "resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_channel",
        ),
        Axis::new(
            "resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_height",
        ),
        Axis::new(
            "resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_width",
        ),
        Axis::new(
            "resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_batch",
        ),
    );
    let axes = ResNetAxes {
        channel,
        height,
        width,
    };
    let input = Tensor::from_slice(
        &RESNET_TINY_INPUT,
        [batch.of(2), channel.of(3), height.of(8), width.of(8)],
        &device,
    )?;
    let mut model = Sequential::new(vec![
        Box::new(
            Conv2d::new(channel, channel.of(4), [height, width], [7, 7])
                .stride([2, 2])
                .padding([3, 3])
                .bias(false),
        ) as Box<dyn Module>,
        Box::new(BatchNorm::new(channel)),
        Box::new(ReLU),
        Box::new(
            MaxPool2d::new(channel, [height, width], [3, 3])
                .stride([2, 2])
                .padding([1, 1]),
        ),
        Box::new(BasicBlock::new(axes, 4, 6, 2)),
        Box::new(BasicBlock::new(axes, 6, 6, 1)),
        Box::new(AdaptiveAvgPool2d::new([height, width], [1, 1])),
        Box::new(Flatten::new([channel, height, width], channel)),
        Box::new(Linear::new(channel, channel.of(4))),
    ]);
    model.build(input.shape(), &device, 0)?;
    for (path, values) in RESNET_TINY_PARAMS {
        model.parameter(path)?.set_values(values)?;
    }
    let logits = model.forward(&input)?;
    close(
        "resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient logits",
        &logits.to_vec()?,
        &RESNET_TINY_LOGITS,
    );
    logits.sum([batch, channel])?.backward()?;
    close(
        "resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient first conv weight gradient",
        &model.parameter("0.weight")?.grad().unwrap().to_vec()?,
        &RESNET_TINY_CONV1_GRAD,
    );
    println!(
        "resnet34_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient: tiny-width equivalence PASS (6-wide head, BasicBlock)"
    );
    Ok(())
}
const RESNET50_TINY_INPUT: [f32; 384] = [
    -1.5255959e+00,
    -7.502318e-01,
    -6.539809e-01,
    -1.6094848e+00,
    -1.0016718e-01,
    -6.091889e-01,
    -9.7977227e-01,
    -1.6090963e+00,
    -7.121446e-01,
    3.03722e-01,
    -7.773143e-01,
    -2.5145525e-01,
    -2.2227049e-01,
    1.6871134e+00,
    2.2842517e-01,
    4.676355e-01,
    -6.969724e-01,
    -1.1607615e+00,
    6.995424e-01,
    1.9908163e-01,
    8.656924e-01,
    2.444039e-01,
    -6.6291136e-01,
    8.0730826e-01,
    1.1016806e+00,
    -1.7593604e-01,
    -2.2455578e+00,
    -1.446458e+00,
    6.1155282e-02,
    -6.1774445e-01,
    -7.9806983e-01,
    -1.3162321e-01,
    1.8793458e+00,
    -7.213178e-02,
    1.577706e-01,
    -7.734549e-01,
    1.990565e-01,
    4.5702778e-02,
    1.5295692e-01,
    -4.756788e-01,
    -1.1101983e-01,
    2.9273525e-01,
    -1.5784515e-01,
    -2.878714e-02,
    2.357111e+00,
    -1.0373387e+00,
    1.5747981e+00,
    -6.298472e-01,
    -9.273917e-01,
    5.451415e-01,
    6.628026e-02,
    -4.3704012e-01,
    7.626006e-01,
    4.4151092e-01,
    1.1651385e+00,
    2.0153918e+00,
    1.3741246e-01,
    9.386447e-01,
    -1.8600109e-01,
    -6.446393e-01,
    1.5392458e+00,
    -8.695876e-01,
    -3.3311536e+00,
    -7.4787223e-01,
    -2.5502462e-02,
    -1.0233306e+00,
    -5.9618515e-01,
    -1.0055307e+00,
    -2.106063e-01,
    -7.5475276e-03,
    1.6734272e+00,
    1.0342831e-02,
    -7.0395666e-01,
    -1.852658e-01,
    -9.962351e-01,
    -8.3125526e-01,
    -4.6102202e-01,
    -5.6008244e-01,
    3.9557618e-01,
    -9.822771e-01,
    -5.064866e-01,
    9.9775404e-02,
    -6.539734e-01,
    7.316937e-01,
    -1.4343859e+00,
    -5.0081307e-01,
    1.7163314e-01,
    -1.5999313e-01,
    2.546335e-01,
    -5.019573e-01,
    -1.0412e+00,
    7.322672e-01,
    -1.0483401e+00,
    -4.708772e-01,
    2.9113635e-01,
    1.9907043e+00,
    6.614453e-01,
    1.1899205e+00,
    8.165339e-01,
    -9.135236e-01,
    1.3851457e+00,
    -8.1384623e-01,
    -9.2757654e-01,
    1.1119633e+00,
    1.3352057e+00,
    6.042736e-01,
    -1.0344208e-01,
    -1.5121692e-01,
    -2.102083e+00,
    -6.2002194e-01,
    -1.478231e+00,
    -1.1334175e+00,
    8.737963e-01,
    -5.602594e-01,
    1.2857845e+00,
    8.1682384e-01,
    2.053041e-01,
    3.0510718e-01,
    5.35687e-01,
    -4.31185e-01,
    2.5581384e+00,
    -2.3336388e-01,
    -1.347213e-02,
    1.8606348e+00,
    -1.9804063e+00,
    1.7985829e+00,
    1.0181159e-01,
    3.40006e-01,
    7.1236455e-01,
    -1.7765073e+00,
    3.5386458e-01,
    1.1996132e+00,
    -3.029974e-01,
    -1.7618417e+00,
    6.348446e-01,
    -8.043591e-01,
    -1.6111118e+00,
    -1.8716129e+00,
    5.430836e-01,
    6.606786e-01,
    2.2952116e+00,
    6.7490596e-01,
    1.7133216e+00,
    -1.7942734e+00,
    -1.3632672e+00,
    -9.832196e-01,
    1.5112667e+00,
    6.4187074e-01,
    4.729638e-01,
    -4.285901e-01,
    5.5137074e-01,
    -1.5473709e+00,
    5.181121e-01,
    1.0653535e-01,
    2.6924077e-01,
    1.324768e+00,
    1.7460191e+00,
    1.854969e+00,
    -7.063691e-01,
    2.5570862e+00,
    4.175343e-01,
    -2.127186e-01,
    -8.39958e-01,
    -4.2001787e-01,
    -6.240363e-01,
    -9.772961e-01,
    8.748428e-01,
    9.872814e-01,
    3.0957633e-01,
    1.52069e+00,
    1.2052339e+00,
    -1.815591e+00,
    -4.0346155e-01,
    -9.591452e-01,
    -5.207704e-03,
    -7.886314e-02,
    8.436543e-01,
    1.1657013e+00,
    5.269322e-01,
    1.6192533e+00,
    -9.6397626e-01,
    1.4152038e-01,
    -1.6366096e-01,
    -3.5822257e-01,
    1.7222793e+00,
    -3.035756e-01,
    2.388742e-01,
    1.3440012e+00,
    1.0322569e-01,
    1.1003542e+00,
    -3.416802e-01,
    9.473389e-01,
    -5.6851596e-01,
    8.375962e-01,
    1.7836607e+00,
    -1.9542466e-01,
    5.149161e-01,
    -1.8474776e+00,
    -2.9167426e+00,
    -5.673299e-01,
    -5.412802e-01,
    8.95174e-01,
    -8.82507e-01,
    5.3181124e-01,
    -1.5457772e+00,
    -1.7329982e-01,
    7.2824633e-01,
    5.706102e-02,
    9.055172e-01,
    1.0462948e+00,
    -5.20597e-01,
    1.3547838e+00,
    2.3519313e-01,
    1.9142433e+00,
    1.8364111e+00,
    1.3245324e+00,
    -9.6900916e-01,
    1.2516364e+00,
    1.2103242e+00,
    -5.279206e-01,
    2.1856615e-01,
    -5.7430726e-01,
    1.4571251e+00,
    1.7709557e+00,
    1.6499138e+00,
    -4.3200457e-01,
    -2.710269e-01,
    -1.4391626e+00,
    1.2470404e+00,
    1.2738512e+00,
    3.9094925e-01,
    3.872105e-01,
    -7.982872e-02,
    3.417243e-01,
    9.488273e-01,
    -1.3839359e+00,
    1.7240863e+00,
    -2.3647652e+00,
    -9.294909e-01,
    2.936253e-01,
    2.1513203e-01,
    9.384637e-01,
    1.4657077e+00,
    -5.564744e-01,
    -7.448408e-01,
    -2.0215721e-01,
    -2.296679e-01,
    1.3313366e-03,
    3.752759e-01,
    -5.810679e-01,
    -5.7230884e-01,
    1.0097175e+00,
    -1.0564939e-01,
    -1.179696e+00,
    -9.0779595e-02,
    5.6311435e-01,
    -1.2560141e+00,
    8.95555e-01,
    1.6747737e-01,
    7.5142086e-01,
    2.4142299e+00,
    1.020584e+00,
    -4.404838e-01,
    -1.7341677e+00,
    -1.236225e+00,
    1.5785813e+00,
    -1.1160507e+00,
    7.6777023e-01,
    -5.8820677e-01,
    2.1188903e+00,
    -5.4219025e-01,
    -2.4592547e+00,
    -1.1108288e+00,
    -1.1187209e+00,
    7.5799555e-01,
    -4.9565765e-01,
    -1.9700006e-01,
    -3.3396218e-02,
    7.192915e-01,
    1.0644146e+00,
    8.3402544e-01,
    -1.9162164e+00,
    -3.4202927e-01,
    -6.6049206e-01,
    3.1508535e-01,
    1.1422518e+00,
    3.0550566e-01,
    -5.7888174e-01,
    -2.3828252e-01,
    -1.3541743e+00,
    2.686894e-01,
    1.1455697e-01,
    -1.5562972e+00,
    -1.0757437e+00,
    -8.751946e-01,
    -4.7281876e-01,
    9.912368e-01,
    -5.8622282e-02,
    1.1787646e+00,
    6.22185e-01,
    7.878501e-01,
    1.3685523e+00,
    -8.506898e-01,
    5.1260746e-01,
    1.0476325e+00,
    -3.1758463e-01,
    1.3948506e-01,
    2.3402624e+00,
    -6.1160916e-01,
    8.160271e-01,
    2.47723e-01,
    -3.867267e-01,
    1.9948451e-01,
    7.9926956e-01,
    -2.619034e-01,
    1.5132962e-01,
    1.1981666e+00,
    -2.2832582e+00,
    -1.0129594e+00,
    -8.878909e-01,
    6.5221924e-01,
    -8.726203e-01,
    3.5253752e-02,
    -3.365303e-01,
    1.402332e+00,
    4.841214e-01,
    -7.0304507e-01,
    -8.267661e-01,
    7.74396e-01,
    6.9199395e-01,
    -1.01848e+00,
    -8.0337167e-01,
    -7.071132e-01,
    7.521183e-01,
    -1.9208279e-02,
    1.103333e+00,
    -6.0679215e-01,
    -5.252235e-01,
    -5.6618774e-01,
    6.6039857e-04,
    7.224588e-01,
    1.526352e-01,
    1.4495978e-01,
    -2.3442194e+00,
    3.60003e-01,
    4.6668175e-01,
    1.2830665e+00,
    1.2678007e+00,
    1.9883296e-01,
    5.440877e-01,
    -3.9781693e-01,
    -1.9291055e+00,
    2.323687e-01,
    8.6146563e-01,
    6.217573e-01,
    -1.7811896e+00,
    -7.820609e-01,
    -1.42367e+00,
    1.6090765e+00,
    -3.2787595e-02,
    8.532334e-01,
    5.506365e-02,
    -1.7425371e+00,
    8.7500376e-01,
    -2.7188172e+00,
    -2.2192061e-01,
    3.4208494e-01,
    1.1093477e+00,
    -5.731476e-01,
    9.577846e-01,
    9.820232e-04,
    -1.3847686e+00,
    -9.965023e-01,
    8.0734813e-01,
    1.1738863e+00,
    -9.3984646e-01,
    1.3109189e+00,
    -3.1670693e-01,
    -1.861041e-01,
    -5.764602e-01,
    6.866518e-01,
    4.2086706e-01,
    -1.0213808e+00,
    9.8856664e-01,
];
const RESNET50_TINY_PARAMS: &[(&str, &[f32])] = &[
    (
        "0.weight",
        &[
            -5.329979e-02,
            2.244175e-02,
            5.6216065e-02,
            -3.5644952e-02,
            5.288132e-02,
            3.660904e-02,
            -3.9688766e-02,
            -1.9641967e-02,
            2.9292367e-02,
            2.4510998e-02,
            5.8018327e-02,
            -1.9475901e-02,
            2.0410877e-02,
            -1.25817275e-02,
            6.687414e-02,
            -5.771635e-02,
            -1.6353896e-02,
            -6.384741e-02,
            -5.7680003e-02,
            -2.5168823e-02,
            -2.5739122e-02,
            5.2506573e-02,
            -3.802596e-02,
            2.7754657e-02,
            1.3996495e-02,
            -7.0701505e-04,
            9.928828e-03,
            6.0548995e-02,
            6.3825995e-02,
            -5.4814234e-02,
            -7.7766284e-02,
            6.265098e-03,
            1.1811275e-02,
            7.1128726e-02,
            2.5329815e-02,
            -7.9872414e-02,
            -3.899984e-02,
            4.598577e-02,
            -3.4365702e-02,
            4.3681707e-02,
            8.1933275e-02,
            -6.482237e-02,
            -3.656918e-02,
            -2.4956781e-02,
            1.7639784e-03,
            -4.1191228e-02,
            -6.376147e-02,
            7.365762e-02,
            6.520079e-02,
            -2.8067833e-02,
            3.8585022e-02,
            -7.969743e-02,
            2.0101672e-02,
            -4.7613524e-02,
            -5.2559257e-02,
            -1.929725e-02,
            7.393024e-02,
            3.8516432e-02,
            -6.592531e-02,
            6.396402e-02,
            4.7588e-02,
            -5.8839772e-02,
            -2.168986e-02,
            3.568032e-02,
            -4.2049836e-02,
            7.665306e-02,
            -4.5916e-02,
            -4.1149255e-02,
            5.7937536e-02,
            -3.4091245e-02,
            -8.018553e-02,
            -7.141882e-02,
            5.305956e-04,
            4.8685096e-02,
            3.324007e-02,
            -8.858631e-04,
            -6.6207014e-02,
            2.8087988e-03,
            -3.138029e-02,
            3.18624e-02,
            -1.7283062e-02,
            -3.627336e-02,
            5.7773426e-02,
            -6.84735e-02,
            -7.4446194e-02,
            5.50778e-02,
            5.805811e-02,
            -6.901839e-02,
            -4.2732347e-02,
            -6.293395e-02,
            -5.260244e-02,
            -1.0382723e-03,
            -5.843081e-02,
            -4.3637946e-02,
            -4.8082765e-02,
            -1.6915346e-02,
            5.641138e-02,
            9.88592e-03,
            -2.927636e-02,
            5.3012405e-02,
            -1.7327778e-02,
            6.542819e-02,
            4.8312258e-02,
            -7.9916365e-02,
            -1.4250402e-02,
            -3.53384e-02,
            -2.8391609e-03,
            -1.3118635e-02,
            3.952502e-02,
            -5.022377e-02,
            -2.946505e-02,
            -6.391505e-02,
            3.356637e-02,
            -5.2655652e-02,
            5.109391e-02,
            -1.2012944e-02,
            -5.4433238e-02,
            -3.636293e-02,
            7.821828e-02,
            -4.2140204e-02,
            3.0396229e-02,
            -2.8490981e-02,
            6.0976245e-02,
            2.595772e-02,
            3.5038777e-02,
            -1.921102e-02,
            -6.3071854e-02,
            -6.8997905e-02,
            -6.557952e-02,
            -4.692092e-02,
            -1.713076e-02,
            -6.488272e-02,
            7.170995e-02,
            2.5778294e-02,
            -1.58204e-02,
            -4.6805855e-02,
            -5.0186664e-02,
            6.698493e-04,
            7.0387475e-02,
            -7.223783e-02,
            -3.225333e-02,
            -5.3888906e-02,
            -4.2658627e-02,
            5.206608e-02,
            -6.793645e-02,
            -4.7100615e-02,
            -4.7270987e-02,
            -1.1803586e-02,
            7.538363e-02,
            1.7465016e-02,
            1.2904392e-03,
            -4.5824e-02,
            1.5705373e-02,
            -1.1782339e-03,
            -1.9159677e-02,
            5.779512e-02,
            6.9449864e-02,
            5.8374047e-02,
            3.12196e-02,
            -1.8712478e-02,
            5.3859213e-03,
            -1.2503433e-02,
            1.002543e-02,
            4.265715e-02,
            4.483073e-02,
            -6.681607e-02,
            -4.9244117e-02,
            -2.55675e-02,
            -2.6334226e-02,
            -6.100221e-02,
            5.9974223e-02,
            -6.9005676e-02,
            -6.579943e-02,
            -2.4209533e-02,
            5.1341172e-02,
            1.8114561e-02,
            1.3240092e-02,
            -1.8267117e-02,
            1.8061694e-02,
            5.135871e-02,
            -7.355324e-03,
            9.412676e-03,
            -7.4653395e-02,
            -8.041927e-02,
            -8.32779e-03,
            -1.8497517e-02,
            3.8758285e-02,
            8.5857175e-03,
            -4.050647e-02,
            3.0432638e-02,
            5.480395e-02,
            1.6438907e-02,
            1.7074529e-03,
            -2.7431048e-02,
            2.2962129e-02,
            9.466518e-03,
            2.3686519e-02,
            6.0535505e-02,
            -3.1881552e-02,
            -2.041001e-03,
            2.9753989e-02,
            1.8944303e-02,
            5.5315755e-03,
            -2.5638076e-02,
            -6.644183e-02,
            -2.5356874e-02,
            -3.168494e-02,
            -6.40311e-02,
            3.9847083e-02,
            7.350501e-02,
            7.764983e-02,
            1.3452695e-02,
            8.481516e-03,
            2.1056293e-02,
            7.964267e-02,
            -5.5393882e-02,
            5.6886956e-02,
            -5.6051638e-02,
            -9.611592e-03,
            -6.8147756e-02,
            5.8288783e-02,
            -4.590294e-02,
            -5.54914e-02,
            -2.6479999e-02,
            -6.495312e-02,
            5.1715326e-02,
            4.8518255e-02,
            -5.646125e-02,
            7.920587e-02,
            3.766971e-02,
            4.2095613e-02,
            2.5279721e-02,
            -8.111366e-02,
            6.439882e-02,
            4.3719955e-02,
            5.874964e-02,
            4.7414135e-02,
            -6.3356543e-03,
            -6.206127e-02,
            6.409988e-02,
            5.8051865e-03,
            5.7687238e-02,
            2.7967367e-02,
            -5.112205e-02,
            3.9029606e-02,
            4.1006587e-02,
            -3.5197772e-02,
            4.4967033e-02,
            -4.460166e-02,
            2.55071e-02,
            -7.249479e-02,
            5.8610395e-02,
            4.9585845e-02,
            -1.9089337e-02,
            -5.759291e-02,
            9.252214e-04,
            -4.081312e-02,
            7.950844e-02,
            3.4963224e-02,
            3.4796786e-02,
            -7.2902516e-02,
            1.6690679e-02,
            3.0878235e-02,
            4.6669807e-02,
            2.7633071e-02,
            -2.1280615e-02,
            -7.2388746e-02,
            6.310832e-02,
            4.514545e-02,
            -1.1726492e-03,
            4.6771698e-02,
            4.413209e-02,
            8.175773e-02,
            8.1080474e-02,
            6.319841e-02,
            3.5205588e-02,
            -8.620062e-03,
            5.5396833e-02,
            6.261183e-02,
            -3.1304687e-02,
            7.934266e-02,
            -6.112178e-03,
            4.6570946e-02,
            -4.5984954e-02,
            5.2987378e-02,
            8.0859184e-02,
            -1.2947408e-02,
            -3.911011e-02,
            -4.0906426e-02,
            3.621928e-02,
            5.5266093e-02,
            6.786203e-02,
            -6.361942e-02,
            -4.391302e-02,
            6.457083e-02,
            -1.9976852e-02,
            4.625926e-02,
            -7.5061224e-02,
            4.2994574e-02,
            2.9499894e-03,
            -4.5188043e-02,
            4.7931507e-02,
            -5.9322767e-02,
            -7.691833e-02,
            3.7832513e-02,
            7.735114e-02,
            -2.8785624e-02,
            -3.6636805e-03,
            -7.063773e-02,
            -4.8054095e-02,
            6.534108e-02,
            -2.3267213e-02,
            6.0173914e-02,
            3.3875853e-02,
            6.287974e-02,
            -5.3218357e-02,
            2.4720307e-02,
            3.5112223e-03,
            5.2627854e-02,
            6.387664e-02,
            -9.481974e-03,
            -1.763537e-02,
            -3.4472786e-02,
            -6.6775635e-02,
            3.7827007e-02,
            -2.2426795e-02,
            -1.6334744e-02,
            -1.1790283e-02,
            -5.872037e-02,
            2.294568e-02,
            -6.1447352e-02,
            -5.848035e-03,
            -1.8030398e-02,
            1.8182926e-02,
            -4.7004644e-02,
            -7.4668206e-02,
            8.08572e-04,
            5.5706773e-02,
            6.331337e-02,
            4.9867727e-02,
            -4.7658347e-02,
            3.0193292e-02,
            -6.76967e-02,
            -2.624369e-02,
            -2.107511e-02,
            3.7532493e-02,
            -2.038255e-02,
            3.7053347e-02,
            -3.5569303e-02,
            7.2759084e-02,
            1.7507639e-02,
            4.6032876e-02,
            7.86051e-02,
            2.5682626e-02,
            -4.9488183e-02,
            7.272869e-02,
            7.006852e-02,
            1.2243332e-02,
            1.9225819e-02,
            2.0401044e-02,
            7.1136184e-02,
            -3.0027531e-02,
            -1.3258804e-02,
            6.1919644e-02,
            6.7412764e-02,
            6.048894e-02,
            3.298515e-02,
            7.617289e-02,
            -7.899034e-02,
            1.2180554e-02,
            -6.3292734e-02,
            1.6346827e-02,
            2.758929e-02,
            5.316311e-02,
            1.1382384e-02,
            -1.6805353e-02,
            3.0400684e-02,
            7.964519e-02,
            -7.519784e-02,
            -2.2582645e-02,
            -6.0496304e-02,
            -1.8446585e-02,
            5.116911e-02,
            1.3362248e-02,
            3.3431947e-02,
            5.807326e-02,
            8.0117255e-02,
            2.9681673e-02,
            -4.4754274e-02,
            -1.6749801e-02,
            -2.9438453e-02,
            -7.780883e-02,
            1.311548e-02,
            7.443115e-02,
            1.2196127e-02,
            6.1797608e-02,
            2.7319336e-02,
            6.4918526e-02,
            -4.249384e-02,
            1.3006312e-02,
            7.117038e-02,
            -2.9563274e-02,
            -5.2760344e-02,
            -7.68313e-02,
            -4.3269493e-02,
            2.9580027e-03,
            -4.177178e-02,
            2.6418626e-02,
            -6.48921e-02,
            -3.6019057e-02,
            -3.9292555e-02,
            5.2342054e-02,
            7.631402e-02,
            -3.3358242e-02,
            1.5656251e-02,
            3.0818729e-02,
            3.4621872e-02,
            -1.6699234e-02,
            -9.454286e-03,
            6.732509e-02,
            4.0088505e-02,
            -5.623143e-04,
            5.7830936e-03,
            1.7051228e-03,
            2.968938e-02,
            -2.8751476e-02,
            -6.320593e-02,
            -6.842523e-02,
            -2.9194191e-02,
            -7.785967e-02,
            -5.1818173e-02,
            -1.8709725e-04,
            4.961489e-02,
            -4.9885325e-02,
            -7.140343e-02,
            5.994958e-02,
            -7.2006665e-02,
            -7.0481263e-03,
            -5.928099e-02,
            -3.9670948e-02,
            1.1478091e-02,
            -2.6329674e-02,
            6.906898e-02,
            3.517967e-02,
            3.728361e-02,
            6.1261427e-02,
            3.0642841e-02,
            1.9012488e-02,
            5.6353074e-02,
            6.704331e-02,
            -7.865977e-02,
            -1.3382197e-02,
            -4.1187413e-02,
            5.2544035e-02,
            -1.3766028e-02,
            5.8615743e-04,
            -5.5746436e-02,
            3.6503512e-02,
            6.6193924e-03,
            3.5090562e-02,
            3.091542e-02,
            -6.5386035e-02,
            4.4062108e-02,
            -5.6440778e-02,
            -6.550156e-02,
            7.792287e-02,
            7.6495245e-02,
            5.20754e-02,
            -6.854494e-02,
            -7.268099e-02,
            1.9916581e-02,
            5.583811e-02,
            -7.417453e-02,
            -2.7115405e-02,
            5.81008e-02,
            4.6282478e-02,
            -3.999674e-02,
            -5.8049724e-02,
            -2.934953e-02,
            8.241866e-02,
            -1.5743336e-02,
            2.9659314e-02,
            -2.6949123e-03,
            -3.6015508e-03,
            -4.8843287e-02,
            -6.863173e-02,
            -4.978957e-02,
            -5.8241352e-02,
            -3.4874115e-02,
            -4.0661495e-02,
            1.900063e-02,
            3.7661716e-03,
            -7.3033735e-02,
            -5.239136e-02,
            7.613235e-02,
            -2.392896e-02,
            5.069495e-02,
            -5.069843e-02,
            5.179482e-02,
            2.9115917e-02,
            -1.8881818e-02,
            -3.8502548e-02,
            8.207293e-02,
            6.976784e-02,
            -5.2039504e-02,
            8.107799e-02,
            -3.716394e-02,
            -4.5143317e-02,
            2.1473719e-02,
            -6.736701e-03,
            -1.0319924e-02,
            3.4538317e-03,
            -6.1669916e-02,
            2.5141824e-02,
            1.7013147e-02,
            8.022168e-02,
            5.7734225e-02,
            -8.18752e-02,
            6.631055e-02,
            -6.584244e-02,
            4.7426485e-02,
            7.333297e-02,
            -7.581773e-02,
            -1.4232773e-02,
            -8.105268e-02,
            -1.7960235e-02,
            -4.0506158e-02,
            8.00916e-02,
            -6.474126e-02,
            -7.569423e-02,
            1.1964392e-02,
            5.763088e-02,
            -5.438446e-02,
            5.684091e-02,
            -4.767294e-02,
            2.3938626e-02,
            -4.4648245e-02,
            -2.230171e-02,
            3.0810766e-02,
            -3.5573117e-02,
            1.0298431e-02,
            7.490146e-02,
            -5.622976e-03,
            -7.6885577e-03,
            -8.155954e-02,
            -5.6022123e-02,
            -2.1372516e-02,
            -7.39606e-02,
            -6.114691e-02,
            -7.657583e-02,
            4.4233974e-02,
            -7.765102e-02,
            3.766675e-02,
            -6.1962366e-02,
            5.8945134e-02,
            -7.727707e-02,
            -2.7012777e-02,
            3.5727944e-02,
            8.0226004e-02,
            7.921842e-02,
            -3.2820087e-02,
            -9.249717e-03,
            3.7349712e-02,
            6.5234825e-02,
            -6.8233006e-02,
            -7.640977e-02,
            -2.2555694e-03,
            3.271689e-02,
            3.3429526e-02,
            1.5292931e-02,
            6.743506e-02,
            -2.071649e-02,
            5.82432e-03,
            -5.2573355e-03,
            6.564001e-02,
            7.101341e-02,
            -3.188047e-02,
            2.9797016e-02,
            -4.133634e-02,
            -2.1786178e-02,
            -5.050857e-02,
            -4.1955218e-02,
            3.756881e-02,
            5.7259046e-02,
            1.754092e-02,
            -5.8880534e-02,
            9.664617e-03,
            -2.6844016e-03,
        ],
    ),
    ("1.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("1.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "4.conv1.weight",
        &[
            -2.3520494e-01,
            1.4951986e-01,
            -7.9494834e-02,
            -3.2245433e-01,
            -1.3954836e-01,
            -1.5234011e-01,
            2.4248993e-01,
            -4.3897748e-02,
            -1.0676402e-01,
            2.2971839e-01,
            9.989786e-02,
            -1.6661495e-01,
            -4.576682e-01,
            3.1879234e-01,
            3.3578455e-01,
            4.3811607e-01,
        ],
    ),
    ("4.bn1.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("4.bn1.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "4.conv2.weight",
        &[
            -1.4959875e-01,
            2.7629038e-02,
            6.320268e-02,
            -1.6594514e-01,
            -9.367888e-02,
            6.7893267e-03,
            8.229383e-02,
            -2.0603657e-02,
            1.8815001e-02,
            2.5433421e-02,
            -1.6542596e-01,
            -4.001244e-02,
            1.4795595e-01,
            -9.328129e-02,
            1.1268e-02,
            2.5140047e-03,
            1.1276835e-01,
            -1.6324809e-01,
            -1.5545896e-01,
            6.975144e-02,
            -1.559597e-01,
            3.6475859e-03,
            -1.907283e-02,
            6.413114e-02,
            -1.1717749e-01,
            -6.799241e-02,
            -8.8107195e-03,
            -1.2752979e-01,
            1.6160154e-01,
            -5.618938e-02,
            -1.1725335e-01,
            -1.8267235e-03,
            2.4289291e-02,
            1.2269765e-01,
            -1.6503754e-01,
            -8.259543e-02,
            -1.427511e-01,
            2.7219694e-02,
            1.0969441e-03,
            1.1273806e-01,
            -5.2508395e-02,
            7.6119184e-02,
            1.4988321e-01,
            8.253046e-02,
            7.286426e-02,
            -4.6578687e-02,
            1.5020412e-01,
            9.148028e-02,
            -3.4851234e-02,
            -1.3224773e-01,
            -1.0147023e-01,
            1.5152392e-01,
            -1.2802145e-01,
            3.323277e-02,
            -2.96069e-02,
            8.1867084e-02,
            -1.0290265e-01,
            -1.1037445e-01,
            -1.393739e-01,
            -1.6654865e-01,
            -8.633914e-02,
            2.0650547e-02,
            -1.3015896e-01,
            4.2054992e-02,
            -6.5282984e-03,
            -1.3007042e-01,
            -4.569773e-02,
            -1.1436425e-01,
            6.531525e-02,
            1.2564856e-01,
            1.618098e-01,
            2.2256812e-02,
            8.0021426e-02,
            -1.1353564e-01,
            -4.3480836e-02,
            1.7814636e-03,
            -8.0965266e-02,
            1.4469302e-01,
            9.643018e-03,
            1.6530353e-01,
            -2.7964374e-02,
            1.1773388e-01,
            1.5679412e-01,
            -1.1387043e-01,
            -1.2868099e-01,
            1.2599045e-01,
            2.9701214e-02,
            -1.6329354e-01,
            1.2604056e-01,
            1.0000716e-01,
            1.344497e-01,
            -4.467986e-02,
            1.6632569e-01,
            3.8576625e-02,
            1.1074853e-01,
            2.6857397e-02,
            1.6199559e-01,
            -8.12308e-03,
            1.5628515e-01,
            1.2209068e-01,
            6.8112515e-02,
            -2.660374e-03,
            2.8581679e-02,
            1.3513361e-01,
            -3.5539508e-02,
            -6.628692e-02,
            -1.3478039e-01,
            1.283577e-01,
            -4.1547418e-03,
            -1.5319942e-01,
            -1.4337721e-01,
            5.2612185e-02,
            6.0130656e-02,
            1.5191239e-01,
            -8.611296e-02,
            -4.763007e-02,
            -1.4446817e-01,
            8.635402e-03,
            1.3756268e-01,
            -3.5240494e-02,
            1.0346115e-02,
            7.838653e-02,
            9.5838726e-02,
            -9.114548e-02,
            -1.3547581e-01,
            6.1171455e-03,
            4.877939e-02,
            3.259176e-02,
            -1.6663788e-01,
            4.7895335e-02,
            -1.2692152e-01,
            -1.15906775e-01,
            -5.5705905e-03,
            -1.02688454e-01,
            -1.2275078e-01,
            5.7301044e-02,
            -5.3058527e-02,
            -1.2956095e-01,
            1.3597488e-01,
            1.076351e-01,
            -1.0557672e-01,
            1.2710707e-01,
            -1.7300626e-02,
            1.4142787e-01,
        ],
    ),
    ("4.bn2.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("4.bn2.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "4.conv3.weight",
        &[
            -4.9283105e-01,
            4.5643705e-01,
            -4.8631835e-01,
            8.228958e-02,
            -3.3204734e-02,
            -3.253373e-01,
            4.628244e-01,
            -4.079367e-01,
            -1.8587202e-01,
            -4.0738255e-01,
            2.7079284e-01,
            2.0546746e-01,
            -4.876619e-01,
            8.5127175e-02,
            3.1270784e-01,
            -1.5305936e-02,
            4.1023487e-01,
            2.7095252e-01,
            3.9089024e-01,
            -1.1732477e-01,
            -4.090572e-01,
            -2.71973e-01,
            -1.8049973e-01,
            2.5757903e-01,
            4.6663386e-01,
            -4.4502276e-01,
            -1.5465283e-01,
            4.6326143e-01,
            4.7913164e-01,
            2.5089246e-01,
            7.090002e-02,
            -2.2332358e-01,
            -3.1409633e-01,
            -1.586014e-01,
            4.8535913e-01,
            -3.165264e-01,
            -4.041071e-01,
            -4.3239594e-03,
            -2.7801508e-01,
            4.2959774e-01,
            -3.6169708e-01,
            7.5656176e-04,
            4.9263835e-03,
            -7.5164855e-02,
            2.1073836e-01,
            1.7755508e-01,
            4.3066907e-01,
            3.314531e-02,
            -4.082495e-02,
            2.951641e-01,
            -2.2705853e-02,
            -2.2615635e-01,
            5.4873586e-02,
            -2.6350147e-01,
            -1.3320345e-01,
            -2.883557e-01,
            1.3541979e-01,
            -2.853688e-01,
            -2.610392e-01,
            -3.0860525e-01,
            2.0649308e-01,
            2.4413788e-01,
            3.109101e-01,
            3.894058e-01,
        ],
    ),
    (
        "4.bn3.scale",
        &[
            1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00,
            1e+00, 1e+00, 1e+00, 1e+00,
        ],
    ),
    (
        "4.bn3.bias",
        &[
            0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00,
            0e+00, 0e+00, 0e+00, 0e+00,
        ],
    ),
    (
        "4.downsample.0.weight",
        &[
            3.1673926e-01,
            1.883905e-01,
            1.1555934e-01,
            -3.4783125e-01,
            -4.034829e-01,
            4.566043e-01,
            -6.1674774e-02,
            2.4926245e-01,
            -1.9021338e-01,
            4.1741377e-01,
            -1.174708e-01,
            6.8378925e-02,
            -1.8126166e-01,
            -3.9767796e-01,
            3.4428775e-01,
            -1.1601025e-01,
            -2.0261222e-01,
            4.7612047e-01,
            -4.698993e-01,
            -4.3808383e-01,
            2.967602e-02,
            -1.9770151e-01,
            -3.1799775e-01,
            -2.6300734e-01,
            -2.7389723e-01,
            -2.745558e-01,
            -1.3461703e-01,
            -3.2226938e-01,
            -3.4922785e-01,
            -2.7283573e-01,
            -3.8332844e-01,
            -2.3040593e-02,
            -1.0438573e-01,
            2.427935e-01,
            -1.8420881e-01,
            3.373915e-01,
            2.9446137e-01,
            2.2263587e-01,
            -9.520787e-02,
            -1.2398422e-01,
            3.6714917e-01,
            3.661431e-01,
            -2.4312496e-02,
            2.0564401e-01,
            3.3728456e-01,
            4.4749206e-01,
            2.2566223e-01,
            3.6458927e-01,
            1.9852018e-01,
            1.59293e-01,
            5.2918136e-02,
            1.7005295e-01,
            2.8803742e-01,
            -4.0559113e-01,
            2.846418e-01,
            -1.0002762e-01,
            3.4561467e-01,
            -1.5161556e-01,
            -2.1894926e-01,
            -4.193114e-01,
            3.4352416e-01,
            1.4569551e-01,
            3.1827033e-01,
            1.3633984e-01,
        ],
    ),
    (
        "4.downsample.1.scale",
        &[
            1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00,
            1e+00, 1e+00, 1e+00, 1e+00,
        ],
    ),
    (
        "4.downsample.1.bias",
        &[
            0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00,
            0e+00, 0e+00, 0e+00, 0e+00,
        ],
    ),
    (
        "5.conv1.weight",
        &[
            7.173315e-02,
            -1.7760077e-01,
            1.7611945e-01,
            -7.927054e-02,
            7.6040626e-04,
            -2.1225959e-01,
            -2.3706585e-02,
            1.570841e-01,
            -1.0437173e-01,
            1.6037372e-01,
            2.167672e-03,
            -7.337144e-02,
            -1.5301985e-01,
            -1.544796e-01,
            2.100361e-01,
            -2.1287724e-01,
            2.4948087e-01,
            3.2671392e-03,
            -2.3602274e-01,
            1.7854333e-02,
            -1.9749612e-02,
            -3.3962727e-04,
            2.4939865e-02,
            1.3008705e-01,
            2.218574e-01,
            4.276824e-02,
            2.2834212e-01,
            -9.79799e-03,
            2.0454964e-01,
            -1.462695e-02,
            -1.25453e-01,
            3.245145e-03,
            -1.3072771e-01,
            1.2973428e-01,
            -4.6114504e-02,
            8.8584423e-04,
            1.6992867e-02,
            3.4333408e-02,
            -1.5062004e-02,
            1.3268337e-01,
            -6.394687e-02,
            1.7927182e-01,
            8.236176e-02,
            1.9561142e-02,
            -4.5393407e-03,
            5.3964794e-02,
            9.449136e-02,
            -2.2333476e-01,
            1.6476518e-01,
            -3.1181574e-03,
            5.6634903e-02,
            -1.7829117e-01,
            1.3766298e-01,
            3.5425454e-02,
            9.861866e-02,
            1.4777571e-02,
            -1.4199737e-01,
            1.6662613e-01,
            -1.0726407e-01,
            -2.0074397e-02,
            -1.23702675e-01,
            3.7733525e-02,
            1.05177194e-01,
            2.3896438e-01,
        ],
    ),
    ("5.bn1.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("5.bn1.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "5.conv2.weight",
        &[
            -3.9179526e-02,
            2.8607905e-02,
            -1.4979796e-01,
            1.07033096e-01,
            1.2755135e-01,
            1.16282724e-01,
            -7.95155e-02,
            9.029269e-03,
            -1.4056109e-01,
            -7.589062e-02,
            -3.3939302e-02,
            1.6107331e-01,
            -1.5840243e-01,
            1.2146936e-01,
            1.1548718e-02,
            -1.8431783e-02,
            -1.5331626e-02,
            1.6026625e-01,
            -2.706631e-02,
            1.6490243e-01,
            -1.0552965e-01,
            -4.297539e-02,
            -7.618569e-02,
            -9.729364e-02,
            -4.8603993e-02,
            1.0518853e-01,
            -7.433935e-02,
            8.376368e-02,
            1.877312e-02,
            9.1953024e-02,
            -1.013725e-01,
            -1.2433414e-01,
            -1.4457023e-01,
            1.5608045e-01,
            -1.1633426e-01,
            1.1442226e-01,
            1.145286e-01,
            -3.8789928e-02,
            -1.0383692e-01,
            3.0253768e-02,
            -1.01149246e-01,
            -9.914424e-02,
            1.5982161e-01,
            9.462317e-02,
            1.2787896e-01,
            -6.186682e-02,
            -7.852852e-02,
            -1.435297e-01,
            5.447954e-02,
            -1.13391384e-01,
            -1.3859801e-01,
            -1.6579203e-01,
            -3.1256538e-02,
            5.717814e-03,
            1.1455046e-01,
            8.030629e-02,
            -1.4585821e-01,
            1.6069028e-01,
            6.4659834e-02,
            -6.8568826e-02,
            2.758511e-02,
            1.6168949e-01,
            1.1464417e-01,
            6.600714e-02,
            -9.310971e-02,
            1.2589239e-01,
            -1.6578248e-01,
            -9.8881446e-02,
            1.4404383e-01,
            1.6491732e-02,
            -1.58647e-01,
            -5.1522575e-02,
            -1.4020853e-01,
            3.6180816e-03,
            -1.18291065e-01,
            7.466674e-04,
            -4.40256e-02,
            1.10353455e-01,
            -8.780056e-02,
            1.0036759e-01,
            -9.1853306e-02,
            9.8963976e-02,
            -3.6702555e-04,
            9.411951e-02,
            -1.071929e-01,
            -1.6539257e-02,
            7.2093055e-02,
            -1.4887731e-02,
            -8.692833e-02,
            -4.2152684e-02,
            9.7576976e-02,
            -4.7141813e-02,
            -1.6523441e-02,
            6.5901875e-02,
            -1.4335082e-01,
            -2.890428e-03,
            1.1274954e-02,
            -5.783121e-03,
            6.646862e-02,
            8.590039e-02,
            2.8232336e-02,
            -1.1708971e-01,
            -9.638475e-02,
            -1.4126168e-01,
            8.516648e-02,
            -7.8392826e-02,
            3.8310014e-02,
            -5.0927244e-02,
            -6.807642e-02,
            1.3516489e-01,
            5.232787e-02,
            -1.6066608e-01,
            -1.0340941e-01,
            1.3880005e-01,
            2.6671946e-02,
            2.4057647e-02,
            -3.5116456e-02,
            -6.4943515e-02,
            5.8967512e-02,
            -1.971386e-02,
            1.5706025e-01,
            -3.5766404e-02,
            -8.321981e-03,
            9.441583e-02,
            -2.036212e-02,
            -8.260387e-02,
            1.1003131e-01,
            1.3314508e-01,
            -1.4031044e-01,
            -5.4412227e-02,
            3.4760356e-02,
            -1.4250556e-01,
            1.3077652e-01,
            1.5192644e-01,
            1.4601521e-01,
            -7.3760375e-02,
            4.0552855e-02,
            -1.4823388e-01,
            3.6128126e-02,
            -2.6793262e-02,
            2.8926432e-02,
            7.953086e-02,
            -8.822928e-02,
            -1.6619762e-03,
        ],
    ),
    ("5.bn2.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("5.bn2.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "5.conv3.weight",
        &[
            -1.1413401e-01,
            4.0208858e-01,
            2.235967e-02,
            -2.2894496e-01,
            -3.4845293e-02,
            -2.1158898e-01,
            -2.9820025e-02,
            -2.9230356e-02,
            -1.6732764e-01,
            2.9340982e-02,
            3.355583e-01,
            -3.4402633e-01,
            3.856032e-01,
            2.5234222e-02,
            -1.8078369e-01,
            4.4474316e-01,
            -3.143729e-01,
            -1.2742656e-01,
            -3.9018583e-01,
            3.6810708e-01,
            -8.54193e-02,
            2.2075266e-01,
            2.5968856e-01,
            -4.5474672e-01,
            -2.0082384e-01,
            -2.0320177e-02,
            1.4120978e-01,
            3.4839308e-01,
            4.692493e-01,
            8.262861e-02,
            3.3364785e-01,
            -1.5669763e-01,
            -1.9319946e-01,
            3.3019614e-01,
            -4.6918327e-01,
            3.9493024e-01,
            -2.8498787e-01,
            1.1888242e-01,
            9.6688986e-02,
            -1.1482596e-02,
            -1.09317124e-01,
            -1.3168788e-01,
            -1.8416119e-01,
            4.6701467e-01,
            -3.0042768e-02,
            -4.6283615e-01,
            -1.4352858e-02,
            1.07626975e-01,
            -1.2876511e-02,
            -4.0417504e-01,
            4.955107e-02,
            -4.9766183e-01,
            -3.4907132e-01,
            3.009283e-01,
            -1.09028995e-01,
            -1.6425782e-01,
            -3.6913878e-01,
            4.340359e-01,
            6.3773096e-02,
            -7.4652076e-02,
            -1.8566304e-01,
            -3.1190932e-02,
            3.2896763e-01,
            -4.277122e-02,
        ],
    ),
    (
        "5.bn3.scale",
        &[
            1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00,
            1e+00, 1e+00, 1e+00, 1e+00,
        ],
    ),
    (
        "5.bn3.bias",
        &[
            0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00,
            0e+00, 0e+00, 0e+00, 0e+00,
        ],
    ),
    (
        "8.weight",
        &[
            -1.0895002e-01,
            1.6464046e-01,
            -2.4912575e-01,
            1.6612962e-01,
            -1.3868266e-01,
            1.4777675e-01,
            -2.0429492e-02,
            1.8963814e-02,
            5.7849884e-03,
            -2.3632964e-01,
            8.84071e-03,
            1.1429071e-02,
            7.595652e-02,
            1.4445424e-01,
            -1.1675462e-01,
            -1.9388175e-01,
            -4.3970197e-02,
            -5.7044446e-02,
            1.971843e-01,
            2.5448442e-02,
            4.2455554e-02,
            5.38615e-02,
            -6.636232e-02,
            2.4316353e-01,
            1.5623036e-01,
            -1.5660867e-01,
            -2.3834935e-01,
            1.3216138e-02,
            1.0422039e-01,
            -3.794998e-02,
            -1.4926937e-01,
            -2.2309631e-01,
            1.03093e-01,
            5.5564106e-02,
            8.015752e-03,
            1.3721627e-01,
            7.0259094e-02,
            1.9062218e-01,
            -1.8604228e-01,
            2.24365e-01,
            -1.9217986e-01,
            1.5159956e-01,
            5.710283e-02,
            2.0958582e-01,
            -1.5125126e-01,
            -1.3594481e-01,
            2.4320069e-01,
            1.1791199e-02,
            -1.8539459e-01,
            -1.6398436e-01,
            1.0234177e-02,
            4.6965808e-02,
            1.6428435e-01,
            -1.521343e-02,
            -6.1588526e-02,
            4.0457636e-02,
            -2.2019213e-01,
            7.8460425e-02,
            2.3552665e-01,
            -1.0246304e-01,
            -1.8664473e-01,
            1.3707465e-01,
            1.8455964e-01,
            -3.6477745e-02,
        ],
    ),
    (
        "8.bias",
        &[2.0395163e-01, 2.5505751e-02, 2.356214e-01, -1.6601676e-01],
    ),
];
const RESNET50_TINY_LOGITS: [f64; 8] = [
    4.08214033e-02,
    1.21684060e-01,
    3.67549688e-01,
    -1.25134856e-01,
    5.84951043e-02,
    1.46950632e-01,
    2.68771321e-01,
    -1.61930457e-01,
];
const RESNET50_TINY_CONV1_GRAD: [f64; 588] = [
    1.32464431e-03,
    -1.54524494e-03,
    -3.60176689e-03,
    -1.25102773e-02,
    2.41643726e-03,
    -1.36589853e-03,
    -3.48288496e-03,
    6.73982035e-03,
    -4.67468332e-03,
    1.03676636e-02,
    1.51916884e-03,
    3.18175182e-03,
    -1.45750195e-02,
    1.20636320e-03,
    -6.28956070e-04,
    2.64024604e-02,
    -5.07454388e-03,
    2.95788841e-03,
    1.65265321e-03,
    1.23594992e-03,
    -8.19945894e-03,
    -2.15417676e-04,
    -4.19309083e-03,
    -8.64416454e-03,
    -3.29950592e-03,
    -6.96859715e-05,
    -5.09618409e-03,
    -3.30003537e-03,
    -1.04874466e-03,
    1.35993492e-02,
    1.15713803e-04,
    -1.32629853e-02,
    7.54125509e-03,
    6.65064156e-02,
    -3.71228874e-01,
    1.65757276e-02,
    4.14535822e-03,
    -6.85519651e-02,
    -1.90918103e-01,
    -1.91260520e-02,
    1.25344051e-03,
    -1.91866770e-01,
    -2.69990027e-01,
    1.01982988e-01,
    1.90266408e-04,
    -6.65302202e-03,
    -2.71662563e-01,
    2.37425156e-02,
    2.04159506e-03,
    -8.26811790e-02,
    2.52952576e-01,
    2.76349522e-02,
    1.06167300e-02,
    1.06963277e-01,
    -1.81383282e-01,
    1.02226570e-01,
    7.61981495e-03,
    -1.60390232e-03,
    2.08476000e-03,
    -2.05106605e-02,
    -6.03136979e-03,
    4.10756171e-02,
    -1.86687812e-01,
    -1.61625855e-02,
    1.11977067e-02,
    -8.29394758e-02,
    7.76351467e-02,
    6.78112311e-03,
    1.02051906e-02,
    3.82414274e-02,
    -2.76392967e-01,
    4.95908894e-02,
    -2.29623145e-03,
    -4.04470414e-02,
    8.51998329e-02,
    -2.04431824e-02,
    1.01748817e-02,
    1.13540992e-01,
    -1.89849764e-01,
    4.34977971e-02,
    -7.08608329e-03,
    7.52439629e-03,
    4.96930003e-01,
    -1.60442330e-02,
    8.99103703e-04,
    4.01138002e-03,
    5.11544151e-03,
    4.07285523e-03,
    6.67945528e-03,
    -8.87570530e-02,
    -1.74171239e-01,
    2.96545625e-02,
    -1.78148597e-02,
    -5.88847138e-02,
    -2.72661030e-01,
    -1.26828663e-02,
    -8.43857899e-02,
    6.35633394e-02,
    3.26167524e-01,
    -3.68409194e-02,
    7.68616572e-02,
    -1.34589642e-01,
    2.20640004e-01,
    7.06058070e-02,
    1.97918832e-01,
    1.29399747e-02,
    1.32097721e-01,
    -8.02684575e-02,
    -1.59726758e-02,
    -1.47850737e-01,
    2.71754384e-01,
    5.96384555e-02,
    4.75978106e-03,
    8.33120756e-03,
    -1.37356785e-03,
    -4.51186672e-03,
    -2.54781917e-02,
    1.22776121e-01,
    2.73692250e-01,
    1.90741885e-02,
    9.05030034e-03,
    -1.11902714e-01,
    -4.94625904e-02,
    -2.63391361e-02,
    -5.68047315e-02,
    -6.52447641e-02,
    -6.94638073e-01,
    -6.10038713e-02,
    9.32446644e-02,
    3.18770260e-02,
    -1.46347418e-01,
    2.33219564e-02,
    -7.84449950e-02,
    -7.26640746e-02,
    2.04406485e-01,
    2.37672508e-01,
    4.16651815e-02,
    1.97862424e-02,
    -2.35945016e-01,
    1.00882746e-01,
    9.55213141e-03,
    -1.81392059e-02,
    -8.30280688e-03,
    -1.36555890e-02,
    5.05913980e-04,
    -1.40870616e-01,
    4.69122797e-01,
    -5.66071160e-02,
    1.73054785e-02,
    7.16622174e-03,
    -1.03803296e-02,
    -3.66813838e-02,
    1.14048891e-01,
    7.07261264e-02,
    2.98834115e-01,
    -9.74612385e-02,
    1.08373784e-01,
    1.28355503e-01,
    -2.55376160e-01,
    -4.54004742e-02,
    -6.11980930e-02,
    -9.86605734e-02,
    6.57842681e-03,
    -1.83024667e-02,
    1.45452976e-01,
    -1.27750516e-01,
    -2.13636518e-01,
    -2.78410278e-02,
    -8.87123169e-04,
    4.61011380e-03,
    -7.98419304e-03,
    6.34485949e-03,
    -9.66843683e-03,
    1.08371619e-02,
    -1.73412114e-02,
    5.36116175e-02,
    -6.16088556e-03,
    -3.32694799e-02,
    5.83092272e-02,
    -6.95665181e-02,
    -7.82211646e-02,
    -1.15241423e-01,
    -5.66327795e-02,
    -5.19221537e-02,
    1.54679254e-01,
    8.72543454e-02,
    4.80129272e-02,
    1.55963395e-02,
    1.26826510e-01,
    -1.57529846e-01,
    7.27614045e-01,
    2.94923503e-03,
    -6.31194711e-02,
    2.01930121e-01,
    -4.71210361e-01,
    1.93992686e-02,
    4.37898049e-03,
    -5.10823680e-03,
    5.51419286e-03,
    -2.31303722e-02,
    -3.34485457e-03,
    -2.83307652e-03,
    9.84239276e-04,
    1.71601214e-02,
    9.64905322e-03,
    -3.44182341e-03,
    1.90066255e-03,
    -2.01769155e-02,
    -1.21399732e-02,
    2.23581004e-03,
    -9.06704227e-04,
    7.33869942e-03,
    1.21534336e-02,
    -6.08764030e-03,
    -3.28699686e-03,
    -3.36082280e-02,
    -7.20383180e-03,
    -2.76087347e-04,
    -8.79137253e-04,
    8.55457969e-03,
    9.76451766e-03,
    -2.30366364e-04,
    1.66581688e-03,
    3.75293866e-02,
    -3.85449873e-03,
    1.48347355e-02,
    8.23192671e-03,
    1.67853571e-02,
    4.01709555e-03,
    9.99030769e-02,
    6.88666012e-03,
    -1.50176156e-02,
    1.06999371e-03,
    -7.38595575e-02,
    -2.45968267e-01,
    -8.42898909e-04,
    6.34253863e-03,
    -3.90883256e-03,
    -3.40853214e-01,
    6.07261434e-03,
    -5.34236524e-03,
    -8.69641304e-02,
    -1.14129722e-01,
    7.30300695e-02,
    4.20388859e-03,
    -2.25403488e-01,
    -2.30784416e-02,
    2.58543491e-02,
    -3.28361662e-03,
    -1.27406299e-01,
    1.16915375e-01,
    5.24393730e-02,
    -3.85751994e-03,
    1.89154055e-02,
    4.59810859e-03,
    2.73144841e-02,
    1.51605820e-02,
    9.26208869e-02,
    -1.79812536e-01,
    6.02028426e-03,
    -5.67000033e-03,
    -1.29345894e-01,
    -3.51030082e-02,
    7.16075115e-03,
    -3.79371690e-03,
    7.08011091e-02,
    -4.43422675e-01,
    5.17592691e-02,
    -1.46662304e-02,
    -3.96165140e-02,
    2.42120326e-02,
    6.29878230e-03,
    -6.29184023e-03,
    1.73739009e-02,
    -2.85251081e-01,
    7.00629577e-02,
    3.86840128e-03,
    -2.20792413e-01,
    -1.91744566e-02,
    7.26207346e-02,
    7.95268361e-03,
    -2.50358768e-02,
    -1.94564823e-03,
    1.56714022e-02,
    1.10691618e-02,
    8.62579867e-02,
    -1.26028329e-01,
    5.92054799e-02,
    -7.78816684e-05,
    7.68649653e-02,
    1.75968893e-02,
    -4.38229963e-02,
    -1.26603991e-01,
    -8.14135969e-02,
    -3.28399330e-01,
    4.67470177e-02,
    8.07739124e-02,
    7.24502653e-02,
    8.83638859e-03,
    -1.18683726e-02,
    1.40614826e-02,
    3.28341872e-03,
    -2.34836400e-01,
    -4.90342267e-02,
    7.55652785e-02,
    3.09734643e-02,
    -1.99429795e-01,
    -8.60524103e-02,
    6.00179564e-03,
    -2.61250865e-02,
    -6.11052103e-03,
    6.16510864e-04,
    1.96004659e-02,
    -7.51632527e-02,
    6.51583597e-02,
    3.30397300e-02,
    1.56798847e-02,
    1.31912023e-01,
    -1.17222749e-01,
    -7.43504465e-02,
    -1.22904070e-01,
    3.48807871e-02,
    -1.27287224e-01,
    1.43731292e-02,
    1.65688127e-01,
    6.31300732e-02,
    -1.16193563e-01,
    -1.77843794e-02,
    -1.21518224e-01,
    -5.44485301e-02,
    -3.08201015e-01,
    1.15724519e-01,
    9.40332338e-02,
    -8.01706985e-02,
    -2.16385081e-01,
    -1.60258904e-01,
    -7.70934857e-03,
    4.92615346e-03,
    6.60901249e-04,
    6.21597841e-03,
    1.66994426e-03,
    2.00423449e-02,
    1.53599456e-01,
    4.71268483e-02,
    -1.88685267e-03,
    1.21384434e-01,
    2.86266893e-01,
    1.83992609e-02,
    -1.18739888e-01,
    -1.57667156e-02,
    1.58416659e-01,
    -7.01012984e-02,
    -9.77840647e-02,
    -2.74077617e-03,
    -4.38855082e-01,
    -7.48420432e-02,
    7.80109540e-02,
    1.49794370e-01,
    3.82440716e-01,
    -5.26325740e-02,
    -4.88527864e-02,
    6.21554665e-02,
    -1.82137385e-01,
    7.12872297e-02,
    -1.58143044e-02,
    1.92202479e-02,
    3.58950510e-03,
    -5.64957177e-03,
    1.14171496e-02,
    -8.32321420e-02,
    3.32728297e-01,
    -3.76802459e-02,
    4.96844016e-03,
    9.77971219e-03,
    1.56421125e-01,
    8.58308971e-02,
    8.69129673e-02,
    -5.26015013e-02,
    1.24548219e-01,
    -9.54861939e-02,
    -2.10946813e-01,
    -1.27209183e-02,
    -4.33703549e-02,
    1.61455832e-02,
    -2.93146838e-02,
    -7.58443177e-02,
    -3.32492828e-01,
    7.10051041e-03,
    -7.83721730e-02,
    -1.16475046e-01,
    -5.51797822e-02,
    -2.68732905e-02,
    -3.48039740e-03,
    4.06000810e-03,
    -1.34294503e-03,
    -7.31325056e-03,
    -5.89205557e-03,
    1.41045535e-02,
    -3.69221158e-03,
    2.13694107e-02,
    6.25602435e-04,
    4.14743787e-03,
    -3.64828203e-03,
    1.45354541e-02,
    -1.35555994e-02,
    1.00822300e-02,
    -5.72170946e-04,
    1.97481308e-02,
    1.01180021e-02,
    -1.15448656e-02,
    -1.56569201e-03,
    -9.25351586e-03,
    2.79635261e-03,
    1.50505148e-04,
    1.14477647e-03,
    1.13529584e-03,
    1.11719305e-02,
    1.83094337e-04,
    5.55128325e-03,
    -1.21520553e-02,
    -3.38131911e-03,
    -7.01851957e-03,
    -4.42050165e-03,
    8.70308653e-03,
    -4.21054522e-03,
    -5.10241054e-02,
    1.81394711e-01,
    9.80858970e-03,
    -1.41805504e-03,
    7.62632787e-02,
    -4.28732842e-01,
    2.62979325e-03,
    -9.19417664e-03,
    2.76500247e-02,
    1.87081188e-01,
    -5.57440892e-02,
    6.55439962e-03,
    -2.53851712e-02,
    1.57835126e-01,
    1.11719154e-01,
    1.56305195e-03,
    -1.13499902e-01,
    -7.35963061e-02,
    -3.04596592e-02,
    -2.89165997e-03,
    -1.24080963e-02,
    -4.85535443e-01,
    -6.03831410e-02,
    -6.97876234e-03,
    1.03130601e-02,
    -4.79325978e-03,
    -2.50795996e-03,
    -6.05424680e-03,
    -7.14686438e-02,
    -3.95063609e-01,
    -8.33970867e-03,
    -1.43087818e-03,
    -5.57380430e-02,
    -4.58206505e-01,
    -1.13773085e-02,
    1.53134130e-02,
    2.43342333e-02,
    2.62198389e-01,
    1.01878099e-01,
    -1.34733301e-02,
    4.29556146e-02,
    2.77517945e-01,
    1.36387363e-01,
    -4.52853506e-04,
    7.55465925e-02,
    4.04022515e-01,
    -4.52929623e-02,
    -6.87041320e-04,
    -3.68919298e-02,
    3.71670127e-02,
    -2.23400425e-02,
    -4.38008923e-03,
    -4.99064988e-03,
    -2.75235530e-03,
    -1.29046645e-02,
    1.88838001e-02,
    2.57278979e-02,
    -3.30915153e-01,
    -4.05313857e-02,
    2.06272490e-02,
    3.47888507e-02,
    -2.36802176e-01,
    6.32867888e-02,
    6.45808652e-02,
    2.54786350e-02,
    2.80349731e-01,
    1.07633375e-01,
    -1.12094030e-01,
    -5.35934512e-03,
    5.98528013e-02,
    9.98336226e-02,
    -2.70170043e-03,
    -1.73430555e-02,
    2.44056210e-02,
    -1.93300188e-01,
    -4.50330302e-02,
    -1.11281872e-02,
    -1.02536321e-01,
    -6.23750463e-02,
    -6.75478484e-04,
    -4.02362365e-03,
    8.22555739e-03,
    -5.93906455e-03,
    9.41365585e-03,
    -2.30202526e-02,
    1.29030526e-01,
    -6.44348189e-03,
    1.12496838e-02,
    -5.28020561e-02,
    3.09833437e-02,
    -4.56836261e-02,
    7.93178976e-02,
    -8.96410719e-02,
    1.18156880e-01,
    4.17652316e-02,
    9.38753486e-02,
    -1.23915158e-01,
    3.86196256e-01,
    2.78473478e-02,
    -1.19229391e-01,
    -1.16370535e-02,
    6.32576704e-01,
    9.97428596e-03,
    -8.69762674e-02,
    -2.23197788e-02,
    6.51186466e-01,
    -1.39131501e-01,
    -1.08069750e-02,
    1.76738966e-02,
    -1.02760261e-02,
    -8.37938860e-03,
    1.07612498e-02,
    -2.70811860e-02,
    9.81147662e-02,
    2.33765598e-02,
    7.08924141e-03,
    -8.25781226e-02,
    -6.04979806e-02,
    4.14777845e-02,
    -6.17724806e-02,
    -5.23195527e-02,
    -1.60864249e-01,
    7.38313422e-03,
    -7.99579099e-02,
    1.56297565e-01,
    3.11782062e-02,
    3.88455298e-03,
    -6.46217987e-02,
    5.38095050e-02,
    -5.62332943e-02,
    1.28423795e-02,
    8.83814413e-04,
    1.09135143e-01,
    -5.18523455e-01,
    3.55836824e-02,
    2.47232663e-03,
    1.43095385e-03,
    5.04806545e-03,
    7.47269951e-03,
    8.69170949e-03,
    -8.46956298e-02,
    6.04060367e-02,
    -1.33451475e-02,
    -4.56510112e-04,
    5.84309129e-03,
    3.72493267e-01,
    -1.00822272e-02,
    4.03628573e-02,
    1.40810817e-01,
    4.31528687e-01,
    -8.71147811e-02,
    3.18493284e-02,
    -4.15750965e-03,
    -4.31254148e-01,
    -1.66242585e-01,
    1.39129892e-01,
    2.67900705e-01,
    -3.71439993e-01,
    -8.05142000e-02,
    1.43623129e-01,
    -2.49619614e-02,
    -9.90305394e-02,
    9.21310335e-02,
];

#[test]
#[ignore = "requires CUDA"]
fn resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new(
            "resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_channel",
        ),
        Axis::new(
            "resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_height",
        ),
        Axis::new(
            "resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_width",
        ),
        Axis::new(
            "resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_batch",
        ),
    );
    let axes = ResNetAxes {
        channel,
        height,
        width,
    };
    let input = Tensor::from_slice(
        &RESNET50_TINY_INPUT,
        [batch.of(2), channel.of(3), height.of(8), width.of(8)],
        &device,
    )?;
    let mut model = Sequential::new(vec![
        Box::new(
            Conv2d::new(channel, channel.of(4), [height, width], [7, 7])
                .stride([2, 2])
                .padding([3, 3])
                .bias(false),
        ) as Box<dyn Module>,
        Box::new(BatchNorm::new(channel)),
        Box::new(ReLU),
        Box::new(
            MaxPool2d::new(channel, [height, width], [3, 3])
                .stride([2, 2])
                .padding([1, 1]),
        ),
        Box::new(Bottleneck::new(axes, 4, 4, 2)),
        Box::new(Bottleneck::new(axes, 16, 4, 1)),
        Box::new(AdaptiveAvgPool2d::new([height, width], [1, 1])),
        Box::new(Flatten::new([channel, height, width], channel)),
        Box::new(Linear::new(channel, channel.of(4))),
    ]);
    model.build(input.shape(), &device, 0)?;
    for (path, values) in RESNET50_TINY_PARAMS {
        model.parameter(path)?.set_values(values)?;
    }
    let logits = model.forward(&input)?;
    close(
        "resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient logits",
        &logits.to_vec()?,
        &RESNET50_TINY_LOGITS,
    );
    logits.sum([batch, channel])?.backward()?;
    close(
        "resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient first conv weight gradient",
        &model.parameter("0.weight")?.grad().unwrap().to_vec()?,
        &RESNET50_TINY_CONV1_GRAD,
    );
    println!(
        "resnet50_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient: tiny-width equivalence PASS (16-wide head, Bottleneck)"
    );
    Ok(())
}
const VGG_TINY_INPUT: [f32; 384] = [
    -1.5255959e+00,
    -7.502318e-01,
    -6.539809e-01,
    -1.6094848e+00,
    -1.0016718e-01,
    -6.091889e-01,
    -9.7977227e-01,
    -1.6090963e+00,
    -7.121446e-01,
    3.03722e-01,
    -7.773143e-01,
    -2.5145525e-01,
    -2.2227049e-01,
    1.6871134e+00,
    2.2842517e-01,
    4.676355e-01,
    -6.969724e-01,
    -1.1607615e+00,
    6.995424e-01,
    1.9908163e-01,
    8.656924e-01,
    2.444039e-01,
    -6.6291136e-01,
    8.0730826e-01,
    1.1016806e+00,
    -1.7593604e-01,
    -2.2455578e+00,
    -1.446458e+00,
    6.1155282e-02,
    -6.1774445e-01,
    -7.9806983e-01,
    -1.3162321e-01,
    1.8793458e+00,
    -7.213178e-02,
    1.577706e-01,
    -7.734549e-01,
    1.990565e-01,
    4.5702778e-02,
    1.5295692e-01,
    -4.756788e-01,
    -1.1101983e-01,
    2.9273525e-01,
    -1.5784515e-01,
    -2.878714e-02,
    2.357111e+00,
    -1.0373387e+00,
    1.5747981e+00,
    -6.298472e-01,
    -9.273917e-01,
    5.451415e-01,
    6.628026e-02,
    -4.3704012e-01,
    7.626006e-01,
    4.4151092e-01,
    1.1651385e+00,
    2.0153918e+00,
    1.3741246e-01,
    9.386447e-01,
    -1.8600109e-01,
    -6.446393e-01,
    1.5392458e+00,
    -8.695876e-01,
    -3.3311536e+00,
    -7.4787223e-01,
    -2.5502462e-02,
    -1.0233306e+00,
    -5.9618515e-01,
    -1.0055307e+00,
    -2.106063e-01,
    -7.5475276e-03,
    1.6734272e+00,
    1.0342831e-02,
    -7.0395666e-01,
    -1.852658e-01,
    -9.962351e-01,
    -8.3125526e-01,
    -4.6102202e-01,
    -5.6008244e-01,
    3.9557618e-01,
    -9.822771e-01,
    -5.064866e-01,
    9.9775404e-02,
    -6.539734e-01,
    7.316937e-01,
    -1.4343859e+00,
    -5.0081307e-01,
    1.7163314e-01,
    -1.5999313e-01,
    2.546335e-01,
    -5.019573e-01,
    -1.0412e+00,
    7.322672e-01,
    -1.0483401e+00,
    -4.708772e-01,
    2.9113635e-01,
    1.9907043e+00,
    6.614453e-01,
    1.1899205e+00,
    8.165339e-01,
    -9.135236e-01,
    1.3851457e+00,
    -8.1384623e-01,
    -9.2757654e-01,
    1.1119633e+00,
    1.3352057e+00,
    6.042736e-01,
    -1.0344208e-01,
    -1.5121692e-01,
    -2.102083e+00,
    -6.2002194e-01,
    -1.478231e+00,
    -1.1334175e+00,
    8.737963e-01,
    -5.602594e-01,
    1.2857845e+00,
    8.1682384e-01,
    2.053041e-01,
    3.0510718e-01,
    5.35687e-01,
    -4.31185e-01,
    2.5581384e+00,
    -2.3336388e-01,
    -1.347213e-02,
    1.8606348e+00,
    -1.9804063e+00,
    1.7985829e+00,
    1.0181159e-01,
    3.40006e-01,
    7.1236455e-01,
    -1.7765073e+00,
    3.5386458e-01,
    1.1996132e+00,
    -3.029974e-01,
    -1.7618417e+00,
    6.348446e-01,
    -8.043591e-01,
    -1.6111118e+00,
    -1.8716129e+00,
    5.430836e-01,
    6.606786e-01,
    2.2952116e+00,
    6.7490596e-01,
    1.7133216e+00,
    -1.7942734e+00,
    -1.3632672e+00,
    -9.832196e-01,
    1.5112667e+00,
    6.4187074e-01,
    4.729638e-01,
    -4.285901e-01,
    5.5137074e-01,
    -1.5473709e+00,
    5.181121e-01,
    1.0653535e-01,
    2.6924077e-01,
    1.324768e+00,
    1.7460191e+00,
    1.854969e+00,
    -7.063691e-01,
    2.5570862e+00,
    4.175343e-01,
    -2.127186e-01,
    -8.39958e-01,
    -4.2001787e-01,
    -6.240363e-01,
    -9.772961e-01,
    8.748428e-01,
    9.872814e-01,
    3.0957633e-01,
    1.52069e+00,
    1.2052339e+00,
    -1.815591e+00,
    -4.0346155e-01,
    -9.591452e-01,
    -5.207704e-03,
    -7.886314e-02,
    8.436543e-01,
    1.1657013e+00,
    5.269322e-01,
    1.6192533e+00,
    -9.6397626e-01,
    1.4152038e-01,
    -1.6366096e-01,
    -3.5822257e-01,
    1.7222793e+00,
    -3.035756e-01,
    2.388742e-01,
    1.3440012e+00,
    1.0322569e-01,
    1.1003542e+00,
    -3.416802e-01,
    9.473389e-01,
    -5.6851596e-01,
    8.375962e-01,
    1.7836607e+00,
    -1.9542466e-01,
    5.149161e-01,
    -1.8474776e+00,
    -2.9167426e+00,
    -5.673299e-01,
    -5.412802e-01,
    8.95174e-01,
    -8.82507e-01,
    5.3181124e-01,
    -1.5457772e+00,
    -1.7329982e-01,
    7.2824633e-01,
    5.706102e-02,
    9.055172e-01,
    1.0462948e+00,
    -5.20597e-01,
    1.3547838e+00,
    2.3519313e-01,
    1.9142433e+00,
    1.8364111e+00,
    1.3245324e+00,
    -9.6900916e-01,
    1.2516364e+00,
    1.2103242e+00,
    -5.279206e-01,
    2.1856615e-01,
    -5.7430726e-01,
    1.4571251e+00,
    1.7709557e+00,
    1.6499138e+00,
    -4.3200457e-01,
    -2.710269e-01,
    -1.4391626e+00,
    1.2470404e+00,
    1.2738512e+00,
    3.9094925e-01,
    3.872105e-01,
    -7.982872e-02,
    3.417243e-01,
    9.488273e-01,
    -1.3839359e+00,
    1.7240863e+00,
    -2.3647652e+00,
    -9.294909e-01,
    2.936253e-01,
    2.1513203e-01,
    9.384637e-01,
    1.4657077e+00,
    -5.564744e-01,
    -7.448408e-01,
    -2.0215721e-01,
    -2.296679e-01,
    1.3313366e-03,
    3.752759e-01,
    -5.810679e-01,
    -5.7230884e-01,
    1.0097175e+00,
    -1.0564939e-01,
    -1.179696e+00,
    -9.0779595e-02,
    5.6311435e-01,
    -1.2560141e+00,
    8.95555e-01,
    1.6747737e-01,
    7.5142086e-01,
    2.4142299e+00,
    1.020584e+00,
    -4.404838e-01,
    -1.7341677e+00,
    -1.236225e+00,
    1.5785813e+00,
    -1.1160507e+00,
    7.6777023e-01,
    -5.8820677e-01,
    2.1188903e+00,
    -5.4219025e-01,
    -2.4592547e+00,
    -1.1108288e+00,
    -1.1187209e+00,
    7.5799555e-01,
    -4.9565765e-01,
    -1.9700006e-01,
    -3.3396218e-02,
    7.192915e-01,
    1.0644146e+00,
    8.3402544e-01,
    -1.9162164e+00,
    -3.4202927e-01,
    -6.6049206e-01,
    3.1508535e-01,
    1.1422518e+00,
    3.0550566e-01,
    -5.7888174e-01,
    -2.3828252e-01,
    -1.3541743e+00,
    2.686894e-01,
    1.1455697e-01,
    -1.5562972e+00,
    -1.0757437e+00,
    -8.751946e-01,
    -4.7281876e-01,
    9.912368e-01,
    -5.8622282e-02,
    1.1787646e+00,
    6.22185e-01,
    7.878501e-01,
    1.3685523e+00,
    -8.506898e-01,
    5.1260746e-01,
    1.0476325e+00,
    -3.1758463e-01,
    1.3948506e-01,
    2.3402624e+00,
    -6.1160916e-01,
    8.160271e-01,
    2.47723e-01,
    -3.867267e-01,
    1.9948451e-01,
    7.9926956e-01,
    -2.619034e-01,
    1.5132962e-01,
    1.1981666e+00,
    -2.2832582e+00,
    -1.0129594e+00,
    -8.878909e-01,
    6.5221924e-01,
    -8.726203e-01,
    3.5253752e-02,
    -3.365303e-01,
    1.402332e+00,
    4.841214e-01,
    -7.0304507e-01,
    -8.267661e-01,
    7.74396e-01,
    6.9199395e-01,
    -1.01848e+00,
    -8.0337167e-01,
    -7.071132e-01,
    7.521183e-01,
    -1.9208279e-02,
    1.103333e+00,
    -6.0679215e-01,
    -5.252235e-01,
    -5.6618774e-01,
    6.6039857e-04,
    7.224588e-01,
    1.526352e-01,
    1.4495978e-01,
    -2.3442194e+00,
    3.60003e-01,
    4.6668175e-01,
    1.2830665e+00,
    1.2678007e+00,
    1.9883296e-01,
    5.440877e-01,
    -3.9781693e-01,
    -1.9291055e+00,
    2.323687e-01,
    8.6146563e-01,
    6.217573e-01,
    -1.7811896e+00,
    -7.820609e-01,
    -1.42367e+00,
    1.6090765e+00,
    -3.2787595e-02,
    8.532334e-01,
    5.506365e-02,
    -1.7425371e+00,
    8.7500376e-01,
    -2.7188172e+00,
    -2.2192061e-01,
    3.4208494e-01,
    1.1093477e+00,
    -5.731476e-01,
    9.577846e-01,
    9.820232e-04,
    -1.3847686e+00,
    -9.965023e-01,
    8.0734813e-01,
    1.1738863e+00,
    -9.3984646e-01,
    1.3109189e+00,
    -3.1670693e-01,
    -1.861041e-01,
    -5.764602e-01,
    6.866518e-01,
    4.2086706e-01,
    -1.0213808e+00,
    9.8856664e-01,
];
const VGG_TINY_PARAMS: &[(&str, &[f32])] = &[
    (
        "0.weight",
        &[
            -1.2436617e-01,
            9.2225045e-02,
            1.3273622e-01,
            8.826301e-02,
            1.2338974e-01,
            7.832153e-02,
            1.3600716e-01,
            -1.370142e-01,
            6.8348855e-02,
            -1.2701088e-01,
            -1.5155727e-01,
            -4.207093e-02,
            4.7625374e-02,
            7.0924535e-02,
            1.848137e-01,
            1.886668e-03,
            -3.815909e-02,
            8.175714e-02,
            -1.8926519e-01,
            -1.11202806e-01,
            -6.005795e-02,
            -1.5301888e-01,
            1.1063298e-01,
            -4.917526e-02,
            3.2658488e-02,
            1.6732323e-01,
            1.3545435e-02,
            -8.2995035e-02,
            1.4892732e-01,
            -1.1710221e-01,
            9.106907e-02,
            1.834119e-01,
            2.755964e-02,
            -7.525776e-02,
            -1.0407054e-01,
            1.6349323e-01,
            -9.0999626e-02,
            -1.5851837e-01,
            1.15700305e-01,
            1.6598444e-01,
            1.9117765e-01,
            1.7589514e-01,
            -9.523061e-02,
            1.5729645e-01,
            4.1159494e-03,
            3.664587e-02,
            -1.7010586e-01,
            -1.8431078e-01,
            1.5213518e-01,
            1.6204968e-01,
            6.447717e-02,
            6.4375006e-02,
            4.69039e-02,
            1.25671495e-02,
            1.0533938e-01,
            7.093493e-02,
            1.7250389e-01,
            1.04605034e-01,
            1.9076805e-01,
            -1.4115803e-01,
            1.1103866e-01,
            -6.1446525e-02,
            -2.0113477e-02,
            7.800787e-02,
            -9.811628e-02,
            -1.53532e-01,
            1.8513289e-01,
            -1.0442664e-01,
            1.3518758e-01,
            3.089355e-02,
            1.23637214e-01,
            3.0602783e-02,
            1.2380563e-03,
            -1.7162422e-02,
            -9.544833e-02,
            6.374511e-02,
            -1.5448302e-01,
            -1.9431507e-02,
            -1.4844531e-01,
            1.6606422e-01,
            -4.032714e-02,
            -9.45151e-02,
            1.07938275e-01,
            -1.0096215e-01,
            -1.7370778e-01,
            3.984057e-03,
            -1.0543876e-01,
            -1.514149e-01,
            -9.970881e-02,
            5.526854e-02,
            8.827586e-02,
            1.7806605e-01,
            -1.3633855e-01,
            6.942597e-02,
            -1.6482137e-01,
            8.0784366e-02,
            1.3162655e-01,
            -1.5503092e-01,
            1.4040579e-01,
            9.353984e-02,
            -4.043148e-02,
            9.2976525e-02,
            5.7680715e-02,
            6.927522e-02,
            -3.325094e-02,
            1.9790204e-02,
            -2.2124605e-02,
            -6.811978e-02,
        ],
    ),
    (
        "0.bias",
        &[1.15768075e-01, -1.6801554e-01, 2.678221e-02, 8.699509e-02],
    ),
    (
        "2.weight",
        &[
            1.1387412e-01,
            7.397684e-02,
            3.529201e-02,
            -8.873618e-02,
            -8.322847e-02,
            4.953001e-02,
            -2.3808877e-03,
            -1.5167817e-01,
            -1.1264827e-01,
            -2.5424223e-02,
            1.1795795e-01,
            9.685644e-02,
            6.247159e-02,
            -1.2901811e-01,
            -2.5266012e-02,
            1.5630546e-01,
            -1.323607e-01,
            1.0610139e-01,
            -1.3501696e-01,
            -9.7104155e-02,
            -1.3851053e-01,
            -1.4286836e-03,
            -1.23268746e-01,
            6.845381e-02,
            -1.498864e-01,
            -1.1076455e-01,
            -4.8920833e-02,
            7.0952177e-03,
            -8.082245e-02,
            1.4373165e-01,
            -3.691284e-02,
            -3.5636246e-02,
            -3.1812966e-02,
            9.292464e-02,
            1.9020438e-02,
            -4.5318406e-02,
            -9.86989e-02,
            -1.3098824e-01,
            -3.737841e-02,
            4.636693e-02,
            -7.047103e-02,
            -8.323618e-02,
            6.149602e-02,
            3.674271e-02,
            -1.4758116e-01,
            -5.6717396e-02,
            -5.5430632e-02,
            1.12568125e-01,
            1.02440596e-01,
            -9.621388e-02,
            1.2232566e-01,
            6.101237e-02,
            -3.8154982e-02,
            7.783115e-02,
            3.8281243e-02,
            7.584288e-02,
            -1.0515758e-01,
            -1.1889905e-01,
            -5.1239293e-02,
            1.4702617e-01,
            4.33925e-02,
            1.5489483e-01,
            1.4853346e-01,
            5.189755e-02,
            -1.24617994e-01,
            -6.888907e-02,
            4.2548995e-02,
            2.4740418e-02,
            1.1666505e-01,
            9.837924e-02,
            -1.13265e-01,
            -6.067741e-02,
            9.5835924e-02,
            5.6758127e-03,
            -9.275727e-02,
            1.22231565e-01,
            -1.6378526e-01,
            -7.3298514e-02,
            1.045025e-01,
            2.461356e-02,
            -1.3082436e-01,
            1.1129713e-01,
            7.612016e-02,
            1.07428074e-01,
            -1.09896086e-01,
            -1.2717228e-01,
            1.3013236e-01,
            1.609411e-01,
            -9.022187e-02,
            -8.8180326e-02,
            -1.2802621e-02,
            -3.7275493e-02,
            2.0810306e-02,
            1.9976735e-02,
            1.165701e-01,
            1.17350146e-01,
            -1.6480947e-01,
            1.3221245e-01,
            8.286308e-02,
            -3.384676e-02,
            -1.23561144e-01,
            -7.1409225e-02,
            5.154286e-02,
            1.5040496e-01,
            7.611418e-02,
            -1.0148847e-01,
            -3.857432e-02,
            1.3118255e-01,
            -5.458542e-02,
            -1.06402636e-01,
            1.6066477e-01,
            -5.9739273e-02,
            -6.6320404e-02,
            -7.347952e-02,
            3.372729e-02,
            5.977313e-03,
            -1.378802e-01,
            -5.7572465e-02,
            -4.300229e-02,
            -7.278465e-02,
            6.755191e-02,
            -3.8820207e-02,
            -2.3696027e-03,
            -6.740787e-02,
            1.1769354e-02,
            -9.481432e-02,
            1.6384141e-01,
            -3.3744574e-02,
            -6.44217e-02,
            5.2090865e-02,
            1.11941814e-01,
            -1.1362832e-03,
            -1.02063976e-01,
            1.3535818e-03,
            -1.2351036e-02,
            -5.8098853e-02,
            3.5445392e-02,
            -1.08894706e-01,
            1.6339426e-01,
            -1.5733306e-01,
            4.5348626e-02,
            -9.5177434e-02,
            7.318924e-02,
            -1.0080457e-01,
        ],
    ),
    (
        "2.bias",
        &[
            -1.42423315e-02,
            -5.3205054e-02,
            1.2379255e-01,
            1.3547616e-01,
        ],
    ),
    (
        "5.weight",
        &[
            1.0617709e-01,
            1.1723914e-01,
            -3.8716458e-02,
            8.688025e-02,
            1.3956949e-01,
            1.9529542e-02,
            7.3763594e-02,
            1.3513431e-01,
            6.308626e-02,
            -1.1987506e-01,
            6.1920784e-02,
            -7.202868e-02,
            -1.3212726e-01,
            -1.16555475e-01,
            2.0258646e-02,
            -5.8167856e-02,
            -1.5894984e-01,
            -3.969103e-02,
            1.5746078e-01,
            -7.684004e-02,
            -9.950887e-02,
            1.3203642e-01,
            -2.7817369e-02,
            -3.9355457e-02,
            -1.4686835e-01,
            2.0063441e-02,
            1.21191464e-01,
            1.2706272e-01,
            1.3375978e-02,
            -1.1662893e-01,
            -5.4792803e-02,
            -1.5714435e-01,
            1.0374644e-01,
            1.0634647e-01,
            8.903744e-02,
            -5.0859295e-02,
            -1.1730258e-01,
            5.1184617e-02,
            3.6497734e-02,
            -6.966005e-02,
            1.5457593e-01,
            5.6084555e-02,
            5.9933346e-02,
            -6.9443665e-02,
            -1.5085405e-01,
            -3.300804e-02,
            4.024595e-02,
            1.2235292e-01,
            -1.3868593e-01,
            -7.389629e-02,
            7.831987e-02,
            -1.2416826e-01,
            1.17405795e-01,
            1.2660047e-02,
            -8.216574e-02,
            -1.2884447e-01,
            1.1074376e-01,
            -9.49835e-02,
            -5.930736e-02,
            -1.6140026e-01,
            -1.0586858e-01,
            7.796975e-02,
            4.640017e-02,
            1.2793897e-01,
            -5.445679e-03,
            8.826876e-02,
            -1.0244763e-01,
            -1.0620785e-01,
            -6.442388e-02,
            -1.3679647e-01,
            -1.00611076e-01,
            -5.0430816e-02,
            -7.78031e-02,
            -1.3321698e-01,
            1.1177799e-02,
            -4.1187547e-02,
            3.839507e-02,
            1.4884186e-01,
            1.638364e-01,
            -4.3829262e-02,
            -6.402658e-02,
            3.537814e-02,
            1.5384261e-01,
            -1.6104667e-01,
            -1.3613026e-02,
            -9.278365e-02,
            1.5690903e-01,
            -1.0000205e-01,
            1.0466313e-01,
            -3.8994454e-02,
            5.0804734e-02,
            -1.6203299e-01,
            1.60936e-01,
            3.885011e-02,
            1.6584688e-01,
            1.2925375e-01,
            -1.6544734e-01,
            6.716907e-02,
            -1.9422393e-02,
            -2.6792407e-02,
            -7.509814e-02,
            7.210015e-02,
            1.4818583e-01,
            -6.341096e-02,
            -1.1213291e-01,
            6.665395e-02,
            -2.085374e-02,
            -8.315136e-02,
            -3.6292713e-02,
            1.16744265e-01,
            9.804209e-02,
            -1.2789726e-01,
            3.437891e-02,
            -1.4431787e-01,
            -1.529573e-01,
            1.17319524e-01,
            8.5063696e-02,
            2.3000678e-02,
            1.3399544e-01,
            -1.7900865e-03,
            1.148599e-01,
            -1.0629511e-01,
            8.834605e-02,
            -1.5195422e-01,
            -1.5320686e-01,
            6.4385176e-02,
            -4.506564e-02,
            -9.716209e-02,
            -1.2540881e-01,
            1.0339874e-01,
            -8.185184e-02,
            -1.383662e-01,
            1.5135533e-01,
            -5.91595e-02,
            5.6514382e-02,
            1.6189502e-01,
            2.4176758e-02,
            -1.3946724e-01,
            -1.1320536e-01,
            9.762584e-02,
            -7.112505e-02,
            -5.9487045e-02,
            -9.633394e-02,
            -2.0980637e-03,
            -1.5473877e-01,
            -5.737166e-03,
            -1.4649211e-01,
            2.464503e-02,
            6.2260114e-02,
            -3.418128e-02,
            -1.2520896e-01,
            -5.954079e-02,
            -1.16379485e-01,
            -8.58684e-02,
            -1.1362493e-02,
            1.0712354e-01,
            7.219639e-02,
            1.0324679e-01,
            7.06511e-02,
            -1.06614195e-01,
            -4.3188e-02,
            -1.6148907e-01,
            -1.8691143e-02,
            1.5805772e-01,
            6.2396448e-02,
            -8.440932e-02,
            8.9384735e-02,
            -2.6509166e-02,
            -1.5440321e-01,
            1.2321627e-01,
            -1.4627782e-01,
            -7.939949e-02,
            1.19111955e-01,
            -1.291548e-01,
            3.0902823e-02,
            -1.2745093e-01,
            9.451278e-02,
            3.1636994e-02,
            1.6211477e-01,
            -2.4274867e-02,
            -1.0623634e-02,
            -3.461657e-02,
            1.2770668e-01,
            -1.9104522e-02,
            7.547353e-02,
            -8.51538e-02,
            6.02116e-02,
            -3.1968694e-02,
            1.2652135e-01,
            1.1686048e-02,
            -4.557888e-03,
            5.245344e-02,
            -8.478e-02,
            1.4223379e-01,
            9.4107114e-02,
            -1.2772185e-01,
            1.3626778e-01,
            -1.3942586e-01,
            -1.1898142e-01,
            -8.620139e-02,
            -2.616316e-02,
            -1.04710326e-01,
            1.3264047e-01,
            -1.3111019e-01,
            1.1359726e-01,
            -9.552171e-02,
            1.1167763e-01,
            -1.4428675e-01,
            -8.352941e-02,
            -9.4581805e-02,
            -8.020012e-02,
            2.6076238e-03,
            1.3047996e-01,
            -1.19790636e-01,
            7.5916275e-02,
            -1.4597285e-01,
        ],
    ),
    (
        "5.bias",
        &[
            1.0521128e-01,
            -2.3851812e-02,
            -9.2597745e-02,
            1.1678809e-01,
            -3.781279e-02,
            8.619841e-02,
        ],
    ),
    (
        "7.weight",
        &[
            -4.2184222e-02,
            2.2046585e-02,
            -2.9057486e-02,
            -1.3329151e-01,
            1.09777726e-01,
            1.24226436e-01,
            -1.1385355e-01,
            4.8972264e-02,
            -1.2456151e-01,
            2.9782413e-03,
            9.04258e-02,
            7.374394e-02,
            2.9887501e-02,
            -1.2837802e-01,
            4.0694155e-02,
            -5.5515572e-02,
            1.2760629e-01,
            -4.3165836e-02,
            8.4737554e-02,
            1.01960845e-01,
            -4.146173e-02,
            -4.5878436e-02,
            2.3336845e-02,
            8.03335e-02,
            -1.3268504e-01,
            2.145932e-02,
            6.252143e-02,
            1.0018221e-01,
            -1.1004773e-01,
            -1.3235909e-01,
            1.4165712e-02,
            -1.2676518e-01,
            8.676429e-02,
            2.2224788e-02,
            -1.17067e-01,
            1.06386855e-01,
            2.712281e-02,
            4.3588508e-02,
            -2.1635754e-02,
            6.2151056e-02,
            -7.0310935e-02,
            1.3209803e-01,
            1.5618957e-02,
            8.635998e-02,
            6.59974e-02,
            -3.803134e-02,
            1.12319455e-01,
            -6.179751e-03,
            -3.36748e-03,
            5.084831e-02,
            2.7188756e-02,
            -1.0797982e-01,
            7.8251995e-02,
            2.2396388e-02,
            -4.2300668e-02,
            1.11080736e-01,
            9.1388986e-02,
            2.7134443e-02,
            3.9828204e-02,
            -3.1931758e-02,
            -1.0564593e-01,
            2.813309e-03,
            -8.7760955e-02,
            -9.012037e-02,
            -1.0363099e-01,
            -8.6147584e-02,
            2.2195814e-02,
            -1.1289587e-01,
            -1.1947454e-02,
            1.68611e-02,
            -1.002256e-01,
            -6.1551966e-02,
            -9.1395244e-02,
            -3.0869473e-04,
            -4.534685e-02,
            -1.0620205e-01,
            1.1102303e-01,
            -9.0371845e-03,
            -1.1243806e-01,
            9.891176e-02,
            1.192401e-01,
            1.0259162e-01,
            -1.4125902e-02,
            -1.1133127e-01,
            -4.368977e-02,
            -6.545373e-02,
            -1.2214686e-01,
            -9.2701465e-02,
            -1.3549364e-01,
            -1.09984025e-01,
            -9.315631e-02,
            5.80435e-02,
            -7.648848e-02,
            1.1814136e-01,
            -1.6822817e-02,
            1.4934699e-02,
            4.1709412e-02,
            3.1369004e-02,
            1.5362385e-02,
            9.612931e-02,
            -3.267002e-02,
            -8.85456e-02,
            9.693196e-02,
            -2.20795e-02,
            1.2080553e-01,
            1.0287078e-01,
            2.0526764e-03,
            -7.402168e-02,
            1.05759405e-01,
            9.6711045e-04,
            9.2074975e-02,
            8.16555e-02,
            5.6951817e-02,
            -1.1768327e-03,
            -8.4347084e-02,
            5.7896473e-02,
            -1.2734056e-01,
            3.1497683e-02,
            5.236286e-02,
            -7.171602e-02,
            7.419182e-02,
            -9.3122534e-02,
            -9.567502e-02,
            -6.6324673e-03,
            -1.0412763e-01,
            1.2596485e-01,
            9.670222e-02,
            8.592003e-02,
            1.3194712e-01,
            -2.1721863e-03,
            -1.4915135e-03,
            -4.9125805e-02,
            1.5265375e-03,
            9.212819e-02,
            1.9832123e-02,
            -5.4123048e-02,
            -6.7438886e-02,
            -7.566612e-02,
            5.7411768e-02,
            7.636219e-02,
            -1.1655579e-01,
            -1.250868e-01,
            9.205024e-02,
            -3.625339e-02,
            7.700125e-02,
            1.3598384e-01,
            -4.2872924e-02,
            1.24035954e-01,
            6.738584e-02,
            -1.1102631e-01,
            1.0412341e-01,
            -5.9422557e-03,
            5.9493415e-02,
            7.0507764e-03,
            7.469334e-02,
            7.010414e-02,
            7.281423e-02,
            -9.609333e-02,
            -2.8455913e-02,
            6.4002335e-02,
            1.2371876e-01,
            1.169217e-01,
            5.808626e-02,
            6.213866e-03,
            -1.0452907e-01,
            4.9946285e-03,
            6.6844195e-02,
            -7.848049e-02,
            -5.16501e-02,
            -3.9480772e-02,
            -8.401967e-02,
            3.9106376e-02,
            -1.359864e-01,
            -5.0587956e-02,
            -7.58713e-02,
            4.8038814e-02,
            -7.049561e-02,
            -8.384477e-02,
            3.433776e-02,
            1.2700166e-01,
            -6.452839e-02,
            1.1511106e-01,
            -5.330333e-03,
            -1.0578608e-01,
            -9.337802e-02,
            -9.844148e-02,
            1.1196664e-01,
            -7.448267e-02,
            5.3329676e-02,
            1.0378249e-01,
            1.817261e-02,
            3.68566e-02,
            -3.2960124e-02,
            5.6985314e-03,
            6.533722e-02,
            5.1604774e-02,
            1.454559e-03,
            -1.1087549e-01,
            4.8672343e-03,
            1.3235903e-01,
            -6.610786e-02,
            6.719263e-02,
            1.3496977e-01,
            -1.2111986e-01,
            -1.2690876e-01,
            -1.0863449e-01,
            -2.2832816e-02,
            -1.3506973e-01,
            -9.297481e-02,
            2.0591004e-04,
            -6.0447645e-03,
            -2.3482878e-02,
            -1.0506759e-01,
            9.200283e-03,
            -1.3332862e-01,
            -7.766755e-02,
            -3.8388945e-02,
            1.321444e-01,
            1.0291169e-01,
            -1.269317e-01,
            -3.6480956e-02,
            7.370048e-02,
            -8.7805815e-02,
            9.50861e-02,
            1.3580436e-01,
            -1.5572902e-02,
            2.1928973e-02,
            -4.2091172e-02,
            1.0539108e-01,
            3.949672e-02,
            1.3226885e-01,
            -7.193922e-03,
            9.968662e-02,
            1.3407918e-03,
            -1.1017418e-01,
            -5.869265e-02,
            5.5613637e-02,
            -9.573696e-02,
            1.1033613e-01,
            -7.1045876e-02,
            -1.9452974e-02,
            -1.2685473e-02,
            -2.9017888e-02,
            -1.3475259e-01,
            1.04803614e-01,
            5.5921163e-02,
            -9.648766e-03,
            -1.2202876e-01,
            -3.3923325e-03,
            8.9565106e-04,
            4.295767e-02,
            1.2608379e-01,
            -1.2319625e-01,
            -1.2811764e-01,
            4.9096476e-02,
            1.2237913e-01,
            -3.888979e-02,
            -2.0457283e-02,
            8.227755e-02,
            -1.2750067e-01,
            -1.1795777e-01,
            1.2264115e-01,
            -2.8773742e-02,
            -8.3991714e-02,
            -4.329988e-02,
            1.3070372e-01,
            8.447568e-03,
            -8.28501e-02,
            -7.4419975e-02,
            -1.3272476e-01,
            6.1134905e-02,
            1.0763197e-01,
            -1.10615544e-01,
            -2.4173932e-02,
            2.6611062e-02,
            1.3040312e-01,
            7.595037e-02,
            5.3980112e-02,
            -1.3605927e-01,
            -1.1379831e-01,
            -9.463749e-02,
            5.735572e-02,
            1.1999622e-01,
            -3.4180466e-02,
            -4.548368e-03,
            -1.0627435e-01,
            4.6786107e-02,
            5.62003e-02,
            3.3660006e-02,
            1.17166154e-01,
            -4.3322105e-02,
            -3.7312042e-02,
            8.7883696e-02,
            2.3168683e-02,
            1.021622e-01,
            -3.594536e-02,
            -8.620303e-02,
            1.3211715e-01,
            1.1547538e-01,
            6.828428e-02,
            1.2567885e-01,
            9.44726e-02,
            2.2559015e-02,
            -3.5501953e-02,
            -1.3413163e-01,
            4.8324376e-02,
            2.697089e-02,
            -4.4290368e-03,
            5.543462e-03,
            7.873491e-03,
            1.1165179e-01,
            6.644592e-02,
            -2.7727418e-02,
            -6.401468e-02,
            2.0766303e-02,
            1.2802187e-01,
            -8.5486196e-02,
            8.5108295e-02,
            -3.7259463e-02,
            -3.7980255e-02,
            -7.616386e-02,
            2.425094e-02,
            -1.1111144e-02,
            1.9296544e-02,
        ],
    ),
    (
        "7.bias",
        &[
            1.1721328e-01,
            8.4619015e-02,
            -4.1657486e-03,
            -6.0780983e-02,
            9.02101e-03,
            1.0598283e-01,
        ],
    ),
    (
        "12.weight",
        &[
            1.2930827e-01,
            -2.5178622e-02,
            -7.399976e-02,
            -1.0673873e-01,
            1.4380093e-01,
            7.232888e-04,
            -1.1403564e-01,
            1.9628526e-01,
            -8.271609e-02,
            -1.2982205e-01,
            -1.4257167e-01,
            1.3874618e-02,
            -1.9356346e-02,
            1.0833552e-01,
            1.7641693e-01,
            -5.2633885e-02,
            -4.26153e-02,
            -3.8868453e-02,
            1.3769585e-01,
            -5.2212406e-02,
            1.7698968e-03,
            1.5971607e-02,
            -1.7171967e-01,
            1.2882912e-01,
            8.104553e-02,
            1.1620453e-01,
            1.4024316e-01,
            -3.7063563e-03,
            1.7149377e-01,
            -1.8235208e-01,
            -5.392013e-02,
            1.1261899e-01,
            7.69101e-02,
            1.01760976e-01,
            -1.6235135e-01,
            1.3453022e-01,
            -1.9271177e-01,
            -1.4557414e-01,
            -1.1249686e-01,
            1.9115873e-01,
            1.9437538e-01,
            -1.073723e-01,
            -1.1138473e-01,
            1.1240135e-01,
            2.0363316e-02,
            1.2065836e-02,
            -1.3128395e-01,
            -4.7507767e-02,
            9.9120036e-02,
            -5.0616346e-02,
            1.8268788e-01,
            -1.1594037e-01,
            1.8644056e-01,
            -1.6390678e-02,
            -1.0646503e-01,
            -1.214264e-01,
            6.50311e-02,
            -4.0836107e-02,
            5.9479944e-02,
            -1.0100281e-01,
            -1.02431946e-01,
            1.951136e-01,
            -2.0237e-02,
            -7.577107e-02,
            4.7176905e-02,
            -7.7654295e-02,
            1.4055489e-01,
            -1.4501043e-01,
            -3.7652336e-02,
            -4.7984924e-02,
            1.3808942e-02,
            -1.3887551e-01,
            -1.918356e-01,
            -1.1181808e-01,
            -1.5649319e-01,
            -1.7330924e-01,
            -1.22980755e-02,
            1.5621786e-01,
            3.457741e-02,
            7.0028636e-03,
            -7.5202934e-02,
            1.4988802e-01,
            9.212622e-02,
            1.309446e-01,
            6.72481e-02,
            -1.7215148e-01,
            1.04307204e-01,
            1.9680458e-01,
            2.1603739e-02,
            1.410966e-01,
            1.2993333e-01,
            -1.2613207e-01,
            7.715188e-02,
            -1.9400257e-01,
            -8.3376236e-02,
            1.9802837e-01,
            -1.4200152e-01,
            1.7040846e-01,
            -4.7360986e-02,
            2.667608e-03,
            4.6242207e-02,
            -1.8777331e-02,
            -1.2665015e-01,
            1.5418606e-01,
            -1.7884699e-01,
            -1.12086944e-01,
            -9.406283e-03,
            -2.773045e-04,
            8.05218e-02,
            -1.2924689e-01,
            -4.30087e-02,
            2.0198165e-02,
            1.3773951e-01,
            1.494773e-01,
            1.4884296e-01,
            3.4920122e-02,
            -8.758075e-02,
            -5.952749e-02,
            1.9235873e-01,
            4.4312268e-03,
            6.9423825e-02,
            -6.1896797e-02,
            5.566051e-02,
            -1.1942855e-02,
            8.587682e-02,
            2.2992281e-02,
            -2.49384e-02,
            1.3515483e-01,
            -1.6472122e-01,
            -4.7957256e-02,
            5.856987e-02,
            1.059276e-01,
            -6.4724125e-02,
            -1.7706166e-01,
            -1.7184448e-01,
            1.2120563e-01,
            1.2115185e-02,
            -5.4957174e-02,
            6.208691e-04,
            2.8033111e-02,
            1.2825865e-01,
            1.4026831e-01,
            1.6016789e-01,
            -2.025637e-02,
            1.2021335e-01,
            -9.925535e-03,
            -8.521916e-02,
            1.4637484e-01,
            -5.9907533e-02,
            -1.2388202e-01,
            4.9666904e-02,
            -5.1626284e-02,
            1.17590785e-01,
            -8.9385666e-02,
            -1.2494019e-01,
            4.406207e-02,
            -1.7381355e-01,
            1.5661912e-01,
            3.54275e-02,
            8.071299e-02,
            1.8640794e-01,
            2.791558e-02,
            2.0370029e-01,
            -2.545965e-03,
            1.45780025e-02,
            6.672354e-02,
            3.5037387e-02,
            -7.082848e-03,
            -8.0711305e-02,
            -1.3156593e-01,
            -1.612549e-02,
            2.8924763e-02,
            1.0621563e-01,
            -3.8281284e-02,
            1.4241667e-01,
            -1.4340502e-01,
            9.089071e-02,
            8.395382e-02,
            1.8114582e-01,
            1.3604967e-01,
            -8.0000255e-03,
            -1.786391e-01,
            -9.294665e-02,
            -9.6011214e-02,
            -1.655819e-01,
            -1.7118317e-01,
            1.6701409e-01,
            3.0809294e-02,
            2.64965e-03,
            3.378472e-02,
            1.4876898e-01,
            1.655425e-01,
        ],
    ),
    (
        "12.bias",
        &[
            1.6999465e-01,
            -7.953924e-02,
            -4.380472e-02,
            -1.0116867e-01,
            -6.66411e-02,
            1.8607113e-01,
            -1.8154868e-01,
            9.740501e-02,
        ],
    ),
    (
        "15.weight",
        &[
            -3.1776944e-01,
            -2.4678223e-01,
            -3.5167772e-01,
            1.4100122e-01,
            7.373785e-02,
            3.4981087e-01,
            -3.5169798e-01,
            1.9965762e-01,
            -1.6867784e-01,
            -2.2027135e-01,
            -3.3654112e-01,
            -2.0446292e-01,
            3.09745e-01,
            -2.0639096e-01,
            1.7035536e-01,
            -3.1581644e-02,
            -7.199613e-02,
            3.3903283e-01,
            -2.5093323e-01,
            8.1267804e-02,
            7.6639324e-02,
            1.776896e-01,
            -1.4545645e-01,
            -1.0000288e-01,
            2.449853e-02,
            -1.6658415e-01,
            -1.8625312e-01,
            1.11004174e-01,
            -1.8716256e-01,
            -2.6375252e-01,
            1.4002228e-01,
            -6.1315233e-03,
            -5.7416312e-02,
            -2.9401076e-01,
            -7.7857875e-04,
            5.6579743e-02,
            2.2705148e-01,
            2.4272627e-01,
            -2.0975922e-01,
            1.8222223e-01,
            -1.6161424e-01,
            2.429982e-01,
            1.5293245e-01,
            1.2508897e-01,
            1.9153971e-02,
            6.417793e-02,
            -1.0929588e-01,
            -2.9966128e-01,
            -1.5769757e-01,
            1.3716422e-01,
            2.0699202e-01,
            -1.7653586e-02,
            3.4168807e-01,
            2.0072605e-01,
            1.5839207e-03,
            -1.08032994e-01,
            -2.1504354e-01,
            2.4319701e-01,
            -3.04093e-01,
            2.3341164e-01,
            -3.9099716e-02,
            -3.0447245e-01,
            2.129118e-01,
            -3.408242e-01,
        ],
    ),
    (
        "15.bias",
        &[
            5.1033974e-02,
            -4.181941e-02,
            2.0028621e-01,
            2.8244334e-01,
            -3.0229992e-01,
            -1.5646937e-01,
            -5.683709e-02,
            -3.5255838e-03,
        ],
    ),
    (
        "18.weight",
        &[
            -8.0704935e-02,
            1.5810674e-02,
            -2.4639342e-02,
            -2.1085942e-02,
            -2.2229521e-01,
            -2.7590305e-01,
            -6.0400564e-02,
            1.8362753e-01,
            -1.3661264e-01,
            -3.3176267e-01,
            -2.0151685e-01,
            6.836943e-02,
            -9.105068e-03,
            3.5037898e-02,
            -2.468307e-01,
            -7.709514e-02,
            2.8431955e-01,
            -1.6188852e-01,
            -1.49616e-01,
            -2.0668983e-02,
            -9.0104185e-02,
            2.60291e-01,
            1.560957e-01,
            -3.2155448e-01,
            2.3348393e-01,
            2.7925783e-01,
            8.406256e-02,
            -8.119422e-03,
            -2.857949e-01,
            -3.5190004e-01,
            2.1278843e-01,
            -1.16147816e-01,
        ],
    ),
    (
        "18.bias",
        &[
            -1.18318506e-01,
            -1.420039e-01,
            -7.729888e-02,
            -2.6102054e-01,
        ],
    ),
];
const VGG_TINY_LOGITS: [f64; 8] = [
    -1.33673057e-01,
    -1.57822818e-01,
    -1.87630236e-01,
    -2.84010530e-01,
    -1.32109001e-01,
    -1.53807625e-01,
    -1.86208814e-01,
    -2.85103142e-01,
];
const VGG_TINY_CONV1_GRAD: [f64; 108] = [
    6.83790352e-03,
    7.15985801e-03,
    -1.90641906e-03,
    -2.03808909e-03,
    1.90590266e-02,
    3.86920548e-03,
    9.59987566e-03,
    -6.72674831e-03,
    -1.56336967e-02,
    -2.20761891e-03,
    -2.39308062e-03,
    -4.51806607e-03,
    -2.72940518e-03,
    -3.46688554e-03,
    2.06005182e-02,
    5.44820260e-03,
    1.59314610e-02,
    6.25213375e-03,
    2.27553817e-03,
    -1.77987921e-03,
    -1.47165544e-02,
    -4.91069444e-03,
    6.59511425e-03,
    -3.01206973e-03,
    -1.39790103e-02,
    -2.85172858e-03,
    -6.94842124e-03,
    -2.49519083e-03,
    2.83796806e-04,
    -2.24667997e-03,
    9.00204293e-03,
    7.46099977e-05,
    -7.81917572e-03,
    -9.42824781e-03,
    1.49930399e-02,
    -7.13419076e-03,
    -5.29192761e-03,
    1.96744385e-03,
    1.44161582e-02,
    -3.19176447e-03,
    2.63101398e-03,
    1.77834113e-03,
    6.79219607e-04,
    4.99758171e-04,
    7.20879156e-03,
    4.85581672e-03,
    -7.48479459e-03,
    4.97081317e-03,
    -7.62084918e-03,
    9.40532051e-03,
    -3.36808967e-03,
    -1.23877581e-02,
    -7.96627812e-03,
    1.94017659e-03,
    9.43025667e-03,
    -8.73526186e-03,
    3.11643304e-03,
    2.09544823e-02,
    7.42538646e-03,
    3.30614485e-03,
    1.33295432e-02,
    4.89228591e-03,
    1.08555518e-02,
    -2.50443444e-03,
    -1.47717632e-02,
    -3.01360269e-03,
    9.49193165e-03,
    -6.22550957e-03,
    2.69177668e-02,
    6.80161454e-03,
    1.34833856e-04,
    -4.28700261e-03,
    -7.35951588e-03,
    6.84530241e-03,
    1.05061121e-02,
    -9.25065577e-03,
    7.12522399e-03,
    -7.82894064e-03,
    -6.96860906e-03,
    -1.24332001e-02,
    -1.03243003e-02,
    -2.65025650e-03,
    1.54795181e-02,
    -2.65903363e-04,
    -5.30659966e-03,
    1.17898732e-02,
    3.93077871e-03,
    6.67799357e-03,
    -1.24709401e-02,
    2.00519920e-04,
    1.24819968e-02,
    -6.20251475e-03,
    9.11673810e-03,
    1.52563108e-02,
    -1.76247042e-02,
    -4.47541184e-04,
    3.61742335e-03,
    -1.54473144e-03,
    9.09620430e-03,
    -2.24402011e-03,
    -1.56167150e-03,
    5.44539467e-03,
    -5.79801016e-03,
    6.55097188e-03,
    9.98449977e-03,
    3.39663029e-03,
    2.12053629e-03,
    3.11839767e-03,
];

#[test]
#[ignore = "requires CUDA"]
fn vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new(
            "vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_channel",
        ),
        Axis::new(
            "vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_height",
        ),
        Axis::new(
            "vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_width",
        ),
        Axis::new(
            "vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_batch",
        ),
    );
    let input = Tensor::from_slice(
        &VGG_TINY_INPUT,
        [batch.of(2), channel.of(3), height.of(8), width.of(8)],
        &device,
    )?;
    let mut model = Sequential::new(vec![
        Box::new(Conv2d::new(channel, channel.of(4), [height, width], [3, 3]).padding([1, 1]))
            as Box<dyn Module>,
        Box::new(ReLU),
        Box::new(Conv2d::new(channel, channel.of(4), [height, width], [3, 3]).padding([1, 1])),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(channel, [height, width], [2, 2])),
        Box::new(Conv2d::new(channel, channel.of(6), [height, width], [3, 3]).padding([1, 1])),
        Box::new(ReLU),
        Box::new(Conv2d::new(channel, channel.of(6), [height, width], [3, 3]).padding([1, 1])),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(channel, [height, width], [2, 2])),
        Box::new(AdaptiveAvgPool2d::new([height, width], [2, 2])),
        Box::new(Flatten::new([channel, height, width], channel)),
        Box::new(Linear::new(channel, channel.of(8))),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5).expect("0.5 is a valid dropout probability")),
        Box::new(Linear::new(channel, channel.of(8))),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5).expect("0.5 is a valid dropout probability")),
        Box::new(Linear::new(channel, channel.of(4))),
    ]);
    model.build(input.shape(), &device, 0)?;
    for (path, values) in VGG_TINY_PARAMS {
        model.parameter(path)?.set_values(values)?;
    }
    let logits = model.forward(&input)?;
    close(
        "vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient logits",
        &logits.to_vec()?,
        &VGG_TINY_LOGITS,
    );
    logits.sum([batch, channel])?.backward()?;
    close(
        "vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient first conv weight gradient",
        &model.parameter("0.weight")?.grad().unwrap().to_vec()?,
        &VGG_TINY_CONV1_GRAD,
    );
    println!(
        "vgg16_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient: tiny-width equivalence PASS (batch_norm=False)"
    );
    Ok(())
}
const VGG_BN_TINY_INPUT: [f32; 384] = [
    -1.5255959e+00,
    -7.502318e-01,
    -6.539809e-01,
    -1.6094848e+00,
    -1.0016718e-01,
    -6.091889e-01,
    -9.7977227e-01,
    -1.6090963e+00,
    -7.121446e-01,
    3.03722e-01,
    -7.773143e-01,
    -2.5145525e-01,
    -2.2227049e-01,
    1.6871134e+00,
    2.2842517e-01,
    4.676355e-01,
    -6.969724e-01,
    -1.1607615e+00,
    6.995424e-01,
    1.9908163e-01,
    8.656924e-01,
    2.444039e-01,
    -6.6291136e-01,
    8.0730826e-01,
    1.1016806e+00,
    -1.7593604e-01,
    -2.2455578e+00,
    -1.446458e+00,
    6.1155282e-02,
    -6.1774445e-01,
    -7.9806983e-01,
    -1.3162321e-01,
    1.8793458e+00,
    -7.213178e-02,
    1.577706e-01,
    -7.734549e-01,
    1.990565e-01,
    4.5702778e-02,
    1.5295692e-01,
    -4.756788e-01,
    -1.1101983e-01,
    2.9273525e-01,
    -1.5784515e-01,
    -2.878714e-02,
    2.357111e+00,
    -1.0373387e+00,
    1.5747981e+00,
    -6.298472e-01,
    -9.273917e-01,
    5.451415e-01,
    6.628026e-02,
    -4.3704012e-01,
    7.626006e-01,
    4.4151092e-01,
    1.1651385e+00,
    2.0153918e+00,
    1.3741246e-01,
    9.386447e-01,
    -1.8600109e-01,
    -6.446393e-01,
    1.5392458e+00,
    -8.695876e-01,
    -3.3311536e+00,
    -7.4787223e-01,
    -2.5502462e-02,
    -1.0233306e+00,
    -5.9618515e-01,
    -1.0055307e+00,
    -2.106063e-01,
    -7.5475276e-03,
    1.6734272e+00,
    1.0342831e-02,
    -7.0395666e-01,
    -1.852658e-01,
    -9.962351e-01,
    -8.3125526e-01,
    -4.6102202e-01,
    -5.6008244e-01,
    3.9557618e-01,
    -9.822771e-01,
    -5.064866e-01,
    9.9775404e-02,
    -6.539734e-01,
    7.316937e-01,
    -1.4343859e+00,
    -5.0081307e-01,
    1.7163314e-01,
    -1.5999313e-01,
    2.546335e-01,
    -5.019573e-01,
    -1.0412e+00,
    7.322672e-01,
    -1.0483401e+00,
    -4.708772e-01,
    2.9113635e-01,
    1.9907043e+00,
    6.614453e-01,
    1.1899205e+00,
    8.165339e-01,
    -9.135236e-01,
    1.3851457e+00,
    -8.1384623e-01,
    -9.2757654e-01,
    1.1119633e+00,
    1.3352057e+00,
    6.042736e-01,
    -1.0344208e-01,
    -1.5121692e-01,
    -2.102083e+00,
    -6.2002194e-01,
    -1.478231e+00,
    -1.1334175e+00,
    8.737963e-01,
    -5.602594e-01,
    1.2857845e+00,
    8.1682384e-01,
    2.053041e-01,
    3.0510718e-01,
    5.35687e-01,
    -4.31185e-01,
    2.5581384e+00,
    -2.3336388e-01,
    -1.347213e-02,
    1.8606348e+00,
    -1.9804063e+00,
    1.7985829e+00,
    1.0181159e-01,
    3.40006e-01,
    7.1236455e-01,
    -1.7765073e+00,
    3.5386458e-01,
    1.1996132e+00,
    -3.029974e-01,
    -1.7618417e+00,
    6.348446e-01,
    -8.043591e-01,
    -1.6111118e+00,
    -1.8716129e+00,
    5.430836e-01,
    6.606786e-01,
    2.2952116e+00,
    6.7490596e-01,
    1.7133216e+00,
    -1.7942734e+00,
    -1.3632672e+00,
    -9.832196e-01,
    1.5112667e+00,
    6.4187074e-01,
    4.729638e-01,
    -4.285901e-01,
    5.5137074e-01,
    -1.5473709e+00,
    5.181121e-01,
    1.0653535e-01,
    2.6924077e-01,
    1.324768e+00,
    1.7460191e+00,
    1.854969e+00,
    -7.063691e-01,
    2.5570862e+00,
    4.175343e-01,
    -2.127186e-01,
    -8.39958e-01,
    -4.2001787e-01,
    -6.240363e-01,
    -9.772961e-01,
    8.748428e-01,
    9.872814e-01,
    3.0957633e-01,
    1.52069e+00,
    1.2052339e+00,
    -1.815591e+00,
    -4.0346155e-01,
    -9.591452e-01,
    -5.207704e-03,
    -7.886314e-02,
    8.436543e-01,
    1.1657013e+00,
    5.269322e-01,
    1.6192533e+00,
    -9.6397626e-01,
    1.4152038e-01,
    -1.6366096e-01,
    -3.5822257e-01,
    1.7222793e+00,
    -3.035756e-01,
    2.388742e-01,
    1.3440012e+00,
    1.0322569e-01,
    1.1003542e+00,
    -3.416802e-01,
    9.473389e-01,
    -5.6851596e-01,
    8.375962e-01,
    1.7836607e+00,
    -1.9542466e-01,
    5.149161e-01,
    -1.8474776e+00,
    -2.9167426e+00,
    -5.673299e-01,
    -5.412802e-01,
    8.95174e-01,
    -8.82507e-01,
    5.3181124e-01,
    -1.5457772e+00,
    -1.7329982e-01,
    7.2824633e-01,
    5.706102e-02,
    9.055172e-01,
    1.0462948e+00,
    -5.20597e-01,
    1.3547838e+00,
    2.3519313e-01,
    1.9142433e+00,
    1.8364111e+00,
    1.3245324e+00,
    -9.6900916e-01,
    1.2516364e+00,
    1.2103242e+00,
    -5.279206e-01,
    2.1856615e-01,
    -5.7430726e-01,
    1.4571251e+00,
    1.7709557e+00,
    1.6499138e+00,
    -4.3200457e-01,
    -2.710269e-01,
    -1.4391626e+00,
    1.2470404e+00,
    1.2738512e+00,
    3.9094925e-01,
    3.872105e-01,
    -7.982872e-02,
    3.417243e-01,
    9.488273e-01,
    -1.3839359e+00,
    1.7240863e+00,
    -2.3647652e+00,
    -9.294909e-01,
    2.936253e-01,
    2.1513203e-01,
    9.384637e-01,
    1.4657077e+00,
    -5.564744e-01,
    -7.448408e-01,
    -2.0215721e-01,
    -2.296679e-01,
    1.3313366e-03,
    3.752759e-01,
    -5.810679e-01,
    -5.7230884e-01,
    1.0097175e+00,
    -1.0564939e-01,
    -1.179696e+00,
    -9.0779595e-02,
    5.6311435e-01,
    -1.2560141e+00,
    8.95555e-01,
    1.6747737e-01,
    7.5142086e-01,
    2.4142299e+00,
    1.020584e+00,
    -4.404838e-01,
    -1.7341677e+00,
    -1.236225e+00,
    1.5785813e+00,
    -1.1160507e+00,
    7.6777023e-01,
    -5.8820677e-01,
    2.1188903e+00,
    -5.4219025e-01,
    -2.4592547e+00,
    -1.1108288e+00,
    -1.1187209e+00,
    7.5799555e-01,
    -4.9565765e-01,
    -1.9700006e-01,
    -3.3396218e-02,
    7.192915e-01,
    1.0644146e+00,
    8.3402544e-01,
    -1.9162164e+00,
    -3.4202927e-01,
    -6.6049206e-01,
    3.1508535e-01,
    1.1422518e+00,
    3.0550566e-01,
    -5.7888174e-01,
    -2.3828252e-01,
    -1.3541743e+00,
    2.686894e-01,
    1.1455697e-01,
    -1.5562972e+00,
    -1.0757437e+00,
    -8.751946e-01,
    -4.7281876e-01,
    9.912368e-01,
    -5.8622282e-02,
    1.1787646e+00,
    6.22185e-01,
    7.878501e-01,
    1.3685523e+00,
    -8.506898e-01,
    5.1260746e-01,
    1.0476325e+00,
    -3.1758463e-01,
    1.3948506e-01,
    2.3402624e+00,
    -6.1160916e-01,
    8.160271e-01,
    2.47723e-01,
    -3.867267e-01,
    1.9948451e-01,
    7.9926956e-01,
    -2.619034e-01,
    1.5132962e-01,
    1.1981666e+00,
    -2.2832582e+00,
    -1.0129594e+00,
    -8.878909e-01,
    6.5221924e-01,
    -8.726203e-01,
    3.5253752e-02,
    -3.365303e-01,
    1.402332e+00,
    4.841214e-01,
    -7.0304507e-01,
    -8.267661e-01,
    7.74396e-01,
    6.9199395e-01,
    -1.01848e+00,
    -8.0337167e-01,
    -7.071132e-01,
    7.521183e-01,
    -1.9208279e-02,
    1.103333e+00,
    -6.0679215e-01,
    -5.252235e-01,
    -5.6618774e-01,
    6.6039857e-04,
    7.224588e-01,
    1.526352e-01,
    1.4495978e-01,
    -2.3442194e+00,
    3.60003e-01,
    4.6668175e-01,
    1.2830665e+00,
    1.2678007e+00,
    1.9883296e-01,
    5.440877e-01,
    -3.9781693e-01,
    -1.9291055e+00,
    2.323687e-01,
    8.6146563e-01,
    6.217573e-01,
    -1.7811896e+00,
    -7.820609e-01,
    -1.42367e+00,
    1.6090765e+00,
    -3.2787595e-02,
    8.532334e-01,
    5.506365e-02,
    -1.7425371e+00,
    8.7500376e-01,
    -2.7188172e+00,
    -2.2192061e-01,
    3.4208494e-01,
    1.1093477e+00,
    -5.731476e-01,
    9.577846e-01,
    9.820232e-04,
    -1.3847686e+00,
    -9.965023e-01,
    8.0734813e-01,
    1.1738863e+00,
    -9.3984646e-01,
    1.3109189e+00,
    -3.1670693e-01,
    -1.861041e-01,
    -5.764602e-01,
    6.866518e-01,
    4.2086706e-01,
    -1.0213808e+00,
    9.8856664e-01,
];
const VGG_BN_TINY_PARAMS: &[(&str, &[f32])] = &[
    (
        "0.weight",
        &[
            -1.2436617e-01,
            9.2225045e-02,
            1.3273622e-01,
            8.826301e-02,
            1.2338974e-01,
            7.832153e-02,
            1.3600716e-01,
            -1.370142e-01,
            6.8348855e-02,
            -1.2701088e-01,
            -1.5155727e-01,
            -4.207093e-02,
            4.7625374e-02,
            7.0924535e-02,
            1.848137e-01,
            1.886668e-03,
            -3.815909e-02,
            8.175714e-02,
            -1.8926519e-01,
            -1.11202806e-01,
            -6.005795e-02,
            -1.5301888e-01,
            1.1063298e-01,
            -4.917526e-02,
            3.2658488e-02,
            1.6732323e-01,
            1.3545435e-02,
            -8.2995035e-02,
            1.4892732e-01,
            -1.1710221e-01,
            9.106907e-02,
            1.834119e-01,
            2.755964e-02,
            -7.525776e-02,
            -1.0407054e-01,
            1.6349323e-01,
            -9.0999626e-02,
            -1.5851837e-01,
            1.15700305e-01,
            1.6598444e-01,
            1.9117765e-01,
            1.7589514e-01,
            -9.523061e-02,
            1.5729645e-01,
            4.1159494e-03,
            3.664587e-02,
            -1.7010586e-01,
            -1.8431078e-01,
            1.5213518e-01,
            1.6204968e-01,
            6.447717e-02,
            6.4375006e-02,
            4.69039e-02,
            1.25671495e-02,
            1.0533938e-01,
            7.093493e-02,
            1.7250389e-01,
            1.04605034e-01,
            1.9076805e-01,
            -1.4115803e-01,
            1.1103866e-01,
            -6.1446525e-02,
            -2.0113477e-02,
            7.800787e-02,
            -9.811628e-02,
            -1.53532e-01,
            1.8513289e-01,
            -1.0442664e-01,
            1.3518758e-01,
            3.089355e-02,
            1.23637214e-01,
            3.0602783e-02,
            1.2380563e-03,
            -1.7162422e-02,
            -9.544833e-02,
            6.374511e-02,
            -1.5448302e-01,
            -1.9431507e-02,
            -1.4844531e-01,
            1.6606422e-01,
            -4.032714e-02,
            -9.45151e-02,
            1.07938275e-01,
            -1.0096215e-01,
            -1.7370778e-01,
            3.984057e-03,
            -1.0543876e-01,
            -1.514149e-01,
            -9.970881e-02,
            5.526854e-02,
            8.827586e-02,
            1.7806605e-01,
            -1.3633855e-01,
            6.942597e-02,
            -1.6482137e-01,
            8.0784366e-02,
            1.3162655e-01,
            -1.5503092e-01,
            1.4040579e-01,
            9.353984e-02,
            -4.043148e-02,
            9.2976525e-02,
            5.7680715e-02,
            6.927522e-02,
            -3.325094e-02,
            1.9790204e-02,
            -2.2124605e-02,
            -6.811978e-02,
        ],
    ),
    (
        "0.bias",
        &[1.15768075e-01, -1.6801554e-01, 2.678221e-02, 8.699509e-02],
    ),
    ("1.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("1.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "3.weight",
        &[
            1.1387412e-01,
            7.397684e-02,
            3.529201e-02,
            -8.873618e-02,
            -8.322847e-02,
            4.953001e-02,
            -2.3808877e-03,
            -1.5167817e-01,
            -1.1264827e-01,
            -2.5424223e-02,
            1.1795795e-01,
            9.685644e-02,
            6.247159e-02,
            -1.2901811e-01,
            -2.5266012e-02,
            1.5630546e-01,
            -1.323607e-01,
            1.0610139e-01,
            -1.3501696e-01,
            -9.7104155e-02,
            -1.3851053e-01,
            -1.4286836e-03,
            -1.23268746e-01,
            6.845381e-02,
            -1.498864e-01,
            -1.1076455e-01,
            -4.8920833e-02,
            7.0952177e-03,
            -8.082245e-02,
            1.4373165e-01,
            -3.691284e-02,
            -3.5636246e-02,
            -3.1812966e-02,
            9.292464e-02,
            1.9020438e-02,
            -4.5318406e-02,
            -9.86989e-02,
            -1.3098824e-01,
            -3.737841e-02,
            4.636693e-02,
            -7.047103e-02,
            -8.323618e-02,
            6.149602e-02,
            3.674271e-02,
            -1.4758116e-01,
            -5.6717396e-02,
            -5.5430632e-02,
            1.12568125e-01,
            1.02440596e-01,
            -9.621388e-02,
            1.2232566e-01,
            6.101237e-02,
            -3.8154982e-02,
            7.783115e-02,
            3.8281243e-02,
            7.584288e-02,
            -1.0515758e-01,
            -1.1889905e-01,
            -5.1239293e-02,
            1.4702617e-01,
            4.33925e-02,
            1.5489483e-01,
            1.4853346e-01,
            5.189755e-02,
            -1.24617994e-01,
            -6.888907e-02,
            4.2548995e-02,
            2.4740418e-02,
            1.1666505e-01,
            9.837924e-02,
            -1.13265e-01,
            -6.067741e-02,
            9.5835924e-02,
            5.6758127e-03,
            -9.275727e-02,
            1.22231565e-01,
            -1.6378526e-01,
            -7.3298514e-02,
            1.045025e-01,
            2.461356e-02,
            -1.3082436e-01,
            1.1129713e-01,
            7.612016e-02,
            1.07428074e-01,
            -1.09896086e-01,
            -1.2717228e-01,
            1.3013236e-01,
            1.609411e-01,
            -9.022187e-02,
            -8.8180326e-02,
            -1.2802621e-02,
            -3.7275493e-02,
            2.0810306e-02,
            1.9976735e-02,
            1.165701e-01,
            1.17350146e-01,
            -1.6480947e-01,
            1.3221245e-01,
            8.286308e-02,
            -3.384676e-02,
            -1.23561144e-01,
            -7.1409225e-02,
            5.154286e-02,
            1.5040496e-01,
            7.611418e-02,
            -1.0148847e-01,
            -3.857432e-02,
            1.3118255e-01,
            -5.458542e-02,
            -1.06402636e-01,
            1.6066477e-01,
            -5.9739273e-02,
            -6.6320404e-02,
            -7.347952e-02,
            3.372729e-02,
            5.977313e-03,
            -1.378802e-01,
            -5.7572465e-02,
            -4.300229e-02,
            -7.278465e-02,
            6.755191e-02,
            -3.8820207e-02,
            -2.3696027e-03,
            -6.740787e-02,
            1.1769354e-02,
            -9.481432e-02,
            1.6384141e-01,
            -3.3744574e-02,
            -6.44217e-02,
            5.2090865e-02,
            1.11941814e-01,
            -1.1362832e-03,
            -1.02063976e-01,
            1.3535818e-03,
            -1.2351036e-02,
            -5.8098853e-02,
            3.5445392e-02,
            -1.08894706e-01,
            1.6339426e-01,
            -1.5733306e-01,
            4.5348626e-02,
            -9.5177434e-02,
            7.318924e-02,
            -1.0080457e-01,
        ],
    ),
    (
        "3.bias",
        &[
            -1.42423315e-02,
            -5.3205054e-02,
            1.2379255e-01,
            1.3547616e-01,
        ],
    ),
    ("4.scale", &[1e+00, 1e+00, 1e+00, 1e+00]),
    ("4.bias", &[0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "7.weight",
        &[
            1.0617709e-01,
            1.1723914e-01,
            -3.8716458e-02,
            8.688025e-02,
            1.3956949e-01,
            1.9529542e-02,
            7.3763594e-02,
            1.3513431e-01,
            6.308626e-02,
            -1.1987506e-01,
            6.1920784e-02,
            -7.202868e-02,
            -1.3212726e-01,
            -1.16555475e-01,
            2.0258646e-02,
            -5.8167856e-02,
            -1.5894984e-01,
            -3.969103e-02,
            1.5746078e-01,
            -7.684004e-02,
            -9.950887e-02,
            1.3203642e-01,
            -2.7817369e-02,
            -3.9355457e-02,
            -1.4686835e-01,
            2.0063441e-02,
            1.21191464e-01,
            1.2706272e-01,
            1.3375978e-02,
            -1.1662893e-01,
            -5.4792803e-02,
            -1.5714435e-01,
            1.0374644e-01,
            1.0634647e-01,
            8.903744e-02,
            -5.0859295e-02,
            -1.1730258e-01,
            5.1184617e-02,
            3.6497734e-02,
            -6.966005e-02,
            1.5457593e-01,
            5.6084555e-02,
            5.9933346e-02,
            -6.9443665e-02,
            -1.5085405e-01,
            -3.300804e-02,
            4.024595e-02,
            1.2235292e-01,
            -1.3868593e-01,
            -7.389629e-02,
            7.831987e-02,
            -1.2416826e-01,
            1.17405795e-01,
            1.2660047e-02,
            -8.216574e-02,
            -1.2884447e-01,
            1.1074376e-01,
            -9.49835e-02,
            -5.930736e-02,
            -1.6140026e-01,
            -1.0586858e-01,
            7.796975e-02,
            4.640017e-02,
            1.2793897e-01,
            -5.445679e-03,
            8.826876e-02,
            -1.0244763e-01,
            -1.0620785e-01,
            -6.442388e-02,
            -1.3679647e-01,
            -1.00611076e-01,
            -5.0430816e-02,
            -7.78031e-02,
            -1.3321698e-01,
            1.1177799e-02,
            -4.1187547e-02,
            3.839507e-02,
            1.4884186e-01,
            1.638364e-01,
            -4.3829262e-02,
            -6.402658e-02,
            3.537814e-02,
            1.5384261e-01,
            -1.6104667e-01,
            -1.3613026e-02,
            -9.278365e-02,
            1.5690903e-01,
            -1.0000205e-01,
            1.0466313e-01,
            -3.8994454e-02,
            5.0804734e-02,
            -1.6203299e-01,
            1.60936e-01,
            3.885011e-02,
            1.6584688e-01,
            1.2925375e-01,
            -1.6544734e-01,
            6.716907e-02,
            -1.9422393e-02,
            -2.6792407e-02,
            -7.509814e-02,
            7.210015e-02,
            1.4818583e-01,
            -6.341096e-02,
            -1.1213291e-01,
            6.665395e-02,
            -2.085374e-02,
            -8.315136e-02,
            -3.6292713e-02,
            1.16744265e-01,
            9.804209e-02,
            -1.2789726e-01,
            3.437891e-02,
            -1.4431787e-01,
            -1.529573e-01,
            1.17319524e-01,
            8.5063696e-02,
            2.3000678e-02,
            1.3399544e-01,
            -1.7900865e-03,
            1.148599e-01,
            -1.0629511e-01,
            8.834605e-02,
            -1.5195422e-01,
            -1.5320686e-01,
            6.4385176e-02,
            -4.506564e-02,
            -9.716209e-02,
            -1.2540881e-01,
            1.0339874e-01,
            -8.185184e-02,
            -1.383662e-01,
            1.5135533e-01,
            -5.91595e-02,
            5.6514382e-02,
            1.6189502e-01,
            2.4176758e-02,
            -1.3946724e-01,
            -1.1320536e-01,
            9.762584e-02,
            -7.112505e-02,
            -5.9487045e-02,
            -9.633394e-02,
            -2.0980637e-03,
            -1.5473877e-01,
            -5.737166e-03,
            -1.4649211e-01,
            2.464503e-02,
            6.2260114e-02,
            -3.418128e-02,
            -1.2520896e-01,
            -5.954079e-02,
            -1.16379485e-01,
            -8.58684e-02,
            -1.1362493e-02,
            1.0712354e-01,
            7.219639e-02,
            1.0324679e-01,
            7.06511e-02,
            -1.06614195e-01,
            -4.3188e-02,
            -1.6148907e-01,
            -1.8691143e-02,
            1.5805772e-01,
            6.2396448e-02,
            -8.440932e-02,
            8.9384735e-02,
            -2.6509166e-02,
            -1.5440321e-01,
            1.2321627e-01,
            -1.4627782e-01,
            -7.939949e-02,
            1.19111955e-01,
            -1.291548e-01,
            3.0902823e-02,
            -1.2745093e-01,
            9.451278e-02,
            3.1636994e-02,
            1.6211477e-01,
            -2.4274867e-02,
            -1.0623634e-02,
            -3.461657e-02,
            1.2770668e-01,
            -1.9104522e-02,
            7.547353e-02,
            -8.51538e-02,
            6.02116e-02,
            -3.1968694e-02,
            1.2652135e-01,
            1.1686048e-02,
            -4.557888e-03,
            5.245344e-02,
            -8.478e-02,
            1.4223379e-01,
            9.4107114e-02,
            -1.2772185e-01,
            1.3626778e-01,
            -1.3942586e-01,
            -1.1898142e-01,
            -8.620139e-02,
            -2.616316e-02,
            -1.04710326e-01,
            1.3264047e-01,
            -1.3111019e-01,
            1.1359726e-01,
            -9.552171e-02,
            1.1167763e-01,
            -1.4428675e-01,
            -8.352941e-02,
            -9.4581805e-02,
            -8.020012e-02,
            2.6076238e-03,
            1.3047996e-01,
            -1.19790636e-01,
            7.5916275e-02,
            -1.4597285e-01,
        ],
    ),
    (
        "7.bias",
        &[
            1.0521128e-01,
            -2.3851812e-02,
            -9.2597745e-02,
            1.1678809e-01,
            -3.781279e-02,
            8.619841e-02,
        ],
    ),
    ("8.scale", &[1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00]),
    ("8.bias", &[0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "10.weight",
        &[
            -4.2184222e-02,
            2.2046585e-02,
            -2.9057486e-02,
            -1.3329151e-01,
            1.09777726e-01,
            1.24226436e-01,
            -1.1385355e-01,
            4.8972264e-02,
            -1.2456151e-01,
            2.9782413e-03,
            9.04258e-02,
            7.374394e-02,
            2.9887501e-02,
            -1.2837802e-01,
            4.0694155e-02,
            -5.5515572e-02,
            1.2760629e-01,
            -4.3165836e-02,
            8.4737554e-02,
            1.01960845e-01,
            -4.146173e-02,
            -4.5878436e-02,
            2.3336845e-02,
            8.03335e-02,
            -1.3268504e-01,
            2.145932e-02,
            6.252143e-02,
            1.0018221e-01,
            -1.1004773e-01,
            -1.3235909e-01,
            1.4165712e-02,
            -1.2676518e-01,
            8.676429e-02,
            2.2224788e-02,
            -1.17067e-01,
            1.06386855e-01,
            2.712281e-02,
            4.3588508e-02,
            -2.1635754e-02,
            6.2151056e-02,
            -7.0310935e-02,
            1.3209803e-01,
            1.5618957e-02,
            8.635998e-02,
            6.59974e-02,
            -3.803134e-02,
            1.12319455e-01,
            -6.179751e-03,
            -3.36748e-03,
            5.084831e-02,
            2.7188756e-02,
            -1.0797982e-01,
            7.8251995e-02,
            2.2396388e-02,
            -4.2300668e-02,
            1.11080736e-01,
            9.1388986e-02,
            2.7134443e-02,
            3.9828204e-02,
            -3.1931758e-02,
            -1.0564593e-01,
            2.813309e-03,
            -8.7760955e-02,
            -9.012037e-02,
            -1.0363099e-01,
            -8.6147584e-02,
            2.2195814e-02,
            -1.1289587e-01,
            -1.1947454e-02,
            1.68611e-02,
            -1.002256e-01,
            -6.1551966e-02,
            -9.1395244e-02,
            -3.0869473e-04,
            -4.534685e-02,
            -1.0620205e-01,
            1.1102303e-01,
            -9.0371845e-03,
            -1.1243806e-01,
            9.891176e-02,
            1.192401e-01,
            1.0259162e-01,
            -1.4125902e-02,
            -1.1133127e-01,
            -4.368977e-02,
            -6.545373e-02,
            -1.2214686e-01,
            -9.2701465e-02,
            -1.3549364e-01,
            -1.09984025e-01,
            -9.315631e-02,
            5.80435e-02,
            -7.648848e-02,
            1.1814136e-01,
            -1.6822817e-02,
            1.4934699e-02,
            4.1709412e-02,
            3.1369004e-02,
            1.5362385e-02,
            9.612931e-02,
            -3.267002e-02,
            -8.85456e-02,
            9.693196e-02,
            -2.20795e-02,
            1.2080553e-01,
            1.0287078e-01,
            2.0526764e-03,
            -7.402168e-02,
            1.05759405e-01,
            9.6711045e-04,
            9.2074975e-02,
            8.16555e-02,
            5.6951817e-02,
            -1.1768327e-03,
            -8.4347084e-02,
            5.7896473e-02,
            -1.2734056e-01,
            3.1497683e-02,
            5.236286e-02,
            -7.171602e-02,
            7.419182e-02,
            -9.3122534e-02,
            -9.567502e-02,
            -6.6324673e-03,
            -1.0412763e-01,
            1.2596485e-01,
            9.670222e-02,
            8.592003e-02,
            1.3194712e-01,
            -2.1721863e-03,
            -1.4915135e-03,
            -4.9125805e-02,
            1.5265375e-03,
            9.212819e-02,
            1.9832123e-02,
            -5.4123048e-02,
            -6.7438886e-02,
            -7.566612e-02,
            5.7411768e-02,
            7.636219e-02,
            -1.1655579e-01,
            -1.250868e-01,
            9.205024e-02,
            -3.625339e-02,
            7.700125e-02,
            1.3598384e-01,
            -4.2872924e-02,
            1.24035954e-01,
            6.738584e-02,
            -1.1102631e-01,
            1.0412341e-01,
            -5.9422557e-03,
            5.9493415e-02,
            7.0507764e-03,
            7.469334e-02,
            7.010414e-02,
            7.281423e-02,
            -9.609333e-02,
            -2.8455913e-02,
            6.4002335e-02,
            1.2371876e-01,
            1.169217e-01,
            5.808626e-02,
            6.213866e-03,
            -1.0452907e-01,
            4.9946285e-03,
            6.6844195e-02,
            -7.848049e-02,
            -5.16501e-02,
            -3.9480772e-02,
            -8.401967e-02,
            3.9106376e-02,
            -1.359864e-01,
            -5.0587956e-02,
            -7.58713e-02,
            4.8038814e-02,
            -7.049561e-02,
            -8.384477e-02,
            3.433776e-02,
            1.2700166e-01,
            -6.452839e-02,
            1.1511106e-01,
            -5.330333e-03,
            -1.0578608e-01,
            -9.337802e-02,
            -9.844148e-02,
            1.1196664e-01,
            -7.448267e-02,
            5.3329676e-02,
            1.0378249e-01,
            1.817261e-02,
            3.68566e-02,
            -3.2960124e-02,
            5.6985314e-03,
            6.533722e-02,
            5.1604774e-02,
            1.454559e-03,
            -1.1087549e-01,
            4.8672343e-03,
            1.3235903e-01,
            -6.610786e-02,
            6.719263e-02,
            1.3496977e-01,
            -1.2111986e-01,
            -1.2690876e-01,
            -1.0863449e-01,
            -2.2832816e-02,
            -1.3506973e-01,
            -9.297481e-02,
            2.0591004e-04,
            -6.0447645e-03,
            -2.3482878e-02,
            -1.0506759e-01,
            9.200283e-03,
            -1.3332862e-01,
            -7.766755e-02,
            -3.8388945e-02,
            1.321444e-01,
            1.0291169e-01,
            -1.269317e-01,
            -3.6480956e-02,
            7.370048e-02,
            -8.7805815e-02,
            9.50861e-02,
            1.3580436e-01,
            -1.5572902e-02,
            2.1928973e-02,
            -4.2091172e-02,
            1.0539108e-01,
            3.949672e-02,
            1.3226885e-01,
            -7.193922e-03,
            9.968662e-02,
            1.3407918e-03,
            -1.1017418e-01,
            -5.869265e-02,
            5.5613637e-02,
            -9.573696e-02,
            1.1033613e-01,
            -7.1045876e-02,
            -1.9452974e-02,
            -1.2685473e-02,
            -2.9017888e-02,
            -1.3475259e-01,
            1.04803614e-01,
            5.5921163e-02,
            -9.648766e-03,
            -1.2202876e-01,
            -3.3923325e-03,
            8.9565106e-04,
            4.295767e-02,
            1.2608379e-01,
            -1.2319625e-01,
            -1.2811764e-01,
            4.9096476e-02,
            1.2237913e-01,
            -3.888979e-02,
            -2.0457283e-02,
            8.227755e-02,
            -1.2750067e-01,
            -1.1795777e-01,
            1.2264115e-01,
            -2.8773742e-02,
            -8.3991714e-02,
            -4.329988e-02,
            1.3070372e-01,
            8.447568e-03,
            -8.28501e-02,
            -7.4419975e-02,
            -1.3272476e-01,
            6.1134905e-02,
            1.0763197e-01,
            -1.10615544e-01,
            -2.4173932e-02,
            2.6611062e-02,
            1.3040312e-01,
            7.595037e-02,
            5.3980112e-02,
            -1.3605927e-01,
            -1.1379831e-01,
            -9.463749e-02,
            5.735572e-02,
            1.1999622e-01,
            -3.4180466e-02,
            -4.548368e-03,
            -1.0627435e-01,
            4.6786107e-02,
            5.62003e-02,
            3.3660006e-02,
            1.17166154e-01,
            -4.3322105e-02,
            -3.7312042e-02,
            8.7883696e-02,
            2.3168683e-02,
            1.021622e-01,
            -3.594536e-02,
            -8.620303e-02,
            1.3211715e-01,
            1.1547538e-01,
            6.828428e-02,
            1.2567885e-01,
            9.44726e-02,
            2.2559015e-02,
            -3.5501953e-02,
            -1.3413163e-01,
            4.8324376e-02,
            2.697089e-02,
            -4.4290368e-03,
            5.543462e-03,
            7.873491e-03,
            1.1165179e-01,
            6.644592e-02,
            -2.7727418e-02,
            -6.401468e-02,
            2.0766303e-02,
            1.2802187e-01,
            -8.5486196e-02,
            8.5108295e-02,
            -3.7259463e-02,
            -3.7980255e-02,
            -7.616386e-02,
            2.425094e-02,
            -1.1111144e-02,
            1.9296544e-02,
        ],
    ),
    (
        "10.bias",
        &[
            1.1721328e-01,
            8.4619015e-02,
            -4.1657486e-03,
            -6.0780983e-02,
            9.02101e-03,
            1.0598283e-01,
        ],
    ),
    ("11.scale", &[1e+00, 1e+00, 1e+00, 1e+00, 1e+00, 1e+00]),
    ("11.bias", &[0e+00, 0e+00, 0e+00, 0e+00, 0e+00, 0e+00]),
    (
        "16.weight",
        &[
            1.2930827e-01,
            -2.5178622e-02,
            -7.399976e-02,
            -1.0673873e-01,
            1.4380093e-01,
            7.232888e-04,
            -1.1403564e-01,
            1.9628526e-01,
            -8.271609e-02,
            -1.2982205e-01,
            -1.4257167e-01,
            1.3874618e-02,
            -1.9356346e-02,
            1.0833552e-01,
            1.7641693e-01,
            -5.2633885e-02,
            -4.26153e-02,
            -3.8868453e-02,
            1.3769585e-01,
            -5.2212406e-02,
            1.7698968e-03,
            1.5971607e-02,
            -1.7171967e-01,
            1.2882912e-01,
            8.104553e-02,
            1.1620453e-01,
            1.4024316e-01,
            -3.7063563e-03,
            1.7149377e-01,
            -1.8235208e-01,
            -5.392013e-02,
            1.1261899e-01,
            7.69101e-02,
            1.01760976e-01,
            -1.6235135e-01,
            1.3453022e-01,
            -1.9271177e-01,
            -1.4557414e-01,
            -1.1249686e-01,
            1.9115873e-01,
            1.9437538e-01,
            -1.073723e-01,
            -1.1138473e-01,
            1.1240135e-01,
            2.0363316e-02,
            1.2065836e-02,
            -1.3128395e-01,
            -4.7507767e-02,
            9.9120036e-02,
            -5.0616346e-02,
            1.8268788e-01,
            -1.1594037e-01,
            1.8644056e-01,
            -1.6390678e-02,
            -1.0646503e-01,
            -1.214264e-01,
            6.50311e-02,
            -4.0836107e-02,
            5.9479944e-02,
            -1.0100281e-01,
            -1.02431946e-01,
            1.951136e-01,
            -2.0237e-02,
            -7.577107e-02,
            4.7176905e-02,
            -7.7654295e-02,
            1.4055489e-01,
            -1.4501043e-01,
            -3.7652336e-02,
            -4.7984924e-02,
            1.3808942e-02,
            -1.3887551e-01,
            -1.918356e-01,
            -1.1181808e-01,
            -1.5649319e-01,
            -1.7330924e-01,
            -1.22980755e-02,
            1.5621786e-01,
            3.457741e-02,
            7.0028636e-03,
            -7.5202934e-02,
            1.4988802e-01,
            9.212622e-02,
            1.309446e-01,
            6.72481e-02,
            -1.7215148e-01,
            1.04307204e-01,
            1.9680458e-01,
            2.1603739e-02,
            1.410966e-01,
            1.2993333e-01,
            -1.2613207e-01,
            7.715188e-02,
            -1.9400257e-01,
            -8.3376236e-02,
            1.9802837e-01,
            -1.4200152e-01,
            1.7040846e-01,
            -4.7360986e-02,
            2.667608e-03,
            4.6242207e-02,
            -1.8777331e-02,
            -1.2665015e-01,
            1.5418606e-01,
            -1.7884699e-01,
            -1.12086944e-01,
            -9.406283e-03,
            -2.773045e-04,
            8.05218e-02,
            -1.2924689e-01,
            -4.30087e-02,
            2.0198165e-02,
            1.3773951e-01,
            1.494773e-01,
            1.4884296e-01,
            3.4920122e-02,
            -8.758075e-02,
            -5.952749e-02,
            1.9235873e-01,
            4.4312268e-03,
            6.9423825e-02,
            -6.1896797e-02,
            5.566051e-02,
            -1.1942855e-02,
            8.587682e-02,
            2.2992281e-02,
            -2.49384e-02,
            1.3515483e-01,
            -1.6472122e-01,
            -4.7957256e-02,
            5.856987e-02,
            1.059276e-01,
            -6.4724125e-02,
            -1.7706166e-01,
            -1.7184448e-01,
            1.2120563e-01,
            1.2115185e-02,
            -5.4957174e-02,
            6.208691e-04,
            2.8033111e-02,
            1.2825865e-01,
            1.4026831e-01,
            1.6016789e-01,
            -2.025637e-02,
            1.2021335e-01,
            -9.925535e-03,
            -8.521916e-02,
            1.4637484e-01,
            -5.9907533e-02,
            -1.2388202e-01,
            4.9666904e-02,
            -5.1626284e-02,
            1.17590785e-01,
            -8.9385666e-02,
            -1.2494019e-01,
            4.406207e-02,
            -1.7381355e-01,
            1.5661912e-01,
            3.54275e-02,
            8.071299e-02,
            1.8640794e-01,
            2.791558e-02,
            2.0370029e-01,
            -2.545965e-03,
            1.45780025e-02,
            6.672354e-02,
            3.5037387e-02,
            -7.082848e-03,
            -8.0711305e-02,
            -1.3156593e-01,
            -1.612549e-02,
            2.8924763e-02,
            1.0621563e-01,
            -3.8281284e-02,
            1.4241667e-01,
            -1.4340502e-01,
            9.089071e-02,
            8.395382e-02,
            1.8114582e-01,
            1.3604967e-01,
            -8.0000255e-03,
            -1.786391e-01,
            -9.294665e-02,
            -9.6011214e-02,
            -1.655819e-01,
            -1.7118317e-01,
            1.6701409e-01,
            3.0809294e-02,
            2.64965e-03,
            3.378472e-02,
            1.4876898e-01,
            1.655425e-01,
        ],
    ),
    (
        "16.bias",
        &[
            1.6999465e-01,
            -7.953924e-02,
            -4.380472e-02,
            -1.0116867e-01,
            -6.66411e-02,
            1.8607113e-01,
            -1.8154868e-01,
            9.740501e-02,
        ],
    ),
    (
        "19.weight",
        &[
            -3.1776944e-01,
            -2.4678223e-01,
            -3.5167772e-01,
            1.4100122e-01,
            7.373785e-02,
            3.4981087e-01,
            -3.5169798e-01,
            1.9965762e-01,
            -1.6867784e-01,
            -2.2027135e-01,
            -3.3654112e-01,
            -2.0446292e-01,
            3.09745e-01,
            -2.0639096e-01,
            1.7035536e-01,
            -3.1581644e-02,
            -7.199613e-02,
            3.3903283e-01,
            -2.5093323e-01,
            8.1267804e-02,
            7.6639324e-02,
            1.776896e-01,
            -1.4545645e-01,
            -1.0000288e-01,
            2.449853e-02,
            -1.6658415e-01,
            -1.8625312e-01,
            1.11004174e-01,
            -1.8716256e-01,
            -2.6375252e-01,
            1.4002228e-01,
            -6.1315233e-03,
            -5.7416312e-02,
            -2.9401076e-01,
            -7.7857875e-04,
            5.6579743e-02,
            2.2705148e-01,
            2.4272627e-01,
            -2.0975922e-01,
            1.8222223e-01,
            -1.6161424e-01,
            2.429982e-01,
            1.5293245e-01,
            1.2508897e-01,
            1.9153971e-02,
            6.417793e-02,
            -1.0929588e-01,
            -2.9966128e-01,
            -1.5769757e-01,
            1.3716422e-01,
            2.0699202e-01,
            -1.7653586e-02,
            3.4168807e-01,
            2.0072605e-01,
            1.5839207e-03,
            -1.08032994e-01,
            -2.1504354e-01,
            2.4319701e-01,
            -3.04093e-01,
            2.3341164e-01,
            -3.9099716e-02,
            -3.0447245e-01,
            2.129118e-01,
            -3.408242e-01,
        ],
    ),
    (
        "19.bias",
        &[
            5.1033974e-02,
            -4.181941e-02,
            2.0028621e-01,
            2.8244334e-01,
            -3.0229992e-01,
            -1.5646937e-01,
            -5.683709e-02,
            -3.5255838e-03,
        ],
    ),
    (
        "22.weight",
        &[
            -8.0704935e-02,
            1.5810674e-02,
            -2.4639342e-02,
            -2.1085942e-02,
            -2.2229521e-01,
            -2.7590305e-01,
            -6.0400564e-02,
            1.8362753e-01,
            -1.3661264e-01,
            -3.3176267e-01,
            -2.0151685e-01,
            6.836943e-02,
            -9.105068e-03,
            3.5037898e-02,
            -2.468307e-01,
            -7.709514e-02,
            2.8431955e-01,
            -1.6188852e-01,
            -1.49616e-01,
            -2.0668983e-02,
            -9.0104185e-02,
            2.60291e-01,
            1.560957e-01,
            -3.2155448e-01,
            2.3348393e-01,
            2.7925783e-01,
            8.406256e-02,
            -8.119422e-03,
            -2.857949e-01,
            -3.5190004e-01,
            2.1278843e-01,
            -1.16147816e-01,
        ],
    ),
    (
        "22.bias",
        &[
            -1.18318506e-01,
            -1.420039e-01,
            -7.729888e-02,
            -2.6102054e-01,
        ],
    ),
];
const VGG_BN_TINY_LOGITS: [f64; 8] = [
    -1.33673146e-01,
    -1.57823026e-01,
    -1.87630296e-01,
    -2.84010470e-01,
    -1.32109106e-01,
    -1.53807908e-01,
    -1.86208889e-01,
    -2.85103083e-01,
];
const VGG_BN_TINY_CONV1_GRAD: [f64; 108] = [
    6.83776708e-03,
    7.15971459e-03,
    -1.90638017e-03,
    -2.03805044e-03,
    1.90586485e-02,
    3.86912841e-03,
    9.59968474e-03,
    -6.72661234e-03,
    -1.56333856e-02,
    -2.20757560e-03,
    -2.39303079e-03,
    -4.51797619e-03,
    -2.72934977e-03,
    -3.46681755e-03,
    2.06001103e-02,
    5.44809271e-03,
    1.59311406e-02,
    6.25200709e-03,
    2.27548974e-03,
    -1.77984126e-03,
    -1.47162601e-02,
    -4.91059572e-03,
    6.59498479e-03,
    -3.01200850e-03,
    -1.39787327e-02,
    -2.85167294e-03,
    -6.94828015e-03,
    -2.49514147e-03,
    2.83790869e-04,
    -2.24663410e-03,
    9.00186133e-03,
    7.46087171e-05,
    -7.81901740e-03,
    -9.42805968e-03,
    1.49927400e-02,
    -7.13405013e-03,
    -5.29182283e-03,
    1.96740264e-03,
    1.44158695e-02,
    -3.19170114e-03,
    2.63096113e-03,
    1.77830574e-03,
    6.79205637e-04,
    4.99748858e-04,
    7.20864441e-03,
    4.85571846e-03,
    -7.48464465e-03,
    4.97071538e-03,
    -7.62069738e-03,
    9.40513052e-03,
    -3.36802215e-03,
    -1.23875113e-02,
    -7.96611700e-03,
    1.94013712e-03,
    9.43006668e-03,
    -8.73508863e-03,
    3.11637251e-03,
    2.09540632e-02,
    7.42523931e-03,
    3.30608059e-03,
    1.33292768e-02,
    4.89218673e-03,
    1.08553320e-02,
    -2.50438368e-03,
    -1.47714652e-02,
    -3.01354332e-03,
    9.49174259e-03,
    -6.22538384e-03,
    2.69172247e-02,
    6.80147856e-03,
    1.34831062e-04,
    -4.28691600e-03,
    -7.35936733e-03,
    6.84516365e-03,
    1.05058998e-02,
    -9.25046764e-03,
    7.12508103e-03,
    -7.82878418e-03,
    -6.96847169e-03,
    -1.24329505e-02,
    -1.03240926e-02,
    -2.65020388e-03,
    1.54792070e-02,
    -2.65897484e-04,
    -5.30649349e-03,
    1.17896358e-02,
    3.93070094e-03,
    6.67785946e-03,
    -1.24706924e-02,
    2.00516544e-04,
    1.24817472e-02,
    -6.20239275e-03,
    9.11655836e-03,
    1.52560063e-02,
    -1.76243447e-02,
    -4.47532628e-04,
    3.61735118e-03,
    -1.54469884e-03,
    9.09601990e-03,
    -2.24397564e-03,
    -1.56164076e-03,
    5.44528756e-03,
    -5.79789747e-03,
    6.55083917e-03,
    9.98430327e-03,
    3.39656207e-03,
    2.12049345e-03,
    3.11833527e-03,
];

#[test]
#[ignore = "requires CUDA"]
fn vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient() -> Result<()>
{
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new(
            "vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_channel",
        ),
        Axis::new(
            "vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_height",
        ),
        Axis::new(
            "vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_width",
        ),
        Axis::new(
            "vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient_batch",
        ),
    );
    let input = Tensor::from_slice(
        &VGG_BN_TINY_INPUT,
        [batch.of(2), channel.of(3), height.of(8), width.of(8)],
        &device,
    )?;
    let mut model = Sequential::new(vec![
        Box::new(Conv2d::new(channel, channel.of(4), [height, width], [3, 3]).padding([1, 1]))
            as Box<dyn Module>,
        Box::new(BatchNorm::new(channel)),
        Box::new(ReLU),
        Box::new(Conv2d::new(channel, channel.of(4), [height, width], [3, 3]).padding([1, 1])),
        Box::new(BatchNorm::new(channel)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(channel, [height, width], [2, 2])),
        Box::new(Conv2d::new(channel, channel.of(6), [height, width], [3, 3]).padding([1, 1])),
        Box::new(BatchNorm::new(channel)),
        Box::new(ReLU),
        Box::new(Conv2d::new(channel, channel.of(6), [height, width], [3, 3]).padding([1, 1])),
        Box::new(BatchNorm::new(channel)),
        Box::new(ReLU),
        Box::new(MaxPool2d::new(channel, [height, width], [2, 2])),
        Box::new(AdaptiveAvgPool2d::new([height, width], [2, 2])),
        Box::new(Flatten::new([channel, height, width], channel)),
        Box::new(Linear::new(channel, channel.of(8))),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5).expect("0.5 is a valid dropout probability")),
        Box::new(Linear::new(channel, channel.of(8))),
        Box::new(ReLU),
        Box::new(Dropout::new(0.5).expect("0.5 is a valid dropout probability")),
        Box::new(Linear::new(channel, channel.of(4))),
    ]);
    model.build(input.shape(), &device, 0)?;
    for (path, values) in VGG_BN_TINY_PARAMS {
        model.parameter(path)?.set_values(values)?;
    }
    let logits = model.forward(&input)?;
    close(
        "vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient logits",
        &logits.to_vec()?,
        &VGG_BN_TINY_LOGITS,
    );
    logits.sum([batch, channel])?.backward()?;
    close(
        "vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient first conv weight gradient",
        &model.parameter("0.weight")?.grad().unwrap().to_vec()?,
        &VGG_BN_TINY_CONV1_GRAD,
    );
    println!(
        "vgg16_bn_tiny_width_matches_a_hand_rolled_reference_logits_and_first_conv_gradient: tiny-width equivalence PASS (batch_norm=True)"
    );
    Ok(())
}
#[test]
#[ignore = "requires CUDA"]
fn resnet34_parameter_count_matches_torchvision_at_real_size() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new("resnet34_count_channel"),
        Axis::new("resnet34_count_height"),
        Axis::new("resnet34_count_width"),
        Axis::new("resnet34_count_batch"),
    );
    let axes = ResNetAxes {
        channel,
        height,
        width,
    };
    let mut model = resnet34(axes, 1000);
    let shape = Shape::new([batch.of(1), channel.of(3), height.of(224), width.of(224)])?;
    model.build(&shape, &device, 0)?;
    let count: usize = model
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    assert_eq!(
        count, 21_797_672,
        "resnet34 parameter count vs torchvision.models.resnet34(weights=None)"
    );
    println!("resnet34 parameter count PASS: {count} == torchvision's 21797672");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn resnet50_parameter_count_matches_torchvision_at_real_size() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new("resnet50_count_channel"),
        Axis::new("resnet50_count_height"),
        Axis::new("resnet50_count_width"),
        Axis::new("resnet50_count_batch"),
    );
    let axes = ResNetAxes {
        channel,
        height,
        width,
    };
    let mut model = resnet50(axes, 1000);
    let shape = Shape::new([batch.of(1), channel.of(3), height.of(224), width.of(224)])?;
    model.build(&shape, &device, 0)?;
    let count: usize = model
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    assert_eq!(
        count, 25_557_032,
        "resnet50 parameter count vs torchvision.models.resnet50(weights=None)"
    );
    println!("resnet50 parameter count PASS: {count} == torchvision's 25557032");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn vgg16_parameter_count_matches_torchvision_at_real_size() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new("vgg16_count_channel"),
        Axis::new("vgg16_count_height"),
        Axis::new("vgg16_count_width"),
        Axis::new("vgg16_count_batch"),
    );
    let axes = VggAxes {
        channel,
        height,
        width,
    };
    let mut model = vgg16(axes, 1000);
    let shape = Shape::new([batch.of(1), channel.of(3), height.of(224), width.of(224)])?;
    model.build(&shape, &device, 0)?;
    let count: usize = model
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    assert_eq!(
        count, 138_357_544,
        "vgg16 parameter count vs torchvision.models.vgg16(weights=None)"
    );
    println!("vgg16 parameter count PASS: {count} == torchvision's 138357544");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn vgg16_bn_parameter_count_matches_torchvision_at_real_size() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new("vgg16_bn_count_channel"),
        Axis::new("vgg16_bn_count_height"),
        Axis::new("vgg16_bn_count_width"),
        Axis::new("vgg16_bn_count_batch"),
    );
    let axes = VggAxes {
        channel,
        height,
        width,
    };
    let mut model = vgg16_bn(axes, 1000);
    let shape = Shape::new([batch.of(1), channel.of(3), height.of(224), width.of(224)])?;
    model.build(&shape, &device, 0)?;
    let count: usize = model
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    assert_eq!(
        count, 138_365_992,
        "vgg16_bn parameter count vs torchvision.models.vgg16_bn(weights=None)"
    );
    println!("vgg16_bn parameter count PASS: {count} == torchvision's 138365992");
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn resnet34_training_step_updates_batch_norm_running_statistics() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new("resnet34_step_channel"),
        Axis::new("resnet34_step_height"),
        Axis::new("resnet34_step_width"),
        Axis::new("resnet34_step_batch"),
    );
    let axes = ResNetAxes {
        channel,
        height,
        width,
    };
    let mut model = resnet34_small_input(axes, 4);
    let shape = Shape::new([batch.of(2), channel.of(3), height.of(16), width.of(16)])?;
    model.build(&shape, &device, 0)?;
    let stem_running_mean_before = model.state("1.running_mean")?.tensor().to_vec()?;
    assert_eq!(
        stem_running_mean_before,
        vec![0.0; 64],
        "BatchNorm running_mean starts at zero"
    );

    let input = Tensor::from_slice(
        &(0..2 * 3 * 16 * 16)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect::<Vec<_>>(),
        [batch.of(2), channel.of(3), height.of(16), width.of(16)],
        &device,
    )?;
    let mut trainer = Trainer::new(SGD::new(0.01)?).with_seed(1);
    let step = trainer.step_training(&mut model, |m, pass| {
        m.forward_training(&input, pass)?.mean([batch, channel])
    })?;
    assert!(step.pre_update_loss()?.is_finite());
    assert!(
        step.committed_states() > 0,
        "step_training must stage and commit at least one BatchNorm running-statistics update"
    );
    let stem_running_mean_after = model.state("1.running_mean")?.tensor().to_vec()?;
    assert_ne!(
        stem_running_mean_after, stem_running_mean_before,
        "one step_training call must update the stem BatchNorm's running_mean"
    );
    println!(
        "resnet34 training step PASS: committed {} state updates",
        step.committed_states()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn vgg16_bn_training_step_updates_batch_norm_running_statistics() -> Result<()> {
    let device = Device::cuda(0)?;
    let (channel, height, width, batch) = (
        Axis::new("vgg16_bn_step_channel"),
        Axis::new("vgg16_bn_step_height"),
        Axis::new("vgg16_bn_step_width"),
        Axis::new("vgg16_bn_step_batch"),
    );
    let axes = VggAxes {
        channel,
        height,
        width,
    };
    let mut model = vgg16_bn(axes, 4);
    // vgg16_bn's five 2x2 stride-2 max pools need at least 32x32 to survive
    // to a nonempty spatial extent; 8x8 is fine for the tiny-width oracle
    // test above (which builds a 2-pool mini VGG by hand), but real vgg16_bn
    // needs real headroom.
    let shape = Shape::new([batch.of(2), channel.of(3), height.of(32), width.of(32)])?;
    model.build(&shape, &device, 0)?;
    let running_mean_before = model.state("1.running_mean")?.tensor().to_vec()?;
    assert_eq!(
        running_mean_before,
        vec![0.0; 64],
        "BatchNorm running_mean starts at zero"
    );

    let input = Tensor::from_slice(
        &(0..2 * 3 * 32 * 32)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
            .collect::<Vec<_>>(),
        [batch.of(2), channel.of(3), height.of(32), width.of(32)],
        &device,
    )?;
    let mut trainer = Trainer::new(SGD::new(0.01)?).with_seed(1);
    let step = trainer.step_training(&mut model, |m, pass| {
        m.forward_training(&input, pass)?.mean([batch, channel])
    })?;
    assert!(step.pre_update_loss()?.is_finite());
    assert!(
        step.committed_states() > 0,
        "step_training must stage and commit at least one BatchNorm running-statistics update"
    );
    let running_mean_after = model.state("1.running_mean")?.tensor().to_vec()?;
    assert_ne!(
        running_mean_after, running_mean_before,
        "one step_training call must update the first BatchNorm's running_mean"
    );
    println!(
        "vgg16_bn training step PASS: committed {} state updates",
        step.committed_states()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dcgan_generator_and_discriminator_parameter_counts_match_pytorch_tutorial_reference()
-> Result<()> {
    // Independently confirmed against the PyTorch DCGAN tutorial's own `Generator`/
    // `Discriminator` classes (`bias=False` throughout, nz=100, ngf=ndf=64, nc=3):
    // `sum(p.numel() for p in netG.parameters())` == 3_576_704,
    // `sum(p.numel() for p in netD.parameters())` == 2_765_568.
    let device = Device::cuda(0)?;
    let (batch, channel, height, width) = (
        Axis::new("dcgan_count_batch"),
        Axis::new("dcgan_count_channel"),
        Axis::new("dcgan_count_height"),
        Axis::new("dcgan_count_width"),
    );

    let mut generator = DcganGenerator::new(channel, [height, width]);
    let latent_shape = Shape::new([
        batch.of(1),
        channel.of(DCGAN_LATENT),
        height.of(1),
        width.of(1),
    ])?;
    let image_shape = generator.build(&latent_shape, &device, 1)?;
    assert_eq!(
        image_shape,
        Shape::new([
            batch.of(1),
            height.of(64),
            width.of(64),
            channel.of(DCGAN_IMAGE_CHANNELS)
        ])?
    );
    let generator_params: usize = generator
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    assert_eq!(generator_params, 3_576_704);

    let mut discriminator = DcganDiscriminator::new(channel, [height, width]);
    discriminator.build(&image_shape, &device, 2)?;
    let discriminator_params: usize = discriminator
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.tensor().shape().len())
        .sum();
    assert_eq!(discriminator_params, 2_765_568);

    println!(
        "check DCGAN parameter counts: PASS generator={generator_params} discriminator={discriminator_params}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dcgan_tutorial_init_draws_normal_weights_and_zero_bias() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, channel, height, width) = (
        Axis::new("dcgan_init_batch"),
        Axis::new("dcgan_init_channel"),
        Axis::new("dcgan_init_height"),
        Axis::new("dcgan_init_width"),
    );
    const NGF: usize = 8;
    let mut generator =
        DcganGenerator::with_features(channel, [height, width], DCGAN_IMAGE_CHANNELS, NGF);
    let latent_shape = Shape::new([batch.of(1), channel.of(100), height.of(1), width.of(1)])?;
    generator.build(&latent_shape, &device, 7)?;

    // Axis's own default convolution init is Xavier/Glorot-uniform-shaped; record it first so
    // the assertion below is a genuine before/after change, not a coincidence.
    let before = generator.parameter("0.weight")?.tensor().to_vec()?;

    crate::architectures::dcgan_tutorial_init(&generator, &device, 1234)?;

    let after = generator.parameter("0.weight")?.tensor().to_vec()?;
    assert_ne!(
        before, after,
        "tutorial init must overwrite the default init"
    );

    let mut weight_values = Vec::new();
    let mut scale_values = Vec::new();
    for (path, parameter) in generator.named_parameters() {
        let values = parameter.tensor().to_vec()?;
        if path.ends_with(".bias") {
            assert!(
                values.iter().all(|&v| v == 0.0),
                "{path} must be exactly zero after tutorial init"
            );
        } else if path.ends_with(".scale") {
            scale_values.extend(values);
        } else if path.ends_with(".weight") {
            weight_values.extend(values);
        }
    }

    let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
    let std = |values: &[f32], mean: f32| {
        (values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt()
    };
    let weight_mean = mean(&weight_values);
    let weight_std = std(&weight_values, weight_mean);
    assert!(
        weight_mean.abs() < 0.002,
        "conv weight mean should be near 0, got {weight_mean}"
    );
    assert!(
        (weight_std - 0.02).abs() < 0.002,
        "conv weight std should be near 0.02, got {weight_std}"
    );

    let scale_mean = mean(&scale_values);
    let scale_std = std(&scale_values, scale_mean);
    assert!(
        (scale_mean - 1.0).abs() < 0.005,
        "BatchNorm scale mean should be near 1.0, got {scale_mean}"
    );
    assert!(
        (scale_std - 0.02).abs() < 0.005,
        "BatchNorm scale std should be near 0.02, got {scale_std}"
    );

    println!(
        "check dcgan_tutorial_init: PASS weight_mean={weight_mean:.4} weight_std={weight_std:.4} scale_mean={scale_mean:.4} scale_std={scale_std:.4}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dcgan_matches_pytorch_reference_at_tiny_width() -> Result<()> {
    // Oracle from a one-off PyTorch script building the IDENTICAL five-layer DCGAN topology
    // this file's own `DcganGenerator`/`DcganDiscriminator` build (full 64x64 spatial pipeline;
    // only nz/ngf/ndf/nc shrink to 2/1/1/1, batch 1), with layer order matching exactly. Weight
    // arrays are PyTorch's own `[out,in,kh,kw]` (Conv2d) / `[in,out,kh,kw]` (ConvTranspose2d),
    // permuted to Axis's `[patch(other_side, kh, kw), self_side]` flat layout via
    // `weight.permute(1, 2, 3, 0).flatten()` -- the same index formulas
    // `conv2d_lowers_valid_patches_...` and `conv_transpose2d_matches_scalar_forward_...` above
    // assert directly. The oracle-generating script is kept out of this repo, per its own
    // convention for other baked-literal oracles; it is reproducible from this comment's
    // description and `docs/design/library.md`'s convolution weight-layout contract.
    const NZ: usize = 2;
    const NGF: usize = 1;
    const NDF: usize = 1;
    const NC: usize = 1;

    const Z: &[f32] = &[-0.67533, 0.220349];
    const G_W1: &[f32] = &[
        -0.000662, -0.030921, 0.047415, -0.072445, -0.072748, -0.018802, -0.065048, 0.018894,
        -0.034043, -0.057582, 0.023702, -0.004536, -0.001751, 0.063273, 0.070082, -0.009086,
        -0.007844, 0.002457, 0.023389, -0.007625, -0.026712, 0.017888, -0.017374, 0.056201,
        -0.084442, 0.083726, -0.058538, 0.056132, -0.036436, 0.083917, 0.003274, -0.006392,
        0.034943, -0.079401, 0.053035, -0.041903, -0.059922, 0.060184, -0.03849, -0.000573,
        0.032104, -0.043933, 0.073397, -0.067733, -0.01819, -0.082718, 0.066142, -0.0746,
        -0.014247, -0.017928, 0.009353, 0.048473, 0.080034, 0.047786, -0.081995, -0.085245,
        -0.055644, 0.055135, -0.022377, -0.069165, -0.034454, -0.018686, 0.076368, -0.035839,
        -0.057292, -0.017025, -0.040688, -0.017354, -0.061752, -0.079315, -0.082781, -0.076318,
        -0.051596, -0.013831, 0.075978, 0.001143, 0.039441, -0.040153, 0.042839, 0.033296,
        0.004648, -0.079555, -0.045315, -0.005965, 0.014954, 0.07773, -0.082528, -0.036053,
        -0.063866, 0.079815, -0.045567, 0.03201, 0.055768, -0.079767, 0.051824, 0.055923, -0.0392,
        -0.0102, -0.003189, -0.039457, 0.05653, 0.07068, 0.08787, -0.071426, 0.03508, 0.009485,
        0.011941, -0.018506, 0.059263, 0.063119, -0.052043, 0.024673, 0.016471, 0.042471,
        -0.068528, 0.031215, -0.061261, -0.021255, -0.04566, -0.018589, 0.039993, -0.072839,
        0.035546, 0.047893, -0.052357, 0.070178, 0.026703, 0.060478, 0.048523, -0.062347,
        -0.011156, 0.003942, 0.003375, -0.062308, 0.02048, -0.048656, 0.054834, -0.051504, 0.08487,
        0.030206, -0.068114, -0.052672, -0.032392, -0.001928, 0.034737, 0.003718, 0.073234,
        0.056977, 0.076916, -0.066815, 0.07799, -0.06068, 0.017591, -0.051324, -0.076861, 0.061866,
        0.008131, -0.031773, -0.055296, 0.074555, -0.082374, 0.031962, 0.078532, 0.011192,
        0.067207, -0.000658, -0.08817, -0.017473, 0.016544, 0.01109, -0.01489, -0.020183,
        -0.014545, -0.000621, -0.04046, 0.011278, 0.03399, -0.069138, -0.052353, -0.046327,
        0.032402, 0.071373, 0.044699, -0.071731, 0.063275, -0.006347, 0.033049, 0.087437,
        -0.087481, 0.031929, -0.057337, 0.002503, 0.044134, -0.076598, 0.0185, 0.043786, -0.06895,
        -0.062957, -0.050896, -0.02509, 0.083151, -0.029656, 0.059558, -0.013089, -0.03854,
        0.000967, -0.022246, 0.072903, -0.084199, 0.011034, -0.001589, 0.079169, -0.066562,
        0.054068, -0.068179, -0.05588, -0.00487, 0.039643, 0.013271, -0.062481, -0.036198,
        -0.037461, 0.052448, 0.025997, -0.053788, 0.029185, 0.080201, 0.066311, 0.060573,
        -0.028454, -0.074536, 0.000141, -0.021998, 0.045504, 0.003988, -0.08548, 0.012896,
        0.063903, 0.020963, -0.07309, 0.034686, 0.001218, 0.005294, -0.015028, -0.043127,
        -0.046552, 0.041824, 0.011682, -0.084786, 0.07309, -0.052388, -0.025838, -0.022126,
        -0.052476, -0.043055, -0.032689,
    ];
    const G_W2: &[f32] = &[
        -0.123894, 0.010441, 0.041352, 0.0057, -0.034682, -0.041952, -0.074014, -0.079244,
        0.056424, 0.015608, 0.12496, -0.112715, -0.046599, -0.115921, 0.017933, -0.013517,
        -0.060033, -0.098268, 0.064833, 0.103666, 0.031469, 0.051241, 0.039899, 0.016079,
        -0.083418, 0.009824, 0.077706, 0.067305, 0.044337, 0.121684, 0.00885, 0.106493, -0.072013,
        0.086559, -0.043749, 0.124249, -0.061071, -0.035585, -0.081044, -0.05964, 0.07187,
        0.112639, 0.059978, 0.063152, 0.01105, -0.103503, 0.119541, 0.08008, 0.066197, 0.073469,
        0.014363, -0.082509, 0.072442, -0.113382, -0.072687, -0.015879, 0.09594, 0.01675,
        -0.029852, 0.104323, -0.012437, 0.031325, 0.102813, -0.059366, 0.04534, 0.058377,
        -0.070464, 0.006718, 0.038042, -0.009465, -0.099439, -0.10885, -0.041745, -0.06081,
        -0.07014, 0.059277, -0.030148, -0.063123, -0.030069, -0.11469, -0.034931, -0.103586,
        -0.096185, -0.100229, 0.043812, 0.025266, 0.067996, 0.122076, 0.036929, -0.10749, 0.083916,
        -0.035953, -0.090549, 0.04747, -0.051073, -0.031175, 0.102754, 0.124701, 0.088866,
        -0.122735, -0.073504, 0.099415, 0.105005, 0.006244, 0.033987, 0.079349, -0.014227,
        -0.048687, -0.063449, 0.097048, -0.08602, 0.033889, -0.059143, -0.086404, -0.072336,
        0.026966, 0.114876, -0.01871, -0.104971, 0.084967, -0.05876, 0.048905, 0.096613, -0.098145,
        -0.033636, -0.110221, -0.056365, 0.106684, -0.118184, 0.094396, 0.079936, 0.039846,
        -0.000341, -0.112953, 0.020206, 0.101377, 0.027006, 0.124952, 0.009292, 0.067101,
        -0.060562, 0.116711, 0.115094, -0.09261, -0.070148, 0.109299, -0.059017, 0.017414,
        0.124786, 0.055257, -0.059675, -0.020027, -0.111447, 0.096844, 0.114881, -0.083635,
        0.122084, 0.054488, 0.044704, -0.073967, 0.10961, -0.028657, 0.051118, -0.096915,
        -0.094273, -0.108153, -0.031338, -0.071419, -0.081176, -0.043868, -0.094893, -0.038565,
        -0.101334, 0.115752, -0.027108, 0.029651, -0.01422, 0.102631, 0.119635, 0.05487, -0.094749,
        0.118418, 0.091913, 0.117331, 0.035812, 0.070047, 0.094922, 0.1233, -0.000603, 0.112858,
        -0.096873, -0.100138, 0.003982, -0.075222, -0.045561, 0.071879, -0.031863, -0.105448,
        0.013277, 0.075651, -0.08411, 0.112377, 0.070269, -0.014076, -0.081818, -0.047165,
        0.117551, -0.064808, -0.10104, 0.060395, -0.071024, 0.043827, -0.044834, -0.08597,
        -0.017177, -0.024347, 0.099635, 0.068142, -0.019589, -0.122633, 0.023616, 0.118371,
        0.097055, 0.099221, 0.020355, -0.078347, 0.106138, -0.106763, -0.065311, -0.05371,
        -0.038495, -0.028271, 0.103703, 0.035862, 0.005165, 0.058326, 0.027698, -0.057069, 0.10062,
        0.011382, -0.041905, -0.04382, -0.088402, -0.070802, -0.028664, 0.065489, -0.120916,
        -0.087379, 0.036819, 0.09767, -0.041779, 0.060137, -0.06057, -0.057824, -0.018015,
        0.106405, -0.028585, -0.022493, -0.033931, -0.088242, 0.017173, -0.061565, -0.021956,
        -0.016159, -0.005559, 0.048664, -0.024116, -0.061914, 0.102782, -0.010934, 0.040511,
        -0.091424, -0.076131, 0.022203, 0.011963, -0.102961, -0.08451, -0.012015, 0.049041,
        0.036601, 0.042275, 0.053184, 0.115604, 0.06523, 0.005804, -0.097372, 0.095977, -0.088877,
        0.039521, -0.042478, 0.006693, -0.012737, -0.046096, 0.104209, -0.018623, -0.099188,
        -0.002579, 0.060946, -0.07718, 0.0962, 0.122663, -0.05514, -0.004957, 0.007615, -0.028113,
        -0.087309, 0.006407, 0.077359, -0.118595, 0.044339, 0.085601, 0.099105, -0.077054,
        0.028232, 0.059936, 0.069168, -0.119833, 0.108723, -0.033822, -0.035373, 0.086443,
        -0.084574, 0.062005, 0.00402, 0.123173, 0.06304, 0.109579, 0.058833, -0.093053, -0.123317,
        -0.114242, -0.038647, -0.079086, 0.017697, -0.083227, 0.107415, 0.051208, -0.100382,
        -0.022368, -0.027179, 0.023965, 0.106357, -0.013526, 0.082908, -0.042032, 0.098684,
        -0.092893, 0.016613, -0.010788, 0.016803, -0.006714, -0.06557, -0.060309, 0.067629,
        -0.053336, 0.061964, -0.026332, -0.057832, 0.055776, -0.013702, 0.022456, 0.117277,
        0.045037, -0.087574, -0.027918, 0.11825, 0.08546, -0.039336, -0.064932, 0.10014, -0.088766,
        0.10491, 0.07943, 0.029585, -0.019813, -0.100515, 0.028801, -0.111631, 0.046466, -0.013591,
        0.005975, -0.12196, -0.10356, 5.2e-05, 0.024548, -0.085304, 0.106097, -0.104743, -0.121702,
        -0.035586, 0.061937, 0.094054, -0.092811, -0.020198, 0.008199, -0.067632, -0.073795,
        -0.085146, 0.037391, 0.105314, 0.020812, -0.081179, -0.083311, 0.110602, -0.042613,
        0.109594, 0.050212, 0.011663, 0.053241, 0.0868, -0.044786, 0.114316, 0.062902, -0.020643,
        -0.077108, 0.02839, 0.049479, -0.094501, 0.027296, -0.115785, -0.080892, -0.113941,
        0.080449, -0.054115, -0.015735, -0.060988, -0.09529, 0.088162, 0.117865, -0.007858,
        0.118394, 0.094355, -0.102473, -0.120761, 0.062101, 0.062645, -0.027841, 0.078502,
        0.010846, -0.052014, -0.019269, -0.070968, -0.113484, 0.07399, -0.022445, 0.032475,
        -0.116756, -0.086832, 0.043413, 0.10281, -0.120162, 0.105816, 0.097945, 0.039526, 0.087749,
        0.019249, -0.04561, 0.102345, -0.121458, -0.067369, 0.062826, 0.011609, -0.092681,
        0.074923, 0.047448, 0.08948, -0.025358, 0.039472, 0.106015, 0.046602, 0.028734, -0.112699,
        0.083245, 0.096516, 0.084054, 0.051154, 0.072306, -0.030456, 0.018165, 0.112996, -0.065269,
        0.111148, -0.11831, -0.036937, -0.037919, -0.049725, -0.058505, 0.044966, 0.001233,
        -0.032008, 0.103898, 0.041832, -0.082934, -0.116839, 0.043524, -0.087577, 0.051687, 0.055,
        -0.050003, -0.035964, -0.0093, -0.094167, -0.111804, -0.026926, 0.009797, 0.111366,
        0.036611, 0.077283, 0.103455, 0.054175, 0.02847, 0.108447,
    ];
    const G_W3: &[f32] = &[
        -0.135629, 0.078641, 0.017727, 0.17401, -0.051668, -0.075561, -0.065759, -0.075546,
        0.057979, 0.140927, -0.148042, -0.014166, -0.154871, -0.124041, 0.071536, -0.077411,
        0.096909, 0.000502, 0.02262, -0.024402, 0.092021, 0.158908, -0.070796, 0.03842, 0.109638,
        0.175784, -0.059769, 0.023611, -0.112703, -0.1024, 0.046231, -0.033389, 0.17607, 0.031489,
        -0.031964, -0.176749, -0.104786, 0.020881, 0.115519, 0.009052, 0.176483, -0.082882,
        0.010013, -0.005236, -0.169651, -0.061075, 0.066775, 0.026594, -0.157503, 0.047881,
        0.077021, 0.135455, 0.108576, -0.122924, -0.043375, 0.171829, 0.018475, 0.029167, 0.076485,
        -0.104879, 0.010198, 0.076496, 0.130449, -0.011023, -0.097892, -0.069662, 0.008135,
        -0.07083, -0.074153, 0.146838, 0.034594, -0.162925, -0.051687, -0.011634, 0.006423,
        -0.128633, -0.172208, 0.080206, 0.122394, 0.165916, 0.009187, 0.175063, -0.074634,
        0.103072, 0.031265, -0.054035, -0.093969, 0.094956, -0.00015, 0.096062, 0.077031, 0.098242,
        0.057087, -0.050563, -0.153851, -0.140445, 0.167735, -0.025822, 0.002834, 0.018565,
        0.047002, -0.029959, -0.081298, 0.164529, -0.064713, -0.001098, 0.116679, -0.098657,
        -0.072752, -0.066781, -0.164586, 0.15726, -0.113103, 0.041434, 0.107185, 0.046464,
        -0.122545, 0.006661, 0.174895, 0.12437, -0.02847, 0.112057, 0.035769, -0.075729, -0.031266,
        -0.035781, -0.011778, 0.081497,
    ];
    const G_W4: &[f32] = &[
        -0.221622, 0.040338, -0.015874, 0.06039, 0.083388, 0.219613, 0.074974, 0.091058, 0.209202,
        -0.219235, 0.245664, -0.18147, 0.227439, 0.110187, 0.167848, 0.03407, 0.175766, 0.121923,
        -0.032257, -0.24968, -0.194012, -0.230717, -0.092807, 0.153307, -0.028066, 0.159911,
        -0.135097, -0.226178, 0.127793, 0.094896, 0.087757, -0.195211,
    ];
    const G_W5: &[f32] = &[
        0.189327, 0.078448, 0.247197, -0.246498, -0.240669, 0.169026, 0.221956, 0.15317, 0.166207,
        -0.210568, 0.180953, -0.236037, 0.05585, -0.163759, -0.099299, 0.044267,
    ];
    const G_BN1_MEAN: &[f32] = &[
        0.163925, 0.016793, 0.100173, 0.135883, -0.091888, 0.099005, -0.00079, -0.048434,
    ];
    const G_BN1_VAR: &[f32] = &[
        0.725659, 0.843648, 1.18204, 1.39933, 1.4783, 1.08003, 1.02107, 0.775982,
    ];
    const G_BN1_SCALE: &[f32] = &[
        0.984586, 1.24317, 1.04492, 1.06576, 1.05544, 1.12366, 1.06993, 1.06945,
    ];
    const G_BN1_BIAS: &[f32] = &[
        -0.036982, -0.025984, 0.128958, 0.050514, 0.050746, 0.038344, 0.006657, 0.000256,
    ];
    const G_BN2_MEAN: &[f32] = &[0.049622, 0.080409, 0.103266, -0.050927];
    const G_BN2_VAR: &[f32] = &[0.504331, 0.86997, 1.19142, 1.43053];
    const G_BN2_SCALE: &[f32] = &[0.948457, 1.04381, 0.950099, 1.00156];
    const G_BN2_BIAS: &[f32] = &[0.065465, 0.036938, 0.042914, 0.106698];
    const G_BN3_MEAN: &[f32] = &[0.03304, 0.127533];
    const G_BN3_VAR: &[f32] = &[1.48913, 0.796144];
    const G_BN3_SCALE: &[f32] = &[0.768852, 1.14152];
    const G_BN3_BIAS: &[f32] = &[0.08727, -0.019373];
    const G_BN4_MEAN: &[f32] = &[0.08502];
    const G_BN4_VAR: &[f32] = &[1.48859];
    const G_BN4_SCALE: &[f32] = &[1.11745];
    const G_BN4_BIAS: &[f32] = &[0.08903];
    const D_W1: &[f32] = &[
        -0.098295, -0.039556, -0.236956, -0.235718, 0.096841, -0.10582, -0.152097, -0.098646,
        -0.204392, -0.19985, 0.149593, 0.127061, 0.033301, -0.136565, 0.083175, 0.011231,
    ];
    const D_W2: &[f32] = &[
        -0.178668, 0.127718, 0.053797, 0.220477, 0.227636, -0.248733, -0.245902, 0.135376,
        -0.248598, -0.072583, 0.032031, -0.244805, 0.188256, 0.055494, -0.152122, -0.136103,
        0.055034, 0.035428, 0.175746, -0.162312, 0.077079, -0.188132, 0.054103, -0.021213,
        0.249431, -0.083981, 0.162868, 0.146216, -0.140607, 0.021555, 0.021609, 0.19514,
    ];
    const D_W3: &[f32] = &[
        0.033128, -0.012386, 0.14517, -0.010889, -0.056845, -0.166965, 0.068463, -0.110871,
        0.119737, -0.101927, 0.148929, -0.06394, 0.00966, 0.071689, -0.060581, 0.114876, 0.03318,
        -0.059958, -0.097518, -0.070887, -0.056855, 0.122016, 0.152012, 0.109785, 0.000634,
        0.139749, 0.073685, -0.070098, -0.033989, 0.025716, 0.169703, -0.041171, -0.015029,
        -0.013979, -0.073839, 0.003762, -0.137864, -0.056114, -0.113502, -0.162222, -0.001087,
        -0.008939, -0.020721, -0.001765, -0.146732, 0.032177, -0.166449, -0.017818, -0.03803,
        -0.135031, 0.069292, -0.015881, 0.10245, -0.042249, 0.130388, -0.035338, -0.122922,
        -0.147043, 0.042436, 0.139367, -0.119157, 0.108498, -0.017467, 0.130459, -0.043165,
        -0.112578, 0.087647, -0.119812, -0.048461, 0.161493, -0.112209, 0.082118, 0.029926,
        -0.04557, 0.172931, -0.138661, 0.029348, -0.101271, -0.175777, -0.150494, 0.089643,
        0.084933, -0.169342, 0.054037, -0.121778, 0.026354, -0.041773, 0.002598, -0.012475,
        0.12241, 0.144399, -0.082467, 0.09887, 0.073799, 0.0177, -0.170502, 0.152907, -0.170296,
        0.06789, -0.068437, -0.012895, 0.11181, -0.129579, 0.059043, -0.102287, -0.033289,
        0.064468, -0.163625, 0.050817, -0.07813, -0.019754, -0.171732, 0.10067, 0.112263, 0.070856,
        0.022428, -0.110441, 0.128929, 0.124847, -0.127351, -0.102582, -0.155374, 0.076841,
        -0.154871, 0.078135, -0.015985, -0.015037, -0.06811,
    ];
    const D_W4: &[f32] = &[
        -0.020125, -0.079732, 0.103442, -0.005262, 0.010104, -0.059615, -0.052602, -0.07052,
        -0.113588, -0.101699, -0.024881, -0.090837, 0.039388, -0.022545, -0.026071, 0.052145,
        -0.083466, -0.022845, 0.071794, 0.038376, -0.013918, -0.033444, -0.046268, -0.116466,
        0.008196, 0.097409, 0.11871, 0.079248, 0.030613, -0.109813, 0.068495, -0.07128, -0.117871,
        0.109567, 0.056967, -0.123239, 0.123698, -0.022037, -0.02375, 0.093716, -0.017569,
        -0.108819, -0.111901, -0.088015, -0.080882, 0.038265, 0.094777, -0.064106, -0.080713,
        -0.005416, 0.062903, 0.092588, -0.006825, -0.050757, 0.028523, -0.052019, -0.086655,
        0.073371, 0.047679, 0.029596, -0.047715, -0.005327, -0.019314, -0.035097, 0.049129, 0.0551,
        0.096883, -0.007793, 0.109679, -0.000598, -0.0722, 0.053432, -0.096211, 0.102388,
        -0.123304, 0.022345, 0.107005, -0.124438, -0.114459, -0.081996, -0.094729, -0.003467,
        0.076877, -0.019005, -0.003409, 0.09084, -0.036465, 0.012533, 0.071285, -0.083987,
        0.102318, -0.087371, -0.043329, 0.100972, -0.115754, 0.05105, 0.023371, 0.082691,
        -0.116296, -0.116804, -0.077233, 0.085226, -0.023159, -0.009074, 0.113839, -0.042864,
        0.013078, 0.082213, 0.089442, -0.04105, 0.002043, 0.022602, -0.033412, 0.064723, 0.106709,
        -0.014867, -0.008608, 0.06532, -0.008055, -0.095356, -0.060377, -0.030805, -0.090914,
        -0.046423, 0.117726, -0.117115, -0.061402, 0.056512, -0.011568, 0.053137, 0.041015,
        -0.032264, 0.071283, -0.030923, 0.086758, 0.046716, 0.000632, 0.112428, 0.055235,
        -0.068178, 0.109413, -0.025423, 0.070397, -0.015679, 0.095331, 0.116232, 0.032761,
        -0.037023, 0.098865, -0.044144, -0.030797, -0.103359, -0.12451, -0.113758, 0.124284,
        -0.107302, -0.08123, 0.0525, -0.049388, 0.049997, -0.0089, -0.082685, 0.102347, 0.093543,
        0.036588, -0.056957, 0.078586, 0.081568, 0.055294, -0.051683, 0.107244, 0.051043,
        -0.051832, -0.014488, 0.108753, -0.021863, -0.068835, -0.079713, -0.015581, -0.06304,
        0.062719, -0.036094, -0.114455, 0.058917, -0.063518, 0.119938, 0.104839, -0.042664,
        -0.011239, -0.117385, -0.027007, 0.046332, 0.060408, -0.01104, -0.116489, 0.05608,
        0.010821, -0.020643, -0.0116, -0.049996, -0.119773, -0.082918, -0.067454, -0.046288,
        -0.086451, -0.105935, -0.045258, -0.008678, 0.029619, -0.055624, -0.026735, 0.032962,
        -0.111716, 0.072888, 0.106714, 0.12443, -0.096773, -0.069046, -0.037132, 0.034418,
        0.047072, 0.066808, -0.018464, 0.00059, 0.062133, 0.050999, 0.115462, -0.078687, 0.029048,
        0.117099, 0.105981, -0.099514, -0.10688, -0.109031, 0.058958, 0.031313, 0.010858,
        -0.041975, 0.000618, -0.009116, 0.076851, -0.11655, 0.080477, 0.007648, 0.084899, 0.006357,
        -0.013559, 0.062187, -0.118329, -0.12322, 0.085026, 0.109173, -0.052197, 0.024486,
        -0.072913, -0.040803, -0.105071, 0.053816, -0.093375, -0.045227, 0.044373, 0.124777,
        -0.041563, -0.06546, 0.001842, 0.061807, -0.014478, -0.032134, -0.068845, 0.114577,
        0.040279, 0.102972, 0.098499, -0.077635, 0.018253, 0.1077, 0.0259, -0.064692, 0.055292,
        0.066126, -0.107049, 0.030997, 0.032512, -0.102075, -0.028537, -0.106363, -0.004448,
        -0.021237, -0.055757, 0.092273, 0.064279, 0.009031, 0.087541, -0.112661, 0.119105,
        0.006386, 0.109785, 0.018163, -0.049738, 0.120438, 0.10931, -0.060195, -0.086517, -0.01322,
        -0.020962, -0.026448, -0.037692, -0.056046, 0.108565, 0.105098, 0.047927, 0.029021,
        0.051786, 0.009235, 0.104593, 0.119835, 0.005623, 0.041981, 0.112584, 0.004442, -0.12025,
        -0.013403, -0.033548, 0.110918, 0.073712, 0.120607, -0.021735, -0.075642, -0.03931,
        0.062937, -0.08683, 0.116292, -0.124892, 0.092653, 0.099239, 0.023159, -0.063479,
        -0.084876, 0.084755, -0.011928, -0.073324, 0.001812, 0.002061, 0.003269, 0.105383,
        0.063388, 0.084644, 0.113079, 0.02234, -0.077472, -0.093606, 0.008082, 0.05188, 0.070453,
        -0.039196, -0.054465, 0.000697, 0.122063, -0.068577, -0.047636, -0.091209, -0.075734,
        -0.02499, -0.042788, -0.098134, 0.100271, 0.046527, 0.032028, 0.020153, 0.10207, -0.092235,
        0.047279, -0.055326, 0.054118, -0.054094, -0.101081, -0.001437, 0.122773, -0.104387,
        0.026406, -0.058382, -0.068283, 0.030064, 0.037365, -0.051359, -0.049279, 0.047006,
        0.068909, -0.123845, 0.098472, 0.123032, 0.067737, -0.071385, -0.035012, -0.053399,
        0.11879, 0.011921, 0.011377, 0.04246, -0.013757, -0.04148, 0.062294, 0.099512, -0.078249,
        0.085212, 0.082654, -0.011699, -0.0971, 0.098528, -0.097928, -0.100089, 0.066341, 0.050733,
        0.052171, 0.031244, -0.000454, -0.089607, 0.049432, -0.017278, 0.041821, -0.059392,
        0.066376, -0.072018, 0.000177, -0.089043, 0.029163, 0.002699, 0.065946, -0.114783,
        -0.109684, -0.105032, -0.108026, -0.038565, 0.078791, 0.064243, 0.059272, -0.100847,
        -0.06931, -0.093962, -0.088412, 0.002175, -0.108576, 0.101166, -0.046759, 0.002319,
        0.074971, -0.104888, 0.115617, 0.034026, -0.103848, 0.084651, -0.098795, 0.085989,
        -0.026042, 0.025652, 0.093477, 0.003751, 0.056141, 0.055025, 0.100048, -0.050622,
        -0.108747, 0.024348, 0.069275, -0.030091, 0.045026, -0.057374, -0.071279, -0.068018,
        0.113675, -0.053121, 0.086595, -0.075794, -0.116346, -0.089268, 0.079916, -0.066682,
        -0.002703, -0.122269, 0.04812, -0.022429, -0.069707, -0.111827, -0.11908, -0.010821,
        0.100835, 0.067713, 0.070349, -0.050319, -0.114372, 0.006201, -0.021625, -0.118707,
        -0.097702, -0.070926, 0.08279, 0.100991, -0.089553, 0.121235, 0.001325, -0.110306,
        0.043037, -0.020868, -0.029201, -0.121676, -0.0564, -0.023743, 0.021628, 0.123049,
        -0.047084, -0.10481, -0.01589,
    ];
    const D_W5: &[f32] = &[
        0.063025, -3.2e-05, -0.057513, -0.068536, 0.049433, 0.073214, 0.068643, 0.01471, 0.061013,
        -0.081152, -0.063681, -0.08505, 0.071213, -0.0008, -0.0407, 0.029908, -0.033087, -0.056687,
        0.034091, 0.016184, -0.063297, -0.058218, 0.041443, 0.067208, 0.007195, -0.044396, 0.04268,
        0.015415, -0.077382, -0.05728, 0.019346, 0.054692, -0.028396, 0.042739, -0.05474, 0.084582,
        0.062981, 0.016556, 0.039323, 0.030773, -0.052158, -0.000771, 0.02357, -0.019951, 0.08476,
        -0.034086, -0.078937, -0.04208, -0.082032, 0.072836, -0.039678, -0.062188, -0.009555,
        -0.059557, -0.046172, 0.045061, -0.066362, 0.041683, 0.064456, -0.000198, 0.039248,
        -0.049511, 0.068927, 0.085864, 0.021004, 0.080546, -0.087784, -0.009705, 0.017562,
        -0.076826, 0.043013, -0.033686, -0.031969, -0.070129, 0.036135, -0.064314, -0.018757,
        0.039265, 0.066587, 0.019987, -0.032973, -0.038446, 0.049611, 0.016399, 0.074406, 0.033286,
        0.071204, -0.071224, 0.071058, -0.04061, -0.02366, 0.000529, -0.016576, 0.017484, 0.078763,
        -0.052811, 0.027311, -0.071038, 0.013878, 0.072815, 0.059599, 0.075692, 0.047847,
        -0.007313, 0.023078, -0.036933, 0.075026, 0.044256, -0.064693, 0.076426, -0.001312,
        -0.003929, -0.024191, -0.02332, 0.004424, -0.04315, -0.082219, -0.045529, -0.017589,
        0.066746, 0.065023, -0.052958, -0.003394, -0.02551, -0.047285, -0.015543, -0.030669,
        0.049405,
    ];
    const D_BN1_MEAN: &[f32] = &[-0.18594, 0.003889];
    const D_BN1_VAR: &[f32] = &[0.87473, 0.835963];
    const D_BN1_SCALE: &[f32] = &[1.14035, 0.919943];
    const D_BN1_BIAS: &[f32] = &[0.124758, 0.103195];
    const D_BN2_MEAN: &[f32] = &[-0.024432, -0.034092, 0.024997, 0.062286];
    const D_BN2_VAR: &[f32] = &[0.744562, 1.07503, 1.45944, 1.46836];
    const D_BN2_SCALE: &[f32] = &[0.96639, 0.951472, 0.857381, 1.00955];
    const D_BN2_BIAS: &[f32] = &[0.01301, 0.156966, 0.196363, 0.11355];
    const D_BN3_MEAN: &[f32] = &[
        -0.031571, -0.029798, -0.121395, 0.150365, -0.213366, 0.012636, 0.01758, -0.000288,
    ];
    const D_BN3_VAR: &[f32] = &[
        1.34741, 0.996283, 1.15341, 1.01739, 1.01357, 0.52419, 0.562883, 1.04157,
    ];
    const D_BN3_SCALE: &[f32] = &[
        0.947618, 1.06124, 0.992159, 0.946438, 0.898145, 1.14488, 1.06682, 0.853798,
    ];
    const D_BN3_BIAS: &[f32] = &[
        -0.058574, -0.139271, 0.047026, 0.007257, -0.111808, 0.1047, -0.11613, 0.037924,
    ];
    const D_OUT: &[f32] = &[0.003951];
    const G_L1_WEIGHT_GRAD: &[f32] = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3e-06, -1e-06, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-06, -0.0, 0.0, -0.0, -0.0, 0.0,
        -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2e-06, -1e-06, 1e-06, -0.0, 3e-06, -1e-06, 1e-06,
        -0.0, -3e-06, 1e-06, -1e-06, 0.0, -2e-06, 1e-06, 0.0, 0.0, 3e-06, -1e-06, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -1e-06, 0.0, 1e-06, -0.0,
        0.0, -0.0, -0.0, 0.0, -1e-06, 0.0, 0.0, -0.0, -1e-06, 0.0, 1e-06, -0.0, -2e-06, 1e-06,
        1e-06, -0.0, 1e-06, -0.0, -1e-06, 0.0, 0.0, -0.0, 1e-06, -0.0, -2e-06, 1e-06, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, -1e-06, 0.0,
        3e-06, -1e-06, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -1e-06, 0.0, 1e-06, -0.0, 1e-06, -0.0, 1e-06,
        -0.0, -3e-06, 1e-06, 0.0, -0.0, -1e-06, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0,
        -0.0, 1e-06, -0.0, 3e-06, -1e-06, 1e-06, -0.0, -0.0, 0.0, 0.0, -0.0, 1e-06, -0.0, 1e-06,
        -0.0, 0.0, -0.0, 1e-06, -0.0, 4e-06, -1e-06, -1e-06, 0.0, 1e-06, -0.0, -2e-06, 1e-06,
        4e-06, -1e-06,
    ];

    fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name} length");
        let mut max_error = 0.0_f32;
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let error = (a - e).abs();
            assert!(
                a.is_finite() && error < 1e-5 + 1e-3 * e.abs(),
                "{name}[{i}]: axis={a} pytorch={e} error={error}"
            );
            max_error = max_error.max(error);
        }
        println!(
            "check {name}: PASS {} elements max_abs_error={max_error:.2e}",
            actual.len()
        );
    }

    let device = Device::cuda(0)?;
    let (batch, channel, height, width) = (
        Axis::new("dcgan_ref_batch"),
        Axis::new("dcgan_ref_channel"),
        Axis::new("dcgan_ref_height"),
        Axis::new("dcgan_ref_width"),
    );

    let mut generator = DcganGenerator::with_features(channel, [height, width], NC, NGF);
    let z = Tensor::from_slice(
        Z,
        [batch.of(1), channel.of(NZ), height.of(1), width.of(1)],
        &device,
    )?;
    let image_shape = generator.build(z.shape(), &device, 1)?;
    assert_eq!(
        image_shape,
        Shape::new([batch.of(1), height.of(64), width.of(64), channel.of(NC)])?
    );

    generator.parameter("0.weight")?.set_values(G_W1)?;
    generator.parameter("3.weight")?.set_values(G_W2)?;
    generator.parameter("6.weight")?.set_values(G_W3)?;
    generator.parameter("9.weight")?.set_values(G_W4)?;
    generator.parameter("12.weight")?.set_values(G_W5)?;
    let generator_bn = [
        (1, G_BN1_MEAN, G_BN1_VAR, G_BN1_SCALE, G_BN1_BIAS),
        (4, G_BN2_MEAN, G_BN2_VAR, G_BN2_SCALE, G_BN2_BIAS),
        (7, G_BN3_MEAN, G_BN3_VAR, G_BN3_SCALE, G_BN3_BIAS),
        (10, G_BN4_MEAN, G_BN4_VAR, G_BN4_SCALE, G_BN4_BIAS),
    ];
    for (index, mean, var, scale, bias) in generator_bn {
        generator
            .state(&format!("{index}.running_mean"))?
            .replace(Tensor::from_slice(mean, [channel.of(mean.len())], &device)?);
        generator
            .state(&format!("{index}.running_var"))?
            .replace(Tensor::from_slice(var, [channel.of(var.len())], &device)?);
        generator
            .parameter(&format!("{index}.scale"))?
            .set_values(scale)?;
        generator
            .parameter(&format!("{index}.bias"))?
            .set_values(bias)?;
    }

    let fake = generator.forward(&z)?;
    assert_eq!(fake.shape(), &image_shape);

    let mut discriminator = DcganDiscriminator::with_features(channel, [height, width], NDF);
    let logits_shape = discriminator.build(&image_shape, &device, 2)?;
    assert_eq!(
        logits_shape,
        Shape::new([batch.of(1), height.of(1), width.of(1), channel.of(1)])?
    );
    discriminator.parameter("0.weight")?.set_values(D_W1)?;
    discriminator.parameter("2.weight")?.set_values(D_W2)?;
    discriminator.parameter("5.weight")?.set_values(D_W3)?;
    discriminator.parameter("8.weight")?.set_values(D_W4)?;
    discriminator.parameter("11.weight")?.set_values(D_W5)?;
    let discriminator_bn = [
        (3, D_BN1_MEAN, D_BN1_VAR, D_BN1_SCALE, D_BN1_BIAS),
        (6, D_BN2_MEAN, D_BN2_VAR, D_BN2_SCALE, D_BN2_BIAS),
        (9, D_BN3_MEAN, D_BN3_VAR, D_BN3_SCALE, D_BN3_BIAS),
    ];
    for (index, mean, var, scale, bias) in discriminator_bn {
        discriminator
            .state(&format!("{index}.running_mean"))?
            .replace(Tensor::from_slice(mean, [channel.of(mean.len())], &device)?);
        discriminator
            .state(&format!("{index}.running_var"))?
            .replace(Tensor::from_slice(var, [channel.of(var.len())], &device)?);
        discriminator
            .parameter(&format!("{index}.scale"))?
            .set_values(scale)?;
        discriminator
            .parameter(&format!("{index}.bias"))?
            .set_values(bias)?;
    }

    let d_out = discriminator.forward(&fake)?;
    assert_close(
        "DCGAN eval-mode discriminator logits",
        &d_out.to_vec()?,
        D_OUT,
    );

    let loss = d_out.mean([batch, height, width, channel])?;
    loss.backward()?;
    let g_l1_grad = generator
        .parameter("0.weight")?
        .grad()
        .expect("generator's first conv weight must receive a gradient through the discriminator")
        .to_vec()?;
    assert_close(
        "DCGAN generator first ConvTranspose2d weight gradient",
        &g_l1_grad,
        G_L1_WEIGHT_GRAD,
    );

    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn dcgan_forward_training_step_updates_both_networks_and_batch_norm_state() -> Result<()> {
    // One complete alternating step exactly as this file's module doc describes: a
    // discriminator step on a real batch and a detached fake batch, then a generator step
    // backpropagated through the (otherwise untouched) discriminator. Both networks' parameters
    // and both networks' BatchNorm running statistics must change, and only once each.
    const NZ: usize = 8;
    const NGF: usize = 4;
    const NDF: usize = 4;
    const NC: usize = 3;
    const BATCH: usize = 4;

    let device = Device::cuda(0)?;
    let (batch, channel, height, width) = (
        Axis::new("dcgan_step_batch"),
        Axis::new("dcgan_step_channel"),
        Axis::new("dcgan_step_height"),
        Axis::new("dcgan_step_width"),
    );

    let mut generator = DcganGenerator::with_features(channel, [height, width], NC, NGF);
    let mut discriminator = DcganDiscriminator::with_features(channel, [height, width], NDF);

    let latent_shape = Shape::new([batch.of(BATCH), channel.of(NZ), height.of(1), width.of(1)])?;
    let image_shape = generator.build(&latent_shape, &device, 11)?;
    discriminator.build(&image_shape, &device, 13)?;

    // The paper's own hyperparameters: lr=2e-4, beta1=0.5, beta2=0.999.
    let mut g_trainer =
        Trainer::new(Adam::with_hyperparameters(2e-4, 0.5, 0.999, 1e-8)?).with_seed(101);
    let mut d_trainer =
        Trainer::new(Adam::with_hyperparameters(2e-4, 0.5, 0.999, 1e-8)?).with_seed(202);

    let real = Tensor::uniform(image_shape.dims().iter().copied(), 5, -1.0, 1.0, &device)?;
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

    let before_g: Vec<f32> = generator
        .named_parameters()
        .into_iter()
        .flat_map(|(_, p)| p.tensor().to_vec().unwrap())
        .collect();
    let before_d: Vec<f32> = discriminator
        .named_parameters()
        .into_iter()
        .flat_map(|(_, p)| p.tensor().to_vec().unwrap())
        .collect();
    let before_g_running: Vec<f32> = generator
        .named_states()
        .into_iter()
        .filter(|(name, _)| name.ends_with("running_mean") || name.ends_with("running_var"))
        .flat_map(|(_, s)| s.tensor().to_vec().unwrap())
        .collect();
    let before_d_running: Vec<f32> = discriminator
        .named_states()
        .into_iter()
        .filter(|(name, _)| name.ends_with("running_mean") || name.ends_with("running_var"))
        .flat_map(|(_, s)| s.tensor().to_vec().unwrap())
        .collect();

    // 1) Discriminator step: real against the real label, detached fake against the fake label.
    let noise_seed = 17_u64;
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
    assert!(d_step.pre_update_loss()?.is_finite());
    assert!(
        d_step.committed_states() > 0,
        "discriminator BatchNorm state must commit"
    );

    // 2) Generator step: the non-saturating objective, backpropagated through the (read-only
    // here) discriminator.
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
    assert!(g_step.pre_update_loss()?.is_finite());
    assert!(
        g_step.committed_states() > 0,
        "generator BatchNorm state must commit"
    );

    let after_g: Vec<f32> = generator
        .named_parameters()
        .into_iter()
        .flat_map(|(_, p)| p.tensor().to_vec().unwrap())
        .collect();
    let after_d: Vec<f32> = discriminator
        .named_parameters()
        .into_iter()
        .flat_map(|(_, p)| p.tensor().to_vec().unwrap())
        .collect();
    let after_g_running: Vec<f32> = generator
        .named_states()
        .into_iter()
        .filter(|(name, _)| name.ends_with("running_mean") || name.ends_with("running_var"))
        .flat_map(|(_, s)| s.tensor().to_vec().unwrap())
        .collect();
    let after_d_running: Vec<f32> = discriminator
        .named_states()
        .into_iter()
        .filter(|(name, _)| name.ends_with("running_mean") || name.ends_with("running_var"))
        .flat_map(|(_, s)| s.tensor().to_vec().unwrap())
        .collect();

    assert_ne!(before_g, after_g, "generator parameters must update");
    assert_ne!(before_d, after_d, "discriminator parameters must update");
    assert_ne!(
        before_g_running, after_g_running,
        "generator BatchNorm running statistics must update"
    );
    assert_ne!(
        before_d_running, after_d_running,
        "discriminator BatchNorm running statistics must update"
    );

    println!(
        "check dcgan alternating step: PASS d_loss={:.4} g_loss={:.4} d_states={} g_states={}",
        d_step.pre_update_loss()?,
        g_step.pre_update_loss()?,
        d_step.committed_states(),
        g_step.committed_states(),
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn broadcast_bias_gradient_plan_is_cached_by_shape_and_layout_signature() -> Result<()> {
    // The ImageNet64 profile (data/evidence/imagenet64/profile.md in axis-benchmarks)
    // found align()'s broadcast scatter-add plan for a bias rebuilt from scratch, and
    // re-uploaded to the device, on every training step even though a bias's shape and
    // layout never change once a model is built. align_reverse_plan now caches that plan
    // by (source shape, source layout, target shape); this checks that repeating the same
    // broadcast reuses one plan, a distinct broadcast shape builds and caches its own, and
    // every call -- cached or not -- still produces the hand-computed forward value and
    // gradient.
    let device = Device::cuda(0)?;
    let (batch, channel) = (Axis::new("batch"), Axis::new("channel"));
    let bias_values = [0.5f32, -1.5, 2.0];
    let bias = Tensor::from_slice(&bias_values, [channel.of(3)], &device)?.with_grad();
    let field_values: Vec<_> = (0..12).map(|v| v as f32 * 0.25).collect();
    let field =
        Tensor::from_slice(&field_values, [batch.of(4), channel.of(3)], &device)?.with_grad();

    let check_biased =
        |field: &Tensor, bias: &Tensor, field_values: &[f32], step: usize| -> Result<()> {
            // Gradients accumulate onto an existing leaf grad, so each check starts clean.
            bias.zero_grad();
            field.zero_grad();
            let biased = field.add(bias)?;
            let out = biased.shape().clone();
            let channel_extent = out.extent(channel)?;
            let mut expected = Vec::with_capacity(out.len());
            for index in 0..out.len() {
                let coords = out.coords(index);
                let (b, c) = (coords[out.index(batch)?], coords[out.index(channel)?]);
                expected.push(f64::from(field_values[b * 3 + c] + bias_values[c]));
            }
            close(
                &format!("cached broadcast values, step {step}"),
                &biased.to_vec()?,
                &expected,
            );
            // mean() divides by every reduced element (batch * channel); summing that
            // constant back over the batch axis a bias broadcasts across always leaves
            // exactly 1 / channel_extent, independent of the batch extent -- so the same
            // expected gradient holds for both the original and the "wide" batch below.
            biased.mean([batch, channel])?.backward()?;
            close(
                &format!("cached broadcast bias gradient, step {step}"),
                &bias.grad().expect("bias gradient").to_vec()?,
                &[1.0 / channel_extent as f64; 3],
            );
            Ok(())
        };

    let builds = Tensor::align_plan_build_count();
    for step in 0..3 {
        check_biased(&field, &bias, &field_values, step)?;
    }
    assert_eq!(
        Tensor::align_plan_build_count(),
        builds + 1,
        "repeated broadcasts of the same shape and layout must reuse one plan"
    );

    // A distinct target shape (a different batch extent) must build and cache its own plan
    // rather than reusing the first one.
    let wide_values: Vec<_> = (0..15).map(|v| v as f32 * 0.1).collect();
    let wide = Tensor::from_slice(&wide_values, [batch.of(5), channel.of(3)], &device)?.with_grad();
    check_biased(&wide, &bias, &wide_values, 3)?;
    assert_eq!(
        Tensor::align_plan_build_count(),
        builds + 2,
        "a distinct broadcast target shape must build its own plan"
    );

    // The original signature is still cached after a different one was built in between.
    check_biased(&field, &bias, &field_values, 4)?;
    assert_eq!(
        Tensor::align_plan_build_count(),
        builds + 2,
        "the first signature must still be cached after a different signature was built"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn grouped_reduction_spans_multiple_chunks_for_a_large_group() -> Result<()> {
    // GROUPED_CHUNK (runtime/backend.rs) splits a CSR group's contributions across many
    // blocks once a group exceeds 512 entries, combining the per-chunk partials back
    // together in a second kernel; every other test in this suite stays well under that
    // per-group count, so this specifically exercises the multi-chunk combine path (both
    // directions: align's backward Group plan for the bias gradient, and reduce_grouped's
    // forward Group plan for the sum itself) with a group large enough to span several
    // chunks (600 > 512, so two chunks per channel).
    let device = Device::cuda(0)?;
    let (batch, channel) = (Axis::new("batch"), Axis::new("channel"));
    let (batch_extent, channel_extent) = (600usize, 2usize);
    let field_values: Vec<f32> = (0..batch_extent * channel_extent)
        .map(|i| (i % 97) as f32 * 0.01 - 0.3)
        .collect();
    let bias_values = [0.25f32, -0.75];
    let field = Tensor::from_slice(
        &field_values,
        [batch.of(batch_extent), channel.of(channel_extent)],
        &device,
    )?
    .with_grad();
    let bias = Tensor::from_slice(&bias_values, [channel.of(channel_extent)], &device)?.with_grad();
    let biased = field.add(&bias)?;
    let out = biased.shape().clone();
    let mut expected = Vec::with_capacity(out.len());
    for index in 0..out.len() {
        let coords = out.coords(index);
        let (b, c) = (coords[out.index(batch)?], coords[out.index(channel)?]);
        expected.push(f64::from(
            field_values[b * channel_extent + c] + bias_values[c],
        ));
    }
    close("large-group broadcast values", &biased.to_vec()?, &expected);

    let summed = biased.sum(batch)?;
    let mut expected_sum = vec![0.0f64; channel_extent];
    for b in 0..batch_extent {
        for c in 0..channel_extent {
            expected_sum[c] += f64::from(field_values[b * channel_extent + c] + bias_values[c]);
        }
    }
    close("large-group forward sum", &summed.to_vec()?, &expected_sum);

    summed.sum(channel)?.backward()?;
    // sum's gradient is exactly 1 per contributing element, so the bias gradient -- summed
    // over every batch position a bias broadcasts across -- is exactly the batch extent: a
    // large exact integer that a correct per-chunk-partial reassociation still hits exactly
    // (summing many 1.0s stays exact in f32 far past 600).
    close(
        "large-group broadcast bias gradient",
        &bias.grad().expect("bias gradient").to_vec()?,
        &[batch_extent as f64; 2],
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_device_creation_enables_the_jit_disk_cache_once() -> Result<()> {
    let _first = Device::cuda(0)?;
    assert!(
        cutile::jit_cache::is_enabled(),
        "creating a CUDA device must enable cuTile's persistent kernel cache by default"
    );

    let _second = Device::cuda(0)?;
    // `ensure_jit_cache_enabled`'s `OnceLock` runs its closure at most once per
    // process; every `Device::cuda` call after the very first one in this
    // process -- this test's second device, or any earlier ignored test's
    // first one -- must be a no-op on the cache switch. cuTile exposes no
    // public way to inspect *which* store is currently installed, so call the
    // guard directly to observe the "ran only once" return value itself
    // rather than an indirect symptom of it.
    assert!(
        !crate::backend::ensure_jit_cache_enabled(),
        "a later Device::cuda call must not re-run cache enablement"
    );

    let stats = jit_cache_stats();
    println!(
        "check jit cache stats after two devices: PASS hits={} misses={}",
        stats.hits, stats.misses
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn axis_jit_cache_off_leaves_the_cache_disabled() -> Result<()> {
    // `AXIS_JIT_CACHE` is only ever read once per process (inside the same
    // `OnceLock` the test above exercises), and every ignored CUDA test in
    // this binary shares one process (`scripts/check.sh` runs them with
    // `--test-threads=1`). By the time this test runs, an earlier test has
    // very likely already created a device with the cache on, so this
    // process's own switch is already latched. Only a fresh child process,
    // started with `AXIS_JIT_CACHE=off` before it creates any device, can
    // observe the opt-out actually apply -- so this test re-execs itself,
    // selecting only itself, with the marker env var below set to switch to
    // the child's half of the check.
    const CHILD_MARKER: &str = "AXIS_JIT_CACHE_OFF_CHILD";
    if std::env::var_os(CHILD_MARKER).is_some() {
        let _device = Device::cuda(0)?;
        assert!(
            !cutile::jit_cache::is_enabled(),
            "AXIS_JIT_CACHE=off must leave cuTile's persistent kernel cache disabled"
        );
        println!("check axis_jit_cache_off_leaves_the_cache_disabled (child): PASS");
        return Ok(());
    }

    let exe = std::env::current_exe()?;
    let output = std::process::Command::new(&exe)
        .args([
            "--ignored",
            "--exact",
            "--test-threads=1",
            "tests::axis_jit_cache_off_leaves_the_cache_disabled",
        ])
        .env("AXIS_JIT_CACHE", "off")
        .env(CHILD_MARKER, "1")
        .output()?;
    assert!(
        output.status.success(),
        "isolated child process failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    println!("check axis_jit_cache_off_leaves_the_cache_disabled (parent): PASS");
    Ok(())
}
