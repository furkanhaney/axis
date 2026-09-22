use crate::prelude::*;

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

    assert!(ELU::new(0.0).is_err());
    assert!(ELU::new(-1.0).is_err());
    assert!(ELU::new(f32::NAN).is_err());
    assert!(ELU::new(f32::INFINITY).is_err());
    assert!(ELU::new(1.0).is_ok());

    assert!(CELU::new(0.0).is_err());
    assert!(CELU::new(-0.5).is_err());
    assert!(CELU::new(f32::NAN).is_err());
    assert!(CELU::new(1.0).is_ok());

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

    let distance = x1.pairwise_distance(&x2, feature, eps as f32)?;
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
            .pairwise_distance(&wrong, feature, eps as f32)
            .is_err()
    );
    assert!(
        x1.detach()
            .pairwise_distance(&x2.detach(), feature, f32::NAN)
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

    let distance = x1.pairwise_distance(&x2, feature, eps as f32)?;
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
        let loss =
            anchor.triplet_margin_loss(&positive, &negative, feature, margin, eps as f32, false)?;
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
        let loss =
            anchor.triplet_margin_loss(&positive, &negative, feature, margin, eps as f32, true)?;
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
            .triplet_margin_loss(&p_err, &n_err, feature, margin, eps as f32, false)
            .is_err()
    );
    assert!(
        a_err
            .triplet_margin_loss(&p_err, &n_err, feature, f32::NAN, eps as f32, false)
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
        |a: &Tensor, b: &Tensor| a.pairwise_distance(b, feature, eps),
    )?;
    let via_direct =
        anchor2.triplet_margin_loss(&positive2, &negative2, feature, 1.0, eps, false)?;
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
                |a: &Tensor, b: &Tensor| { a.pairwise_distance(b, feature, eps) }
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

    let loss = x.multi_margin_loss(&target, class, margin)?;
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
            .multi_margin_loss(&target.with_grad(), class, margin)
            .is_err()
    );
    let bad_target = Tensor::from_slice(&[0.5_f32; 6], [batch.of(2), class.of(3)], &device)?;
    assert!(
        x.detach()
            .multi_margin_loss(&bad_target, class, margin)
            .is_err()
    );
    let solo_class = Axis::new("mml_solo_class");
    let solo_x = Tensor::from_slice(&[1.0_f32, 2.0], [batch.of(2), solo_class.of(1)], &device)?;
    let solo_t = Tensor::from_slice(&[1.0_f32, 1.0], [batch.of(2), solo_class.of(1)], &device)?;
    assert!(
        solo_x
            .multi_margin_loss(&solo_t, solo_class, margin)
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
    // -[y*log(x) + (1-y)*log(1-x)] with the probability floored at `f32::MIN_POSITIVE` before
    // its logarithm -- not PyTorch's own `exp(-100)` -- because `exp(-100)` rounds to an `f32`
    // subnormal so small that `ln`'s backward `1 / x` overflows to infinity and then hits the
    // clamp's zeroing multiplier as `inf * 0 -> NaN` (see the doc comment on
    // `binary_cross_entropy` for the full trace and why an output-side `-100` clamp has the
    // same NaN problem at `x == 0` for a different reason). This oracle uses that same
    // achieved floor (`ln(f32::MIN_POSITIVE) ~ -87.3`) rather than PyTorch's literal `-100`.
    // The composed gradient matches PyTorch's interior formula (x-y)/(x*(1-x)) away from the
    // floor, but is exactly zero -- not PyTorch's own large finite value -- at a saturated
    // wrong-side prediction (x == 0, y == 1 and its mirror x == 1, y == 0), since `clamp`'s
    // ordinary boundary rule zeroes the moved term's gradient there.
    let floor: f32 = f32::MIN_POSITIVE;
    fn bce(x: f32, y: f32, floor: f32) -> f64 {
        let log_x = x.max(floor).ln();
        let log_1mx = (1.0 - x).max(floor).ln();
        f64::from(-(y * log_x + (1.0 - y) * log_1mx))
    }
    fn bce_grad(x: f32, y: f32, floor: f32) -> f64 {
        // Composition trace: term1 = y * ln(max(x, floor)), term2 = (1-y) * ln(max(1-x,
        // floor)); the input-side clamp's own boundary rule zeroes a term's local derivative
        // wherever its *own* value fell below the floor, independent of the other term's
        // weight.
        let d_term1 = if x >= floor {
            f64::from(y) / f64::from(x)
        } else {
            0.0
        };
        let complement = 1.0 - x;
        let d_term2 = if complement >= floor {
            -f64::from(1.0 - y) / f64::from(complement)
        } else {
            0.0
        };
        -(d_term1 + d_term2)
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
        .map(|(&x, y)| bce(x, y, floor))
        .collect();
    let expected_grad: Vec<f64> = x_values
        .iter()
        .zip(y_values)
        .map(|(&x, y)| bce_grad(x, y, floor) / n)
        .collect();

    let x = Tensor::from_slice(&x_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?
        .with_grad();
    let y = Tensor::from_slice(&y_values, [batch.of(2), feature.of(3)], &device)?
        .with_layout([feature, batch])?;

    let loss = x.binary_cross_entropy(&y)?;
    close(
        "BCELoss forward under reordered, asymmetric storage",
        &loss.to_vec()?,
        &expected_loss,
    );
    loss.mean([batch, feature])?.backward()?;
    close(
        "BCELoss gradient (exact zero at the floor boundary)",
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
    assert!(LPPool1d::new(channel, height, 0, 2).is_err());
    assert!(LPPool2d::new(channel, [height, width], 0, [2, 2]).is_err());
    assert!(LPPool3d::new(channel, [height, width, channel.role("d")], 0, [2, 2, 2]).is_err());
    let pool = LPPool2d::new(channel, [height, width], 2, [2, 2]).unwrap();
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

    let mut pool = LPPool1d::new(channel, length, 1, 2)?;
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

    let mut pool = LPPool2d::new(channel, [height, width], 2, [2, 2])?;
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

    let mut pool = LPPool3d::new(channel, [depth, height, width], 3, [1, 1, 3])?;
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

#[test]
fn multihead_attention_rejects_invalid_configuration_before_allocation() {
    let (feature, time) = (Axis::new("feature"), Axis::new("time"));
    assert!(MultiheadAttention::new(feature, feature, 4, 2, 0.0).is_err());
    assert!(MultiheadAttention::new(feature, time, 4, 2, 0.1).is_err());
    assert!(MultiheadAttention::new(feature, time, 0, 2, 0.0).is_err());
    assert!(MultiheadAttention::new(feature, time, 4, 0, 0.0).is_err());
    assert!(MultiheadAttention::new(feature, time, 5, 2, 0.0).is_err());

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
