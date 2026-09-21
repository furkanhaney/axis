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
fn merge_plan_cache_reuses_stable_shape_without_conflating_axis_order() -> Result<()> {
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
    let builds = Tensor::merge_plan_build_count();

    let expected_forward: Vec<_> = (0..24).map(|value| value as f32).collect();
    for _ in 0..4 {
        assert_eq!(
            input.merge([head, depth], feature)?.to_vec()?,
            expected_forward
        );
    }
    assert_eq!(
        Tensor::merge_plan_build_count(),
        builds + 1,
        "four stable-shape merges should build one reusable host plan"
    );

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
    assert_eq!(
        Tensor::merge_plan_build_count(),
        builds + 2,
        "reversing selected axes has distinct merge semantics and one reusable plan"
    );
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
