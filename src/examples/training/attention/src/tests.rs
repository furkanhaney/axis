use super::{attention::AttentionAxes, reference};
use axis::prelude::*;

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

#[test]
#[ignore = "requires CUDA; workspace check enables this"]
fn attention_forward_gradients_layouts_and_causality() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, feature, head, d) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("feature"),
        Axis::new("head"),
        Axis::new("head_feature"),
    );
    let axes = AttentionAxes::new(time, feature, head.of(2), d.of(3));
    let shape = reference::Shape {
        batch: 2,
        time: 5,
        heads: 2,
        width: 3,
    };
    let n = 60;
    let values: Vec<Vec<f64>> = (0..3)
        .map(|seed| {
            (0..n)
                .map(|i| {
                    // Round first so both implementations receive exactly the same inputs.
                    f64::from(((i * (7 + seed * 2) + seed * 11) % 31) as f32 / 19.0 - 0.7)
                })
                .collect()
        })
        .collect();
    let weights: Vec<_> = (0..n)
        .map(|i| f64::from((i % 11) as f32 / 13.0 - 0.4))
        .collect();
    let expected = shape.attention(&values[0], &values[1], &values[2]);
    let objective = |values: &[Vec<f64>]| {
        shape
            .attention(&values[0], &values[1], &values[2])
            .iter()
            .zip(&weights)
            .map(|(x, w)| x * w)
            .sum::<f64>()
            / n as f64
    };
    let mut gradients = vec![vec![0.0; n]; 3];
    let epsilon = 1e-5;
    for slot in 0..3 {
        for i in 0..n {
            let mut perturbed = values.clone();
            perturbed[slot][i] += epsilon;
            let plus = objective(&perturbed);
            perturbed[slot][i] -= 2.0 * epsilon;
            gradients[slot][i] = (plus - objective(&perturbed)) / (2.0 * epsilon);
        }
    }
    for leading_feature in [false, true] {
        let order = |values: &[f64]| -> Vec<f64> {
            if leading_feature {
                (0..n)
                    .map(|i| {
                        let f = i / 10;
                        let b = (i / 5) % 2;
                        let t = i % 5;
                        values[(b * 5 + t) * 6 + f]
                    })
                    .collect()
            } else {
                values.to_vec()
            }
        };
        let dims = if leading_feature {
            [feature.of(6), batch.of(2), time.of(5)]
        } else {
            [batch.of(2), time.of(5), feature.of(6)]
        };
        let tensor = |values: &[f64]| {
            Tensor::from_slice(
                &order(values).iter().map(|&v| v as f32).collect::<Vec<_>>(),
                dims,
                &device,
            )
        };
        let leaves: Vec<_> = values
            .iter()
            .map(|v| tensor(v).map(|t| t.with_grad()))
            .collect::<Result<_>>()?;
        let q = leaves[0].with_layout([feature, time, batch])?;
        let k = leaves[1].with_layout([time, batch, feature])?;
        let v = leaves[2].with_layout([batch, feature, time])?;
        let prediction = axes.attend(&q, &k, &v)?;
        assert_eq!(prediction.shape().dims(), dims);
        close(
            "causal attention forward",
            &prediction.to_vec()?,
            &order(&expected),
        );
        prediction
            .mul(&tensor(&weights)?)?
            .mean([batch, time, feature])?
            .backward()?;
        for (i, leaf) in leaves.iter().enumerate() {
            close(
                &format!("attention finite-difference gradient {i}"),
                &leaf.grad().unwrap().to_vec()?,
                &order(&gradients[i]),
            );
        }
    }
    // Perturb only future keys/values: earlier output tokens cannot change.
    let dims = [batch.of(2), time.of(5), feature.of(6)];
    let tensor = |values: &[f64]| {
        Tensor::from_slice(
            &values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            dims,
            &device,
        )
    };
    let mut future = values.clone();
    for slot in [1, 2] {
        for b in 0..2 {
            for f in 0..6 {
                future[slot][(b * 5 + 4) * 6 + f] += 10.0;
            }
        }
    }
    let changed = axes
        .attend(
            &tensor(&future[0])?,
            &tensor(&future[1])?,
            &tensor(&future[2])?,
        )?
        .to_vec()?;
    for b in 0..2 {
        close(
            "future token isolation",
            &changed[b * 30..b * 30 + 24],
            &expected[b * 30..b * 30 + 24],
        );
    }
    // A loss on the first query can use only the first value; Q and K derivatives vanish.
    let leaves: Vec<_> = values
        .iter()
        .map(|v| tensor(v).map(|t| t.with_grad()))
        .collect::<Result<_>>()?;
    let mut first = vec![0.0; n];
    first[..6].fill(1.0);
    let loss = axes
        .attend(&leaves[0], &leaves[1], &leaves[2])?
        .mul(&tensor(&first)?)?
        .mean([batch, time, feature])?;
    loss.backward()?;
    close(
        "one-key query gradient",
        &leaves[0].grad().unwrap().to_vec()?,
        &vec![0.0; n],
    );
    close(
        "one-key key gradient",
        &leaves[1].grad().unwrap().to_vec()?,
        &vec![0.0; n],
    );
    close(
        "masked value gradients",
        &leaves[2].grad().unwrap().to_vec()?,
        &first.iter().map(|x| x / n as f64).collect::<Vec<_>>(),
    );
    let incompatible = Tensor::from_slice(
        &[0.0; 30],
        [batch.of(1), time.of(5), feature.of(6)],
        &device,
    )?;
    assert!(axes.attend(&incompatible, &leaves[1], &leaves[2]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA; workspace check enables this"]
fn named_softmax_stability_and_mask_contract() -> Result<()> {
    let device = Device::cuda(0)?;
    let (key, row) = (Axis::new("key"), Axis::new("row"));
    let data = [1000., -1000., 1001., -1002., -1000., -1001.];
    let x = Tensor::from_slice(&data, [key.of(3), row.of(2)], &device)?.with_grad();
    let p = x.with_layout([row, key])?.softmax(key)?;
    let mut expected = vec![0.0; 6];
    for r in 0..2 {
        let maximum = (0..3)
            .map(|k| f64::from(data[k * 2 + r]))
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = (0..3)
            .map(|k| (f64::from(data[k * 2 + r]) - maximum).exp())
            .sum();
        for k in 0..3 {
            expected[k * 2 + r] = (f64::from(data[k * 2 + r]) - maximum).exp() / sum;
        }
    }
    close("stable nonfinal-axis softmax", &p.to_vec()?, &expected);
    let weights = [1., -2., 3., 5., -7., 11.];
    p.mul(&Tensor::from_slice(
        &weights,
        [key.of(3), row.of(2)],
        &device,
    )?)?
    .mean([row, key])?
    .backward()?;
    let grad: Vec<_> = (0..6)
        .map(|i| {
            let r = i % 2;
            let dot: f64 = (0..3)
                .map(|k| expected[k * 2 + r] * f64::from(weights[k * 2 + r]))
                .sum();
            expected[i] * (f64::from(weights[i]) - dot) / 6.0
        })
        .collect();
    close("softmax derivative", &x.grad().unwrap().to_vec()?, &grad);
    assert!(x.softmax(Axis::new("key")).is_err());
    assert!(x.causal_mask(key, key).is_err());
    assert!(x.causal_mask(row, key).is_err());
    let query = key.role("query");
    let scores = Tensor::from_slice(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [key.of(3), query.of(3)],
        &device,
    )?
    .with_grad();
    let masked = scores.with_layout([query, key])?.causal_mask(query, key)?;
    let actual = masked.to_vec()?;
    assert_eq!(actual[3], f32::NEG_INFINITY);
    assert_eq!(actual[6], f32::NEG_INFINITY);
    assert_eq!(actual[7], f32::NEG_INFINITY);
    let p = masked.softmax(key)?.to_vec()?;
    assert_eq!([p[3], p[6], p[7]], [0.0; 3]);
    for q in 0..3 {
        assert!(((0..3).map(|k| p[k * 3 + q]).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    // This shape crossed the former scalar-reduction contribution cap even
    // though the tensor itself is small enough for an ordinary attention batch.
    let rows = 2_560;
    let width = 81;
    let large = Tensor::from_slice(
        &vec![0.0; rows * width],
        [row.of(rows), key.of(width)],
        &device,
    )?
    .softmax(key)?
    .to_vec()?;
    for values in large.chunks_exact(width) {
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    Ok(())
}
