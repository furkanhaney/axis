use super::{HIDDEN, INPUT, OUTPUT, reference};
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
#[ignore = "requires CUDA; scripts/check.sh runs these explicitly"]
fn mlp_matches_retained_f64_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, input, hidden, output) = (
        Axis::new("batch"),
        Axis::new("input"),
        Axis::new("hidden"),
        Axis::new("output"),
    );
    let weights = reference::Weights::new(7);
    let (values, targets) = reference::dataset(32, 11);
    let oracle = reference::forward_backward(&weights, &values, &targets);
    reference::finite_difference_check(&weights, &values, &targets, &oracle)?;
    let x = Tensor::from_slice(&values, [batch.of(32), input.of(INPUT)], &device)?.with_grad();
    let y = Tensor::from_slice(&targets, [batch.of(32), output.of(OUTPUT)], &device)?;
    let mut model = Sequential::new((
        Linear::new(input, hidden.of(HIDDEN)),
        ReLU,
        Linear::new(hidden, output.of(OUTPUT)),
    ));
    model.build(x.shape(), &device, 7)?;
    let params =
        ["0.weight", "0.bias", "2.weight", "2.bias"].map(|name| model.parameter(name).unwrap());
    let initial = [&weights.w1, &weights.b1, &weights.w2, &weights.b2];
    for (p, v) in params.iter().zip(initial) {
        p.set_values(v)?;
    }
    let prediction = model.forward(&x)?;
    close("MLP predictions", &prediction.to_vec()?, &oracle.prediction);
    let loss = prediction.squared_error(&y)?.mean([batch, output])?;
    loss.backward()?;
    assert!(loss.backward().is_err());
    let gradients = [&oracle.dw1, &oracle.db1, &oracle.dw2, &oracle.db2];
    for (i, (p, g)) in params.iter().zip(gradients).enumerate() {
        close(
            &format!("MLP gradient {i}"),
            &p.grad().unwrap().to_vec()?,
            g,
        );
    }
    let mut dx = vec![0.0; values.len()];
    for r in 0..32 {
        for h in 0..HIDDEN {
            let pre = f64::from(weights.b1[h])
                + (0..INPUT)
                    .map(|i| {
                        f64::from(values[r * INPUT + i]) * f64::from(weights.w1[i * HIDDEN + h])
                    })
                    .sum::<f64>();
            if pre > 0.0 {
                let dh = (0..OUTPUT)
                    .map(|o| {
                        2.0 * (oracle.prediction[r * OUTPUT + o]
                            - f64::from(targets[r * OUTPUT + o]))
                            / targets.len() as f64
                            * f64::from(weights.w2[h * OUTPUT + o])
                    })
                    .sum::<f64>();
                for i in 0..INPUT {
                    dx[r * INPUT + i] += dh * f64::from(weights.w1[i * HIDDEN + h]);
                }
            }
        }
    }
    close("MLP input gradients", &x.grad().unwrap().to_vec()?, &dx);
    SGD::new(0.5)?.step(&mut model)?;
    for (i, ((p, v), g)) in params.iter().zip(initial).zip(gradients).enumerate() {
        let expected: Vec<_> = v
            .iter()
            .zip(g)
            .map(|(&v, &g)| f64::from(v) - 0.5 * g)
            .collect();
        close(&format!("MLP update {i}"), &p.tensor().to_vec()?, &expected);
    }
    Ok(())
}
