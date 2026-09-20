//! Scalar f64 oracle, independent of the GPU's tiled matrix operations.
use super::{HIDDEN, INPUT, OUTPUT, Result};

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0
    }
}

#[derive(Clone)]
pub struct Weights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}
impl Weights {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            w1: (0..INPUT * HIDDEN)
                .map(|_| rng.uniform() * (6.0 / INPUT as f32).sqrt())
                .collect(),
            b1: (0..HIDDEN).map(|_| rng.uniform() * 0.1).collect(),
            w2: (0..HIDDEN * OUTPUT)
                .map(|_| rng.uniform() * (3.0 / HIDDEN as f32).sqrt())
                .collect(),
            b2: (0..OUTPUT).map(|_| rng.uniform() * 0.1).collect(),
        }
    }
}

fn forward(w: &Weights, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let rows = x.len() / INPUT;
    let mut hidden = vec![0.0; rows * HIDDEN];
    let mut prediction = vec![0.0; rows * OUTPUT];
    for r in 0..rows {
        for h in 0..HIDDEN {
            let mut v = f64::from(w.b1[h]);
            for i in 0..INPUT {
                v += f64::from(x[r * INPUT + i]) * f64::from(w.w1[i * HIDDEN + h]);
            }
            hidden[r * HIDDEN + h] = v.max(0.0);
        }
        for o in 0..OUTPUT {
            let mut v = f64::from(w.b2[o]);
            for h in 0..HIDDEN {
                v += hidden[r * HIDDEN + h] * f64::from(w.w2[h * OUTPUT + o]);
            }
            prediction[r * OUTPUT + o] = v;
        }
    }
    (hidden, prediction)
}

pub fn dataset(rows: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let x: Vec<f32> = (0..rows * INPUT).map(|_| rng.uniform()).collect();
    // A fixed teacher produces a nonlinear, noise-free regression problem.
    // Train and validation differ in inputs, while the teacher stays fixed.
    let (_, y) = forward(&Weights::new(2026), &x);
    (x, y.into_iter().map(|v| v as f32).collect())
}

pub struct Gradients {
    pub prediction: Vec<f64>,
    pub dw1: Vec<f64>,
    pub db1: Vec<f64>,
    pub dw2: Vec<f64>,
    pub db2: Vec<f64>,
}

pub fn forward_backward(w: &Weights, x: &[f32], y: &[f32]) -> Gradients {
    let (hidden, prediction) = forward(w, x);
    let mut result = Gradients {
        prediction,
        dw1: vec![0.0; INPUT * HIDDEN],
        db1: vec![0.0; HIDDEN],
        dw2: vec![0.0; HIDDEN * OUTPUT],
        db2: vec![0.0; OUTPUT],
    };
    for r in 0..x.len() / INPUT {
        for o in 0..OUTPUT {
            let dy = 2.0 * (result.prediction[r * OUTPUT + o] - f64::from(y[r * OUTPUT + o]))
                / y.len() as f64;
            result.db2[o] += dy;
            for h in 0..HIDDEN {
                result.dw2[h * OUTPUT + o] += hidden[r * HIDDEN + h] * dy;
                if hidden[r * HIDDEN + h] > 0.0 {
                    let dh = dy * f64::from(w.w2[h * OUTPUT + o]);
                    result.db1[h] += dh;
                    for i in 0..INPUT {
                        result.dw1[i * HIDDEN + h] += f64::from(x[r * INPUT + i]) * dh;
                    }
                }
            }
        }
    }
    result
}

fn loss(w: &Weights, x: &[f32], y: &[f32]) -> f64 {
    let (_, p) = forward(w, x);
    p.iter()
        .zip(y)
        .map(|(&p, &y)| (p - f64::from(y)).powi(2))
        .sum::<f64>()
        / y.len() as f64
}

pub fn finite_difference_check(w: &Weights, x: &[f32], y: &[f32], grads: &Gradients) -> Result<()> {
    let mut checked = 0;
    let mut max_error = 0.0_f64;
    for (group, analytic) in [&grads.dw1, &grads.db1, &grads.dw2, &grads.db2]
        .iter()
        .enumerate()
    {
        for index in [
            0,
            analytic.len() / 3,
            analytic.len() / 2,
            analytic.len() - 1,
        ] {
            let mut plus = w.clone();
            let mut minus = w.clone();
            let (p, m) = match group {
                0 => (&mut plus.w1, &mut minus.w1),
                1 => (&mut plus.b1, &mut minus.b1),
                2 => (&mut plus.w2, &mut minus.w2),
                _ => (&mut plus.b2, &mut minus.b2),
            };
            p[index] += 1e-4;
            m[index] -= 1e-4;
            let delta = f64::from(p[index]) - f64::from(m[index]);
            let numeric = (loss(&plus, x, y) - loss(&minus, x, y)) / delta;
            let error = (numeric - analytic[index]).abs();
            if !error.is_finite() || error > 2e-5 + 2e-3 * analytic[index].abs() {
                return Err(format!(
                    "Finite difference group {group} index {index}: analytic={}, numeric={numeric}",
                    analytic[index]
                )
                .into());
            }
            max_error = max_error.max(error);
            checked += 1;
        }
    }
    println!("check finite differences: PASS {checked} parameters max_abs_error={max_error:.2e}");
    Ok(())
}
