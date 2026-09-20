//! Direct scalar f64 convolution oracle: no im2col or tiled matrix multiplication.
use super::{CHANNELS, FILTER, IMAGE, PATCH, Result, SIDE, SPATIAL};

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
    pub conv_weight: Vec<f32>,
    pub conv_bias: Vec<f32>,
    pub head_weight: Vec<f32>,
    pub head_bias: Vec<f32>,
}

impl Weights {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut conv_weight = vec![0.0; PATCH * CHANNELS];
        for weight in &mut conv_weight[..FILTER * FILTER * CHANNELS] {
            *weight = rng.uniform() * (6.0 / (FILTER * FILTER) as f32).sqrt();
        }
        Self {
            conv_weight,
            conv_bias: (0..CHANNELS).map(|_| rng.uniform() * 0.1).collect(),
            head_weight: (0..CHANNELS)
                .map(|_| rng.uniform() * (3.0 / CHANNELS as f32).sqrt())
                .collect(),
            head_bias: vec![rng.uniform() * 0.1],
        }
    }
}

pub fn dataset(rows: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    assert!(rows.is_multiple_of(2));
    let mut rng = Rng::new(seed);
    let mut images = vec![0.0; rows * IMAGE * IMAGE];
    let mut labels = vec![0.0; rows];
    for (n, label) in labels.iter_mut().enumerate() {
        *label = (n % 2) as f32; // Balanced: horizontal = 0, vertical = 1.
        let position = 2 + ((rng.uniform() + 1.0) * 3.0) as usize;
        let amplitude = 0.8 + 0.2 * rng.uniform();
        let background = 0.05 * rng.uniform();
        for y in 0..IMAGE {
            for x in 0..IMAGE {
                let stripe = if *label == 1.0 {
                    x == position
                } else {
                    y == position
                };
                images[(n * IMAGE + y) * IMAGE + x] =
                    background + 0.08 * rng.uniform() + if stripe { amplitude } else { 0.0 };
            }
        }
    }
    (images, labels)
}

pub struct Gradients {
    pub activation: Vec<f64>,
    pub pooled: Vec<f64>,
    pub logits: Vec<f64>,
    pub dconv_weight: Vec<f64>,
    pub dconv_bias: Vec<f64>,
    pub dhead_weight: Vec<f64>,
    pub dhead_bias: Vec<f64>,
}

fn forward(w: &Weights, images: &[f32]) -> Gradients {
    let rows = images.len() / (IMAGE * IMAGE);
    let mut result = Gradients {
        activation: vec![0.0; rows * SPATIAL * CHANNELS],
        pooled: vec![0.0; rows * CHANNELS],
        logits: vec![0.0; rows],
        dconv_weight: vec![0.0; PATCH * CHANNELS],
        dconv_bias: vec![0.0; CHANNELS],
        dhead_weight: vec![0.0; CHANNELS],
        dhead_bias: vec![0.0],
    };
    for n in 0..rows {
        for oy in 0..SIDE {
            for ox in 0..SIDE {
                for c in 0..CHANNELS {
                    let mut value = f64::from(w.conv_bias[c]);
                    for ky in 0..FILTER {
                        for kx in 0..FILTER {
                            let pixel = images[(n * IMAGE + oy + ky) * IMAGE + ox + kx];
                            value += f64::from(pixel)
                                * f64::from(w.conv_weight[(ky * FILTER + kx) * CHANNELS + c]);
                        }
                    }
                    let activation = value.max(0.0);
                    result.activation[((n * SIDE + oy) * SIDE + ox) * CHANNELS + c] = activation;
                    result.pooled[n * CHANNELS + c] += activation / SPATIAL as f64;
                }
            }
        }
        result.logits[n] = f64::from(w.head_bias[0]);
        for c in 0..CHANNELS {
            result.logits[n] += result.pooled[n * CHANNELS + c] * f64::from(w.head_weight[c]);
        }
    }
    result
}

pub fn forward_backward(w: &Weights, images: &[f32], labels: &[f32]) -> Gradients {
    let mut result = forward(w, images);
    for (n, &label) in labels.iter().enumerate() {
        let z = result.logits[n];
        let probability = if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            z.exp() / (1.0 + z.exp())
        };
        let dz = (probability - f64::from(label)) / labels.len() as f64;
        result.dhead_bias[0] += dz;
        for c in 0..CHANNELS {
            result.dhead_weight[c] += result.pooled[n * CHANNELS + c] * dz;
            let dh = dz * f64::from(w.head_weight[c]) / SPATIAL as f64;
            for oy in 0..SIDE {
                for ox in 0..SIDE {
                    if result.activation[((n * SIDE + oy) * SIDE + ox) * CHANNELS + c] <= 0.0 {
                        continue;
                    }
                    result.dconv_bias[c] += dh;
                    for ky in 0..FILTER {
                        for kx in 0..FILTER {
                            let pixel = images[(n * IMAGE + oy + ky) * IMAGE + ox + kx];
                            result.dconv_weight[(ky * FILTER + kx) * CHANNELS + c] +=
                                f64::from(pixel) * dh;
                        }
                    }
                }
            }
        }
    }
    result
}

fn loss(w: &Weights, images: &[f32], labels: &[f32]) -> f64 {
    let result = forward(w, images);
    result
        .logits
        .iter()
        .zip(labels)
        .map(|(&z, &y)| z.max(0.0) - z * f64::from(y) + (-z.abs()).exp().ln_1p())
        .sum::<f64>()
        / labels.len() as f64
}

pub fn finite_difference_check(w: &Weights, x: &[f32], y: &[f32], grads: &Gradients) -> Result<()> {
    let mut checked = 0;
    let mut max_error = 0.0_f64;
    for (group, analytic) in [
        &grads.dconv_weight,
        &grads.dconv_bias,
        &grads.dhead_weight,
        &grads.dhead_bias,
    ]
    .iter()
    .enumerate()
    {
        let mut indices = vec![
            0,
            analytic.len() / 3,
            analytic.len() / 2,
            analytic.len() - 1,
        ];
        indices.sort_unstable();
        indices.dedup();
        for index in indices {
            let mut plus = w.clone();
            let mut minus = w.clone();
            let (p, m) = match group {
                0 => (&mut plus.conv_weight, &mut minus.conv_weight),
                1 => (&mut plus.conv_bias, &mut minus.conv_bias),
                2 => (&mut plus.head_weight, &mut minus.head_weight),
                _ => (&mut plus.head_bias, &mut minus.head_bias),
            };
            // Small enough to stay in the local ReLU region for this fixture.
            p[index] += 1e-6;
            m[index] -= 1e-6;
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
