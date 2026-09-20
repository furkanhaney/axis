//! Learn the running mean of a sequence with one causal multi-head attention block.
use axis::prelude::*;
use std::env;

mod attention;
use attention::{AttentionAxes, CausalAttention};
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

const BATCH: usize = 16;
const TIME: usize = 5;
const HEADS: usize = 2;
const HEAD_FEATURE: usize = 3;
const FEATURES: usize = HEADS * HEAD_FEATURE;

fn dataset(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = seed.max(1);
    let x: Vec<_> = (0..BATCH * TIME * FEATURES)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / (1_u32 << 23) as f32 - 1.0
        })
        .collect();
    let mut y = vec![0.0; x.len()];
    for b in 0..BATCH {
        for t in 0..TIME {
            for f in 0..FEATURES {
                y[(b * TIME + t) * FEATURES + f] = (0..=t)
                    .map(|k| x[(b * TIME + k) * FEATURES + f])
                    .sum::<f32>()
                    / (t + 1) as f32;
            }
        }
    }
    (x, y)
}

fn main() -> Result<()> {
    let mut steps = 200;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => steps = 10,
            "--steps" => {
                steps = args
                    .next()
                    .ok_or("--steps requires a value")?
                    .parse::<usize>()?
            }
            "--help" | "-h" => {
                println!(
                    "axis-attention [--steps N | --smoke]\nLearn sequence prefix means with causal attention, seed 42, SGD 0.3."
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
    let (batch, time, feature, head, head_feature) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("feature"),
        Axis::new("head"),
        Axis::new("head_feature"),
    );
    let dims = [batch.of(BATCH), time.of(TIME), feature.of(FEATURES)];
    let (train_x, train_y) = dataset(11);
    let (valid_x, valid_y) = dataset(29);
    let x = Tensor::from_slice(&train_x, dims, &device)?;
    let y = Tensor::from_slice(&train_y, dims, &device)?;
    let vx = Tensor::from_slice(&valid_x, dims, &device)?;
    let vy = Tensor::from_slice(&valid_y, dims, &device)?;
    let mut model = CausalAttention::new(AttentionAxes::new(
        time,
        feature,
        head.of(HEADS),
        head_feature.of(HEAD_FEATURE),
    ))?;
    model.build(x.shape(), &device, 42)?;
    let mut trainer = Trainer::new(SGD::new(0.3)?);
    let loss = |model: &CausalAttention, x: &Tensor, y: &Tensor| {
        model
            .forward(x)?
            .squared_error(y)?
            .mean([batch, time, feature])?
            .item()
    };
    let initial = loss(&model, &x, &y)?;
    let initial_valid = loss(&model, &vx, &vy)?;
    println!("initial train_mse={initial:.2e} validation_mse={initial_valid:.2e}");
    for _ in 0..steps {
        let report = trainer.step(&mut model, |model| {
            model
                .forward(&x)?
                .squared_error(&y)?
                .mean([batch, time, feature])
        })?;
        let step = report.step();
        if step == 1 || step % 50 == 0 || step == steps {
            println!(
                "step={step} pre_update_train_mse={:.2e}",
                report.pre_update_loss()?
            );
        }
    }
    let final_loss = loss(&model, &x, &y)?;
    let final_valid = loss(&model, &vx, &vy)?;
    println!("final train_mse={final_loss:.2e} validation_mse={final_valid:.2e}");
    if !final_loss.is_finite()
        || !final_valid.is_finite()
        || final_loss >= initial
        || final_valid >= initial_valid
    {
        return Err("attention training did not improve both losses".into());
    }
    println!("PASS: causal attention learned prefix means on cuTile");
    Ok(())
}
