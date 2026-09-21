//! The original regression experiment, expressed through the named-axis library.
use axis::prelude::*;
use std::{env, time::Instant};

const INPUT: usize = 16;
const HIDDEN: usize = 32;
const OUTPUT: usize = 16;
const BATCH: usize = 256;
// Keep data and initial weights identical to the standalone trainer and its oracle.
#[allow(dead_code)]
mod reference;

#[cfg(test)]
mod tests;

fn main() -> Result<()> {
    let mut steps = 500;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => {
                steps = args
                    .next()
                    .ok_or("--steps requires a value")?
                    .parse::<usize>()?
            }
            "--smoke" => steps = 10,
            "--help" | "-h" => {
                println!(
                    "axis-mlp [--steps N | --smoke]\nNamed-axis cuTile MLP; deterministic teacher data, seed 42, SGD 0.5."
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
    let (batch, input, hidden, output) = (
        Axis::new("batch"),
        Axis::new("input"),
        Axis::new("hidden"),
        Axis::new("output"),
    );
    let (train_x, train_y) = reference::dataset(BATCH, 1001);
    let (valid_x, valid_y) = reference::dataset(BATCH, 2002);
    let x = Tensor::from_slice(&train_x, [batch.of(BATCH), input.of(INPUT)], &device)?;
    let y = Tensor::from_slice(&train_y, [batch.of(BATCH), output.of(OUTPUT)], &device)?;
    let vx = Tensor::from_slice(&valid_x, [batch.of(BATCH), input.of(INPUT)], &device)?;
    let vy = Tensor::from_slice(&valid_y, [batch.of(BATCH), output.of(OUTPUT)], &device)?;
    let mut model = Sequential::new((
        Linear::new(input, hidden.of(HIDDEN)),
        ReLU,
        Linear::new(hidden, output.of(OUTPUT)),
    ));
    model.build(x.shape(), &device, 42)?;
    let weights = reference::Weights::new(42);
    model.parameter("0.weight")?.set_values(&weights.w1)?;
    model.parameter("0.bias")?.set_values(&weights.b1)?;
    model.parameter("2.weight")?.set_values(&weights.w2)?;
    model.parameter("2.bias")?.set_values(&weights.b2)?;
    let mut trainer = Trainer::new(SGD::new(0.5)?);
    let loss = |x: &Tensor, y: &Tensor| -> Result<Tensor> {
        model.forward(x)?.squared_error(y)?.mean([batch, output])
    };
    let initial = loss(&x, &y)?.item()?;
    let initial_valid = loss(&vx, &vy)?.item()?;
    println!("initial train_mse={initial:.8} validation_mse={initial_valid:.8}");
    let mut train_learning = LearningProgress::new(
        LearningDirection::Decrease,
        LearningLimits::any_improvement(),
        LearningObservation::new(
            "mean squared error",
            "fixed training set",
            "optimizer steps",
            0,
            f64::from(initial),
        )?,
    )?;
    let mut validation_learning = LearningProgress::new(
        LearningDirection::Decrease,
        LearningLimits::any_improvement(),
        LearningObservation::new(
            "mean squared error",
            "fixed validation set",
            "optimizer steps",
            0,
            f64::from(initial_valid),
        )?,
    )?;
    let started = Instant::now();
    for _ in 0..steps {
        let report = trainer.step(&mut model, |model| {
            model.forward(&x)?.squared_error(&y)?.mean([batch, output])
        })?;
        let step = report.step();
        if step == 1 || step % 100 == 0 || step == steps {
            println!(
                "step={step} pre_update_train_mse={:.8}",
                report.pre_update_loss()?
            );
        }
    }
    let final_loss = model
        .forward(&x)?
        .squared_error(&y)?
        .mean([batch, output])?
        .item()?;
    let final_valid = model
        .forward(&vx)?
        .squared_error(&vy)?
        .mean([batch, output])?
        .item()?;
    println!(
        "final train_mse={final_loss:.8} validation_mse={final_valid:.8} elapsed_s={:.2}",
        started.elapsed().as_secs_f64()
    );
    let terminal_step = u64::try_from(steps)?;
    train_learning.observe(LearningObservation::new(
        "mean squared error",
        "fixed training set",
        "optimizer steps",
        terminal_step,
        f64::from(final_loss),
    )?)?;
    validation_learning.observe(LearningObservation::new(
        "mean squared error",
        "fixed validation set",
        "optimizer steps",
        terminal_step,
        f64::from(final_valid),
    )?)?;
    println!("{}", train_learning.assert_learning()?);
    println!("{}", validation_learning.assert_learning()?);
    println!("PASS: named-axis MLP learned on cuTile");
    Ok(())
}
