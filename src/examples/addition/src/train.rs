//! Learn addition from an inexhaustible generated stream under an IDR assertion.
use axis::prelude::*;
use std::{env, time::Instant};

const BATCH: usize = 256;
const HIDDEN: usize = 16;
const TRAIN_SEED: u64 = 0xadd1_7100;
const EVAL_SEED: u64 = 0xe7a1_0000;

fn tensors(
    samples: &[AdditionSample],
    batch: Axis,
    input: Axis,
    output: Axis,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut operands = Vec::with_capacity(samples.len() * 2);
    let mut sums = Vec::with_capacity(samples.len());
    for sample in samples {
        operands.extend([sample.left, sample.right]);
        sums.push(sample.sum);
    }
    Ok((
        Tensor::from_slice(&operands, [batch.of(samples.len()), input.of(2)], device)?,
        Tensor::from_slice(&sums, [batch.of(samples.len()), output.of(1)], device)?,
    ))
}

fn monotonicity_audit(
    model: &impl Module,
    contexts: &[AdditionSample],
    batch: Axis,
    input: Axis,
    output: Axis,
    device: &Device,
) -> Result<(EmpiricalMonotonicityReceipt, EmpiricalMonotonicityReceipt)> {
    fn bounds(value: f32) -> (f32, f32) {
        ((value - 0.05).max(-1.0), (value + 0.05).min(1.0))
    }
    fn predictions(
        model: &impl Module,
        samples: &[AdditionSample],
        batch: Axis,
        input: Axis,
        output: Axis,
        device: &Device,
    ) -> Result<Vec<f32>> {
        let (inputs, _) = tensors(samples, batch, input, output, device)?;
        model.forward(&inputs)?.to_vec()
    }

    let mut left_lower = Vec::with_capacity(contexts.len());
    let mut left_upper = Vec::with_capacity(contexts.len());
    let mut right_lower = Vec::with_capacity(contexts.len());
    let mut right_upper = Vec::with_capacity(contexts.len());
    for sample in contexts {
        let (lower, upper) = bounds(sample.left);
        left_lower.push(AdditionSample {
            left: lower,
            right: sample.right,
            sum: lower + sample.right,
        });
        left_upper.push(AdditionSample {
            left: upper,
            right: sample.right,
            sum: upper + sample.right,
        });
        let (lower, upper) = bounds(sample.right);
        right_lower.push(AdditionSample {
            left: sample.left,
            right: lower,
            sum: sample.left + lower,
        });
        right_upper.push(AdditionSample {
            left: sample.left,
            right: upper,
            sum: sample.left + upper,
        });
    }

    let left_lower_predictions = predictions(model, &left_lower, batch, input, output, device)?;
    let left_upper_predictions = predictions(model, &left_upper, batch, input, output, device)?;
    let right_lower_predictions = predictions(model, &right_lower, batch, input, output, device)?;
    let right_upper_predictions = predictions(model, &right_upper, batch, input, output, device)?;
    let limits = MonotonicityLimits::strict(1e-6)?;
    let mut left = EmpiricalMonotonicity::new(
        "left operand",
        "predicted sum",
        MonotoneDirection::Increasing,
        limits,
    )?;
    let mut right = EmpiricalMonotonicity::new(
        "right operand",
        "predicted sum",
        MonotoneDirection::Increasing,
        limits,
    )?;
    left.observe_ordered_batch(
        left_lower
            .iter()
            .zip(&left_upper)
            .zip(left_lower_predictions.iter().zip(&left_upper_predictions))
            .map(|((lower, upper), (lower_output, upper_output))| {
                (
                    f64::from(lower.left),
                    f64::from(upper.left),
                    f64::from(*lower_output),
                    f64::from(*upper_output),
                )
            }),
    )?;
    right.observe_ordered_batch(
        right_lower
            .iter()
            .zip(&right_upper)
            .zip(right_lower_predictions.iter().zip(&right_upper_predictions))
            .map(|((lower, upper), (lower_output, upper_output))| {
                (
                    f64::from(lower.right),
                    f64::from(upper.right),
                    f64::from(*lower_output),
                    f64::from(*upper_output),
                )
            }),
    )?;
    Ok((left.receipt(), right.receipt()))
}

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
                    "addition-train [--steps N | --smoke]\nFresh generated addition samples, named axes, IDR assertion, SGD 0.25."
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
    let mut train = DataLoader::new(AdditionDataset::new(TRAIN_SEED), BATCH)?
        .assert_idr(IdrLimits::generated(0.0)?)?;
    let mut evaluation = DataLoader::new(AdditionDataset::new(EVAL_SEED), BATCH)?
        .assert_idr(IdrLimits::generated(0.0)?)?;
    let evaluation = evaluation
        .next_batch()?
        .expect("generated evaluation source never ends");
    let mut disjoint = TrainEvalDisjoint::new();
    disjoint.observe_evaluation(evaluation.sample_ids.iter().copied())?;
    let (eval_x, eval_y) = tensors(&evaluation.samples, batch, input, output, &device)?;

    let mut model = Sequential::new((
        Linear::new(input, hidden.of(HIDDEN)),
        ReLU,
        Linear::new(hidden, output.of(1)),
    ));
    model.build(&Shape::new([batch.of(BATCH), input.of(2)])?, &device, 42)?;
    let initial_eval = model
        .forward(&eval_x)?
        .squared_error(&eval_y)?
        .mean([batch, output])?
        .item()?;
    println!("initial evaluation_mse={initial_eval:.8}");

    let mut trainer = Trainer::new(SGD::new(0.25)?);
    let started = Instant::now();
    let mut final_receipt = None;
    for _ in 0..steps {
        let fresh = train
            .next_batch()?
            .expect("generated training source never ends");
        disjoint.observe_train(fresh.sample_ids.iter().copied())?;
        let (x, y) = tensors(&fresh.samples, batch, input, output, &device)?;
        let report = trainer.step(&mut model, |model| {
            model.forward(&x)?.squared_error(&y)?.mean([batch, output])
        })?;
        let step = report.step();
        if step == 1 || step % 100 == 0 || step == steps {
            println!(
                "step={step} samples={} pre_update_mse={:.8}",
                train.samples_delivered(),
                report.pre_update_loss()?
            );
        }
        final_receipt = Some(fresh.regime);
    }

    let final_eval = model
        .forward(&eval_x)?
        .squared_error(&eval_y)?
        .mean([batch, output])?
        .item()?;
    println!(
        "final evaluation_mse={final_eval:.8} elapsed_s={:.2}",
        started.elapsed().as_secs_f64()
    );
    println!("{}", final_receipt.expect("positive step count"));
    println!("{}", disjoint.receipt());
    let (left_monotonicity, right_monotonicity) =
        monotonicity_audit(&model, &evaluation.samples, batch, input, output, &device)?;
    println!("{left_monotonicity}");
    println!("{right_monotonicity}");
    if !final_eval.is_finite() || final_eval >= initial_eval {
        return Err("training did not improve held-out generated addition loss".into());
    }
    println!("PASS: learned addition from fresh samples without epochs");
    Ok(())
}
