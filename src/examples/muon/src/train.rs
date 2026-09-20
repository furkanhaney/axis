use axis::prelude::*;
use std::env;

const INPUTS: usize = 4;
const HIDDEN: usize = 8;
const OUTPUTS: usize = 2;
const BATCH: usize = 128;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn dataset(rows: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let inputs: Vec<_> = (0..rows * INPUTS).map(|_| rng.uniform()).collect();
    let mut targets = Vec::with_capacity(rows * OUTPUTS);
    for row in inputs.chunks_exact(INPUTS) {
        let hidden = (0.8 * row[0] - 0.4 * row[1] + 0.6 * row[2]).max(0.0);
        targets.push(hidden + 0.25 * row[3]);
        targets.push(-0.5 * hidden + 0.3 * row[1] - 0.2 * row[3]);
    }
    (inputs, targets)
}

fn model(input: Axis, hidden: Axis, output: Axis) -> Sequential {
    Sequential::new((
        Linear::new(input, hidden.of(HIDDEN)),
        ReLU,
        Linear::new(hidden, output.of(OUTPUTS)),
    ))
}

fn mse(model: &Sequential, x: &Tensor, y: &Tensor, batch: Axis, output: Axis) -> Result<f32> {
    model
        .forward(x)?
        .squared_error(y)?
        .mean([batch, output])?
        .item()
}

fn main() -> Result<()> {
    let mut steps = 100;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--steps" => steps = args.next().ok_or("--steps requires a value")?.parse()?,
            "--smoke" => steps = 20,
            "--help" | "-h" => {
                println!("axis-muon-acceptance [--steps N | --smoke]");
                return Ok(());
            }
            _ => return Err(format!("unknown option {argument}").into()),
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
    let (train_inputs, train_targets) = dataset(BATCH, 701);
    let (validation_inputs, validation_targets) = dataset(BATCH, 1701);
    let train_x = Tensor::from_slice(&train_inputs, [batch.of(BATCH), input.of(INPUTS)], &device)?;
    let train_y = Tensor::from_slice(
        &train_targets,
        [batch.of(BATCH), output.of(OUTPUTS)],
        &device,
    )?;
    let validation_x = Tensor::from_slice(
        &validation_inputs,
        [batch.of(BATCH), input.of(INPUTS)],
        &device,
    )?;
    let validation_y = Tensor::from_slice(
        &validation_targets,
        [batch.of(BATCH), output.of(OUTPUTS)],
        &device,
    )?;

    let mut adamw_model = model(input, hidden, output);
    let mut muon_model = model(input, hidden, output);
    adamw_model.build(train_x.shape(), &device, 42)?;
    muon_model.build(train_x.shape(), &device, 42)?;

    let adamw_baseline = mse(&adamw_model, &validation_x, &validation_y, batch, output)?;
    let muon_baseline = mse(&muon_model, &validation_x, &validation_y, batch, output)?;
    let mut adamw_trainer = Trainer::new(AdamW::new(0.02, 0.001)?);
    let selected = muon_model.parameter("0.weight")?;
    let mut muon_trainer = Trainer::new(MuonWithAuxAdamW::new(
        [MuonMatrix::axis_linear(&selected)],
        0.05,
        0.02,
        0.001,
    )?);

    for _ in 0..steps {
        adamw_trainer.step(&mut adamw_model, |model| {
            model
                .forward(&train_x)?
                .squared_error(&train_y)?
                .mean([batch, output])
        })?;
        muon_trainer.step(&mut muon_model, |model| {
            model
                .forward(&train_x)?
                .squared_error(&train_y)?
                .mean([batch, output])
        })?;
    }

    let adamw_final = mse(&adamw_model, &validation_x, &validation_y, batch, output)?;
    let muon_final = mse(&muon_model, &validation_x, &validation_y, batch, output)?;
    let terminal = u64::try_from(steps)?;
    for (optimizer, baseline, final_value) in [
        ("AdamW", adamw_baseline, adamw_final),
        ("MuonWithAuxAdamW", muon_baseline, muon_final),
    ] {
        let mut progress = LearningProgress::new(
            LearningDirection::Decrease,
            LearningLimits::any_improvement(),
            LearningObservation::new(
                "mean squared error",
                format!("{optimizer} fixed validation population"),
                "optimizer steps",
                0,
                f64::from(baseline),
            )?,
        )?;
        progress.observe(LearningObservation::new(
            "mean squared error",
            format!("{optimizer} fixed validation population"),
            "optimizer steps",
            terminal,
            f64::from(final_value),
        )?)?;
        println!("optimizer={optimizer} baseline_mse={baseline:.8} final_mse={final_value:.8}");
        println!("{}", progress.assert_learning()?);
    }
    println!("PASS: AdamW and explicitly partitioned Muon both learned");
    Ok(())
}
