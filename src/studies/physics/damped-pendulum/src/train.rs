//! Learn a nonlinear damped-pendulum trajectory from its differential law.
use axis::prelude::*;
use std::{collections::HashSet, env, time::Instant};

mod reference;
use reference::Pendulum;

const BATCH: usize = 64;
const HIDDEN: usize = 32;
const END_TIME: f32 = 2.0;
const DIFFERENCE_STEP: f32 = 1e-2;
const TRAIN_SEED: u64 = 0x5049_4e4e;
const EVALUATION_POINTS: usize = 257;

const SYSTEM: Pendulum = Pendulum {
    gravity: 9.81,
    length: 1.0,
    damping_rate: 0.2,
    initial_angle: 0.7,
    initial_angular_velocity: 0.0,
};

struct CollocationStream {
    seed: u64,
    next_id: u64,
    excluded_coordinates: HashSet<u32>,
}

impl CollocationStream {
    fn new(seed: u64) -> Self {
        let excluded_coordinates = evaluation_times()
            .into_iter()
            .flat_map(|time| {
                [
                    (time - DIFFERENCE_STEP).to_bits(),
                    time.to_bits(),
                    (time + DIFFERENCE_STEP).to_bits(),
                ]
            })
            .collect();
        Self {
            seed,
            next_id: 0,
            excluded_coordinates,
        }
    }

    fn mix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

impl DataSource for CollocationStream {
    type Item = f32;

    fn next_sample(&mut self) -> Result<Option<Sample<Self::Item>>> {
        loop {
            let draw = self.next_id;
            self.next_id = self
                .next_id
                .checked_add(1)
                .ok_or("collocation identity space exhausted")?;
            let unit = (Self::mix(self.seed ^ draw) >> 40) as f32 / (1_u32 << 24) as f32;
            let time = DIFFERENCE_STEP + unit * (END_TIME - 2.0 * DIFFERENCE_STEP);
            let support = [
                (time - DIFFERENCE_STEP).to_bits(),
                time.to_bits(),
                (time + DIFFERENCE_STEP).to_bits(),
            ];
            if support
                .iter()
                .any(|coordinate| self.excluded_coordinates.contains(coordinate))
            {
                continue;
            }
            return Ok(Some(Sample {
                id: (u128::from(self.seed) << 64) | u128::from(draw),
                value: time,
            }));
        }
    }

    fn available(&self) -> Option<usize> {
        None
    }
}

#[derive(Clone, Copy)]
struct Axes {
    batch: Axis,
    input: Axis,
    hidden: Axis,
    solution: Axis,
}

fn times(values: &[f32], axes: Axes, device: &Device) -> Result<Tensor> {
    Tensor::from_slice(
        values,
        [axes.batch.of(values.len()), axes.input.of(1)],
        device,
    )
}

/// Both initial conditions are structural: theta(t) = theta(0) + omega(0)t + t²N(t).
fn angle(model: &impl Module, time: &Tensor, axes: Axes, initial_angle: &Tensor) -> Result<Tensor> {
    let correction = model.forward(time)?.select(axes.solution, 0)?;
    let time = time.select(axes.input, 0)?;
    correction
        .mul(&time)?
        .mul(&time)?
        .add(&time.scale(SYSTEM.initial_angular_velocity as f32)?)?
        .add(initial_angle)
}

fn shifted_angles(
    model: &impl Module,
    centers: &[f32],
    axes: Axes,
    initial_angle: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let lower = centers
        .iter()
        .map(|time| time - DIFFERENCE_STEP)
        .collect::<Vec<_>>();
    let upper = centers
        .iter()
        .map(|time| time + DIFFERENCE_STEP)
        .collect::<Vec<_>>();
    Ok((
        angle(model, &times(&lower, axes, device)?, axes, initial_angle)?,
        angle(model, &times(centers, axes, device)?, axes, initial_angle)?,
        angle(model, &times(&upper, axes, device)?, axes, initial_angle)?,
    ))
}

fn dynamics_residual(
    model: &impl Module,
    centers: &[f32],
    axes: Axes,
    initial_angle: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let (lower, center, upper) = shifted_angles(model, centers, axes, initial_angle, device)?;
    let stencil = CentralDifference::new(Axis::new("time"), DIFFERENCE_STEP)?;
    let velocity = stencil.first(&lower, &upper)?;
    let acceleration = stencil.second(&lower, &center, &upper)?;
    acceleration
        .add(&velocity.scale(SYSTEM.damping_rate as f32)?)?
        .add(
            &center
                .sin()?
                .scale((SYSTEM.gravity / SYSTEM.length) as f32)?,
        )
}

fn physics_loss(
    model: &impl Module,
    centers: &[f32],
    axes: Axes,
    initial_angle: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    dynamics_residual(model, centers, axes, initial_angle, device)?.mean_square(axes.batch)
}

fn evaluation_times() -> Vec<f32> {
    (0..EVALUATION_POINTS)
        .map(|index| END_TIME * index as f32 / (EVALUATION_POINTS - 1) as f32)
        .collect()
}

fn trajectory_rmse(
    model: &impl Module,
    axes: Axes,
    initial_angle: &Tensor,
    device: &Device,
) -> Result<f64> {
    let evaluation = evaluation_times();
    let (lower, center, upper) = shifted_angles(model, &evaluation, axes, initial_angle, device)?;
    let velocity = CentralDifference::new(Axis::new("time"), DIFFERENCE_STEP)?
        .first(&lower, &upper)?
        .to_vec()?;
    let predicted = center.to_vec()?;
    let squared_error = evaluation
        .iter()
        .zip(predicted.iter().zip(&velocity))
        .map(|(&time, (&angle, &velocity))| {
            let expected = SYSTEM.state_at(f64::from(time), 1e-3);
            (f64::from(angle) - expected[0]).powi(2) + (f64::from(velocity) - expected[1]).powi(2)
        })
        .sum::<f64>();
    Ok((squared_error / (evaluation.len() * 2) as f64).sqrt())
}

fn residual_receipts(
    model: &impl Module,
    axes: Axes,
    initial_angle: &Tensor,
    device: &Device,
) -> Result<(EmpiricalResidualReceipt, EmpiricalResidualReceipt)> {
    let centers = evaluation_times()[1..EVALUATION_POINTS - 1].to_vec();
    let dynamic = dynamics_residual(model, &centers, axes, initial_angle, device)?;
    let law = LawIdentity::new("nonlinear damped pendulum", "caliper-r5@1")?;
    let limits = ResidualLimits::strict(0.3)?;
    let evaluator = format!("second-order central difference; h={DIFFERENCE_STEP}");
    let region = format!("fixed held-out grid; 0 < t < {END_TIME} seconds");
    let mut initial_check = EmpiricalResidual::new(
        ResidualScope::new(
            law.clone(),
            "theta(0) = theta_0 and dtheta/dt(0) = omega_0",
            "initial condition at t = 0 seconds",
            evaluator.clone(),
        )?,
        ResidualLimits::strict(1e-4)?,
    );
    let mut dynamic_check = EmpiricalResidual::new(
        ResidualScope::new(
            law,
            "d2theta/dt2 + gamma*dtheta/dt + (g/L)*sin(theta) = 0",
            region,
            evaluator,
        )?,
        limits,
    );
    let zero = [0.0_f32];
    let (lower, center, upper) = shifted_angles(model, &zero, axes, initial_angle, device)?;
    let initial_velocity = CentralDifference::new(Axis::new("time"), DIFFERENCE_STEP)?
        .first(&lower, &upper)?
        .to_vec()?[0];
    let initial_angle_observed = center.to_vec()?[0];
    let initial_receipt = initial_check.observe_batch([
        f64::from(initial_angle_observed) - SYSTEM.initial_angle,
        f64::from(initial_velocity) - SYSTEM.initial_angular_velocity,
    ])?;
    let dynamic_receipt =
        dynamic_check.observe_batch(dynamic.to_vec()?.into_iter().map(f64::from))?;
    Ok((initial_receipt, dynamic_receipt))
}

fn main() -> Result<()> {
    let mut steps = 4_000;
    let mut smoke = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => {
                steps = args
                    .next()
                    .ok_or("--steps requires a value")?
                    .parse::<usize>()?;
            }
            "--smoke" => {
                steps = 5;
                smoke = true;
            }
            "--help" | "-h" => {
                println!("damped-pendulum [--steps N | --smoke]");
                return Ok(());
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if steps == 0 {
        return Err("steps must be positive".into());
    }

    let device = Device::cuda(0)?;
    let axes = Axes {
        batch: Axis::new("collocation"),
        input: Axis::new("coordinate"),
        hidden: Axis::new("hidden"),
        solution: Axis::new("solution"),
    };
    let initial_angle = Tensor::from_slice(&[SYSTEM.initial_angle as f32], [], &device)?;
    let mut model = Sequential::new((
        Linear::new(axes.input, axes.hidden.of(HIDDEN)),
        Tanh,
        Linear::new(axes.hidden, axes.hidden.of(HIDDEN)),
        Tanh,
        Linear::new(axes.hidden, axes.solution.of(1)),
    ));
    model.build(
        &Shape::new([axes.batch.of(BATCH), axes.input.of(1)])?,
        &device,
        42,
    )?;

    let initial_rmse = trajectory_rmse(&model, axes, &initial_angle, &device)?;
    println!("initial trajectory_rmse={initial_rmse:.8}");
    let mut learning = LearningProgress::new(
        LearningDirection::Decrease,
        LearningLimits::any_improvement(),
        LearningObservation::new(
            "state RMSE against independent f64 RK4",
            "fixed held-out time grid",
            "optimizer steps",
            0,
            initial_rmse,
        )?,
    )?;
    let mut loader = DataLoader::new(CollocationStream::new(TRAIN_SEED), BATCH)?
        .assert_idr(IdrLimits::generated(0.0)?)?;
    let mut disjoint = Disjointness::new(
        IdentityScheme::new(
            "pendulum-collocation-coordinate",
            "1",
            "f32 time-coordinate bits, including central-stencil support",
        )?,
        [
            PopulationSpec::streaming("training"),
            PopulationSpec::retained("evaluation"),
        ],
    )?;
    disjoint.observe(
        "evaluation",
        evaluation_times().into_iter().flat_map(|time| {
            [
                (time - DIFFERENCE_STEP).to_bits(),
                time.to_bits(),
                (time + DIFFERENCE_STEP).to_bits(),
            ]
        }),
    )?;
    let mut trainer = Trainer::new(Adam::new(1e-3)?);
    let started = Instant::now();
    let mut final_regime = None;
    for _ in 0..steps {
        let batch = loader.next_batch()?.expect("generated stream never ends");
        disjoint.observe(
            "training",
            batch.samples.iter().flat_map(|time| {
                [
                    (*time - DIFFERENCE_STEP).to_bits(),
                    time.to_bits(),
                    (*time + DIFFERENCE_STEP).to_bits(),
                ]
            }),
        )?;
        let report = trainer.step(&mut model, |model| {
            physics_loss(model, &batch.samples, axes, &initial_angle, &device)
        })?;
        let step = report.step();
        if step == 1 || step % 250 == 0 || step == steps {
            println!(
                "step={step} samples={} pre_update_physics_loss={:.8}",
                loader.samples_delivered(),
                report.pre_update_loss()?
            );
        }
        final_regime = Some(batch.regime);
    }

    let final_rmse = trajectory_rmse(&model, axes, &initial_angle, &device)?;
    learning.observe(LearningObservation::new(
        "state RMSE against independent f64 RK4",
        "fixed held-out time grid",
        "optimizer steps",
        u64::try_from(steps)?,
        final_rmse,
    )?)?;
    println!(
        "final trajectory_rmse={final_rmse:.8} elapsed_s={:.2}",
        started.elapsed().as_secs_f64()
    );
    println!("{}", final_regime.expect("at least one step"));
    println!("{}", disjoint.assert_disjoint()?);
    println!("{}", learning.assert_learning()?);
    if smoke {
        println!(
            "SMOKE PASS: training mechanics ran; scientific residual acceptance was not requested"
        );
        return Ok(());
    }
    let (initial_condition, dynamics) = residual_receipts(&model, axes, &initial_angle, &device)?;
    println!("{initial_condition}");
    println!("{dynamics}");
    println!("PASS: learned a damped-pendulum trajectory from sampled differential residuals");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_stencils_exclude_the_sealed_evaluation_support() -> Result<()> {
        let excluded = CollocationStream::new(TRAIN_SEED).excluded_coordinates;
        let mut stream = CollocationStream::new(TRAIN_SEED);
        let mut identities = HashSet::new();
        for _ in 0..100_000 {
            let sample = stream.next_sample()?.expect("generated stream");
            assert!(identities.insert(sample.id));
            assert!((DIFFERENCE_STEP..=END_TIME - DIFFERENCE_STEP).contains(&sample.value));
            for coordinate in [
                sample.value - DIFFERENCE_STEP,
                sample.value,
                sample.value + DIFFERENCE_STEP,
            ] {
                assert!(!excluded.contains(&coordinate.to_bits()));
            }
        }
        Ok(())
    }
}
