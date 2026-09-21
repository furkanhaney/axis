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
const COLLOCATION_POPULATION: u64 = 1 << 22;
const MAX_ANGLE_RMSE: f64 = 0.01;
const MAX_ANGULAR_VELOCITY_RMSE: f64 = 0.02;

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
    available: usize,
}

impl CollocationStream {
    fn new(seed: u64) -> Self {
        let excluded_coordinates: HashSet<u32> = evaluation_times()
            .into_iter()
            .flat_map(|time| {
                [
                    (time - DIFFERENCE_STEP).to_bits(),
                    time.to_bits(),
                    (time + DIFFERENCE_STEP).to_bits(),
                ]
            })
            .collect();
        let excluded = (0..COLLOCATION_POPULATION)
            .filter(|draw| {
                let time = Self::coordinate(seed, *draw);
                [
                    (time - DIFFERENCE_STEP).to_bits(),
                    time.to_bits(),
                    (time + DIFFERENCE_STEP).to_bits(),
                ]
                .iter()
                .any(|coordinate| excluded_coordinates.contains(coordinate))
            })
            .count();
        Self {
            seed,
            next_id: 0,
            excluded_coordinates,
            available: usize::try_from(COLLOCATION_POPULATION).expect("population fits usize")
                - excluded,
        }
    }

    fn coordinate(seed: u64, draw: u64) -> f32 {
        let mask = COLLOCATION_POPULATION - 1;
        let lattice_index = draw.wrapping_mul(0x9e37_79b9).wrapping_add(seed) & mask;
        let unit = lattice_index as f32 / mask as f32;
        DIFFERENCE_STEP + unit * (END_TIME - 2.0 * DIFFERENCE_STEP)
    }
}

impl DataSource for CollocationStream {
    type Item = f32;

    fn next_sample(&mut self) -> Result<Option<Sample<Self::Item>>> {
        loop {
            if self.next_id == COLLOCATION_POPULATION {
                return Ok(None);
            }
            let draw = self.next_id;
            self.next_id = self
                .next_id
                .checked_add(1)
                .ok_or("collocation identity space exhausted")?;
            let time = Self::coordinate(self.seed, draw);
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
                id: u128::from(time.to_bits()),
                value: time,
            }));
        }
    }

    fn available(&self) -> Option<usize> {
        Some(self.available)
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
    let stencil = CentralDifference::new(axes.input, DIFFERENCE_STEP)?;
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

#[derive(Clone, Copy, Debug)]
struct TrajectoryErrors {
    angle_rmse: f64,
    angular_velocity_rmse: f64,
}

fn trajectory_errors(
    model: &impl Module,
    axes: Axes,
    initial_angle: &Tensor,
    device: &Device,
) -> Result<TrajectoryErrors> {
    let evaluation = evaluation_times();
    let (lower, center, upper) = shifted_angles(model, &evaluation, axes, initial_angle, device)?;
    let velocity = CentralDifference::new(axes.input, DIFFERENCE_STEP)?
        .first(&lower, &upper)?
        .to_vec()?;
    let predicted = center.to_vec()?;
    let (angle_squared_error, velocity_squared_error) =
        evaluation.iter().zip(predicted.iter().zip(&velocity)).fold(
            (0.0, 0.0),
            |(angle_error, velocity_error), (&time, (&angle, &velocity))| {
                let expected = SYSTEM.state_at(f64::from(time), 1e-3);
                (
                    angle_error + (f64::from(angle) - expected[0]).powi(2),
                    velocity_error + (f64::from(velocity) - expected[1]).powi(2),
                )
            },
        );
    Ok(TrajectoryErrors {
        angle_rmse: (angle_squared_error / evaluation.len() as f64).sqrt(),
        angular_velocity_rmse: (velocity_squared_error / evaluation.len() as f64).sqrt(),
    })
}

fn residual_receipts(
    model: &impl Module,
    axes: Axes,
    initial_angle: &Tensor,
    device: &Device,
) -> Result<(
    EmpiricalResidualReceipt,
    EmpiricalResidualReceipt,
    EmpiricalResidualReceipt,
)> {
    let centers = evaluation_times()[1..EVALUATION_POINTS - 1].to_vec();
    let dynamic = dynamics_residual(model, &centers, axes, initial_angle, device)?;
    let law = LawIdentity::new("nonlinear damped pendulum", "caliper-r5@1")?;
    let limits = ResidualLimits::strict(0.25)?;
    let evaluator = format!("fp32 second-order central difference; h={DIFFERENCE_STEP} seconds");
    let region = format!("fixed held-out grid; 0 < t < {END_TIME} seconds; residual unit rad/s^2");
    let mut initial_angle_check = EmpiricalResidual::new(
        ResidualScope::new(
            law.clone(),
            "theta(0) - theta_0 = 0 [rad]",
            "initial angle at t = 0 seconds",
            "direct fp32 evaluation",
        )?,
        ResidualLimits::strict(1e-6)?,
    );
    let mut initial_velocity_check = EmpiricalResidual::new(
        ResidualScope::new(
            law.clone(),
            "dtheta/dt(0) - omega_0 = 0 [rad/s]",
            "initial angular velocity at t = 0 seconds",
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
    let initial_velocity = CentralDifference::new(axes.input, DIFFERENCE_STEP)?
        .first(&lower, &upper)?
        .to_vec()?[0];
    let initial_angle_observed = center.to_vec()?[0];
    let initial_angle_receipt =
        initial_angle_check.observe(f64::from(initial_angle_observed) - SYSTEM.initial_angle)?;
    let initial_velocity_receipt = initial_velocity_check
        .observe(f64::from(initial_velocity) - SYSTEM.initial_angular_velocity)?;
    let dynamic_receipt =
        dynamic_check.observe_batch(dynamic.to_vec()?.into_iter().map(f64::from))?;
    Ok((
        initial_angle_receipt,
        initial_velocity_receipt,
        dynamic_receipt,
    ))
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
    run_experiment(steps, smoke)
}

fn run_experiment(steps: usize, smoke: bool) -> Result<()> {
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

    let initial_errors = trajectory_errors(&model, axes, &initial_angle, &device)?;
    println!(
        "initial angle_rmse_rad={:.8} angular_velocity_rmse_rad_s={:.8}",
        initial_errors.angle_rmse, initial_errors.angular_velocity_rmse
    );
    let progress_limits = if smoke {
        LearningLimits::any_improvement()
    } else {
        LearningLimits::new(0.0, Some(0.95))?
    };
    let mut angle_learning = LearningProgress::new(
        LearningDirection::Decrease,
        progress_limits,
        LearningObservation::new(
            "angle RMSE [rad] against independent f64 RK4",
            "fixed held-out time grid",
            "optimizer steps",
            0,
            initial_errors.angle_rmse,
        )?,
    )?;
    let mut velocity_learning = LearningProgress::new(
        LearningDirection::Decrease,
        progress_limits,
        LearningObservation::new(
            "angular velocity RMSE [rad/s] against independent f64 RK4",
            "fixed held-out time grid",
            "optimizer steps",
            0,
            initial_errors.angular_velocity_rmse,
        )?,
    )?;
    let mut loader = DataLoader::new(CollocationStream::new(TRAIN_SEED), BATCH)?
        .assert_idr(IdrLimits::fixed(0.1, 0.0)?)?;
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
        let batch = loader
            .next_batch()?
            .ok_or("collocation population exhausted before the training budget")?;
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

    let final_errors = trajectory_errors(&model, axes, &initial_angle, &device)?;
    angle_learning.observe(LearningObservation::new(
        "angle RMSE [rad] against independent f64 RK4",
        "fixed held-out time grid",
        "optimizer steps",
        u64::try_from(steps)?,
        final_errors.angle_rmse,
    )?)?;
    velocity_learning.observe(LearningObservation::new(
        "angular velocity RMSE [rad/s] against independent f64 RK4",
        "fixed held-out time grid",
        "optimizer steps",
        u64::try_from(steps)?,
        final_errors.angular_velocity_rmse,
    )?)?;
    println!(
        "final angle_rmse_rad={:.8} angular_velocity_rmse_rad_s={:.8} elapsed_s={:.2}",
        final_errors.angle_rmse,
        final_errors.angular_velocity_rmse,
        started.elapsed().as_secs_f64()
    );
    println!("{}", final_regime.expect("at least one step"));
    println!("{}", disjoint.assert_disjoint()?);
    println!("{}", angle_learning.assert_learning()?);
    println!("{}", velocity_learning.assert_learning()?);
    if smoke {
        println!(
            "SMOKE PASS: training mechanics ran; scientific residual acceptance was not requested"
        );
        return Ok(());
    }
    if final_errors.angle_rmse > MAX_ANGLE_RMSE
        || final_errors.angular_velocity_rmse > MAX_ANGULAR_VELOCITY_RMSE
    {
        return Err(format!(
            "independent RK4 quality gate failed: angle RMSE {:.8} rad (max {MAX_ANGLE_RMSE}), angular-velocity RMSE {:.8} rad/s (max {MAX_ANGULAR_VELOCITY_RMSE})",
            final_errors.angle_rmse, final_errors.angular_velocity_rmse
        )
        .into());
    }
    let (initial_angle_receipt, initial_velocity_receipt, dynamics) =
        residual_receipts(&model, axes, &initial_angle, &device)?;
    println!("{initial_angle_receipt}");
    println!("{initial_velocity_receipt}");
    println!("{dynamics}");
    println!("PASS: learned a damped-pendulum trajectory from sampled differential residuals");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_stencils_exclude_the_sealed_evaluation_support() -> Result<()> {
        let declared = CollocationStream::new(TRAIN_SEED);
        assert!(declared.available > 4_000_000);
        let excluded = declared.excluded_coordinates;
        let mut stream = CollocationStream::new(TRAIN_SEED);
        let mut identities = HashSet::new();
        for _ in 0..100_000 {
            let sample = stream.next_sample()?.expect("generated stream");
            assert!(identities.insert(sample.id));
            assert_eq!(sample.id, u128::from(sample.value.to_bits()));
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

    #[test]
    #[ignore = "requires CUDA; scripts/check.sh runs the scientific acceptance"]
    fn cuda_acceptance_meets_oracle_and_residual_limits() -> Result<()> {
        run_experiment(4_000, false)
    }
}
