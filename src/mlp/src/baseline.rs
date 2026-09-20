//! A complete GPU training loop: y = relu(x W1 + b1) W2 + b2, MSE, SGD.
//! CPU code only generates data, checks correctness, and reports metrics.
use cutile::prelude::*;
use std::{env, time::Instant};

mod reference;

const INPUT: usize = 16;
const HIDDEN: usize = 32;
const OUTPUT: usize = 16;
const BATCH: usize = 256;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    // All dimensions are multiples of 16. Each block owns one output tile.
    // Transposing tiles on load also implements both backpropagation GEMMs.
    #[cutile::entry()]
    fn matmul<const TA: i32, const TB: i32>(
        out: &mut Tensor<f32, { [16, 16] }>,
        a: &Tensor<f32, { [-1, -1] }>,
        b: &Tensor<f32, { [-1, -1] }>,
        inner: i32,
    ) {
        let pid = get_tile_block_id();
        let ap = a.partition(shape![16, 16]);
        let bp = b.partition(shape![16, 16]);
        let mut acc = constant(0.0f32, shape![16, 16]);
        for k in 0i32..(inner / 16i32) {
            let at = if TA == 0 {
                ap.load([pid.0, k])
            } else {
                ap.load([k, pid.0]).transpose()
            };
            let bt = if TB == 0 {
                bp.load([k, pid.1])
            } else {
                bp.load([pid.1, k]).transpose()
            };
            acc = mma(at, bt, acc);
        }
        out.store(acc);
    }

    #[cutile::entry()]
    fn bias_activation<const RELU: i32>(
        out: &mut Tensor<f32, { [16, 16] }>,
        bias: &Tensor<f32, { [1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let b = bias.partition(shape![1, 16]).load([0i32, pid.1]);
        let value = out.load() + b.broadcast(shape![16, 16]);
        if RELU == 1 {
            let zero: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
            out.store(max_tile(value, zero));
        } else {
            out.store(value);
        }
    }

    #[cutile::entry()]
    fn mse_gradient(
        gradient: &mut Tensor<f32, { [16, 16] }>,
        prediction: &Tensor<f32, { [-1, -1] }>,
        target: &Tensor<f32, { [-1, -1] }>,
        scale: f32,
    ) {
        let delta = prediction.load_like(gradient) - target.load_like(gradient);
        let factor: Tile<f32, { [16, 16] }> = broadcast_scalar(scale, shape![16, 16]);
        gradient.store(delta * factor);
    }

    #[cutile::entry()]
    fn relu_backward(
        gradient: &mut Tensor<f32, { [16, 16] }>,
        activation: &Tensor<f32, { [-1, -1] }>,
    ) {
        let zero = constant(0.0f32, shape![16, 16]);
        let mask = gt_tile(activation.load_like(gradient), zero);
        let value = select(mask, gradient.load(), zero);
        gradient.store(value);
    }

    #[cutile::entry()]
    fn bias_gradient(
        out: &mut Tensor<f32, { [1, 16] }>,
        gradient: &Tensor<f32, { [-1, -1] }>,
        rows: i32,
    ) {
        let pid = get_tile_block_id();
        let parts = gradient.partition(shape![16, 16]);
        let mut acc = constant(0.0f32, shape![1, 16]);
        for r in 0i32..(rows / 16i32) {
            let sum: Tile<f32, { [16] }> = reduce_sum(parts.load([r, pid.1]), 0i32);
            acc = acc + sum.reshape(shape![1, 16]);
        }
        out.store(acc);
    }

    #[cutile::entry()]
    fn sgd<const M: i32, const N: i32>(
        parameter: &mut Tensor<f32, { [M, N] }>,
        gradient: &Tensor<f32, { [-1, -1] }>,
        learning_rate: f32,
    ) {
        let factor: Tile<f32, { [M, N] }> = broadcast_scalar(learning_rate, shape![M, N]);
        let value = parameter.load() - gradient.load_like(parameter) * factor;
        parameter.store(value);
    }
}

fn upload(values: Vec<f32>, rows: usize, cols: usize) -> Result<Tensor<f32>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(values))
        .reshape(&[rows, cols])
        .sync()?)
}

fn zeros(rows: usize, cols: usize) -> Result<Tensor<f32>> {
    Ok(api::zeros(&[rows, cols]).sync()?)
}

fn matmul(
    out: &mut Tensor<f32>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    inner: usize,
    ta: bool,
    tb: bool,
) -> Result<()> {
    kernels::matmul(out.partition([16, 16]), a, b, inner as i32)
        .generics(vec![i32::from(ta).to_string(), i32::from(tb).to_string()])
        .sync()?;
    Ok(())
}

struct Model {
    w1: Tensor<f32>,
    b1: Tensor<f32>,
    w2: Tensor<f32>,
    b2: Tensor<f32>,
}

impl Model {
    fn new(weights: &reference::Weights) -> Result<Self> {
        Ok(Self {
            w1: upload(weights.w1.clone(), INPUT, HIDDEN)?,
            b1: upload(weights.b1.clone(), 1, HIDDEN)?,
            w2: upload(weights.w2.clone(), HIDDEN, OUTPUT)?,
            b2: upload(weights.b2.clone(), 1, OUTPUT)?,
        })
    }

    fn forward(&self, x: &Tensor<f32>, state: &mut State) -> Result<()> {
        matmul(&mut state.hidden, x, &self.w1, INPUT, false, false)?;
        kernels::bias_activation((&mut state.hidden).partition([16, 16]), &self.b1)
            .generics(vec!["1".into()])
            .sync()?;
        matmul(
            &mut state.prediction,
            &state.hidden,
            &self.w2,
            HIDDEN,
            false,
            false,
        )?;
        kernels::bias_activation((&mut state.prediction).partition([16, 16]), &self.b2)
            .generics(vec!["0".into()])
            .sync()?;
        Ok(())
    }

    fn backward(&self, x: &Tensor<f32>, target: &Tensor<f32>, state: &mut State) -> Result<()> {
        kernels::mse_gradient(
            (&mut state.dy).partition([16, 16]),
            &state.prediction,
            target,
            2.0 / (state.rows * OUTPUT) as f32,
        )
        .sync()?;
        matmul(
            &mut state.dw2,
            &state.hidden,
            &state.dy,
            state.rows,
            true,
            false,
        )?;
        kernels::bias_gradient(
            (&mut state.db2).partition([1, 16]),
            &state.dy,
            state.rows as i32,
        )
        .sync()?;
        // W2 must still be the pre-update weights when propagating into layer 1.
        matmul(&mut state.dh, &state.dy, &self.w2, OUTPUT, false, true)?;
        kernels::relu_backward((&mut state.dh).partition([16, 16]), &state.hidden).sync()?;
        matmul(&mut state.dw1, x, &state.dh, state.rows, true, false)?;
        kernels::bias_gradient(
            (&mut state.db1).partition([1, 16]),
            &state.dh,
            state.rows as i32,
        )
        .sync()?;
        Ok(())
    }

    fn update(&mut self, state: &State, lr: f32) -> Result<()> {
        for (parameter, gradient, rows) in [
            (&mut self.w1, &state.dw1, 16),
            (&mut self.b1, &state.db1, 1),
            (&mut self.w2, &state.dw2, 16),
            (&mut self.b2, &state.db2, 1),
        ] {
            kernels::sgd(parameter.partition([rows, 16]), gradient, lr).sync()?;
        }
        Ok(())
    }
}

struct State {
    rows: usize,
    hidden: Tensor<f32>,
    prediction: Tensor<f32>,
    dy: Tensor<f32>,
    dh: Tensor<f32>,
    dw1: Tensor<f32>,
    db1: Tensor<f32>,
    dw2: Tensor<f32>,
    db2: Tensor<f32>,
}

impl State {
    fn new(rows: usize) -> Result<Self> {
        assert!(rows > 0 && rows.is_multiple_of(16));
        Ok(Self {
            rows,
            hidden: zeros(rows, HIDDEN)?,
            prediction: zeros(rows, OUTPUT)?,
            dy: zeros(rows, OUTPUT)?,
            dh: zeros(rows, HIDDEN)?,
            dw1: zeros(INPUT, HIDDEN)?,
            db1: zeros(1, HIDDEN)?,
            dw2: zeros(HIDDEN, OUTPUT)?,
            db2: zeros(1, OUTPUT)?,
        })
    }
}

fn compare(name: &str, gpu: &Tensor<f32>, cpu: &[f64]) -> Result<()> {
    let actual = api::dup(gpu).to_host_vec().sync()?;
    assert_eq!(actual.len(), cpu.len());
    let mut max_error = 0.0_f64;
    for (i, (&a, &b)) in actual.iter().zip(cpu).enumerate() {
        let error = (f64::from(a) - b).abs();
        // f32 tensor-core arithmetic need not match scalar f64 bit for bit.
        if !a.is_finite() || !b.is_finite() || error > 2e-4 + 2e-3 * b.abs() {
            return Err(format!("{name}[{i}]: GPU={a}, CPU={b}, abs_error={error}").into());
        }
        max_error = max_error.max(error);
    }
    println!("check {name}: PASS max_abs_error={max_error:.2e}");
    Ok(())
}

fn verify() -> Result<()> {
    let weights = reference::Weights::new(7);
    let (x, y) = reference::dataset(32, 11);
    let cpu = reference::forward_backward(&weights, &x, &y);
    reference::finite_difference_check(&weights, &x, &y, &cpu)?;
    let mut model = Model::new(&weights)?;
    let mut state = State::new(32)?;
    let dx = upload(x, 32, INPUT)?;
    let dy = upload(y, 32, OUTPUT)?;
    model.forward(&dx, &mut state)?;
    model.backward(&dx, &dy, &mut state)?;
    for (name, gpu, cpu) in [
        ("prediction", &state.prediction, &cpu.prediction),
        ("dW1", &state.dw1, &cpu.dw1),
        ("db1", &state.db1, &cpu.db1),
        ("dW2", &state.dw2, &cpu.dw2),
        ("db2", &state.db2, &cpu.db2),
    ] {
        compare(name, gpu, cpu)?;
    }
    let lr = 0.1_f32;
    model.update(&state, lr)?;
    for (name, gpu, original, grad) in [
        ("updated W1", &model.w1, &weights.w1, &cpu.dw1),
        ("updated b1", &model.b1, &weights.b1, &cpu.db1),
        ("updated W2", &model.w2, &weights.w2, &cpu.dw2),
        ("updated b2", &model.b2, &weights.b2, &cpu.db2),
    ] {
        let expected: Vec<f64> = original
            .iter()
            .zip(grad)
            .map(|(&p, &g)| f64::from(p) - f64::from(lr) * g)
            .collect();
        compare(name, gpu, &expected)?;
    }
    Ok(())
}

fn mse(prediction: &Tensor<f32>, target: &[f32]) -> Result<f64> {
    let values = api::dup(prediction).to_host_vec().sync()?;
    let loss = values
        .iter()
        .zip(target)
        .map(|(&p, &y)| (f64::from(p) - f64::from(y)).powi(2))
        .sum::<f64>()
        / target.len() as f64;
    if !loss.is_finite() {
        return Err("Non-finite MSE".into());
    }
    Ok(loss)
}

fn main() -> Result<()> {
    let mut steps = 500_usize;
    let mut lr = 0.5_f32;
    let mut seed = 42_u64;
    let mut verify_only = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--lr" => lr = args.next().ok_or("--lr needs a value")?.parse()?,
            "--seed" => seed = args.next().ok_or("--seed needs a value")?.parse()?,
            "--verify-only" => verify_only = true,
            "--smoke" => steps = 10,
            "--help" | "-h" => {
                println!(
                    "cutile-mlp [--steps 500] [--lr 0.5] [--seed 42] [--smoke] [--verify-only]"
                );
                return Ok(());
            }
            _ => return Err(format!("Unknown argument: {arg}").into()),
        }
    }
    if steps == 0 || !lr.is_finite() || lr <= 0.0 {
        return Err("steps and learning rate must be positive and finite".into());
    }
    println!("GPU: {}", Device::new(0)?.name()?);
    println!("cuTile 0.3.1 | MLP {INPUT} -> {HIDDEN} ReLU -> {OUTPUT} | f32 | SGD");
    let setup = Instant::now();
    verify()?;
    if verify_only {
        return Ok(());
    }
    let weights = reference::Weights::new(seed);
    let mut model = Model::new(&weights)?;
    let (train_x, train_y) = reference::dataset(BATCH, 1001);
    let (valid_x, valid_y) = reference::dataset(BATCH, 2002);
    let x = upload(train_x, BATCH, INPUT)?;
    let y = upload(train_y.clone(), BATCH, OUTPUT)?;
    let vx = upload(valid_x, BATCH, INPUT)?;
    let mut state = State::new(BATCH)?;
    let mut validation = State::new(BATCH)?;
    // Warm up all kernels before measuring the training loop; do not update weights.
    model.forward(&x, &mut state)?;
    model.backward(&x, &y, &mut state)?;
    model.forward(&vx, &mut validation)?;
    let initial = mse(&state.prediction, &train_y)?;
    let initial_valid = mse(&validation.prediction, &valid_y)?;
    println!(
        "setup_and_verification_seconds={:.2} steps={steps} lr={lr:.2} seed={seed}",
        setup.elapsed().as_secs_f64()
    );
    println!("step,train_mse,validation_mse,elapsed_seconds");
    println!("0,{initial:.2e},{initial_valid:.2e},0.00");
    let start = Instant::now();
    let log_every = (steps / 10).max(1);
    let mut final_loss = initial;
    let mut final_valid = initial_valid;
    for step in 1..=steps {
        model.forward(&x, &mut state)?;
        model.backward(&x, &y, &mut state)?;
        model.update(&state, lr)?;
        if step % log_every == 0 || step == steps {
            // Evaluate the UPDATED model, including after the final update.
            model.forward(&x, &mut state)?;
            model.forward(&vx, &mut validation)?;
            final_loss = mse(&state.prediction, &train_y)?;
            final_valid = mse(&validation.prediction, &valid_y)?;
            println!(
                "{step},{final_loss:.2e},{final_valid:.2e},{:.2}",
                start.elapsed().as_secs_f64()
            );
        }
    }
    if final_loss >= initial || final_valid >= initial_valid {
        return Err("Training failed: both training and held-out MSE must decrease".into());
    }
    println!(
        "PASS train_reduction={:.2}% validation_reduction={:.2}% elapsed_seconds={:.2}",
        100.0 * (1.0 - final_loss / initial),
        100.0 * (1.0 - final_valid / initial_valid),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
