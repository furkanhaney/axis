//! CNN: valid 3x3 convolution -> ReLU -> global mean -> binary linear head.
//! CPU code only generates data, checks correctness, and reports metrics.
use cutile::prelude::*;
use std::{env, time::Instant};

#[path = "reference.rs"]
mod reference;

const IMAGE: usize = 10;
const FILTER: usize = 3;
const SIDE: usize = IMAGE - FILTER + 1;
const SPATIAL: usize = SIDE * SIDE;
const CHANNELS: usize = 16;
const PATCH: usize = 16; // Nine actual filter coefficients; seven zero-padded rows.
const BATCH: usize = 128;
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
    fn bias_gradient<const N: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
        gradient: &Tensor<f32, { [-1, -1] }>,
        rows: i32,
    ) {
        let pid = get_tile_block_id();
        let parts = gradient.partition(shape![16, N]);
        let mut acc = constant(0.0f32, shape![1, N]);
        for r in 0i32..(rows / 16i32) {
            let sum: Tile<f32, { [N] }> = reduce_sum(parts.load([r, pid.1]), 0i32);
            acc = acc + sum.reshape(shape![1, N]);
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
    // Image-major patches, valid 3x3 stride-1 cross-correlation, no padding.
    // Each block owns one patch row. Safe tensor loads avoid raw pointers.
    #[cutile::entry()]
    fn im2col(out: &mut Tensor<f32, { [1, 16] }>, images: &Tensor<f32, { [-1, 100] }>) {
        let pid = get_tile_block_id();
        let image = pid.0 / 64i32;
        let position = pid.0 % 64i32;
        let oy = position / 8i32;
        let ox = position % 8i32;
        let pixels = images.partition(shape![1, 1]);
        let columns: Tile<i32, { [16] }> = iota(shape![16]);
        let columns2 = columns.reshape(shape![1, 16]);
        let mut patch = constant(0.0f32, shape![1, 16]);
        for ky in 0i32..3i32 {
            for kx in 0i32..3i32 {
                let offset: Tile<i32, { [1, 16] }> =
                    broadcast_scalar(ky * 3i32 + kx, shape![1, 16]);
                let pixel = pixels.load([image, (oy + ky) * 10i32 + ox + kx]);
                patch = select(
                    eq_tile(columns2, offset),
                    pixel.broadcast(shape![1, 16]),
                    patch,
                );
            }
        }
        out.store(patch);
    }

    #[cutile::entry()]
    fn mean_pool(out: &mut Tensor<f32, { [1, 16] }>, activation: &Tensor<f32, { [-1, 16] }>) {
        let pid = get_tile_block_id();
        let patch = activation.partition(shape![64, 16]).load([pid.0, 0i32]);
        let sum: Tile<f32, { [16] }> = reduce_sum(patch, 0i32);
        let scale: Tile<f32, { [1, 16] }> = constant(0.015625f32, shape![1, 16]); // 1 / 64
        out.store(sum.reshape(shape![1, 16]) * scale);
    }

    #[cutile::entry()]
    fn mean_pool_backward(
        out: &mut Tensor<f32, { [16, 16] }>,
        gradient: &Tensor<f32, { [-1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let image = pid.0 / 4i32;
        let g = gradient.partition(shape![1, 16]).load([image, 0i32]);
        let scale: Tile<f32, { [16, 16] }> = constant(0.015625f32, shape![16, 16]); // 1 / 64
        out.store(g.broadcast(shape![16, 16]) * scale);
    }

    // A one-output linear layer uses a dot reduction rather than padding logits.
    #[cutile::entry()]
    fn linear(
        out: &mut Tensor<f32, { [16, 1] }>,
        input: &Tensor<f32, { [-1, 16] }>,
        weight: &Tensor<f32, { [1, 16] }>,
        bias: &Tensor<f32, { [1, 1] }>,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![16, 16]).load([pid.0, 0i32]);
        let w = weight.partition(shape![1, 16]).load([0i32, 0i32]);
        let b = bias.partition(shape![1, 1]).load([0i32, 0i32]);
        let product = x * w.broadcast(shape![16, 16]);
        let sum: Tile<f32, { [16] }> = reduce_sum(product, 1i32);
        out.store(sum.reshape(shape![16, 1]) + b.broadcast(shape![16, 1]));
    }

    #[cutile::entry()]
    fn bce_gradient(
        out: &mut Tensor<f32, { [16, 1] }>,
        logits: &Tensor<f32, { [-1, 1] }>,
        labels: &Tensor<f32, { [-1, 1] }>,
        scale: f32,
    ) {
        let z = logits.load_like(out);
        let one = constant(1.0f32, shape![16, 1]);
        let zero = constant(0.0f32, shape![16, 1]);
        // Stable sigmoid: every exponential has a nonpositive argument.
        let magnitude = max_tile(z, zero - z);
        let e = exp(zero - magnitude);
        let probability = select(gt_tile(z, zero), one / (one + e), e / (one + e));
        let factor: Tile<f32, { [16, 1] }> = broadcast_scalar(scale, shape![16, 1]);
        let delta = probability - labels.load_like(out);
        out.store(delta * factor);
    }

    #[cutile::entry()]
    fn linear_input_gradient(
        out: &mut Tensor<f32, { [16, 16] }>,
        gradient: &Tensor<f32, { [-1, 1] }>,
        weight: &Tensor<f32, { [1, 16] }>,
    ) {
        let pid = get_tile_block_id();
        let g = gradient.partition(shape![16, 1]).load([pid.0, 0i32]);
        let w = weight.partition(shape![1, 16]).load([0i32, 0i32]);
        out.store(g.broadcast(shape![16, 16]) * w.broadcast(shape![16, 16]));
    }

    #[cutile::entry()]
    fn linear_weight_gradient(
        out: &mut Tensor<f32, { [1, 16] }>,
        input: &Tensor<f32, { [-1, 16] }>,
        gradient: &Tensor<f32, { [-1, 1] }>,
        rows: i32,
    ) {
        let xp = input.partition(shape![16, 16]);
        let gp = gradient.partition(shape![16, 1]);
        let mut acc = constant(0.0f32, shape![1, 16]);
        for r in 0i32..(rows / 16i32) {
            let product = xp.load([r, 0i32]) * gp.load([r, 0i32]).broadcast(shape![16, 16]);
            let sum: Tile<f32, { [16] }> = reduce_sum(product, 0i32);
            acc = acc + sum.reshape(shape![1, 16]);
        }
        out.store(acc);
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
    conv_weight: Tensor<f32>,
    conv_bias: Tensor<f32>,
    head_weight: Tensor<f32>,
    head_bias: Tensor<f32>,
}

impl Model {
    fn new(weights: &reference::Weights) -> Result<Self> {
        Ok(Self {
            conv_weight: upload(weights.conv_weight.clone(), PATCH, CHANNELS)?,
            conv_bias: upload(weights.conv_bias.clone(), 1, CHANNELS)?,
            head_weight: upload(weights.head_weight.clone(), 1, CHANNELS)?,
            head_bias: upload(weights.head_bias.clone(), 1, 1)?,
        })
    }

    fn forward(&self, state: &mut State) -> Result<()> {
        matmul(
            &mut state.activation,
            &state.patches,
            &self.conv_weight,
            PATCH,
            false,
            false,
        )?;
        kernels::bias_activation((&mut state.activation).partition([16, 16]), &self.conv_bias)
            .generics(vec!["1".into()])
            .sync()?;
        kernels::mean_pool((&mut state.pooled).partition([1, 16]), &state.activation).sync()?;
        kernels::linear(
            (&mut state.logits).partition([16, 1]),
            &state.pooled,
            &self.head_weight,
            &self.head_bias,
        )
        .sync()?;
        Ok(())
    }

    fn backward(&self, labels: &Tensor<f32>, state: &mut State) -> Result<()> {
        kernels::bce_gradient(
            (&mut state.dlogits).partition([16, 1]),
            &state.logits,
            labels,
            1.0 / state.rows as f32,
        )
        .sync()?;
        kernels::linear_weight_gradient(
            (&mut state.dhead_weight).partition([1, 16]),
            &state.pooled,
            &state.dlogits,
            state.rows as i32,
        )
        .sync()?;
        kernels::bias_gradient(
            (&mut state.dhead_bias).partition([1, 1]),
            &state.dlogits,
            state.rows as i32,
        )
        .sync()?;
        // Backpropagate using the old head weights; update only after all gradients exist.
        kernels::linear_input_gradient(
            (&mut state.dpooled).partition([16, 16]),
            &state.dlogits,
            &self.head_weight,
        )
        .sync()?;
        kernels::mean_pool_backward((&mut state.dactivation).partition([16, 16]), &state.dpooled)
            .sync()?;
        kernels::relu_backward(
            (&mut state.dactivation).partition([16, 16]),
            &state.activation,
        )
        .sync()?;
        matmul(
            &mut state.dconv_weight,
            &state.patches,
            &state.dactivation,
            state.rows * SPATIAL,
            true,
            false,
        )?;
        kernels::bias_gradient(
            (&mut state.dconv_bias).partition([1, 16]),
            &state.dactivation,
            (state.rows * SPATIAL) as i32,
        )
        .sync()?;
        Ok(())
    }

    fn update(&mut self, state: &State, lr: f32) -> Result<()> {
        for (parameter, gradient, shape) in [
            (&mut self.conv_weight, &state.dconv_weight, [16, 16]),
            (&mut self.conv_bias, &state.dconv_bias, [1, 16]),
            (&mut self.head_weight, &state.dhead_weight, [1, 16]),
            (&mut self.head_bias, &state.dhead_bias, [1, 1]),
        ] {
            kernels::sgd(parameter.partition(shape), gradient, lr).sync()?;
        }
        Ok(())
    }
}

struct State {
    rows: usize,
    patches: Tensor<f32>,
    activation: Tensor<f32>,
    pooled: Tensor<f32>,
    logits: Tensor<f32>,
    dlogits: Tensor<f32>,
    dpooled: Tensor<f32>,
    dactivation: Tensor<f32>,
    dconv_weight: Tensor<f32>,
    dconv_bias: Tensor<f32>,
    dhead_weight: Tensor<f32>,
    dhead_bias: Tensor<f32>,
}

impl State {
    fn new(images: Vec<f32>, rows: usize) -> Result<Self> {
        assert!(rows > 0 && rows.is_multiple_of(16));
        // Geometry is fixed in the device kernels; fail instead of silently mismatching it.
        assert_eq!(
            (IMAGE, FILTER, SIDE, SPATIAL, CHANNELS, PATCH),
            (10, 3, 8, 64, 16, 16)
        );
        assert_eq!(images.len(), rows * IMAGE * IMAGE);
        let images = upload(images, rows, IMAGE * IMAGE)?;
        let mut patches = zeros(rows * SPATIAL, PATCH)?;
        kernels::im2col((&mut patches).partition([1, 16]), &images).sync()?;
        // Inputs stay fixed in this full-batch demo, so patches can be cached once.
        Ok(Self {
            rows,
            patches,
            activation: zeros(rows * SPATIAL, CHANNELS)?,
            pooled: zeros(rows, CHANNELS)?,
            logits: zeros(rows, 1)?,
            dlogits: zeros(rows, 1)?,
            dpooled: zeros(rows, CHANNELS)?,
            dactivation: zeros(rows * SPATIAL, CHANNELS)?,
            dconv_weight: zeros(PATCH, CHANNELS)?,
            dconv_bias: zeros(1, CHANNELS)?,
            dhead_weight: zeros(1, CHANNELS)?,
            dhead_bias: zeros(1, 1)?,
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
    let (x, y) = reference::dataset(16, 11);
    let cpu = reference::forward_backward(&weights, &x, &y);
    reference::finite_difference_check(&weights, &x, &y, &cpu)?;
    let mut model = Model::new(&weights)?;
    let mut state = State::new(x, 16)?;
    let labels = upload(y, 16, 1)?;
    model.forward(&mut state)?;
    model.backward(&labels, &mut state)?;
    for (name, gpu, cpu) in [
        ("conv+ReLU", &state.activation, &cpu.activation),
        ("mean pooling", &state.pooled, &cpu.pooled),
        ("logits", &state.logits, &cpu.logits),
        ("d conv weight", &state.dconv_weight, &cpu.dconv_weight),
        ("d conv bias", &state.dconv_bias, &cpu.dconv_bias),
        ("d head weight", &state.dhead_weight, &cpu.dhead_weight),
        ("d head bias", &state.dhead_bias, &cpu.dhead_bias),
    ] {
        compare(name, gpu, cpu)?;
    }
    let lr = 0.1_f32;
    model.update(&state, lr)?;
    for (name, gpu, original, grad) in [
        (
            "updated conv weight",
            &model.conv_weight,
            &weights.conv_weight,
            &cpu.dconv_weight,
        ),
        (
            "updated conv bias",
            &model.conv_bias,
            &weights.conv_bias,
            &cpu.dconv_bias,
        ),
        (
            "updated head weight",
            &model.head_weight,
            &weights.head_weight,
            &cpu.dhead_weight,
        ),
        (
            "updated head bias",
            &model.head_bias,
            &weights.head_bias,
            &cpu.dhead_bias,
        ),
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

fn metrics(logits: &Tensor<f32>, target: &[f32]) -> Result<(f64, f64)> {
    let values = api::dup(logits).to_host_vec().sync()?;
    assert_eq!(values.len(), target.len());
    let mut loss = 0.0;
    let mut correct = 0;
    for (&z, &y) in values.iter().zip(target) {
        if !z.is_finite() {
            return Err("Non-finite logit".into());
        }
        let z = f64::from(z);
        loss += z.max(0.0) - z * f64::from(y) + (-z.abs()).exp().ln_1p();
        correct += usize::from((z >= 0.0) == (y == 1.0));
    }
    Ok((
        loss / target.len() as f64,
        100.0 * correct as f64 / target.len() as f64,
    ))
}

fn main() -> Result<()> {
    let mut steps = 500_usize;
    let mut lr = 1.0_f32;
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
                    "cutile-cnn [--steps 500] [--lr 1.0] [--seed 42] [--smoke] [--verify-only]"
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
    println!("cuTile 0.3.1 | CNN 1x10x10 -> conv3x3/16 -> ReLU -> mean -> linear/1 | BCE | SGD");
    let setup = Instant::now();
    verify()?;
    if verify_only {
        return Ok(());
    }
    let mut model = Model::new(&reference::Weights::new(seed))?;
    let (train_x, train_y) = reference::dataset(BATCH, 1001);
    let (valid_x, valid_y) = reference::dataset(BATCH, 2002);
    let labels = upload(train_y.clone(), BATCH, 1)?;
    let mut state = State::new(train_x, BATCH)?;
    let mut validation = State::new(valid_x, BATCH)?;
    model.forward(&mut state)?;
    model.backward(&labels, &mut state)?;
    model.forward(&mut validation)?;
    let initial = metrics(&state.logits, &train_y)?;
    let initial_valid = metrics(&validation.logits, &valid_y)?;
    println!(
        "setup_and_verification_seconds={:.2} steps={steps} lr={lr:.2} seed={seed}",
        setup.elapsed().as_secs_f64()
    );
    println!("step,train_bce,validation_bce,train_accuracy,validation_accuracy,elapsed_seconds");
    println!(
        "0,{:.2e},{:.2e},{:.2},{:.2},0.00",
        initial.0, initial_valid.0, initial.1, initial_valid.1
    );
    let start = Instant::now();
    let log_every = (steps / 10).max(1);
    let mut final_train = initial;
    let mut final_valid = initial_valid;
    for step in 1..=steps {
        model.forward(&mut state)?;
        model.backward(&labels, &mut state)?;
        model.update(&state, lr)?;
        if step % log_every == 0 || step == steps {
            model.forward(&mut state)?;
            model.forward(&mut validation)?;
            final_train = metrics(&state.logits, &train_y)?;
            final_valid = metrics(&validation.logits, &valid_y)?;
            println!(
                "{step},{:.2e},{:.2e},{:.2},{:.2},{:.2}",
                final_train.0,
                final_valid.0,
                final_train.1,
                final_valid.1,
                start.elapsed().as_secs_f64()
            );
        }
    }
    if final_train.0 >= initial.0 || final_valid.0 >= initial_valid.0 {
        return Err("Training failed: both training and held-out BCE must decrease".into());
    }
    println!(
        "PASS train_reduction={:.2}% validation_reduction={:.2}% validation_accuracy={:.2}% elapsed_seconds={:.2}",
        100.0 * (1.0 - final_train.0 / initial.0),
        100.0 * (1.0 - final_valid.0 / initial_valid.0),
        final_valid.1,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
