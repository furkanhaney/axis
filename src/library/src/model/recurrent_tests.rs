use crate::prelude::*;

const B: usize = 2;
const T: usize = 3;
const I: usize = 2;
const H: usize = 2;

#[derive(Clone)]
struct Case {
    input: Vec<f64>,
    initial_hidden: Vec<f64>,
    initial_cell: Vec<f64>,
    input_weight: Vec<f64>,
    recurrent_weight: Vec<f64>,
    bias: Vec<f64>,
}

struct Reference {
    sequence: Vec<f64>,
    hidden: Vec<f64>,
    cell: Vec<f64>,
}

fn case() -> Case {
    Case {
        input: vec![
            0.2, -0.4, 0.7, 0.1, -0.3, 0.8, -0.6, 0.5, 0.9, -0.2, 0.4, 0.3,
        ],
        initial_hidden: vec![0.1, -0.2, 0.3, 0.05],
        initial_cell: vec![-0.15, 0.25, 0.2, -0.1],
        input_weight: vec![
            0.15, -0.2, 0.3, 0.1, -0.25, 0.4, 0.05, -0.3, -0.1, 0.35, -0.2, 0.25, 0.45, -0.15, 0.2,
            0.1,
        ],
        recurrent_weight: vec![
            -0.2, 0.1, 0.25, -0.35, 0.3, 0.15, -0.1, 0.2, 0.4, -0.25, 0.05, 0.3, -0.15, 0.2, 0.35,
            -0.05,
        ],
        bias: vec![0.05, -0.1, 0.2, 0.15, -0.05, 0.08, 0.12, -0.07],
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

fn reference_step(case: &Case, input: &[f64], hidden: &mut [f64], cell: &mut [f64]) {
    let mut gates = case.bias.clone();
    for (input_feature, &input_value) in input.iter().enumerate() {
        for (gate, value) in gates.iter_mut().enumerate() {
            *value += input_value * case.input_weight[input_feature * 4 * H + gate];
        }
    }
    for (previous, &hidden_value) in hidden.iter().enumerate() {
        for (gate, value) in gates.iter_mut().enumerate() {
            *value += hidden_value * case.recurrent_weight[previous * 4 * H + gate];
        }
    }
    for feature in 0..H {
        let input_gate = sigmoid(gates[feature]);
        let forget_gate = sigmoid(gates[H + feature]);
        let candidate = gates[2 * H + feature].tanh();
        let output_gate = sigmoid(gates[3 * H + feature]);
        cell[feature] = forget_gate * cell[feature] + input_gate * candidate;
        hidden[feature] = output_gate * cell[feature].tanh();
    }
}

fn reference(case: &Case) -> Reference {
    let mut hidden = case.initial_hidden.clone();
    let mut cell = case.initial_cell.clone();
    let mut sequence = vec![0.0; B * T * H];
    for batch in 0..B {
        let mut h = hidden[batch * H..(batch + 1) * H].to_vec();
        let mut c = cell[batch * H..(batch + 1) * H].to_vec();
        for step in 0..T {
            let offset = (batch * T + step) * I;
            reference_step(case, &case.input[offset..offset + I], &mut h, &mut c);
            for feature in 0..H {
                sequence[(batch * T + step) * H + feature] = h[feature];
            }
        }
        hidden[batch * H..(batch + 1) * H].copy_from_slice(&h);
        cell[batch * H..(batch + 1) * H].copy_from_slice(&c);
    }
    Reference {
        sequence,
        hidden,
        cell,
    }
}

fn objective(case: &Case) -> f64 {
    let result = reference(case);
    let sequence_coefficients = [
        0.2, -0.3, 0.5, 0.1, -0.4, 0.7, -0.6, 0.8, 0.25, -0.15, 0.45, -0.35,
    ];
    let hidden_coefficients = [0.3, -0.2, 0.6, 0.1];
    let cell_coefficients = [-0.4, 0.5, 0.2, -0.3];
    result
        .sequence
        .iter()
        .zip(sequence_coefficients)
        .map(|(value, coefficient)| value * coefficient)
        .sum::<f64>()
        + result
            .hidden
            .iter()
            .zip(hidden_coefficients)
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>()
        + result
            .cell
            .iter()
            .zip(cell_coefficients)
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>()
}

fn finite_difference(case: &Case, field: fn(&mut Case) -> &mut Vec<f64>) -> Vec<f64> {
    let epsilon = 1e-5;
    (0..field(&mut case.clone()).len())
        .map(|index| {
            let mut high = case.clone();
            field(&mut high)[index] += epsilon;
            let mut low = case.clone();
            field(&mut low)[index] -= epsilon;
            (objective(&high) - objective(&low)) / (2.0 * epsilon)
        })
        .collect()
}

fn close(name: &str, actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(actual) - expected).abs();
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && error <= tolerance * (1.0 + expected.abs()),
            "{name}[{index}]: {actual} != {expected}, error={error}"
        );
    }
}

fn f32s(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

fn install(model: &Lstm, case: &Case) -> Result<()> {
    model
        .parameter("input_weight")?
        .set_values(&f32s(&case.input_weight))?;
    model
        .parameter("recurrent_weight")?
        .set_values(&f32s(&case.recurrent_weight))?;
    model.parameter("bias")?.set_values(&f32s(&case.bias))?;
    Ok(())
}

fn weighted_sum(value: &Tensor, coefficients: &[f32], axes: &[Axis]) -> Result<Tensor> {
    let coefficient = Tensor::from_slice(
        coefficients,
        value.shape().dims().iter().copied(),
        value.device(),
    )?;
    value
        .mul(&coefficient)?
        .mean(axes)?
        .scale(coefficients.len() as f32)
}

#[test]
#[ignore = "requires CUDA"]
fn lstm_matches_independent_f64_forward_and_all_central_differences() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let case = case();
    let expected = reference(&case);
    let mut model = Lstm::new(input, hidden.of(H), time)?;
    model.build(
        &Shape::new([batch.of(B), time.of(T), input.of(I)])?,
        &device,
        17,
    )?;
    install(&model, &case)?;

    let x = Tensor::from_slice(
        &f32s(&case.input),
        [batch.of(B), time.of(T), input.of(I)],
        &device,
    )?
    .with_layout([input, time, batch])?
    .with_grad();
    let initial = LstmState::new(
        Tensor::from_slice(
            &f32s(&case.initial_hidden),
            [batch.of(B), hidden.of(H)],
            &device,
        )?
        .with_layout([hidden, batch])?
        .with_grad(),
        Tensor::from_slice(
            &f32s(&case.initial_cell),
            [batch.of(B), hidden.of(H)],
            &device,
        )?
        .with_layout([hidden, batch])?
        .with_grad(),
    )?;
    let run = model.run_from(&x, &[initial.clone()])?;
    assert_eq!(
        run.sequence.shape(),
        &Shape::new([batch.of(B), time.of(T), hidden.of(H)])?
    );
    close(
        "LSTM sequence",
        &run.sequence.to_vec()?,
        &expected.sequence,
        4e-5,
    );
    close(
        "LSTM final hidden",
        &run.state[0].hidden.to_vec()?,
        &expected.hidden,
        4e-5,
    );
    close(
        "LSTM final cell",
        &run.state[0].cell.to_vec()?,
        &expected.cell,
        4e-5,
    );
    close(
        "LSTM final hidden equals last sequence coordinate",
        &run.sequence.select(time, T - 1)?.to_vec()?,
        &expected.hidden,
        4e-5,
    );

    let sequence_coefficients = [
        0.2, -0.3, 0.5, 0.1, -0.4, 0.7, -0.6, 0.8, 0.25, -0.15, 0.45, -0.35,
    ];
    let hidden_coefficients = [0.3, -0.2, 0.6, 0.1];
    let cell_coefficients = [-0.4, 0.5, 0.2, -0.3];
    let loss = weighted_sum(
        &run.sequence,
        &sequence_coefficients,
        &[batch, time, hidden],
    )?
    .add(&weighted_sum(
        &run.state[0].hidden,
        &hidden_coefficients,
        &[batch, hidden],
    )?)?
    .add(&weighted_sum(
        &run.state[0].cell,
        &cell_coefficients,
        &[batch, hidden],
    )?)?;
    loss.backward()?;

    let gradients = [
        (
            "LSTM input gradient",
            x.grad().unwrap().to_vec()?,
            finite_difference(&case, |case| &mut case.input),
        ),
        (
            "LSTM initial hidden gradient",
            initial.hidden.grad().unwrap().to_vec()?,
            finite_difference(&case, |case| &mut case.initial_hidden),
        ),
        (
            "LSTM initial cell gradient",
            initial.cell.grad().unwrap().to_vec()?,
            finite_difference(&case, |case| &mut case.initial_cell),
        ),
        (
            "LSTM input-weight gradient",
            model.parameter("input_weight")?.grad().unwrap().to_vec()?,
            finite_difference(&case, |case| &mut case.input_weight),
        ),
        (
            "LSTM recurrent-weight gradient",
            model
                .parameter("recurrent_weight")?
                .grad()
                .unwrap()
                .to_vec()?,
            finite_difference(&case, |case| &mut case.recurrent_weight),
        ),
        (
            "LSTM bias gradient",
            model.parameter("bias")?.grad().unwrap().to_vec()?,
            finite_difference(&case, |case| &mut case.bias),
        ),
    ];
    for (name, actual, expected) in gradients {
        close(name, &actual, &expected, 3e-3);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lstm_executes_two_unrelated_stream_axes_across_layouts() -> Result<()> {
    const REPLICAS: usize = 2;
    const PATIENTS: usize = 3;

    let device = Device::cuda(0)?;
    let (replica, time, patient, input, hidden) = (
        Axis::new("replica"),
        Axis::new("time"),
        Axis::new("patient"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let case = case();
    let input_values: Vec<_> = (0..REPLICAS * T * PATIENTS * I)
        .map(|index| ((index * 7 % 29) as f64 - 14.0) / 10.0)
        .collect();
    let input_shape = Shape::new([
        replica.of(REPLICAS),
        time.of(T),
        patient.of(PATIENTS),
        input.of(I),
    ])?;
    let mut model = Lstm::new(input, hidden.of(H), time)?;
    model.build(&input_shape, &device, 41)?;
    install(&model, &case)?;
    let tensor = Tensor::from_slice(
        &f32s(&input_values),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_layout([patient, input, replica, time])?;
    let run = model.run(&tensor)?;

    assert_eq!(
        run.sequence.shape(),
        &Shape::new([
            replica.of(REPLICAS),
            time.of(T),
            patient.of(PATIENTS),
            hidden.of(H),
        ])?
    );
    let state_shape = Shape::new([replica.of(REPLICAS), patient.of(PATIENTS), hidden.of(H)])?;
    assert_eq!(run.state[0].hidden.shape(), &state_shape);
    assert_eq!(run.state[0].cell.shape(), &state_shape);

    let mut expected_sequence = vec![0.0; REPLICAS * T * PATIENTS * H];
    let mut expected_hidden = vec![0.0; REPLICAS * PATIENTS * H];
    let mut expected_cell = vec![0.0; REPLICAS * PATIENTS * H];
    for replica_index in 0..REPLICAS {
        for patient_index in 0..PATIENTS {
            let mut stream_hidden = vec![0.0; H];
            let mut stream_cell = vec![0.0; H];
            for step in 0..T {
                let input_offset = ((replica_index * T + step) * PATIENTS + patient_index) * I;
                reference_step(
                    &case,
                    &input_values[input_offset..input_offset + I],
                    &mut stream_hidden,
                    &mut stream_cell,
                );
                let output_offset = ((replica_index * T + step) * PATIENTS + patient_index) * H;
                expected_sequence[output_offset..output_offset + H].copy_from_slice(&stream_hidden);
            }
            let state_offset = (replica_index * PATIENTS + patient_index) * H;
            expected_hidden[state_offset..state_offset + H].copy_from_slice(&stream_hidden);
            expected_cell[state_offset..state_offset + H].copy_from_slice(&stream_cell);
        }
    }
    close(
        "two-stream-axis LSTM sequence",
        &run.sequence.to_vec()?,
        &expected_sequence,
        4e-5,
    );
    close(
        "two-stream-axis LSTM hidden state",
        &run.state[0].hidden.to_vec()?,
        &expected_hidden,
        4e-5,
    );
    close(
        "two-stream-axis LSTM cell state",
        &run.state[0].cell.to_vec()?,
        &expected_cell,
        4e-5,
    );
    Ok(())
}

fn terminal_loss(state: &LstmState, batch: Axis, hidden: Axis) -> Result<Tensor> {
    state
        .hidden
        .mean([batch, hidden])?
        .add(&state.cell.mean([batch, hidden])?)
}

fn chunk_values(input: &[f64], start: usize, end: usize) -> Vec<f64> {
    let mut result = vec![];
    for batch in 0..B {
        for step in start..end {
            result.extend_from_slice(&input[(batch * T + step) * I..(batch * T + step + 1) * I]);
        }
    }
    result
}

fn combine_chunks(first: &[f32], second: &[f32]) -> Vec<f32> {
    let mut result = vec![];
    for batch in 0..B {
        result.extend_from_slice(&first[batch * 2 * H..(batch + 1) * 2 * H]);
        result.extend_from_slice(&second[batch * H..(batch + 1) * H]);
    }
    result
}

#[test]
#[ignore = "requires CUDA"]
fn lstm_causality_streaming_layout_and_state_contracts() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden, patient) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
        Axis::new("patient"),
    );
    assert!(Lstm::new(time, hidden.of(H), time).is_err());
    assert!(Lstm::new(input, time.of(H), time).is_err());
    let case = case();
    let input_shape = Shape::new([batch.of(B), time.of(T), input.of(I)])?;
    let mut model = Lstm::new(input, hidden.of(H), time)?;
    model.build(&input_shape, &device, 23)?;
    install(&model, &case)?;
    assert_eq!(
        model
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["input_weight", "recurrent_weight", "bias"]
    );
    let parameter_ids: Vec<_> = model.parameters().iter().map(Parameter::id).collect();
    assert_eq!(
        model.build(
            &Shape::new([patient.of(2), time.of(1), batch.of(1), input.of(I)])?,
            &device,
            999,
        )?,
        Shape::new([patient.of(2), time.of(1), batch.of(1), hidden.of(H)])?
    );
    assert_eq!(
        parameter_ids,
        model
            .parameters()
            .iter()
            .map(Parameter::id)
            .collect::<Vec<_>>()
    );
    assert!(
        model
            .output_shape(&Shape::new([batch.of(B), input.of(I)])?)
            .is_err()
    );
    assert!(
        model
            .output_shape(&Shape::new([
                batch.of(B),
                time.of(T),
                input.of(I),
                hidden.of(H),
            ])?)
            .is_err()
    );
    assert!(
        model
            .output_shape(&Shape::new([batch.of(B), time.of(T), input.of(I + 1)])?)
            .is_err()
    );

    let base = Tensor::from_slice(
        &f32s(&case.input),
        input_shape.dims().iter().copied(),
        &device,
    )?;
    let baseline = model.run(&base)?;
    let mut changed = case.input.clone();
    for batch_index in 0..B {
        for feature in 0..I {
            changed[(batch_index * T + 2) * I + feature] += 20.0 + feature as f64;
        }
    }
    let changed = Tensor::from_slice(&f32s(&changed), input_shape.dims().iter().copied(), &device)?
        .with_layout([time, input, batch])?;
    let changed_run = model.run(&changed)?;
    for step in 0..2 {
        close(
            &format!("causal prefix {step}"),
            &changed_run.sequence.select(time, step)?.to_vec()?,
            &baseline
                .sequence
                .select(time, step)?
                .to_vec()?
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>(),
            1e-6,
        );
    }
    let causal_input = Tensor::from_slice(
        &f32s(&case.input),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_grad();
    model
        .run(&causal_input)?
        .sequence
        .select(time, 0)?
        .mean([batch, hidden])?
        .backward()?;
    let causal_gradient = causal_input.grad().unwrap().to_vec()?;
    for batch_index in 0..B {
        for step in 1..T {
            for feature in 0..I {
                assert_eq!(
                    causal_gradient[(batch_index * T + step) * I + feature],
                    0.0,
                    "future input gradient must be exactly zero"
                );
            }
        }
    }

    let initial = LstmState::new(
        Tensor::from_slice(
            &f32s(&case.initial_hidden),
            [batch.of(B), hidden.of(H)],
            &device,
        )?,
        Tensor::from_slice(
            &f32s(&case.initial_cell),
            [batch.of(B), hidden.of(H)],
            &device,
        )?,
    )?;
    let full = model.run_from(&base, &[initial.clone()])?;
    let first_values = chunk_values(&case.input, 0, 2);
    let second_values = chunk_values(&case.input, 2, 3);
    let first = Tensor::from_slice(
        &f32s(&first_values),
        [batch.of(B), time.of(2), input.of(I)],
        &device,
    )?;
    let second = Tensor::from_slice(
        &f32s(&second_values),
        [batch.of(B), time.of(1), input.of(I)],
        &device,
    )?;
    let first_run = model.run_from(&first, &[initial.clone()])?;
    let second_run = model.run_from(&second, &[first_run.state[0].clone()])?;
    close(
        "streamed sequence",
        &combine_chunks(
            &first_run.sequence.to_vec()?,
            &second_run.sequence.to_vec()?,
        ),
        &full
            .sequence
            .to_vec()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
        1e-6,
    );
    close(
        "streamed final hidden",
        &second_run.state[0].hidden.to_vec()?,
        &full.state[0]
            .hidden
            .to_vec()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
        1e-6,
    );
    close(
        "streamed final cell",
        &second_run.state[0].cell.to_vec()?,
        &full.state[0]
            .cell
            .to_vec()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
        1e-6,
    );

    model.zero_grad();
    let full_input = Tensor::from_slice(
        &f32s(&case.input),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_grad();
    let full_initial = LstmState::new(
        Tensor::from_slice(
            &f32s(&case.initial_hidden),
            [batch.of(B), hidden.of(H)],
            &device,
        )?
        .with_grad(),
        Tensor::from_slice(
            &f32s(&case.initial_cell),
            [batch.of(B), hidden.of(H)],
            &device,
        )?
        .with_grad(),
    )?;
    let full_run = model.run_from(&full_input, &[full_initial.clone()])?;
    terminal_loss(&full_run.state[0].clone(), batch, hidden)?.backward()?;
    let full_input_gradient = full_input.grad().unwrap().to_vec()?;
    let full_hidden_gradient = full_initial.hidden.grad().unwrap().to_vec()?;
    let full_cell_gradient = full_initial.cell.grad().unwrap().to_vec()?;
    let full_parameter_gradients: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| parameter.grad().unwrap().to_vec())
        .collect::<Result<_>>()?;

    model.zero_grad();
    let first_input = Tensor::from_slice(
        &f32s(&first_values),
        [batch.of(B), time.of(2), input.of(I)],
        &device,
    )?
    .with_grad();
    let second_input = Tensor::from_slice(
        &f32s(&second_values),
        [batch.of(B), time.of(1), input.of(I)],
        &device,
    )?
    .with_grad();
    let streamed_initial = LstmState::new(
        Tensor::from_slice(
            &f32s(&case.initial_hidden),
            [batch.of(B), hidden.of(H)],
            &device,
        )?
        .with_grad(),
        Tensor::from_slice(
            &f32s(&case.initial_cell),
            [batch.of(B), hidden.of(H)],
            &device,
        )?
        .with_grad(),
    )?;
    let first_run = model.run_from(&first_input, &[streamed_initial.clone()])?;
    let second_run = model.run_from(&second_input, &[first_run.state[0].clone()])?;
    terminal_loss(&second_run.state[0], batch, hidden)?.backward()?;
    close(
        "connected streamed input gradient",
        &combine_chunks(
            &first_input.grad().unwrap().to_vec()?,
            &second_input.grad().unwrap().to_vec()?,
        ),
        &full_input_gradient
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>(),
        5e-5,
    );
    close(
        "connected streamed initial hidden gradient",
        &streamed_initial.hidden.grad().unwrap().to_vec()?,
        &full_hidden_gradient
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>(),
        5e-5,
    );
    close(
        "connected streamed initial cell gradient",
        &streamed_initial.cell.grad().unwrap().to_vec()?,
        &full_cell_gradient
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>(),
        5e-5,
    );
    for (index, (streamed, full)) in model
        .parameters()
        .iter()
        .map(|parameter| parameter.grad().unwrap().to_vec())
        .collect::<Result<Vec<_>>>()?
        .iter()
        .zip(&full_parameter_gradients)
        .enumerate()
    {
        close(
            &format!("connected streamed parameter gradient {index}"),
            streamed,
            &full.iter().copied().map(f64::from).collect::<Vec<_>>(),
            5e-5,
        );
    }

    model.zero_grad();
    let detached_first = Tensor::from_slice(
        &f32s(&first_values),
        [batch.of(B), time.of(2), input.of(I)],
        &device,
    )?
    .with_grad();
    let detached_second = Tensor::from_slice(
        &f32s(&second_values),
        [batch.of(B), time.of(1), input.of(I)],
        &device,
    )?
    .with_grad();
    let detached_initial = LstmState::new(initial.hidden.with_grad(), initial.cell.with_grad())?;
    let first_run = model.run_from(&detached_first, &[detached_initial.clone()])?;
    let detached_state = first_run.state[0].detach();
    let detached_run = model.run_from(&detached_second, &[detached_state])?;
    close(
        "detached streaming preserves forward hidden",
        &detached_run.state[0].hidden.to_vec()?,
        &second_run.state[0]
            .hidden
            .to_vec()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
        1e-6,
    );
    terminal_loss(&detached_run.state[0], batch, hidden)?.backward()?;
    assert!(detached_first.grad().is_none());
    assert!(detached_initial.hidden.grad().is_none());
    assert!(detached_initial.cell.grad().is_none());
    assert!(detached_second.grad().is_some());

    let wrong = LstmState::new(
        Tensor::zeros([batch.of(B), hidden.of(H + 1)], &device)?,
        Tensor::zeros([batch.of(B), hidden.of(H + 1)], &device)?,
    )?;
    assert!(model.run_from(&base, &[wrong]).is_err());
    let other_device = Device::cuda(0)?;
    let wrong_device = LstmState::new(
        Tensor::zeros([batch.of(B), hidden.of(H)], &other_device)?,
        Tensor::zeros([batch.of(B), hidden.of(H)], &other_device)?,
    )?;
    assert!(model.run_from(&base, &[wrong_device]).is_err());

    let feature = Axis::new("feature");
    let mut square = Lstm::new(feature, feature.of(H), time)?;
    assert_eq!(
        square.build(
            &Shape::new([batch.of(B), time.of(T), feature.of(H)])?,
            &device,
            9,
        )?,
        Shape::new([batch.of(B), time.of(T), feature.of(H)])?
    );
    Ok(())
}

/// Forward sequence and terminal hidden state shared by the RNN and GRU
/// independent oracles below (neither family carries LSTM's separate cell
/// state).
struct SequenceReference {
    sequence: Vec<f64>,
    hidden: Vec<f64>,
}

// ---------------------------------------------------------------------
// RNN (Elman)
// ---------------------------------------------------------------------

#[derive(Clone)]
struct RnnCase {
    input: Vec<f64>,
    initial_hidden: Vec<f64>,
    input_weight: Vec<f64>,
    recurrent_weight: Vec<f64>,
    bias: Vec<f64>,
}

fn rnn_case() -> RnnCase {
    RnnCase {
        input: vec![
            0.2, -0.4, 0.7, 0.1, -0.3, 0.8, -0.6, 0.5, 0.9, -0.2, 0.4, 0.3,
        ],
        initial_hidden: vec![0.1, -0.2, 0.3, 0.05],
        input_weight: vec![0.15, -0.2, 0.3, 0.1],
        recurrent_weight: vec![-0.2, 0.1, 0.25, -0.35],
        bias: vec![0.05, -0.1],
    }
}

fn rnn_reference_step(
    case: &RnnCase,
    input: &[f64],
    hidden: &mut [f64],
    nonlinearity: fn(f64) -> f64,
) {
    let mut total = case.bias.clone();
    for (input_feature, &input_value) in input.iter().enumerate() {
        for (feature, value) in total.iter_mut().enumerate() {
            *value += input_value * case.input_weight[input_feature * H + feature];
        }
    }
    let previous = hidden.to_vec();
    for (previous_feature, &previous_value) in previous.iter().enumerate() {
        for (feature, value) in total.iter_mut().enumerate() {
            *value += previous_value * case.recurrent_weight[previous_feature * H + feature];
        }
    }
    for feature in 0..H {
        hidden[feature] = nonlinearity(total[feature]);
    }
}

fn rnn_reference(case: &RnnCase, nonlinearity: fn(f64) -> f64) -> SequenceReference {
    let mut hidden = case.initial_hidden.clone();
    let mut sequence = vec![0.0; B * T * H];
    for batch in 0..B {
        let mut h = hidden[batch * H..(batch + 1) * H].to_vec();
        for step in 0..T {
            let offset = (batch * T + step) * I;
            rnn_reference_step(case, &case.input[offset..offset + I], &mut h, nonlinearity);
            for feature in 0..H {
                sequence[(batch * T + step) * H + feature] = h[feature];
            }
        }
        hidden[batch * H..(batch + 1) * H].copy_from_slice(&h);
    }
    SequenceReference { sequence, hidden }
}

fn rnn_objective(case: &RnnCase, nonlinearity: fn(f64) -> f64) -> f64 {
    let result = rnn_reference(case, nonlinearity);
    let sequence_coefficients = [
        0.2, -0.3, 0.5, 0.1, -0.4, 0.7, -0.6, 0.8, 0.25, -0.15, 0.45, -0.35,
    ];
    let hidden_coefficients = [0.3, -0.2, 0.6, 0.1];
    result
        .sequence
        .iter()
        .zip(sequence_coefficients)
        .map(|(value, coefficient)| value * coefficient)
        .sum::<f64>()
        + result
            .hidden
            .iter()
            .zip(hidden_coefficients)
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>()
}

fn rnn_finite_difference(
    case: &RnnCase,
    nonlinearity: fn(f64) -> f64,
    field: fn(&mut RnnCase) -> &mut Vec<f64>,
) -> Vec<f64> {
    let epsilon = 1e-5;
    (0..field(&mut case.clone()).len())
        .map(|index| {
            let mut high = case.clone();
            field(&mut high)[index] += epsilon;
            let mut low = case.clone();
            field(&mut low)[index] -= epsilon;
            (rnn_objective(&high, nonlinearity) - rnn_objective(&low, nonlinearity))
                / (2.0 * epsilon)
        })
        .collect()
}

fn install_rnn(model: &Rnn, case: &RnnCase) -> Result<()> {
    model
        .parameter("input_weight")?
        .set_values(&f32s(&case.input_weight))?;
    model
        .parameter("recurrent_weight")?
        .set_values(&f32s(&case.recurrent_weight))?;
    model.parameter("bias")?.set_values(&f32s(&case.bias))?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn rnn_matches_independent_f64_forward_and_all_central_differences() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let case = rnn_case();
    let expected = rnn_reference(&case, f64::tanh);
    let mut model = Rnn::new(input, hidden.of(H), time)?;
    model.build(
        &Shape::new([batch.of(B), time.of(T), input.of(I)])?,
        &device,
        17,
    )?;
    install_rnn(&model, &case)?;

    let x = Tensor::from_slice(
        &f32s(&case.input),
        [batch.of(B), time.of(T), input.of(I)],
        &device,
    )?
    .with_layout([input, time, batch])?
    .with_grad();
    let initial = Tensor::from_slice(
        &f32s(&case.initial_hidden),
        [batch.of(B), hidden.of(H)],
        &device,
    )?
    .with_layout([hidden, batch])?
    .with_grad();
    let run = model.run_from(&x, &[initial.clone()])?;
    assert_eq!(
        run.sequence.shape(),
        &Shape::new([batch.of(B), time.of(T), hidden.of(H)])?
    );
    close(
        "RNN sequence",
        &run.sequence.to_vec()?,
        &expected.sequence,
        4e-5,
    );
    close(
        "RNN final hidden",
        &run.state[0].to_vec()?,
        &expected.hidden,
        4e-5,
    );
    close(
        "RNN final hidden equals last sequence coordinate",
        &run.sequence.select(time, T - 1)?.to_vec()?,
        &expected.hidden,
        4e-5,
    );

    let sequence_coefficients = [
        0.2, -0.3, 0.5, 0.1, -0.4, 0.7, -0.6, 0.8, 0.25, -0.15, 0.45, -0.35,
    ];
    let hidden_coefficients = [0.3, -0.2, 0.6, 0.1];
    let loss = weighted_sum(
        &run.sequence,
        &sequence_coefficients,
        &[batch, time, hidden],
    )?
    .add(&weighted_sum(
        &run.state[0],
        &hidden_coefficients,
        &[batch, hidden],
    )?)?;
    loss.backward()?;

    let gradients = [
        (
            "RNN input gradient",
            x.grad().unwrap().to_vec()?,
            rnn_finite_difference(&case, f64::tanh, |case| &mut case.input),
        ),
        (
            "RNN initial hidden gradient",
            initial.grad().unwrap().to_vec()?,
            rnn_finite_difference(&case, f64::tanh, |case| &mut case.initial_hidden),
        ),
        (
            "RNN input-weight gradient",
            model.parameter("input_weight")?.grad().unwrap().to_vec()?,
            rnn_finite_difference(&case, f64::tanh, |case| &mut case.input_weight),
        ),
        (
            "RNN recurrent-weight gradient",
            model
                .parameter("recurrent_weight")?
                .grad()
                .unwrap()
                .to_vec()?,
            rnn_finite_difference(&case, f64::tanh, |case| &mut case.recurrent_weight),
        ),
        (
            "RNN bias gradient",
            model.parameter("bias")?.grad().unwrap().to_vec()?,
            rnn_finite_difference(&case, f64::tanh, |case| &mut case.bias),
        ),
    ];
    for (name, actual, expected) in gradients {
        close(name, &actual, &expected, 3e-3);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn rnn_relu_executes_asymmetric_geometry_with_reordered_layout() -> Result<()> {
    const REPLICAS: usize = 2;
    const PATIENTS: usize = 3;

    let device = Device::cuda(0)?;
    let (replica, time, patient, input, hidden) = (
        Axis::new("replica"),
        Axis::new("time"),
        Axis::new("patient"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    assert!(Rnn::new(time, hidden.of(H), time).is_err());
    assert!(Rnn::new(input, time.of(H), time).is_err());

    let case = rnn_case();
    let input_values: Vec<_> = (0..REPLICAS * T * PATIENTS * I)
        .map(|index| ((index * 7 % 29) as f64 - 14.0) / 10.0)
        .collect();
    let input_shape = Shape::new([
        replica.of(REPLICAS),
        time.of(T),
        patient.of(PATIENTS),
        input.of(I),
    ])?;
    let mut model = Rnn::new(input, hidden.of(H), time)?.nonlinearity(RnnNonlinearity::Relu);
    model.build(&input_shape, &device, 41)?;
    install_rnn(&model, &case)?;
    assert_eq!(
        model
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["input_weight", "recurrent_weight", "bias"]
    );
    let tensor = Tensor::from_slice(
        &f32s(&input_values),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_layout([patient, input, replica, time])?;
    let run = model.run(&tensor)?;

    assert_eq!(
        run.sequence.shape(),
        &Shape::new([
            replica.of(REPLICAS),
            time.of(T),
            patient.of(PATIENTS),
            hidden.of(H),
        ])?
    );
    let state_shape = Shape::new([replica.of(REPLICAS), patient.of(PATIENTS), hidden.of(H)])?;
    assert_eq!(run.state[0].shape(), &state_shape);

    let relu = |value: f64| value.max(0.0);
    let mut expected_sequence = vec![0.0; REPLICAS * T * PATIENTS * H];
    let mut expected_hidden = vec![0.0; REPLICAS * PATIENTS * H];
    for replica_index in 0..REPLICAS {
        for patient_index in 0..PATIENTS {
            let mut stream_hidden = vec![0.0; H];
            for step in 0..T {
                let input_offset = ((replica_index * T + step) * PATIENTS + patient_index) * I;
                rnn_reference_step(
                    &case,
                    &input_values[input_offset..input_offset + I],
                    &mut stream_hidden,
                    relu,
                );
                let output_offset = ((replica_index * T + step) * PATIENTS + patient_index) * H;
                expected_sequence[output_offset..output_offset + H].copy_from_slice(&stream_hidden);
            }
            let state_offset = (replica_index * PATIENTS + patient_index) * H;
            expected_hidden[state_offset..state_offset + H].copy_from_slice(&stream_hidden);
        }
    }
    close(
        "asymmetric RNN sequence",
        &run.sequence.to_vec()?,
        &expected_sequence,
        4e-5,
    );
    close(
        "asymmetric RNN hidden state",
        &run.state[0].to_vec()?,
        &expected_hidden,
        4e-5,
    );

    let wrong = Tensor::zeros(
        [replica.of(REPLICAS), patient.of(PATIENTS), hidden.of(H + 1)],
        &device,
    )?;
    assert!(model.run_from(&tensor, &[wrong]).is_err());
    let other_device = Device::cuda(0)?;
    let wrong_device = Tensor::zeros(
        [replica.of(REPLICAS), patient.of(PATIENTS), hidden.of(H)],
        &other_device,
    )?;
    assert!(model.run_from(&tensor, &[wrong_device]).is_err());
    Ok(())
}

// ---------------------------------------------------------------------
// GRU
// ---------------------------------------------------------------------

#[derive(Clone)]
struct GruCase {
    input: Vec<f64>,
    initial_hidden: Vec<f64>,
    input_weight: Vec<f64>,
    recurrent_weight: Vec<f64>,
    bias: Vec<f64>,
}

fn gru_case() -> GruCase {
    GruCase {
        input: vec![
            0.2, -0.4, 0.7, 0.1, -0.3, 0.8, -0.6, 0.5, 0.9, -0.2, 0.4, 0.3,
        ],
        initial_hidden: vec![0.1, -0.2, 0.3, 0.05],
        input_weight: vec![
            0.15, -0.2, 0.3, 0.1, -0.25, 0.4, 0.05, -0.3, -0.1, 0.35, -0.2, 0.25,
        ],
        recurrent_weight: vec![
            -0.2, 0.1, 0.25, -0.35, 0.3, 0.15, -0.1, 0.2, 0.4, -0.25, 0.05, 0.3,
        ],
        bias: vec![0.05, -0.1, 0.2, 0.15, -0.05, 0.08],
    }
}

/// Independent host reference for PyTorch's GRU gate order (`r, z, n`) with
/// the reset gate multiplying the hidden-side candidate contribution after
/// its own matrix product, matching `GruCell`'s doc comment exactly.
fn gru_reference_step(case: &GruCase, input: &[f64], hidden: &mut [f64]) {
    let mut input_gates = case.bias.clone();
    for (input_feature, &input_value) in input.iter().enumerate() {
        for (gate, value) in input_gates.iter_mut().enumerate() {
            *value += input_value * case.input_weight[input_feature * 3 * H + gate];
        }
    }
    let mut hidden_gates = [0.0; 3 * H];
    for (previous_feature, &previous_value) in hidden.iter().enumerate() {
        for (gate, value) in hidden_gates.iter_mut().enumerate() {
            *value += previous_value * case.recurrent_weight[previous_feature * 3 * H + gate];
        }
    }
    for feature in 0..H {
        let reset = sigmoid(input_gates[feature] + hidden_gates[feature]);
        let update = sigmoid(input_gates[H + feature] + hidden_gates[H + feature]);
        let candidate =
            (input_gates[2 * H + feature] + reset * hidden_gates[2 * H + feature]).tanh();
        hidden[feature] = (1.0 - update) * candidate + update * hidden[feature];
    }
}

fn gru_reference(case: &GruCase) -> SequenceReference {
    let mut hidden = case.initial_hidden.clone();
    let mut sequence = vec![0.0; B * T * H];
    for batch in 0..B {
        let mut h = hidden[batch * H..(batch + 1) * H].to_vec();
        for step in 0..T {
            let offset = (batch * T + step) * I;
            gru_reference_step(case, &case.input[offset..offset + I], &mut h);
            for feature in 0..H {
                sequence[(batch * T + step) * H + feature] = h[feature];
            }
        }
        hidden[batch * H..(batch + 1) * H].copy_from_slice(&h);
    }
    SequenceReference { sequence, hidden }
}

fn gru_objective(case: &GruCase) -> f64 {
    let result = gru_reference(case);
    let sequence_coefficients = [
        0.2, -0.3, 0.5, 0.1, -0.4, 0.7, -0.6, 0.8, 0.25, -0.15, 0.45, -0.35,
    ];
    let hidden_coefficients = [0.3, -0.2, 0.6, 0.1];
    result
        .sequence
        .iter()
        .zip(sequence_coefficients)
        .map(|(value, coefficient)| value * coefficient)
        .sum::<f64>()
        + result
            .hidden
            .iter()
            .zip(hidden_coefficients)
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>()
}

fn gru_finite_difference(case: &GruCase, field: fn(&mut GruCase) -> &mut Vec<f64>) -> Vec<f64> {
    let epsilon = 1e-5;
    (0..field(&mut case.clone()).len())
        .map(|index| {
            let mut high = case.clone();
            field(&mut high)[index] += epsilon;
            let mut low = case.clone();
            field(&mut low)[index] -= epsilon;
            (gru_objective(&high) - gru_objective(&low)) / (2.0 * epsilon)
        })
        .collect()
}

fn install_gru(model: &Gru, case: &GruCase) -> Result<()> {
    model
        .parameter("input_weight")?
        .set_values(&f32s(&case.input_weight))?;
    model
        .parameter("recurrent_weight")?
        .set_values(&f32s(&case.recurrent_weight))?;
    model.parameter("bias")?.set_values(&f32s(&case.bias))?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gru_matches_independent_f64_forward_and_all_central_differences() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let case = gru_case();
    let expected = gru_reference(&case);
    let mut model = Gru::new(input, hidden.of(H), time)?;
    model.build(
        &Shape::new([batch.of(B), time.of(T), input.of(I)])?,
        &device,
        17,
    )?;
    install_gru(&model, &case)?;

    let x = Tensor::from_slice(
        &f32s(&case.input),
        [batch.of(B), time.of(T), input.of(I)],
        &device,
    )?
    .with_layout([input, time, batch])?
    .with_grad();
    let initial = Tensor::from_slice(
        &f32s(&case.initial_hidden),
        [batch.of(B), hidden.of(H)],
        &device,
    )?
    .with_layout([hidden, batch])?
    .with_grad();
    let run = model.run_from(&x, &[initial.clone()])?;
    assert_eq!(
        run.sequence.shape(),
        &Shape::new([batch.of(B), time.of(T), hidden.of(H)])?
    );
    close(
        "GRU sequence",
        &run.sequence.to_vec()?,
        &expected.sequence,
        4e-5,
    );
    close(
        "GRU final hidden",
        &run.state[0].to_vec()?,
        &expected.hidden,
        4e-5,
    );
    close(
        "GRU final hidden equals last sequence coordinate",
        &run.sequence.select(time, T - 1)?.to_vec()?,
        &expected.hidden,
        4e-5,
    );

    let sequence_coefficients = [
        0.2, -0.3, 0.5, 0.1, -0.4, 0.7, -0.6, 0.8, 0.25, -0.15, 0.45, -0.35,
    ];
    let hidden_coefficients = [0.3, -0.2, 0.6, 0.1];
    let loss = weighted_sum(
        &run.sequence,
        &sequence_coefficients,
        &[batch, time, hidden],
    )?
    .add(&weighted_sum(
        &run.state[0],
        &hidden_coefficients,
        &[batch, hidden],
    )?)?;
    loss.backward()?;

    let gradients = [
        (
            "GRU input gradient",
            x.grad().unwrap().to_vec()?,
            gru_finite_difference(&case, |case| &mut case.input),
        ),
        (
            "GRU initial hidden gradient",
            initial.grad().unwrap().to_vec()?,
            gru_finite_difference(&case, |case| &mut case.initial_hidden),
        ),
        (
            "GRU input-weight gradient",
            model.parameter("input_weight")?.grad().unwrap().to_vec()?,
            gru_finite_difference(&case, |case| &mut case.input_weight),
        ),
        (
            "GRU recurrent-weight gradient",
            model
                .parameter("recurrent_weight")?
                .grad()
                .unwrap()
                .to_vec()?,
            gru_finite_difference(&case, |case| &mut case.recurrent_weight),
        ),
        (
            "GRU bias gradient",
            model.parameter("bias")?.grad().unwrap().to_vec()?,
            gru_finite_difference(&case, |case| &mut case.bias),
        ),
    ];
    for (name, actual, expected) in gradients {
        close(name, &actual, &expected, 3e-3);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gru_executes_asymmetric_geometry_with_reordered_layout() -> Result<()> {
    const REPLICAS: usize = 2;
    const PATIENTS: usize = 3;

    let device = Device::cuda(0)?;
    let (replica, time, patient, input, hidden) = (
        Axis::new("replica"),
        Axis::new("time"),
        Axis::new("patient"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    assert!(Gru::new(time, hidden.of(H), time).is_err());
    assert!(Gru::new(input, time.of(H), time).is_err());

    let case = gru_case();
    let input_values: Vec<_> = (0..REPLICAS * T * PATIENTS * I)
        .map(|index| ((index * 7 % 29) as f64 - 14.0) / 10.0)
        .collect();
    let input_shape = Shape::new([
        replica.of(REPLICAS),
        time.of(T),
        patient.of(PATIENTS),
        input.of(I),
    ])?;
    let mut model = Gru::new(input, hidden.of(H), time)?;
    model.build(&input_shape, &device, 41)?;
    install_gru(&model, &case)?;
    assert_eq!(
        model
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["input_weight", "recurrent_weight", "bias"]
    );
    let tensor = Tensor::from_slice(
        &f32s(&input_values),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_layout([patient, input, replica, time])?;
    let run = model.run(&tensor)?;

    assert_eq!(
        run.sequence.shape(),
        &Shape::new([
            replica.of(REPLICAS),
            time.of(T),
            patient.of(PATIENTS),
            hidden.of(H),
        ])?
    );
    let state_shape = Shape::new([replica.of(REPLICAS), patient.of(PATIENTS), hidden.of(H)])?;
    assert_eq!(run.state[0].shape(), &state_shape);

    let mut expected_sequence = vec![0.0; REPLICAS * T * PATIENTS * H];
    let mut expected_hidden = vec![0.0; REPLICAS * PATIENTS * H];
    for replica_index in 0..REPLICAS {
        for patient_index in 0..PATIENTS {
            let mut stream_hidden = vec![0.0; H];
            for step in 0..T {
                let input_offset = ((replica_index * T + step) * PATIENTS + patient_index) * I;
                gru_reference_step(
                    &case,
                    &input_values[input_offset..input_offset + I],
                    &mut stream_hidden,
                );
                let output_offset = ((replica_index * T + step) * PATIENTS + patient_index) * H;
                expected_sequence[output_offset..output_offset + H].copy_from_slice(&stream_hidden);
            }
            let state_offset = (replica_index * PATIENTS + patient_index) * H;
            expected_hidden[state_offset..state_offset + H].copy_from_slice(&stream_hidden);
        }
    }
    close(
        "asymmetric GRU sequence",
        &run.sequence.to_vec()?,
        &expected_sequence,
        4e-5,
    );
    close(
        "asymmetric GRU hidden state",
        &run.state[0].to_vec()?,
        &expected_hidden,
        4e-5,
    );

    let wrong = Tensor::zeros(
        [replica.of(REPLICAS), patient.of(PATIENTS), hidden.of(H + 1)],
        &device,
    )?;
    assert!(model.run_from(&tensor, &[wrong]).is_err());
    let other_device = Device::cuda(0)?;
    let wrong_device = Tensor::zeros(
        [replica.of(REPLICAS), patient.of(PATIENTS), hidden.of(H)],
        &other_device,
    )?;
    assert!(model.run_from(&tensor, &[wrong_device]).is_err());
    Ok(())
}

// ---------------------------------------------------------------------
// bias = false
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA"]
fn recurrent_families_bias_false_omit_bias_parameter_and_run() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let shape = Shape::new([batch.of(B), time.of(T), input.of(I)])?;
    let x = Tensor::from_slice(&f32s(&case().input), shape.dims().iter().copied(), &device)?;

    let config_no_bias = RecurrentConfig {
        num_layers: 1,
        bidirectional: false,
        bias: false,
    };

    // Single-layer, unidirectional: unprefixed names, exactly as before
    // `bias` existed, just without the "bias" entry.
    let mut rnn = Rnn::with_config(input, hidden.of(H), time, config_no_bias)?;
    rnn.build(&shape, &device, 5)?;
    assert_eq!(
        rnn.named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["input_weight", "recurrent_weight"]
    );
    rnn.run(&x)?;

    let mut gru = Gru::with_config(input, hidden.of(H), time, config_no_bias)?;
    gru.build(&shape, &device, 6)?;
    assert_eq!(
        gru.named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["input_weight", "recurrent_weight"]
    );
    gru.run(&x)?;

    let mut lstm = Lstm::with_config(input, hidden.of(H), time, config_no_bias)?;
    lstm.build(&shape, &device, 7)?;
    assert_eq!(
        lstm.named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["input_weight", "recurrent_weight"]
    );
    lstm.run(&x)?;

    // The default (bias = true, single layer, unidirectional) still carries
    // an unprefixed "bias" entry: bit-exact with this type before `bias`
    // existed as a toggle.
    let mut rnn_with_bias = Rnn::new(input, hidden.of(H), time)?;
    rnn_with_bias.build(&shape, &device, 5)?;
    assert_eq!(
        rnn_with_bias
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["input_weight", "recurrent_weight", "bias"]
    );

    // A stacked or bidirectional configuration prefixes every name with its
    // layer and direction, since it then owns more than one of each, and
    // `bias = false` still omits every "bias" entry.
    let config_stacked_no_bias = RecurrentConfig {
        num_layers: 2,
        bidirectional: true,
        bias: false,
    };
    let mut stacked = Gru::with_config(input, hidden.of(H), time, config_stacked_no_bias)?;
    stacked.build(&shape, &device, 8)?;
    assert_eq!(
        stacked
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        [
            "layer0_forward_input_weight",
            "layer0_forward_recurrent_weight",
            "layer0_backward_input_weight",
            "layer0_backward_recurrent_weight",
            "layer1_forward_input_weight",
            "layer1_forward_recurrent_weight",
            "layer1_backward_input_weight",
            "layer1_backward_recurrent_weight",
        ]
    );
    stacked.run(&x)?;
    Ok(())
}

// ---------------------------------------------------------------------
// Standalone Cell tests: asymmetric geometry with reordered storage
// ---------------------------------------------------------------------

fn sigmoid_f64(value: f64) -> f64 {
    sigmoid(value)
}

fn rnn_cell_step(
    hidden_extent: usize,
    input: &[f64],
    hidden: &mut [f64],
    input_weight: &[f64],
    recurrent_weight: &[f64],
    bias: &[f64],
    nonlinearity: fn(f64) -> f64,
) {
    let mut total = bias.to_vec();
    for (input_feature, &input_value) in input.iter().enumerate() {
        for feature in 0..hidden_extent {
            total[feature] += input_value * input_weight[input_feature * hidden_extent + feature];
        }
    }
    let previous = hidden.to_vec();
    for (previous_feature, &previous_value) in previous.iter().enumerate() {
        for feature in 0..hidden_extent {
            total[feature] +=
                previous_value * recurrent_weight[previous_feature * hidden_extent + feature];
        }
    }
    for feature in 0..hidden_extent {
        hidden[feature] = nonlinearity(total[feature]);
    }
}

fn gru_cell_step(
    hidden_extent: usize,
    input: &[f64],
    hidden: &mut [f64],
    input_weight: &[f64],
    recurrent_weight: &[f64],
    bias: &[f64],
) {
    let gate_extent = 3 * hidden_extent;
    let mut input_gates = bias.to_vec();
    for (input_feature, &input_value) in input.iter().enumerate() {
        for gate in 0..gate_extent {
            input_gates[gate] += input_value * input_weight[input_feature * gate_extent + gate];
        }
    }
    let mut hidden_gates = vec![0.0; gate_extent];
    for (previous_feature, &previous_value) in hidden.iter().enumerate() {
        for gate in 0..gate_extent {
            hidden_gates[gate] +=
                previous_value * recurrent_weight[previous_feature * gate_extent + gate];
        }
    }
    for feature in 0..hidden_extent {
        let reset = sigmoid_f64(input_gates[feature] + hidden_gates[feature]);
        let update = sigmoid_f64(
            input_gates[hidden_extent + feature] + hidden_gates[hidden_extent + feature],
        );
        let candidate = (input_gates[2 * hidden_extent + feature]
            + reset * hidden_gates[2 * hidden_extent + feature])
            .tanh();
        hidden[feature] = (1.0 - update) * candidate + update * hidden[feature];
    }
}

fn lstm_cell_step(
    hidden_extent: usize,
    input: &[f64],
    hidden: &mut [f64],
    cell: &mut [f64],
    input_weight: &[f64],
    recurrent_weight: &[f64],
    bias: &[f64],
) {
    let gate_extent = 4 * hidden_extent;
    let mut gates = bias.to_vec();
    for (input_feature, &input_value) in input.iter().enumerate() {
        for gate in 0..gate_extent {
            gates[gate] += input_value * input_weight[input_feature * gate_extent + gate];
        }
    }
    let previous = hidden.to_vec();
    for (previous_feature, &previous_value) in previous.iter().enumerate() {
        for gate in 0..gate_extent {
            gates[gate] += previous_value * recurrent_weight[previous_feature * gate_extent + gate];
        }
    }
    for feature in 0..hidden_extent {
        let input_gate = sigmoid_f64(gates[feature]);
        let forget_gate = sigmoid_f64(gates[hidden_extent + feature]);
        let candidate = gates[2 * hidden_extent + feature].tanh();
        let output_gate = sigmoid_f64(gates[3 * hidden_extent + feature]);
        cell[feature] = forget_gate * cell[feature] + input_gate * candidate;
        hidden[feature] = output_gate * cell[feature].tanh();
    }
}

/// Deterministic synthetic parameter values for the multi-layer/bidirectional
/// oracle tests below. These are inputs to the independent host reference,
/// not the expected values under test, so a simple reproducible generator
/// (rather than typed-out literals) is sufficient; it deliberately differs
/// from the crate's own xorshift build-time initializer.
fn synth(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            (bits as f64 / u32::MAX as f64 - 0.5) * 0.6
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA"]
fn rnn_cell_standalone_asymmetric_geometry_with_reordered_layout() -> Result<()> {
    const REPLICAS: usize = 2;
    const PATIENTS: usize = 3;
    const CELL_I: usize = 3;
    const CELL_H: usize = 2;

    let device = Device::cuda(0)?;
    let (replica, patient, input, hidden) = (
        Axis::new("replica"),
        Axis::new("patient"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let input_weight = synth(CELL_I * CELL_H, 101);
    let recurrent_weight = synth(CELL_H * CELL_H, 102);
    let bias = synth(CELL_H, 103);
    let input_values = synth(REPLICAS * PATIENTS * CELL_I, 104);
    let initial_values = synth(REPLICAS * PATIENTS * CELL_H, 105);

    let input_shape = Shape::new([replica.of(REPLICAS), patient.of(PATIENTS), input.of(CELL_I)])?;
    let mut cell = RnnCell::new(input, hidden.of(CELL_H))?;
    cell.build(&input_shape, &device, 41)?;
    cell.parameter("input_weight")?
        .set_values(&f32s(&input_weight))?;
    cell.parameter("recurrent_weight")?
        .set_values(&f32s(&recurrent_weight))?;
    cell.parameter("bias")?.set_values(&f32s(&bias))?;

    let x = Tensor::from_slice(
        &f32s(&input_values),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_layout([patient, input, replica])?;
    let state = Tensor::from_slice(
        &f32s(&initial_values),
        [
            replica.of(REPLICAS),
            patient.of(PATIENTS),
            hidden.of(CELL_H),
        ],
        &device,
    )?
    .with_layout([hidden, replica, patient])?;
    let next = cell.step(&x, &state)?;

    let mut expected = vec![0.0; REPLICAS * PATIENTS * CELL_H];
    for replica_index in 0..REPLICAS {
        for patient_index in 0..PATIENTS {
            let stream = replica_index * PATIENTS + patient_index;
            let mut h = initial_values[stream * CELL_H..(stream + 1) * CELL_H].to_vec();
            rnn_cell_step(
                CELL_H,
                &input_values[stream * CELL_I..(stream + 1) * CELL_I],
                &mut h,
                &input_weight,
                &recurrent_weight,
                &bias,
                f64::tanh,
            );
            expected[stream * CELL_H..(stream + 1) * CELL_H].copy_from_slice(&h);
        }
    }
    close(
        "standalone RnnCell asymmetric+reordered",
        &next.to_vec()?,
        &expected,
        4e-5,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gru_cell_standalone_asymmetric_geometry_with_reordered_layout() -> Result<()> {
    const REPLICAS: usize = 2;
    const PATIENTS: usize = 3;
    const CELL_I: usize = 3;
    const CELL_H: usize = 2;

    let device = Device::cuda(0)?;
    let (replica, patient, input, hidden) = (
        Axis::new("replica"),
        Axis::new("patient"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let input_weight = synth(CELL_I * 3 * CELL_H, 111);
    let recurrent_weight = synth(CELL_H * 3 * CELL_H, 112);
    let bias = synth(3 * CELL_H, 113);
    let input_values = synth(REPLICAS * PATIENTS * CELL_I, 114);
    let initial_values = synth(REPLICAS * PATIENTS * CELL_H, 115);

    let input_shape = Shape::new([replica.of(REPLICAS), patient.of(PATIENTS), input.of(CELL_I)])?;
    let mut cell = GruCell::new(input, hidden.of(CELL_H))?;
    cell.build(&input_shape, &device, 43)?;
    cell.parameter("input_weight")?
        .set_values(&f32s(&input_weight))?;
    cell.parameter("recurrent_weight")?
        .set_values(&f32s(&recurrent_weight))?;
    cell.parameter("bias")?.set_values(&f32s(&bias))?;

    let x = Tensor::from_slice(
        &f32s(&input_values),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_layout([patient, input, replica])?;
    let state = Tensor::from_slice(
        &f32s(&initial_values),
        [
            replica.of(REPLICAS),
            patient.of(PATIENTS),
            hidden.of(CELL_H),
        ],
        &device,
    )?
    .with_layout([hidden, replica, patient])?;
    let next = cell.step(&x, &state)?;

    let mut expected = vec![0.0; REPLICAS * PATIENTS * CELL_H];
    for replica_index in 0..REPLICAS {
        for patient_index in 0..PATIENTS {
            let stream = replica_index * PATIENTS + patient_index;
            let mut h = initial_values[stream * CELL_H..(stream + 1) * CELL_H].to_vec();
            gru_cell_step(
                CELL_H,
                &input_values[stream * CELL_I..(stream + 1) * CELL_I],
                &mut h,
                &input_weight,
                &recurrent_weight,
                &bias,
            );
            expected[stream * CELL_H..(stream + 1) * CELL_H].copy_from_slice(&h);
        }
    }
    close(
        "standalone GruCell asymmetric+reordered",
        &next.to_vec()?,
        &expected,
        4e-5,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lstm_cell_standalone_asymmetric_geometry_with_reordered_layout() -> Result<()> {
    const REPLICAS: usize = 2;
    const PATIENTS: usize = 3;
    const CELL_I: usize = 3;
    const CELL_H: usize = 2;

    let device = Device::cuda(0)?;
    let (replica, patient, input, hidden) = (
        Axis::new("replica"),
        Axis::new("patient"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let input_weight = synth(CELL_I * 4 * CELL_H, 121);
    let recurrent_weight = synth(CELL_H * 4 * CELL_H, 122);
    let bias = synth(4 * CELL_H, 123);
    let input_values = synth(REPLICAS * PATIENTS * CELL_I, 124);
    let initial_hidden = synth(REPLICAS * PATIENTS * CELL_H, 125);
    let initial_cell = synth(REPLICAS * PATIENTS * CELL_H, 126);

    let input_shape = Shape::new([replica.of(REPLICAS), patient.of(PATIENTS), input.of(CELL_I)])?;
    let mut cell = LstmCell::new(input, hidden.of(CELL_H))?;
    cell.build(&input_shape, &device, 45)?;
    cell.parameter("input_weight")?
        .set_values(&f32s(&input_weight))?;
    cell.parameter("recurrent_weight")?
        .set_values(&f32s(&recurrent_weight))?;
    cell.parameter("bias")?.set_values(&f32s(&bias))?;

    let x = Tensor::from_slice(
        &f32s(&input_values),
        input_shape.dims().iter().copied(),
        &device,
    )?
    .with_layout([patient, input, replica])?;
    let hidden_state = Tensor::from_slice(
        &f32s(&initial_hidden),
        [
            replica.of(REPLICAS),
            patient.of(PATIENTS),
            hidden.of(CELL_H),
        ],
        &device,
    )?
    .with_layout([hidden, replica, patient])?;
    let cell_state = Tensor::from_slice(
        &f32s(&initial_cell),
        [
            replica.of(REPLICAS),
            patient.of(PATIENTS),
            hidden.of(CELL_H),
        ],
        &device,
    )?
    .with_layout([replica, hidden, patient])?;
    let state = LstmState::new(hidden_state, cell_state)?;
    let next = cell.step(&x, &state)?;

    let mut expected_hidden = vec![0.0; REPLICAS * PATIENTS * CELL_H];
    let mut expected_cell = vec![0.0; REPLICAS * PATIENTS * CELL_H];
    for replica_index in 0..REPLICAS {
        for patient_index in 0..PATIENTS {
            let stream = replica_index * PATIENTS + patient_index;
            let mut h = initial_hidden[stream * CELL_H..(stream + 1) * CELL_H].to_vec();
            let mut c = initial_cell[stream * CELL_H..(stream + 1) * CELL_H].to_vec();
            lstm_cell_step(
                CELL_H,
                &input_values[stream * CELL_I..(stream + 1) * CELL_I],
                &mut h,
                &mut c,
                &input_weight,
                &recurrent_weight,
                &bias,
            );
            expected_hidden[stream * CELL_H..(stream + 1) * CELL_H].copy_from_slice(&h);
            expected_cell[stream * CELL_H..(stream + 1) * CELL_H].copy_from_slice(&c);
        }
    }
    close(
        "standalone LstmCell asymmetric+reordered hidden",
        &next.hidden.to_vec()?,
        &expected_hidden,
        4e-5,
    );
    close(
        "standalone LstmCell asymmetric+reordered cell",
        &next.cell.to_vec()?,
        &expected_cell,
        4e-5,
    );
    Ok(())
}

// ---------------------------------------------------------------------
// Two-layer bidirectional independent f64 oracle, one per family
// ---------------------------------------------------------------------

#[derive(Clone)]
struct DirectionWeights {
    input_weight: Vec<f64>,
    recurrent_weight: Vec<f64>,
    bias: Vec<f64>,
}

fn direction_weights(gate_multiple: usize, input_extent: usize, seed: u64) -> DirectionWeights {
    let gate_extent = gate_multiple * H;
    DirectionWeights {
        input_weight: synth(input_extent * gate_extent, seed),
        recurrent_weight: synth(H * gate_extent, seed + 1),
        bias: synth(gate_extent, seed + 2),
    }
}

#[derive(Clone)]
struct BiStack {
    layer0_forward: DirectionWeights,
    layer0_backward: DirectionWeights,
    layer1_forward: DirectionWeights,
    layer1_backward: DirectionWeights,
}

fn bi_stack(gate_multiple: usize, seed: u64) -> BiStack {
    BiStack {
        layer0_forward: direction_weights(gate_multiple, I, seed),
        layer0_backward: direction_weights(gate_multiple, I, seed + 10),
        layer1_forward: direction_weights(gate_multiple, 2 * H, seed + 20),
        layer1_backward: direction_weights(gate_multiple, 2 * H, seed + 30),
    }
}

#[derive(Clone)]
struct BiCase {
    input: Vec<f64>,
    initial: [Vec<f64>; 4],
    initial_cell: [Vec<f64>; 4],
    stack: BiStack,
}

fn bi_case(gate_multiple: usize, seed: u64) -> BiCase {
    BiCase {
        input: synth(B * T * I, seed),
        initial: [
            synth(B * H, seed + 1),
            synth(B * H, seed + 2),
            synth(B * H, seed + 3),
            synth(B * H, seed + 4),
        ],
        initial_cell: [
            synth(B * H, seed + 5),
            synth(B * H, seed + 6),
            synth(B * H, seed + 7),
            synth(B * H, seed + 8),
        ],
        stack: bi_stack(gate_multiple, seed + 100),
    }
}

struct BiReference {
    sequence: Vec<f64>,
    finals_hidden: [Vec<f64>; 4],
    finals_cell: [Vec<f64>; 4],
}

/// Independent host reference for a 2-layer, bidirectional RNN/GRU/LSTM
/// stack, matching the semantics `Rnn`/`Gru`/`Lstm::with_config` implement:
/// each layer runs a forward transition over `0..T` and a backward
/// transition over `T-1..0`, concatenates `[forward; backward]` on the
/// hidden axis (PyTorch's order) to form that layer's `2H`-wide output, and
/// feeds it to the next layer verbatim. `step` is one of `rnn_cell_step`,
/// `gru_cell_step` or a wrapped `lstm_cell_step` (LSTM alone tracks cell
/// state), never the Axis code under test.
fn bi_reference<F>(case: &BiCase, mut step: F) -> BiReference
where
    F: FnMut(usize, &[f64], &mut [f64], &mut [f64], &DirectionWeights),
{
    let mut sequence = vec![0.0; B * T * 2 * H];
    let mut finals_hidden: [Vec<f64>; 4] = Default::default();
    let mut finals_cell: [Vec<f64>; 4] = Default::default();
    for slot in 0..4 {
        finals_hidden[slot] = vec![0.0; B * H];
        finals_cell[slot] = vec![0.0; B * H];
    }
    for batch in 0..B {
        let mut layer0_seq = vec![0.0; T * 2 * H];
        let mut h = case.initial[0][batch * H..(batch + 1) * H].to_vec();
        let mut c = case.initial_cell[0][batch * H..(batch + 1) * H].to_vec();
        for t in 0..T {
            let off = (batch * T + t) * I;
            step(
                H,
                &case.input[off..off + I],
                &mut h,
                &mut c,
                &case.stack.layer0_forward,
            );
            layer0_seq[t * 2 * H..t * 2 * H + H].copy_from_slice(&h);
        }
        finals_hidden[0][batch * H..(batch + 1) * H].copy_from_slice(&h);
        finals_cell[0][batch * H..(batch + 1) * H].copy_from_slice(&c);

        let mut h = case.initial[1][batch * H..(batch + 1) * H].to_vec();
        let mut c = case.initial_cell[1][batch * H..(batch + 1) * H].to_vec();
        for t in (0..T).rev() {
            let off = (batch * T + t) * I;
            step(
                H,
                &case.input[off..off + I],
                &mut h,
                &mut c,
                &case.stack.layer0_backward,
            );
            layer0_seq[t * 2 * H + H..t * 2 * H + 2 * H].copy_from_slice(&h);
        }
        finals_hidden[1][batch * H..(batch + 1) * H].copy_from_slice(&h);
        finals_cell[1][batch * H..(batch + 1) * H].copy_from_slice(&c);

        let mut layer1_seq = vec![0.0; T * 2 * H];
        let mut h = case.initial[2][batch * H..(batch + 1) * H].to_vec();
        let mut c = case.initial_cell[2][batch * H..(batch + 1) * H].to_vec();
        for t in 0..T {
            let off = t * 2 * H;
            step(
                H,
                &layer0_seq[off..off + 2 * H],
                &mut h,
                &mut c,
                &case.stack.layer1_forward,
            );
            layer1_seq[t * 2 * H..t * 2 * H + H].copy_from_slice(&h);
        }
        finals_hidden[2][batch * H..(batch + 1) * H].copy_from_slice(&h);
        finals_cell[2][batch * H..(batch + 1) * H].copy_from_slice(&c);

        let mut h = case.initial[3][batch * H..(batch + 1) * H].to_vec();
        let mut c = case.initial_cell[3][batch * H..(batch + 1) * H].to_vec();
        for t in (0..T).rev() {
            let off = t * 2 * H;
            step(
                H,
                &layer0_seq[off..off + 2 * H],
                &mut h,
                &mut c,
                &case.stack.layer1_backward,
            );
            layer1_seq[t * 2 * H + H..t * 2 * H + 2 * H].copy_from_slice(&h);
        }
        finals_hidden[3][batch * H..(batch + 1) * H].copy_from_slice(&h);
        finals_cell[3][batch * H..(batch + 1) * H].copy_from_slice(&c);

        for t in 0..T {
            let out_off = (batch * T + t) * 2 * H;
            sequence[out_off..out_off + 2 * H]
                .copy_from_slice(&layer1_seq[t * 2 * H..t * 2 * H + 2 * H]);
        }
    }
    BiReference {
        sequence,
        finals_hidden,
        finals_cell,
    }
}

fn bi_objective(result: &BiReference) -> f64 {
    let sequence_coefficients = synth(B * T * 2 * H, 900);
    let mut total: f64 = result
        .sequence
        .iter()
        .zip(&sequence_coefficients)
        .map(|(value, coefficient)| value * coefficient)
        .sum();
    for slot in 0..4 {
        let hidden_coefficients = synth(B * H, 910 + slot as u64);
        total += result.finals_hidden[slot]
            .iter()
            .zip(&hidden_coefficients)
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>();
        let cell_coefficients = synth(B * H, 920 + slot as u64);
        total += result.finals_cell[slot]
            .iter()
            .zip(&cell_coefficients)
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>();
    }
    total
}

fn bi_finite_difference<F>(
    case: &BiCase,
    step: F,
    field: impl Fn(&mut BiCase) -> &mut Vec<f64>,
) -> Vec<f64>
where
    F: Fn(usize, &[f64], &mut [f64], &mut [f64], &DirectionWeights) + Copy,
{
    let epsilon = 1e-5;
    let mut probe = case.clone();
    let len = field(&mut probe).len();
    (0..len)
        .map(|index| {
            let mut high = case.clone();
            field(&mut high)[index] += epsilon;
            let mut low = case.clone();
            field(&mut low)[index] -= epsilon;
            (bi_objective(&bi_reference(&high, step)) - bi_objective(&bi_reference(&low, step)))
                / (2.0 * epsilon)
        })
        .collect()
}

fn install_bi_direction(cell: &dyn Module, weights: &DirectionWeights, prefix: &str) -> Result<()> {
    cell.parameter(&format!("{prefix}_input_weight"))?
        .set_values(&f32s(&weights.input_weight))?;
    cell.parameter(&format!("{prefix}_recurrent_weight"))?
        .set_values(&f32s(&weights.recurrent_weight))?;
    cell.parameter(&format!("{prefix}_bias"))?
        .set_values(&f32s(&weights.bias))?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn rnn_two_layer_bidirectional_matches_independent_f64_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let case = bi_case(1, 200);
    let step = |h: usize, x: &[f64], hid: &mut [f64], _cell: &mut [f64], w: &DirectionWeights| {
        rnn_cell_step(
            h,
            x,
            hid,
            &w.input_weight,
            &w.recurrent_weight,
            &w.bias,
            f64::tanh,
        );
    };
    let expected = bi_reference(&case, step);

    let config = RecurrentConfig {
        num_layers: 2,
        bidirectional: true,
        bias: true,
    };
    let mut model = Rnn::with_config(input, hidden.of(H), time, config)?;
    model.build(
        &Shape::new([batch.of(B), time.of(T), input.of(I)])?,
        &device,
        11,
    )?;
    install_bi_direction(&model, &case.stack.layer0_forward, "layer0_forward")?;
    install_bi_direction(&model, &case.stack.layer0_backward, "layer0_backward")?;
    install_bi_direction(&model, &case.stack.layer1_forward, "layer1_forward")?;
    install_bi_direction(&model, &case.stack.layer1_backward, "layer1_backward")?;

    let x = Tensor::from_slice(
        &f32s(&case.input),
        [batch.of(B), time.of(T), input.of(I)],
        &device,
    )?
    .with_grad();
    let initial: Vec<Tensor> = case
        .initial
        .iter()
        .map(|values| {
            Tensor::from_slice(&f32s(values), [batch.of(B), hidden.of(H)], &device)
                .map(|tensor| tensor.with_grad())
        })
        .collect::<Result<_>>()?;
    let run = model.run_from(&x, &initial)?;
    assert_eq!(
        run.sequence.shape(),
        &Shape::new([batch.of(B), time.of(T), hidden.of(2 * H)])?
    );
    close(
        "2-layer bidirectional RNN sequence",
        &run.sequence.to_vec()?,
        &expected.sequence,
        4e-5,
    );
    for slot in 0..4 {
        close(
            &format!("2-layer bidirectional RNN final state {slot}"),
            &run.state[slot].to_vec()?,
            &expected.finals_hidden[slot],
            4e-5,
        );
    }

    let sequence_coefficients = f32s(&synth(B * T * 2 * H, 900));
    let mut loss = weighted_sum(
        &run.sequence,
        &sequence_coefficients,
        &[batch, time, hidden],
    )?;
    for slot in 0..4 {
        let hidden_coefficients = f32s(&synth(B * H, 910 + slot as u64));
        loss = loss.add(&weighted_sum(
            &run.state[slot],
            &hidden_coefficients,
            &[batch, hidden],
        )?)?;
    }
    loss.backward()?;

    let input_gradient = x.grad().unwrap().to_vec()?;
    close(
        "2-layer bidirectional RNN input gradient",
        &input_gradient,
        &bi_finite_difference(&case, step, |case| &mut case.input),
        3e-3,
    );
    for slot in 0..4 {
        close(
            &format!("2-layer bidirectional RNN initial state gradient {slot}"),
            &initial[slot].grad().unwrap().to_vec()?,
            &bi_finite_difference(&case, step, |case| &mut case.initial[slot]),
            3e-3,
        );
    }
    let weight_fields: [(&str, fn(&mut BiCase) -> &mut Vec<f64>); 12] = [
        ("layer0_forward_input_weight", |c| {
            &mut c.stack.layer0_forward.input_weight
        }),
        ("layer0_forward_recurrent_weight", |c| {
            &mut c.stack.layer0_forward.recurrent_weight
        }),
        ("layer0_forward_bias", |c| &mut c.stack.layer0_forward.bias),
        ("layer0_backward_input_weight", |c| {
            &mut c.stack.layer0_backward.input_weight
        }),
        ("layer0_backward_recurrent_weight", |c| {
            &mut c.stack.layer0_backward.recurrent_weight
        }),
        ("layer0_backward_bias", |c| {
            &mut c.stack.layer0_backward.bias
        }),
        ("layer1_forward_input_weight", |c| {
            &mut c.stack.layer1_forward.input_weight
        }),
        ("layer1_forward_recurrent_weight", |c| {
            &mut c.stack.layer1_forward.recurrent_weight
        }),
        ("layer1_forward_bias", |c| &mut c.stack.layer1_forward.bias),
        ("layer1_backward_input_weight", |c| {
            &mut c.stack.layer1_backward.input_weight
        }),
        ("layer1_backward_recurrent_weight", |c| {
            &mut c.stack.layer1_backward.recurrent_weight
        }),
        ("layer1_backward_bias", |c| {
            &mut c.stack.layer1_backward.bias
        }),
    ];
    for (name, field) in weight_fields {
        let actual = model.parameter(name)?.grad().unwrap().to_vec()?;
        close(
            &format!("2-layer bidirectional RNN {name} gradient"),
            &actual,
            &bi_finite_difference(&case, step, field),
            3e-3,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn gru_two_layer_bidirectional_matches_independent_f64_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let case = bi_case(3, 300);
    let step = |h: usize, x: &[f64], hid: &mut [f64], _cell: &mut [f64], w: &DirectionWeights| {
        gru_cell_step(h, x, hid, &w.input_weight, &w.recurrent_weight, &w.bias);
    };
    let expected = bi_reference(&case, step);

    let config = RecurrentConfig {
        num_layers: 2,
        bidirectional: true,
        bias: true,
    };
    let mut model = Gru::with_config(input, hidden.of(H), time, config)?;
    model.build(
        &Shape::new([batch.of(B), time.of(T), input.of(I)])?,
        &device,
        12,
    )?;
    install_bi_direction(&model, &case.stack.layer0_forward, "layer0_forward")?;
    install_bi_direction(&model, &case.stack.layer0_backward, "layer0_backward")?;
    install_bi_direction(&model, &case.stack.layer1_forward, "layer1_forward")?;
    install_bi_direction(&model, &case.stack.layer1_backward, "layer1_backward")?;

    let x = Tensor::from_slice(
        &f32s(&case.input),
        [batch.of(B), time.of(T), input.of(I)],
        &device,
    )?
    .with_grad();
    let initial: Vec<Tensor> = case
        .initial
        .iter()
        .map(|values| {
            Tensor::from_slice(&f32s(values), [batch.of(B), hidden.of(H)], &device)
                .map(|tensor| tensor.with_grad())
        })
        .collect::<Result<_>>()?;
    let run = model.run_from(&x, &initial)?;
    assert_eq!(
        run.sequence.shape(),
        &Shape::new([batch.of(B), time.of(T), hidden.of(2 * H)])?
    );
    close(
        "2-layer bidirectional GRU sequence",
        &run.sequence.to_vec()?,
        &expected.sequence,
        4e-5,
    );
    for slot in 0..4 {
        close(
            &format!("2-layer bidirectional GRU final state {slot}"),
            &run.state[slot].to_vec()?,
            &expected.finals_hidden[slot],
            4e-5,
        );
    }

    let sequence_coefficients = f32s(&synth(B * T * 2 * H, 900));
    let mut loss = weighted_sum(
        &run.sequence,
        &sequence_coefficients,
        &[batch, time, hidden],
    )?;
    for slot in 0..4 {
        let hidden_coefficients = f32s(&synth(B * H, 910 + slot as u64));
        loss = loss.add(&weighted_sum(
            &run.state[slot],
            &hidden_coefficients,
            &[batch, hidden],
        )?)?;
    }
    loss.backward()?;

    let input_gradient = x.grad().unwrap().to_vec()?;
    close(
        "2-layer bidirectional GRU input gradient",
        &input_gradient,
        &bi_finite_difference(&case, step, |case| &mut case.input),
        3e-3,
    );
    for slot in 0..4 {
        close(
            &format!("2-layer bidirectional GRU initial state gradient {slot}"),
            &initial[slot].grad().unwrap().to_vec()?,
            &bi_finite_difference(&case, step, |case| &mut case.initial[slot]),
            3e-3,
        );
    }
    let weight_fields: [(&str, fn(&mut BiCase) -> &mut Vec<f64>); 12] = [
        ("layer0_forward_input_weight", |c| {
            &mut c.stack.layer0_forward.input_weight
        }),
        ("layer0_forward_recurrent_weight", |c| {
            &mut c.stack.layer0_forward.recurrent_weight
        }),
        ("layer0_forward_bias", |c| &mut c.stack.layer0_forward.bias),
        ("layer0_backward_input_weight", |c| {
            &mut c.stack.layer0_backward.input_weight
        }),
        ("layer0_backward_recurrent_weight", |c| {
            &mut c.stack.layer0_backward.recurrent_weight
        }),
        ("layer0_backward_bias", |c| {
            &mut c.stack.layer0_backward.bias
        }),
        ("layer1_forward_input_weight", |c| {
            &mut c.stack.layer1_forward.input_weight
        }),
        ("layer1_forward_recurrent_weight", |c| {
            &mut c.stack.layer1_forward.recurrent_weight
        }),
        ("layer1_forward_bias", |c| &mut c.stack.layer1_forward.bias),
        ("layer1_backward_input_weight", |c| {
            &mut c.stack.layer1_backward.input_weight
        }),
        ("layer1_backward_recurrent_weight", |c| {
            &mut c.stack.layer1_backward.recurrent_weight
        }),
        ("layer1_backward_bias", |c| {
            &mut c.stack.layer1_backward.bias
        }),
    ];
    for (name, field) in weight_fields {
        let actual = model.parameter(name)?.grad().unwrap().to_vec()?;
        close(
            &format!("2-layer bidirectional GRU {name} gradient"),
            &actual,
            &bi_finite_difference(&case, step, field),
            3e-3,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn lstm_two_layer_bidirectional_matches_independent_f64_oracle() -> Result<()> {
    let device = Device::cuda(0)?;
    let (batch, time, input, hidden) = (
        Axis::new("batch"),
        Axis::new("time"),
        Axis::new("input"),
        Axis::new("hidden"),
    );
    let case = bi_case(4, 400);
    let step = |h: usize, x: &[f64], hid: &mut [f64], cell: &mut [f64], w: &DirectionWeights| {
        lstm_cell_step(
            h,
            x,
            hid,
            cell,
            &w.input_weight,
            &w.recurrent_weight,
            &w.bias,
        );
    };
    let expected = bi_reference(&case, step);

    let config = RecurrentConfig {
        num_layers: 2,
        bidirectional: true,
        bias: true,
    };
    let mut model = Lstm::with_config(input, hidden.of(H), time, config)?;
    model.build(
        &Shape::new([batch.of(B), time.of(T), input.of(I)])?,
        &device,
        13,
    )?;
    install_bi_direction(&model, &case.stack.layer0_forward, "layer0_forward")?;
    install_bi_direction(&model, &case.stack.layer0_backward, "layer0_backward")?;
    install_bi_direction(&model, &case.stack.layer1_forward, "layer1_forward")?;
    install_bi_direction(&model, &case.stack.layer1_backward, "layer1_backward")?;

    let x = Tensor::from_slice(
        &f32s(&case.input),
        [batch.of(B), time.of(T), input.of(I)],
        &device,
    )?
    .with_grad();
    let initial: Vec<LstmState> = (0..4)
        .map(|slot| {
            LstmState::new(
                Tensor::from_slice(
                    &f32s(&case.initial[slot]),
                    [batch.of(B), hidden.of(H)],
                    &device,
                )?
                .with_grad(),
                Tensor::from_slice(
                    &f32s(&case.initial_cell[slot]),
                    [batch.of(B), hidden.of(H)],
                    &device,
                )?
                .with_grad(),
            )
        })
        .collect::<Result<_>>()?;
    let run = model.run_from(&x, &initial)?;
    assert_eq!(
        run.sequence.shape(),
        &Shape::new([batch.of(B), time.of(T), hidden.of(2 * H)])?
    );
    close(
        "2-layer bidirectional LSTM sequence",
        &run.sequence.to_vec()?,
        &expected.sequence,
        4e-5,
    );
    for slot in 0..4 {
        close(
            &format!("2-layer bidirectional LSTM final hidden {slot}"),
            &run.state[slot].hidden.to_vec()?,
            &expected.finals_hidden[slot],
            4e-5,
        );
        close(
            &format!("2-layer bidirectional LSTM final cell {slot}"),
            &run.state[slot].cell.to_vec()?,
            &expected.finals_cell[slot],
            4e-5,
        );
    }

    let sequence_coefficients = f32s(&synth(B * T * 2 * H, 900));
    let mut loss = weighted_sum(
        &run.sequence,
        &sequence_coefficients,
        &[batch, time, hidden],
    )?;
    for slot in 0..4 {
        let hidden_coefficients = f32s(&synth(B * H, 910 + slot as u64));
        let cell_coefficients = f32s(&synth(B * H, 920 + slot as u64));
        loss = loss
            .add(&weighted_sum(
                &run.state[slot].hidden,
                &hidden_coefficients,
                &[batch, hidden],
            )?)?
            .add(&weighted_sum(
                &run.state[slot].cell,
                &cell_coefficients,
                &[batch, hidden],
            )?)?;
    }
    loss.backward()?;

    let input_gradient = x.grad().unwrap().to_vec()?;
    close(
        "2-layer bidirectional LSTM input gradient",
        &input_gradient,
        &bi_finite_difference(&case, step, |case| &mut case.input),
        3e-3,
    );
    for slot in 0..4 {
        close(
            &format!("2-layer bidirectional LSTM initial hidden gradient {slot}"),
            &initial[slot].hidden.grad().unwrap().to_vec()?,
            &bi_finite_difference(&case, step, |case| &mut case.initial[slot]),
            3e-3,
        );
        close(
            &format!("2-layer bidirectional LSTM initial cell gradient {slot}"),
            &initial[slot].cell.grad().unwrap().to_vec()?,
            &bi_finite_difference(&case, step, |case| &mut case.initial_cell[slot]),
            3e-3,
        );
    }
    let weight_fields: [(&str, fn(&mut BiCase) -> &mut Vec<f64>); 12] = [
        ("layer0_forward_input_weight", |c| {
            &mut c.stack.layer0_forward.input_weight
        }),
        ("layer0_forward_recurrent_weight", |c| {
            &mut c.stack.layer0_forward.recurrent_weight
        }),
        ("layer0_forward_bias", |c| &mut c.stack.layer0_forward.bias),
        ("layer0_backward_input_weight", |c| {
            &mut c.stack.layer0_backward.input_weight
        }),
        ("layer0_backward_recurrent_weight", |c| {
            &mut c.stack.layer0_backward.recurrent_weight
        }),
        ("layer0_backward_bias", |c| {
            &mut c.stack.layer0_backward.bias
        }),
        ("layer1_forward_input_weight", |c| {
            &mut c.stack.layer1_forward.input_weight
        }),
        ("layer1_forward_recurrent_weight", |c| {
            &mut c.stack.layer1_forward.recurrent_weight
        }),
        ("layer1_forward_bias", |c| &mut c.stack.layer1_forward.bias),
        ("layer1_backward_input_weight", |c| {
            &mut c.stack.layer1_backward.input_weight
        }),
        ("layer1_backward_recurrent_weight", |c| {
            &mut c.stack.layer1_backward.recurrent_weight
        }),
        ("layer1_backward_bias", |c| {
            &mut c.stack.layer1_backward.bias
        }),
    ];
    for (name, field) in weight_fields {
        let actual = model.parameter(name)?.grad().unwrap().to_vec()?;
        close(
            &format!("2-layer bidirectional LSTM {name} gradient"),
            &actual,
            &bi_finite_difference(&case, step, field),
            3e-3,
        );
    }
    Ok(())
}
