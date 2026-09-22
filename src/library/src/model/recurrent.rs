use crate::{Axis, Device, Dim, Result, Shape, Tensor, nn::Module, nn::Parameter};

fn same_named_shape(actual: &Shape, expected: &Shape) -> bool {
    actual.rank() == expected.rank()
        && expected.dims().iter().all(|dim| {
            actual
                .extent(dim.axis)
                .is_ok_and(|extent| extent == dim.extent)
        })
}

fn without(shape: &Shape, axis: Axis) -> Result<Shape> {
    shape.extent(axis)?;
    Shape::new(shape.dims().iter().copied().filter(|dim| dim.axis != axis))
}

/// Explicit recurrent state. Both tensors contain the same named stream axes
/// and the LSTM's hidden axis.
#[derive(Clone)]
pub struct LstmState {
    pub hidden: Tensor,
    pub cell: Tensor,
}

impl LstmState {
    pub fn new(hidden: Tensor, cell: Tensor) -> Result<Self> {
        if !hidden.device().same(cell.device()) {
            return Err("LSTM hidden and cell states must use the same Device handle".into());
        }
        if !same_named_shape(hidden.shape(), cell.shape()) {
            return Err("LSTM hidden and cell states must have identical named shapes".into());
        }
        Ok(Self { hidden, cell })
    }

    /// Truncate recurrent differentiation while preserving current values.
    pub fn detach(&self) -> Self {
        Self {
            hidden: self.hidden.detach(),
            cell: self.cell.detach(),
        }
    }
}

/// Sequence output and the two terminal recurrent states from one LSTM run.
pub struct LstmRun {
    pub sequence: Tensor,
    pub state: LstmState,
}

struct BoundLstmCell {
    input_extent: usize,
    input_weight: Parameter,
    recurrent_weight: Parameter,
    bias: Parameter,
}

/// One standard IFGO LSTM transition with explicit hidden and cell state.
pub struct LstmCell {
    input: Axis,
    hidden: Dim,
    input_role: Axis,
    hidden_input_role: Axis,
    gate_feature: Axis,
    gate: Axis,
    gate_hidden: Axis,
    bound: Option<BoundLstmCell>,
}

impl LstmCell {
    pub fn new(input: Axis, hidden: Dim) -> Result<Self> {
        if hidden.extent == 0 {
            return Err("LSTM hidden extent must be positive".into());
        }
        Ok(Self {
            input,
            hidden,
            input_role: input.role("lstm_input"),
            hidden_input_role: hidden.axis.role("lstm_hidden_input"),
            gate_feature: hidden.axis.role("lstm_gate_feature"),
            gate: hidden.axis.role("lstm_gate"),
            gate_hidden: hidden.axis.role("lstm_gate_hidden"),
            bound: None,
        })
    }

    fn output_shape_for(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.input)?;
        if let Some(bound) = &self.bound
            && bound.input_extent != extent
        {
            return Err("LSTM input extent differs from its built extent".into());
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.input)
            .collect();
        dims.push(self.hidden);
        Shape::new(dims)
    }

    fn validate_state(&self, input: &Tensor, state: &LstmState) -> Result<Shape> {
        let expected = self.output_shape_for(input.shape())?;
        if !input.device().same(state.hidden.device()) || !input.device().same(state.cell.device())
        {
            return Err("LSTM input and state must use the same Device handle".into());
        }
        if !same_named_shape(state.hidden.shape(), &expected)
            || !same_named_shape(state.cell.shape(), &expected)
        {
            return Err(format!("LSTM state named shape mismatch: expected {:?}", expected).into());
        }
        Ok(expected)
    }

    pub fn zero_state(&self, input: &Tensor) -> Result<LstmState> {
        let shape = self.output_shape_for(input.shape())?;
        LstmState::new(
            Tensor::zeros(shape.dims().iter().copied(), input.device())?,
            Tensor::zeros(shape.dims().iter().copied(), input.device())?,
        )
    }

    fn project_input(&self, input: &Tensor) -> Result<Tensor> {
        let bound = self
            .bound
            .as_ref()
            .ok_or("LstmCell must be built before projecting input")?;
        if input.extent(self.input)? != bound.input_extent {
            return Err("LSTM input extent differs from its built extent".into());
        }
        input
            .rename(self.input, self.input_role)?
            .contract(&bound.input_weight.tensor(), self.input_role)?
            .add(&bound.bias.tensor())
    }

    fn step_projected(&self, projected: &Tensor, state: &LstmState) -> Result<LstmState> {
        let bound = self
            .bound
            .as_ref()
            .ok_or("LstmCell must be built before step")?;
        if !projected.device().same(state.hidden.device())
            || !projected.device().same(state.cell.device())
        {
            return Err("LSTM projected input and state must use the same Device handle".into());
        }
        let gate_extent = self
            .hidden
            .extent
            .checked_mul(4)
            .ok_or("LSTM gate extent overflow")?;
        let expected = Shape::new(
            state
                .hidden
                .shape()
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != self.hidden.axis)
                .chain([self.gate_feature.of(gate_extent)]),
        )?;
        if !same_named_shape(projected.shape(), &expected) {
            return Err(format!(
                "LSTM projected input named shape mismatch: expected {:?}",
                expected
            )
            .into());
        }
        let gates = projected
            .add(
                &state
                    .hidden
                    .rename(self.hidden.axis, self.hidden_input_role)?
                    .contract(&bound.recurrent_weight.tensor(), self.hidden_input_role)?,
            )?
            .split(
                self.gate_feature,
                [self.gate.of(4), self.gate_hidden.of(self.hidden.extent)],
            )?;
        let gate = |index, sigmoid: bool| -> Result<Tensor> {
            let value = gates
                .select(self.gate, index)?
                .rename(self.gate_hidden, self.hidden.axis)?;
            if sigmoid {
                value.sigmoid()
            } else {
                value.tanh()
            }
        };
        let input_gate = gate(0, true)?;
        let forget_gate = gate(1, true)?;
        let candidate = gate(2, false)?;
        let output_gate = gate(3, true)?;
        let cell = forget_gate
            .mul(&state.cell)?
            .add(&input_gate.mul(&candidate)?)?;
        let hidden = output_gate.mul(&cell.tanh()?)?;
        LstmState::new(hidden, cell)
    }

    pub fn step(&self, input: &Tensor, state: &LstmState) -> Result<LstmState> {
        self.validate_state(input, state)?;
        self.step_projected(&self.project_input(input)?, state)
    }
}

impl Module for LstmCell {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape_for(input)?;
        if let Some(bound) = &self.bound {
            if !bound.input_weight.tensor().device().same(device) {
                return Err("LstmCell is already built on a different Device".into());
            }
            return Ok(output);
        }
        let input_extent = input.extent(self.input)?;
        let gate_extent = self
            .hidden
            .extent
            .checked_mul(4)
            .ok_or("LSTM gate extent overflow")?;
        let mut rng = seed.max(1);
        let mut initialized = |len: usize, scale: f32| {
            (0..len)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (((rng >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * scale
                })
                .collect::<Vec<_>>()
        };
        let input_shape = Shape::new([
            self.input_role.of(input_extent),
            self.gate_feature.of(gate_extent),
        ])?;
        let recurrent_shape = Shape::new([
            self.hidden_input_role.of(self.hidden.extent),
            self.gate_feature.of(gate_extent),
        ])?;
        let input_scale = (6.0 / (input_extent + gate_extent) as f32).sqrt();
        let recurrent_scale = (6.0 / (self.hidden.extent + gate_extent) as f32).sqrt();
        let input_weight = Parameter::new(Tensor::from_slice(
            &initialized(input_shape.len(), input_scale),
            input_shape.dims().iter().copied(),
            device,
        )?);
        let recurrent_weight = Parameter::new(Tensor::from_slice(
            &initialized(recurrent_shape.len(), recurrent_scale),
            recurrent_shape.dims().iter().copied(),
            device,
        )?);
        let bias = Parameter::new(Tensor::zeros([self.gate_feature.of(gate_extent)], device)?);
        self.bound = Some(BoundLstmCell {
            input_extent,
            input_weight,
            recurrent_weight,
            bias,
        });
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let state = self.zero_state(input)?;
        Ok(self.step(input, &state)?.hidden)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|bound| {
                vec![
                    ("input_weight".into(), bound.input_weight.clone()),
                    ("recurrent_weight".into(), bound.recurrent_weight.clone()),
                    ("bias".into(), bound.bias.clone()),
                ]
            })
            .unwrap_or_default()
    }
}

/// A single-layer, forward, eager LSTM over one named time axis.
///
/// The input projection is performed once with the time axis intact. The
/// recurrence then submits O(T) eager recurrent transitions and retains their
/// activations for reverse mode. This is a bounded correctness path, not a
/// fused scan or a sequence-throughput claim.
pub struct Lstm {
    time: Axis,
    cell: LstmCell,
}

impl Lstm {
    pub fn new(input: Axis, hidden: Dim, time: Axis) -> Result<Self> {
        if time == input || time == hidden.axis {
            return Err("LSTM time must be distinct from its input and hidden axes".into());
        }
        Ok(Self {
            time,
            cell: LstmCell::new(input, hidden)?,
        })
    }

    fn state_shape(&self, input: &Shape) -> Result<Shape> {
        self.cell.output_shape_for(&without(input, self.time)?)
    }

    fn validate_state(&self, input: &Tensor, state: &LstmState) -> Result<()> {
        let expected = self.state_shape(input.shape())?;
        if !input.device().same(state.hidden.device()) || !input.device().same(state.cell.device())
        {
            return Err("LSTM input and state must use the same Device handle".into());
        }
        if !same_named_shape(state.hidden.shape(), &expected)
            || !same_named_shape(state.cell.shape(), &expected)
        {
            return Err(format!("LSTM state named shape mismatch: expected {:?}", expected).into());
        }
        Ok(())
    }

    pub fn run(&self, input: &Tensor) -> Result<LstmRun> {
        let shape = self.state_shape(input.shape())?;
        let state = LstmState::new(
            Tensor::zeros(shape.dims().iter().copied(), input.device())?,
            Tensor::zeros(shape.dims().iter().copied(), input.device())?,
        )?;
        self.run_from(input, &state)
    }

    pub fn run_from(&self, input: &Tensor, initial: &LstmState) -> Result<LstmRun> {
        let output = self.output_shape(input.shape())?;
        self.validate_state(input, initial)?;
        let steps = input.extent(self.time)?;
        let projected = self.cell.project_input(input)?;
        let mut state = initial.clone();
        let mut hidden = Vec::with_capacity(steps);
        for step in 0..steps {
            state = self
                .cell
                .step_projected(&projected.select(self.time, step)?, &state)?;
            hidden.push(state.hidden.clone());
        }
        let time_position = output.index(self.time)?;
        let sequence = Tensor::stack(&hidden, self.time, time_position)?;
        Ok(LstmRun { sequence, state })
    }
}

impl Module for Lstm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.time)?;
        self.cell.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        self.cell.build(&without(input, self.time)?, device, seed)?;
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.run(input)?.sequence)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.cell.named_parameters()
    }
}

/// The pointwise nonlinearity applied at each [`Rnn`]/[`RnnCell`] transition.
/// PyTorch's default is `Tanh`; `Relu` matches `nonlinearity="relu"`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RnnNonlinearity {
    Tanh,
    Relu,
}

struct BoundRnnCell {
    input_extent: usize,
    input_weight: Parameter,
    recurrent_weight: Parameter,
    bias: Parameter,
}

/// One Elman RNN transition, `h' = nonlinearity(W_ih x + b + W_hh h)`, with
/// explicit hidden state. `nonlinearity` defaults to `Tanh`
/// (PyTorch's `RNNCell(nonlinearity="tanh")` default); `.nonlinearity(Relu)`
/// matches `nonlinearity="relu"`. Mirrors [`LstmCell`]'s single combined bias
/// (rather than PyTorch's separate `bias_ih`/`bias_hh`) and its Xavier-uniform
/// initialization (rather than PyTorch's `uniform(-1/sqrt(hidden),
/// 1/sqrt(hidden))`); Axis's own `Lstm` does not match PyTorch's parameter
/// layout or init either, so `RnnCell` follows `LstmCell`'s convention
/// instead of introducing a second one.
pub struct RnnCell {
    input: Axis,
    hidden: Dim,
    nonlinearity: RnnNonlinearity,
    input_role: Axis,
    hidden_input_role: Axis,
    hidden_output_role: Axis,
    bound: Option<BoundRnnCell>,
}

impl RnnCell {
    pub fn new(input: Axis, hidden: Dim) -> Result<Self> {
        if hidden.extent == 0 {
            return Err("RNN hidden extent must be positive".into());
        }
        Ok(Self {
            input,
            hidden,
            nonlinearity: RnnNonlinearity::Tanh,
            input_role: input.role("rnn_input"),
            hidden_input_role: hidden.axis.role("rnn_hidden_input"),
            hidden_output_role: hidden.axis.role("rnn_hidden_output"),
            bound: None,
        })
    }

    /// Choose the pointwise nonlinearity. Has no effect once `build` has
    /// already allocated parameters; call it before `build`.
    pub fn nonlinearity(mut self, nonlinearity: RnnNonlinearity) -> Self {
        self.nonlinearity = nonlinearity;
        self
    }

    fn output_shape_for(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.input)?;
        if let Some(bound) = &self.bound
            && bound.input_extent != extent
        {
            return Err("RNN input extent differs from its built extent".into());
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.input)
            .collect();
        dims.push(self.hidden);
        Shape::new(dims)
    }

    fn validate_state(&self, input: &Tensor, state: &Tensor) -> Result<Shape> {
        let expected = self.output_shape_for(input.shape())?;
        if !input.device().same(state.device()) {
            return Err("RNN input and state must use the same Device handle".into());
        }
        if !same_named_shape(state.shape(), &expected) {
            return Err(format!("RNN state named shape mismatch: expected {:?}", expected).into());
        }
        Ok(expected)
    }

    pub fn zero_state(&self, input: &Tensor) -> Result<Tensor> {
        let shape = self.output_shape_for(input.shape())?;
        Tensor::zeros(shape.dims().iter().copied(), input.device())
    }

    fn project_input(&self, input: &Tensor) -> Result<Tensor> {
        let bound = self
            .bound
            .as_ref()
            .ok_or("RnnCell must be built before projecting input")?;
        if input.extent(self.input)? != bound.input_extent {
            return Err("RNN input extent differs from its built extent".into());
        }
        input
            .rename(self.input, self.input_role)?
            .contract(&bound.input_weight.tensor(), self.input_role)?
            .add(&bound.bias.tensor())
    }

    fn step_projected(&self, projected: &Tensor, state: &Tensor) -> Result<Tensor> {
        let bound = self
            .bound
            .as_ref()
            .ok_or("RnnCell must be built before step")?;
        if !projected.device().same(state.device()) {
            return Err("RNN projected input and state must use the same Device handle".into());
        }
        let expected = Shape::new(
            state
                .shape()
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != self.hidden.axis)
                .chain([self.hidden_output_role.of(self.hidden.extent)]),
        )?;
        if !same_named_shape(projected.shape(), &expected) {
            return Err(format!(
                "RNN projected input named shape mismatch: expected {:?}",
                expected
            )
            .into());
        }
        let total = projected.add(
            &state
                .rename(self.hidden.axis, self.hidden_input_role)?
                .contract(&bound.recurrent_weight.tensor(), self.hidden_input_role)?,
        )?;
        let activated = match self.nonlinearity {
            RnnNonlinearity::Tanh => total.tanh()?,
            RnnNonlinearity::Relu => total.relu()?,
        };
        activated.rename(self.hidden_output_role, self.hidden.axis)
    }

    pub fn step(&self, input: &Tensor, state: &Tensor) -> Result<Tensor> {
        self.validate_state(input, state)?;
        self.step_projected(&self.project_input(input)?, state)
    }
}

impl Module for RnnCell {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape_for(input)?;
        if let Some(bound) = &self.bound {
            if !bound.input_weight.tensor().device().same(device) {
                return Err("RnnCell is already built on a different Device".into());
            }
            return Ok(output);
        }
        let input_extent = input.extent(self.input)?;
        let mut rng = seed.max(1);
        let mut initialized = |len: usize, scale: f32| {
            (0..len)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (((rng >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * scale
                })
                .collect::<Vec<_>>()
        };
        let input_shape = Shape::new([
            self.input_role.of(input_extent),
            self.hidden_output_role.of(self.hidden.extent),
        ])?;
        let recurrent_shape = Shape::new([
            self.hidden_input_role.of(self.hidden.extent),
            self.hidden_output_role.of(self.hidden.extent),
        ])?;
        let input_scale = (6.0 / (input_extent + self.hidden.extent) as f32).sqrt();
        let recurrent_scale = (6.0 / (self.hidden.extent + self.hidden.extent) as f32).sqrt();
        let input_weight = Parameter::new(Tensor::from_slice(
            &initialized(input_shape.len(), input_scale),
            input_shape.dims().iter().copied(),
            device,
        )?);
        let recurrent_weight = Parameter::new(Tensor::from_slice(
            &initialized(recurrent_shape.len(), recurrent_scale),
            recurrent_shape.dims().iter().copied(),
            device,
        )?);
        let bias = Parameter::new(Tensor::zeros(
            [self.hidden_output_role.of(self.hidden.extent)],
            device,
        )?);
        self.bound = Some(BoundRnnCell {
            input_extent,
            input_weight,
            recurrent_weight,
            bias,
        });
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let state = self.zero_state(input)?;
        self.step(input, &state)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|bound| {
                vec![
                    ("input_weight".into(), bound.input_weight.clone()),
                    ("recurrent_weight".into(), bound.recurrent_weight.clone()),
                    ("bias".into(), bound.bias.clone()),
                ]
            })
            .unwrap_or_default()
    }
}

/// Sequence output and the terminal hidden state from one [`Rnn`] run.
pub struct RnnRun {
    pub sequence: Tensor,
    pub state: Tensor,
}

/// A single-layer, forward, eager Elman RNN over one named time axis. See
/// [`Lstm`]'s doc comment: an eager per-step transition retained for reverse
/// mode, not a fused scan or a sequence-throughput claim.
pub struct Rnn {
    time: Axis,
    cell: RnnCell,
}

impl Rnn {
    pub fn new(input: Axis, hidden: Dim, time: Axis) -> Result<Self> {
        if time == input || time == hidden.axis {
            return Err("RNN time must be distinct from its input and hidden axes".into());
        }
        Ok(Self {
            time,
            cell: RnnCell::new(input, hidden)?,
        })
    }

    /// Choose the pointwise nonlinearity before `build`. Default `Tanh`.
    pub fn nonlinearity(mut self, nonlinearity: RnnNonlinearity) -> Self {
        self.cell = self.cell.nonlinearity(nonlinearity);
        self
    }

    fn state_shape(&self, input: &Shape) -> Result<Shape> {
        self.cell.output_shape_for(&without(input, self.time)?)
    }

    fn validate_state(&self, input: &Tensor, state: &Tensor) -> Result<()> {
        let expected = self.state_shape(input.shape())?;
        if !input.device().same(state.device()) {
            return Err("RNN input and state must use the same Device handle".into());
        }
        if !same_named_shape(state.shape(), &expected) {
            return Err(format!("RNN state named shape mismatch: expected {:?}", expected).into());
        }
        Ok(())
    }

    pub fn run(&self, input: &Tensor) -> Result<RnnRun> {
        let shape = self.state_shape(input.shape())?;
        let state = Tensor::zeros(shape.dims().iter().copied(), input.device())?;
        self.run_from(input, &state)
    }

    pub fn run_from(&self, input: &Tensor, initial: &Tensor) -> Result<RnnRun> {
        let output = self.output_shape(input.shape())?;
        self.validate_state(input, initial)?;
        let steps = input.extent(self.time)?;
        let projected = self.cell.project_input(input)?;
        let mut state = initial.clone();
        let mut hidden = Vec::with_capacity(steps);
        for step in 0..steps {
            state = self
                .cell
                .step_projected(&projected.select(self.time, step)?, &state)?;
            hidden.push(state.clone());
        }
        let time_position = output.index(self.time)?;
        let sequence = Tensor::stack(&hidden, self.time, time_position)?;
        Ok(RnnRun { sequence, state })
    }
}

impl Module for Rnn {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.time)?;
        self.cell.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        self.cell.build(&without(input, self.time)?, device, seed)?;
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.run(input)?.sequence)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.cell.named_parameters()
    }
}

struct BoundGruCell {
    input_extent: usize,
    input_weight: Parameter,
    recurrent_weight: Parameter,
    bias: Parameter,
}

/// One PyTorch-form GRU transition with explicit hidden state.
///
/// Gate order is PyTorch's `reset, update, new` (`r, z, n`), and the reset
/// gate multiplies the hidden-side candidate contribution after its own
/// matrix product -- `r * (W_hn h + b_hn)`, PyTorch's form -- rather than the
/// paper's `r * h` before any projection:
/// `r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)`,
/// `z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)`,
/// `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`,
/// `h' = (1 - z) * n + z * h`. Mirrors [`LstmCell`]'s single combined bias
/// (rather than PyTorch's separate `bias_ih`/`bias_hh`) and its Xavier-uniform
/// initialization (rather than PyTorch's `uniform(-1/sqrt(hidden),
/// 1/sqrt(hidden))`); Axis's own `Lstm` does not match PyTorch's parameter
/// layout or init either, so `GruCell` follows `LstmCell`'s convention
/// instead of introducing a second one.
pub struct GruCell {
    input: Axis,
    hidden: Dim,
    input_role: Axis,
    hidden_input_role: Axis,
    gate_feature: Axis,
    gate: Axis,
    gate_hidden: Axis,
    bound: Option<BoundGruCell>,
}

impl GruCell {
    pub fn new(input: Axis, hidden: Dim) -> Result<Self> {
        if hidden.extent == 0 {
            return Err("GRU hidden extent must be positive".into());
        }
        Ok(Self {
            input,
            hidden,
            input_role: input.role("gru_input"),
            hidden_input_role: hidden.axis.role("gru_hidden_input"),
            gate_feature: hidden.axis.role("gru_gate_feature"),
            gate: hidden.axis.role("gru_gate"),
            gate_hidden: hidden.axis.role("gru_gate_hidden"),
            bound: None,
        })
    }

    fn output_shape_for(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.input)?;
        if let Some(bound) = &self.bound
            && bound.input_extent != extent
        {
            return Err("GRU input extent differs from its built extent".into());
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.input)
            .collect();
        dims.push(self.hidden);
        Shape::new(dims)
    }

    fn validate_state(&self, input: &Tensor, state: &Tensor) -> Result<Shape> {
        let expected = self.output_shape_for(input.shape())?;
        if !input.device().same(state.device()) {
            return Err("GRU input and state must use the same Device handle".into());
        }
        if !same_named_shape(state.shape(), &expected) {
            return Err(format!("GRU state named shape mismatch: expected {:?}", expected).into());
        }
        Ok(expected)
    }

    pub fn zero_state(&self, input: &Tensor) -> Result<Tensor> {
        let shape = self.output_shape_for(input.shape())?;
        Tensor::zeros(shape.dims().iter().copied(), input.device())
    }

    fn project_input(&self, input: &Tensor) -> Result<Tensor> {
        let bound = self
            .bound
            .as_ref()
            .ok_or("GruCell must be built before projecting input")?;
        if input.extent(self.input)? != bound.input_extent {
            return Err("GRU input extent differs from its built extent".into());
        }
        input
            .rename(self.input, self.input_role)?
            .contract(&bound.input_weight.tensor(), self.input_role)?
            .add(&bound.bias.tensor())
    }

    fn step_projected(&self, projected: &Tensor, state: &Tensor) -> Result<Tensor> {
        let bound = self
            .bound
            .as_ref()
            .ok_or("GruCell must be built before step")?;
        if !projected.device().same(state.device()) {
            return Err("GRU projected input and state must use the same Device handle".into());
        }
        let gate_extent = self
            .hidden
            .extent
            .checked_mul(3)
            .ok_or("GRU gate extent overflow")?;
        let expected = Shape::new(
            state
                .shape()
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != self.hidden.axis)
                .chain([self.gate_feature.of(gate_extent)]),
        )?;
        if !same_named_shape(projected.shape(), &expected) {
            return Err(format!(
                "GRU projected input named shape mismatch: expected {:?}",
                expected
            )
            .into());
        }
        let input_gates = projected.split(
            self.gate_feature,
            [self.gate.of(3), self.gate_hidden.of(self.hidden.extent)],
        )?;
        let hidden_gates = state
            .rename(self.hidden.axis, self.hidden_input_role)?
            .contract(&bound.recurrent_weight.tensor(), self.hidden_input_role)?
            .split(
                self.gate_feature,
                [self.gate.of(3), self.gate_hidden.of(self.hidden.extent)],
            )?;
        let component = |gates: &Tensor, index: usize| -> Result<Tensor> {
            gates
                .select(self.gate, index)?
                .rename(self.gate_hidden, self.hidden.axis)
        };
        let reset = component(&input_gates, 0)?
            .add(&component(&hidden_gates, 0)?)?
            .sigmoid()?;
        let update = component(&input_gates, 1)?
            .add(&component(&hidden_gates, 1)?)?
            .sigmoid()?;
        let candidate = component(&input_gates, 2)?
            .add(&reset.mul(&component(&hidden_gates, 2)?)?)?
            .tanh()?;
        let one = Tensor::from_slice(&[1.0], [], state.device())?;
        one.sub(&update)?.mul(&candidate)?.add(&update.mul(state)?)
    }

    pub fn step(&self, input: &Tensor, state: &Tensor) -> Result<Tensor> {
        self.validate_state(input, state)?;
        self.step_projected(&self.project_input(input)?, state)
    }
}

impl Module for GruCell {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape_for(input)?;
        if let Some(bound) = &self.bound {
            if !bound.input_weight.tensor().device().same(device) {
                return Err("GruCell is already built on a different Device".into());
            }
            return Ok(output);
        }
        let input_extent = input.extent(self.input)?;
        let gate_extent = self
            .hidden
            .extent
            .checked_mul(3)
            .ok_or("GRU gate extent overflow")?;
        let mut rng = seed.max(1);
        let mut initialized = |len: usize, scale: f32| {
            (0..len)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (((rng >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * scale
                })
                .collect::<Vec<_>>()
        };
        let input_shape = Shape::new([
            self.input_role.of(input_extent),
            self.gate_feature.of(gate_extent),
        ])?;
        let recurrent_shape = Shape::new([
            self.hidden_input_role.of(self.hidden.extent),
            self.gate_feature.of(gate_extent),
        ])?;
        let input_scale = (6.0 / (input_extent + gate_extent) as f32).sqrt();
        let recurrent_scale = (6.0 / (self.hidden.extent + gate_extent) as f32).sqrt();
        let input_weight = Parameter::new(Tensor::from_slice(
            &initialized(input_shape.len(), input_scale),
            input_shape.dims().iter().copied(),
            device,
        )?);
        let recurrent_weight = Parameter::new(Tensor::from_slice(
            &initialized(recurrent_shape.len(), recurrent_scale),
            recurrent_shape.dims().iter().copied(),
            device,
        )?);
        let bias = Parameter::new(Tensor::zeros([self.gate_feature.of(gate_extent)], device)?);
        self.bound = Some(BoundGruCell {
            input_extent,
            input_weight,
            recurrent_weight,
            bias,
        });
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let state = self.zero_state(input)?;
        self.step(input, &state)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|bound| {
                vec![
                    ("input_weight".into(), bound.input_weight.clone()),
                    ("recurrent_weight".into(), bound.recurrent_weight.clone()),
                    ("bias".into(), bound.bias.clone()),
                ]
            })
            .unwrap_or_default()
    }
}

/// Sequence output and the terminal hidden state from one [`Gru`] run.
pub struct GruRun {
    pub sequence: Tensor,
    pub state: Tensor,
}

/// A single-layer, forward, eager GRU over one named time axis. See [`Lstm`]'s
/// doc comment: an eager per-step transition retained for reverse mode, not a
/// fused scan or a sequence-throughput claim.
pub struct Gru {
    time: Axis,
    cell: GruCell,
}

impl Gru {
    pub fn new(input: Axis, hidden: Dim, time: Axis) -> Result<Self> {
        if time == input || time == hidden.axis {
            return Err("GRU time must be distinct from its input and hidden axes".into());
        }
        Ok(Self {
            time,
            cell: GruCell::new(input, hidden)?,
        })
    }

    fn state_shape(&self, input: &Shape) -> Result<Shape> {
        self.cell.output_shape_for(&without(input, self.time)?)
    }

    fn validate_state(&self, input: &Tensor, state: &Tensor) -> Result<()> {
        let expected = self.state_shape(input.shape())?;
        if !input.device().same(state.device()) {
            return Err("GRU input and state must use the same Device handle".into());
        }
        if !same_named_shape(state.shape(), &expected) {
            return Err(format!("GRU state named shape mismatch: expected {:?}", expected).into());
        }
        Ok(())
    }

    pub fn run(&self, input: &Tensor) -> Result<GruRun> {
        let shape = self.state_shape(input.shape())?;
        let state = Tensor::zeros(shape.dims().iter().copied(), input.device())?;
        self.run_from(input, &state)
    }

    pub fn run_from(&self, input: &Tensor, initial: &Tensor) -> Result<GruRun> {
        let output = self.output_shape(input.shape())?;
        self.validate_state(input, initial)?;
        let steps = input.extent(self.time)?;
        let projected = self.cell.project_input(input)?;
        let mut state = initial.clone();
        let mut hidden = Vec::with_capacity(steps);
        for step in 0..steps {
            state = self
                .cell
                .step_projected(&projected.select(self.time, step)?, &state)?;
            hidden.push(state.clone());
        }
        let time_position = output.index(self.time)?;
        let sequence = Tensor::stack(&hidden, self.time, time_position)?;
        Ok(GruRun { sequence, state })
    }
}

impl Module for Gru {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.time)?;
        self.cell.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        self.cell.build(&without(input, self.time)?, device, seed)?;
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.run(input)?.sequence)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.cell.named_parameters()
    }
}
