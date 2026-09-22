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

/// Shared depth/direction/bias configuration for [`Rnn`], [`Gru`] and
/// [`Lstm`]. PyTorch's `RNNBase` plays this role for its three subclasses,
/// but it also owns shared weight storage and flattening behavior that Axis's
/// three families do not share (each keeps its own gate layout and init
/// convention, matching `LstmCell`'s existing precedent of not introducing a
/// second parameter convention). `RecurrentConfig` is the honest analogue:
/// the three fields every family's constructor actually shares, reused
/// verbatim by `Rnn::with_config`, `Gru::with_config` and `Lstm::with_config`,
/// with no shared class behavior invented beyond that.
///
/// `num_layers` stacks that many transitions, each layer's output feeding the
/// next verbatim (PyTorch's stacking convention). `bidirectional` runs a
/// second transition per layer over the reversed time order and concatenates
/// the two directions' hidden output on the hidden axis (PyTorch's
/// `[forward; backward]` order); every layer after the first then reads a
/// hidden extent of `hidden * 2`. `bias` toggles whether each transition
/// allocates its (combined) bias parameter; `bias = false` matches PyTorch's
/// `bias=False`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrentConfig {
    pub num_layers: usize,
    pub bidirectional: bool,
    pub bias: bool,
}

impl Default for RecurrentConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            bidirectional: false,
            bias: true,
        }
    }
}

impl RecurrentConfig {
    fn directions(self) -> usize {
        if self.bidirectional { 2 } else { 1 }
    }
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

/// Sequence output and the per-layer, per-direction terminal recurrent
/// states from one [`Lstm`] run. `state[layer * directions + direction]`
/// (`direction` 0 = forward, 1 = backward when bidirectional), matching
/// PyTorch's `(num_layers * num_directions, batch, hidden)` stacking order.
pub struct LstmRun {
    pub sequence: Tensor,
    pub state: Vec<LstmState>,
}

struct BoundLstmCell {
    input_extent: usize,
    input_weight: Parameter,
    recurrent_weight: Parameter,
    bias: Option<Parameter>,
}

/// One standard IFGO LSTM transition with explicit hidden and cell state.
///
/// `h', c' = LSTMCell(x, (h, c))` per PyTorch 2.14's documented form:
/// `i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)`,
/// `f = sigmoid(W_if x + b_if + W_hf h + b_hf)`,
/// `g = tanh(W_ig x + b_ig + W_hg h + b_hg)`,
/// `o = sigmoid(W_io x + b_io + W_ho h + b_ho)`,
/// `c' = f * c + i * g`, `h' = o * tanh(c')`. `bias` defaults to `true`
/// (PyTorch's default); `.bias(false)` matches `bias=False`.
pub struct LstmCell {
    input: Axis,
    hidden: Dim,
    bias: bool,
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
            bias: true,
            input_role: input.role("lstm_input"),
            hidden_input_role: hidden.axis.role("lstm_hidden_input"),
            gate_feature: hidden.axis.role("lstm_gate_feature"),
            gate: hidden.axis.role("lstm_gate"),
            gate_hidden: hidden.axis.role("lstm_gate_hidden"),
            bound: None,
        })
    }

    /// Toggle the combined bias parameter. Has no effect once `build` has
    /// already allocated parameters; call it before `build`. Defaults to
    /// `true`.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
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
        let projected = input
            .rename(self.input, self.input_role)?
            .contract(&bound.input_weight.tensor(), self.input_role)?;
        match &bound.bias {
            Some(bias) => projected.add(&bias.tensor()),
            None => Ok(projected),
        }
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
        let bias = self
            .bias
            .then(|| -> Result<Parameter> {
                Ok(Parameter::new(Tensor::zeros(
                    [self.gate_feature.of(gate_extent)],
                    device,
                )?))
            })
            .transpose()?;
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
                let mut params = vec![
                    ("input_weight".into(), bound.input_weight.clone()),
                    ("recurrent_weight".into(), bound.recurrent_weight.clone()),
                ];
                if let Some(bias) = &bound.bias {
                    params.push(("bias".into(), bias.clone()));
                }
                params
            })
            .unwrap_or_default()
    }
}

struct LstmLayer {
    forward: LstmCell,
    backward: Option<LstmCell>,
}

/// A stacked, optionally bidirectional, eager LSTM over one named time axis.
///
/// The input projection of each direction of each layer is performed once
/// with the time axis intact. The recurrence then submits O(T) eager
/// recurrent transitions and retains their activations for reverse mode:
/// this is a bounded correctness path, not a fused scan or a sequence-
/// throughput claim. `num_layers` stacks that many transitions, each layer's
/// (possibly concatenated) hidden output feeding the next verbatim. When
/// `bidirectional`, each layer also runs a reversed-time transition and
/// concatenates `[forward; backward]` on the hidden axis (PyTorch's order),
/// so every layer after the first reads a hidden extent of `hidden * 2`.
/// The default configuration (one layer, unidirectional, bias on) is
/// bit-exact with the single-layer LSTM this type used to be.
pub struct Lstm {
    time: Axis,
    hidden: Dim,
    config: RecurrentConfig,
    layers: Vec<LstmLayer>,
}

impl Lstm {
    pub fn new(input: Axis, hidden: Dim, time: Axis) -> Result<Self> {
        Self::with_config(input, hidden, time, RecurrentConfig::default())
    }

    pub fn with_config(input: Axis, hidden: Dim, time: Axis, config: RecurrentConfig) -> Result<Self> {
        if time == input || time == hidden.axis {
            return Err("LSTM time must be distinct from its input and hidden axes".into());
        }
        if config.num_layers == 0 {
            return Err("LSTM num_layers must be positive".into());
        }
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let layer_input = if index == 0 { input } else { hidden.axis };
            let forward = LstmCell::new(layer_input, hidden)?.bias(config.bias);
            let backward = config
                .bidirectional
                .then(|| LstmCell::new(layer_input, hidden).map(|cell| cell.bias(config.bias)))
                .transpose()?;
            layers.push(LstmLayer { forward, backward });
        }
        Ok(Self {
            time,
            hidden,
            config,
            layers,
        })
    }

    pub fn config(&self) -> RecurrentConfig {
        self.config
    }

    fn directions(&self) -> usize {
        self.config.directions()
    }

    fn output_shape_for(&self, input: &Shape) -> Result<Shape> {
        let per_direction = self.layers[0].forward.output_shape_for(input)?;
        if !self.config.bidirectional {
            return Ok(per_direction);
        }
        let mut dims: Vec<_> = per_direction
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.hidden.axis)
            .collect();
        dims.push(self.hidden.axis.of(self.hidden.extent * 2));
        Shape::new(dims)
    }

    /// Zero initial state for every layer and direction, ordered
    /// `layer * directions + direction`.
    pub fn zero_states(&self, input: &Tensor) -> Result<Vec<LstmState>> {
        let directions = self.directions();
        let mut states = Vec::with_capacity(self.layers.len() * directions);
        let mut layer_input_shape = input.shape().clone();
        for layer in &self.layers {
            let per_direction = layer
                .forward
                .output_shape_for(&without(&layer_input_shape, self.time)?)?;
            let zero = || -> Result<LstmState> {
                LstmState::new(
                    Tensor::zeros(per_direction.dims().iter().copied(), input.device())?,
                    Tensor::zeros(per_direction.dims().iter().copied(), input.device())?,
                )
            };
            states.push(zero()?);
            if self.config.bidirectional {
                states.push(zero()?);
            }
            let mut dims: Vec<_> = layer_input_shape
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != layer.forward.input)
                .collect();
            dims.push(self.hidden.axis.of(self.hidden.extent * directions));
            layer_input_shape = Shape::new(dims)?;
        }
        Ok(states)
    }

    pub fn run(&self, input: &Tensor) -> Result<LstmRun> {
        let initial = self.zero_states(input)?;
        self.run_from(input, &initial)
    }

    pub fn run_from(&self, input: &Tensor, initial: &[LstmState]) -> Result<LstmRun> {
        let directions = self.directions();
        let expected_states = self.layers.len() * directions;
        if initial.len() != expected_states {
            return Err(format!(
                "LSTM expected {expected_states} initial states (num_layers * num_directions), got {}",
                initial.len()
            )
            .into());
        }
        let steps = input.extent(self.time)?;
        let mut layer_input = input.clone();
        let mut final_states = Vec::with_capacity(expected_states);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let stripped = without(layer_input.shape(), self.time)?;
            let full_layer_shape = {
                let mut dims: Vec<_> = layer_input
                    .shape()
                    .dims()
                    .iter()
                    .copied()
                    .filter(|dim| dim.axis != layer.forward.input)
                    .collect();
                dims.push(self.hidden);
                Shape::new(dims)?
            };
            let time_position = full_layer_shape.index(self.time)?;

            let run_direction = |cell: &LstmCell, reverse: bool, state0: &LstmState| -> Result<(Tensor, LstmState)> {
                let expected = cell.output_shape_for(&stripped)?;
                if !layer_input.device().same(state0.hidden.device())
                    || !layer_input.device().same(state0.cell.device())
                {
                    return Err("LSTM input and state must use the same Device handle".into());
                }
                if !same_named_shape(state0.hidden.shape(), &expected)
                    || !same_named_shape(state0.cell.shape(), &expected)
                {
                    return Err(
                        format!("LSTM state named shape mismatch: expected {:?}", expected).into(),
                    );
                }
                let projected = cell.project_input(&layer_input)?;
                let mut state = state0.clone();
                let mut hidden = vec![Tensor::zeros([], layer_input.device())?; steps];
                let order: Vec<usize> = if reverse {
                    (0..steps).rev().collect()
                } else {
                    (0..steps).collect()
                };
                for step in order {
                    state = cell.step_projected(&projected.select(self.time, step)?, &state)?;
                    hidden[step] = state.hidden.clone();
                }
                let sequence = Tensor::stack(&hidden, self.time, time_position)?;
                Ok((sequence, state))
            };

            let (forward_sequence, forward_final) =
                run_direction(&layer.forward, false, &initial[layer_index * directions])?;
            final_states.push(forward_final);

            let layer_output = if let Some(backward) = &layer.backward {
                let (backward_sequence, backward_final) =
                    run_direction(backward, true, &initial[layer_index * directions + 1])?;
                final_states.push(backward_final);
                Tensor::concat(&[forward_sequence, backward_sequence], self.hidden.axis)?
            } else {
                forward_sequence
            };
            layer_input = layer_output;
        }
        Ok(LstmRun {
            sequence: layer_input,
            state: final_states,
        })
    }
}

impl Module for Lstm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.time)?;
        self.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        let directions = self.config.directions();
        let mut layer_input_shape = without(input, self.time)?;
        let mut layer_seed = seed;
        for layer in &mut self.layers {
            layer.forward.build(&layer_input_shape, device, layer_seed)?;
            layer_seed = layer_seed.wrapping_add(1);
            if let Some(backward) = &mut layer.backward {
                backward.build(&layer_input_shape, device, layer_seed)?;
                layer_seed = layer_seed.wrapping_add(1);
            }
            let mut dims: Vec<_> = layer_input_shape
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != layer.forward.input)
                .collect();
            dims.push(self.hidden.axis.of(self.hidden.extent * directions));
            layer_input_shape = Shape::new(dims)?;
        }
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.run(input)?.sequence)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        // The single-layer, unidirectional default keeps its original,
        // unprefixed parameter names ("input_weight", "recurrent_weight",
        // "bias") bit-exact with this type before `num_layers`/
        // `bidirectional` existed. Any other configuration prefixes each
        // name with its layer and direction, since it then owns more than
        // one of each.
        if let [layer] = self.layers.as_slice()
            && layer.backward.is_none()
        {
            return layer.forward.named_parameters();
        }
        let mut params = vec![];
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, parameter) in layer.forward.named_parameters() {
                params.push((format!("layer{index}_forward_{name}"), parameter));
            }
            if let Some(backward) = &layer.backward {
                for (name, parameter) in backward.named_parameters() {
                    params.push((format!("layer{index}_backward_{name}"), parameter));
                }
            }
        }
        params
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
    bias: Option<Parameter>,
}

/// One Elman RNN transition, `h' = nonlinearity(W_ih x + b + W_hh h)`, with
/// explicit hidden state. `nonlinearity` defaults to `Tanh`
/// (PyTorch's `RNNCell(nonlinearity="tanh")` default); `.nonlinearity(Relu)`
/// matches `nonlinearity="relu"`. Mirrors [`LstmCell`]'s single combined bias
/// (rather than PyTorch's separate `bias_ih`/`bias_hh`) and its Xavier-uniform
/// initialization (rather than PyTorch's `uniform(-1/sqrt(hidden),
/// 1/sqrt(hidden))`); Axis's own `Lstm` does not match PyTorch's parameter
/// layout or init either, so `RnnCell` follows `LstmCell`'s convention
/// instead of introducing a second one. `bias` defaults to `true`
/// (PyTorch's default); `.bias(false)` matches `bias=False`.
pub struct RnnCell {
    input: Axis,
    hidden: Dim,
    nonlinearity: RnnNonlinearity,
    bias: bool,
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
            bias: true,
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

    /// Toggle the combined bias parameter. Has no effect once `build` has
    /// already allocated parameters; call it before `build`. Defaults to
    /// `true`.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
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
        let projected = input
            .rename(self.input, self.input_role)?
            .contract(&bound.input_weight.tensor(), self.input_role)?;
        match &bound.bias {
            Some(bias) => projected.add(&bias.tensor()),
            None => Ok(projected),
        }
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
        let bias = self
            .bias
            .then(|| -> Result<Parameter> {
                Ok(Parameter::new(Tensor::zeros(
                    [self.hidden_output_role.of(self.hidden.extent)],
                    device,
                )?))
            })
            .transpose()?;
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
                let mut params = vec![
                    ("input_weight".into(), bound.input_weight.clone()),
                    ("recurrent_weight".into(), bound.recurrent_weight.clone()),
                ];
                if let Some(bias) = &bound.bias {
                    params.push(("bias".into(), bias.clone()));
                }
                params
            })
            .unwrap_or_default()
    }
}

/// Sequence output and the per-layer, per-direction terminal hidden states
/// from one [`Rnn`] run, ordered as [`LstmRun::state`] documents.
pub struct RnnRun {
    pub sequence: Tensor,
    pub state: Vec<Tensor>,
}

struct RnnLayer {
    forward: RnnCell,
    backward: Option<RnnCell>,
}

/// A stacked, optionally bidirectional, eager Elman RNN over one named time
/// axis. See [`Lstm`]'s doc comment for the eager-transition contract and
/// [`RecurrentConfig`] for `num_layers`/`bidirectional`/`bias`. The default
/// configuration (one layer, unidirectional, bias on) is bit-exact with the
/// single-layer RNN this type used to be.
pub struct Rnn {
    time: Axis,
    hidden: Dim,
    config: RecurrentConfig,
    nonlinearity: RnnNonlinearity,
    layers: Vec<RnnLayer>,
}

impl Rnn {
    pub fn new(input: Axis, hidden: Dim, time: Axis) -> Result<Self> {
        Self::with_config(input, hidden, time, RecurrentConfig::default())
    }

    pub fn with_config(input: Axis, hidden: Dim, time: Axis, config: RecurrentConfig) -> Result<Self> {
        if time == input || time == hidden.axis {
            return Err("RNN time must be distinct from its input and hidden axes".into());
        }
        if config.num_layers == 0 {
            return Err("RNN num_layers must be positive".into());
        }
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let layer_input = if index == 0 { input } else { hidden.axis };
            let forward = RnnCell::new(layer_input, hidden)?.bias(config.bias);
            let backward = config
                .bidirectional
                .then(|| RnnCell::new(layer_input, hidden).map(|cell| cell.bias(config.bias)))
                .transpose()?;
            layers.push(RnnLayer { forward, backward });
        }
        Ok(Self {
            time,
            hidden,
            config,
            nonlinearity: RnnNonlinearity::Tanh,
            layers,
        })
    }

    /// Choose the pointwise nonlinearity before `build`. Default `Tanh`,
    /// applied identically to every layer and direction. Has no effect once
    /// any layer has already been built.
    pub fn nonlinearity(mut self, nonlinearity: RnnNonlinearity) -> Self {
        self.nonlinearity = nonlinearity;
        let bias = self.config.bias;
        for layer in &mut self.layers {
            let input_axis = layer.forward.input;
            layer.forward = RnnCell::new(input_axis, self.hidden)
                .expect("hidden extent already validated by with_config")
                .nonlinearity(nonlinearity)
                .bias(bias);
            if layer.backward.is_some() {
                layer.backward = Some(
                    RnnCell::new(input_axis, self.hidden)
                        .expect("hidden extent already validated by with_config")
                        .nonlinearity(nonlinearity)
                        .bias(bias),
                );
            }
        }
        self
    }

    pub fn config(&self) -> RecurrentConfig {
        self.config
    }

    fn directions(&self) -> usize {
        self.config.directions()
    }

    fn output_shape_for(&self, input: &Shape) -> Result<Shape> {
        let per_direction = self.layers[0].forward.output_shape_for(input)?;
        if !self.config.bidirectional {
            return Ok(per_direction);
        }
        let mut dims: Vec<_> = per_direction
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.hidden.axis)
            .collect();
        dims.push(self.hidden.axis.of(self.hidden.extent * 2));
        Shape::new(dims)
    }

    /// Zero initial state for every layer and direction, ordered
    /// `layer * directions + direction`.
    pub fn zero_states(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        let directions = self.directions();
        let mut states = Vec::with_capacity(self.layers.len() * directions);
        let mut layer_input_shape = input.shape().clone();
        for layer in &self.layers {
            let per_direction = layer
                .forward
                .output_shape_for(&without(&layer_input_shape, self.time)?)?;
            states.push(Tensor::zeros(
                per_direction.dims().iter().copied(),
                input.device(),
            )?);
            if self.config.bidirectional {
                states.push(Tensor::zeros(
                    per_direction.dims().iter().copied(),
                    input.device(),
                )?);
            }
            let mut dims: Vec<_> = layer_input_shape
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != layer.forward.input)
                .collect();
            dims.push(self.hidden.axis.of(self.hidden.extent * directions));
            layer_input_shape = Shape::new(dims)?;
        }
        Ok(states)
    }

    pub fn run(&self, input: &Tensor) -> Result<RnnRun> {
        let initial = self.zero_states(input)?;
        self.run_from(input, &initial)
    }

    pub fn run_from(&self, input: &Tensor, initial: &[Tensor]) -> Result<RnnRun> {
        let directions = self.directions();
        let expected_states = self.layers.len() * directions;
        if initial.len() != expected_states {
            return Err(format!(
                "RNN expected {expected_states} initial states (num_layers * num_directions), got {}",
                initial.len()
            )
            .into());
        }
        let steps = input.extent(self.time)?;
        let mut layer_input = input.clone();
        let mut final_states = Vec::with_capacity(expected_states);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let stripped = without(layer_input.shape(), self.time)?;
            let full_layer_shape = {
                let mut dims: Vec<_> = layer_input
                    .shape()
                    .dims()
                    .iter()
                    .copied()
                    .filter(|dim| dim.axis != layer.forward.input)
                    .collect();
                dims.push(self.hidden);
                Shape::new(dims)?
            };
            let time_position = full_layer_shape.index(self.time)?;

            let run_direction = |cell: &RnnCell, reverse: bool, state0: &Tensor| -> Result<(Tensor, Tensor)> {
                let expected = cell.output_shape_for(&stripped)?;
                if !layer_input.device().same(state0.device()) {
                    return Err("RNN input and state must use the same Device handle".into());
                }
                if !same_named_shape(state0.shape(), &expected) {
                    return Err(
                        format!("RNN state named shape mismatch: expected {:?}", expected).into(),
                    );
                }
                let projected = cell.project_input(&layer_input)?;
                let mut state = state0.clone();
                let mut hidden = vec![Tensor::zeros([], layer_input.device())?; steps];
                let order: Vec<usize> = if reverse {
                    (0..steps).rev().collect()
                } else {
                    (0..steps).collect()
                };
                for step in order {
                    state = cell.step_projected(&projected.select(self.time, step)?, &state)?;
                    hidden[step] = state.clone();
                }
                let sequence = Tensor::stack(&hidden, self.time, time_position)?;
                Ok((sequence, state))
            };

            let (forward_sequence, forward_final) =
                run_direction(&layer.forward, false, &initial[layer_index * directions])?;
            final_states.push(forward_final);

            let layer_output = if let Some(backward) = &layer.backward {
                let (backward_sequence, backward_final) =
                    run_direction(backward, true, &initial[layer_index * directions + 1])?;
                final_states.push(backward_final);
                Tensor::concat(&[forward_sequence, backward_sequence], self.hidden.axis)?
            } else {
                forward_sequence
            };
            layer_input = layer_output;
        }
        Ok(RnnRun {
            sequence: layer_input,
            state: final_states,
        })
    }
}

impl Module for Rnn {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.time)?;
        self.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        let directions = self.config.directions();
        let mut layer_input_shape = without(input, self.time)?;
        let mut layer_seed = seed;
        for layer in &mut self.layers {
            layer.forward.build(&layer_input_shape, device, layer_seed)?;
            layer_seed = layer_seed.wrapping_add(1);
            if let Some(backward) = &mut layer.backward {
                backward.build(&layer_input_shape, device, layer_seed)?;
                layer_seed = layer_seed.wrapping_add(1);
            }
            let mut dims: Vec<_> = layer_input_shape
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != layer.forward.input)
                .collect();
            dims.push(self.hidden.axis.of(self.hidden.extent * directions));
            layer_input_shape = Shape::new(dims)?;
        }
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.run(input)?.sequence)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        // The single-layer, unidirectional default keeps its original,
        // unprefixed parameter names ("input_weight", "recurrent_weight",
        // "bias") bit-exact with this type before `num_layers`/
        // `bidirectional` existed. Any other configuration prefixes each
        // name with its layer and direction, since it then owns more than
        // one of each.
        if let [layer] = self.layers.as_slice()
            && layer.backward.is_none()
        {
            return layer.forward.named_parameters();
        }
        let mut params = vec![];
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, parameter) in layer.forward.named_parameters() {
                params.push((format!("layer{index}_forward_{name}"), parameter));
            }
            if let Some(backward) = &layer.backward {
                for (name, parameter) in backward.named_parameters() {
                    params.push((format!("layer{index}_backward_{name}"), parameter));
                }
            }
        }
        params
    }
}

struct BoundGruCell {
    input_extent: usize,
    input_weight: Parameter,
    recurrent_weight: Parameter,
    bias: Option<Parameter>,
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
/// instead of introducing a second one. `bias` defaults to `true`
/// (PyTorch's default); `.bias(false)` matches `bias=False`.
pub struct GruCell {
    input: Axis,
    hidden: Dim,
    bias: bool,
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
            bias: true,
            input_role: input.role("gru_input"),
            hidden_input_role: hidden.axis.role("gru_hidden_input"),
            gate_feature: hidden.axis.role("gru_gate_feature"),
            gate: hidden.axis.role("gru_gate"),
            gate_hidden: hidden.axis.role("gru_gate_hidden"),
            bound: None,
        })
    }

    /// Toggle the combined bias parameter. Has no effect once `build` has
    /// already allocated parameters; call it before `build`. Defaults to
    /// `true`.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
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
        let projected = input
            .rename(self.input, self.input_role)?
            .contract(&bound.input_weight.tensor(), self.input_role)?;
        match &bound.bias {
            Some(bias) => projected.add(&bias.tensor()),
            None => Ok(projected),
        }
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
        let bias = self
            .bias
            .then(|| -> Result<Parameter> {
                Ok(Parameter::new(Tensor::zeros(
                    [self.gate_feature.of(gate_extent)],
                    device,
                )?))
            })
            .transpose()?;
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
                let mut params = vec![
                    ("input_weight".into(), bound.input_weight.clone()),
                    ("recurrent_weight".into(), bound.recurrent_weight.clone()),
                ];
                if let Some(bias) = &bound.bias {
                    params.push(("bias".into(), bias.clone()));
                }
                params
            })
            .unwrap_or_default()
    }
}

/// Sequence output and the per-layer, per-direction terminal hidden states
/// from one [`Gru`] run, ordered as [`LstmRun::state`] documents.
pub struct GruRun {
    pub sequence: Tensor,
    pub state: Vec<Tensor>,
}

struct GruLayer {
    forward: GruCell,
    backward: Option<GruCell>,
}

/// A stacked, optionally bidirectional, eager GRU over one named time axis.
/// See [`Lstm`]'s doc comment for the eager-transition contract and
/// [`RecurrentConfig`] for `num_layers`/`bidirectional`/`bias`. The default
/// configuration (one layer, unidirectional, bias on) is bit-exact with the
/// single-layer GRU this type used to be.
pub struct Gru {
    time: Axis,
    hidden: Dim,
    config: RecurrentConfig,
    layers: Vec<GruLayer>,
}

impl Gru {
    pub fn new(input: Axis, hidden: Dim, time: Axis) -> Result<Self> {
        Self::with_config(input, hidden, time, RecurrentConfig::default())
    }

    pub fn with_config(input: Axis, hidden: Dim, time: Axis, config: RecurrentConfig) -> Result<Self> {
        if time == input || time == hidden.axis {
            return Err("GRU time must be distinct from its input and hidden axes".into());
        }
        if config.num_layers == 0 {
            return Err("GRU num_layers must be positive".into());
        }
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            let layer_input = if index == 0 { input } else { hidden.axis };
            let forward = GruCell::new(layer_input, hidden)?.bias(config.bias);
            let backward = config
                .bidirectional
                .then(|| GruCell::new(layer_input, hidden).map(|cell| cell.bias(config.bias)))
                .transpose()?;
            layers.push(GruLayer { forward, backward });
        }
        Ok(Self {
            time,
            hidden,
            config,
            layers,
        })
    }

    pub fn config(&self) -> RecurrentConfig {
        self.config
    }

    fn directions(&self) -> usize {
        self.config.directions()
    }

    fn output_shape_for(&self, input: &Shape) -> Result<Shape> {
        let per_direction = self.layers[0].forward.output_shape_for(input)?;
        if !self.config.bidirectional {
            return Ok(per_direction);
        }
        let mut dims: Vec<_> = per_direction
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.hidden.axis)
            .collect();
        dims.push(self.hidden.axis.of(self.hidden.extent * 2));
        Shape::new(dims)
    }

    /// Zero initial state for every layer and direction, ordered
    /// `layer * directions + direction`.
    pub fn zero_states(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        let directions = self.directions();
        let mut states = Vec::with_capacity(self.layers.len() * directions);
        let mut layer_input_shape = input.shape().clone();
        for layer in &self.layers {
            let per_direction = layer
                .forward
                .output_shape_for(&without(&layer_input_shape, self.time)?)?;
            states.push(Tensor::zeros(
                per_direction.dims().iter().copied(),
                input.device(),
            )?);
            if self.config.bidirectional {
                states.push(Tensor::zeros(
                    per_direction.dims().iter().copied(),
                    input.device(),
                )?);
            }
            let mut dims: Vec<_> = layer_input_shape
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != layer.forward.input)
                .collect();
            dims.push(self.hidden.axis.of(self.hidden.extent * directions));
            layer_input_shape = Shape::new(dims)?;
        }
        Ok(states)
    }

    pub fn run(&self, input: &Tensor) -> Result<GruRun> {
        let initial = self.zero_states(input)?;
        self.run_from(input, &initial)
    }

    pub fn run_from(&self, input: &Tensor, initial: &[Tensor]) -> Result<GruRun> {
        let directions = self.directions();
        let expected_states = self.layers.len() * directions;
        if initial.len() != expected_states {
            return Err(format!(
                "GRU expected {expected_states} initial states (num_layers * num_directions), got {}",
                initial.len()
            )
            .into());
        }
        let steps = input.extent(self.time)?;
        let mut layer_input = input.clone();
        let mut final_states = Vec::with_capacity(expected_states);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let stripped = without(layer_input.shape(), self.time)?;
            let full_layer_shape = {
                let mut dims: Vec<_> = layer_input
                    .shape()
                    .dims()
                    .iter()
                    .copied()
                    .filter(|dim| dim.axis != layer.forward.input)
                    .collect();
                dims.push(self.hidden);
                Shape::new(dims)?
            };
            let time_position = full_layer_shape.index(self.time)?;

            let run_direction = |cell: &GruCell, reverse: bool, state0: &Tensor| -> Result<(Tensor, Tensor)> {
                let expected = cell.output_shape_for(&stripped)?;
                if !layer_input.device().same(state0.device()) {
                    return Err("GRU input and state must use the same Device handle".into());
                }
                if !same_named_shape(state0.shape(), &expected) {
                    return Err(
                        format!("GRU state named shape mismatch: expected {:?}", expected).into(),
                    );
                }
                let projected = cell.project_input(&layer_input)?;
                let mut state = state0.clone();
                let mut hidden = vec![Tensor::zeros([], layer_input.device())?; steps];
                let order: Vec<usize> = if reverse {
                    (0..steps).rev().collect()
                } else {
                    (0..steps).collect()
                };
                for step in order {
                    state = cell.step_projected(&projected.select(self.time, step)?, &state)?;
                    hidden[step] = state.clone();
                }
                let sequence = Tensor::stack(&hidden, self.time, time_position)?;
                Ok((sequence, state))
            };

            let (forward_sequence, forward_final) =
                run_direction(&layer.forward, false, &initial[layer_index * directions])?;
            final_states.push(forward_final);

            let layer_output = if let Some(backward) = &layer.backward {
                let (backward_sequence, backward_final) =
                    run_direction(backward, true, &initial[layer_index * directions + 1])?;
                final_states.push(backward_final);
                Tensor::concat(&[forward_sequence, backward_sequence], self.hidden.axis)?
            } else {
                forward_sequence
            };
            layer_input = layer_output;
        }
        Ok(GruRun {
            sequence: layer_input,
            state: final_states,
        })
    }
}

impl Module for Gru {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.time)?;
        self.output_shape_for(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        let directions = self.config.directions();
        let mut layer_input_shape = without(input, self.time)?;
        let mut layer_seed = seed;
        for layer in &mut self.layers {
            layer.forward.build(&layer_input_shape, device, layer_seed)?;
            layer_seed = layer_seed.wrapping_add(1);
            if let Some(backward) = &mut layer.backward {
                backward.build(&layer_input_shape, device, layer_seed)?;
                layer_seed = layer_seed.wrapping_add(1);
            }
            let mut dims: Vec<_> = layer_input_shape
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != layer.forward.input)
                .collect();
            dims.push(self.hidden.axis.of(self.hidden.extent * directions));
            layer_input_shape = Shape::new(dims)?;
        }
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.run(input)?.sequence)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        // The single-layer, unidirectional default keeps its original,
        // unprefixed parameter names ("input_weight", "recurrent_weight",
        // "bias") bit-exact with this type before `num_layers`/
        // `bidirectional` existed. Any other configuration prefixes each
        // name with its layer and direction, since it then owns more than
        // one of each.
        if let [layer] = self.layers.as_slice()
            && layer.backward.is_none()
        {
            return layer.forward.named_parameters();
        }
        let mut params = vec![];
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, parameter) in layer.forward.named_parameters() {
                params.push((format!("layer{index}_forward_{name}"), parameter));
            }
            if let Some(backward) = &layer.backward {
                for (name, parameter) in backward.named_parameters() {
                    params.push((format!("layer{index}_backward_{name}"), parameter));
                }
            }
        }
        params
    }
}
