use crate::{Axis, Device, Dim, Result, Shape, Tensor};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_PARAMETER: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParamId(u64);

/// Cloning explicitly shares a parameter, including its accumulated gradient.
#[derive(Clone)]
pub struct Parameter(Rc<RefCell<ParameterState>>);
struct ParameterState {
    id: ParamId,
    tensor: Tensor,
    version: Rc<Cell<u64>>,
}
impl Parameter {
    pub fn new(tensor: Tensor) -> Self {
        let version = Rc::new(Cell::new(0));
        Self(Rc::new(RefCell::new(ParameterState {
            id: ParamId(NEXT_PARAMETER.fetch_add(1, Ordering::Relaxed)),
            tensor: tensor.parameter_leaf(version.clone()),
            version,
        })))
    }
    pub fn id(&self) -> ParamId {
        self.0.borrow().id
    }
    pub fn tensor(&self) -> Tensor {
        self.0.borrow().tensor.clone()
    }
    pub fn grad(&self) -> Option<Tensor> {
        self.0.borrow().tensor.grad()
    }
    pub fn zero_grad(&self) {
        self.0.borrow().tensor.zero_grad();
    }
    pub(crate) fn scale_grad(&self, factor: f32) -> Result<()> {
        self.0.borrow().tensor.scale_grad(factor)
    }
    pub fn set_values(&self, values: &[f32]) -> Result<()> {
        let current = self.tensor();
        let new = Tensor::from_slice(
            values,
            current.shape().dims().iter().copied(),
            current.device(),
        )?;
        self.replace(new);
        Ok(())
    }
    pub(crate) fn replace(&self, tensor: Tensor) {
        let mut state = self.0.borrow_mut();
        state.version.set(
            state
                .version
                .get()
                .checked_add(1)
                .expect("parameter version overflow"),
        );
        state.tensor = tensor.parameter_leaf(state.version.clone());
    }
}

/// Deterministic uniform values in `[-scale, scale)` from the shared xorshift stream
/// (`crate::tensor::xorshift_unit_stream`, also used by `Tensor::uniform`/`Tensor::normal`).
/// Every parameter initializer shares this generator so a seed is reproducible.
/// A seed below 2^40 has no high bits yet, so the first sample is exactly
/// `-scale`; the stream is well mixed from the second sample on.
fn uniform_values(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    crate::tensor::xorshift_unit_stream(seed, count)
        .into_iter()
        .map(|raw| (raw * 2.0 - 1.0) * scale)
        .collect()
}

pub trait Module {
    fn output_shape(&self, input: &Shape) -> Result<Shape>;
    /// Infer extents, validate contracts, allocate and initialize parameters; never run forward.
    /// Rebuilding with compatible extents/device preserves parameters and ignores the new seed.
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape>;
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    /// Structural paths identify slots; ParamId identifies shared storage within this process.
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        vec![]
    }
    fn parameter(&self, name: &str) -> Result<Parameter> {
        let mut matches = self
            .named_parameters()
            .into_iter()
            .filter(|(path, _)| path == name);
        let (_, parameter) = matches.next().ok_or_else(|| {
            format!("unknown parameter {name:?}; build the model before accessing parameters")
        })?;
        if matches.next().is_some() {
            return Err(format!("ambiguous parameter path {name:?}").into());
        }
        Ok(parameter)
    }
    fn parameters(&self) -> Vec<Parameter> {
        self.named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect()
    }
    fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.zero_grad();
        }
    }
}

/// Contract `input`, introduce `output`, and preserve all unrelated axes.
/// Cloning a built Linear explicitly ties its parameters. Carries a learned
/// `[output]` bias by default; `.bias(false)` before `build` omits it
/// entirely, so `named_parameters` then has only `weight` and no zero-filled
/// bias tensor is ever allocated.
#[derive(Clone)]
pub struct Linear {
    input: Axis,
    output: Dim,
    input_role: Axis,
    output_role: Axis,
    bias: bool,
    bound: Option<(usize, Parameter, Option<Parameter>)>,
}
impl Linear {
    pub fn new(input: Axis, output: Dim) -> Self {
        Self {
            input,
            output,
            input_role: input.role("linear_input"),
            output_role: output.axis.role("linear_output"),
            bias: true,
            bound: None,
        }
    }
    /// Disable the learned bias term. Has no effect once `build` has already
    /// allocated parameters; call it before `build`.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}
impl Module for Linear {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.input)?;
        if let Some((bound, _, _)) = &self.bound
            && *bound != extent
        {
            return Err("Linear input extent differs from its built extent".into());
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .copied()
            .filter(|d| d.axis != self.input)
            .collect();
        dims.push(self.output);
        Shape::new(dims)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some((_, weight, _)) = &self.bound {
            if !weight.tensor().device().same(device) {
                return Err("Linear is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let extent = input.extent(self.input)?;
        let weight_shape = Shape::new([
            self.input_role.of(extent),
            self.output_role.of(self.output.extent),
        ])?;
        let scale = (6.0 / (extent + self.output.extent) as f32).sqrt();
        let values = uniform_values(seed, weight_shape.len(), scale);
        let weight = Parameter::new(Tensor::from_slice(
            &values,
            weight_shape.dims().iter().copied(),
            device,
        )?);
        let bias = if self.bias {
            Some(Parameter::new(Tensor::from_slice(
                &vec![0.0; self.output.extent],
                [self.output_role.of(self.output.extent)],
                device,
            )?))
        } else {
            None
        };
        self.bound = Some((extent, weight, bias));
        Ok(shape)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let (_, weight, bias) = self
            .bound
            .as_ref()
            .ok_or("Linear must be built before forward")?;
        let projected = input
            .rename(self.input, self.input_role)?
            .contract(&weight.tensor(), self.input_role)?;
        let projected = match bias {
            Some(bias) => projected.add(&bias.tensor())?,
            None => projected,
        };
        projected.rename(self.output_role, self.output.axis)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, weight, bias)| {
                let mut params = vec![("weight".into(), weight.clone())];
                if let Some(bias) = bias {
                    params.push(("bias".into(), bias.clone()));
                }
                params
            })
            .unwrap_or_default()
    }
}

/// Look up one learned feature vector per vocabulary entry.
///
/// Axis tensors are floating point, so a token is spelled as a one-hot
/// coordinate on the named `vocabulary` axis rather than as an integer index.
/// The lookup contracts that axis with a `[vocabulary, feature]` table, which
/// has the same value and derivative as an index gather. The input is not
/// checked to be one-hot on the device; a soft input is a weighted mixture.
/// Entries start uniform in `[-0.02, 0.02)`.
#[derive(Clone)]
pub struct Embedding {
    vocabulary: Axis,
    feature: Dim,
    vocabulary_role: Axis,
    feature_role: Axis,
    bound: Option<(usize, Parameter)>,
}
impl Embedding {
    pub fn new(vocabulary: Axis, feature: Dim) -> Self {
        Self {
            vocabulary,
            feature,
            vocabulary_role: vocabulary.role("embedding_vocabulary"),
            feature_role: feature.axis.role("embedding_feature"),
            bound: None,
        }
    }
}
impl Module for Embedding {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.vocabulary)?;
        if let Some((bound, _)) = &self.bound
            && *bound != extent
        {
            return Err("Embedding vocabulary extent differs from its built extent".into());
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.vocabulary)
            .collect();
        dims.push(self.feature);
        Shape::new(dims)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some((_, table)) = &self.bound {
            if !table.tensor().device().same(device) {
                return Err("Embedding is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let extent = input.extent(self.vocabulary)?;
        let table_shape = Shape::new([
            self.vocabulary_role.of(extent),
            self.feature_role.of(self.feature.extent),
        ])?;
        let table = Parameter::new(Tensor::from_slice(
            &uniform_values(seed, table_shape.len(), 0.02),
            table_shape.dims().iter().copied(),
            device,
        )?);
        self.bound = Some((extent, table));
        Ok(shape)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let (_, table) = self
            .bound
            .as_ref()
            .ok_or("Embedding must be built before forward")?;
        input
            .rename(self.vocabulary, self.vocabulary_role)?
            .contract(&table.tensor(), self.vocabulary_role)?
            .rename(self.feature_role, self.feature.axis)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, table)| vec![("table".into(), table.clone())])
            .unwrap_or_default()
    }
}

/// Add one learned vector per position: a `[position, feature]` table whose
/// extents are read from the input at build and broadcast over every other
/// axis. The output shape equals the input shape. Entries start uniform in
/// `[-0.02, 0.02)`.
#[derive(Clone)]
pub struct PositionEmbedding {
    position: Axis,
    feature: Axis,
    bound: Option<(usize, usize, Parameter)>,
}
impl PositionEmbedding {
    pub fn new(position: Axis, feature: Axis) -> Self {
        Self {
            position,
            feature,
            bound: None,
        }
    }
}
impl Module for PositionEmbedding {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        if self.position == self.feature {
            return Err("PositionEmbedding requires distinct position and feature axes".into());
        }
        let extents = (input.extent(self.position)?, input.extent(self.feature)?);
        if let Some((positions, features, _)) = &self.bound
            && (*positions, *features) != extents
        {
            return Err("PositionEmbedding input extents differ from its built extents".into());
        }
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some((_, _, table)) = &self.bound {
            if !table.tensor().device().same(device) {
                return Err("PositionEmbedding is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let (positions, features) = (input.extent(self.position)?, input.extent(self.feature)?);
        let table = Parameter::new(Tensor::from_slice(
            &uniform_values(seed, positions * features, 0.02),
            [self.position.of(positions), self.feature.of(features)],
            device,
        )?);
        self.bound = Some((positions, features, table));
        Ok(shape)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let (_, _, table) = self
            .bound
            .as_ref()
            .ok_or("PositionEmbedding must be built before forward")?;
        input.add(&table.tensor())
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, _, table)| vec![("table".into(), table.clone())])
            .unwrap_or_default()
    }
}

/// An additive (`f32`, added to pre-softmax scores; `-infinity` or a large
/// negative forbids a query/key pair) or boolean `attn_mask` for
/// [`MultiheadAttention`]. The boolean variant uses PyTorch's own convention:
/// a `1.0` entry forbids that query/key pair from attending, `0.0` allows it
/// -- the opposite sense from `Tensor::masked_softmax`'s own validity mask,
/// which `MultiheadAttention` inverts internally before calling it.
pub enum AttentionMask {
    Additive(Tensor),
    Boolean(Tensor),
}

/// Scaled dot-product multi-head attention, `torch.nn.MultiheadAttention` at
/// its own default `dropout=0.0`: `Attention(Q, K, V) = softmax(QK^T /
/// sqrt(head_feature)) V`, computed independently per head and concatenated
/// before one output projection (PyTorch's `forward`, `batch_first`-agnostic
/// since Axis has no positional batch axis to begin with).
///
/// `query`, `key`, and `value` are separate tensors sharing one `feature`
/// embedding axis (`embed_dim`) and one `time` sequence axis; passing the
/// same tensor for all three is self-attention, and `key`/`value` may carry a
/// different `time` extent than `query` for cross-attention as long as `key`
/// and `value` agree with each other. `feature` splits into `num_heads`
/// independent `head_feature`-wide heads (`embed_dim / num_heads`, which must
/// divide evenly); any other axis present on the inputs (a batch or
/// population axis) passes through unconsumed and broadcasts across
/// attention the same way `AttentionAxes::attend` already does in
/// `examples/training/attention`, whose self-attention composition this
/// generalizes to separate Q/K/V and explicit masks.
///
/// In-projection and out-projection are four separate [`Linear`] layers with
/// a learned bias by default (`bias=True`, PyTorch's own default); `.bias(false)`
/// before `build` omits every projection's bias parameter, matching
/// `Linear::bias(false)`. Weights use `Linear`'s own Xavier-uniform
/// initialization, not PyTorch's `nn.MultiheadAttention` in-place
/// `xavier_uniform_`/zero reset, the same documented departure `Linear`
/// itself already takes from PyTorch's `reset_parameters`.
///
/// `attn_mask` (additive or boolean, either or neither present) and
/// `key_padding_mask` (always boolean, `1.0` marks a key that must be
/// ignored for every query in its batch/population entry -- PyTorch's own
/// `key_padding_mask` convention) compose by validity AND before a single
/// `Tensor::masked_softmax` call over the key axis, so a position excluded by
/// either mask gets exactly zero probability and zero gradient; supplying
/// neither runs ordinary `softmax`. Build a mask tensor against
/// [`MultiheadAttention::query_time`] and [`MultiheadAttention::key_time`] --
/// the private per-call roles the shared `time` axis is renamed to before the
/// attention score matrix is formed, exposed so a caller can name them.
///
/// `forward` returns only the output; `forward_with_weights`'s `need_weights`
/// additionally returns the attention probabilities averaged over heads
/// (PyTorch's own default `average_attn_weights=True`), indexed by
/// `query_time` and `key_time`.
///
/// Only `dropout == 0.0`, PyTorch's own default, is supported: nonzero
/// dropout needs a seeded draw that is reproducible per training step, which
/// is the not-yet-landed training-pass contract (`Module::forward_training`,
/// `TrainingPass`, Axis issue #127); `new` rejects a nonzero value
/// immediately rather than silently ignoring it.
pub struct MultiheadAttention {
    feature: Axis,
    time: Axis,
    head: Dim,
    head_feature: Dim,
    query_time: Axis,
    key_time: Axis,
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    out_proj: Linear,
}
impl MultiheadAttention {
    /// `feature` and `time` must be distinct axes. `embed_dim` must be a
    /// nonzero multiple of a nonzero `num_heads`. `dropout` must be exactly
    /// `0.0`; see the training-pass contract note above.
    pub fn new(
        feature: Axis,
        time: Axis,
        embed_dim: usize,
        num_heads: usize,
        dropout: f32,
    ) -> Result<Self> {
        if feature == time {
            return Err("MultiheadAttention requires distinct feature and time axes".into());
        }
        if dropout != 0.0 {
            return Err(
                "MultiheadAttention only supports dropout=0.0 until the seeded, per-step \
                 training-pass contract lands (Axis issue #127); nonzero dropout is rejected \
                 rather than silently ignored"
                    .into(),
            );
        }
        if num_heads == 0 || embed_dim == 0 || !embed_dim.is_multiple_of(num_heads) {
            return Err(
                "MultiheadAttention requires a positive embed_dim divisible by a positive \
                 num_heads"
                    .into(),
            );
        }
        let head = Axis::new("mha_head").of(num_heads);
        let head_feature = Axis::new("mha_head_feature").of(embed_dim / num_heads);
        let projection = || Linear::new(feature, feature.of(embed_dim));
        Ok(Self {
            feature,
            time,
            head,
            head_feature,
            query_time: time.role("mha_query_time"),
            key_time: time.role("mha_key_time"),
            query_proj: projection(),
            key_proj: projection(),
            value_proj: projection(),
            out_proj: projection(),
        })
    }
    /// Disable every projection's learned bias term. Has no effect once
    /// `build` has already allocated parameters; call it before `build`.
    pub fn bias(mut self, bias: bool) -> Self {
        self.query_proj = self.query_proj.bias(bias);
        self.key_proj = self.key_proj.bias(bias);
        self.value_proj = self.value_proj.bias(bias);
        self.out_proj = self.out_proj.bias(bias);
        self
    }
    /// The private role `time` is renamed to for `query` positions once
    /// projected; build an `attn_mask` tensor against this axis.
    pub fn query_time(&self) -> Axis {
        self.query_time
    }
    /// The private role `time` is renamed to for `key`/`value` positions once
    /// projected; build an `attn_mask` or `key_padding_mask` tensor against
    /// this axis.
    pub fn key_time(&self) -> Axis {
        self.key_time
    }
    pub fn output_shape(&self, query: &Shape, key: &Shape, value: &Shape) -> Result<Shape> {
        query.extent(self.time)?;
        if key.extent(self.time)? != value.extent(self.time)? {
            return Err("MultiheadAttention key and value must share the same time extent".into());
        }
        self.out_proj
            .output_shape(&self.query_proj.output_shape(query)?)
    }
    pub fn build(
        &mut self,
        query: &Shape,
        key: &Shape,
        value: &Shape,
        device: &Device,
        seed: u64,
    ) -> Result<Shape> {
        let output = self.output_shape(query, key, value)?;
        let projected = self.query_proj.build(query, device, seed)?;
        self.key_proj.build(key, device, seed.wrapping_add(1))?;
        self.value_proj.build(value, device, seed.wrapping_add(2))?;
        self.out_proj
            .build(&projected, device, seed.wrapping_add(3))?;
        Ok(output)
    }
    fn split_heads(&self, x: &Tensor, role: Axis) -> Result<Tensor> {
        x.split(self.feature, [self.head, self.head_feature])?
            .rename(self.time, role)
    }
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&AttentionMask>,
        key_padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        Ok(self
            .forward_with_weights(query, key, value, attn_mask, key_padding_mask, false)?
            .0)
    }
    /// `need_weights` additionally computes and returns the head-averaged
    /// attention probabilities; pass `false` to skip that reduction entirely
    /// when only the output is needed.
    pub fn forward_with_weights(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&AttentionMask>,
        key_padding_mask: Option<&Tensor>,
        need_weights: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        self.output_shape(query.shape(), key.shape(), value.shape())?;
        let q = self.split_heads(&self.query_proj.forward(query)?, self.query_time)?;
        let k = self.split_heads(&self.key_proj.forward(key)?, self.key_time)?;
        let v = self.split_heads(&self.value_proj.forward(value)?, self.key_time)?;
        let mut scores = q
            .contract(&k, self.head_feature.axis)?
            .scale(1.0 / (self.head_feature.extent as f32).sqrt())?;
        if let Some(AttentionMask::Additive(mask)) = attn_mask {
            scores = scores.add(mask)?;
        }
        let mut validity: Option<Tensor> = None;
        if let Some(mask) = key_padding_mask {
            validity = Some(mask.logical_not()?.broadcast_to(scores.shape())?);
        }
        if let Some(AttentionMask::Boolean(mask)) = attn_mask {
            let allowed = mask.logical_not()?.broadcast_to(scores.shape())?;
            validity = Some(match validity {
                Some(v) => v.mul(&allowed)?,
                None => allowed,
            });
        }
        let probabilities = match &validity {
            Some(v) => scores.masked_softmax(self.key_time, v)?,
            None => scores.softmax(self.key_time)?,
        };
        let attended = probabilities
            .contract(&v, self.key_time)?
            .merge([self.head.axis, self.head_feature.axis], self.feature)?
            .rename(self.query_time, self.time)?;
        let output = self.out_proj.forward(&attended)?;
        let weights = if need_weights {
            Some(
                probabilities
                    .mean(self.head.axis)?
                    .rename(self.query_time, self.time)?,
            )
        } else {
            None
        };
        Ok((output, weights))
    }
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [
            ("query", &self.query_proj),
            ("key", &self.key_proj),
            ("value", &self.value_proj),
            ("output", &self.out_proj),
        ]
        .into_iter()
        .flat_map(|(prefix, layer)| {
            layer
                .named_parameters()
                .into_iter()
                .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
        })
        .collect()
    }
}

/// A family of independent linear maps sharing one explicit population axis.
/// Inputs without the population axis are broadcast to every member; inputs
/// already carrying it keep memberwise independence through later layers.
pub struct PopulationLinear {
    population: Dim,
    input: Axis,
    output: Dim,
    input_role: Axis,
    output_role: Axis,
    bound: Option<(usize, Parameter, Parameter)>,
}

impl PopulationLinear {
    pub fn new(population: Dim, input: Axis, output: Dim) -> Self {
        Self {
            population,
            input,
            output,
            input_role: input.role("population_linear_input"),
            output_role: output.axis.role("population_linear_output"),
            bound: None,
        }
    }
}

impl Module for PopulationLinear {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.input)?;
        if let Some((bound, _, _)) = &self.bound
            && *bound != extent
        {
            return Err("PopulationLinear input extent differs from its built extent".into());
        }
        if input.contains(self.population.axis)
            && input.extent(self.population.axis)? != self.population.extent
        {
            return Err("PopulationLinear population extent mismatch".into());
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != self.input)
            .collect();
        if !input.contains(self.population.axis) {
            dims.push(self.population);
        }
        dims.push(self.output);
        Shape::new(dims)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some((_, weight, _)) = &self.bound {
            if !weight.tensor().device().same(device) {
                return Err("PopulationLinear is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let extent = input.extent(self.input)?;
        let weight_shape = Shape::new([
            self.population,
            self.input_role.of(extent),
            self.output_role.of(self.output.extent),
        ])?;
        let scale = (6.0 / (extent + self.output.extent) as f32).sqrt();
        let values = uniform_values(seed, weight_shape.len(), scale);
        let weight = Parameter::new(Tensor::from_slice(
            &values,
            weight_shape.dims().iter().copied(),
            device,
        )?);
        let bias = Parameter::new(Tensor::from_slice(
            &vec![0.0; self.population.extent * self.output.extent],
            [self.population, self.output_role.of(self.output.extent)],
            device,
        )?);
        self.bound = Some((extent, weight, bias));
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let (_, weight, bias) = self
            .bound
            .as_ref()
            .ok_or("PopulationLinear must be built before forward")?;
        input
            .rename(self.input, self.input_role)?
            .contract(&weight.tensor(), self.input_role)?
            .add(&bias.tensor())?
            .rename(self.output_role, self.output.axis)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, weight, bias)| {
                vec![
                    ("weight".into(), weight.clone()),
                    ("bias".into(), bias.clone()),
                ]
            })
            .unwrap_or_default()
    }
}

#[derive(Clone, Copy)]
pub struct ReLU;
impl Module for ReLU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }
}

/// Parameter-free tanh-form GELU. PyTorch's default `nn.GELU()` is the erf form,
/// [`ExactGELU`]; a faithful port of an un-annotated PyTorch model uses that one.
#[derive(Clone, Copy)]
pub struct GELU;
impl Module for GELU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.gelu()
    }
}

/// Elementwise ELU with an explicit, finite, positive `alpha`. See
/// [`Tensor::elu`] for the exact forward/backward split at `x == 0`.
#[derive(Clone, Copy)]
pub struct ELU {
    alpha: f32,
}
impl ELU {
    pub fn new(alpha: f32) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err("ELU alpha must be finite and positive".into());
        }
        Ok(Self { alpha })
    }
}
impl Module for ELU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.elu(self.alpha)
    }
}

/// Elementwise CELU with an explicit, finite, positive `alpha`. See
/// [`Tensor::celu`] for the exact forward/backward split at `x == 0`.
#[derive(Clone, Copy)]
pub struct CELU {
    alpha: f32,
}
impl CELU {
    pub fn new(alpha: f32) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err("CELU alpha must be finite and positive".into());
        }
        Ok(Self { alpha })
    }
}
impl Module for CELU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.celu(self.alpha)
    }
}

/// Parameter-free SELU: PyTorch's fixed-constant self-normalizing activation.
/// See [`Tensor::selu`] for the exact constants.
#[derive(Clone, Copy)]
pub struct SELU;
impl Module for SELU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.selu()
    }
}

/// Elementwise Softplus with explicit `beta`/`threshold`, defaulting to
/// PyTorch's own `beta = 1`, `threshold = 20`. See [`Tensor::softplus`] for the
/// exact branch seam.
#[derive(Clone, Copy)]
pub struct Softplus {
    beta: f32,
    threshold: f32,
}
impl Softplus {
    pub fn new() -> Self {
        Self {
            beta: 1.0,
            threshold: 20.0,
        }
    }
    pub fn beta(mut self, beta: f32) -> Result<Self> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err("Softplus beta must be finite and positive".into());
        }
        self.beta = beta;
        Ok(self)
    }
    pub fn threshold(mut self, threshold: f32) -> Result<Self> {
        if !threshold.is_finite() {
            return Err("Softplus threshold must be finite".into());
        }
        self.threshold = threshold;
        Ok(self)
    }
}
impl Default for Softplus {
    fn default() -> Self {
        Self::new()
    }
}
impl Module for Softplus {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softplus(self.beta, self.threshold)
    }
}

/// Parameter-free numerically stable log-sigmoid. See [`Tensor::log_sigmoid`].
#[derive(Clone, Copy)]
pub struct LogSigmoid;
impl Module for LogSigmoid {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.log_sigmoid()
    }
}

/// Parameter-free Mish, `x * tanh(softplus(x))`. See [`Tensor::mish`].
#[derive(Clone, Copy)]
pub struct Mish;
impl Module for Mish {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.mish()
    }
}

/// Gated Linear Unit over an explicit named `axis`, halved by the gate. See
/// [`Tensor::glu`].
#[derive(Clone, Copy)]
pub struct GLU {
    axis: Axis,
}
impl GLU {
    pub fn new(axis: Axis) -> Self {
        Self { axis }
    }
}
impl Module for GLU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.axis)?;
        if extent % 2 != 0 {
            return Err("GLU requires an even extent on the split axis".into());
        }
        let dims: Vec<_> = input
            .dims()
            .iter()
            .copied()
            .map(|dim| {
                if dim.axis == self.axis {
                    self.axis.of(extent / 2)
                } else {
                    dim
                }
            })
            .collect();
        Shape::new(dims)
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        input.glu(self.axis)
    }
}

/// Parametric ReLU with a learnable weight, either shared across every element
/// ([`PReLU::shared`], PyTorch's default `num_parameters=1`) or one per entry
/// of an explicit named channel axis ([`PReLU::channel`], PyTorch's
/// `num_parameters=C`); both start at PyTorch's default init, `0.25`. See
/// [`Tensor::prelu`].
#[derive(Clone)]
pub struct PReLU {
    channel: Option<Axis>,
    bound: Option<(usize, Parameter)>,
}
impl PReLU {
    /// One weight shared by every element (PyTorch's `num_parameters=1` default).
    pub fn shared() -> Self {
        Self {
            channel: None,
            bound: None,
        }
    }
    /// One weight per entry of `channel` (PyTorch's `num_parameters=C`).
    pub fn channel(channel: Axis) -> Self {
        Self {
            channel: Some(channel),
            bound: None,
        }
    }
}
impl Module for PReLU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        if let Some(channel) = self.channel {
            input.extent(channel)?;
        }
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, device: &Device, _seed: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some((_, weight)) = &self.bound {
            if !weight.tensor().device().same(device) {
                return Err("PReLU is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let extent = match self.channel {
            Some(channel) => input.extent(channel)?,
            None => 1,
        };
        let weight_dims: Vec<Dim> = match self.channel {
            Some(channel) => vec![channel.of(extent)],
            None => vec![],
        };
        let weight = Parameter::new(Tensor::from_slice(
            &vec![0.25; extent],
            weight_dims,
            device,
        )?);
        self.bound = Some((extent, weight));
        Ok(shape)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let (_, weight) = self
            .bound
            .as_ref()
            .ok_or("PReLU must be built before forward")?;
        input.prelu(&weight.tensor())
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, weight)| vec![("weight".into(), weight.clone())])
            .unwrap_or_default()
    }
}

/// Numerically stable log-softmax over an explicit named `axis`. See
/// [`Tensor::log_softmax`].
#[derive(Clone, Copy)]
pub struct LogSoftmax {
    axis: Axis,
}
impl LogSoftmax {
    pub fn new(axis: Axis) -> Self {
        Self { axis }
    }
}
impl Module for LogSoftmax {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.axis)?;
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        input.log_softmax(self.axis)
    }
}

/// Softmin over an explicit named `axis`. See [`Tensor::softmin`].
#[derive(Clone, Copy)]
pub struct Softmin {
    axis: Axis,
}
impl Softmin {
    pub fn new(axis: Axis) -> Self {
        Self { axis }
    }
}
impl Module for Softmin {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.axis)?;
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        input.softmin(self.axis)
    }
}

/// Softmax over an explicit named `channel` axis of a `[channel, height,
/// width]` input (PyTorch's `Softmax2d`, which requires exactly that rank).
/// Delegates to the existing named-axis [`Tensor::softmax`]; `height`/`width`
/// exist to state and check the contract, since `softmax` already treats every
/// other axis independently.
#[derive(Clone, Copy)]
pub struct Softmax2d {
    channel: Axis,
    height: Axis,
    width: Axis,
}
impl Softmax2d {
    pub fn new(channel: Axis, height: Axis, width: Axis) -> Result<Self> {
        if channel == height || channel == width || height == width {
            return Err("Softmax2d requires distinct channel, height, and width axes".into());
        }
        Ok(Self {
            channel,
            height,
            width,
        })
    }
}
impl Module for Softmax2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        if input.rank() != 3
            || !input.contains(self.channel)
            || !input.contains(self.height)
            || !input.contains(self.width)
        {
            return Err("Softmax2d requires exactly the channel, height, and width axes".into());
        }
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        input.softmax(self.channel)
    }
}

/// Parameter-free erf-form GELU for checkpoints trained with exact GELU, which is
/// PyTorch's default `nn.GELU()`.
#[derive(Clone, Copy)]
pub struct ExactGELU;
impl Module for ExactGELU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.gelu_exact()
    }
}

/// Parameter-free sigmoid linear unit, `x * sigmoid(x)`.
#[derive(Clone, Copy)]
pub struct SiLU;
impl Module for SiLU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.silu()
    }
}

/// Parameter-free leaky ReLU with an explicit negative-region slope.
/// The derivative at exactly zero is the negative slope, matching PyTorch.
#[derive(Clone, Copy)]
pub struct LeakyReLU {
    negative_slope: f32,
}
impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Result<Self> {
        if !negative_slope.is_finite() || negative_slope < 0.0 {
            return Err("LeakyReLU negative slope must be finite and non-negative".into());
        }
        Ok(Self { negative_slope })
    }
}
impl Module for LeakyReLU {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.leaky_relu(self.negative_slope)
    }
}

/// Parameter-free elementwise hyperbolic tangent module.
#[derive(Clone, Copy)]
pub struct Tanh;
impl Module for Tanh {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.tanh()
    }
}

/// Parameter-free `ReLU6`: [`Hardtanh`] with fixed bounds `0` and `6`. See
/// [`Tensor::relu6`] for the exact forward and (strictly-interior) backward semantics.
#[derive(Clone, Copy)]
pub struct ReLU6;
impl Module for ReLU6 {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu6()
    }
}

/// Parameter-free `Hardtanh` with explicit `min_val`/`max_val` bounds. See
/// [`Tensor::hardtanh`] for the exact forward and gradient-at-the-kinks semantics.
#[derive(Clone, Copy)]
pub struct Hardtanh {
    min_val: f32,
    max_val: f32,
}
impl Hardtanh {
    pub fn new(min_val: f32, max_val: f32) -> Result<Self> {
        if !min_val.is_finite() || !max_val.is_finite() {
            return Err("Hardtanh bounds must be finite".into());
        }
        if min_val > max_val {
            return Err(format!(
                "Hardtanh requires min_val <= max_val, got min_val={min_val} max_val={max_val}"
            )
            .into());
        }
        Ok(Self { min_val, max_val })
    }
}
impl Module for Hardtanh {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardtanh(self.min_val, self.max_val)
    }
}

/// Parameter-free `Hardsigmoid`. See [`Tensor::hardsigmoid`] for the exact forward and
/// gradient-at-the-kinks semantics.
#[derive(Clone, Copy)]
pub struct Hardsigmoid;
impl Module for Hardsigmoid {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardsigmoid()
    }
}

/// Parameter-free `Hardswish`. See [`Tensor::hardswish`] for the exact forward and
/// gradient-at-the-kinks semantics, including its asymmetric boundary convention.
#[derive(Clone, Copy)]
pub struct Hardswish;
impl Module for Hardswish {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardswish()
    }
}

/// Parameter-free `Hardshrink` with an explicit `lambd`. See [`Tensor::hardshrink`] for
/// the exact forward and gradient-at-the-band semantics.
#[derive(Clone, Copy)]
pub struct Hardshrink {
    lambd: f32,
}
impl Hardshrink {
    pub fn new(lambd: f32) -> Result<Self> {
        if !lambd.is_finite() || lambd < 0.0 {
            return Err("Hardshrink lambd must be finite and non-negative".into());
        }
        Ok(Self { lambd })
    }
}
impl Module for Hardshrink {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.hardshrink(self.lambd)
    }
}

/// Parameter-free `Softshrink` with an explicit `lambd`. See [`Tensor::softshrink`] for
/// the exact forward and gradient-at-the-band semantics.
#[derive(Clone, Copy)]
pub struct Softshrink {
    lambd: f32,
}
impl Softshrink {
    pub fn new(lambd: f32) -> Result<Self> {
        if !lambd.is_finite() || lambd < 0.0 {
            return Err("Softshrink lambd must be finite and non-negative".into());
        }
        Ok(Self { lambd })
    }
}
impl Module for Softshrink {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softshrink(self.lambd)
    }
}

/// Parameter-free `Threshold` with an explicit `threshold`/`value`. PyTorch requires
/// both explicitly (no defaults); see [`Tensor::threshold`] for the exact forward and
/// gradient-at-the-kink semantics.
#[derive(Clone, Copy)]
pub struct Threshold {
    threshold: f32,
    value: f32,
}
impl Threshold {
    pub fn new(threshold: f32, value: f32) -> Result<Self> {
        if !threshold.is_finite() || !value.is_finite() {
            return Err("Threshold threshold and value must both be finite".into());
        }
        Ok(Self { threshold, value })
    }
}
impl Module for Threshold {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.threshold(self.threshold, self.value)
    }
}

/// Parameter-free `Softsign`. See [`Tensor::softsign`] for the exact forward and
/// backward semantics (smooth everywhere, no true kink).
#[derive(Clone, Copy)]
pub struct Softsign;
impl Module for Softsign {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.softsign()
    }
}

/// Parameter-free `Tanhshrink`. See [`Tensor::tanhshrink`] for the exact forward and
/// backward semantics (smooth everywhere, no true kink).
#[derive(Clone, Copy)]
pub struct Tanhshrink;
impl Module for Tanhshrink {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.tanhshrink()
    }
}

/// Parameter-free sign quantization to `{-1, +1}` with a straight-through
/// backward. See [`Tensor::sign_straight_through`] for the exact forward and
/// backward semantics this reproduces from bae's `bitae.quantize`.
#[derive(Clone, Copy)]
pub struct SignStraightThrough;
impl Module for SignStraightThrough {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.sign_straight_through()
    }
}

/// Every padded axis and its `(before, after)` extents, shared by all five
/// padding modes below. PyTorch's `nn.*Pad1d/2d/3d` classes take one flat
/// tuple ordered *last-dim-first* (`ZeroPad2d((left, right, top, bottom))`
/// pads width before height); Axis instead takes one `(Axis, before, after)`
/// entry per padded axis, in any order, since axis identity -- not tuple
/// position -- selects which dimension moves. A `ZeroPad2d((left, right,
/// top, bottom))` call becomes `ZeroPad::new([(height, top, bottom),
/// (width, left, right)])`; the "1d"/"2d"/"3d" split in PyTorch's names is
/// purely how many entries the list has (one, two, or three), not a
/// different type or contract here. Every axis not listed is preserved
/// unchanged, matching how `Conv2d`/`MaxPool2d` append or resize only their
/// own named axes.
fn validate_pad_list(name: &str, pads: &[(Axis, usize, usize)]) -> Result<()> {
    if pads.is_empty() {
        return Err(format!("{name} requires at least one padded axis").into());
    }
    for (index, &(axis, _, _)) in pads.iter().enumerate() {
        if pads[..index].iter().any(|&(other, _, _)| other == axis) {
            return Err(format!("{name} lists axis {axis:?} more than once").into());
        }
    }
    Ok(())
}

/// Add `before + after` to each listed axis's extent, checked for overflow, and
/// preserve every other axis in place. Shared shape math for all five padding
/// modes: they differ only in what forward writes into the new coordinates.
fn padded_shape(name: &str, input: &Shape, pads: &[(Axis, usize, usize)]) -> Result<Shape> {
    let mut dims = input.dims().to_vec();
    for &(axis, before, after) in pads {
        let slot = dims
            .iter_mut()
            .find(|dim| dim.axis == axis)
            .ok_or_else(|| format!("{name} axis {axis:?} is not in the input shape"))?;
        let extent = slot
            .extent
            .checked_add(before)
            .and_then(|extent| extent.checked_add(after))
            .ok_or_else(|| format!("{name} padded extent overflow"))?;
        *slot = axis.of(extent);
    }
    Shape::new(dims)
}

/// Exact zero padding, covering `ZeroPad1d`/`ZeroPad2d`/`ZeroPad3d` (see the
/// tuple-to-axis mapping above `validate_pad_list`). Composed entirely from
/// repeated [`Tensor::pad_zeros`], one call per listed axis; backward is
/// `pad_zeros`'s own exact crop, so a source element's gradient is unaffected
/// by padding.
pub struct ZeroPad {
    pads: Vec<(Axis, usize, usize)>,
}
impl ZeroPad {
    pub fn new(pads: impl IntoIterator<Item = (Axis, usize, usize)>) -> Result<Self> {
        let pads: Vec<_> = pads.into_iter().collect();
        validate_pad_list("ZeroPad", &pads)?;
        Ok(Self { pads })
    }
}
impl Module for ZeroPad {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        padded_shape("ZeroPad", input, &self.pads)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.pads
            .iter()
            .try_fold(input.clone(), |value, &(axis, before, after)| {
                value.pad_zeros(axis, before, after)
            })
    }
}

/// Padding at an arbitrary constant value, covering `ConstantPad1d/2d/3d`.
/// `ZeroPad`'s own zero-padded tensor already has the correct interior
/// (`pad_zeros` never touches it) and exact zeros in the border; a second,
/// independent zero-padding of a constant ones mask (`Tensor::zeros(..).ge(0.0)`,
/// which carries no gradient edge of its own) turns into an interior/border
/// indicator, and its `logical_not` -- exactly zero inside, exactly one in the
/// border -- scaled by `value` and added supplies the constant. Backward is
/// therefore identical to `ZeroPad`'s: the border term is a detached constant
/// and contributes no gradient.
pub struct ConstantPad {
    pads: Vec<(Axis, usize, usize)>,
    value: f32,
}
impl ConstantPad {
    pub fn new(pads: impl IntoIterator<Item = (Axis, usize, usize)>, value: f32) -> Result<Self> {
        let pads: Vec<_> = pads.into_iter().collect();
        validate_pad_list("ConstantPad", &pads)?;
        if !value.is_finite() {
            return Err("ConstantPad value must be finite".into());
        }
        Ok(Self { pads, value })
    }
}
impl Module for ConstantPad {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        padded_shape("ConstantPad", input, &self.pads)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let interior = Tensor::zeros(input.shape().dims().to_vec(), input.device())?.ge(0.0)?;
        let (zero_padded, interior_padded) = self.pads.iter().try_fold(
            (input.clone(), interior),
            |(value, mask), &(axis, before, after)| -> Result<_> {
                Ok((
                    value.pad_zeros(axis, before, after)?,
                    mask.pad_zeros(axis, before, after)?,
                ))
            },
        )?;
        let border = interior_padded.logical_not()?.scale(self.value)?;
        zero_padded.add(&border)
    }
}

/// A source coordinate function for `edge_pad`: maps an extended-domain
/// coordinate (which may be negative or `>= extent`) back into `[0, extent)`.
type EdgeSource = fn(usize, isize) -> usize;

/// PyTorch's whole-sample reflection: mirror around each edge coordinate
/// without repeating it (`-1` reflects to `1`, not `0`). Only ever called
/// with `before < extent` and `after < extent` (enforced before launch), so
/// one bounce off each edge always lands back inside `[0, extent)`; a larger
/// pad would need a second bounce, which is exactly PyTorch's own
/// restriction on `ReflectionPad*`.
fn reflect_source(extent: usize, coordinate: isize) -> usize {
    if coordinate < 0 {
        (-coordinate) as usize
    } else if coordinate as usize >= extent {
        2 * (extent - 1) - coordinate as usize
    } else {
        coordinate as usize
    }
}

/// PyTorch's edge replication: clamp into `[0, extent)`. Valid for any pad
/// size, since clamping never leaves the axis.
fn replicate_source(extent: usize, coordinate: isize) -> usize {
    coordinate.clamp(0, extent as isize - 1) as usize
}

/// Pad one named axis by building a host-side source index for every output
/// coordinate and reading it back with [`Tensor::gather`]. Gather's backward
/// is an exact scatter-add over repeated indices (`Plan::gather`/`Plan::reverse`),
/// which is exactly the gradient reflection and replication padding need: an
/// edge-adjacent source element that feeds more than one output coordinate
/// accumulates every contribution instead of losing all but one.
fn edge_pad(
    value: &Tensor,
    axis: Axis,
    before: usize,
    after: usize,
    source: EdgeSource,
) -> Result<Tensor> {
    if before == 0 && after == 0 {
        return Ok(value.clone());
    }
    let extent = value.extent(axis)?;
    let out_len = extent
        .checked_add(before)
        .and_then(|extent| extent.checked_add(after))
        .ok_or("edge padding extent overflow")?;
    let index: Vec<usize> = (0..out_len)
        .map(|position| source(extent, position as isize - before as isize))
        .collect();
    let role = axis.role("edge_pad");
    value.gather(axis, &index, role)?.rename(role, axis)
}

/// Reflection padding, covering `ReflectionPad1d/2d/3d`. Matches PyTorch's
/// own constraint that each side's padding be strictly less than the axis's
/// extent (checked before launch, in `reflect_source`'s doc); see
/// `edge_pad` for the composition and its exact gradient.
pub struct ReflectionPad {
    pads: Vec<(Axis, usize, usize)>,
}
impl ReflectionPad {
    pub fn new(pads: impl IntoIterator<Item = (Axis, usize, usize)>) -> Result<Self> {
        let pads: Vec<_> = pads.into_iter().collect();
        validate_pad_list("ReflectionPad", &pads)?;
        Ok(Self { pads })
    }
}
impl Module for ReflectionPad {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        for &(axis, before, after) in &self.pads {
            let extent = input.extent(axis)?;
            if before >= extent || after >= extent {
                return Err(format!(
                    "ReflectionPad padding on axis {axis:?} must be less than its extent {extent}, matching PyTorch's own constraint"
                )
                .into());
            }
        }
        padded_shape("ReflectionPad", input, &self.pads)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.pads
            .iter()
            .try_fold(input.clone(), |value, &(axis, before, after)| {
                edge_pad(&value, axis, before, after, reflect_source)
            })
    }
}

/// Edge replication padding, covering `ReplicationPad1d/2d/3d`. No pad-size
/// constraint beyond the shared overflow check: see `edge_pad` for the
/// composition and its exact gradient.
pub struct ReplicationPad {
    pads: Vec<(Axis, usize, usize)>,
}
impl ReplicationPad {
    pub fn new(pads: impl IntoIterator<Item = (Axis, usize, usize)>) -> Result<Self> {
        let pads: Vec<_> = pads.into_iter().collect();
        validate_pad_list("ReplicationPad", &pads)?;
        Ok(Self { pads })
    }
}
impl Module for ReplicationPad {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        padded_shape("ReplicationPad", input, &self.pads)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.pads
            .iter()
            .try_fold(input.clone(), |value, &(axis, before, after)| {
                edge_pad(&value, axis, before, after, replicate_source)
            })
    }
}

/// Pad one named axis by wrapping values from its opposite edge: the
/// composition `roll` itself uses (`narrow` the wrap-around slice, `concat`
/// it onto the un-narrowed original). Backward needs no dedicated rule:
/// `concat`'s backward narrows the upstream gradient back to each operand's
/// own output slice, and `narrow`'s backward zero-scatters that slice into
/// the source axis, so a source element used by both the wrapped copy and
/// the interior copy (whenever `before` or `after` is positive) accumulates
/// gradient from both through ordinary multi-use accumulation.
fn circular_pad_axis(value: &Tensor, axis: Axis, before: usize, after: usize) -> Result<Tensor> {
    if before == 0 && after == 0 {
        return Ok(value.clone());
    }
    let extent = value.extent(axis)?;
    let mut parts = Vec::with_capacity(3);
    if before > 0 {
        parts.push(value.narrow(axis, extent - before, before)?);
    }
    parts.push(value.clone());
    if after > 0 {
        parts.push(value.narrow(axis, 0, after)?);
    }
    Tensor::concat(&parts, axis)
}

/// Circular (wrap-around) padding, covering `CircularPad1d/2d/3d`. Matches
/// PyTorch's own constraint that each side's padding be at most the axis's
/// extent (a full wrap is allowed, unlike reflection's strict `<`); see
/// `circular_pad_axis` for the composition and its exact gradient.
pub struct CircularPad {
    pads: Vec<(Axis, usize, usize)>,
}
impl CircularPad {
    pub fn new(pads: impl IntoIterator<Item = (Axis, usize, usize)>) -> Result<Self> {
        let pads: Vec<_> = pads.into_iter().collect();
        validate_pad_list("CircularPad", &pads)?;
        Ok(Self { pads })
    }
}
impl Module for CircularPad {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        for &(axis, before, after) in &self.pads {
            let extent = input.extent(axis)?;
            if before > extent || after > extent {
                return Err(format!(
                    "CircularPad padding on axis {axis:?} must be at most its extent {extent}, matching PyTorch's own constraint"
                )
                .into());
            }
        }
        padded_shape("CircularPad", input, &self.pads)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.pads
            .iter()
            .try_fold(input.clone(), |value, &(axis, before, after)| {
                circular_pad_axis(&value, axis, before, after)
            })
    }
}

/// Sub-pixel upscaling: rearrange `channel` (extent `C * factor^2`) into
/// `factor` blocks along each of the two named `spatial` axes, expanding
/// each by `factor` and shrinking `channel` to `C`. PyTorch's own
/// `pixel_shuffle` decomposition -- reshape the channel axis to
/// `(C, factor, factor)`, permute to interleave each `factor` axis with its
/// spatial axis, then flatten -- is exactly [`Tensor::split`] (channel into
/// `[out_channel, row_factor, col_factor]`, outermost first) followed by two
/// [`Tensor::merge`] calls (`[height, row_factor]`, `[width, col_factor]`,
/// spatial axis outermost so it varies slower than its own sub-pixel
/// offset). No dedicated kernel or backward rule: `split`/`merge` are both
/// already differentiable.
pub struct PixelShuffle {
    channel: Axis,
    spatial: [Axis; 2],
    factor: usize,
}
impl PixelShuffle {
    pub fn new(channel: Axis, spatial: [Axis; 2], factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err("PixelShuffle upscale_factor must be at least 1".into());
        }
        if spatial[0] == spatial[1] || spatial.contains(&channel) {
            return Err("PixelShuffle requires distinct channel and spatial axes".into());
        }
        Ok(Self {
            channel,
            spatial,
            factor,
        })
    }
}
impl Module for PixelShuffle {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let square = self
            .factor
            .checked_mul(self.factor)
            .ok_or("PixelShuffle upscale_factor squared overflow")?;
        let channel_extent = input.extent(self.channel)?;
        if channel_extent % square != 0 {
            return Err(format!(
                "PixelShuffle channel axis {:?} extent {channel_extent} is not divisible by upscale_factor^2 ({square})",
                self.channel
            )
            .into());
        }
        let out_channel = channel_extent / square;
        let [height, width] = self.spatial;
        let new_height = input
            .extent(height)?
            .checked_mul(self.factor)
            .ok_or("PixelShuffle height overflow")?;
        let new_width = input
            .extent(width)?
            .checked_mul(self.factor)
            .ok_or("PixelShuffle width overflow")?;
        let dims = input.dims().iter().map(|dim| {
            if dim.axis == self.channel {
                self.channel.of(out_channel)
            } else if dim.axis == height {
                height.of(new_height)
            } else if dim.axis == width {
                width.of(new_width)
            } else {
                *dim
            }
        });
        Shape::new(dims)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let channel_extent = input.extent(self.channel)?;
        let out_channel = channel_extent / (self.factor * self.factor);
        let (out_channel_axis, row_factor, col_factor) = (
            self.channel.role("pixel_shuffle_channel"),
            self.channel.role("pixel_shuffle_row_factor"),
            self.channel.role("pixel_shuffle_col_factor"),
        );
        let [height, width] = self.spatial;
        let (new_height, new_width) = (
            height.role("pixel_shuffle_height"),
            width.role("pixel_shuffle_width"),
        );
        let split = input.split(
            self.channel,
            [
                out_channel_axis.of(out_channel),
                row_factor.of(self.factor),
                col_factor.of(self.factor),
            ],
        )?;
        let merged = split
            .merge([height, row_factor], new_height)?
            .merge([width, col_factor], new_width)?;
        merged
            .rename(out_channel_axis, self.channel)?
            .rename(new_height, height)?
            .rename(new_width, width)
    }
}

/// The exact inverse of [`PixelShuffle`]: shrink `channel`'s two named
/// `spatial` axes by `factor` each and grow `channel` by `factor^2`.
/// Composed as [`Tensor::split`] on each spatial axis (`[outer, sub_pixel]`,
/// spatial axis outermost, undoing `PixelShuffle`'s merge order exactly)
/// followed by one [`Tensor::merge`] of `[channel, row_factor, col_factor]`
/// into the new channel axis (undoing `PixelShuffle`'s split order exactly).
pub struct PixelUnshuffle {
    channel: Axis,
    spatial: [Axis; 2],
    factor: usize,
}
impl PixelUnshuffle {
    pub fn new(channel: Axis, spatial: [Axis; 2], factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err("PixelUnshuffle downscale_factor must be at least 1".into());
        }
        if spatial[0] == spatial[1] || spatial.contains(&channel) {
            return Err("PixelUnshuffle requires distinct channel and spatial axes".into());
        }
        Ok(Self {
            channel,
            spatial,
            factor,
        })
    }
}
impl Module for PixelUnshuffle {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let [height, width] = self.spatial;
        let height_extent = input.extent(height)?;
        let width_extent = input.extent(width)?;
        if height_extent % self.factor != 0 || width_extent % self.factor != 0 {
            return Err(format!(
                "PixelUnshuffle spatial extents ({height_extent}, {width_extent}) must both be divisible by downscale_factor {}",
                self.factor
            )
            .into());
        }
        let square = self
            .factor
            .checked_mul(self.factor)
            .ok_or("PixelUnshuffle downscale_factor squared overflow")?;
        let channel_extent = input.extent(self.channel)?;
        let new_channel = channel_extent
            .checked_mul(square)
            .ok_or("PixelUnshuffle channel overflow")?;
        let dims = input.dims().iter().map(|dim| {
            if dim.axis == self.channel {
                self.channel.of(new_channel)
            } else if dim.axis == height {
                height.of(height_extent / self.factor)
            } else if dim.axis == width {
                width.of(width_extent / self.factor)
            } else {
                *dim
            }
        });
        Shape::new(dims)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let [height, width] = self.spatial;
        let (height_outer, row_factor) = (
            height.role("pixel_unshuffle_height"),
            height.role("pixel_unshuffle_row_factor"),
        );
        let (width_outer, col_factor) = (
            width.role("pixel_unshuffle_width"),
            width.role("pixel_unshuffle_col_factor"),
        );
        let height_extent = input.extent(height)?;
        let width_extent = input.extent(width)?;
        let split = input
            .split(
                height,
                [
                    height_outer.of(height_extent / self.factor),
                    row_factor.of(self.factor),
                ],
            )?
            .split(
                width,
                [
                    width_outer.of(width_extent / self.factor),
                    col_factor.of(self.factor),
                ],
            )?;
        let new_channel = self.channel.role("pixel_unshuffle_channel");
        let merged = split.merge([self.channel, row_factor, col_factor], new_channel)?;
        merged
            .rename(new_channel, self.channel)?
            .rename(height_outer, height)?
            .rename(width_outer, width)
    }
}

/// `torch.nn.ChannelShuffle(groups)`: split `channel` (extent `C`) into
/// `[group (groups), within (C / groups)]` -- PyTorch's actual internal
/// reshape order, `(N, groups, C/groups, *)`, which is the reverse of its
/// own prose ("divides ... into g groups as (N, C/g, g, *)"), verified
/// against a real `channel_shuffle` run rather than trusted from the text
/// -- then merge back in swapped order `[within, group]`, reproducing its
/// documented transpose-and-flatten exactly (`[ch0, ch1, ch2, ch3]` at
/// `groups=2` becomes `[ch0, ch2, ch1, ch3]`, matching the PyTorch docs'
/// own example; the two group/within orderings only coincide when
/// `groups == C / groups`, which is why that example alone cannot
/// distinguish them). Every other axis is untouched, so this works on any
/// `(N, C, *)` shape without naming the trailing axes. No dedicated kernel
/// or backward rule: `split` and `merge` are both already differentiable.
pub struct ChannelShuffle {
    channel: Axis,
    groups: usize,
}
impl ChannelShuffle {
    pub fn new(channel: Axis, groups: usize) -> Result<Self> {
        if groups == 0 {
            return Err("ChannelShuffle groups must be at least 1".into());
        }
        Ok(Self { channel, groups })
    }
}
impl Module for ChannelShuffle {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.channel)?;
        if extent % self.groups != 0 {
            return Err(format!(
                "ChannelShuffle channel axis {:?} extent {extent} is not divisible by groups {}",
                self.channel, self.groups
            )
            .into());
        }
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let extent = input.extent(self.channel)?;
        let within_extent = extent / self.groups;
        let (group, within) = (
            self.channel.role("channel_shuffle_group"),
            self.channel.role("channel_shuffle_within"),
        );
        let split = input.split(
            self.channel,
            [group.of(self.groups), within.of(within_extent)],
        )?;
        let new_channel = self.channel.role("channel_shuffle_merged");
        let merged = split.merge([within, group], new_channel)?;
        merged.rename(new_channel, self.channel)
    }
}

pub trait IntoLayers {
    fn into_layers(self) -> Vec<Box<dyn Module>>;
}
impl IntoLayers for Vec<Box<dyn Module>> {
    fn into_layers(self) -> Vec<Box<dyn Module>> {
        self
    }
}
macro_rules! layers {
    ($($name:ident),+) => {
        impl<$($name: Module + 'static),+> IntoLayers for ($($name,)+) {
            #[allow(non_snake_case)]
            fn into_layers(self) -> Vec<Box<dyn Module>> { let ($($name,)+) = self; vec![$(Box::new($name)),+] }
        }
    };
}
layers!(A);
layers!(A, B);
layers!(A, B, C);
layers!(A, B, C, D);
layers!(A, B, C, D, E);

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}
impl Sequential {
    pub fn new(layers: impl IntoLayers) -> Self {
        Self {
            layers: layers.into_layers(),
        }
    }
}
impl Module for Sequential {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.layers
            .iter()
            .try_fold(input.clone(), |shape, layer| layer.output_shape(&shape))
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.output_shape(input)?; // Validate the complete transformation before allocating any parameters.
        let mut shape = input.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            shape = layer.build(&shape, device, seed.wrapping_add(i as u64))?;
        }
        Ok(shape)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        self.layers
            .iter()
            .try_fold(input.clone(), |value, layer| layer.forward(&value))
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| {
                layer
                    .named_parameters()
                    .into_iter()
                    .map(move |(name, parameter)| (format!("{i}.{name}"), parameter))
            })
            .collect()
    }
}

/// Ordered collection of modules with no forward of its own, PyTorch's
/// `nn.ModuleList`: PyTorch's own `nn.Module.forward` on it raises
/// `NotImplementedError`, since only a subclass that actually threads data
/// through its children -- such as [`Sequential`] -- has one, so
/// `output_shape`/`build`/`forward` below each return an explicit error
/// instead of guessing an order. Build and run a held module directly
/// through [`ModuleList::get`]/[`ModuleList::get_mut`]/[`ModuleList::iter`].
/// `named_parameters` still aggregates every held module's own parameters
/// under [`Sequential`]'s own slot-path convention, `"{index}.{name}"` (e.g.
/// `"0.weight"`), so they stay addressable through `model.parameter(path)`
/// even though nothing here threads a single input through them in order.
/// [`ModuleDict`], [`ParameterList`], and [`ParameterDict`] share this exact
/// contract, keyed or indexed differently.
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}
impl ModuleList {
    pub fn new(modules: Vec<Box<dyn Module>>) -> Self {
        Self { modules }
    }
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.modules.push(module);
    }
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|module| module.as_ref())
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut (dyn Module + '_)> {
        let module = self.modules.get_mut(index)?;
        Some(module.as_mut())
    }
    pub fn iter(&self) -> impl Iterator<Item = &dyn Module> {
        self.modules.iter().map(|module| module.as_ref())
    }
}
impl Module for ModuleList {
    fn output_shape(&self, _input: &Shape) -> Result<Shape> {
        Err(
            "ModuleList has no forward of its own; call output_shape on a held module directly"
                .into(),
        )
    }
    fn build(&mut self, _input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        Err("ModuleList has no forward of its own; build each held module directly".into())
    }
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err("ModuleList has no forward of its own, matching PyTorch's nn.ModuleList".into())
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.modules
            .iter()
            .enumerate()
            .flat_map(|(i, module)| {
                module
                    .named_parameters()
                    .into_iter()
                    .map(move |(name, parameter)| (format!("{i}.{name}"), parameter))
            })
            .collect()
    }
}

/// Named collection of modules with no forward of its own, insertion-ordered
/// like PyTorch's `OrderedDict`-backed `nn.ModuleDict`. See [`ModuleList`]'s
/// own doc for the shared no-forward/aggregation contract; slots here are
/// addressable as `"{key}.{name}"`. Duplicate keys are rejected before any
/// module is built.
pub struct ModuleDict {
    entries: Vec<(String, Box<dyn Module>)>,
}
impl ModuleDict {
    pub fn new(entries: Vec<(String, Box<dyn Module>)>) -> Result<Self> {
        for (index, (name, _)) in entries.iter().enumerate() {
            if entries[..index].iter().any(|(other, _)| other == name) {
                return Err(format!("ModuleDict contains duplicate key {name:?}").into());
            }
        }
        Ok(Self { entries })
    }
    pub fn insert(&mut self, name: impl Into<String>, module: Box<dyn Module>) -> Result<()> {
        let name = name.into();
        if self.entries.iter().any(|(other, _)| *other == name) {
            return Err(format!("ModuleDict contains duplicate key {name:?}").into());
        }
        self.entries.push((name, module));
        Ok(())
    }
    pub fn get(&self, name: &str) -> Option<&dyn Module> {
        self.entries
            .iter()
            .find(|(other, _)| other == name)
            .map(|(_, module)| module.as_ref())
    }
    pub fn get_mut(&mut self, name: &str) -> Option<&mut (dyn Module + '_)> {
        let (_, module) = self.entries.iter_mut().find(|(other, _)| other == name)?;
        Some(module.as_mut())
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
impl Module for ModuleDict {
    fn output_shape(&self, _input: &Shape) -> Result<Shape> {
        Err(
            "ModuleDict has no forward of its own; call output_shape on a held module directly"
                .into(),
        )
    }
    fn build(&mut self, _input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        Err("ModuleDict has no forward of its own; build each held module directly".into())
    }
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err("ModuleDict has no forward of its own, matching PyTorch's nn.ModuleDict".into())
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.entries
            .iter()
            .flat_map(|(name, module)| {
                module
                    .named_parameters()
                    .into_iter()
                    .map(move |(sub, parameter)| (format!("{name}.{sub}"), parameter))
            })
            .collect()
    }
}

/// Ordered collection of standalone parameters, not owned by any module:
/// PyTorch's `nn.ParameterList`. See [`ModuleList`]'s own doc for the shared
/// no-forward/aggregation contract; slots here are addressable by index,
/// `"{index}"`.
#[derive(Clone)]
pub struct ParameterList {
    parameters: Vec<Parameter>,
}
impl ParameterList {
    pub fn new(parameters: Vec<Parameter>) -> Self {
        Self { parameters }
    }
    pub fn push(&mut self, parameter: Parameter) {
        self.parameters.push(parameter);
    }
    pub fn get(&self, index: usize) -> Option<&Parameter> {
        self.parameters.get(index)
    }
    pub fn len(&self) -> usize {
        self.parameters.len()
    }
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }
}
impl Module for ParameterList {
    fn output_shape(&self, _input: &Shape) -> Result<Shape> {
        Err("ParameterList has no forward of its own".into())
    }
    fn build(&mut self, _input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        Err("ParameterList has no forward of its own".into())
    }
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err("ParameterList has no forward of its own, matching PyTorch's nn.ParameterList".into())
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.parameters
            .iter()
            .enumerate()
            .map(|(i, parameter)| (i.to_string(), parameter.clone()))
            .collect()
    }
}

/// Named collection of standalone parameters, not owned by any module:
/// PyTorch's `nn.ParameterDict`. See [`ModuleList`]'s own doc for the shared
/// no-forward/aggregation contract; slots here are addressable by key.
/// Duplicate keys are rejected immediately.
#[derive(Clone)]
pub struct ParameterDict {
    entries: Vec<(String, Parameter)>,
}
impl ParameterDict {
    pub fn new(entries: Vec<(String, Parameter)>) -> Result<Self> {
        for (index, (name, _)) in entries.iter().enumerate() {
            if entries[..index].iter().any(|(other, _)| other == name) {
                return Err(format!("ParameterDict contains duplicate key {name:?}").into());
            }
        }
        Ok(Self { entries })
    }
    pub fn insert(&mut self, name: impl Into<String>, parameter: Parameter) -> Result<()> {
        let name = name.into();
        if self.entries.iter().any(|(other, _)| *other == name) {
            return Err(format!("ParameterDict contains duplicate key {name:?}").into());
        }
        self.entries.push((name, parameter));
        Ok(())
    }
    pub fn get(&self, name: &str) -> Option<&Parameter> {
        self.entries
            .iter()
            .find(|(other, _)| other == name)
            .map(|(_, parameter)| parameter)
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
impl Module for ParameterDict {
    fn output_shape(&self, _input: &Shape) -> Result<Shape> {
        Err("ParameterDict has no forward of its own".into())
    }
    fn build(&mut self, _input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        Err("ParameterDict has no forward of its own".into())
    }
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err("ParameterDict has no forward of its own, matching PyTorch's nn.ParameterDict".into())
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.entries.clone()
    }
}

/// Merge a run of named axes into one, PyTorch's `nn.Flatten` expressed over
/// identities instead of a positional `start_dim`/`end_dim` range: the
/// caller names the exact axes to fold, in the physical order they should
/// flatten in, and the single `output` axis that replaces them. A thin
/// `Module` wrapper over [`Tensor::merge`]; see its own doc for the exact
/// selected-axis-order and physical-layout contract this reuses verbatim,
/// including which position the merged axis is inserted at.
#[derive(Clone)]
pub struct Flatten {
    axes: Vec<Axis>,
    output: Axis,
}
impl Flatten {
    pub fn new(axes: impl IntoIterator<Item = Axis>, output: Axis) -> Self {
        Self {
            axes: axes.into_iter().collect(),
            output,
        }
    }
}
impl Module for Flatten {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let axes = input.select_axes(self.axes.clone())?;
        if axes.is_empty() {
            return Err("Flatten requires at least one axis to merge".into());
        }
        let mut total = 1usize;
        for &axis in &axes {
            total = total
                .checked_mul(input.extent(axis)?)
                .ok_or("Flatten merged extent overflow")?;
        }
        let mut dims = Vec::with_capacity(input.rank());
        let mut inserted = false;
        for &dim in input.dims() {
            if axes.contains(&dim.axis) {
                if !inserted {
                    dims.push(self.output.of(total));
                    inserted = true;
                }
            } else {
                dims.push(dim);
            }
        }
        Shape::new(dims)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.merge(self.axes.clone(), self.output)
    }
}

/// Split one named axis into an ordered list of named axes whose extents
/// multiply back to it, PyTorch's `nn.Unflatten` over identities instead of
/// a positional dim and an unnamed size tuple. A thin `Module` wrapper over
/// [`Tensor::split`]; see its own doc for the exact validation and physical
/// layout it reuses verbatim.
#[derive(Clone)]
pub struct Unflatten {
    axis: Axis,
    dims: Vec<Dim>,
}
impl Unflatten {
    pub fn new(axis: Axis, dims: impl IntoIterator<Item = Dim>) -> Self {
        Self {
            axis,
            dims: dims.into_iter().collect(),
        }
    }
}
impl Module for Unflatten {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let index = input.index(self.axis)?;
        let parts = Shape::new(self.dims.clone())?;
        if parts.rank() == 0 || parts.len() != input.extent(self.axis)? {
            return Err("Unflatten output extents must multiply to the source extent".into());
        }
        let mut dims = input.dims().to_vec();
        dims.splice(index..=index, parts.dims().iter().copied());
        Shape::new(dims)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.split(self.axis, self.dims.clone())
    }
}

/// Parameter-free passthrough, PyTorch's `nn.Identity`: returns its input
/// unchanged.
#[derive(Clone, Copy)]
pub struct Identity;
impl Module for Identity {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(input.clone())
    }
    fn build(&mut self, input: &Shape, _: &Device, _: u64) -> Result<Shape> {
        Ok(input.clone())
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
}

/// Learned bilinear form over two named axes, `y = x1^T A x2 + b`, PyTorch's
/// `nn.Bilinear`. Composed from two [`Tensor::contract`] calls -- `x1`
/// contracted against `weight` over `in1`, then that intermediate contracted
/// against `x2` over `in2` -- the same contraction [`Linear`] already uses,
/// applied twice, so it needs no dedicated kernel. Any axis `x1` and `x2`
/// share beyond `in1`/`in2` (such as a common batch axis) aligns
/// automatically through that shared contraction, exactly like `Linear`'s
/// own unrelated axes; an axis unique to either input passes through
/// untouched.
///
/// Does not implement [`Module`], which only threads a single input tensor
/// through `forward`: call [`Bilinear::build`] then [`Bilinear::forward`]
/// directly with both operands.
///
/// Weight has logical shape `[in1, in2, output]` and starts uniform in
/// `[-scale, scale)` with `scale = sqrt(6 / (in1_extent + in2_extent +
/// output_extent))`, the same Xavier-style rule `Linear` uses generalized
/// over both inputs (Axis does not reproduce PyTorch's own default
/// `reset_parameters`, matching `Linear`'s own departure from it). Carries a
/// learned `[output]` bias, zero-initialized like `Linear`'s; call
/// `.bias(false)` before `build` to omit it entirely.
#[derive(Clone)]
pub struct Bilinear {
    in1: Axis,
    in2: Axis,
    output: Dim,
    in1_role: Axis,
    in2_role: Axis,
    output_role: Axis,
    bias: bool,
    bound: Option<(usize, usize, Parameter, Option<Parameter>)>,
}
impl Bilinear {
    pub fn new(in1: Axis, in2: Axis, output: Dim) -> Self {
        Self {
            in1,
            in2,
            output,
            in1_role: in1.role("bilinear_in1"),
            in2_role: in2.role("bilinear_in2"),
            output_role: output.axis.role("bilinear_output"),
            bias: true,
            bound: None,
        }
    }
    /// Disable the learned bias term. Has no effect once `build` has already
    /// allocated parameters; call it before `build`.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
    pub fn output_shape(&self, x1: &Shape, x2: &Shape) -> Result<Shape> {
        let extent1 = x1.extent(self.in1)?;
        let extent2 = x2.extent(self.in2)?;
        for dim in x1.dims() {
            if dim.axis != self.in1 && x2.contains(dim.axis) && x2.extent(dim.axis)? != dim.extent {
                return Err(format!("Bilinear shared axis {:?} extent mismatch", dim.axis).into());
            }
        }
        if let Some((bound1, bound2, _, _)) = &self.bound
            && (*bound1, *bound2) != (extent1, extent2)
        {
            return Err("Bilinear input extents differ from its built extents".into());
        }
        let mut dims: Vec<_> = x1
            .dims()
            .iter()
            .copied()
            .filter(|d| d.axis != self.in1)
            .collect();
        for &dim in x2.dims() {
            if dim.axis != self.in2 && !dims.iter().any(|d| d.axis == dim.axis) {
                dims.push(dim);
            }
        }
        dims.push(self.output);
        Shape::new(dims)
    }
    pub fn build(&mut self, x1: &Shape, x2: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let shape = self.output_shape(x1, x2)?;
        if let Some((_, _, weight, _)) = &self.bound {
            if !weight.tensor().device().same(device) {
                return Err("Bilinear is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let extent1 = x1.extent(self.in1)?;
        let extent2 = x2.extent(self.in2)?;
        let weight_shape = Shape::new([
            self.in1_role.of(extent1),
            self.in2_role.of(extent2),
            self.output_role.of(self.output.extent),
        ])?;
        let scale = (6.0 / (extent1 + extent2 + self.output.extent) as f32).sqrt();
        let weight = Parameter::new(Tensor::from_slice(
            &uniform_values(seed, weight_shape.len(), scale),
            weight_shape.dims().iter().copied(),
            device,
        )?);
        let bias = if self.bias {
            Some(Parameter::new(Tensor::from_slice(
                &vec![0.0; self.output.extent],
                [self.output_role.of(self.output.extent)],
                device,
            )?))
        } else {
            None
        };
        self.bound = Some((extent1, extent2, weight, bias));
        Ok(shape)
    }
    pub fn forward(&self, x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
        self.output_shape(x1.shape(), x2.shape())?;
        let (_, _, weight, bias) = self
            .bound
            .as_ref()
            .ok_or("Bilinear must be built before forward")?;
        let projected = x1
            .rename(self.in1, self.in1_role)?
            .contract(&weight.tensor(), self.in1_role)?
            .contract(&x2.rename(self.in2, self.in2_role)?, self.in2_role)?;
        let projected = match bias {
            Some(bias) => projected.add(&bias.tensor())?,
            None => projected,
        };
        projected.rename(self.output_role, self.output.axis)
    }
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, _, weight, bias)| {
                let mut params = vec![("weight".into(), weight.clone())];
                if let Some(bias) = bias {
                    params.push(("bias".into(), bias.clone()));
                }
                params
            })
            .unwrap_or_default()
    }
}

/// Pooling mode for [`EmbeddingBag`], matching PyTorch's `mode` argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingBagMode {
    Sum,
    Mean,
    Max,
}

/// Expand PyTorch's CSR-style `offsets` (bag `b` starts at `offsets[b]`) into
/// one bag id per position of a length-`total` index, exactly the shape
/// [`Tensor::scatter_add`]'s own `index` parameter wants.
fn embedding_bag_assignment(offsets: &[usize], total: usize) -> Result<Vec<usize>> {
    if offsets.is_empty() {
        return Err("EmbeddingBag requires at least one bag".into());
    }
    if offsets[0] != 0 {
        return Err("EmbeddingBag offsets must start at zero".into());
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("EmbeddingBag offsets must be non-decreasing".into());
    }
    if *offsets.last().expect("checked nonempty above") > total {
        return Err("EmbeddingBag offsets exceed the index length".into());
    }
    let mut bag = vec![0usize; total];
    for (b, &start) in offsets.iter().enumerate() {
        let end = offsets.get(b + 1).copied().unwrap_or(total);
        bag[start..end].fill(b);
    }
    Ok(bag)
}

/// One learned feature vector per vocabulary entry, pooled per bag: PyTorch's
/// `nn.EmbeddingBag`. Unlike [`Embedding`]'s dense one-hot contraction, the
/// input is a host-side flat row-index array together with `offsets` (bag
/// `b` covers positions `offsets[b]..offsets[b + 1]`, or `..index.len()` for
/// the last bag -- exactly `torch.nn.EmbeddingBag.forward`'s own `input`/
/// `offsets` pair), so it scales to a large table the same way
/// [`Tensor::gather`] itself does. Does not implement [`Module`], which only
/// threads a single *tensor* input: the index and offsets are host-side
/// integer arrays, not tensors, exactly as [`Tensor::gather`]'s own `index`
/// and [`Tensor::scatter_add`]'s own `index`/`bucket` already are.
///
/// `forward` gathers one table row per position ([`Tensor::gather`]), then
/// pools rows sharing a bag: `Sum` and `Mean` reuse [`Tensor::scatter_add`]
/// directly (`Mean` additionally divides by each bag's own item count from
/// [`Tensor::bincount`], clamped to at least one so an empty bag reads back
/// as an exact zero row rather than `0 / 0`); `Max` instead broadcasts the
/// gathered rows onto an explicit `[bag, position, feature]` cube, adds a
/// host-built additive offset that is exactly `f32::NEG_INFINITY` at every
/// `(bag, position)` pair whose position is not in that bag, and reduces
/// with [`Tensor::max`], which already ignores non-finite candidates and
/// documents the "no finite candidate" case as `NaN` with zero gradient --
/// so an empty bag's max row is honestly `NaN`, a deliberate divergence from
/// PyTorch's zero-filled empty bag for that one mode. No new kernel is
/// needed for any mode. Entries start uniform in `[-0.02, 0.02)`, matching
/// [`Embedding`].
#[derive(Clone)]
pub struct EmbeddingBag {
    vocabulary: Dim,
    feature: Dim,
    position_role: Axis,
    mode: EmbeddingBagMode,
    table: Option<Parameter>,
}
impl EmbeddingBag {
    pub fn new(vocabulary: Dim, feature: Dim, mode: EmbeddingBagMode) -> Self {
        Self {
            vocabulary,
            feature,
            position_role: vocabulary.axis.role("embedding_bag_position"),
            mode,
            table: None,
        }
    }
    pub fn build(&mut self, device: &Device, seed: u64) -> Result<()> {
        if let Some(table) = &self.table {
            if !table.tensor().device().same(device) {
                return Err("EmbeddingBag is already built on a different Device".into());
            }
            return Ok(());
        }
        let table_shape = Shape::new([self.vocabulary, self.feature])?;
        let table = Parameter::new(Tensor::from_slice(
            &uniform_values(seed, table_shape.len(), 0.02),
            table_shape.dims().iter().copied(),
            device,
        )?);
        self.table = Some(table);
        Ok(())
    }
    pub fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.table
            .as_ref()
            .map(|table| vec![("table".into(), table.clone())])
            .unwrap_or_default()
    }
    /// `index[i]` names the vocabulary row position `i` reads; `offsets`
    /// splits `index` into bags as documented on [`EmbeddingBag`] itself.
    /// `output` is the bag axis of the `[output, feature]` result.
    pub fn forward(&self, index: &[usize], offsets: &[usize], output: Axis) -> Result<Tensor> {
        let table = self
            .table
            .as_ref()
            .ok_or("EmbeddingBag must be built before forward")?;
        if index.is_empty() {
            return Err("EmbeddingBag requires a nonempty index".into());
        }
        let bag = embedding_bag_assignment(offsets, index.len())?;
        let bag_count = offsets.len();
        let device = table.tensor().device().clone();
        let rows = table
            .tensor()
            .gather(self.vocabulary.axis, index, self.position_role)?;
        match self.mode {
            EmbeddingBagMode::Sum => rows.scatter_add(self.position_role, &bag, output, bag_count),
            EmbeddingBagMode::Mean => {
                let summed = rows.scatter_add(self.position_role, &bag, output, bag_count)?;
                let counts =
                    Tensor::bincount(&bag, output, bag_count, &device)?.clamp(Some(1.0), None)?;
                summed.div(&counts)
            }
            EmbeddingBagMode::Max => {
                let mut offset_values = vec![f32::NEG_INFINITY; bag_count * index.len()];
                for (position, &b) in bag.iter().enumerate() {
                    offset_values[b * index.len() + position] = 0.0;
                }
                let offset = Tensor::from_slice(
                    &offset_values,
                    [output.of(bag_count), self.position_role.of(index.len())],
                    &device,
                )?;
                let broadcast_shape = Shape::new([
                    output.of(bag_count),
                    self.position_role.of(index.len()),
                    self.feature.axis.of(self.feature.extent),
                ])?;
                rows.broadcast_to(&broadcast_shape)?
                    .add(&offset.broadcast_to(&broadcast_shape)?)?
                    .max(self.position_role)
            }
        }
    }
}
