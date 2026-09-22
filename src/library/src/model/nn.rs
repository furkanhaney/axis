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

/// Parameter-free erf-form GELU for checkpoints trained with exact GELU.
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
