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
        Err("ModuleList has no forward of its own; call output_shape on a held module directly"
            .into())
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
        Err("ModuleDict has no forward of its own; call output_shape on a held module directly"
            .into())
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
            if dim.axis != self.in1 && x2.contains(dim.axis) && x2.extent(dim.axis)? != dim.extent
            {
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
            EmbeddingBagMode::Sum => {
                rows.scatter_add(self.position_role, &bag, output, bag_count)
            }
            EmbeddingBagMode::Mean => {
                let summed = rows.scatter_add(self.position_role, &bag, output, bag_count)?;
                let counts = Tensor::bincount(&bag, output, bag_count, &device)?
                    .clamp(Some(1.0), None)?;
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
