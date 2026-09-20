use crate::{Axis, Device, Dim, Result, Shape, Tensor, backend::Buffer};
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
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
    fn replace(&self, tensor: Tensor) {
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
/// Cloning a built Linear explicitly ties its parameters.
#[derive(Clone)]
pub struct Linear {
    input: Axis,
    output: Dim,
    input_role: Axis,
    output_role: Axis,
    bound: Option<(usize, Parameter, Parameter)>,
}
impl Linear {
    pub fn new(input: Axis, output: Dim) -> Self {
        Self {
            input,
            output,
            input_role: input.role("linear_input"),
            output_role: output.axis.role("linear_output"),
            bound: None,
        }
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
        let mut rng = seed.max(1);
        let scale = (6.0 / (extent + self.output.extent) as f32).sqrt();
        let values: Vec<_> = (0..weight_shape.len())
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (((rng >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect();
        let weight = Parameter::new(Tensor::from_slice(
            &values,
            weight_shape.dims().iter().copied(),
            device,
        )?);
        let bias = Parameter::new(Tensor::from_slice(
            &vec![0.0; self.output.extent],
            [self.output_role.of(self.output.extent)],
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
            .ok_or("Linear must be built before forward")?;
        input
            .rename(self.input, self.input_role)?
            .contract(&weight.tensor(), self.input_role)?
            .add(&bias.tensor())?
            .rename(self.output_role, self.output.axis)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, w, b)| vec![("weight".into(), w.clone()), ("bias".into(), b.clone())])
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
        let mut rng = seed.max(1);
        let scale = (6.0 / (extent + self.output.extent) as f32).sqrt();
        let values: Vec<_> = (0..weight_shape.len())
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (((rng >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect();
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

/// Valid, stride-one 2D cross-correlation followed by bias addition.
pub struct Conv2d {
    input: Axis,
    spatial: [Axis; 2],
    kernel: [usize; 2],
    patch: Axis,
    linear: Linear,
}
impl Conv2d {
    pub fn new(input: Axis, output: Dim, spatial: [Axis; 2], kernel: [usize; 2]) -> Self {
        let patch = input.role("conv_patch");
        Self {
            input,
            spatial,
            kernel,
            patch,
            linear: Linear::new(patch, output),
        }
    }
    fn patch_shape(&self, input: &Shape) -> Result<Shape> {
        let channels = input.extent(self.input)?;
        let height = input.extent(self.spatial[0])?;
        let width = input.extent(self.spatial[1])?;
        if self.kernel.contains(&0) || self.kernel[0] > height || self.kernel[1] > width {
            return Err("Conv2d kernel must be positive and fit both spatial axes".into());
        }
        let patch = channels
            .checked_mul(self.kernel[0])
            .and_then(|n| n.checked_mul(self.kernel[1]))
            .ok_or("Conv2d patch extent overflow")?;
        Shape::new(input.dims().iter().map(|dim| {
            if dim.axis == self.input {
                self.patch.of(patch)
            } else if dim.axis == self.spatial[0] {
                self.spatial[0].of(height - self.kernel[0] + 1)
            } else if dim.axis == self.spatial[1] {
                self.spatial[1].of(width - self.kernel[1] + 1)
            } else {
                *dim
            }
        }))
    }
}
impl Module for Conv2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.linear.output_shape(&self.patch_shape(input)?)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.linear.build(&self.patch_shape(input)?, device, seed)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let patch_extent = input
            .extent(self.input)?
            .checked_mul(self.kernel[0])
            .and_then(|n| n.checked_mul(self.kernel[1]))
            .ok_or("Conv2d patch extent overflow")?;
        self.linear.forward(&input.unfold2d(
            self.input,
            self.spatial,
            self.patch.of(patch_extent),
            self.kernel,
        )?)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.linear.named_parameters()
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

pub struct SGD {
    learning_rate: f32,
}

/// Device-resident Adam with bias correction and one state pair per ParamId.
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: i32,
    states: HashMap<ParamId, (Buffer, Buffer)>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Result<Self> {
        Self::with_hyperparameters(learning_rate, 0.9, 0.999, 1e-8)
    }

    pub fn with_hyperparameters(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("Adam learning rate must be finite and positive".into());
        }
        if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
            return Err("Adam beta1 must be finite and within 0..1".into());
        }
        if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
            return Err("Adam beta2 must be finite and within 0..1".into());
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("Adam epsilon must be finite and positive".into());
        }
        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            step: 0,
            states: HashMap::new(),
        })
    }

    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        self.step_parameters(model.parameters())
    }

    pub fn step_parameters(
        &mut self,
        parameters: impl IntoIterator<Item = Parameter>,
    ) -> Result<()> {
        let next_step = self.step.checked_add(1).ok_or("Adam step overflow")?;
        let correction1 = 1.0 - self.beta1.powi(next_step);
        let correction2 = 1.0 - self.beta2.powi(next_step);
        let mut seen = HashSet::new();
        let mut updates = vec![];
        for parameter in parameters {
            let id = parameter.id();
            if seen.insert(id) {
                let state = self.states.get(&id);
                let (tensor, first, second) = parameter.tensor().adam_updated(
                    state.map(|state| &state.0),
                    state.map(|state| &state.1),
                    self.learning_rate,
                    self.beta1,
                    self.beta2,
                    correction1,
                    correction2,
                    self.epsilon,
                    self.weight_decay,
                )?;
                updates.push((parameter, id, tensor, first, second));
            }
        }
        for (parameter, id, tensor, first, second) in updates {
            parameter.replace(tensor);
            self.states.insert(id, (first, second));
        }
        self.step = next_step;
        Ok(())
    }

    pub fn completed_steps(&self) -> i32 {
        self.step
    }
}

/// Adam with decoupled weight decay; moments and updates remain device-resident.
pub struct AdamW(Adam);

impl AdamW {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Result<Self> {
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err("AdamW weight decay must be finite and nonnegative".into());
        }
        let mut adam = Adam::new(learning_rate)?;
        adam.weight_decay = weight_decay;
        Ok(Self(adam))
    }

    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        self.0.step(model)
    }

    pub fn step_parameters(
        &mut self,
        parameters: impl IntoIterator<Item = Parameter>,
    ) -> Result<()> {
        self.0.step_parameters(parameters)
    }

    pub fn completed_steps(&self) -> i32 {
        self.0.completed_steps()
    }
}
impl SGD {
    pub fn new(learning_rate: f32) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        Ok(Self { learning_rate })
    }
    pub fn step(&mut self, model: &mut impl Module) -> Result<()> {
        self.step_parameters(model.parameters())
    }
    /// All uses accumulate into one leaf; each shared ParamId is updated exactly once.
    pub fn step_parameters(
        &mut self,
        parameters: impl IntoIterator<Item = Parameter>,
    ) -> Result<()> {
        let mut seen = HashSet::new();
        let mut updates = vec![];
        for parameter in parameters {
            if seen.insert(parameter.id()) {
                let new = parameter.tensor().updated(self.learning_rate)?;
                updates.push((parameter, new));
            }
        }
        for (parameter, tensor) in updates {
            parameter.replace(tensor);
        }
        Ok(())
    }
}
