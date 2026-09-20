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

/// Named-channel 2D cross-correlation followed by bias addition.
///
/// Stride defaults to `[1, 1]`, padding to `[0, 0]`, and groups to `1`.
/// A depthwise convolution sets groups equal to both the input and output
/// channel extents. Each spatial output extent is
/// `floor((input + 2 * padding - kernel) / stride) + 1`.
///
/// Padding is symmetric and may produce windows containing only zeros.
/// Dilation and asymmetric padding are not currently supported. The cuTile
/// backend computes patch indices from compact geometry and uses deterministic
/// input-centric col2im for its derivative. It still materializes the patch
/// tensor before tiled grouped contraction, so this is a correctness path
/// rather than a fused or throughput-competitive convolution kernel.
pub struct Conv2d {
    input: Axis,
    output: Dim,
    spatial: [Axis; 2],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    groups: usize,
    group: Axis,
    patch: Axis,
    output_in_group: Axis,
    output_role: Axis,
    bound: Option<BoundConv2d>,
}

struct BoundConv2d {
    input_channels: usize,
    groups: usize,
    patch: usize,
    output_in_group: usize,
    weight: Parameter,
    bias: Parameter,
}
impl Conv2d {
    pub fn new(input: Axis, output: Dim, spatial: [Axis; 2], kernel: [usize; 2]) -> Self {
        Self {
            input,
            output,
            spatial,
            kernel,
            stride: [1, 1],
            padding: [0, 0],
            groups: 1,
            group: input.role("conv_group"),
            patch: input.role("conv_patch_in_group"),
            output_in_group: output.axis.role("conv_output_in_group"),
            output_role: output.axis.role("conv_output"),
            bound: None,
        }
    }

    /// Set vertical and horizontal stride. Zero is rejected by shape validation.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.stride = stride;
        self
    }

    /// Set symmetric vertical and horizontal zero-padding.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.padding = padding;
        self
    }

    /// Partition input and output channels into independent convolution groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    fn geometry(&self, input: &Shape) -> Result<(Shape, usize, usize, usize)> {
        if self.spatial[0] == self.spatial[1]
            || self.input == self.spatial[0]
            || self.input == self.spatial[1]
        {
            return Err("Conv2d requires distinct input-channel and spatial axes".into());
        }
        let channels = input.extent(self.input)?;
        let height = input.extent(self.spatial[0])?;
        let width = input.extent(self.spatial[1])?;
        if self.kernel.contains(&0) {
            return Err("Conv2d kernel extents must be positive".into());
        }
        if self.stride.contains(&0) {
            return Err("Conv2d stride extents must be positive".into());
        }
        if self.groups == 0 {
            return Err("Conv2d groups must be positive".into());
        }
        if !channels.is_multiple_of(self.groups) {
            return Err("Conv2d input channels must be divisible by groups".into());
        }
        if !self.output.extent.is_multiple_of(self.groups) {
            return Err("Conv2d output channels must be divisible by groups".into());
        }
        let padded_height = height
            .checked_add(
                self.padding[0]
                    .checked_mul(2)
                    .ok_or("Conv2d padding overflow")?,
            )
            .ok_or("Conv2d padded height overflow")?;
        let padded_width = width
            .checked_add(
                self.padding[1]
                    .checked_mul(2)
                    .ok_or("Conv2d padding overflow")?,
            )
            .ok_or("Conv2d padded width overflow")?;
        if self.kernel[0] > padded_height || self.kernel[1] > padded_width {
            return Err("Conv2d kernel must fit the padded spatial axes".into());
        }
        let output_height = (padded_height - self.kernel[0]) / self.stride[0] + 1;
        let output_width = (padded_width - self.kernel[1]) / self.stride[1] + 1;
        let channels_per_group = channels / self.groups;
        let patch = channels_per_group
            .checked_mul(self.kernel[0])
            .and_then(|n| n.checked_mul(self.kernel[1]))
            .ok_or("Conv2d patch extent overflow")?;
        let output_in_group = self.output.extent / self.groups;
        if let Some(bound) = &self.bound {
            if bound.input_channels != channels {
                return Err("Conv2d input channels differ from its built extent".into());
            }
            if bound.groups != self.groups
                || bound.patch != patch
                || bound.output_in_group != output_in_group
            {
                return Err("Conv2d group geometry differs from its built parameters".into());
            }
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .filter_map(|dim| {
                if dim.axis == self.input {
                    None
                } else if dim.axis == self.spatial[0] {
                    Some(self.spatial[0].of(output_height))
                } else if dim.axis == self.spatial[1] {
                    Some(self.spatial[1].of(output_width))
                } else {
                    Some(*dim)
                }
            })
            .collect();
        dims.push(self.output);
        Ok((Shape::new(dims)?, channels, patch, output_in_group))
    }
}
impl Module for Conv2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(self.geometry(input)?.0)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let (shape, channels, patch, output_in_group) = self.geometry(input)?;
        if let Some(bound) = &self.bound {
            if !bound.weight.tensor().device().same(device) {
                return Err("Conv2d is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let weight_shape = Shape::new([
            self.group.of(self.groups),
            self.patch.of(patch),
            self.output_in_group.of(output_in_group),
        ])?;
        let mut rng = seed.max(1);
        let scale = (6.0 / (patch + output_in_group) as f32).sqrt();
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
        self.bound = Some(BoundConv2d {
            input_channels: channels,
            groups: self.groups,
            patch,
            output_in_group,
            weight,
            bias,
        });
        Ok(shape)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, _, patch, _) = self.geometry(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or("Conv2d must be built before forward")?;
        input
            .unfold2d_grouped(
                self.input,
                self.spatial,
                self.group.of(self.groups),
                self.patch.of(patch),
                self.kernel,
                self.stride,
                self.padding,
            )?
            .contract(&bound.weight.tensor(), self.patch)?
            .merge([self.group, self.output_in_group], self.output_role)?
            .add(&bound.bias.tensor())?
            .rename(self.output_role, self.output.axis)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|bound| {
                vec![
                    ("weight".into(), bound.weight.clone()),
                    ("bias".into(), bound.bias.clone()),
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

/// Affine layer normalization over one named feature axis.
pub struct LayerNorm {
    feature: Axis,
    epsilon: f32,
    bound: Option<(usize, Parameter, Parameter)>,
}

impl LayerNorm {
    pub fn new(feature: Axis, epsilon: f32) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("LayerNorm epsilon must be finite and positive".into());
        }
        Ok(Self {
            feature,
            epsilon,
            bound: None,
        })
    }
}

impl Module for LayerNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.feature)?;
        if let Some((bound, _, _)) = &self.bound
            && *bound != extent
        {
            return Err("LayerNorm feature extent differs from its built extent".into());
        }
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, device: &Device, _: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some((_, scale, _)) = &self.bound {
            if !scale.tensor().device().same(device) {
                return Err("LayerNorm is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let extent = input.extent(self.feature)?;
        let scale = Parameter::new(Tensor::from_slice(
            &vec![1.0; extent],
            [self.feature.of(extent)],
            device,
        )?);
        let bias = Parameter::new(Tensor::from_slice(
            &vec![0.0; extent],
            [self.feature.of(extent)],
            device,
        )?);
        self.bound = Some((extent, scale, bias));
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let (_, scale, bias) = self
            .bound
            .as_ref()
            .ok_or("LayerNorm must be built before forward")?;
        let centered = input.sub(&input.mean(self.feature)?)?;
        let inverse_std = centered
            .mul(&centered)?
            .mean(self.feature)?
            .inverse_sqrt(self.epsilon)?;
        centered
            .mul(&inverse_std)?
            .mul(&scale.tensor())?
            .add(&bias.tensor())
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|(_, scale, bias)| {
                vec![
                    ("scale".into(), scale.clone()),
                    ("bias".into(), bias.clone()),
                ]
            })
            .unwrap_or_default()
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
