//! Named-axis normalization modules with no running state.
use crate::{Axis, Device, Dim, IntoAxes, Module, Parameter, Result, Shape, Tensor};

const DEFAULT_EPSILON: f32 = 1e-5;
const DEFAULT_RMS_EPSILON: f32 = 1e-6;

fn declared_axes(axes: impl IntoAxes, owner: &str, allow_empty: bool) -> Result<Vec<Axis>> {
    let axes = axes.into_axes();
    if !allow_empty && axes.is_empty() {
        return Err(format!("{owner} requires at least one normalized axis").into());
    }
    for (index, axis) in axes.iter().enumerate() {
        if axes[..index].contains(axis) {
            return Err(format!("{owner} contains duplicate axis {axis:?}").into());
        }
    }
    Ok(axes)
}

fn validate_epsilon(owner: &str, epsilon: f32) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(format!("{owner} epsilon must be finite and positive").into());
    }
    Ok(())
}

fn selected_dims(input: &Shape, axes: &[Axis]) -> Result<Vec<Dim>> {
    axes.iter()
        .map(|&axis| Ok(axis.of(input.extent(axis)?)))
        .collect()
}

fn centered_normalize(input: &Tensor, axes: &[Axis], epsilon: f32) -> Result<Tensor> {
    let (mean, variance) = input.moments(axes)?;
    input.sub(&mean)?.mul(&variance.inverse_sqrt(epsilon)?)
}

fn rms_normalize(input: &Tensor, axes: &[Axis], epsilon: f32) -> Result<Tensor> {
    input.mul(&input.mean_square(axes)?.inverse_sqrt(epsilon)?)
}

#[derive(Clone)]
struct Affine {
    scale: Parameter,
    bias: Option<Parameter>,
}

impl Affine {
    fn new(dims: &[Dim], bias: bool, device: &Device) -> Result<Self> {
        let shape = Shape::new(dims.iter().copied())?;
        let scale = Parameter::new(Tensor::from_slice(
            &vec![1.0; shape.len()],
            dims.iter().copied(),
            device,
        )?);
        let bias = bias
            .then(|| {
                Tensor::from_slice(&vec![0.0; shape.len()], dims.iter().copied(), device)
                    .map(Parameter::new)
            })
            .transpose()?;
        Ok(Self { scale, bias })
    }

    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        let scaled = input.mul(&self.scale.tensor())?;
        match &self.bias {
            Some(bias) => scaled.add(&bias.tensor()),
            None => Ok(scaled),
        }
    }

    fn parameters(&self) -> Vec<(String, Parameter)> {
        let mut parameters = vec![("scale".into(), self.scale.clone())];
        if let Some(bias) = &self.bias {
            parameters.push(("bias".into(), bias.clone()));
        }
        parameters
    }
}

#[derive(Clone)]
struct AxisNormBound {
    dims: Vec<Dim>,
    affine: Option<Affine>,
}

/// Population layer normalization over one or more declared named axes.
///
/// `Clone` is provided so a caller (e.g. `TransformerEncoderLayer`) can
/// build one configured, unbuilt instance and clone it before `build`; a
/// clone of an already-built instance instead shares the original's
/// `Parameter`s (ties weights), matching every other `Clone` module in the
/// crate.
#[derive(Clone)]
pub struct LayerNorm {
    axes: Vec<Axis>,
    epsilon: f32,
    affine: bool,
    bound: Option<AxisNormBound>,
}

impl LayerNorm {
    /// Defaults to epsilon `1e-5` and learnable scale and bias.
    pub fn new(axes: impl IntoAxes) -> Result<Self> {
        Ok(Self {
            axes: declared_axes(axes, "LayerNorm", false)?,
            epsilon: DEFAULT_EPSILON,
            affine: true,
            bound: None,
        })
    }

    pub fn epsilon(mut self, epsilon: f32) -> Result<Self> {
        validate_epsilon("LayerNorm", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }
}

impl Module for LayerNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let dims = selected_dims(input, &self.axes)?;
        if let Some(bound) = &self.bound {
            if bound.dims != dims {
                return Err("LayerNorm normalized extents differ from its built extents".into());
            }
            if bound.affine.is_some() != self.affine {
                return Err("LayerNorm affine configuration changed after build".into());
            }
        }
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, device: &Device, _: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some(bound) = &self.bound {
            if let Some(affine) = &bound.affine
                && !affine.scale.tensor().device().same(device)
            {
                return Err("LayerNorm is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let dims = selected_dims(input, &self.axes)?;
        let affine = self
            .affine
            .then(|| Affine::new(&dims, true, device))
            .transpose()?;
        self.bound = Some(AxisNormBound { dims, affine });
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or("LayerNorm must be built before forward")?;
        let normalized = centered_normalize(input, &self.axes, self.epsilon)?;
        match &bound.affine {
            Some(affine) => affine.apply(&normalized),
            None => Ok(normalized),
        }
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .and_then(|bound| bound.affine.as_ref())
            .map(Affine::parameters)
            .unwrap_or_default()
    }
}

/// Root-mean-square normalization over one or more declared named axes.
pub struct RmsNorm {
    axes: Vec<Axis>,
    epsilon: f32,
    affine: bool,
    bound: Option<AxisNormBound>,
}

impl RmsNorm {
    /// Defaults to epsilon `1e-6` and a learnable scale without bias.
    pub fn new(axes: impl IntoAxes) -> Result<Self> {
        Ok(Self {
            axes: declared_axes(axes, "RmsNorm", false)?,
            epsilon: DEFAULT_RMS_EPSILON,
            affine: true,
            bound: None,
        })
    }

    pub fn epsilon(mut self, epsilon: f32) -> Result<Self> {
        validate_epsilon("RmsNorm", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }
}

impl Module for RmsNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let dims = selected_dims(input, &self.axes)?;
        if let Some(bound) = &self.bound {
            if bound.dims != dims {
                return Err("RmsNorm normalized extents differ from its built extents".into());
            }
            if bound.affine.is_some() != self.affine {
                return Err("RmsNorm affine configuration changed after build".into());
            }
        }
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, device: &Device, _: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some(bound) = &self.bound {
            if let Some(affine) = &bound.affine
                && !affine.scale.tensor().device().same(device)
            {
                return Err("RmsNorm is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let dims = selected_dims(input, &self.axes)?;
        let affine = self
            .affine
            .then(|| Affine::new(&dims, false, device))
            .transpose()?;
        self.bound = Some(AxisNormBound { dims, affine });
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or("RmsNorm must be built before forward")?;
        let normalized = rms_normalize(input, &self.axes, self.epsilon)?;
        match &bound.affine {
            Some(affine) => affine.apply(&normalized),
            None => Ok(normalized),
        }
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .and_then(|bound| bound.affine.as_ref())
            .map(Affine::parameters)
            .unwrap_or_default()
    }
}

enum GroupCount {
    Fixed(usize),
    PerChannel,
}

struct ChannelNormBound {
    channels: usize,
    groups: usize,
    affine: Option<Affine>,
}

struct ChannelNorm {
    owner: &'static str,
    channel: Axis,
    groups: GroupCount,
    sample_axes: Vec<Axis>,
    group: Axis,
    within_group: Axis,
    epsilon: f32,
    affine: bool,
    bound: Option<ChannelNormBound>,
}

impl ChannelNorm {
    fn new(
        owner: &'static str,
        channel: Axis,
        groups: GroupCount,
        sample_axes: impl IntoAxes,
        affine: bool,
        require_samples: bool,
    ) -> Result<Self> {
        let sample_axes = declared_axes(sample_axes, owner, !require_samples)?;
        if sample_axes.contains(&channel) {
            return Err(format!("{owner} channel cannot also be a sample axis").into());
        }
        Ok(Self {
            owner,
            channel,
            groups,
            sample_axes,
            group: channel.role("norm_group"),
            within_group: channel.role("norm_channel_in_group"),
            epsilon: DEFAULT_EPSILON,
            affine,
            bound: None,
        })
    }

    fn group_count(&self, channels: usize) -> Result<usize> {
        let groups = match self.groups {
            GroupCount::Fixed(groups) => groups,
            GroupCount::PerChannel => channels,
        };
        if groups == 0 {
            return Err(format!("{} groups must be positive", self.owner).into());
        }
        if !channels.is_multiple_of(groups) {
            return Err(format!("{} channels must be divisible by groups", self.owner).into());
        }
        Ok(groups)
    }

    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let channels = input.extent(self.channel)?;
        let groups = self.group_count(channels)?;
        for &axis in &self.sample_axes {
            input.extent(axis)?;
        }
        if let Some(bound) = &self.bound
            && (bound.channels != channels || bound.groups != groups)
        {
            return Err(format!(
                "{} channel geometry differs from its built geometry",
                self.owner
            )
            .into());
        }
        if let Some(bound) = &self.bound
            && bound.affine.is_some() != self.affine
        {
            return Err(format!("{} affine configuration changed after build", self.owner).into());
        }
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, device: &Device) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some(bound) = &self.bound {
            if let Some(affine) = &bound.affine
                && !affine.scale.tensor().device().same(device)
            {
                return Err(
                    format!("{} is already built on a different Device", self.owner).into(),
                );
            }
            return Ok(shape);
        }
        let channels = input.extent(self.channel)?;
        let groups = self.group_count(channels)?;
        let affine = self
            .affine
            .then(|| Affine::new(&[self.channel.of(channels)], true, device))
            .transpose()?;
        self.bound = Some(ChannelNormBound {
            channels,
            groups,
            affine,
        });
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or_else(|| format!("{} must be built before forward", self.owner))?;
        let channels_per_group = bound.channels / bound.groups;
        let grouped = input.split(
            self.channel,
            [
                self.group.of(bound.groups),
                self.within_group.of(channels_per_group),
            ],
        )?;
        let mut reduced = self.sample_axes.clone();
        reduced.push(self.within_group);
        let normalized = centered_normalize(&grouped, &reduced, self.epsilon)?
            .merge([self.group, self.within_group], self.channel)?;
        match &bound.affine {
            Some(affine) => affine.apply(&normalized),
            None => Ok(normalized),
        }
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .and_then(|bound| bound.affine.as_ref())
            .map(Affine::parameters)
            .unwrap_or_default()
    }
}

/// Group normalization over channel groups and explicitly named sample axes.
pub struct GroupNorm(ChannelNorm);

impl GroupNorm {
    /// Defaults to epsilon `1e-5` and per-channel learnable scale and bias.
    pub fn new(channel: Axis, groups: usize, sample_axes: impl IntoAxes) -> Result<Self> {
        if groups == 0 {
            return Err("GroupNorm groups must be positive".into());
        }
        Ok(Self(ChannelNorm::new(
            "GroupNorm",
            channel,
            GroupCount::Fixed(groups),
            sample_axes,
            true,
            false,
        )?))
    }

    pub fn epsilon(mut self, epsilon: f32) -> Result<Self> {
        validate_epsilon("GroupNorm", epsilon)?;
        self.0.epsilon = epsilon;
        Ok(self)
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.0.affine = affine;
        self
    }
}

impl Module for GroupNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, _: u64) -> Result<Shape> {
        self.0.build(input, device)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.0.named_parameters()
    }
}

/// Per-channel instance normalization over explicitly named sample axes.
pub struct InstanceNorm(ChannelNorm);

impl InstanceNorm {
    /// Defaults to epsilon `1e-5` with no affine parameters or running state.
    pub fn new(channel: Axis, sample_axes: impl IntoAxes) -> Result<Self> {
        Ok(Self(ChannelNorm::new(
            "InstanceNorm",
            channel,
            GroupCount::PerChannel,
            sample_axes,
            false,
            true,
        )?))
    }

    pub fn epsilon(mut self, epsilon: f32) -> Result<Self> {
        validate_epsilon("InstanceNorm", epsilon)?;
        self.0.epsilon = epsilon;
        Ok(self)
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.0.affine = affine;
        self
    }
}

impl Module for InstanceNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }

    fn build(&mut self, input: &Shape, device: &Device, _: u64) -> Result<Shape> {
        self.0.build(input, device)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.0.named_parameters()
    }
}

fn validate_finite(owner: &str, field: &str, value: f32) -> Result<()> {
    if !value.is_finite() {
        return Err(format!("{owner} {field} must be finite").into());
    }
    Ok(())
}

/// Response normalization across a fixed-size sliding window of a named
/// channel axis, PyTorch's `nn.LocalResponseNorm`: each element divides by
/// `(k + alpha / size * sum(a_c'^2))^beta`, where the sum runs over a window
/// of `size` channels around `c`, padded with implicit zeros at the
/// channel-axis boundary (`size / 2` channels before `c`, `(size - 1) / 2`
/// after, PyTorch's own asymmetric split for an even `size`). No learned
/// parameters, matching PyTorch's own stateless module.
///
/// Composed entirely from existing ops: `mul` (square), `pad_zeros`,
/// `size - 1` pairs of `narrow`/`add` for the sliding sum, `scale` for the
/// mean and the `alpha` factor, and `exp(beta * ln(x))` for the fractional
/// power -- no dedicated kernel. The power composition inherits `Tensor::ln`'s
/// ordinary IEEE domain: a `k`/`alpha` choice that drives the base
/// non-positive produces the same `-inf`/`NaN` propagation `Tensor::ln`
/// documents, rather than a clamp.
#[derive(Clone, Copy)]
pub struct LocalResponseNorm {
    channel: Axis,
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
}

impl LocalResponseNorm {
    /// Defaults to `alpha = 1e-4`, `beta = 0.75`, `k = 1.0`, matching PyTorch.
    pub fn new(channel: Axis, size: usize) -> Result<Self> {
        if size == 0 {
            return Err("LocalResponseNorm size must be positive".into());
        }
        Ok(Self {
            channel,
            size,
            alpha: 1e-4,
            beta: 0.75,
            k: 1.0,
        })
    }

    pub fn alpha(mut self, alpha: f32) -> Result<Self> {
        validate_finite("LocalResponseNorm", "alpha", alpha)?;
        self.alpha = alpha;
        Ok(self)
    }

    pub fn beta(mut self, beta: f32) -> Result<Self> {
        validate_finite("LocalResponseNorm", "beta", beta)?;
        self.beta = beta;
        Ok(self)
    }

    pub fn k(mut self, k: f32) -> Result<Self> {
        validate_finite("LocalResponseNorm", "k", k)?;
        self.k = k;
        Ok(self)
    }
}

impl Module for LocalResponseNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.channel)?;
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let extent = input.extent(self.channel)?;
        let before = self.size / 2;
        let after = (self.size - 1) / 2;
        let padded = input.mul(input)?.pad_zeros(self.channel, before, after)?;
        let mut window = padded.narrow(self.channel, 0, extent)?;
        for offset in 1..self.size {
            window = window.add(&padded.narrow(self.channel, offset, extent)?)?;
        }
        let mean_square = window.scale(1.0 / self.size as f32)?;
        let base = mean_square.scale(self.alpha)?.add(&Tensor::from_slice(
            &[self.k],
            [],
            input.device(),
        )?)?;
        let denominator = base.ln()?.scale(self.beta)?.exp()?;
        input.div(&denominator)
    }
}
