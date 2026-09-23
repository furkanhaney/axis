//! Named-axis normalization modules, with and without persistent running state.
use crate::{
    Axis, Device, Dim, IntoAxes, Module, Parameter, Result, Shape, State, Tensor, TrainingPass,
};

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

/// `None` selects PyTorch's cumulative moving average; `Some(m)` must be
/// finite and in `[0, 1]`.
fn validate_momentum(owner: &str, momentum: Option<f32>) -> Result<()> {
    match momentum {
        Some(m) if !m.is_finite() || !(0.0..=1.0).contains(&m) => {
            Err(format!("{owner} momentum must be finite and in [0, 1]").into())
        }
        _ => Ok(()),
    }
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

/// Persistent per-feature running statistics shared by [`BatchNorm`] and
/// [`InstanceNorm`]'s optional tracking: PyTorch's `running_mean`,
/// `running_var`, and `num_batches_tracked`, one entry per feature. Neither
/// caller writes a `State` directly -- both stage an update through
/// [`RunningStats::stage_update`], and only [`crate::TrainingPass::commit`]
/// (called by `Trainer::step_training` after the optimizer step succeeds)
/// actually writes it.
struct RunningStats {
    extent: usize,
    mean: State,
    var: State,
    num_batches_tracked: State,
}

impl RunningStats {
    /// `running_mean` starts at zero, `running_var` at one, and
    /// `num_batches_tracked` at zero, matching PyTorch's own initial buffers.
    fn new(feature: Dim, device: &Device) -> Result<Self> {
        Ok(Self {
            extent: feature.extent,
            mean: State::new(Tensor::from_slice(
                &vec![0.0; feature.extent],
                [feature],
                device,
            )?),
            var: State::new(Tensor::from_slice(
                &vec![1.0; feature.extent],
                [feature],
                device,
            )?),
            num_batches_tracked: State::new(Tensor::from_slice(&[0.0], [], device)?),
        })
    }

    /// Evaluation-mode normalization from the committed running statistics:
    /// `(x - running_mean) * rsqrt(running_var + epsilon)`, broadcasting the
    /// feature-shaped statistics against every other axis in `input`.
    fn eval_normalize(&self, input: &Tensor, epsilon: f32) -> Result<Tensor> {
        input
            .sub(&self.mean.tensor())?
            .mul(&self.var.tensor().inverse_sqrt(epsilon)?)
    }

    /// Stage this step's EMA update from `pooled_mean`/`pooled_var` (already
    /// reduced to one value per feature by the caller): `running = (1 -
    /// factor) * running + factor * pooled`, where `factor` is `momentum`
    /// when `Some`, or `1 / num_batches_tracked` (after incrementing) for
    /// PyTorch's `momentum = None` cumulative moving average.
    /// `num_batches_tracked` itself is always staged as a plain increment,
    /// regardless of `momentum`. Detaches every staged value: staging never
    /// keeps this step's forward graph alive past the call.
    fn stage_update(
        &self,
        pass: &mut TrainingPass,
        pooled_mean: Tensor,
        pooled_var: Tensor,
        momentum: Option<f32>,
        device: &Device,
    ) -> Result<()> {
        let one = Tensor::from_slice(&[1.0], [], device)?;
        let new_count = self.num_batches_tracked.tensor().add(&one)?;
        let (new_mean, new_var) = match momentum {
            Some(m) => (
                self.mean
                    .tensor()
                    .scale(1.0 - m)?
                    .add(&pooled_mean.scale(m)?)?,
                self.var
                    .tensor()
                    .scale(1.0 - m)?
                    .add(&pooled_var.scale(m)?)?,
            ),
            None => {
                let factor = one.div(&new_count)?;
                let keep = one.sub(&factor)?;
                (
                    self.mean
                        .tensor()
                        .mul(&keep)?
                        .add(&pooled_mean.mul(&factor)?)?,
                    self.var
                        .tensor()
                        .mul(&keep)?
                        .add(&pooled_var.mul(&factor)?)?,
                )
            }
        };
        pass.stage(self.mean.clone(), new_mean.detach());
        pass.stage(self.var.clone(), new_var.detach());
        pass.stage(self.num_batches_tracked.clone(), new_count.detach());
        Ok(())
    }

    fn named_states(&self) -> Vec<(String, State)> {
        vec![
            ("running_mean".into(), self.mean.clone()),
            ("running_var".into(), self.var.clone()),
            (
                "num_batches_tracked".into(),
                self.num_batches_tracked.clone(),
            ),
        ]
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

/// Every axis of `input` other than `feature`, in whatever order `input`
/// happens to carry them. Errors first if `feature` itself is missing, the
/// same contract `output_shape` checks.
fn non_feature_axes(input: &Shape, feature: Axis) -> Result<Vec<Axis>> {
    input.extent(feature)?;
    Ok(input
        .axes()
        .into_iter()
        .filter(|&axis| axis != feature)
        .collect())
}

/// Batch normalization over one explicitly named feature axis, PyTorch's
/// `BatchNorm1d`/`BatchNorm2d`/`BatchNorm3d` in a single named-axis module:
/// every OTHER axis present at call time -- batch and any spatial axes --
/// normalizes together, whatever it is named. Defaults to epsilon `1e-5`,
/// momentum `Some(0.1)`, a learnable per-feature affine scale and bias, and
/// `track_running_stats = true`, matching PyTorch's own defaults.
///
/// `forward_training` normalizes with this call's own batch statistics
/// (BIASED variance, PyTorch's normalization convention) and, when tracking,
/// stages `running = (1 - momentum) * running + momentum * batch` using the
/// UNBIASED batch variance for `running_var` (`n / (n - 1)` against this
/// call's own biased variance, `n` the element count reduced -- exactly `1 /
/// 0` when `n == 1`, propagated rather than special-cased, matching how the
/// rest of Axis lets domain edges through). `momentum = None` selects
/// PyTorch's cumulative moving average (`1 / num_batches_tracked` after
/// incrementing, in place of a fixed factor) instead; `num_batches_tracked`
/// is staged as a plain increment either way. `forward` (evaluation)
/// normalizes with the committed running statistics when tracking is on, or
/// recomputes this call's own batch statistics when `track_running_stats =
/// false`, exactly matching PyTorch's behavior for that flag. See
/// [`crate::State`] and [`crate::TrainingPass`]: nothing here ever writes a
/// running statistic outside [`crate::TrainingPass::commit`].
pub struct BatchNorm {
    feature: Axis,
    epsilon: f32,
    momentum: Option<f32>,
    affine: bool,
    track_running_stats: bool,
    bound: Option<BatchNormBound>,
}

struct BatchNormBound {
    extent: usize,
    affine: Option<Affine>,
    running: Option<RunningStats>,
}

impl BatchNorm {
    pub fn new(feature: Axis) -> Self {
        Self {
            feature,
            epsilon: DEFAULT_EPSILON,
            momentum: Some(0.1),
            affine: true,
            track_running_stats: true,
            bound: None,
        }
    }

    pub fn epsilon(mut self, epsilon: f32) -> Result<Self> {
        validate_epsilon("BatchNorm", epsilon)?;
        self.epsilon = epsilon;
        Ok(self)
    }

    /// `None` selects PyTorch's cumulative moving average in place of a
    /// fixed exponential factor.
    pub fn momentum(mut self, momentum: Option<f32>) -> Result<Self> {
        validate_momentum("BatchNorm", momentum)?;
        self.momentum = momentum;
        Ok(self)
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }

    /// `false` disables running-statistics tracking entirely: no `State` is
    /// allocated, and both `forward` and `forward_training` normalize with
    /// this call's own batch statistics, matching PyTorch.
    pub fn track_running_stats(mut self, track: bool) -> Self {
        self.track_running_stats = track;
        self
    }
}

impl Module for BatchNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let extent = input.extent(self.feature)?;
        if let Some(bound) = &self.bound {
            if bound.extent != extent {
                return Err("BatchNorm feature extent differs from its built extent".into());
            }
            if bound.affine.is_some() != self.affine {
                return Err("BatchNorm affine configuration changed after build".into());
            }
            if bound.running.is_some() != self.track_running_stats {
                return Err("BatchNorm track_running_stats changed after build".into());
            }
        }
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, device: &Device, _seed: u64) -> Result<Shape> {
        let shape = self.output_shape(input)?;
        if let Some(bound) = &self.bound {
            let existing_device = bound
                .affine
                .as_ref()
                .map(|affine| affine.scale.tensor().device().clone())
                .or_else(|| {
                    bound
                        .running
                        .as_ref()
                        .map(|r| r.mean.tensor().device().clone())
                });
            if let Some(existing) = existing_device
                && !existing.same(device)
            {
                return Err("BatchNorm is already built on a different Device".into());
            }
            return Ok(shape);
        }
        let extent = input.extent(self.feature)?;
        let dim = self.feature.of(extent);
        let affine = self
            .affine
            .then(|| Affine::new(&[dim], true, device))
            .transpose()?;
        let running = self
            .track_running_stats
            .then(|| RunningStats::new(dim, device))
            .transpose()?;
        self.bound = Some(BatchNormBound {
            extent,
            affine,
            running,
        });
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or("BatchNorm must be built before forward")?;
        let normalized = match &bound.running {
            Some(running) => running.eval_normalize(input, self.epsilon)?,
            None => {
                let reduce = non_feature_axes(input.shape(), self.feature)?;
                centered_normalize(input, &reduce, self.epsilon)?
            }
        };
        match &bound.affine {
            Some(affine) => affine.apply(&normalized),
            None => Ok(normalized),
        }
    }

    fn forward_training(&self, input: &Tensor, pass: &mut TrainingPass) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or("BatchNorm must be built before forward")?;
        let reduce = non_feature_axes(input.shape(), self.feature)?;
        let (mean, biased_var) = input.moments(reduce)?;
        let normalized = input
            .sub(&mean)?
            .mul(&biased_var.inverse_sqrt(self.epsilon)?)?;
        if let Some(running) = &bound.running {
            let count = input.shape().len() / bound.extent;
            let unbiased_var = biased_var.scale(count as f32 / (count as f32 - 1.0))?;
            running.stage_update(pass, mean, unbiased_var, self.momentum, input.device())?;
        }
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

    fn named_states(&self) -> Vec<(String, State)> {
        self.bound
            .as_ref()
            .and_then(|bound| bound.running.as_ref())
            .map(RunningStats::named_states)
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
///
/// `track_running_stats` (PyTorch default `false`, so the default path stays
/// exactly the bit-for-bit instance normalization above) adds PyTorch's
/// optional running-statistics tracking on the same [`crate::State`] /
/// [`crate::TrainingPass`] mechanism as [`BatchNorm`]. PyTorch's own
/// `InstanceNorm` computes each running-statistics update by momentum-EMA-ing
/// every individual instance (one `(sample, channel)` pair) against the
/// SAME shared running buffer and then averaging the resulting per-instance
/// estimates back down to one value per channel; because averaging commutes
/// with the EMA's linear combination, that is exactly `running = (1 -
/// factor) * running + factor * pooled`, where `pooled_mean` is the mean of
/// every instance's own mean (equivalently, the grand mean over every
/// non-channel axis) and `pooled_var` is the mean, over every instance, of
/// that instance's own UNBIASED variance (`sample_size / (sample_size - 1)`
/// against its own biased variance, `sample_size` the extent product of the
/// declared sample axes only -- deliberately NOT `BatchNorm`'s pooled
/// variance over the whole batch, since each instance's own unbiasing uses
/// only its own sample count, matching PyTorch's per-instance normalization
/// exactly).
pub struct InstanceNorm {
    inner: ChannelNorm,
    momentum: Option<f32>,
    track_running_stats: bool,
    running: Option<RunningStats>,
}

impl InstanceNorm {
    /// Defaults to epsilon `1e-5`, momentum `Some(0.1)`, no affine
    /// parameters, and `track_running_stats = false`.
    pub fn new(channel: Axis, sample_axes: impl IntoAxes) -> Result<Self> {
        Ok(Self {
            inner: ChannelNorm::new(
                "InstanceNorm",
                channel,
                GroupCount::PerChannel,
                sample_axes,
                false,
                true,
            )?,
            momentum: Some(0.1),
            track_running_stats: false,
            running: None,
        })
    }

    pub fn epsilon(mut self, epsilon: f32) -> Result<Self> {
        validate_epsilon("InstanceNorm", epsilon)?;
        self.inner.epsilon = epsilon;
        Ok(self)
    }

    pub fn affine(mut self, affine: bool) -> Self {
        self.inner.affine = affine;
        self
    }

    /// `None` selects PyTorch's cumulative moving average in place of a
    /// fixed exponential factor. Only consulted when `track_running_stats`
    /// is enabled.
    pub fn momentum(mut self, momentum: Option<f32>) -> Result<Self> {
        validate_momentum("InstanceNorm", momentum)?;
        self.momentum = momentum;
        Ok(self)
    }

    /// `true` allocates and maintains `running_mean`/`running_var`/
    /// `num_batches_tracked` as in PyTorch; evaluation then uses the
    /// committed running statistics instead of this call's own instance
    /// statistics, exactly like `BatchNorm`.
    pub fn track_running_stats(mut self, track: bool) -> Self {
        self.track_running_stats = track;
        self
    }
}

impl Module for InstanceNorm {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.inner.output_shape(input)?;
        if let Some(running) = &self.running {
            if input.extent(self.inner.channel)? != running.extent {
                return Err("InstanceNorm channel extent differs from its built extent".into());
            }
            if !self.track_running_stats {
                return Err("InstanceNorm track_running_stats changed after build".into());
            }
        }
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, device: &Device, _seed: u64) -> Result<Shape> {
        let shape = self.inner.build(input, device)?;
        match &self.running {
            Some(running) if !running.mean.tensor().device().same(device) => {
                return Err("InstanceNorm is already built on a different Device".into());
            }
            Some(_) => {}
            None if self.track_running_stats => {
                let channels = input.extent(self.inner.channel)?;
                self.running = Some(RunningStats::new(self.inner.channel.of(channels), device)?);
            }
            None => {}
        }
        Ok(shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match &self.running {
            Some(running) if self.track_running_stats => {
                self.inner.output_shape(input.shape())?;
                let bound = self
                    .inner
                    .bound
                    .as_ref()
                    .ok_or("InstanceNorm must be built before forward")?;
                let normalized = running.eval_normalize(input, self.inner.epsilon)?;
                match &bound.affine {
                    Some(affine) => affine.apply(&normalized),
                    None => Ok(normalized),
                }
            }
            _ => self.inner.forward(input),
        }
    }

    fn forward_training(&self, input: &Tensor, pass: &mut TrainingPass) -> Result<Tensor> {
        let normalized = self.inner.forward(input)?;
        if let Some(running) = &self.running {
            let sample_axes = &self.inner.sample_axes;
            let channel = self.inner.channel;
            let extra_axes: Vec<Axis> = input
                .shape()
                .axes()
                .into_iter()
                .filter(|axis| *axis != channel && !sample_axes.contains(axis))
                .collect();
            let (instance_mean, instance_var) = input.moments(sample_axes.clone())?;
            let mut sample_size = 1usize;
            for &axis in sample_axes {
                sample_size = sample_size
                    .checked_mul(input.extent(axis)?)
                    .ok_or("InstanceNorm sample size overflow")?;
            }
            let unbiased_instance_var =
                instance_var.scale(sample_size as f32 / (sample_size as f32 - 1.0))?;
            let (pooled_mean, pooled_var) = if extra_axes.is_empty() {
                (instance_mean, unbiased_instance_var)
            } else {
                (
                    instance_mean.mean(extra_axes.clone())?,
                    unbiased_instance_var.mean(extra_axes)?,
                )
            };
            running.stage_update(pass, pooled_mean, pooled_var, self.momentum, input.device())?;
        }
        Ok(normalized)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.inner.named_parameters()
    }

    fn named_states(&self) -> Vec<(String, State)> {
        self.running
            .as_ref()
            .map(RunningStats::named_states)
            .unwrap_or_default()
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
