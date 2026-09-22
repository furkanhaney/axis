use crate::{
    Axis, Device, Dim, IntoAxes, Result, Shape,
    axis::Layout,
    backend::{Buffer, Plan, SelectSpec, UnfoldSpec, WindowSpec},
};
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

fn profile(label: &str, started: Instant) {
    if std::env::var_os("AXIS_PROFILE").is_some() {
        eprintln!(
            "axis_profile {label} {:.6}",
            started.elapsed().as_secs_f64()
        );
    }
}

/// Deterministic raw stream in `[0, 1)` from a xorshift64 generator, seeded `seed.max(1)`.
/// Parameter initialization (`model::nn::uniform_values`) and `Tensor::uniform`/`Tensor::normal`
/// all draw from this one stream, so a seed reproduces bit-exact values everywhere it is used.
/// A seed below 2^40 has no high bits set yet, so the first raw sample is exactly `0.0`; the
/// stream is well mixed from the second sample on. This quirk is not fixed here: recorded
/// initialization baselines depend on it.
pub(crate) fn xorshift_unit_stream(seed: u64, count: usize) -> Vec<f32> {
    let mut rng = seed.max(1);
    (0..count)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / (1_u32 << 24) as f32
        })
        .collect()
}

/// Host-side values for [`Tensor::uniform`]: each raw `[0, 1)` sample rescaled to `[low, high)`.
/// Kept as a pure function so its exact sequence is checkable against an independent oracle
/// without a device.
pub(crate) fn uniform_host_values(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    xorshift_unit_stream(seed, count)
        .into_iter()
        .map(|raw| low + raw * (high - low))
        .collect()
}

/// Host-side values for [`Tensor::normal`]: consecutive raw `[0, 1)` pairs `(u1, u2)` become one
/// Box-Muller pair of independent standard-normal values, `z0 = sqrt(-2 * ln(1 - u1)) *
/// cos(2*pi*u2)` and `z1 = sqrt(-2 * ln(1 - u1)) * sin(2*pi*u2)`, each scaled to `mean + std *
/// z`. Using `1 - u1` rather than `u1` keeps the logarithm's argument in `(0, 1]` even on the
/// shared stream's documented first-sample-exactly-zero seeds (where it collapses the first
/// pair to exactly `mean`), so no draw ever requires discarding or resampling. An odd element
/// count drops the unused second value of the final pair. Kept as a pure function for the same
/// reason as [`uniform_host_values`].
pub(crate) fn normal_host_values(seed: u64, count: usize, mean: f32, std: f32) -> Vec<f32> {
    let raw = xorshift_unit_stream(seed, count.div_ceil(2) * 2);
    let mut values = Vec::with_capacity(count);
    for pair in raw.chunks_exact(2) {
        let radius = (-2.0 * (1.0 - pair[0]).ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * pair[1];
        values.push(mean + std * (radius * angle.cos()));
        if values.len() < count {
            values.push(mean + std * (radius * angle.sin()));
        }
    }
    values
}

#[derive(Clone)]
enum UnfoldPlans {
    Implicit(Rc<UnfoldSpec>),
    Zero,
}

#[derive(Clone)]
struct ReductionPlans {
    output: Shape,
    layout: Layout,
    factor: f32,
    forward: Rc<Plan>,
    reverse: Rc<Plan>,
}

type ReductionPlanKey = (u8, Shape, Layout, Vec<Axis>);
#[derive(Clone, Eq, Hash, PartialEq)]
struct UnfoldPlanKey {
    input_shape: Shape,
    input_layout: Layout,
    output_shape: Shape,
    channels: Axis,
    spatial_rank: usize,
    spatial: [Option<Axis>; 3],
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    group: Option<Dim>,
    fill_bits: u32,
}

thread_local! {
    static REDUCTION_PLANS: RefCell<HashMap<ReductionPlanKey, ReductionPlans>> = RefCell::new(HashMap::new());
    static UNFOLD_PLANS: RefCell<HashMap<UnfoldPlanKey, UnfoldPlans>> = RefCell::new(HashMap::new());
}

#[cfg(test)]
thread_local! {
    static UNFOLD_PLAN_BUILDS: Cell<usize> = const { Cell::new(0) };
    static UNFOLD_PLAN_METADATA_MAX: Cell<usize> = const { Cell::new(0) };
    static LAYOUT_METADATA_MAX: Cell<usize> = const { Cell::new(0) };
}

static NEXT_NODE: AtomicU64 = AtomicU64::new(1);

/// Immutable values. Clones share a graph node; `detach` starts a new, untracked leaf.
#[derive(Clone)]
pub struct Tensor(Rc<Node>);
struct Node {
    id: u64,
    shape: Shape,
    layout: Layout,
    value: Buffer,
    device: Device,
    tracked: bool,
    edges: RefCell<Option<Rc<Vec<Edge>>>>,
    consumed: Cell<bool>,
    grad: RefCell<Option<Buffer>>,
    version: Option<(Rc<Cell<u64>>, u64)>,
}
struct Edge {
    input: Tensor,
    rule: Rule,
}
enum Rule {
    Identity,
    Zero(usize),
    Scale(f32),
    Multiply(Buffer),
    Divide(Buffer),
    DivideDenominator {
        numerator: Buffer,
        denominator: Buffer,
    },
    Relu(Buffer),
    Sigmoid(Buffer),
    Silu(Buffer),
    LeakyRelu {
        input: Buffer,
        negative_slope: f32,
    },
    Tanh(Buffer),
    Sin(Buffer),
    Abs(Buffer),
    Gelu(Buffer),
    GeluExact(Buffer),
    InverseSqrt {
        input: Buffer,
        epsilon: f32,
    },
    Clamp {
        input: Buffer,
        min: f32,
        max: f32,
    },
    BinaryCrossEntropy {
        logits: Buffer,
        targets: Buffer,
    },
    BinaryCrossEntropyWeighted {
        logits: Buffer,
        targets: Buffer,
        pos_weight: Buffer,
    },
    CategoricalCrossEntropy {
        probability: Buffer,
        targets: Buffer,
        width: usize,
    },
    Softmax {
        probability: Buffer,
        width: usize,
    },
    MatmulLeft {
        rhs: Buffer,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    MatmulRight {
        lhs: Buffer,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    Group {
        plan: Rc<Plan>,
        rhs: Option<Buffer>,
        factor: f32,
    },
    Minimum {
        plan: Rc<Plan>,
        winners: Buffer,
    },
    Maximum {
        plan: Rc<Plan>,
        winners: Buffer,
    },
    Unfold(Rc<UnfoldSpec>),
    Select(Rc<SelectSpec>),
    Window(Rc<WindowSpec>),
    StackSlice {
        offset: usize,
        len: usize,
    },
    Exp(Buffer),
    Ln(Buffer),
    Softplus {
        input: Buffer,
        beta: f32,
        threshold: f32,
    },
}
impl Edge {
    fn new(input: &Tensor, rule: Rule) -> Self {
        Self {
            input: input.clone(),
            rule,
        }
    }
}

impl Tensor {
    fn node(
        shape: Shape,
        layout: Layout,
        value: Buffer,
        device: &Device,
        edges: Vec<Edge>,
        leaf: bool,
        version: Option<(Rc<Cell<u64>>, u64)>,
    ) -> Self {
        let edges: Vec<_> = edges.into_iter().filter(|e| e.input.0.tracked).collect();
        let tracked = leaf || !edges.is_empty();
        Self(Rc::new(Node {
            id: NEXT_NODE.fetch_add(1, Ordering::Relaxed),
            shape,
            layout,
            value,
            device: device.clone(),
            tracked,
            edges: RefCell::new(if edges.is_empty() {
                None
            } else {
                Some(Rc::new(edges))
            }),
            consumed: Cell::new(false),
            grad: RefCell::new(None),
            version,
        }))
    }
    pub fn from_slice(
        values: &[f32],
        dims: impl IntoIterator<Item = Dim>,
        device: &Device,
    ) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if values.len() != shape.len() {
            return Err("data length does not match shape".into());
        }
        let value = device.upload(values.to_vec())?;
        Ok(Self::node(
            shape.clone(),
            Layout::contiguous(&shape),
            value,
            device,
            vec![],
            false,
            None,
        ))
    }
    /// Allocate a device-resident zero tensor without constructing a host vector.
    pub fn zeros(dims: impl IntoIterator<Item = Dim>, device: &Device) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let value = device.zeros_buffer(shape.len())?;
        Ok(Self::node(
            shape.clone(),
            Layout::contiguous(&shape),
            value,
            device,
            vec![],
            false,
            None,
        ))
    }
    /// Deterministic uniform draw in `[low, high)` from the shared xorshift stream that
    /// initializes parameters, generated host-side then uploaded like [`Tensor::from_slice`].
    /// A random draw has no upstream input, so the result carries no gradient edge; it is a
    /// constant, not a parameter. The same seed, shape and range reproduce identical values on
    /// any run or machine; distinct seeds diverge. Inherits the shared stream's documented
    /// quirk: a seed below 2^40 draws exactly `low` first.
    pub fn uniform(
        dims: impl IntoIterator<Item = Dim>,
        seed: u64,
        low: f32,
        high: f32,
        device: &Device,
    ) -> Result<Self> {
        if !(low.is_finite() && high.is_finite() && low < high) {
            return Err(
                format!("uniform requires finite low < high, got low={low} high={high}").into(),
            );
        }
        let shape = Shape::new(dims)?;
        let values = uniform_host_values(seed, shape.len(), low, high);
        Self::from_slice(&values, shape.dims().iter().copied(), device)
    }
    /// Deterministic normal draw with the given `mean` and standard deviation `std`, from the
    /// same shared xorshift stream as [`Tensor::uniform`] via the Box-Muller transform: two raw
    /// `[0, 1)` stream samples `u1, u2` become one pair `z0 = sqrt(-2 * ln(1 - u1)) *
    /// cos(2*pi*u2)`, `z1 = sqrt(-2 * ln(1 - u1)) * sin(2*pi*u2)` of independent standard-normal
    /// values, each scaled to `mean + std * z`; an odd element count drops the unused second
    /// value of the final pair. Generated host-side then uploaded, with no gradient edge,
    /// exactly like [`Tensor::uniform`].
    pub fn normal(
        dims: impl IntoIterator<Item = Dim>,
        seed: u64,
        mean: f32,
        std: f32,
        device: &Device,
    ) -> Result<Self> {
        if !(mean.is_finite() && std.is_finite() && std >= 0.0) {
            return Err(format!(
                "normal requires finite mean and non-negative std, got mean={mean} std={std}"
            )
            .into());
        }
        let shape = Shape::new(dims)?;
        let values = normal_host_values(seed, shape.len(), mean, std);
        Self::from_slice(&values, shape.dims().iter().copied(), device)
    }
    pub fn shape(&self) -> &Shape {
        &self.0.shape
    }
    pub fn extent(&self, axis: Axis) -> Result<usize> {
        self.shape().extent(axis)
    }
    pub fn device(&self) -> &Device {
        &self.0.device
    }
    pub fn requires_grad(&self) -> bool {
        self.0.tracked
    }
    pub fn with_grad(&self) -> Self {
        Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            self.0.value.clone(),
            self.device(),
            vec![],
            true,
            None,
        )
    }
    pub fn detach(&self) -> Self {
        Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            self.0.value.clone(),
            self.device(),
            vec![],
            false,
            None,
        )
    }
    /// Values in logical Shape order, independent of physical storage order.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let physical = self.device().read(&self.0.value)?;
        Ok((0..self.shape().len())
            .map(|i| physical[self.0.layout.offset(&self.shape().coords(i))])
            .collect())
    }
    pub fn item(&self) -> Result<f32> {
        if self.shape().rank() != 0 {
            return Err("item requires a scalar tensor".into());
        }
        Ok(self.to_vec()?[0])
    }
    /// Leaf gradients accumulate until explicitly cleared. Intermediates do not retain gradients.
    pub fn grad(&self) -> Option<Self> {
        self.0.grad.borrow().as_ref().map(|value| {
            Self::node(
                self.shape().clone(),
                self.0.layout.clone(),
                value.clone(),
                self.device(),
                vec![],
                false,
                None,
            )
        })
    }
    pub fn zero_grad(&self) {
        self.0.grad.borrow_mut().take();
    }
    /// Rescale this leaf's accumulated gradient in place by a global factor,
    /// without touching the forward value or building any autograd edge. A
    /// no-op when there is no gradient yet. Used by [`crate::clip_grad_norm`]
    /// to apply one global-norm scale factor to every parameter after the
    /// norm has already been computed from the unscaled gradients.
    pub(crate) fn scale_grad(&self, factor: f32) -> Result<()> {
        let mut grad = self.0.grad.borrow_mut();
        if let Some(buffer) = grad.as_ref() {
            *grad = Some(self.device().scale(buffer, factor)?);
        }
        Ok(())
    }
    pub(crate) fn parameter_leaf(&self, version: Rc<Cell<u64>>) -> Self {
        let expected = version.get();
        Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            self.0.value.clone(),
            self.device(),
            vec![],
            true,
            Some((version, expected)),
        )
    }
    pub(crate) fn updated(&self, learning_rate: f32) -> Result<Self> {
        let gradient = self
            .0
            .grad
            .borrow()
            .clone()
            .ok_or("parameter has no gradient; run backward before step")?;
        let delta = self.device().scale(&gradient, learning_rate)?;
        let value = self.device().binary(&self.0.value, &delta, 1)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![],
            false,
            None,
        ))
    }
    pub(crate) fn expanded_axis_values(&self, axis: Axis, values: &[f32]) -> Result<Buffer> {
        if self.extent(axis)? != values.len() {
            return Err(format!(
                "axis learning-rate count {} does not match {:?} extent {}",
                values.len(),
                axis,
                self.extent(axis)?
            )
            .into());
        }
        Ok(
            Tensor::from_slice(values, [axis.of(values.len())], self.device())?
                .align(self.shape())?
                .0
                .value
                .clone(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adam_updated(
        &self,
        first: Option<&Buffer>,
        second: Option<&Buffer>,
        learning_rate: f32,
        learning_rates: Option<&Buffer>,
        beta1: f32,
        beta2: f32,
        correction1: f32,
        correction2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<(Self, Buffer, Buffer)> {
        let gradient = self
            .0
            .grad
            .borrow()
            .clone()
            .ok_or("parameter has no gradient; run backward before step")?;
        let zero_first;
        let first = match first {
            Some(value) => value,
            None => {
                zero_first = self.device().zeros_buffer(self.shape().len())?;
                &zero_first
            }
        };
        let zero_second;
        let second = match second {
            Some(value) => value,
            None => {
                zero_second = self.device().zeros_buffer(self.shape().len())?;
                &zero_second
            }
        };
        let (value, next_first, next_second) = self.device().adam(
            &self.0.value,
            &gradient,
            first,
            second,
            learning_rate,
            learning_rates,
            beta1,
            beta2,
            correction1,
            correction2,
            epsilon,
            weight_decay,
        )?;
        Ok((
            Self::node(
                self.shape().clone(),
                self.0.layout.clone(),
                value,
                self.device(),
                vec![],
                false,
                None,
            ),
            next_first,
            next_second,
        ))
    }
    fn compatible_device(&self, rhs: &Self) -> Result<()> {
        if !self.device().same(rhs.device()) {
            return Err("tensors must use the same Device handle".into());
        }
        Ok(())
    }
    fn shared_extents(&self, rhs: &Self) -> Result<()> {
        for d in self.shape().dims() {
            if rhs.shape().contains(d.axis) && rhs.extent(d.axis)? != d.extent {
                return Err(format!("extent mismatch for {:?}", d.axis).into());
            }
        }
        Ok(())
    }
    fn unfolded_with_spec(
        &self,
        shape: Shape,
        layout: Layout,
        spec: Rc<UnfoldSpec>,
    ) -> Result<Self> {
        let value = self.device().unfold(&self.0.value, spec.as_ref())?;
        let edges = self
            .requires_grad()
            .then(|| Edge::new(self, Rule::Unfold(spec)));
        Ok(Self::node(
            shape,
            layout,
            value,
            self.device(),
            edges.into_iter().collect(),
            false,
            None,
        ))
    }
    fn zero_gathered(&self, shape: Shape, layout: Layout) -> Result<Self> {
        let value = self.device().zeros_buffer(shape.len())?;
        let edges = self
            .requires_grad()
            .then(|| Edge::new(self, Rule::Zero(self.shape().len())));
        Ok(Self::node(
            shape.clone(),
            layout,
            value,
            self.device(),
            edges.into_iter().collect(),
            false,
            None,
        ))
    }
    #[cfg(test)]
    pub(crate) fn unfold_plan_build_count() -> usize {
        UNFOLD_PLAN_BUILDS.with(Cell::get)
    }
    #[cfg(test)]
    pub(crate) fn unfold_plan_metadata_max() -> usize {
        UNFOLD_PLAN_METADATA_MAX.with(Cell::get)
    }
    #[cfg(test)]
    pub(crate) fn layout_metadata_max() -> usize {
        LAYOUT_METADATA_MAX.with(Cell::get)
    }
    #[cfg(test)]
    pub(crate) fn layout_strides(&self) -> &[usize] {
        &self.0.layout.strides
    }
    #[cfg(test)]
    pub(crate) fn shares_buffer(&self, other: &Self) -> bool {
        std::ptr::eq(self.0.value.as_ref(), other.0.value.as_ref())
    }
    /// `self` presented in `shape`'s axis order, contiguous. Axes missing from `self`
    /// broadcast. Both cases run the rank-sized compact copier: a permutation is
    /// bijective and keeps the copier's inverse for its gradient; a broadcast reads
    /// with stride 0 and, only when a gradient is required, builds the scatter-add
    /// plan that sums over the broadcast axes. No element-sized plan is built or
    /// uploaded in inference.
    fn align(&self, shape: &Shape) -> Result<Self> {
        let started = Instant::now();
        let layout = Layout::contiguous(shape);
        if self.shape() == shape && self.0.layout.strides == layout.strides {
            return Ok(self.clone());
        }
        let positions: Vec<_> = self
            .shape()
            .axes()
            .iter()
            .map(|&a| shape.index(a))
            .collect::<Result<_>>()?;
        if self.shape().rank() == shape.rank() {
            let ordered = self.with_layout(shape.axes())?;
            let result = Self::node(
                shape.clone(),
                layout,
                ordered.0.value.clone(),
                self.device(),
                vec![Edge::new(&ordered, Rule::Identity)],
                false,
                None,
            );
            profile("align", started);
            return Ok(result);
        }
        let mut metadata = Vec::with_capacity(shape.rank() * 3);
        for (index, dim) in shape.dims().iter().enumerate() {
            let input_stride = positions
                .iter()
                .position(|&p| p == index)
                .map_or(0, |own| self.0.layout.strides[own]);
            metadata.extend([
                i32::try_from(dim.extent)?,
                i32::try_from(input_stride)?,
                i32::try_from(layout.strides[index])?,
            ]);
        }
        let spec = SelectSpec {
            input_len: self.shape().len(),
            output_len: shape.len(),
            rank: i32::try_from(shape.rank())?,
            coordinate: 0,
            metadata,
        };
        let value = self.device().select_axis(&self.0.value, &spec)?;
        let edges = self
            .requires_grad()
            .then(|| {
                let map = (0..shape.len())
                    .map(|i| {
                        let coords = shape.coords(i);
                        self.0
                            .layout
                            .offset(&positions.iter().map(|&p| coords[p]).collect::<Vec<_>>())
                    })
                    .collect::<Vec<_>>();
                Plan::reverse(&map, self.shape().len()).map(|plan| {
                    Edge::new(
                        self,
                        Rule::Group {
                            plan: Rc::new(plan),
                            rhs: None,
                            factor: 1.0,
                        },
                    )
                })
            })
            .transpose()?;
        let result = Self::node(
            shape.clone(),
            layout,
            value,
            self.device(),
            edges.into_iter().collect(),
            false,
            None,
        );
        profile("align", started);
        Ok(result)
    }
    fn binary(&self, rhs: &Self, op: i32) -> Result<Self> {
        self.compatible_device(rhs)?;
        self.shared_extents(rhs)?;
        let a_in_b = self.shape().axes().iter().all(|&a| rhs.shape().contains(a));
        let b_in_a = rhs.shape().axes().iter().all(|&a| self.shape().contains(a));
        let shape = if b_in_a {
            self.shape()
        } else if a_in_b {
            rhs.shape()
        } else {
            return Err("incomparable axis sets: elementwise broadcasting cannot introduce an implicit outer product".into());
        };
        let a = self.align(shape)?;
        let b = rhs.align(shape)?;
        let value = self.device().binary(&a.0.value, &b.0.value, op)?;
        let rules = match op {
            0 => (Rule::Identity, Rule::Identity),
            1 => (Rule::Identity, Rule::Scale(-1.0)),
            2 => (
                Rule::Multiply(b.0.value.clone()),
                Rule::Multiply(a.0.value.clone()),
            ),
            _ => (
                Rule::Divide(b.0.value.clone()),
                Rule::DivideDenominator {
                    numerator: a.0.value.clone(),
                    denominator: b.0.value.clone(),
                },
            ),
        };
        Ok(Self::node(
            shape.clone(),
            Layout::contiguous(shape),
            value,
            self.device(),
            vec![Edge::new(&a, rules.0), Edge::new(&b, rules.1)],
            false,
            None,
        ))
    }
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, 0)
    }
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, 1)
    }
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, 2)
    }
    /// Elementwise `self / rhs`. Same shape-agreement contract as [`Tensor::mul`]:
    /// identical axis sets, no implicit alignment. Division by zero follows IEEE
    /// float semantics (produces `inf`/`nan`, as in PyTorch); callers that need a
    /// safe denominator must clamp it themselves before dividing.
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, 3)
    }
    pub fn squared_error(&self, rhs: &Self) -> Result<Self> {
        if self.shape().rank() != rhs.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&a| !rhs.shape().contains(a))
        {
            return Err("squared_error requires identical axis sets".into());
        }
        let delta = self.sub(rhs)?;
        delta.mul(&delta)
    }
    /// Elementwise absolute value. Backward is `gradient * sign(x)`, matching PyTorch's `abs`
    /// backward: the gradient is exactly zero at `x == 0` (PyTorch's own subgradient choice
    /// there, not `NaN` or either one-sided slope).
    pub fn abs(&self) -> Result<Self> {
        let value = self.device().abs(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Abs(self.0.value.clone()))],
            false,
            None,
        ))
    }
    /// Elementwise mean absolute error (PyTorch's unreduced `F.l1_loss`): `(self -
    /// rhs).abs()`. Same axis-agreement contract as [`Tensor::squared_error`] — identical axis
    /// sets, equal extents, unreduced shape — leaving the reduction (typically `.mean(axes)`) to
    /// the caller.
    pub fn absolute_error(&self, rhs: &Self) -> Result<Self> {
        if self.shape().rank() != rhs.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&a| !rhs.shape().contains(a))
        {
            return Err("absolute_error requires identical axis sets".into());
        }
        self.sub(rhs)?.abs()
    }
    /// Stable elementwise binary cross-entropy. Targets are constants in reverse mode.
    pub fn binary_cross_entropy_with_logits(&self, targets: &Self) -> Result<Self> {
        if targets.requires_grad() {
            return Err("binary cross-entropy targets cannot require gradients".into());
        }
        if self.shape().rank() != targets.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&axis| !targets.shape().contains(axis))
        {
            return Err("binary_cross_entropy_with_logits requires identical axis sets".into());
        }
        self.compatible_device(targets)?;
        self.shared_extents(targets)?;
        let logits = self.align(self.shape())?;
        let targets = targets.align(self.shape())?;
        let value = self
            .device()
            .binary_cross_entropy(&logits.0.value, &targets.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            Layout::contiguous(self.shape()),
            value,
            self.device(),
            vec![Edge::new(
                &logits,
                Rule::BinaryCrossEntropy {
                    logits: logits.0.value.clone(),
                    targets: targets.0.value.clone(),
                },
            )],
            false,
            None,
        ))
    }
    /// Stable elementwise binary cross-entropy with an explicit positive-class weight,
    /// matching `torch.nn.BCEWithLogitsLoss(pos_weight=...)`: the positive term of each
    /// element's loss is scaled by `pos_weight` before the negative term is added, so
    /// `pos_weight` changes the loss value itself and is not reachable by scaling the
    /// unweighted loss afterward. `pos_weight` may be a scalar or may broadcast over any
    /// subset of `self`'s axes (for example one weight per class, broadcasting over batch
    /// and spatial axes); it follows the same subset-axis broadcasting as elementwise
    /// `add`/`mul`, never an implicit outer product. Targets and `pos_weight` are
    /// constants in reverse mode.
    pub fn binary_cross_entropy_with_logits_weighted(
        &self,
        targets: &Self,
        pos_weight: &Self,
    ) -> Result<Self> {
        if targets.requires_grad() {
            return Err("binary cross-entropy targets cannot require gradients".into());
        }
        if pos_weight.requires_grad() {
            return Err("binary cross-entropy pos_weight cannot require gradients".into());
        }
        if self.shape().rank() != targets.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&axis| !targets.shape().contains(axis))
        {
            return Err(
                "binary_cross_entropy_with_logits_weighted requires identical axis sets for logits and targets"
                    .into(),
            );
        }
        if pos_weight
            .shape()
            .axes()
            .iter()
            .any(|&axis| !self.shape().contains(axis))
        {
            return Err(
                "binary_cross_entropy_with_logits_weighted pos_weight axes must be a subset of the logits axes"
                    .into(),
            );
        }
        self.compatible_device(targets)?;
        self.compatible_device(pos_weight)?;
        self.shared_extents(targets)?;
        self.shared_extents(pos_weight)?;
        let logits = self.align(self.shape())?;
        let targets = targets.align(self.shape())?;
        let pos_weight = pos_weight.align(self.shape())?;
        let value = self.device().binary_cross_entropy_weighted(
            &logits.0.value,
            &targets.0.value,
            &pos_weight.0.value,
        )?;
        Ok(Self::node(
            self.shape().clone(),
            Layout::contiguous(self.shape()),
            value,
            self.device(),
            vec![Edge::new(
                &logits,
                Rule::BinaryCrossEntropyWeighted {
                    logits: logits.0.value.clone(),
                    targets: targets.0.value.clone(),
                    pos_weight: pos_weight.0.value.clone(),
                },
            )],
            false,
            None,
        ))
    }
    /// Stable categorical cross-entropy for constant one-hot or probability targets.
    /// The named class axis is reduced; every other logits axis is preserved.
    pub fn categorical_cross_entropy_with_logits(
        &self,
        targets: &Self,
        class: Axis,
    ) -> Result<Self> {
        if targets.requires_grad() {
            return Err("categorical cross-entropy targets cannot require gradients".into());
        }
        if !targets.shape().contains(class)
            || targets
                .shape()
                .axes()
                .iter()
                .any(|&axis| !self.shape().contains(axis))
        {
            return Err("categorical_cross_entropy_with_logits targets must contain the class axis and may omit only broadcast axes".into());
        }
        self.compatible_device(targets)?;
        self.shared_extents(targets)?;
        let width = self.extent(class)?;
        if width < 2 {
            return Err("categorical cross-entropy requires at least two classes".into());
        }
        let mut ordered_dims: Vec<_> = self
            .shape()
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != class)
            .collect();
        let output = Shape::new(ordered_dims.iter().copied())?;
        ordered_dims.push(class.of(width));
        let ordered = Shape::new(ordered_dims)?;
        let logits = self.align(&ordered)?;
        let targets = targets.align(&ordered)?;
        for (row, distribution) in targets.to_vec()?.chunks_exact(width).enumerate() {
            if distribution
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(format!(
                    "categorical target row {row} must contain finite nonnegative probabilities"
                )
                .into());
            }
            let sum: f32 = distribution.iter().sum();
            if (sum - 1.0).abs() > 1e-5 {
                return Err(
                    format!("categorical target row {row} must sum to 1; observed {sum}").into(),
                );
            }
        }
        let probability = self.device().softmax(&logits.0.value, width)?;
        let value =
            self.device()
                .categorical_cross_entropy(&logits.0.value, &targets.0.value, width)?;
        Ok(Self::node(
            output.clone(),
            Layout::contiguous(&output),
            value,
            self.device(),
            vec![Edge::new(
                &logits,
                Rule::CategoricalCrossEntropy {
                    probability,
                    targets: targets.0.value.clone(),
                    width,
                },
            )],
            false,
            None,
        ))
    }
    pub(crate) fn categorical_correct_flags(&self, targets: &Self, class: Axis) -> Result<Self> {
        if self.shape().rank() != targets.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&axis| !targets.shape().contains(axis))
        {
            return Err("categorical accuracy requires identical axis sets".into());
        }
        self.compatible_device(targets)?;
        self.shared_extents(targets)?;
        let width = self.extent(class)?;
        if width < 2 {
            return Err("categorical accuracy requires at least two classes".into());
        }
        let mut ordered_dims: Vec<_> = self
            .shape()
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != class)
            .collect();
        let output = Shape::new(ordered_dims.iter().copied())?;
        ordered_dims.push(class.of(width));
        let ordered = Shape::new(ordered_dims)?;
        let logits = self.align(&ordered)?;
        let targets = targets.align(&ordered)?;
        let value = self
            .device()
            .categorical_correct(&logits.0.value, &targets.0.value, width)?;
        Ok(Self::node(
            output.clone(),
            Layout::contiguous(&output),
            value,
            self.device(),
            vec![],
            false,
            None,
        ))
    }
    pub fn scale(&self, factor: f32) -> Result<Self> {
        if !factor.is_finite() {
            return Err("scale must be finite".into());
        }
        let value = self.device().scale(&self.0.value, factor)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Scale(factor))],
            false,
            None,
        ))
    }
    pub(crate) fn normalized_l2(&self, epsilon: f32) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("L2 normalization epsilon must be finite and positive".into());
        }
        let scalar_shape = Shape::new([])?;
        let norm_squared = Self::node(
            scalar_shape.clone(),
            Layout::contiguous(&scalar_shape),
            self.device().sum_squares(&self.0.value)?,
            self.device(),
            vec![],
            false,
            None,
        );
        // Keep the normalizer device-resident. `inverse_sqrt` is the backend's
        // unary square-root primitive; the minimum positive adjustment is far
        // below the public Muon epsilon for every representable nonzero norm.
        let inverse_root = norm_squared.inverse_sqrt(f32::MIN_POSITIVE)?;
        let norm = norm_squared.mul(&inverse_root)?;
        let denominator = norm.add(&Self::from_slice(&[epsilon], [], self.device())?)?;
        let inverse_half = denominator.inverse_sqrt(f32::MIN_POSITIVE)?;
        let inverse = inverse_half.mul(&inverse_half)?;
        let value = self
            .device()
            .multiply_scalar(&self.0.value, &inverse.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![],
            false,
            None,
        ))
    }

    pub fn relu(&self) -> Result<Self> {
        let value = self.device().relu(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Relu(self.0.value.clone()))],
            false,
            None,
        ))
    }
    /// Stable elementwise logistic sigmoid.
    pub fn sigmoid(&self) -> Result<Self> {
        let value = self.device().sigmoid(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value.clone(),
            self.device(),
            vec![Edge::new(self, Rule::Sigmoid(value))],
            false,
            None,
        ))
    }
    /// Elementwise sigmoid linear unit, `x * sigmoid(x)`.
    pub fn silu(&self) -> Result<Self> {
        let value = self.device().silu(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Silu(self.0.value.clone()))],
            false,
            None,
        ))
    }

    /// Elementwise leaky ReLU with an explicit finite, non-negative slope.
    /// Its derivative at exactly zero is the negative slope, matching PyTorch.
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<Self> {
        if !negative_slope.is_finite() || negative_slope < 0.0 {
            return Err("leaky_relu negative slope must be finite and non-negative".into());
        }
        let value = self.device().leaky_relu(&self.0.value, negative_slope)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::LeakyRelu {
                    input: self.0.value.clone(),
                    negative_slope,
                },
            )],
            false,
            None,
        ))
    }
    /// Elementwise hyperbolic tangent.
    pub fn tanh(&self) -> Result<Self> {
        let value = self.device().tanh(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value.clone(),
            self.device(),
            vec![Edge::new(self, Rule::Tanh(value))],
            false,
            None,
        ))
    }
    /// Elementwise sine in radians.
    pub fn sin(&self) -> Result<Self> {
        let value = self.device().sin(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Sin(self.0.value.clone()))],
            false,
            None,
        ))
    }
    /// Tanh-approximated GELU, matching the common transformer formulation.
    ///
    /// This is PyTorch's `F.gelu(x, approximate="tanh")`, NOT its default. A model ported
    /// from an un-annotated `F.gelu(x)` or `nn.GELU()` wants [`Self::gelu_exact`]; using this
    /// form instead gives plausible numbers that differ by about `1e-3` in the tails.
    pub fn gelu(&self) -> Result<Self> {
        let value = self.device().gelu(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Gelu(self.0.value.clone()))],
            false,
            None,
        ))
    }
    /// Erf-form GELU, `x * Phi(x)`, rather than the tanh formulation of [`Self::gelu`].
    ///
    /// The CUDA backend numerically evaluates the normal CDF in FP32 (not bitwise
    /// identical to a particular libm). Backward uses `Phi(x) + x * phi(x)`.
    /// This distinction matters when importing pretrained exact-GELU models: this is
    /// PyTorch's default `F.gelu`/`nn.GELU` (`approximate="none"`).
    pub fn gelu_exact(&self) -> Result<Self> {
        let value = self.device().gelu_exact(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::GeluExact(self.0.value.clone()))],
            false,
            None,
        ))
    }
    /// Elementwise sign quantization to `{-1, +1}` with a straight-through backward.
    ///
    /// Forward: `+1` where `x > 0`, otherwise `-1`, so `x == 0` maps to `-1`. This
    /// reproduces bae's `bitae.quantize` two-level branch,
    /// `torch.where(z > 0, 1.0, -1.0)`, rather than `torch.sign`, which would give
    /// exactly `0` at `x == 0`. Backward passes the upstream gradient through
    /// unchanged (the straight-through estimator, `Rule::Identity`), matching
    /// `z + (hard - z).detach()` in the same reference: the hard threshold has no
    /// gradient of its own, so the whole local Jacobian is the identity. Output
    /// has the same [`Shape`] as the input.
    pub fn sign_straight_through(&self) -> Result<Self> {
        let value = self.device().sign(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Identity)],
            false,
            None,
        ))
    }
    /// Elementwise exponential, `exp(x)`. Backward is `g * exp(x)`, computed from
    /// the already-produced output rather than re-evaluating `exp` from the input,
    /// the same trade [`Self::tanh`] makes. Output has the same [`Shape`] as the
    /// input.
    pub fn exp(&self) -> Result<Self> {
        let value = self.device().exp(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value.clone(),
            self.device(),
            vec![Edge::new(self, Rule::Exp(value))],
            false,
            None,
        ))
    }
    /// Elementwise natural logarithm, `ln(x)`. IEEE behaviour, no clamping: `ln`
    /// of a non-positive input is `-inf` at exactly `x == 0` and `NaN` for
    /// `x < 0`, matching `f32::ln`/PyTorch's `torch.log` rather than any epsilon
    /// or absolute-value guard. Backward is `g / x`, which inherits the same
    /// non-finite behaviour at and below zero. Output has the same [`Shape`] as
    /// the input.
    pub fn ln(&self) -> Result<Self> {
        let value = self.device().ln(&self.0.value)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Ln(self.0.value.clone()))],
            false,
            None,
        ))
    }
    /// PyTorch-exact `Softplus`: `(1 / beta) * ln(1 + exp(beta * x))`, except
    /// where `beta * x > threshold`, which returns `x` itself (the identity
    /// function's large-`x` asymptote, computed exactly rather than through the
    /// logarithm) to keep both branches finite and match
    /// `torch.nn.functional.softplus(x, beta, threshold)` bit for bit at the seam.
    /// Backward is `sigmoid(beta * x)` on the logarithmic branch and exactly `1`
    /// on the linear branch, mirroring the forward's own branch selection.
    /// `beta` must be finite and positive; `threshold` must be finite.
    pub fn softplus(&self, beta: f32, threshold: f32) -> Result<Self> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err("softplus beta must be finite and positive".into());
        }
        if !threshold.is_finite() {
            return Err("softplus threshold must be finite".into());
        }
        let value = self.device().softplus(&self.0.value, beta, threshold)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::Softplus {
                    input: self.0.value.clone(),
                    beta,
                    threshold,
                },
            )],
            false,
            None,
        ))
    }
    /// Elementwise ELU (`torch.nn.functional.elu`): `x` where `x > 0`, otherwise
    /// `alpha * (exp(x) - 1)`. `alpha` must be finite and positive. Composed from
    /// [`Self::gt`]/[`Self::logical_not`], [`Self::clamp`] and [`Self::exp`]
    /// rather than a dedicated kernel: clamping the input to `(-inf, 0]` before
    /// `exp` keeps the positive branch -- masked out of the result anyway -- from
    /// ever overflowing. Backward matches PyTorch's own `x > 0` split exactly,
    /// including at `x == 0` (grouped with the negative branch, so the gradient
    /// there is `alpha`, not `1`): `clamp`'s own pass-through gradient at its
    /// upper bound keeps that boundary case exact through the chain rule.
    pub fn elu(&self, alpha: f32) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err("elu alpha must be finite and positive".into());
        }
        let positive = self.gt(0.0)?;
        let negative = positive.logical_not()?;
        let one = Self::from_slice(&[1.0], [], self.device())?;
        let branch = self
            .clamp(None, Some(0.0))?
            .exp()?
            .sub(&one)?
            .scale(alpha)?;
        positive.mul(self)?.add(&negative.mul(&branch)?)
    }
    /// Elementwise CELU (`torch.nn.functional.celu`, the continuously
    /// differentiable exponential linear unit): `x` where `x > 0`, otherwise
    /// `alpha * (exp(x / alpha) - 1)`. `alpha` must be finite and positive --
    /// PyTorch's own definition allows any nonzero `alpha`, but a negative one
    /// makes the negative branch diverge instead of saturate, so Axis narrows
    /// the accepted range to the saturating case every consumer wants. Composed
    /// exactly like [`Self::elu`], scaling the clamped input by `1 / alpha`
    /// before `exp` and the branch result back up by `alpha`; the same
    /// `x == 0` boundary reasoning applies.
    pub fn celu(&self, alpha: f32) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err("celu alpha must be finite and positive".into());
        }
        let positive = self.gt(0.0)?;
        let negative = positive.logical_not()?;
        let one = Self::from_slice(&[1.0], [], self.device())?;
        let branch = self
            .clamp(None, Some(0.0))?
            .scale(1.0 / alpha)?
            .exp()?
            .sub(&one)?
            .scale(alpha)?;
        positive.mul(self)?.add(&negative.mul(&branch)?)
    }
    /// Elementwise SELU (`torch.nn.functional.selu`): PyTorch's fixed-constant
    /// self-normalizing activation, `scale * elu(x, alpha)` with
    /// `alpha = 1.6732632423543772` and `scale = 1.0507009873554805` --
    /// PyTorch's own literals, solved so a standard-normal input keeps zero mean
    /// and unit variance through the nonlinearity. No configurable parameters:
    /// unlike [`Self::elu`]/[`Self::celu`], `selu` takes no `alpha`.
    pub fn selu(&self) -> Result<Self> {
        const SELU_ALPHA: f32 = 1.673_263_2;
        const SELU_SCALE: f32 = 1.050_701;
        self.elu(SELU_ALPHA)?.scale(SELU_SCALE)
    }
    /// Numerically stable elementwise log-sigmoid, `ln(sigmoid(x))`
    /// (`torch.nn.functional.logsigmoid`), computed as `-softplus(-x)`: PyTorch's
    /// own stabilization. `softplus`'s existing linear seam keeps this finite for
    /// very negative `x`, where a literal `sigmoid` then `ln` would underflow to
    /// `ln(0) = -inf`, and it asymptotes to `x` exactly as `log_sigmoid` should
    /// when `x -> -inf`. Backward is `sigmoid(-x)`, which falls out of the chain
    /// rule through [`Self::scale`] and [`Self::softplus`] with no dedicated rule.
    pub fn log_sigmoid(&self) -> Result<Self> {
        self.scale(-1.0)?.softplus(1.0, 20.0)?.scale(-1.0)
    }
    /// Elementwise Mish (`torch.nn.functional.mish`), `x * tanh(softplus(x))`,
    /// composed from the existing stable [`Self::softplus`] (`beta = 1`,
    /// `threshold = 20`, PyTorch's own defaults) and [`Self::tanh`] rather than a
    /// dedicated kernel.
    pub fn mish(&self) -> Result<Self> {
        self.mul(&self.softplus(1.0, 20.0)?.tanh()?)
    }
    /// Gated Linear Unit (`torch.nn.functional.glu`): split `axis` into two
    /// equal halves and gate the first half by the sigmoid of the second,
    /// `a * sigmoid(b)`. `axis`'s extent must be even; both halves keep `axis`'s
    /// own identity, at half the extent. Composed from [`Self::narrow`] and
    /// [`Self::sigmoid`]/[`Self::mul`].
    pub fn glu(&self, axis: Axis) -> Result<Self> {
        let extent = self.extent(axis)?;
        if extent % 2 != 0 {
            return Err("glu requires an even extent on the split axis".into());
        }
        let half = extent / 2;
        let a = self.narrow(axis, 0, half)?;
        let b = self.narrow(axis, half, half)?;
        a.mul(&b.sigmoid()?)
    }
    /// Elementwise parametric ReLU (`torch.nn.functional.prelu`): `x` where
    /// `x > 0`, otherwise `weight * x`. `weight` is a constant here -- the
    /// [`crate::PReLU`] module supplies a differentiable [`crate::Parameter`]
    /// and composes with this method -- and follows the same subset-axis
    /// broadcasting as [`Self::mul`], so a single shared weight (shape `[]`) or
    /// one weight per entry of a named channel axis both work unchanged.
    /// Composed entirely from [`Self::gt`]/[`Self::logical_not`] and
    /// [`Self::mul`]/[`Self::add`], with the same `x > 0` boundary convention as
    /// [`Self::elu`]; `weight`'s own gradient (when it requires one) falls out
    /// of `mul`'s existing broadcast-sum backward with no dedicated rule.
    pub fn prelu(&self, weight: &Self) -> Result<Self> {
        let positive = self.gt(0.0)?;
        let negative = positive.logical_not()?;
        positive.mul(self)?.add(&negative.mul(self)?.mul(weight)?)
    }
    /// Numerically stable log-softmax along one named `axis`
    /// (`torch.nn.functional.log_softmax`), `x - logsumexp(x, axis)`, composed
    /// entirely from the existing [`Self::logsumexp`] (itself `max`-shifted) and
    /// [`Self::sub`]'s broadcast over the axis `logsumexp` removes -- no
    /// dedicated kernel or backward rule. Shares `logsumexp`'s all-non-finite
    /// group convention (`NaN`, not PyTorch's `-infinity`).
    pub fn log_softmax(&self, axis: Axis) -> Result<Self> {
        self.sub(&self.logsumexp(axis)?)
    }
    /// Softmin along one named `axis` (`torch.nn.functional.softmin`),
    /// `softmax(-x, axis)`, composed from [`Self::scale`] and the existing
    /// stable [`Self::softmax`].
    pub fn softmin(&self, axis: Axis) -> Result<Self> {
        self.scale(-1.0)?.softmax(axis)
    }
    /// Shared body for the scalar comparison family below. `op` selects the cuTile
    /// comparison the same way [`Self::binary`]'s `op` selects add/sub/mul/div: 0
    /// (`>`), 1 (`>=`), 2 (`<`), 3 (`<=`), 4 (`==`). Output has the same [`Shape`]
    /// as the input, holds exactly `0.0` or `1.0`, and carries no autograd edge:
    /// like PyTorch's comparison operators, a comparison is not differentiable, so
    /// the result is built the same way a leaf constant is (`Tensor::from_slice`) --
    /// empty edges, not a leaf -- rather than routing a `Rule` through `self`. A
    /// downstream op such as `x.mul(&mask)` still back-propagates correctly to `x`:
    /// only the mask's own gradient path is absent, matching the masking
    /// convention `causal_mask` and `minimum_winner_mask` already use for their
    /// constant 0/1 outputs.
    fn compare_scalar(&self, scalar: f32, op: i32) -> Result<Self> {
        let value = self.device().compare_scalar(&self.0.value, scalar, op)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![],
            false,
            None,
        ))
    }
    /// Elementwise `self > scalar` as a `{0.0, 1.0}` mask, IEEE-ordered so any
    /// comparison against `NaN` is `false` (`0.0`), exactly like PyTorch's `>`.
    /// No autograd edge, even when `self` requires grad: like PyTorch, a
    /// comparison is not differentiable.
    pub fn gt(&self, scalar: f32) -> Result<Self> {
        self.compare_scalar(scalar, 0)
    }
    /// Elementwise `self >= scalar`. Same NaN and no-gradient contract as [`Self::gt`].
    pub fn ge(&self, scalar: f32) -> Result<Self> {
        self.compare_scalar(scalar, 1)
    }
    /// Elementwise `self < scalar`. Same NaN and no-gradient contract as [`Self::gt`].
    pub fn lt(&self, scalar: f32) -> Result<Self> {
        self.compare_scalar(scalar, 2)
    }
    /// Elementwise `self <= scalar`. Same NaN and no-gradient contract as [`Self::gt`].
    pub fn le(&self, scalar: f32) -> Result<Self> {
        self.compare_scalar(scalar, 3)
    }
    /// Elementwise `self == scalar`, bit-for-bit IEEE equality (no epsilon). Same
    /// NaN and no-gradient contract as [`Self::gt`].
    pub fn eq(&self, scalar: f32) -> Result<Self> {
        self.compare_scalar(scalar, 4)
    }
    /// Elementwise logical AND of two `{0.0, 1.0}` masks, spelled as their product
    /// (`mask * mask` is exactly PyTorch's `&` on 0/1 tensors). Same axis-agreement
    /// contract as [`Self::mul`]; the result carries a gradient only if `mul` would
    /// give one, which is never the case for two comparison-derived masks since
    /// neither operand has an autograd edge to begin with.
    pub fn logical_and(&self, rhs: &Self) -> Result<Self> {
        self.mul(rhs)
    }
    /// Elementwise logical NOT of a `{0.0, 1.0}` mask, spelled as `1.0 - mask`
    /// (PyTorch's `~` on a 0/1 tensor). Logical OR has no named method because no
    /// migrated consumer calls it; on 0/1 masks it is an elementwise maximum
    /// (`a | b` == `max(a, b)`, since both are `{0.0, 1.0}`), which Axis has no
    /// tensor/tensor primitive for yet -- add one only when a consumer needs it.
    pub fn logical_not(&self) -> Result<Self> {
        let one = Self::from_slice(&[1.0], [], self.device())?;
        one.sub(self)
    }
    /// Elementwise `(x + epsilon)^-1/2`; inputs plus epsilon must be positive.
    pub fn inverse_sqrt(&self, epsilon: f32) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("inverse_sqrt epsilon must be finite and positive".into());
        }
        let value = self.device().inverse_sqrt(&self.0.value, epsilon)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::InverseSqrt {
                    input: self.0.value.clone(),
                    epsilon,
                },
            )],
            false,
            None,
        ))
    }
    /// Elementwise clamp into `[min, max]`; either bound may be `None` to leave that side
    /// unbounded, matching `torch.clamp`'s optional `min`/`max` keywords (`sites.clamp_(0,
    /// 1)` in `vision/image-encode`'s Stage A and `counts.clamp(min=1)` in
    /// `world/energy-output`'s reconstruction loss are the two consumers -- the second gives
    /// only `min`). Rejects a `NaN` bound or `min > max` before any device launch. A `NaN`
    /// element of `self` propagates unclamped: like [`Self::gt`] and its siblings, every
    /// ordered comparison against `NaN` is `false`, so neither the low nor the high branch
    /// ever fires for it.
    ///
    /// Gradient matches PyTorch's `clamp` convention, not "zero at the boundary too": the
    /// upstream gradient passes through unchanged where `min <= x && x <= max` -- including
    /// exactly at either bound -- and is zero everywhere `x` was actually moved by clamping.
    /// A `NaN` input therefore also gets a zero gradient, since `x >= min` is itself `false`
    /// for `NaN` under the same ordered-comparison rule.
    pub fn clamp(&self, min: Option<f32>, max: Option<f32>) -> Result<Self> {
        if min.is_some_and(f32::is_nan) {
            return Err("clamp min must not be NaN".into());
        }
        if max.is_some_and(f32::is_nan) {
            return Err("clamp max must not be NaN".into());
        }
        if let (Some(min), Some(max)) = (min, max)
            && min > max
        {
            return Err(format!("clamp requires min <= max, got min={min} max={max}").into());
        }
        let lo = min.unwrap_or(f32::NEG_INFINITY);
        let hi = max.unwrap_or(f32::INFINITY);
        let value = self.device().clamp(&self.0.value, lo, hi)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::Clamp {
                    input: self.0.value.clone(),
                    min: lo,
                    max: hi,
                },
            )],
            false,
            None,
        ))
    }
    /// A scalar constant broadcast against any shape, built the same way
    /// [`Self::logical_not`] and `normalized_l2`'s epsilon term already are
    /// (`Tensor::from_slice` with an empty axis list): no autograd edge, and
    /// [`Self::binary`]'s subset-axis rule aligns an axis-less operand onto
    /// any other shape for free.
    fn constant(&self, value: f32) -> Result<Self> {
        Self::from_slice(&[value], [], self.device())
    }
    /// Elementwise `Hardtanh`, PyTorch's `F.hardtanh(x, min_val, max_val)`: the same
    /// values as [`Self::clamp`]`(Some(min_val), Some(max_val))`, but with `hardtanh`'s
    /// OWN kink convention rather than `clamp`'s. PyTorch's `hardtanh_backward` tests
    /// `self <= min_val || self >= max_val` (zero there, at BOTH bounds, not just
    /// outside them), so the gradient passes through only strictly inside
    /// `(min_val, max_val)` -- unlike [`Self::clamp`], whose own gradient is inclusive of
    /// both bounds. Composed from three disjoint region masks (`<= min_val`, the open
    /// interior, `>= max_val`) rather than reusing `clamp`'s `Rule`, so the boundary
    /// convention is exact even when `min_val == max_val`. Rejects a non-finite bound or
    /// `min_val > max_val` before any device launch, matching `clamp`.
    pub fn hardtanh(&self, min_val: f32, max_val: f32) -> Result<Self> {
        if !min_val.is_finite() || !max_val.is_finite() {
            return Err("hardtanh bounds must be finite".into());
        }
        if min_val > max_val {
            return Err(format!(
                "hardtanh requires min_val <= max_val, got min_val={min_val} max_val={max_val}"
            )
            .into());
        }
        let mask_lo = self.le(min_val)?;
        let mask_mid = self.gt(min_val)?.logical_and(&self.lt(max_val)?)?;
        // `1 - mask_lo - mask_mid` rather than `self.ge(max_val)`: the two subtractions
        // stay exact (and mutually exclusive with `mask_lo`) even when `min_val ==
        // max_val`, where a literal `ge(max_val)` would double-count the single point
        // `x == min_val == max_val` against `mask_lo`.
        let mask_hi = mask_lo.logical_not()?.sub(&mask_mid)?;
        let low = mask_lo.scale(min_val)?;
        let mid = self.mul(&mask_mid)?;
        let high = mask_hi.scale(max_val)?;
        low.add(&mid)?.add(&high)
    }
    /// Elementwise `ReLU6`, PyTorch's `F.hardtanh(x, 0., 6.)`: [`Self::hardtanh`] with
    /// fixed bounds `0` and `6`, including its strictly-interior gradient at both kinks.
    pub fn relu6(&self) -> Result<Self> {
        self.hardtanh(0.0, 6.0)
    }
    /// Elementwise `Hardsigmoid`, PyTorch's piecewise-linear sigmoid approximation:
    /// `clamp(x + 3, 0, 6) / 6`. Backward matches PyTorch's `hardsigmoid_backward`:
    /// `grad / 6` strictly inside `(-3, 3)`, exactly `0` at and outside either bound
    /// (`self > -3 && self < 3`, both strict). Composed from an interior mask and a
    /// saturated-high mask rather than `clamp`, for the same boundary reason as
    /// [`Self::hardtanh`].
    pub fn hardsigmoid(&self) -> Result<Self> {
        let mask_hi = self.ge(3.0)?;
        let mask_mid = self.gt(-3.0)?.logical_and(&self.lt(3.0)?)?;
        let mid_value = self.scale(1.0 / 6.0)?.add(&self.constant(0.5)?)?;
        mask_mid.mul(&mid_value)?.add(&mask_hi)
    }
    /// Elementwise `Hardswish`, PyTorch's `x * relu6(x + 3) / 6`. Backward matches
    /// PyTorch's `hardswish_backward` exactly, including its asymmetric kinks: `0` for
    /// `x <= -3`, `grad * (x / 3 + 0.5)` strictly inside `(-3, 3)`, and `grad` (pass-through,
    /// derivative `1`) for `x >= 3` -- so, unlike [`Self::hardtanh`]/[`Self::hardsigmoid`],
    /// the two kinks are NOT symmetric: `x == -3` routes to the zero branch but `x == 3`
    /// routes to the pass-through branch, not the interior formula (whose limit there is
    /// `1.5`, not `1`). The interior region is composed as `x * (x + 3) / 6`, whose own
    /// derivative `x / 3 + 0.5` falls out of the product rule for free.
    pub fn hardswish(&self) -> Result<Self> {
        let mask_hi = self.ge(3.0)?;
        let mask_mid = self.gt(-3.0)?.logical_and(&self.lt(3.0)?)?;
        let mid_value = self
            .mul(&self.add(&self.constant(3.0)?)?)?
            .scale(1.0 / 6.0)?;
        mask_mid.mul(&mid_value)?.add(&mask_hi.mul(self)?)
    }
    /// Elementwise `Hardshrink`: `0` where `|x| <= lambd`, otherwise `x` (PyTorch's
    /// `torch.where(x.abs() <= lambd, 0, x)`). Backward matches PyTorch's shared
    /// `shrink_backward_kernel` (also behind [`Self::softshrink`]): `0` on the closed
    /// band `-lambd <= x <= lambd` (inclusive of both bounds), `grad` outside it. Composed
    /// as `x * outside_mask`, so this natural product-rule derivative already equals that
    /// target with no extra masking trick. Rejects a non-finite or negative `lambd`
    /// before any device launch.
    pub fn hardshrink(&self, lambd: f32) -> Result<Self> {
        if !lambd.is_finite() || lambd < 0.0 {
            return Err("hardshrink lambd must be finite and non-negative".into());
        }
        let inside = self.abs()?.le(lambd)?;
        self.mul(&inside.logical_not()?)
    }
    /// Elementwise `Softshrink`: `x - lambd` where `x > lambd`, `x + lambd` where
    /// `x < -lambd`, otherwise `0` on the closed band `-lambd <= x <= lambd` (PyTorch's
    /// `torch.where(x.abs() > lambd, x - x.sign() * lambd, 0)`). Backward shares
    /// [`Self::hardshrink`]'s exact `shrink_backward_kernel` convention: `0` on that same
    /// closed band, `grad` outside it. Composed from two directional shift terms (each
    /// `(x -+ lambd) * mask`) whose masks are already mutually exclusive and complementary
    /// to the zero band, so the natural derivative equals the target directly. Rejects a
    /// non-finite or negative `lambd` before any device launch.
    pub fn softshrink(&self, lambd: f32) -> Result<Self> {
        if !lambd.is_finite() || lambd < 0.0 {
            return Err("softshrink lambd must be finite and non-negative".into());
        }
        let mask_pos = self.gt(lambd)?;
        let mask_neg = self.lt(-lambd)?;
        let shifted_pos = self.sub(&self.constant(lambd)?)?.mul(&mask_pos)?;
        let shifted_neg = self.add(&self.constant(lambd)?)?.mul(&mask_neg)?;
        shifted_pos.add(&shifted_neg)
    }
    /// Elementwise `Threshold(threshold, value)`: `x` where `x > threshold`, otherwise the
    /// constant `value` (PyTorch's `torch.where(x <= threshold, value, x)`; note the
    /// `<=`/`>` split, shared by forward and backward alike). Backward matches PyTorch's
    /// `threshold_backward`: `grad` where `x > threshold`, exactly `0` at and below it.
    /// Composed as `x * mask + value * (1 - mask)`, so the natural product-rule derivative
    /// already equals that target with no extra masking trick. Rejects a non-finite
    /// `threshold` or `value` before any device launch.
    pub fn threshold(&self, threshold: f32, value: f32) -> Result<Self> {
        if !threshold.is_finite() || !value.is_finite() {
            return Err("threshold and value must both be finite".into());
        }
        let mask_hi = self.gt(threshold)?;
        let mask_lo = mask_hi.logical_not()?;
        self.mul(&mask_hi)?.add(&mask_lo.scale(value)?)
    }
    /// Elementwise `Softsign`, PyTorch's `x / (1 + x.abs())`. No true kink: although the
    /// forward composes through [`Self::abs`], its derivative `1 / (1 + x.abs())^2` is
    /// continuous through `x == 0` (both one-sided limits agree), and the ordinary
    /// composed chain rule already reproduces it exactly there because `abs`'s own
    /// backward is exactly zero at `x == 0` (its documented convention) rather than a
    /// one-sided slope, which is exactly the missing term the analytic derivative also
    /// drops at that point.
    pub fn softsign(&self) -> Result<Self> {
        let denominator = self.abs()?.add(&self.constant(1.0)?)?;
        self.div(&denominator)
    }
    /// Elementwise `Tanhshrink`, PyTorch's `x - x.tanh()`. Smooth everywhere; composed
    /// directly from [`Self::tanh`], so the chain rule gives the exact derivative
    /// `1 - (1 - tanh(x)^2) == tanh(x)^2` for free, with no dedicated rule.
    pub fn tanhshrink(&self) -> Result<Self> {
        self.sub(&self.tanh()?)
    }
    /// Shared plan for the group-sum kernel behind [`Tensor::mean`] and [`Tensor::sum`]:
    /// remove exactly `axes`, keep every other axis, and scale the summed contributions by
    /// `factor` (computed from the output/input element counts so callers can pick `1.0` for
    /// an exact sum or `output/input` for a mean). `discriminant` keeps the two ops' cached
    /// plans apart, since they share a shape/layout/axes key but not a factor.
    fn reduction_plans(
        &self,
        discriminant: u8,
        axes: &[Axis],
        factor: fn(usize, usize) -> f32,
    ) -> Result<ReductionPlans> {
        let key = (
            discriminant,
            self.shape().clone(),
            self.0.layout.clone(),
            axes.to_vec(),
        );
        let cached = REDUCTION_PLANS.with(|cache| cache.borrow().get(&key).cloned());
        if let Some(plans) = cached {
            return Ok(plans);
        }
        let output = Shape::new(
            self.shape()
                .dims()
                .iter()
                .copied()
                .filter(|d| !axes.contains(&d.axis)),
        )?;
        let layout = Layout::contiguous(&output);
        let factor = factor(output.len(), self.shape().len());
        let mut map = vec![0; self.shape().len()];
        for i in 0..self.shape().len() {
            let coords = self.shape().coords(i);
            let retained: Vec<_> = self
                .shape()
                .dims()
                .iter()
                .zip(&coords)
                .filter(|(d, _)| !axes.contains(&d.axis))
                .map(|(_, &coordinate)| coordinate)
                .collect();
            map[self.0.layout.offset(&coords)] = layout.offset(&retained);
        }
        let plans = ReductionPlans {
            output: output.clone(),
            layout,
            factor,
            forward: Rc::new(Plan::reverse(&map, output.len())?),
            reverse: Rc::new(Plan::gather(&map)?),
        };
        REDUCTION_PLANS.with(|cache| {
            cache.borrow_mut().insert(key, plans.clone());
        });
        Ok(plans)
    }
    /// Run the group-sum kernel for a [`reduction_plans`](Self::reduction_plans) plan and wrap
    /// the result in a graph node whose backward broadcasts the upstream gradient back across
    /// the reduced axes, scaled by the same `factor`.
    fn reduce_grouped(
        &self,
        discriminant: u8,
        axes: Vec<Axis>,
        factor: fn(usize, usize) -> f32,
        label: &str,
    ) -> Result<Self> {
        let started = Instant::now();
        let plans = self.reduction_plans(discriminant, &axes, factor)?;
        let value =
            self.device()
                .grouped(&self.0.value, None, plans.forward.as_ref(), plans.factor)?;
        let rule = Rule::Group {
            plan: plans.reverse,
            rhs: None,
            factor: plans.factor,
        };
        let result = Self::node(
            plans.output,
            plans.layout,
            value,
            self.device(),
            vec![Edge::new(self, rule)],
            false,
            None,
        );
        profile(label, started);
        Ok(result)
    }
    pub fn mean(&self, axes: impl IntoAxes) -> Result<Self> {
        let axes = self.shape().select_axes(axes)?;
        self.reduce_grouped(
            0,
            axes,
            |output, input| output as f32 / input as f32,
            "mean",
        )
    }
    /// Reduce precisely the named `axes` to their sum; every other axis is preserved
    /// unchanged. Backward broadcasts the upstream gradient across the reduced axes without
    /// scaling it, since each contributing element has unit local derivative. This runs the
    /// same group-sum kernel `mean` uses, with the constant scale `1.0` in place of `mean`'s
    /// `1 / extent` — the two are the same primitive, not a division after the fact.
    pub fn sum(&self, axes: impl IntoAxes) -> Result<Self> {
        let axes = self.shape().select_axes(axes)?;
        self.reduce_grouped(2, axes, |_, _| 1.0, "sum")
    }
    /// Population mean and variance over exactly the declared named axes.
    pub fn moments(&self, axes: impl IntoAxes) -> Result<(Self, Self)> {
        let axes = self.shape().select_axes(axes)?;
        if axes.is_empty() {
            return Err("moments requires at least one axis".into());
        }
        let mean = self.mean(axes.clone())?;
        let centered = self.sub(&mean)?;
        let variance = centered.mul(&centered)?.mean(axes)?;
        Ok((mean, variance))
    }
    /// Population mean of squares over exactly the declared named axes.
    pub fn mean_square(&self, axes: impl IntoAxes) -> Result<Self> {
        let axes = self.shape().select_axes(axes)?;
        if axes.is_empty() {
            return Err("mean_square requires at least one axis".into());
        }
        self.mul(self)?.mean(axes)
    }
    /// Reduce one named axis to its minimum finite value.
    ///
    /// Non-finite candidates are ignored. A group without a finite candidate returns NaN and
    /// has zero derivative even if its upstream derivative is non-finite. Ties route the
    /// derivative to the first logical coordinate along `axis`, independently of physical
    /// layout.
    pub fn min(&self, axis: Axis) -> Result<Self> {
        let started = Instant::now();
        let reduced_index = self.shape().index(axis)?;
        let extent = self.extent(axis)?;
        let key = (1, self.shape().clone(), self.0.layout.clone(), vec![axis]);
        let cached = REDUCTION_PLANS.with(|cache| cache.borrow().get(&key).cloned());
        let plans = match cached {
            Some(plans) => plans,
            None => {
                let output = Shape::new(
                    self.shape()
                        .dims()
                        .iter()
                        .copied()
                        .filter(|d| d.axis != axis),
                )?;
                let layout = Layout::contiguous(&output);
                let mut map = vec![0; self.shape().len()];
                let mut groups = Vec::with_capacity(output.len());
                for output_index in 0..output.len() {
                    let output_coords = output.coords(output_index);
                    let mut group = Vec::with_capacity(extent);
                    for coordinate in 0..extent {
                        let mut input_coords = output_coords.clone();
                        input_coords.insert(reduced_index, coordinate);
                        let physical = self.0.layout.offset(&input_coords);
                        map[physical] = output_index;
                        group.push((physical, 0));
                    }
                    groups.push(group);
                }
                let plans = ReductionPlans {
                    output: output.clone(),
                    layout,
                    factor: 1.0,
                    forward: Rc::new(Plan::groups(groups, false)?),
                    reverse: Rc::new(Plan::gather(&map)?),
                };
                REDUCTION_PLANS.with(|cache| {
                    cache.borrow_mut().insert(key, plans.clone());
                });
                plans
            }
        };
        let (value, winners) = self.device().grouped_minimum(
            &self.0.value,
            plans.forward.as_ref(),
            plans.reverse.as_ref(),
        )?;
        let result = Self::node(
            plans.output,
            plans.layout,
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::Minimum {
                    plan: plans.reverse,
                    winners,
                },
            )],
            false,
            None,
        );
        profile("min", started);
        Ok(result)
    }
    /// Reduce three named spatial axes to a fixed target extent each, using PyTorch's adaptive
    /// average-pooling bin formula per axis: `start = floor(i * input / output)`,
    /// `end = ceil((i + 1) * input / output)`. Every other axis (such as batch or channel) is
    /// preserved. When an axis's input extent does not divide evenly by its target extent,
    /// adjacent bins can share one boundary element; that element is averaged, at full weight,
    /// into each bin it falls in, exactly as `F.adaptive_avg_pool3d` computes it. Unlike a fixed
    /// kernel/stride pool, bins have no padding: every bin is a nonempty range of real input
    /// coordinates.
    pub fn adaptive_avg_pool3d(&self, spatial: [Axis; 3], target: [usize; 3]) -> Result<Self> {
        self.adaptive_avg_pool(spatial, target)
    }
    fn adaptive_avg_pool<const N: usize>(
        &self,
        spatial: [Axis; N],
        target: [usize; N],
    ) -> Result<Self> {
        let started = Instant::now();
        let name = format!("adaptive_avg_pool{N}d");
        if !(N == 2 || N == 3) {
            return Err(
                "Axis adaptive average pooling supports exactly two or three spatial axes".into(),
            );
        }
        for (index, &axis) in spatial.iter().enumerate() {
            if spatial[..index].contains(&axis) {
                return Err(format!("{name} requires distinct spatial axes").into());
            }
        }
        if target.contains(&0) {
            return Err(format!("{name} target extents must be positive").into());
        }
        let mut bins: [Vec<(usize, usize)>; N] = std::array::from_fn(|_| Vec::new());
        for index in 0..N {
            let input_extent = self.extent(spatial[index])?;
            let out_extent = target[index];
            let mut axis_bins = Vec::with_capacity(out_extent);
            for i in 0..out_extent {
                let start = i * input_extent / out_extent;
                let end = ((i + 1) * input_extent).div_ceil(out_extent);
                axis_bins.push((start, end));
            }
            bins[index] = axis_bins;
        }
        let dims: Vec<_> = self
            .shape()
            .dims()
            .iter()
            .map(|dim| {
                spatial
                    .iter()
                    .position(|&axis| axis == dim.axis)
                    .map_or(*dim, |index| dim.axis.of(target[index]))
            })
            .collect();
        let output = Shape::new(dims)?;
        let layout = Layout::contiguous(&output);
        let mut spatial_positions = [0usize; N];
        for index in 0..N {
            spatial_positions[index] = self.shape().index(spatial[index])?;
        }
        let mut forward_groups: Vec<Vec<(usize, usize)>> = Vec::with_capacity(output.len());
        let mut reverse_groups: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.shape().len()];
        let mut weights = Vec::with_capacity(output.len());
        for output_index in 0..output.len() {
            let output_coords = output.coords(output_index);
            let mut ranges = [(0usize, 0usize); N];
            let mut extents = [0usize; N];
            for index in 0..N {
                ranges[index] = bins[index][output_coords[spatial_positions[index]]];
                extents[index] = ranges[index].1 - ranges[index].0;
            }
            let window_len: usize = extents.iter().product();
            let mut group = Vec::with_capacity(window_len);
            for w in 0..window_len {
                let mut remainder = w;
                let mut offsets = [0usize; N];
                for index in (0..N).rev() {
                    offsets[index] = remainder % extents[index];
                    remainder /= extents[index];
                }
                let mut input_coords = output_coords.clone();
                for index in 0..N {
                    input_coords[spatial_positions[index]] = ranges[index].0 + offsets[index];
                }
                let physical = self.0.layout.offset(&input_coords);
                group.push((physical, output_index));
                reverse_groups[physical].push((output_index, output_index));
            }
            weights.push(1.0 / window_len as f32);
            forward_groups.push(group);
        }
        let weight_axis = Axis::new("adaptive_avg_pool_weight");
        let weights = Self::from_slice(&weights, [weight_axis.of(output.len())], self.device())?;
        let forward_plan = Plan::groups(forward_groups, true)?;
        let reverse_plan = Rc::new(Plan::groups(reverse_groups, true)?);
        let value =
            self.device()
                .grouped(&self.0.value, Some(&weights.0.value), &forward_plan, 1.0)?;
        let result = Self::node(
            output,
            layout,
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::Group {
                    plan: reverse_plan,
                    rhs: Some(weights.0.value.clone()),
                    factor: 1.0,
                },
            )],
            false,
            None,
        );
        profile(&name, started);
        Ok(result)
    }
    /// Reduce one named axis to its maximum finite value.
    ///
    /// Non-finite candidates are ignored. A group without a finite candidate returns NaN and
    /// has zero derivative even if its upstream derivative is non-finite. Ties route the
    /// derivative to the first logical coordinate along `axis`, independently of physical
    /// layout. Exact mirror of [`Self::min`].
    pub fn max(&self, axis: Axis) -> Result<Self> {
        let started = Instant::now();
        let reduced_index = self.shape().index(axis)?;
        let extent = self.extent(axis)?;
        let key = (1, self.shape().clone(), self.0.layout.clone(), vec![axis]);
        let cached = REDUCTION_PLANS.with(|cache| cache.borrow().get(&key).cloned());
        let plans = match cached {
            Some(plans) => plans,
            None => {
                let output = Shape::new(
                    self.shape()
                        .dims()
                        .iter()
                        .copied()
                        .filter(|d| d.axis != axis),
                )?;
                let layout = Layout::contiguous(&output);
                let mut map = vec![0; self.shape().len()];
                let mut groups = Vec::with_capacity(output.len());
                for output_index in 0..output.len() {
                    let output_coords = output.coords(output_index);
                    let mut group = Vec::with_capacity(extent);
                    for coordinate in 0..extent {
                        let mut input_coords = output_coords.clone();
                        input_coords.insert(reduced_index, coordinate);
                        let physical = self.0.layout.offset(&input_coords);
                        map[physical] = output_index;
                        group.push((physical, 0));
                    }
                    groups.push(group);
                }
                let plans = ReductionPlans {
                    output: output.clone(),
                    layout,
                    factor: 1.0,
                    forward: Rc::new(Plan::groups(groups, false)?),
                    reverse: Rc::new(Plan::gather(&map)?),
                };
                REDUCTION_PLANS.with(|cache| {
                    cache.borrow_mut().insert(key, plans.clone());
                });
                plans
            }
        };
        let (value, winners) = self.device().grouped_maximum(
            &self.0.value,
            plans.forward.as_ref(),
            plans.reverse.as_ref(),
        )?;
        let result = Self::node(
            plans.output,
            plans.layout,
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::Maximum {
                    plan: plans.reverse,
                    winners,
                },
            )],
            false,
            None,
        );
        profile("max", started);
        Ok(result)
    }
    /// Numerically stable log-sum-exp reduction over one named axis: `ln(sum(exp(x)))`
    /// computed as `m + ln(sum(exp(x - m)))` with `m = self.max(axis)`, matching
    /// `torch.logsumexp`. `m` is [`Self::detach`]ed before the subtraction, so no gradient
    /// flows back through it; differentiating the remaining composition by hand collapses
    /// to exactly `softmax(x)` along `axis` (`exp(x - m) / sum(exp(x - m))`), independent of
    /// `m`'s own derivative -- the standard stabilizing-shift trick. Composed entirely from
    /// [`Self::max`], [`Self::sub`], [`Self::exp`], [`Self::sum`], [`Self::ln`], and
    /// [`Self::add`] -- no dedicated kernel or backward rule -- so even a row whose entries
    /// sit near a large shared magnitude (e.g. near 1000) never exponentiates anything
    /// larger than zero and cannot overflow. Shares `max`'s axis-removal contract and its
    /// no-finite-candidate convention: a group with every entry non-finite gives `m = NaN`,
    /// so `logsumexp` returns `NaN` there too, unlike PyTorch's `torch.logsumexp`, which
    /// returns `-infinity` for an all-`-infinity` group. Unlike `max`, this composition has
    /// no dedicated backward rule to zero that group's gradient, so `NaN` also propagates
    /// through it ordinarily, rather than landing on zero.
    pub fn logsumexp(&self, axis: Axis) -> Result<Self> {
        let shift = self.max(axis)?.detach();
        self.sub(&shift)?.exp()?.sum(axis)?.ln()?.add(&shift)
    }
    /// Reduce a tensor to the mean of elements selected by a constant binary mask.
    /// The mask must have the same named axes, and every axis must be reduced.
    pub fn masked_mean(&self, mask: &Self) -> Result<Self> {
        if mask.requires_grad() {
            return Err("masked_mean mask cannot require gradients".into());
        }
        if self.shape().rank() != mask.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&axis| !mask.shape().contains(axis))
        {
            return Err("masked_mean requires identical axis sets".into());
        }
        self.compatible_device(mask)?;
        self.shared_extents(mask)?;
        let mask = mask.align(self.shape())?;
        let values = mask.to_vec()?;
        if values
            .iter()
            .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
        {
            return Err("masked_mean mask must contain only finite zero or one values".into());
        }
        let selected = values.iter().filter(|&&value| value == 1.0).count();
        if selected == 0 {
            return Err("masked_mean mask selects no elements".into());
        }
        self.mul(&mask)?
            .mean(self.shape().axes())?
            .scale(self.shape().len() as f32 / selected as f32)
    }
    /// Softmax over one named `axis`, with positions `mask` marks `0.0` receiving
    /// exactly zero probability and exactly zero gradient, and a group whose mask
    /// is entirely zero along `axis` returning all zeros for that group instead of
    /// `NaN`. `mask` is a constant `{0.0, 1.0}` tensor sharing every one of
    /// `self`'s named axes, validated exactly like [`Self::masked_mean`]'s mask
    /// (finite `0`/`1` values only, rejected if it requires gradients) -- except
    /// an all-zero *group* along `axis` is the expected fully-masked case here,
    /// not an error the way an entirely empty mask is for `masked_mean`.
    ///
    /// This is the named-axis equivalent of
    /// `torch.softmax(score.masked_fill(~keep, -inf), dim)` followed by
    /// `torch.nan_to_num(..., neginf=0.0)` for a fully-masked row -- the exact
    /// pair every gated-attention consumer with a variable-length validity mask
    /// (`InterfaceMIL`, `PhaseSeparableFusion`) otherwise repeats by hand. Folding
    /// the `nan_to_num` cleanup into the op's own contract, rather than leaving a
    /// `NaN` for the caller to catch, is a deliberate difference from PyTorch's
    /// two-call composition.
    ///
    /// Composed as `softmax(self + (1 - mask) * MASKED_SOFTMAX_OFFSET) *
    /// any_valid`, where `any_valid` (`mask.max(axis)`) is `1.0` for a group with
    /// any valid position and `0.0` for a fully-masked one, and
    /// `MASKED_SOFTMAX_OFFSET` is a large *finite* negative constant rather than
    /// literal `-inf`. An actually-infinite offset would make a fully-masked
    /// row's own maximum `-inf` too, so the stable-softmax step computes
    /// `exp(-inf - -inf)`, i.e. `NaN`, for every position in that row -- and
    /// `NaN * 0.0` is still `NaN`, so the trailing `any_valid` multiply could
    /// never clean it up. The finite offset keeps every intermediate value
    /// finite: a masked position's probability underflows `exp` to exactly
    /// `0.0f32` once its row has any valid position (the offset sits far past
    /// `f32`'s roughly -104 underflow threshold at any realistic score
    /// magnitude, with wide margin below overflowing the addition itself to
    /// `-inf`), and a fully-masked row instead computes an ordinary finite (if
    /// meaningless) distribution that the trailing multiply by `any_valid` zeroes
    /// out cleanly, since it is never `NaN`.
    ///
    /// Gradient into `self` at a masked position is exactly zero either way:
    /// softmax's backward rule scales the upstream gradient by the
    /// (exactly-zero) probability there, so it does not matter that the offset's
    /// own local derivative into `self` is nominally `1.0`. A fully-masked
    /// group's gradient is exactly zero throughout, for the same reason applied
    /// to the `any_valid` multiply: it zeros the upstream gradient into that
    /// group's raw softmax before the softmax backward rule ever runs.
    pub fn masked_softmax(&self, axis: Axis, mask: &Self) -> Result<Self> {
        const MASKED_SOFTMAX_OFFSET: f32 = -1.0e9;
        if mask.requires_grad() {
            return Err("masked_softmax mask cannot require gradients".into());
        }
        if self.shape().rank() != mask.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&own_axis| !mask.shape().contains(own_axis))
        {
            return Err("masked_softmax requires identical axis sets".into());
        }
        self.compatible_device(mask)?;
        self.shared_extents(mask)?;
        let mask = mask.align(self.shape())?;
        let values = mask.to_vec()?;
        if values
            .iter()
            .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
        {
            return Err("masked_softmax mask must contain only finite zero or one values".into());
        }
        let offset = mask.logical_not()?.scale(MASKED_SOFTMAX_OFFSET)?;
        let any_valid = mask.max(axis)?;
        self.add(&offset)?.softmax(axis)?.mul(&any_valid)
    }
    /// Mask key positions greater than query positions with -infinity.
    /// This is square, zero-offset self-attention; cached/offset attention is not supported.
    pub fn causal_mask(&self, query: Axis, key: Axis) -> Result<Self> {
        self.prefix_causal_mask(query, key, 0)
    }
    /// Causal mask whose first `prefix` key positions stay visible to every query.
    /// A block of memory or code tokens placed before the sequence is attended
    /// freely, and the remaining positions are strictly causal. This is a
    /// logical dense mask, not a sparse kernel; `prefix == 0` is `causal_mask`.
    pub fn prefix_causal_mask(&self, query: Axis, key: Axis, prefix: usize) -> Result<Self> {
        if query == key || self.extent(query)? != self.extent(key)? {
            return Err("causal_mask requires distinct query/key axes of equal extent".into());
        }
        if prefix > self.extent(key)? {
            return Err("prefix_causal_mask prefix exceeds the key extent".into());
        }
        let q = self.shape().index(query)?;
        let k = self.shape().index(key)?;
        Plan::check_size(self.shape().len())?;
        let mut keep = vec![0.0; self.shape().len()];
        for i in 0..self.shape().len() {
            let coords = self.shape().coords(i);
            keep[self.0.layout.offset(&coords)] =
                f32::from(coords[k] <= coords[q] || coords[k] < prefix);
        }
        let keep = self.device().upload(keep)?;
        let value = self.device().mask(&self.0.value, &keep)?;
        Ok(Self::node(
            self.shape().clone(),
            self.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(self, Rule::Multiply(keep))],
            false,
            None,
        ))
    }
    /// Numerically stable normalization along one named axis, preserving the logical shape.
    /// Rows must contain finite values or -infinity and at least one finite value.
    pub fn softmax(&self, axis: Axis) -> Result<Self> {
        let width = self.extent(axis)?;
        let mut physical = self.shape().axes();
        physical.retain(|&a| a != axis);
        physical.push(axis);
        let input = self.with_layout(physical)?;
        let value = self.device().softmax(&input.0.value, width)?;
        let rule = Rule::Softmax {
            probability: value.clone(),
            width,
        };
        Ok(Self::node(
            self.shape().clone(),
            input.0.layout.clone(),
            value,
            self.device(),
            vec![Edge::new(&input, rule)],
            false,
            None,
        ))
    }
    /// Extract valid stride-one 2D patches, replacing channels with one flattened patch axis.
    /// Reverse mode scatters and sums overlapping patch contributions into the input.
    pub fn unfold2d(
        &self,
        channels: Axis,
        spatial: [Axis; 2],
        patch: Dim,
        kernel: [usize; 2],
    ) -> Result<Self> {
        self.unfold_configured(channels, spatial, None, patch, kernel, [1, 1], [0, 0], 0.0)
    }

    /// Lower a configured 2D or 3D convolution or pooling op to grouped patches. The group
    /// and patch axes are appended so merging grouped output channels restores
    /// convolution's public rule that output channels are the final logical axis.
    /// `fill` is written at every patch position outside the input (padding); convolution
    /// passes `0.0`, and windowed max pooling passes `f32::NEG_INFINITY` so a padded position
    /// can never win the reduction.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn unfold_grouped<const N: usize>(
        &self,
        channels: Axis,
        spatial: [Axis; N],
        group: Dim,
        patch: Dim,
        kernel: [usize; N],
        stride: [usize; N],
        padding: [usize; N],
        fill: f32,
    ) -> Result<Self> {
        self.unfold_configured(
            channels,
            spatial,
            Some(group),
            patch,
            kernel,
            stride,
            padding,
            fill,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn unfold_configured<const N: usize>(
        &self,
        channels: Axis,
        spatial: [Axis; N],
        group: Option<Dim>,
        patch: Dim,
        kernel: [usize; N],
        stride: [usize; N],
        padding: [usize; N],
        fill: f32,
    ) -> Result<Self> {
        let name = format!("unfold{N}d");
        if !fill.is_finite() && fill != f32::NEG_INFINITY {
            return Err(format!("{name} fill must be finite or negative infinity").into());
        }
        if !(N == 2 || N == 3) {
            return Err("Axis unfold supports exactly two or three spatial axes".into());
        }
        for (index, &axis) in spatial.iter().enumerate() {
            if axis == channels || spatial[..index].contains(&axis) {
                return Err(format!("{name} requires distinct channel and spatial axes").into());
            }
        }
        if kernel.contains(&0) {
            return Err(format!("{name} kernel extents must be positive").into());
        }
        if stride.contains(&0) {
            return Err(format!("{name} stride extents must be positive").into());
        }
        let channel_extent = self.extent(channels)?;
        let mut input_spatial = [0; N];
        for index in 0..N {
            input_spatial[index] = self.extent(spatial[index])?;
        }
        let groups = group.map_or(1, |dim| dim.extent);
        if groups == 0 {
            return Err(format!("{name} groups must be positive").into());
        }
        if !channel_extent.is_multiple_of(groups) {
            return Err(format!("{name} channels must be divisible by groups").into());
        }
        let mut output_spatial = [0; N];
        for index in 0..N {
            let doubled_padding = padding[index]
                .checked_mul(2)
                .ok_or_else(|| format!("{name} padding overflow"))?;
            let padded = input_spatial[index]
                .checked_add(doubled_padding)
                .ok_or_else(|| format!("{name} padded spatial extent overflow"))?;
            if padded > i32::MAX as usize {
                return Err(format!(
                    "{name} padded spatial extent exceeds the i32 kernel index range"
                )
                .into());
            }
            if kernel[index] > padded {
                return Err(format!("{name} kernel exceeds the padded spatial extent").into());
            }
            output_spatial[index] = (padded - kernel[index]) / stride[index] + 1;
        }
        let channels_per_group = channel_extent / groups;
        let expected_patch = kernel
            .iter()
            .try_fold(channels_per_group, |extent, &kernel| {
                extent.checked_mul(kernel)
            });
        let expected_patch =
            expected_patch.ok_or_else(|| format!("{name} patch extent overflow"))?;
        if patch.extent != expected_patch {
            return Err(format!(
                "{name} patch extent must equal channels per group * kernel volume"
            )
            .into());
        }

        let mut dims = Vec::with_capacity(self.shape().rank() + usize::from(group.is_some()));
        for dim in self.shape().dims() {
            if dim.axis == channels {
                if group.is_none() {
                    dims.push(patch);
                }
            } else if let Some(index) = spatial.iter().position(|&axis| axis == dim.axis) {
                dims.push(dim.axis.of(output_spatial[index]));
            } else {
                dims.push(*dim);
            }
        }
        if let Some(group) = group {
            dims.extend([group, patch]);
        }
        let shape = Shape::new(dims)?;
        let mut spatial_key = [None; 3];
        let mut kernel_key = [1; 3];
        let mut stride_key = [1; 3];
        let mut padding_key = [0; 3];
        for index in 0..N {
            spatial_key[index] = Some(spatial[index]);
            kernel_key[index] = kernel[index];
            stride_key[index] = stride[index];
            padding_key[index] = padding[index];
        }
        let key = UnfoldPlanKey {
            input_shape: self.shape().clone(),
            input_layout: self.0.layout.clone(),
            output_shape: shape.clone(),
            channels,
            spatial_rank: N,
            spatial: spatial_key,
            kernel: kernel_key,
            stride: stride_key,
            padding: padding_key,
            group,
            fill_bits: fill.to_bits(),
        };
        let mut output_order = vec![];
        if let Some(group) = group {
            output_order.push(group.axis);
        }
        output_order.extend(
            shape
                .axes()
                .into_iter()
                .filter(|axis| Some(*axis) != group.map(|dim| dim.axis) && *axis != patch.axis),
        );
        output_order.push(patch.axis);
        let output_layout = Layout::new(&shape, &output_order)?;
        const TILE_TAIL: usize = 127;
        if self.shape().len() > i32::MAX as usize - TILE_TAIL
            || shape.len() > i32::MAX as usize - TILE_TAIL
        {
            return Err(
                format!("{name} tensor is too large for its 128-lane index arithmetic").into(),
            );
        }
        if let Some(plans) = UNFOLD_PLANS.with(|cache| cache.borrow().get(&key).cloned()) {
            return match plans {
                UnfoldPlans::Implicit(spec) => self.unfolded_with_spec(shape, output_layout, spec),
                UnfoldPlans::Zero => self.zero_gathered(shape, output_layout),
            };
        }

        let has_input = (0..N).all(|dimension| {
            (0..output_spatial[dimension]).any(|output| {
                (0..kernel[dimension]).any(|offset| {
                    (output * stride[dimension] + offset)
                        .checked_sub(padding[dimension])
                        .is_some_and(|input| input < input_spatial[dimension])
                })
            })
        });
        if !has_input {
            let result = self.zero_gathered(shape, output_layout)?;
            UNFOLD_PLANS.with(|cache| {
                cache.borrow_mut().insert(key, UnfoldPlans::Zero);
            });
            #[cfg(test)]
            UNFOLD_PLAN_BUILDS.with(|builds| builds.set(builds.get() + 1));
            return Ok(result);
        }

        let to_i32 = |value: usize| -> Result<i32> { Ok(i32::try_from(value)?) };
        let mut forward_metadata = Vec::with_capacity(shape.rank() * 4);
        for (index, dim) in shape.dims().iter().enumerate() {
            let (input_stride, role) =
                if let Some(spatial_index) = spatial.iter().position(|&axis| axis == dim.axis) {
                    (0, to_i32(spatial_index + 1)?)
                } else if group.is_some_and(|group| dim.axis == group.axis) {
                    (0, to_i32(N + 1)?)
                } else if dim.axis == patch.axis {
                    (0, to_i32(N + 2)?)
                } else {
                    let input_index = self.shape().index(dim.axis)?;
                    (to_i32(self.0.layout.strides[input_index])?, 0)
                };
            forward_metadata.extend([
                to_i32(dim.extent)?,
                to_i32(output_layout.strides[index])?,
                input_stride,
                role,
            ]);
        }
        let mut backward_metadata = Vec::with_capacity(self.shape().rank() * 4);
        for (index, dim) in self.shape().dims().iter().enumerate() {
            let (output_stride, role) = if dim.axis == channels {
                (0, 1)
            } else if let Some(spatial_index) = spatial.iter().position(|&axis| axis == dim.axis) {
                (0, to_i32(spatial_index + 2)?)
            } else {
                let output_index = shape.index(dim.axis)?;
                (to_i32(output_layout.strides[output_index])?, 0)
            };
            backward_metadata.extend([
                to_i32(dim.extent)?,
                to_i32(self.0.layout.strides[index])?,
                output_stride,
                role,
            ]);
        }
        let input_stride =
            |axis| -> Result<i32> { to_i32(self.0.layout.strides[self.shape().index(axis)?]) };
        let output_stride =
            |axis| -> Result<i32> { to_i32(output_layout.strides[shape.index(axis)?]) };
        let mut kernel_spec = [1; 3];
        let mut stride_spec = [1; 3];
        let mut padding_spec = [0; 3];
        let mut input_spatial_spec = [1; 3];
        let mut output_spatial_spec = [1; 3];
        let mut input_special_strides = [0; 4];
        let mut output_special_strides = [0; 5];
        input_special_strides[0] = input_stride(channels)?;
        for index in 0..N {
            kernel_spec[index] = to_i32(kernel[index])?;
            stride_spec[index] = to_i32(stride[index])?;
            padding_spec[index] = to_i32(padding[index])?;
            input_spatial_spec[index] = to_i32(input_spatial[index])?;
            output_spatial_spec[index] = to_i32(output_spatial[index])?;
            input_special_strides[index + 1] = input_stride(spatial[index])?;
            output_special_strides[index] = output_stride(spatial[index])?;
        }
        output_special_strides[3] = group.map_or(Ok(0), |group| output_stride(group.axis))?;
        output_special_strides[4] = output_stride(patch.axis)?;
        let spec = Rc::new(UnfoldSpec {
            spatial_rank: to_i32(N)?,
            input_len: self.shape().len(),
            output_len: shape.len(),
            input_rank: to_i32(self.shape().rank())?,
            output_rank: to_i32(shape.rank())?,
            forward_metadata,
            backward_metadata,
            channels_per_group: to_i32(channels_per_group)?,
            fill,
            kernel: kernel_spec,
            stride: stride_spec,
            padding: padding_spec,
            input_spatial: input_spatial_spec,
            output_spatial: output_spatial_spec,
            input_special_strides,
            output_special_strides,
        });
        let result = self.unfolded_with_spec(shape, output_layout, spec.clone())?;
        UNFOLD_PLANS.with(|cache| {
            cache
                .borrow_mut()
                .insert(key, UnfoldPlans::Implicit(spec.clone()));
        });
        #[cfg(test)]
        {
            UNFOLD_PLAN_BUILDS.with(|builds| builds.set(builds.get() + 1));
            UNFOLD_PLAN_METADATA_MAX
                .with(|maximum| maximum.set(maximum.get().max(spec.metadata_len())));
        }
        Ok(result)
    }
    /// Sum the named shared axes; align remaining shared axes, retain distinct ones.
    pub fn contract(&self, rhs: &Self, axes: impl IntoAxes) -> Result<Self> {
        self.compatible_device(rhs)?;
        self.shared_extents(rhs)?;
        let axes = self.shape().select_axes(axes)?;
        rhs.shape().select_axes(axes.clone())?;
        let reduction = Shape::new(
            axes.iter()
                .map(|&a| a.of(self.extent(a).expect("validated"))),
        )?;
        let mut dims: Vec<_> = self
            .shape()
            .dims()
            .iter()
            .copied()
            .filter(|d| !axes.contains(&d.axis))
            .collect();
        for &d in rhs.shape().dims() {
            if !axes.contains(&d.axis) && !dims.iter().any(|e| e.axis == d.axis) {
                dims.push(d);
            }
        }
        let shape = Shape::new(dims)?;
        if axes.len() == 1 {
            let batch_axes: Vec<_> = self
                .shape()
                .axes()
                .into_iter()
                .filter(|axis| !axes.contains(axis) && rhs.shape().contains(*axis))
                .collect();
            let left_axes: Vec<_> = self
                .shape()
                .axes()
                .into_iter()
                .filter(|axis| !axes.contains(axis) && !rhs.shape().contains(*axis))
                .collect();
            let right_axes: Vec<_> = rhs
                .shape()
                .axes()
                .into_iter()
                .filter(|axis| !axes.contains(axis) && !self.shape().contains(*axis))
                .collect();
            let left_order: Vec<_> = batch_axes
                .iter()
                .chain(&left_axes)
                .chain(&axes)
                .copied()
                .collect();
            let right_order: Vec<_> = batch_axes
                .iter()
                .chain(&axes)
                .chain(&right_axes)
                .copied()
                .collect();
            let ordered = |tensor: &Self, order: &[Axis]| -> Result<Self> {
                let layout = Layout::new(tensor.shape(), order)?;
                if tensor.0.layout.strides == layout.strides {
                    Ok(tensor.clone())
                } else {
                    tensor.with_layout(order.to_vec())
                }
            };
            let left = ordered(self, &left_order)?;
            let right = ordered(rhs, &right_order)?;
            let k = reduction.len();
            let batch = batch_axes
                .iter()
                .map(|axis| self.extent(*axis).expect("validated shared axis"))
                .product();
            let m = left_axes
                .iter()
                .map(|axis| self.extent(*axis).expect("validated left axis"))
                .product();
            let n = right_axes
                .iter()
                .map(|axis| rhs.extent(*axis).expect("validated right axis"))
                .product();
            let value = self
                .device()
                .matmul(&left.0.value, &right.0.value, batch, m, k, n)?;
            let mut edges = vec![];
            if left.requires_grad() {
                edges.push(Edge::new(
                    &left,
                    Rule::MatmulLeft {
                        rhs: right.0.value.clone(),
                        batch,
                        m,
                        k,
                        n,
                    },
                ));
            }
            if right.requires_grad() {
                edges.push(Edge::new(
                    &right,
                    Rule::MatmulRight {
                        lhs: left.0.value.clone(),
                        batch,
                        m,
                        k,
                        n,
                    },
                ));
            }
            let output_order: Vec<_> = batch_axes
                .iter()
                .chain(&left_axes)
                .chain(&right_axes)
                .copied()
                .collect();
            return Ok(Self::node(
                shape.clone(),
                Layout::new(&shape, &output_order)?,
                value,
                self.device(),
                edges,
                false,
                None,
            ));
        }
        Plan::check_size(
            shape
                .len()
                .checked_mul(reduction.len())
                .ok_or("contraction size overflow")?,
        )?;
        let positions = |s: &Shape| -> Vec<(bool, usize)> {
            s.axes()
                .iter()
                .map(|a| {
                    if let Ok(i) = reduction.index(*a) {
                        (true, i)
                    } else {
                        (false, shape.index(*a).expect("retained axis"))
                    }
                })
                .collect()
        };
        let lp = positions(self.shape());
        let rp = positions(rhs.shape());
        let mut groups = vec![vec![]; shape.len()];
        let mut dl = vec![
            vec![];
            if self.requires_grad() {
                self.shape().len()
            } else {
                0
            }
        ];
        let mut dr = vec![
            vec![];
            if rhs.requires_grad() {
                rhs.shape().len()
            } else {
                0
            }
        ];
        for (o, group) in groups.iter_mut().enumerate() {
            let output = shape.coords(o);
            for k in 0..reduction.len() {
                let red = reduction.coords(k);
                let offset = |p: &[(bool, usize)], layout: &Layout| {
                    layout.offset(
                        &p.iter()
                            .map(|&(r, i)| if r { red[i] } else { output[i] })
                            .collect::<Vec<_>>(),
                    )
                };
                let l = offset(&lp, &self.0.layout);
                let r = offset(&rp, &rhs.0.layout);
                group.push((l, r));
                if self.requires_grad() {
                    dl[l].push((o, r));
                }
                if rhs.requires_grad() {
                    dr[r].push((o, l));
                }
            }
        }
        let value = self.device().grouped(
            &self.0.value,
            Some(&rhs.0.value),
            &Plan::groups(groups, true)?,
            1.0,
        )?;
        let mut edges = vec![];
        if self.requires_grad() {
            edges.push(Edge::new(
                self,
                Rule::Group {
                    plan: Rc::new(Plan::groups(dl, true)?),
                    rhs: Some(rhs.0.value.clone()),
                    factor: 1.0,
                },
            ));
        }
        if rhs.requires_grad() {
            edges.push(Edge::new(
                rhs,
                Rule::Group {
                    plan: Rc::new(Plan::groups(dr, true)?),
                    rhs: Some(self.0.value.clone()),
                    factor: 1.0,
                },
            ));
        }
        Ok(Self::node(
            shape.clone(),
            Layout::contiguous(&shape),
            value,
            self.device(),
            edges,
            false,
            None,
        ))
    }
    /// Explicit expansion: shared axes still align, but no axes are reduced.
    pub fn outer(&self, rhs: &Self) -> Result<Self> {
        self.contract(rhs, [])
    }
    pub fn rename(&self, from: Axis, to: Axis) -> Result<Self> {
        self.shape().index(from)?;
        let shape = Shape::new(
            self.shape()
                .dims()
                .iter()
                .map(|d| if d.axis == from { to.of(d.extent) } else { *d }),
        )?;
        Ok(Self::node(
            shape,
            self.0.layout.clone(),
            self.0.value.clone(),
            self.device(),
            vec![Edge::new(self, Rule::Identity)],
            false,
            None,
        ))
    }
    /// Add exact zeros before and after one named axis, preserving logical axis order.
    /// Zero padding on both sides shares the original storage. Otherwise the result
    /// is materialized on-device using rank-sized geometry, including for backward.
    pub fn pad_zeros(&self, axis: Axis, before: usize, after: usize) -> Result<Self> {
        let extent = self.extent(axis)?;
        let padded = extent
            .checked_add(before)
            .and_then(|n| n.checked_add(after))
            .ok_or("padding extent overflow")?;
        self.window(axis, padded, -i32::try_from(before)?)
    }

    /// Keep a contiguous, nonempty interval of one named axis without removing it.
    /// Reject out-of-range or overflowing intervals. A full-axis interval shares
    /// storage; other intervals use device-side copies and zero-scattered gradients.
    pub fn narrow(&self, axis: Axis, start: usize, length: usize) -> Result<Self> {
        let extent = self.extent(axis)?;
        let end = start
            .checked_add(length)
            .ok_or("narrow interval overflow")?;
        if length == 0 || end > extent {
            return Err("narrow requires a nonempty interval inside the named axis".into());
        }
        self.window(axis, length, i32::try_from(start)?)
    }

    fn window(&self, axis: Axis, extent: usize, shift: i32) -> Result<Self> {
        let dimension = self.shape().index(axis)?;
        if extent == self.extent(axis)? && shift == 0 {
            return Ok(self.clone());
        }
        let mut dims = self.shape().dims().to_vec();
        dims[dimension] = axis.of(extent);
        let shape = Shape::new(dims)?;
        let layout = Layout::contiguous(&shape);
        // (destination extent, destination stride, source extent, source stride,
        // source coordinate - destination coordinate), per logical dimension.
        let mut forward = Vec::with_capacity(shape.rank() * 5);
        let mut reverse = Vec::with_capacity(shape.rank() * 5);
        for (index, dim) in shape.dims().iter().enumerate() {
            let output_extent = i32::try_from(dim.extent)?;
            let output_stride = i32::try_from(layout.strides[index])?;
            let input_extent = i32::try_from(self.shape().dims()[index].extent)?;
            let input_stride = i32::try_from(self.0.layout.strides[index])?;
            let offset = if index == dimension { shift } else { 0 };
            forward.extend([
                output_extent,
                output_stride,
                input_extent,
                input_stride,
                offset,
            ]);
            reverse.extend([
                input_extent,
                input_stride,
                output_extent,
                output_stride,
                -offset,
            ]);
        }
        let spec = WindowSpec {
            output_len: shape.len(),
            rank: i32::try_from(shape.rank())?,
            metadata: forward,
        };
        let reverse = Rc::new(WindowSpec {
            output_len: self.shape().len(),
            rank: spec.rank,
            metadata: reverse,
        });
        let value = self.device().copy_window(&self.0.value, &spec)?;
        Ok(Self::node(
            shape,
            layout,
            value,
            self.device(),
            vec![Edge::new(self, Rule::Window(reverse))],
            false,
            None,
        ))
    }

    /// Select one logical coordinate of a named axis and remove that axis.
    ///
    /// The compact backend computes offsets from rank-sized metadata rather
    /// than constructing one host index per tensor element.
    pub fn select(&self, axis: Axis, coordinate: usize) -> Result<Self> {
        let selected_index = self.shape().index(axis)?;
        if coordinate >= self.shape().dims()[selected_index].extent {
            return Err(format!(
                "coordinate {coordinate} is outside {:?} extent {}",
                axis,
                self.shape().dims()[selected_index].extent
            )
            .into());
        }
        let shape = Shape::new(
            self.shape()
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != axis),
        )?;
        let output_layout = Layout::contiguous(&shape);
        let mut output_index = 0;
        let mut metadata = Vec::with_capacity(self.shape().rank() * 3);
        for (index, dim) in self.shape().dims().iter().enumerate() {
            metadata.push(i32::try_from(dim.extent)?);
            metadata.push(i32::try_from(self.0.layout.strides[index])?);
            if index == selected_index {
                metadata.push(-1);
            } else {
                metadata.push(i32::try_from(output_layout.strides[output_index])?);
                output_index += 1;
            }
        }
        let spec = Rc::new(SelectSpec {
            input_len: self.shape().len(),
            output_len: shape.len(),
            rank: i32::try_from(self.shape().rank())?,
            coordinate: i32::try_from(coordinate)?,
            metadata,
        });
        let value = self.device().select_axis(&self.0.value, spec.as_ref())?;
        Ok(Self::node(
            shape,
            output_layout,
            value,
            self.device(),
            vec![Edge::new(self, Rule::Select(spec))],
            false,
            None,
        ))
    }

    /// Host-side index of the first logical coordinate achieving the minimum along one named
    /// `axis`, per remaining position, in the same order [`Self::min`] itself removes that axis.
    ///
    /// This is a discrete evaluation-path primitive: it names *which* coordinate won a
    /// reduction, so unlike [`Self::min`] it carries no gradient of its own and stays entirely
    /// host-side. It never invents a second comparator: it reads back [`Self::min`]'s own
    /// device-computed minimum for each remaining position alongside this tensor's raw storage,
    /// then rescans each group in ascending coordinate order for the first element bit-equal to
    /// that minimum -- exactly `min`'s own tie rule (`backend.rs`'s `grouped_minimum` keeps the
    /// earlier candidate on a strict `<` comparison, so the first logical coordinate wins any
    /// tie) and its own finite-only comparison (a non-finite candidate never matches). A group
    /// with no finite candidate is an error: there is no coordinate a NaN result could name.
    pub fn argmin(&self, axis: Axis) -> Result<Vec<usize>> {
        let started = Instant::now();
        let reduced_index = self.shape().index(axis)?;
        let extent = self.extent(axis)?;
        let output = Shape::new(
            self.shape()
                .dims()
                .iter()
                .copied()
                .filter(|d| d.axis != axis),
        )?;
        let minimum = self.min(axis)?.to_vec()?;
        let values = self.device().read(&self.0.value)?;
        let mut indices = Vec::with_capacity(output.len());
        for (output_index, &target) in minimum.iter().enumerate() {
            let output_coords = output.coords(output_index);
            let mut winner = None;
            for coordinate in 0..extent {
                let mut input_coords = output_coords.clone();
                input_coords.insert(reduced_index, coordinate);
                let physical = self.0.layout.offset(&input_coords);
                let value = values[physical];
                if value.is_finite() && value == target {
                    winner = Some(coordinate);
                    break;
                }
            }
            indices.push(winner.ok_or_else(|| {
                format!(
                    "argmin found no finite candidate along {axis:?} at output position {output_index}"
                )
            })?);
        }
        profile("argmin", started);
        Ok(indices)
    }

    /// Sum values into buckets named by a host-side integer label per position of `axis`,
    /// replacing `axis` with a new `bucket` axis of extent `bucket_count`. `index[i]` names the
    /// bucket position `i` of `axis` contributes to; every other axis carries through unchanged,
    /// and a bucket no position names is an exact zero, not an error, matching PyTorch's
    /// `Tensor.index_add_`.
    ///
    /// This is the exact transpose of [`Self::gather`]: gather's forward picks one source row
    /// per output position, and its backward scatter-adds the upstream gradient back into every
    /// row that picked it. `scatter_add`'s forward *is* that scatter-add, built from the same
    /// host index array; its own backward is exactly gather's forward pick over the identical
    /// index (reading each source position's own bucket back out of the upstream gradient). The
    /// pairing needs no new backend kernel: it reuses `Plan::gather`/`Plan::reverse`'s CSR
    /// machinery the other way around from `gather`, keyed by this tensor's own physical
    /// storage so reordered layouts read correctly. `index` is checked against `axis`'s extent
    /// and `bucket_count` before any device work, and shares `Plan`'s 16,777,216-contribution
    /// limit, counted against this tensor's own size.
    pub fn scatter_add(
        &self,
        axis: Axis,
        index: &[usize],
        bucket: Axis,
        bucket_count: usize,
    ) -> Result<Self> {
        let started = Instant::now();
        let dimension = self.shape().index(axis)?;
        let extent = self.shape().dims()[dimension].extent;
        if bucket_count == 0 {
            return Err("scatter_add requires a positive bucket count".into());
        }
        if index.len() != extent {
            return Err(format!(
                "scatter_add index length {} does not match {axis:?} extent {extent}",
                index.len()
            )
            .into());
        }
        for (position, &target) in index.iter().enumerate() {
            if target >= bucket_count {
                return Err(format!(
                    "scatter_add bucket {target} at position {position} is outside bucket count {bucket_count}"
                )
                .into());
            }
        }
        let mut dims = self.shape().dims().to_vec();
        dims[dimension] = bucket.of(bucket_count);
        let output_shape = Shape::new(dims)?;
        let output_layout = Layout::contiguous(&output_shape);
        // One entry per physical position of `self`: `map[physical] = output_physical` reads
        // through `self.0.layout` (so reordered storage is handled the same way `gather` reads
        // it), and writes to the freshly allocated, always-contiguous output.
        let mut map = vec![0usize; self.shape().len()];
        for source_position in 0..self.shape().len() {
            let mut coords = self.shape().coords(source_position);
            let physical = self.0.layout.offset(&coords);
            coords[dimension] = index[coords[dimension]];
            map[physical] = output_layout.offset(&coords);
        }
        // Forward groups source positions by their target bucket and sums them (the transpose
        // of gather's own `Plan::gather` pick); backward picks each source position's bucket
        // straight out of the upstream gradient (the transpose of gather's own `Plan::reverse`
        // scatter-add) -- the same `map`, read the other way around in each direction.
        let forward_plan = Rc::new(Plan::reverse(&map, output_shape.len())?);
        let backward_plan = Rc::new(Plan::gather(&map)?);
        let value = self
            .device()
            .grouped(&self.0.value, None, forward_plan.as_ref(), 1.0)?;
        let result = Self::node(
            output_shape,
            output_layout,
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::Group {
                    plan: backward_plan,
                    rhs: None,
                    factor: 1.0,
                },
            )],
            false,
            None,
        );
        profile("scatter_add", started);
        Ok(result)
    }

    /// Per-bucket counts of a host-side bucket-index array: the constant-ones case of
    /// [`Self::scatter_add`], so `hard_eval`'s `torch.bincount(lab, minlength=k)` has a direct
    /// spelling. Adds no backend machinery of its own; a caller that already has a values tensor
    /// to scatter can divide its own `scatter_add` sum by this count instead of calling both.
    pub fn bincount(
        index: &[usize],
        bucket: Axis,
        bucket_count: usize,
        device: &Device,
    ) -> Result<Self> {
        if index.is_empty() {
            return Err("bincount requires a nonempty index".into());
        }
        let source = Axis::new("bincount_source");
        let ones = Self::from_slice(&vec![1.0f32; index.len()], [source.of(index.len())], device)?;
        ones.scatter_add(source, index, bucket, bucket_count)
    }

    /// Gather rows of a named axis by an arbitrary host-computed integer index,
    /// replacing that axis with a new named axis laid out along the index.
    ///
    /// `index[i]` selects one logical coordinate of `axis` for output position
    /// `i` of `output`; every other axis carries through unchanged and every
    /// coordinate is checked against `axis`'s extent before any device work.
    /// This is a row copy, not `Embedding`'s dense one-hot contraction, so it
    /// stays affordable at table sizes a one-hot vector could never reach: the
    /// index lives on the host as plain integers, never as device-side
    /// vocabulary weight. Backward scatter-adds the upstream gradient into the
    /// picked rows using the same host-built, CSR-grouped reduction plan
    /// `mean` and `min` already use for their own broadcast and reduction
    /// gradients (`Plan::gather`/`Plan::reverse` in `runtime::backend`), so a
    /// repeated index accumulates every contribution deterministically and a
    /// row that is never picked gets an exact zero gradient. It shares that
    /// plan's 16,777,216-contribution limit, counted against the gathered
    /// output's size (`index.len()` times the extent of every other axis),
    /// never against the table's own row count.
    pub fn gather(&self, axis: Axis, index: &[usize], output: Axis) -> Result<Self> {
        let started = Instant::now();
        let dimension = self.shape().index(axis)?;
        let extent = self.shape().dims()[dimension].extent;
        if index.is_empty() {
            return Err("gather requires a nonempty index".into());
        }
        for (position, &row) in index.iter().enumerate() {
            if row >= extent {
                return Err(format!(
                    "gather index {row} at position {position} is outside {axis:?} extent {extent}"
                )
                .into());
            }
        }
        let mut dims = self.shape().dims().to_vec();
        dims[dimension] = output.of(index.len());
        let shape = Shape::new(dims)?;
        let layout = Layout::contiguous(&shape);
        let mut map = vec![0usize; shape.len()];
        for (output_position, slot) in map.iter_mut().enumerate() {
            let mut coords = shape.coords(output_position);
            coords[dimension] = index[coords[dimension]];
            *slot = self.0.layout.offset(&coords);
        }
        let forward_plan = Rc::new(Plan::gather(&map)?);
        let reverse_plan = Rc::new(Plan::reverse(&map, self.shape().len())?);
        let value = self
            .device()
            .grouped(&self.0.value, None, forward_plan.as_ref(), 1.0)?;
        let result = Self::node(
            shape,
            layout,
            value,
            self.device(),
            vec![Edge::new(
                self,
                Rule::Group {
                    plan: reverse_plan,
                    rhs: None,
                    factor: 1.0,
                },
            )],
            false,
            None,
        );
        profile("gather", started);
        Ok(result)
    }

    /// Broadcast `self` onto `shape`, an explicit target that must carry every one
    /// of `self`'s axes at `self`'s own extent; `shape` may add axes `self` lacks
    /// entirely (they read with stride zero) and may reorder `self`'s existing axes.
    /// This is the outer-broadcast primitive: elementwise `add`/`sub`/`mul`/`div`
    /// only ever align one operand's axis set onto the other's when it is already a
    /// subset (`binary`, `algebra/tensor.rs`), and reject two operands that each have
    /// an axis the other lacks as "cannot introduce an implicit outer product". A
    /// caller with genuinely disjoint axis sets -- a `pixel`-indexed operand and a
    /// `site`-indexed operand, say -- broadcasts each one explicitly onto a shared
    /// `[pixel, site, ...]` shape first, then combines the results with an ordinary
    /// elementwise op: `a.broadcast_to(&shape)?.sub(&b.broadcast_to(&shape)?)`
    /// composes a `torch.cdist`-style outer pairwise difference from `broadcast_to`
    /// and `sub` alone, with no dedicated outer-product op. Backward sums the
    /// upstream gradient over every axis this call added, exactly as every other
    /// broadcast in the crate reduces an introduced axis (elementwise add/mul/div,
    /// `binary_cross_entropy_with_logits_weighted`'s `pos_weight`).
    pub fn broadcast_to(&self, shape: &Shape) -> Result<Self> {
        for dim in self.shape().dims() {
            if !shape.contains(dim.axis) {
                return Err(
                    format!("broadcast_to target shape is missing axis {:?}", dim.axis).into(),
                );
            }
            if shape.extent(dim.axis)? != dim.extent {
                return Err(format!("broadcast_to extent mismatch for {:?}", dim.axis).into());
            }
        }
        self.align(shape)
    }

    // --- Distances and margin losses ----------------------------------------------------
    //
    // `docs/nn/catalog.md`'s Distance Functions and margin-family Loss Functions rows. Every
    // method below is a plain tensor-level function, not a `Module`, exactly like
    // `categorical_cross_entropy_with_logits` and `binary_cross_entropy_with_logits`: a
    // PyTorch default that is itself a parameter (`margin`, `eps`, `swap`, ...) is an explicit
    // argument, its PyTorch default cited in the doc comment, never a Rust `Default`. A
    // `target`/label argument is always a constant (rejected if it requires gradients),
    // matching how cross-entropy already treats its targets. Every function below reduces only
    // the one named axis its own definition consumes (the compared feature axis, or the class
    // axis) and leaves every other axis unreduced for the caller's own `.mean(axes)` /
    // `.sum(axes)`, exactly as `binary_cross_entropy_with_logits` leaves batch reduction to its
    // caller.

    /// Same axis-agreement contract [`Self::squared_error`] already checks inline: identical
    /// axis sets (equal rank, every axis of `self` present in `rhs`), rejected before any
    /// device work.
    fn require_identical_axes(&self, rhs: &Self, context: &str) -> Result<()> {
        if self.shape().rank() != rhs.shape().rank()
            || self
                .shape()
                .axes()
                .iter()
                .any(|&a| !rhs.shape().contains(a))
        {
            return Err(format!("{context} requires identical axis sets").into());
        }
        Ok(())
    }

    /// Elementwise `x * (x + epsilon)^-1/2`: an ordinary `sqrt(x)` for any `x` far above
    /// `epsilon`, whose gradient stays finite as `x -> 0` (a literal `pow(0.5)` composition
    /// divides by zero there). This is exactly the norm Muon's `normalized_l2` already computes
    /// this way; `epsilon` here is a numerical floor rather than a public contract, so every
    /// caller below passes `f32::MIN_POSITIVE`, matching `normalized_l2`'s own choice.
    fn stable_sqrt(&self, epsilon: f32) -> Result<Self> {
        let inverse_root = self.inverse_sqrt(epsilon)?;
        self.mul(&inverse_root)
    }

    /// Cosine similarity along one named feature axis: [PyTorch's `CosineSimilarity`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.CosineSimilarity.html),
    /// `cos = (x1 . x2) / (max(||x1||_2, eps) * max(||x2||_2, eps))`. Each L2 norm is clamped to
    /// `eps` INDIVIDUALLY before the product -- PyTorch's own C++ kernel instead clamps the
    /// product of the squared norms to `eps^2` before one shared square root; the two formulas
    /// agree whenever either input has a norm above `eps`, and diverge only in the degenerate
    /// near-zero-vector regime neither treats as a meaningful similarity. `eps` matches
    /// PyTorch's default `1e-8` and must be finite and positive. `self`/`rhs` require identical
    /// axis sets including `axis`; every other axis is preserved unreduced.
    pub fn cosine_similarity(&self, rhs: &Self, axis: Axis, eps: f32) -> Result<Self> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err("cosine_similarity eps must be finite and positive".into());
        }
        self.require_identical_axes(rhs, "cosine_similarity")?;
        self.extent(axis)?;
        let dot = self.mul(rhs)?.sum(axis)?;
        let norm_self = self
            .mul(self)?
            .sum(axis)?
            .stable_sqrt(f32::MIN_POSITIVE)?
            .clamp(Some(eps), None)?;
        let norm_rhs = rhs
            .mul(rhs)?
            .sum(axis)?
            .stable_sqrt(f32::MIN_POSITIVE)?
            .clamp(Some(eps), None)?;
        dot.div(&norm_self.mul(&norm_rhs)?)
    }

    /// Euclidean pairwise distance along one named feature axis: [PyTorch's `PairwiseDistance`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.PairwiseDistance.html) at its
    /// default `p=2`, `((self - rhs + eps)^2).sum(axis).sqrt()`. `eps` (PyTorch's default
    /// `1e-6`) is added to the raw difference before squaring -- exactly where PyTorch's own
    /// `F.pairwise_distance` adds it, not as a denominator floor. `keepdim=False` is automatic:
    /// `axis` is removed like every other Axis reduction. Only `p=2` is implemented; PyTorch's
    /// general `p`-norm is not, since every consumer below uses the Euclidean default.
    /// `self`/`rhs` require identical axis sets including `axis`.
    pub fn pairwise_distance(&self, rhs: &Self, axis: Axis, eps: f32) -> Result<Self> {
        if !eps.is_finite() {
            return Err("pairwise_distance eps must be finite".into());
        }
        self.require_identical_axes(rhs, "pairwise_distance")?;
        self.extent(axis)?;
        let eps_tensor = Self::from_slice(&[eps], [], self.device())?;
        let diff = self.sub(rhs)?.add(&eps_tensor)?;
        diff.mul(&diff)?.sum(axis)?.stable_sqrt(f32::MIN_POSITIVE)
    }

    /// [PyTorch's `MarginRankingLoss`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.MarginRankingLoss.html):
    /// `max(0, -target * (self - rhs) + margin)`, elementwise and unreduced. `target` is a
    /// constant `1.0`/`-1.0` per element (checked before launch) and carries no gradient,
    /// matching `categorical_cross_entropy_with_logits`'s constant targets. `margin` (PyTorch
    /// default `0.0`) must be finite. `self`, `rhs`, and `target` require identical axis sets.
    pub fn margin_ranking_loss(&self, rhs: &Self, target: &Self, margin: f32) -> Result<Self> {
        if !margin.is_finite() {
            return Err("margin_ranking_loss margin must be finite".into());
        }
        if target.requires_grad() {
            return Err("margin_ranking_loss target cannot require gradients".into());
        }
        self.require_identical_axes(rhs, "margin_ranking_loss")?;
        self.require_identical_axes(target, "margin_ranking_loss")?;
        if target
            .to_vec()?
            .iter()
            .any(|&value| !value.is_finite() || (value != 1.0 && value != -1.0))
        {
            return Err("margin_ranking_loss target must be exactly 1.0 or -1.0".into());
        }
        let margin_tensor = Self::from_slice(&[margin], [], self.device())?;
        let scaled = self.sub(rhs)?.mul(target)?;
        margin_tensor.sub(&scaled)?.relu()
    }

    /// [PyTorch's `HingeEmbeddingLoss`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.HingeEmbeddingLoss.html): per
    /// element, `self` where `target == 1`, else `max(0, margin - self)` where `target == -1`.
    /// `target` is a constant `1.0`/`-1.0` per element (checked before launch, no gradient).
    /// `margin` (PyTorch default `1.0`) must be finite. `self` and `target` require identical
    /// axis sets. Unreduced.
    pub fn hinge_embedding_loss(&self, target: &Self, margin: f32) -> Result<Self> {
        if !margin.is_finite() {
            return Err("hinge_embedding_loss margin must be finite".into());
        }
        if target.requires_grad() {
            return Err("hinge_embedding_loss target cannot require gradients".into());
        }
        self.require_identical_axes(target, "hinge_embedding_loss")?;
        if target
            .to_vec()?
            .iter()
            .any(|&value| !value.is_finite() || (value != 1.0 && value != -1.0))
        {
            return Err("hinge_embedding_loss target must be exactly 1.0 or -1.0".into());
        }
        let margin_tensor = Self::from_slice(&[margin], [], self.device())?;
        let positive_mask = target.gt(0.0)?;
        let negative_mask = positive_mask.logical_not()?;
        let relu_term = margin_tensor.sub(self)?.relu()?;
        positive_mask
            .mul(self)?
            .add(&negative_mask.mul(&relu_term)?)
    }

    /// [PyTorch's `CosineEmbeddingLoss`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.CosineEmbeddingLoss.html):
    /// `1 - cos(self, rhs)` where `target == 1`, else `max(0, cos(self, rhs) - margin)` where
    /// `target == -1`, reducing the named `feature` axis via [`Self::cosine_similarity`] at its
    /// own default `eps = 1e-8`. PyTorch's public documented formula names no `eps` at all; its
    /// C++ kernel instead adds a fixed, undocumented `1e-12` inside the sum of squares. Reusing
    /// `CosineSimilarity`'s own public default keeps one canonical epsilon across the family
    /// rather than inventing a second undocumented constant. `margin` (PyTorch default `0.0`)
    /// must be finite. `target` is a constant `1.0`/`-1.0` (checked before launch): it must not
    /// contain `feature`, and its remaining axes must be a subset of `self`'s. Unreduced over
    /// every axis but `feature`.
    pub fn cosine_embedding_loss(
        &self,
        rhs: &Self,
        target: &Self,
        feature: Axis,
        margin: f32,
    ) -> Result<Self> {
        if !margin.is_finite() {
            return Err("cosine_embedding_loss margin must be finite".into());
        }
        if target.requires_grad() {
            return Err("cosine_embedding_loss target cannot require gradients".into());
        }
        if target.shape().contains(feature) {
            return Err(
                "cosine_embedding_loss target must not contain the reduced feature axis".into(),
            );
        }
        if target
            .shape()
            .axes()
            .iter()
            .any(|&a| !self.shape().contains(a))
        {
            return Err(
                "cosine_embedding_loss target axes must be a subset of the input axes".into(),
            );
        }
        if target
            .to_vec()?
            .iter()
            .any(|&value| !value.is_finite() || (value != 1.0 && value != -1.0))
        {
            return Err("cosine_embedding_loss target must be exactly 1.0 or -1.0".into());
        }
        let cosine = self.cosine_similarity(rhs, feature, 1e-8)?;
        let margin_tensor = Self::from_slice(&[margin], [], self.device())?;
        let one = Self::from_slice(&[1.0], [], self.device())?;
        let positive_mask = target.gt(0.0)?;
        let negative_mask = positive_mask.logical_not()?;
        let positive_term = one.sub(&cosine)?;
        let negative_term = cosine.sub(&margin_tensor)?.relu()?;
        positive_mask
            .mul(&positive_term)?
            .add(&negative_mask.mul(&negative_term)?)
    }

    /// [PyTorch's `TripletMarginLoss`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginLoss.html) at its
    /// default `p=2`: `max(0, margin + d(self, positive) - d(self, negative))`, where `d` is
    /// [`Self::pairwise_distance`] along `axis` with the given `eps`. When `swap` is true
    /// (PyTorch default `false`), `d(self, negative)` is instead the smaller of itself and
    /// `d(positive, negative)` (Balntas et al.'s swap term, penalizing a positive that sits
    /// closer to the negative than the anchor does), computed by stacking the two distances on
    /// a fresh axis and reducing with [`Self::min`]. `margin` (PyTorch default `1.0`) must be
    /// finite. Unreduced over every axis but `axis`.
    pub fn triplet_margin_loss(
        &self,
        positive: &Self,
        negative: &Self,
        axis: Axis,
        margin: f32,
        eps: f32,
        swap: bool,
    ) -> Result<Self> {
        if !margin.is_finite() {
            return Err("triplet_margin_loss margin must be finite".into());
        }
        let distance_positive = self.pairwise_distance(positive, axis, eps)?;
        let mut distance_negative = self.pairwise_distance(negative, axis, eps)?;
        if swap {
            let distance_swap = positive.pairwise_distance(negative, axis, eps)?;
            let pair = axis.role("triplet_margin_loss_swap_pair");
            distance_negative =
                Tensor::stack(&[distance_negative, distance_swap], pair, 0)?.min(pair)?;
        }
        let margin_tensor = Self::from_slice(&[margin], [], self.device())?;
        margin_tensor
            .add(&distance_positive)?
            .sub(&distance_negative)?
            .relu()
    }

    /// [PyTorch's `TripletMarginWithDistanceLoss`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginWithDistanceLoss.html):
    /// exactly [`Self::triplet_margin_loss`]'s `max(0, margin + d(self, positive) - d(self,
    /// negative))` and `swap` rule, with `d` an arbitrary caller-supplied Rust closure over two
    /// tensors in place of a fixed `p`-norm. PyTorch defaults `distance_function` to `None`,
    /// meaning `PairwiseDistance()`; Rust has no `Option`-shaped default that keeps a plain,
    /// statically dispatched closure parameter, so the default is spelled explicitly by the
    /// caller, e.g. `anchor.triplet_margin_with_distance_loss(&pos, &neg, margin, swap, |a, b|
    /// a.pairwise_distance(b, axis, eps))?`. `margin` (PyTorch default `1.0`) must be finite.
    pub fn triplet_margin_with_distance_loss(
        &self,
        positive: &Self,
        negative: &Self,
        margin: f32,
        swap: bool,
        distance: impl Fn(&Self, &Self) -> Result<Self>,
    ) -> Result<Self> {
        if !margin.is_finite() {
            return Err("triplet_margin_with_distance_loss margin must be finite".into());
        }
        let distance_positive = distance(self, positive)?;
        let mut distance_negative = distance(self, negative)?;
        if swap {
            let distance_swap = distance(positive, negative)?;
            let pair = Axis::new("triplet_margin_with_distance_loss_swap_pair");
            distance_negative =
                Tensor::stack(&[distance_negative, distance_swap], pair, 0)?.min(pair)?;
        }
        let margin_tensor = Self::from_slice(&[margin], [], self.device())?;
        margin_tensor
            .add(&distance_positive)?
            .sub(&distance_negative)?
            .relu()
    }

    /// [PyTorch's `MultiMarginLoss`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiMarginLoss.html) at its
    /// default `p=1`: for each class `i`, `max(0, margin - self[target] + self[i])`, summed over
    /// `i != target` and scaled by `1 / class.extent()`. `target` is a constant one-hot
    /// indicator over `class` (checked before launch: every value `0.0`/`1.0`, summing to
    /// exactly `1.0`), matching how `categorical_cross_entropy_with_logits` already spells a
    /// class label as a tensor rather than a host index. `margin` (PyTorch default `1.0`) must
    /// be finite; `class` must have at least two classes. PyTorch's optional per-class `weight`
    /// and its general `p` are not implemented -- every consumer below uses the defaults.
    /// Unreduced over every axis but `class`.
    pub fn multi_margin_loss(&self, target: &Self, class: Axis, margin: f32) -> Result<Self> {
        if !margin.is_finite() {
            return Err("multi_margin_loss margin must be finite".into());
        }
        if target.requires_grad() {
            return Err("multi_margin_loss target cannot require gradients".into());
        }
        if !target.shape().contains(class)
            || target
                .shape()
                .axes()
                .iter()
                .any(|&a| !self.shape().contains(a))
        {
            return Err("multi_margin_loss target must contain the class axis and may omit only broadcast axes".into());
        }
        self.compatible_device(target)?;
        self.shared_extents(target)?;
        let width = self.extent(class)?;
        if width < 2 {
            return Err("multi_margin_loss requires at least two classes".into());
        }
        let mut ordered_dims: Vec<_> = self
            .shape()
            .dims()
            .iter()
            .copied()
            .filter(|dim| dim.axis != class)
            .collect();
        ordered_dims.push(class.of(width));
        let ordered = Shape::new(ordered_dims)?;
        let target_ordered = target.align(&ordered)?;
        for distribution in target_ordered.to_vec()?.chunks_exact(width) {
            if distribution
                .iter()
                .any(|&v| !v.is_finite() || (v != 0.0 && v != 1.0))
            {
                return Err("multi_margin_loss target must be one-hot (values 0.0 or 1.0)".into());
            }
            let sum: f32 = distribution.iter().sum();
            if (sum - 1.0).abs() > 1e-5 {
                return Err(
                    format!("multi_margin_loss target row must sum to 1; observed {sum}").into(),
                );
            }
        }
        let target_score = self.mul(target)?.sum(class)?.broadcast_to(self.shape())?;
        let margin_tensor = Self::from_slice(&[margin], [], self.device())?;
        let hinge = margin_tensor.sub(&target_score)?.add(self)?.relu()?;
        let not_target = target.logical_not()?;
        hinge
            .mul(&not_target)?
            .sum(class)?
            .scale(1.0 / width as f32)
    }

    /// [PyTorch's `MultiLabelMarginLoss`](
    /// https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiLabelMarginLoss.html):
    /// `(1 / class.extent()) * sum_{i,j} max(0, 1 - (self[j] - self[i]))`, summed over class
    /// positions `i` that are NOT a positive label and `j` that ARE. PyTorch spells the
    /// positive-label set as a fixed-width index array terminated by `-1`; Axis instead takes
    /// `target` as a constant multi-hot `{0.0, 1.0}` indicator over `class` (`1.0` at every
    /// positive label) -- the same floating-point-tensor spelling
    /// `categorical_cross_entropy_with_logits` already uses for a single label, generalized to a
    /// set. The two encodings name the same label sets; `target` is checked before launch
    /// (every value exactly `0.0` or `1.0`) and carries no gradient. `class` must have at least
    /// two classes. Neither `margin` (PyTorch fixes it at `1.0`; it is not a parameter of this
    /// class) nor `weight`/`reduction` exist to configure. Unreduced over every axis but
    /// `class`.
    pub fn multi_label_margin_loss(&self, target: &Self, class: Axis) -> Result<Self> {
        if target.requires_grad() {
            return Err("multi_label_margin_loss target cannot require gradients".into());
        }
        if !target.shape().contains(class)
            || target
                .shape()
                .axes()
                .iter()
                .any(|&a| !self.shape().contains(a))
        {
            return Err("multi_label_margin_loss target must contain the class axis and may omit only broadcast axes".into());
        }
        self.compatible_device(target)?;
        self.shared_extents(target)?;
        let width = self.extent(class)?;
        if width < 2 {
            return Err("multi_label_margin_loss requires at least two classes".into());
        }
        if target
            .to_vec()?
            .iter()
            .any(|&value| !value.is_finite() || (value != 0.0 && value != 1.0))
        {
            return Err("multi_label_margin_loss target must be 0.0 or 1.0".into());
        }
        let pair = class.role("multi_label_margin_loss_pair");
        let mut combined_dims: Vec<_> = self.shape().dims().to_vec();
        combined_dims.push(pair.of(width));
        let combined = Shape::new(combined_dims)?;

        let score_i = self.broadcast_to(&combined)?;
        let score_j = self.rename(class, pair)?.broadcast_to(&combined)?;
        let target_i = target.broadcast_to(&combined)?;
        let target_j = target.rename(class, pair)?.broadcast_to(&combined)?;

        let one = Self::from_slice(&[1.0], [], self.device())?;
        let hinge = one.sub(&score_j.sub(&score_i)?)?.relu()?;
        let not_target_i = one.sub(&target_i)?;
        let mask = target_j.mul(&not_target_i)?;
        hinge
            .mul(&mask)?
            .sum([class, pair])?
            .scale(1.0 / width as f32)
    }

    /// Stack equal named shapes, inserting a new logical axis at `position`.
    /// Physical storage is stack-major so each source remains one contiguous copy.
    pub fn stack(values: &[Self], axis: Axis, position: usize) -> Result<Self> {
        let first = values.first().ok_or("stack requires at least one tensor")?;
        if position > first.shape().rank() {
            return Err("stack position is outside the output rank".into());
        }
        if first.shape().contains(axis) {
            return Err(format!("stack axis {axis:?} already exists in its inputs").into());
        }
        let mut aligned = Vec::with_capacity(values.len());
        for value in values {
            first.compatible_device(value)?;
            if value.shape().rank() != first.shape().rank()
                || first
                    .shape()
                    .axes()
                    .iter()
                    .any(|candidate| !value.shape().contains(*candidate))
            {
                return Err("stack requires identical input axis sets".into());
            }
            first.shared_extents(value)?;
            aligned.push(value.align(first.shape())?);
        }
        let mut dims = first.shape().dims().to_vec();
        dims.insert(position, axis.of(values.len()));
        let shape = Shape::new(dims)?;
        let mut physical = vec![axis];
        physical.extend(first.shape().axes());
        let layout = Layout::new(&shape, &physical)?;
        let buffers: Vec<_> = aligned.iter().map(|value| value.0.value.clone()).collect();
        let value = first.device().stack_contiguous(&buffers)?;
        let slice_len = first.shape().len();
        let edges = aligned
            .iter()
            .enumerate()
            .map(|(index, input)| {
                Edge::new(
                    input,
                    Rule::StackSlice {
                        offset: index * slice_len,
                        len: slice_len,
                    },
                )
            })
            .collect();
        Ok(Self::node(
            shape,
            layout,
            value,
            first.device(),
            edges,
            false,
            None,
        ))
    }
    /// Concatenate two or more tensors along a named axis they already share,
    /// extending that axis's extent by the sum of each operand's extent. Every
    /// other axis must match exactly across operands (same identity, same
    /// extent); the output's axis order follows the first operand's. This is
    /// the wave-1-proven composition of `pad_zeros` and `add`: each operand is
    /// zero-padded into its own slice of the concatenated axis, then the
    /// padded tensors are summed, so backward automatically narrows the
    /// incoming gradient back to each operand's slice with no dedicated rule.
    pub fn concat(values: &[Self], axis: Axis) -> Result<Self> {
        let first = values
            .first()
            .ok_or("concat requires at least one tensor")?;
        first.shape().index(axis)?;
        let mut total = 0usize;
        for value in values {
            first.compatible_device(value)?;
            if value.shape().rank() != first.shape().rank()
                || first
                    .shape()
                    .axes()
                    .iter()
                    .any(|candidate| !value.shape().contains(*candidate))
            {
                return Err("concat requires identical input axis sets".into());
            }
            for dim in first.shape().dims() {
                if dim.axis != axis && value.extent(dim.axis)? != dim.extent {
                    return Err(
                        format!("concat requires matching extent for {:?}", dim.axis).into(),
                    );
                }
            }
            total = total
                .checked_add(value.extent(axis)?)
                .ok_or("concat extent overflow")?;
        }
        let mut offset = 0usize;
        let mut result: Option<Self> = None;
        for value in values {
            let extent = value.extent(axis)?;
            let after = total - offset - extent;
            let padded = value.pad_zeros(axis, offset, after)?;
            result = Some(match result {
                Some(accumulated) => accumulated.add(&padded)?,
                None => padded,
            });
            offset += extent;
        }
        Ok(result.expect("validated at least one tensor above"))
    }
    /// Cyclically shift one named axis's values by `shift`, wrapping values
    /// that run off one end onto the other. PyTorch's `torch.roll` sign
    /// convention: the element at logical index `i` moves to
    /// `(i + shift).rem_euclid(extent)`, so a positive shift moves values
    /// toward higher indices and a negative shift moves them toward lower
    /// indices. Shifting by 0 or by any multiple of the axis's extent is the
    /// identity. Rolling one axis at a time composes exactly with PyTorch's
    /// multi-axis `dims=`: `torch.roll(x, shifts=(-s, -s), dims=(1, 2))` is
    /// two calls, one per axis (SwinIR's shifted-window attention,
    /// `network_swinir.py:251,271`).
    ///
    /// Implemented as `concat` of two `narrow` slices -- the wrapped tail
    /// moved ahead of the untouched head -- rather than a dedicated backend
    /// rule: both primitives already carry exact device kernels and exact
    /// gradients, so backward falls out for free as the inverse roll (the
    /// same call with `shift` negated), with no new `Rule` variant to
    /// maintain.
    pub fn roll(&self, axis: Axis, shift: isize) -> Result<Self> {
        let extent = self.extent(axis)?;
        let offset = shift.rem_euclid(isize::try_from(extent)?);
        if offset == 0 {
            return Ok(self.clone());
        }
        let split = extent - usize::try_from(offset)?;
        let head = self.narrow(axis, 0, split)?;
        let tail = self.narrow(axis, split, extent - split)?;
        Self::concat(&[tail, head], axis)
    }
    /// Materialize a storage order without changing logical axes or values.
    pub fn with_layout(&self, order: impl IntoAxes) -> Result<Self> {
        let started = Instant::now();
        let layout = Layout::new(self.shape(), &order.into_axes())?;
        if self.0.layout.strides == layout.strides {
            return Ok(self.clone());
        }
        if self
            .shape()
            .dims()
            .iter()
            .enumerate()
            .all(|(i, dim)| dim.extent == 1 || self.0.layout.strides[i] == layout.strides[i])
        {
            return Ok(Self::node(
                self.shape().clone(),
                layout,
                self.0.value.clone(),
                self.device(),
                vec![Edge::new(self, Rule::Identity)],
                false,
                None,
            ));
        }
        // The existing compact selection copier also implements a permutation:
        // no dimension is removed when all output strides are nonnegative.
        // Forward/backward each compute one bijective offset per device element.
        let mut metadata = Vec::with_capacity(self.shape().rank() * 3);
        for (index, dim) in self.shape().dims().iter().enumerate() {
            metadata.extend([
                i32::try_from(dim.extent)?,
                i32::try_from(self.0.layout.strides[index])?,
                i32::try_from(layout.strides[index])?,
            ]);
        }
        #[cfg(test)]
        LAYOUT_METADATA_MAX.with(|max| max.set(max.get().max(metadata.len())));
        let spec = Rc::new(SelectSpec {
            input_len: self.shape().len(),
            output_len: self.shape().len(),
            rank: i32::try_from(self.shape().rank())?,
            coordinate: 0,
            metadata,
        });
        let value = self.device().select_axis(&self.0.value, spec.as_ref())?;
        let result = Self::node(
            self.shape().clone(),
            layout,
            value,
            self.device(),
            vec![Edge::new(self, Rule::Select(spec))],
            false,
            None,
        );
        profile("with_layout", started);
        Ok(result)
    }
    pub fn split(&self, axis: Axis, dims: impl IntoIterator<Item = Dim>) -> Result<Self> {
        let index = self.shape().index(axis)?;
        let parts = Shape::new(dims)?;
        if parts.rank() == 0 || parts.len() != self.extent(axis)? {
            return Err("split extents must multiply to the source extent".into());
        }
        let mut dims = self.shape().dims().to_vec();
        dims.splice(index..=index, parts.dims().iter().copied());
        let shape = Shape::new(dims)?;
        let ordered = self.with_layout(self.shape().axes())?;
        Ok(Self::node(
            shape.clone(),
            Layout::contiguous(&shape),
            ordered.0.value.clone(),
            self.device(),
            vec![Edge::new(&ordered, Rule::Identity)],
            false,
            None,
        ))
    }
    /// Selected-axis order defines flattening; insert the merged axis at the first selected position.
    pub fn merge(&self, axes: impl IntoAxes, axis: Axis) -> Result<Self> {
        let started = Instant::now();
        let axes = self.shape().select_axes(axes)?;
        if axes.is_empty() {
            return Err("merge requires at least one axis".into());
        }
        let parts = Shape::new(
            axes.iter()
                .map(|&a| a.of(self.extent(a).expect("validated"))),
        )?;
        let mut dims = vec![];
        let mut inserted = false;
        for &dim in self.shape().dims() {
            if axes.contains(&dim.axis) {
                if !inserted {
                    dims.push(axis.of(parts.len()));
                    inserted = true;
                }
            } else {
                dims.push(dim);
            }
        }
        let shape = Shape::new(dims)?;
        let layout = Layout::contiguous(&shape);
        // Expand the desired merged physical order back into the input axes.
        // This keeps selected-axis order significant, including nonadjacent axes,
        // but costs O(rank) host metadata instead of O(elements) index vectors.
        let order = shape
            .axes()
            .into_iter()
            .flat_map(|a| if a == axis { axes.clone() } else { vec![a] })
            .collect::<Vec<_>>();
        let ordered = self.with_layout(order)?;
        let result = Self::node(
            shape,
            layout,
            ordered.0.value.clone(),
            self.device(),
            vec![Edge::new(&ordered, Rule::Identity)],
            false,
            None,
        );
        profile("merge", started);
        Ok(result)
    }
    /// `F.interpolate(mode="nearest")` for one named spatial axis at an exact
    /// integer scale factor. Composed entirely from `stack` (duplicate the
    /// input `factor` times along a fresh axis) and `merge` (fold that axis
    /// into the named spatial axis, spatial-axis major, so each source
    /// element becomes `factor` adjacent identical copies) — the composition
    /// wave 1 of the research migration proved bit-exact against a real
    /// `F.interpolate(mode="nearest")` oracle
    /// (`research/src/vision/morpheus/axis/tests/partitioner_trunk_fpn_obj.rs`,
    /// `upsample2x`, filed as axis issue #66). `morpheus`'s
    /// `mobilesam/scale10/train_partitioner.py:107` and
    /// `detector/train_detector.py:237,239` both call this at `factor = 2` on
    /// `height` then `width` separately (stride-16 to stride-8, stride-8 to
    /// stride-4 in an FPN fusion); call this once per spatial axis to
    /// reproduce that. Backward needs no dedicated rule: `stack` and `merge`
    /// are already differentiable, so the gradient for a source element is
    /// the sum of its `factor` output copies' incoming gradients.
    pub fn upsample_nearest(&self, axis: Axis, factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err("upsample_nearest factor must be at least 1".into());
        }
        self.shape().index(axis)?;
        if factor == 1 {
            return Ok(self.clone());
        }
        let repeat = axis.role("upsample_nearest_repeat");
        let merged = axis.role("upsample_nearest_merged");
        let position = self.shape().rank();
        let copies = vec![self.clone(); factor];
        let stacked = Self::stack(&copies, repeat, position)?;
        stacked.merge([axis, repeat], merged)?.rename(merged, axis)
    }
    /// `torch.nn.Upsample(mode="bilinear", align_corners=False)` for one
    /// named axis, at any positive output extent (`world/fluid`'s
    /// `scripts/train.py:101-102` only ever calls this at `scale_factor=2`,
    /// i.e. `out_extent = 2 * self.extent(axis)`, applied to `height` then
    /// `width` separately before each of the U-Net decoder's two skip
    /// concatenations; filed as axis issue #66). The interpolation weights
    /// are fixed by the input/output extents alone (never by tensor data),
    /// so this is one `contract` against a host-built `[axis, resampled]`
    /// weight matrix rather than a dedicated kernel; `contract`'s existing
    /// backward is already the transpose of that same matrix, so the
    /// gradient is exact for free and distributes to up to two source
    /// coordinates per output coordinate (up to four source cells when a
    /// height and a width axis are each resampled, one call per axis).
    ///
    /// Weights follow PyTorch's own `align_corners=False` half-pixel
    /// formula: for output coordinate `j`,
    /// `source = max(0, (j + 0.5) * in_extent / out_extent - 0.5)`; let
    /// `lower = floor(source)` clamped to the last valid input coordinate and
    /// `upper = lower + 1` (or `lower` again at the last input coordinate, so
    /// its weight is exactly 1). `lower` receives weight `1 - fract(source)`
    /// and `upper` receives weight `fract(source)`.
    pub fn resample_bilinear(&self, axis: Axis, out_extent: usize) -> Result<Self> {
        let in_extent = self.extent(axis)?;
        if out_extent == 0 {
            return Err("resample_bilinear output extent must be at least 1".into());
        }
        if out_extent == in_extent {
            return Ok(self.clone());
        }
        let resampled = axis.role("resample_bilinear_resampled");
        let scale = in_extent as f64 / out_extent as f64;
        let mut weights = vec![0f32; in_extent * out_extent];
        for j in 0..out_extent {
            let source = (scale * (j as f64 + 0.5) - 0.5).max(0.0);
            let lower = (source.floor() as usize).min(in_extent - 1);
            let upper = if lower < in_extent - 1 {
                lower + 1
            } else {
                lower
            };
            let lambda1 = (source - lower as f64) as f32;
            let lambda0 = 1.0 - lambda1;
            weights[lower * out_extent + j] += lambda0;
            weights[upper * out_extent + j] += lambda1;
        }
        let weight = Self::from_slice(
            &weights,
            [axis.of(in_extent), resampled.of(out_extent)],
            self.device(),
        )?;
        self.contract(&weight, axis)?.rename(resampled, axis)
    }
    /// Reverse mode from a scalar. Releases the graph after success; rebuild it for another backward.
    pub fn backward(&self) -> Result<()> {
        if self.shape().rank() != 0 {
            return Err("backward requires a scalar; reduce the loss axes explicitly".into());
        }
        if !self.requires_grad() {
            return Err("tensor does not require gradients".into());
        }
        let mut order = vec![];
        let mut seen = HashSet::new();
        let mut stack = vec![(self.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                order.push(node);
                continue;
            }
            if !seen.insert(node.0.id) {
                continue;
            }
            if node.0.consumed.get() {
                return Err("backward graph already consumed; run forward again".into());
            }
            if let Some((version, expected)) = &node.0.version
                && version.get() != *expected
            {
                return Err("parameter changed between forward and backward".into());
            }
            stack.push((node.clone(), true));
            if let Some(edges) = node.0.edges.borrow().as_ref() {
                for edge in edges.iter() {
                    stack.push((edge.input.clone(), false));
                }
            }
        }
        let mut adjoints: HashMap<u64, Buffer> = HashMap::new();
        adjoints.insert(self.0.id, self.device().upload(vec![1.0])?);
        let mut leaves = vec![];
        for node in order.iter().rev() {
            let gradient = adjoints.remove(&node.0.id).expect("reachable gradient");
            if let Some(edges) = node.0.edges.borrow().as_ref() {
                for edge in edges.iter() {
                    let contribution = match &edge.rule {
                        Rule::Identity => gradient.clone(),
                        Rule::Zero(len) => self.device().zeros_buffer(*len)?,
                        Rule::Scale(f) => self.device().scale(&gradient, *f)?,
                        Rule::Multiply(rhs) => self.device().binary(&gradient, rhs, 2)?,
                        Rule::Divide(rhs) => self.device().binary(&gradient, rhs, 3)?,
                        Rule::DivideDenominator {
                            numerator,
                            denominator,
                        } => self.device().divide_backward_denominator(
                            &gradient,
                            numerator,
                            denominator,
                        )?,
                        Rule::Relu(x) => self.device().relu_backward(&gradient, x)?,
                        Rule::Sigmoid(probability) => {
                            self.device().sigmoid_backward(&gradient, probability)?
                        }
                        Rule::Silu(input) => self.device().silu_backward(&gradient, input)?,
                        Rule::LeakyRelu {
                            input,
                            negative_slope,
                        } => {
                            self.device()
                                .leaky_relu_backward(&gradient, input, *negative_slope)?
                        }
                        Rule::Tanh(output) => self.device().tanh_backward(&gradient, output)?,
                        Rule::Sin(input) => self.device().sin_backward(&gradient, input)?,
                        Rule::Abs(input) => self.device().abs_backward(&gradient, input)?,
                        Rule::Gelu(x) => self.device().gelu_backward(&gradient, x)?,
                        Rule::GeluExact(x) => self.device().gelu_exact_backward(&gradient, x)?,
                        Rule::InverseSqrt { input, epsilon } => self
                            .device()
                            .inverse_sqrt_backward(&gradient, input, *epsilon)?,
                        Rule::Clamp { input, min, max } => {
                            self.device().clamp_backward(&gradient, input, *min, *max)?
                        }
                        Rule::BinaryCrossEntropy { logits, targets } => self
                            .device()
                            .binary_cross_entropy_backward(&gradient, logits, targets)?,
                        Rule::BinaryCrossEntropyWeighted {
                            logits,
                            targets,
                            pos_weight,
                        } => self.device().binary_cross_entropy_weighted_backward(
                            &gradient, logits, targets, pos_weight,
                        )?,
                        Rule::CategoricalCrossEntropy {
                            probability,
                            targets,
                            width,
                        } => self.device().categorical_cross_entropy_backward(
                            &gradient,
                            probability,
                            targets,
                            *width,
                        )?,
                        Rule::Softmax { probability, width } => {
                            self.device()
                                .softmax_backward(&gradient, probability, *width)?
                        }
                        Rule::MatmulLeft {
                            rhs,
                            batch,
                            m,
                            k,
                            n,
                        } => self
                            .device()
                            .matmul_left_backward(&gradient, rhs, *batch, *m, *k, *n)?,
                        Rule::MatmulRight {
                            lhs,
                            batch,
                            m,
                            k,
                            n,
                        } => self
                            .device()
                            .matmul_right_backward(lhs, &gradient, *batch, *m, *k, *n)?,
                        Rule::Group { plan, rhs, factor } => self.device().grouped(
                            &gradient,
                            rhs.as_ref(),
                            plan.as_ref(),
                            *factor,
                        )?,
                        Rule::Minimum { plan, winners } | Rule::Maximum { plan, winners } => {
                            let expanded =
                                self.device().grouped(&gradient, None, plan.as_ref(), 1.0)?;
                            self.device().mask_gradient(&expanded, winners)?
                        }
                        Rule::Unfold(spec) => {
                            self.device().unfold_backward(&gradient, spec.as_ref())?
                        }
                        Rule::Select(spec) => self
                            .device()
                            .select_axis_backward(&gradient, spec.as_ref())?,
                        Rule::Window(spec) => {
                            self.device().copy_window(&gradient, spec.as_ref())?
                        }
                        Rule::StackSlice { offset, len } => {
                            self.device().contiguous_slice(&gradient, *offset, *len)?
                        }
                        Rule::Exp(output) => self.device().exp_backward(&gradient, output)?,
                        Rule::Ln(input) => self.device().ln_backward(&gradient, input)?,
                        Rule::Softplus {
                            input,
                            beta,
                            threshold,
                        } => self
                            .device()
                            .softplus_backward(&gradient, input, *beta, *threshold)?,
                    };
                    let id = edge.input.0.id;
                    let sum = if let Some(existing) = adjoints.remove(&id) {
                        self.device().binary(&existing, &contribution, 0)?
                    } else {
                        contribution
                    };
                    adjoints.insert(id, sum);
                }
            } else {
                let sum = if let Some(existing) = node.0.grad.borrow().as_ref() {
                    self.device().binary(existing, &gradient, 0)?
                } else {
                    gradient
                };
                leaves.push((node.clone(), sum));
            }
        }
        // Commit only after every GPU operation succeeds.
        for (leaf, gradient) in leaves {
            *leaf.0.grad.borrow_mut() = Some(gradient);
        }
        for node in order {
            if node.0.edges.borrow_mut().take().is_some() {
                node.0.consumed.set(true);
            }
        }
        Ok(())
    }
}
