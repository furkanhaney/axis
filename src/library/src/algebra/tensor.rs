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
    Gelu(Buffer),
    GeluExact(Buffer),
    InverseSqrt {
        input: Buffer,
        epsilon: f32,
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
    Unfold(Rc<UnfoldSpec>),
    Select(Rc<SelectSpec>),
    Window(Rc<WindowSpec>),
    StackSlice {
        offset: usize,
        len: usize,
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
    /// This distinction matters when importing pretrained exact-GELU models.
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
    pub fn mean(&self, axes: impl IntoAxes) -> Result<Self> {
        let started = Instant::now();
        let axes = self.shape().select_axes(axes)?;
        let key = (0, self.shape().clone(), self.0.layout.clone(), axes.clone());
        let cached = REDUCTION_PLANS.with(|cache| cache.borrow().get(&key).cloned());
        let plans = match cached {
            Some(plans) => plans,
            None => {
                let output = Shape::new(
                    self.shape()
                        .dims()
                        .iter()
                        .copied()
                        .filter(|d| !axes.contains(&d.axis)),
                )?;
                let layout = Layout::contiguous(&output);
                let factor = output.len() as f32 / self.shape().len() as f32;
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
                plans
            }
        };
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
        profile("mean", started);
        Ok(result)
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
                        Rule::Gelu(x) => self.device().gelu_backward(&gradient, x)?,
                        Rule::GeluExact(x) => self.device().gelu_exact_backward(&gradient, x)?,
                        Rule::InverseSqrt { input, epsilon } => self
                            .device()
                            .inverse_sqrt_backward(&gradient, input, *epsilon)?,
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
                        Rule::Minimum { plan, winners } => {
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
