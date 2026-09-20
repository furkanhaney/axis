use crate::{
    Axis, Device, Dim, IntoAxes, Result, Shape,
    axis::Layout,
    backend::{Buffer, Plan},
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
struct GatherPlans {
    forward: Rc<Plan>,
    reverse: Rc<Plan>,
}

#[derive(Clone)]
struct MeanPlans {
    output: Shape,
    layout: Layout,
    factor: f32,
    forward: Rc<Plan>,
    reverse: Rc<Plan>,
}

type GatherPlanKey = (u8, Shape, Layout, Shape, Layout);
type MeanPlanKey = (Shape, Layout, Vec<Axis>);

thread_local! {
    static GATHER_PLANS: RefCell<HashMap<GatherPlanKey, GatherPlans>> = RefCell::new(HashMap::new());
    static MEAN_PLANS: RefCell<HashMap<MeanPlanKey, MeanPlans>> = RefCell::new(HashMap::new());
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
    Scale(f32),
    Multiply(Buffer),
    Relu(Buffer),
    Gelu(Buffer),
    InverseSqrt {
        input: Buffer,
        epsilon: f32,
    },
    BinaryCrossEntropy {
        logits: Buffer,
        targets: Buffer,
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
    fn gathered(&self, shape: Shape, layout: Layout, map: Vec<usize>) -> Result<Self> {
        if map
            .iter()
            .enumerate()
            .all(|(output, &input)| output == input)
        {
            return Ok(Self::node(
                shape,
                layout,
                self.0.value.clone(),
                self.device(),
                vec![Edge::new(self, Rule::Identity)],
                false,
                None,
            ));
        }
        let forward = Rc::new(Plan::gather(&map)?);
        let reverse = Rc::new(Plan::reverse(&map, self.shape().len())?);
        let value = self
            .device()
            .grouped(&self.0.value, None, forward.as_ref(), 1.0)?;
        let edges = self.requires_grad().then(|| {
            Edge::new(
                self,
                Rule::Group {
                    plan: reverse,
                    rhs: None,
                    factor: 1.0,
                },
            )
        });
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
    fn align(&self, shape: &Shape) -> Result<Self> {
        let started = Instant::now();
        let layout = Layout::contiguous(shape);
        if self.shape() == shape && self.0.layout.strides == layout.strides {
            return Ok(self.clone());
        }
        let key = (
            0,
            self.shape().clone(),
            self.0.layout.clone(),
            shape.clone(),
            layout.clone(),
        );
        let cached = GATHER_PLANS.with(|cache| cache.borrow().get(&key).cloned());
        let plans = match cached {
            Some(plans) => plans,
            None => {
                let positions: Vec<_> = self
                    .shape()
                    .axes()
                    .iter()
                    .map(|&a| shape.index(a))
                    .collect::<Result<_>>()?;
                let map = (0..shape.len())
                    .map(|i| {
                        let coords = shape.coords(i);
                        self.0
                            .layout
                            .offset(&positions.iter().map(|&p| coords[p]).collect::<Vec<_>>())
                    })
                    .collect::<Vec<_>>();
                if map
                    .iter()
                    .enumerate()
                    .all(|(output, &input)| output == input)
                {
                    return Ok(Self::node(
                        shape.clone(),
                        layout,
                        self.0.value.clone(),
                        self.device(),
                        vec![Edge::new(self, Rule::Identity)],
                        false,
                        None,
                    ));
                }
                let plans = GatherPlans {
                    forward: Rc::new(Plan::gather(&map)?),
                    reverse: Rc::new(Plan::reverse(&map, self.shape().len())?),
                };
                GATHER_PLANS.with(|cache| {
                    cache.borrow_mut().insert(key, plans.clone());
                });
                plans
            }
        };
        let value = self
            .device()
            .grouped(&self.0.value, None, plans.forward.as_ref(), 1.0)?;
        let edges = self.requires_grad().then(|| {
            Edge::new(
                self,
                Rule::Group {
                    plan: plans.reverse,
                    rhs: None,
                    factor: 1.0,
                },
            )
        });
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
            _ => (
                Rule::Multiply(b.0.value.clone()),
                Rule::Multiply(a.0.value.clone()),
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
        let key = (self.shape().clone(), self.0.layout.clone(), axes.clone());
        let cached = MEAN_PLANS.with(|cache| cache.borrow().get(&key).cloned());
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
                let plans = MeanPlans {
                    output: output.clone(),
                    layout,
                    factor,
                    forward: Rc::new(Plan::reverse(&map, output.len())?),
                    reverse: Rc::new(Plan::gather(&map)?),
                };
                MEAN_PLANS.with(|cache| {
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
        if query == key || self.extent(query)? != self.extent(key)? {
            return Err("causal_mask requires distinct query/key axes of equal extent".into());
        }
        let q = self.shape().index(query)?;
        let k = self.shape().index(key)?;
        Plan::check_size(self.shape().len())?;
        let mut keep = vec![0.0; self.shape().len()];
        for i in 0..self.shape().len() {
            let coords = self.shape().coords(i);
            keep[self.0.layout.offset(&coords)] = f32::from(coords[k] <= coords[q]);
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
        if spatial[0] == spatial[1] || channels == spatial[0] || channels == spatial[1] {
            return Err("unfold2d requires distinct channel and spatial axes".into());
        }
        if kernel.contains(&0) {
            return Err("unfold2d kernel extents must be positive".into());
        }
        let channel_extent = self.extent(channels)?;
        let input_height = self.extent(spatial[0])?;
        let input_width = self.extent(spatial[1])?;
        if kernel[0] > input_height || kernel[1] > input_width {
            return Err("unfold2d kernel exceeds the spatial extent".into());
        }
        let expected_patch = channel_extent
            .checked_mul(kernel[0])
            .and_then(|n| n.checked_mul(kernel[1]))
            .ok_or("unfold2d patch extent overflow")?;
        if patch.extent != expected_patch {
            return Err("unfold2d patch extent must equal channels * kernel area".into());
        }
        let output_height = input_height - kernel[0] + 1;
        let output_width = input_width - kernel[1] + 1;
        let shape = Shape::new(self.shape().dims().iter().map(|dim| {
            if dim.axis == channels {
                patch
            } else if dim.axis == spatial[0] {
                spatial[0].of(output_height)
            } else if dim.axis == spatial[1] {
                spatial[1].of(output_width)
            } else {
                *dim
            }
        }))?;
        let patch_index = shape.index(patch.axis)?;
        let output_height_index = shape.index(spatial[0])?;
        let output_width_index = shape.index(spatial[1])?;
        let map = (0..shape.len())
            .map(|i| {
                let output = shape.coords(i);
                let flattened = output[patch_index];
                let kx = flattened % kernel[1];
                let rest = flattened / kernel[1];
                let ky = rest % kernel[0];
                let channel = rest / kernel[0];
                let input = self
                    .shape()
                    .dims()
                    .iter()
                    .map(|dim| {
                        if dim.axis == channels {
                            channel
                        } else if dim.axis == spatial[0] {
                            output[output_height_index] + ky
                        } else if dim.axis == spatial[1] {
                            output[output_width_index] + kx
                        } else {
                            output[shape.index(dim.axis).expect("retained axis")]
                        }
                    })
                    .collect::<Vec<_>>();
                self.0.layout.offset(&input)
            })
            .collect();
        self.gathered(shape.clone(), Layout::contiguous(&shape), map)
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
    /// Materialize a storage order without changing logical axes or values.
    pub fn with_layout(&self, order: impl IntoAxes) -> Result<Self> {
        let layout = Layout::new(self.shape(), &order.into_axes())?;
        if self.0.layout.strides == layout.strides {
            return Ok(self.clone());
        }
        let key = (
            1,
            self.shape().clone(),
            self.0.layout.clone(),
            self.shape().clone(),
            layout.clone(),
        );
        let cached = GATHER_PLANS.with(|cache| cache.borrow().get(&key).cloned());
        let plans = match cached {
            Some(plans) => plans,
            None => {
                let mut map = vec![0; self.shape().len()];
                for i in 0..self.shape().len() {
                    let coords = self.shape().coords(i);
                    map[layout.offset(&coords)] = self.0.layout.offset(&coords);
                }
                let plans = GatherPlans {
                    forward: Rc::new(Plan::gather(&map)?),
                    reverse: Rc::new(Plan::reverse(&map, self.shape().len())?),
                };
                GATHER_PLANS.with(|cache| {
                    cache.borrow_mut().insert(key, plans.clone());
                });
                plans
            }
        };
        let value = self
            .device()
            .grouped(&self.0.value, None, plans.forward.as_ref(), 1.0)?;
        let edges = self.requires_grad().then(|| {
            Edge::new(
                self,
                Rule::Group {
                    plan: plans.reverse,
                    rhs: None,
                    factor: 1.0,
                },
            )
        });
        Ok(Self::node(
            self.shape().clone(),
            layout,
            value,
            self.device(),
            edges.into_iter().collect(),
            false,
            None,
        ))
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
        if self.0.layout == Layout::contiguous(self.shape()) {
            return Ok(Self::node(
                shape.clone(),
                Layout::contiguous(&shape),
                self.0.value.clone(),
                self.device(),
                vec![Edge::new(self, Rule::Identity)],
                false,
                None,
            ));
        }
        let pl = Layout::contiguous(&parts);
        let map = (0..shape.len())
            .map(|i| {
                let mut coords = shape.coords(i);
                let merged = pl.offset(&coords[index..index + parts.rank()]);
                coords.splice(index..index + parts.rank(), [merged]);
                self.0.layout.offset(&coords)
            })
            .collect();
        self.gathered(shape.clone(), Layout::contiguous(&shape), map)
    }
    /// Selected-axis order defines flattening; insert the merged axis at the first selected position.
    pub fn merge(&self, axes: impl IntoAxes, axis: Axis) -> Result<Self> {
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
        let merged_index = shape.index(axis)?;
        let map = (0..shape.len())
            .map(|i| {
                let output = shape.coords(i);
                let part = parts.coords(output[merged_index]);
                let input: Vec<_> = self
                    .shape()
                    .axes()
                    .iter()
                    .map(|&a| {
                        if let Ok(p) = parts.index(a) {
                            part[p]
                        } else {
                            output[shape.index(a).expect("retained axis")]
                        }
                    })
                    .collect();
                self.0.layout.offset(&input)
            })
            .collect();
        self.gathered(shape.clone(), Layout::contiguous(&shape), map)
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
                        Rule::Scale(f) => self.device().scale(&gradient, *f)?,
                        Rule::Multiply(rhs) => self.device().binary(&gradient, rhs, 2)?,
                        Rule::Relu(x) => self.device().relu_backward(&gradient, x)?,
                        Rule::Gelu(x) => self.device().gelu_backward(&gradient, x)?,
                        Rule::InverseSqrt { input, epsilon } => self
                            .device()
                            .inverse_sqrt_backward(&gradient, input, *epsilon)?,
                        Rule::BinaryCrossEntropy { logits, targets } => self
                            .device()
                            .binary_cross_entropy_backward(&gradient, logits, targets)?,
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
