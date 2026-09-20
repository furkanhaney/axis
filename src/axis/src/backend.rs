//! A synchronous, correctness-first cuTile backend. Plans contain indices, never values.
use crate::Result;
use cutile::prelude::*;
use std::rc::Rc;

pub(crate) type Buffer = Arc<cutile::tensor::Tensor<f32>>;

#[derive(Clone)]
pub struct Device(Rc<Context>);
struct Context {
    stream: Arc<cutile::cuda_core::Stream>,
}

impl Device {
    pub fn cuda(ordinal: usize) -> Result<Self> {
        let device = cutile::cuda_core::Device::new(ordinal)?;
        Ok(Self(Rc::new(Context {
            stream: device.new_stream()?,
        })))
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn upload(&self, values: Vec<f32>) -> Result<Buffer> {
        Ok(Arc::new(
            api::copy_host_vec_to_device(&Arc::new(values)).sync_on(&self.0.stream)?,
        ))
    }
    pub(crate) fn read(&self, buffer: &Buffer) -> Result<Vec<f32>> {
        Ok(buffer.to_host_vec().sync_on(&self.0.stream)?)
    }
    fn zeros(&self, len: usize) -> Result<cutile::tensor::Tensor<f32>> {
        Ok(api::zeros(&[len]).sync_on(&self.0.stream)?)
    }
    pub(crate) fn zeros_buffer(&self, len: usize) -> Result<Buffer> {
        Ok(Arc::new(self.zeros(len)?))
    }
    pub(crate) fn binary(&self, a: &Buffer, b: &Buffer, op: i32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::binary((&mut out).partition([128]), a.as_ref(), b.as_ref())
            .generics(vec![op.to_string()])
            .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn scale(&self, a: &Buffer, scale: f32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::scale((&mut out).partition([128]), a.as_ref(), scale).sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn relu(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::relu((&mut out).partition([128]), a.as_ref()).sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn relu_backward(&self, gradient: &Buffer, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::relu_backward((&mut out).partition([128]), gradient.as_ref(), a.as_ref())
            .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn gelu(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::gelu((&mut out).partition([128]), a.as_ref()).sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn gelu_backward(&self, gradient: &Buffer, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::gelu_backward((&mut out).partition([128]), gradient.as_ref(), a.as_ref())
            .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn inverse_sqrt(&self, a: &Buffer, epsilon: f32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::inverse_sqrt((&mut out).partition([128]), a.as_ref(), epsilon)
            .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn inverse_sqrt_backward(
        &self,
        gradient: &Buffer,
        a: &Buffer,
        epsilon: f32,
    ) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::inverse_sqrt_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            a.as_ref(),
            epsilon,
        )
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn binary_cross_entropy(&self, logits: &Buffer, targets: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(logits.shape()[0] as usize)?;
        kernels::binary_cross_entropy(
            (&mut out).partition([128]),
            logits.as_ref(),
            targets.as_ref(),
        )
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn binary_cross_entropy_backward(
        &self,
        gradient: &Buffer,
        logits: &Buffer,
        targets: &Buffer,
    ) -> Result<Buffer> {
        let mut out = self.zeros(logits.shape()[0] as usize)?;
        kernels::binary_cross_entropy_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            logits.as_ref(),
            targets.as_ref(),
        )
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn categorical_cross_entropy(
        &self,
        logits: &Buffer,
        targets: &Buffer,
        width: usize,
    ) -> Result<Buffer> {
        let rows = logits.shape()[0] as usize / width;
        let mut out = self.zeros(rows)?;
        kernels::categorical_cross_entropy(
            (&mut out).partition([1]),
            logits.as_ref(),
            targets.as_ref(),
            width as i32,
        )
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn categorical_cross_entropy_backward(
        &self,
        gradient: &Buffer,
        probability: &Buffer,
        targets: &Buffer,
        width: usize,
    ) -> Result<Buffer> {
        let mut out = self.zeros(probability.shape()[0] as usize)?;
        kernels::categorical_cross_entropy_backward(
            (&mut out).partition([1]),
            gradient.as_ref(),
            probability.as_ref(),
            targets.as_ref(),
            width as i32,
        )
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn mask(&self, a: &Buffer, keep: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::mask(
            (&mut out).partition([128]),
            a.as_ref(),
            keep.as_ref(),
            f32::NEG_INFINITY,
        )
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    /// Rows are contiguous after the tensor layer puts the reduced axis innermost.
    pub(crate) fn softmax(&self, a: &Buffer, width: usize) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::softmax((&mut out).partition([1]), a.as_ref(), width as i32)
            .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    pub(crate) fn softmax_backward(
        &self,
        gradient: &Buffer,
        probability: &Buffer,
        width: usize,
    ) -> Result<Buffer> {
        let mut out = self.zeros(probability.shape()[0] as usize)?;
        kernels::softmax_backward(
            (&mut out).partition([1]),
            gradient.as_ref(),
            probability.as_ref(),
            width as i32,
        )
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adam(
        &self,
        value: &Buffer,
        gradient: &Buffer,
        first: &Buffer,
        second: &Buffer,
        learning_rate: f32,
        learning_rates: Option<&Buffer>,
        beta1: f32,
        beta2: f32,
        correction1: f32,
        correction2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<(Buffer, Buffer, Buffer)> {
        let len = value.shape()[0] as usize;
        let mut next_first = self.zeros(len)?;
        kernels::adam_moment(
            (&mut next_first).partition([128]),
            gradient.as_ref(),
            first.as_ref(),
            beta1,
        )
        .generics(vec!["0".into()])
        .sync_on(&self.0.stream)?;
        let next_first = Arc::new(next_first);
        let mut next_second = self.zeros(len)?;
        kernels::adam_moment(
            (&mut next_second).partition([128]),
            gradient.as_ref(),
            second.as_ref(),
            beta2,
        )
        .generics(vec!["1".into()])
        .sync_on(&self.0.stream)?;
        let next_second = Arc::new(next_second);
        let mut updated = self.zeros(len)?;
        kernels::adam_update(
            (&mut updated).partition([128]),
            value.as_ref(),
            next_first.as_ref(),
            next_second.as_ref(),
            learning_rates.unwrap_or(first).as_ref(),
            learning_rate,
            correction1,
            correction2,
            epsilon,
            weight_decay,
        )
        .generics(vec![i32::from(learning_rates.is_some()).to_string()])
        .sync_on(&self.0.stream)?;
        Ok((Arc::new(updated), next_first, next_second))
    }
    /// One group per output; each contribution gathers one or two input elements.
    pub(crate) fn grouped(
        &self,
        a: &Buffer,
        b: Option<&Buffer>,
        plan: &Plan,
        scale: f32,
    ) -> Result<Buffer> {
        let upload = |v: &Vec<i32>| {
            api::copy_host_vec_to_device(&Arc::new(v.clone())).sync_on(&self.0.stream)
        };
        let offsets = upload(&plan.offsets)?;
        let left = upload(&plan.left)?;
        let right = upload(if b.is_some() { &plan.right } else { &plan.left })?;
        let mut out = self.zeros(plan.offsets.len() - 1)?;
        kernels::grouped(
            (&mut out).partition([1]),
            a.as_ref(),
            b.unwrap_or(a).as_ref(),
            &offsets,
            &left,
            &right,
            scale,
        )
        .generics(vec![i32::from(b.is_some()).to_string()])
        .sync_on(&self.0.stream)?;
        Ok(Arc::new(out))
    }
}

/// CSR gather/reduction plan. Deliberately bounded until a tiled lowering replaces it.
#[derive(Clone)]
pub(crate) struct Plan {
    pub offsets: Vec<i32>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
}
impl Plan {
    pub fn groups(groups: Vec<Vec<(usize, usize)>>, product: bool) -> Result<Self> {
        let count: usize = groups.iter().map(Vec::len).sum();
        Self::check_size(count)?;
        let mut result = Self {
            offsets: vec![0],
            left: Vec::with_capacity(count),
            right: Vec::new(),
        };
        for group in groups {
            for (a, b) in group {
                result.left.push(i32::try_from(a)?);
                if product {
                    result.right.push(i32::try_from(b)?);
                }
            }
            result.offsets.push(result.left.len() as i32);
        }
        Ok(result)
    }
    pub fn gather(map: &[usize]) -> Result<Self> {
        Self::check_size(map.len())?;
        Ok(Self {
            offsets: (0..=map.len() as i32).collect(),
            left: map
                .iter()
                .map(|&i| i32::try_from(i))
                .collect::<std::result::Result<_, _>>()?,
            right: vec![],
        })
    }
    pub fn reverse(map: &[usize], len: usize) -> Result<Self> {
        let mut groups = vec![vec![]; len];
        for (out, &input) in map.iter().enumerate() {
            groups[input].push((out, 0));
        }
        Self::groups(groups, false)
    }
    pub fn check_size(count: usize) -> Result<()> {
        if count == 0 || count > 16_777_216 {
            return Err(
                "index plan must have 1..=16777216 contributions (experimental backend limit)"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry()]
    fn binary_cross_entropy(
        out: &mut Tensor<f32, { [128] }>,
        logits: &Tensor<f32, { [-1] }>,
        targets: &Tensor<f32, { [-1] }>,
    ) {
        let z = logits.load_like(out);
        let y = targets.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let magnitude = max_tile(z, zero - z);
        out.store(max_tile(z, zero) - z * y + log(one + exp(zero - magnitude)));
    }
    #[cutile::entry()]
    fn binary_cross_entropy_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        logits: &Tensor<f32, { [-1] }>,
        targets: &Tensor<f32, { [-1] }>,
    ) {
        let z = logits.load_like(out);
        let y = targets.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let magnitude = max_tile(z, zero - z);
        let e = exp(zero - magnitude);
        let probability = select(gt_tile(z, zero), one / (one + e), e / (one + e));
        out.store(gradient.load_like(out) * (probability - y));
    }
    #[cutile::entry()]
    fn categorical_cross_entropy(
        out: &mut Tensor<f32, { [1] }>,
        logits: &Tensor<f32, { [-1] }>,
        targets: &Tensor<f32, { [-1] }>,
        width: i32,
    ) {
        let row = get_tile_block_id().0;
        let start = row * width;
        let zp = logits.partition(shape![1]);
        let yp = targets.partition(shape![1]);
        let mut maximum = zp.load([start]);
        for j in 1i32..width {
            maximum = max_tile(maximum, zp.load([start + j]));
        }
        let mut exponential_sum = constant(0.0f32, shape![1]);
        for j in 0i32..width {
            exponential_sum = exponential_sum + exp(zp.load([start + j]) - maximum);
        }
        let log_normalizer = maximum + log(exponential_sum);
        let mut loss = constant(0.0f32, shape![1]);
        for j in 0i32..width {
            loss = loss + yp.load([start + j]) * (log_normalizer - zp.load([start + j]));
        }
        out.store(loss);
    }
    #[cutile::entry()]
    fn categorical_cross_entropy_backward(
        out: &mut Tensor<f32, { [1] }>,
        gradient: &Tensor<f32, { [-1] }>,
        probability: &Tensor<f32, { [-1] }>,
        targets: &Tensor<f32, { [-1] }>,
        width: i32,
    ) {
        let position = get_tile_block_id().0;
        let row = position / width;
        let gp = gradient.partition(shape![1]);
        let pp = probability.partition(shape![1]);
        let yp = targets.partition(shape![1]);
        out.store(gp.load([row]) * (pp.load([position]) - yp.load([position])));
    }
    #[cutile::entry()]
    fn mask(
        out: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        keep: &Tensor<f32, { [-1] }>,
        masked: f32,
    ) {
        let zero = constant(0.0f32, shape![128]);
        out.store(select(
            gt_tile(keep.load_like(out), zero),
            a.load_like(out),
            broadcast_scalar(masked, shape![128]),
        ));
    }
    #[cutile::entry()]
    fn softmax(out: &mut Tensor<f32, { [1] }>, a: &Tensor<f32, { [-1] }>, width: i32) {
        let pid = get_tile_block_id().0;
        let start = (pid / width) * width;
        let ap = a.partition(shape![1]);
        let mut maximum = ap.load([start]);
        for j in 1i32..width {
            maximum = max_tile(maximum, ap.load([start + j]));
        }
        let mut sum = constant(0.0f32, shape![1]);
        for j in 0i32..width {
            sum = sum + exp(ap.load([start + j]) - maximum);
        }
        out.store(exp(ap.load([pid]) - maximum) / sum);
    }
    #[cutile::entry()]
    fn softmax_backward(
        out: &mut Tensor<f32, { [1] }>,
        gradient: &Tensor<f32, { [-1] }>,
        probability: &Tensor<f32, { [-1] }>,
        width: i32,
    ) {
        let pid = get_tile_block_id().0;
        let start = (pid / width) * width;
        let gp = gradient.partition(shape![1]);
        let pp = probability.partition(shape![1]);
        let mut dot = constant(0.0f32, shape![1]);
        for j in 0i32..width {
            dot = dot + gp.load([start + j]) * pp.load([start + j]);
        }
        out.store(pp.load([pid]) * (gp.load([pid]) - dot));
    }
    #[cutile::entry()]
    fn adam_moment<const SQUARE: i32>(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        previous: &Tensor<f32, { [-1] }>,
        beta: f32,
    ) {
        let g = gradient.load_like(out);
        let observation = if SQUARE == 1 { g * g } else { g };
        let b = broadcast_scalar(beta, shape![128]);
        let one = constant(1.0f32, shape![128]);
        out.store(b * previous.load_like(out) + (one - b) * observation);
    }
    #[cutile::entry()]
    fn adam_update<const PER_ELEMENT_RATE: i32>(
        out: &mut Tensor<f32, { [128] }>,
        value: &Tensor<f32, { [-1] }>,
        first: &Tensor<f32, { [-1] }>,
        second: &Tensor<f32, { [-1] }>,
        learning_rates: &Tensor<f32, { [-1] }>,
        learning_rate: f32,
        correction1: f32,
        correction2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) {
        let lr = if PER_ELEMENT_RATE == 1 {
            learning_rates.load_like(out)
        } else {
            broadcast_scalar(learning_rate, shape![128])
        };
        let c1 = broadcast_scalar(correction1, shape![128]);
        let c2 = broadcast_scalar(correction2, shape![128]);
        let eps = broadcast_scalar(epsilon, shape![128]);
        let decay = broadcast_scalar(weight_decay, shape![128]);
        let m = first.load_like(out) / c1;
        let v = second.load_like(out) / c2;
        out.store(
            value.load_like(out)
                - lr * (m / (sqrt(v, rounding::NearestEven, ftz::Disabled) + eps)
                    + decay * value.load_like(out)),
        );
    }
    #[cutile::entry()]
    fn binary<const OP: i32>(
        out: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let x = a.load_like(out);
        let y = b.load_like(out);
        if OP == 0 {
            out.store(x + y);
        } else if OP == 1 {
            out.store(x - y);
        } else {
            out.store(x * y);
        }
    }
    #[cutile::entry()]
    fn scale(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>, factor: f32) {
        let value = a.load_like(out) * broadcast_scalar(factor, shape![128]);
        out.store(value);
    }
    #[cutile::entry()]
    fn relu(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let zero: Tile<f32, { [128] }> = constant(0.0f32, shape![128]);
        out.store(max_tile(a.load_like(out), zero));
    }
    #[cutile::entry()]
    fn relu_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
    ) {
        let zero: Tile<f32, { [128] }> = constant(0.0f32, shape![128]);
        out.store(select(
            gt_tile(a.load_like(out), zero),
            gradient.load_like(out),
            zero,
        ));
    }
    #[cutile::entry()]
    fn gelu(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let x = a.load_like(out);
        let half = constant(0.5f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let c = constant(0.7978845608f32, shape![128]);
        let cubic = constant(0.044715f32, shape![128]);
        out.store(half * x * (one + tanh(c * (x + cubic * x * x * x))));
    }
    #[cutile::entry()]
    fn gelu_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
    ) {
        let x = a.load_like(out);
        let half = constant(0.5f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let three = constant(3.0f32, shape![128]);
        let c = constant(0.7978845608f32, shape![128]);
        let cubic = constant(0.044715f32, shape![128]);
        let t = tanh(c * (x + cubic * x * x * x));
        let derivative =
            half * (one + t) + half * x * (one - t * t) * c * (one + three * cubic * x * x);
        out.store(gradient.load_like(out) * derivative);
    }
    #[cutile::entry()]
    fn inverse_sqrt(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>, epsilon: f32) {
        let eps = broadcast_scalar(epsilon, shape![128]);
        let one: Tile<f32, { [128] }> = constant(1.0f32, shape![128]);
        out.store(one / sqrt(a.load_like(out) + eps, rounding::NearestEven, ftz::Disabled));
    }
    #[cutile::entry()]
    fn inverse_sqrt_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        epsilon: f32,
    ) {
        let eps = broadcast_scalar(epsilon, shape![128]);
        let root = sqrt(a.load_like(out) + eps, rounding::NearestEven, ftz::Disabled);
        let scale = broadcast_scalar(-0.5f32, shape![128]);
        out.store(gradient.load_like(out) * scale / (root * root * root));
    }
    #[cutile::entry()]
    fn grouped<const PRODUCT: i32>(
        out: &mut Tensor<f32, { [1] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        offsets: &Tensor<i32, { [-1] }>,
        left: &Tensor<i32, { [-1] }>,
        right: &Tensor<i32, { [-1] }>,
        scale: f32,
    ) {
        let pid = get_tile_block_id().0;
        let op = offsets.partition(shape![1]);
        let start: i32 = tile_to_scalar(op.load([pid]).reshape(shape![]));
        let end: i32 = tile_to_scalar(op.load([pid + 1i32]).reshape(shape![]));
        let lp = left.partition(shape![1]);
        let rp = right.partition(shape![1]);
        let ap = a.partition(shape![1]);
        let bp = b.partition(shape![1]);
        let mut sum = constant(0.0f32, shape![1]);
        for i in start..end {
            let li: i32 = tile_to_scalar(lp.load([i]).reshape(shape![]));
            let value = ap.load([li]);
            if PRODUCT == 1 {
                let ri: i32 = tile_to_scalar(rp.load([i]).reshape(shape![]));
                sum = sum + value * bp.load([ri]);
            } else {
                sum = sum + value;
            }
        }
        out.store(sum * broadcast_scalar(scale, shape![1]));
    }
}
