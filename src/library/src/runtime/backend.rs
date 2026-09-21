//! A correctness-first cuTile backend. Plans contain indices, never values.
use crate::Result;
use cutile::prelude::*;
use std::time::Instant;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

fn profile(label: &str, started: Instant) {
    if std::env::var_os("AXIS_PROFILE").is_some() {
        eprintln!(
            "axis_profile {label} {:.6}",
            started.elapsed().as_secs_f64()
        );
    }
}

pub(crate) type Buffer = Arc<cutile::tensor::Tensor<f32>>;

#[derive(Clone)]
pub struct Device(Rc<Context>);
struct Context {
    stream: Arc<cutile::cuda_core::Stream>,
    bf16_matmul: bool,
    pending: RefCell<Vec<Buffer>>,
    pending_f32: RefCell<Vec<Arc<Vec<f32>>>>,
    pending_host_i32: RefCell<Vec<Arc<Vec<i32>>>>,
    pending_device_i32: RefCell<Vec<Arc<cutile::tensor::Tensor<i32>>>>,
    plans: RefCell<HashMap<u64, DevicePlan>>,
    plan_bytes: Cell<usize>,
}

#[derive(Clone)]
struct DevicePlan {
    offsets: Arc<cutile::tensor::Tensor<i32>>,
    left: Arc<cutile::tensor::Tensor<i32>>,
    right: Arc<cutile::tensor::Tensor<i32>>,
}

trait Enqueue: DeviceOp + Sized {
    fn enqueue_on(
        self,
        stream: &Arc<cutile::cuda_core::Stream>,
    ) -> std::result::Result<<Self as DeviceOp>::Output, DeviceError> {
        // Axis owns every buffer until `Device::synchronize`; all work is
        // submitted to this one stream, so dependency order is preserved.
        unsafe { self.async_on(stream) }
    }
}
impl<T: DeviceOp> Enqueue for T {}

impl Device {
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Self::cuda_with_bf16(ordinal, false)
    }
    /// CUDA device with BF16 inputs and FP32 accumulation inside matrix products.
    /// Parameters, activations outside GEMM, optimizer state, and reductions remain FP32.
    pub fn cuda_bf16(ordinal: usize) -> Result<Self> {
        Self::cuda_with_bf16(ordinal, true)
    }
    fn cuda_with_bf16(ordinal: usize, bf16_matmul: bool) -> Result<Self> {
        let device = cutile::cuda_core::Device::new(ordinal)?;
        Ok(Self(Rc::new(Context {
            stream: device.new_stream()?,
            bf16_matmul,
            pending: RefCell::new(Vec::new()),
            pending_f32: RefCell::new(Vec::new()),
            pending_host_i32: RefCell::new(Vec::new()),
            pending_device_i32: RefCell::new(Vec::new()),
            plans: RefCell::new(HashMap::new()),
            plan_bytes: Cell::new(0),
        })))
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    fn track(&self, tensor: cutile::tensor::Tensor<f32>) -> Buffer {
        let buffer = Arc::new(tensor);
        self.0.pending.borrow_mut().push(buffer.clone());
        buffer
    }
    pub fn synchronize(&self) -> Result<()> {
        let started = Instant::now();
        unsafe { self.0.stream.synchronize()? };
        profile("synchronize", started);
        self.0.pending.borrow_mut().clear();
        self.0.pending_f32.borrow_mut().clear();
        self.0.pending_host_i32.borrow_mut().clear();
        self.0.pending_device_i32.borrow_mut().clear();
        Ok(())
    }
    pub(crate) fn upload(&self, values: Vec<f32>) -> Result<Buffer> {
        let values = Arc::new(values);
        let tensor = api::copy_host_vec_to_device(&values).enqueue_on(&self.0.stream)?;
        self.0.pending_f32.borrow_mut().push(values);
        Ok(self.track(tensor))
    }
    pub(crate) fn read(&self, buffer: &Buffer) -> Result<Vec<f32>> {
        let started = Instant::now();
        let values = buffer.to_host_vec().sync_on(&self.0.stream)?;
        profile("read", started);
        self.0.pending.borrow_mut().clear();
        self.0.pending_f32.borrow_mut().clear();
        self.0.pending_host_i32.borrow_mut().clear();
        self.0.pending_device_i32.borrow_mut().clear();
        Ok(values)
    }
    fn zeros(&self, len: usize) -> Result<cutile::tensor::Tensor<f32>> {
        Ok(api::zeros(&[len]).enqueue_on(&self.0.stream)?)
    }
    pub(crate) fn zeros_buffer(&self, len: usize) -> Result<Buffer> {
        let tensor = self.zeros(len)?;
        Ok(self.track(tensor))
    }
    /// Reduce a flat FP32 buffer to one device-resident sum of squares.
    ///
    /// This deliberately does not use Axis's general indexed reduction plan:
    /// Muon matrices can exceed that plan's contribution-count bound.
    pub(crate) fn sum_squares(&self, a: &Buffer) -> Result<Buffer> {
        const TILE_WIDTH: usize = 256;
        let len: usize = a.shape().iter().map(|&extent| extent as usize).product();
        if len == 0 {
            return Err("sum of squares requires a nonempty buffer".into());
        }
        let logical_len = i32::try_from(len).map_err(|_| "sum of squares exceeds i32 indexing")?;
        let partial_count = len.div_ceil(TILE_WIDTH);
        let mut partials = self.zeros(partial_count)?;
        kernels::sum_squares_tiles((&mut partials).partition([1]), a.as_ref(), logical_len)
            .generics(vec![(TILE_WIDTH as i32).to_string()])
            .enqueue_on(&self.0.stream)?;
        let mut current = self.track(partials);
        let mut current_len = partial_count;
        while current_len > 1 {
            let next_len = current_len.div_ceil(TILE_WIDTH);
            let mut next = self.zeros(next_len)?;
            kernels::sum_tiles(
                (&mut next).partition([1]),
                current.as_ref(),
                i32::try_from(current_len)
                    .map_err(|_| "sum of squares partial count exceeds i32 indexing")?,
            )
            .generics(vec![(TILE_WIDTH as i32).to_string()])
            .enqueue_on(&self.0.stream)?;
            current = self.track(next);
            current_len = next_len;
        }
        Ok(current)
    }
    pub(crate) fn binary(&self, a: &Buffer, b: &Buffer, op: i32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::binary((&mut out).partition([128]), a.as_ref(), b.as_ref())
            .generics(vec![op.to_string()])
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn mask_gradient(&self, gradient: &Buffer, winners: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(gradient.shape()[0] as usize)?;
        kernels::mask_gradient(
            (&mut out).partition([128]),
            gradient.as_ref(),
            winners.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn scale(&self, a: &Buffer, scale: f32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::scale((&mut out).partition([128]), a.as_ref(), scale)
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn multiply_scalar(&self, a: &Buffer, scalar: &Buffer) -> Result<Buffer> {
        if scalar
            .shape()
            .iter()
            .map(|&extent| extent as usize)
            .product::<usize>()
            != 1
        {
            return Err("device scalar multiplier must contain exactly one value".into());
        }
        let len = a.shape().iter().map(|&extent| extent as usize).product();
        let mut out = self.zeros(len)?;
        kernels::multiply_scalar((&mut out).partition([128]), a.as_ref(), scalar.as_ref())
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn relu(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::relu((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn relu_backward(&self, gradient: &Buffer, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::relu_backward((&mut out).partition([128]), gradient.as_ref(), a.as_ref())
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn gelu(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::gelu((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn gelu_backward(&self, gradient: &Buffer, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::gelu_backward((&mut out).partition([128]), gradient.as_ref(), a.as_ref())
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn inverse_sqrt(&self, a: &Buffer, epsilon: f32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::inverse_sqrt((&mut out).partition([128]), a.as_ref(), epsilon)
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
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
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn binary_cross_entropy(&self, logits: &Buffer, targets: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(logits.shape()[0] as usize)?;
        kernels::binary_cross_entropy(
            (&mut out).partition([128]),
            logits.as_ref(),
            targets.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
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
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
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
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
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
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn categorical_correct(
        &self,
        logits: &Buffer,
        targets: &Buffer,
        width: usize,
    ) -> Result<Buffer> {
        let rows = logits.shape()[0] as usize / width;
        let mut out = self.zeros(rows)?;
        kernels::categorical_correct(
            (&mut out).partition([1]),
            logits.as_ref(),
            targets.as_ref(),
            width as i32,
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn mask(&self, a: &Buffer, keep: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::mask(
            (&mut out).partition([128]),
            a.as_ref(),
            keep.as_ref(),
            f32::NEG_INFINITY,
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    /// Rows are contiguous after the tensor layer puts the reduced axis innermost.
    pub(crate) fn softmax(&self, a: &Buffer, width: usize) -> Result<Buffer> {
        let len = a.shape().iter().map(|&extent| extent as usize).product();
        let rows = len / width;
        let tile_width = width.next_power_of_two();
        let input = a.reshape(&[rows, width])?;
        let mut out = self.zeros(len)?.reshape(&[rows, width])?;
        kernels::softmax(
            (&mut out).partition([1, tile_width]),
            input.as_ref(),
            width as i32,
        )
        .generics(vec![(tile_width as i32).to_string()])
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out.reshape(&[len])?))
    }
    pub(crate) fn softmax_backward(
        &self,
        gradient: &Buffer,
        probability: &Buffer,
        width: usize,
    ) -> Result<Buffer> {
        let len = probability
            .shape()
            .iter()
            .map(|&extent| extent as usize)
            .product();
        let rows = len / width;
        let tile_width = width.next_power_of_two();
        let gradient = gradient.reshape(&[rows, width])?;
        let probability = probability.reshape(&[rows, width])?;
        let mut out = self.zeros(len)?.reshape(&[rows, width])?;
        kernels::softmax_backward(
            (&mut out).partition([1, tile_width]),
            gradient.as_ref(),
            probability.as_ref(),
            width as i32,
        )
        .generics(vec![(tile_width as i32).to_string()])
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out.reshape(&[len])?))
    }
    pub(crate) fn matmul(
        &self,
        a: &Buffer,
        b: &Buffer,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Buffer> {
        let a = a.reshape(&[batch, m, k])?;
        let b = b.reshape(&[batch, k, n])?;
        let mut out = self.zeros(batch * m * n)?.reshape(&[batch, m, n])?;
        let bk = contraction_tile(k);
        if self.0.bf16_matmul {
            kernels::matmul_bf16((&mut out).partition([1, 64, 64]), a.as_ref(), b.as_ref())
                .generics(vec![bk.to_string(), (k as i32).to_string()])
                .enqueue_on(&self.0.stream)?;
        } else {
            kernels::matmul((&mut out).partition([1, 64, 64]), a.as_ref(), b.as_ref())
                .generics(vec![bk.to_string(), (k as i32).to_string()])
                .enqueue_on(&self.0.stream)?;
        }
        Ok(self.track(out.reshape(&[batch * m * n])?))
    }
    pub(crate) fn matmul_left_backward(
        &self,
        gradient: &Buffer,
        rhs: &Buffer,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Buffer> {
        let gradient = gradient.reshape(&[batch, m, n])?;
        let rhs = rhs.reshape(&[batch, k, n])?;
        let mut out = self.zeros(batch * m * k)?.reshape(&[batch, m, k])?;
        let bn = contraction_tile(n);
        if self.0.bf16_matmul {
            kernels::matmul_left_backward_bf16(
                (&mut out).partition([1, 64, 64]),
                gradient.as_ref(),
                rhs.as_ref(),
            )
            .generics(vec![bn.to_string(), (n as i32).to_string()])
            .enqueue_on(&self.0.stream)?;
        } else {
            kernels::matmul_left_backward(
                (&mut out).partition([1, 64, 64]),
                gradient.as_ref(),
                rhs.as_ref(),
            )
            .generics(vec![bn.to_string(), (n as i32).to_string()])
            .enqueue_on(&self.0.stream)?;
        }
        Ok(self.track(out.reshape(&[batch * m * k])?))
    }
    pub(crate) fn matmul_right_backward(
        &self,
        lhs: &Buffer,
        gradient: &Buffer,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Buffer> {
        let lhs = lhs.reshape(&[batch, m, k])?;
        let gradient = gradient.reshape(&[batch, m, n])?;
        let mut out = self.zeros(batch * k * n)?.reshape(&[batch, k, n])?;
        let bm = contraction_tile(m);
        if self.0.bf16_matmul {
            kernels::matmul_right_backward_bf16(
                (&mut out).partition([1, 64, 64]),
                lhs.as_ref(),
                gradient.as_ref(),
            )
            .generics(vec![bm.to_string(), (m as i32).to_string()])
            .enqueue_on(&self.0.stream)?;
        } else {
            kernels::matmul_right_backward(
                (&mut out).partition([1, 64, 64]),
                lhs.as_ref(),
                gradient.as_ref(),
            )
            .generics(vec![bm.to_string(), (m as i32).to_string()])
            .enqueue_on(&self.0.stream)?;
        }
        Ok(self.track(out.reshape(&[batch * k * n])?))
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
        .enqueue_on(&self.0.stream)?;
        let next_first = self.track(next_first);
        let mut next_second = self.zeros(len)?;
        kernels::adam_moment(
            (&mut next_second).partition([128]),
            gradient.as_ref(),
            second.as_ref(),
            beta2,
        )
        .generics(vec!["1".into()])
        .enqueue_on(&self.0.stream)?;
        let next_second = self.track(next_second);
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
        .enqueue_on(&self.0.stream)?;
        Ok((self.track(updated), next_first, next_second))
    }
    /// One group per output; each contribution gathers one or two input elements.
    pub(crate) fn grouped(
        &self,
        a: &Buffer,
        b: Option<&Buffer>,
        plan: &Plan,
        scale: f32,
    ) -> Result<Buffer> {
        let started = Instant::now();
        let device_plan = self.device_plan(plan, b.is_some())?;
        let mut out = self.zeros(plan.offsets.len() - 1)?;
        kernels::grouped(
            (&mut out).partition([1]),
            a.as_ref(),
            b.unwrap_or(a).as_ref(),
            device_plan.offsets.as_ref(),
            device_plan.left.as_ref(),
            device_plan.right.as_ref(),
            scale,
        )
        .generics(vec![i32::from(b.is_some()).to_string()])
        .enqueue_on(&self.0.stream)?;
        profile("grouped_submit", started);
        Ok(self.track(out))
    }

    pub(crate) fn unfold(&self, input: &Buffer, spec: &UnfoldSpec) -> Result<Buffer> {
        let metadata = self.upload_i32(&spec.forward_metadata)?;
        let mut out = self.zeros(spec.output_len)?;
        match spec.spatial_rank {
            2 => {
                unsafe {
                    kernels::unfold2d(
                        (&mut out).partition([128]),
                        input.as_ref().device_pointer(),
                        metadata.as_ref(),
                        spec.output_len as i32,
                        spec.output_rank,
                        spec.channels_per_group,
                        spec.kernel[0],
                        spec.kernel[1],
                        spec.stride[0],
                        spec.stride[1],
                        spec.padding[0],
                        spec.padding[1],
                        spec.input_spatial[0],
                        spec.input_spatial[1],
                        spec.input_special_strides[0],
                        spec.input_special_strides[1],
                        spec.input_special_strides[2],
                    )
                }
                .enqueue_on(&self.0.stream)?;
            }
            3 => {
                unsafe {
                    kernels::unfold3d(
                        (&mut out).partition([128]),
                        input.as_ref().device_pointer(),
                        metadata.as_ref(),
                        spec.output_len as i32,
                        spec.output_rank,
                        spec.channels_per_group,
                        spec.kernel[0],
                        spec.kernel[1],
                        spec.kernel[2],
                        spec.stride[0],
                        spec.stride[1],
                        spec.stride[2],
                        spec.padding[0],
                        spec.padding[1],
                        spec.padding[2],
                        spec.input_spatial[0],
                        spec.input_spatial[1],
                        spec.input_spatial[2],
                        spec.input_special_strides[0],
                        spec.input_special_strides[1],
                        spec.input_special_strides[2],
                        spec.input_special_strides[3],
                    )
                }
                .enqueue_on(&self.0.stream)?;
            }
            _ => return Err("unfold supports exactly two or three spatial axes".into()),
        }
        Ok(self.track(out))
    }

    pub(crate) fn unfold_backward(&self, gradient: &Buffer, spec: &UnfoldSpec) -> Result<Buffer> {
        let metadata = self.upload_i32(&spec.backward_metadata)?;
        let mut out = self.zeros(spec.input_len)?;
        match spec.spatial_rank {
            2 => {
                unsafe {
                    kernels::unfold2d_backward(
                        (&mut out).partition([128]),
                        gradient.as_ref().device_pointer(),
                        metadata.as_ref(),
                        spec.input_len as i32,
                        spec.input_rank,
                        spec.channels_per_group,
                        spec.kernel[0],
                        spec.kernel[1],
                        spec.stride[0],
                        spec.stride[1],
                        spec.padding[0],
                        spec.padding[1],
                        spec.output_spatial[0],
                        spec.output_spatial[1],
                        spec.output_special_strides[0],
                        spec.output_special_strides[1],
                        spec.output_special_strides[3],
                        spec.output_special_strides[4],
                    )
                }
                .enqueue_on(&self.0.stream)?;
            }
            3 => {
                unsafe {
                    kernels::unfold3d_backward(
                        (&mut out).partition([128]),
                        gradient.as_ref().device_pointer(),
                        metadata.as_ref(),
                        spec.input_len as i32,
                        spec.input_rank,
                        spec.channels_per_group,
                        spec.kernel[0],
                        spec.kernel[1],
                        spec.kernel[2],
                        spec.stride[0],
                        spec.stride[1],
                        spec.stride[2],
                        spec.padding[0],
                        spec.padding[1],
                        spec.padding[2],
                        spec.output_spatial[0],
                        spec.output_spatial[1],
                        spec.output_spatial[2],
                        spec.output_special_strides[0],
                        spec.output_special_strides[1],
                        spec.output_special_strides[2],
                        spec.output_special_strides[3],
                        spec.output_special_strides[4],
                    )
                }
                .enqueue_on(&self.0.stream)?;
            }
            _ => return Err("unfold supports exactly two or three spatial axes".into()),
        }
        Ok(self.track(out))
    }

    /// One minimum per CSR group plus a one-hot winner mask in input storage order.
    pub(crate) fn grouped_minimum(
        &self,
        a: &Buffer,
        forward: &Plan,
        reverse: &Plan,
    ) -> Result<(Buffer, Buffer)> {
        let forward_plan = self.device_plan(forward, false)?;
        let reverse_plan = self.device_plan(reverse, false)?;
        let groups = forward.offsets.len() - 1;
        let mut out = self.zeros(groups)?;
        let mut winner_indices = api::zeros::<i32>(&[groups]).enqueue_on(&self.0.stream)?;
        kernels::grouped_minimum(
            (&mut out).partition([1]),
            (&mut winner_indices).partition([1]),
            a.as_ref(),
            forward_plan.offsets.as_ref(),
            forward_plan.left.as_ref(),
            f32::MIN,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        )
        .enqueue_on(&self.0.stream)?;
        let mut winner_mask = self.zeros(a.shape()[0] as usize)?;
        kernels::minimum_winner_mask(
            (&mut winner_mask).partition([1]),
            &winner_indices,
            reverse_plan.left.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        self.0
            .pending_device_i32
            .borrow_mut()
            .push(Arc::new(winner_indices));
        Ok((self.track(out), self.track(winner_mask)))
    }

    fn upload_i32(&self, values: &[i32]) -> Result<Arc<cutile::tensor::Tensor<i32>>> {
        let host = Arc::new(values.to_vec());
        let device = Arc::new(api::copy_host_vec_to_device(&host).enqueue_on(&self.0.stream)?);
        self.0.pending_host_i32.borrow_mut().push(host);
        self.0.pending_device_i32.borrow_mut().push(device.clone());
        Ok(device)
    }

    fn device_plan(&self, plan: &Plan, product: bool) -> Result<DevicePlan> {
        let cached = self.0.plans.borrow().get(&plan.id).cloned();
        if let Some(cached) = cached {
            return Ok(cached);
        }
        let uploaded = DevicePlan {
            offsets: self.upload_i32(&plan.offsets)?,
            left: self.upload_i32(&plan.left)?,
            right: self.upload_i32(if product { &plan.right } else { &plan.left })?,
        };
        let bytes = plan
            .offsets
            .len()
            .saturating_add(plan.left.len())
            .saturating_add(if product {
                plan.right.len()
            } else {
                plan.left.len()
            })
            .saturating_mul(std::mem::size_of::<i32>());
        const PLAN_CACHE_BYTES: usize = 256 * 1024 * 1024;
        if self.0.plan_bytes.get().saturating_add(bytes) <= PLAN_CACHE_BYTES {
            self.0.plan_bytes.set(self.0.plan_bytes.get() + bytes);
            self.0.plans.borrow_mut().insert(plan.id, uploaded.clone());
        }
        Ok(uploaded)
    }
}

/// Compact convolution patch geometry. Per-element source indices are computed
/// by the kernels, so storage grows with tensor rank rather than patch count.
#[derive(Clone)]
pub(crate) struct UnfoldSpec {
    pub spatial_rank: i32,
    pub input_len: usize,
    pub output_len: usize,
    pub input_rank: i32,
    pub output_rank: i32,
    pub forward_metadata: Vec<i32>,
    pub backward_metadata: Vec<i32>,
    pub channels_per_group: i32,
    pub kernel: [i32; 3],
    pub stride: [i32; 3],
    pub padding: [i32; 3],
    pub input_spatial: [i32; 3],
    pub output_spatial: [i32; 3],
    pub input_special_strides: [i32; 4],
    pub output_special_strides: [i32; 5],
}

impl UnfoldSpec {
    #[cfg(test)]
    pub(crate) fn metadata_len(&self) -> usize {
        self.forward_metadata.len() + self.backward_metadata.len()
    }
}

fn contraction_tile(extent: usize) -> i32 {
    [16, 8, 4, 2]
        .into_iter()
        .find(|tile| extent.is_multiple_of(*tile))
        .unwrap_or(1) as i32
}

/// CSR gather/reduction plan. Deliberately bounded until a tiled lowering replaces it.
static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub(crate) struct Plan {
    id: u64,
    pub offsets: Vec<i32>,
    pub left: Vec<i32>,
    pub right: Vec<i32>,
}
impl Plan {
    pub(crate) fn retained_bytes(&self) -> usize {
        [&self.offsets, &self.left, &self.right]
            .into_iter()
            .map(|values| values.capacity() * std::mem::size_of::<i32>())
            .sum()
    }

    pub fn groups(groups: Vec<Vec<(usize, usize)>>, product: bool) -> Result<Self> {
        let count: usize = groups.iter().map(Vec::len).sum();
        Self::check_size(count)?;
        let mut result = Self {
            id: NEXT_PLAN.fetch_add(1, Ordering::Relaxed),
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
            id: NEXT_PLAN.fetch_add(1, Ordering::Relaxed),
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
    unsafe fn unfold2d(
        out: &mut Tensor<f32, { [128] }>,
        input: *const f32,
        metadata: &Tensor<i32, { [-1] }>,
        output_len: i32,
        rank: i32,
        channels_per_group: i32,
        kernel_y: i32,
        kernel_x: i32,
        stride_y: i32,
        stride_x: i32,
        padding_y: i32,
        padding_x: i32,
        input_height: i32,
        input_width: i32,
        input_channel_stride: i32,
        input_height_stride: i32,
        input_width_stride: i32,
    ) {
        let output_index: Tile<i32, { [128] }> =
            iota(shape![128]) + broadcast_scalar(get_tile_block_id().0 * 128i32, shape![128]);
        let live = lt_tile(output_index, broadcast_scalar(output_len, shape![128]));
        let mp = metadata.partition(shape![1]);
        let mut input_index = constant(0i32, shape![128]);
        let mut output_y = constant(0i32, shape![128]);
        let mut output_x = constant(0i32, shape![128]);
        let mut group = constant(0i32, shape![128]);
        let mut patch = constant(0i32, shape![128]);
        for dimension in 0i32..rank {
            let base = dimension * 4i32;
            let extent: i32 = tile_to_scalar(mp.load([base]).reshape(shape![]));
            let output_stride: i32 = tile_to_scalar(mp.load([base + 1i32]).reshape(shape![]));
            let input_stride: i32 = tile_to_scalar(mp.load([base + 2i32]).reshape(shape![]));
            let role: i32 = tile_to_scalar(mp.load([base + 3i32]).reshape(shape![]));
            let coordinate = (output_index / broadcast_scalar(output_stride, shape![128]))
                % broadcast_scalar(extent, shape![128]);
            if role == 0i32 {
                input_index =
                    input_index + coordinate * broadcast_scalar(input_stride, shape![128]);
            } else if role == 1i32 {
                output_y = coordinate;
            } else if role == 2i32 {
                output_x = coordinate;
            } else if role == 3i32 {
                group = coordinate;
            } else {
                patch = coordinate;
            }
        }
        let kernel_column = patch % broadcast_scalar(kernel_x, shape![128]);
        let rest = patch / broadcast_scalar(kernel_x, shape![128]);
        let kernel_row = rest % broadcast_scalar(kernel_y, shape![128]);
        let channel_in_group = rest / broadcast_scalar(kernel_y, shape![128]);
        let channel = group * broadcast_scalar(channels_per_group, shape![128]) + channel_in_group;
        let input_y = output_y * broadcast_scalar(stride_y, shape![128]) + kernel_row
            - broadcast_scalar(padding_y, shape![128]);
        let input_x = output_x * broadcast_scalar(stride_x, shape![128]) + kernel_column
            - broadcast_scalar(padding_x, shape![128]);
        let zero = constant(0i32, shape![128]);
        let valid = live
            & ge_tile(input_y, zero)
            & lt_tile(input_y, broadcast_scalar(input_height, shape![128]))
            & ge_tile(input_x, zero)
            & lt_tile(input_x, broadcast_scalar(input_width, shape![128]));
        input_index = input_index
            + channel * broadcast_scalar(input_channel_stride, shape![128])
            + input_y * broadcast_scalar(input_height_stride, shape![128])
            + input_x * broadcast_scalar(input_width_stride, shape![128]);
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [128] }> = base.broadcast(shape![128]);
        let addresses = addptr_tile(base, select(valid, input_index, zero));
        let (values, _token): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                addresses,
                ordering::Relaxed,
                Some(scope::Device),
                Some(valid),
                Some(0.0f32),
                None,
                Latency::<0>,
            )
        };
        out.store(values);
    }

    #[cutile::entry()]
    unsafe fn unfold3d(
        out: &mut Tensor<f32, { [128] }>,
        input: *const f32,
        metadata: &Tensor<i32, { [-1] }>,
        output_len: i32,
        rank: i32,
        channels_per_group: i32,
        kernel_z: i32,
        kernel_y: i32,
        kernel_x: i32,
        stride_z: i32,
        stride_y: i32,
        stride_x: i32,
        padding_z: i32,
        padding_y: i32,
        padding_x: i32,
        input_depth: i32,
        input_height: i32,
        input_width: i32,
        input_channel_stride: i32,
        input_depth_stride: i32,
        input_height_stride: i32,
        input_width_stride: i32,
    ) {
        let output_index: Tile<i32, { [128] }> =
            iota(shape![128]) + broadcast_scalar(get_tile_block_id().0 * 128i32, shape![128]);
        let live = lt_tile(output_index, broadcast_scalar(output_len, shape![128]));
        let mp = metadata.partition(shape![1]);
        let mut input_index = constant(0i32, shape![128]);
        let mut output_z = constant(0i32, shape![128]);
        let mut output_y = constant(0i32, shape![128]);
        let mut output_x = constant(0i32, shape![128]);
        let mut group = constant(0i32, shape![128]);
        let mut patch = constant(0i32, shape![128]);
        for dimension in 0i32..rank {
            let base = dimension * 4i32;
            let extent: i32 = tile_to_scalar(mp.load([base]).reshape(shape![]));
            let output_stride: i32 = tile_to_scalar(mp.load([base + 1i32]).reshape(shape![]));
            let input_stride: i32 = tile_to_scalar(mp.load([base + 2i32]).reshape(shape![]));
            let role: i32 = tile_to_scalar(mp.load([base + 3i32]).reshape(shape![]));
            let coordinate = (output_index / broadcast_scalar(output_stride, shape![128]))
                % broadcast_scalar(extent, shape![128]);
            if role == 0i32 {
                input_index =
                    input_index + coordinate * broadcast_scalar(input_stride, shape![128]);
            } else if role == 1i32 {
                output_z = coordinate;
            } else if role == 2i32 {
                output_y = coordinate;
            } else if role == 3i32 {
                output_x = coordinate;
            } else if role == 4i32 {
                group = coordinate;
            } else {
                patch = coordinate;
            }
        }
        let kernel_column = patch % broadcast_scalar(kernel_x, shape![128]);
        let rest = patch / broadcast_scalar(kernel_x, shape![128]);
        let kernel_row = rest % broadcast_scalar(kernel_y, shape![128]);
        let rest = rest / broadcast_scalar(kernel_y, shape![128]);
        let kernel_depth = rest % broadcast_scalar(kernel_z, shape![128]);
        let channel_in_group = rest / broadcast_scalar(kernel_z, shape![128]);
        let channel = group * broadcast_scalar(channels_per_group, shape![128]) + channel_in_group;
        let input_z = output_z * broadcast_scalar(stride_z, shape![128]) + kernel_depth
            - broadcast_scalar(padding_z, shape![128]);
        let input_y = output_y * broadcast_scalar(stride_y, shape![128]) + kernel_row
            - broadcast_scalar(padding_y, shape![128]);
        let input_x = output_x * broadcast_scalar(stride_x, shape![128]) + kernel_column
            - broadcast_scalar(padding_x, shape![128]);
        let zero = constant(0i32, shape![128]);
        let valid = live
            & ge_tile(input_z, zero)
            & lt_tile(input_z, broadcast_scalar(input_depth, shape![128]))
            & ge_tile(input_y, zero)
            & lt_tile(input_y, broadcast_scalar(input_height, shape![128]))
            & ge_tile(input_x, zero)
            & lt_tile(input_x, broadcast_scalar(input_width, shape![128]));
        input_index = input_index
            + channel * broadcast_scalar(input_channel_stride, shape![128])
            + input_z * broadcast_scalar(input_depth_stride, shape![128])
            + input_y * broadcast_scalar(input_height_stride, shape![128])
            + input_x * broadcast_scalar(input_width_stride, shape![128]);
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [128] }> = base.broadcast(shape![128]);
        let addresses = addptr_tile(base, select(valid, input_index, zero));
        let (values, _token): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                addresses,
                ordering::Relaxed,
                Some(scope::Device),
                Some(valid),
                Some(0.0f32),
                None,
                Latency::<0>,
            )
        };
        out.store(values);
    }

    #[cutile::entry()]
    unsafe fn unfold2d_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: *const f32,
        metadata: &Tensor<i32, { [-1] }>,
        input_len: i32,
        rank: i32,
        channels_per_group: i32,
        kernel_y: i32,
        kernel_x: i32,
        stride_y: i32,
        stride_x: i32,
        padding_y: i32,
        padding_x: i32,
        output_height: i32,
        output_width: i32,
        output_height_stride: i32,
        output_width_stride: i32,
        output_group_stride: i32,
        output_patch_stride: i32,
    ) {
        let input_index: Tile<i32, { [128] }> =
            iota(shape![128]) + broadcast_scalar(get_tile_block_id().0 * 128i32, shape![128]);
        let live = lt_tile(input_index, broadcast_scalar(input_len, shape![128]));
        let mp = metadata.partition(shape![1]);
        let mut output_base = constant(0i32, shape![128]);
        let mut channel = constant(0i32, shape![128]);
        let mut input_y = constant(0i32, shape![128]);
        let mut input_x = constant(0i32, shape![128]);
        for dimension in 0i32..rank {
            let base = dimension * 4i32;
            let extent: i32 = tile_to_scalar(mp.load([base]).reshape(shape![]));
            let input_stride: i32 = tile_to_scalar(mp.load([base + 1i32]).reshape(shape![]));
            let output_stride: i32 = tile_to_scalar(mp.load([base + 2i32]).reshape(shape![]));
            let role: i32 = tile_to_scalar(mp.load([base + 3i32]).reshape(shape![]));
            let coordinate = (input_index / broadcast_scalar(input_stride, shape![128]))
                % broadcast_scalar(extent, shape![128]);
            if role == 0i32 {
                output_base =
                    output_base + coordinate * broadcast_scalar(output_stride, shape![128]);
            } else if role == 1i32 {
                channel = coordinate;
            } else if role == 2i32 {
                input_y = coordinate;
            } else {
                input_x = coordinate;
            }
        }
        let group = channel / broadcast_scalar(channels_per_group, shape![128]);
        let channel_in_group = channel % broadcast_scalar(channels_per_group, shape![128]);
        let gradient_base: PointerTile<*const f32, { [] }> = pointer_to_tile(gradient);
        let gradient_base: PointerTile<*const f32, { [1] }> = gradient_base.reshape(shape![1]);
        let gradient_base: PointerTile<*const f32, { [128] }> =
            gradient_base.broadcast(shape![128]);
        let mut sum = constant(0.0f32, shape![128]);
        let zero: Tile<i32, { [128] }> = constant(0i32, shape![128]);
        for kernel_row in 0i32..kernel_y {
            let padded_y = input_y + broadcast_scalar(padding_y - kernel_row, shape![128]);
            let output_y = padded_y / broadcast_scalar(stride_y, shape![128]);
            let valid_y = ge_tile(padded_y, zero)
                & eq_tile(padded_y % broadcast_scalar(stride_y, shape![128]), zero)
                & lt_tile(output_y, broadcast_scalar(output_height, shape![128]));
            for kernel_column in 0i32..kernel_x {
                let padded_x = input_x + broadcast_scalar(padding_x - kernel_column, shape![128]);
                let output_x = padded_x / broadcast_scalar(stride_x, shape![128]);
                let valid = live
                    & valid_y
                    & ge_tile(padded_x, zero)
                    & eq_tile(padded_x % broadcast_scalar(stride_x, shape![128]), zero)
                    & lt_tile(output_x, broadcast_scalar(output_width, shape![128]));
                let patch = (channel_in_group * broadcast_scalar(kernel_y, shape![128])
                    + broadcast_scalar(kernel_row, shape![128]))
                    * broadcast_scalar(kernel_x, shape![128])
                    + broadcast_scalar(kernel_column, shape![128]);
                let output_index = output_base
                    + output_y * broadcast_scalar(output_height_stride, shape![128])
                    + output_x * broadcast_scalar(output_width_stride, shape![128])
                    + group * broadcast_scalar(output_group_stride, shape![128])
                    + patch * broadcast_scalar(output_patch_stride, shape![128]);
                let addresses = addptr_tile(gradient_base, select(valid, output_index, zero));
                let (values, _token): (Tile<f32, { [128] }>, Token) = unsafe {
                    load_ptr_tko(
                        addresses,
                        ordering::Relaxed,
                        Some(scope::Device),
                        Some(valid),
                        Some(0.0f32),
                        None,
                        Latency::<0>,
                    )
                };
                sum = sum + values;
            }
        }
        out.store(sum);
    }
    #[cutile::entry()]
    unsafe fn unfold3d_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: *const f32,
        metadata: &Tensor<i32, { [-1] }>,
        input_len: i32,
        rank: i32,
        channels_per_group: i32,
        kernel_z: i32,
        kernel_y: i32,
        kernel_x: i32,
        stride_z: i32,
        stride_y: i32,
        stride_x: i32,
        padding_z: i32,
        padding_y: i32,
        padding_x: i32,
        output_depth: i32,
        output_height: i32,
        output_width: i32,
        output_depth_stride: i32,
        output_height_stride: i32,
        output_width_stride: i32,
        output_group_stride: i32,
        output_patch_stride: i32,
    ) {
        let input_index: Tile<i32, { [128] }> =
            iota(shape![128]) + broadcast_scalar(get_tile_block_id().0 * 128i32, shape![128]);
        let live = lt_tile(input_index, broadcast_scalar(input_len, shape![128]));
        let mp = metadata.partition(shape![1]);
        let mut output_base = constant(0i32, shape![128]);
        let mut channel = constant(0i32, shape![128]);
        let mut input_z = constant(0i32, shape![128]);
        let mut input_y = constant(0i32, shape![128]);
        let mut input_x = constant(0i32, shape![128]);
        for dimension in 0i32..rank {
            let base = dimension * 4i32;
            let extent: i32 = tile_to_scalar(mp.load([base]).reshape(shape![]));
            let input_stride: i32 = tile_to_scalar(mp.load([base + 1i32]).reshape(shape![]));
            let output_stride: i32 = tile_to_scalar(mp.load([base + 2i32]).reshape(shape![]));
            let role: i32 = tile_to_scalar(mp.load([base + 3i32]).reshape(shape![]));
            let coordinate = (input_index / broadcast_scalar(input_stride, shape![128]))
                % broadcast_scalar(extent, shape![128]);
            if role == 0i32 {
                output_base =
                    output_base + coordinate * broadcast_scalar(output_stride, shape![128]);
            } else if role == 1i32 {
                channel = coordinate;
            } else if role == 2i32 {
                input_z = coordinate;
            } else if role == 3i32 {
                input_y = coordinate;
            } else {
                input_x = coordinate;
            }
        }
        let group = channel / broadcast_scalar(channels_per_group, shape![128]);
        let channel_in_group = channel % broadcast_scalar(channels_per_group, shape![128]);
        let gradient_base: PointerTile<*const f32, { [] }> = pointer_to_tile(gradient);
        let gradient_base: PointerTile<*const f32, { [1] }> = gradient_base.reshape(shape![1]);
        let gradient_base: PointerTile<*const f32, { [128] }> =
            gradient_base.broadcast(shape![128]);
        let mut sum = constant(0.0f32, shape![128]);
        let zero: Tile<i32, { [128] }> = constant(0i32, shape![128]);
        for kernel_depth in 0i32..kernel_z {
            let padded_z = input_z + broadcast_scalar(padding_z - kernel_depth, shape![128]);
            let output_z = padded_z / broadcast_scalar(stride_z, shape![128]);
            let valid_z = ge_tile(padded_z, zero)
                & eq_tile(padded_z % broadcast_scalar(stride_z, shape![128]), zero)
                & lt_tile(output_z, broadcast_scalar(output_depth, shape![128]));
            for kernel_row in 0i32..kernel_y {
                let padded_y = input_y + broadcast_scalar(padding_y - kernel_row, shape![128]);
                let output_y = padded_y / broadcast_scalar(stride_y, shape![128]);
                let valid_y = valid_z
                    & ge_tile(padded_y, zero)
                    & eq_tile(padded_y % broadcast_scalar(stride_y, shape![128]), zero)
                    & lt_tile(output_y, broadcast_scalar(output_height, shape![128]));
                for kernel_column in 0i32..kernel_x {
                    let padded_x =
                        input_x + broadcast_scalar(padding_x - kernel_column, shape![128]);
                    let output_x = padded_x / broadcast_scalar(stride_x, shape![128]);
                    let valid = live
                        & valid_y
                        & ge_tile(padded_x, zero)
                        & eq_tile(padded_x % broadcast_scalar(stride_x, shape![128]), zero)
                        & lt_tile(output_x, broadcast_scalar(output_width, shape![128]));
                    let patch = ((channel_in_group * broadcast_scalar(kernel_z, shape![128])
                        + broadcast_scalar(kernel_depth, shape![128]))
                        * broadcast_scalar(kernel_y, shape![128])
                        + broadcast_scalar(kernel_row, shape![128]))
                        * broadcast_scalar(kernel_x, shape![128])
                        + broadcast_scalar(kernel_column, shape![128]);
                    let output_index = output_base
                        + output_z * broadcast_scalar(output_depth_stride, shape![128])
                        + output_y * broadcast_scalar(output_height_stride, shape![128])
                        + output_x * broadcast_scalar(output_width_stride, shape![128])
                        + group * broadcast_scalar(output_group_stride, shape![128])
                        + patch * broadcast_scalar(output_patch_stride, shape![128]);
                    let addresses = addptr_tile(gradient_base, select(valid, output_index, zero));
                    let (values, _token): (Tile<f32, { [128] }>, Token) = unsafe {
                        load_ptr_tko(
                            addresses,
                            ordering::Relaxed,
                            Some(scope::Device),
                            Some(valid),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        )
                    };
                    sum = sum + values;
                }
            }
        }
        out.store(sum);
    }

    #[cutile::entry()]
    fn sum_squares_tiles<const TILE_WIDTH: i32>(
        out: &mut Tensor<f32, { [1] }>,
        input: &Tensor<f32, { [-1] }>,
        len: i32,
    ) {
        let block = get_tile_block_id().0;
        let offsets: Tile<i32, { [TILE_WIDTH] }> =
            iota(shape![TILE_WIDTH]) + broadcast_scalar(block * TILE_WIDTH, shape![TILE_WIDTH]);
        let valid = lt_tile(offsets, broadcast_scalar(len, shape![TILE_WIDTH]));
        let loaded: Tile<f32, { [TILE_WIDTH] }> = input.partition(shape![TILE_WIDTH]).load([block]);
        let zero = constant(0.0f32, shape![TILE_WIDTH]);
        let values = select(valid, loaded, zero);
        let sum: Tile<f32, { [] }> = reduce_sum(values * values, 0i32);
        out.store(sum.reshape(shape![1]));
    }
    #[cutile::entry()]
    fn sum_tiles<const TILE_WIDTH: i32>(
        out: &mut Tensor<f32, { [1] }>,
        input: &Tensor<f32, { [-1] }>,
        len: i32,
    ) {
        let block = get_tile_block_id().0;
        let offsets: Tile<i32, { [TILE_WIDTH] }> =
            iota(shape![TILE_WIDTH]) + broadcast_scalar(block * TILE_WIDTH, shape![TILE_WIDTH]);
        let valid = lt_tile(offsets, broadcast_scalar(len, shape![TILE_WIDTH]));
        let loaded: Tile<f32, { [TILE_WIDTH] }> = input.partition(shape![TILE_WIDTH]).load([block]);
        let zero = constant(0.0f32, shape![TILE_WIDTH]);
        let sum: Tile<f32, { [] }> = reduce_sum(select(valid, loaded, zero), 0i32);
        out.store(sum.reshape(shape![1]));
    }
    #[cutile::entry()]
    fn multiply_scalar(
        out: &mut Tensor<f32, { [128] }>,
        input: &Tensor<f32, { [-1] }>,
        scalar: &Tensor<f32, { [-1] }>,
    ) {
        let value: Tile<f32, { [1] }> = scalar.partition(shape![1]).load([0i32]);
        out.store(input.load_like(out) * value.broadcast(shape![128]));
    }
    #[cutile::entry()]
    fn matmul<const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [1, 64, 64] }>,
        a: &Tensor<f32, { [-1, -1, K] }>,
        b: &Tensor<f32, { [-1, K, -1] }>,
    ) {
        let ap = a.partition(shape![1, 64, BK]);
        let bp = b.partition(shape![1, BK, 64]);
        let pid = get_tile_block_id();
        let mut value = load_tile_mut(out).reshape(shape![64, 64]);
        for block in 0i32..(K / BK) {
            value = mma(
                ap.load([pid.0, pid.1, block]).reshape(shape![64, BK]),
                bp.load([pid.0, block, pid.2]).reshape(shape![BK, 64]),
                value,
            );
        }
        out.store(value.reshape(shape![1, 64, 64]));
    }
    #[cutile::entry()]
    fn matmul_bf16<const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [1, 64, 64] }>,
        a: &Tensor<f32, { [-1, -1, K] }>,
        b: &Tensor<f32, { [-1, K, -1] }>,
    ) {
        let ap = a.partition(shape![1, 64, BK]);
        let bp = b.partition(shape![1, BK, 64]);
        let pid = get_tile_block_id();
        let mut value = load_tile_mut(out).reshape(shape![64, 64]);
        for block in 0i32..(K / BK) {
            let left: Tile<bf16, { [64, BK] }> =
                convert_tile(ap.load([pid.0, pid.1, block]).reshape(shape![64, BK]));
            let right: Tile<bf16, { [BK, 64] }> =
                convert_tile(bp.load([pid.0, block, pid.2]).reshape(shape![BK, 64]));
            value = mma(left, right, value);
        }
        out.store(value.reshape(shape![1, 64, 64]));
    }
    #[cutile::entry()]
    fn matmul_left_backward<const BN: i32, const N: i32>(
        out: &mut Tensor<f32, { [1, 64, 64] }>,
        gradient: &Tensor<f32, { [-1, -1, N] }>,
        rhs: &Tensor<f32, { [-1, -1, N] }>,
    ) {
        let gp = gradient.partition(shape![1, 64, BN]);
        let rp = rhs.partition(shape![1, 64, BN]);
        let pid = get_tile_block_id();
        let mut value = load_tile_mut(out).reshape(shape![64, 64]);
        for block in 0i32..(N / BN) {
            value = mma(
                gp.load([pid.0, pid.1, block]).reshape(shape![64, BN]),
                rp.load([pid.0, pid.2, block])
                    .reshape(shape![64, BN])
                    .transpose(),
                value,
            );
        }
        out.store(value.reshape(shape![1, 64, 64]));
    }
    #[cutile::entry()]
    fn matmul_left_backward_bf16<const BN: i32, const N: i32>(
        out: &mut Tensor<f32, { [1, 64, 64] }>,
        gradient: &Tensor<f32, { [-1, -1, N] }>,
        rhs: &Tensor<f32, { [-1, -1, N] }>,
    ) {
        let gp = gradient.partition(shape![1, 64, BN]);
        let rp = rhs.partition(shape![1, 64, BN]);
        let pid = get_tile_block_id();
        let mut value = load_tile_mut(out).reshape(shape![64, 64]);
        for block in 0i32..(N / BN) {
            let left: Tile<bf16, { [64, BN] }> =
                convert_tile(gp.load([pid.0, pid.1, block]).reshape(shape![64, BN]));
            let right: Tile<bf16, { [BN, 64] }> = convert_tile(
                rp.load([pid.0, pid.2, block])
                    .reshape(shape![64, BN])
                    .transpose(),
            );
            value = mma(left, right, value);
        }
        out.store(value.reshape(shape![1, 64, 64]));
    }
    #[cutile::entry()]
    fn matmul_right_backward<const BM: i32, const M: i32>(
        out: &mut Tensor<f32, { [1, 64, 64] }>,
        lhs: &Tensor<f32, { [-1, M, -1] }>,
        gradient: &Tensor<f32, { [-1, M, -1] }>,
    ) {
        let lp = lhs.partition(shape![1, BM, 64]);
        let gp = gradient.partition(shape![1, BM, 64]);
        let pid = get_tile_block_id();
        let mut value = load_tile_mut(out).reshape(shape![64, 64]);
        for block in 0i32..(M / BM) {
            value = mma(
                lp.load([pid.0, block, pid.1])
                    .reshape(shape![BM, 64])
                    .transpose(),
                gp.load([pid.0, block, pid.2]).reshape(shape![BM, 64]),
                value,
            );
        }
        out.store(value.reshape(shape![1, 64, 64]));
    }
    #[cutile::entry()]
    fn matmul_right_backward_bf16<const BM: i32, const M: i32>(
        out: &mut Tensor<f32, { [1, 64, 64] }>,
        lhs: &Tensor<f32, { [-1, M, -1] }>,
        gradient: &Tensor<f32, { [-1, M, -1] }>,
    ) {
        let lp = lhs.partition(shape![1, BM, 64]);
        let gp = gradient.partition(shape![1, BM, 64]);
        let pid = get_tile_block_id();
        let mut value = load_tile_mut(out).reshape(shape![64, 64]);
        for block in 0i32..(M / BM) {
            let left: Tile<bf16, { [64, BM] }> = convert_tile(
                lp.load([pid.0, block, pid.1])
                    .reshape(shape![BM, 64])
                    .transpose(),
            );
            let right: Tile<bf16, { [BM, 64] }> =
                convert_tile(gp.load([pid.0, block, pid.2]).reshape(shape![BM, 64]));
            value = mma(left, right, value);
        }
        out.store(value.reshape(shape![1, 64, 64]));
    }
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
    fn categorical_correct(
        out: &mut Tensor<f32, { [1] }>,
        logits: &Tensor<f32, { [-1] }>,
        targets: &Tensor<f32, { [-1] }>,
        width: i32,
    ) {
        let row = get_tile_block_id().0;
        let start = row * width;
        let zp = logits.partition(shape![1]);
        let yp = targets.partition(shape![1]);
        let mut predicted_value = zp.load([start]);
        let mut target_value = yp.load([start]);
        let mut predicted = constant(0i32, shape![1]);
        let mut target = constant(0i32, shape![1]);
        for j in 1i32..width {
            let z = zp.load([start + j]);
            let y = yp.load([start + j]);
            let index = broadcast_scalar(j, shape![1]);
            let z_better = gt_tile(z, predicted_value);
            let y_better = gt_tile(y, target_value);
            predicted_value = select(z_better, z, predicted_value);
            target_value = select(y_better, y, target_value);
            predicted = select(z_better, index, predicted);
            target = select(y_better, index, target);
        }
        let one = constant(1.0f32, shape![1]);
        let zero = constant(0.0f32, shape![1]);
        out.store(select(eq_tile(predicted, target), one, zero));
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
    fn softmax<const TILE_WIDTH: i32>(
        out: &mut Tensor<f32, { [1, TILE_WIDTH] }>,
        a: &Tensor<f32, { [-1, -1] }>,
        width: i32,
    ) {
        let columns: Tile<i32, { [TILE_WIDTH] }> = iota(shape![TILE_WIDTH]);
        let columns = columns.reshape(shape![1, TILE_WIDTH]);
        let valid = lt_tile(columns, broadcast_scalar(width, shape![1, TILE_WIDTH]));
        let loaded: Tile<f32, { [1, TILE_WIDTH] }> = a.load_like(out);
        let negative_infinity: Tile<f32, { [1, TILE_WIDTH] }> =
            constant(f32::NEG_INFINITY, shape![1, TILE_WIDTH]);
        let values = select(valid, loaded, negative_infinity);
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 1i32);
        let shifted = values - maximum.reshape(shape![1, 1]).broadcast(out.shape());
        let zero: Tile<f32, { [1, TILE_WIDTH] }> = constant(0.0f32, shape![1, TILE_WIDTH]);
        let numerator = select(valid, exp(shifted), zero);
        let sum: Tile<f32, { [1] }> = reduce_sum(numerator, 1i32);
        out.store(numerator / sum.reshape(shape![1, 1]).broadcast(out.shape()));
    }
    #[cutile::entry()]
    fn softmax_backward<const TILE_WIDTH: i32>(
        out: &mut Tensor<f32, { [1, TILE_WIDTH] }>,
        gradient: &Tensor<f32, { [-1, -1] }>,
        probability: &Tensor<f32, { [-1, -1] }>,
        width: i32,
    ) {
        let columns: Tile<i32, { [TILE_WIDTH] }> = iota(shape![TILE_WIDTH]);
        let columns = columns.reshape(shape![1, TILE_WIDTH]);
        let valid = lt_tile(columns, broadcast_scalar(width, shape![1, TILE_WIDTH]));
        let zero: Tile<f32, { [1, TILE_WIDTH] }> = constant(0.0f32, shape![1, TILE_WIDTH]);
        let g: Tile<f32, { [1, TILE_WIDTH] }> = select(valid, gradient.load_like(out), zero);
        let p: Tile<f32, { [1, TILE_WIDTH] }> = select(valid, probability.load_like(out), zero);
        let dot: Tile<f32, { [1] }> = reduce_sum(g * p, 1i32);
        out.store(p * (g - dot.reshape(shape![1, 1]).broadcast(out.shape())));
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
    fn mask_gradient(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        winners: &Tensor<f32, { [-1] }>,
    ) {
        let zero = constant(0.0f32, shape![128]);
        out.store(select(
            gt_tile(winners.load_like(out), zero),
            gradient.load_like(out),
            zero,
        ));
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

    #[cutile::entry()]
    fn grouped_minimum(
        out: &mut Tensor<f32, { [1] }>,
        winners: &mut Tensor<i32, { [1] }>,
        a: &Tensor<f32, { [-1] }>,
        offsets: &Tensor<i32, { [-1] }>,
        left: &Tensor<i32, { [-1] }>,
        minimum_finite: f32,
        maximum_finite: f32,
        positive_infinity: f32,
        no_finite_value: f32,
    ) {
        let pid = get_tile_block_id().0;
        let op = offsets.partition(shape![1]);
        let start: i32 = tile_to_scalar(op.load([pid]).reshape(shape![]));
        let end: i32 = tile_to_scalar(op.load([pid + 1i32]).reshape(shape![]));
        let lp = left.partition(shape![1]);
        let ap = a.partition(shape![1]);
        let mut winner = constant(-1i32, shape![1]);
        let mut minimum = broadcast_scalar(positive_infinity, shape![1]);
        let finite_low = broadcast_scalar(minimum_finite, shape![1]);
        let finite_high = broadcast_scalar(maximum_finite, shape![1]);
        for i in start..end {
            let candidate: i32 = tile_to_scalar(lp.load([i]).reshape(shape![]));
            let value = ap.load([candidate]);
            let finite = ge_tile(value, finite_low) & le_tile(value, finite_high);
            let better = finite & lt_tile(value, minimum);
            minimum = select(better, value, minimum);
            let candidate_tile: Tile<i32, { [1] }> = scalar_to_tile(candidate).reshape(shape![1]);
            winner = select(better, candidate_tile, winner);
        }
        let found = le_tile(minimum, finite_high);
        winners.store(winner);
        out.store(select(
            found,
            minimum,
            broadcast_scalar(no_finite_value, shape![1]),
        ));
    }

    #[cutile::entry()]
    fn minimum_winner_mask(
        out: &mut Tensor<f32, { [1] }>,
        winners: &Tensor<i32, { [-1] }>,
        groups: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let gp = groups.partition(shape![1]);
        let group: i32 = tile_to_scalar(gp.load([pid]).reshape(shape![]));
        let wp = winners.partition(shape![1]);
        let winner = wp.load([group]);
        let input: Tile<i32, { [1] }> = scalar_to_tile(pid).reshape(shape![1]);
        let one = constant(1.0f32, shape![1]);
        let zero = constant(0.0f32, shape![1]);
        out.store(select(eq_tile(winner, input), one, zero));
    }
}
