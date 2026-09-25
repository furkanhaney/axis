//! A correctness-first cuTile backend. Plans contain indices, never values.
use crate::Result;
use cutile::prelude::*;
use std::time::Instant;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

fn profile(label: &str, started: Instant) {
    if std::env::var_os("AXIS_PROFILE").is_some() {
        eprintln!(
            "axis_profile {label} {:.6}",
            started.elapsed().as_secs_f64()
        );
    }
}

/// Contributions per chunk in [`Device::grouped`]'s stage-one launch. A group's contribution
/// range never crosses into the next group's, so a group of `n` contributions always becomes
/// `n.div_ceil(GROUPED_CHUNK)` chunks -- exactly one for a group at or under this size.
const GROUPED_CHUNK: i32 = 512;

/// Splits each CSR group's `[offsets[g], offsets[g + 1])` contribution range into chunks of
/// at most `GROUPED_CHUNK` so [`Device::grouped`] can launch one block per chunk instead of
/// one block per group. Returns `(chunk_offsets, chunk_group)`: `chunk_offsets` are absolute
/// boundaries into the same `left`/`right` arrays `offsets` already indexes, one level finer
/// (`chunk_offsets[c]..chunk_offsets[c + 1]` is chunk `c`'s own range); `chunk_group` is the
/// matching per-group index into that chunk list (`chunk_group[g]..chunk_group[g + 1]` are
/// group `g`'s chunk indices, empty for an empty group). Both are `O(offsets.len() +
/// total contributions / GROUPED_CHUNK)` to build, not `O(total contributions)`.
fn chunked_offsets(offsets: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut chunk_offsets = vec![offsets[0]];
    let mut chunk_group = vec![0i32];
    for window in offsets.windows(2) {
        let (start, end) = (window[0], window[1]);
        let mut cursor = start;
        while cursor < end {
            cursor = (cursor + GROUPED_CHUNK).min(end);
            chunk_offsets.push(cursor);
        }
        chunk_group.push(chunk_offsets.len() as i32 - 1);
    }
    (chunk_offsets, chunk_group)
}

/// Enables cuTile's persistent on-disk cubin cache the first time any CUDA
/// `Device` is created in this process. cuTile JIT-compiles every kernel
/// through its `tileiras` subprocess on first use with a new shape (roughly
/// 290 ms each, #152); a cache hit skips that recompile on the next process.
/// cuTile's disk cache is off by default and has no environment
/// switch of its own (`cutile::jit_cache`), so Axis turns it on here, at
/// cuTile's own default location (`~/.cache/cutile/kernels`, or
/// `$XDG_CACHE_HOME/cutile/kernels`). `AXIS_JIT_CACHE=off` opts out entirely;
/// `AXIS_JIT_CACHE_DIR` redirects the store to an explicit directory instead.
/// A cache that cannot be enabled -- an unwritable directory, or no
/// resolvable per-user cache path -- is logged once to stderr and left
/// disabled; every cuTile store I/O failure afterward is already soft
/// (`cutile::jit_cache` never turns a working launch into a failing one), and
/// enabling the store itself must not fail device creation either.
///
/// Returns whether this call was the one that ran the check -- `true` at most
/// once per process, `false` for every call after -- so callers (and tests)
/// can observe the once-per-process guarantee directly; `Device::cuda`
/// itself ignores it.
pub(crate) fn ensure_jit_cache_enabled() -> bool {
    static ENABLED: OnceLock<()> = OnceLock::new();
    let mut ran = false;
    ENABLED.get_or_init(|| {
        ran = true;
        if std::env::var("AXIS_JIT_CACHE").ok().as_deref() == Some("off") {
            return;
        }
        let result = match std::env::var_os("AXIS_JIT_CACHE_DIR") {
            Some(dir) => cutile::jit_cache::FileSystemJitStore::new(dir)
                .map(|store| cutile::jit_cache::enable(Arc::new(store))),
            None => cutile::jit_cache::enable_default(),
        };
        if let Err(error) = result {
            eprintln!("axis: cuTile JIT disk cache disabled: {error}");
        }
    });
    ran
}

/// Cumulative cuTile JIT disk-cache hit/miss counts for this process, zero for
/// both fields before any CUDA device has compiled a kernel, or when the cache
/// is disabled (`AXIS_JIT_CACHE=off`) or could not be enabled. See
/// `Device::cuda`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JitCacheStats {
    /// Kernel compiles served from the on-disk cache instead of `tileiras`.
    pub hits: u64,
    /// Kernel compiles that missed the disk cache and ran `tileiras`.
    pub misses: u64,
}

/// Snapshot of the cuTile JIT disk cache's cumulative hit/miss counts for
/// this process, for a receipt or `AXIS_PROFILE` consumer to record.
pub fn jit_cache_stats() -> JitCacheStats {
    let stats = cutile::jit_cache::stats();
    JitCacheStats {
        hits: stats.hits,
        misses: stats.misses,
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
    /// Compile declared forward kernels before the first step, without allocating
    /// tensor storage or executing those kernels. Uses this device's FP32/BF16
    /// matrix mode and the same specialization keys as ordinary execution.
    ///
    /// Call during startup: preparation is synchronous and front-loads compilation;
    /// it does not promise lower total startup time. Only the listed matrix and
    /// softmax kernels are covered; layout copies, other operators and backward
    /// remain lazy. Every declaration is validated before any is compiled.
    pub fn prepare_kernels(&self, specs: &[crate::KernelSpec]) -> Result<()> {
        use crate::KernelSpec;
        for spec in specs {
            spec.validate()?;
        }
        for &spec in specs {
            // Ordinary execution zero-fills each flat output before reshaping.
            // Prepare that upstream kernel too, using its real launch geometry.
            let output_len = match spec {
                KernelSpec::Matmul {
                    batch,
                    rows,
                    columns,
                    ..
                } => batch * rows * columns,
                KernelSpec::Softmax { rows, width } => rows * width,
            };
            let mut flat = api::meta::<f32>(&[output_len]).sync_on(&self.0.stream)?;
            cutile::kernels::creation::full(0.0_f32, (&mut flat).partition([128]))
                .compile_on(&self.0.stream)?;
            match spec {
                KernelSpec::Matmul {
                    batch,
                    rows: m,
                    inner: k,
                    columns: n,
                } => {
                    let left = api::meta::<f32>(&[batch, m, k]).sync_on(&self.0.stream)?;
                    let right = api::meta::<f32>(&[batch, k, n]).sync_on(&self.0.stream)?;
                    let mut out = api::meta::<f32>(&[batch, m, n]).sync_on(&self.0.stream)?;
                    let generics = vec![contraction_tile(k).to_string(), k.to_string()];
                    if self.0.bf16_matmul {
                        kernels::matmul_bf16((&mut out).partition([1, 64, 64]), &left, &right)
                            .generics(generics)
                            .compile_on(&self.0.stream)?;
                    } else {
                        kernels::matmul((&mut out).partition([1, 64, 64]), &left, &right)
                            .generics(generics)
                            .compile_on(&self.0.stream)?;
                    }
                }
                KernelSpec::Softmax { rows, width } => {
                    let input = api::meta::<f32>(&[rows, width]).sync_on(&self.0.stream)?;
                    let mut out = api::meta::<f32>(&[rows, width]).sync_on(&self.0.stream)?;
                    let tile = width.next_power_of_two();
                    kernels::softmax((&mut out).partition([1, tile]), &input, width as i32)
                        .generics(vec![tile.to_string()])
                        .compile_on(&self.0.stream)?;
                }
            }
        }
        Ok(())
    }
    /// The first CUDA `Device` created in a process also enables cuTile's
    /// persistent on-disk kernel cache; `AXIS_JIT_CACHE=off` opts out and
    /// `AXIS_JIT_CACHE_DIR` redirects it. See `jit_cache_stats`.
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Self::cuda_with_bf16(ordinal, false)
    }
    /// CUDA device with BF16 inputs and FP32 accumulation inside matrix products.
    /// Parameters, activations outside GEMM, optimizer state, and reductions remain FP32.
    /// Also enables the JIT disk cache on first use, exactly as [`Device::cuda`] does.
    pub fn cuda_bf16(ordinal: usize) -> Result<Self> {
        Self::cuda_with_bf16(ordinal, true)
    }
    fn cuda_with_bf16(ordinal: usize, bf16_matmul: bool) -> Result<Self> {
        ensure_jit_cache_enabled();
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
    /// Deterministic uniform draw in `[0, 1)` generated entirely on the device: each output
    /// element `i` is `to_unit_float(splitmix64(seed ^ i.wrapping_mul(GOLDEN)))`, the same
    /// SplitMix64 mixer and golden-ratio constant `TrainingPass::next_seed` already uses (see
    /// `runtime::train`), applied to `seed` and the element's own flat index rather than to a
    /// pass seed and a draw counter. Every element depends only on `(seed, i)`, never on tile
    /// width or launch shape, so the result is bit-identical across any partition. Used by
    /// [`Tensor::uniform_device`] for training-time random draws (currently `Dropout`) that must
    /// not round-trip a host-generated mask through PCIe every step; `Tensor::uniform`'s
    /// host-side xorshift stream is untouched and keeps feeding recorded initialization.
    pub(crate) fn uniform_device(&self, seed: u64, len: usize) -> Result<Buffer> {
        let mut out = self.zeros(len)?;
        kernels::uniform_device((&mut out).partition([128]), seed).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
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
    /// `-gradient * numerator / denominator^2`: the divisor's gradient in `a / b`.
    pub(crate) fn divide_backward_denominator(
        &self,
        gradient: &Buffer,
        numerator: &Buffer,
        denominator: &Buffer,
    ) -> Result<Buffer> {
        let mut out = self.zeros(denominator.shape()[0] as usize)?;
        kernels::divide_backward_denominator(
            (&mut out).partition([128]),
            gradient.as_ref(),
            numerator.as_ref(),
            denominator.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    /// Elementwise absolute value.
    pub(crate) fn abs(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::abs((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    /// `gradient * sign(input)`, zero at exactly `input == 0`.
    pub(crate) fn abs_backward(&self, gradient: &Buffer, input: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(input.shape()[0] as usize)?;
        kernels::abs_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            input.as_ref(),
        )
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
    pub(crate) fn sigmoid(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::sigmoid((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn sigmoid_backward(
        &self,
        gradient: &Buffer,
        probability: &Buffer,
    ) -> Result<Buffer> {
        let mut out = self.zeros(probability.shape()[0] as usize)?;
        kernels::sigmoid_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            probability.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn silu(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::silu((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn silu_backward(&self, gradient: &Buffer, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::silu_backward((&mut out).partition([128]), gradient.as_ref(), a.as_ref())
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn leaky_relu(&self, a: &Buffer, negative_slope: f32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::leaky_relu((&mut out).partition([128]), a.as_ref(), negative_slope)
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn leaky_relu_backward(
        &self,
        gradient: &Buffer,
        a: &Buffer,
        negative_slope: f32,
    ) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::leaky_relu_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            a.as_ref(),
            negative_slope,
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn tanh(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::tanh_forward((&mut out).partition([128]), a.as_ref())
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn tanh_backward(&self, gradient: &Buffer, output: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(output.shape()[0] as usize)?;
        kernels::tanh_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            output.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn sin(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::sin_forward((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn sin_backward(&self, gradient: &Buffer, input: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(input.shape()[0] as usize)?;
        kernels::sin_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            input.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn gelu(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::gelu((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn gelu_exact(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::gelu_exact((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn gelu_exact_backward(&self, gradient: &Buffer, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::gelu_exact_backward((&mut out).partition([128]), gradient.as_ref(), a.as_ref())
            .enqueue_on(&self.0.stream)?;
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
    pub(crate) fn sign(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::sign((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn exp(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::exp_forward((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn exp_backward(&self, gradient: &Buffer, output: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(output.shape()[0] as usize)?;
        kernels::exp_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            output.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn ln(&self, a: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::ln((&mut out).partition([128]), a.as_ref()).enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn ln_backward(&self, gradient: &Buffer, input: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(input.shape()[0] as usize)?;
        kernels::ln_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            input.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn softplus(&self, a: &Buffer, beta: f32, threshold: f32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::softplus((&mut out).partition([128]), a.as_ref(), beta, threshold)
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn softplus_backward(
        &self,
        gradient: &Buffer,
        input: &Buffer,
        beta: f32,
        threshold: f32,
    ) -> Result<Buffer> {
        let mut out = self.zeros(input.shape()[0] as usize)?;
        kernels::softplus_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            input.as_ref(),
            beta,
            threshold,
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    /// `op` selects the comparison the same way [`Self::binary`]'s `op` selects
    /// add/sub/mul/div: 0 (`>`), 1 (`>=`), 2 (`<`), 3 (`<=`), 4 (`==`).
    pub(crate) fn compare_scalar(&self, a: &Buffer, scalar: f32, op: i32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::compare_scalar((&mut out).partition([128]), a.as_ref(), scalar)
            .generics(vec![op.to_string()])
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    /// `min`/`max` are already-resolved bounds: [`Tensor::clamp`] maps an omitted side to
    /// `f32::NEG_INFINITY`/`f32::INFINITY` before calling here, so the kernel only ever sees
    /// two concrete scalars, the same resolved-parameter shape as [`Self::leaky_relu`].
    pub(crate) fn clamp(&self, a: &Buffer, min: f32, max: f32) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::clamp((&mut out).partition([128]), a.as_ref(), min, max)
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn clamp_backward(
        &self,
        gradient: &Buffer,
        a: &Buffer,
        min: f32,
        max: f32,
    ) -> Result<Buffer> {
        let mut out = self.zeros(a.shape()[0] as usize)?;
        kernels::clamp_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            a.as_ref(),
            min,
            max,
        )
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
    pub(crate) fn binary_cross_entropy_weighted(
        &self,
        logits: &Buffer,
        targets: &Buffer,
        pos_weight: &Buffer,
    ) -> Result<Buffer> {
        let mut out = self.zeros(logits.shape()[0] as usize)?;
        kernels::binary_cross_entropy_weighted(
            (&mut out).partition([128]),
            logits.as_ref(),
            targets.as_ref(),
            pos_weight.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn binary_cross_entropy_weighted_backward(
        &self,
        gradient: &Buffer,
        logits: &Buffer,
        targets: &Buffer,
        pos_weight: &Buffer,
    ) -> Result<Buffer> {
        let mut out = self.zeros(logits.shape()[0] as usize)?;
        kernels::binary_cross_entropy_weighted_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            logits.as_ref(),
            targets.as_ref(),
            pos_weight.as_ref(),
        )
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn bce_loss(&self, x: &Buffer, y: &Buffer) -> Result<Buffer> {
        let mut out = self.zeros(x.shape()[0] as usize)?;
        kernels::bce_loss((&mut out).partition([128]), x.as_ref(), y.as_ref())
            .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }
    pub(crate) fn bce_loss_backward(
        &self,
        gradient: &Buffer,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<Buffer> {
        let mut out = self.zeros(x.shape()[0] as usize)?;
        kernels::bce_loss_backward(
            (&mut out).partition([128]),
            gradient.as_ref(),
            x.as_ref(),
            y.as_ref(),
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
    /// Each output sums its group's contributions in two stages so a group with many
    /// contributions still spreads across many CUDA blocks instead of one: stage one
    /// (`grouped_partial`) launches one block per fixed-size chunk of the plan's flat
    /// `left`/`right` arrays (`chunked_offsets`, split at group boundaries so a chunk never
    /// mixes two groups) and writes each chunk's own sequential sum; stage two
    /// (`grouped_combine`) launches one block per group and sums that group's chunk
    /// partials, in chunk order, into the final scaled output. Chunk boundaries are fixed
    /// by the plan alone, and both stages accumulate in a fixed left-to-right order, so the
    /// result is deterministic and reproducible -- it is a reassociation of the same terms
    /// the single-block version summed, not a race. A group with at most `GROUPED_CHUNK`
    /// contributions gets exactly the one chunk it always got, so small plans (most tests,
    /// most groups outside a broadcast gradient) pay for an unchanged single-block sum plus
    /// one pass-through combine.
    pub(crate) fn grouped(
        &self,
        a: &Buffer,
        b: Option<&Buffer>,
        plan: &Plan,
        scale: f32,
    ) -> Result<Buffer> {
        let started = Instant::now();
        let device_plan = self.device_plan(plan, b.is_some())?;
        let group_count = plan.offsets.len() - 1;
        let (chunk_offsets, chunk_group) = chunked_offsets(&plan.offsets);
        let chunk_count = chunk_offsets.len() - 1;
        let chunk_offsets = self.upload_i32(&chunk_offsets)?;
        let chunk_group = self.upload_i32(&chunk_group)?;
        let mut partial = self.zeros(chunk_count)?;
        kernels::grouped_partial(
            (&mut partial).partition([1]),
            a.as_ref(),
            b.unwrap_or(a).as_ref(),
            chunk_offsets.as_ref(),
            device_plan.left.as_ref(),
            device_plan.right.as_ref(),
        )
        .generics(vec![i32::from(b.is_some()).to_string()])
        .enqueue_on(&self.0.stream)?;
        let mut out = self.zeros(group_count)?;
        kernels::grouped_combine(
            (&mut out).partition([1]),
            &partial,
            chunk_group.as_ref(),
            scale,
        )
        .enqueue_on(&self.0.stream)?;
        self.track(partial);
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
                        spec.fill,
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
                        spec.fill,
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

    pub(crate) fn copy_window(&self, input: &Buffer, spec: &WindowSpec) -> Result<Buffer> {
        let metadata = self.upload_i32(&spec.metadata)?;
        let mut out = self.zeros(spec.output_len)?;
        unsafe {
            kernels::copy_window(
                (&mut out).partition([128]),
                input.as_ref().device_pointer(),
                metadata.as_ref(),
                i32::try_from(spec.output_len)?,
                spec.rank,
            )
        }
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }

    pub(crate) fn select_axis(&self, input: &Buffer, spec: &SelectSpec) -> Result<Buffer> {
        let metadata = self.upload_i32(&spec.metadata)?;
        let mut out = self.zeros(spec.output_len)?;
        unsafe {
            kernels::select_axis(
                (&mut out).partition([128]),
                input.as_ref().device_pointer(),
                metadata.as_ref(),
                i32::try_from(spec.output_len)?,
                spec.rank,
                spec.coordinate,
            )
        }
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }

    pub(crate) fn select_axis_backward(
        &self,
        gradient: &Buffer,
        spec: &SelectSpec,
    ) -> Result<Buffer> {
        let metadata = self.upload_i32(&spec.metadata)?;
        let mut out = self.zeros(spec.input_len)?;
        unsafe {
            kernels::select_axis_backward(
                (&mut out).partition([128]),
                gradient.as_ref().device_pointer(),
                metadata.as_ref(),
                i32::try_from(spec.input_len)?,
                spec.rank,
                spec.coordinate,
            )
        }
        .enqueue_on(&self.0.stream)?;
        Ok(self.track(out))
    }

    /// Concatenate equally sized contiguous buffers. The caller describes the
    /// resulting named layout; this backend operation only owns physical order.
    pub(crate) fn stack_contiguous(&self, inputs: &[Buffer]) -> Result<Buffer> {
        let first = inputs.first().ok_or("stack requires at least one tensor")?;
        let slice_len = first.shape()[0] as usize;
        let output_len = slice_len
            .checked_mul(inputs.len())
            .ok_or("stack size overflow")?;
        let out = self.zeros(output_len)?;
        let destination = out.device_pointer().cu_deviceptr();
        for (index, input) in inputs.iter().enumerate() {
            if input.shape()[0] as usize != slice_len {
                return Err("stack buffers must have equal lengths".into());
            }
            let offset = index
                .checked_mul(slice_len)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .ok_or("stack byte offset overflow")?;
            unsafe {
                cutile::cuda_core::memcpy_dtod_async::<f32>(
                    destination + u64::try_from(offset)?,
                    input.device_pointer().cu_deviceptr(),
                    slice_len,
                    &self.0.stream,
                )?;
            }
        }
        Ok(self.track(out))
    }

    pub(crate) fn contiguous_slice(
        &self,
        input: &Buffer,
        offset: usize,
        len: usize,
    ) -> Result<Buffer> {
        let input_len = input.shape()[0] as usize;
        if offset.checked_add(len).is_none_or(|end| end > input_len) {
            return Err("contiguous slice is outside its input".into());
        }
        let out = self.zeros(len)?;
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("slice byte offset overflow")?;
        unsafe {
            cutile::cuda_core::memcpy_dtod_async::<f32>(
                out.device_pointer().cu_deviceptr(),
                input.device_pointer().cu_deviceptr() + u64::try_from(byte_offset)?,
                len,
                &self.0.stream,
            )?;
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
        kernels::reduction_winner_mask(
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

    /// One maximum per CSR group plus a one-hot winner mask in input storage order.
    /// Exact mirror of [`Self::grouped_minimum`].
    pub(crate) fn grouped_maximum(
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
        kernels::grouped_maximum(
            (&mut out).partition([1]),
            (&mut winner_indices).partition([1]),
            a.as_ref(),
            forward_plan.offsets.as_ref(),
            forward_plan.left.as_ref(),
            f32::MIN,
            f32::MAX,
            f32::NEG_INFINITY,
            f32::NAN,
        )
        .enqueue_on(&self.0.stream)?;
        let mut winner_mask = self.zeros(a.shape()[0] as usize)?;
        kernels::reduction_winner_mask(
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

/// Compact convolution/pooling patch geometry. Per-element source indices are computed
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
    /// Value written for a patch position outside the input (padding). Convolution and the
    /// public `unfold2d` use `0.0`, which is correct for a linear contraction; max pooling uses
    /// `f32::NEG_INFINITY` so a padded position can never win the windowed maximum.
    pub fill: f32,
    pub kernel: [i32; 3],
    pub stride: [i32; 3],
    pub padding: [i32; 3],
    pub input_spatial: [i32; 3],
    pub output_spatial: [i32; 3],
    pub input_special_strides: [i32; 4],
    pub output_special_strides: [i32; 5],
}

/// Rank-sized destination-to-source translation used by padding, cropping and
/// their inverse derivatives. Each dimension stores five signed coordinates.
pub(crate) struct WindowSpec {
    pub output_len: usize,
    pub rank: i32,
    pub metadata: Vec<i32>,
}

/// Compact metadata for a layout permutation, optionally selecting one coordinate.
/// Each logical input dimension contributes `(extent, input_stride,
/// output_stride)`, with `-1` marking the selected dimension.
/// If no dimension is selected, the same copier and inverse preserve all values.
#[derive(Clone)]
pub(crate) struct SelectSpec {
    pub input_len: usize,
    pub output_len: usize,
    pub rank: i32,
    pub coordinate: i32,
    pub metadata: Vec<i32>,
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
        fill: f32,
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
                Some(fill),
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
        fill: f32,
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
                Some(fill),
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
    unsafe fn copy_window(
        out: &mut Tensor<f32, { [128] }>,
        input: *const f32,
        metadata: &Tensor<i32, { [-1] }>,
        output_len: i32,
        rank: i32,
    ) {
        let output_index: Tile<i32, { [128] }> =
            iota(shape![128]) + broadcast_scalar(get_tile_block_id().0 * 128i32, shape![128]);
        let mut live = lt_tile(output_index, broadcast_scalar(output_len, shape![128]));
        let mp = metadata.partition(shape![1]);
        let mut input_index = constant(0i32, shape![128]);
        let zero = constant(0i32, shape![128]);
        for dimension in 0i32..rank {
            let base = dimension * 5i32;
            let output_extent: i32 = tile_to_scalar(mp.load([base]).reshape(shape![]));
            let output_stride: i32 = tile_to_scalar(mp.load([base + 1i32]).reshape(shape![]));
            let input_extent: i32 = tile_to_scalar(mp.load([base + 2i32]).reshape(shape![]));
            let input_stride: i32 = tile_to_scalar(mp.load([base + 3i32]).reshape(shape![]));
            let shift: i32 = tile_to_scalar(mp.load([base + 4i32]).reshape(shape![]));
            let coordinate = (output_index / broadcast_scalar(output_stride, shape![128]))
                % broadcast_scalar(output_extent, shape![128])
                + broadcast_scalar(shift, shape![128]);
            let valid = ge_tile(coordinate, zero)
                & lt_tile(coordinate, broadcast_scalar(input_extent, shape![128]));
            live = live & valid;
            // Invalid coordinates must not overflow intermediate address arithmetic.
            input_index = input_index
                + select(valid, coordinate, zero) * broadcast_scalar(input_stride, shape![128]);
        }
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [128] }> = base.broadcast(shape![128]);
        let addresses = addptr_tile(base, select(live, input_index, zero));
        let (values, _token): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                addresses,
                ordering::Relaxed,
                Some(scope::Device),
                Some(live),
                Some(0.0f32),
                None,
                Latency::<0>,
            )
        };
        out.store(values);
    }

    #[cutile::entry()]
    unsafe fn select_axis(
        out: &mut Tensor<f32, { [128] }>,
        input: *const f32,
        metadata: &Tensor<i32, { [-1] }>,
        output_len: i32,
        rank: i32,
        selected: i32,
    ) {
        let output_index: Tile<i32, { [128] }> =
            iota(shape![128]) + broadcast_scalar(get_tile_block_id().0 * 128i32, shape![128]);
        let live = lt_tile(output_index, broadcast_scalar(output_len, shape![128]));
        let mp = metadata.partition(shape![1]);
        let mut input_index = constant(0i32, shape![128]);
        for dimension in 0i32..rank {
            let base = dimension * 3i32;
            let extent: i32 = tile_to_scalar(mp.load([base]).reshape(shape![]));
            let input_stride: i32 = tile_to_scalar(mp.load([base + 1i32]).reshape(shape![]));
            let output_stride: i32 = tile_to_scalar(mp.load([base + 2i32]).reshape(shape![]));
            if output_stride < 0i32 {
                input_index = input_index + broadcast_scalar(selected * input_stride, shape![128]);
            } else {
                let coordinate = (output_index / broadcast_scalar(output_stride, shape![128]))
                    % broadcast_scalar(extent, shape![128]);
                input_index =
                    input_index + coordinate * broadcast_scalar(input_stride, shape![128]);
            }
        }
        let zero = constant(0i32, shape![128]);
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [128] }> = base.broadcast(shape![128]);
        let addresses = addptr_tile(base, select(live, input_index, zero));
        let (values, _token): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                addresses,
                ordering::Relaxed,
                Some(scope::Device),
                Some(live),
                Some(0.0f32),
                None,
                Latency::<0>,
            )
        };
        out.store(values);
    }

    #[cutile::entry()]
    unsafe fn select_axis_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: *const f32,
        metadata: &Tensor<i32, { [-1] }>,
        input_len: i32,
        rank: i32,
        selected: i32,
    ) {
        let input_index: Tile<i32, { [128] }> =
            iota(shape![128]) + broadcast_scalar(get_tile_block_id().0 * 128i32, shape![128]);
        let mut live = lt_tile(input_index, broadcast_scalar(input_len, shape![128]));
        let mp = metadata.partition(shape![1]);
        let mut gradient_index = constant(0i32, shape![128]);
        for dimension in 0i32..rank {
            let base = dimension * 3i32;
            let extent: i32 = tile_to_scalar(mp.load([base]).reshape(shape![]));
            let input_stride: i32 = tile_to_scalar(mp.load([base + 1i32]).reshape(shape![]));
            let output_stride: i32 = tile_to_scalar(mp.load([base + 2i32]).reshape(shape![]));
            let coordinate = (input_index / broadcast_scalar(input_stride, shape![128]))
                % broadcast_scalar(extent, shape![128]);
            if output_stride < 0i32 {
                live = live & eq_tile(coordinate, broadcast_scalar(selected, shape![128]));
            } else {
                gradient_index =
                    gradient_index + coordinate * broadcast_scalar(output_stride, shape![128]);
            }
        }
        let zero = constant(0i32, shape![128]);
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(gradient);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [128] }> = base.broadcast(shape![128]);
        let addresses = addptr_tile(base, select(live, gradient_index, zero));
        let (values, _token): (Tile<f32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                addresses,
                ordering::Relaxed,
                Some(scope::Device),
                Some(live),
                Some(0.0f32),
                None,
                Latency::<0>,
            )
        };
        out.store(values);
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
    // Weighted binary cross-entropy: -[p*y*log(sigma(x)) + (1-y)*log(1-sigma(x))], which
    // expands (see `binary_cross_entropy_with_logits_weighted`'s derivation) to
    // `log_weight * softplus_stable(x) - p*y*x` with `log_weight = 1 + (p-1)*y`; reduces to
    // the unweighted kernel above at `p == 1`, where `log_weight == 1`.
    #[cutile::entry()]
    fn binary_cross_entropy_weighted(
        out: &mut Tensor<f32, { [128] }>,
        logits: &Tensor<f32, { [-1] }>,
        targets: &Tensor<f32, { [-1] }>,
        pos_weight: &Tensor<f32, { [-1] }>,
    ) {
        let z = logits.load_like(out);
        let y = targets.load_like(out);
        let p = pos_weight.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let magnitude = max_tile(z, zero - z);
        let softplus = max_tile(z, zero) + log(one + exp(zero - magnitude));
        let log_weight = one + (p - one) * y;
        out.store(log_weight * softplus - p * y * z);
    }
    // d/dx of the weighted loss above is `log_weight * sigma(x) - p*y`; reduces to the
    // unweighted backward kernel above at `p == 1`.
    #[cutile::entry()]
    fn binary_cross_entropy_weighted_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        logits: &Tensor<f32, { [-1] }>,
        targets: &Tensor<f32, { [-1] }>,
        pos_weight: &Tensor<f32, { [-1] }>,
    ) {
        let z = logits.load_like(out);
        let y = targets.load_like(out);
        let p = pos_weight.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let magnitude = max_tile(z, zero - z);
        let e = exp(zero - magnitude);
        let probability = select(gt_tile(z, zero), one / (one + e), e / (one + e));
        let log_weight = one + (p - one) * y;
        out.store(gradient.load_like(out) * (log_weight * probability - p * y));
    }
    // `BCELoss` for probability inputs, PyTorch's own `binary_cross_entropy` kernel shape:
    // `-[y*log(x) + (1-y)*log(1-x)]` with each logarithm floored at exactly `-100` (PyTorch's
    // own literal), rather than flooring the probability before the logarithm the way
    // `Tensor::clamp`+`Tensor::ln` composition would. This kernel and its backward below are
    // the dedicated, non-differentiated-through-`ln` pair `binary_cross_entropy` (the
    // probability-input `BCELoss`) now calls, replacing that composition.
    #[cutile::entry()]
    fn bce_loss(
        out: &mut Tensor<f32, { [128] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.load_like(out);
        let t = y.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let floor = constant(-100.0f32, shape![128]);
        let log_p = max_tile(log(p), floor);
        let log_1mp = max_tile(log(one - p), floor);
        out.store(zero - (t * log_p + (one - t) * log_1mp));
    }
    // PyTorch's own `binary_cross_entropy_backward`: `grad * (x - y) / max((1 - x) * x, eps)`
    // with `eps = 1e-12`, computed directly rather than falling out of the forward's own
    // composition -- so it stays finite at `x == 0` and `x == 1`, where a floored-input `ln`
    // composition's own backward would divide by exactly zero.
    #[cutile::entry()]
    fn bce_loss_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let p = x.load_like(out);
        let t = y.load_like(out);
        let one = constant(1.0f32, shape![128]);
        let eps = constant(1e-12f32, shape![128]);
        let denominator = max_tile((one - p) * p, eps);
        out.store(gradient.load_like(out) * (p - t) / denominator);
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
        } else if OP == 2 {
            out.store(x * y);
        } else {
            out.store(x / y);
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
    fn sigmoid(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let x = a.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let magnitude = max_tile(x, zero - x);
        let e = exp(zero - magnitude);
        out.store(select(gt_tile(x, zero), one / (one + e), e / (one + e)));
    }
    #[cutile::entry()]
    fn sigmoid_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        probability: &Tensor<f32, { [-1] }>,
    ) {
        let p = probability.load_like(out);
        let one = constant(1.0f32, shape![128]);
        out.store(gradient.load_like(out) * p * (one - p));
    }
    fn sigmoid_tile(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let e = exp(zero - max_tile(x, zero - x));
        select(gt_tile(x, zero), one / (one + e), e / (one + e))
    }
    #[cutile::entry()]
    fn silu(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let x = a.load_like(out);
        out.store(x * sigmoid_tile(x));
    }
    #[cutile::entry()]
    fn silu_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
    ) {
        let x = a.load_like(out);
        let p = sigmoid_tile(x);
        let one = constant(1.0f32, shape![128]);
        out.store(gradient.load_like(out) * (p + x * p * (one - p)));
    }
    #[cutile::entry()]
    fn leaky_relu(
        out: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        negative_slope: f32,
    ) {
        let x = a.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let slope = broadcast_scalar(negative_slope, shape![128]);
        out.store(select(gt_tile(x, zero), x, slope * x));
    }
    #[cutile::entry()]
    fn leaky_relu_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        negative_slope: f32,
    ) {
        let x = a.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let slope = broadcast_scalar(negative_slope, shape![128]);
        out.store(gradient.load_like(out) * select(gt_tile(x, zero), one, slope));
    }
    #[cutile::entry()]
    fn tanh_forward(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        out.store(tanh(a.load_like(out)));
    }
    #[cutile::entry()]
    fn tanh_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        output: &Tensor<f32, { [-1] }>,
    ) {
        let y = output.load_like(out);
        let one = constant(1.0f32, shape![128]);
        out.store(gradient.load_like(out) * (one - y * y));
    }
    #[cutile::entry()]
    fn sin_forward(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        out.store(sin(a.load_like(out)));
    }
    #[cutile::entry()]
    fn sin_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        out.store(gradient.load_like(out) * cos(input.load_like(out)));
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
    // Normal-CDF evaluation (Abramowitz & Stegun 26.2.17). Evaluate the small
    // tail directly for negative x rather than subtracting nearly equal values.
    // Mathematical approximation error is < 7.5e-8 for the CDF; FP32 rounding
    // is additional and covered by the independent quadrature CUDA oracle.
    fn normal_cdf(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let one = constant(1.0f32, shape![128]);
        let zero = constant(0.0f32, shape![128]);
        let half = constant(0.5f32, shape![128]);
        let t = one / (one + constant(0.2316419f32, shape![128]) * absf(x));
        let polynomial = (((constant(1.330274429f32, shape![128]) * t
            + constant(-1.821255978f32, shape![128]))
            * t
            + constant(1.781477937f32, shape![128]))
            * t
            + constant(-0.356563782f32, shape![128]))
            * t
            + constant(0.319381530f32, shape![128]);
        let density = constant(0.3989422804f32, shape![128]) * exp(zero - half * x * x);
        let tail = density * t * polynomial;
        let cdf = select(lt_tile(x, zero), tail, one - tail);
        select(eq_tile(x, zero), half, cdf)
    }
    #[cutile::entry()]
    fn gelu_exact(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let x = a.load_like(out);
        out.store(x * normal_cdf(x));
    }
    #[cutile::entry()]
    fn gelu_exact_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
    ) {
        let x = a.load_like(out);
        let half = constant(-0.5f32, shape![128]);
        let density = constant(0.3989422804f32, shape![128]) * exp(half * x * x);
        out.store(gradient.load_like(out) * (normal_cdf(x) + x * density));
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
    fn sign(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        let zero: Tile<f32, { [128] }> = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let negative_one = constant(-1.0f32, shape![128]);
        out.store(select(gt_tile(a.load_like(out), zero), one, negative_one));
    }
    // Named `exp_forward`/`exp_backward` rather than bare `exp` because the body
    // calls cutile's own `exp` primitive; a same-named entry point would shadow
    // it and recurse into itself instead, the same reason `tanh`/`sin` above are
    // `tanh_forward`/`sin_forward`.
    #[cutile::entry()]
    fn exp_forward(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        out.store(exp(a.load_like(out)));
    }
    #[cutile::entry()]
    fn exp_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        output: &Tensor<f32, { [-1] }>,
    ) {
        out.store(gradient.load_like(out) * output.load_like(out));
    }
    // `ln` calls cutile's `log` primitive (a different name), so no `exp`-style
    // shadowing renamed is needed here.
    #[cutile::entry()]
    fn ln(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        out.store(log(a.load_like(out)));
    }
    #[cutile::entry()]
    fn ln_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        out.store(gradient.load_like(out) / input.load_like(out));
    }
    // PyTorch-exact `Softplus(beta, threshold)`: `(1/beta) * ln(1 + exp(beta*x))`
    // on the logarithmic branch, `x` itself once `beta*x > threshold`. This
    // mirrors PyTorch's own kernel, which evaluates the logarithmic branch
    // directly (no `sign`-style magnitude subtraction) because the threshold
    // already keeps `exp(beta*x)` finite before the branch matters; unlike
    // `binary_cross_entropy`'s internal softplus, this one is user-facing with
    // an unbounded `x`, so the linear branch -- not a stability trick -- is what
    // keeps it finite past the threshold.
    #[cutile::entry()]
    fn softplus(
        out: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        beta: f32,
        threshold: f32,
    ) {
        let x = a.load_like(out);
        let beta_tile = broadcast_scalar(beta, shape![128]);
        let threshold_tile = broadcast_scalar(threshold, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let scaled = beta_tile * x;
        let value = log(one + exp(scaled)) / beta_tile;
        out.store(select(gt_tile(scaled, threshold_tile), x, value));
    }
    // Backward is `sigmoid(beta*x)` on the logarithmic branch (computed the same
    // stable way as `sigmoid`/`binary_cross_entropy_backward` above, since unlike
    // the forward there is no threshold protecting this expression's exponent)
    // and exactly `1` on the linear branch, since the linear branch's output is
    // `x` itself.
    #[cutile::entry()]
    fn softplus_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        beta: f32,
        threshold: f32,
    ) {
        let x = a.load_like(out);
        let beta_tile = broadcast_scalar(beta, shape![128]);
        let threshold_tile = broadcast_scalar(threshold, shape![128]);
        let zero = constant(0.0f32, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let scaled = beta_tile * x;
        let magnitude = max_tile(scaled, zero - scaled);
        let e = exp(zero - magnitude);
        let sigmoid = select(gt_tile(scaled, zero), one / (one + e), e / (one + e));
        out.store(gradient.load_like(out) * select(gt_tile(scaled, threshold_tile), one, sigmoid));
    }
    /// One elementwise comparison kernel for the whole `gt`/`ge`/`lt`/`le`/`eq`
    /// family, mirroring `sign`'s `select`-of-a-comparison shape. `OP` picks the
    /// operator (0=`>`, 1=`>=`, 2=`<`, 3=`<=`, 4=`==`); each `gt_tile`/`ge_tile`/
    /// `lt_tile`/`le_tile`/`eq_tile` is an ordered IEEE comparison, so any
    /// comparison against `NaN` is `false` on either input.
    #[cutile::entry()]
    fn compare_scalar<const OP: i32>(
        out: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        scalar: f32,
    ) {
        let x = a.load_like(out);
        let s = broadcast_scalar(scalar, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let zero = constant(0.0f32, shape![128]);
        let condition = if OP == 0 {
            gt_tile(x, s)
        } else if OP == 1 {
            ge_tile(x, s)
        } else if OP == 2 {
            lt_tile(x, s)
        } else if OP == 3 {
            le_tile(x, s)
        } else {
            eq_tile(x, s)
        };
        out.store(select(condition, one, zero));
    }
    /// Clamp low then high, mirroring `sign`'s comparison-then-`select` shape. A `NaN`
    /// input satisfies neither `lt_tile` nor `gt_tile` (every ordered IEEE comparison
    /// against `NaN` is false), so it falls through both selects unclamped and the stored
    /// output stays `NaN`.
    #[cutile::entry()]
    fn clamp(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>, min: f32, max: f32) {
        let x = a.load_like(out);
        let lo = broadcast_scalar(min, shape![128]);
        let hi = broadcast_scalar(max, shape![128]);
        let after_low = select(lt_tile(x, lo), lo, x);
        out.store(select(gt_tile(after_low, hi), hi, after_low));
    }
    /// Gradient matches PyTorch's `clamp`: passes through where `min <= x && x <= max`,
    /// including exactly at either bound, and is zero elsewhere -- including a `NaN` input,
    /// since `ge_tile`/`le_tile` against `NaN` are both false.
    #[cutile::entry()]
    fn clamp_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        a: &Tensor<f32, { [-1] }>,
        min: f32,
        max: f32,
    ) {
        let x = a.load_like(out);
        let lo = broadcast_scalar(min, shape![128]);
        let hi = broadcast_scalar(max, shape![128]);
        let one = constant(1.0f32, shape![128]);
        let zero = constant(0.0f32, shape![128]);
        let within_high = select(le_tile(x, hi), one, zero);
        let mask = select(ge_tile(x, lo), within_high, zero);
        out.store(gradient.load_like(out) * mask);
    }
    /// Stage one of [`crate::backend::Device::grouped`]'s two-stage reduction: one block per
    /// chunk of a group's contributions (never per group), summing that chunk alone in the
    /// same sequential order the single-stage kernel this replaces summed a whole group.
    /// `chunk_offsets` is `offsets` one level finer -- a chunk boundary is always also a
    /// group boundary, never crossing into another group's contributions.
    #[cutile::entry()]
    fn grouped_partial<const PRODUCT: i32>(
        out: &mut Tensor<f32, { [1] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        chunk_offsets: &Tensor<i32, { [-1] }>,
        left: &Tensor<i32, { [-1] }>,
        right: &Tensor<i32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let op = chunk_offsets.partition(shape![1]);
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
        out.store(sum);
    }

    /// Stage two of [`crate::backend::Device::grouped`]'s two-stage reduction: one block per
    /// group, summing that group's own chunk partials from stage one in a fixed,
    /// increasing-chunk-index order (deterministic and reproducible, though a reassociation
    /// of the terms a single stage-one chunk summed on its own), then applying `scale`
    /// exactly once at the end, matching the single-stage kernel this replaces.
    #[cutile::entry()]
    fn grouped_combine(
        out: &mut Tensor<f32, { [1] }>,
        partial: &Tensor<f32, { [-1] }>,
        chunk_group: &Tensor<i32, { [-1] }>,
        scale: f32,
    ) {
        let pid = get_tile_block_id().0;
        let gp = chunk_group.partition(shape![1]);
        let start: i32 = tile_to_scalar(gp.load([pid]).reshape(shape![]));
        let end: i32 = tile_to_scalar(gp.load([pid + 1i32]).reshape(shape![]));
        let pp = partial.partition(shape![1]);
        let mut sum = constant(0.0f32, shape![1]);
        for i in start..end {
            sum = sum + pp.load([i]);
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
    fn grouped_maximum(
        out: &mut Tensor<f32, { [1] }>,
        winners: &mut Tensor<i32, { [1] }>,
        a: &Tensor<f32, { [-1] }>,
        offsets: &Tensor<i32, { [-1] }>,
        left: &Tensor<i32, { [-1] }>,
        minimum_finite: f32,
        maximum_finite: f32,
        negative_infinity: f32,
        no_finite_value: f32,
    ) {
        let pid = get_tile_block_id().0;
        let op = offsets.partition(shape![1]);
        let start: i32 = tile_to_scalar(op.load([pid]).reshape(shape![]));
        let end: i32 = tile_to_scalar(op.load([pid + 1i32]).reshape(shape![]));
        let lp = left.partition(shape![1]);
        let ap = a.partition(shape![1]);
        let mut winner = constant(-1i32, shape![1]);
        let mut maximum = broadcast_scalar(negative_infinity, shape![1]);
        let finite_low = broadcast_scalar(minimum_finite, shape![1]);
        let finite_high = broadcast_scalar(maximum_finite, shape![1]);
        for i in start..end {
            let candidate: i32 = tile_to_scalar(lp.load([i]).reshape(shape![]));
            let value = ap.load([candidate]);
            let finite = ge_tile(value, finite_low) & le_tile(value, finite_high);
            let better = finite & gt_tile(value, maximum);
            maximum = select(better, value, maximum);
            let candidate_tile: Tile<i32, { [1] }> = scalar_to_tile(candidate).reshape(shape![1]);
            winner = select(better, candidate_tile, winner);
        }
        let found = ge_tile(maximum, finite_low);
        winners.store(winner);
        out.store(select(
            found,
            maximum,
            broadcast_scalar(no_finite_value, shape![1]),
        ));
    }

    /// One-hot mask, in input storage order, of the position each group's `grouped_minimum`/
    /// `grouped_maximum` winner index names. Shared by both reductions: it only compares winner
    /// indices, so it carries no min/max-specific logic.
    #[cutile::entry()]
    fn reduction_winner_mask(
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

    /// `-gradient * numerator / denominator^2`, the divisor's gradient in `a / b`.
    #[cutile::entry()]
    fn divide_backward_denominator(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        numerator: &Tensor<f32, { [-1] }>,
        denominator: &Tensor<f32, { [-1] }>,
    ) {
        let g = gradient.load_like(out);
        let a = numerator.load_like(out);
        let b = denominator.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        out.store((zero - g) * a / (b * b));
    }

    #[cutile::entry()]
    fn abs(out: &mut Tensor<f32, { [128] }>, a: &Tensor<f32, { [-1] }>) {
        out.store(absf(a.load_like(out)));
    }

    /// `gradient * sign(input)`; PyTorch's `abs` backward convention gives exactly zero at
    /// `input == 0`, so neither branch of the select fires there.
    #[cutile::entry()]
    fn abs_backward(
        out: &mut Tensor<f32, { [128] }>,
        gradient: &Tensor<f32, { [-1] }>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let x = input.load_like(out);
        let g = gradient.load_like(out);
        let zero = constant(0.0f32, shape![128]);
        out.store(select(
            gt_tile(x, zero),
            g,
            select(lt_tile(x, zero), zero - g, zero),
        ));
    }

    /// Counter-based uniform draw in `[0, 1)`, entirely on the device: `out[i] =
    /// to_unit_float(splitmix64(seed ^ i * GOLDEN))` for the output's own flat index `i`, with no
    /// input tensor and no host round trip. `GOLDEN` and the mixer's own two multipliers are the
    /// same three constants `runtime::train::GOLDEN` and its private `splitmix64` use, each
    /// built here from two `u32`-range halves (see the comment at their construction below --
    /// `constant()` silently truncates a `u64` literal at or above `2^63` in this cuTile
    /// version). `i` is `block_id * 128 + local_lane`, computed the same way `unfold2d` computes
    /// its own output index; elements past the tensor's true length are never stored (the padded
    /// tail of the last tile), exactly like every other kernel that partitions by `128`.
    /// `splitmix64` runs in `u64` tiles so `>>` is a logical (unsigned) shift, matching the host
    /// `u64` mixer bit for bit. `to_unit_float` keeps the top 24 bits of the 64-bit hash
    /// (`>> 40`) and divides by `2^24`, mirroring the host xorshift stream's own
    /// `(rng >> 40) as f32 / 2^24` -- an exact float since every value below `2^24` round-trips
    /// through `f32` without rounding.
    #[cutile::entry()]
    fn uniform_device(out: &mut Tensor<f32, { [128] }>, seed: u64) {
        let block_id = get_tile_block_id().0;
        let local: Tile<i32, { [128] }> = iota(shape![128]);
        let index_i32 = local + broadcast_scalar(block_id * 128i32, shape![128]);
        // cuTile's `convert` lowering does not implement integer-to-integer width changes at all
        // (only the identity case and a handful of same-width bitcasts), despite the crate's
        // documented "all conversions supported" matrix -- confirmed against both `i32 -> u64`
        // and `i32 -> u32`. Integer <-> float conversions have no such restriction, so `i32 ->
        // f64 -> u64` widens exactly: every index below `2^53` round-trips through `f64` without
        // rounding, and this kernel's indices never approach that.
        let index_f64: Tile<f64, { [128] }> = convert_tile(index_i32);
        let index: Tile<u64, { [128] }> = convert_tile(index_f64);

        // `constant()` silently truncates a `u64` literal at or above `2^63` to `0` in this
        // cuTile version (its literal-parsing path assumes a signed 64-bit range); confirmed by a
        // device probe, since the failure is silent, not a compile error. Every constant here
        // that needs its top bit set -- `GOLDEN` and both SplitMix64 multipliers -- is therefore
        // built from two `u32`-range halves (each comfortably under `2^63`) combined with a
        // shift and an or, instead of one `constant()` call with the full 64-bit literal.
        let shift32: Tile<u64, { [128] }> = constant(32u64, shape![128]);
        let golden_hi: Tile<u64, { [128] }> = constant(0x9E37_79B9u64, shape![128]);
        let golden_lo: Tile<u64, { [128] }> = constant(0x7F4A_7C15u64, shape![128]);
        let golden: Tile<u64, { [128] }> = (golden_hi << shift32) | golden_lo;
        let mul1_hi: Tile<u64, { [128] }> = constant(0xBF58_476Du64, shape![128]);
        let mul1_lo: Tile<u64, { [128] }> = constant(0x1CE4_E5B9u64, shape![128]);
        let mul1: Tile<u64, { [128] }> = (mul1_hi << shift32) | mul1_lo;
        let mul2_hi: Tile<u64, { [128] }> = constant(0x94D0_49BBu64, shape![128]);
        let mul2_lo: Tile<u64, { [128] }> = constant(0x1331_11EBu64, shape![128]);
        let mul2: Tile<u64, { [128] }> = (mul2_hi << shift32) | mul2_lo;

        let seed_tile: Tile<u64, { [128] }> = broadcast_scalar(seed, shape![128]);
        let shift30: Tile<u64, { [128] }> = constant(30u64, shape![128]);
        let shift27: Tile<u64, { [128] }> = constant(27u64, shape![128]);
        let shift31: Tile<u64, { [128] }> = constant(31u64, shape![128]);
        let shift40: Tile<u64, { [128] }> = constant(40u64, shape![128]);

        // Plain operators (`*`, `^`, `>>`, `<<`, `|`), not `muli`/`xori`/`shri`/`shli`/`ori`: the
        // explicit functions with an `overflow::None` mode fail to serialize in this cuTile
        // version ("missing attribute 'overflow' on op MulI"), while the operator overloads --
        // already exercised by every other kernel's `i32` index arithmetic in this file -- lower
        // without that attribute at all, giving ordinary two's-complement wraparound (`u64` here,
        // so `>>` is logical).
        let mut z = seed_tile ^ (index * golden);
        z = z ^ (z >> shift30);
        z = z * mul1;
        z = z ^ (z >> shift27);
        z = z * mul2;
        z = z ^ (z >> shift31);

        // `u64 -> f32` is an integer-to-float conversion, not an integer width change, so it
        // needs no detour through another integer type; the shifted value is already below
        // `2^24`, which is exactly representable in `f32`.
        let top24: Tile<u64, { [128] }> = z >> shift40;
        let raw: Tile<f32, { [128] }> = convert_tile(top24);
        // 1.0 / 2^24, exact in f32.
        let scale: Tile<f32, { [128] }> = constant(5.960464477539063e-8f32, shape![128]);
        out.store(raw * scale);
    }
}
