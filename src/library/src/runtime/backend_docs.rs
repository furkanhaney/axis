//! Type-compatible backend used only while docs.rs renders Axis's public API.
//!
//! docs.rs does not provide the CUDA toolkit needed to build cuTile's generated
//! bindings. This module keeps the public API visible to rustdoc without
//! becoming a selectable runtime backend: `build.rs` enables it only when the
//! docs.rs service sets `DOCS_RS`.

use crate::Result;
use std::{rc::Rc, sync::Arc};

pub(crate) type Buffer = Arc<Vec<f32>>;

#[derive(Clone)]
pub struct Device(Rc<()>);

fn unavailable<T>() -> Result<T> {
    Err("the docs.rs build has no CUDA runtime; use Axis with its default `cuda` feature".into())
}

impl Device {
    pub fn cuda(_ordinal: usize) -> Result<Self> {
        unavailable()
    }

    /// CUDA device with BF16 inputs and FP32 accumulation inside matrix products.
    /// Parameters, activations outside GEMM, optimizer state, and reductions remain FP32.
    pub fn cuda_bf16(_ordinal: usize) -> Result<Self> {
        unavailable()
    }

    pub(crate) fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    pub fn synchronize(&self) -> Result<()> {
        unavailable()
    }

    pub(crate) fn upload(&self, _values: Vec<f32>) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn read(&self, _buffer: &Buffer) -> Result<Vec<f32>> {
        unavailable()
    }

    pub(crate) fn zeros_buffer(&self, _len: usize) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn binary(&self, _a: &Buffer, _b: &Buffer, _op: i32) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn mask_gradient(&self, _gradient: &Buffer, _winners: &Buffer) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn scale(&self, _a: &Buffer, _scale: f32) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn relu(&self, _a: &Buffer) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn relu_backward(&self, _gradient: &Buffer, _a: &Buffer) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn gelu(&self, _a: &Buffer) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn gelu_backward(&self, _gradient: &Buffer, _a: &Buffer) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn inverse_sqrt(&self, _a: &Buffer, _epsilon: f32) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn inverse_sqrt_backward(
        &self,
        _gradient: &Buffer,
        _a: &Buffer,
        _epsilon: f32,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn binary_cross_entropy(
        &self,
        _logits: &Buffer,
        _targets: &Buffer,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn binary_cross_entropy_backward(
        &self,
        _gradient: &Buffer,
        _logits: &Buffer,
        _targets: &Buffer,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn categorical_cross_entropy(
        &self,
        _logits: &Buffer,
        _targets: &Buffer,
        _width: usize,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn categorical_cross_entropy_backward(
        &self,
        _gradient: &Buffer,
        _probability: &Buffer,
        _targets: &Buffer,
        _width: usize,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn categorical_correct(
        &self,
        _logits: &Buffer,
        _targets: &Buffer,
        _width: usize,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn mask(&self, _a: &Buffer, _keep: &Buffer) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn softmax(&self, _a: &Buffer, _width: usize) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn softmax_backward(
        &self,
        _gradient: &Buffer,
        _probability: &Buffer,
        _width: usize,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn matmul(
        &self,
        _a: &Buffer,
        _b: &Buffer,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn matmul_left_backward(
        &self,
        _gradient: &Buffer,
        _rhs: &Buffer,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn matmul_right_backward(
        &self,
        _lhs: &Buffer,
        _gradient: &Buffer,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Buffer> {
        unavailable()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adam(
        &self,
        _value: &Buffer,
        _gradient: &Buffer,
        _first: &Buffer,
        _second: &Buffer,
        _learning_rate: f32,
        _learning_rates: Option<&Buffer>,
        _beta1: f32,
        _beta2: f32,
        _correction1: f32,
        _correction2: f32,
        _epsilon: f32,
        _weight_decay: f32,
    ) -> Result<(Buffer, Buffer, Buffer)> {
        unavailable()
    }

    pub(crate) fn grouped(
        &self,
        _a: &Buffer,
        _b: Option<&Buffer>,
        _plan: &Plan,
        _scale: f32,
    ) -> Result<Buffer> {
        unavailable()
    }

    pub(crate) fn grouped_minimum(
        &self,
        _a: &Buffer,
        _forward: &Plan,
        _reverse: &Plan,
    ) -> Result<(Buffer, Buffer)> {
        unavailable()
    }
}

#[derive(Clone)]
pub(crate) struct Plan;

impl Plan {
    pub fn groups(groups: Vec<Vec<(usize, usize)>>, _product: bool) -> Result<Self> {
        Self::check_size(groups.iter().map(Vec::len).sum())?;
        Ok(Self)
    }

    pub fn gather(map: &[usize]) -> Result<Self> {
        Self::check_size(map.len())?;
        Ok(Self)
    }

    pub fn reverse(map: &[usize], len: usize) -> Result<Self> {
        if map.iter().any(|&input| input >= len) {
            return Err("reverse index is outside the input extent".into());
        }
        Self::check_size(map.len())?;
        Ok(Self)
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
