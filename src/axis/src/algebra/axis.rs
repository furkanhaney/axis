use crate::Result;
use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_AXIS: AtomicU64 = AtomicU64::new(1);

/// Identity is independent of both the diagnostic name and tensor-local extent.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Axis {
    id: u64,
    name: &'static str,
    parent: Option<u64>,
}

impl Axis {
    pub fn new(name: &'static str) -> Self {
        Self {
            id: NEXT_AXIS.fetch_add(1, Ordering::Relaxed),
            name,
            parent: None,
        }
    }
    pub fn of(self, extent: usize) -> Dim {
        Dim { axis: self, extent }
    }
    /// A distinct alignment identity with ancestry retained for diagnostics.
    pub fn role(self, name: &'static str) -> Self {
        Self {
            parent: Some(self.id),
            ..Self::new(name)
        }
    }
    pub fn name(self) -> &'static str {
        self.name
    }
    pub fn parent_id(self) -> Option<u64> {
        self.parent
    }
}

impl fmt::Debug for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", self.name, self.id)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Dim {
    pub axis: Axis,
    pub extent: usize,
}

pub trait IntoAxes {
    fn into_axes(self) -> Vec<Axis>;
}
impl IntoAxes for Axis {
    fn into_axes(self) -> Vec<Axis> {
        vec![self]
    }
}
impl<const N: usize> IntoAxes for [Axis; N] {
    fn into_axes(self) -> Vec<Axis> {
        self.to_vec()
    }
}
impl IntoAxes for Vec<Axis> {
    fn into_axes(self) -> Vec<Axis> {
        self
    }
}
impl IntoAxes for &[Axis] {
    fn into_axes(self) -> Vec<Axis> {
        self.to_vec()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Shape {
    dims: Vec<Dim>,
    len: usize,
}

impl Shape {
    pub fn new(dims: impl IntoIterator<Item = Dim>) -> Result<Self> {
        let dims: Vec<Dim> = dims.into_iter().collect();
        let mut len = 1_usize;
        for (i, dim) in dims.iter().enumerate() {
            if dim.extent == 0 {
                return Err(format!("zero extent for {:?}", dim.axis).into());
            }
            if dims[..i].iter().any(|d| d.axis == dim.axis) {
                return Err(format!("duplicate axis {:?}", dim.axis).into());
            }
            len = len.checked_mul(dim.extent).ok_or("shape size overflow")?;
        }
        if len > i32::MAX as usize {
            return Err("tensor exceeds cuTile index range".into());
        }
        Ok(Self { dims, len })
    }
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }
    pub fn axes(&self) -> Vec<Axis> {
        self.dims.iter().map(|d| d.axis).collect()
    }
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        false
    } // Scalars contain one element; zero extents are rejected.
    pub fn extent(&self, axis: Axis) -> Result<usize> {
        Ok(self.dims[self.index(axis)?].extent)
    }
    pub(crate) fn index(&self, axis: Axis) -> Result<usize> {
        self.dims
            .iter()
            .position(|d| d.axis == axis)
            .ok_or_else(|| format!("missing axis {axis:?} in {self:?}").into())
    }
    pub(crate) fn contains(&self, axis: Axis) -> bool {
        self.dims.iter().any(|d| d.axis == axis)
    }
    pub(crate) fn select_axes(&self, axes: impl IntoAxes) -> Result<Vec<Axis>> {
        let axes = axes.into_axes();
        for (i, &axis) in axes.iter().enumerate() {
            self.index(axis)?;
            if axes[..i].contains(&axis) {
                return Err(format!("duplicate selected axis {axis:?}").into());
            }
        }
        Ok(axes)
    }
    pub(crate) fn coords(&self, mut flat: usize) -> Vec<usize> {
        let mut result = vec![0; self.rank()];
        for i in (0..self.rank()).rev() {
            result[i] = flat % self.dims[i].extent;
            flat /= self.dims[i].extent;
        }
        result
    }
}

/// Physical strides indexed by the logical axes in Shape.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct Layout {
    pub strides: Vec<usize>,
}
impl Layout {
    pub fn new(shape: &Shape, order: &[Axis]) -> Result<Self> {
        let axes = shape.select_axes(order)?;
        if axes.len() != shape.rank() {
            return Err("layout must contain every axis exactly once".into());
        }
        let mut strides = vec![0; shape.rank()];
        let mut stride = 1;
        for &axis in axes.iter().rev() {
            let i = shape.index(axis)?;
            strides[i] = stride;
            stride *= shape.dims[i].extent;
        }
        Ok(Self { strides })
    }
    pub fn contiguous(shape: &Shape) -> Self {
        Self::new(shape, &shape.axes()).expect("valid shape")
    }
    pub fn offset(&self, coords: &[usize]) -> usize {
        coords.iter().zip(&self.strides).map(|(c, s)| c * s).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_binding_and_validation() -> Result<()> {
        let x = Axis::new("x");
        assert_ne!(x, Axis::new("x"));
        let role = x.role("query");
        assert_ne!(x, role);
        assert!(role.parent_id().is_some());
        assert!(Shape::new([x.of(2), x.of(2)]).is_err());
        assert!(Shape::new([x.of(0)]).is_err());
        assert!(Shape::new([x.of(usize::MAX), role.of(2)]).is_err());
        assert_eq!(Shape::new([x.of(32)])?.extent(x)?, 32);
        assert_eq!(Shape::new([x.of(8)])?.extent(x)?, 8);
        assert_eq!(Shape::new([])?.len(), 1);
        Ok(())
    }
}
