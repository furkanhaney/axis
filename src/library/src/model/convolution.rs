use crate::{
    Axis, Device, Dim, Result, Shape, Tensor,
    nn::{Module, Parameter},
};

struct Convolution<const N: usize> {
    input: Axis,
    output: Dim,
    spatial: [Axis; N],
    kernel: [usize; N],
    stride: [usize; N],
    padding: [usize; N],
    groups: usize,
    group: Axis,
    patch: Axis,
    output_in_group: Axis,
    output_role: Axis,
    bound: Option<BoundConvolution>,
}

struct BoundConvolution {
    input_channels: usize,
    groups: usize,
    patch: usize,
    output_in_group: usize,
    weight: Parameter,
    bias: Parameter,
}

struct Geometry {
    shape: Shape,
    channels: usize,
    patch: usize,
    output_in_group: usize,
}

impl<const N: usize> Convolution<N> {
    fn new(input: Axis, output: Dim, spatial: [Axis; N], kernel: [usize; N]) -> Self {
        Self {
            input,
            output,
            spatial,
            kernel,
            stride: [1; N],
            padding: [0; N],
            groups: 1,
            group: input.role("conv_group"),
            patch: input.role("conv_patch_in_group"),
            output_in_group: output.axis.role("conv_output_in_group"),
            output_role: output.axis.role("conv_output"),
            bound: None,
        }
    }

    fn name() -> &'static str {
        match N {
            2 => "Conv2d",
            3 => "Conv3d",
            _ => "convolution",
        }
    }

    fn geometry(&self, input: &Shape) -> Result<Geometry> {
        let name = Self::name();
        if !(N == 2 || N == 3) {
            return Err("Axis convolution supports exactly two or three spatial axes".into());
        }
        for (index, &axis) in self.spatial.iter().enumerate() {
            if axis == self.input || self.spatial[..index].contains(&axis) {
                return Err(
                    format!("{name} requires distinct input-channel and spatial axes").into(),
                );
            }
        }
        let channels = input.extent(self.input)?;
        let input_spatial: Vec<_> = self
            .spatial
            .iter()
            .map(|&axis| input.extent(axis))
            .collect::<Result<_>>()?;
        if self.kernel.contains(&0) {
            return Err(format!("{name} kernel extents must be positive").into());
        }
        if self.stride.contains(&0) {
            return Err(format!("{name} stride extents must be positive").into());
        }
        if self.groups == 0 {
            return Err(format!("{name} groups must be positive").into());
        }
        if !channels.is_multiple_of(self.groups) {
            return Err(format!("{name} input channels must be divisible by groups").into());
        }
        if !self.output.extent.is_multiple_of(self.groups) {
            return Err(format!("{name} output channels must be divisible by groups").into());
        }
        let mut output_spatial = [0; N];
        for index in 0..N {
            let doubled_padding = self.padding[index]
                .checked_mul(2)
                .ok_or_else(|| format!("{name} padding overflow"))?;
            let padded = input_spatial[index]
                .checked_add(doubled_padding)
                .ok_or_else(|| format!("{name} padded spatial extent overflow"))?;
            if self.kernel[index] > padded {
                return Err(format!("{name} kernel must fit the padded spatial axes").into());
            }
            output_spatial[index] = (padded - self.kernel[index]) / self.stride[index] + 1;
        }
        let channels_per_group = channels / self.groups;
        let patch = self
            .kernel
            .iter()
            .try_fold(channels_per_group, |extent, &kernel| {
                extent.checked_mul(kernel)
            });
        let patch = patch.ok_or_else(|| format!("{name} patch extent overflow"))?;
        let output_in_group = self.output.extent / self.groups;
        if let Some(bound) = &self.bound {
            if bound.input_channels != channels {
                return Err(format!("{name} input channels differ from its built extent").into());
            }
            if bound.groups != self.groups
                || bound.patch != patch
                || bound.output_in_group != output_in_group
            {
                return Err(
                    format!("{name} group geometry differs from its built parameters").into(),
                );
            }
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .filter_map(|dim| {
                if dim.axis == self.input {
                    None
                } else if let Some(index) = self.spatial.iter().position(|&axis| axis == dim.axis) {
                    Some(dim.axis.of(output_spatial[index]))
                } else {
                    Some(*dim)
                }
            })
            .collect();
        dims.push(self.output);
        Ok(Geometry {
            shape: Shape::new(dims)?,
            channels,
            patch,
            output_in_group,
        })
    }

    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(self.geometry(input)?.shape)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let geometry = self.geometry(input)?;
        if let Some(bound) = &self.bound {
            if !bound.weight.tensor().device().same(device) {
                return Err(
                    format!("{} is already built on a different Device", Self::name()).into(),
                );
            }
            return Ok(geometry.shape);
        }
        let weight_shape = Shape::new([
            self.group.of(self.groups),
            self.patch.of(geometry.patch),
            self.output_in_group.of(geometry.output_in_group),
        ])?;
        let mut rng = seed.max(1);
        let scale = (6.0 / (geometry.patch + geometry.output_in_group) as f32).sqrt();
        let values: Vec<_> = (0..weight_shape.len())
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (((rng >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect();
        let weight = Parameter::new(Tensor::from_slice(
            &values,
            weight_shape.dims().iter().copied(),
            device,
        )?);
        let bias = Parameter::new(Tensor::from_slice(
            &vec![0.0; self.output.extent],
            [self.output_role.of(self.output.extent)],
            device,
        )?);
        self.bound = Some(BoundConvolution {
            input_channels: geometry.channels,
            groups: self.groups,
            patch: geometry.patch,
            output_in_group: geometry.output_in_group,
            weight,
            bias,
        });
        Ok(geometry.shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let geometry = self.geometry(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or_else(|| format!("{} must be built before forward", Self::name()))?;
        input
            .unfold_grouped(
                self.input,
                self.spatial,
                self.group.of(self.groups),
                self.patch.of(geometry.patch),
                self.kernel,
                self.stride,
                self.padding,
                0.0,
            )?
            .contract(&bound.weight.tensor(), self.patch)?
            .merge([self.group, self.output_in_group], self.output_role)?
            .add(&bound.bias.tensor())?
            .rename(self.output_role, self.output.axis)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|bound| {
                vec![
                    ("weight".into(), bound.weight.clone()),
                    ("bias".into(), bound.bias.clone()),
                ]
            })
            .unwrap_or_default()
    }
}

/// Named-channel 2D cross-correlation followed by bias addition.
///
/// Stride defaults to `[1, 1]`, padding to `[0, 0]`, and groups to `1`.
/// Spatial arrays are ordered as the supplied named axes. Padding is symmetric.
/// Dilation and asymmetric padding are not supported.
///
/// The backend materializes the patch tensor before tiled grouped contraction;
/// this is a correctness path rather than a fused or throughput-competitive kernel.
pub struct Conv2d(Convolution<2>);

impl Conv2d {
    pub fn new(input: Axis, output: Dim, spatial: [Axis; 2], kernel: [usize; 2]) -> Self {
        Self(Convolution::new(input, output, spatial, kernel))
    }

    /// Set stride in the order of the supplied spatial axes.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric zero-padding in the order of the supplied spatial axes.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.0.padding = padding;
        self
    }

    /// Partition input and output channels into independent convolution groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.0.groups = groups;
        self
    }
}

impl Module for Conv2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.0.build(input, device, seed)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.0.named_parameters()
    }
}

/// Named-channel 3D cross-correlation followed by bias addition.
///
/// The three spatial axes conventionally represent depth, height, and width;
/// kernel, stride, and symmetric padding arrays use that same supplied order.
/// Stride defaults to `[1, 1, 1]`, padding to `[0, 0, 0]`, and groups to `1`.
/// Dilation and asymmetric padding are not supported.
///
/// The backend computes patch indices from compact geometry and uses a
/// deterministic input-centric derivative, but materializes the FP32 patch
/// tensor before contraction. Downstream layout operations still have the
/// generic index-plan limits documented by Axis.
pub struct Conv3d(Convolution<3>);

impl Conv3d {
    pub fn new(input: Axis, output: Dim, spatial: [Axis; 3], kernel: [usize; 3]) -> Self {
        Self(Convolution::new(input, output, spatial, kernel))
    }

    /// Set stride in the order of the supplied spatial axes.
    pub fn stride(mut self, stride: [usize; 3]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric zero-padding in the order of the supplied spatial axes.
    pub fn padding(mut self, padding: [usize; 3]) -> Self {
        self.0.padding = padding;
        self
    }

    /// Partition input and output channels into independent convolution groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.0.groups = groups;
        self
    }
}

impl Module for Conv3d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.0.build(input, device, seed)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.0.named_parameters()
    }
}

/// Deterministic uniform weight matching [`Convolution::build`]'s own xorshift init, scaled by
/// `sqrt(6 / (patch + other_side))` (a Glorot/Xavier-uniform-shaped bound, not PyTorch's default
/// kaiming-uniform init). Shared by `Convolution` and `TransposedConvolution` so both draw from
/// the exact same formula and stream.
fn xavier_uniform_weight(len: usize, patch: usize, other_side: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed.max(1);
    let scale = (6.0 / (patch + other_side) as f32).sqrt();
    (0..len)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (((rng >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Named-channel 1D cross-correlation. Reuses [`Conv2d`]'s exact geometry, weight layout, and
/// xorshift initialization over a synthetic unit spatial axis appended for the call: a kernel of
/// `1`, stride of `1`, and padding of `0` on that axis is an exact no-op (it always contracts a
/// single, always-in-bounds position), and `Tensor::broadcast_to`/`Tensor::select` add and remove
/// an extent-1 axis with an exact identity gradient, so no separate rank-1 backend path exists.
/// Stride and padding default to `1` and `0`; groups default to `1`, matching [`Conv2d`].
pub struct Conv1d {
    dummy: Axis,
    inner: Conv2d,
}

impl Conv1d {
    pub fn new(input: Axis, output: Dim, spatial: Axis, kernel: usize) -> Self {
        let dummy = spatial.role("conv1d_unit");
        Self {
            dummy,
            inner: Conv2d::new(input, output, [spatial, dummy], [kernel, 1]),
        }
    }

    /// Set stride along the one real spatial axis. Defaults to `1`.
    pub fn stride(mut self, stride: usize) -> Self {
        self.inner = self.inner.stride([stride, 1]);
        self
    }

    /// Set symmetric zero-padding along the one real spatial axis. Defaults to `0`.
    pub fn padding(mut self, padding: usize) -> Self {
        self.inner = self.inner.padding([padding, 0]);
        self
    }

    /// Partition input and output channels into independent convolution groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.inner = self.inner.groups(groups);
        self
    }

    fn expanded(&self, input: &Shape) -> Result<Shape> {
        let mut dims = input.dims().to_vec();
        dims.push(self.dummy.of(1));
        Shape::new(dims)
    }
}

impl Module for Conv1d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let out = self.inner.output_shape(&self.expanded(input)?)?;
        Shape::new(out.dims().iter().copied().filter(|d| d.axis != self.dummy))
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let out = self.inner.build(&self.expanded(input)?, device, seed)?;
        Shape::new(out.dims().iter().copied().filter(|d| d.axis != self.dummy))
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let expanded = input.broadcast_to(&self.expanded(input.shape())?)?;
        self.inner.forward(&expanded)?.select(self.dummy, 0)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.inner.named_parameters()
    }
}

struct TransposeGeometry {
    shape: Shape,
    input_channels: usize,
    patch: usize,
    input_in_group: usize,
}

/// Transposed (fractionally-strided) named-channel cross-correlation: the adjoint of
/// [`Convolution`] built the same way its own backward gradient is (a contraction against the
/// weight followed by `fold`, the scatter/col2im that `unfold`'s own backward already
/// implements), so no separate scatter kernel exists for it.
///
/// Weight layout is `[group, patch(out_per_group, kh, kw), input_in_group]` -- the same
/// `[group, patch, other_side]` convention [`Convolution`] uses, with "the axis being
/// patch-extracted" and "the axis being contracted to" swapped: `patch` here is sized from the
/// *output* channels per group (PyTorch's own ConvTranspose weight is `[in_channels,
/// out_channels/groups, kH, kW]`; this is the "roles swapped" analogue of Axis's own `Conv2d`
/// divergence from PyTorch's `[out, in, kh, kw]`, not a copy of PyTorch's own layout).
struct TransposedConvolution<const N: usize> {
    input: Axis,
    output: Dim,
    spatial: [Axis; N],
    kernel: [usize; N],
    stride: [usize; N],
    padding: [usize; N],
    output_padding: [usize; N],
    groups: usize,
    group: Axis,
    patch: Axis,
    input_in_group: Axis,
    output_role: Axis,
    bound: Option<BoundConvolution>,
}

impl<const N: usize> TransposedConvolution<N> {
    fn new(input: Axis, output: Dim, spatial: [Axis; N], kernel: [usize; N]) -> Self {
        Self {
            input,
            output,
            spatial,
            kernel,
            stride: [1; N],
            padding: [0; N],
            output_padding: [0; N],
            groups: 1,
            group: input.role("conv_transpose_group"),
            patch: output.axis.role("conv_transpose_patch_in_group"),
            input_in_group: input.role("conv_transpose_input_in_group"),
            output_role: output.axis.role("conv_transpose_output"),
            bound: None,
        }
    }

    fn name() -> &'static str {
        match N {
            2 => "ConvTranspose2d",
            3 => "ConvTranspose3d",
            _ => "transposed convolution",
        }
    }

    fn geometry(&self, input: &Shape) -> Result<TransposeGeometry> {
        let name = Self::name();
        if !(N == 2 || N == 3) {
            return Err(
                "Axis transposed convolution supports exactly two or three spatial axes".into(),
            );
        }
        for (index, &axis) in self.spatial.iter().enumerate() {
            if axis == self.input || self.spatial[..index].contains(&axis) {
                return Err(
                    format!("{name} requires distinct input-channel and spatial axes").into(),
                );
            }
        }
        let channels = input.extent(self.input)?;
        let input_spatial: Vec<_> = self
            .spatial
            .iter()
            .map(|&axis| input.extent(axis))
            .collect::<Result<_>>()?;
        if self.kernel.contains(&0) {
            return Err(format!("{name} kernel extents must be positive").into());
        }
        if self.stride.contains(&0) {
            return Err(format!("{name} stride extents must be positive").into());
        }
        if self.groups == 0 {
            return Err(format!("{name} groups must be positive").into());
        }
        if !channels.is_multiple_of(self.groups) {
            return Err(format!("{name} input channels must be divisible by groups").into());
        }
        if !self.output.extent.is_multiple_of(self.groups) {
            return Err(format!("{name} output channels must be divisible by groups").into());
        }
        // Matches PyTorch's own ConvTranspose constraint (with dilation pinned at 1, where it
        // reduces to exactly this): output_padding must be strictly less than stride, so it can
        // only disambiguate which of several valid input sizes produced this output, never add
        // a whole extra window.
        for index in 0..N {
            if self.output_padding[index] >= self.stride[index] {
                return Err(format!("{name} output_padding must be less than stride").into());
            }
        }
        let mut output_spatial = [0usize; N];
        for index in 0..N {
            let expanded = (input_spatial[index] - 1)
                .checked_mul(self.stride[index])
                .ok_or_else(|| format!("{name} expanded spatial extent overflow"))?;
            let with_kernel = expanded
                .checked_add(self.kernel[index])
                .ok_or_else(|| format!("{name} expanded spatial extent overflow"))?;
            let doubled_padding = self.padding[index]
                .checked_mul(2)
                .ok_or_else(|| format!("{name} padding overflow"))?;
            let unpadded = with_kernel
                .checked_sub(doubled_padding)
                .ok_or_else(|| format!("{name} padding exceeds the expanded spatial extent"))?;
            output_spatial[index] = unpadded
                .checked_add(self.output_padding[index])
                .ok_or_else(|| format!("{name} expanded spatial extent overflow"))?;
        }
        let output_channels_per_group = self.output.extent / self.groups;
        let patch = self
            .kernel
            .iter()
            .try_fold(output_channels_per_group, |extent, &kernel| {
                extent.checked_mul(kernel)
            });
        let patch = patch.ok_or_else(|| format!("{name} patch extent overflow"))?;
        let input_in_group = channels / self.groups;
        if let Some(bound) = &self.bound {
            if bound.input_channels != channels {
                return Err(format!("{name} input channels differ from its built extent").into());
            }
            if bound.groups != self.groups
                || bound.patch != patch
                || bound.output_in_group != input_in_group
            {
                return Err(
                    format!("{name} group geometry differs from its built parameters").into(),
                );
            }
        }
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .filter_map(|dim| {
                if dim.axis == self.input {
                    None
                } else if let Some(index) = self.spatial.iter().position(|&axis| axis == dim.axis) {
                    Some(dim.axis.of(output_spatial[index]))
                } else {
                    Some(*dim)
                }
            })
            .collect();
        dims.push(self.output);
        Ok(TransposeGeometry {
            shape: Shape::new(dims)?,
            input_channels: channels,
            patch,
            input_in_group,
        })
    }

    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(self.geometry(input)?.shape)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let geometry = self.geometry(input)?;
        if let Some(bound) = &self.bound {
            if !bound.weight.tensor().device().same(device) {
                return Err(
                    format!("{} is already built on a different Device", Self::name()).into(),
                );
            }
            return Ok(geometry.shape);
        }
        let weight_shape = Shape::new([
            self.group.of(self.groups),
            self.patch.of(geometry.patch),
            self.input_in_group.of(geometry.input_in_group),
        ])?;
        let values = xavier_uniform_weight(
            weight_shape.len(),
            geometry.patch,
            geometry.input_in_group,
            seed,
        );
        let weight = Parameter::new(Tensor::from_slice(
            &values,
            weight_shape.dims().iter().copied(),
            device,
        )?);
        let bias = Parameter::new(Tensor::from_slice(
            &vec![0.0; self.output.extent],
            [self.output_role.of(self.output.extent)],
            device,
        )?);
        self.bound = Some(BoundConvolution {
            input_channels: geometry.input_channels,
            groups: self.groups,
            patch: geometry.patch,
            output_in_group: geometry.input_in_group,
            weight,
            bias,
        });
        Ok(geometry.shape)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let geometry = self.geometry(input.shape())?;
        let bound = self
            .bound
            .as_ref()
            .ok_or_else(|| format!("{} must be built before forward", Self::name()))?;
        let split = input.split(
            self.input,
            [
                self.group.of(self.groups),
                self.input_in_group.of(geometry.input_in_group),
            ],
        )?;
        let patches = split.contract(&bound.weight.tensor(), self.input_in_group)?;
        // Declared axis order matches `geometry`'s own (spatial axes in place, the channel axis
        // last): callers rely on `output_shape`/`build` naming the order `forward` actually
        // produces, and `Tensor::to_vec` follows a tensor's own declared `Shape.dims()` order.
        let mut core_dims = Vec::with_capacity(input.shape().rank());
        for dim in input.shape().dims() {
            if dim.axis == self.input {
                continue;
            } else if let Some(index) = self.spatial.iter().position(|&axis| axis == dim.axis) {
                let full = geometry.shape.extent(dim.axis)?;
                core_dims.push(dim.axis.of(full - self.output_padding[index]));
            } else {
                core_dims.push(*dim);
            }
        }
        core_dims.push(self.output_role.of(self.output.extent));
        let core_shape = Shape::new(core_dims)?;
        let folded = patches.fold_grouped(
            &core_shape,
            self.output_role,
            self.spatial,
            self.group.of(self.groups),
            self.patch.of(geometry.patch),
            self.kernel,
            self.stride,
            self.padding,
        )?;
        let mut result = folded;
        for index in 0..N {
            if self.output_padding[index] > 0 {
                result = result.pad_zeros(self.spatial[index], 0, self.output_padding[index])?;
            }
        }
        result
            .add(&bound.bias.tensor())?
            .rename(self.output_role, self.output.axis)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.bound
            .as_ref()
            .map(|bound| {
                vec![
                    ("weight".into(), bound.weight.clone()),
                    ("bias".into(), bound.bias.clone()),
                ]
            })
            .unwrap_or_default()
    }
}

/// Named-channel 1D transposed convolution; see [`ConvTranspose2d`] for the shared contract.
/// Reuses `ConvTranspose2d`'s exact geometry and initialization over a synthetic unit spatial
/// axis, exactly as [`Conv1d`] reuses [`Conv2d`].
pub struct ConvTranspose1d {
    dummy: Axis,
    inner: ConvTranspose2d,
}

impl ConvTranspose1d {
    pub fn new(input: Axis, output: Dim, spatial: Axis, kernel: usize) -> Self {
        let dummy = spatial.role("conv_transpose1d_unit");
        Self {
            dummy,
            inner: ConvTranspose2d::new(input, output, [spatial, dummy], [kernel, 1]),
        }
    }

    /// Set stride along the one real spatial axis. Defaults to `1`.
    pub fn stride(mut self, stride: usize) -> Self {
        self.inner = self.inner.stride([stride, 1]);
        self
    }

    /// Set symmetric zero-padding along the one real spatial axis. Defaults to `0`.
    pub fn padding(mut self, padding: usize) -> Self {
        self.inner = self.inner.padding([padding, 0]);
        self
    }

    /// Set output padding along the one real spatial axis. Must be less than stride. Defaults to `0`.
    pub fn output_padding(mut self, output_padding: usize) -> Self {
        self.inner = self.inner.output_padding([output_padding, 0]);
        self
    }

    /// Partition input and output channels into independent convolution groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.inner = self.inner.groups(groups);
        self
    }

    fn expanded(&self, input: &Shape) -> Result<Shape> {
        let mut dims = input.dims().to_vec();
        dims.push(self.dummy.of(1));
        Shape::new(dims)
    }
}

impl Module for ConvTranspose1d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let out = self.inner.output_shape(&self.expanded(input)?)?;
        Shape::new(out.dims().iter().copied().filter(|d| d.axis != self.dummy))
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let out = self.inner.build(&self.expanded(input)?, device, seed)?;
        Shape::new(out.dims().iter().copied().filter(|d| d.axis != self.dummy))
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let expanded = input.broadcast_to(&self.expanded(input.shape())?)?;
        self.inner.forward(&expanded)?.select(self.dummy, 0)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.inner.named_parameters()
    }
}

/// Named-channel 2D transposed convolution (PyTorch's `nn.ConvTranspose2d`): stride, padding,
/// output_padding and groups are supported; dilation is pinned at `1`, matching [`Conv2d`].
/// Stride defaults to `[1, 1]`, padding and output_padding to `[0, 0]`, and groups to `1`.
pub struct ConvTranspose2d(TransposedConvolution<2>);

impl ConvTranspose2d {
    pub fn new(input: Axis, output: Dim, spatial: [Axis; 2], kernel: [usize; 2]) -> Self {
        Self(TransposedConvolution::new(input, output, spatial, kernel))
    }

    /// Set stride in the order of the supplied spatial axes.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric zero-padding in the order of the supplied spatial axes.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.0.padding = padding;
        self
    }

    /// Set output padding (each entry must be less than the matching stride) in the order of
    /// the supplied spatial axes.
    pub fn output_padding(mut self, output_padding: [usize; 2]) -> Self {
        self.0.output_padding = output_padding;
        self
    }

    /// Partition input and output channels into independent convolution groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.0.groups = groups;
        self
    }
}

impl Module for ConvTranspose2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.0.build(input, device, seed)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.0.named_parameters()
    }
}

/// Named-channel 3D transposed convolution; see [`ConvTranspose2d`] for the shared contract.
/// The three spatial axes conventionally represent depth, height, and width, matching [`Conv3d`].
pub struct ConvTranspose3d(TransposedConvolution<3>);

impl ConvTranspose3d {
    pub fn new(input: Axis, output: Dim, spatial: [Axis; 3], kernel: [usize; 3]) -> Self {
        Self(TransposedConvolution::new(input, output, spatial, kernel))
    }

    /// Set stride in the order of the supplied spatial axes.
    pub fn stride(mut self, stride: [usize; 3]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric zero-padding in the order of the supplied spatial axes.
    pub fn padding(mut self, padding: [usize; 3]) -> Self {
        self.0.padding = padding;
        self
    }

    /// Set output padding (each entry must be less than the matching stride) in the order of
    /// the supplied spatial axes.
    pub fn output_padding(mut self, output_padding: [usize; 3]) -> Self {
        self.0.output_padding = output_padding;
        self
    }

    /// Partition input and output channels into independent convolution groups.
    pub fn groups(mut self, groups: usize) -> Self {
        self.0.groups = groups;
        self
    }
}

impl Module for ConvTranspose3d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.0.build(input, device, seed)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        self.0.named_parameters()
    }
}

/// Named-axis `im2col` (PyTorch's `nn.Unfold`): extracts sliding local kernel-sized blocks,
/// replacing `channels` in place with `patch` (extent `channels * kernel volume`) and shrinking
/// `spatial` to their windowed count. Out-of-bounds (padding) positions read `0.0`. Stride
/// defaults to `[1, 1]` and padding to `[0, 0]`. Stateless: `build` only validates and infers the
/// output shape.
pub struct Unfold {
    channels: Axis,
    spatial: [Axis; 2],
    patch: Axis,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl Unfold {
    pub fn new(channels: Axis, spatial: [Axis; 2], patch: Axis, kernel: [usize; 2]) -> Self {
        Self {
            channels,
            spatial,
            patch,
            kernel,
            stride: [1, 1],
            padding: [0, 0],
        }
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to `[1, 1]`.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.stride = stride;
        self
    }

    /// Set symmetric zero-padding in the order of the supplied spatial axes. Defaults to `[0, 0]`.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for Unfold {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let (shape, ..) = Tensor::unfold_shape::<2>(
            "Unfold",
            input,
            self.channels,
            self.spatial,
            None,
            self.patch,
            self.kernel,
            self.stride,
            self.padding,
        )?;
        Ok(shape)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, patch_extent, ..) = Tensor::unfold_shape::<2>(
            "Unfold",
            input.shape(),
            self.channels,
            self.spatial,
            None,
            self.patch,
            self.kernel,
            self.stride,
            self.padding,
        )?;
        input.unfold_ungrouped(
            self.channels,
            self.spatial,
            self.patch.of(patch_extent),
            self.kernel,
            self.stride,
            self.padding,
        )
    }
}

/// Named-axis `col2im` (PyTorch's `nn.Fold`), the adjoint of [`Unfold`]: combines an array of
/// sliding local kernel-sized blocks (`patch` in place of a channel axis) by summing overlapping
/// contributions into `output_size`-shaped `spatial` axes, reconstructing `channels` (inferred
/// from the input's `patch` extent divided by the kernel volume). Rejects an input whose extents
/// are inconsistent with `output_size`/`kernel`/`stride`/`padding` before any device call. Stride
/// defaults to `[1, 1]` and padding to `[0, 0]`.
pub struct Fold {
    channels: Axis,
    spatial: [Axis; 2],
    image_spatial: [usize; 2],
    patch: Axis,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl Fold {
    pub fn new(
        channels: Axis,
        spatial: [Axis; 2],
        output_size: [usize; 2],
        patch: Axis,
        kernel: [usize; 2],
    ) -> Self {
        Self {
            channels,
            spatial,
            image_spatial: output_size,
            patch,
            kernel,
            stride: [1, 1],
            padding: [0, 0],
        }
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to `[1, 1]`.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.stride = stride;
        self
    }

    /// Set symmetric zero-padding in the order of the supplied spatial axes. Defaults to `[0, 0]`.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.padding = padding;
        self
    }

    /// The reconstructed image shape and the input's own patch extent, after checking that the
    /// input's extents are consistent with `output_size`/`kernel`/`stride`/`padding`.
    fn geometry(&self, input: &Shape) -> Result<(Shape, usize)> {
        let kernel_volume: usize = self.kernel.iter().product();
        let patch_extent = input.extent(self.patch)?;
        if kernel_volume == 0 || patch_extent == 0 || !patch_extent.is_multiple_of(kernel_volume) {
            return Err(
                "Fold patch extent must be a positive multiple of the kernel volume".into(),
            );
        }
        let channel_extent = patch_extent / kernel_volume;
        let mut dims: Vec<Dim> = input
            .dims()
            .iter()
            .copied()
            .filter(|d| d.axis != self.patch && !self.spatial.contains(&d.axis))
            .collect();
        dims.push(self.channels.of(channel_extent));
        for (&axis, &extent) in self.spatial.iter().zip(self.image_spatial.iter()) {
            dims.push(axis.of(extent));
        }
        let image_shape = Shape::new(dims)?;
        let (expected_patches, expected_patch_extent, ..) = Tensor::unfold_shape::<2>(
            "Fold",
            &image_shape,
            self.channels,
            self.spatial,
            None,
            self.patch,
            self.kernel,
            self.stride,
            self.padding,
        )?;
        if expected_patch_extent != patch_extent {
            return Err(
                "Fold patch extent does not match output_size, kernel, stride and padding".into(),
            );
        }
        for dim in expected_patches.dims() {
            if input.extent(dim.axis)? != dim.extent {
                return Err(format!(
                    "Fold input extent for {:?} does not match output_size, kernel, stride and padding",
                    dim.axis
                )
                .into());
            }
        }
        Ok((image_shape, patch_extent))
    }
}

impl Module for Fold {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(self.geometry(input)?.0)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (image_shape, patch_extent) = self.geometry(input.shape())?;
        input.fold_ungrouped(
            &image_shape,
            self.channels,
            self.spatial,
            self.patch.of(patch_extent),
            self.kernel,
            self.stride,
            self.padding,
        )
    }
}

/// Which reduction a [`Pooling`] window applies. Shared geometry (kernel/stride/padding/ceil
/// validation and the windowed patch extraction) is identical across the family; only the
/// per-window reduction differs, so one generic struct serves `MaxPool*d`, `AvgPool*d`, and
/// `LPPool*d` instead of three near-duplicate ones.
#[derive(Clone, Copy)]
enum PoolReduce {
    Max,
    /// `sum(window) / divisor`, PyTorch's `avg_pool` divisor rule: `divisor_override` when
    /// set; otherwise, with `count_include_pad`, the window clipped to the padded input (the
    /// full kernel volume except for a `ceil_mode` window overhanging the right padding);
    /// without it, the count of real (non-padding) positions.
    Avg {
        count_include_pad: bool,
        divisor_override: Option<usize>,
    },
    /// Power-average pooling, `(sum(x^p))^(1/p)`, for any finite positive real `p`.
    Lp(f32),
}

/// Host-side geometry of one [`Pooling`] call, computed before any device work.
struct PoolGeometry<const N: usize> {
    shape: Shape,
    channels: usize,
    input: [usize; N],
    output: [usize; N],
    /// Zero positions appended past the right padding so every `ceil_mode` window exists.
    overhang: [usize; N],
}

struct Pooling<const N: usize> {
    channels: Axis,
    spatial: [Axis; N],
    kernel: [usize; N],
    stride: [usize; N],
    padding: [usize; N],
    ceil_mode: bool,
    patch: Axis,
    reduce: PoolReduce,
    name: &'static str,
}

impl<const N: usize> Pooling<N> {
    fn new(
        channels: Axis,
        spatial: [Axis; N],
        kernel: [usize; N],
        reduce: PoolReduce,
        name: &'static str,
    ) -> Self {
        Self {
            channels,
            spatial,
            kernel,
            // PyTorch's own `MaxPool*d`/`AvgPool*d`/`LPPool*d` default stride to the kernel
            // extent.
            stride: kernel,
            padding: [0; N],
            ceil_mode: false,
            patch: channels.role("pool_patch"),
            reduce,
            name,
        }
    }

    fn geometry(&self, input: &Shape) -> Result<PoolGeometry<N>> {
        let name = self.name;
        if !(N == 2 || N == 3) {
            return Err("Axis pooling supports exactly two or three spatial axes".into());
        }
        for (index, &axis) in self.spatial.iter().enumerate() {
            if axis == self.channels || self.spatial[..index].contains(&axis) {
                return Err(format!("{name} requires distinct channel and spatial axes").into());
            }
        }
        if self.kernel.contains(&0) {
            return Err(format!("{name} kernel extents must be positive").into());
        }
        if self.stride.contains(&0) {
            return Err(format!("{name} stride extents must be positive").into());
        }
        if let PoolReduce::Avg {
            divisor_override: Some(0),
            ..
        } = self.reduce
        {
            return Err(format!("{name} divisor_override must be positive").into());
        }
        let channels = input.extent(self.channels)?;
        let mut sizes = [0; N];
        let mut output = [0; N];
        let mut overhang = [0; N];
        // Several parallel small arrays share one index; a `.zip()` chain would read worse
        // than the loop it replaces.
        #[allow(clippy::needless_range_loop)]
        for index in 0..N {
            let (kernel, stride) = (self.kernel[index], self.stride[index]);
            let doubled_padding = self.padding[index]
                .checked_mul(2)
                .ok_or_else(|| format!("{name} padding overflow"))?;
            if doubled_padding > kernel {
                // PyTorch's own MaxPool/AvgPool constraint. It also guarantees every window
                // keeps at least one real, unpadded element, so a fully-padding window is
                // unreachable.
                return Err(
                    format!("{name} padding must be at most half the kernel extent").into(),
                );
            }
            let extent = input.extent(self.spatial[index])?;
            let padded = extent
                .checked_add(doubled_padding)
                .ok_or_else(|| format!("{name} padded spatial extent overflow"))?;
            // PyTorch's `pooling_output_shape`: `ceil_mode` rounds the window count up, then
            // drops a last window that would start past the input and its left padding (in
            // the right padding or beyond), so every window keeps a real element.
            let span = if self.ceil_mode {
                padded + (stride - 1)
            } else {
                padded
            };
            if kernel > span {
                return Err(format!("{name} kernel must fit the padded spatial axes").into());
            }
            let mut count = (span - kernel) / stride + 1;
            if self.ceil_mode && (count - 1) * stride >= extent + self.padding[index] {
                count -= 1;
            }
            sizes[index] = extent;
            output[index] = count;
            overhang[index] = ((count - 1) * stride + kernel).saturating_sub(padded);
        }
        // Matches `unfold_grouped`'s own layout: unrelated axes (spatial axes replaced in
        // place, other axes untouched) come first, then the channel axis is appended, exactly
        // as `Conv2d`/`Conv3d` append their new output-channel axis.
        let mut dims: Vec<_> = input
            .dims()
            .iter()
            .filter(|dim| dim.axis != self.channels)
            .map(|dim| {
                self.spatial
                    .iter()
                    .position(|&axis| axis == dim.axis)
                    .map_or(*dim, |index| dim.axis.of(output[index]))
            })
            .collect();
        dims.push(self.channels.of(channels));
        Ok(PoolGeometry {
            shape: Shape::new(dims)?,
            channels,
            input: sizes,
            output,
            overhang,
        })
    }

    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(self.geometry(input)?.shape)
    }

    /// Per-output-window element counts over the spatial axes (supplied order, last axis
    /// fastest): each window clipped to the padded input when `include_padding`, else to the
    /// real input. `None` when every window counts the full kernel volume, so callers keep
    /// the plain uniform reduction (and its exact bits).
    fn window_counts(
        &self,
        geometry: &PoolGeometry<N>,
        include_padding: bool,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let volume: usize = self.kernel.iter().product();
        let windows: usize = geometry.output.iter().product();
        let mut counts = Vec::with_capacity(windows);
        for window in 0..windows {
            let mut remainder = window;
            let mut count = 1;
            for index in (0..N).rev() {
                let position = remainder % geometry.output[index];
                remainder /= geometry.output[index];
                let (extent, padding) = (geometry.input[index], self.padding[index]);
                // Window bounds in padded coordinates.
                let start = position * self.stride[index];
                let end = (start + self.kernel[index]).min(extent + 2 * padding);
                count *= if include_padding {
                    end - start
                } else {
                    end.min(extent + padding) - start.max(padding)
                };
            }
            counts.push(count);
        }
        if counts.iter().all(|&count| count == volume) {
            return Ok(None);
        }
        let counts: Vec<f32> = counts.into_iter().map(|count| count as f32).collect();
        Ok(Some(Tensor::from_slice(
            &counts,
            self.spatial_dims(geometry),
            device,
        )?))
    }

    /// The pooled spatial axes alone, in supplied order: the shape of a per-window divisor.
    fn spatial_dims(&self, geometry: &PoolGeometry<N>) -> Vec<Dim> {
        (0..N)
            .map(|index| self.spatial[index].of(geometry.output[index]))
            .collect()
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let geometry = self.geometry(input.shape())?;
        let kernel_volume = self.kernel.iter().product();
        // Max pads with negative infinity so a padded position can never win; average and
        // power-average pad with zero (a sum-neutral value) exactly as `Conv2d` does, and
        // count padding through their explicit divisors instead.
        let fill = match self.reduce {
            PoolReduce::Max => f32::NEG_INFINITY,
            PoolReduce::Avg { .. } | PoolReduce::Lp(_) => 0.0,
        };
        // A `ceil_mode` window overhanging the right padding reads zeros appended past the
        // real input: only average and LP pooling expose `ceil_mode`, and for their sums an
        // appended zero is indistinguishable from padding.
        let mut source = input.clone();
        for index in 0..N {
            if geometry.overhang[index] > 0 {
                source = source.pad_zeros(self.spatial[index], 0, geometry.overhang[index])?;
            }
        }
        // Materialize the windowed patches (as `Conv2d`/`Conv3d` do); `unfold_grouped`'s
        // backward (col2im) already sums overlapping window contributions exactly, so no new
        // backend kernel is needed for either padding or a stride smaller than the kernel, for
        // any of the three reductions below.
        let patches = source.unfold_grouped(
            self.channels,
            self.spatial,
            self.channels.of(geometry.channels),
            self.patch.of(kernel_volume),
            self.kernel,
            self.stride,
            self.padding,
            fill,
        )?;
        match self.reduce {
            // `-min(-patch)`: `Tensor::min` already ignores non-finite candidates and breaks
            // ties toward the first logical coordinate.
            PoolReduce::Max => patches.scale(-1.0)?.min(self.patch)?.scale(-1.0),
            PoolReduce::Avg {
                count_include_pad,
                divisor_override,
            } => {
                if let Some(divisor) = divisor_override {
                    let windows = geometry.output.iter().product();
                    let divisor = Tensor::from_slice(
                        &vec![divisor as f32; windows],
                        self.spatial_dims(&geometry),
                        input.device(),
                    )?;
                    return patches.sum(self.patch)?.div(&divisor);
                }
                match self.window_counts(&geometry, count_include_pad, input.device())? {
                    None => patches.mean(self.patch),
                    Some(counts) => patches.sum(self.patch)?.div(&counts),
                }
            }
            PoolReduce::Lp(p) => {
                let power = if p.fract() == 0.0 {
                    // Integer `p`: repeated products, the exact path every integer `p` has
                    // always taken (its bits must not move).
                    let mut power = patches.clone();
                    for _ in 1..p as u64 {
                        power = power.mul(&patches)?;
                    }
                    power
                } else {
                    // Real `p`: `exp(p * ln x)`. A zero reads as one inside the logarithm and
                    // is masked back to zero afterwards, so `0^p = 0` with a zero (not NaN)
                    // gradient; a negative `x` stays NaN, as `x.pow(p)` is in PyTorch.
                    let zero = patches.eq(0.0)?;
                    patches
                        .add(&zero)?
                        .ln()?
                        .scale(p)?
                        .exp()?
                        .mul(&zero.logical_not()?)?
                };
                let mut sum = power.sum(self.patch)?;
                // PyTorch composes LP pooling as `avg_pool(x^p, ceil_mode) * kernel_volume`: a
                // `ceil_mode` window clipped at the input edge is its sum rescaled by
                // `kernel_volume / clipped_count`, not its plain sum.
                if let Some(counts) = self.window_counts(&geometry, true, input.device())? {
                    sum = sum.div(&counts)?.scale(kernel_volume as f32)?;
                }
                // sign(sum) as a constant (no gradient of its own, matching `torch.sign`):
                // +1/-1/0. An odd integer `p` can leave `sum` negative; the signed root keeps
                // it real where a literal `1/p` power would be NaN.
                let sign = sum.gt(0.0)?.sub(&sum.lt(0.0)?)?;
                let root = sum.abs()?.ln()?.scale(1.0 / p)?.exp()?;
                sign.mul(&root)
            }
        }
    }

    /// [`Self::forward`] for max pooling, plus PyTorch's `return_indices=True` indices: per
    /// output element in its logical order, the row-major offset (supplied spatial order,
    /// last axis fastest) of the winning input position within its spatial volume. The
    /// winner is the first position in window scan order (last spatial axis innermost)
    /// holding the output value, exactly the element the gradient routes to.
    fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        let geometry = self.geometry(input.shape())?;
        let output = self.forward(input)?;
        let maxima = output.to_vec()?;
        let values = input.to_vec()?;
        let input_shape = input.shape();
        let output_shape = output.shape();
        let mut input_positions = Vec::with_capacity(input_shape.rank());
        for dim in input_shape.dims() {
            input_positions.push(output_shape.index(dim.axis)?);
        }
        let mut spatial_positions = [0usize; N];
        for (slot, &axis) in spatial_positions.iter_mut().zip(&self.spatial) {
            *slot = input_shape.index(axis)?;
        }
        let volume: usize = self.kernel.iter().product();
        let mut indices = Vec::with_capacity(maxima.len());
        for (element, &maximum) in maxima.iter().enumerate() {
            let output_coords = output_shape.coords(element);
            let mut coords: Vec<usize> = input_positions
                .iter()
                .map(|&position| output_coords[position])
                .collect();
            let window_start = coords.clone();
            let mut winner = None;
            'scan: for offset in 0..volume {
                let mut remainder = offset;
                let mut flat = 0;
                let mut stride = 1;
                for index in (0..N).rev() {
                    let kernel_offset = remainder % self.kernel[index];
                    remainder /= self.kernel[index];
                    let padded =
                        window_start[spatial_positions[index]] * self.stride[index] + kernel_offset;
                    let Some(position) = padded
                        .checked_sub(self.padding[index])
                        .filter(|&position| position < geometry.input[index])
                    else {
                        continue 'scan;
                    };
                    coords[spatial_positions[index]] = position;
                    flat += position * stride;
                    stride *= geometry.input[index];
                }
                let logical = input_shape
                    .dims()
                    .iter()
                    .zip(&coords)
                    .fold(0, |index, (dim, &coord)| index * dim.extent + coord);
                let value = values[logical];
                if value.is_finite() && value == maximum {
                    winner = Some(flat);
                    break;
                }
            }
            indices.push(winner.ok_or_else(|| {
                format!(
                    "{} found no finite maximum at output element {element}",
                    self.name
                )
            })?);
        }
        Ok((output, indices))
    }
}

/// Named-channel 2D max pooling: reduce each kernel window to its maximum, independently per
/// channel. Stride defaults to the kernel extent and padding to `[0, 0]`, matching PyTorch's
/// `nn.MaxPool2d`. Spatial arrays are ordered as the supplied named axes, preserved (at their
/// new, pooled extent) in their original position; the channel axis is appended, exactly as
/// `Conv2d` appends its output-channel axis. Padding must be at most half the kernel extent in
/// each dimension (PyTorch's own `MaxPool` constraint); a padded position can never win the
/// maximum. Ties route the gradient to the first logical coordinate along each spatial axis,
/// independently of physical layout, matching `Tensor::min`.
///
/// The backend materializes the windowed patch tensor (as `Conv2d` does) before a negated
/// minimum reduction; this is a correctness path rather than a fused kernel.
pub struct MaxPool2d(Pooling<2>);

impl MaxPool2d {
    pub fn new(channels: Axis, spatial: [Axis; 2], kernel: [usize; 2]) -> Self {
        Self(Pooling::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Max,
            "MaxPool2d",
        ))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric padding in the order of the supplied spatial axes. Defaults to `[0, 0]`.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.0.padding = padding;
        self
    }

    /// [`Module::forward`] plus PyTorch's `return_indices=True` indices, the input to
    /// [`MaxUnpool2d`]: one host-side index per output element, in the output's logical
    /// ([`Tensor::to_vec`]) order, each the row-major offset of the winning input position
    /// within its spatial volume (supplied spatial order, last axis fastest), per slice of
    /// every other axis. The winner is the first maximum in window scan order (last spatial
    /// axis innermost), the same element the gradient routes to. A window with no finite
    /// candidate is an error: there is no position a NaN result could name.
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        self.0.forward_with_indices(input)
    }
}

impl Module for MaxPool2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-channel 3D max pooling; see [`MaxPool2d`] for the shared contract. The three spatial
/// axes conventionally represent depth, height, and width, and its kernel/stride/padding arrays
/// use that same supplied order, matching [`Conv3d`].
pub struct MaxPool3d(Pooling<3>);

impl MaxPool3d {
    pub fn new(channels: Axis, spatial: [Axis; 3], kernel: [usize; 3]) -> Self {
        Self(Pooling::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Max,
            "MaxPool3d",
        ))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 3]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric padding in the order of the supplied spatial axes. Defaults to `[0, 0, 0]`.
    pub fn padding(mut self, padding: [usize; 3]) -> Self {
        self.0.padding = padding;
        self
    }

    /// [`Module::forward`] plus PyTorch's `return_indices=True` indices, the input to
    /// [`MaxUnpool3d`]: one host-side index per output element, in the output's logical
    /// ([`Tensor::to_vec`]) order, each the row-major offset of the winning input position
    /// within its spatial volume (supplied spatial order, last axis fastest), per slice of
    /// every other axis. The winner is the first maximum in window scan order (last spatial
    /// axis innermost), the same element the gradient routes to. A window with no finite
    /// candidate is an error: there is no position a NaN result could name.
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        self.0.forward_with_indices(input)
    }
}

impl Module for MaxPool3d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Lifts a single named spatial axis through [`Pooling<2>`]'s own machinery via a fresh unit
/// axis: broadcast on (stride zero, extent one) before pooling, removed by [`Tensor::select`]
/// after. There is no dedicated one-spatial-axis unfold kernel, so every 1D fixed-kernel pool
/// reuses the 2D machinery instead of a new one, per this file's rule to extend shared machinery
/// rather than duplicate it. The lifted axis is always padding-free, stride-one, kernel-one, so
/// it can never itself win a reduction or change the real axis's value or gradient.
struct Pooling1d {
    unit: Axis,
    inner: Pooling<2>,
}

impl Pooling1d {
    fn new(
        channels: Axis,
        spatial: Axis,
        kernel: usize,
        reduce: PoolReduce,
        name: &'static str,
    ) -> Self {
        let unit = spatial.role("pool1d_unit");
        Self {
            unit,
            inner: Pooling::new(channels, [spatial, unit], [kernel, 1], reduce, name),
        }
    }

    /// Add the lifted unit axis (extent one, so it changes nothing else) to a real input shape.
    fn lift(&self, input: &Shape) -> Result<Shape> {
        let mut dims = input.dims().to_vec();
        dims.push(self.unit.of(1));
        Shape::new(dims)
    }

    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let shape = self.inner.output_shape(&self.lift(input)?)?;
        Shape::new(
            shape
                .dims()
                .iter()
                .copied()
                .filter(|dim| dim.axis != self.unit),
        )
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let lifted = input.broadcast_to(&self.lift(input.shape())?)?;
        self.inner.forward(&lifted)?.select(self.unit, 0)
    }

    /// The lifted unit axis is last and has extent one, so the inner flat index is already
    /// the real axis's own coordinate.
    fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        let lifted = input.broadcast_to(&self.lift(input.shape())?)?;
        let (output, indices) = self.inner.forward_with_indices(&lifted)?;
        Ok((output.select(self.unit, 0)?, indices))
    }
}

/// Named-channel 1D max pooling; see [`MaxPool2d`] for the shared contract, applied along one
/// spatial axis.
pub struct MaxPool1d(Pooling1d);

impl MaxPool1d {
    pub fn new(channels: Axis, spatial: Axis, kernel: usize) -> Self {
        Self(Pooling1d::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Max,
            "MaxPool1d",
        ))
    }

    /// Set stride. Defaults to the kernel extent.
    pub fn stride(mut self, stride: usize) -> Self {
        self.0.inner.stride[0] = stride;
        self
    }

    /// Set symmetric padding. Defaults to `0`.
    pub fn padding(mut self, padding: usize) -> Self {
        self.0.inner.padding[0] = padding;
        self
    }

    /// [`Module::forward`] plus PyTorch's `return_indices=True` indices; see
    /// [`MaxPool2d::forward_with_indices`]. Each index is the winning coordinate along the
    /// spatial axis, the input to [`MaxUnpool1d`].
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        self.0.forward_with_indices(input)
    }
}

impl Module for MaxPool1d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-channel 1D average pooling: `sum(window) / divisor`, independently per channel,
/// matching PyTorch's `nn.AvgPool1d` and all of its options. The divisor is
/// `divisor_override` when set; otherwise, with `count_include_pad=True` (the default), the
/// window's extent clipped to the padded input -- the full kernel extent except for a
/// `ceil_mode` window overhanging the right padding; with `count_include_pad=False`, the count
/// of real (non-padding) positions. The window count is
/// `floor((L + 2p - k) / s) + 1`, or with `ceil_mode=True` `ceil((L + 2p - k) / s) + 1` minus
/// one when that last window would start at or past `L + p` (PyTorch's own rule: every
/// window starts inside the input or its left padding). Stride defaults to the kernel extent
/// and padding to `0`; padding must be at most half the kernel extent, the same constraint
/// [`MaxPool1d`] enforces.
pub struct AvgPool1d(Pooling1d);

impl AvgPool1d {
    pub fn new(channels: Axis, spatial: Axis, kernel: usize) -> Self {
        Self(Pooling1d::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Avg {
                count_include_pad: true,
                divisor_override: None,
            },
            "AvgPool1d",
        ))
    }

    /// Set stride. Defaults to the kernel extent.
    pub fn stride(mut self, stride: usize) -> Self {
        self.0.inner.stride[0] = stride;
        self
    }

    /// Set symmetric padding. Defaults to `0`.
    pub fn padding(mut self, padding: usize) -> Self {
        self.0.inner.padding[0] = padding;
        self
    }

    /// Round the window count up (PyTorch's `ceil_mode=True`); a last window that would
    /// start in the right padding is still dropped. Defaults to `false`.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.0.inner.ceil_mode = ceil_mode;
        self
    }

    /// Whether padding positions count toward the divisor. Defaults to `true`.
    pub fn count_include_pad(mut self, include: bool) -> Self {
        if let PoolReduce::Avg {
            count_include_pad, ..
        } = &mut self.0.inner.reduce
        {
            *count_include_pad = include;
        }
        self
    }

    /// Divide every window's sum by this constant instead. Must be positive (checked before
    /// launch). Unset by default.
    pub fn divisor_override(mut self, divisor: usize) -> Self {
        if let PoolReduce::Avg {
            divisor_override, ..
        } = &mut self.0.inner.reduce
        {
            *divisor_override = Some(divisor);
        }
        self
    }
}

impl Module for AvgPool1d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-channel 2D average pooling; see [`AvgPool1d`] for the shared `ceil_mode`,
/// `count_include_pad` and `divisor_override` contract, applied per axis over two spatial axes
/// as [`MaxPool2d`] is; a window's divisor is the product of its per-axis counts.
pub struct AvgPool2d(Pooling<2>);

impl AvgPool2d {
    pub fn new(channels: Axis, spatial: [Axis; 2], kernel: [usize; 2]) -> Self {
        Self(Pooling::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Avg {
                count_include_pad: true,
                divisor_override: None,
            },
            "AvgPool2d",
        ))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric padding in the order of the supplied spatial axes. Defaults to `[0, 0]`.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.0.padding = padding;
        self
    }

    /// Round the window count up (PyTorch's `ceil_mode=True`); a last window that would
    /// start in the right padding is still dropped. Defaults to `false`.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.0.ceil_mode = ceil_mode;
        self
    }

    /// Whether padding positions count toward the divisor. Defaults to `true`.
    pub fn count_include_pad(mut self, include: bool) -> Self {
        if let PoolReduce::Avg {
            count_include_pad, ..
        } = &mut self.0.reduce
        {
            *count_include_pad = include;
        }
        self
    }

    /// Divide every window's sum by this constant instead. Must be positive (checked before
    /// launch). Unset by default.
    pub fn divisor_override(mut self, divisor: usize) -> Self {
        if let PoolReduce::Avg {
            divisor_override, ..
        } = &mut self.0.reduce
        {
            *divisor_override = Some(divisor);
        }
        self
    }
}

impl Module for AvgPool2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-channel 3D average pooling; see [`AvgPool1d`] for the shared contract, applied over
/// three spatial axes as [`MaxPool3d`] is.
pub struct AvgPool3d(Pooling<3>);

impl AvgPool3d {
    pub fn new(channels: Axis, spatial: [Axis; 3], kernel: [usize; 3]) -> Self {
        Self(Pooling::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Avg {
                count_include_pad: true,
                divisor_override: None,
            },
            "AvgPool3d",
        ))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 3]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set symmetric padding in the order of the supplied spatial axes. Defaults to `[0, 0, 0]`.
    pub fn padding(mut self, padding: [usize; 3]) -> Self {
        self.0.padding = padding;
        self
    }

    /// Round the window count up (PyTorch's `ceil_mode=True`); a last window that would
    /// start in the right padding is still dropped. Defaults to `false`.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.0.ceil_mode = ceil_mode;
        self
    }

    /// Whether padding positions count toward the divisor. Defaults to `true`.
    pub fn count_include_pad(mut self, include: bool) -> Self {
        if let PoolReduce::Avg {
            count_include_pad, ..
        } = &mut self.0.reduce
        {
            *count_include_pad = include;
        }
        self
    }

    /// Divide every window's sum by this constant instead. Must be positive (checked before
    /// launch). Unset by default.
    pub fn divisor_override(mut self, divisor: usize) -> Self {
        if let PoolReduce::Avg {
            divisor_override, ..
        } = &mut self.0.reduce
        {
            *divisor_override = Some(divisor);
        }
        self
    }
}

impl Module for AvgPool3d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-channel 1D power-average pooling: `(sum(x^p))^(1/p)` over each kernel window,
/// independently per channel, matching PyTorch's `nn.LPPool1d`
/// (`f(X) = (sum_{x in X} x^p)^(1/p)`; at `p = 1` this is exactly sum pooling, and there is no
/// division by the window size). `p` (PyTorch's `norm_type`) is any finite positive real. An
/// integer `p` uses repeated products; a real `p` uses `exp(p ln x)` with `0^p = 0`, so a
/// negative input under a non-integer `p` is NaN, as `x.pow(p)` is in PyTorch. One deliberate
/// difference remains: an odd integer `p` whose window sum is negative returns the real
/// signed root `sign(s) |s|^(1/p)`, where PyTorch's `pow(1/p)` returns NaN. With
/// `ceil_mode=True`, windows are counted as in [`AvgPool1d`] and a window clipped at the input
/// edge is `(sum * k / clipped)^(1/p)`, exactly PyTorch's `avg_pool(x^p) * k` composition.
/// PyTorch's `LPPool` has no `padding` parameter, so none is exposed here either. Stride
/// defaults to the kernel extent, matching every other pooling family in this file.
pub struct LPPool1d(Pooling1d);

impl LPPool1d {
    pub fn new(channels: Axis, spatial: Axis, p: f32, kernel: usize) -> Result<Self> {
        if !(p.is_finite() && p > 0.0) {
            return Err("LPPool1d p must be finite and positive".into());
        }
        Ok(Self(Pooling1d::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Lp(p),
            "LPPool1d",
        )))
    }

    /// Set stride. Defaults to the kernel extent.
    pub fn stride(mut self, stride: usize) -> Self {
        self.0.inner.stride[0] = stride;
        self
    }

    /// Round the window count up (PyTorch's `ceil_mode=True`); see [`LPPool1d`]. Defaults to
    /// `false`.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.0.inner.ceil_mode = ceil_mode;
        self
    }
}

impl Module for LPPool1d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-channel 2D power-average pooling; see [`LPPool1d`] for the shared contract, applied
/// over two spatial axes as [`MaxPool2d`] is.
pub struct LPPool2d(Pooling<2>);

impl LPPool2d {
    pub fn new(channels: Axis, spatial: [Axis; 2], p: f32, kernel: [usize; 2]) -> Result<Self> {
        if !(p.is_finite() && p > 0.0) {
            return Err("LPPool2d p must be finite and positive".into());
        }
        Ok(Self(Pooling::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Lp(p),
            "LPPool2d",
        )))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Round the window count up (PyTorch's `ceil_mode=True`); see [`LPPool1d`]. Defaults to
    /// `false`.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.0.ceil_mode = ceil_mode;
        self
    }
}

impl Module for LPPool2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-channel 3D power-average pooling; see [`LPPool1d`] for the shared contract, applied
/// over three spatial axes as [`MaxPool3d`] is.
pub struct LPPool3d(Pooling<3>);

impl LPPool3d {
    pub fn new(channels: Axis, spatial: [Axis; 3], p: f32, kernel: [usize; 3]) -> Result<Self> {
        if !(p.is_finite() && p > 0.0) {
            return Err("LPPool3d p must be finite and positive".into());
        }
        Ok(Self(Pooling::new(
            channels,
            spatial,
            kernel,
            PoolReduce::Lp(p),
            "LPPool3d",
        )))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 3]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Round the window count up (PyTorch's `ceil_mode=True`); see [`LPPool1d`]. Defaults to
    /// `false`.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.0.ceil_mode = ceil_mode;
        self
    }
}

impl Module for LPPool3d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Shared geometry and dispatch for the adaptive pooling family: validates the spatial/target
/// contract (mirroring [`Pooling::geometry`]) and defers the actual reduction to
/// [`Tensor::adaptive_avg_pool`]/[`Tensor::adaptive_max_pool`], which already implement the bin
/// math generically over `N`.
struct AdaptivePooling<const N: usize> {
    spatial: [Axis; N],
    target: [usize; N],
    name: &'static str,
    max: bool,
}

impl<const N: usize> AdaptivePooling<N> {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let name = self.name;
        for (index, &axis) in self.spatial.iter().enumerate() {
            if self.spatial[..index].contains(&axis) {
                return Err(format!("{name} requires distinct spatial axes").into());
            }
        }
        if self.target.contains(&0) {
            return Err(format!("{name} target extents must be positive").into());
        }
        for &axis in &self.spatial {
            input.extent(axis)?;
        }
        let dims = input.dims().iter().copied().map(|dim| {
            self.spatial
                .iter()
                .position(|&axis| axis == dim.axis)
                .map_or(dim, |index| dim.axis.of(self.target[index]))
        });
        Shape::new(dims)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.max {
            input.adaptive_max_pool(self.spatial, self.target)
        } else {
            input.adaptive_avg_pool(self.spatial, self.target)
        }
    }
}

/// Named-axis adaptive average pooling over one spatial axis: reduces it to a fixed target
/// extent using PyTorch's own per-axis bin formula. See [`Tensor::adaptive_avg_pool3d`] for the
/// exact bin math and its uneven-division behavior.
pub struct AdaptiveAvgPool1d(AdaptivePooling<1>);

impl AdaptiveAvgPool1d {
    pub fn new(spatial: Axis, target: usize) -> Self {
        Self(AdaptivePooling {
            spatial: [spatial],
            target: [target],
            name: "AdaptiveAvgPool1d",
            max: false,
        })
    }
}

impl Module for AdaptiveAvgPool1d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-axis adaptive average pooling over two spatial axes; see [`AdaptiveAvgPool1d`] and
/// [`Tensor::adaptive_avg_pool3d`] for the shared bin contract.
pub struct AdaptiveAvgPool2d(AdaptivePooling<2>);

impl AdaptiveAvgPool2d {
    pub fn new(spatial: [Axis; 2], target: [usize; 2]) -> Self {
        Self(AdaptivePooling {
            spatial,
            target,
            name: "AdaptiveAvgPool2d",
            max: false,
        })
    }
}

impl Module for AdaptiveAvgPool2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-axis adaptive average pooling over three spatial axes, wrapping
/// [`Tensor::adaptive_avg_pool3d`] as a stateless [`Module`]; see [`AdaptiveAvgPool1d`] for the
/// shared contract.
pub struct AdaptiveAvgPool3d(AdaptivePooling<3>);

impl AdaptiveAvgPool3d {
    pub fn new(spatial: [Axis; 3], target: [usize; 3]) -> Self {
        Self(AdaptivePooling {
            spatial,
            target,
            name: "AdaptiveAvgPool3d",
            max: false,
        })
    }
}

impl Module for AdaptiveAvgPool3d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-axis adaptive max pooling over one spatial axis: reduces it to a fixed target extent
/// using the same bin formula as [`AdaptiveAvgPool1d`], taking each bin's maximum instead of its
/// mean. See [`Tensor::adaptive_max_pool3d`] for the exact composition and its tie-break.
pub struct AdaptiveMaxPool1d(AdaptivePooling<1>);

impl AdaptiveMaxPool1d {
    pub fn new(spatial: Axis, target: usize) -> Self {
        Self(AdaptivePooling {
            spatial: [spatial],
            target: [target],
            name: "AdaptiveMaxPool1d",
            max: true,
        })
    }
}

impl Module for AdaptiveMaxPool1d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-axis adaptive max pooling over two spatial axes; see [`AdaptiveMaxPool1d`] and
/// [`Tensor::adaptive_max_pool3d`] for the shared contract.
pub struct AdaptiveMaxPool2d(AdaptivePooling<2>);

impl AdaptiveMaxPool2d {
    pub fn new(spatial: [Axis; 2], target: [usize; 2]) -> Self {
        Self(AdaptivePooling {
            spatial,
            target,
            name: "AdaptiveMaxPool2d",
            max: true,
        })
    }
}

impl Module for AdaptiveMaxPool2d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Named-axis adaptive max pooling over three spatial axes, wrapping
/// [`Tensor::adaptive_max_pool3d`] as a stateless [`Module`]; see [`AdaptiveMaxPool1d`] for the
/// shared contract.
pub struct AdaptiveMaxPool3d(AdaptivePooling<3>);

impl AdaptiveMaxPool3d {
    pub fn new(spatial: [Axis; 3], target: [usize; 3]) -> Self {
        Self(AdaptivePooling {
            spatial,
            target,
            name: "AdaptiveMaxPool3d",
            max: true,
        })
    }
}

impl Module for AdaptiveMaxPool3d {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn build(&mut self, input: &Shape, _device: &Device, _seed: u64) -> Result<Shape> {
        self.0.output_shape(input)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.0.forward(input)
    }
}

/// Shared geometry for the max-unpooling family: PyTorch's `MaxUnpoolNd` default output size
/// `(in - 1) * stride - 2 * padding + kernel` per axis, an explicit size validated against
/// PyTorch's own `default - stride < size < default + stride` window, and dispatch to
/// [`Tensor::max_unpool`].
struct Unpooling<const N: usize> {
    spatial: [Axis; N],
    kernel: [usize; N],
    stride: [usize; N],
    padding: [usize; N],
    name: &'static str,
}

impl<const N: usize> Unpooling<N> {
    fn new(spatial: [Axis; N], kernel: [usize; N], name: &'static str) -> Self {
        Self {
            spatial,
            kernel,
            stride: kernel,
            padding: [0; N],
            name,
        }
    }

    fn sizes(&self, input: &Shape, output_size: Option<[usize; N]>) -> Result<[usize; N]> {
        let name = self.name;
        for (index, &axis) in self.spatial.iter().enumerate() {
            if self.spatial[..index].contains(&axis) {
                return Err(format!("{name} requires distinct spatial axes").into());
            }
        }
        if self.kernel.contains(&0) || self.stride.contains(&0) {
            return Err(format!("{name} kernel and stride extents must be positive").into());
        }
        let mut sizes = [0; N];
        for index in 0..N {
            let extent = input.extent(self.spatial[index])?;
            let default = extent
                .checked_sub(1)
                .and_then(|steps| steps.checked_mul(self.stride[index]))
                .and_then(|span| span.checked_add(self.kernel[index]))
                .and_then(|span| span.checked_sub(2 * self.padding[index]))
                .filter(|&size| size > 0)
                .ok_or_else(|| format!("{name} default output size is not positive"))?;
            sizes[index] = match output_size {
                None => default,
                Some(requested) => {
                    let size = requested[index];
                    if size + self.stride[index] <= default || size >= default + self.stride[index]
                    {
                        return Err(format!(
                            "{name} output size {size} is outside ({}, {}) for axis {index}",
                            default as isize - self.stride[index] as isize,
                            default + self.stride[index]
                        )
                        .into());
                    }
                    size
                }
            };
        }
        Ok(sizes)
    }

    fn output_shape(&self, input: &Shape, output_size: Option<[usize; N]>) -> Result<Shape> {
        let sizes = self.sizes(input, output_size)?;
        Shape::new(input.dims().iter().map(|dim| {
            self.spatial
                .iter()
                .position(|&axis| axis == dim.axis)
                .map_or(*dim, |index| dim.axis.of(sizes[index]))
        }))
    }

    fn forward(
        &self,
        input: &Tensor,
        indices: &[usize],
        output_size: Option<[usize; N]>,
    ) -> Result<Tensor> {
        let sizes = self.sizes(input.shape(), output_size)?;
        input.max_unpool(self.spatial, indices, sizes)
    }
}

/// Named-axis 1D max unpooling, PyTorch's `nn.MaxUnpool1d`: the partial inverse of
/// [`MaxPool1d`], scattering each input value to the position its index names in a zero
/// output and leaving every other position zero. `indices` are
/// [`MaxPool1d::forward_with_indices`]'s, one per input element in its logical order; see
/// [`Tensor::max_unpool3d`] for duplicates and the gradient (a gather of the upstream gradient
/// by the same indices). Stride defaults to the kernel extent and padding to `0`; the output
/// extent defaults to `(in - 1) * stride - 2 * padding + kernel`, and an explicit one (to undo
/// a pool whose floor division dropped a remainder) must lie strictly within `stride` of that
/// default, as PyTorch requires. Not a [`Module`]: like PyTorch's, its forward takes the
/// indices as a second input.
pub struct MaxUnpool1d(Unpooling<1>);

impl MaxUnpool1d {
    pub fn new(spatial: Axis, kernel: usize) -> Self {
        Self(Unpooling::new([spatial], [kernel], "MaxUnpool1d"))
    }

    /// Set stride. Defaults to the kernel extent.
    pub fn stride(mut self, stride: usize) -> Self {
        self.0.stride = [stride];
        self
    }

    /// Set padding. Defaults to `0`.
    pub fn padding(mut self, padding: usize) -> Self {
        self.0.padding = [padding];
        self
    }

    /// The output shape at the default output extent.
    pub fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input, None)
    }

    pub fn forward(&self, input: &Tensor, indices: &[usize]) -> Result<Tensor> {
        self.0.forward(input, indices, None)
    }

    /// [`Self::forward`] at an explicit output extent (PyTorch's `output_size`).
    pub fn forward_sized(&self, input: &Tensor, indices: &[usize], size: usize) -> Result<Tensor> {
        self.0.forward(input, indices, Some([size]))
    }
}

/// Named-axis 2D max unpooling, PyTorch's `nn.MaxUnpool2d`; see [`MaxUnpool1d`] for the
/// shared contract. Its indices are [`MaxPool2d::forward_with_indices`]'s, row-major offsets
/// into the output's spatial plane in the supplied spatial order.
pub struct MaxUnpool2d(Unpooling<2>);

impl MaxUnpool2d {
    pub fn new(spatial: [Axis; 2], kernel: [usize; 2]) -> Self {
        Self(Unpooling::new(spatial, kernel, "MaxUnpool2d"))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 2]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set padding in the order of the supplied spatial axes. Defaults to `[0, 0]`.
    pub fn padding(mut self, padding: [usize; 2]) -> Self {
        self.0.padding = padding;
        self
    }

    /// The output shape at the default output extents.
    pub fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input, None)
    }

    pub fn forward(&self, input: &Tensor, indices: &[usize]) -> Result<Tensor> {
        self.0.forward(input, indices, None)
    }

    /// [`Self::forward`] at explicit output extents (PyTorch's `output_size`).
    pub fn forward_sized(
        &self,
        input: &Tensor,
        indices: &[usize],
        size: [usize; 2],
    ) -> Result<Tensor> {
        self.0.forward(input, indices, Some(size))
    }
}

/// Named-axis 3D max unpooling, PyTorch's `nn.MaxUnpool3d`; see [`MaxUnpool1d`] for the
/// shared contract. Its indices are [`MaxPool3d::forward_with_indices`]'s.
pub struct MaxUnpool3d(Unpooling<3>);

impl MaxUnpool3d {
    pub fn new(spatial: [Axis; 3], kernel: [usize; 3]) -> Self {
        Self(Unpooling::new(spatial, kernel, "MaxUnpool3d"))
    }

    /// Set stride in the order of the supplied spatial axes. Defaults to the kernel extent.
    pub fn stride(mut self, stride: [usize; 3]) -> Self {
        self.0.stride = stride;
        self
    }

    /// Set padding in the order of the supplied spatial axes. Defaults to `[0, 0, 0]`.
    pub fn padding(mut self, padding: [usize; 3]) -> Self {
        self.0.padding = padding;
        self
    }

    /// The output shape at the default output extents.
    pub fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.0.output_shape(input, None)
    }

    pub fn forward(&self, input: &Tensor, indices: &[usize]) -> Result<Tensor> {
        self.0.forward(input, indices, None)
    }

    /// [`Self::forward`] at explicit output extents (PyTorch's `output_size`).
    pub fn forward_sized(
        &self,
        input: &Tensor,
        indices: &[usize],
        size: [usize; 3],
    ) -> Result<Tensor> {
        self.0.forward(input, indices, Some(size))
    }
}
