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

struct Pooling<const N: usize> {
    channels: Axis,
    spatial: [Axis; N],
    kernel: [usize; N],
    stride: [usize; N],
    padding: [usize; N],
    patch: Axis,
}

impl<const N: usize> Pooling<N> {
    fn new(channels: Axis, spatial: [Axis; N], kernel: [usize; N]) -> Self {
        Self {
            channels,
            spatial,
            kernel,
            // PyTorch's own `MaxPool2d`/`MaxPool3d` default stride to the kernel extent.
            stride: kernel,
            padding: [0; N],
            patch: channels.role("pool_patch"),
        }
    }

    fn name() -> &'static str {
        match N {
            2 => "MaxPool2d",
            3 => "MaxPool3d",
            _ => "max pooling",
        }
    }

    /// Output shape and the input's channel extent (needed again by `forward`).
    fn geometry(&self, input: &Shape) -> Result<(Shape, usize)> {
        let name = Self::name();
        if !(N == 2 || N == 3) {
            return Err("Axis max pooling supports exactly two or three spatial axes".into());
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
        let channel_extent = input.extent(self.channels)?;
        let mut output_spatial = [0; N];
        // Four parallel small arrays share one index; a `.zip()` chain would read worse than
        // the loop it replaces.
        #[allow(clippy::needless_range_loop)]
        for index in 0..N {
            let doubled_padding = self.padding[index]
                .checked_mul(2)
                .ok_or_else(|| format!("{name} padding overflow"))?;
            if doubled_padding > self.kernel[index] {
                // PyTorch's own MaxPool constraint. It also guarantees every window keeps at
                // least one real, unpadded element, so a fully-padding window is unreachable.
                return Err(
                    format!("{name} padding must be at most half the kernel extent").into(),
                );
            }
            let input_extent = input.extent(self.spatial[index])?;
            let padded = input_extent
                .checked_add(doubled_padding)
                .ok_or_else(|| format!("{name} padded spatial extent overflow"))?;
            if self.kernel[index] > padded {
                return Err(format!("{name} kernel must fit the padded spatial axes").into());
            }
            output_spatial[index] = (padded - self.kernel[index]) / self.stride[index] + 1;
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
                    .map_or(*dim, |index| dim.axis.of(output_spatial[index]))
            })
            .collect();
        dims.push(self.channels.of(channel_extent));
        Ok((Shape::new(dims)?, channel_extent))
    }

    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        Ok(self.geometry(input)?.0)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, channel_extent) = self.geometry(input.shape())?;
        let kernel_volume = self.kernel.iter().product();
        // Materialize the windowed patches (as `Conv2d`/`Conv3d` do), padded with negative
        // infinity so a padded position can never be selected, then take the maximum of each
        // patch as `-min(-patch)`: `Tensor::min` already ignores non-finite candidates and
        // breaks ties toward the first logical coordinate, and `unfold_grouped`'s backward
        // (col2im) already sums overlapping window contributions exactly, so no new backend
        // kernel is needed for either the padding or the stride-less-than-kernel overlap case.
        input
            .unfold_grouped(
                self.channels,
                self.spatial,
                self.channels.of(channel_extent),
                self.patch.of(kernel_volume),
                self.kernel,
                self.stride,
                self.padding,
                f32::NEG_INFINITY,
            )?
            .scale(-1.0)?
            .min(self.patch)?
            .scale(-1.0)
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
        Self(Pooling::new(channels, spatial, kernel))
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
        Self(Pooling::new(channels, spatial, kernel))
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
