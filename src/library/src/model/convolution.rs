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
