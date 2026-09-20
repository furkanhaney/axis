//! The first attention consumer. Keep composition here until another program needs it.
use axis::prelude::*;

pub struct AttentionAxes {
    pub time: Axis,
    pub feature: Axis,
    pub head: Dim,
    pub head_feature: Dim,
    query_time: Axis,
    key_time: Axis,
}

impl AttentionAxes {
    pub fn new(time: Axis, feature: Axis, head: Dim, head_feature: Dim) -> Self {
        Self {
            time,
            feature,
            head,
            head_feature,
            query_time: time.role("query_time"),
            key_time: time.role("key_time"),
        }
    }

    /// Q/K/V carry the same logical axes and extents, in any order or physical layout.
    pub fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        for other in [k, v] {
            if q.shape().rank() != other.shape().rank()
                || q.shape()
                    .dims()
                    .iter()
                    .any(|d| other.extent(d.axis).ok() != Some(d.extent))
            {
                return Err(
                    "self-attention Q/K/V must have identical axis sets and extents".into(),
                );
            }
        }
        let split = |x: &Tensor, role| {
            x.split(self.feature, [self.head, self.head_feature])?
                .rename(self.time, role)
        };
        let q = split(q, self.query_time)?;
        let k = split(k, self.key_time)?;
        let v = split(v, self.key_time)?;
        let probabilities = q
            .contract(&k, self.head_feature.axis)?
            .scale(1.0 / (self.head_feature.extent as f32).sqrt())?
            .causal_mask(self.query_time, self.key_time)?
            .softmax(self.key_time)?;
        probabilities
            .contract(&v, self.key_time)?
            .merge([self.head.axis, self.head_feature.axis], self.feature)?
            .rename(self.query_time, self.time)
    }
}

pub struct CausalAttention {
    axes: AttentionAxes,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

impl CausalAttention {
    pub fn new(axes: AttentionAxes) -> Result<Self> {
        let extent = axes
            .head
            .extent
            .checked_mul(axes.head_feature.extent)
            .ok_or("head size overflow")?;
        Shape::new([
            axes.time.of(1),
            axes.feature.of(extent),
            axes.head,
            axes.head_feature,
        ])?;
        let projection = || Linear::new(axes.feature, axes.feature.of(extent));
        Ok(Self {
            query: projection(),
            key: projection(),
            value: projection(),
            output: projection(),
            axes,
        })
    }
}

impl Module for CausalAttention {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.axes.time)?;
        for dim in [self.axes.head, self.axes.head_feature] {
            if input.axes().contains(&dim.axis) {
                return Err("input already contains a split head axis".into());
            }
        }
        self.output.output_shape(&self.query.output_shape(input)?)
    }
    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        let projected = self.query.build(input, device, seed)?;
        self.key.build(input, device, seed.wrapping_add(1))?;
        self.value.build(input, device, seed.wrapping_add(2))?;
        self.output
            .build(&projected, device, seed.wrapping_add(3))?;
        Ok(output)
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let q = self.query.forward(input)?;
        let k = self.key.forward(input)?;
        let v = self.value.forward(input)?;
        self.output.forward(&self.axes.attend(&q, &k, &v)?)
    }
    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [
            ("query", &self.query),
            ("key", &self.key),
            ("value", &self.value),
            ("output", &self.output),
        ]
        .into_iter()
        .flat_map(|(prefix, layer)| {
            layer
                .named_parameters()
                .into_iter()
                .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
        })
        .collect()
    }
}
