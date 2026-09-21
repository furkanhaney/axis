//! Differentiable finite-difference stencils with named coordinate provenance.
use crate::{Axis, Result, Tensor};

/// A centered finite-difference stencil evaluated at a positive coordinate step.
///
/// The coordinate identifies what was shifted to produce the supplied tensors;
/// it does not need to be an axis of their output shape. Callers remain
/// responsible for evaluating the same function and holding every other input
/// fixed at `x - step`, `x`, and `x + step`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentralDifference {
    coordinate: Axis,
    step: f32,
}

impl CentralDifference {
    pub fn new(coordinate: Axis, step: f32) -> Result<Self> {
        if !step.is_finite() || step <= 0.0 {
            return Err("central-difference step must be finite and positive".into());
        }
        Ok(Self { coordinate, step })
    }

    pub fn coordinate(self) -> Axis {
        self.coordinate
    }

    pub fn step(self) -> f32 {
        self.step
    }

    fn validate(&self, tensors: &[&Tensor]) -> Result<()> {
        let first = tensors
            .first()
            .ok_or("central difference requires at least two tensors")?;
        for tensor in &tensors[1..] {
            if tensor.shape() != first.shape() {
                return Err(format!(
                    "central difference over {:?} requires identically ordered shapes; observed {:?} and {:?}",
                    self.coordinate,
                    first.shape(),
                    tensor.shape()
                )
                .into());
            }
            if !tensor.device().same(first.device()) {
                return Err(format!(
                    "central difference over {:?} requires one Device handle",
                    self.coordinate
                )
                .into());
            }
        }
        Ok(())
    }

    /// Approximate the first derivative as `(upper - lower) / (2 * step)`.
    pub fn first(&self, lower: &Tensor, upper: &Tensor) -> Result<Tensor> {
        self.validate(&[lower, upper])?;
        upper.sub(lower)?.scale(0.5 / self.step)
    }

    /// Approximate the second derivative as `(lower - 2*center + upper) / step²`.
    pub fn second(&self, lower: &Tensor, center: &Tensor, upper: &Tensor) -> Result<Tensor> {
        self.validate(&[lower, center, upper])?;
        let inverse_square = 1.0 / (self.step * self.step);
        lower
            .add(upper)?
            .sub(&center.scale(2.0)?)?
            .scale(inverse_square)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Shape};

    fn close(name: &str, actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{name} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(actual) - expected).abs();
            assert!(
                error <= 2e-5,
                "{name}[{index}]: {actual} != {expected} ({error})"
            );
        }
    }

    #[test]
    fn declaration_rejects_ambiguous_steps() {
        let coordinate = Axis::new("time");
        for step in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(CentralDifference::new(coordinate, step).is_err());
        }
        let stencil = CentralDifference::new(coordinate, 0.25).unwrap();
        assert_eq!(stencil.coordinate(), coordinate);
        assert_eq!(stencil.step(), 0.25);
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn first_and_second_stencils_match_scalar_forward_and_gradient_oracles() -> Result<()> {
        let device = Device::cuda(0)?;
        let sample = Axis::new("sample");
        let time = Axis::new("time");
        let lower_values = [0.3_f32, -0.8, 1.2];
        let center_values = [0.5_f32, -0.2, 0.7];
        let upper_values = [1.1_f32, 0.4, -0.3];
        let lower = Tensor::from_slice(&lower_values, [sample.of(3)], &device)?.with_grad();
        let center = Tensor::from_slice(&center_values, [sample.of(3)], &device)?.with_grad();
        let upper = Tensor::from_slice(&upper_values, [sample.of(3)], &device)?.with_grad();
        let stencil = CentralDifference::new(time, 0.2)?;

        let first = stencil.first(&lower, &upper)?;
        let second = stencil.second(&lower, &center, &upper)?;
        let expected_first = lower_values
            .iter()
            .zip(upper_values)
            .map(|(&lower, upper)| (f64::from(upper) - f64::from(lower)) / 0.4)
            .collect::<Vec<_>>();
        let expected_second = lower_values
            .iter()
            .zip(center_values)
            .zip(upper_values)
            .map(|((&lower, center), upper)| {
                (f64::from(lower) - 2.0 * f64::from(center) + f64::from(upper)) / 0.04
            })
            .collect::<Vec<_>>();
        close(
            "central first difference",
            &first.to_vec()?,
            &expected_first,
        );
        close(
            "central second difference",
            &second.to_vec()?,
            &expected_second,
        );
        let combined = first.add(&second)?;
        let expected = expected_first
            .iter()
            .zip(&expected_second)
            .map(|(first, second)| first + second)
            .collect::<Vec<_>>();
        close("central difference", &combined.to_vec()?, &expected);

        combined.mean(sample)?.backward()?;
        let n = 3.0_f64;
        close(
            "lower derivative",
            &lower.grad().unwrap().to_vec()?,
            &[(-1.0 / 0.4 + 1.0 / 0.04) / n; 3],
        );
        close(
            "center derivative",
            &center.grad().unwrap().to_vec()?,
            &[(-2.0 / 0.04) / n; 3],
        );
        close(
            "upper derivative",
            &upper.grad().unwrap().to_vec()?,
            &[(1.0 / 0.4 + 1.0 / 0.04) / n; 3],
        );

        let other = Axis::new("other");
        let wrong = Tensor::from_slice(&[1.0, 2.0, 3.0], [other.of(3)], &device)?;
        assert!(stencil.first(&lower.detach(), &wrong).is_err());
        assert_eq!(first.shape(), &Shape::new([sample.of(3)])?);
        Ok(())
    }
}
