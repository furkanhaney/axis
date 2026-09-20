use crate::Result;

/// A train-fitted affine standardization transform.
#[derive(Clone, Copy, Debug)]
pub struct Standardizer {
    mean: f64,
    standard_deviation: f64,
    correction: usize,
}

impl Standardizer {
    /// Fit with Bessel's correction (`N - 1`), matching `torch.std()`'s default.
    pub fn fit(training: &[f32]) -> Result<Self> {
        Self::fit_with_correction(training, 1)
    }

    pub fn fit_with_correction(training: &[f32], correction: usize) -> Result<Self> {
        if training.len() <= correction {
            return Err("training values are too few for the variance correction".into());
        }
        if training.iter().any(|value| !value.is_finite()) {
            return Err("training values must be finite".into());
        }
        let mean = training.iter().copied().map(f64::from).sum::<f64>() / training.len() as f64;
        let variance = training
            .iter()
            .map(|&value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            / (training.len() - correction) as f64;
        let standard_deviation = variance.sqrt();
        if !standard_deviation.is_finite() || standard_deviation == 0.0 {
            return Err("training normalization has invalid standard deviation".into());
        }
        Ok(Self {
            mean,
            standard_deviation,
            correction,
        })
    }

    pub fn transform_in_place(&self, values: &mut [f32]) -> Result<()> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err("values to standardize must be finite".into());
        }
        for value in values {
            *value = ((f64::from(*value) - self.mean) / self.standard_deviation) as f32;
        }
        Ok(())
    }

    pub fn mean(self) -> f64 {
        self.mean
    }

    pub fn standard_deviation(self) -> f64 {
        self.standard_deviation
    }

    pub fn correction(self) -> usize {
        self.correction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardizer_fits_training_only_and_matches_sample_statistics() -> Result<()> {
        let fit = Standardizer::fit(&[1.0, 2.0, 3.0])?;
        assert_eq!(fit.mean(), 2.0);
        assert_eq!(fit.standard_deviation(), 1.0);
        assert_eq!(fit.correction(), 1);
        let mut held_out = [2.0, 4.0];
        fit.transform_in_place(&mut held_out)?;
        assert_eq!(held_out, [0.0, 2.0]);
        assert!(Standardizer::fit(&[1.0]).is_err());
        assert!(Standardizer::fit(&[1.0, 1.0]).is_err());
        Ok(())
    }
}
