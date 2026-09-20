use crate::{Axis, Result, Tensor};

/// Exact categorical classification counts. Counts compose across evaluation batches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CategoricalAccuracy {
    correct: usize,
    total: usize,
}

impl CategoricalAccuracy {
    pub fn correct(self) -> usize {
        self.correct
    }

    pub fn total(self) -> usize {
        self.total
    }

    pub fn fraction(self) -> f32 {
        self.correct as f32 / self.total as f32
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            correct: self.correct + other.correct,
            total: self.total + other.total,
        }
    }
}

impl Tensor {
    /// Compare categorical argmaxes on the device and return exact row counts.
    /// `targets` may contain one-hot or probability distributions.
    pub fn categorical_accuracy(
        &self,
        targets: &Tensor,
        class: Axis,
    ) -> Result<CategoricalAccuracy> {
        let flags = self.categorical_correct_flags(targets, class)?;
        let values = flags.to_vec()?;
        let correct = values.iter().filter(|&&value| value == 1.0).count();
        if values.iter().any(|&value| value != 0.0 && value != 1.0) {
            return Err("categorical accuracy kernel returned a nonbinary flag".into());
        }
        Ok(CategoricalAccuracy {
            correct,
            total: values.len(),
        })
    }

    /// Compare categorical argmaxes only where `mask` is one.
    ///
    /// The mask must have the logits' shape with the class axis removed and
    /// contain only zeroes and ones. Returning counts keeps evaluation exactly
    /// composable across batches.
    pub fn masked_categorical_accuracy(
        &self,
        targets: &Tensor,
        mask: &Tensor,
        class: Axis,
    ) -> Result<CategoricalAccuracy> {
        let flags = self.categorical_correct_flags(targets, class)?;
        if flags.shape() != mask.shape() {
            return Err("categorical accuracy mask must match the non-class axes".into());
        }
        let flags = flags.to_vec()?;
        let mask = mask.to_vec()?;
        let mut correct = 0;
        let mut total = 0;
        for (flag, selected) in flags.into_iter().zip(mask) {
            if flag != 0.0 && flag != 1.0 {
                return Err("categorical accuracy kernel returned a nonbinary flag".into());
            }
            if selected != 0.0 && selected != 1.0 {
                return Err("categorical accuracy mask must be binary".into());
            }
            if selected == 1.0 {
                correct += usize::from(flag == 1.0);
                total += 1;
            }
        }
        if total == 0 {
            return Err("categorical accuracy mask must select at least one row".into());
        }
        Ok(CategoricalAccuracy { correct, total })
    }
}
