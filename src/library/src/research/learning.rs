//! Empirical progress checks for a declared metric and evaluation population.
use crate::Result;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LearningDirection {
    Increase,
    Decrease,
}

impl fmt::Display for LearningDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Increase => f.write_str("increase"),
            Self::Decrease => f.write_str("decrease"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearningLimits {
    minimum_absolute_improvement: f64,
    minimum_relative_improvement: Option<f64>,
}

impl LearningLimits {
    /// Require any strictly positive improvement in the declared direction.
    pub const fn any_improvement() -> Self {
        Self {
            minimum_absolute_improvement: 0.0,
            minimum_relative_improvement: None,
        }
    }

    pub fn new(
        minimum_absolute_improvement: f64,
        minimum_relative_improvement: Option<f64>,
    ) -> Result<Self> {
        if !minimum_absolute_improvement.is_finite() || minimum_absolute_improvement < 0.0 {
            return Err("minimum absolute improvement must be finite and nonnegative".into());
        }
        if minimum_relative_improvement.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err("minimum relative improvement must be finite and nonnegative".into());
        }
        Ok(Self {
            minimum_absolute_improvement,
            minimum_relative_improvement,
        })
    }

    pub fn minimum_absolute_improvement(self) -> f64 {
        self.minimum_absolute_improvement
    }

    pub fn minimum_relative_improvement(self) -> Option<f64> {
        self.minimum_relative_improvement
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearningObservation {
    metric: String,
    population: String,
    budget_unit: String,
    budget: u64,
    value: f64,
}

impl LearningObservation {
    pub fn new(
        metric: impl Into<String>,
        population: impl Into<String>,
        budget_unit: impl Into<String>,
        budget: u64,
        value: f64,
    ) -> Result<Self> {
        if !value.is_finite() {
            return Err("learning-progress metric value must be finite".into());
        }
        Ok(Self {
            metric: label("learning-progress metric", metric.into())?,
            population: label("learning-progress population", population.into())?,
            budget_unit: label("learning-progress budget unit", budget_unit.into())?,
            budget,
            value,
        })
    }

    pub fn metric(&self) -> &str {
        &self.metric
    }

    pub fn population(&self) -> &str {
        &self.population
    }

    pub fn budget_unit(&self) -> &str {
        &self.budget_unit
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

fn label(kind: &str, value: String) -> Result<String> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(
            format!("{kind} must be nonempty, trimmed, and contain no control characters").into(),
        );
    }
    Ok(value)
}

/// The strength of the claim made by a learning-progress receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LearningEvidence {
    /// The caller-supplied observations met their declared progress limits. This
    /// does not prove the population label, generalization, or convergence.
    VerifiedObservations,
}

impl fmt::Display for LearningEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerifiedObservations => f.write_str("verified observations"),
        }
    }
}

/// Compares one declared metric on one declared population over an ordered budget.
pub struct LearningProgress {
    direction: LearningDirection,
    limits: LearningLimits,
    baseline: LearningObservation,
    terminal: LearningObservation,
    observations: usize,
}

impl LearningProgress {
    pub fn new(
        direction: LearningDirection,
        limits: LearningLimits,
        baseline: LearningObservation,
    ) -> Result<Self> {
        if limits.minimum_relative_improvement.is_some() && baseline.value == 0.0 {
            return Err(
                "relative learning progress is undefined for a zero-valued baseline".into(),
            );
        }
        Ok(Self {
            direction,
            limits,
            terminal: baseline.clone(),
            baseline,
            observations: 1,
        })
    }

    /// Record a compatible observation after validating it completely.
    /// A rejected observation leaves the previous terminal observation intact.
    pub fn observe(&mut self, observation: LearningObservation) -> Result<()> {
        if observation.metric != self.baseline.metric {
            return Err(format!(
                "learning-progress metric mismatch: expected {:?}, observed {:?}",
                self.baseline.metric, observation.metric
            )
            .into());
        }
        if observation.population != self.baseline.population {
            return Err(format!(
                "learning-progress population mismatch: expected {:?}, observed {:?}",
                self.baseline.population, observation.population
            )
            .into());
        }
        if observation.budget_unit != self.baseline.budget_unit {
            return Err(format!(
                "learning-progress budget-unit mismatch: expected {:?}, observed {:?}",
                self.baseline.budget_unit, observation.budget_unit
            )
            .into());
        }
        if observation.budget <= self.terminal.budget {
            return Err(format!(
                "learning-progress budget must increase strictly: previous {} {}, observed {} {}",
                self.terminal.budget,
                self.baseline.budget_unit,
                observation.budget,
                self.baseline.budget_unit,
            )
            .into());
        }
        let next = self
            .observations
            .checked_add(1)
            .ok_or("learning-progress observation count overflow")?;
        self.terminal = observation;
        self.observations = next;
        Ok(())
    }

    pub fn assert_learning(&self) -> Result<LearningProgressReceipt> {
        if self.observations < 2 {
            return Err("learning progress requires a baseline and a later observation".into());
        }
        let receipt = self.receipt();
        if !receipt.passed() {
            return Err(LearningProgressViolation(receipt).into());
        }
        Ok(receipt)
    }

    pub fn receipt(&self) -> LearningProgressReceipt {
        let absolute_improvement = match self.direction {
            LearningDirection::Increase => self.terminal.value - self.baseline.value,
            LearningDirection::Decrease => self.baseline.value - self.terminal.value,
        };
        LearningProgressReceipt {
            metric: self.baseline.metric.clone(),
            population: self.baseline.population.clone(),
            budget_unit: self.baseline.budget_unit.clone(),
            direction: self.direction,
            limits: self.limits,
            observations: self.observations,
            baseline_budget: self.baseline.budget,
            baseline_value: self.baseline.value,
            terminal_budget: self.terminal.budget,
            terminal_value: self.terminal.value,
            absolute_improvement,
            relative_improvement: (self.baseline.value != 0.0)
                .then_some(absolute_improvement / self.baseline.value.abs()),
            evidence: LearningEvidence::VerifiedObservations,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearningProgressReceipt {
    pub metric: String,
    pub population: String,
    pub budget_unit: String,
    pub direction: LearningDirection,
    pub limits: LearningLimits,
    pub observations: usize,
    pub baseline_budget: u64,
    pub baseline_value: f64,
    pub terminal_budget: u64,
    pub terminal_value: f64,
    pub absolute_improvement: f64,
    pub relative_improvement: Option<f64>,
    pub evidence: LearningEvidence,
}

impl LearningProgressReceipt {
    pub fn budget_delta(&self) -> u64 {
        self.terminal_budget - self.baseline_budget
    }

    pub fn passed(&self) -> bool {
        self.observations >= 2
            && self.absolute_improvement > 0.0
            && meets_threshold(
                self.absolute_improvement,
                self.limits.minimum_absolute_improvement,
            )
            && self
                .limits
                .minimum_relative_improvement
                .is_none_or(|minimum| {
                    self.relative_improvement
                        .is_some_and(|observed| meets_threshold(observed, minimum))
                })
    }
}

// A difference and an optional division separate an observation from its
// declared threshold. Admit only their immediate representation noise: there
// is deliberately no absolute epsilon that could erase a tiny threshold.
const MAX_THRESHOLD_ULPS: u64 = 4;

fn meets_threshold(observed: f64, minimum: f64) -> bool {
    observed.is_finite()
        && minimum.is_finite()
        && (observed >= minimum
            || (observed >= 0.0
                && minimum.to_bits().saturating_sub(observed.to_bits()) <= MAX_THRESHOLD_ULPS))
}

impl fmt::Display for LearningProgressReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EMPIRICAL LEARNING PROGRESS\n")?;
        writeln!(f, "metric:                       {}", self.metric)?;
        writeln!(f, "evaluation population:        {}", self.population)?;
        writeln!(f, "direction:                    {}", self.direction)?;
        writeln!(f, "observations:                 {}", self.observations)?;
        writeln!(
            f,
            "baseline:                     {:.8} at {} {}",
            self.baseline_value, self.baseline_budget, self.budget_unit
        )?;
        writeln!(
            f,
            "terminal:                     {:.8} at {} {}",
            self.terminal_value, self.terminal_budget, self.budget_unit
        )?;
        writeln!(
            f,
            "absolute improvement:         {:.8}",
            self.absolute_improvement
        )?;
        match self.relative_improvement {
            Some(value) => writeln!(f, "relative improvement:         {:.4}%", value * 100.0)?,
            None => writeln!(f, "relative improvement:         n/a (zero baseline)")?,
        }
        writeln!(
            f,
            "minimum absolute improvement: {:.8}",
            self.limits.minimum_absolute_improvement
        )?;
        match self.limits.minimum_relative_improvement {
            Some(value) => writeln!(f, "minimum relative improvement: {:.4}%", value * 100.0)?,
            None => writeln!(f, "minimum relative improvement: none")?,
        }
        writeln!(
            f,
            "budget delta:                 {} {}",
            self.budget_delta(),
            self.budget_unit
        )?;
        writeln!(f, "evidence:                     {}", self.evidence)?;
        write!(
            f,
            "scope:                        supplied evaluations; not generalization or convergence\n\n{}",
            if self.passed() { "PASS" } else { "FAIL" }
        )
    }
}

#[derive(Debug)]
struct LearningProgressViolation(LearningProgressReceipt);

impl fmt::Display for LearningProgressViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "empirical learning-progress violation\n\n{}\n\nthe supplied observations did not meet their declared improvement limits",
            self.0
        )
    }
}

impl std::error::Error for LearningProgressViolation {}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        metric: &str,
        population: &str,
        unit: &str,
        budget: u64,
        value: f64,
    ) -> Result<LearningObservation> {
        LearningObservation::new(metric, population, unit, budget, value)
    }

    #[test]
    fn decreasing_loss_and_increasing_accuracy_pass_at_exact_limits() -> Result<()> {
        let limits = LearningLimits::new(0.5, Some(0.25))?;
        let mut loss = LearningProgress::new(
            LearningDirection::Decrease,
            limits,
            observation("MSE", "held-out-v1", "optimizer steps", 0, 2.0)?,
        )?;
        loss.observe(observation(
            "MSE",
            "held-out-v1",
            "optimizer steps",
            100,
            1.5,
        )?)?;
        let loss = loss.assert_learning()?;
        assert_eq!(loss.absolute_improvement, 0.5);
        assert_eq!(loss.relative_improvement, Some(0.25));
        assert_eq!(loss.budget_delta(), 100);
        assert_eq!(loss.evidence, LearningEvidence::VerifiedObservations);

        let mut accuracy = LearningProgress::new(
            LearningDirection::Increase,
            LearningLimits::new(0.25, Some(0.5))?,
            observation("accuracy", "test-v1", "samples", 1_000, 0.5)?,
        )?;
        accuracy.observe(observation("accuracy", "test-v1", "samples", 2_000, 0.75)?)?;
        assert!(accuracy.assert_learning()?.passed());

        let mut decimal_absolute = LearningProgress::new(
            LearningDirection::Decrease,
            LearningLimits::new(0.1, None)?,
            observation("loss", "decimal-absolute", "steps", 0, 1.0)?,
        )?;
        decimal_absolute.observe(observation("loss", "decimal-absolute", "steps", 1, 0.9)?)?;
        assert_eq!(
            decimal_absolute.receipt().absolute_improvement,
            0.09999999999999998
        );
        assert!(decimal_absolute.assert_learning()?.passed());

        let mut decimal_relative = LearningProgress::new(
            LearningDirection::Decrease,
            LearningLimits::new(0.0, Some(0.1))?,
            observation("loss", "decimal-relative", "steps", 0, 2.0)?,
        )?;
        decimal_relative.observe(observation("loss", "decimal-relative", "steps", 1, 1.8)?)?;
        assert_eq!(
            decimal_relative.receipt().relative_improvement,
            Some(0.09999999999999998)
        );
        assert!(decimal_relative.assert_learning()?.passed());
        Ok(())
    }

    #[test]
    fn threshold_slack_is_ulps_not_an_absolute_floor() {
        let threshold = 0.1_f64;
        assert!(meets_threshold(
            f64::from_bits(threshold.to_bits() - MAX_THRESHOLD_ULPS),
            threshold
        ));
        assert!(!meets_threshold(
            f64::from_bits(threshold.to_bits() - MAX_THRESHOLD_ULPS - 1),
            threshold
        ));
        assert!(!meets_threshold(5e-301, 1e-300));
        assert!(!meets_threshold(f64::INFINITY, 1.0));
    }

    #[test]
    fn direction_and_threshold_failures_report_the_full_receipt() -> Result<()> {
        let mut wrong_direction = LearningProgress::new(
            LearningDirection::Increase,
            LearningLimits::any_improvement(),
            observation("accuracy", "audit", "steps", 0, 0.8)?,
        )?;
        wrong_direction.observe(observation("accuracy", "audit", "steps", 10, 0.7)?)?;
        let error = wrong_direction.assert_learning().unwrap_err().to_string();
        assert!(
            error.contains("absolute improvement:         -0.10000000"),
            "{error}"
        );
        assert!(
            error.contains("evidence:                     verified observations"),
            "{error}"
        );
        assert!(error.contains("FAIL"), "{error}");

        let mut below_absolute = LearningProgress::new(
            LearningDirection::Decrease,
            LearningLimits::new(0.5, None)?,
            observation("loss", "validation", "tokens", 0, 2.0)?,
        )?;
        below_absolute.observe(observation("loss", "validation", "tokens", 10, 1.6)?)?;
        let receipt = below_absolute.receipt();
        assert!(!receipt.passed());
        assert!(below_absolute.assert_learning().is_err());

        let mut below_relative = LearningProgress::new(
            LearningDirection::Decrease,
            LearningLimits::new(0.5, Some(0.2))?,
            observation("loss", "validation", "tokens", 0, 4.0)?,
        )?;
        below_relative.observe(observation("loss", "validation", "tokens", 10, 3.5)?)?;
        assert_eq!(below_relative.receipt().absolute_improvement, 0.5);
        assert_eq!(below_relative.receipt().relative_improvement, Some(0.125));
        assert!(below_relative.assert_learning().is_err());
        Ok(())
    }

    #[test]
    fn incompatible_and_nonfinite_observations_fail_atomically() -> Result<()> {
        let mut progress = LearningProgress::new(
            LearningDirection::Decrease,
            LearningLimits::any_improvement(),
            observation("loss", "validation-v1", "steps", 0, 4.0)?,
        )?;

        assert!(
            progress
                .observe(observation("accuracy", "validation-v1", "steps", 1, 3.0)?)
                .is_err()
        );
        assert!(
            progress
                .observe(observation("loss", "training-v1", "steps", 1, 3.0)?)
                .is_err()
        );
        assert!(
            progress
                .observe(observation("loss", "validation-v1", "samples", 1, 3.0)?)
                .is_err()
        );
        assert!(LearningObservation::new("loss", "validation-v1", "steps", 1, f64::NAN).is_err());
        assert!(
            LearningObservation::new("loss", "validation-v1", "steps", 1, f64::INFINITY,).is_err()
        );
        assert_eq!(progress.receipt().observations, 1);
        assert_eq!(progress.receipt().terminal_budget, 0);

        progress.observe(observation("loss", "validation-v1", "steps", 2, 3.0)?)?;
        let before = progress.receipt();
        assert!(
            progress
                .observe(observation("loss", "validation-v1", "steps", 2, 2.0)?)
                .is_err()
        );
        assert!(
            progress
                .observe(observation("loss", "validation-v1", "steps", 1, 2.0)?)
                .is_err()
        );
        assert_eq!(progress.receipt(), before);
        assert!(progress.assert_learning()?.passed());
        Ok(())
    }

    #[test]
    fn declarations_reject_ambiguous_evidence() -> Result<()> {
        assert!(LearningLimits::new(-0.1, None).is_err());
        assert!(LearningLimits::new(f64::NAN, None).is_err());
        assert!(LearningLimits::new(0.0, Some(-0.1)).is_err());
        assert!(LearningLimits::new(0.0, Some(f64::INFINITY)).is_err());
        assert!(LearningObservation::new("", "population", "steps", 0, 1.0).is_err());
        assert!(LearningObservation::new("loss", " population", "steps", 0, 1.0).is_err());
        assert!(LearningObservation::new("loss", "population", "step\n", 0, 1.0).is_err());
        assert!(
            LearningProgress::new(
                LearningDirection::Decrease,
                LearningLimits::new(0.0, Some(0.1))?,
                observation("loss", "validation", "steps", 0, 0.0)?,
            )
            .is_err()
        );

        let no_relative = LearningLimits::any_improvement();
        assert_eq!(no_relative.minimum_absolute_improvement(), 0.0);
        assert_eq!(no_relative.minimum_relative_improvement(), None);
        let progress = LearningProgress::new(
            LearningDirection::Increase,
            no_relative,
            observation("score", "audit", "samples", 0, 0.0)?,
        )?;
        assert!(progress.assert_learning().is_err());
        Ok(())
    }

    #[test]
    fn receipt_names_scope_budget_thresholds_and_claim_level() -> Result<()> {
        let mut progress = LearningProgress::new(
            LearningDirection::Decrease,
            LearningLimits::new(0.25, Some(0.1))?,
            observation("cross entropy", "fixed audit split", "tokens", 50, 2.0)?,
        )?;
        progress.observe(observation(
            "cross entropy",
            "fixed audit split",
            "tokens",
            1_050,
            1.5,
        )?)?;
        let text = progress.assert_learning()?.to_string();
        for expected in [
            "EMPIRICAL LEARNING PROGRESS",
            "metric:                       cross entropy",
            "evaluation population:        fixed audit split",
            "direction:                    decrease",
            "baseline:                     2.00000000 at 50 tokens",
            "terminal:                     1.50000000 at 1050 tokens",
            "absolute improvement:         0.50000000",
            "relative improvement:         25.0000%",
            "minimum absolute improvement: 0.25000000",
            "minimum relative improvement: 10.0000%",
            "budget delta:                 1000 tokens",
            "evidence:                     verified observations",
            "not generalization or convergence",
            "PASS",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in {text}");
        }
        Ok(())
    }
}
