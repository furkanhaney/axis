//! Empirical monotonicity checks over explicitly ordered input pairs.
use crate::Result;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MonotoneDirection {
    Increasing,
    Decreasing,
}

impl fmt::Display for MonotoneDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Increasing => f.write_str("increasing"),
            Self::Decreasing => f.write_str("decreasing"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MonotonicityLimits {
    absolute_tolerance: f64,
    max_violation_rate: f64,
}

impl MonotonicityLimits {
    /// Require every observed relation to hold within an absolute output tolerance.
    pub fn strict(absolute_tolerance: f64) -> Result<Self> {
        Self::new(absolute_tolerance, 0.0)
    }

    pub fn new(absolute_tolerance: f64, max_violation_rate: f64) -> Result<Self> {
        if !absolute_tolerance.is_finite() || absolute_tolerance < 0.0 {
            return Err("monotonicity tolerance must be finite and nonnegative".into());
        }
        if !max_violation_rate.is_finite() || !(0.0..=1.0).contains(&max_violation_rate) {
            return Err("monotonicity max violation rate must be finite and within 0..=1".into());
        }
        Ok(Self {
            absolute_tolerance,
            max_violation_rate,
        })
    }

    pub fn absolute_tolerance(self) -> f64 {
        self.absolute_tolerance
    }

    pub fn max_violation_rate(self) -> f64 {
        self.max_violation_rate
    }
}

/// A sampled check, never a global proof about the represented function.
pub struct EmpiricalMonotonicity {
    input: &'static str,
    output: &'static str,
    direction: MonotoneDirection,
    limits: MonotonicityLimits,
    comparisons: usize,
    violations: usize,
    worst_violation: f64,
    smallest_input_span: f64,
}

impl EmpiricalMonotonicity {
    pub fn new(
        input: &'static str,
        output: &'static str,
        direction: MonotoneDirection,
        limits: MonotonicityLimits,
    ) -> Result<Self> {
        if input.trim().is_empty() || output.trim().is_empty() {
            return Err("monotonicity input and output names must be nonempty".into());
        }
        Ok(Self {
            input,
            output,
            direction,
            limits,
            comparisons: 0,
            violations: 0,
            worst_violation: 0.0,
            smallest_input_span: f64::INFINITY,
        })
    }

    /// Observe outputs evaluated at a strictly ordered pair of one controlled input.
    /// All other inputs must be held fixed by the caller.
    pub fn observe_ordered(
        &mut self,
        lower_input: f64,
        upper_input: f64,
        lower_output: f64,
        upper_output: f64,
    ) -> Result<EmpiricalMonotonicityReceipt> {
        self.observe_ordered_batch([(lower_input, upper_input, lower_output, upper_output)])
    }

    /// Record a batch before enforcing its declared violation-rate limit.
    pub fn observe_ordered_batch(
        &mut self,
        observations: impl IntoIterator<Item = (f64, f64, f64, f64)>,
    ) -> Result<EmpiricalMonotonicityReceipt> {
        let observations = observations.into_iter().collect::<Vec<_>>();
        if observations.is_empty() {
            return Err("monotonicity observation batch must be nonempty".into());
        }
        for &(lower_input, upper_input, lower_output, upper_output) in &observations {
            if !lower_input.is_finite()
                || !upper_input.is_finite()
                || !lower_output.is_finite()
                || !upper_output.is_finite()
            {
                return Err("monotonicity observations must be finite".into());
            }
            if lower_input >= upper_input {
                return Err(format!(
                    "monotonicity input pair must be strictly ordered; observed {lower_input} >= {upper_input}"
                )
                .into());
            }
        }

        for (lower_input, upper_input, lower_output, upper_output) in observations {
            self.comparisons = self
                .comparisons
                .checked_add(1)
                .ok_or("monotonicity comparison count overflow")?;
            self.smallest_input_span = self.smallest_input_span.min(upper_input - lower_input);
            let violation = match self.direction {
                MonotoneDirection::Increasing => (lower_output - upper_output).max(0.0),
                MonotoneDirection::Decreasing => (upper_output - lower_output).max(0.0),
            };
            self.worst_violation = self.worst_violation.max(violation);
            if violation > self.limits.absolute_tolerance {
                self.violations = self
                    .violations
                    .checked_add(1)
                    .ok_or("monotonicity violation count overflow")?;
            }
        }
        self.assert_invariant()
    }

    pub fn assert_invariant(&self) -> Result<EmpiricalMonotonicityReceipt> {
        if self.comparisons == 0 {
            return Err("empirical monotonicity requires at least one ordered comparison".into());
        }
        let receipt = self.receipt();
        if receipt.violation_rate() > self.limits.max_violation_rate {
            return Err(MonotonicityViolation(receipt).into());
        }
        Ok(receipt)
    }

    pub fn receipt(&self) -> EmpiricalMonotonicityReceipt {
        EmpiricalMonotonicityReceipt {
            input: self.input,
            output: self.output,
            direction: self.direction,
            limits: self.limits,
            comparisons: self.comparisons,
            violations: self.violations,
            worst_violation: self.worst_violation,
            smallest_input_span: self.smallest_input_span,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmpiricalMonotonicityReceipt {
    pub input: &'static str,
    pub output: &'static str,
    pub direction: MonotoneDirection,
    pub limits: MonotonicityLimits,
    pub comparisons: usize,
    pub violations: usize,
    pub worst_violation: f64,
    pub smallest_input_span: f64,
}

impl EmpiricalMonotonicityReceipt {
    pub fn violation_rate(self) -> f64 {
        if self.comparisons == 0 {
            0.0
        } else {
            self.violations as f64 / self.comparisons as f64
        }
    }
}

impl fmt::Display for EmpiricalMonotonicityReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EMPIRICAL MONOTONICITY\n")?;
        writeln!(
            f,
            "relation:               {} -> {}",
            self.input, self.output
        )?;
        writeln!(f, "direction:              {}", self.direction)?;
        writeln!(f, "ordered comparisons:    {}", self.comparisons)?;
        writeln!(f, "observed violations:    {}", self.violations)?;
        writeln!(
            f,
            "violation rate:         {:.4}%",
            self.violation_rate() * 100.0
        )?;
        writeln!(
            f,
            "max violation rate:     {:.4}%",
            self.limits.max_violation_rate * 100.0
        )?;
        writeln!(f, "worst output violation: {:.8}", self.worst_violation)?;
        writeln!(
            f,
            "absolute tolerance:     {:.8}",
            self.limits.absolute_tolerance
        )?;
        writeln!(f, "smallest input span:    {:.8}", self.smallest_input_span)?;
        write!(
            f,
            "evidence:                sampled ordered pairs; not a global guarantee\n\nPASS"
        )
    }
}

#[derive(Debug)]
struct MonotonicityViolation(EmpiricalMonotonicityReceipt);

impl fmt::Display for MonotonicityViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let receipt = self.0;
        write!(
            f,
            "empirical monotonicity violation\n\nrelation:               {} -> {}\ndirection:              {}\nordered comparisons:    {}\nobserved violations:    {}\nviolation rate:         {:.4}%\nmax violation rate:     {:.4}%\nworst output violation: {:.8}\nabsolute tolerance:     {:.8}\n\nthe sampled relation left its declared monotonicity limits",
            receipt.input,
            receipt.output,
            receipt.direction,
            receipt.comparisons,
            receipt.violations,
            receipt.violation_rate() * 100.0,
            receipt.limits.max_violation_rate * 100.0,
            receipt.worst_violation,
            receipt.limits.absolute_tolerance,
        )
    }
}

impl std::error::Error for MonotonicityViolation {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increasing_and_decreasing_checks_emit_scoped_receipts() -> Result<()> {
        let mut increasing = EmpiricalMonotonicity::new(
            "income",
            "score",
            MonotoneDirection::Increasing,
            MonotonicityLimits::strict(1e-9)?,
        )?;
        let receipt =
            increasing.observe_ordered_batch([(1.0, 2.0, 4.0, 4.5), (2.0, 3.0, 4.5, 4.5)])?;
        assert_eq!(receipt.comparisons, 2);
        assert_eq!(receipt.violations, 0);
        assert_eq!(receipt.smallest_input_span, 1.0);
        assert!(receipt.to_string().contains("not a global guarantee"));

        let mut decreasing = EmpiricalMonotonicity::new(
            "price",
            "demand",
            MonotoneDirection::Decreasing,
            MonotonicityLimits::strict(0.0)?,
        )?;
        assert_eq!(
            decreasing.observe_ordered(10.0, 11.0, 5.0, 4.0)?.direction,
            MonotoneDirection::Decreasing
        );
        Ok(())
    }

    #[test]
    fn violations_are_recorded_before_the_limit_fails() -> Result<()> {
        let mut check = EmpiricalMonotonicity::new(
            "quality",
            "score",
            MonotoneDirection::Increasing,
            MonotonicityLimits::new(0.05, 0.25)?,
        )?;
        let error = check
            .observe_ordered_batch([
                (0.0, 1.0, 2.0, 2.2),
                (1.0, 2.0, 2.2, 2.1),
                (2.0, 3.0, 2.1, 2.3),
            ])
            .unwrap_err()
            .to_string();
        assert!(error.contains("observed violations:    1"), "{error}");
        assert!(
            error.contains("violation rate:         33.3333%"),
            "{error}"
        );
        assert!(
            error.contains("worst output violation: 0.10000000"),
            "{error}"
        );
        assert_eq!(check.receipt().comparisons, 3);
        Ok(())
    }

    #[test]
    fn declarations_and_observations_reject_ambiguous_evidence() -> Result<()> {
        assert!(MonotonicityLimits::strict(-1.0).is_err());
        assert!(MonotonicityLimits::new(0.0, 1.1).is_err());
        assert!(
            EmpiricalMonotonicity::new(
                "",
                "output",
                MonotoneDirection::Increasing,
                MonotonicityLimits::strict(0.0)?,
            )
            .is_err()
        );

        let mut check = EmpiricalMonotonicity::new(
            "x",
            "y",
            MonotoneDirection::Increasing,
            MonotonicityLimits::strict(0.0)?,
        )?;
        assert!(check.assert_invariant().is_err());
        assert!(check.observe_ordered(1.0, 1.0, 0.0, 0.0).is_err());
        assert!(check.observe_ordered(0.0, 1.0, f64::NAN, 0.0).is_err());
        assert_eq!(check.receipt().comparisons, 0);
        Ok(())
    }
}
