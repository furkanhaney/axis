//! Empirical checks for caller-evaluated scientific residuals.
use crate::Result;
use std::fmt;

fn label(kind: &str, value: String) -> Result<String> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(
            format!("{kind} must be nonempty, trimmed, and contain no control characters").into(),
        );
    }
    Ok(value)
}

/// Stable identity for the scientific law whose residual is being evaluated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LawIdentity {
    name: String,
    version: String,
}

impl LawIdentity {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Result<Self> {
        Ok(Self {
            name: label("law name", name.into())?,
            version: label("law version", version.into())?,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

/// Declares what residual was evaluated, where, and by which numerical method.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidualScope {
    law: LawIdentity,
    expression: String,
    region: String,
    evaluator: String,
}

impl ResidualScope {
    pub fn new(
        law: LawIdentity,
        expression: impl Into<String>,
        region: impl Into<String>,
        evaluator: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            law,
            expression: label("residual expression", expression.into())?,
            region: label("residual region", region.into())?,
            evaluator: label("residual evaluator", evaluator.into())?,
        })
    }

    pub fn law(&self) -> &LawIdentity {
        &self.law
    }

    pub fn expression(&self) -> &str {
        &self.expression
    }

    pub fn region(&self) -> &str {
        &self.region
    }

    pub fn evaluator(&self) -> &str {
        &self.evaluator
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualLimits {
    absolute_tolerance: f64,
    max_violation_rate: f64,
}

impl ResidualLimits {
    pub fn strict(absolute_tolerance: f64) -> Result<Self> {
        Self::new(absolute_tolerance, 0.0)
    }

    pub fn new(absolute_tolerance: f64, max_violation_rate: f64) -> Result<Self> {
        if !absolute_tolerance.is_finite() || absolute_tolerance < 0.0 {
            return Err("residual absolute tolerance must be finite and nonnegative".into());
        }
        if !max_violation_rate.is_finite() || !(0.0..=1.0).contains(&max_violation_rate) {
            return Err("residual max violation rate must be finite and within 0..=1".into());
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

/// The claim made by this receipt. A guaranteed law requires a separate,
/// construction-backed API and cannot be selected by a caller here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidualClaim {
    Empirical,
}

impl fmt::Display for ResidualClaim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empirical => f.write_str("empirical"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidualEvidence {
    VerifiedObservations,
}

impl fmt::Display for ResidualEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerifiedObservations => f.write_str("verified sampled discretized observations"),
        }
    }
}

/// A sampled residual check, never a proof that the represented law holds.
pub struct EmpiricalResidual {
    scope: ResidualScope,
    limits: ResidualLimits,
    observations: usize,
    violations: usize,
    maximum_absolute_residual: f64,
    rms_scale: f64,
    rms_scaled_squares: f64,
}

impl EmpiricalResidual {
    pub fn new(scope: ResidualScope, limits: ResidualLimits) -> Self {
        Self {
            scope,
            limits,
            observations: 0,
            violations: 0,
            maximum_absolute_residual: 0.0,
            rms_scale: 0.0,
            rms_scaled_squares: 0.0,
        }
    }

    pub fn observe(&mut self, residual: f64) -> Result<EmpiricalResidualReceipt> {
        self.observe_batch([residual])
    }

    /// Validate an entire batch before recording any of it. A limit violation is
    /// still recorded so the failing receipt describes the evidence observed.
    pub fn observe_batch(
        &mut self,
        residuals: impl IntoIterator<Item = f64>,
    ) -> Result<EmpiricalResidualReceipt> {
        let residuals = residuals.into_iter().collect::<Vec<_>>();
        if residuals.is_empty() {
            return Err("residual observation batch must be nonempty".into());
        }
        if residuals.iter().any(|value| !value.is_finite()) {
            return Err("residual observations must be finite".into());
        }
        let observations = self
            .observations
            .checked_add(residuals.len())
            .ok_or("residual observation count overflow")?;
        let additional_violations = residuals
            .iter()
            .filter(|value| value.abs() > self.limits.absolute_tolerance)
            .count();
        let violations = self
            .violations
            .checked_add(additional_violations)
            .ok_or("residual violation count overflow")?;

        let mut scale = self.rms_scale;
        let mut scaled_squares = self.rms_scaled_squares;
        let mut maximum = self.maximum_absolute_residual;
        for residual in residuals {
            let magnitude = residual.abs();
            maximum = maximum.max(magnitude);
            if magnitude == 0.0 {
                continue;
            }
            if scale < magnitude {
                let ratio = scale / magnitude;
                scaled_squares = 1.0 + scaled_squares * ratio * ratio;
                scale = magnitude;
            } else {
                let ratio = magnitude / scale;
                scaled_squares += ratio * ratio;
            }
        }

        self.observations = observations;
        self.violations = violations;
        self.maximum_absolute_residual = maximum;
        self.rms_scale = scale;
        self.rms_scaled_squares = scaled_squares;
        self.assert_within_limits()
    }

    pub fn assert_within_limits(&self) -> Result<EmpiricalResidualReceipt> {
        if self.observations == 0 {
            return Err("empirical residual requires at least one observation".into());
        }
        let receipt = self.receipt();
        if receipt.violation_rate() > self.limits.max_violation_rate {
            return Err(ResidualViolation(receipt).into());
        }
        Ok(receipt)
    }

    pub fn receipt(&self) -> EmpiricalResidualReceipt {
        let rms_residual = if self.observations == 0 || self.rms_scale == 0.0 {
            0.0
        } else {
            self.rms_scale * (self.rms_scaled_squares / self.observations as f64).sqrt()
        };
        EmpiricalResidualReceipt {
            scope: self.scope.clone(),
            limits: self.limits,
            observations: self.observations,
            violations: self.violations,
            maximum_absolute_residual: self.maximum_absolute_residual,
            rms_residual,
            claim: ResidualClaim::Empirical,
            evidence: ResidualEvidence::VerifiedObservations,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalResidualReceipt {
    pub scope: ResidualScope,
    pub limits: ResidualLimits,
    pub observations: usize,
    pub violations: usize,
    pub maximum_absolute_residual: f64,
    pub rms_residual: f64,
    pub claim: ResidualClaim,
    pub evidence: ResidualEvidence,
}

impl EmpiricalResidualReceipt {
    pub fn violation_rate(&self) -> f64 {
        if self.observations == 0 {
            0.0
        } else {
            self.violations as f64 / self.observations as f64
        }
    }
}

impl fmt::Display for EmpiricalResidualReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EMPIRICAL RESIDUAL\n")?;
        writeln!(f, "law:                       {}", self.scope.law.name)?;
        writeln!(f, "law version:               {}", self.scope.law.version)?;
        writeln!(f, "residual:                  {}", self.scope.expression)?;
        writeln!(f, "region:                    {}", self.scope.region)?;
        writeln!(f, "evaluator:                 {}", self.scope.evaluator)?;
        writeln!(f, "observations:              {}", self.observations)?;
        writeln!(f, "violations:                {}", self.violations)?;
        writeln!(
            f,
            "violation rate:            {:.4}%",
            self.violation_rate() * 100.0
        )?;
        writeln!(
            f,
            "maximum allowed rate:      {:.4}%",
            self.limits.max_violation_rate * 100.0
        )?;
        writeln!(
            f,
            "maximum |residual|:        {:.8}",
            self.maximum_absolute_residual
        )?;
        writeln!(f, "RMS residual:              {:.8}", self.rms_residual)?;
        writeln!(
            f,
            "absolute tolerance:        {:.8}",
            self.limits.absolute_tolerance
        )?;
        writeln!(f, "claim:                     {}", self.claim)?;
        writeln!(f, "evidence:                  {}", self.evidence)?;
        write!(
            f,
            "\nThese sampled discretized residual observations met their declared limits. This does not prove the law, evaluator accuracy, unsampled-domain behavior, boundary satisfaction, or a global guarantee.\n\nPASS"
        )
    }
}

#[derive(Debug)]
struct ResidualViolation(EmpiricalResidualReceipt);

impl fmt::Display for ResidualViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let receipt = &self.0;
        write!(
            f,
            "empirical residual limit violation\n\nlaw:                  {}\nlaw version:          {}\nresidual:             {}\nregion:               {}\nevaluator:            {}\nobservations:         {}\nviolations:           {}\nviolation rate:       {:.4}%\nmaximum allowed rate: {:.4}%\nmaximum |residual|:   {:.8}\nRMS residual:         {:.8}\nabsolute tolerance:   {:.8}\n\nthe sampled discretized residual observations left their declared limits; no global law claim was established",
            receipt.scope.law.name,
            receipt.scope.law.version,
            receipt.scope.expression,
            receipt.scope.region,
            receipt.scope.evaluator,
            receipt.observations,
            receipt.violations,
            receipt.violation_rate() * 100.0,
            receipt.limits.max_violation_rate * 100.0,
            receipt.maximum_absolute_residual,
            receipt.rms_residual,
            receipt.limits.absolute_tolerance,
        )
    }
}

impl std::error::Error for ResidualViolation {}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> Result<ResidualScope> {
        ResidualScope::new(
            LawIdentity::new("unit-rate exponential decay", "ode-v1")?,
            "du/dt + u",
            "held-out interior t in [0.01, 0.99]",
            "second-order central difference; h=0.01; fp32",
        )
    }

    #[test]
    fn receipt_reports_scope_statistics_and_claim_boundary() -> Result<()> {
        let mut audit = EmpiricalResidual::new(scope()?, ResidualLimits::new(0.01, 0.25)?);
        let receipt = audit.observe_batch([0.0, 0.005, -0.02, 0.01])?;
        assert_eq!(receipt.observations, 4);
        assert_eq!(receipt.violations, 1);
        assert_eq!(receipt.violation_rate(), 0.25);
        assert_eq!(receipt.maximum_absolute_residual, 0.02);
        assert!((receipt.rms_residual - 0.011456439).abs() < 1e-9);
        assert_eq!(receipt.claim, ResidualClaim::Empirical);
        assert_eq!(receipt.evidence, ResidualEvidence::VerifiedObservations);
        let text = receipt.to_string();
        assert!(text.contains("sampled discretized residual observations"));
        assert!(text.contains("does not prove the law"));
        assert!(text.contains("boundary satisfaction"));
        Ok(())
    }

    #[test]
    fn invalid_batches_are_rejected_atomically() -> Result<()> {
        let mut audit = EmpiricalResidual::new(scope()?, ResidualLimits::strict(0.1)?);
        audit.observe(0.05)?;
        let before = audit.receipt();
        assert!(audit.observe_batch([]).is_err());
        assert!(audit.observe_batch([0.01, f64::NAN, 0.02]).is_err());
        assert!(audit.observe_batch([f64::INFINITY]).is_err());
        assert_eq!(audit.receipt(), before);
        Ok(())
    }

    #[test]
    fn violation_is_recorded_before_failure_and_rms_stays_stable() -> Result<()> {
        let mut audit = EmpiricalResidual::new(scope()?, ResidualLimits::strict(1.0)?);
        let error = audit
            .observe_batch([f64::MAX, f64::MAX / 2.0])
            .unwrap_err()
            .to_string();
        let receipt = audit.receipt();
        assert_eq!(receipt.observations, 2);
        assert_eq!(receipt.violations, 2);
        assert!(receipt.rms_residual.is_finite());
        assert!(error.contains("no global law claim"));
        Ok(())
    }

    #[test]
    fn declarations_reject_ambiguous_text_and_limits() -> Result<()> {
        assert!(LawIdentity::new("", "v1").is_err());
        assert!(LawIdentity::new("law", " v1").is_err());
        let law = LawIdentity::new("law", "v1")?;
        assert!(ResidualScope::new(law.clone(), "", "region", "method").is_err());
        assert!(ResidualScope::new(law.clone(), "r", "region\n", "method").is_err());
        assert!(ResidualLimits::strict(-1.0).is_err());
        assert!(ResidualLimits::strict(f64::NAN).is_err());
        assert!(ResidualLimits::new(0.0, 1.01).is_err());
        let scope = ResidualScope::new(law, "r", "region", "method")?;
        assert_eq!(scope.law().name(), "law");
        assert_eq!(scope.law().version(), "v1");
        assert_eq!(scope.expression(), "r");
        assert_eq!(scope.region(), "region");
        assert_eq!(scope.evaluator(), "method");
        assert!(
            EmpiricalResidual::new(scope, ResidualLimits::strict(0.0)?)
                .assert_within_limits()
                .is_err()
        );
        Ok(())
    }
}
