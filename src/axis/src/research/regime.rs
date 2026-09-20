//! Executable assumptions about how a training run consumes data.
use crate::Result;
use std::{collections::HashSet, fmt};

/// A count-only invariant for a loader that promises unique examples within one pass.
/// This proves the consumption budget, not the loader's uniqueness promise.
#[derive(Clone, Copy, Debug)]
pub struct SinglePass {
    available: usize,
    consumed: usize,
}

impl SinglePass {
    pub fn new(available: usize) -> Result<Self> {
        if available == 0 {
            return Err("single-pass data availability must be positive".into());
        }
        Ok(Self {
            available,
            consumed: 0,
        })
    }

    pub fn consume(&mut self, count: usize) -> Result<RegimeSnapshot> {
        self.consumed = self
            .consumed
            .checked_add(count)
            .ok_or("single-pass sample count overflow")?;
        let snapshot = RegimeSnapshot::fixed(self.available, self.consumed, 0);
        if self.consumed > self.available {
            return Err(RegimeViolation::SinglePass(snapshot).into());
        }
        Ok(snapshot)
    }

    pub fn snapshot(self) -> RegimeSnapshot {
        RegimeSnapshot::fixed(self.available, self.consumed, 0)
    }
}

/// Declared operational limits for an infinite-data approximation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdrLimits {
    max_coverage: Option<f64>,
    max_repeat_rate: f64,
}

impl IdrLimits {
    pub fn fixed(max_coverage: f64, max_repeat_rate: f64) -> Result<Self> {
        if !max_coverage.is_finite() || !(0.0..=1.0).contains(&max_coverage) {
            return Err("IDR max coverage must be finite and within 0..=1".into());
        }
        Self::validate_repeat_rate(max_repeat_rate)?;
        Ok(Self {
            max_coverage: Some(max_coverage),
            max_repeat_rate,
        })
    }

    /// Generated streams have no finite-corpus coverage; observed repeats remain measurable.
    pub fn generated(max_repeat_rate: f64) -> Result<Self> {
        Self::validate_repeat_rate(max_repeat_rate)?;
        Ok(Self {
            max_coverage: None,
            max_repeat_rate,
        })
    }

    fn validate_repeat_rate(rate: f64) -> Result<()> {
        if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
            return Err("IDR max repeat rate must be finite and within 0..=1".into());
        }
        Ok(())
    }

    pub fn max_coverage(self) -> Option<f64> {
        self.max_coverage
    }

    pub fn max_repeat_rate(self) -> f64 {
        self.max_repeat_rate
    }
}

/// Exact observed-ID accounting. Callers choose a stable ID whose equality means reuse.
pub struct Idr {
    available: Option<usize>,
    limits: IdrLimits,
    seen: HashSet<u128>,
    consumed: usize,
    repeats: usize,
}

impl Idr {
    pub fn fixed(available: usize, limits: IdrLimits) -> Result<Self> {
        if available == 0 {
            return Err("IDR data availability must be positive".into());
        }
        if limits.max_coverage.is_none() {
            return Err("fixed IDR requires a coverage limit".into());
        }
        Ok(Self {
            available: Some(available),
            limits,
            seen: HashSet::new(),
            consumed: 0,
            repeats: 0,
        })
    }

    pub fn generated(limits: IdrLimits) -> Result<Self> {
        if limits.max_coverage.is_some() {
            return Err("generated IDR cannot declare finite-corpus coverage".into());
        }
        Ok(Self {
            available: None,
            limits,
            seen: HashSet::new(),
            consumed: 0,
            repeats: 0,
        })
    }

    /// Record IDs after a batch is drawn, before its examples affect an optimizer step.
    pub fn observe(&mut self, ids: impl IntoIterator<Item = u128>) -> Result<IdrReceipt> {
        for id in ids {
            self.consumed = self
                .consumed
                .checked_add(1)
                .ok_or("IDR sample count overflow")?;
            if !self.seen.insert(id) {
                self.repeats = self
                    .repeats
                    .checked_add(1)
                    .ok_or("IDR repeat count overflow")?;
            }
        }
        self.assert_regime()
    }

    pub fn assert_regime(&self) -> Result<IdrReceipt> {
        let snapshot = match self.available {
            Some(available) => RegimeSnapshot::fixed(available, self.consumed, self.repeats),
            None => RegimeSnapshot::generated(self.consumed, self.repeats),
        };
        let coverage_exceeded = self
            .limits
            .max_coverage
            .is_some_and(|limit| snapshot.coverage().expect("fixed corpus") > limit);
        if coverage_exceeded || snapshot.repeat_rate() > self.limits.max_repeat_rate {
            return Err(RegimeViolation::Idr {
                snapshot,
                limits: self.limits,
            }
            .into());
        }
        Ok(IdrReceipt {
            snapshot,
            limits: self.limits,
        })
    }

    pub fn snapshot(&self) -> RegimeSnapshot {
        match self.available {
            Some(available) => RegimeSnapshot::fixed(available, self.consumed, self.repeats),
            None => RegimeSnapshot::generated(self.consumed, self.repeats),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdrReceipt {
    pub snapshot: RegimeSnapshot,
    pub limits: IdrLimits,
}

impl fmt::Display for IdrReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IDR STATUS\n")?;
        writeln!(f, "samples consumed:      {}", self.snapshot.consumed)?;
        writeln!(f, "unique sample IDs:     {}", self.snapshot.unique)?;
        match self.snapshot.available {
            Some(available) => {
                writeln!(f, "available population:  {available}")?;
                writeln!(
                    f,
                    "coverage:              {:.2}%",
                    self.snapshot.coverage().expect("fixed corpus") * 100.0
                )?;
                writeln!(
                    f,
                    "max coverage:          {:.2}%",
                    self.limits.max_coverage.expect("fixed corpus") * 100.0
                )?;
            }
            None => writeln!(f, "available population:  generated stream")?,
        }
        writeln!(
            f,
            "observed reuse:         {:.4}%",
            self.snapshot.repeat_rate() * 100.0
        )?;
        writeln!(
            f,
            "max observed reuse:     {:.4}%",
            self.limits.max_repeat_rate * 100.0
        )?;
        write!(f, "\nPASS")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeSnapshot {
    pub available: Option<usize>,
    pub consumed: usize,
    pub unique: usize,
    pub repeats: usize,
}

impl RegimeSnapshot {
    fn fixed(available: usize, consumed: usize, repeats: usize) -> Self {
        Self {
            available: Some(available),
            consumed,
            unique: consumed - repeats,
            repeats,
        }
    }
    fn generated(consumed: usize, repeats: usize) -> Self {
        Self {
            available: None,
            consumed,
            unique: consumed - repeats,
            repeats,
        }
    }
    pub fn coverage(self) -> Option<f64> {
        self.available
            .map(|available| self.consumed as f64 / available as f64)
    }
    pub fn epochs(self) -> Option<f64> {
        self.coverage()
    }
    pub fn repeat_rate(self) -> f64 {
        if self.consumed == 0 {
            0.0
        } else {
            self.repeats as f64 / self.consumed as f64
        }
    }
}

#[derive(Debug)]
enum RegimeViolation {
    SinglePass(RegimeSnapshot),
    Idr {
        snapshot: RegimeSnapshot,
        limits: IdrLimits,
    },
}

impl fmt::Display for RegimeViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SinglePass(snapshot) => write!(
                f,
                "single-pass violation\n\nsamples available:  {}\nsamples consumed:   {}\ncoverage:           {:.2}%\nepochs:             {:.4}\n\ntraining has begun reusing the finite corpus",
                snapshot.available.expect("fixed corpus"),
                snapshot.consumed,
                snapshot.coverage().expect("fixed corpus") * 100.0,
                snapshot.epochs().expect("fixed corpus"),
            ),
            Self::Idr { snapshot, limits } => {
                writeln!(f, "IDR violation\n")?;
                if let Some(available) = snapshot.available {
                    writeln!(f, "samples available:  {available}")?;
                    writeln!(
                        f,
                        "coverage:           {:.2}%",
                        snapshot.coverage().expect("fixed corpus") * 100.0
                    )?;
                    writeln!(
                        f,
                        "max coverage:       {:.2}%",
                        limits.max_coverage.expect("fixed corpus") * 100.0
                    )?;
                }
                write!(
                    f,
                    "samples consumed:   {}\nunique IDs:         {}\nrepeated IDs:       {}\nrepeat rate:        {:.4}%\nmax repeat rate:    {:.4}%\n\nthe run has left its declared infinite-data approximation",
                    snapshot.consumed,
                    snapshot.unique,
                    snapshot.repeats,
                    snapshot.repeat_rate() * 100.0,
                    limits.max_repeat_rate * 100.0,
                )
            }
        }
    }
}

impl std::error::Error for RegimeViolation {}

/// Exact sample-identity separation between training and evaluation populations.
#[derive(Default)]
pub struct TrainEvalDisjoint {
    train: HashSet<u128>,
    evaluation: HashSet<u128>,
}

impl TrainEvalDisjoint {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe_train(
        &mut self,
        ids: impl IntoIterator<Item = u128>,
    ) -> Result<TrainEvalReceipt> {
        for id in ids {
            if self.evaluation.contains(&id) {
                return Err(format!(
                    "train/evaluation contamination: sample ID {id} was already observed in evaluation"
                )
                .into());
            }
            self.train.insert(id);
        }
        Ok(self.receipt())
    }

    pub fn observe_evaluation(
        &mut self,
        ids: impl IntoIterator<Item = u128>,
    ) -> Result<TrainEvalReceipt> {
        for id in ids {
            if self.train.contains(&id) {
                return Err(format!(
                    "train/evaluation contamination: sample ID {id} was already observed in training"
                )
                .into());
            }
            self.evaluation.insert(id);
        }
        Ok(self.receipt())
    }

    pub fn receipt(&self) -> TrainEvalReceipt {
        TrainEvalReceipt {
            unique_train_ids: self.train.len(),
            unique_evaluation_ids: self.evaluation.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainEvalReceipt {
    pub unique_train_ids: usize,
    pub unique_evaluation_ids: usize,
}

/// Exact accounting for a declared number of complete finite-corpus passes.
pub struct FinitePasses {
    population: usize,
    declared_passes: usize,
    completed_passes: usize,
    consumed: usize,
    current_ids: HashSet<u128>,
    population_ids: Option<HashSet<u128>>,
}

impl FinitePasses {
    pub fn new(population: usize, declared_passes: usize) -> Result<Self> {
        if population == 0 || declared_passes == 0 {
            return Err("finite passes require a positive population and pass count".into());
        }
        Ok(Self {
            population,
            declared_passes,
            completed_passes: 0,
            consumed: 0,
            current_ids: HashSet::with_capacity(population),
            population_ids: None,
        })
    }

    pub fn observe(&mut self, ids: impl IntoIterator<Item = u128>) -> Result<FinitePassesReceipt> {
        for id in ids {
            if self.completed_passes == self.declared_passes {
                return Err(
                    "finite-pass violation: observations continued after the declared passes"
                        .into(),
                );
            }
            if !self.current_ids.insert(id) {
                return Err(format!(
                    "finite-pass violation: sample ID {id} repeated within pass {}",
                    self.completed_passes + 1
                )
                .into());
            }
            self.consumed = self
                .consumed
                .checked_add(1)
                .ok_or("finite-pass observation count overflow")?;
            if self.current_ids.len() == self.population {
                if let Some(population_ids) = &self.population_ids {
                    if &self.current_ids != population_ids {
                        return Err(format!(
                            "finite-pass violation: pass {} did not contain the established population",
                            self.completed_passes + 1
                        )
                        .into());
                    }
                } else {
                    self.population_ids = Some(self.current_ids.clone());
                }
                self.completed_passes += 1;
                self.current_ids.clear();
            }
        }
        Ok(self.receipt())
    }

    pub fn finish(&self) -> Result<FinitePassesReceipt> {
        if self.completed_passes != self.declared_passes || !self.current_ids.is_empty() {
            return Err(format!(
                "finite-pass violation: source ended after {} complete passes and {} samples of pass {}; declared {} complete passes",
                self.completed_passes,
                self.current_ids.len(),
                self.completed_passes + 1,
                self.declared_passes
            )
            .into());
        }
        Ok(self.receipt())
    }

    pub fn receipt(&self) -> FinitePassesReceipt {
        FinitePassesReceipt {
            population: self.population,
            declared_passes: self.declared_passes,
            completed_passes: self.completed_passes,
            samples_in_current_pass: self.current_ids.len(),
            observations: self.consumed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinitePassesReceipt {
    pub population: usize,
    pub declared_passes: usize,
    pub completed_passes: usize,
    pub samples_in_current_pass: usize,
    pub observations: usize,
}

impl fmt::Display for FinitePassesReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FINITE PASSES STATUS\n\npopulation:             {}\ndeclared passes:        {}\ncompleted passes:       {}\nsamples in next pass:   {}\ntotal observations:     {}\n\nPASS",
            self.population,
            self.declared_passes,
            self.completed_passes,
            self.samples_in_current_pass,
            self.observations,
        )
    }
}

impl fmt::Display for TrainEvalReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TRAIN/EVALUATION DISJOINTNESS\n\nunique training IDs:    {}\nunique evaluation IDs:  {}\nobserved overlap:        0\n\nPASS",
            self.unique_train_ids, self.unique_evaluation_ids
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_pass_fails_at_the_first_example_past_the_corpus() -> Result<()> {
        let mut pass = SinglePass::new(10)?;
        assert_eq!(pass.consume(6)?.coverage(), Some(0.6));
        assert_eq!(pass.consume(4)?.epochs(), Some(1.0));
        let error = pass.consume(1).unwrap_err().to_string();
        assert!(error.contains("samples available:  10"), "{error}");
        assert!(error.contains("samples consumed:   11"), "{error}");
        assert!(error.contains("110.00%"), "{error}");
        Ok(())
    }

    #[test]
    fn fixed_idr_checks_coverage_and_observed_repeats() -> Result<()> {
        let limits = IdrLimits::fixed(0.5, 0.2)?;
        let mut coverage = Idr::fixed(10, limits)?;
        coverage.observe([0, 1, 2, 3, 4])?;
        let error = coverage.observe([5]).unwrap_err().to_string();
        assert!(error.contains("coverage:           60.00%"), "{error}");
        assert!(error.contains("max coverage:       50.00%"), "{error}");

        let mut repeats = Idr::fixed(100, IdrLimits::fixed(1.0, 0.2)?)?;
        repeats.observe([10, 11, 12, 13])?;
        let error = repeats.observe([10, 10]).unwrap_err().to_string();
        assert!(error.contains("repeated IDs:       2"), "{error}");
        assert!(error.contains("repeat rate:        33.3333%"), "{error}");
        Ok(())
    }

    #[test]
    fn generated_idr_checks_identity_without_inventing_coverage() -> Result<()> {
        let mut idr = Idr::generated(IdrLimits::generated(0.0)?)?;
        let receipt = idr.observe([101, 102, 103])?;
        assert_eq!(receipt.snapshot.available, None);
        assert_eq!(receipt.snapshot.repeat_rate(), 0.0);
        assert!(receipt.to_string().ends_with("PASS"));
        let error = idr.observe([102]).unwrap_err().to_string();
        assert!(!error.contains("coverage"), "{error}");
        assert!(error.contains("repeated IDs:       1"), "{error}");
        Ok(())
    }

    #[test]
    fn invalid_regime_declarations_fail() {
        assert!(IdrLimits::fixed(1.1, 0.0).is_err());
        assert!(IdrLimits::fixed(0.1, -0.1).is_err());
        assert!(IdrLimits::generated(f64::NAN).is_err());
        assert!(SinglePass::new(0).is_err());
    }

    #[test]
    fn train_and_evaluation_identity_overlap_fails_in_either_order() -> Result<()> {
        let mut evaluation_first = TrainEvalDisjoint::new();
        evaluation_first.observe_evaluation([100, 101])?;
        let error = evaluation_first
            .observe_train([1, 100])
            .unwrap_err()
            .to_string();
        assert!(error.contains("sample ID 100"), "{error}");

        let mut train_first = TrainEvalDisjoint::new();
        train_first.observe_train([7, 8])?;
        let error = train_first
            .observe_evaluation([9, 8])
            .unwrap_err()
            .to_string();
        assert!(error.contains("sample ID 8"), "{error}");
        Ok(())
    }

    #[test]
    fn finite_passes_require_the_same_population_once_per_pass() -> Result<()> {
        let mut passes = FinitePasses::new(3, 2)?;
        assert_eq!(passes.observe([2, 0])?.completed_passes, 0);
        assert_eq!(passes.observe([1])?.completed_passes, 1);
        let receipt = passes.observe([1, 2, 0])?;
        assert_eq!(receipt.completed_passes, 2);
        assert_eq!(receipt.observations, 6);
        assert!(passes.finish()?.to_string().ends_with("PASS"));

        let mut duplicate = FinitePasses::new(3, 1)?;
        let error = duplicate.observe([0, 1, 0]).unwrap_err().to_string();
        assert!(error.contains("repeated within pass 1"), "{error}");

        let mut short = FinitePasses::new(3, 2)?;
        short.observe([0, 1, 2, 0])?;
        let error = short.finish().unwrap_err().to_string();
        assert!(error.contains("1 samples of pass 2"), "{error}");
        Ok(())
    }
}
