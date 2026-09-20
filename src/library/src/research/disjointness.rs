//! Executable separation contracts for caller-defined semantic identities.
use crate::Result;
use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    hash::Hash,
};

/// Names the equivalence relation used to decide whether two observations are
/// the same scientific unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityScheme {
    name: String,
    version: String,
    unit: String,
}

impl IdentityScheme {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        unit: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            name: label("identity scheme name", name.into())?,
            version: label("identity scheme version", version.into())?,
            unit: label("identity unit", unit.into())?,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn unit(&self) -> &str {
        &self.unit
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

/// The strength of the evidence recorded by a disjointness receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisjointnessEvidence {
    /// Every identity delivered to this ledger was compared exactly. This says
    /// nothing about observations that were not delivered or the soundness of
    /// the caller's canonicalization function.
    VerifiedObservations,
}

impl fmt::Display for DisjointnessEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerifiedObservations => write!(f, "verified observed identities"),
        }
    }
}

/// Whether the ledger keeps identities after checking an observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PopulationMode {
    /// Retain identities for exact comparisons with populations observed later.
    Retained,
    /// Compare against sealed retained populations, then discard the identity.
    /// Exactly one streaming population may be declared.
    Streaming,
}

impl fmt::Display for PopulationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retained => write!(f, "retained"),
            Self::Streaming => write!(f, "streaming"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PopulationSpec {
    name: String,
    mode: PopulationMode,
}

impl PopulationSpec {
    pub fn retained(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mode: PopulationMode::Retained,
        }
    }

    pub fn streaming(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mode: PopulationMode::Streaming,
        }
    }
}

impl From<&str> for PopulationSpec {
    fn from(name: &str) -> Self {
        Self::retained(name)
    }
}

impl From<String> for PopulationSpec {
    fn from(name: String) -> Self {
        Self::retained(name)
    }
}

#[derive(Debug)]
struct Population<K> {
    mode: PopulationMode,
    identities: HashSet<K>,
    observations: usize,
    repeats: usize,
}

impl<K> Population<K> {
    fn new(mode: PopulationMode) -> Self {
        Self {
            mode,
            identities: HashSet::new(),
            observations: 0,
            repeats: 0,
        }
    }
}

/// Exact separation across two or more declared populations.
///
/// `K` is the scientific identity. A raw source ID checks draw separation;
/// a canonicalized problem value can check a stronger semantic equivalence.
/// Retained populations are pairwise compared. One optional streaming
/// population can be checked against sealed retained populations without
/// retaining an unbounded history.
pub struct Disjointness<K> {
    scheme: IdentityScheme,
    populations: BTreeMap<String, Population<K>>,
}

impl<K> Disjointness<K>
where
    K: Eq + Hash,
{
    pub fn new(
        scheme: IdentityScheme,
        populations: impl IntoIterator<Item = impl Into<PopulationSpec>>,
    ) -> Result<Self> {
        let mut declared = BTreeMap::new();
        for population in populations {
            let population = population.into();
            let name = label("population name", population.name)?;
            if declared
                .insert(name.clone(), Population::new(population.mode))
                .is_some()
            {
                return Err(format!("population {name:?} was declared more than once").into());
            }
        }
        if declared.len() < 2 {
            return Err("disjointness requires at least two declared populations".into());
        }
        if declared
            .values()
            .filter(|population| population.mode == PopulationMode::Streaming)
            .count()
            > 1
        {
            return Err("disjointness permits at most one streaming population".into());
        }
        Ok(Self {
            scheme,
            populations: declared,
        })
    }

    /// Atomically compare and record one batch of canonical identities.
    /// A contaminated batch leaves the ledger unchanged.
    pub fn observe(
        &mut self,
        population: &str,
        identities: impl IntoIterator<Item = K>,
    ) -> Result<DisjointnessReceipt> {
        if !self.populations.contains_key(population) {
            return Err(format!("undeclared disjointness population {population:?}").into());
        }
        let mode = self
            .populations
            .get(population)
            .expect("population checked above")
            .mode;
        if mode == PopulationMode::Streaming {
            let empty_references = self
                .populations
                .iter()
                .filter(|(_, state)| {
                    state.mode == PopulationMode::Retained && state.observations == 0
                })
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>();
            if !empty_references.is_empty() {
                return Err(format!(
                    "streaming population {population:?} cannot start before retained populations are observed: {}",
                    empty_references.join(", ")
                )
                .into());
            }
        } else if self
            .populations
            .values()
            .any(|state| state.mode == PopulationMode::Streaming && state.observations > 0)
        {
            return Err(format!(
                "retained population {population:?} is sealed after streaming observations begin"
            )
            .into());
        }

        let incoming = identities.into_iter().collect::<Vec<_>>();

        for (index, identity) in incoming.iter().enumerate() {
            if let Some((other, _)) = self
                .populations
                .iter()
                .filter(|(name, _)| name.as_str() != population)
                .filter(|(_, state)| state.mode == PopulationMode::Retained)
                .find(|(_, state)| state.identities.contains(identity))
            {
                return Err(format!(
                    "semantic contamination under {}@{}: {population:?} identity at batch index {index} was already observed in {other:?}\nidentity unit: {}",
                    self.scheme.name, self.scheme.version, self.scheme.unit
                )
                .into());
            }
        }

        let state = self
            .populations
            .get_mut(population)
            .expect("population checked above");
        let observations = state
            .observations
            .checked_add(incoming.len())
            .ok_or("disjointness observation count overflow")?;
        if state.mode == PopulationMode::Retained {
            for identity in incoming {
                if !state.identities.insert(identity) {
                    state.repeats = state
                        .repeats
                        .checked_add(1)
                        .ok_or("disjointness repeat count overflow")?;
                }
            }
        }
        state.observations = observations;
        Ok(self.receipt())
    }

    /// Require at least one observed identity in every declared population.
    pub fn assert_disjoint(&self) -> Result<DisjointnessReceipt> {
        let empty = self
            .populations
            .iter()
            .filter(|(_, state)| state.observations == 0)
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        if !empty.is_empty() {
            return Err(format!(
                "disjointness evidence is incomplete; no identities were observed for {}",
                empty.join(", ")
            )
            .into());
        }
        Ok(self.receipt())
    }

    pub fn receipt(&self) -> DisjointnessReceipt {
        DisjointnessReceipt {
            scheme: self.scheme.clone(),
            evidence: DisjointnessEvidence::VerifiedObservations,
            populations: self
                .populations
                .iter()
                .map(|(name, state)| PopulationReceipt {
                    name: name.clone(),
                    mode: state.mode,
                    observations: state.observations,
                    unique_identities: (state.mode == PopulationMode::Retained)
                        .then_some(state.identities.len()),
                    within_population_repeats: (state.mode == PopulationMode::Retained)
                        .then_some(state.repeats),
                })
                .collect(),
            complete: self
                .populations
                .values()
                .all(|state| state.observations > 0),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PopulationReceipt {
    pub name: String,
    pub mode: PopulationMode,
    pub observations: usize,
    pub unique_identities: Option<usize>,
    pub within_population_repeats: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisjointnessReceipt {
    pub scheme: IdentityScheme,
    pub evidence: DisjointnessEvidence,
    pub populations: Vec<PopulationReceipt>,
    pub complete: bool,
}

impl DisjointnessReceipt {
    pub fn population(&self, name: &str) -> Option<&PopulationReceipt> {
        self.populations
            .iter()
            .find(|population| population.name == name)
    }
}

impl fmt::Display for DisjointnessReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SEMANTIC DISJOINTNESS\n")?;
        writeln!(
            f,
            "identity scheme:       {}@{}",
            self.scheme.name, self.scheme.version
        )?;
        writeln!(f, "identity unit:         {}", self.scheme.unit)?;
        writeln!(f, "evidence:              {}", self.evidence)?;
        for population in &self.populations {
            match (
                population.unique_identities,
                population.within_population_repeats,
            ) {
                (Some(unique), Some(repeats)) => writeln!(
                    f,
                    "{}: mode={} observations={} unique={} repeats={}",
                    population.name, population.mode, population.observations, unique, repeats
                )?,
                _ => writeln!(
                    f,
                    "{}: mode={} observations={} unique=not-retained repeats=not-measured",
                    population.name, population.mode, population.observations
                )?,
            }
        }
        writeln!(f, "observed cross-population overlap: 0")?;
        write!(f, "\n{}", if self.complete { "PASS" } else { "INCOMPLETE" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheme() -> Result<IdentityScheme> {
        IdentityScheme::new(
            "case-folded-label",
            "1",
            "label after Unicode lowercase canonicalization",
        )
    }

    #[test]
    fn semantic_identity_catches_overlap_hidden_by_distinct_source_ids() -> Result<()> {
        let mut guard = Disjointness::new(scheme()?, ["training", "evaluation"])?;
        guard.observe("evaluation", ["same problem".to_owned()])?;
        let error = guard
            .observe("training", ["SAME PROBLEM".to_lowercase()])
            .unwrap_err()
            .to_string();
        assert!(error.contains("semantic contamination"), "{error}");
        assert!(error.contains("case-folded-label@1"), "{error}");
        assert!(error.contains("evaluation"), "{error}");
        Ok(())
    }

    #[test]
    fn rejected_batches_are_atomic() -> Result<()> {
        let mut guard = Disjointness::new(scheme()?, ["training", "evaluation"])?;
        guard.observe("evaluation", ["held out".to_owned()])?;
        guard
            .observe("training", ["new".to_owned(), "held out".to_owned()])
            .unwrap_err();
        let receipt = guard.receipt();
        assert_eq!(receipt.population("training").unwrap().observations, 0);
        assert_eq!(
            receipt.population("training").unwrap().unique_identities,
            Some(0)
        );
        assert!(!receipt.complete);
        Ok(())
    }

    #[test]
    fn receipt_counts_reuse_without_confusing_it_with_cross_population_overlap() -> Result<()> {
        let mut guard = Disjointness::new(scheme()?, ["training", "audit", "evaluation"])?;
        guard.observe("training", ["a".to_owned(), "a".to_owned(), "b".to_owned()])?;
        guard.observe("evaluation", ["c".to_owned()])?;
        guard.observe("audit", ["d".to_owned(), "d".to_owned()])?;
        let receipt = guard.assert_disjoint()?;
        let training = receipt.population("training").unwrap();
        assert_eq!(
            (training.observations, training.unique_identities),
            (3, Some(2))
        );
        assert_eq!(training.within_population_repeats, Some(1));
        assert_eq!(receipt.evidence, DisjointnessEvidence::VerifiedObservations);
        assert!(receipt.complete);
        assert!(receipt.to_string().ends_with("PASS"));
        Ok(())
    }

    #[test]
    fn declarations_and_completion_are_checked() -> Result<()> {
        assert!(IdentityScheme::new("", "1", "sample").is_err());
        assert!(IdentityScheme::new("raw", " 1", "sample").is_err());
        assert!(IdentityScheme::new("raw", "1", "sample\nidentity").is_err());
        assert!(Disjointness::<u8>::new(scheme()?, ["training"]).is_err());
        assert!(Disjointness::<u8>::new(scheme()?, ["training", "training"]).is_err());

        let mut guard = Disjointness::new(scheme()?, ["training", "evaluation"])?;
        assert!(guard.observe("unknown", [1_u8]).is_err());
        guard.observe("training", [1])?;
        let error = guard.assert_disjoint().unwrap_err().to_string();
        assert!(error.contains("evaluation"), "{error}");
        assert!(guard.receipt().to_string().ends_with("INCOMPLETE"));
        Ok(())
    }

    #[test]
    fn streaming_population_is_bounded_and_requires_sealed_references() -> Result<()> {
        let specs = [
            PopulationSpec::streaming("training"),
            PopulationSpec::retained("evaluation"),
        ];
        let mut guard = Disjointness::new(scheme()?, specs)?;
        let error = guard.observe("training", [1_u32]).unwrap_err().to_string();
        assert!(error.contains("evaluation"), "{error}");

        guard.observe("evaluation", [10, 11])?;
        guard.observe("training", 100..10_100)?;
        let receipt = guard.assert_disjoint()?;
        let training = receipt.population("training").unwrap();
        assert_eq!(training.mode, PopulationMode::Streaming);
        assert_eq!(training.observations, 10_000);
        assert_eq!(training.unique_identities, None);
        assert_eq!(training.within_population_repeats, None);
        assert!(guard.populations["training"].identities.is_empty());
        assert_eq!(guard.populations["evaluation"].identities.len(), 2);
        assert!(receipt.to_string().contains("unique=not-retained"));

        let error = guard.observe("evaluation", [12]).unwrap_err().to_string();
        assert!(error.contains("sealed"), "{error}");
        let error = guard.observe("training", [10]).unwrap_err().to_string();
        assert!(error.contains("semantic contamination"), "{error}");
        assert_eq!(
            guard.receipt().population("training").unwrap().observations,
            10_000
        );
        Ok(())
    }

    #[test]
    fn only_one_streaming_population_is_permitted() -> Result<()> {
        let specs = [
            PopulationSpec::streaming("training-a"),
            PopulationSpec::streaming("training-b"),
        ];
        assert!(Disjointness::<u8>::new(scheme()?, specs).is_err());
        Ok(())
    }
}
