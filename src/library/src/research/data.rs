//! Identity-aware sources and batching. Tensor construction remains explicit.
use crate::{
    FinitePasses, FinitePassesReceipt, Idr, IdrLimits, IdrReceipt, RegimeSnapshot, Result,
    SinglePass,
};
use std::fmt;

#[derive(Debug)]
pub struct Sample<T> {
    pub id: u128,
    pub value: T,
}

/// A source may be finite or generated. `None` availability means no finite corpus exists.
pub trait DataSource {
    type Item;

    fn next_sample(&mut self) -> Result<Option<Sample<Self::Item>>>;
    fn available(&self) -> Option<usize>;
}

pub struct InMemoryDataset<T> {
    samples: Vec<T>,
    ids: Vec<u128>,
    cursor: usize,
    repeat: bool,
}

impl<T> InMemoryDataset<T> {
    pub fn new(samples: Vec<T>) -> Result<Self> {
        let ids = (0..samples.len())
            .map(u128::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Self::with_ids(samples, ids)
    }

    pub fn with_ids(samples: Vec<T>, ids: Vec<u128>) -> Result<Self> {
        if samples.is_empty() {
            return Err("dataset must contain at least one sample".into());
        }
        if samples.len() != ids.len() {
            return Err("dataset samples and IDs must have equal lengths".into());
        }
        Ok(Self {
            samples,
            ids,
            cursor: 0,
            repeat: false,
        })
    }

    /// Explicitly turn a finite pass into a repeating source.
    pub fn repeat(mut self) -> Self {
        self.repeat = true;
        self
    }
}

impl<T: Clone> DataSource for InMemoryDataset<T> {
    type Item = T;

    fn next_sample(&mut self) -> Result<Option<Sample<Self::Item>>> {
        if self.cursor == self.samples.len() {
            if !self.repeat {
                return Ok(None);
            }
            self.cursor = 0;
        }
        let index = self.cursor;
        self.cursor += 1;
        Ok(Some(Sample {
            id: self.ids[index],
            value: self.samples[index].clone(),
        }))
    }

    fn available(&self) -> Option<usize> {
        Some(self.samples.len())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdditionSample {
    pub left: f32,
    pub right: f32,
    pub sum: f32,
}

/// An inexhaustible deterministic stream of fresh addition problems.
/// IDs are unique draw counters; values can still coincide after f32 projection.
pub struct AdditionDataset {
    seed: u64,
    next_id: u64,
}

impl AdditionDataset {
    pub fn new(seed: u64) -> Self {
        Self { seed, next_id: 0 }
    }

    fn mix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn unit(value: u64) -> f32 {
        (value >> 40) as f32 / (1_u32 << 24) as f32
    }
}

impl DataSource for AdditionDataset {
    type Item = AdditionSample;

    fn next_sample(&mut self) -> Result<Option<Sample<Self::Item>>> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("AdditionDataset exhausted its u64 sample identity space")?;
        let left = Self::unit(Self::mix(self.seed ^ id.wrapping_mul(2))) * 2.0 - 1.0;
        let right =
            Self::unit(Self::mix(self.seed ^ id.wrapping_mul(2).wrapping_add(1))) * 2.0 - 1.0;
        Ok(Some(Sample {
            id: (u128::from(self.seed) << 64) | u128::from(id),
            value: AdditionSample {
                left,
                right,
                sum: left + right,
            },
        }))
    }

    fn available(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug)]
pub struct Batch<T> {
    pub samples: Vec<T>,
    pub sample_ids: Vec<u128>,
    /// Total observations delivered before this batch.
    pub samples_seen_before: usize,
    pub regime: DataRegimeReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataRegimeReceipt {
    Unchecked,
    SinglePass(RegimeSnapshot),
    FinitePasses(FinitePassesReceipt),
    Idr(IdrReceipt),
}

impl fmt::Display for DataRegimeReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unchecked => write!(f, "DATA REGIME\n\nunchecked"),
            Self::SinglePass(snapshot) => write!(
                f,
                "SINGLE PASS STATUS\n\nsamples consumed:      {}\navailable population:  {}\ncoverage:              {:.2}%\n\nPASS",
                snapshot.consumed,
                snapshot.available.expect("finite source"),
                snapshot.coverage().expect("finite source") * 100.0,
            ),
            Self::FinitePasses(receipt) => fmt::Display::fmt(receipt, f),
            Self::Idr(receipt) => fmt::Display::fmt(receipt, f),
        }
    }
}

/// Batches a finite population for exactly N independently shuffled complete passes.
pub struct FinitePassesLoader<T> {
    samples: Vec<T>,
    ids: Vec<u128>,
    order: Vec<usize>,
    batch_size: usize,
    seed: u64,
    pass: usize,
    cursor: usize,
    delivered: usize,
    guard: FinitePasses,
}

impl<T: Clone> FinitePassesLoader<T> {
    pub fn new(samples: Vec<T>, batch_size: usize, passes: usize, seed: u64) -> Result<Self> {
        let ids = (0..samples.len())
            .map(u128::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Self::with_ids(samples, ids, batch_size, passes, seed)
    }

    pub fn with_ids(
        samples: Vec<T>,
        ids: Vec<u128>,
        batch_size: usize,
        passes: usize,
        seed: u64,
    ) -> Result<Self> {
        if samples.is_empty() || samples.len() != ids.len() {
            return Err("finite-pass samples and IDs must be nonempty and equal in length".into());
        }
        if batch_size == 0 {
            return Err("finite-pass batch size must be positive".into());
        }
        let mut loader = Self {
            order: (0..samples.len()).collect(),
            guard: FinitePasses::new(samples.len(), passes)?,
            samples,
            ids,
            batch_size,
            seed,
            pass: 0,
            cursor: 0,
            delivered: 0,
        };
        loader.shuffle();
        Ok(loader)
    }

    fn shuffle(&mut self) {
        let mut state = self
            .seed
            .wrapping_add(self.pass as u64)
            .wrapping_add(1)
            .max(1);
        for end in (1..self.order.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.order.swap(end, state as usize % (end + 1));
        }
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch<T>>> {
        if self.pass == self.guard.receipt().declared_passes {
            self.guard.finish()?;
            return Ok(None);
        }
        let seen_before = self.delivered;
        let end = (self.cursor + self.batch_size).min(self.samples.len());
        let indices = &self.order[self.cursor..end];
        let samples = indices
            .iter()
            .map(|&index| self.samples[index].clone())
            .collect::<Vec<_>>();
        let sample_ids = indices
            .iter()
            .map(|&index| self.ids[index])
            .collect::<Vec<_>>();
        let receipt = self.guard.observe(sample_ids.iter().copied())?;
        self.delivered = self
            .delivered
            .checked_add(samples.len())
            .ok_or("finite-pass delivered-sample count overflow")?;
        self.cursor = end;
        if self.cursor == self.samples.len() {
            self.pass += 1;
            self.cursor = 0;
            if self.pass < receipt.declared_passes {
                self.order = (0..self.samples.len()).collect();
                self.shuffle();
            }
        }
        Ok(Some(Batch {
            samples,
            sample_ids,
            samples_seen_before: seen_before,
            regime: DataRegimeReceipt::FinitePasses(receipt),
        }))
    }

    pub fn receipt(&self) -> FinitePassesReceipt {
        self.guard.receipt()
    }

    pub fn samples_delivered(&self) -> usize {
        self.delivered
    }
}

enum Guard {
    None,
    SinglePass(SinglePass),
    Idr(Idr),
}

pub struct DataLoader<S> {
    source: S,
    batch_size: usize,
    delivered: usize,
    guard: Guard,
}

impl<S: DataSource> DataLoader<S> {
    pub fn new(source: S, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err("DataLoader batch size must be positive".into());
        }
        Ok(Self {
            source,
            batch_size,
            delivered: 0,
            guard: Guard::None,
        })
    }

    pub fn assert_single_pass(mut self) -> Result<Self> {
        let available = self
            .source
            .available()
            .ok_or("single-pass assertion requires a finite source")?;
        self.guard = Guard::SinglePass(SinglePass::new(available)?);
        Ok(self)
    }

    /// Select finite-coverage or generated-stream IDR semantics from the source itself.
    pub fn assert_idr(mut self, limits: IdrLimits) -> Result<Self> {
        self.guard = Guard::Idr(match self.source.available() {
            Some(available) => Idr::fixed(available, limits)?,
            None => Idr::generated(limits)?,
        });
        Ok(self)
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch<S::Item>>> {
        let seen_before = self.delivered;
        let mut samples = Vec::with_capacity(self.batch_size);
        let mut sample_ids = Vec::with_capacity(self.batch_size);
        while samples.len() < self.batch_size {
            let Some(sample) = self.source.next_sample()? else {
                break;
            };
            samples.push(sample.value);
            sample_ids.push(sample.id);
        }
        if samples.is_empty() {
            return Ok(None);
        }
        let regime = match &mut self.guard {
            Guard::None => DataRegimeReceipt::Unchecked,
            Guard::SinglePass(guard) => {
                DataRegimeReceipt::SinglePass(guard.consume(samples.len())?)
            }
            Guard::Idr(guard) => DataRegimeReceipt::Idr(guard.observe(sample_ids.iter().copied())?),
        };
        self.delivered = self
            .delivered
            .checked_add(samples.len())
            .ok_or("DataLoader delivered-sample count overflow")?;
        Ok(Some(Batch {
            samples,
            sample_ids,
            samples_seen_before: seen_before,
            regime,
        }))
    }

    pub fn samples_delivered(&self) -> usize {
        self.delivered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_sources_validate_ids_and_repeat_only_when_requested() -> Result<()> {
        assert!(InMemoryDataset::<u8>::new(vec![]).is_err());
        assert!(InMemoryDataset::with_ids(vec![10, 20], vec![7]).is_err());

        let mut source = InMemoryDataset::with_ids(vec![10, 20], vec![70, 90])?;
        assert_eq!(source.available(), Some(2));
        let first = source.next_sample()?.expect("first sample");
        assert_eq!((first.id, first.value), (70, 10));
        let second = source.next_sample()?.expect("second sample");
        assert_eq!((second.id, second.value), (90, 20));
        assert!(source.next_sample()?.is_none());

        let mut repeated = InMemoryDataset::with_ids(vec![10, 20], vec![70, 90])?.repeat();
        let observed = (0..5)
            .map(|_| {
                let sample = repeated.next_sample()?.expect("repeating source");
                Ok((sample.id, sample.value))
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(observed, [(70, 10), (90, 20), (70, 10), (90, 20), (70, 10)]);
        Ok(())
    }

    #[test]
    fn data_regime_receipts_name_the_guarantee_they_record() -> Result<()> {
        assert_eq!(
            DataRegimeReceipt::Unchecked.to_string(),
            "DATA REGIME\n\nunchecked"
        );

        let mut single = SinglePass::new(4)?;
        let single = DataRegimeReceipt::SinglePass(single.consume(1)?).to_string();
        assert!(single.contains("SINGLE PASS STATUS"), "{single}");
        assert!(single.contains("coverage:              25.00%"), "{single}");

        let mut passes = FinitePasses::new(2, 1)?;
        let finite = DataRegimeReceipt::FinitePasses(passes.observe([1, 2])?).to_string();
        assert!(finite.contains("FINITE PASSES STATUS"), "{finite}");

        let mut idr = Idr::generated(IdrLimits::generated(0.0)?)?;
        let idr = DataRegimeReceipt::Idr(idr.observe([1, 2])?).to_string();
        assert!(idr.contains("IDR STATUS"), "{idr}");
        Ok(())
    }

    #[test]
    fn finite_source_yields_a_short_final_batch_and_stops() -> Result<()> {
        let source = InMemoryDataset::new(vec![10, 20, 30, 40, 50])?;
        let mut loader = DataLoader::new(source, 3)?;
        let first = loader.next_batch()?.unwrap();
        assert_eq!(first.samples, [10, 20, 30]);
        assert_eq!(first.sample_ids, [0, 1, 2]);
        let last = loader.next_batch()?.unwrap();
        assert_eq!(last.samples, [40, 50]);
        assert_eq!(last.samples_seen_before, 3);
        assert!(loader.next_batch()?.is_none());
        Ok(())
    }

    #[test]
    fn idr_violation_never_escapes_the_loader() -> Result<()> {
        let source = InMemoryDataset::new(vec![0, 1, 2, 3])?.repeat();
        let mut loader = DataLoader::new(source, 3)?.assert_idr(IdrLimits::fixed(1.0, 0.0)?)?;
        assert!(loader.next_batch()?.is_some());
        let error = loader.next_batch().unwrap_err().to_string();
        assert!(error.contains("IDR violation"), "{error}");
        assert!(error.contains("repeated IDs:       2"), "{error}");
        assert_eq!(loader.samples_delivered(), 3);
        Ok(())
    }

    #[test]
    fn addition_source_never_exhausts_and_passes_generated_idr() -> Result<()> {
        let source = AdditionDataset::new(42);
        let mut loader = DataLoader::new(source, 7)?.assert_idr(IdrLimits::generated(0.0)?)?;
        for batch_index in 0..100 {
            let batch = loader.next_batch()?.expect("generated source");
            assert_eq!(batch.samples.len(), 7);
            assert_eq!(batch.samples_seen_before, batch_index * 7);
            for sample in batch.samples {
                assert_eq!(sample.sum, sample.left + sample.right);
            }
            assert!(batch.regime.to_string().ends_with("PASS"));
        }
        assert_eq!(loader.samples_delivered(), 700);
        Ok(())
    }

    #[test]
    fn source_kind_and_limit_kind_must_agree() -> Result<()> {
        assert!(DataLoader::new(InMemoryDataset::new(vec![1])?, 0).is_err());
        assert!(
            DataLoader::new(AdditionDataset::new(1), 2)?
                .assert_single_pass()
                .is_err()
        );
        assert!(
            DataLoader::new(AdditionDataset::new(1), 2)?
                .assert_idr(IdrLimits::fixed(0.1, 0.0)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn single_pass_loader_reports_progress_without_reusing_samples() -> Result<()> {
        let source = InMemoryDataset::with_ids(vec![10, 20, 30], vec![7, 8, 9])?;
        let mut loader = DataLoader::new(source, 2)?.assert_single_pass()?;

        let first = loader.next_batch()?.expect("first batch");
        assert_eq!(first.samples, [10, 20]);
        assert_eq!(first.sample_ids, [7, 8]);
        assert_eq!(first.samples_seen_before, 0);
        assert!(first.regime.to_string().contains("66.67%"));

        let last = loader.next_batch()?.expect("short final batch");
        assert_eq!(last.samples, [30]);
        assert_eq!(last.samples_seen_before, 2);
        assert!(last.regime.to_string().contains("100.00%"));
        assert_eq!(loader.samples_delivered(), 3);
        assert!(loader.next_batch()?.is_none());
        Ok(())
    }

    #[test]
    fn finite_pass_loader_validates_inputs_and_preserves_custom_id_sets() -> Result<()> {
        assert!(FinitePassesLoader::<u8>::new(vec![], 1, 1, 0).is_err());
        assert!(FinitePassesLoader::with_ids(vec![1, 2], vec![10], 1, 1, 0).is_err());
        assert!(FinitePassesLoader::new(vec![1, 2], 0, 1, 0).is_err());
        assert!(FinitePassesLoader::new(vec![1, 2], 1, 0, 0).is_err());

        let mut loader =
            FinitePassesLoader::with_ids(vec!["a", "b", "c"], vec![100, 200, 300], 8, 2, 17)?;
        for expected_pass in 1..=2 {
            let batch = loader.next_batch()?.expect("complete pass");
            assert_eq!(batch.samples.len(), 3);
            let mut ids = batch.sample_ids;
            ids.sort_unstable();
            assert_eq!(ids, [100, 200, 300]);
            let DataRegimeReceipt::FinitePasses(receipt) = batch.regime else {
                panic!("finite-pass receipt expected");
            };
            assert_eq!(receipt.completed_passes, expected_pass);
        }
        assert!(loader.next_batch()?.is_none());
        Ok(())
    }

    #[test]
    fn finite_pass_loader_never_crosses_a_pass_boundary() -> Result<()> {
        let mut loader = FinitePassesLoader::new(vec![10, 20, 30, 40, 50], 3, 2, 7)?;
        let sizes = [3, 2, 3, 2];
        let mut completed = vec![];
        for size in sizes {
            let batch = loader.next_batch()?.expect("declared batch");
            assert_eq!(batch.samples.len(), size);
            let DataRegimeReceipt::FinitePasses(receipt) = batch.regime else {
                panic!("finite-pass receipt expected");
            };
            completed.push(receipt.completed_passes);
        }
        assert_eq!(completed, [0, 1, 1, 2]);
        assert!(loader.next_batch()?.is_none());
        assert_eq!(loader.samples_delivered(), 10);
        assert_eq!(loader.receipt().observations, 10);
        Ok(())
    }
}
