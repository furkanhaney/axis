//! Minimal step ordering shared by concrete training programs.
use crate::{Adam, AdamW, Module, Muon, MuonWithAuxAdamW, Result, SGD, State, Tensor};
use std::{cell::RefCell, rc::Rc, time::Instant};

fn profile(label: &str, started: Instant) {
    if std::env::var_os("AXIS_PROFILE").is_some() {
        eprintln!(
            "axis_profile {label} {:.6}",
            started.elapsed().as_secs_f64()
        );
    }
}
pub trait Optimizer {
    fn step<M: Module>(&mut self, model: &mut M) -> Result<()>;
}

impl Optimizer for SGD {
    fn step<M: Module>(&mut self, model: &mut M) -> Result<()> {
        SGD::step(self, model)
    }
}

impl Optimizer for Adam {
    fn step<M: Module>(&mut self, model: &mut M) -> Result<()> {
        Adam::step(self, model)
    }
}

impl Optimizer for AdamW {
    fn step<M: Module>(&mut self, model: &mut M) -> Result<()> {
        AdamW::step(self, model)
    }
}

impl Optimizer for Muon {
    fn step<M: Module>(&mut self, model: &mut M) -> Result<()> {
        Muon::step(self, model)
    }
}

impl Optimizer for MuonWithAuxAdamW {
    fn step<M: Module>(&mut self, model: &mut M) -> Result<()> {
        MuonWithAuxAdamW::step(self, model)
    }
}

/// SplitMix64's output mixer (http://xoshiro.di.unimi.it/splitmix64.c): three
/// xor/multiply rounds that avalanche an arbitrary 64-bit input into a
/// well-distributed 64-bit output. [`TrainingPass::seed`] applies it to
/// `run_seed ^ step_index.wrapping_mul(GOLDEN)`, and [`TrainingPass::next_seed`]
/// applies it again to `seed ^ draw_counter.wrapping_mul(GOLDEN)`; both call
/// sites share this one mixer and the one golden-ratio constant.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The explicit per-step training context threaded through
/// [`Module::forward_training`]. `Trainer::step_training` builds one from
/// `(run_seed, completed_steps)`; a caller driving `forward_training` outside
/// the `Trainer` -- a test, or a custom loop -- builds one directly with
/// [`TrainingPass::new`].
///
/// `seed` is this step's pass seed: `splitmix64(run_seed ^
/// step_index.wrapping_mul(0x9E3779B97F4A7C15))`. Each random consumer then
/// calls [`TrainingPass::next_seed`], which returns
/// `splitmix64(seed ^ draw_counter.wrapping_mul(0x9E3779B97F4A7C15))` for the
/// current draw counter and increments the counter. So the same model, run
/// seed, step and forward order give bit-identical draws, and two random
/// consumers in one step draw different values.
///
/// Also carries the pending list of persistent-state updates a
/// `forward_training` call queues with [`TrainingPass::stage`] --
/// `BatchNorm`-style running statistics. Nothing writes a [`State`] during
/// forward: [`TrainingPass::commit`] is the only writer, called by
/// `Trainer::step_training` after the optimizer step succeeds, so a forward
/// pass stays side-effect free and a failed step leaves every `State`
/// untouched.
pub struct TrainingPass {
    seed: u64,
    draws: u64,
    pending: Vec<(State, Tensor)>,
}

impl TrainingPass {
    /// Build a pass directly from its seed, for driving `forward_training`
    /// outside the `Trainer` -- a test or a custom loop.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            draws: 0,
            pending: Vec::new(),
        }
    }

    fn from_run(run_seed: u64, step_index: u64) -> Self {
        Self::new(splitmix64(run_seed ^ step_index.wrapping_mul(GOLDEN)))
    }

    /// This pass's own seed, the value `Trainer::step_training` records as
    /// `TrainStep::pass_seed`.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// A distinct deterministic seed for the next random draw in this pass,
    /// derived from `(seed, counter)`. Advances the counter, so repeated
    /// calls in one pass never repeat a seed.
    pub fn next_seed(&mut self) -> u64 {
        let draw = self.draws;
        self.draws = self.draws.wrapping_add(1);
        splitmix64(self.seed ^ draw.wrapping_mul(GOLDEN))
    }

    /// Queue a persistent-state update for `state` without writing it: a
    /// `forward_training` implementation (`BatchNorm`'s running statistics)
    /// calls this instead of ever mutating a [`State`] directly. Updates
    /// commit in staging order.
    pub fn stage(&mut self, state: State, value: Tensor) {
        self.pending.push((state, value));
    }

    /// Write every staged update to its `State`, in staging order, detaching
    /// each new value first. `Trainer::step_training` calls this once, after
    /// its optimizer step succeeds; public so a custom training loop driving
    /// `forward_training` directly can do the same. Returns how many updates
    /// were committed.
    pub fn commit(&mut self) -> usize {
        let count = self.pending.len();
        for (state, value) in self.pending.drain(..) {
            state.replace(value);
        }
        count
    }
}

pub struct Trainer<O> {
    optimizer: O,
    completed_steps: usize,
    run_seed: Option<u64>,
}

impl<O: Optimizer> Trainer<O> {
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            completed_steps: 0,
            run_seed: None,
        }
    }

    /// Set this trainer's run seed, enabling `step_training`. Builder style:
    /// `Trainer::new(optimizer).with_seed(1234)`.
    pub fn with_seed(mut self, run_seed: u64) -> Self {
        self.run_seed = Some(run_seed);
        self
    }

    /// Clear gradients, construct a fresh scalar loss, run backward, then update.
    pub fn step<M, F>(&mut self, model: &mut M, loss: F) -> Result<TrainStep>
    where
        M: Module,
        F: FnOnce(&M) -> Result<Tensor>,
    {
        self.run_step(model, None, loss)
    }

    /// Same ordering as `step`, plus an explicit [`TrainingPass`] built from
    /// this trainer's run seed and completed-step count, threaded to `loss`
    /// so it can drive `Module::forward_training`. Errors before any device
    /// work if `with_seed` was never called.
    ///
    /// Commits the pass's staged persistent-state updates
    /// (`TrainStep::committed_states`) once the whole step -- loss, backward,
    /// optimizer step -- has succeeded, and only then: an error anywhere
    /// before that point returns early and never commits, leaving every
    /// [`State`] exactly as it was.
    pub fn step_training<M, F>(&mut self, model: &mut M, loss: F) -> Result<TrainStep>
    where
        M: Module,
        F: FnOnce(&M, &mut TrainingPass) -> Result<Tensor>,
    {
        let run_seed = self
            .run_seed
            .ok_or("Trainer::step_training requires with_seed before use")?;
        let pass = Rc::new(RefCell::new(TrainingPass::from_run(
            run_seed,
            self.completed_steps as u64,
        )));
        let pass_seed = pass.borrow().seed();
        let closure_pass = pass.clone();
        let mut step = self.run_step(model, Some(pass_seed), move |model| {
            loss(model, &mut closure_pass.borrow_mut())
        })?;
        step.committed_states = pass.borrow_mut().commit();
        Ok(step)
    }

    /// The update ordering both `step` and `step_training` share: zero
    /// gradients, construct the scalar loss, backward, optimizer step,
    /// synchronize once at the boundary.
    fn run_step<M, F>(
        &mut self,
        model: &mut M,
        pass_seed: Option<u64>,
        loss: F,
    ) -> Result<TrainStep>
    where
        M: Module,
        F: FnOnce(&M) -> Result<Tensor>,
    {
        let next = self
            .completed_steps
            .checked_add(1)
            .ok_or("trainer step count overflow")?;
        let started = Instant::now();
        model.zero_grad();
        profile("train_zero_grad", started);
        let started = Instant::now();
        let loss = loss(model)?;
        profile("train_loss", started);
        if loss.shape().rank() != 0 {
            return Err(
                "Trainer loss closure must return a scalar; reduce loss axes explicitly".into(),
            );
        }
        let started = Instant::now();
        loss.backward()?;
        profile("train_backward", started);
        let started = Instant::now();
        self.optimizer.step(model)?;
        profile("train_optimizer", started);
        // The eager graph enqueues one stream-ordered step. Synchronize once
        // here instead of after every allocation and kernel launch.
        let started = Instant::now();
        loss.device().synchronize()?;
        profile("train_boundary", started);
        self.completed_steps = next;
        Ok(TrainStep {
            step: next,
            pre_update_loss: loss,
            pass_seed,
            committed_states: 0,
        })
    }

    pub fn completed_steps(&self) -> usize {
        self.completed_steps
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

pub struct TrainStep {
    step: usize,
    pre_update_loss: Tensor,
    pass_seed: Option<u64>,
    committed_states: usize,
}

impl TrainStep {
    pub fn step(&self) -> usize {
        self.step
    }

    /// Synchronize and read the loss produced before this step's update.
    pub fn pre_update_loss(&self) -> Result<f32> {
        self.pre_update_loss.item()
    }

    /// The `TrainingPass` seed this step used: `None` for `step`, `Some` for
    /// `step_training`. The receipt `docs/direction/next.md` asks for.
    pub fn pass_seed(&self) -> Option<u64> {
        self.pass_seed
    }

    /// How many staged persistent-state updates `step_training` committed
    /// after this step's optimizer step succeeded. Always `0` for `step`,
    /// which never builds a `TrainingPass`.
    pub fn committed_states(&self) -> usize {
        self.committed_states
    }
}
