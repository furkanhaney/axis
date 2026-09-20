//! Minimal step ordering shared by concrete training programs.
use crate::{Adam, AdamW, Module, Result, SGD, Tensor};

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

pub struct Trainer<O> {
    optimizer: O,
    completed_steps: usize,
}

impl<O: Optimizer> Trainer<O> {
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            completed_steps: 0,
        }
    }

    /// Clear gradients, construct a fresh scalar loss, run backward, then update.
    pub fn step<M, F>(&mut self, model: &mut M, loss: F) -> Result<TrainStep>
    where
        M: Module,
        F: FnOnce(&M) -> Result<Tensor>,
    {
        let next = self
            .completed_steps
            .checked_add(1)
            .ok_or("trainer step count overflow")?;
        model.zero_grad();
        let loss = loss(model)?;
        if loss.shape().rank() != 0 {
            return Err(
                "Trainer loss closure must return a scalar; reduce loss axes explicitly".into(),
            );
        }
        loss.backward()?;
        self.optimizer.step(model)?;
        // The eager graph enqueues one stream-ordered step. Synchronize once
        // here instead of after every allocation and kernel launch.
        loss.device().synchronize()?;
        self.completed_steps = next;
        Ok(TrainStep {
            step: next,
            pre_update_loss: loss,
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
}

impl TrainStep {
    pub fn step(&self) -> usize {
        self.step
    }

    /// Synchronize and read the loss produced before this step's update.
    pub fn pre_update_loss(&self) -> Result<f32> {
        self.pre_update_loss.item()
    }
}
