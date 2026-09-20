//! Experimental training with named axes on a stream-ordered cuTile GPU backend.
mod axis;
mod backend;
mod data;
mod metrics;
mod nn;
mod preprocess;
mod regime;
mod tensor;
mod train;

pub use axis::{Axis, Dim, IntoAxes, Shape};
pub use backend::Device;
pub use data::{
    AdditionDataset, AdditionSample, Batch, DataLoader, DataRegimeReceipt, DataSource,
    FinitePassesLoader, InMemoryDataset, Sample,
};
pub use metrics::CategoricalAccuracy;
pub use nn::{
    Adam, AdamW, Conv2d, GELU, IntoLayers, LayerNorm, Linear, Module, ParamId, Parameter,
    PopulationLinear, ReLU, SGD, Sequential,
};
pub use preprocess::Standardizer;
pub use regime::{
    FinitePasses, FinitePassesReceipt, Idr, IdrLimits, IdrReceipt, RegimeSnapshot, SinglePass,
    TrainEvalDisjoint, TrainEvalReceipt,
};
pub use tensor::Tensor;
pub use train::{Optimizer, TrainStep, Trainer};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::{
        Adam, AdamW, AdditionDataset, AdditionSample, Axis, Batch, CategoricalAccuracy, Conv2d,
        DataLoader, DataRegimeReceipt, DataSource, Device, Dim, FinitePasses, FinitePassesLoader,
        FinitePassesReceipt, GELU, Idr, IdrLimits, IdrReceipt, InMemoryDataset, LayerNorm, Linear,
        Module, Optimizer, ParamId, Parameter, PopulationLinear, ReLU, RegimeSnapshot, Result, SGD,
        Sample, Sequential, Shape, SinglePass, Standardizer, Tensor, TrainEvalDisjoint,
        TrainEvalReceipt, TrainStep, Trainer,
    };
}
