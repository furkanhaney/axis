//! Experimental training with named axes on a stream-ordered cuTile GPU backend.
#[path = "algebra/axis.rs"]
mod axis;
#[cfg(feature = "cuda")]
#[path = "runtime/backend.rs"]
mod backend;
#[cfg(all(not(feature = "cuda"), axis_docs_rs))]
#[path = "runtime/backend_docs.rs"]
mod backend;
#[path = "research/data.rs"]
mod data;
#[path = "research/disjointness.rs"]
mod disjointness;
#[path = "model/metrics.rs"]
mod metrics;
#[path = "research/monotonicity.rs"]
mod monotonicity;
#[path = "model/nn.rs"]
mod nn;
#[path = "model/preprocess.rs"]
mod preprocess;
#[path = "research/regime.rs"]
mod regime;
#[path = "algebra/tensor.rs"]
mod tensor;
#[path = "runtime/train.rs"]
mod train;

pub use axis::{Axis, Dim, IntoAxes, Shape};
pub use backend::Device;
pub use data::{
    AdditionDataset, AdditionSample, Batch, DataLoader, DataRegimeReceipt, DataSource,
    FinitePassesLoader, InMemoryDataset, Sample,
};
pub use disjointness::{
    Disjointness, DisjointnessEvidence, DisjointnessReceipt, IdentityScheme, PopulationMode,
    PopulationReceipt, PopulationSpec,
};
pub use metrics::CategoricalAccuracy;
pub use monotonicity::{
    EmpiricalMonotonicity, EmpiricalMonotonicityReceipt, MonotoneDirection, MonotonicityLimits,
};
pub use nn::{
    Adam, AdamW, Conv2d, GELU, IntoLayers, LayerNorm, Linear, Module, ParamId, Parameter,
    PopulationLinear, ReLU, SGD, Sequential,
};
pub use preprocess::Standardizer;
pub use regime::{
    FinitePasses, FinitePassesReceipt, Idr, IdrLimits, IdrReceipt, RegimeSnapshot, SinglePass,
};
pub use tensor::Tensor;
pub use train::{Optimizer, TrainStep, Trainer};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(all(not(feature = "cuda"), not(axis_docs_rs)))]
compile_error!(
    "Axis requires its default `cuda` feature; disabling it is reserved for the docs.rs build"
);

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::{
        Adam, AdamW, AdditionDataset, AdditionSample, Axis, Batch, CategoricalAccuracy, Conv2d,
        DataLoader, DataRegimeReceipt, DataSource, Device, Dim, Disjointness, DisjointnessEvidence,
        DisjointnessReceipt, EmpiricalMonotonicity, EmpiricalMonotonicityReceipt, FinitePasses,
        FinitePassesLoader, FinitePassesReceipt, GELU, IdentityScheme, Idr, IdrLimits, IdrReceipt,
        InMemoryDataset, LayerNorm, Linear, Module, MonotoneDirection, MonotonicityLimits,
        Optimizer, ParamId, Parameter, PopulationLinear, PopulationMode, PopulationReceipt,
        PopulationSpec, ReLU, RegimeSnapshot, Result, SGD, Sample, Sequential, Shape, SinglePass,
        Standardizer, Tensor, TrainStep, Trainer,
    };
}
