mod data;
mod model;

pub use data::{ImageNet64, Sample, prepare_npz_shards};
pub use model::{Axes, Classifier, tensors};
