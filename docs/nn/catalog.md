# The torch.nn catalog, row by row

`classes 161 · yes 20 · partial 7 · refused 0 · no 134`

The denominator is every class in the layer sections of PyTorch 2.14's
[`torch.nn` page](https://docs.pytorch.org/docs/2.14/nn.html) on 2026-09-22:
Containers through DataParallel, plus `Flatten` and `Unflatten` from the
utilities table. The 40 utility *functions* and `LazyModuleMixin` are not rows.
A verdict is about the six-point definition in
[module-backlog.md](../direction/module-backlog.md): **yes** = the named-axis
module or tensor-level function exists with forward and gradient tests on
CUDA; **partial** = exists with a named gap; **refused** = a doc says never,
scores zero, stays in the denominator (none at the freeze: the backlog defers,
it does not refuse); **no** = absent. `#open` names an open Axis issue.

| Section | Class | Verdict | Evidence |
|---|---|---|---|
| Containers | `Module` | yes | the `Module` trait: named-axis contract, `named_parameters`, `forward` (`src/library/src/model/nn.rs`) |
| Containers | `Sequential` | yes | `Sequential` composes modules; used by every training example |
| Containers | `ModuleList` | no | absent |
| Containers | `ModuleDict` | no | absent |
| Containers | `ParameterList` | no | absent |
| Containers | `ParameterDict` | no | absent |
| Convolution Layers | `Conv1d` | no | absent |
| Convolution Layers | `Conv2d` | yes | named-axis `Conv2d` (`model/convolution.rs`); CUDA witness for overlapping patches, forward and reverse (docs/contracts/coverage.md) |
| Convolution Layers | `Conv3d` | yes | named-axis `Conv3d` (`model/convolution.rs`) with CUDA tests |
| Convolution Layers | `ConvTranspose1d` | no | absent |
| Convolution Layers | `ConvTranspose2d` | no | absent |
| Convolution Layers | `ConvTranspose3d` | no | absent |
| Convolution Layers | `LazyConv1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConv2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConv3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConvTranspose1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConvTranspose2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConvTranspose3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `Unfold` | no | absent |
| Convolution Layers | `Fold` | no | absent |
| Pooling layers | `MaxPool1d` | no | #open: pooling family |
| Pooling layers | `MaxPool2d` | no | #open: pooling family |
| Pooling layers | `MaxPool3d` | no | #open: pooling family |
| Pooling layers | `MaxUnpool1d` | no | absent |
| Pooling layers | `MaxUnpool2d` | no | absent |
| Pooling layers | `MaxUnpool3d` | no | absent |
| Pooling layers | `AvgPool1d` | no | #open: pooling family |
| Pooling layers | `AvgPool2d` | no | #open: pooling family |
| Pooling layers | `AvgPool3d` | no | #open: pooling family |
| Pooling layers | `FractionalMaxPool2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `FractionalMaxPool3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `LPPool1d` | no | absent |
| Pooling layers | `LPPool2d` | no | absent |
| Pooling layers | `LPPool3d` | no | absent |
| Pooling layers | `AdaptiveMaxPool1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `AdaptiveMaxPool2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `AdaptiveMaxPool3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `AdaptiveAvgPool1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `AdaptiveAvgPool2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `AdaptiveAvgPool3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Padding Layers | `ReflectionPad1d` | no | absent |
| Padding Layers | `ReflectionPad2d` | no | absent |
| Padding Layers | `ReflectionPad3d` | no | absent |
| Padding Layers | `ReplicationPad1d` | no | absent |
| Padding Layers | `ReplicationPad2d` | no | absent |
| Padding Layers | `ReplicationPad3d` | no | absent |
| Padding Layers | `ZeroPad1d` | no | absent |
| Padding Layers | `ZeroPad2d` | no | absent |
| Padding Layers | `ZeroPad3d` | no | absent |
| Padding Layers | `ConstantPad1d` | no | absent |
| Padding Layers | `ConstantPad2d` | no | absent |
| Padding Layers | `ConstantPad3d` | no | absent |
| Padding Layers | `CircularPad1d` | no | absent |
| Padding Layers | `CircularPad2d` | no | absent |
| Padding Layers | `CircularPad3d` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `ELU` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardshrink` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardsigmoid` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardtanh` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardswish` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `LeakyReLU` | yes | `LeakyReLU` (`model/nn.rs`), landed from vision consumer pressure |
| Non-linear Activations (weighted sum, nonlinearity) | `LogSigmoid` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `MultiheadAttention` | partial | `CausalAttention` and attention composition (`examples/training/attention`); no general heads-and-mask module |
| Non-linear Activations (weighted sum, nonlinearity) | `PReLU` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `ReLU` | yes | `ReLU` (`model/nn.rs`); MLP, CNN and MNIST acceptances |
| Non-linear Activations (weighted sum, nonlinearity) | `ReLU6` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `RReLU` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `SELU` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `CELU` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `GELU` | yes | `GELU` (tanh form) and `ExactGELU` (`model/nn.rs`); open issue asks the docs to say which is PyTorch's default |
| Non-linear Activations (weighted sum, nonlinearity) | `Sigmoid` | yes | `Tensor::sigmoid`; the BCE-with-logits witness exercises it |
| Non-linear Activations (weighted sum, nonlinearity) | `SiLU` | yes | `SiLU` (`model/nn.rs`) |
| Non-linear Activations (weighted sum, nonlinearity) | `Mish` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Softplus` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Softshrink` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Softsign` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Tanh` | yes | `Tanh` (`model/nn.rs`) |
| Non-linear Activations (weighted sum, nonlinearity) | `Tanhshrink` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `Threshold` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `GLU` | no | absent |
| Non-linear Activations (other) | `Softmin` | no | absent |
| Non-linear Activations (other) | `Softmax` | yes | `Tensor::softmax` over a named axis; causal softmax has a hand-checked CUDA witness |
| Non-linear Activations (other) | `Softmax2d` | no | absent |
| Non-linear Activations (other) | `LogSoftmax` | no | #open: numerically stable logsumexp reduction |
| Non-linear Activations (other) | `AdaptiveLogSoftmaxWithLoss` | no | absent |
| Normalization Layers | `BatchNorm1d` | no | #open: trained BatchNorm with persistent running statistics |
| Normalization Layers | `BatchNorm2d` | no | same open issue |
| Normalization Layers | `BatchNorm3d` | no | same open issue |
| Normalization Layers | `LazyBatchNorm1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Normalization Layers | `LazyBatchNorm2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Normalization Layers | `LazyBatchNorm3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Normalization Layers | `GroupNorm` | yes | named-axis `GroupNorm` (`model/normalization.rs`) |
| Normalization Layers | `SyncBatchNorm` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Normalization Layers | `InstanceNorm1d` | partial | stateless named-axis `InstanceNorm`; no running statistics (needs the stateful contract) |
| Normalization Layers | `InstanceNorm2d` | partial | same stateless `InstanceNorm` |
| Normalization Layers | `InstanceNorm3d` | partial | same stateless `InstanceNorm` |
| Normalization Layers | `LazyInstanceNorm1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Normalization Layers | `LazyInstanceNorm2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Normalization Layers | `LazyInstanceNorm3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Normalization Layers | `LayerNorm` | yes | named-axis `LayerNorm` (`model/normalization.rs`) |
| Normalization Layers | `LocalResponseNorm` | no | absent |
| Normalization Layers | `RMSNorm` | yes | `RmsNorm` (`model/normalization.rs`) |
| Recurrent Layers | `RNNBase` | no | absent |
| Recurrent Layers | `RNN` | no | absent |
| Recurrent Layers | `LSTM` | partial | `Lstm` (`model/recurrent.rs`): eager IFGO correctness; fused recurrence, direction and layer stacking remain (backlog) |
| Recurrent Layers | `GRU` | no | absent |
| Recurrent Layers | `RNNCell` | no | absent |
| Recurrent Layers | `LSTMCell` | partial | `LstmCell` with forward and gradient oracles in `recurrent_tests.rs`; not yet under the six-point definition's CUDA geometry coverage |
| Recurrent Layers | `GRUCell` | no | absent |
| Transformer Layers | `Transformer` | no | absent |
| Transformer Layers | `TransformerEncoder` | no | absent |
| Transformer Layers | `TransformerDecoder` | no | absent |
| Transformer Layers | `TransformerEncoderLayer` | no | absent |
| Transformer Layers | `TransformerDecoderLayer` | no | absent |
| Linear Layers | `Identity` | no | absent |
| Linear Layers | `Linear` | yes | `Linear` with `.bias(false)`; `PopulationLinear` beside it |
| Linear Layers | `Bilinear` | no | absent |
| Linear Layers | `LazyLinear` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Dropout Layers | `Dropout` | no | #open: dropout and explicit train/eval mode |
| Dropout Layers | `Dropout1d` | no | absent |
| Dropout Layers | `Dropout2d` | no | absent |
| Dropout Layers | `Dropout3d` | no | absent |
| Dropout Layers | `AlphaDropout` | no | absent |
| Dropout Layers | `FeatureAlphaDropout` | no | absent |
| Sparse Layers | `Embedding` | yes | one-hot `Embedding` (`model/nn.rs`); arbitrary-index `gather` for large tables |
| Sparse Layers | `EmbeddingBag` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Distance Functions | `CosineSimilarity` | no | absent |
| Distance Functions | `PairwiseDistance` | no | #open: outer-broadcast for elementwise ops |
| Loss Functions | `L1Loss` | no | #open: elementwise absolute value / L1 loss |
| Loss Functions | `MSELoss` | no | absent |
| Loss Functions | `CrossEntropyLoss` | yes | `categorical_cross_entropy_with_logits` |
| Loss Functions | `LinearCrossEntropyLoss` | no | absent |
| Loss Functions | `LinearCrossEntropyOptions` | no | absent |
| Loss Functions | `CTCLoss` | no | absent |
| Loss Functions | `NLLLoss` | no | absent |
| Loss Functions | `PoissonNLLLoss` | no | absent |
| Loss Functions | `GaussianNLLLoss` | no | absent |
| Loss Functions | `KLDivLoss` | no | absent |
| Loss Functions | `BCELoss` | no | absent |
| Loss Functions | `BCEWithLogitsLoss` | yes | `binary_cross_entropy_with_logits` and the weighted form; hand-checked CUDA witness |
| Loss Functions | `MarginRankingLoss` | no | absent |
| Loss Functions | `HingeEmbeddingLoss` | no | absent |
| Loss Functions | `MultiLabelMarginLoss` | no | absent |
| Loss Functions | `HuberLoss` | no | absent |
| Loss Functions | `SmoothL1Loss` | no | absent |
| Loss Functions | `SoftMarginLoss` | no | absent |
| Loss Functions | `MultiLabelSoftMarginLoss` | no | absent |
| Loss Functions | `CosineEmbeddingLoss` | no | absent |
| Loss Functions | `MultiMarginLoss` | no | absent |
| Loss Functions | `TripletMarginLoss` | no | absent |
| Loss Functions | `TripletMarginWithDistanceLoss` | no | absent |
| Vision Layers | `PixelShuffle` | no | absent |
| Vision Layers | `PixelUnshuffle` | no | absent |
| Vision Layers | `Upsample` | partial | nearest and bilinear only; no trilinear or bicubic, no scale-factor form |
| Vision Layers | `UpsamplingNearest2d` | yes | `Tensor::upsample_nearest`, named-axis |
| Vision Layers | `UpsamplingBilinear2d` | yes | `Tensor::resample_bilinear`, named-axis |
| Shuffle Layers | `ChannelShuffle` | no | absent |
| DataParallel Layers (multi-GPU, distributed) | `DataParallel` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| DataParallel Layers (multi-GPU, distributed) | `DistributedDataParallel` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Utilities (modules) | `Flatten` | no | absent |
| Utilities (modules) | `Unflatten` | no | absent |

Score: mean of strict (20/161) and half-credit ((20+7/2)/161) = 13.51% = 1351 bp.
