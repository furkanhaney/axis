# The torch.nn catalog, row by row

`classes 161 · yes 87 · partial 33 · refused 0 · no 41`

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
| Containers | `ModuleList` | yes | #129: aggregates named_parameters in Sequential's slot paths; forward errors like PyTorch |
| Containers | `ModuleDict` | yes | #129: same |
| Containers | `ParameterList` | yes | #129: same |
| Containers | `ParameterDict` | yes | #129: same |
| Convolution Layers | `Conv1d` | yes | #131: unit-axis lift of Conv2d, exact, scalar oracle |
| Convolution Layers | `Conv2d` | yes | named-axis `Conv2d` (`model/convolution.rs`); CUDA witness for overlapping patches, forward and reverse (docs/contracts/coverage.md) |
| Convolution Layers | `Conv3d` | yes | named-axis `Conv3d` (`model/convolution.rs`) with CUDA tests |
| Convolution Layers | `ConvTranspose1d` | yes | #131: scalar oracle, reuses ConvTranspose2d exactly |
| Convolution Layers | `ConvTranspose2d` | yes | #131: fwd + all gradients, reordered storage |
| Convolution Layers | `ConvTranspose3d` | yes | #131: fwd + all gradients |
| Convolution Layers | `LazyConv1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConv2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConv3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConvTranspose1d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConvTranspose2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `LazyConvTranspose3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Convolution Layers | `Unfold` | yes | #131: patches + gradient, reordered storage |
| Convolution Layers | `Fold` | yes | #131: col2im, adjoint of Unfold |
| Pooling layers | `MaxPool1d` | yes | #130: unit-axis lift over MaxPool2d path, oracle |
| Pooling layers | `MaxPool2d` | yes | #130: audit: asymmetric-kernel CUDA test added |
| Pooling layers | `MaxPool3d` | yes | #130: audit: reordered-storage CUDA test added |
| Pooling layers | `MaxUnpool1d` | no | absent |
| Pooling layers | `MaxUnpool2d` | no | absent |
| Pooling layers | `MaxUnpool3d` | no | absent |
| Pooling layers | `AvgPool1d` | partial | #130: count_include_pad=True only; no ceil_mode/divisor_override |
| Pooling layers | `AvgPool2d` | partial | #130: same |
| Pooling layers | `AvgPool3d` | partial | #130: same |
| Pooling layers | `FractionalMaxPool2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `FractionalMaxPool3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `LPPool1d` | partial | #130: integer p only |
| Pooling layers | `LPPool2d` | partial | #130: integer p only |
| Pooling layers | `LPPool3d` | partial | #130: integer p only |
| Pooling layers | `AdaptiveMaxPool1d` | partial | #130: separable gather+max; cross-axis tie order may differ from PyTorch |
| Pooling layers | `AdaptiveMaxPool2d` | partial | #130: same |
| Pooling layers | `AdaptiveMaxPool3d` | partial | #130: same |
| Pooling layers | `AdaptiveAvgPool1d` | yes | #130: PyTorch bin formula, oracle |
| Pooling layers | `AdaptiveAvgPool2d` | yes | #130: same |
| Pooling layers | `AdaptiveAvgPool3d` | yes | #130: module over generalized adaptive_avg_pool |
| Padding Layers | `ReflectionPad1d` | yes | #128: generic named-axis ReflectionPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ReflectionPad2d` | yes | #128: generic named-axis ReflectionPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ReflectionPad3d` | yes | #128: generic named-axis ReflectionPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ReplicationPad1d` | yes | #128: generic named-axis ReplicationPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ReplicationPad2d` | yes | #128: generic named-axis ReplicationPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ReplicationPad3d` | yes | #128: generic named-axis ReplicationPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ZeroPad1d` | yes | #128: generic named-axis ZeroPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ZeroPad2d` | yes | #128: generic named-axis ZeroPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ZeroPad3d` | yes | #128: generic named-axis ZeroPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ConstantPad1d` | yes | #128: generic named-axis ConstantPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ConstantPad2d` | yes | #128: generic named-axis ConstantPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `ConstantPad3d` | yes | #128: generic named-axis ConstantPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `CircularPad1d` | yes | #128: generic named-axis CircularPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `CircularPad2d` | yes | #128: generic named-axis CircularPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Padding Layers | `CircularPad3d` | yes | #128: generic named-axis CircularPad module; oracle across 1d/2d/3d from PyTorch 2.14 runs |
| Non-linear Activations (weighted sum, nonlinearity) | `ELU` | partial | #123: CUDA oracle fwd+grad; alpha restricted to finite positive (PyTorch allows any nonzero) |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardshrink` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardsigmoid` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardtanh` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `Hardswish` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `LeakyReLU` | yes | `LeakyReLU` (`model/nn.rs`), landed from vision consumer pressure |
| Non-linear Activations (weighted sum, nonlinearity) | `LogSigmoid` | yes | #123: CUDA oracle |
| Non-linear Activations (weighted sum, nonlinearity) | `MultiheadAttention` | partial | `CausalAttention` and attention composition (`examples/training/attention`); no general heads-and-mask module |
| Non-linear Activations (weighted sum, nonlinearity) | `PReLU` | yes | #123: shared and per-channel weight, weight-gradient oracle |
| Non-linear Activations (weighted sum, nonlinearity) | `ReLU` | yes | `ReLU` (`model/nn.rs`); MLP, CNN and MNIST acceptances |
| Non-linear Activations (weighted sum, nonlinearity) | `ReLU6` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `RReLU` | no | absent |
| Non-linear Activations (weighted sum, nonlinearity) | `SELU` | yes | #123: CUDA oracle fwd+grad, reordered storage |
| Non-linear Activations (weighted sum, nonlinearity) | `CELU` | partial | #123: CUDA oracle fwd+grad; alpha restricted to finite positive |
| Non-linear Activations (weighted sum, nonlinearity) | `GELU` | yes | `GELU` (tanh form) and `ExactGELU` (`model/nn.rs`); open issue asks the docs to say which is PyTorch's default |
| Non-linear Activations (weighted sum, nonlinearity) | `Sigmoid` | yes | `Tensor::sigmoid`; the BCE-with-logits witness exercises it |
| Non-linear Activations (weighted sum, nonlinearity) | `SiLU` | yes | `SiLU` (`model/nn.rs`) |
| Non-linear Activations (weighted sum, nonlinearity) | `Mish` | yes | #123: CUDA oracle |
| Non-linear Activations (weighted sum, nonlinearity) | `Softplus` | yes | #123: module over exact softplus kernel |
| Non-linear Activations (weighted sum, nonlinearity) | `Softshrink` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `Softsign` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `Tanh` | yes | `Tanh` (`model/nn.rs`) |
| Non-linear Activations (weighted sum, nonlinearity) | `Tanhshrink` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `Threshold` | yes | #124: CUDA oracle fwd+grad incl. PyTorch ATen kink conventions |
| Non-linear Activations (weighted sum, nonlinearity) | `GLU` | yes | #123: CUDA oracle, narrow-split on named axis |
| Non-linear Activations (other) | `Softmin` | yes | #123: oracle |
| Non-linear Activations (other) | `Softmax` | yes | `Tensor::softmax` over a named axis; causal softmax has a hand-checked CUDA witness |
| Non-linear Activations (other) | `Softmax2d` | yes | #123: softmax over named channel axis, oracle |
| Non-linear Activations (other) | `LogSoftmax` | yes | #123: x - logsumexp, oracle |
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
| Normalization Layers | `LocalResponseNorm` | yes | #129: composed, oracle, PyTorch defaults |
| Normalization Layers | `RMSNorm` | yes | `RmsNorm` (`model/normalization.rs`) |
| Recurrent Layers | `RNNBase` | no | absent |
| Recurrent Layers | `RNN` | partial | #122: rnn_matches_independent_f64_forward_and_all_central_differences, rnn_relu_executes_asymmetric_geometry_with_reordered_layout; missing num_layers, bidirectional, bias toggle, fused scan |
| Recurrent Layers | `LSTM` | partial | `Lstm` (`model/recurrent.rs`): eager IFGO correctness; fused recurrence, direction and layer stacking remain (backlog) |
| Recurrent Layers | `GRU` | partial | #122: gru_matches_independent_f64_forward_and_all_central_differences, gru_executes_asymmetric_geometry_with_reordered_layout; missing num_layers, bidirectional, bias toggle, fused scan |
| Recurrent Layers | `RNNCell` | partial | #122: via rnn oracle; missing standalone CUDA geometry test (same gap as LSTMCell), bias toggle |
| Recurrent Layers | `LSTMCell` | partial | `LstmCell` with forward and gradient oracles in `recurrent_tests.rs`; not yet under the six-point definition's CUDA geometry coverage |
| Recurrent Layers | `GRUCell` | partial | #122: via gru oracle; missing standalone CUDA geometry test, bias toggle |
| Transformer Layers | `Transformer` | no | absent |
| Transformer Layers | `TransformerEncoder` | no | absent |
| Transformer Layers | `TransformerDecoder` | no | absent |
| Transformer Layers | `TransformerEncoderLayer` | no | absent |
| Transformer Layers | `TransformerDecoderLayer` | no | absent |
| Linear Layers | `Identity` | yes | #129: oracle |
| Linear Layers | `Linear` | yes | `Linear` with `.bias(false)`; `PopulationLinear` beside it |
| Linear Layers | `Bilinear` | partial | #129: oracle fwd+grad incl. weight; not a Module (two inputs), Linear-style init not PyTorch's |
| Linear Layers | `LazyLinear` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Dropout Layers | `Dropout` | no | #open: dropout and explicit train/eval mode |
| Dropout Layers | `Dropout1d` | no | absent |
| Dropout Layers | `Dropout2d` | no | absent |
| Dropout Layers | `Dropout3d` | no | absent |
| Dropout Layers | `AlphaDropout` | no | absent |
| Dropout Layers | `FeatureAlphaDropout` | no | absent |
| Sparse Layers | `Embedding` | yes | one-hot `Embedding` (`model/nn.rs`); arbitrary-index `gather` for large tables |
| Sparse Layers | `EmbeddingBag` | partial | #129: sum/mean/max via gather+scatter_add; not a Module; empty Max bag gives NaN where PyTorch gives 0 |
| Distance Functions | `CosineSimilarity` | partial | #125: oracle+central-difference; eps clamps each norm, PyTorch 2.14 clamps the norm product (my brief's error) |
| Distance Functions | `PairwiseDistance` | partial | #125: p=2 only |
| Loss Functions | `L1Loss` | yes | #126: absolute_error + new reordered-storage CUDA evidence |
| Loss Functions | `MSELoss` | yes | #126: squared_error + new reordered-storage CUDA evidence |
| Loss Functions | `CrossEntropyLoss` | yes | `categorical_cross_entropy_with_logits` |
| Loss Functions | `LinearCrossEntropyLoss` | no | absent |
| Loss Functions | `LinearCrossEntropyOptions` | no | absent |
| Loss Functions | `CTCLoss` | no | absent |
| Loss Functions | `NLLLoss` | partial | #126: no weight / ignore_index |
| Loss Functions | `PoissonNLLLoss` | partial | #126: no log_input=False / full=True |
| Loss Functions | `GaussianNLLLoss` | partial | #126: no full=True |
| Loss Functions | `KLDivLoss` | partial | #126: no log_target=True |
| Loss Functions | `BCELoss` | partial | #126: log floor ~-87.3 (f32::MIN_POSITIVE) instead of PyTorch's -100, to keep gradients finite at 0 and 1 |
| Loss Functions | `BCEWithLogitsLoss` | yes | `binary_cross_entropy_with_logits` and the weighted form; hand-checked CUDA witness |
| Loss Functions | `MarginRankingLoss` | yes | #125: oracle+central-difference |
| Loss Functions | `HingeEmbeddingLoss` | yes | #125: oracle+central-difference |
| Loss Functions | `MultiLabelMarginLoss` | yes | #125: multi-hot target like CrossEntropy's one-hot |
| Loss Functions | `HuberLoss` | yes | #126: oracle fwd+grad, delta=1 default |
| Loss Functions | `SmoothL1Loss` | yes | #126: oracle fwd+grad, beta=1 default |
| Loss Functions | `SoftMarginLoss` | yes | #126: oracle fwd+grad |
| Loss Functions | `MultiLabelSoftMarginLoss` | yes | #126: mean of BCE-with-logits, oracle |
| Loss Functions | `CosineEmbeddingLoss` | yes | #125: oracle+central-difference |
| Loss Functions | `MultiMarginLoss` | partial | #125: p=1 only |
| Loss Functions | `TripletMarginLoss` | partial | #125: p=2 only |
| Loss Functions | `TripletMarginWithDistanceLoss` | yes | #125: closure distance, PairwiseDistance default |
| Vision Layers | `PixelShuffle` | yes | #128: split+merge, PyTorch oracle |
| Vision Layers | `PixelUnshuffle` | yes | #128: split+merge, PyTorch oracle |
| Vision Layers | `Upsample` | partial | nearest and bilinear only; no trilinear or bicubic, no scale-factor form |
| Vision Layers | `UpsamplingNearest2d` | yes | `Tensor::upsample_nearest`, named-axis |
| Vision Layers | `UpsamplingBilinear2d` | yes | `Tensor::resample_bilinear`, named-axis |
| Shuffle Layers | `ChannelShuffle` | yes | #128: groups=3 PyTorch oracle |
| DataParallel Layers (multi-GPU, distributed) | `DataParallel` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| DataParallel Layers (multi-GPU, distributed) | `DistributedDataParallel` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Utilities (modules) | `Flatten` | yes | #129: merge of named axes, oracle |
| Utilities (modules) | `Unflatten` | yes | #129: split of a named axis, oracle |

Score: mean of strict (87/161) and half-credit ((87+33/2)/161) = 59.16% = 5916 bp.
