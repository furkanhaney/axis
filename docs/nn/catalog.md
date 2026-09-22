# The torch.nn catalog, row by row

`classes 161 · yes 111 · partial 16 · refused 0 · no 34`

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
| Pooling layers | `MaxUnpool1d` | partial | #134: scatter by MaxPool forward_with_indices, duplicates last-write like PyTorch CPU; not a Module (two inputs), same rule as Bilinear |
| Pooling layers | `MaxUnpool2d` | partial | #134: same |
| Pooling layers | `MaxUnpool3d` | partial | #134: same, reordered storage |
| Pooling layers | `AvgPool1d` | yes | #134: ceil_mode, count_include_pad, divisor_override per PyTorch; defaults bit-exact |
| Pooling layers | `AvgPool2d` | yes | #134: same, reordered storage |
| Pooling layers | `AvgPool3d` | yes | #134: count_include_pad=False case |
| Pooling layers | `FractionalMaxPool2d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `FractionalMaxPool3d` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Pooling layers | `LPPool1d` | partial | #130: integer p only |
| Pooling layers | `LPPool2d` | partial | #130: integer p only |
| Pooling layers | `LPPool3d` | partial | #130: integer p only |
| Pooling layers | `AdaptiveMaxPool1d` | yes | #134: PyTorch scan order, first maximum |
| Pooling layers | `AdaptiveMaxPool2d` | yes | #134: cross-axis ties route in PyTorch's scan order |
| Pooling layers | `AdaptiveMaxPool3d` | yes | #134: same, reordered storage |
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
| Non-linear Activations (weighted sum, nonlinearity) | `ELU` | yes | #133: any nonzero finite alpha, positive alpha bit-exact |
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
| Non-linear Activations (weighted sum, nonlinearity) | `CELU` | yes | #133: any nonzero finite alpha |
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
| Non-linear Activations (other) | `AdaptiveLogSoftmaxWithLoss` | partial | #136: loss only; no per-row output, log_prob(), predict() |
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
| Recurrent Layers | `RNN` | yes | #137: num_layers, bidirectional, bias toggle; 2-layer bidirectional f64 oracle over all weights and states |
| Recurrent Layers | `LSTM` | yes | #137: depth and direction closed; fused recurrence is a performance item, inter-layer dropout defaults to 0 |
| Recurrent Layers | `GRU` | yes | #137: same as RNN |
| Recurrent Layers | `RNNCell` | yes | #137: standalone reordered-storage asymmetric CUDA test |
| Recurrent Layers | `LSTMCell` | yes | #137: standalone CUDA test |
| Recurrent Layers | `GRUCell` | yes | #137: standalone CUDA test |
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
| Distance Functions | `CosineSimilarity` | yes | #133: joint clamp on the squared-norm product, as PyTorch; test shows divergence from old clamp |
| Distance Functions | `PairwiseDistance` | yes | #133: any p>0 and inf-norm, eps on the difference, p=2 bit-exact; general-p oracle |
| Loss Functions | `L1Loss` | yes | #126: absolute_error + new reordered-storage CUDA evidence |
| Loss Functions | `MSELoss` | yes | #126: squared_error + new reordered-storage CUDA evidence |
| Loss Functions | `CrossEntropyLoss` | yes | `categorical_cross_entropy_with_logits` |
| Loss Functions | `LinearCrossEntropyLoss` | partial | #136: unfused Linear+CE composition stores full logits (PyTorch's point is not to); no reduction/weight/ignore_index |
| Loss Functions | `LinearCrossEntropyOptions` | partial | #136: label_smoothing only |
| Loss Functions | `CTCLoss` | partial | #136: exact gradient via autodiff through log-space recursion, brute-force alignment oracle; no per-row input/target lengths |
| Loss Functions | `NLLLoss` | yes | #138: per-class weight and ignore_index; mean-reduction weighting rule documented |
| Loss Functions | `PoissonNLLLoss` | yes | #138: log_input=False and full=True Stirling term; default bit-exact |
| Loss Functions | `GaussianNLLLoss` | yes | #138: full=True constant; default bit-exact |
| Loss Functions | `KLDivLoss` | yes | #138: log_target=True |
| Loss Functions | `BCELoss` | yes | #138: exact -100 log floor with PyTorch's explicit clamped backward via a dedicated kernel |
| Loss Functions | `BCEWithLogitsLoss` | yes | `binary_cross_entropy_with_logits` and the weighted form; hand-checked CUDA witness |
| Loss Functions | `MarginRankingLoss` | yes | #125: oracle+central-difference |
| Loss Functions | `HingeEmbeddingLoss` | yes | #125: oracle+central-difference |
| Loss Functions | `MultiLabelMarginLoss` | yes | #125: multi-hot target like CrossEntropy's one-hot |
| Loss Functions | `HuberLoss` | yes | #126: oracle fwd+grad, delta=1 default |
| Loss Functions | `SmoothL1Loss` | yes | #126: oracle fwd+grad, beta=1 default |
| Loss Functions | `SoftMarginLoss` | yes | #126: oracle fwd+grad |
| Loss Functions | `MultiLabelSoftMarginLoss` | yes | #126: mean of BCE-with-logits, oracle |
| Loss Functions | `CosineEmbeddingLoss` | yes | #125: oracle+central-difference |
| Loss Functions | `MultiMarginLoss` | yes | #133: p in {1,2} and per-class weight (a constant, as in PyTorch) |
| Loss Functions | `TripletMarginLoss` | yes | #133: p passed through to every distance incl. swap |
| Loss Functions | `TripletMarginWithDistanceLoss` | yes | #125: closure distance, PairwiseDistance default |
| Vision Layers | `PixelShuffle` | yes | #128: split+merge, PyTorch oracle |
| Vision Layers | `PixelUnshuffle` | yes | #128: split+merge, PyTorch oracle |
| Vision Layers | `Upsample` | yes | #136: module with size= or scale_factor=, nearest/bilinear/trilinear/bicubic (A=-0.75); no align_corners=True |
| Vision Layers | `UpsamplingNearest2d` | yes | `Tensor::upsample_nearest`, named-axis |
| Vision Layers | `UpsamplingBilinear2d` | yes | `Tensor::resample_bilinear`, named-axis |
| Shuffle Layers | `ChannelShuffle` | yes | #128: groups=3 PyTorch oracle |
| DataParallel Layers (multi-GPU, distributed) | `DataParallel` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| DataParallel Layers (multi-GPU, distributed) | `DistributedDataParallel` | no | specialized stage of the backlog: needs its own representation or execution contract; a name alone would be false parity |
| Utilities (modules) | `Flatten` | yes | #129: merge of named axes, oracle |
| Utilities (modules) | `Unflatten` | yes | #129: split of a named axis, oracle |

Score: mean of strict (111/161) and half-credit ((111+16/2)/161) = 71.43% = 7143 bp.
