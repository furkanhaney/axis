# API sketch

This historical, non-compiling sketch supplied the four golden programs that
shaped Axis. Current contracts and implemented behavior live in
[library.md](library.md); this document preserves design intent rather than an
alternate source tree.

```rust
use axis::prelude::*;

// The four "golden programs" for the API.
//
// Core rules:
//
// 1. Axes have identity and extent.
// 2. Operators name the axes they transform.
// 3. Logical tensor axes are independent of physical memory layout.
// 4. No dim=-1, transpose(1, 2), view(B, T, H, D), etc.
// 5. Modules describe axis transformations, not positional conventions.
// 6. Optimizers do not hold long-lived mutable borrows of model parameters.
// 7. Losses expose their unreduced structure; reductions are explicit.
// 8. The same axis object means the same mathematical dimension.

// -----------------------------------------------------------------------------
// MLP
// -----------------------------------------------------------------------------

fn mlp() -> Result<()> {
    let batch = Axis::new("batch", 256);

    let input = Axis::new("input", 16);
    let hidden = Axis::new("hidden", 32);
    let output = Axis::new("output", 16);

    let x = Tensor::randn([batch, input], Device::Cuda(0))?;
    let target = Tensor::randn([batch, output], Device::Cuda(0))?;

    let mut model = Sequential::new((
        Linear::new(input, hidden),
        ReLU,
        Linear::new(hidden, output),
    ));

    let mut optimizer = SGD::new(0.5);

    for _ in 0..500 {
        model.zero_grad();

        let prediction = model.forward(&x);

        // [batch, output]
        let error = prediction.squared_error(&target);

        // []
        let loss = error.mean([batch, output]);

        loss.backward();
        optimizer.step(&mut model)?;
    }

    Ok(())
}

// Shapes:
//
// x                         [batch, input]
// Linear(input -> hidden)   [batch, hidden]
// ReLU                      [batch, hidden]
// Linear(hidden -> output)  [batch, output]
// squared_error             [batch, output]
// mean(batch, output)       []
//
// Linear does NOT mean "operate on the final dimension."
//
// Linear::new(input, hidden)
//
// means:
//
//     contract input
//     introduce hidden
//     preserve every other axis
//
// Therefore:
//
// [batch, time, input]
//       Linear(input -> hidden)
// [batch, time, hidden]
//
// works without Linear knowing that batch or time exist.


// -----------------------------------------------------------------------------
// CNN
// -----------------------------------------------------------------------------

fn cnn() -> Result<()> {
    let batch = Axis::new("batch", 128);

    let rgb = Axis::new("rgb", 3);
    let feature1 = Axis::new("feature1", 32);
    let feature2 = Axis::new("feature2", 64);

    let height = Axis::new("height", 32);
    let width = Axis::new("width", 32);

    let class = Axis::new("class", 10);

    let x = Tensor::randn(
        [batch, rgb, height, width],
        Device::Cuda(0),
    )?;

    let target = Tensor::randint(
        0..class.len(),
        [batch],
        Device::Cuda(0),
    )?;

    let spatial = [height, width];

    let mut model = Sequential::new((
        Conv::new(rgb, feature1, spatial)
            .kernel([3, 3])
            .padding(Padding::Same),

        ReLU,

        Conv::new(feature1, feature2, spatial)
            .kernel([3, 3])
            .padding(Padding::Same),

        ReLU,

        AdaptiveMeanPool::new(spatial, [8, 8]),

        Flatten::new(
            [feature2, height, width],
            Axis::new("features", 64 * 8 * 8),
        ),

        Linear::new(
            Axis::named("features"),
            class,
        ),
    ));

    let mut optimizer = AdamW::new(1e-3);

    for _ in 0..1000 {
        model.zero_grad();

        let logits = model.forward(&x);

        // [batch]
        //
        // "class" says which axis contains the categorical distribution.
        // No assumption that classes occupy the last dimension.
        let error = logits.cross_entropy(&target, class);

        let loss = error.mean(batch);

        loss.backward();
        optimizer.step(&mut model)?;
    }

    Ok(())
}

// But Flatten above is still too primitive.
//
// The preferable API is:
//
//     let features = Axis::new("features");
//
//     Collapse::new(
//         [feature2, height, width],
//         features,
//     )
//
// The extent of `features` is inferred from the incoming tensor.
//
// After pooling:
//
//     [batch, feature2=64, height=8, width=8]
//
// Collapse:
//
//     [feature2, height, width] -> features
//
// gives:
//
//     [batch, features=4096]
//
// We should therefore ultimately allow:
//
//     let features = Axis::symbol("features");
//
//     ...
//
//     Collapse::new([feature2, height, width], features)
//
//     Linear::new(features, class)
//
// This lets axis identity exist before its extent is known.
//
// More generally, spatial operations should transform axis extents without
// replacing their identities:
//
//     height=32 -> height=16 -> height=8
//     width=32  -> width=16  -> width=8
//
// "height" remains height.
//
// Its current extent is tensor-specific.
//
// Therefore an Axis probably should NOT fundamentally be:
//
//     Axis { name, size }
//
// but closer to:
//
//     Axis { id, name }
//
// while:
//
//     Shape = [(Axis, Extent)]
//
// Thus:
//
//     x.shape()
//
// might be:
//
//     [
//         (batch, 128),
//         (rgb, 3),
//         (height, 32),
//         (width, 32),
//     ]
//
// This is cleaner than putting mutable/changing size semantics into Axis itself.


// -----------------------------------------------------------------------------
// LSTM
// -----------------------------------------------------------------------------

fn lstm() -> Result<()> {
    let batch = Axis::new("batch", 64);
    let time = Axis::new("time", 128);

    let input = Axis::new("input", 256);
    let hidden = Axis::new("hidden", 512);

    let class = Axis::new("class", 20);

    let x = Tensor::randn(
        [batch(64), time(128), input(256)],
        Device::Cuda(0),
    )?;

    let target = Tensor::randint(
        0..20,
        [batch(64)],
        Device::Cuda(0),
    )?;

    let mut model = SequenceClassifier::new(
        Lstm::new(input, hidden).over(time),
        Take::last(time),
        Linear::new(hidden, class),
    );

    let mut optimizer = AdamW::new(1e-3);

    for _ in 0..500 {
        model.zero_grad();

        let logits = model.forward(&x);

        // logits: [batch, class]
        let error = logits.cross_entropy(&target, class);

        // []
        let loss = error.mean(batch);

        loss.backward();
        optimizer.step(&mut model)?;
    }

    Ok(())
}

// Here the LSTM contract is:
//
// input:
//
//     [*, time, input]
//
// Lstm::new(input, hidden).over(time)
//
// output:
//
//     [*, time, hidden]
//
// "*" means every unrelated axis is preserved.
//
// There is no batch_first=true.
// There is no batch_first=false.
// There is no assumption that time is dimension 0 or dimension 1.
//
// These should all mean the same mathematical thing:
//
//     [batch, time, input]
//     [time, batch, input]
//     [patient, batch, time, input]
//
// because:
//
//     input   identifies the feature axis
//     time    identifies the recurrent axis
//
// The physical backend is free to reorder memory if one representation is faster.
//
// The explicit version of the model would be:
//
//     let sequence = lstm.forward(&x);
//     // [batch, time, hidden]
//
//     let final_state = sequence.select(time, Last);
//     // [batch, hidden]
//
//     let logits = classifier.forward(&final_state);
//     // [batch, class]
//
// `select(time, Last)` is preferable to:
//
//     x[:, -1, :]
//     x.select(1, sequence_length - 1)
//
// because the operation says what it means.


// -----------------------------------------------------------------------------
// TRANSFORMER
// -----------------------------------------------------------------------------

fn transformer() -> Result<()> {
    let batch = Axis::new("batch");
    let time = Axis::new("time");

    let vocab = Axis::new("vocab");
    let model = Axis::new("model");

    let head = Axis::new("head");
    let head_feature = Axis::new("head_feature");

    let mlp = Axis::new("mlp");

    let tokens = Tensor::randint(
        0..50_000,
        [batch(32), time(512)],
        Device::Cuda(0),
    )?;

    let target = Tensor::randint(
        0..50_000,
        [batch(32), time(512)],
        Device::Cuda(0),
    )?;

    let mut network = Transformer::new(
        TokenEmbedding::new(vocab(50_000), model(768)),

        Repeat::new(
            12,
            TransformerBlock::new(
                CausalSelfAttention::new(model)
                    .split(model, [head(12), head_feature(64)])
                    .over(time),

                FeedForward::new(
                    Linear::new(model, mlp(3072)),
                    Gelu,
                    Linear::new(mlp, model),
                ),
            ),
        ),

        LayerNorm::new(model),
        Linear::new(model, vocab),
    );

    let mut optimizer = AdamW::new(3e-4);

    for _ in 0..10_000 {
        network.zero_grad();

        // [batch, time, vocab]
        let logits = network.forward(&tokens);

        // [batch, time]
        let token_loss = logits.cross_entropy(&target, vocab);

        // []
        let loss = token_loss.mean([batch, time]);

        loss.backward();
        optimizer.step(&mut network)?;
    }

    Ok(())
}


// -----------------------------------------------------------------------------
// WHAT SELF-ATTENTION SHOULD LOOK LIKE INTERNALLY
// -----------------------------------------------------------------------------

struct SelfAttention {
    model: Axis,
    head: Axis,
    head_feature: Axis,
    time: Axis,

    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Module for SelfAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x:
        //
        // [batch, time, model]

        let q = self
            .q
            .forward(x)
            .split(self.model, [self.head, self.head_feature]);

        let k = self
            .k
            .forward(x)
            .split(self.model, [self.head, self.head_feature]);

        let v = self
            .v
            .forward(x)
            .split(self.model, [self.head, self.head_feature]);

        // q/k/v:
        //
        // [batch, time, head, head_feature]

        // Attention needs TWO logical time axes.
        //
        // This is an important place where merely naming every sequence axis
        // "time" is insufficient.
        //
        // q has a query-time axis.
        // k/v have a key-time axis.
        //
        // They originate from the same time axis but play different roles in
        // the attention product.

        let query_time = self.time.alias("query_time");
        let key_time = self.time.alias("key_time");

        let q = q.rename(self.time, query_time);
        let k = k.rename(self.time, key_time);
        let v = v.rename(self.time, key_time);

        // q:
        // [batch, query_time, head, head_feature]
        //
        // k:
        // [batch, key_time, head, head_feature]

        let scores = q.contract(&k, self.head_feature);

        // [batch, head, query_time, key_time]
        //
        // contract() removes head_feature.
        //
        // Shared non-contracted axes:
        //
        //     batch
        //     head
        //
        // align automatically.
        //
        // Distinct axes:
        //
        //     query_time
        //     key_time
        //
        // survive.

        let scores = scores
            .scale(self.head_feature.extent().rsqrt())
            .causal_mask(query_time, key_time)
            .softmax(key_time);

        // scores:
        // [batch, head, query_time, key_time]
        //
        // v:
        // [batch, key_time, head, head_feature]

        let context = scores.contract(&v, key_time);

        // [batch, head, query_time, head_feature]

        let context = context
            .rename(query_time, self.time)
            .merge(
                [self.head, self.head_feature],
                self.model,
            );

        // [batch, time, model]

        self.out.forward(&context)
    }
}


// -----------------------------------------------------------------------------
// THIS SUGGESTS A BETTER AXIS MODEL
// -----------------------------------------------------------------------------

// Axis identity and axis extent should probably be separate.
//
// Instead of:
//
//     let batch = Axis::new("batch", 32);
//
// make the fundamental object:
//
//     let batch = Axis::new("batch");
//
// and bind an extent when constructing a shape:
//
//     [batch(32), time(512), model(768)]
//
// Why?
//
// Because the same semantic axis can change extent:
//
//     image:
//         [height=256, width=256]
//
//     pooled:
//         [height=128, width=128]
//
// They're still the same height and width axes.
//
// And sometimes the extent is unknown until runtime:
//
//     [batch=?, time=?, model=768]
//
// The natural types become:
//
//     Axis
//         identity + human-readable name
//
//     Dim
//         axis + extent
//
//     Shape
//         ordered collection of Dims
//
// Conceptually:
//
//     struct Axis {
//         id: AxisId,
//         name: &'static str,
//     }
//
//     struct Dim {
//         axis: Axis,
//         extent: usize,
//     }
//
// Then:
//
//     let batch = Axis::new("batch");
//     let time = Axis::new("time");
//     let model = Axis::new("model");
//
//     let x = Tensor::randn(
//         [batch(32), time(512), model(768)],
//         Cuda(0),
//     )?;
//
// This distinction matters enormously.


// -----------------------------------------------------------------------------
// BASIC TENSOR ALGEBRA
// -----------------------------------------------------------------------------

fn tensor_algebra() {
    let batch = Axis::new("batch");
    let time = Axis::new("time");
    let feature = Axis::new("feature");
    let hidden = Axis::new("hidden");

    // A:
    // [batch=32, time=512, feature=768]

    // B:
    // [feature=768, hidden=3072]

    let c = a.contract(&b, feature);

    // C:
    // [batch=32, time=512, hidden=3072]


    // REDUCTION

    let x = x.mean(time);

    // [batch, time, hidden]
    // ->
    // [batch, hidden]


    let x = x.mean([time, hidden]);

    // [batch, time, hidden]
    // ->
    // [batch]


    // SOFTMAX

    let probabilities = logits.softmax(class);

    // Not softmax(-1).
    //
    // The class axis can be physically anywhere.


    // BROADCASTING

    // x:
    // [batch, time, hidden]
    //
    // bias:
    // [hidden]

    let y = x + bias;

    // Alignment occurs because both tensors contain the SAME hidden axis.
    //
    // No positional broadcasting rules are required.


    // OUTER PRODUCT

    // a:
    // [batch]
    //
    // b:
    // [feature]
    //
    // Unmatched axes should NOT silently produce an outer product through
    // broadcasting.

    let c = a.outer(&b);

    // [batch, feature]
    //
    // Structural expansion is explicit.


    // SPLIT

    // x:
    // [batch, time, model=768]

    let x = x.split(
        model,
        [head(12), head_feature(64)],
    );

    // [batch, time, head=12, head_feature=64]


    // MERGE

    let x = x.merge(
        [head, head_feature],
        model,
    );

    // [batch, time, model=768]


    // RENAME / ROLE SPECIALIZATION

    let query_time = time.alias("query_time");

    let q = q.rename(time, query_time);

    // This preserves ancestry:
    //
    // query_time derives from time.
    //
    // But query_time != time.
    //
    // That makes operations such as attention mathematically expressible
    // without positional hacks.
}


// -----------------------------------------------------------------------------
// THE FOUR USER-FACING PROGRAMS, COMPRESSED
// -----------------------------------------------------------------------------

fn final_mlp() -> Result<()> {
    let batch = Axis::new("batch");
    let input = Axis::new("input");
    let hidden = Axis::new("hidden");
    let output = Axis::new("output");

    let x = Tensor::randn([batch(256), input(16)], Cuda(0))?;
    let y = Tensor::randn([batch(256), output(16)], Cuda(0))?;

    let mut model = Sequential::new((
        Linear::new(input, hidden(32)),
        ReLU,
        Linear::new(hidden, output(16)),
    ));

    let mut optimizer = SGD::new(0.5);

    for _ in 0..500 {
        model.zero_grad();

        let loss = model
            .forward(&x)
            .squared_error(&y)
            .mean([batch, output]);

        loss.backward();
        optimizer.step(&mut model)?;
    }

    Ok(())
}


fn final_cnn() -> Result<()> {
    let batch = Axis::new("batch");

    let channel = Axis::new("channel");
    let feature = Axis::new("feature");

    let height = Axis::new("height");
    let width = Axis::new("width");

    let flattened = Axis::new("flattened");
    let class = Axis::new("class");

    let x = Tensor::randn(
        [
            batch(128),
            channel(3),
            height(32),
            width(32),
        ],
        Cuda(0),
    )?;

    let y = Tensor::randint(
        0..10,
        [batch(128)],
        Cuda(0),
    )?;

    let mut model = Sequential::new((
        Conv::new(channel, feature(32), [height, width])
            .kernel([3, 3])
            .padding(Same),

        ReLU,

        Conv::new(feature, feature(64), [height, width])
            .kernel([3, 3])
            .padding(Same),

        ReLU,

        AdaptiveMeanPool::new([height, width], [8, 8]),

        Collapse::new(
            [feature, height, width],
            flattened,
        ),

        Linear::new(flattened, class(10)),
    ));

    let mut optimizer = AdamW::new(1e-3);

    for _ in 0..1000 {
        model.zero_grad();

        let loss = model
            .forward(&x)
            .cross_entropy(&y, class)
            .mean(batch);

        loss.backward();
        optimizer.step(&mut model)?;
    }

    Ok(())
}


fn final_lstm() -> Result<()> {
    let batch = Axis::new("batch");
    let time = Axis::new("time");

    let input = Axis::new("input");
    let hidden = Axis::new("hidden");
    let class = Axis::new("class");

    let x = Tensor::randn(
        [
            batch(64),
            time(128),
            input(256),
        ],
        Cuda(0),
    )?;

    let y = Tensor::randint(
        0..20,
        [batch(64)],
        Cuda(0),
    )?;

    let mut lstm = Lstm::new(input, hidden(512)).over(time);
    let mut classifier = Linear::new(hidden, class(20));

    let mut optimizer = AdamW::new(1e-3);

    for _ in 0..500 {
        lstm.zero_grad();
        classifier.zero_grad();

        let sequence = lstm.forward(&x);

        let state = sequence.select(time, Last);

        let logits = classifier.forward(&state);

        let loss = logits
            .cross_entropy(&y, class)
            .mean(batch);

        loss.backward();

        optimizer.step((&mut lstm, &mut classifier))?;
    }

    Ok(())
}


fn final_transformer() -> Result<()> {
    let batch = Axis::new("batch");
    let time = Axis::new("time");

    let vocab = Axis::new("vocab");
    let model = Axis::new("model");

    let head = Axis::new("head");
    let head_feature = Axis::new("head_feature");

    let mlp = Axis::new("mlp");

    let tokens = Tensor::randint(
        0..50_000,
        [batch(32), time(512)],
        Cuda(0),
    )?;

    let targets = Tensor::randint(
        0..50_000,
        [batch(32), time(512)],
        Cuda(0),
    )?;

    let mut transformer = Transformer::new(
        TokenEmbedding::new(vocab(50_000), model(768)),

        Repeat::new(
            12,
            TransformerBlock::new(
                CausalSelfAttention::new(model)
                    .heads([head(12), head_feature(64)])
                    .over(time),

                FeedForward::new(
                    model,
                    mlp(3072),
                ),
            ),
        ),

        LayerNorm::new(model),
        Linear::new(model, vocab),
    );

    let mut optimizer = AdamW::new(3e-4);

    for _ in 0..10_000 {
        transformer.zero_grad();

        let logits = transformer.forward(&tokens);

        let loss = logits
            .cross_entropy(&targets, vocab)
            .mean([batch, time]);

        loss.backward();
        optimizer.step(&mut transformer)?;
    }

    Ok(())
}


// -----------------------------------------------------------------------------
// THE CENTRAL IDEA
// -----------------------------------------------------------------------------

// PyTorch fundamentally gives the programmer:
//
//     tensor + numbered dimensions
//
// and then builds semantic conventions on top:
//
//     dimension 0 is probably batch
//     dimension 1 might be sequence
//     dimension -1 is probably feature
//     transpose 1 and 2 before attention
//     flatten dimensions 1 through 3
//     softmax dimension -1
//
// This framework should invert that:
//
//     tensor + semantic axes
//
// and let numbered physical dimensions become a backend implementation detail.
//
// The programmer says:
//
//     mean(time)
//     softmax(vocab)
//     contract(feature)
//     split(model, [head, head_feature])
//     merge([head, head_feature], model)
//     select(time, Last)
//     causal_mask(query_time, key_time)
//
// The CUDA backend decides:
//
//     strides
//     permutations
//     contiguous layouts
//     tile dimensions
//     MMA layouts
//     shared-memory placement
//     kernel fusion
//     whether a transpose exists physically at all
//
// In other words:
//
//     USERS MANIPULATE AXES.
//     BACKENDS MANIPULATE DIMENSIONS.
//
// That should be a foundational rule, not just nicer syntax.
```
