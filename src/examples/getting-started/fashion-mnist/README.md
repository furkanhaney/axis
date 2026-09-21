# Fashion-MNIST with Axis

This is the next step after the [MNIST example](../mnist/README.md): the same
small named-axis classifier and exact finite-pass receipt, now on ten clothing
categories whose silhouettes overlap enough to make the task less toy-like.

Download and extract Fashion-MNIST's four IDX files, then pass the directory
containing the uncompressed files:

```sh
bash src/examples/getting-started/fashion-mnist/scripts/train.sh --data /path/to/FashionMNIST/raw --smoke
bash src/examples/getting-started/fashion-mnist/scripts/train.sh --data /path/to/FashionMNIST/raw
```

`--smoke` deterministically selects 2,048 training and 512 held-out examples,
trains for five complete shuffled passes, and requires held-out categorical
accuracy to improve. The default uses all 60,000/10,000 examples for twenty
passes. Pixels are average-pooled from 28x28 to 7x7, and normalization is fit
only on training pixels before being applied to both populations.

The downloaded dataset remains local and is ignored by Git. The shared strict
IDX parser and training loop live in `../vision-data`; model construction and
the Fashion-MNIST defaults remain here where a reader can see them.

On the local RTX 5060, the bounded smoke completed in 5.61 seconds and improved
held-out accuracy from 9.77% to 47.66%. The exact finite-pass and learning
receipts are captured in [data/runs/smoke.log](data/runs/smoke.log). This is a
mechanics witness on a fixed prefix, not a full-dataset benchmark.
