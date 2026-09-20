# Migration notes

## Mechanics and architecture kept

The source is a hello-world baseline for treating training trajectories as
data. Its constrained question is whether a useful MNIST classifier fits in
about one thousand parameters. This first port preserves the experiment's data
path, architecture, iteration structure, and evaluation mechanics. Both
programs:

- decode the canonical 60,000/10,000 MNIST train/test split;
- average each non-overlapping 4x4 image block, reducing 28x28 to 7x7;
- normalize both splits with the scalar mean and standard deviation measured
  from training pixels only;
- train a 49 -> 16 -> 10 ReLU MLP with exactly 970 parameters;
- shuffle all training examples once per epoch, retain the short final batch,
  and report accuracy only on the untouched test split.

The Rust executable defaults to the source program's 20 epochs, batch size 512,
learning rate 0.003, and seed 0. `--smoke` uses a bounded subset and five
passes; it requires held-out accuracy to improve, but is not the full baseline
result.

## Library behavior established by the migration

The consumer now uses stable categorical cross-entropy with one-hot targets and
reduces only the named class axis. Its `Adam` uses the source defaults (beta1
0.9, beta2 0.999, epsilon 1e-8); parameter values and both moment buffers remain
on the GPU during updates.

The finite source is intentionally reused. Applying `assert_idr()` would be a
false scientific claim. `FinitePassesLoader` instead declares the exact pass
count, keeps stable sample identities through deterministic shuffles, prevents
batches from crossing pass boundaries, rejects duplicates within a pass, and
checks that every pass contains the same population. Its final receipt reports
population, completed passes, and total observations.

The port does not reproduce PyTorch's random-number generator or its exact
parameter initialization, so individual curves and final accuracy are not
claimed numerically identical. The objective, optimizer equations, data path,
topology, iteration structure, and evaluation protocol now match.
