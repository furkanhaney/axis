# Vision example support

This internal crate keeps the introductory vision programs consistent without
hiding their models. It owns strict IDX and CIFAR-100 binary decoding,
train-fitted per-channel normalization, named-axis tensor packing, and the
finite-pass categorical training/evaluation loop.

Dataset-specific model construction, defaults, and claims stay in MNIST,
Fashion-MNIST, CIFAR-100, and later consumers. This crate is a workspace helper
and is not part of the published `axis` package.
