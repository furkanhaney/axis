# Axis — executable research assumptions on named tensors

Axis is the Rust training framework born from concrete Rasat research programs.
The workspace's `src/` has three roles at one semantic level: the reusable
crate in `src/library/`, compact consumers in `src/examples/`, and research
ports in `src/studies/`. Preserve that shape: shared behavior enters through
the library, while experiment-specific policy stays with its consumer.

The original Python studies remain in `../research/`. Migrations link to those
sources and state what they preserve, change, and have actually measured. A
migration result is a mechanics witness unless its document explicitly claims
scientific parity.

Read the relevant spoke in `docs/` before changing its contract. Put durable
cross-consumer reasoning there as the context grows. Keep launchers in each
consumer's `scripts/`, Rust in its `src/`, and measured output in `data/evidence/` or `data/runs/`.

Run `bash scripts/check.sh` before pushing. Use `bash scripts/cargo.sh` for Cargo
so the local CUDA toolkit and library path are selected consistently. Smoke a
training program before a longer run. Commit small working increments.

Filesystem structure is architecture: put a thing at the lowest common
ancestor of its consumers, and make the tree resemble the project it explains.
Do not preserve compatibility paths after a move; fix consumers directly.
