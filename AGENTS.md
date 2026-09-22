# Axis

Axis is a public Rust framework for named-axis tensor programs, GPU training,
and executable research contracts on NVIDIA cuTile. Maintain this file directly.
Do not add a `HUB.md` or harness-specific instruction aliases.

README.md is for users and outside contributors. This file is for agents working
on the repository. Keep public claims tied to reproducible tests, receipts, or
run artifacts; distinguish mechanics witnesses from scientific results.

## Project shape

- `src/library/` is the published `axis` crate and the owner of reusable tensor,
  module, optimizer, data, metric, and research-contract behavior.
- `src/examples/getting-started/` contains approachable end-to-end programs;
  MNIST is the first path for a new user.
- `src/examples/training/` contains ordinary model and optimizer flows such as
  MLP, CNN, attention, and Muon. They are building blocks, not research claims.
- `src/studies/contracts/` contains researcher-level data-regime and execution
  contracts such as IDR and population-axis training. Verification and
  monotonicity work belongs here or in an outside research consumer, never in
  the introductory shelf.
- Other `src/studies/` families contain applied research ports whose
  task-specific architecture, metrics, and evidence do not belong in the library.
- `docs/` explains contracts, design decisions, and measured limits. `docs/nn/`
  is the torch.nn spec: every PyTorch module class as a row with a verdict, a
  number and a floor (`scripts/checks/nn_gap.sh --check`).
- `scripts/` contains shell tooling. Axis contains no Python.
- `tests/` contains repository and packaging gates.
- Tracked data belongs only in `data/evidence/` or `data/runs/`.

Filesystem structure is architecture. Put code at the lowest common ancestor of
its consumers, keep every direct directory under `docs/`, `src/`, `scripts/`,
and `tests/` to at most eight entries, and never place Rust source in `docs/`.
The structure gate enforces these rules.

## Working contract

- Read the relevant document under `docs/` before changing its contract.
- Make clean migrations: update callers and remove retired paths. Do not keep
  compatibility shims for internal APIs.
- Leave materially touched code slightly better. The requested change itself
  can supply that improvement; do not widen scope to manufacture cleanup.
- Fold durable findings into the owning documentation as context grows.
- Keep framework policy in Axis and experiment-specific policy in its consumer.
- Do not expose credentials or commit generated toolkits, build output, large
  datasets, or scratch runs.
- Nothing is sent to NVIDIA or another upstream without explicit owner approval.

## Verification and delivery

Run `bash scripts/check.sh` before merging. Use `bash scripts/cargo.sh` for Cargo
when a command needs the repository CUDA environment. GPU behavior needs a real
CUDA witness; host-only compilation is not runtime evidence.

The public crate is package-allowlisted in `src/library/Cargo.toml`. Verify the
package boundary when adding files or features. Main is protected; changes land
through focused PRs with `host-and-compile`, `documentation-contract`, and
`release-readiness` green. Agents may open Axis issues and PRs and should state
exact evidence and limits in them.

Before a release, resolve every Axis issue and unrelated PR. The version-bump PR
must be the sole open PR; after it merges, strict release readiness requires zero
open issues and zero open PRs. Inspect `git status -sb` before pushing because the
public GitHub remote and the GitLab mirror have different roles.
