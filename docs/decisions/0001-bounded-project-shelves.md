# ADR 0001: Bound project shelves and separate durable data lanes

- Status: accepted
- Date: 2026-09-20
- Scope: the Axis repository

## Context

Flat directories initially made Axis easy to start, but `docs/` and the core
crate source grew into scrolling inventories whose filenames carried several
different architectural roles. A `docs/sample.rs` file also looked like Rust
source despite being a non-compiling historical sketch. Run logs, verification
receipts, datasets, and transient debugging output shared names such as `data/`
and `runs/`, which obscured what the repository intended to preserve.

The filesystem is one of the longest-lived interfaces between maintainers and
agents. It should expose project boundaries before a reader opens a file.

The working hypothesis is that a hard local capacity limit forces an agent
with broad temporary context to identify and name the semantic groups it has
already discovered. Those names persist after the context window is gone.

## Decision

Every directory at or below `docs/`, `src/`, `scripts/`, and `tests/` may have
at most eight direct entries, counting files and subdirectories together.
When a shelf reaches the limit, group its contents by an actual subsystem or
responsibility. Do not create compatibility paths after a move.

Rust source does not live under any `docs/` directory. Non-compiling API
sketches use Markdown code fences. Direct children of `src/` name roles at the
same level: `library/` is the published crate, `examples/` holds compact
teaching and acceptance consumers, and `studies/` holds research migrations
grouped by research area. The reusable crate groups its own source into
`algebra/`, `model/`, `research/`, and `runtime/` while keeping their Rust
modules private and re-exporting the public API from `lib.rs`.

Repository automation under `scripts/` and `tests/` is shell. The structure
gate rejects Python source and bytecode so a second scripting toolchain cannot
quietly return.

`data/` is ignored by default. Two descendants are repository artifacts:

- `data/evidence/` for compact audits, oracle comparisons, and verification
  receipts; and
- `data/runs/` for selected reproducible training and profiling records.

Consumers own their records under their own `data/` directory. Cross-workspace
records use the root lanes. A transient debug log remains local unless a
documented claim depends on it.

`scripts/checks/structure.sh` enforces these rules in the versioned pre-commit
hook, the local check, and GitHub CI.

## Alternatives considered

- **Leave flat directories and rely on naming.** Cheap immediately, but the
  tree stops communicating subsystem boundaries and every reader reconstructs
  them independently.
- **Use a larger numerical ceiling.** This postpones the same problem. Eight is
  small enough to keep a shelf visible without scrolling and large enough for
  the current real groupings.
- **Ignore all data.** This removes clutter but also discards the receipts that
  support public claims.
- **Track all data.** This makes raw datasets, checkpoints, and exploratory
  output accidental repository interfaces.

## Consequences

Adding a ninth entry requires a small structural decision rather than another
flat file. A directory whose children mix a library, individual examples, and
whole study families has failed even when its count is below eight: the cap is
supposed to expose semantic structure, not excuse uneven abstraction levels.
Moves require link and caller updates in the same change. Durable evidence
remains reviewable, while ordinary experiment data stays local.

The number eight is a local navigation budget, not a universal software law.
If the rule forces artificial groupings, amend this ADR and the checker together
with a concrete counterexample. Repeated catch-all folders, concepts scattered
across unrelated groups, or frequent moves without a change in meaning are
evidence against the hypothesis rather than reasons to game the count.
