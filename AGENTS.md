# GENERATED — edit HUB.md, then: scripts/hubgen.py --write
- Depth lives in docs/ spokes: read the spoke before CHANGING what it covers.
- Launch from a node, never inside a src/. Only src/ recurses.
- Slots exist only when nonempty. Ephemera (build/, scratch/, worktrees) are
  gitignored; archive/ is frozen.
- Put a thing at the LCA of its consumers. Everything enters at the bottom.
- Filesystem should resemble the shape of the project, Conway style:
  give real crates, programs, and subsystems their own folders.
- Internal structure changes are clean migrations: update callers and
  remove retired paths; do not retain compatibility shims.
- DER00: Leave the code you touched a tiny bit better. The requested change
  can supply the improvement; no extra cleanup is owed. Preserve required
  behavior except intended changes. Do not expand scope, invent abstractions,
  or move complexity elsewhere merely to satisfy this rule.
- Read and improve the owning scope's docs/ as you learn. Roughly every 5-25
  turns, fold useful findings, ideas, corrections, or next steps into local
  docs; a small edit is enough. Do not rely on chat or compaction alone.
- Child hubs only ADD — never restate or override an ancestor.
- Ratchet baselines only go down; an exception needs a written waiver.
- Verify by driving the real thing; report red gates verbatim.
- Commit frequently: small working increments, in the child repo that owns
  the change. Never end a task with the work only on disk.
- Shell cwd persists across tool calls and resets between them unpredictably:
  use absolute paths; never trust a bare `cd`.
- Doctrine: <root>/docs/ · why hubs look like this: <root>/docs/hubs.md
# --- end invariant — this node's half follows ---
# Axis — executable research assumptions on named tensors

Axis is the Rust training framework born from concrete Rasat research programs.
The reusable crate lives at `src/axis/`; sibling folders under `src/` are real
consumers and migration witnesses. Preserve that shape: shared behavior enters
through the library, while experiment-specific policy stays with its consumer.

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
