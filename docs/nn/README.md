# The torch.nn spec — how far Axis reaches PyTorch's module catalog

A spec, not a gate. A gate is a wall that only tightens; a spec is a
battering ram that demands rows. This folder is the ram for the question
"when does Axis reach PyTorch": one row per class in the `torch.nn` layer
catalog, a verdict with evidence, a number, and a floor the number may not
fall below.

- [catalog.md](catalog.md) — the 161 rows, frozen from PyTorch 2.14 on
  2026-09-22, with the verdict and the evidence for each.
- `floor` — basis points. **1351 bp (13.51%)** at the freeze: 20 yes,
  7 partial, 0 refused, 134 no.
- `scripts/checks/nn_gap.sh --check` — recomputes the number from the table,
  fails when the table disagrees with its summary line or the number is below
  the floor. Runs from `scripts/check.sh`, from CI, and from the family's root
  ratchet.

The number is the one the family uses for GNOME Core: the mean of strict
(yes/N) and half-credit ((yes + partial/2)/N). The definition of a complete
module is the six-point one in
[module-backlog.md](../direction/module-backlog.md); the backlog stays the
place that says in what ORDER rows should flip, and why some wait for a
consumer. This table only says which have.

## What flips a row

A verdict flips on a test, not a claim: the module or function exists in
`src/library`, and a CUDA test checks forward values and every gradient
against an independent oracle. A row that lands with a consumer flips to yes;
a row that lands without one may only reach partial until a second use
exists, per the admission rule.

The freeze graded existence plus tests. The first honest climb is the other
way: re-read the 20 yes rows against the six points (independent oracle,
reordered storage, asymmetric geometry). Some may fall to partial. When that
happens the floor takes a written step down here, dated, never a silent one.

## Refusals

None at the freeze. The backlog defers sparse, quantized, distributed,
adaptive and fractional pooling, and lazy initialization to a specialized
stage because "names alone would provide false parity"; deferred is `no`,
not `refused`. A row becomes `refused` only when a design doc says never.

## Changes to the denominator

- 2026-09-22 — frozen: 161 classes from the PyTorch 2.14 `torch.nn` page
  (layer sections, Containers, `Flatten`, `Unflatten`; utility functions
  and `LazyModuleMixin` excluded). Re-freezing against a newer PyTorch is a
  dated line here and a new table.

## Waves

- 2026-09-22 — freeze at 1351 bp.
