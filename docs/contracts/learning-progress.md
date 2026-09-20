# Learning progress is an observed relation

A training loop can finish every forward, backward, and optimizer call without
learning anything. Programs that claim learning therefore need evidence beyond
successful mechanics. `LearningProgress` makes the narrow relation they already
test explicit: one declared metric on one declared evaluation population
improved between a baseline and a later budget point.

```rust
use axis::prelude::*;

let baseline = LearningObservation::new(
    "mean squared error",
    "fixed validation set v1",
    "optimizer steps",
    0,
    1.25,
)?;
let mut learning = LearningProgress::new(
    LearningDirection::Decrease,
    LearningLimits::new(0.10, Some(0.08))?,
    baseline,
)?;
learning.observe(LearningObservation::new(
    "mean squared error",
    "fixed validation set v1",
    "optimizer steps",
    500,
    1.05,
)?)?;
println!("{}", learning.assert_learning()?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Every observation repeats its metric, population, and budget-unit identity.
Axis rejects a mismatch rather than comparing two numbers that describe
different evaluations. Budgets must increase strictly. Metric values and
thresholds must be finite, and relative thresholds require a nonzero baseline.
Any-improvement limits still require a strictly positive change; configured
absolute and relative threshold boundaries are inclusive. Threshold comparison
admits at most four adjacent `f64` representations below the declared value to
cover subtraction and division rounding. It uses no fixed epsilon, so this does
not weaken thresholds at small scales.

The terminal observation is the latest accepted observation. A rejected
observation changes no state, so a malformed checkpoint cannot silently replace
the last valid one. The receipt records the baseline and terminal values,
absolute and relative change, budget delta, thresholds, direction, and evidence
level. Separate training and held-out checks produce separate receipts; one PASS
does not imply the other.

## Claim boundary

The receipt says `VerifiedObservations`. Axis verifies the supplied values and
metadata against the declared relation. The caller still owns the evaluator and
population labels. The receipt does not establish:

- that the evaluation population is representative or disjoint;
- that fresh-batch training loss estimates expected population loss;
- monotonic improvement at every intermediate checkpoint;
- convergence, generalization, or useful compute efficiency; or
- finite gradients and parameter updates.

Compose learning progress with IDR, finite-pass, and disjointness receipts when
those assumptions matter. Use empirical monotonicity for a controlled
input-output relation; learning progress compares one metric through a training
budget.

A mechanics-only acceptance may intentionally omit this contract. The current
Chessformer smoke executes two finite, game-disjoint AdamW updates and makes no
model-quality claim. Adding a progress PASS to such a run would strengthen its
claim and therefore requires an evaluation protocol and meaningful budget.
