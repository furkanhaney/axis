---
title: Residual laws and the first physics-informed path
status: implemented primitives; consumer pending
---

# Residual laws without overstating them

Axis separates three claims that physics-informed training often compresses
into the word "satisfied":

1. a residual expression was included in an optimization objective;
2. sampled residual values met declared numerical limits; and
3. a law is guaranteed throughout a declared domain.

These claims need different evidence. Optimizing a residual does not establish
that it became small. Checking a discretized residual at finite locations does
not prove the differential equation between those locations. Axis currently
implements the second claim as `EmpiricalResidual`. A construction-backed
guarantee will require a separate proof-bearing API; callers cannot select a
`Guaranteed` variant on the empirical receipt.

```rust,ignore
let scope = ResidualScope::new(
    LawIdentity::new("unit-rate exponential decay", "ode-v1")?,
    "du/dt + u",
    "held-out interior t in [0.01, 0.99]",
    "second-order central difference; h=0.01; fp32",
)?;
let mut audit = EmpiricalResidual::new(
    scope,
    ResidualLimits::new(1e-3, 0.01)?,
);
audit.observe_batch(residual_values)?;
println!("{}", audit.assert_within_limits()?);
```

The scope records the stable law identity and version, residual expression,
evaluated region, and evaluator description. Axis verifies the supplied finite
values, tolerance, violation rate, maximum absolute residual, and stable RMS.
It does not parse the expression or prove that the evaluator description is
truthful. The receipt therefore says `empirical` and `verified sampled
discretized observations`, followed by the proof boundary.

Invalid batches are atomic: empty or non-finite input changes no counters or
statistics. A valid batch that exceeds its limits remains recorded so the
error can report the evidence that failed. RMS uses a scaled sum of squares,
which remains finite for finite inputs whose naive squares would overflow.

## Differentiable finite differences

`CentralDifference` records the named coordinate and a finite positive step.
It computes first and second centered stencils from tensors evaluated at
shifted coordinates:

\[
  D_h f(x) = \frac{f(x+h)-f(x-h)}{2h},
  \qquad
  D_h^2 f(x) = \frac{f(x-h)-2f(x)+f(x+h)}{h^2}.
\]

The supplied tensors must have identical ordered shapes and share one `Device`
handle. The coordinate is provenance for what the caller shifted; it need not
be an output axis. The stencils compose existing tensor operations, so reverse
mode reaches every shifted model evaluation and shared parameter.

The primitive does not claim a discretization order, boundary treatment, or
smoothness assumption. Those depend on how the caller produced the tensors and
belong in the residual evaluator description and experiment documentation.

## Why the first consumer uses a stencil

Axis reverse mode currently releases a graph after scalar backward and does
not construct a differentiable gradient graph. A classical automatic-
differentiation PINN loss containing `du/dx` or `d²u/dx²` would therefore fail
to propagate the required mixed derivatives into model parameters. Pretending
otherwise would create a plausible run with the wrong training objective.

The first consumer should instead train a small smooth network on

\[
  u'(t)+u(t)=0,\qquad u(0)=1,
\]

using a centered residual over generated interior points and an explicit
initial-condition loss. `Tanh` supplies the smooth module activation and
`Tensor::sin` supports the pendulum consumer that follows. Evaluation should
compose separate interior and boundary residual receipts, data-regime and
disjointness receipts, and an independent comparison with the analytic
solution `exp(-t)`. None of those fragments alone implies a globally correct
solution.

Higher-order automatic differentiation is a later algebra project. It must
retain a gradient graph and independently verify mixed and second derivatives
before an autodiff residual API enters Axis.
