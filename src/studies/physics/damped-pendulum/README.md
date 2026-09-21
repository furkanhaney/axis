# Damped pendulum PINN

This study learns a nonlinear damped-pendulum trajectory from the differential
law itself:

\[
\theta''(t) + \gamma\theta'(t) + \frac{g}{L}\sin\theta(t)=0.
\]

It migrates the equation and parameterization used by the archived Caliper R5
pendulum generator. The original MuJoCo scene used
`b = gamma * mass * length²`; after dividing the torque equation by
`mass * length²`, absolute mass disappears. This study evaluates the ideal
point-mass equation and does not claim exact MuJoCo rod-inertia parity.

```bash
bash src/studies/physics/damped-pendulum/scripts/train.sh --smoke
bash src/studies/physics/damped-pendulum/scripts/train.sh
```

The smoke command checks the complete CUDA training path without claiming that
five updates satisfy the scientific limits. The default run performs 4,000
Adam updates over 256,000 fresh, exact `f32` center-coordinate identities, then
applies the declared residual gates on a fixed 255-point interior grid. The
generator rejects center reuse and every train/evaluation stencil overlap.

```mermaid
flowchart LR
    A[Fresh times] --> B[t-h, t, t+h]
    B --> C[Shared Tanh network]
    C --> D[Structural initial conditions]
    D --> E[Central first and second stencils]
    E --> F[Pendulum residual]
    F --> G[Adam update]
    D --> H[Held-out trajectory]
    H --> I[Independent f64 RK4 oracle]
    E --> J[Empirical residual receipt]
```

The solution is parameterized as

\[
\theta(t)=\theta_0+\omega_0t+t^2N(t),
\]

so the value and first-derivative initial conditions hold by construction. The
angle receipt evaluates the value directly; the angular-velocity receipt uses
the declared finite-difference evaluator. Neither silently promotes the
construction into a claim about other laws.

On the checked-in deterministic run, held-out angle RMSE fell from `1.62756939`
to `0.00291016` radians and angular-velocity RMSE fell from `1.60544326` to
`0.00890289` radians/second against an independent f64 RK4 solver. The executable
requires at least 95% improvement plus absolute gates of `0.01` radians and
`0.02` radians/second. The dynamics audit observed 255 points: RMS residual
`0.07254200` radians/second², maximum absolute residual `0.29985142`, and no
observation exceeded the strict `0.35` radians/second² tolerance. The receipt
covers sampled discretized observations,
not the continuous interval between them. See
[the complete run](data/runs/training.log).
