---
title: Make scientific claims executable
status: direction, admitted bottom-up through real consumers
---

# Axis vision

Axis should become the framework where a researcher can state an assumption
next to the model or data path, and the framework can justify exactly what the
statement means. Named tensor axes are the substrate. The product is trustworthy
claims about data, functions, and computation.

The claim level is part of every contract:

```text
Guaranteed       construction makes a violation impossible within its domain
Verified         a mechanical proof or exhaustive declared domain was checked
Empirical        declared observations passed a measured limit
```

These words are not interchangeable. Testing one hundred million ordered pairs
without finding a monotonicity violation is strong empirical evidence; it is
not a proof that the represented function is globally monotone. An API must
encode that distinction rather than leave it to prose.

## The first function contract: monotonicity

Axis 0.2 starts with an empirical monotonicity invariant. A caller declares an
input, output, direction, absolute numerical tolerance, and maximum observed
violation rate. Every observation supplies the lower and upper value of the
controlled input together with the corresponding outputs. Axis rejects
unordered or non-finite evidence, counts all comparisons, records the worst
violation, and emits a receipt that says explicitly that sampled evidence is
not a global guarantee.

```rust
let mut check = EmpiricalMonotonicity::new(
    "price",
    "demand",
    MonotoneDirection::Decreasing,
    MonotonicityLimits::strict(1e-6)?,
)?;

check.observe_ordered(lower_price, upper_price, lower_demand, upper_demand)?;
println!("{}", check.assert_invariant()?);
```

The caller is responsible for holding all other inputs fixed. The ordered
input values make that test auditable instead of accepting two unexplained
output arrays.

A later architectural contract must use a different API and evidence type. A
positive-weight parameterization, monotone lattice, or another constrained
function class may eventually produce `Guaranteed<Monotone>`. Adversarial
search and finite-difference checks should still attack such a model: a found
counterexample would expose a framework bug, not an unlucky trained model.

## Contract families

The intended families are broader than training-time checks.

### Data and access

- infinite-data-regime limits and exact observed identity reuse;
- train/evaluation disjointness and explicit finite passes;
- causal access and no lookahead; and
- forbidden dependence and information firewalls.

### Function shape

- increasing or decreasing relations;
- positivity and bounded ranges;
- convexity or concavity;
- Lipschitz sensitivity bounds; and
- simplex membership or another declared total.

### Structure and transformation

- permutation invariance or equivariance for sets and populations;
- geometric equivariance for rotations, translations, reflections, and
  declared transformation groups;
- conservation of mass, charge, stocks, or accounting identities; and
- graph-level causal and forbidden-path guarantees.

The distinction between invariant and equivariant matters. Reordering a set may
leave a pooled output unchanged, while a per-entity output must reorder in the
same way. Likewise, augmentation supplies examples of a rotation; an
equivariant architecture supplies a transformation law.

## Admission rule

Axis grows these contracts from experiments rather than an operator checklist.
A new contract needs:

1. a concrete consumer whose conclusion depends on the property;
2. a precise domain and claim level;
3. an independent oracle, construction argument, or adversarial check;
4. a useful failure report and durable receipt when applicable; and
5. composition rules before the property is propagated through arbitrary
   modules or graph operations.

For example, increasing composed with increasing remains increasing, two
increasing functions can be added, and positive scaling preserves direction.
An unrestricted negative path can destroy the property. Axis should propagate
a guarantee only when every relevant operation has a justified rule.

The target experience may eventually look like this:

```rust,ignore
model.guarantees([
    Increasing(income),
    Decreasing(price),
    Causal(time),
    NonNegative(output),
    Lipschitz(1.0),
    Conserves(total_mass),
]);
```

That is a direction, not current API. Each word earns implementation only when
Axis can explain why it is true and a real research program tests the boundary.
