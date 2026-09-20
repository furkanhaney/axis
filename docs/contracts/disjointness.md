# Semantic disjointness

Different source IDs do not imply different scientific examples. A generator
can emit the same problem from two seeds, augmentation can produce two byte
representations of one object, and copied documents can arrive through
different shards. Axis therefore lets the experiment define the identity that
matters to its claim.

```rust
let scheme = IdentityScheme::new(
    "sudoku-puzzle",
    "1",
    "81 cells after first-occurrence digit relabeling; positions retained",
)?;
let mut split = Disjointness::new(scheme, ["training", "tuning", "audit"])?;

split.observe("tuning", tuning.iter().map(canonical_puzzle))?;
split.observe("audit", audit.iter().map(canonical_puzzle))?;
split.observe("training", batch.iter().map(canonical_puzzle))?;
let receipt = split.assert_disjoint()?;
```

The key type is generic and equality is exact. Axis does not hash a sample or
guess its semantics. The caller supplies a canonical key and names the
equivalence relation in a versioned `IdentityScheme`. This makes the proof
boundary visible in the receipt and allows a later canonicalizer change to be
distinguished from a data change.

Every observation is compared with every other declared population. Repeats
within one population are allowed and counted separately. If any identity
crosses a boundary, the complete incoming batch is rejected without changing
the ledger. `assert_disjoint` also requires at least one observation from every
declared population, catching misspelled or skipped evaluation paths.

The receipt says `VerifiedObservations`: it proves zero equality overlap among
the identities delivered to this ledger. It does not prove that unobserved data
is disjoint, that the canonicalizer captures every relevant equivalence, or
that the populations are statistically independent. A construction-time
guarantee needs a separate proof-producing partition API; a label saying
“different seed namespaces” is not sufficient when two seeds can generate the
same semantic problem.

The addition acceptance uses exact operand bit patterns rather than generator
draw IDs. The Sudoku acceptance uses the clue board after canonical digit
relabeling. Thus both consumers can detect a duplicate problem even when it was
drawn under a distinct raw ID, while keeping the identity unit appropriate to
the experiment.
