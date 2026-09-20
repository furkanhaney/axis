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
let mut split = Disjointness::new(
    scheme,
    [
        PopulationSpec::streaming("training"),
        PopulationSpec::retained("tuning"),
        PopulationSpec::retained("audit"),
    ],
)?;

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

Retained populations are compared pairwise, and their repeats are counted.
Exactly one population may instead be `streaming`. Its observations are checked
against every retained identity and then discarded, so an endless training run
uses memory proportional to the finite tuning/audit populations. All retained
populations must be observed before streaming begins and are sealed afterward;
otherwise later reference data could overlap discarded training history.

Streaming mode deliberately does not claim unique identities or repeat counts
inside the stream. Pair it with IDR when raw draw reuse matters. If all
populations are bounded and pairwise history is needed, declare all of them
retained by passing names directly. In either mode, a contaminated incoming
batch is rejected without changing the ledger. `assert_disjoint` requires at
least one observation from every declared population.

The receipt says `VerifiedObservations`: it proves zero equality overlap among
the identities delivered to this ledger. It does not prove that unobserved data
is disjoint, that the canonicalizer captures every relevant equivalence, or
that the populations are statistically independent. A construction-time
guarantee needs a separate proof-producing partition API; a label saying
“different seed namespaces” is not sufficient when two seeds can generate the
same semantic problem.

The addition acceptance retains exact operand bit patterns for evaluation and
streams training problems against them. The Sudoku acceptance retains its
tuning and audit keys while streaming training keys. The Sudoku program owns
its clue-board canonicalizer; Axis verifies the identities it is given. Both
consumers can detect a reference problem under a distinct raw draw ID without
retaining an unbounded training history.
