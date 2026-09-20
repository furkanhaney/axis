# Contributing to Axis

Axis welcomes contributions from people working directly and from people
working with coding agents. What matters is that every change has a clear
reason, a reviewable implementation, and evidence proportional to the claim it
makes.

The project is experimental and its API is still moving. Before doing a large
piece of work, open an issue with the problem, the consumer that needs it, and
the evidence that would show it works. Small fixes, documentation improvements,
and focused tests can go straight to a pull request.

## Choose work from a real experiment

Axis grows from concrete research programs rather than an operator checklist.
Framework behavior belongs in `src/axis/`; architecture, metrics, data policy,
and evidence specific to one experiment stay in that consumer's crate. A new
abstraction should normally arrive with a program that needs it and an
independent reference, oracle, or invariant that can catch a wrong result.

Good contributions include:

- a minimal reproduction and fix for incorrect tensor, gradient, or data-regime
  behavior;
- a measured performance improvement that preserves the existing correctness
  witnesses;
- a sharper diagnostic or executable research assumption;
- a consumer-driven capability with an independent correctness check; and
- documentation that makes a guarantee, limit, or result easier to assess.

Please keep scientific claims narrow. A passing smoke test establishes the path
it exercised. It does not by itself establish benchmark parity, model quality,
statistical independence, or general correctness.

## Set up the workspace

Axis currently targets Linux on an NVIDIA GPU, Rust 1.89 or newer, `libclang`,
and CUDA 13.2. If a compatible CUDA toolkit is not already installed, the setup
helper downloads a repository-local toolkit from NVIDIA and verifies the
archives by SHA-256:

```bash
python3 scripts/setup_cuda.py
bash scripts/check.sh
```

Use `bash scripts/cargo.sh ...` for individual Cargo commands so the local CUDA
toolkit is selected consistently. The full check runs formatting, Clippy, CPU
tests, and serialized CUDA tests. If you cannot run the GPU checks, say exactly
which checks you ran and which remain unverified in the pull request.

Run commands from the repository root. The member READMEs under `src/` contain
their smoke commands and measured acceptance criteria.

## Make a focused change

1. Read the README and the relevant contract in `docs/` before changing its
   behavior.
2. Add or update the smallest useful witness. Prefer an independent scalar
   reference, finite differences, identity accounting, or a real consumer over
   a test that repeats the implementation.
3. Keep generated toolkits, build output, downloaded datasets, and scratch runs
   out of Git.
4. Update the owning documentation when a guarantee, limitation, interface, or
   measured result changes.
5. Run `bash scripts/check.sh` and a relevant training smoke when the change
   affects an executable path.

Format commit messages as short imperatives that describe the outcome. Pull
requests should explain the problem, the resulting behavior, the evidence run,
and any limits that remain. Link the motivating issue when there is one.

## Contributions made with agents

Agent-assisted and agent-authored contributions are welcome. The person opening
the pull request remains accountable for the change and should be able to
explain its behavior and evidence. In the pull request:

- say that an agent materially contributed and name the tool or model when
  known;
- describe the human review performed;
- include the commands and real outputs used to verify the change; and
- call out any generated code, unverified path, or external source used.

Do not include prompts, credentials, private data, or unrelated agent
transcripts. Treat generated code like any other contribution: check its
provenance, licensing, security, and correctness before submitting it.

## Licensing

By contributing material you have the right to submit, you agree that your
contribution may be distributed under the repository's [MIT license](LICENSE.md).
Do not copy code, data, documentation, model artifacts, or test vectors whose
license is incompatible or unclear. Identify adapted material in the pull
request and preserve any notices its license requires.

Dependencies and externally obtained datasets, toolchains, and artifacts keep
their own licenses; see [the license boundaries](LICENSE.md#scope-and-third-party-material).
