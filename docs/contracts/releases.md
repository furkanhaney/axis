# Release readiness

An Axis release is a statement that the repository's known work has been
resolved or deliberately incorporated. Publishing while an issue or another
pull request remains open makes that boundary ambiguous: the release can omit a
known correction or race an already-reviewed change.

`scripts/checks/release-readiness.sh` therefore requires zero open GitHub issues
and zero open pull requests. A release PR is the sole exception while it is
being reviewed:

```bash
# In release PR 12: issue count must be zero and PR 12 must be the only open PR.
bash scripts/checks/release-readiness.sh --allow-pr 12

# After merge, before tagging or publishing: nothing may remain open.
bash scripts/checks/release-readiness.sh
```

Passing `--base-ref` limits the check to a workspace-version change. GitHub CI
uses that mode on every pull request and main-branch push, so ordinary changes
report a visible skip while version bumps enter the gate. On a pull request CI
permits that pull request's number; after merge it uses strict mode. The manual
workflow dispatch always checks readiness and can optionally name an open
release PR.

The gate checks state, not quality. Closed issues and merged pull requests do
not prove the implementation is correct; formatting, Clippy, unit tests, CUDA
oracles, package inspection, acceptance runs, and human review retain their own
roles. The release check also does not close anything automatically. It makes
unresolved work visible and blocks publication until a maintainer resolves it.
