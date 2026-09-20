#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checker="$workspace_root/scripts/checks/release-readiness.sh"
temporary="$(mktemp -d)"
trap 'rm -rf -- "$temporary"' EXIT

cat > "$temporary/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$1" >> "$MOCK_CALLS"
case "$1" in
    issue)
        [[ "${MOCK_FAIL:-}" != issue ]] || exit 17
        cat "$MOCK_ISSUES"
        ;;
    pr)
        [[ "${MOCK_FAIL:-}" != pr ]] || exit 19
        cat "$MOCK_PRS"
        ;;
    *) exit 23 ;;
esac
EOF
chmod +x "$temporary/gh"
: > "$temporary/issues"
: > "$temporary/pulls"
: > "$temporary/calls"

export AXIS_GH_BIN="$temporary/gh"
export MOCK_ISSUES="$temporary/issues"
export MOCK_PRS="$temporary/pulls"
export MOCK_CALLS="$temporary/calls"

expect_pass() {
    local expected="$1"
    shift
    output="$("$@" 2>&1)" || {
        printf 'expected success; got:\n%s\n' "$output" >&2
        exit 1
    }
    [[ "$output" == *"$expected"* ]] || {
        printf 'missing success text %q in:\n%s\n' "$expected" "$output" >&2
        exit 1
    }
}

expect_fail() {
    local expected="$1"
    shift
    if output="$("$@" 2>&1)"; then
        printf 'expected failure; got:\n%s\n' "$output" >&2
        exit 1
    fi
    [[ "$output" == *"$expected"* ]] || {
        printf 'missing failure text %q in:\n%s\n' "$expected" "$output" >&2
        exit 1
    }
}

run_check() {
    bash "$checker" --repository furkanhaney/axis "$@"
}

expect_pass 'PASS (0 open issues; 0 open PRs)' run_check

printf '17\n' > "$MOCK_ISSUES"
expect_fail '#17 https://github.com/furkanhaney/axis/issues/17' run_check
: > "$MOCK_ISSUES"

printf '8\n' > "$MOCK_PRS"
expect_fail '#8 https://github.com/furkanhaney/axis/pull/8' run_check
expect_pass 'release PR #8 is the only open PR' run_check --allow-pr 8

printf '8\n9\n' > "$MOCK_PRS"
expect_fail 'other than release PR #8' run_check --allow-pr 8

: > "$MOCK_PRS"
expect_fail 'Release PR #8 is not open' run_check --allow-pr 8
expect_fail 'must be a positive pull-request number' run_check --allow-pr nope

MOCK_FAIL=issue expect_fail 'GitHub issue query failed' run_check
MOCK_FAIL=pr expect_fail 'GitHub pull-request query failed' run_check

# An ordinary PR with no workspace-version change must not query GitHub at all.
: > "$MOCK_CALLS"
expect_pass 'SKIP (workspace version remains' run_check --base-ref HEAD
[[ ! -s "$MOCK_CALLS" ]] || {
    printf 'unchanged version unexpectedly queried GitHub:\n' >&2
    cat "$MOCK_CALLS" >&2
    exit 1
}

# A version change reaches the repository gate.
fixture="$temporary/version-fixture"
mkdir -p "$fixture/scripts/checks"
cp "$checker" "$fixture/scripts/checks/release-readiness.sh"
git -C "$fixture" init -q
git -C "$fixture" config user.name 'Axis test'
git -C "$fixture" config user.email 'axis-test@example.invalid'
cat > "$fixture/Cargo.toml" <<'EOF'
[workspace]

[workspace.package]
version = "0.2.0"
EOF
git -C "$fixture" add Cargo.toml scripts/checks/release-readiness.sh
git -C "$fixture" commit -qm baseline
base="$(git -C "$fixture" rev-parse HEAD)"
sed -i 's/version = "0.2.0"/version = "0.3.0"/' "$fixture/Cargo.toml"
: > "$MOCK_CALLS"
expect_pass 'Release candidate version: 0.2.0 -> 0.3.0' \
    bash "$fixture/scripts/checks/release-readiness.sh" \
        --repository furkanhaney/axis --base-ref "$base"
[[ "$(wc -l < "$MOCK_CALLS")" -eq 2 ]] || {
    printf 'version change should query issues and PRs exactly once each\n' >&2
    exit 1
}

printf 'RELEASE READINESS TESTS: PASS\n'
