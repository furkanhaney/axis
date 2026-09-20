#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
repository="${GITHUB_REPOSITORY:-}"
allow_pr=""
base_ref=""
gh_bin="${AXIS_GH_BIN:-gh}"

usage() {
    cat <<'EOF'
Usage: scripts/checks/release-readiness.sh [OPTIONS]

Require zero open GitHub issues and pull requests before an Axis release.

Options:
  --repository OWNER/REPO  Repository to inspect (defaults to GITHUB_REPOSITORY
                           or the current gh repository)
  --allow-pr NUMBER        Permit exactly this open release PR and no other PR
  --base-ref REVISION      Skip the gate unless the workspace version changed
                           relative to REVISION
  -h, --help               Show this help
EOF
}

fail_usage() {
    printf 'RELEASE READINESS: ERROR: %s\n' "$1" >&2
    usage >&2
    exit 2
}

while (( $# )); do
    case "$1" in
        --repository)
            (( $# >= 2 )) || fail_usage "--repository requires a value"
            repository="$2"
            shift 2
            ;;
        --allow-pr)
            (( $# >= 2 )) || fail_usage "--allow-pr requires a value"
            allow_pr="$2"
            shift 2
            ;;
        --base-ref)
            (( $# >= 2 )) || fail_usage "--base-ref requires a value"
            base_ref="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) fail_usage "unknown argument: $1" ;;
    esac
done

if [[ -n "$allow_pr" && ! "$allow_pr" =~ ^[1-9][0-9]*$ ]]; then
    fail_usage "--allow-pr must be a positive pull-request number"
fi

workspace_version() {
    awk '
        /^\[workspace\.package\]$/ { in_package = 1; next }
        /^\[/ { in_package = 0 }
        in_package && /^version[[:space:]]*=/ {
            value = $0
            sub(/^[^=]*=[[:space:]]*"/, "", value)
            sub(/"[[:space:]]*$/, "", value)
            print value
            exit
        }
    '
}

if [[ -n "$base_ref" ]]; then
    current_version="$(workspace_version < "$workspace_root/Cargo.toml")"
    if [[ -z "$current_version" ]]; then
        printf 'RELEASE READINESS: ERROR: workspace version is missing from Cargo.toml\n' >&2
        exit 2
    fi
    if ! base_manifest="$(git -C "$workspace_root" show "$base_ref:Cargo.toml" 2>/dev/null)"; then
        printf 'RELEASE READINESS: ERROR: cannot read Cargo.toml at base ref %s\n' "$base_ref" >&2
        exit 2
    fi
    base_version="$(workspace_version <<<"$base_manifest")"
    if [[ -z "$base_version" ]]; then
        printf 'RELEASE READINESS: ERROR: workspace version is missing at base ref %s\n' "$base_ref" >&2
        exit 2
    fi
    if [[ "$base_version" == "$current_version" ]]; then
        printf 'RELEASE READINESS: SKIP (workspace version remains %s)\n' "$current_version"
        exit 0
    fi
    printf 'Release candidate version: %s -> %s\n' "$base_version" "$current_version"
fi

if [[ -z "$repository" ]]; then
    if ! repository="$("$gh_bin" repo view --json nameWithOwner --jq .nameWithOwner)"; then
        printf 'RELEASE READINESS: ERROR: cannot determine the GitHub repository\n' >&2
        exit 2
    fi
fi
if [[ ! "$repository" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
    fail_usage "repository must have OWNER/REPO form"
fi

temporary="$(mktemp -d)"
trap 'rm -rf -- "$temporary"' EXIT
issues_file="$temporary/issues"
prs_file="$temporary/pulls"
number_template='{{range .}}{{printf "%v\n" .number}}{{end}}'

if ! "$gh_bin" issue list --repo "$repository" --state open --limit 1000 \
    --json number --template "$number_template" > "$issues_file"; then
    printf 'RELEASE READINESS: ERROR: GitHub issue query failed for %s\n' "$repository" >&2
    exit 2
fi
if ! "$gh_bin" pr list --repo "$repository" --state open --limit 1000 \
    --json number --template "$number_template" > "$prs_file"; then
    printf 'RELEASE READINESS: ERROR: GitHub pull-request query failed for %s\n' "$repository" >&2
    exit 2
fi

issues=()
pulls=()
while IFS= read -r number; do
    [[ -z "$number" ]] && continue
    if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then
        printf 'RELEASE READINESS: ERROR: invalid issue number from GitHub: %s\n' "$number" >&2
        exit 2
    fi
    issues+=("$number")
done < "$issues_file"
while IFS= read -r number; do
    [[ -z "$number" ]] && continue
    if [[ ! "$number" =~ ^[1-9][0-9]*$ ]]; then
        printf 'RELEASE READINESS: ERROR: invalid pull-request number from GitHub: %s\n' "$number" >&2
        exit 2
    fi
    pulls+=("$number")
done < "$prs_file"

blocked=0
if (( ${#issues[@]} )); then
    blocked=1
    printf 'Open issues block release:\n' >&2
    for number in "${issues[@]}"; do
        printf '  #%s https://github.com/%s/issues/%s\n' "$number" "$repository" "$number" >&2
    done
fi

if [[ -z "$allow_pr" ]]; then
    if (( ${#pulls[@]} )); then
        blocked=1
        printf 'Open pull requests block release:\n' >&2
        for number in "${pulls[@]}"; do
            printf '  #%s https://github.com/%s/pull/%s\n' "$number" "$repository" "$number" >&2
        done
    fi
else
    allowed_seen=0
    other_pulls=()
    for number in "${pulls[@]}"; do
        if [[ "$number" == "$allow_pr" ]]; then
            allowed_seen=1
        else
            other_pulls+=("$number")
        fi
    done
    if (( ! allowed_seen )); then
        blocked=1
        printf 'Release PR #%s is not open in %s.\n' "$allow_pr" "$repository" >&2
    fi
    if (( ${#other_pulls[@]} )); then
        blocked=1
        printf 'Pull requests other than release PR #%s block release:\n' "$allow_pr" >&2
        for number in "${other_pulls[@]}"; do
            printf '  #%s https://github.com/%s/pull/%s\n' "$number" "$repository" "$number" >&2
        done
    fi
fi

if (( blocked )); then
    printf 'RELEASE READINESS: FAIL (%s open issue(s), %s open pull request(s))\n' \
        "${#issues[@]}" "${#pulls[@]}" >&2
    exit 1
fi

if [[ -n "$allow_pr" ]]; then
    printf 'RELEASE READINESS: PASS (0 open issues; release PR #%s is the only open PR)\n' "$allow_pr"
else
    printf 'RELEASE READINESS: PASS (0 open issues; 0 open PRs)\n'
fi
