#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-host}"

if ! cargo llvm-cov --version >/dev/null 2>&1; then
    echo "coverage requires cargo-llvm-cov: cargo install cargo-llvm-cov --locked" >&2
    exit 1
fi

common=(
    -p axis
    --lib
    --locked
    --summary-only
    --ignore-filename-regex '(^|/)(target|\.cargo/registry)/'
)

case "$mode" in
    host)
        bash "$workspace_root/scripts/cargo.sh" llvm-cov "${common[@]}"
        ;;
    cuda)
        bash "$workspace_root/scripts/cargo.sh" llvm-cov "${common[@]}" -- \
            --include-ignored --test-threads=1
        ;;
    *)
        echo "usage: scripts/coverage.sh [host|cuda]" >&2
        exit 2
        ;;
esac
