#!/usr/bin/env bash
set -euo pipefail
workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$workspace_root/scripts/checks/structure.sh"
bash "$workspace_root/scripts/checks/nn_gap.sh" --check
bash "$workspace_root/tests/setup_cuda.sh"
bash "$workspace_root/tests/release_readiness.sh"
bash "$workspace_root/tests/docsrs.sh"
runner="$workspace_root/scripts/cargo.sh"
bash "$runner" fmt --all -- --check
bash "$runner" clippy --workspace --release --locked --all-targets -- -D warnings
bash "$runner" test --workspace --release --locked
bash "$runner" test --workspace --release --locked -- --ignored --test-threads=1 --nocapture
