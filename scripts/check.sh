#!/usr/bin/env bash
set -euo pipefail
workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$workspace_root/scripts/check_structure.py"
python3 -m unittest discover -s "$workspace_root/tests"
runner="$workspace_root/scripts/cargo.sh"
bash "$runner" fmt --all -- --check
bash "$runner" clippy --workspace --release --locked --all-targets -- -D warnings
bash "$runner" test --workspace --release --locked
bash "$runner" test --workspace --release --locked -- --ignored --test-threads=1 --nocapture
