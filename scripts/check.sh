#!/usr/bin/env bash
set -euo pipefail
runner="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/cargo.sh"
bash "$runner" fmt --all -- --check
bash "$runner" clippy --workspace --release --locked --all-targets -- -D warnings
bash "$runner" test --workspace --release --locked
bash "$runner" test --workspace --release --locked -- --ignored --test-threads=1 --nocapture
