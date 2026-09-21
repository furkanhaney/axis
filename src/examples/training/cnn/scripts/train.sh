#!/usr/bin/env bash
set -euo pipefail
workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
exec bash "$workspace_root/scripts/cargo.sh" run --release --locked -p cnn --bin axis-cnn -- "$@"
