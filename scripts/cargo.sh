#!/usr/bin/env bash
set -euo pipefail
workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CUDA_TOOLKIT_PATH:-}${CUDA_HOME:-}" && -d "$workspace_root/.cuda" ]]; then
    export CUDA_TOOLKIT_PATH="$workspace_root/.cuda"
fi
cargo_command="${1:?usage: cargo.sh <command> [arguments...]}"
shift
exec cargo "$cargo_command" --manifest-path "$workspace_root/Cargo.toml" "$@"
