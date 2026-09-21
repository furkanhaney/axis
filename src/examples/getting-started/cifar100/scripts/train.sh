#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
exec bash "$workspace/scripts/cargo.sh" run --release --locked -p axis-cifar100 -- "$@"
