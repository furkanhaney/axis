#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$WORKSPACE/scripts/cargo.sh" run --release --locked -p axis-mnist-population -- "$@"
