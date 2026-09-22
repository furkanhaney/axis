#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
proof_root="$(mktemp -d "${TMPDIR:-/tmp}/axis-docsrs.XXXXXX")"
trap 'rm -rf -- "$proof_root"' EXIT

version="$(
    awk '
        /^\[workspace\.package\]$/ { in_package = 1; next }
        /^\[/ { in_package = 0 }
        in_package && /^version = / {
            gsub(/version = |"/, "")
            print
            exit
        }
    ' "$root/Cargo.toml"
)"
package_target="$proof_root/package-target"

CARGO_TARGET_DIR="$package_target" \
    cargo package --manifest-path "$root/Cargo.toml" -p axis --locked --allow-dirty --no-verify

mkdir -p "$proof_root/unpacked"
tar -xzf "$package_target/package/axis-$version.crate" -C "$proof_root/unpacked"
manifest="$proof_root/unpacked/axis-$version/Cargo.toml"

env -u CUDA_TOOLKIT_PATH -u CUDA_HOME \
    DOCS_RS=1 \
    CARGO_TARGET_DIR="$proof_root/docs-target" \
    cargo doc --manifest-path "$manifest" --locked --no-default-features --no-deps

env -u CUDA_TOOLKIT_PATH -u CUDA_HOME \
    DOCS_RS=1 \
    CARGO_TARGET_DIR="$proof_root/docs-target" \
    cargo check --manifest-path "$manifest" --locked --no-default-features

set +e
CUDA_TOOLKIT_PATH="$proof_root/missing-cuda" \
    CUDA_HOME= \
    CARGO_TARGET_DIR="$proof_root/default-target" \
    cargo check --manifest-path "$manifest" --locked \
    >"$proof_root/default-build.log" 2>&1
default_status=$?

env -u DOCS_RS -u CUDA_TOOLKIT_PATH -u CUDA_HOME \
    CARGO_TARGET_DIR="$proof_root/no-feature-target" \
    cargo check --manifest-path "$manifest" --locked --no-default-features \
    >"$proof_root/no-feature-build.log" 2>&1
no_feature_status=$?
set -e

if (( default_status == 0 )); then
    printf 'default Axis build unexpectedly succeeded without CUDA\n' >&2
    exit 1
fi
if ! grep -Fq "is invalid: $proof_root/missing-cuda is not a directory" \
    "$proof_root/default-build.log"; then
    cat "$proof_root/default-build.log" >&2
    printf 'default Axis build did not report the missing CUDA toolkit clearly\n' >&2
    exit 1
fi

if (( no_feature_status == 0 )); then
    printf 'Axis unexpectedly built without either CUDA or the docs.rs environment\n' >&2
    exit 1
fi
if ! grep -Fq 'disabling it is reserved for the docs.rs build' \
    "$proof_root/no-feature-build.log"; then
    cat "$proof_root/no-feature-build.log" >&2
    printf 'Axis did not reject the docs-only configuration outside docs.rs\n' >&2
    exit 1
fi

printf 'DOCS.RS CONTRACT: PASS\n'
