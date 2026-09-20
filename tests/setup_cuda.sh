#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../scripts/setup_cuda.sh
source "$root/scripts/setup_cuda.sh"

temporary="$(mktemp -d)"
trap 'rm -r -- "$temporary"' EXIT

test_fail() {
    printf 'setup_cuda test failed: %s\n' "$*" >&2
    exit 1
}

make_toolkit() {
    local destination="$1" symlinks="${2:-true}" row alias_relative source_relative source alias index=0
    mkdir -p -- "$destination"
    for row in "${CUDA_ALIAS_ROWS[@]}"; do
        IFS='|' read -r alias_relative source_relative <<<"$row"
        source="$destination/$source_relative"
        alias="$destination/$alias_relative"
        mkdir -p -- "$(dirname -- "$source")" "$(dirname -- "$alias")"
        printf 'versioned CUDA library %d\n' "$index" >"$source"
        if "$symlinks"; then
            ln -s -- "$(realpath --relative-to="$(dirname -- "$alias")" "$source")" "$alias"
        fi
        index=$((index + 1))
    done
}

case_one="$temporary/materialize"
make_toolkit "$case_one"
materialize_cuda_aliases "$case_one"
[[ -z "$(find "$case_one" -type l -print -quit)" ]] || test_fail 'aliases remained symlinks'
for row in "${CUDA_ALIAS_ROWS[@]}"; do
    IFS='|' read -r alias_relative source_relative <<<"$row"
    alias="$case_one/$alias_relative"
    source="$case_one/$source_relative"
    [[ -f "$alias" && ! -L "$alias" ]] || test_fail "$alias is not a regular file"
    cmp -s -- "$source" "$alias" || test_fail "$alias differs from its versioned source"
    [[ "$(stat -c '%d:%i' -- "$source")" != "$(stat -c '%d:%i' -- "$alias")" ]] \
        || test_fail "$alias shares an inode with its source"
done

stable="$case_one/lib/libcudart.so"
broken="$case_one/lib/libcudart.so.13"
stable_inode="$(stat -c %i -- "$stable")"
printf 'corrupt\n' >"$broken"
materialize_cuda_aliases "$case_one"
[[ "$(stat -c %i -- "$stable")" == "$stable_inode" ]] || test_fail 'valid alias was rewritten'
cmp -s -- "$case_one/lib/libcudart.so.13.2.51" "$broken" || test_fail 'corrupt alias was not repaired'

case_two="$temporary/unexpected"
make_toolkit "$case_two"
ln -s lib "$case_two/unexpected"
if materialize_cuda_aliases "$case_two" >"$temporary/unexpected.log" 2>&1; then
    test_fail 'unexpected symlink was accepted'
fi
grep -q 'unexpected CUDA symlink' "$temporary/unexpected.log" \
    || test_fail 'unexpected symlink error was not diagnostic'

case_three="$temporary/markers"
make_toolkit "$case_three"
for row in "${CUDA_PACKAGE_ROWS[@]}"; do
    IFS='|' read -r name _ expected _ <<<"$row"
    printf '%s\n' "$expected" >"$case_three/.$name.sha256"
done
install_cuda "$case_three" >"$temporary/markers.log"
[[ -z "$(find "$case_three" -type l -print -quit)" ]] \
    || test_fail 'marker fast path skipped alias materialization'

printf 'SETUP CUDA TESTS: PASS\n'
