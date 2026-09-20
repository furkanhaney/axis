#!/usr/bin/env bash
set -euo pipefail

CUDA_BASE='https://developer.download.nvidia.com/compute/cuda/redist'
CUDA_DEST_DEFAULT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/build/cuda"

CUDA_PACKAGE_ROWS=(
    'cuda_cudart|cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.2.51-archive.tar.xz|539edc1056e44d319f2112e9971c6415d78d4dde04b3f6ffbd20ec808e718526|1539216'
    'cuda_crt|cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.2.51-archive.tar.xz|fbc31fed55b7255591f3a19f575ca078827f5e6757d317d009f7ec1e69fcde4b|80208'
    'cuda_cccl|cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-13.2.27-archive.tar.xz|56e1bafb29faa87375b0484814870046530b88c0a421909096892f027ec1927b|1218316'
    'cuda_nvcc|cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.2.51-archive.tar.xz|706b996fefc59dc8d64d317fdf48d0aa84c4ae004eff43009dd918f40c5cc66a|31097904'
    'cuda_tileiras|cuda_tileiras/linux-x86_64/cuda_tileiras-linux-x86_64-13.2.51-archive.tar.xz|76cbbcc4458b6175878c3a1168521ca9ce36263e7e450ff8a1d1988e5b0bf792|26094256'
    'libnvvm|libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.2.51-archive.tar.xz|e013fce38130d2337ea695aadc5ddd5dcfb78f9107903d72492b9819539749bb|45458632'
    'libcurand|libcurand/linux-x86_64/libcurand-linux-x86_64-10.4.2.51-archive.tar.xz|a089985ac24fff42b719ab42a015c7df39cd721a3d83bfa4af9249b9fca883dc|86632944'
)

# NVIDIA ships these public names as symlinks. Axis replaces each with an
# independent regular file so the ignored toolkit cannot add links into or
# through the surrounding checkout.
CUDA_ALIAS_ROWS=(
    'lib/libcudart.so|lib/libcudart.so.13.2.51'
    'lib/libcudart.so.13|lib/libcudart.so.13.2.51'
    'lib/libcurand.so|lib/libcurand.so.10.4.2.51'
    'lib/libcurand.so.10|lib/libcurand.so.10.4.2.51'
    'nvvm/lib64/libnvvm.so|nvvm/lib64/libnvvm.so.4.0.0'
    'nvvm/lib64/libnvvm.so.4|nvvm/lib64/libnvvm.so.4.0.0'
)

fail() {
    printf '%s\n' "$*" >&2
    return 1
}

require_real_directory() {
    local path="$1"
    [[ -d "$path" && ! -L "$path" ]] || {
        fail "CUDA toolkit root must be a real directory: $path"
        return 1
    }
}

path_with_real_parents() {
    local root="$1" relative="$2" current="$1" index
    local -a parts
    [[ "$relative" != /* && "$relative" != *'..'* ]] || {
        fail "unsafe CUDA path: $relative"
        return 1
    }
    IFS=/ read -r -a parts <<<"$relative"
    (( ${#parts[@]} )) || {
        fail "empty CUDA path"
        return 1
    }
    for ((index = 0; index + 1 < ${#parts[@]}; index++)); do
        current="$current/${parts[index]}"
        [[ -d "$current" && ! -L "$current" ]] || {
            fail "CUDA path parent must be a real directory: $current"
            return 1
        }
    done
    printf '%s/%s\n' "$current" "${parts[-1]}"
}

materialize_cuda_aliases() {
    local root="$1" row alias_relative source_relative alias source temporary relative
    declare -A expected=()
    require_real_directory "$root" || return

    for row in "${CUDA_ALIAS_ROWS[@]}"; do
        IFS='|' read -r alias_relative source_relative <<<"$row"
        expected[$alias_relative]=1
    done
    while IFS= read -r -d '' alias; do
        relative="${alias#"$root"/}"
        [[ -n "${expected[$relative]+set}" ]] || {
            fail "unexpected CUDA symlink: $relative"
            return 1
        }
    done < <(find "$root" -type l -print0)

    for row in "${CUDA_ALIAS_ROWS[@]}"; do
        IFS='|' read -r alias_relative source_relative <<<"$row"
        source="$(path_with_real_parents "$root" "$source_relative")" || return
        [[ -f "$source" && ! -L "$source" ]] || {
            fail "CUDA alias source must be a real regular file: $source"
            return 1
        }
        alias="$(path_with_real_parents "$root" "$alias_relative")" || return
        if [[ -e "$alias" && ! -f "$alias" && ! -L "$alias" ]]; then
            fail "CUDA alias path must be a file or symlink: $alias"
            return 1
        fi
        if [[ -f "$alias" && ! -L "$alias" ]] \
            && cmp -s -- "$source" "$alias" \
            && [[ "$(stat -c '%d:%i' -- "$source")" != "$(stat -c '%d:%i' -- "$alias")" ]]; then
            continue
        fi
        temporary="$(mktemp "$(dirname -- "$alias")/.${alias##*/}.XXXXXX")"
        cp --preserve=mode -- "$source" "$temporary"
        sync -f "$temporary" 2>/dev/null || true
        mv -fT -- "$temporary" "$alias"
        sync -f "$(dirname -- "$alias")" 2>/dev/null || true
    done

    if find "$root" -type l -print -quit | grep -q .; then
        fail "residual CUDA symlink remains under $root"
        return 1
    fi
}

archive_paths_are_safe() {
    local archive="$1" entry normalized part
    while IFS= read -r entry; do
        normalized="${entry#./}"
        [[ "$normalized" != /* ]] || {
            fail "archive contains an absolute path: $entry"
            return 1
        }
        IFS=/ read -r -a components <<<"$normalized"
        for part in "${components[@]}"; do
            [[ "$part" != .. ]] || {
                fail "archive contains parent traversal: $entry"
                return 1
            }
        done
    done < <(tar -tJf "$archive")
}

install_cuda() (
    local destination="$1" temporary row name relative expected size marker archive actual marker_tmp
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
        fail 'This helper supports Linux x86_64 only; use a system CUDA toolkit.'
        return 1
    }
    if [[ -L "$destination" || ( -e "$destination" && ! -d "$destination" ) ]]; then
        fail "CUDA toolkit root must be a real directory: $destination"
        return 1
    fi
    mkdir -p -- "$destination"
    require_real_directory "$destination" || return
    temporary="$(mktemp -d)"
    trap 'rm -r -- "$temporary"' EXIT

    for row in "${CUDA_PACKAGE_ROWS[@]}"; do
        IFS='|' read -r name relative expected size <<<"$row"
        marker="$destination/.$name.sha256"
        if [[ -f "$marker" && "$(<"$marker")" == "$expected" ]]; then
            printf 'Already installed: %s\n' "$name"
            continue
        fi
        printf 'Downloading %s: %.2f MiB\n' "$name" "$(awk -v bytes="$size" 'BEGIN { print bytes / 1048576 }')"
        archive="$(mktemp "$temporary/$name.XXXXXX")"
        curl --fail --location --retry 3 --connect-timeout 30 --max-time 900 \
            --output "$archive" "$CUDA_BASE/$relative"
        actual="$(sha256sum -- "$archive")"
        actual="${actual%% *}"
        [[ "$actual" == "$expected" ]] || {
            fail "SHA-256 mismatch for $name"
            return 1
        }
        archive_paths_are_safe "$archive" || return
        tar -xJf "$archive" -C "$destination" --strip-components=1 \
            --no-same-owner --no-same-permissions
        marker_tmp="$(mktemp "$destination/.$name.sha256.XXXXXX")"
        printf '%s\n' "$expected" >"$marker_tmp"
        mv -fT -- "$marker_tmp" "$marker"
    done

    materialize_cuda_aliases "$destination" || return
    printf 'CUDA_TOOLKIT_PATH=%s\n' "$destination"
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    install_cuda "${CUDA_DEST:-$CUDA_DEST_DEFAULT}"
fi
