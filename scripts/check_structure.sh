#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
declare -A child_seen=()
declare -A child_count=()
violations=()

managed_directory() {
    case "$1" in
        docs|docs/*|src|src/*|scripts|scripts/*|tests|tests/*) return 0 ;;
        *) return 1 ;;
    esac
}

inspect_visible() {
    local path="$1" parent="" part key
    local -a parts
    IFS=/ read -r -a parts <<<"$path"

    case "$path" in
        *.py|*.pyc|*/__pycache__/*) violations+=("Python artifacts are not part of Axis: $path") ;;
    esac
    [[ "${parts[0]:-}" == runs ]] && violations+=("run records belong under data/runs/: $path")

    for part in "${parts[@]}"; do
        if [[ "$part" == docs && "$path" == *.rs ]]; then
            violations+=("Rust source belongs outside docs/: $path")
            break
        fi
    done

    for part in "${parts[@]}"; do
        if [[ -n "$parent" ]] && managed_directory "$parent"; then
            key="$parent"$'\034'"$part"
            if [[ -z "${child_seen[$key]+set}" ]]; then
                child_seen[$key]=1
                child_count[$parent]=$(( ${child_count[$parent]:-0} + 1 ))
            fi
        fi
        parent="${parent:+$parent/}$part"
    done
}

inspect_tracked_data() {
    local path="$1" index lane
    local -a parts
    IFS=/ read -r -a parts <<<"$path"
    for ((index = 0; index < ${#parts[@]}; index++)); do
        [[ "${parts[index]}" == data ]] || continue
        lane="${parts[index + 1]:-}"
        if [[ "$lane" != evidence && "$lane" != runs ]]; then
            violations+=("tracked data belongs in data/evidence/ or data/runs/: $path")
        fi
        break
    done
}

while IFS= read -r -d '' path; do
    inspect_visible "$path"
    inspect_tracked_data "$path"
done < <(git -C "$root" ls-files --cached -z)

while IFS= read -r -d '' path; do
    inspect_visible "$path"
done < <(git -C "$root" ls-files --others --exclude-standard -z)

while IFS= read -r directory; do
    if (( child_count[$directory] > 8 )); then
        violations+=("directory has ${child_count[$directory]} direct entries; limit is 8: $directory")
    fi
done < <(printf '%s\n' "${!child_count[@]}" | LC_ALL=C sort)

if (( ${#violations[@]} )); then
    printf 'STRUCTURE CHECK: FAIL\n' >&2
    printf -- '- %s\n' "${violations[@]}" >&2
    exit 1
fi

printf 'STRUCTURE CHECK: PASS\n'
