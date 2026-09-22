#!/usr/bin/env bash
# The torch.nn spec: how far Axis reaches PyTorch's module catalog.
#
#   scripts/checks/nn_gap.sh            # print the number
#   scripts/checks/nn_gap.sh --check    # the same, and fail if the table disagrees with
#                                       # its summary line or the number is below docs/nn/floor
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
catalog="$root/docs/nn/catalog.md"
floor_file="$root/docs/nn/floor"

summary="$(grep -oE 'classes [0-9]+ · yes [0-9]+ · partial [0-9]+ · refused [0-9]+ · no [0-9]+' "$catalog" | head -1)"
if [ -z "$summary" ]; then
  echo "FAIL  nn spec: catalog.md has no summary line" >&2; exit 1
fi
read -r _ n _ y _ p _ r _ no <<<"${summary//·/}"
if [ "$n" -ne $((y + p + r + no)) ]; then
  echo "FAIL  nn spec: summary does not add up ($n != $y+$p+$r+$no)" >&2; exit 1
fi

# One table row per class: "| Section | `Class` | verdict | evidence |"
rows="$(grep -E '^\| [^|]+ \| `[^`]+` \| (yes|partial|refused|no) \|' "$catalog")"
count() { printf '%s\n' "$rows" | grep -cE "^\| [^|]+ \| \`[^\`]+\` \| $1 \|" || true; }
ty="$(count yes)"; tp="$(count partial)"; tr_="$(count refused)"; tn="$(count no)"
if [ "$ty $tp $tr_ $tn" != "$y $p $r $no" ]; then
  echo "FAIL  nn spec: table rows ($ty $tp $tr_ $tn) disagree with the summary line ($y $p $r $no)" >&2; exit 1
fi
dups="$(printf '%s\n' "$rows" | sed -E 's/^\| [^|]+ \| `([^`]+)` .*/\1/' | sort | uniq -d)"
if [ -n "$dups" ]; then
  echo "FAIL  nn spec: duplicate rows: $(echo "$dups" | tr '\n' ' ')" >&2; exit 1
fi

# mean of strict (yes/N) and half-credit ((yes + partial/2)/N), in basis points
bp="$(awk -v n="$n" -v y="$y" -v p="$p" 'BEGIN { printf "%d", ((y/n) + ((y + p/2)/n)) / 2 * 10000 + 0.5 }')"
printf 'nn-spec: classes %d  yes %d  partial %d  refused %d  no %d   %d.%02d%% of torch.nn (%d bp)\n' \
  "$n" "$y" "$p" "$r" "$no" $((bp / 100)) $((bp % 100)) "$bp"

if [ "${1:-}" = "--check" ]; then
  floor="$(tr -d '[:space:]' < "$floor_file")"
  if [ "$bp" -lt "$floor" ]; then
    echo "FAIL  nn spec: $bp bp is below the floor $floor (docs/nn/floor) — a verdict went backwards" >&2; exit 1
  fi
  echo "OK    nn spec: $bp bp, floor $floor."
fi
