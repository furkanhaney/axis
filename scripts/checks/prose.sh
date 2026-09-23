#!/usr/bin/env bash
# Public prose carries no em dashes: every Markdown file and every Rust doc
# comment (docs.rs renders them). Owner ruling, 2026-09-22.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
hits="$(git -C "$root" grep -n $'—' -- '*.md' 2>/dev/null || true)
$(git -C "$root" grep -nE $'^\\s*//[/!].*—' -- '*.rs' 2>/dev/null || true)"
hits="$(printf '%s\n' "$hits" | sed '/^$/d')"
if [ -n "$hits" ]; then
  printf 'FAIL  prose: em dash in public text (use a colon, comma, parentheses or a new sentence):\n%s\n' "$hits" >&2
  exit 1
fi
echo "PROSE CHECK: PASS"
