#!/usr/bin/env python3
"""Print exact shared function definitions in the two concrete Rust trainers.

This is a small source census for these files, not a general Rust parser.
It identifies functions in the kernels module, top level, and simple impls.
Matching ignores whitespace and includes signatures, braces, and comments;
reported counts exclude blank lines and function attributes. Similar functions
and common fragments inside differing functions are deliberately uncounted.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def functions(path):
    source = path.read_text()
    result = {}
    for match in re.finditer(r"^( {0,4})fn (\w+)\b", source, re.M):
        indent, name = match.groups()
        scope = "host"
        if indent:
            scopes = list(re.finditer(r"^(?:impl|mod) (\w+)\s*\{", source[:match.start()], re.M))
            scope = scopes[-1].group(1)
        # Skip the parameter list: its const-generic types also contain braces.
        start = source.index("(", match.end())
        depth = 1
        end = start + 1
        while depth:
            depth += (source[end] == "(") - (source[end] == ")")
            end += 1
        body = source.index("{", end)
        depth = 1
        end = body + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        text = source[match.start():end]
        key = f"{scope}::{name}"
        if key in result:
            raise ValueError(f"Ambiguous function key {key} in {path}")
        result[key] = {
            "normalized": re.sub(r"\s+", "", text),
            "lines": sum(bool(line.strip()) for line in text.splitlines()),
            "start": source.count("\n", 0, match.start()) + 1,
        }
    return result


def main():
    left = functions(ROOT / "src/examples/mlp/src/baseline.rs")
    right = functions(ROOT / "src/examples/cnn/src/train.rs")
    same = [name for name in sorted(left.keys() & right.keys())
            if left[name]["normalized"] == right[name]["normalized"]]
    print("| Identical function | MLP line | CNN line | Nonblank lines |")
    print("|---|---:|---:|---:|")
    for name in same:
        print(f"| `{name}` | {left[name]['start']} | {right[name]['start']} | {left[name]['lines']} |")
    total = sum(left[name]["lines"] for name in same)
    print(f"\n{len(same)} identical definitions; {total} nonblank lines per trainer.")


if __name__ == "__main__":
    main()
