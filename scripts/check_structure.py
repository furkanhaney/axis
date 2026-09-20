#!/usr/bin/env python3
"""Reject files whose location contradicts the Axis repository structure."""

from pathlib import Path, PurePosixPath
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def git_paths(*arguments: str) -> set[PurePosixPath]:
    result = subprocess.run(
        ["git", "-C", ROOT, "ls-files", "-z", *arguments],
        check=True,
        capture_output=True,
    )
    return {
        PurePosixPath(raw.decode())
        for raw in result.stdout.split(b"\0")
        if raw
    }


def main() -> int:
    tracked = git_paths("--cached")
    visible = tracked | git_paths("--others", "--exclude-standard")
    violations: list[str] = []
    children: dict[PurePosixPath, set[str]] = {}

    for path in sorted(visible, key=str):
        for index, child in enumerate(path.parts):
            parent = PurePosixPath(*path.parts[:index])
            children.setdefault(parent, set()).add(child)
        if path.suffix == ".rs" and "docs" in path.parts:
            violations.append(f"Rust source belongs outside docs/: {path}")
        if path.parts and path.parts[0] == "runs":
            violations.append(f"run records belong under data/runs/: {path}")

    for path in sorted(tracked, key=str):
        for index, part in enumerate(path.parts):
            if part != "data":
                continue
            lane = path.parts[index + 1] if index + 1 < len(path.parts) else None
            if lane not in {"evidence", "runs"}:
                violations.append(
                    f"tracked data belongs in data/evidence/ or data/runs/: {path}"
                )
            break

    managed_roots = tuple(PurePosixPath(name) for name in ("docs", "src", "scripts", "tests"))
    for directory, entries in sorted(children.items(), key=lambda item: str(item[0])):
        if not any(
            directory == root or directory.is_relative_to(root)
            for root in managed_roots
        ):
            continue
        if len(entries) > 8:
            violations.append(
                f"directory has {len(entries)} direct entries; limit is 8: {directory}"
            )

    if violations:
        print("STRUCTURE CHECK: FAIL", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("STRUCTURE CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
