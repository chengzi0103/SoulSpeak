#!/usr/bin/env python3
"""Sandbox runner that executes scripts constrained to approved roots."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _load_allowed_roots() -> list[Path]:
    raw = os.getenv("SANDBOX_ALLOWED_ROOTS", "")
    roots: list[Path] = []
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        roots.append(Path(entry).expanduser().resolve())
    return roots


def _within_roots(target: Path, roots: list[Path]) -> bool:
    if not roots:
        return True
    for root in roots:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: sandbox_runner.py <script path> [args...]", file=sys.stderr)
        return 2

    script_path = Path(argv[1]).expanduser().resolve()
    allowed_roots = _load_allowed_roots()

    if not _within_roots(script_path, allowed_roots):
        print(f"[sandbox] script {script_path} outside allowed roots", file=sys.stderr)
        return 3

    if not script_path.exists():
        print(f"[sandbox] script not found: {script_path}", file=sys.stderr)
        return 4

    if script_path.suffix == ".py" or not os.access(script_path, os.X_OK):
        command = [sys.executable, str(script_path), *argv[2:]]
    else:
        command = [str(script_path), *argv[2:]]
    try:
        completed = subprocess.run(command, check=False, text=True)
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[sandbox] failed to execute {command}: {exc}", file=sys.stderr)
        return 5

    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
