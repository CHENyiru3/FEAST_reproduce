#!/usr/bin/env python3
"""Run Study 05--07 tests in isolated processes.

The study scripts intentionally remain directly executable and therefore use
local module names such as ``workflow`` and ``run``.  Process isolation keeps
those names from colliding during a repository-wide test gate.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    "05_2d_conditional_transfer/tests",
    "06_3d_stack/tests",
    "07_3d_transfer/tests",
)


def main() -> int:
    for suite in SUITES:
        print(f"conditional test gate: {suite}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", suite],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode:
            return int(completed.returncode)
    print("conditional test gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
