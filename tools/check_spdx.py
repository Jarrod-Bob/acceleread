# SPDX-License-Identifier: Apache-2.0
"""Fail if a Python source file lacks the Apache-2.0 SPDX header in its first three lines."""

import sys
from pathlib import Path

HEADER = "# SPDX-License-Identifier: Apache-2.0"
ROOTS = ("packages", "tools", "examples")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    missing = [
        path.relative_to(root)
        for top in ROOTS
        for path in sorted((root / top).rglob("*.py"))
        if HEADER not in path.read_text(encoding="utf-8").splitlines()[:3]
    ]
    for path in missing:
        print(f"missing SPDX header: {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
