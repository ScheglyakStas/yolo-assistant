#!/usr/bin/env python3
"""Copy the canonical shared modules over the worker's copies.

The worker runs under a different interpreter and imports its own copies of the
protocol and of the translation machinery, so each pair must stay
byte-identical. ``tests/test_protocol.py`` and ``tests/test_i18n.py`` fail if
they drift; this is the fix.

    python packaging/sync_protocol.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "yolo_assistant" / "app" / "core"

#: Canonical source -> the worker's copy.
PAIRS = (
    (CORE / "protocol.py", ROOT / "worker" / "protocol.py"),
    (CORE / "i18n.py", ROOT / "worker" / "i18n.py"),
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report drift without copying")
    args = parser.parse_args()

    status = 0
    for source, target in PAIRS:
        if _hash(source) == _hash(target):
            print(f"{target.relative_to(ROOT)} is in sync")
            continue
        if args.check:
            print(f"DRIFT: {target.relative_to(ROOT)} differs from {source.relative_to(ROOT)}")
            status = 1
            continue
        shutil.copyfile(source, target)
        print(f"copied {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
