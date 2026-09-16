#!/usr/bin/env python3
"""Write a CIF list file (one absolute path per line) from a directory or glob."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def collect(inputs: List[Path]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for item in inputs:
        item = item.expanduser()
        if item.is_dir():
            cifs = sorted(item.rglob("*.cif")) + sorted(item.rglob("*.cif.gz"))
        elif item.is_file():
            cifs = [item]
        else:
            raise SystemExit(f"Not found: {item}")
        for p in cifs:
            key = str(p.resolve())
            if key in seen:
                continue
            if p.name.lower().endswith(".cif") or p.name.lower().endswith(".cif.gz"):
                seen.add(key)
                found.append(p.resolve())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CIF files or directories")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    paths = collect(args.inputs)
    if not paths:
        raise SystemExit("No CIF files found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {len(paths)} paths -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
