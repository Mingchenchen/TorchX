#!/usr/bin/env python3
"""Convert CD-HIT .clstr output to TorchFold clusters-by-entity txt format."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence

SEQ_ID_RE = re.compile(r">(\S+)")


def parse_clstr(clstr_path: Path) -> List[List[str]]:
    clusters: List[List[str]] = []
    current: List[str] = []
    for raw in clstr_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">Cluster"):
            if current:
                clusters.append(current)
            current = []
            continue
        match = SEQ_ID_RE.search(line)
        if match:
            seq_id = match.group(1).rstrip(".")
            current.append(seq_id.lower())
    if current:
        clusters.append(current)
    if not clusters:
        raise ValueError(f"No clusters parsed from {clstr_path}")
    return clusters


def write_torchfold_cluster_txt(clusters: List[List[str]], out_path: Path) -> None:
    lines = []
    n_singleton = 0
    for members in clusters:
        uniq = []
        seen = set()
        for m in members:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        if not uniq:
            continue
        if len(uniq) == 1:
            n_singleton += 1
        lines.append(" ".join(uniq))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} cluster lines -> {out_path}")
    print(f"  singleton clusters: {n_singleton}")
    print(f"  multi-member clusters: {len(lines) - n_singleton}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert CD-HIT .clstr to TorchFold cluster txt."
    )
    parser.add_argument("--input-clstr", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.input_clstr.is_file():
        raise SystemExit(f"Missing .clstr file: {args.input_clstr}")

    clusters = parse_clstr(args.input_clstr)
    write_torchfold_cluster_txt(clusters, args.output_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
