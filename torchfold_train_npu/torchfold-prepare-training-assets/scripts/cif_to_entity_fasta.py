#!/usr/bin/env python3
"""Extract polypeptide entities from a CIF list into one FASTA for CD-HIT.

Member id = {sample_id}_{entity_id} (lowercase), matching TorchFold cluster lookup.
If label_entity_id is missing ('?'), entity_id falls back to asym_id (A/B/...).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cif_utils import extract_entity_sequences, load_cif_paths  # noqa: E402


def _process_one(item: Tuple[int, str]) -> Tuple[int, List[Tuple[str, str]], str]:
    idx, cif_path_str = item
    try:
        records = extract_entity_sequences(Path(cif_path_str))
        out = [(mid.lower(), seq) for _eid, mid, seq in records]
        return idx, out, ""
    except Exception as exc:
        return idx, [], f"{cif_path_str}\t{exc}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-list", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args(argv)

    paths = load_cif_paths(args.cif_list, limit=args.limit)
    if not paths:
        raise SystemExit(f"No CIF paths in {args.cif_list}")

    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    failed_path = args.output_fasta.with_suffix(".failed.txt")

    jobs = max(1, int(args.jobs))
    progress_every = max(1, args.progress_every)
    total = len(paths)
    all_recs: List[Optional[List[Tuple[str, str]]]] = [None] * total
    failures: List[str] = []

    print(f"[INFO] CIFs={total} jobs={jobs}", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_process_one, (i, str(p))): i for i, p in enumerate(paths)}
        for fut in as_completed(futs):
            idx, recs, err = fut.result()
            all_recs[idx] = recs
            if err:
                failures.append(err)
            done += 1
            if done % progress_every == 0 or done == total:
                print(
                    f"PROGRESS {done}/{total} fail={len(failures)} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    n_seq = 0
    with args.output_fasta.open("w", encoding="utf-8") as f:
        for recs in all_recs:
            if not recs:
                continue
            for mid, seq in recs:
                f.write(f">{mid}\n{seq}\n")
                n_seq += 1

    if failures:
        failed_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"[WARN] {len(failures)} failures -> {failed_path}", flush=True)

    print(
        f"[DONE] wrote {n_seq} sequences -> {args.output_fasta} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0 if n_seq > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
