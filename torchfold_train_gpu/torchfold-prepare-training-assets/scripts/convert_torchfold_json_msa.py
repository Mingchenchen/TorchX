#!/usr/bin/env python3
"""Convert TorchFold JSON (inline pairedMsa/unpairedMsa) into TorchFold MSA layout.

Dedup key = protein sequence. Output:
  {outdir}/common/seq_to_pdb_index.json
  {outdir}/mmcif_msa_template/{idx}/pairing.a3m
  {outdir}/mmcif_msa_template/{idx}/non_pairing.a3m
  {outdir}/mmcif_msa_template/{idx}/hmmsearch.a3m   # only if templates yield usable a3m
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_OUT_MSA: Optional[Path] = None


def _chain_id(ids: Any) -> str:
    if isinstance(ids, list):
        return str(ids[0])
    return str(ids)


def _cif_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".cif.gz"):
        return name[:-7]
    if lower.endswith(".cif"):
        return name[:-4]
    return path.stem


def _load_success_names(cif_list: Path) -> List[str]:
    names: List[str] = []
    seen = set()
    for line in cif_list.read_text(encoding="utf-8").splitlines():
        p = line.strip().split("#", 1)[0].strip()
        if not p:
            continue
        name = _cif_stem(Path(p))
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _protein_entry(item: dict) -> Optional[dict]:
    return item.get("protein") or item.get("proteinChain")


def _templates_to_hmmsearch_a3m(templates: Any, query_seq: str) -> str:
    """If TorchFold templates embed hit sequences, build a minimal hmmsearch.a3m."""
    if not templates:
        return ""
    lines = [f">query\n{query_seq}"]
    n_hit = 0
    for i, t in enumerate(templates):
        if not isinstance(t, dict):
            continue
        hit = t.get("hitSequence") or t.get("sequence") or ""
        if not hit:
            continue
        name = t.get("name") or t.get("pdbId") or f"template_{i}"
        lines.append(f">{name}\n{hit}")
        n_hit += 1
    if n_hit == 0:
        return ""
    return "\n".join(lines) + "\n"


def _collect_one_json(name: str, json_dir: str) -> Tuple[Dict[str, Tuple[str, str, str]], List[str], int]:
    """Parse one TorchFold JSON; returns (seq->(paired,unpaired,hmm), warnings, n_chains_ok)."""
    jp = Path(json_dir) / f"{name}.json"
    local: Dict[str, Tuple[str, str, str]] = {}
    warnings: List[str] = []
    n_chain = 0

    if not jp.is_file():
        warnings.append(f"{name}\t.\tmissing_json")
        return local, warnings, n_chain

    try:
        with jp.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        warnings.append(f"{name}\t.\tjson_load_error\t{exc}")
        return local, warnings, n_chain

    for item in data.get("sequences") or []:
        pc = _protein_entry(item) or {}
        if not pc:
            continue
        seq = pc.get("sequence") or ""
        if not seq:
            continue
        paired = pc.get("pairedMsa") or ""
        unpaired = pc.get("unpairedMsa") or ""
        cid = _chain_id(pc.get("id"))
        if not paired and pc.get("pairedMsaPath"):
            warnings.append(f"{name}\t{cid}\tpath_only_paired_skipped")
        if not unpaired and pc.get("unpairedMsaPath"):
            warnings.append(f"{name}\t{cid}\tpath_only_unpaired_skipped")
        if not (paired and unpaired):
            warnings.append(
                f"{name}\t{cid}\tmissing_inline_msa\t"
                f"paired={bool(paired)} unpaired={bool(unpaired)}"
            )
            continue
        hmm = _templates_to_hmmsearch_a3m(pc.get("templates") or [], seq)
        n_chain += 1
        prev = local.get(seq)
        if prev is None:
            local[seq] = (paired, unpaired, hmm)
        elif prev[:2] != (paired, unpaired):
            warnings.append(f"{name}\t{cid}\tmsa_conflict_within_json")
    return local, warnings, n_chain


def _merge_collect_results(
    parts: Sequence[Tuple[Dict[str, Tuple[str, str, str]], List[str], int]],
    limit_seqs: int = 0,
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int], List[str], int, int]:
    seq_to_paths: Dict[str, Tuple[str, str, str]] = {}
    warnings: List[str] = []
    n_chain = 0
    missing_json = 0

    for local, warns, nc in parts:
        n_chain += nc
        for w in warns:
            if w.endswith("\tmissing_json"):
                missing_json += 1
        warnings.extend(warns)
        for seq, paths in local.items():
            prev = seq_to_paths.get(seq)
            if prev is None:
                seq_to_paths[seq] = paths
            elif prev[:2] != paths[:2]:
                warnings.append(f".\t.\tmsa_conflict_keep_first\tseq_len={len(seq)}")
            if limit_seqs > 0 and len(seq_to_paths) >= limit_seqs:
                break
        if limit_seqs > 0 and len(seq_to_paths) >= limit_seqs:
            break

    jobs: List[Tuple[str, str, str, str]] = []
    seq_to_idx: Dict[str, int] = {}
    for idx, (seq, paths) in enumerate(seq_to_paths.items()):
        paired, unpaired, hmm = paths
        seq_to_idx[seq] = idx
        jobs.append((seq, paired, unpaired, hmm))
    return jobs, seq_to_idx, warnings, n_chain, missing_json


def _collect_unique_seqs(
    json_dir: Path,
    success_names: Sequence[str],
    limit_seqs: int = 0,
    workers: int = 1,
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int], List[str]]:
    json_dir_str = str(json_dir)
    names = list(success_names)
    if workers <= 1 or len(names) <= 1:
        parts = [_collect_one_json(n, json_dir_str) for n in names]
    else:
        parts = []
        done = 0
        total = len(names)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_collect_one_json, n, json_dir_str): n for n in names}
            for fut in as_completed(futs):
                parts.append(fut.result())
                done += 1
                if done % 200 == 0 or done == total:
                    elapsed = max(1e-6, time.time() - t0)
                    print(
                        f"COLLECT {done}/{total} rate={done / elapsed:.1f}/s",
                        flush=True,
                    )

    jobs, seq_to_idx, warnings, n_chain, missing_json = _merge_collect_results(
        parts, limit_seqs=limit_seqs
    )
    print(
        f"[INFO] chains_ok={n_chain} unique_seqs={len(seq_to_idx)} "
        f"missing_json={missing_json} collect_workers={workers}",
        flush=True,
    )
    return jobs, seq_to_idx, warnings


def _pool_init(out_msa: str) -> None:
    global _OUT_MSA
    _OUT_MSA = Path(out_msa)


def _write_one(item: Tuple[int, str, str, str, str]) -> Tuple[int, str]:
    """item = (idx, seq, paired, unpaired, hmm). Returns (idx, error)."""
    idx, _seq, paired, unpaired, hmm = item
    assert _OUT_MSA is not None
    out_dir = _OUT_MSA / str(idx)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pairing.a3m").write_text(paired, encoding="utf-8")
        (out_dir / "non_pairing.a3m").write_text(unpaired, encoding="utf-8")
        if hmm:
            (out_dir / "hmmsearch.a3m").write_text(hmm, encoding="utf-8")
        (out_dir / ".done").write_text("ok\n", encoding="utf-8")
        return idx, ""
    except Exception as exc:
        return idx, f"{idx}\t{exc}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-list", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if not args.cif_list.is_file():
        raise SystemExit(f"Missing cif list: {args.cif_list}")
    if not args.json_dir.is_dir():
        raise SystemExit(f"Missing json dir: {args.json_dir}")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 8)
    out_msa = args.outdir / "mmcif_msa_template"
    out_common = args.outdir / "common"
    out_logs = args.outdir / "logs"
    out_msa.mkdir(parents=True, exist_ok=True)
    out_common.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    names = _load_success_names(args.cif_list)
    print(f"[INFO] success names from cif list: {len(names)}", flush=True)
    print(f"[INFO] loading JSONs from {args.json_dir} workers={workers} ...", flush=True)
    t_load = time.time()
    jobs, seq_to_idx, warnings = _collect_unique_seqs(
        args.json_dir, names, limit_seqs=args.limit_seqs, workers=workers
    )
    print(f"[INFO] collect done in {time.time() - t_load:.1f}s", flush=True)

    if warnings:
        warn_path = out_logs / "warnings.txt"
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        print(f"[WARN] {len(warnings)} warnings -> {warn_path}", flush=True)

    index_path = out_common / "seq_to_pdb_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(seq_to_idx, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"[INFO] wrote {index_path} ({len(seq_to_idx)} seqs)", flush=True)

    work_items: List[Tuple[int, str, str, str, str]] = []
    skipped = 0
    for seq, paired, unpaired, hmm in jobs:
        idx = seq_to_idx[seq]
        done = out_msa / str(idx) / ".done"
        if args.skip_existing and done.is_file():
            skipped += 1
            continue
        work_items.append((idx, seq, paired, unpaired, hmm))

    print(
        f"[INFO] to write={len(work_items)} skip_existing={skipped} workers={workers}",
        flush=True,
    )
    if not work_items:
        print("[INFO] nothing to write", flush=True)
        return 0

    t0 = time.time()
    fail = 0
    done_n = 0
    fail_lines: List[str] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_pool_init,
        initargs=(str(out_msa),),
    ) as ex:
        futs = {ex.submit(_write_one, it): it[0] for it in work_items}
        for fut in as_completed(futs):
            idx, err = fut.result()
            done_n += 1
            if err:
                fail += 1
                fail_lines.append(err)
            if done_n % 100 == 0 or done_n == len(work_items):
                elapsed = max(1e-6, time.time() - t0)
                print(
                    f"PROGRESS {done_n}/{len(work_items)} fail={fail} "
                    f"rate={done_n / elapsed:.2f}/s",
                    flush=True,
                )

    if fail_lines:
        fail_path = out_logs / "write_failed.txt"
        fail_path.write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
        print(f"[ERROR] {fail} failures -> {fail_path}", flush=True)

    summary = {
        "success_names": len(names),
        "unique_seqs": len(seq_to_idx),
        "written_ok": len(work_items) - fail,
        "failed": fail,
        "skipped_existing": skipped,
        "workers": workers,
        "elapsed_sec": round(time.time() - t0, 1),
        "outdir": str(args.outdir),
    }
    (out_logs / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("[DONE]", json.dumps(summary), flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
