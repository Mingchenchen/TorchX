#!/usr/bin/env python3
"""CIF helpers: polypeptide entity sequences and CIF-list loading."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA3_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def sample_id(cif_path: Path) -> str:
    """Canonical sample id (CIF stem, lowercase)."""
    name_lower = cif_path.name.lower()
    if name_lower.endswith(".cif.gz"):
        return cif_path.name[:-7].lower()
    if name_lower.endswith(".cif"):
        return cif_path.name[:-4].lower()
    return cif_path.stem.lower()


def member_id(cif_path: Path, entity_id: str) -> str:
    return f"{sample_id(cif_path)}_{entity_id}"


def _parse_entity_poly_loop(
    lines: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Parse _entity_poly loop -> (entity_id->type, entity_id->strand_id)."""
    types: Dict[str, str] = {}
    strands: Dict[str, str] = {}
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers: List[str] = []
        while i < n and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if "_entity_poly.type" not in headers and "_entity_poly.pdbx_strand_id" not in headers:
            continue
        if "_entity_poly.entity_id" not in headers:
            continue
        eid_idx = headers.index("_entity_poly.entity_id")
        type_idx = (
            headers.index("_entity_poly.type")
            if "_entity_poly.type" in headers
            else None
        )
        strand_idx = (
            headers.index("_entity_poly.pdbx_strand_id")
            if "_entity_poly.pdbx_strand_id" in headers
            else None
        )
        while i < n:
            row = lines[i].strip()
            if not row or row == "#" or row == "loop_" or row.startswith("_"):
                break
            cols = row.split()
            need = eid_idx
            if type_idx is not None:
                need = max(need, type_idx)
            if strand_idx is not None:
                need = max(need, strand_idx)
            if len(cols) > need:
                eid = cols[eid_idx]
                if type_idx is not None:
                    types[eid] = cols[type_idx]
                if strand_idx is not None:
                    strands[eid] = cols[strand_idx].split(",")[0].strip()
            i += 1
        return types, strands
    return types, strands


def _parse_entity_poly_types(lines: List[str]) -> Dict[str, str]:
    types, _ = _parse_entity_poly_loop(lines)
    return types


def _parse_entity_poly_seq(lines: List[str]) -> Optional[Dict[str, List[str]]]:
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers: List[str] = []
        while i < n and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if "_entity_poly_seq.mon_id" not in headers:
            continue
        eid_idx = headers.index("_entity_poly_seq.entity_id")
        mon_idx = headers.index("_entity_poly_seq.mon_id")
        num_idx = (
            headers.index("_entity_poly_seq.num")
            if "_entity_poly_seq.num" in headers
            else None
        )
        rows: List[Tuple[int, str, str]] = []
        while i < n:
            row = lines[i].strip()
            if not row or row == "#" or row == "loop_" or row.startswith("_"):
                break
            cols = row.split()
            if len(cols) <= max(eid_idx, mon_idx):
                i += 1
                continue
            num = int(cols[num_idx]) if num_idx is not None else len(rows) + 1
            rows.append((num, cols[eid_idx], cols[mon_idx].upper()))
            i += 1
        entity_mons: Dict[str, List[Tuple[int, str]]] = {}
        for num, eid, mon in rows:
            entity_mons.setdefault(eid, []).append((num, mon))
        out: Dict[str, List[str]] = {}
        for eid, items in entity_mons.items():
            items.sort(key=lambda x: x[0])
            out[eid] = [mon for _, mon in items]
        return out
    return None


def _parse_seq_from_atom_site(lines: List[str]) -> Dict[str, List[str]]:
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers: List[str] = []
        while i < n and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if "_atom_site.label_comp_id" not in headers or "_atom_site.label_seq_id" not in headers:
            continue

        group_idx = (
            headers.index("_atom_site.group_PDB")
            if "_atom_site.group_PDB" in headers
            else None
        )
        comp_idx = headers.index("_atom_site.label_comp_id")
        asym_idx = (
            headers.index("_atom_site.label_asym_id")
            if "_atom_site.label_asym_id" in headers
            else None
        )
        eid_idx = (
            headers.index("_atom_site.label_entity_id")
            if "_atom_site.label_entity_id" in headers
            else None
        )
        seq_idx = headers.index("_atom_site.label_seq_id")
        auth_seq_idx = (
            headers.index("_atom_site.auth_seq_id")
            if "_atom_site.auth_seq_id" in headers
            else None
        )

        residues: Dict[str, Dict[int, str]] = {}
        while i < n:
            row = lines[i].strip()
            if not row or row == "#" or row == "loop_" or row.startswith("_"):
                break
            cols = row.split()
            needed = [comp_idx, seq_idx]
            if group_idx is not None:
                needed.append(group_idx)
            if asym_idx is not None:
                needed.append(asym_idx)
            if eid_idx is not None:
                needed.append(eid_idx)
            if len(cols) <= max(needed):
                i += 1
                continue
            if group_idx is not None and cols[group_idx] != "ATOM":
                i += 1
                continue

            comp = cols[comp_idx].upper()
            entity_raw = cols[eid_idx] if eid_idx is not None else "?"
            if entity_raw in (".", "?", ""):
                if asym_idx is None or len(cols) <= asym_idx:
                    i += 1
                    continue
                entity = cols[asym_idx]
            else:
                entity = entity_raw

            seq_raw = cols[seq_idx]
            if seq_raw in (".", "?"):
                if auth_seq_idx is None or len(cols) <= auth_seq_idx:
                    i += 1
                    continue
                seq_raw = cols[auth_seq_idx]
            try:
                seq_num = int(seq_raw)
            except ValueError:
                i += 1
                continue

            residues.setdefault(entity, {})
            if seq_num not in residues[entity]:
                residues[entity][seq_num] = comp
            i += 1

        if residues:
            out: Dict[str, List[str]] = {}
            for eid, seq_map in residues.items():
                out[eid] = [seq_map[num] for num in sorted(seq_map)]
            return out
        return {}
    return {}


def extract_entity_sequences(cif_path: Path) -> List[Tuple[str, str, str]]:
    """
    Return list of (entity_id, member_id, sequence) for polypeptide entities.
    member_id = {sample_id}_{entity_id}
    """
    lines = cif_path.read_text(encoding="utf-8", errors="replace").splitlines()
    pdb_id = sample_id(cif_path)
    entity_types = _parse_entity_poly_types(lines)
    entity_seq = _parse_entity_poly_seq(lines)
    from_atom_site = entity_seq is None
    if from_atom_site:
        entity_seq = _parse_seq_from_atom_site(lines)
        if not entity_seq:
            raise ValueError("No _entity_poly_seq or _atom_site sequence found in CIF.")

    records: List[Tuple[str, str, str]] = []
    for entity_id, mon_ids in entity_seq.items():
        if not from_atom_site:
            poly_type = entity_types.get(entity_id, "")
            if poly_type and "polypeptide" not in poly_type.lower():
                continue
        else:
            std = sum(1 for m in mon_ids if m in AA3_TO_1)
            if std < 10 or std / len(mon_ids) < 0.5:
                continue
        seq = "".join(AA3_TO_1.get(m, "X") for m in mon_ids)
        if len(seq) < 10:
            continue
        records.append((entity_id, f"{pdb_id}_{entity_id}", seq))
    if not records:
        raise ValueError("No polypeptide entities with length >= 10.")
    return records


def load_cif_paths(list_path: Path, limit: int = 0) -> List[Path]:
    paths: List[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split("#", 1)[0].strip()
        if not p:
            continue
        paths.append(Path(p))
    if limit > 0:
        paths = paths[:limit]
    return paths
