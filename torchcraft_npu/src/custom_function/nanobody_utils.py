import json
import random
from typing import List, Optional, Sequence, Tuple

from torchx.structure import mmcif


def parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated list of ints, e.g. '1,2,3' -> [1,2,3]."""
    if not s:
        return []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        return []
    return [int(p) for p in parts]


def parse_range_list(s: str) -> List[Tuple[int, int]]:
    """Parse 'a-b,c-d,...' into [(a,b),(c,d),...]."""
    if not s:
        return []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    ranges: List[Tuple[int, int]] = []
    for part in parts:
        if '-' not in part:
            raise ValueError(f"Range '{part}' missing '-'")
        lo_str, hi_str = part.split('-', 1)
        lo = int(lo_str)
        hi = int(hi_str)
        if lo > hi:
            raise ValueError(f"Invalid range '{part}': min > max")
        ranges.append((lo, hi))
    return ranges


def parse_two_ranges(s: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Parse side_range string 'a-b,c-d' into ((a,b),(c,d))."""
    ranges = parse_range_list(s)
    if len(ranges) != 2:
        raise ValueError(
            f"side_range should contain exactly two ranges, got {len(ranges)}"
        )
    return ranges[0], ranges[1]


def get_binder_chain_length(json_path: str, target_index: int = -1) -> int:
    """(Deprecated helper) Safely return binder chain length if present, else 0."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return 0

    if "sequences" not in data or not data["sequences"]:
        return 0

    if target_index == -1:
        target_index = len(data["sequences"]) - 1

    if target_index < 0 or target_index >= len(data["sequences"]):
        return 0

    seq_item = data["sequences"][target_index]
    if not seq_item:
        return 0

    chain_data = next(iter(seq_item.values()))
    sequence = chain_data.get("sequence", "")
    return len(sequence or "")


def validate_design_positions(
        design_positions_in_chain: Optional[List[int]],
        binder_length: int,
) -> List[int]:
    """Validate and canonicalise design positions.

    - Removes duplicates and sorts.
    - Ensures all indices are in [0, binder_length).
    """
    if not design_positions_in_chain:
        return []

    uniq = sorted(set(int(p) for p in design_positions_in_chain))
    for pos in uniq:
        if pos < 0 or pos >= binder_length:
            raise ValueError(
                f"design position {pos} out of range for binder length {binder_length}"
            )
    return uniq


def _extract_framework_segment_lengths_from_cif(
        cif_path: str,
) -> List[int]:
    """Extract framework segment lengths from mmCIF by analysing label_seq_id/auth_seq_id gaps.

    Assumes a single polymer chain is present in the file. Framework residues are
    those with coordinates; gaps in label_seq_id/auth_seq_id correspond to missing CDRs.
    Returns a list of framework segment lengths [f0, f1, ..., fK] where
    len(list) - 1 == number of gaps (i.e. number of CDRs).
    """
    # Parse using torchfold.cpp.cif_dict
    with open(cif_path, 'r') as f:
        cif_string = f.read()
    mmcif_obj = mmcif.from_string(cif_string)

    # Get _atom_site data
    auth_seq_ids = mmcif_obj.get('_atom_site.auth_seq_id', [])
    label_asym_ids = mmcif_obj.get('_atom_site.label_asym_id', [])

    if not auth_seq_ids or not label_asym_ids:
        raise ValueError(
            f"Could not find _atom_site.auth_seq_id / label_asym_id in {cif_path}"
        )

    # Find the first chain
    first_chain_id = label_asym_ids[0] if label_asym_ids else None
    if first_chain_id is None:
        raise ValueError(f"No chains found in {cif_path}")

    # Collect unique residue numbers for the first chain
    framework_res_ids: List[int] = []
    last_res_id: Optional[int] = None

    for chain_id, res_id_str in zip(label_asym_ids, auth_seq_ids):
        if chain_id != first_chain_id:
            continue
        try:
            res_id = int(res_id_str)
        except ValueError:
            continue

        if last_res_id is None or res_id != last_res_id:
            framework_res_ids.append(res_id)
            last_res_id = res_id

    if not framework_res_ids:
        raise ValueError(f"No framework residues parsed from {cif_path}")

    # 3) Convert residue ID list into segment lengths between gaps.
    segment_lengths: List[int] = []
    start_idx = 0
    for idx in range(1, len(framework_res_ids)):
        if framework_res_ids[idx] != framework_res_ids[idx - 1] + 1:
            # Gap between two framework residues -> end of a segment.
            segment_lengths.append(idx - start_idx)
            start_idx = idx
    # Tail segment.
    segment_lengths.append(len(framework_res_ids) - start_idx)

    return segment_lengths


_AA3_TO_AA1 = {
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


def extract_framework_aa_from_cif(
        cif_path: str,
) -> List[str]:
    """Extract framework amino-acid types (1-letter) from mmCIF.

    Returns a list of AA (1-letter) for the first chain, in the same order as
    templateIndices 0..N-1 produced by build_design_positions_from_cif_and_ranges.
    """
    # using torchfold.cpp.cif_dict to parse
    with open(cif_path, 'r') as f:
        cif_string = f.read()
    mmcif_obj = mmcif.from_string(cif_string)

    # get _atom_site data
    auth_seq_ids = mmcif_obj.get('_atom_site.auth_seq_id', [])
    label_asym_ids = mmcif_obj.get('_atom_site.label_asym_id', [])
    label_comp_ids = mmcif_obj.get('_atom_site.label_comp_id', [])

    if not auth_seq_ids or not label_asym_ids or not label_comp_ids:
        raise ValueError(
            f"Could not find _atom_site.auth_seq_id / label_asym_id / "
            f"label_comp_id in {cif_path}"
        )

    # Find the first chain
    first_chain_id = label_asym_ids[0] if label_asym_ids else None
    if first_chain_id is None:
        raise ValueError(f"No chains found in {cif_path}")

    # Collect unique residue numbers and amino acid types for the first chain
    framework_res_ids: List[int] = []
    framework_res_aas: List[str] = []
    last_res_id: Optional[int] = None

    for chain_id, res_id_str, res_name in zip(label_asym_ids, auth_seq_ids, label_comp_ids):
        if chain_id != first_chain_id:
            continue
        try:
            res_id = int(res_id_str)
        except ValueError:
            continue

        if last_res_id is None or res_id != last_res_id:
            aa1 = _AA3_TO_AA1.get(res_name.upper(), "X")
            framework_res_ids.append(res_id)
            framework_res_aas.append(aa1)
            last_res_id = res_id

    if not framework_res_ids:
        raise ValueError(f"No framework residues parsed from {cif_path}")

    return framework_res_aas


def build_design_positions_from_cif_and_ranges(
        cif_path: str,
        binder_length: Optional[int],
        cdr_ranges: Sequence[Tuple[int, int]],
        side_ranges: Tuple[Tuple[int, int], Tuple[int, int]],
        seed: int,
) -> Tuple[List[int], dict]:
    """Build design_positions_in_chain given framework CIF + CDR/side ranges.

    Args:
      cif_path: Path to framework-only mmCIF (only contains framework residues).
      binder_length: Length of binder chain sequence in JSON. If None or 0, no
        length check is performed; otherwise, it must equal the computed total
        length from framework + sampled CDR + sampled side segments.
      cdr_ranges: Per-CDR (min,max) length ranges, length must equal the number
        of gaps (i.e. len(framework_segments)-1).
      side_ranges: ((left_min,left_max),(right_min,right_max)).
      seed: Random seed for sampling actual lengths within ranges.

    Returns:
      design_positions: Sorted list of binder-chain indices to be designed
        (left side + all CDRs + right side).
      meta: Dict with sampled lengths, framework segment info, and
        framework/template indices for template JSON injection.
    """
    framework_segments = _extract_framework_segment_lengths_from_cif(cif_path)
    num_segments = len(framework_segments)
    num_gaps = num_segments - 1

    if num_gaps != len(cdr_ranges):
        raise ValueError(
            f"Number of CDR ranges ({len(cdr_ranges)}) does not match number "
            f"of gaps inferred from CIF ({num_gaps})"
        )

    (left_min, left_max), (right_min, right_max) = side_ranges

    rng = random.Random(seed)
    left_len = rng.randint(left_min, left_max)
    right_len = rng.randint(right_min, right_max)
    cdr_lengths = [rng.randint(lo, hi) for (lo, hi) in cdr_ranges]

    design_positions: List[int] = []
    pos = 0

    # Left side (designable)
    for _ in range(left_len):
        design_positions.append(pos)
        pos += 1

    # Framework / CDR alternating segments.
    for seg_idx, fw_len in enumerate(framework_segments):
        # Framework segment (fixed, non-design).
        pos += fw_len
        if seg_idx < num_gaps:
            # Insert CDR (designable).
            Lc = cdr_lengths[seg_idx]
            for _ in range(Lc):
                design_positions.append(pos)
                pos += 1

    # Right side (designable)
    for _ in range(right_len):
        design_positions.append(pos)
        pos += 1

    total_len = pos

    if binder_length and binder_length != total_len:
        raise ValueError(
            f"Binder chain length from JSON ({binder_length}) does not match "
            f"length implied by CIF+CDR+side ({total_len}). "
            f"Please check if the binder sequence length, cdr_ranges, and side_range are consistent."
        )

    # Extra: Derive the framework positions on the binder chain based on the left/framework/CDR/right layout,
    # and the corresponding template indices (0..N_fw-1) for later writing to query/templateIndices in JSON.
    framework_query_indices: List[int] = []
    framework_template_indices: List[int] = []
    fw_pos = 0  # Cursor on the binder chain

    # Left side (design positions, not framework)
    fw_pos += left_len

    # Alternate: framework segment + CDR segment
    template_counter = 0
    for seg_idx, fw_len in enumerate(framework_segments):
        # Current framework segment: fw_len consecutive framework residues
        for _ in range(fw_len):
            framework_query_indices.append(fw_pos)
            framework_template_indices.append(template_counter)
            fw_pos += 1
            template_counter += 1

        # Middle CDRs (all except the last segment)
        if seg_idx < num_gaps:
            fw_pos += cdr_lengths[seg_idx]

    # Right side (design positions, not framework)
    fw_pos += right_len

    # Consistency check (strict only when binder_length is provided)
    if binder_length and fw_pos != total_len:
        raise ValueError(
            "Internal error: Total length derived from layout does not match total_len,"
            f"fw_pos={fw_pos}, total_len={total_len}"
        )

    meta = {
        "left_len": left_len,
        "right_len": right_len,
        "cdr_lengths": cdr_lengths,
        "framework_segment_lengths": framework_segments,
        "num_framework_segments": num_segments,
        "num_gaps": num_gaps,
        "total_length": total_len,
        # Template-related indices
        "framework_query_indices": framework_query_indices,
        "framework_template_indices": framework_template_indices,
    }
    return design_positions, meta


def update_framework_template_in_json(
        json_path: str,
        framework_cif_path: str,
        framework_query_indices: List[int],
        framework_template_indices: List[int],
        target_index: int = -1,
) -> None:
    """Write framework CIF information to the templates section in JSON (without changing the JSON structure).

    Only overwrite the following fields in templates[0] of the target chain:
      - mmcifPath
      - queryIndices
      - templateIndices
    If the chain's templates is empty, create a new template entry.
    """
    if len(framework_query_indices) != len(framework_template_indices):
        raise ValueError(
            "framework_query_indices and framework_template_indices have mismatched lengths: "
            f"{len(framework_query_indices)} != {len(framework_template_indices)}"
        )

    with open(json_path, "r") as f:
        data = json.load(f)

    if "sequences" not in data or not data["sequences"]:
        raise ValueError(f"Missing or empty 'sequences' field in JSON: {json_path}")

    if target_index == -1:
        target_index = len(data["sequences"]) - 1

    if target_index < 0 or target_index >= len(data["sequences"]):
        raise ValueError(
            f"target_index {target_index} is out of range for sequences (len={len(data['sequences'])})"
        )

    seq_item = data["sequences"][target_index]
    if not seq_item:
        raise ValueError(f"In JSON,  index={target_index} sequence entry is empty.")

    # Assume the structure is {"protein": {...}}, extract the inner dict
    chain_data = next(iter(seq_item.values()))
    templates = chain_data.get("templates")
    if templates is None:
        templates = []
        chain_data["templates"] = templates

    if templates:
        template_entry = templates[0]
    else:
        template_entry = {}
        templates.append(template_entry)

    template_entry["mmcifPath"] = framework_cif_path
    template_entry["queryIndices"] = [int(i) for i in framework_query_indices]
    template_entry["templateIndices"] = [int(i) for i in framework_template_indices]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# To implement framework_negative_hotspot_loss, need to extract the list of auth_seq_id for framework residues
def extract_framework_auth_ids_from_cif(
        cif_path: str,
) -> List[int]:
    """Extract framework auth_seq_ids from mmCIF.
    
    Returns a list of auth_seq_ids (ints) for the first chain, strictly in the 
    same order as extract_framework_aa_from_cif.
    """
    with open(cif_path, 'r') as f:
        cif_string = f.read()
    mmcif_obj = mmcif.from_string(cif_string)

    auth_seq_ids = mmcif_obj.get('_atom_site.auth_seq_id', [])
    label_asym_ids = mmcif_obj.get('_atom_site.label_asym_id', [])

    if not auth_seq_ids or not label_asym_ids:
        raise ValueError(f"Could not find _atom_site.auth_seq_id / label_asym_id in {cif_path}")

    first_chain_id = label_asym_ids[0] if label_asym_ids else None

    framework_res_ids: List[int] = []
    last_res_id: Optional[int] = None

    for chain_id, res_id_str in zip(label_asym_ids, auth_seq_ids):
        if chain_id != first_chain_id:
            continue
        try:
            res_id = int(res_id_str)
        except ValueError:
            continue

        # Deduplication logic to ensure each residue records an ID only once
        if last_res_id is None or res_id != last_res_id:
            framework_res_ids.append(res_id)
            last_res_id = res_id

    if not framework_res_ids:
        raise ValueError(f"No framework residues parsed from {cif_path}")

    return framework_res_ids
