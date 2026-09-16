from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from torchfold.model import feat_batch


# AF3 atom-cross-attention subset sizes (alphafold3 defaults).
_QUERIES_SUBSET_SIZE = 32
_KEYS_SUBSET_SIZE = 128

_AF3_NUM_TYPES = 31  # POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
_TORCHFOLD_NUM_TYPES = 32  # STD_RESIDUES_WITH_GAP

# standard index -> AF3 index (POLYMER_TYPES_WITH_UNKNOWN_AND_GAP, GAP=21).
_TORCHFOLD_TO_AF3_RESTYPE = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  # ALA..VAL
    20,                 # UNK
    22, 23, 24, 25,     # RNA A,G,C,U -> 22..25
    30,                 # RNA N (unknown) -> UNK_NUCLEIC(30)
    26, 27, 28, 29,     # DNA DA,DG,DC,DT -> 26..29
    30,                 # DNA DN (unknown) -> UNK_NUCLEIC(30)
    21,                 # GAP '-' -> 21
]


def _restype_remap_table(device) -> torch.Tensor:
    return torch.tensor(_TORCHFOLD_TO_AF3_RESTYPE, dtype=torch.int64, device=device)


def _restype_idx_32_to_31(idx: torch.Tensor) -> torch.Tensor:
    """Remap 32-vocab restype/msa indices to AF3 31-vocab by identity.

    NOT a clamp: AF3 puts GAP at 21 and a single nucleic-unknown at 30
    """
    table = _restype_remap_table(idx.device)
    return table[idx.clamp(min=0, max=_TORCHFOLD_NUM_TYPES - 1).to(torch.int64)]


def _profile_32_to_31(profile32: torch.Tensor) -> torch.Tensor:
    """Reorder/merge 32-wide profile distribution into AF3 31-wide.

    Columns are permuted to AF3 order
    """
    assert profile32.shape[-1] == _TORCHFOLD_NUM_TYPES, profile32.shape
    *lead, _ = profile32.shape
    out = profile32.new_zeros((*lead, _AF3_NUM_TYPES))
    table = _restype_remap_table(profile32.device)
    out.index_add_(-1, table, profile32)
    return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ife(batch: dict) -> dict:
    if "input_feature_dict" in batch:
        return batch["input_feature_dict"]
    return batch


def _to_dev(t: torch.Tensor, device, dtype=None) -> torch.Tensor:
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device=device)


def _gather_info_dict(
    prefix: str,
    gather_idxs: torch.Tensor,
    gather_mask: torch.Tensor,
    input_shape: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Pack a GatherInfo triple under the ``<prefix>:...`` keys the AF3 schema uses.

    ``input_shape`` is forced to a CPU int64 tensor.
    """
    return {
        f"{prefix}:gather_idxs": gather_idxs.to(dtype=torch.int64),
        f"{prefix}:gather_mask": gather_mask.to(dtype=torch.bool),
        f"{prefix}:input_shape": torch.tensor(
            list(input_shape), dtype=torch.int64, device=torch.device("cpu")
        ),
    }


def _compute_gather_idxs_np(
    source_uids: np.ndarray, target_uids: np.ndarray, fill_value: int = 0
):
    """Numpy reimplementation of alphafold3 atom_layout.compute_gather_idxs.

    Atoms are identified by an integer ``uid`` (0 == padding/empty, matches nothing).
    ``source_uids`` is flattened (ravel) before building the uid->index map, so the
    resulting gather_idxs index into the *flattened* source array — exactly what
    ``atom_layout.convert`` expects.

    Returns (gather_idxs[target_shape], gather_mask[target_shape]).
    """
    src_flat = source_uids.ravel()
    uid_to_idx: dict[int, int] = {}
    for idx, uid in enumerate(src_flat):
        if uid == 0:
            continue  # padding cell: not a valid gather source
        # first occurrence wins (mirrors dict-comprehension last-wins? -> AF3 uses
        # last-wins via dict comprehension; uids are globally unique here so order
        # is irrelevant). We keep first to be safe with any accidental dupes.
        if uid not in uid_to_idx:
            uid_to_idx[int(uid)] = idx

    tgt_flat = target_uids.ravel()
    g_idx = np.full(tgt_flat.shape, fill_value, dtype=np.int64)
    g_mask = np.zeros(tgt_flat.shape, dtype=bool)
    for i, uid in enumerate(tgt_flat):
        j = uid_to_idx.get(int(uid))
        if j is not None:
            g_idx[i] = j
            g_mask[i] = True
    return g_idx.reshape(target_uids.shape), g_mask.reshape(target_uids.shape)


def _convert_np(gather_idxs, gather_mask, arr, layout_axes):
    """Minimal numpy port of atom_layout.convert for building tokens_to_* gathers."""
    begin = layout_axes[0]
    end = layout_axes[-1] + 1
    layout_shape = arr.shape[begin:end]
    batch_shape = arr.shape[:begin]
    feat_shape = arr.shape[end:]
    flat = arr.reshape(batch_shape + (int(np.prod(layout_shape)),) + feat_shape)
    if begin == 0:
        out = flat[gather_idxs, ...]
    elif begin == 1:
        out = flat[:, gather_idxs, ...]
    else:
        raise ValueError(begin)
    mask_shape = (
        (1,) * len(batch_shape) + gather_mask.shape + (1,) * len(feat_shape)
    )
    out = out * gather_mask.reshape(mask_shape)
    return out


def _build_token_atom_uid_grid(
    atom_to_token_idx: np.ndarray,
    atom_to_tokatom_idx: np.ndarray,
    n_token: int,
) -> tuple[np.ndarray, int]:
    """Return (uid_grid [N_token, max_dense], max_dense).

    """
    max_dense = int(atom_to_tokatom_idx.max()) + 1
    uid_grid = np.zeros((n_token, max_dense), dtype=np.int64)
    n_atom = atom_to_token_idx.shape[0]
    uid_grid[atom_to_token_idx, atom_to_tokatom_idx] = np.arange(1, n_atom + 1)
    return uid_grid, max_dense


def _scatter_flat_to_dense(
    flat: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_to_tokatom_idx: torch.Tensor,
    n_token: int,
    max_dense: int,
    fill: float = 0.0,
) -> torch.Tensor:
    """Scatter a flat ``[N_atom, *feat]`` tensor into dense ``[N_token, max_dense, *feat]``."""
    feat_shape = tuple(flat.shape[1:])
    out = flat.new_full((n_token, max_dense) + feat_shape, fill)
    out[atom_to_token_idx, atom_to_tokatom_idx] = flat
    return out


# ---------------------------------------------------------------------------
# Atom-cross-attention gathers (mirror AtomCrossAtt.compute_features)
# ---------------------------------------------------------------------------

def _build_atom_cross_att_gathers(uid_grid: np.ndarray):
    """Build the five atom-cross-att GatherInfos from the dense uid grid.

    Mirrors alphafold3.model.features.AtomCrossAtt.compute_features but on integer
    uids instead of string AtomLayouts. ``num_atoms`` is padded up to a multiple of
    ``_QUERIES_SUBSET_SIZE`` (AF3 PaddingShapes uses a multiple of 32).
    """
    n_token, max_dense = uid_grid.shape
    token_atoms_mask = uid_grid != 0  # [N_token, max_dense]

    # Flat list of present atom uids (row-major), then pad with 0.
    flat_uids = uid_grid[token_atoms_mask]  # [num_atoms]
    num_atoms = int(flat_uids.shape[0])
    qss = _QUERIES_SUBSET_SIZE
    kss = _KEYS_SUBSET_SIZE
    num_atoms_padded = int(np.ceil(max(num_atoms, 1) / qss) * qss)
    padded_flat = np.zeros((num_atoms_padded,), dtype=np.int64)
    padded_flat[:num_atoms] = flat_uids

    # queries layout: [num_subsets, qss]
    num_subsets = num_atoms_padded // qss
    queries_uids = padded_flat.reshape((num_subsets, qss))

    # keys layout: key subsets centered on query subsets (AF3 logic).
    subset_centers = np.arange(qss / 2, num_atoms_padded, qss)
    flat_to_key = (
        subset_centers[:, None] + np.arange(-kss / 2, kss / 2)[None, :]
    ).astype(int)
    for row in range(flat_to_key.shape[0]):
        if flat_to_key[row, 0] < 0:
            flat_to_key[row, :] -= flat_to_key[row, 0]
        elif flat_to_key[row, -1] > num_atoms - 1:
            overflow = flat_to_key[row, -1] - (num_atoms - 1)
            flat_to_key[row, :] -= overflow
    flat_to_key = np.clip(flat_to_key, 0, num_atoms_padded - 1)
    keys_uids = padded_flat[flat_to_key]  # [num_subsets, kss]

    # token_atoms_to_queries: token_atoms_layout -> queries_layout
    taq_idx, taq_mask = _compute_gather_idxs_np(uid_grid, queries_uids)
    # token_atoms_to_keys (intermediate, used to build tokens_to_keys)
    tak_idx, tak_mask = _compute_gather_idxs_np(uid_grid, keys_uids)
    # queries_to_keys
    qk_idx, qk_mask = _compute_gather_idxs_np(queries_uids, keys_uids)
    # queries_to_token_atoms
    qta_idx, qta_mask = _compute_gather_idxs_np(queries_uids, uid_grid)

    # tokens_to_queries / tokens_to_keys: route the per-token index through the
    # token_atoms->queries/keys gathers (AF3 builds these via convert()).
    token_idxs = np.broadcast_to(
        np.arange(n_token, dtype=np.int64)[:, None], uid_grid.shape
    )
    t2q_idx = _convert_np(taq_idx, taq_mask, token_idxs, (0, 1))
    t2q_mask = _convert_np(taq_idx, taq_mask, token_atoms_mask, (0, 1)).astype(bool)
    t2k_idx = _convert_np(tak_idx, tak_mask, token_idxs, (0, 1))
    t2k_mask = _convert_np(tak_idx, tak_mask, token_atoms_mask, (0, 1)).astype(bool)

    out = {}
    out.update(_gi_np("token_atoms_to_queries", taq_idx, taq_mask, uid_grid.shape))
    out.update(_gi_np("tokens_to_queries", t2q_idx, t2q_mask, (n_token,)))
    out.update(_gi_np("tokens_to_keys", t2k_idx, t2k_mask, (n_token,)))
    out.update(_gi_np("queries_to_keys", qk_idx, qk_mask, queries_uids.shape))
    out.update(_gi_np("queries_to_token_atoms", qta_idx, qta_mask, queries_uids.shape))
    return out, num_atoms, num_atoms_padded


def _gi_np(prefix, gather_idxs_np, gather_mask_np, input_shape):
    return {
        f"{prefix}:gather_idxs": torch.from_numpy(np.ascontiguousarray(gather_idxs_np)).to(torch.int64),
        f"{prefix}:gather_mask": torch.from_numpy(np.ascontiguousarray(gather_mask_np)).to(torch.bool),
        f"{prefix}:input_shape": torch.tensor(list(input_shape), dtype=torch.int64),
    }


def _build_pseudo_beta_gather(uid_grid: np.ndarray, center_slot: np.ndarray):
    """token_atoms_to_pseudo_beta: token_atoms_layout -> per-token center atom.

    ``center_slot[t]`` is the dense slot of token t's pseudo-beta / representative
    atom. We form the pseudo-beta target layout (one uid per token) and gather.
    """
    n_token, _ = uid_grid.shape
    pb_uids = np.zeros((n_token,), dtype=np.int64)
    for t in range(n_token):
        s = int(center_slot[t])
        pb_uids[t] = uid_grid[t, s]
        if pb_uids[t] == 0:
            # fall back to first present atom in the token
            present = np.nonzero(uid_grid[t])[0]
            if present.size:
                pb_uids[t] = uid_grid[t, present[0]]
    pb_idx, pb_mask = _compute_gather_idxs_np(uid_grid, pb_uids)
    return _gi_np("token_atoms_to_pseudo_beta", pb_idx, pb_mask, uid_grid.shape)


def _empty_gather(prefix: str, input_shape: tuple[int, ...]):
    """An empty (zero-row) GatherInfo — for bond layouts with no bonds."""
    return {
        f"{prefix}:gather_idxs": torch.zeros((0, 2), dtype=torch.int64),
        f"{prefix}:gather_mask": torch.zeros((0, 2), dtype=torch.bool),
        f"{prefix}:input_shape": torch.tensor(list(input_shape), dtype=torch.int64),
    }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

def standard_batch_to_af3(
    batch: dict,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    """Convert a standard batch into the flat AF feature dict.

    Returns a ``dict[str, Tensor]`` suitable for
    ``feat_batch.Batch.from_data_dict`` and the torchfold ``AlphaFold3``
    forward. Only INPUT features are produced (no labels / ground truth).
    """
    feats = _ife(batch)
    if device is None:
        device = feats["ref_pos"].device

    f32 = torch.float32

    # ---- token-level scalars ----
    n_token = int(feats["asym_id"].shape[0])
    n_atom = int(feats["atom_to_token_idx"].shape[0])

    a2t = feats["atom_to_token_idx"].to(torch.int64)
    a2ta = feats["atom_to_tokatom_idx"].to(torch.int64)

    # Dense uid grid (numpy for the gather math).
    uid_grid, max_dense = _build_token_atom_uid_grid(
        a2t.cpu().numpy(), a2ta.cpu().numpy(), n_token
    )

    out: dict[str, torch.Tensor] = {}

    # ====================== TokenFeatures ======================
    # restype is one-hot[N_token,32] over the 32-type vocab -> argmax,
    # then bridge 32-vocab -> AF3 31-vocab (clamp UNK_DNA(31) -> nucleic UNK(30)).
    aatype = _restype_idx_32_to_31(feats["restype"].argmax(dim=-1).to(torch.int64))
    out["aatype"] = aatype.to(device)
    out["residue_index"] = feats["residue_index"].to(torch.int64).to(device)
    out["token_index"] = feats["token_index"].to(torch.int64).to(device)
    out["asym_id"] = feats["asym_id"].to(torch.int64).to(device)
    out["entity_id"] = feats["entity_id"].to(torch.int64).to(device)
    out["sym_id"] = feats["sym_id"].to(torch.int64).to(device)
    out["seq_length"] = torch.tensor(n_token, dtype=torch.int64, device=device)

    # seq_mask: token present (all cropped tokens are real here).
    out["seq_mask"] = torch.ones(n_token, dtype=torch.bool, device=device)

    first_atom_of_token = torch.zeros(n_token, dtype=torch.int64)
    # last write wins is fine; use scatter of atom idx by token (any atom suffices)
    first_atom_of_token[a2t.cpu()] = torch.arange(n_atom, dtype=torch.int64)
    fao = first_atom_of_token.to(device)
    for src, dst in [
        ("is_protein", "is_protein"),
        ("is_rna", "is_rna"),
        ("is_dna", "is_dna"),
        ("is_ligand", "is_ligand"),
    ]:
        per_atom = feats[src].to(torch.bool).to(device)
        out[dst] = per_atom[fao]
    out["is_nonstandard_polymer_chain"] = torch.zeros(
        n_token, dtype=torch.bool, device=device
    )
    out["is_water"] = torch.zeros(n_token, dtype=torch.bool, device=device)

    # ====================== MSA ======================
    # msa rows are 32-vocab indices -> bridge to AF3 31-vocab (model one-hots to 32
    # via NUM+1, so clamping 31->30 keeps the gap channel free + stays in-range).
    out["msa"] = _restype_idx_32_to_31(feats["msa"].to(torch.int64)).to(device)
    msa_mask = torch.ones_like(out["msa"], dtype=torch.bool)
    out["msa_mask"] = msa_mask
    _del_val = feats["deletion_value"].to(f32).to(device).clamp(0.0, 0.999)
    out["deletion_matrix"] = 3.0 * torch.tan(_del_val * (torch.pi / 2.0))
    # profile is a 32-wide distribution -> reorder/merge to AF3 31-wide.
    out["profile"] = _profile_32_to_31(feats["profile"].to(f32)).to(device)
    out["deletion_mean"] = feats["deletion_mean"].to(f32).to(device)
    out["num_alignments"] = torch.tensor(
        int(feats["msa"].shape[0]), dtype=torch.int64, device=device
    )

    # ====================== Templates ======================
    out["template_aatype"] = _restype_idx_32_to_31(
        feats["template_aatype"].to(torch.int64)
    ).to(device)
    out["template_atom_positions"] = feats["template_atom_positions"].to(f32).to(device)
    out["template_atom_mask"] = feats["template_atom_mask"].to(torch.bool).to(device)

    # ====================== RefStructure (dense) ======================
    # positions [N_atom,3] -> [N_token, max_dense, 3]
    ref_pos = feats["ref_pos"].to(f32).to(device)
    out["ref_pos"] = _scatter_flat_to_dense(ref_pos, a2t, a2ta, n_token, max_dense)
    # mask [N_atom] -> [N_token, max_dense] bool
    ref_mask_flat = feats["ref_mask"].to(f32).to(device)
    out["ref_mask"] = _scatter_flat_to_dense(
        ref_mask_flat, a2t, a2ta, n_token, max_dense
    ).to(torch.bool).to(f32)
    _elem_idx = feats["ref_element"].argmax(dim=-1).to(torch.int64).to(device)
    ref_elem_flat = torch.where(
        _elem_idx < 118, _elem_idx + 1, torch.zeros_like(_elem_idx)
    )
    out["ref_element"] = _scatter_flat_to_dense(
        ref_elem_flat, a2t, a2ta, n_token, max_dense, fill=0.0
    ).to(torch.int64)
    # charge: per-atom int -> float
    ref_charge_flat = feats["ref_charge"].to(f32).to(device)
    out["ref_charge"] = _scatter_flat_to_dense(
        ref_charge_flat, a2t, a2ta, n_token, max_dense
    )
    # atom_name_chars: one-hot[N_atom,4,64] -> [N_atom,4] codes -> dense
    ref_name_flat = feats["ref_atom_name_chars"].argmax(dim=-1).to(torch.int64).to(device)
    out["ref_atom_name_chars"] = _scatter_flat_to_dense(
        ref_name_flat, a2t, a2ta, n_token, max_dense, fill=0.0
    ).to(torch.int64)
    # ref_space_uid [N_atom] -> dense int
    ref_space_flat = feats["ref_space_uid"].to(torch.int64).to(device)
    out["ref_space_uid"] = _scatter_flat_to_dense(
        ref_space_flat, a2t, a2ta, n_token, max_dense, fill=0.0
    ).to(torch.int64)

    # ====================== PredictedStructureInfo ======================
    # pred_dense_atom_mask = dense atom-present mask [N_token, max_dense]
    pred_dense_atom_mask = torch.from_numpy(uid_grid != 0).to(torch.bool).to(device)
    out["pred_dense_atom_mask"] = pred_dense_atom_mask

    # residue_center_index: dense slot of each token's representative atom.
    # distogram_rep_atom_mask [N_atom] flags the rep atom; map to slot.
    center_slot = np.zeros(n_token, dtype=np.int64)
    rep_mask = feats["distogram_rep_atom_mask"].to(torch.bool).cpu().numpy()
    a2t_np = a2t.cpu().numpy()
    a2ta_np = a2ta.cpu().numpy()
    rep_atoms = np.nonzero(rep_mask)[0]
    for a in rep_atoms:
        center_slot[a2t_np[a]] = a2ta_np[a]
    out["residue_center_index"] = torch.from_numpy(center_slot).to(torch.int64).to(device)

    # ====================== Frames ======================
    out["frames_mask"] = feats["has_frame"].to(torch.bool).to(device)

    # ====================== Atom-cross-attention gathers ======================
    aca, _, _ = _build_atom_cross_att_gathers(uid_grid)
    for k, v in aca.items():
        if k.endswith(":input_shape"):
            out[k] = v.to(torch.device("cpu"))
        else:
            out[k] = v.to(device)

    # ====================== PseudoBeta gather ======================
    pb = _build_pseudo_beta_gather(uid_grid, center_slot)
    for k, v in pb.items():
        if k.endswith(":input_shape"):
            out[k] = v.to(torch.device("cpu"))
        else:
            out[k] = v.to(device)

    # ====================== Bond layouts ======================
    _token_bonds = feats.get("token_bonds", None)
    if _token_bonds is not None:
        _tb = _token_bonds.to(device)
        _bond_pairs = torch.nonzero(_tb > 0.5, as_tuple=False).to(torch.int64)   # [N_bond, 2]
        out["tokens_to_polymer_ligand_bonds:gather_idxs"] = _bond_pairs.to(device)
        out["tokens_to_polymer_ligand_bonds:gather_mask"] = torch.ones_like(_bond_pairs, dtype=torch.bool).to(device)
        out["tokens_to_polymer_ligand_bonds:input_shape"] = torch.tensor([n_token], dtype=torch.int64)
        _empty_prefixes = ("token_atoms_to_polymer_ligand_bonds", "tokens_to_ligand_ligand_bonds")
    else:
        # no token_bonds available -> fall back to old all-empty behaviour (protein-only safe)
        _empty_prefixes = ("tokens_to_polymer_ligand_bonds", "token_atoms_to_polymer_ligand_bonds", "tokens_to_ligand_ligand_bonds")
    for prefix in _empty_prefixes:
        eg = _empty_gather(prefix, (n_token,))
        for k, v in eg.items():
            out[k] = v.to(torch.device("cpu")) if k.endswith(":input_shape") else v.to(device)

    return out


def standard_batch_to_af3_batch(
    batch: dict,
    device: Optional[torch.device] = None,
) -> "feat_batch.Batch":
    """Convenience: adapter + ``feat_batch.Batch.from_data_dict``."""
    flat = standard_batch_to_af3(batch, device=device)
    return feat_batch.Batch.from_data_dict(flat)
