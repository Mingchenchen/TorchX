"""torchfold.runner.confidence_summary
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from torchfold.metrics.clash_af3 import af3_has_clash
from torchfold.runner.confidence_perchain import (
    calculate_chain_based_gpde,
    calculate_chain_based_plddt,
    calculate_chain_based_ptm,
    calculate_chain_pair_pae,
)
from torchfold.runner.sample_confidence import (
    calculate_iptm,
    calculate_ptm,
    compute_contact_prob,
    logits_to_score,  # re-exported for callers; not used directly here
)

__all__ = ["compute_full_data_and_summary"]

# --------------------------------------------------------------------------- #
# Head-verified bin constants (see module docstring for provenance).
# --------------------------------------------------------------------------- #
_PAE_MIN_BIN = 0.0
_PAE_MAX_BIN = 32.0  # head pae_bin_centers == get_bin_centers(0,32,64)
_PAE_NO_BINS = 64

# PDE bins kept for documentation only -- full_pde is already scored.
_PDE_MIN_BIN = 0.0
_PDE_MAX_BIN = 32.0
_PDE_NO_BINS = 64


# Distogram (contact_probs) bins -- torchfold DistogramHead defaults.
_DISTOGRAM_FIRST_BREAK = 2.3125
_DISTOGRAM_LAST_BREAK = 21.6875
_DISTOGRAM_NUM_BINS = 64


# --------------------------------------------------------------------------- #
# json-ready conversion helpers
# --------------------------------------------------------------------------- #
def _to_jsonable(x: Any) -> Any:
    """Convert a tensor / scalar to a json-ready python float or nested list."""
    if torch.is_tensor(x):
        x = x.detach().to(torch.float32).cpu()
        if x.dim() == 0:
            return float(x.item())
        return x.tolist()
    return x


def _to_numpy(x: Any):
    """Convert a tensor to a (detached, cpu, float32) numpy array."""
    if torch.is_tensor(x):
        return x.detach().to(torch.float32).cpu().numpy()
    return x


# --------------------------------------------------------------------------- #
# Top-level aggregator
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_full_data_and_summary(
    out: dict,
    feats: dict,
    pred_coordinate: torch.Tensor,
    *,
    num_recycles: int,
    need_full_data: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Aggregate a torchfold AF3 inference output into standard dicts.

    Args:
        out: torchfold AF3 inference forward output (one seed). Must contain
            ``"pae_logits"``  [N_sample, N_token, N_token, 64] (raw logits),
            ``"full_pde"``    [N_sample, N_token, N_token]     (ALREADY scored),
            ``"predicted_lddt"`` [N_sample, N_token, max_dense] in 0..100 (dense)
                                 (a [N_token, max_dense] sample-shared tensor is
                                 also accepted and broadcast), and
            ``out["distogram"]["logits"]`` [N_token, N_token, 64] (sample-shared)
                                 (a leading sample axis is also accepted).
        feats: input_feature_dict; uses
            ``"asym_id"``            [N_token]  long  (per-token chain id),
            ``"has_frame"``          [N_token]  bool,
            ``"atom_to_token_idx"``  [N_atom]   long,
            ``"is_ligand"``          [N_atom]   bool  (atom_is_polymer = ~is_ligand),
            ``"atom_to_tokatom_idx"``[N_atom]   long  (dense->flat for plddt).
        pred_coordinate: [N_sample, N_atom, 3] flat predicted coords (for clash).
        num_recycles: int, the recycle count for this prediction.
        need_full_data: if True, also populate per-sample full_data dicts.

    Returns:
        (summary_confidence_list, full_data_list):
          summary_confidence_list: list length N_sample of json-ready dicts.
          full_data_list: list length N_sample; full per-sample arrays if
            need_full_data else [{}] * N_sample.
    """
    device = pred_coordinate.device

    # ---- features ----------------------------------------------------------
    asym_id = feats["asym_id"].to(torch.int64).to(device)            # [N_token]
    has_frame = feats["has_frame"].to(torch.bool).to(device)         # [N_token]
    a2t = feats["atom_to_token_idx"].to(torch.int64).to(device)      # [N_atom]
    is_ligand = feats["is_ligand"].to(torch.bool).to(device)         # [N_atom]
    atom_is_polymer = ~is_ligand                                     # [N_atom]

    N_sample = pred_coordinate.shape[0]

    atom_is_ligand = (~atom_is_polymer).long()                       # [N_atom]
    token_is_ligand = torch.zeros_like(asym_id).scatter_add(
        0, a2t, atom_is_ligand
    )
    token_is_ligand = token_is_ligand > 0                            # [N_token] bool

    # ---- PAE: logits -> score + prob --------------------------------------
    pae_logits = out["pae_logits"].to(torch.float32).to(device)
    if pae_logits.dim() == 3:                       # [N_token,N_token,64] shared
        pae_logits = pae_logits.unsqueeze(0).expand(N_sample, -1, -1, -1)
    token_pair_pae, pae_prob = logits_to_score(
        pae_logits,
        min_bin=_PAE_MIN_BIN,
        max_bin=_PAE_MAX_BIN,
        no_bins=_PAE_NO_BINS,
        return_prob=True,
    )  # token_pair_pae: [N_sample,N_token,N_token]; pae_prob: [...,64]

    # ---- PDE: already scored ----------------------------------------------
    token_pair_pde = out["full_pde"].to(torch.float32).to(device)
    if token_pair_pde.dim() == 2:                   # [N_token,N_token] shared
        token_pair_pde = token_pair_pde.unsqueeze(0).expand(N_sample, -1, -1)

    # ---- contact_probs from distogram logits (sample-shared) ---------------
    distogram_logits = out["distogram"]["logits"].to(torch.float32).to(device)
    if distogram_logits.dim() == 4:                 # [N_sample,...] -> take [0]
        distogram_logits = distogram_logits[0]
    contact_probs = compute_contact_prob(
        distogram_logits,
        min_bin=_DISTOGRAM_FIRST_BREAK,
        max_bin=_DISTOGRAM_LAST_BREAK,
        no_bins=_DISTOGRAM_NUM_BINS,
        threshold=8.0,
    )  # [N_token, N_token]

    # ---- atom_plddt: dense (0..100) -> per-atom, then to 0..1 ---
    predicted_lddt = out["predicted_lddt"].to(torch.float32).to(device)
    if predicted_lddt.dim() == 2:                   # [N_token,max_dense] shared
        predicted_lddt = predicted_lddt.unsqueeze(0).expand(N_sample, -1, -1)
    a2ta = feats["atom_to_tokatom_idx"].to(torch.int64).to(device)   # [N_atom]
    # dense [N_sample,N_token,max_dense] -> [N_sample,N_atom]; 0..100 -> 0..1.
    atom_plddt = predicted_lddt[..., a2t, a2ta] / 100.0              # [N_sample,N_atom]

    # ---- clash per sample (AF3 geometric clash) ----------------------------
    has_clash = af3_has_clash(
        pred_coordinate,
        asym_id,
        a2t,
        atom_is_polymer,
        threshold=1.1,
    ).to(torch.float32)                                             # [N_sample]

    summary: dict[str, torch.Tensor] = {}

    summary["plddt"] = atom_plddt.mean(dim=-1) * 100.0              # [N_sample]

    summary["gpde"] = (token_pair_pde * contact_probs).sum(dim=(-1, -2)) / (
        contact_probs.sum() + 0.0
    ).clamp_min(1e-12)

    summary["ptm"] = calculate_ptm(
        pae_prob,
        has_frame=has_frame,
        min_bin=_PAE_MIN_BIN,
        max_bin=_PAE_MAX_BIN,
        no_bins=_PAE_NO_BINS,
    )
    summary["iptm"] = calculate_iptm(
        pae_prob,
        has_frame=has_frame,
        asym_id=asym_id,
        min_bin=_PAE_MIN_BIN,
        max_bin=_PAE_MAX_BIN,
        no_bins=_PAE_NO_BINS,
    )

    summary.update(
        calculate_chain_based_gpde(
            token_pair_pde=token_pair_pde,
            contact_probs=contact_probs,
            asym_id=asym_id,
        )
    )
    summary.update(
        calculate_chain_based_ptm(
            pae_prob,
            has_frame=has_frame,
            asym_id=asym_id,
            token_is_ligand=token_is_ligand,
            min_bin=_PAE_MIN_BIN,
            max_bin=_PAE_MAX_BIN,
            no_bins=_PAE_NO_BINS,
        )
    )
    summary.update(
        calculate_chain_based_plddt(atom_plddt, asym_id, a2t)
    )
    summary.update(
        calculate_chain_pair_pae(
            token_pair_pae=token_pair_pae,
            asym_id=asym_id,
            token_has_frame=has_frame,
        )
    )

    summary["has_clash"] = has_clash
    summary["disorder"] = torch.zeros_like(summary["ptm"])
    summary["ranking_score"] = (
        0.8 * summary["iptm"]
        + 0.2 * summary["ptm"]
        + 0.5 * summary["disorder"]
        - 100.0 * summary["has_clash"]
    )

    # ---- break down to per-sample json-ready dicts -------------------------
    # Per-sample keys index the leading sample axis; num_recycles is shared.
    per_sample_keys = [k for k in summary if k != "num_recycles"]
    summary_confidence_list: list[dict] = []
    for i in range(N_sample):
        d: dict[str, Any] = {}
        for k in per_sample_keys:
            d[k] = _to_jsonable(summary[k][i])
        d["num_recycles"] = int(num_recycles)
        summary_confidence_list.append(d)

    # ---- full_data (optional) ---------------------------------------------
    if need_full_data:
        contact_probs_np = _to_numpy(contact_probs)  # [N_token,N_token], shared
        full_data_list: list[dict] = []
        for i in range(N_sample):
            full_data_list.append(
                {
                    "atom_plddt": _to_numpy(atom_plddt[i]),          # [N_atom] 0..1
                    "token_pair_pae": _to_numpy(token_pair_pae[i]),  # [N_token,N_token]
                    "token_pair_pde": _to_numpy(token_pair_pde[i]),  # [N_token,N_token]
                    "contact_probs": contact_probs_np,               # [N_token,N_token]
                }
            )
    else:
        full_data_list = [{} for _ in range(N_sample)]

    return summary_confidence_list, full_data_list
