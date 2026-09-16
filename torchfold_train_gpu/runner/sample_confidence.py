"""torchfold.runner.sample_confidence -- per-sample summary confidence.



We only port what `_evaluate` needs:

* ``get_bin_centers(min_bin, max_bin, no_bins)``
* ``logits_to_score(logits, ...)``
* ``compute_contact_prob(distogram_logits, ...)`` -- contact probability from
  the distogram head's logits, used to weight the per-pair PDE in gpde.
* ``calculate_normalization(N)`` -- TM-score d_0 constant (Zhang 2004, eq 5).
* ``calculate_ptm(pae_prob, has_frame, min_bin, max_bin, no_bins)``
* ``calculate_iptm(pae_prob, has_frame, asym_id, min_bin, max_bin, no_bins)``
* ``compute_summary_confidence(out, feats, has_clash_per_sample)`` --
  top-level helper that wires all of the above to a tensor of per-sample
  scalars: ``{"plddt", "gpde", "ptm", "iptm", "ranking_score"}``.

Not ported: chain-pair / chain-based confidence variants, vdW clash (we use
the AF3 geometric clash already produced by ``_compute_batch_metrics``),
disorder.

All helpers operate on the AF3-shape tensors that torchfold's
``AlphaFold3.forward`` returns in inference mode (see
``torchfold/model/alphafold3.py:_confidence_output_keys`` + the distogram
head output at ``out["distogram"]["logits"]``).
"""
from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------
def get_bin_centers(
    min_bin: float, max_bin: float, no_bins: int
) -> torch.Tensor:
    """Centres of `no_bins` equal-width bins spanning ``[min_bin, max_bin]``.

    AF3 bin convention: the rightmost bin tops at ``max_bin``, so the
    leftmost edge starts at ``min_bin`` and each centre sits at
    ``edge + bin_width/2``.
    """
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(
        start=min_bin,
        end=max_bin - bin_width,
        steps=no_bins,
    )
    return boundaries + 0.5 * bin_width


def logits_to_score(
    logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    return_prob: bool = False,
):
    """Softmax over the last axis, then expectation w.r.t. bin centres."""
    prob = torch.nn.functional.softmax(logits, dim=-1)
    bin_centers = get_bin_centers(min_bin, max_bin, no_bins).to(logits.device)
    score = prob @ bin_centers
    if return_prob:
        return score, prob
    return score


def compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    threshold: float = 8.0,
) -> torch.Tensor:
    """Probability that the predicted pair distance is below ``threshold``.

    Sums the softmax of distogram_logits over all bins whose CENTRE is below
    `threshold` (default 8 A -- standard AF3 contact definition).

    Args:
        distogram_logits: ``[..., N_token, N_token, no_bins]``.
    Returns:
        ``[..., N_token, N_token]`` contact probability in [0, 1].
    """
    prob = torch.nn.functional.softmax(distogram_logits, dim=-1)
    centres = get_bin_centers(min_bin, max_bin, no_bins).to(prob.device)
    thres_idx = int((centres < threshold).sum().item())
    return prob[..., :thres_idx].sum(-1)


# ---------------------------------------------------------------------------
# TM-score helpers
# ---------------------------------------------------------------------------
def calculate_normalization(N: int) -> float:
    """TM-score d_0 normalisation (Zhang 2004 eq 5). Used for pTM / ipTM."""
    return 1.24 * (max(N, 19) - 15) ** (1.0 / 3.0) - 1.8


def calculate_ptm(
    pae_prob: torch.Tensor,
    has_frame: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """AF3 pTM score (mean over all token pairs, restricted to has_frame tokens).

    Args:
        pae_prob: ``[..., N_token, N_token, no_bins]`` softmax of pae_logits.
        has_frame: ``[N_token]`` bool -- token has a CA + 2-atom backbone frame.
        token_mask: optional ``[N_token]`` bool to restrict everything first.

    Returns:
        ``[...]`` pTM, higher is better.
    """
    has_frame = has_frame.bool()
    if token_mask is not None:
        token_mask = token_mask.bool()
        pae_prob = pae_prob[..., token_mask, :, :][..., :, token_mask, :]
        has_frame = has_frame[token_mask]
    if has_frame.sum() == 0:
        return torch.zeros(size=pae_prob.shape[:-3], device=pae_prob.device)
    N_d = has_frame.shape[-1]
    ptm_norm = calculate_normalization(N_d)
    bin_center = get_bin_centers(min_bin, max_bin, no_bins).to(pae_prob.device)
    per_bin_weight = 1 / (1 + (bin_center / ptm_norm) ** 2)  # [no_bins]
    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)  # [..., N_d, N_d]
    # AF3 pTM = max over i of mean_j token_token_ptm[i,j], restricted to
    # frame tokens.
    return token_token_ptm.mean(dim=-1)[..., has_frame].max(dim=-1).values


def calculate_iptm(
    pae_prob: torch.Tensor,
    has_frame: torch.Tensor,
    asym_id: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """AF3 ipTM score (pTM restricted to CROSS-CHAIN token pairs).

    Identical to ``calculate_ptm`` except the per-token average over
    `j` is masked to ``asym_id[i] != asym_id[j]`` -- only inter-chain
    pairs contribute. For a monomer (single asym_id) this returns 0.

    Returns:
        ``[...]`` ipTM, higher is better.
    """
    has_frame = has_frame.bool()
    if token_mask is not None:
        token_mask = token_mask.bool()
        pae_prob = pae_prob[..., token_mask, :, :][..., :, token_mask, :]
        has_frame = has_frame[token_mask]
        asym_id = asym_id[token_mask]
    if has_frame.sum() == 0:
        return torch.zeros(size=pae_prob.shape[:-3], device=pae_prob.device)
    N_d = has_frame.shape[-1]
    ptm_norm = calculate_normalization(N_d)
    bin_center = get_bin_centers(min_bin, max_bin, no_bins).to(pae_prob.device)
    per_bin_weight = 1 / (1 + (bin_center / ptm_norm) ** 2)
    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)  # [..., N_d, N_d]
    is_diff_chain = (asym_id[None, :] != asym_id[:, None]).to(
        token_token_ptm.dtype
    )  # [N_d, N_d]
    # Per-token: mean of token_token_ptm over cross-chain partners only.
    iptm_per_token = (token_token_ptm * is_diff_chain).sum(dim=-1) / (
        eps + is_diff_chain.sum(dim=-1)
    )  # [..., N_d]
    return iptm_per_token[..., has_frame].max(dim=-1).values


# ---------------------------------------------------------------------------
# Top-level wrapper used by Trainer._evaluate
# ---------------------------------------------------------------------------
# Bin parameters baked into torchfold's confidence head (see
# torchfold/model/modules/confidence.py): pae bin centers 0.25..31.75 A
# (width 0.5, 64 bins) == get_bin_centers(0,32,64); the head's literal 31.0 is
# the linspace(0,31,63) endpoint, NOT max_bin -> feed max_bin=32 here;
# distogram bins are different and read from the distogram head.
_PAE_MIN_BIN = 0.0
_PAE_MAX_BIN = 32.0  # was 31.0 (AF3-head centers == get_bin_centers(0,32,64))
_PAE_NO_BINS = 64

# Distogram head config (torchfold/model/modules/head.py DistogramHead defaults):
# breaks are linspace(first_break, last_break, num_bins - 1), so the bin layout
# the contact-prob softmax sees has effective edges from 0 to last_break +
# bin_width. We pass the head's first_break / last_break + num_bins and let
# `compute_contact_prob` truncate at 8 A.
_DISTOGRAM_FIRST_BREAK = 2.3125
_DISTOGRAM_LAST_BREAK = 21.6875
_DISTOGRAM_NUM_BINS = 64


def compute_summary_confidence(
    out: dict,
    feats: dict,
    has_clash_per_sample: torch.Tensor,
    *,
    distogram_first_break: float = _DISTOGRAM_FIRST_BREAK,
    distogram_last_break: float = _DISTOGRAM_LAST_BREAK,
    distogram_num_bins: int = _DISTOGRAM_NUM_BINS,
) -> dict:
    """Compute per-sample confidence scalars from an inference forward output.

    Args:
        out: torchfold AF3 inference output dict; must contain
            ``predicted_lddt`` ``[N_sample, N_token, max_dense]`` in [0, 100],
            ``full_pde`` ``[N_sample, N_token, N_token]``,
            ``pae_logits`` ``[N_sample, N_token, N_token, 64]``,
            ``distogram"]["logits"`` ``[N_token, N_token, 64]`` (no sample axis;
            distogram is sample-shared in AF3).
        feats: input feature dict; needs ``asym_id [N_token]``,
            ``has_frame [N_token]``, and the (a2t, a2ta) maps already in scope
            in the caller. Here we read ``asym_id`` + ``has_frame`` directly;
            the caller passes ``a2t`` / ``a2ta`` and per-atom mask via
            ``has_clash_per_sample`` already computed (so we don't recompute
            the geometric clash here).
        has_clash_per_sample: ``[N_sample]`` 0/1 tensor of "any AF3 clash in
            this sample" (the caller produces it from the existing clash
            metric on dense coords).

    Returns:
        dict of per-sample scalar tensors, each shape ``[N_sample]``:
        ``"plddt"``, ``"gpde"``, ``"ptm"``, ``"iptm"``, ``"ranking_score"``.
        Tensors live on whatever device ``out["pae_logits"]`` is on.
    """
    pae_logits = out["pae_logits"].to(torch.float32)
    # PAE: bin axis last, normalize to probabilities.
    pae_prob = torch.nn.functional.softmax(pae_logits, dim=-1)

    asym_id = feats["asym_id"].to(torch.int64).to(pae_logits.device)
    has_frame = feats["has_frame"].to(torch.bool).to(pae_logits.device)

    # pTM / ipTM per sample. calculate_*ptm broadcasts over the leading
    # sample dim of pae_prob.
    ptm = calculate_ptm(
        pae_prob, has_frame=has_frame,
        min_bin=_PAE_MIN_BIN, max_bin=_PAE_MAX_BIN, no_bins=_PAE_NO_BINS,
    )
    iptm = calculate_iptm(
        pae_prob, has_frame=has_frame, asym_id=asym_id,
        min_bin=_PAE_MIN_BIN, max_bin=_PAE_MAX_BIN, no_bins=_PAE_NO_BINS,
    )

    # gpde: contact-probability-weighted PDE. distogram is sample-shared.
    distogram_logits = out["distogram"]["logits"].to(torch.float32)
    contact_prob = compute_contact_prob(
        distogram_logits,
        min_bin=distogram_first_break,
        max_bin=distogram_last_break,
        no_bins=distogram_num_bins,
        threshold=8.0,
    )  # [N_token, N_token]
    full_pde = out["full_pde"].to(torch.float32)  # [N_sample, N_token, N_token]
    gpde = (
        (full_pde * contact_prob).sum(dim=(-1, -2))
        / (contact_prob.sum() + 1e-8)
    )  # [N_sample]

    # pLDDT per sample: mean over per-atom-dense plddt, weighted by the
    # token_mask. Without per-atom-valid masking the padded atom slots dilute
    # the mean uniformly across samples (does not affect argmax for the
    # ranker but skews the absolute number). The caller can pass a more
    # careful mask through feats if needed; default = uniform over the
    # (N_token, max_dense) dense grid.
    predicted_lddt = out["predicted_lddt"].to(torch.float32)
    if predicted_lddt.dim() == 2:
        predicted_lddt = predicted_lddt.unsqueeze(0)
    plddt = predicted_lddt.mean(dim=(-1, -2))  # [N_sample]

    # AF3 ranking_score = 0.8*iptm + 0.2*ptm + 0.5*disorder(=0) - 100*has_clash
    has_clash = has_clash_per_sample.to(torch.float32).to(pae_logits.device)
    # Broadcast scalar pae_prob.shape[:-3] to [N_sample]; pae_logits is
    # already [N_sample, ...], so ptm/iptm are [N_sample].
    ranking_score = 0.8 * iptm + 0.2 * ptm - 100.0 * has_clash

    return {
        "plddt": plddt,
        "gpde": gpde,
        "ptm": ptm,
        "iptm": iptm,
        "ranking_score": ranking_score,
    }
