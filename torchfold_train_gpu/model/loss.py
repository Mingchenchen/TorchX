"""torchfold.model.loss

  * **pred side → torchfold AF3 model output.** The losses read the AF3 forward
    output dict (``out`` of ``torchfold.model.alphafold3.AlphaFold3.forward``):

        out['distogram']['logits']                  [N_token, N_token, 64]
        out['diffusion_samples']['atom_positions']  [S, N_token, max_dense, 3]
        out['diffusion_samples']['mask']             [S, N_token, max_dense] (bool)
        out['predicted_lddt']                        [S, N_token, 24]   (expectation*100)
        out['predicted_experimentally_resolved']     [S, N_token, 24]   (P(resolved))
        out['full_pae']                              [S, N_token, N_token] (Å, expectation)
        out['full_pde']                              [S, N_token, N_token] (Å, expectation)

  * **label side:

        batch['label_dict']['coordinate']        [N_atom, 3]   (FLAT atom layout)
        batch['label_dict']['coordinate_mask']   [N_atom]
        batch['input_feature_dict'][...]          is_rna/is_dna/is_ligand, bond_mask,
                                                   *_rep_atom_mask, has_frame,
                                                   frame_atom_index, resolution,
                                                   atom_to_token_idx, atom_to_tokatom_idx,
                                                   atom_perm_list


"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ===========================================================================
# AF3 constants (strictly AlphaFold3 SI)
# ===========================================================================
# Distogram: bins 2.3125 .. 21.6875 over 64 bins.
_AF3_DISTOGRAM_MIN_BIN = 2.3125
_AF3_DISTOGRAM_MAX_BIN = 21.6875
_AF3_DISTOGRAM_NO_BINS = 64
# pLDDT: 50 bins over [0, 1]
_AF3_PLDDT_NO_BINS = 50
# PAE/PDE error bins: AF3 confidence head uses max_error_bin = 31.0 Å over 64 bins.
_AF3_PAE_MAX_BIN = 31.0
_AF3_PDE_MAX_BIN = 31.0
_AF3_CONF_NO_BINS = 64
# Diffusion: SIGMA_DATA = 16
_AF3_SIGMA_DATA = 16.0
# EDM per-sample noise-weight cap. w(sigma)=(s^2+sd^2)/(s*sd)^2 ~ 1/s^2 explodes as
# sigma->0, so rare small-sigma training samples inject 1e4-1e5 loss spikes (huge
# value, ~zero gradient) that dominate the mean train loss. The reference
# training (compute_noise_weights, max_weight=10.0) clamps this; torchfold was
# missing it. Match the reference. Env TORCHFOLD_NOISE_WEIGHT_MAX=0 disables (old).
_NOISE_WEIGHT_MAX = float(os.environ.get("TORCHFOLD_NOISE_WEIGHT_MAX", "10.0"))
# Weighted-MSE per-atom weights (AF3 eq. for L_MSE).
_AF3_MSE_WEIGHT = 1.0 / 3.0     # the 1/3 prefactor on the aligned-MSE
_AF3_WEIGHT_DNA = 5.0
_AF3_WEIGHT_RNA = 5.0
_AF3_WEIGHT_LIGAND = 10.0
# Bespoke-LDDT inclusion radius (AF3: 30 Å nucleotide / 15 Å otherwise).
_AF3_LDDT_NUC_RADIUS = 30.0
_AF3_LDDT_NOT_NUC_RADIUS = 15.0
# AF3 diffusion-block term weights (AF3 SI eq. for L_diffusion):
#   L_diffusion = alpha_diffusion * ( (1/3) L_MSE + alpha_bond L_bond + L_smooth_lddt )
# AF3 SI fine-tuning: alpha_bond = 1.0; smooth_lddt has weight 1.0; alpha_diffusion = 4.0.
_AF3_ALPHA_DIFFUSION = 4.0
_AF3_ALPHA_BOND = 1.0
_AF3_WEIGHT_SMOOTH_LDDT = 1.0
_AF3_ALPHA_DISTOGRAM = 3e-2
_AF3_ALPHA_CONFIDENCE = 1e-4
_AF3_ALPHA_PAE = 1.0
_AF3_ALPHA_EXCEPT_PAE = 1.0
# Confidence resolution gate (AF3 SI: 0.1 .. 4.0 Å).
_AF3_RESOLUTION_MIN = -2.0    # 0721
_AF3_RESOLUTION_MAX = 4.0


# ===========================================================================
# util helpers.
# ===========================================================================
def cdist(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pairwise Euclidean distance."""
    if b is None:
        b = a
    return torch.cdist(a, b)


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """Expand a tensor at ``dim`` by ``n``."""
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1:]
    return x.expand(*before_shape, n, *after_shape)


@lru_cache(maxsize=8)
def _get_off_diagonal_mask(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached (1 - eye(n))."""
    return 1 - torch.eye(n, device=device, dtype=dtype)


def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Reduction wrapper."""
    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross entropy with soft (probability) labels. Vendored."""
    return -1 * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)


def _af3_error_breaks(max_bin: float, no_bins: int, device, dtype=torch.float32) -> torch.Tensor:
    """AF3 confidence-head error-bin breaks: ``linspace(0, max_bin, no_bins - 1)``.

    Identical to ``ConfidenceHead.distance_breaks`` / ``pae_breaks`` (the head's
    own bin layout used to take the expectation). ``no_bins - 1`` interior breaks
    induce ``no_bins`` bins; ``bin_idx = sum(value > breaks)`` lands in
    ``[0, no_bins-1]`` index-for-index with the head's logit bins.
    """
    return torch.linspace(0.0, max_bin, no_bins - 1, device=device, dtype=dtype)


def _bin_indices_from_breaks(value: torch.Tensor, breaks: torch.Tensor) -> torch.Tensor:
    """``bin_idx = sum(value > breaks)`` in ``[0, len(breaks)]`` (AF3 convention)."""
    return torch.sum(value.unsqueeze(-1) > breaks, dim=-1)


def _align_pred_to_true(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    allowing_reflection: bool = False,
):
    """Optimal weighted rigid alignment (Kabsch)."""
    atom_mask = torch.ones(*pred_pose.shape[:-1], device=pred_pose.device, dtype=pred_pose.dtype)
    if weight is None:
        weight = atom_mask
    else:
        weight = weight * atom_mask

    weighted_n_atoms = torch.sum(weight, dim=-1, keepdim=True).unsqueeze(-1)
    pred_centroid = torch.sum(pred_pose * weight.unsqueeze(-1), dim=-2, keepdim=True) / weighted_n_atoms
    pred_centered = pred_pose - pred_centroid
    true_centroid = torch.sum(true_pose * weight.unsqueeze(-1), dim=-2, keepdim=True) / weighted_n_atoms
    true_centered = true_pose - true_centroid
    H_mat = torch.matmul(
        (pred_centered * weight.unsqueeze(-1)).transpose(-2, -1),
        true_centered * atom_mask.unsqueeze(-1),
    )
    u, s, vh = torch.linalg.svd(H_mat)
    u = u.transpose(-1, -2)
    v = vh.transpose(-1, -2)
    if not allowing_reflection:
        det = torch.linalg.det(torch.matmul(v, u))
        diagonal = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1)
        rot = torch.matmul(torch.diag_embed(diagonal).to(u.device), u)
        rot = torch.matmul(v, rot)
    else:
        rot = torch.matmul(v, u)
    pred_translated = torch.matmul(pred_centered, rot.transpose(-1, -2)) + true_centroid
    return pred_translated


def weighted_rigid_align(
    x: torch.Tensor,
    x_target: torch.Tensor,
    atom_weight: torch.Tensor,
    stop_gradient: bool = True,
) -> torch.Tensor:
    """Algorithm 28 (AF3)."""
    if stop_gradient:
        with torch.no_grad():
            x_aligned = _align_pred_to_true(
                pred_pose=x, true_pose=x_target, weight=atom_weight, allowing_reflection=False
            )
            return x_aligned.detach()
    return _align_pred_to_true(
        pred_pose=x, true_pose=x_target, weight=atom_weight, allowing_reflection=False
    )


def expressCoordinatesInFrame(
    coordinate: torch.Tensor, frames: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Algorithm 29 (AF3)."""
    a, b, c = torch.unbind(frames, dim=-2)
    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    d = coordinate[..., None, :, :] - b[..., None, :]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[..., None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return x_transformed


def gather_frame_atom_by_indices(
    coordinate: torch.Tensor, frame_atom_index: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    """Gather frame atoms by index.
    (naive 2-D ``frame_atom_index`` path only — sufficient for batch-size-1 loss)."""
    assert len(frame_atom_index.shape) == 2
    x1 = torch.index_select(coordinate, dim=dim, index=frame_atom_index[:, 0])
    x2 = torch.index_select(coordinate, dim=dim, index=frame_atom_index[:, 1])
    x3 = torch.index_select(coordinate, dim=dim, index=frame_atom_index[:, 2])
    return torch.stack([x1, x2, x3], dim=dim)


# ===========================================================================
# Diffusion sub-losses).
# ===========================================================================
class SmoothLDDTLoss(nn.Module):
    """Algorithm 27 [SmoothLDDTLoss] (AF3)."""

    def __init__(self, eps: float = 1e-10, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, c_lm=None, w1d=None):
        dist_diff = torch.abs(pred_distance - true_distance)
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)
        if w1d is not None:
            # sparse per-pair weighting: dist_diff_epsilon is [..., N_sample, N_pairs]
            # and w1d is [N_pairs] -> weighted mean over the selected pairs (interface
            # pairs carry w_iface, others 1.0). Keeps the lddt in [0, 1].
            return torch.sum(w1d * dist_diff_epsilon, dim=-1) / (w1d.sum() + self.eps)
        if c_lm is not None:
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (
                torch.sum(c_lm, dim=(-1, -2)) + self.eps
            )
        else:
            lddt = torch.mean(dist_diff_epsilon, dim=-1)
        return lddt

    def dense_forward(self, pred_coordinate, true_coordinate, lddt_mask, true_distance=None,
                      pair_weight=None, diffusion_chunk_size=None):
        """pred_coordinate [..., N_sample, N_atom, 3]; true_coordinate [..., N_atom, 3].

        pair_weight: optional float [..., N_atom, N_atom] per-pair weight (interface
        pairs carry w_iface, others 1.0, masked-out 0). When given it replaces the
        binary lddt_mask so the lddt is a multiplicative interface-weighted average.

        diffusion_chunk_size: if set, the per-sample LDDT is computed in chunks over the
        N_sample dim, each chunk wrapped in activation checkpointing so the
        [chunk, N_atom, N_atom] pred-distance is recomputed in backward rather than
        retained for all N_sample at once.
        math-identical (mean over N_sample is applied after the cat)."""
        if pair_weight is not None:
            c_lm = pair_weight.to(pred_coordinate.dtype).unsqueeze(dim=-3).detach()
        else:
            c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()
        if true_distance is None:
            true_distance = torch.cdist(true_coordinate, true_coordinate)

        def _lddt_for(pc):
            pd = torch.cdist(pc, pc)
            return self._chunk_forward(pred_distance=pd, true_distance=true_distance, c_lm=c_lm)

        N_sample = pred_coordinate.shape[-3]
        if diffusion_chunk_size is None or diffusion_chunk_size >= N_sample:
            lddt = _lddt_for(pred_coordinate)
        else:
            outs = []
            for i in range(0, N_sample, diffusion_chunk_size):
                pc_i = pred_coordinate[..., i:i + diffusion_chunk_size, :, :]
                if pc_i.requires_grad:
                    outs.append(torch.utils.checkpoint.checkpoint(_lddt_for, pc_i, use_reentrant=False))
                else:
                    outs.append(_lddt_for(pc_i))
            lddt = torch.cat(outs, dim=-1)
        lddt = lddt.mean(dim=-1)
        return 1 - loss_reduction(lddt, method=self.reduction)

    def sparse_forward(self, pred_coordinate, true_coordinate, lddt_mask,
                       pair_weight=None, diffusion_chunk_size=None):
        """Sparse SmoothLDDT: only the within-radius (l, m) atom pairs (nonzero
        lddt_mask) contribute, so the [N_atom, N_atom] distance is never built --
        peak is [N_sample, N_pairs] not [N_sample, N_atom, N_atom].

        pair_weight: optional float [N_atom, N_atom]; if given, the selected pairs are
        averaged with these per-pair weights (interface pairs w_iface, others 1.0) ->
        multiplicative interface-weighted lddt (still in [0, 1])."""
        if lddt_mask.sum() == 0:
            return (pred_coordinate.sum() * 0.0).to(pred_coordinate.dtype)
        idx = torch.nonzero(lddt_mask, as_tuple=True)
        true_l = true_coordinate.index_select(-2, idx[0])
        true_m = true_coordinate.index_select(-2, idx[1])
        true_distance_sparse = torch.linalg.vector_norm(true_l - true_m, ord=2, dim=-1)
        w_sparse = None
        if pair_weight is not None:
            w_sparse = pair_weight[idx[0], idx[1]].to(pred_coordinate.dtype).detach()

        def _lddt_for(pc):
            pl = pc.index_select(-2, idx[0])
            pm = pc.index_select(-2, idx[1])
            pd = torch.linalg.vector_norm(pl - pm, ord=2, dim=-1)
            return self._chunk_forward(pred_distance=pd, true_distance=true_distance_sparse,
                                       c_lm=None, w1d=w_sparse)

        N_sample = pred_coordinate.shape[-3]
        if diffusion_chunk_size is None or diffusion_chunk_size >= N_sample:
            lddt = _lddt_for(pred_coordinate)
        else:
            outs = []
            for i in range(0, N_sample, diffusion_chunk_size):
                pc_i = pred_coordinate[..., i:i + diffusion_chunk_size, :, :]
                if pc_i.requires_grad:
                    outs.append(torch.utils.checkpoint.checkpoint(_lddt_for, pc_i, use_reentrant=False))
                else:
                    outs.append(_lddt_for(pc_i))
            lddt = torch.cat(outs, dim=-1)
        lddt = lddt.mean(dim=-1)
        return 1 - loss_reduction(lddt, method=self.reduction)


class BondLoss(nn.Module):
    """Formula 5 [BondLoss] (AF3)."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_coordinate, true_distance, distance_mask, bond_mask,
                per_sample_scale=None, diffusion_chunk_size=None):
        # pred-distance per N_sample chunk inside activation checkpointing rather than
        # materializing [N_sample, N_atom, N_atom] at once (OOM at large crops).
        bm = (bond_mask * distance_mask).unsqueeze(dim=-3)          # [..., 1, N_atom, N_atom]
        bm_denom = torch.sum(bm + self.eps, dim=(-1, -2))           # [..., 1]
        true_d = true_distance.unsqueeze(dim=-3)                    # [..., 1, N_atom, N_atom]

        def _bond_for(pc):
            pd = torch.cdist(pc, pc)
            dist_squared_err = (pd - true_d) ** 2
            return torch.sum(dist_squared_err * bm, dim=(-1, -2)) / bm_denom

        N_sample = pred_coordinate.shape[-3]
        if diffusion_chunk_size is None or diffusion_chunk_size >= N_sample:
            bond_loss = _bond_for(pred_coordinate)
        else:
            outs = []
            for i in range(0, N_sample, diffusion_chunk_size):
                pc_i = pred_coordinate[..., i:i + diffusion_chunk_size, :, :]
                if pc_i.requires_grad:
                    outs.append(torch.utils.checkpoint.checkpoint(_bond_for, pc_i, use_reentrant=False))
                else:
                    outs.append(_bond_for(pc_i))
            bond_loss = torch.cat(outs, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale
        bond_loss = bond_loss.mean(dim=-1)
        return loss_reduction(bond_loss, method=self.reduction)

    def sparse_forward(self, pred_coordinate, true_coordinate, distance_mask, bond_mask,
                       per_sample_scale=None):
        """Sparse BondLoss: only bonded (nonzero bond_mask*distance_mask) atom pairs
        contribute. NOTE:
        This is NOT numerically identical to the dense bond path --
        dense divides by sum(bond_mask)+eps*N^2 whereas sparse divides
        by the exact pair count; sparse is the cleaner of the two."""
        bm = bond_mask * distance_mask
        idx = torch.nonzero(bm, as_tuple=True)
        pred_i = pred_coordinate.index_select(-2, idx[0])
        pred_j = pred_coordinate.index_select(-2, idx[1])
        true_i = true_coordinate.index_select(-2, idx[0])
        true_j = true_coordinate.index_select(-2, idx[1])
        pred_distance_sparse = torch.linalg.vector_norm(pred_i - pred_j, ord=2, dim=-1)
        true_distance_sparse = torch.linalg.vector_norm(true_i - true_j, ord=2, dim=-1)
        dist_squared_err = (pred_distance_sparse - true_distance_sparse) ** 2
        if dist_squared_err.numel() == 0:
            return (pred_coordinate.sum() * 0.0).to(pred_coordinate.dtype)
        bond_loss = torch.mean(dist_squared_err, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale
        bond_loss = bond_loss.mean(dim=-1)
        return loss_reduction(bond_loss, method=self.reduction)


def compute_lddt_mask(
    true_distance, distance_mask, is_nucleotide,
    is_nucleotide_threshold=_AF3_LDDT_NUC_RADIUS,
    is_not_nucleotide_threshold=_AF3_LDDT_NOT_NUC_RADIUS,
):
    """Bespoke-radius atom-pair mask c_lm."""
    is_nucleotide_mask = is_nucleotide.bool()
    c_lm = (true_distance < is_nucleotide_threshold) * is_nucleotide_mask[..., None] + (
        true_distance < is_not_nucleotide_threshold
    ) * (~is_nucleotide_mask[..., None])
    c_lm = c_lm * _get_off_diagonal_mask(c_lm.size(-1), c_lm.device, true_distance.dtype)
    c_lm = c_lm * distance_mask
    return c_lm


def compute_interface_mask(true_distance, lddt_mask, atom_asym_id, interface_radius=5.0):
    """Cross-chain within-radius pair mask (subset of lddt_mask). Vendored from
    Inherits lddt_mask constraints (distance/off-diagonal/bespoke radius)."""
    cross_chain = atom_asym_id[..., :, None] != atom_asym_id[..., None, :]
    within_iface = true_distance < interface_radius
    return lddt_mask * cross_chain * within_iface


def compute_interface_atom_weights(
    true_coordinate, coordinate_mask, atom_asym_id, distance_threshold=10.0,
    mode="ca", rep_atom_mask=None, atom_to_token=None, true_distance=None,
):
    """Per-atom interface indicator [..., N_atom] (1.0 interface else 0.0).

    mode='atom': atom l is interface if some different-chain, resolved atom m has
        true_dist(l, m) < distance_threshold.
    mode='ca'  : token is interface if its representative atom (rep_atom_mask, ~CA
        for protein) is within distance_threshold of another chain's rep atom;
    """
    cmask = coordinate_mask.bool()
    if mode == "ca" and rep_atom_mask is not None and atom_to_token is not None:
        rep = rep_atom_mask.bool() & cmask
        idx = torch.nonzero(rep, as_tuple=True)[0]                    # [N_rep]
        rc = true_coordinate[idx]                                    # [N_rep, 3]
        ra = atom_asym_id[idx]                                       # [N_rep]
        rt = atom_to_token[idx]                                      # [N_rep]
        d = torch.cdist(rc, rc)                                      # [N_rep, N_rep]
        cross = ra[:, None] != ra[None, :]
        within = d < distance_threshold
        iface_rep = (cross & within).any(dim=-1)                    # [N_rep]
        n_tok = int(atom_to_token.max().item()) + 1
        iface_tok = torch.zeros(n_tok, dtype=torch.bool, device=rc.device)
        iface_tok[rt] = iface_rep
        out = iface_tok[atom_to_token]                              # [N_atom]
    else:
        d = true_distance if true_distance is not None else cdist(true_coordinate, true_coordinate)
        cross = atom_asym_id[..., :, None] != atom_asym_id[..., None, :]
        within = (d < distance_threshold) & cmask[..., None, :]
        out = (cross & within).any(dim=-1)                         # [N_atom]
    return (out & cmask).to(true_coordinate.dtype)


class MSELoss(nn.Module):
    """Formula 2-4 [MSELoss] (AF3).

    AF3 weighted aligned-MSE: weight = 1 + w_dna*is_dna + w_rna*is_rna + w_lig*is_ligand,
    GT aligned to pred with weighted Kabsch, prefactor 1/3.
    """

    def __init__(self, weight_mse=_AF3_MSE_WEIGHT, weight_dna=_AF3_WEIGHT_DNA,
                 weight_rna=_AF3_WEIGHT_RNA, weight_ligand=_AF3_WEIGHT_LIGAND,
                 eps=1e-6, reduction="mean") -> None:
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction

    def weighted_rigid_align(self, pred_coordinate, true_coordinate, coordinate_mask,
                             is_dna, is_rna, is_ligand):
        N_sample = pred_coordinate.size(-3)
        weight = 1 + self.weight_dna * is_dna + self.weight_rna * is_rna + self.weight_ligand * is_ligand
        weight = weight * coordinate_mask
        true_coordinate = true_coordinate * coordinate_mask.unsqueeze(dim=-1)
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]
        true_coordinate = expand_at_dim(true_coordinate, dim=-3, n=N_sample)
        if len(weight.shape) > 1:
            weight = expand_at_dim(weight, dim=-2, n=N_sample)
        true_coordinate_aligned = weighted_rigid_align(
            x=true_coordinate.to(torch.float32),
            x_target=pred_coordinate.to(torch.float32),
            atom_weight=weight.to(torch.float32),
            stop_gradient=True,
        ).to(pred_coordinate.dtype)
        return true_coordinate_aligned.detach(), weight.detach()

    def forward(self, pred_coordinate, true_coordinate, coordinate_mask,
                is_dna, is_rna, is_ligand, per_sample_scale=None, restrict_atom_mask=None):
        # restrict_atom_mask [..., N_atom] (optional): when given, the per-atom error
        # sum is restricted to those atoms (e.g. interface atoms) and normalized by
        # their count -> a per-interface-atom mean. Alignment (Kabsch) always uses the
        # full structure so the whole complex is superposed before scoring the subset.
        with torch.no_grad():
            true_coordinate_aligned, weight = self.weighted_rigid_align(
                pred_coordinate=pred_coordinate, true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask, is_dna=is_dna, is_rna=is_rna, is_ligand=is_ligand,
            )
        per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(dim=-1)
        err_weight = weight
        denom_mask = coordinate_mask
        if restrict_atom_mask is not None:
            err_weight = err_weight * restrict_atom_mask
            denom_mask = coordinate_mask * restrict_atom_mask
        per_sample_weighted_mse = (err_weight * per_atom_se).sum(dim=-1) / (
            denom_mask.sum(dim=-1, keepdim=True) + self.eps
        )
        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale
        weighted_align_mse_loss = self.weight_mse * per_sample_weighted_mse.mean(dim=-1)
        return loss_reduction(weighted_align_mse_loss, method=self.reduction)


# ===========================================================================
# Distogram loss
# ===========================================================================
class DistogramLoss(nn.Module):
    """DistogramLoss (AF3). Bins strictly AF3."""

    def __init__(self, min_bin=_AF3_DISTOGRAM_MIN_BIN, max_bin=_AF3_DISTOGRAM_MAX_BIN,
                 no_bins=_AF3_DISTOGRAM_NO_BINS, eps=1e-6, reduction="mean") -> None:
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(self, true_coordinate, coordinate_mask, rep_atom_mask):
        boundaries = torch.linspace(self.min_bin, self.max_bin, steps=self.no_bins - 1,
                                    device=true_coordinate.device)
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]
        gt_dist = cdist(true_coordinate, true_coordinate)
        true_bins = torch.sum(gt_dist.unsqueeze(dim=-1) > boundaries, dim=-1)
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        return F.one_hot(true_bins, self.no_bins), pair_mask

    def forward(self, logits, true_coordinate, coordinate_mask, rep_atom_mask):
        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(true_coordinate, coordinate_mask, rep_atom_mask)
        errors = softmax_cross_entropy(logits=logits, labels=true_bins.to(logits.dtype))
        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = torch.sum(errors * pair_mask, dim=(-1, -2)) / denom
        return loss_reduction(loss, method=self.reduction)


# ===========================================================================
# Confidence GT-target helpers
# ===========================================================================
def calculate_atom_bespoke_lddt(
    pred_coordinate, true_coordinate, is_nucleotide, is_polymer, rep_atom_mask,
    is_nucleotide_threshold=_AF3_LDDT_NUC_RADIUS,
    is_not_nucleotide_threshold=_AF3_LDDT_NOT_NUC_RADIUS,
):
    """Per-atom bespoke LDDT (Sec 4.3.1).

    Returns (per_atom_lddt [..., N_sample, N_atom, 1], per_atom_weight [..., N_sample, N_atom, 1]).
    """
    N_atom = true_coordinate.size(-2)
    atom_m_mask = (rep_atom_mask * is_polymer).bool()
    pred_d_lm = torch.cdist(pred_coordinate, pred_coordinate[..., atom_m_mask, :])
    true_d_lm = torch.cdist(true_coordinate, true_coordinate[..., atom_m_mask, :])
    delta_d_lm = torch.abs(pred_d_lm - true_d_lm.unsqueeze(dim=-3))
    lddt_lm = (
        (delta_d_lm < 0.5).to(dtype=delta_d_lm.dtype)
        + (delta_d_lm < 1.0)
        + (delta_d_lm < 2.0)
        + (delta_d_lm < 4.0)
    ) * 0.25
    is_nucleotide = is_nucleotide[..., atom_m_mask].bool()
    locality_mask = (true_d_lm < is_nucleotide_threshold) * is_nucleotide.unsqueeze(dim=-2) + (
        true_d_lm < is_not_nucleotide_threshold
    ) * (~is_nucleotide.unsqueeze(dim=-2))
    diagonal_mask = _get_off_diagonal_mask(N_atom, true_d_lm.device, true_d_lm.dtype).bool()[..., atom_m_mask]
    pair_mask = (locality_mask * diagonal_mask).unsqueeze(dim=-3)
    per_atom_lddt = torch.sum(lddt_lm * pair_mask, dim=-1, keepdim=True)
    per_atom_weight = torch.sum(pair_mask.to(dtype=lddt_lm.dtype), dim=-1, keepdim=True)
    return per_atom_lddt, per_atom_weight


def compute_alignment_error_squared(pred_coordinate, true_coordinate, pred_frames, true_frames):
    """Algorithm 30 (squared)."""
    x_pred = expressCoordinatesInFrame(coordinate=pred_coordinate, frames=pred_frames)
    x_true = expressCoordinatesInFrame(coordinate=true_coordinate, frames=true_frames)
    return torch.sum((x_pred - x_true.unsqueeze(dim=-4)) ** 2, dim=-1)


# ===========================================================================
# Main loss orchestration.
# ===========================================================================
class _LossConfig:
    """Default AF3 weight/constant bundle used when no config is supplied.
    """

    def __init__(self) -> None:
        self.alpha_confidence = _AF3_ALPHA_CONFIDENCE
        self.alpha_pae = _AF3_ALPHA_PAE
        self.alpha_except_pae = _AF3_ALPHA_EXCEPT_PAE
        self.alpha_diffusion = _AF3_ALPHA_DIFFUSION
        self.alpha_distogram = _AF3_ALPHA_DISTOGRAM
        self.alpha_bond = _AF3_ALPHA_BOND
        self.weight_smooth_lddt = _AF3_WEIGHT_SMOOTH_LDDT
        self.weight_smooth_lddt_interface = 0.0
        self.interface_radius = 5.0
        self.interface_mse_weight = 0.0
        self.interface_mse_distance_threshold = 10.0
        self.interface_mse_mode = "ca"
        self.sigma_data = _AF3_SIGMA_DATA
        self.resolution_min = _AF3_RESOLUTION_MIN
        self.resolution_max = _AF3_RESOLUTION_MAX


class Loss(nn.Module):
    """Training loss, adapted to read the torchfold AF3 model output.

    Call as ``Loss(config)(pred=af3_output_dict, label=batch)``.

    * ``pred``  — the dict returned by ``AlphaFold3.forward`` (AF3 layout / keys).
    * ``label`` — a batch with ``input_feature_dict`` + ``label_dict``.

    Returns ``(total_loss, loss_dict)``
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__()
        if config is None:
            config = _LossConfig()
        self.config = config

        self.alpha_confidence = getattr(config, "alpha_confidence", _AF3_ALPHA_CONFIDENCE)
        self.alpha_pae = getattr(config, "alpha_pae", _AF3_ALPHA_PAE)
        self.alpha_except_pae = getattr(config, "alpha_except_pae", _AF3_ALPHA_EXCEPT_PAE)
        self.alpha_diffusion = getattr(config, "alpha_diffusion", _AF3_ALPHA_DIFFUSION)
        self.alpha_distogram = getattr(config, "alpha_distogram", _AF3_ALPHA_DISTOGRAM)
        self.alpha_bond = getattr(config, "alpha_bond", _AF3_ALPHA_BOND)
        self.weight_smooth_lddt = getattr(config, "weight_smooth_lddt", _AF3_WEIGHT_SMOOTH_LDDT)
        self.weight_smooth_lddt_interface = getattr(config, "weight_smooth_lddt_interface", 0.0)
        self.interface_radius = getattr(config, "interface_radius", 5.0)
        self.interface_mse_weight = getattr(config, "interface_mse_weight", 0.0)
        self.interface_mse_distance_threshold = getattr(config, "interface_mse_distance_threshold", 10.0)
        self.interface_mse_mode = getattr(config, "interface_mse_mode", "ca")
        self.sigma_data = getattr(config, "sigma_data", _AF3_SIGMA_DATA)
        self.resolution_min = getattr(config, "resolution_min", _AF3_RESOLUTION_MIN)
        self.resolution_max = getattr(config, "resolution_max", _AF3_RESOLUTION_MAX)
        self.diffusion_loss_chunk_size = getattr(config, "diffusion_loss_chunk_size", 4)
        self.diffusion_sparse_loss_enable = getattr(config, "diffusion_sparse_loss_enable", True)
        self.diffusion_lddt_loss_dense = getattr(config, "diffusion_lddt_loss_dense", True)

        self.lddt_radius = dict(
            is_nucleotide_threshold=_AF3_LDDT_NUC_RADIUS,
            is_not_nucleotide_threshold=_AF3_LDDT_NOT_NUC_RADIUS,
        )

        self.loss_weight = {
            "smooth_lddt_loss": self.alpha_diffusion * self.weight_smooth_lddt,
            "interface_weighted_mse_loss": self.alpha_diffusion * self.interface_mse_weight,
            "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "mse_loss": self.alpha_diffusion,
            "distogram_loss": self.alpha_distogram,
            "plddt_loss": self.alpha_confidence * self.alpha_except_pae,
            "pde_loss": self.alpha_confidence * self.alpha_except_pae,
            "resolved_loss": self.alpha_confidence * self.alpha_except_pae,
            "pae_loss": self.alpha_confidence * self.alpha_pae,
        }

        self.smooth_lddt_loss = SmoothLDDTLoss()
        self.bond_loss = BondLoss()
        self.mse_loss = MSELoss()
        self.distogram_loss = DistogramLoss()

    # ------------------------------------------------------------------ #
    # dense -> flat
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gather_atoms_xyz(dense: torch.Tensor, a2t: torch.Tensor, a2ta: torch.Tensor) -> torch.Tensor:
        """dense coords [S, N_token, max_dense, 3] -> flat [S, N_atom, 3]."""
        return dense[..., a2t, a2ta, :]

    @staticmethod
    def _gather_atoms_scalar(dense: torch.Tensor, a2t: torch.Tensor, a2ta: torch.Tensor) -> torch.Tensor:
        """dense per-atom [S, N_token, max_dense] -> flat [S, N_atom]."""
        return dense[..., a2t, a2ta]

    @staticmethod
    def _gather_atoms_logits(dense: torch.Tensor, a2t: torch.Tensor, a2ta: torch.Tensor) -> torch.Tensor:
        """dense per-atom logits [S, N_token, max_dense, n_bins] -> flat [S, N_atom, n_bins].

        a2t / a2ta index the (token, slot) axes (-3, -2) of the bin-trailing view,
        the same maps used for coords / scalars; the bin axis rides along.
        """
        return dense[..., a2t, a2ta, :]

    def _aggregate(self, loss_fns, has_valid_resolution=None):
        """Weight + sum the per-term losses."""
        cum_loss = 0.0
        metrics = {}
        for name, fn in loss_fns.items():
            weight = self.loss_weight[name]
            loss = fn()
            if (has_valid_resolution is not None) and (has_valid_resolution.sum() == 0) and (
                name in ("plddt_loss", "pde_loss", "resolved_loss", "pae_loss")
            ):
                loss = 0.0 * loss
            metrics[name] = loss.detach().clone()
            metrics[f"weighted_{name}"] = (weight * loss).detach().clone()
            cum_loss = cum_loss + weight * loss
        if not isinstance(cum_loss, torch.Tensor):
            cum_loss = torch.zeros((), device=metrics_device(metrics))
        metrics["loss"] = cum_loss.detach().clone()
        return cum_loss, metrics

    def forward(self, pred: dict, label: dict, mode: str = "train"):
        feat = label["input_feature_dict"] if "input_feature_dict" in label else label
        label_dict = label["label_dict"] if "label_dict" in label else label

        device = pred["distogram"]["logits"].device
        f32 = torch.float32

        a2t = feat["atom_to_token_idx"].to(torch.int64).to(device)
        a2ta = feat["atom_to_tokatom_idx"].to(torch.int64).to(device)

        # ---- label (flat) ----
        true_coord = label_dict["coordinate"].to(f32).to(device)          # [N_atom, 3]
        coord_mask = label_dict["coordinate_mask"].to(f32).to(device)     # [N_atom]
        N_atom = true_coord.shape[0]

        is_rna = feat["is_rna"].to(f32).to(device)
        is_dna = feat["is_dna"].to(f32).to(device)
        is_ligand = feat["is_ligand"].to(f32).to(device)
        is_nucleotide = (is_rna.bool() | is_dna.bool()).to(f32)
        is_polymer = (1 - is_ligand)
        bond_mask = feat["bond_mask"].to(f32).to(device)                  # [N_atom, N_atom]
        distogram_rep = feat["distogram_rep_atom_mask"].to(device)        # [N_atom]
        pae_rep = feat["pae_rep_atom_mask"].to(device)                    # [N_atom]
        plddt_rep = feat["plddt_m_rep_atom_mask"].to(device)             # [N_atom]
        has_frame = feat["has_frame"].to(device)                          # [N_token]
        frame_atom_index = feat["frame_atom_index"].to(torch.int64).to(device)  # [N_token, 3]
        resolution = float(feat["resolution"].reshape(-1)[0].item())

        # ---- pred (dense, AF3) -> flat ----
        confidence_only = bool(pred.get("confidence_only", False))
        diff = pred.get("diffusion_samples")
        if diff is not None:
            dense_pos = diff["atom_positions"].to(f32).to(device)         # [S, N_token, max_dense, 3]
            pre_perm = diff.get("pred_coord_flat", None) if isinstance(diff, dict) else None
            if pre_perm is not None:
                pred_coord = pre_perm.to(f32).to(device)                  # [S, N_atom, 3]
            else:
                pred_coord = self._gather_atoms_xyz(dense_pos, a2t, a2ta)  # [S, N_atom, 3]
            N_sample = pred_coord.shape[0]
        else:
            dense_pos, pred_coord, N_sample = None, None, 1

        conf_dense = pred.get("confidence_atom_positions", None)
        if conf_dense is not None:
            conf_coord = self._gather_atoms_xyz(
                conf_dense.to(f32).to(device), a2t, a2ta
            )                                                              # [S_conf, N_atom, 3]
        else:
            conf_coord = pred_coord

        # ---- diffusion-MSE per-sample noise weighting (TRAINING path only) ----
        # AF3 single-step diffusion training applies the EDM loss weight
        #   w(t) = (t^2 + sigma_data^2) / (t * sigma_data)^2
        # per noised sample (t = the sampled noise level). The training forward
        # exposes `noise_levels` [S] inside diffusion_samples; inference does not.
        # When present, scale the bond + weighted-MSE terms by w(t) (one scalar
        # per diffusion sample); otherwise keep the eval behaviour (no scale).
        per_sample_scale = None
        noise_levels = diff.get("noise_levels", None) if isinstance(diff, dict) else None
        if noise_levels is not None:
            t = noise_levels.to(f32).to(device).reshape(-1)               # [S]
            sd = _AF3_SIGMA_DATA
            per_sample_scale = (t ** 2 + sd ** 2) / ((t * sd) ** 2 + 1e-12)  # [S]
            if _NOISE_WEIGHT_MAX and _NOISE_WEIGHT_MAX > 0:
                # cap the EDM weight
                per_sample_scale = per_sample_scale.clamp(max=_NOISE_WEIGHT_MAX)

        loss_fns: dict[str, Callable] = {}

        if not confidence_only:
            # ---- distance / lddt-mask labels ----
            with torch.no_grad():
                distance_mask = coord_mask[..., None] * coord_mask[..., None, :]     # [N_atom, N_atom]
                true_distance = (cdist(true_coord, true_coord) * distance_mask)
                lddt_mask = compute_lddt_mask(
                    true_distance=true_distance, distance_mask=distance_mask,
                    is_nucleotide=is_nucleotide, **self.lddt_radius,
                )
            cs = self.diffusion_loss_chunk_size

            # ---- Diffusion: smooth-LDDT (dense-chunked OR sparse), bond, weighted-MSE ----
            smooth_pair_weight = None
            if self.weight_smooth_lddt_interface > 0:
                with torch.no_grad():
                    atom_asym_id = feat["asym_id"].to(torch.int64).to(device)[a2t]
                    interface_mask = compute_interface_mask(
                        true_distance=true_distance, lddt_mask=lddt_mask,
                        atom_asym_id=atom_asym_id, interface_radius=self.interface_radius,
                    )
                    # valid pairs -> 1.0; interface pairs -> w_iface; masked-out -> 0.
                    smooth_pair_weight = lddt_mask.to(true_distance.dtype) + (
                        self.weight_smooth_lddt_interface - 1.0
                    ) * interface_mask.to(true_distance.dtype)

            if self.diffusion_lddt_loss_dense or not self.diffusion_sparse_loss_enable:
                loss_fns["smooth_lddt_loss"] = lambda: self.smooth_lddt_loss.dense_forward(
                    pred_coordinate=pred_coord, true_coordinate=true_coord,
                    lddt_mask=lddt_mask, true_distance=true_distance,
                    pair_weight=smooth_pair_weight, diffusion_chunk_size=cs,
                )
            else:
                loss_fns["smooth_lddt_loss"] = lambda: self.smooth_lddt_loss.sparse_forward(
                    pred_coordinate=pred_coord, true_coordinate=true_coord,
                    lddt_mask=lddt_mask, pair_weight=smooth_pair_weight, diffusion_chunk_size=cs,
                )
            # Interface-weighted MSE (configurable CA-CA residue-level or atom-atom).
            # Separate additive term: the weighted aligned-MSE restricted to interface
            # atoms, so interface atoms get (1 + interface_mse_weight)x total weight in
            # the diffusion coordinate loss. Registered only when its weight is nonzero.
            if self.interface_mse_weight > 0:
                with torch.no_grad():
                    atom_asym_id_mse = feat["asym_id"].to(torch.int64).to(device)[a2t]
                    iface_atom_w = compute_interface_atom_weights(
                        true_coordinate=true_coord, coordinate_mask=coord_mask,
                        atom_asym_id=atom_asym_id_mse,
                        distance_threshold=self.interface_mse_distance_threshold,
                        mode=self.interface_mse_mode,
                        rep_atom_mask=distogram_rep, atom_to_token=a2t,
                        true_distance=true_distance,
                    )
                loss_fns["interface_weighted_mse_loss"] = lambda: self.mse_loss(
                    pred_coordinate=pred_coord, true_coordinate=true_coord, coordinate_mask=coord_mask,
                    is_rna=is_rna, is_dna=is_dna, is_ligand=is_ligand,
                    per_sample_scale=per_sample_scale, restrict_atom_mask=iface_atom_w,
                )
            if self.diffusion_sparse_loss_enable:
                loss_fns["bond_loss"] = lambda: self.bond_loss.sparse_forward(
                    pred_coordinate=pred_coord, true_coordinate=true_coord,
                    distance_mask=distance_mask, bond_mask=bond_mask,
                    per_sample_scale=per_sample_scale,   # training noise weight w(t); None at eval
                )
            else:
                loss_fns["bond_loss"] = lambda: self.bond_loss(
                    pred_coordinate=pred_coord, true_distance=true_distance,
                    distance_mask=distance_mask, bond_mask=bond_mask,
                    per_sample_scale=per_sample_scale,
                    diffusion_chunk_size=cs,
                )
            loss_fns["mse_loss"] = lambda: self.mse_loss(
                pred_coordinate=pred_coord, true_coordinate=true_coord, coordinate_mask=coord_mask,
                is_rna=is_rna, is_dna=is_dna, is_ligand=is_ligand, per_sample_scale=per_sample_scale,
            )

            # ---- Distogram CE (AF3 logits) ----
            loss_fns["distogram_loss"] = lambda: self.distogram_loss(
                logits=pred["distogram"]["logits"].to(f32),
                true_coordinate=true_coord, coordinate_mask=coord_mask, rep_atom_mask=distogram_rep,
            )

        # ---- Confidence (strict AF3 cross-entropy over the exposed per-bin logits) ----
        has_valid_resolution = (resolution >= self.resolution_min) and (resolution <= self.resolution_max)
        hvr = torch.tensor([1.0 if has_valid_resolution else 0.0], dtype=f32, device=device)

        # AF3 confidence-head bin edges (identical to ConfidenceHead.distance_breaks /
        # pae_breaks). 64 error bins over <=31 Å; 50 pLDDT bins over [0, 1].
        pde_breaks = _af3_error_breaks(_AF3_PDE_MAX_BIN, _AF3_CONF_NO_BINS, device, f32)   # [63]
        pae_breaks = _af3_error_breaks(_AF3_PAE_MAX_BIN, _AF3_CONF_NO_BINS, device, f32)   # [63]

        conf_available = all(
            k in pred for k in (
                "plddt_logits", "pae_logits", "pde_logits", "experimentally_resolved_logits"
            )
        )
        if conf_available:
            cm = coord_mask.bool()

            def _plddt_loss():
                # plddt_logits dense [S, N_token, 24, 50] -> flat [S, N_atom, 50].
                logits = self._gather_atoms_logits(
                    pred["plddt_logits"].to(f32).to(device), a2t, a2ta
                )                                                                # [S, N_atom, 50]
                logits = logits[..., cm, :]                                      # atoms with coords
                with torch.no_grad():
                    # GT per-atom bespoke LDDT.
                    per_atom_lddt, per_atom_weight = calculate_atom_bespoke_lddt(
                        pred_coordinate=conf_coord[..., cm, :].detach(),
                        true_coordinate=true_coord[..., cm, :],
                        is_nucleotide=is_nucleotide[cm],
                        is_polymer=is_polymer[cm],
                        rep_atom_mask=plddt_rep[cm].to(f32),
                        **self.lddt_radius,
                    )
                    gt_lddt = (per_atom_lddt / (per_atom_weight + 1e-6)).squeeze(-1)  # [S, N_atom_cm] in [0,1]
                    # 50 bins over [0, 1]: bin = clamp(floor(lddt * 50), 0, 49).
                    bin_idx = torch.clamp(
                        (gt_lddt * _AF3_PLDDT_NO_BINS).floor().long(),
                        min=0, max=_AF3_PLDDT_NO_BINS - 1,
                    )
                    labels = F.one_hot(bin_idx, _AF3_PLDDT_NO_BINS).to(f32)      # [S, N_atom_cm, 50]
                ce = softmax_cross_entropy(logits=logits, labels=labels)         # [S, N_atom_cm]
                return ce.mean()

            def _pde_loss():
                # pde_logits (token layout) dense [S, N_token, N_token, 64].
                logits = pred["pde_logits"].to(f32).to(device)                   # [S, N_token, N_token, 64]
                rep = distogram_rep.bool()                                       # one atom per token
                tok_true = true_coord[rep]                                       # [N_token, 3]
                tok_pred = conf_coord[..., rep, :]                               # [S_conf, N_token, 3]
                with torch.no_grad():
                    gt_dist = cdist(tok_true, tok_true)                          # [N_token, N_token]
                    pred_dist = cdist(tok_pred, tok_pred)                        # [S, N_token, N_token]
                    gt_de = torch.abs(pred_dist - gt_dist.unsqueeze(0))          # [S, N_token, N_token] (Å)
                    bin_idx = _bin_indices_from_breaks(gt_de, pde_breaks)        # [S, N_token, N_token] in [0,63]
                    labels = F.one_hot(bin_idx, _AF3_CONF_NO_BINS).to(f32)
                    tok_mask = coord_mask[rep]                                   # [N_token]
                    pair_mask = (tok_mask[..., None] * tok_mask[..., None, :])   # [N_token, N_token]
                ce = softmax_cross_entropy(logits=logits, labels=labels)         # [S_conf, N_token, N_token]
                denom = pair_mask.sum() * conf_coord.shape[0] + 1e-6
                return (ce * pair_mask).sum() / denom

            def _pae_loss():
                # pae_logits dense [S, N_token, N_token, 64]; rows restricted to frames.
                logits = pred["pae_logits"].to(f32).to(device)                   # [S, N_token, N_token, 64]
                hf = has_frame.bool()
                rep = pae_rep.bool()
                with torch.no_grad():
                    fai = frame_atom_index[hf, :]                                # [N_frame, 3]
                    pred_frames = gather_frame_atom_by_indices(conf_coord, fai, dim=-2)   # [S_conf,N_frame,3,3]
                    true_frames = gather_frame_atom_by_indices(true_coord, fai, dim=-2)   # [N_frame,3,3]
                    sq_pae = compute_alignment_error_squared(
                        pred_coordinate=conf_coord[..., rep, :],
                        true_coordinate=true_coord[..., rep, :],
                        pred_frames=pred_frames, true_frames=true_frames,
                    )                                                            # [S, N_frame, N_token]
                    gt_pae = torch.sqrt(sq_pae + 1e-8)                            # Å
                    fr_cmask = gather_frame_atom_by_indices(coord_mask, fai, dim=-1).sum(-1) >= 3  # [N_frame]
                    tok_mask = coord_mask[rep]                                    # [N_token]
                    ft_mask = (fr_cmask[..., None] * tok_mask[..., None, :])      # [N_frame, N_token]
                    bin_idx = _bin_indices_from_breaks(gt_pae, pae_breaks)        # [S, N_frame, N_token] in [0,63]
                    labels = F.one_hot(bin_idx, _AF3_CONF_NO_BINS).to(f32)
                logits_fr = logits[..., hf, :, :]                                # [S, N_frame, N_token, 64]
                ce = softmax_cross_entropy(logits=logits_fr, labels=labels)      # [S_conf, N_frame, N_token]
                denom = ft_mask.sum() * conf_coord.shape[0] + 1e-6
                return (ce * ft_mask).sum() / denom

            def _resolved_loss():
                # experimentally_resolved_logits dense [S, N_token, 24, 2] -> flat [S, N_atom, 2].
                logits = self._gather_atoms_logits(
                    pred["experimentally_resolved_logits"].to(f32).to(device), a2t, a2ta
                )                                                                # [S, N_atom, 2]
                with torch.no_grad():
                    labels = F.one_hot(coord_mask.long(), 2).to(f32)            # [N_atom, 2]
                    labels = labels.unsqueeze(0).expand(conf_coord.shape[0], -1, -1)  # [S_conf, N_atom, 2]
                    atom_mask = coord_mask                                        # [N_atom]
                ce = softmax_cross_entropy(logits=logits, labels=labels)         # [S, N_atom]
                denom = atom_mask.sum() * conf_coord.shape[0] + 1e-6
                return (ce * atom_mask).sum() / denom

            loss_fns["plddt_loss"] = _plddt_loss
            loss_fns["pde_loss"] = _pde_loss
            loss_fns["resolved_loss"] = _resolved_loss
            loss_fns["pae_loss"] = _pae_loss

        cum_loss, metrics = self._aggregate(loss_fns, has_valid_resolution=hvr)
        return cum_loss, metrics


def metrics_device(metrics: dict) -> torch.device:
    for v in metrics.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")


