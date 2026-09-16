#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F

_Dist_Diff_Chunk_Atom_Thre = 13000  # N_atoms=13000, about 541aa


def loss_reduction(loss, method="mean"):
    if method == "mean":
        return loss.mean()
    elif method == "sum":
        return loss.sum()
    elif method == "max":
        return loss.max()
    elif method == "min":
        return loss.min()
    else:
        raise ValueError(f"Unknown reduction method: {method}")


# class WeightedRigidAlign(nn.Module):


#     def __init__(self):
#         super().__init__()

#     def forward(
#         self,
#         pred_coords: torch.Tensor,
#         true_coords: torch.Tensor,
#         weights: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Args:
#             pred_coords: [..., N_atoms, 3]
#             true_coords: [N_atoms, 3]
#             weights: [N_atoms]
#         Returns:
#             aligned_coords: [..., N_atoms, 3]
#         """
#         # Compute weighted centroids
#         weights_sum = weights.sum() + 1e-8
#         pred_center = (pred_coords * weights.unsqueeze(-1)).sum(dim=-2, keepdim=True) / weights_sum
#         true_center = (true_coords * weights.unsqueeze(-1)).sum(dim=-2, keepdim=True) / weights_sum

#         # Center both coordinate sets
#         pred_centered = pred_coords - pred_center
#         true_centered = true_coords - true_center
#         # Solve the optimal rotation via the Kabsch algorithm
#         H = torch.einsum('...ni,nj->...ij', pred_centered * weights.unsqueeze(-1), true_centered)
#         U, S, Vh = torch.linalg.svd(H)
#         R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))

#         # Ensure we keep a proper rotation (determinant = 1)
#         det = torch.det(R)
#         Vh_corrected = Vh.clone()
#         Vh_corrected[..., -1, :] *= det.unsqueeze(-1)
#         R = torch.matmul(Vh_corrected.transpose(-2, -1), U.transpose(-2, -1))

#         # Apply rotation and translation
#         aligned_coords = torch.matmul(pred_centered, R) + true_center

#         return aligned_coords.detach()
# class WeightedRigidAlign(nn.Module):


#     def __init__(self):
#         super().__init__()

#     def forward(
#         self,
#         pred_coords: torch.Tensor,
#         true_coords: torch.Tensor,
#         weights: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Args:
#             pred_coords: [..., N_atoms, 3]
#             true_coords: [N_atoms, 3]
#             weights: [N_atoms]
#         Returns:
#             aligned_coords: [..., N_atoms, 3]
#         """
#         # Key fix: disable AMP to prevent automatic conversion back to BFloat16
#         with torch.npu.amp.autocast(enabled=False):
#             # Preserve original dtype
#             orig_dtype = pred_coords.dtype

#             # Convert everything to float32
#             pred_coords = pred_coords.float()
#             true_coords = true_coords.float()
#             weights = weights.float()

#             # Compute weighted centroids
#             weights_sum = weights.sum() + 1e-8
#             pred_center = (pred_coords * weights.unsqueeze(-1)).sum(dim=-2, keepdim=True) / weights_sum
#             true_center = (true_coords * weights.unsqueeze(-1)).sum(dim=-2, keepdim=True) / weights_sum

#             # Center
#             pred_centered = pred_coords - pred_center
#             true_centered = true_coords - true_center

#             # Compute rotation matrix (Kabsch algorithm)
#             H = torch.einsum('...ni,nj->...ij', pred_centered * weights.unsqueeze(-1), true_centered)

#             # Ensure H is float32 again
#             H = H.float()

#             U, S, Vh = torch.linalg.svd(H)
#             R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))

#             # Ensure proper rotation (determinant = 1)
#             # det shape is [...]
#             det = torch.det(R)
#             # Use torch.sign to determine sign: -1 if det < 0
#             sign = torch.sign(det)  # [...]
#             # Expand dimensions for broadcasting: [...] -> [..., 1], so it can broadcast to [..., 3]
#             sign = sign.unsqueeze(-1)  # [..., 1]

#             Vh_corrected = Vh.clone()
#             # When det < 0, sign = -1, so flip the last column
#             # Vh_corrected[..., :, -1] is [..., 3] shape, sign is [..., 1], can broadcast
#             Vh_corrected[..., :, -1] *= sign
#             R = torch.matmul(Vh_corrected.transpose(-2, -1), U.transpose(-2, -1))

#             # Apply rotation and translation (apply transpose of R)
#             aligned_coords = torch.matmul(pred_centered, R.transpose(-2, -1)) + true_center

#             # Convert back to original dtype
#             return aligned_coords.to(orig_dtype).detach()
class WeightedRigidAlign(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            pred_coords: torch.Tensor,
            true_coords: torch.Tensor,
            weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_coords: [..., N_atoms, 3]
            true_coords: [N_atoms, 3]
            weights: [N_atoms]
        Returns:
            aligned_coords: [..., N_atoms, 3]
        """
        # Key fix: disable AMP to prevent automatic conversion back to BFloat16
        with torch.amp.autocast('npu', enabled=False):
            orig_dtype = pred_coords.dtype

            # Convert everything to float32 (SVD is more stable)
            pred_coords = pred_coords.float()
            true_coords = true_coords.float()
            weights = weights.float()

            # Support batched alignment: true_coords can be [N,3] or [...,N,3]
            if true_coords.ndim == pred_coords.ndim - 1:
                true_coords = true_coords.unsqueeze(0).expand(
                    pred_coords.shape[:-2] + true_coords.shape[-2:]
                )
            elif true_coords.ndim != pred_coords.ndim:
                raise ValueError(
                    f"true_coords.ndim ({true_coords.ndim}) does not match pred_coords.ndim ({pred_coords.ndim})"
                )

            # Compute weighted centroids
            weights_sum = weights.sum() + 1e-8
            weight_shape = (1,) * (pred_coords.ndim - 2) + (weights.shape[0], 1)
            weights_b = weights.view(*weight_shape)
            pred_center = (pred_coords * weights_b).sum(dim=-2, keepdim=True) / weights_sum
            true_center = (true_coords * weights_b).sum(dim=-2, keepdim=True) / weights_sum

            # Center
            pred_centered = pred_coords - pred_center
            true_centered = true_coords - true_center  # [N, 3]

            # true_centered * weights[:, None] : [N, 3]
            # pred_centered                 : [..., N, 3]
            H = torch.einsum(
                "...ni,...nj->...ij",
                true_centered * weights_b,
                pred_centered,
            ).float()  # [..., 3, 3]

            # SVD: H = U @ diag(S) @ Vh  (Vh = V^T)
            U, S, Vh = torch.linalg.svd(H.cpu())  # float32 torch.svd
            U, S, Vh = U.npu(), S.npu(), Vh.npu()
            # U, S, Vh = torch.svd(H) #0123

            # R = U V ; in torch corresponds to R = U @ V^T = U @ Vh
            R = U @ Vh  # [..., 3, 3]

            # === Remove reflection (batched): if det(R)<0 then R = U F V ===
            # detR = torch.det(R)  # [...]
            # detR = torch.det(R.to(torch.float)).to(U.dtype)
            detR = torch.det(R.cpu()).to(R.device)  # [...]
            f = torch.where(detR < 0, -1.0, 1.0).to(dtype=R.dtype)  # [...]

            # F = diag(1,1,f)
            lead = (1,) * (R.ndim - 2)
            Fmat = torch.eye(3, device=R.device, dtype=R.dtype).view(*lead, 3, 3).expand_as(R).clone()
            Fmat[..., 2, 2] = f

            R = U @ Fmat @ Vh  # [..., 3, 3]

            # Apply alignment
            # pred_centered is stored as row vectors [..., N, 3], so use R^T
            aligned_coords = pred_centered @ R.transpose(-2, -1) + true_center

            # Detach: rigid alignment is only used to find the optimal transformation; the transformation itself should not participate in gradient computation
            return aligned_coords.to(orig_dtype).detach()


class SmoothLDDTLoss(nn.Module):

    def __init__(self, eps=1e-10, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.thresholds = [0.5, 1, 2, 4]

    def forward(
            self,
            dist_diff: torch.Tensor,
            distance_mask: torch.Tensor,
            lddt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dist_diff: [..., N_sample, N_atom, N_atom]
            distance_mask: [N_atom, N_atom]
            lddt_mask: [N_atom, N_atom]
        Returns:
            loss: scalar
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()

        # Smooth LDDT term
        dist_diff_epsilon = 0

        def compute_sigmoid(thre):
            return torch.sigmoid(thre - torch.abs(dist_diff))

        if dist_diff.size(-1) > _Dist_Diff_Chunk_Atom_Thre:
            lddt_eps = torch.sum(c_lm, dim=(-1, -2)) + self.eps
            for thre in self.thresholds:
                dist_diff_epsilon += torch.sum(
                    c_lm * torch.utils.checkpoint.checkpoint(compute_sigmoid, thre, use_reentrant=True), dim=(-1, -2))
            lddt = dist_diff_epsilon * (0.25 / lddt_eps)
        else:
            c_lm = c_lm.unsqueeze(dim=-3)
            lddt_eps = torch.sum(c_lm, dim=(-1, -2)) + self.eps
            for thre in self.thresholds:
                dist_diff_epsilon += torch.utils.checkpoint.checkpoint(compute_sigmoid, thre, use_reentrant=True)

            dist_diff_epsilon *= 0.25
            # Average within the mask
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / lddt_eps

        # Loss is 1 - lddt
        loss = 1.0 - lddt

        return loss_reduction(loss, method=self.reduction)


class BondLoss(nn.Module):
    """Sparse bond-pair loss: O(N_bonds) memory instead of O(N_atoms^2)."""

    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
            self,
            pred_bond_dists: torch.Tensor,
            true_bond_dists: torch.Tensor,
            valid_bonds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_bond_dists: [N_sample, N_bonds]
            true_bond_dists: [N_sample, N_bonds] or [1, N_bonds]
            valid_bonds: [N_bonds] bool, both endpoints have valid coords
        Returns:
            loss: scalar (after reduction over N_sample)
        """
        mask = valid_bonds.unsqueeze(0).float()  # [1, N_bonds]
        squared_err = (pred_bond_dists - true_bond_dists) ** 2  # [N_sample, N_bonds]
        bond_loss = torch.sum(squared_err * mask, dim=-1) / (torch.sum(mask) + self.eps)  # [N_sample]
        return loss_reduction(bond_loss, method=self.reduction)


class DistogramLoss(nn.Module):
    def __init__(
            self,
            min_bin: float = 2.3125,
            max_bin: float = 21.6875,
            no_bins: int = 64,
            eps: float = 1e-6,
            reduction: str = "mean",
    ) -> None:
        super(DistogramLoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
            self,
            true_coordinate: torch.Tensor,
            coordinate_mask: torch.Tensor,
            rep_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate one-hot distogram labels and valid pair mask."""
        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins - 1,
            device=true_coordinate.device,
        )

        rep_atom_mask = rep_atom_mask.to(device=true_coordinate.device, dtype=torch.bool)
        true_coordinate = true_coordinate[..., rep_atom_mask, :]
        gt_dist = torch.cdist(true_coordinate, true_coordinate)
        true_bins = torch.sum(gt_dist.unsqueeze(dim=-1) > boundaries, dim=-1)

        token_mask = coordinate_mask[..., rep_atom_mask].to(dtype=torch.bool)
        pair_mask = token_mask[..., :, None] & token_mask[..., None, :]

        return F.one_hot(true_bins, self.no_bins), pair_mask

    def forward(
            self,
            logits: torch.Tensor,
            true_coordinate: torch.Tensor,
            coordinate_mask: torch.Tensor,
            rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        true_bins = true_bins.to(dtype=logits.dtype)
        errors = -torch.sum(true_bins * F.log_softmax(logits, dim=-1), dim=-1)

        pair_mask = pair_mask.to(dtype=errors.dtype)
        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = torch.sum(errors * pair_mask, dim=(-1, -2)) / denom

        return loss_reduction(loss, method=self.reduction)


def compute_noise_weights(noise_levels: torch.Tensor, sigma_data: float = 16.0,
                          max_weight: float = 10.0) -> torch.Tensor:
    """
    Compute per-sample noise weighting for diffusion loss.

    Uses the stable formula: w(σ) = (σ² + σ_data²) / (σ_data + σ)²
    This avoids the weight explosion issue of the original EDM formula
    when σ → 0.

    Additionally clips weights to max_weight for extra safety.

    Args:
        noise_levels: [N_sample] noise level for each sample
        sigma_data: data standard deviation (default 16.0)
        max_weight: maximum allowed weight (default 10.0)

    Returns:
        noise_weights: [N_sample] per-sample weights
    """
    # Stable formula: uses (σ_data + σ)² instead of (σ_data * σ)²
    # When σ → 0: weight → σ_data² / σ_data² = 1.0 (stable!)
    # When σ → ∞: weight → 1.0
    noise_weights = (noise_levels ** 2 + sigma_data ** 2) / (
            (sigma_data * noise_levels) ** 2 + 1e-8
    )
    # Clip to max_weight for additional safety
    noise_weights = torch.clamp(noise_weights, max=max_weight)
    return noise_weights


class MSELoss(nn.Module):

    def __init__(
            self,
            weight_mse: float = 1.0 / 3.0,
            weight_dna: float = 5.0,
            weight_rna: float = 5.0,
            weight_ligand: float = 10.0,
            eps: float = 1e-8,
            reduction: str = "mean",
            max_noise_weight: float = 10.0,
    ):
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction
        self.max_noise_weight = max_noise_weight
        self.rigid_align = WeightedRigidAlign()

    def forward(
            self,
            batch,
            pred_coords: torch.Tensor,
            true_coords: torch.Tensor,
            coord_mask: torch.Tensor,
            is_dna: torch.Tensor,
            is_rna: torch.Tensor,
            is_ligand: torch.Tensor,
            sample_id: str = "unknown",
            noise_levels: torch.Tensor = None,
            interface_weights: torch.Tensor = None,
            interface_atom_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_coords: [N_sample, N_atoms, 3]
            true_coords: Either [N_atoms, 3] or [N_sample, N_atoms, 3] (augmented GT during training)
            coord_mask: [N_atoms]
            is_dna: [N_atoms]
            is_rna: [N_atoms]
            is_ligand: [N_atoms]
            noise_levels: [N_sample] - noise level for each diffusion sample (optional)
                         If provided, applies per-sample noise weighting according to formula (6):
                         w(t) = (t² + σ_data²) / (t + σ_data)², σ_data = 16.0
            interface_weights: [N_atoms] - per-atom multiplier for interface residues (optional)
                         e.g. 5.0 for interface atoms, 1.0 for non-interface atoms
            interface_atom_mask: [N_atoms] - boolean mask, True for interface atoms (optional)
                         Used to compute separate interface/non-interface MSE for monitoring.
        Returns:
            (loss, per_sample_mse, interface_metrics):
                loss: scalar (weighted average over N_sample if noise_levels provided)
                per_sample_mse: [N_sample] detached per-sample losses
                interface_metrics: dict or None, containing interface/non-interface MSE
        """
        valid_mask = coord_mask.bool()

        # Build per-atom weights
        weights = torch.ones_like(coord_mask, dtype=torch.float32)
        weights = weights + is_dna.float() * self.weight_dna
        weights = weights + is_rna.float() * self.weight_rna
        weights = weights + is_ligand.float() * self.weight_ligand
        weights = weights * coord_mask.float()

        # Apply interface loss weighting (multiplicative)
        if interface_weights is not None:
            weights = weights * interface_weights

        # Explicitly filter to valid atoms
        weights_valid = weights[valid_mask]
        coord_mask_valid = coord_mask[valid_mask].float()
        pred_coords_valid = pred_coords[..., valid_mask, :]
        true_coords_valid = true_coords[..., valid_mask, :]
        # Handle the dimensionality of true_coords
        if true_coords.dim() == 3:
            # Training mode: [N_sample, N_atoms, 3] with individual augmentations
            # Process each sample separately
            N_sample = pred_coords.shape[0]
            # That is, align GT to pred, then compute pred - aligned_GT
            true_coords_aligned = self.rigid_align(
                true_coords_valid,  # [N_sample, N_valid, 3] - GT (source)
                pred_coords_valid,  # [N_sample, N_valid, 3] - pred (target coordinate frame)
                weights_valid
            )
            # Compute MSE: pred vs aligned GT

            diff = pred_coords_valid - true_coords_aligned  # [N_sample, N_valid, 3]
            mse_raw = (diff ** 2).sum(dim=-1)  # [N_sample, N_valid]

            # Compute interface-specific metrics (before applying weights)
            interface_metrics = None
            if interface_atom_mask is not None:
                interface_valid = interface_atom_mask[valid_mask]  # [N_valid]
                non_interface_valid = ~interface_valid
                interface_metrics = {}
                if interface_valid.any():
                    # Unweighted mean per-atom MSE on interface atoms, averaged over samples
                    interface_metrics['interface_mse'] = mse_raw[:, interface_valid].mean().detach()
                    interface_metrics['n_interface_atoms'] = int(interface_valid.sum().item())
                if non_interface_valid.any():
                    interface_metrics['non_interface_mse'] = mse_raw[:, non_interface_valid].mean().detach()
                interface_metrics['interface_atom_ratio'] = float(interface_valid.sum().item()) / max(
                    interface_valid.shape[0], 1)

            # Apply weights and mask
            mse = mse_raw * weights_valid.unsqueeze(0)

            # Normalize by coordinate mask sum
            per_sample_losses = mse.sum(dim=-1) / (coord_mask_valid.sum() + self.eps)  # [N_sample]

            # torchfold: per-sample noise weighting
            if noise_levels is not None:
                noise_weights = compute_noise_weights(
                    noise_levels, sigma_data=16.0, max_weight=self.max_noise_weight
                )
                # Weighted average
                loss = (per_sample_losses * noise_weights).mean()
            else:
                loss = per_sample_losses.mean()

            # Return total loss and per-sample losses (for t-interval statistics)
            return self.weight_mse * loss, per_sample_losses.detach(), interface_metrics
        else:

            # Inference mode: pred [N_sample, N_atoms, 3], GT [N_atoms, 3]
            # align GT onto each pred sample
            N_sample = pred_coords_valid.shape[0]
            true_coords_b = true_coords_valid.unsqueeze(0).expand(N_sample, -1, -1)
            true_coords_aligned = self.rigid_align(
                true_coords_b,  # [N_sample, N_valid, 3] - GT (source)
                pred_coords_valid,  # [N_sample, N_valid, 3] - pred (target)
                weights_valid
            )
            diff = pred_coords_valid - true_coords_aligned  # [N_sample, N_valid, 3]
            mse_raw = (diff ** 2).sum(dim=-1)  # [N_sample, N_valid]

            # Compute interface-specific metrics (before applying weights)
            interface_metrics = None
            if interface_atom_mask is not None:
                interface_valid = interface_atom_mask[valid_mask]  # [N_valid]
                non_interface_valid = ~interface_valid
                interface_metrics = {}
                if interface_valid.any():
                    interface_metrics['interface_mse'] = mse_raw[:, interface_valid].mean().detach()
                    interface_metrics['n_interface_atoms'] = int(interface_valid.sum().item())
                if non_interface_valid.any():
                    interface_metrics['non_interface_mse'] = mse_raw[:, non_interface_valid].mean().detach()
                interface_metrics['interface_atom_ratio'] = float(interface_valid.sum().item()) / max(
                    interface_valid.shape[0], 1)

            # Apply weights and mask
            mse = mse_raw * weights_valid.unsqueeze(0)

            # Normalize per sample
            per_sample_mse = mse.sum(dim=-1) / (coord_mask_valid.sum() + self.eps)  # [N_sample]

            # torchfold: per-sample noise weighting
            if noise_levels is not None:
                noise_weights = compute_noise_weights(
                    noise_levels, sigma_data=16.0, max_weight=self.max_noise_weight
                )
                # Weighted average
                loss = (per_sample_mse * noise_weights).mean()
            else:
                loss = per_sample_mse.mean()  # Average over N_sample
            # Return total loss and per-sample losses (for t-interval statistics)
            return self.weight_mse * loss, per_sample_mse.detach(), interface_metrics


# Export all public symbols
__all__ = [
    'WeightedRigidAlign',
    'SmoothLDDTLoss',
    'BondLoss',
    'DistogramLoss',
    'MSELoss',
    'loss_reduction',
    'compute_noise_weights',
]
