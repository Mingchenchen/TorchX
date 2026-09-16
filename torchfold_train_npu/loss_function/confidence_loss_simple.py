#!/usr/bin/env python3
# Simplified confidence loss – consumes prediction scores directly (no logits).
# Uses MSE instead of cross-entropy.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceLossSimple(nn.Module):
    """
    Minimal confidence loss that compares predicted scores with targets via MSE.
    Designed for models that emit scores directly (no logits).
    """
    
    def __init__(
        self,
        plddt_weight: float = 1.0,
        pae_weight: float = 1.0,
        pde_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.plddt_weight = plddt_weight
        self.pae_weight = pae_weight
        self.pde_weight = pde_weight
        self.eps = eps
    
    def forward(
        self,
        # Prediction scores (not logits)
        predicted_lddt: Optional[torch.Tensor],  # [N_sample, N_atoms] pLDDT scores
        predicted_pae: Optional[torch.Tensor],   # [N_sample, N_tokens, N_tokens] PAE distances
        predicted_pde: Optional[torch.Tensor],   # [N_sample, N_tokens, N_tokens] PDE distances
        predicted_resolved: Optional[torch.Tensor] = None,  # [N_sample, N_tokens, atoms_per_token] resolved scores
        # Ground truth coordinates
        pred_coords: torch.Tensor = None,        # [N_sample, N_atoms, 3]
        true_coords: torch.Tensor = None,        # [N_atoms, 3]
        coordinate_mask: torch.Tensor = None,    # [N_atoms]
    ) -> dict:
        """
        Compute confidence-related losses.
        
        Args:
            predicted_lddt: Predicted pLDDT scores (0-100).
            predicted_pae: Predicted PAE distances (Å).
            predicted_pde: Predicted PDE distances (Å).
            pred_coords: Predicted coordinates.
            true_coords: Ground-truth coordinates.
            coordinate_mask: Mask over atoms.
        
        Returns:
            Dictionary containing each loss term.
        """
        device = pred_coords.device
        losses = {
            'plddt_loss': torch.tensor(0.0, device=device),
            'pae_loss': torch.tensor(0.0, device=device),
            'pde_loss': torch.tensor(0.0, device=device),
            'resolved_loss': torch.tensor(0.0, device=device),  # Placeholder when unresolved
        }
        
        # 1. pLDDT Loss – compares predicted scores with RMSD-derived quality
        if predicted_lddt is not None:
            # Predicted scores can be [N_sample, N_tokens, atoms_per_token] or [N_sample, N_atoms]
            if predicted_lddt.ndim == 3:
                # [N_sample, N_tokens, atoms_per_token] -> [N_sample, N_atoms]
                pred_lddt_flat = predicted_lddt.reshape(predicted_lddt.shape[0], -1)
            else:
                pred_lddt_flat = predicted_lddt
            
            with torch.no_grad():
                # true_coords may be [N_sample, N_atoms, 3] or [N_atoms, 3]
                if true_coords.dim() == 3:
                    # Training variant: [N_sample, N_atoms, 3]
                    diff = pred_coords - true_coords
                else:
                    # Inference variant: [N_atoms, 3]
                    diff = pred_coords - true_coords.unsqueeze(0)
                
                # Per-atom RMSD
                rmsd_per_atom = torch.sqrt((diff ** 2).sum(dim=-1))  # [N_sample, N_atoms]
                
                # Convert to pLDDT-like scores (0-100) via score = 100 / (1 + RMSD/scale)
                scale = 5.0  # Å
                true_lddt = 100.0 / (1.0 + rmsd_per_atom / scale)
            
            # MSE loss
            mask = coordinate_mask.unsqueeze(0).expand(pred_lddt_flat.shape[0], -1)
            plddt_mse = F.mse_loss(
                pred_lddt_flat[mask],
                true_lddt[mask],
                reduction='mean'
            )
            losses['plddt_loss'] = plddt_mse
        
        # 2. PAE Loss – compare predicted PAE with ground-truth token distances
        if predicted_pae is not None:
            with torch.no_grad():
                # true_coords can be [N_sample, N_atoms, 3] or [N_atoms, 3]
                N_tokens = predicted_pae.shape[1]
                
                if true_coords.dim() == 3:
                    # Training shape: [N_sample, N_atoms, 3]
                    N_atoms = true_coords.shape[1]
                    atoms_per_token = N_atoms // N_tokens
                    # Use the first atom of each token as its representative
                    rep_coords = true_coords[:, ::atoms_per_token, :]  # [N_sample, N_tokens, 3]
                    true_token_dists = torch.cdist(rep_coords, rep_coords)  # [N_sample, N_tokens, N_tokens]
                else:
                    # Inference shape: [N_atoms, 3]
                    N_atoms = true_coords.shape[0]
                    atoms_per_token = N_atoms // N_tokens
                    # Use the first atom within each token as representative
                    rep_coords = true_coords[::atoms_per_token, :]  # [N_tokens, 3]
                    true_token_dists = torch.cdist(rep_coords, rep_coords)  # [N_tokens, N_tokens]
                    true_token_dists = true_token_dists.unsqueeze(0).expand(predicted_pae.shape[0], -1, -1)
            
            # MSE loss
            pae_mse = F.mse_loss(predicted_pae, true_token_dists, reduction='mean')
            losses['pae_loss'] = pae_mse
        
        # 3. PDE Loss – compare predicted PDE with actual distance errors
        if predicted_pde is not None:
            with torch.no_grad():
                # true_coords may be [N_sample, N_atoms, 3] or [N_atoms, 3]
                N_tokens = predicted_pde.shape[1]
                
                if true_coords.dim() == 3:
                    # Training variant: [N_sample, N_atoms, 3]
                    N_atoms = true_coords.shape[1]
                    atoms_per_token = N_atoms // N_tokens
                    
                    # Representative atom coordinates
                    rep_coords_true = true_coords[:, ::atoms_per_token, :]  # [N_sample, N_tokens, 3]
                    rep_coords_pred = pred_coords[:, ::atoms_per_token, :]  # [N_sample, N_tokens, 3]
                    
                    # Distances per token
                    pred_token_dists = torch.cdist(rep_coords_pred, rep_coords_pred)  # [N_sample, N_tokens, N_tokens]
                    true_token_dists = torch.cdist(rep_coords_true, rep_coords_true)  # [N_sample, N_tokens, N_tokens]
                    
                    # Ground-truth distance errors
                    true_pde = torch.abs(pred_token_dists - true_token_dists)
                else:
                    # Inference variant: [N_atoms, 3]
                    N_atoms = true_coords.shape[0]
                    atoms_per_token = N_atoms // N_tokens
                    
                    # Representative atom coordinates
                    rep_coords_true = true_coords[::atoms_per_token, :]  # [N_tokens, 3]
                    rep_coords_pred = pred_coords[:, ::atoms_per_token, :]  # [N_sample, N_tokens, 3]
                    
                    # Token-level distances
                    pred_token_dists = torch.cdist(rep_coords_pred, rep_coords_pred)  # [N_sample, N_tokens, N_tokens]
                    true_token_dists = torch.cdist(rep_coords_true, rep_coords_true)  # [N_tokens, N_tokens]
                    
                    # Absolute error per token pair
                    true_pde = torch.abs(pred_token_dists - true_token_dists.unsqueeze(0))
            
            # MSE loss
            pde_mse = F.mse_loss(predicted_pde, true_pde, reduction='mean')
            losses['pde_loss'] = pde_mse
        
        # 4. Resolved Loss – predict whether atoms are experimentally observed
        if predicted_resolved is not None and coordinate_mask is not None:
            # predicted_resolved: [N_sample, N_tokens, atoms_per_token]
            # coordinate_mask: [N_atoms] -> reshape to [N_tokens, atoms_per_token]
            N_sample = predicted_resolved.shape[0]
            N_tokens = predicted_resolved.shape[1]
            atoms_per_token = predicted_resolved.shape[2]
            
            # Reshape coordinate_mask
            true_resolved = coordinate_mask.reshape(N_tokens, atoms_per_token)  # [N_tokens, atoms_per_token]
            true_resolved = true_resolved.unsqueeze(0).expand(N_sample, -1, -1).float()  # [N_sample, N_tokens, atoms_per_token]
            
            # Binary cross entropy loss
            # Disable autocast for this operation as binary_cross_entropy is unsafe with autocast
            with torch.npu.amp.autocast(enabled=False):
                resolved_loss = F.binary_cross_entropy(
                    predicted_resolved.float(), 
                    true_resolved, 
                    reduction='mean'
                )
            losses['resolved_loss'] = resolved_loss
        
        # Aggregate confidence loss
        total_loss = (
            self.plddt_weight * losses['plddt_loss'] +
            self.pae_weight * losses['pae_loss'] +
            self.pde_weight * losses['pde_loss']
        )
        losses['total_confidence_loss'] = total_loss
        
        return losses
