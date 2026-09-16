#!/usr/bin/env python3
# Aggregate all loss terms — add chunking capability via chunk_utils.
# Demonstrates how to retrofit chunking support with chunk_utils.py.

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chunk_utils import ChunkProcessor  # Import chunk helper
from confidence_loss_simple import ConfidenceLossSimple
from train_loss_no_chunk import SmoothLDDTLoss, BondLoss, DistogramLoss, MSELoss


def get_default_config():
    return {
        'alpha_dna': 5.0, 'alpha_rna': 5.0, 'alpha_ligand': 10.0,
        'alpha_confidence': 1e-4, 'alpha_pae': 0.0, 'alpha_except_pae': 1.0,
        'alpha_diffusion': 4.0, 'alpha_distogram': 3e-2, 'alpha_bond': 1.0,
        'weight_smooth_lddt': 1.0,
        'distogram_min_bin': 2.3125, 'distogram_max_bin': 21.6875, 'distogram_no_bins': 64,
        # Chunk-specific configuration
        'chunk_size': None,  # None = no chunking, 10 = process 10 samples at a time
    }


class TorchfoldLossWithChunkUtils(nn.Module):
    """
    Simplified loss wrapper that leverages chunk_utils.
    
    Shows how to incorporate chunking with chunk_utils.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if config is None:
            config = get_default_config()
        self.config = config
        
        # Extract weighting parameters
        self.alpha_dna = config['alpha_dna']
        self.alpha_rna = config['alpha_rna']
        self.alpha_ligand = config['alpha_ligand']
        self.alpha_confidence = config['alpha_confidence']
        self.alpha_pae = config['alpha_pae']
        self.alpha_except_pae = config['alpha_except_pae']
        self.alpha_diffusion = config['alpha_diffusion']
        self.alpha_distogram = config['alpha_distogram']
        self.alpha_bond = config['alpha_bond']
        self.weight_smooth_lddt = config['weight_smooth_lddt']
        
        # Build chunk processor helper
        self.chunk_processor = ChunkProcessor(chunk_size=config.get('chunk_size', None))
        
        # Loss weights
        self.loss_weights = {
            "plddt_loss": self.alpha_confidence * self.alpha_except_pae,
            "pae_loss": self.alpha_confidence * self.alpha_pae,
            "pde_loss": self.alpha_confidence * self.alpha_except_pae,
            "resolved_loss": self.alpha_confidence * self.alpha_except_pae,
            "mse_loss": self.alpha_diffusion,
            "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "smooth_lddt_loss": self.alpha_diffusion * self.weight_smooth_lddt,
            "distogram_loss": self.alpha_distogram,
        }
        
        # Instantiate loss modules
        self.smooth_lddt_loss = SmoothLDDTLoss()
        self.bond_loss = BondLoss()
        self.mse_loss = MSELoss(
            weight_mse=1.0/3.0,
            weight_dna=self.alpha_dna,
            weight_rna=self.alpha_rna,
            weight_ligand=self.alpha_ligand,
        )
        self.confidence_loss = ConfidenceLossSimple()
        self.distogram_loss = DistogramLoss(
            min_bin=config['distogram_min_bin'],
            max_bin=config['distogram_max_bin'],
            no_bins=config['distogram_no_bins'],
        )
    
    def _compute_lddt_loss(self, pred_dists, true_dists, distance_mask, lddt_mask):
        """Compute LDDT loss (chunk-friendly)."""
        return self.smooth_lddt_loss(pred_dists, true_dists, distance_mask, lddt_mask)
    
    def _compute_bond_loss(self, pred_bond_dists, true_bond_dists, valid_bonds):
        """Compute bond loss (sparse, chunk-friendly)."""
        return self.bond_loss(pred_bond_dists, true_bond_dists, valid_bonds)
    
    def forward(self, model_output, batch, padding_mask=None):
        """
        Compute all enabled losses (with optional inner chunking).
        
        Args:
            model_output: Model outputs.
                Single sample: atom_positions [N_sample, N_tokens, atoms_per_token, 3]
                Batched: atom_positions [B, N_sample, N_tokens, atoms_per_token, 3]
            batch: Ground-truth tensors.
                Single sample: true_positions [N_tokens, atoms_per_token, 3]
                Batched: true_positions [B, N_tokens, atoms_per_token, 3]
            padding_mask: Optional padding mask.
                Single sample: [N_tokens]
                Batched: [B, N_tokens]
        
        Returns:
            Dictionary containing all loss terms.
        """
        # Determine whether the inputs are batched
        if self._is_batched(model_output):
            return self._forward_batched(model_output, batch, padding_mask)
        else:
            return self._forward_single(model_output, batch, padding_mask)
    
    def _is_batched(self, model_output):
        """Check whether the model output includes a batch dimension."""
        if 'diffusion_samples' in model_output:
            atom_pos = model_output['diffusion_samples']['atom_positions']
        elif 'pred_coords' in model_output:
            atom_pos = model_output['pred_coords']
        else:
            return False
        return atom_pos.ndim == 5
    
    def _forward_batched(self, model_output, batch, padding_mask):
        """Handle the batched case by iterating over samples."""
        if 'diffusion_samples' in model_output:
            B = model_output['diffusion_samples']['atom_positions'].shape[0]
        else:
            B = model_output['pred_coords'].shape[0]
        
        loss_dicts = []
        for b in range(B):
            output_b = self._extract_batch_item(model_output, b)
            batch_b = self._extract_batch_item(batch, b)
            mask_b = padding_mask[b] if padding_mask is not None else None
            
            loss_dict_b = self._forward_single(output_b, batch_b, mask_b)
            loss_dicts.append(loss_dict_b)
        
        return self._average_loss_dicts(loss_dicts)
    
    def _extract_batch_item(self, data_dict, batch_idx):
        """Recursively extract the batch_idx entry from nested dictionaries."""
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                result[key] = self._extract_batch_item(value, batch_idx)
            elif isinstance(value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] > 1:
                    result[key] = value[batch_idx]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def _average_loss_dicts(self, loss_dicts):
        """Average a list of per-sample loss dictionaries."""
        avg_dict = {}
        for key in loss_dicts[0].keys():
            values = [d[key] for d in loss_dicts]
            if isinstance(values[0], torch.Tensor):
                avg_dict[key] = torch.stack(values).mean()
            else:
                avg_dict[key] = values[0]
        return avg_dict
    
    def _forward_single(self, model_output, batch, padding_mask=None):
        """Process a single sample (original forward logic)."""
        # Ensure predicted coordinates exist
        if 'pred_coords' not in model_output:
            model_output['pred_coords'] = model_output['diffusion_samples']['atom_positions']
        
        # Ensure ligand mask exists
        if 'is_ligand' not in batch:
            batch['is_ligand'] = torch.zeros_like(batch['is_dna'])
        
        # Derive bond pairs when absent
        if 'bond_pairs' not in batch:
            # Attempt 1: atom-level polymer/ligand bonds
            key = 'token_atoms_to_polymer_ligand_bonds:gather_idxs'
            mask_key = 'token_atoms_to_polymer_ligand_bonds:gather_mask'
            if key in batch and mask_key in batch:
                bond_idxs = batch[key]
                bond_mask = batch[mask_key]
                valid_mask = bond_mask.all(dim=1)
                if valid_mask.sum() > 0:
                    batch['bond_pairs'] = bond_idxs[valid_mask]
            
            # Attempt 2: token-level ligand/ligand bonds
            if 'bond_pairs' not in batch:
                key2 = 'tokens_to_ligand_ligand_bonds:gather_idxs'
                mask_key2 = 'tokens_to_ligand_ligand_bonds:gather_mask'
                if key2 in batch and mask_key2 in batch:
                    bond_idxs2 = batch[key2]
                    bond_mask2 = batch[mask_key2]
                    valid_mask2 = bond_mask2.all(dim=1)
                    if valid_mask2.sum() > 0:
                        valid_bonds = bond_idxs2[valid_mask2]
                        batch['bond_pairs'] = valid_bonds * 24
        
        # Populate distogram logits if needed
        if 'distogram_logits' not in model_output and 'distogram' in model_output:
            if isinstance(model_output['distogram'], dict) and 'contact_probs' in model_output['distogram']:
                model_output['distogram_logits'] = model_output['distogram']['contact_probs']
        
        # Gather tensors
        pred_coords = model_output['pred_coords']  # [N_sample, N_tokens, atoms_per_token, 3]
        true_positions = batch['true_positions']
        coordinate_mask = batch['true_positions_atom_mask']
        is_dna = batch['is_dna']
        is_rna = batch['is_rna']
        is_ligand = batch['is_ligand']
        bond_pairs = batch.get('bond_pairs', None)
        
        device = pred_coords.device
        
        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask_expanded = padding_mask.unsqueeze(-1).float()
            coordinate_mask = coordinate_mask.float() * padding_mask_expanded
        
        # Reshape
        N_sample = pred_coords.shape[0]
        pred_coords = pred_coords.reshape(N_sample, -1, 3)
        true_coords = true_positions.reshape(-1, 3)
        coord_mask = coordinate_mask.reshape(-1)
        
        # Broadcast is_* flags to atom level
        is_dna_atoms = is_dna.unsqueeze(-1).expand(-1, 24).reshape(-1)
        is_rna_atoms = is_rna.unsqueeze(-1).expand(-1, 24).reshape(-1)
        is_ligand_atoms = is_ligand.unsqueeze(-1).expand(-1, 24).reshape(-1)
        
        # Accumulate loss scalars
        losses = {}
        
        # 1. Smooth LDDT Loss (chunked)
        if self.config.get('use_smooth_lddt_loss', False):
            # Compute distance matrices
            def compute_lddt_for_chunk(pred_coords_chunk):
                """Compute LDDT loss over a chunk."""
                pred_dists_chunk = torch.cdist(pred_coords_chunk, pred_coords_chunk)
                true_dists = torch.cdist(true_coords.unsqueeze(0), true_coords.unsqueeze(0)).squeeze(0)
                
                coord_mask_bool = coord_mask.bool()
                distance_mask = coord_mask_bool.unsqueeze(-1) & coord_mask_bool.unsqueeze(-2)
                lddt_mask = distance_mask & (true_dists < 15.0)
                
                return self._compute_lddt_loss(
                    pred_dists_chunk,
                    true_dists.unsqueeze(0).expand(pred_coords_chunk.shape[0], -1, -1),
                    distance_mask,
                    lddt_mask
                )
            
            # Use chunk_processor to split along N_sample
            lddt_loss = self.chunk_processor.process_mean(
                compute_lddt_for_chunk,
                pred_coords,
                dim=-3  # Chunk along the N_sample dimension
            )
            losses['smooth_lddt'] = lddt_loss
        else:
            losses['smooth_lddt'] = torch.tensor(0.0, device=device)
        
        # 2. Bond Loss (sparse bond-pair, chunked over N_sample)
        if self.config.get('use_bond_loss', False) and bond_pairs is not None:
            idx_i = bond_pairs[:, 0]
            idx_j = bond_pairs[:, 1]
            coord_mask_bool = coord_mask.bool()
            valid_bonds = coord_mask_bool[idx_i] & coord_mask_bool[idx_j]

            if valid_bonds.any():
                true_bond_dists = torch.norm(
                    true_coords[idx_i, :] - true_coords[idx_j, :], dim=-1
                ).unsqueeze(0)  # [1, N_bonds]

                def compute_bond_for_chunk(pred_coords_chunk):
                    pred_i = pred_coords_chunk[:, idx_i, :]
                    pred_j = pred_coords_chunk[:, idx_j, :]
                    pred_bond_dists = torch.norm(pred_i - pred_j, dim=-1)  # [chunk, N_bonds]
                    return self._compute_bond_loss(pred_bond_dists, true_bond_dists, valid_bonds)

                bond_loss_val = self.chunk_processor.process_mean(
                    compute_bond_for_chunk,
                    pred_coords,
                    dim=-3,
                )
                losses['bond'] = bond_loss_val
            else:
                losses['bond'] = torch.tensor(0.0, device=device)
        else:
            losses['bond'] = torch.tensor(0.0, device=device)
        
        # 3. MSE Loss
        mse_loss_val = self.mse_loss(
            pred_coords,
            true_coords,
            coord_mask.bool(),
            is_dna_atoms.bool(),
            is_rna_atoms.bool(),
            is_ligand_atoms.bool(),
        )
        losses['mse_loss'] = mse_loss_val
        
        # 4. Confidence Loss
        if self.config.get('use_confidence_loss', False):
            predicted_lddt = model_output.get('predicted_lddt', None)
            predicted_pae = model_output.get('full_pae', None)
            predicted_pde = model_output.get('full_pde', None)
            predicted_resolved = model_output.get('predicted_experimentally_resolved', None)
            
            confidence_losses = self.confidence_loss(
                predicted_lddt=predicted_lddt,
                predicted_pae=predicted_pae,
                predicted_pde=predicted_pde,
                predicted_resolved=predicted_resolved,
                pred_coords=pred_coords,
                true_coords=true_coords,
                coordinate_mask=coord_mask.bool(),
            )
            losses['confidence'] = confidence_losses
        else:
            losses['confidence'] = {
                'plddt_loss': torch.tensor(0.0, device=device),
                'pae_loss': torch.tensor(0.0, device=device),
                'pde_loss': torch.tensor(0.0, device=device),
                'resolved_loss': torch.tensor(0.0, device=device),
            }
        
        # 5. Distogram Loss
        if self.config.get('use_distogram_loss', False):
            distogram_dict = model_output.get('distogram', None)
            if distogram_dict is not None and 'contact_probs' in distogram_dict:
                contact_probs = distogram_dict['contact_probs']
                N_tokens = contact_probs.shape[0]
                rep_coords = true_coords[::24][:N_tokens]
                true_dists = torch.cdist(rep_coords, rep_coords)
                true_contacts = (true_dists < 8.0).float()
                
                distogram_loss_val = F.binary_cross_entropy(
                    contact_probs, true_contacts, reduction='mean'
                )
                losses['distogram'] = distogram_loss_val
            else:
                losses['distogram'] = torch.tensor(0.0, device=device)
        else:
            losses['distogram'] = torch.tensor(0.0, device=device)
        
        # Apply weights
        conf = losses['confidence']
        loss_smooth_lddt = self.loss_weights['smooth_lddt_loss'] * losses['smooth_lddt']
        loss_bond = self.loss_weights['bond_loss'] * losses['bond']
        loss_mse = self.loss_weights['mse_loss'] * losses['mse_loss']
        loss_plddt = self.loss_weights['plddt_loss'] * conf['plddt_loss']
        loss_pae = self.loss_weights['pae_loss'] * conf['pae_loss']
        loss_pde = self.loss_weights['pde_loss'] * conf['pde_loss']
        loss_resolved = self.loss_weights['resolved_loss'] * conf['resolved_loss']
        loss_conf_total = loss_plddt + loss_pae + loss_pde + loss_resolved
        loss_distogram = self.loss_weights['distogram_loss'] * losses['distogram']
        
        total = loss_smooth_lddt + loss_bond + loss_mse + loss_conf_total + loss_distogram
        
        return {
            'total_loss': total,
            'smooth_lddt_loss': loss_smooth_lddt,
            'bond_loss': loss_bond,
            'mse_loss': loss_mse,
            'total_confidence_loss': loss_conf_total,
            'distogram_loss': loss_distogram,
            'plddt_loss': loss_plddt,        
            'pae_loss': loss_pae,           
            'pde_loss': loss_pde,            
            'resolved_loss': loss_resolved,  
            'chunk_size': self.config.get('chunk_size', None),  
        }

