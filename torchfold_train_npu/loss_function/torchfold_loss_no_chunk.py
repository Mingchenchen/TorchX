#!/usr/bin/env python3
# Aggregate all loss terms — simplified version without chunking.
# Intended for small batches (N_sample < 10).

import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from confidence_loss_simple import ConfidenceLossSimple
from train_loss_no_chunk import SmoothLDDTLoss, BondLoss, DistogramLoss, MSELoss


def get_default_config():
    return {
        'alpha_dna': 5.0, 'alpha_rna': 5.0, 'alpha_ligand': 10.0,
        'alpha_confidence': 1e-4, 'alpha_pae': 1.0, 'alpha_except_pae': 1.0,
        'alpha_diffusion': 4.0, 'alpha_distogram': 3e-2, 'alpha_bond': 1.0,
        'weight_smooth_lddt': 1.0,
        'distogram_min_bin': 2.3125, 'distogram_max_bin': 21.6875, 'distogram_no_bins': 64,
        # Interface loss weighting
        'use_interface_loss_weight': False,
        'interface_loss_weight': 5.0,  # multiplier for interface residues
        'interface_distance_threshold': 10.0,  # Å, CA-CA distance for interface definition
    }


class TorchfoldLossNoChunk(nn.Module):

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

        # Loss weights and scalings
        self.loss_weights = {
            "plddt_loss": self.alpha_confidence * self.alpha_except_pae,
            "pae_loss": self.alpha_confidence * self.alpha_pae,
            "pde_loss": self.alpha_confidence * self.alpha_except_pae,
            "resolved_loss": self.alpha_confidence * self.alpha_except_pae,
            "mse_loss": self.alpha_diffusion / 3.0,  # 1/3 of diffusion loss
            "bond_loss": self.alpha_diffusion * self.alpha_bond / 3.0,  # 1/3 of diffusion loss
            "smooth_lddt_loss": self.alpha_diffusion * self.weight_smooth_lddt / 3.0,  # 1/3 of diffusion loss
            "distogram_loss": self.alpha_distogram,
        }

        # Instantiate loss modules
        self.smooth_lddt_loss = SmoothLDDTLoss()
        self.bond_loss = BondLoss()
        self.mse_loss = MSELoss(
            weight_mse=1.0,  # Weight is managed uniformly in loss_weights
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

        # Interface loss weighting
        self.use_interface_loss_weight = config.get('use_interface_loss_weight', False)
        self.interface_loss_weight = config.get('interface_loss_weight', 5.0)
        self.interface_distance_threshold = config.get('interface_distance_threshold', 10.0)

    def forward(self, model_output, batch, padding_mask=None):
        """
        Compute every enabled loss term.

        Args:
            model_output: Model predictions.
                Single sample: atom_positions [N_sample, N_tokens, atoms_per_token, 3]
                Batched samples: atom_positions [B, N_sample, N_tokens, atoms_per_token, 3]
            batch: Ground-truth supervision tensors.
                Single sample: true_positions [N_tokens, atoms_per_token, 3]
                Batched samples: true_positions [B, N_tokens, atoms_per_token, 3]
            padding_mask: Optional padding mask.
                Single sample: [N_tokens]
                Batched samples: [B, N_tokens]
                1 denotes valid tokens, 0 denotes padding.

        Returns:
            Dictionary containing each component loss.
        """
        # Determine whether we operate on batched tensors
        if self._is_batched(model_output):
            return self._forward_batched(model_output, batch, padding_mask)
        else:
            return self._forward_single(model_output, batch, padding_mask)

    def _is_batched(self, model_output):
        """Check whether model_output contains a batch dimension."""
        if 'diffusion_samples' in model_output and isinstance(model_output['diffusion_samples'], dict):
            if 'atom_positions' in model_output['diffusion_samples']:
                atom_pos = model_output['diffusion_samples']['atom_positions']
            elif 'pred_coords' in model_output:
                atom_pos = model_output['pred_coords']
            else:
                return False
        elif 'pred_coords' in model_output:
            atom_pos = model_output['pred_coords']
        else:
            return False
        return atom_pos.ndim == 5  # [B, N_sample, N_tokens, atoms, 3]

    def _forward_batched(self, model_output, batch, padding_mask):
        """Process the batched case."""
        # Determine batch size
        if 'diffusion_samples' in model_output:
            B = model_output['diffusion_samples']['atom_positions'].shape[0]
        else:
            B = model_output['pred_coords'].shape[0]

        loss_dicts = []
        for b in range(B):
            # Extract the b-th sample
            output_b = self._extract_batch_item(model_output, b)
            batch_b = self._extract_batch_item(batch, b)
            mask_b = padding_mask[b] if padding_mask is not None else None

            # Compute per-sample losses
            loss_dict_b = self._forward_single(output_b, batch_b, mask_b)
            loss_dicts.append(loss_dict_b)

        # Average across samples
        return self._average_loss_dicts(loss_dicts)

    def _extract_batch_item(self, data_dict, batch_idx):
        """Recursively pick the batch_idx entry from a nested dict."""
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                result[key] = self._extract_batch_item(value, batch_idx)
            elif isinstance(value, torch.Tensor):
                # Inspect for a batch dimension (first dim > 1)
                if value.ndim > 0 and value.shape[0] > 1:
                    result[key] = value[batch_idx]
                else:
                    # No batch dimension or singleton
                    result[key] = value
            else:
                result[key] = value
        return result

    def _average_loss_dicts(self, loss_dicts):
        """Average a sequence of per-sample loss dictionaries."""
        avg_dict = {}
        for key in loss_dicts[0].keys():
            values = [d[key] for d in loss_dicts]
            if isinstance(values[0], torch.Tensor):
                avg_dict[key] = torch.stack(values).mean()
            else:
                # Non-tensor values (e.g., metadata) remain untouched
                avg_dict[key] = values[0]
        return avg_dict

    def _compute_interface_weights(self, batch, atoms_per_token: int = 24):
        """
        Compute per-atom interface weight multiplier.

        Interface residues are defined as residues from different chains (asym_id)
        whose CA atoms are within `interface_distance_threshold` Å of each other.

        Args:
            batch: input batch with 'asym_id', 'true_positions', 'true_positions_atom_mask', 'is_protein'
            atoms_per_token: number of atoms per token (default 24)

        Returns:
            (interface_weights, interface_atom_mask):
                interface_weights: [N_atoms] tensor, interface_loss_weight for interface atoms, 1.0 otherwise.
                interface_atom_mask: [N_atoms] boolean tensor, True for interface atoms.
            Returns (None, None) if computation fails or is not applicable.
        """
        asym_id = batch.get('asym_id', None)  # [N_tokens]
        true_pos = batch.get('true_positions', None)  # [N_tokens, atoms_per_token, 3]
        atom_mask = batch.get('true_positions_atom_mask', None)  # [N_tokens, atoms_per_token]

        if asym_id is None or true_pos is None or atom_mask is None:
            return None, None

        N_tokens = asym_id.shape[0]
        device = asym_id.device

        # Extract CA coordinates (index 1 for standard residues)
        # For non-protein tokens, use the first valid atom as representative
        ca_coords = true_pos[:, 1, :]  # [N_tokens, 3]
        ca_mask = atom_mask[:, 1].bool()  # [N_tokens]

        # Build cross-chain mask: different asym_id
        asym_i = asym_id.unsqueeze(1)  # [N_tokens, 1]
        asym_j = asym_id.unsqueeze(0)  # [1, N_tokens]
        cross_chain_mask = (asym_i != asym_j)  # [N_tokens, N_tokens]

        # Compute CA-CA distances
        ca_dists = torch.cdist(ca_coords, ca_coords)  # [N_tokens, N_tokens]

        # Valid pair mask: both residues have valid CA
        valid_pair = ca_mask.unsqueeze(1) & ca_mask.unsqueeze(0)  # [N_tokens, N_tokens]

        # Interface contact: cross-chain, valid, within threshold
        interface_contact = cross_chain_mask & valid_pair & (ca_dists < self.interface_distance_threshold)

        # Interface residues: any cross-chain contact
        interface_token_mask = interface_contact.any(dim=1)  # [N_tokens]

        # Build per-atom weights
        interface_weights = torch.ones(N_tokens * atoms_per_token, device=device, dtype=torch.float32)
        # Expand token mask to atom level
        interface_atom_mask = interface_token_mask.unsqueeze(-1).expand(-1, atoms_per_token).reshape(-1).bool()
        interface_weights[interface_atom_mask] = self.interface_loss_weight

        return interface_weights, interface_atom_mask

    def _build_rep_atom_mask(
            self,
            batch,
            coord_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build a flat representative-atom mask from pseudo-beta gather info."""
        gather_idxs = batch.get('token_atoms_to_pseudo_beta:gather_idxs', None)
        gather_mask = batch.get('token_atoms_to_pseudo_beta:gather_mask', None)
        if gather_idxs is None or gather_mask is None:
            raise KeyError(
                "Missing token_atoms_to_pseudo_beta gather info required for distogram loss."
            )

        rep_atom_mask = torch.zeros(
            coord_mask.shape[-1],
            dtype=torch.bool,
            device=coord_mask.device,
        )
        valid_gather_idxs = gather_idxs.to(device=coord_mask.device, dtype=torch.long)[
            gather_mask.to(device=coord_mask.device, dtype=torch.bool)
        ]
        if valid_gather_idxs.numel() > 0:
            valid_gather_idxs = valid_gather_idxs[
                (valid_gather_idxs >= 0) & (valid_gather_idxs < coord_mask.shape[-1])
                ]
            rep_atom_mask[valid_gather_idxs] = True

        return rep_atom_mask

    def _forward_single(self, model_output, batch, padding_mask=None):
        """Handle a single sample (original forward logic)."""
        confidence_only = bool(model_output.get('confidence_only', False))

        # Ensure predicted coordinates exist
        if 'pred_coords' not in model_output:
            if 'diffusion_samples' in model_output and 'atom_positions' in model_output['diffusion_samples']:
                model_output['pred_coords'] = model_output['diffusion_samples']['atom_positions']
            else:
                raise KeyError("pred_coords is required when diffusion_samples are absent.")

        # Supply is_ligand when missing
        if 'is_ligand' not in batch:
            batch['is_ligand'] = torch.zeros_like(batch['is_dna'])

        # Build bond pairs when missing
        # Force-create bond_pairs for debug: fabricate sparse chain bonds
        coord_mask_atoms = batch['true_positions_atom_mask'].reshape(-1)  # [N_atoms]
        valid_atoms = torch.nonzero(coord_mask_atoms, as_tuple=False).squeeze(-1)
        if valid_atoms.numel() >= 2:
            # To control memory, only keep first K atoms in the chain
            K = min(valid_atoms.numel(), 3000)
            valid_atoms = valid_atoms[:K]
            pairs = torch.stack([valid_atoms[:-1], valid_atoms[1:]], dim=1)  # [K-1, 2]
        else:
            # No valid atoms, fall back to a single self-pair placeholder
            pairs = torch.tensor([[0, 0]], device=coord_mask_atoms.device, dtype=torch.long)
        batch['bond_pairs'] = pairs.to(device=coord_mask_atoms.device, dtype=torch.long)

        # Populate distogram logits when absent
        if 'distogram_logits' not in model_output and 'distogram' in model_output:
            if isinstance(model_output['distogram'], dict) and 'logits' in model_output['distogram']:
                model_output['distogram_logits'] = model_output['distogram']['logits']

        # resolved predictions already available, nothing further required

        # Pull tensors needed below
        pred_coords = model_output['pred_coords']  # [N_sample, N_tokens, atoms_per_token, 3]

        # Original GT (used for MSE loss rigid alignment)
        original_true_positions = batch['true_positions']  # [N_tokens, atoms_per_token, 3]

        # Augmented GT (used for distance-based losses such as LDDT, Bond)
        # Note: Distance-based losses are invariant to rotation/translation, so using augmented or original makes no difference
        if (not confidence_only and 'diffusion_samples' in model_output and
                'gt_positions' in model_output['diffusion_samples']):
            true_positions = model_output['diffusion_samples'][
                'gt_positions']  # [N_sample, N_tokens, atoms_per_token, 3]
        else:
            true_positions = batch['true_positions']  # [N_tokens, atoms_per_token, 3]

        coordinate_mask = batch['true_positions_atom_mask']  # [N_tokens, atoms_per_token]
        is_dna = batch['is_dna']  # [N_tokens]
        is_rna = batch['is_rna']
        is_ligand = batch['is_ligand']
        bond_pairs = batch.get('bond_pairs', None)

        device = pred_coords.device

        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask: [N_tokens], 1=valid, 0=padding
            # Expand to [N_tokens, atoms_per_token]
            padding_mask_expanded = padding_mask.unsqueeze(-1).float()  # [N_tokens, 1]
            coordinate_mask = coordinate_mask.float() * padding_mask_expanded  # [N_tokens, atoms_per_token]

        # Reshape: [N_sample, N_tokens, atoms_per_token, 3] -> [N_sample, N_atoms, 3]
        N_sample = pred_coords.shape[0]
        pred_coords = pred_coords.reshape(N_sample, -1, 3)

        # Original GT (used for MSE loss rigid alignment): always [N_atoms, 3]
        original_true_coords = original_true_positions.reshape(-1, 3)  # [N_atoms, 3]

        # true_positions can be [N_sample, N_tokens, atoms_per_token, 3] or [N_tokens, atoms_per_token, 3]
        # Used for distance-based losses (LDDT, Bond, etc.), distances are invariant to rotation/translation
        if true_positions.dim() == 4:
            # Training mode: augmented GT [N_sample, N_tokens, atoms_per_token, 3]
            true_coords = true_positions.reshape(N_sample, -1, 3)  # [N_sample, N_atoms, 3]
        else:
            # Inference mode: base GT [N_tokens, atoms_per_token, 3]
            true_coords = true_positions.reshape(-1, 3)  # [N_atoms, 3]

        coord_mask = coordinate_mask.reshape(-1)

        # Broadcast is_* indicators to atom level
        is_dna_atoms = is_dna.unsqueeze(-1).expand(-1, 24).reshape(-1)
        is_rna_atoms = is_rna.unsqueeze(-1).expand(-1, 24).reshape(-1)
        is_ligand_atoms = is_ligand.unsqueeze(-1).expand(-1, 24).reshape(-1)

        # Container for loss components
        losses = {}

        # 1. Smooth LDDT Loss + 2. Bond Loss (chunked to reduce OOM for large N_sample)
        N_sample = pred_coords.shape[0]
        Dim_true = true_coords.dim()
        use_lddt = (not confidence_only) and self.config.get('use_smooth_lddt_loss', False)
        use_bond = (not confidence_only) and self.config.get('use_bond_loss', False) and bond_pairs is not None

        coord_mask_bool = coord_mask.bool()
        distance_mask = coord_mask_bool.unsqueeze(-1) & coord_mask_bool.unsqueeze(-2)

        chunk_env = int(os.environ.get("TROCHFOLD_DIFFUSION_SAMPLE_CHUNK", "0"))
        if chunk_env > 0:
            chunk_size = chunk_env
        else:
            chunk_size = 4 if N_sample >= 8 else 0  # 0 => no chunking

        if use_lddt or use_bond:
            if Dim_true == 3:
                true_dists_0 = torch.cdist(true_coords[0], true_coords[0])
            else:
                true_dists_0 = torch.cdist(true_coords, true_coords)
            lddt_mask = distance_mask & (true_dists_0 < 15.0)

        if use_lddt:
            if chunk_size > 0:
                lddt_sum = 0.0
                for i in range(0, N_sample, chunk_size):
                    start = i
                    end = min(N_sample, i + chunk_size)
                    pred_chunk = pred_coords[start:end, :, :]
                    pred_dists = torch.cdist(pred_chunk, pred_chunk)
                    if Dim_true == 3:
                        true_chunk = true_coords[start:end, :, :]
                        true_dists = torch.cdist(true_chunk, true_chunk)
                    else:
                        true_dists = true_dists_0.unsqueeze(0).expand(end - start, -1, -1)
                    pred_dists -= true_dists
                    del true_dists
                    lddt_chunk = self.smooth_lddt_loss(
                        pred_dists,
                        distance_mask,
                        lddt_mask
                    )
                    lddt_sum = lddt_sum + lddt_chunk * (end - start)
                losses['smooth_lddt'] = lddt_sum / N_sample
            else:
                pred_dists = torch.cdist(pred_coords, pred_coords)
                true_dists = torch.cdist(true_coords, true_coords)
                if Dim_true == 2:
                    true_dists = true_dists.unsqueeze(0)
                pred_dists -= true_dists
                del true_dists
                losses['smooth_lddt'] = self.smooth_lddt_loss(
                    pred_dists,
                    distance_mask,
                    lddt_mask
                )
        else:
            losses['smooth_lddt'] = torch.tensor(0.0, device=device)

        if use_bond:
            idx_i = bond_pairs[:, 0]
            idx_j = bond_pairs[:, 1]
            valid_bonds = coord_mask_bool[idx_i] & coord_mask_bool[idx_j]

            if valid_bonds.any():
                if Dim_true == 3:
                    true_bond_dists = torch.norm(
                        true_coords[:, idx_i, :] - true_coords[:, idx_j, :], dim=-1
                    )  # [N_sample, N_bonds]
                else:
                    true_bond_dists = torch.norm(
                        true_coords[idx_i, :] - true_coords[idx_j, :], dim=-1
                    ).unsqueeze(0)  # [1, N_bonds]

                if chunk_size > 0:
                    bond_sum = 0.0
                    for i in range(0, N_sample, chunk_size):
                        start = i
                        end = min(N_sample, i + chunk_size)
                        pred_i = pred_coords[start:end, idx_i, :]
                        pred_j = pred_coords[start:end, idx_j, :]
                        pred_bond_dists = torch.norm(pred_i - pred_j, dim=-1)
                        true_chunk = true_bond_dists[start:end] if Dim_true == 3 else true_bond_dists
                        bond_chunk = self.bond_loss(pred_bond_dists, true_chunk, valid_bonds)
                        bond_sum = bond_sum + bond_chunk * (end - start)
                    losses['bond'] = bond_sum / N_sample
                else:
                    pred_i = pred_coords[:, idx_i, :]
                    pred_j = pred_coords[:, idx_j, :]
                    pred_bond_dists = torch.norm(pred_i - pred_j, dim=-1)
                    losses['bond'] = self.bond_loss(pred_bond_dists, true_bond_dists, valid_bonds)
            else:
                losses['bond'] = torch.tensor(0.0, device=device)
        else:
            losses['bond'] = torch.tensor(0.0, device=device)

        # Retrieve noise_levels for diffusion loss weighting
        # noise_level value range: log-normal distribution, p_mean=-1.2, p_std=1.5
        # Approximate range [0.01, 10.0], median ~0.3
        noise_levels = None
        if (not confidence_only and 'diffusion_samples' in model_output and
                'noise_levels' in model_output['diffusion_samples']):
            noise_levels = model_output['diffusion_samples']['noise_levels']  # [N_sample]
            t_values = model_output['diffusion_samples'].get('t_values',
                                                             None)  # [N_sample] (returned during training sampling)
        else:
            t_values = None

        # 3. MSE Loss (align original GT to pred, not augmented GT)
        # Equation (6): per-sample noise weighting is done inside MSELoss
        # Return: (weighted total loss, per-sample losses for statistics)
        # Compute interface weights if enabled
        interface_weights = None
        interface_atom_mask = None
        if self.use_interface_loss_weight and not confidence_only:
            try:
                interface_weights, interface_atom_mask = self._compute_interface_weights(batch)
            except Exception as e:
                print(f"Warning: Failed to compute interface weights: {e}", flush=True)
                interface_weights = None
                interface_atom_mask = None

        interface_metrics = None
        if confidence_only:
            mse_loss_val = torch.tensor(0.0, device=device)
            per_sample_mse = None
        else:
            mse_loss_val, per_sample_mse, interface_metrics = self.mse_loss(
                batch,
                pred_coords,
                original_true_coords,  # Use original GT, which will be aligned to pred inside MSELoss
                coord_mask.bool(),
                is_dna_atoms.bool(),
                is_rna_atoms.bool(),
                is_ligand_atoms.bool(),
                noise_levels=noise_levels,  # Pass noise_levels for per-sample weighting
                interface_weights=interface_weights,  # Interface residue weighting
                interface_atom_mask=interface_atom_mask,  # Interface atom mask (for separate statistics)
            )
        losses['mse_loss'] = mse_loss_val

        # 4. Confidence Loss
        if self.config.get('use_confidence_loss', False):
            predicted_lddt = model_output.get('predicted_lddt', None)
            predicted_pae = model_output.get('full_pae', None)
            predicted_pde = model_output.get('full_pde', None)
            predicted_resolved = model_output.get('predicted_experimentally_resolved', None)

            # Confidence head may only see 1 mini-rollout sample,
            # while pred_coords comes from diffusion's N_sample (e.g., 32);
            # use confidence_atom_positions to ensure dimension consistency.
            conf_atom_pos = model_output.get('confidence_atom_positions', None)
            if conf_atom_pos is not None:
                conf_pred = conf_atom_pos.reshape(conf_atom_pos.shape[0], -1, 3)
            else:
                conf_pred = pred_coords

            # true_coords may also be [N_sample_diffusion, N_atoms, 3],
            # need to align with the number of samples in confidence
            if true_coords.dim() == 3 and conf_pred.shape[0] != true_coords.shape[0]:
                conf_true = true_coords[:1]  # take GT from the first sample
            else:
                conf_true = true_coords

            confidence_losses = self.confidence_loss(
                predicted_lddt=predicted_lddt,
                predicted_pae=predicted_pae,
                predicted_pde=predicted_pde,
                predicted_resolved=predicted_resolved,
                pred_coords=conf_pred,
                true_coords=conf_true,
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
        if (not confidence_only) and self.config.get('use_distogram_loss', False):
            distogram_logits = model_output.get('distogram_logits', None)
            if distogram_logits is not None:
                rep_atom_mask = self._build_rep_atom_mask(batch, coord_mask)
                losses['distogram'] = self.distogram_loss(
                    logits=distogram_logits,
                    true_coordinate=original_true_coords,
                    coordinate_mask=coord_mask,
                    rep_atom_mask=rep_atom_mask,
                )
            else:
                losses['distogram'] = torch.tensor(0.0, device=device)
        else:
            losses['distogram'] = torch.tensor(0.0, device=device)

        # Apply configured weights
        conf = losses['confidence']

        # Equation (6): L_diffusion = w(t) * (L_MSE + α_bond * L_bond) + L_smooth_lddt
        # w(t) = (t² + σ_data²) / (t + σ_data)², σ_data = 16.0
        sigma_data = 16.0

        # MSE loss already has per-sample noise weighting done inside MSELoss
        # Here we only need to multiply by the global weight
        loss_mse = self.loss_weights['mse_loss'] * losses['mse_loss']

        # Bond loss, if enabled, requires external noise weighting (using average noise_level)
        if noise_levels is not None:
            t_mean = noise_levels.mean()
            noise_weight_mean = (t_mean ** 2 + sigma_data ** 2) / ((sigma_data * t_mean) ** 2)
        else:
            noise_weight_mean = 1.0
        loss_bond = noise_weight_mean * self.loss_weights['bond_loss'] * losses['bond']

        # Smooth LDDT is not weighted
        loss_smooth_lddt = self.loss_weights['smooth_lddt_loss'] * losses['smooth_lddt']

        loss_plddt = self.loss_weights['plddt_loss'] * conf['plddt_loss']
        loss_pae = self.loss_weights['pae_loss'] * conf['pae_loss']
        loss_pde = self.loss_weights['pde_loss'] * conf['pde_loss']
        loss_resolved = self.loss_weights['resolved_loss'] * conf['resolved_loss']
        loss_conf_total = loss_plddt + loss_pae + loss_pde + loss_resolved
        loss_distogram = self.loss_weights['distogram_loss'] * losses['distogram']
        # Note: smooth_lddt is not affected by noise level weighting (Equation 6)
        total = loss_smooth_lddt + loss_bond + loss_mse + loss_conf_total + loss_distogram
        # if losses['mse_loss'] >= 0.8:
        #     print(f"mse_loss: {losses['mse_loss']}")
        #     print(f"batch: {batch['sample_id']}")
        # print(batch['sample_id'])
        # Build return dictionary
        result = {
            'total_loss': total,
            'smooth_lddt_loss': loss_smooth_lddt,
            'bond_loss': loss_bond,
            'mse_loss': losses['mse_loss'],
            'total_confidence_loss': loss_conf_total,
            'distogram_loss': loss_distogram,
            'plddt_loss': loss_plddt,
            'pae_loss': loss_pae,
            'pde_loss': loss_pde,
            'resolved_loss': loss_resolved,
        }

        # Interface loss metrics (for TensorBoard monitoring)
        if interface_metrics is not None:
            for k, v in interface_metrics.items():
                result[k] = v

        # debug dump: expose per-sample metrics for one-off debug dump
        if os.environ.get("TROCHFOLD_DEBUG_0116", "0") == "1" and per_sample_mse is not None:
            if noise_levels is not None:
                noise_weights = (noise_levels ** 2 + sigma_data ** 2) / (
                        (sigma_data * noise_levels) ** 2 + 1e-8
                )
                per_sample_weighted_mse = per_sample_mse * noise_weights
            else:
                noise_weights = None
                per_sample_weighted_mse = None
            result['debug_0116_per_sample_mse'] = per_sample_mse.detach()
            result['debug_0116_per_sample_weighted_mse'] = (
                per_sample_weighted_mse.detach() if per_sample_weighted_mse is not None else None
            )
            result['debug_0116_noise_levels'] = noise_levels.detach() if noise_levels is not None else None
            result['debug_0116_t_values'] = t_values.detach() if t_values is not None else None

        # Partition MSE loss by t value (for TensorBoard visualization)
        # Training phase directly uses sampled t_values, no longer infers from inference schedule
        if t_values is not None and per_sample_mse is not None:
            t_values = t_values.clamp(0.0, 1.0)

            # Partition by t: [0, 0.3) high noise, [0.3, 0.6) medium noise, [0.6, 1] low noise
            T_LOW = 0.3  # t < 0.3: high noise stage
            T_HIGH = 0.6  # t >= 0.6: low noise stage

            high_noise_mask = t_values < T_LOW  # t ∈ [0, 0.3)
            mid_noise_mask = (t_values >= T_LOW) & (t_values < T_HIGH)  # t ∈ [0.3, 0.6)
            low_noise_mask = t_values >= T_HIGH  # t ∈ [0.6, 1.0]
            # Compute noise_weight (align with torchfold)
            noise_weights = (noise_levels ** 2 + sigma_data ** 2) / ((sigma_data * noise_levels) ** 2)
            eps = 1e-8
            # Compute MSE loss per t interval (unweighted + weighted by w(t) then normalized)
            if high_noise_mask.any():
                mse_hi = per_sample_mse[high_noise_mask]
                w_hi = noise_weights[high_noise_mask]
                result['mse_t_0_03'] = mse_hi.mean()
                result['mse_weighted_t_0_03'] = (mse_hi * w_hi).sum() / (w_hi.sum() + eps)
            if mid_noise_mask.any():
                mse_mid = per_sample_mse[mid_noise_mask]
                w_mid = noise_weights[mid_noise_mask]
                result['mse_t_03_06'] = mse_mid.mean()
                result['mse_weighted_t_03_06'] = (mse_mid * w_mid).sum() / (w_mid.sum() + eps)
            if low_noise_mask.any():
                mse_lo = per_sample_mse[low_noise_mask]
                w_lo = noise_weights[low_noise_mask]
                result['mse_t_06_1'] = mse_lo.mean()
                result['mse_weighted_t_06_1'] = (mse_lo * w_lo).sum() / (w_lo.sum() + eps)
        return result
