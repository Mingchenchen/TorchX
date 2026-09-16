"""Process Structure Data."""

import numpy as np
import torch
from typing import Optional, List
import random

from torchfold.processing.cropping import get_continues_crop_index, get_spatial_crop_index


def apply_token_cropping(
        features_tensor: dict,
        crop_size: int,
        *,
        sample_id: str = "unknown",
        crop_method_weights: Optional[List[float]] = None,
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
        interface_minimal_distance: int = 15,
        reference_chain_ids: Optional[List[int]] = None,
        queries_subset_size: int = 32,
        keys_subset_size: int = 128,
        average_num_atoms_per_token: int = 24,
        token_atoms_layout: object = None,
        max_templates: Optional[int] = None,
        remove_unresolved_tokens: bool = False,
        verbose: bool = False,
) -> None:
    """
    Perform contiguous cropping on `features_tensor` in place and remap/filter/pad
    the bond `GatherInfo` entries.

    Updates:
      - Crops Token/Template/MSA/RefStructure/Frames tensors
      - Adjusts both bond `gather_idxs`/`gather_mask`/`input_shape`
      - Recomputes AtomCrossAtt features
      - Updates `seq_length` and `crop_indices`

    New args:
      - queries_subset_size: subset size for queries in atom cross attention (default 32)
      - keys_subset_size: subset size for keys in atom cross attention (default 128)
      - average_num_atoms_per_token: mean atoms per token (default 24)
      - max_templates: maximum number of templates to keep (None = no limit, recommended: 4)
      - remove_unresolved_tokens: whether to filter out unresolved tokens before cropping (default False)
    """
    # Lazily import the heavy modules
    from torchfold.processing import features as torch_features

    tokens = features_tensor['aatype'].to(torch.long)  # [N]
    chain_id = features_tensor['asym_id'].to(torch.long)  # [N]
    centre_idx = features_tensor['residue_center_index'].to(torch.long)  # [N]
    N = int(tokens.shape[0])
    row_ix = torch.arange(N, device=centre_idx.device)
    ref_space_uid_token = features_tensor['ref_space_uid'][row_ix, centre_idx].to(torch.long)  # [N]
    # Compute the total number of atoms per UID (aggregated by UID)
    token_atom_counts = features_tensor['ref_mask'].to(torch.long).sum(dim=-1)  # [N]
    unique_uids, inverse_indices = torch.unique(ref_space_uid_token, return_inverse=True)
    per_uid_atom_sums = torch.zeros_like(unique_uids, dtype=token_atom_counts.dtype)
    per_uid_atom_sums = per_uid_atom_sums.scatter_add(0, inverse_indices, token_atom_counts)
    atom_sums = per_uid_atom_sums[inverse_indices]  # [N], aligned with ref_space_uid_token

    # ========== Filter unresolved tokens before cropping ==========
    resolved_indices = None  # Indices of resolved tokens in the original space
    if remove_unresolved_tokens:
        # Determine which tokens have at least one atom resolved
        atom_mask = features_tensor['true_positions_atom_mask']  # [N, 24]
        token_resolved_mask = atom_mask.sum(dim=-1) > 0  # [N], True if at least one atom is resolved
        resolved_indices = torch.where(token_resolved_mask)[0]  # Original indices of valid tokens

        num_unresolved = N - len(resolved_indices)
        if num_unresolved > 0:
            if verbose:
                print(
                    f"Info: Sample {sample_id}: Filtering out {num_unresolved} unresolved tokens before cropping (N={N} -> {len(resolved_indices)}).",
                    flush=True)

        if len(resolved_indices) == 0:
            if verbose:
                print(f"Error: Sample {sample_id}: All tokens are unresolved! Skipping cropping.", flush=True)
            return

        # Create filtered variables for cropping
        tokens_for_crop = tokens[resolved_indices]
        chain_id_for_crop = chain_id[resolved_indices]
        ref_space_uid_token_for_crop = ref_space_uid_token[resolved_indices]
        atom_sums_for_crop = atom_sums[resolved_indices]
        N_for_crop = len(resolved_indices)
    else:
        # No filtering, use original variables
        tokens_for_crop = tokens
        chain_id_for_crop = chain_id
        ref_space_uid_token_for_crop = ref_space_uid_token
        atom_sums_for_crop = atom_sums
        N_for_crop = N

    # Compute indices used for cropping
    if crop_method_weights is None:
        crop_method_weights = [0.34, 0.33, 0.33]
    assert len(crop_method_weights) == 3, "crop_method_weights must have length 3."
    cand_crop_methods = [
        "ContiguousCropping",
        "SpatialCropping",
        "SpatialInterfaceCropping",
    ]
    crop_method = random.choices(cand_crop_methods, k=1, weights=crop_method_weights)[0]

    def _do_contiguous_cropping():
        """Perform contiguous cropping"""
        return get_continues_crop_index(
            tokens=tokens_for_crop,
            chain_id=chain_id_for_crop,
            ref_space_uid_token=ref_space_uid_token_for_crop,
            atom_sums=atom_sums_for_crop,
            crop_size=int(crop_size),
            crop_complete_ligand_unstdRes=contiguous_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

    def _do_spatial_cropping(is_interface_crop: bool):
        """Perform spatial cropping, return None on failure"""
        try:
            # Spatial cropping uses filtered variables
            reference_mask_for_crop = torch.zeros_like(chain_id_for_crop, dtype=torch.bool)
            for ch in reference_chain_ids:
                reference_mask_for_crop |= chain_id_for_crop == ch
            token_indices_in_ref_for_crop = torch.nonzero(reference_mask_for_crop, as_tuple=False).flatten()

            if len(token_indices_in_ref_for_crop) == 0:
                # No tokens found for reference chain, return None to trigger fallback
                return None

            # Compute coordinates and distances in the filtered space
            if remove_unresolved_tokens and resolved_indices is not None:
                centre_idx_for_crop = centre_idx[resolved_indices]
                row_ix_for_crop = torch.arange(N_for_crop, device=centre_idx.device)
                # Get center coordinates of filtered tokens
                centre_coords_for_crop = features_tensor['true_positions'][resolved_indices]
                centre_coords_for_crop = centre_coords_for_crop[row_ix_for_crop, centre_idx_for_crop].to(torch.float32)
                # Get mask of filtered tokens
                token_dist_mask_1d_for_crop = torch.gather(
                    features_tensor['true_positions_atom_mask'][resolved_indices], 1, centre_idx_for_crop.unsqueeze(-1)
                ).squeeze(-1).to(torch.bool)
            else:
                centre_coords_for_crop = features_tensor['true_positions'][row_ix, centre_idx].to(torch.float32)
                token_dist_mask_1d_for_crop = torch.gather(
                    features_tensor['true_positions_atom_mask'], 1, centre_idx.unsqueeze(-1)
                ).squeeze(-1).to(torch.bool)

            token_distance = torch.cdist(
                centre_coords_for_crop[token_indices_in_ref_for_crop],
                centre_coords_for_crop,
            )
            token_distance_mask = (
                    token_dist_mask_1d_for_crop[token_indices_in_ref_for_crop].unsqueeze(1)
                    & token_dist_mask_1d_for_crop.unsqueeze(0)
            ).to(torch.float32)

            selected, _ = get_spatial_crop_index(
                tokens=tokens_for_crop,
                chain_id=chain_id_for_crop,
                token_distance=token_distance,
                token_distance_mask=token_distance_mask,
                reference_chain_id=reference_chain_ids,
                ref_space_uid_token=ref_space_uid_token_for_crop,
                crop_size=int(crop_size),
                crop_complete_ligand_unstdRes=spatial_crop_complete_lig,
                interface_crop=is_interface_crop,
                interface_minimal_distance=interface_minimal_distance,
            )
            return selected
        except (AssertionError, RuntimeError, IndexError) as e:
            # Spatial cropping failed, return None to trigger fallback
            if verbose:
                print(
                    f"Warning: Sample {sample_id}: Spatial cropping failed ({e}), will fallback to contiguous cropping.",
                    flush=True)
            return None

    if crop_method == "ContiguousCropping":
        selected_in_crop_space = _do_contiguous_cropping()
    else:
        # Try spatial cropping, fallback to contiguous cropping on failure
        selected_in_crop_space = _do_spatial_cropping(
            is_interface_crop=(crop_method == "SpatialInterfaceCropping")
        )
        if selected_in_crop_space is None:
            # Spatial cropping failed, fallback to contiguous cropping
            if verbose:
                print(f"Info: Sample {sample_id}: Falling back to ContiguousCropping.", flush=True)
            crop_method = "ContiguousCropping"
            selected_in_crop_space = _do_contiguous_cropping()

    # ========== Map cropping result back to the original index space ==========
    if remove_unresolved_tokens and resolved_indices is not None:
        # selected_in_crop_space are indices in the filtered space, need to map back to original space
        selected = resolved_indices[selected_in_crop_space]
        if verbose:
            print(
                f"Info: Sample {sample_id}: Mapped {len(selected_in_crop_space)} cropped tokens back to original space.",
                flush=True)
    else:
        selected = selected_in_crop_space

    # bonds GatherInfo: remap + filter + pad
    def _remap_filter_pad_bond_gather(prefix: str):
        idxs_key = f'{prefix}:gather_idxs'
        mask_key = f'{prefix}:gather_mask'
        shape_key = f'{prefix}:input_shape'
        if (idxs_key not in features_tensor) or (mask_key not in features_tensor) or (shape_key not in features_tensor):
            return
        gather_idxs = features_tensor[idxs_key]  # [num_pairs_orig, 2]
        gather_mask = features_tensor[mask_key]  # [num_pairs_orig, 2]
        input_shape_old = features_tensor[shape_key]
        num_pairs_orig = int(gather_idxs.shape[0]) if gather_idxs.ndim >= 2 else 0
        # Build the old->new index mapping
        token_idx_map = torch.full((N,), -1, dtype=torch.long, device=selected.device)
        token_idx_map.index_copy_(0, selected, torch.arange(len(selected), device=selected.device, dtype=torch.long))
        if num_pairs_orig > 0:
            row_valid = (gather_mask.to(torch.bool).sum(dim=1) == 2)
            valid_idxs = gather_idxs[row_valid].to(torch.long)  # [num_valid, 2]

            # Boundary check: filter out bonds with indices out of range
            in_bounds = (valid_idxs[:, 0] >= 0) & (valid_idxs[:, 0] < N) & \
                        (valid_idxs[:, 1] >= 0) & (valid_idxs[:, 1] < N)

            # if not in_bounds.all():
            #     num_out_of_bounds = (~in_bounds).sum().item()
            #     # print(f"Warning: Sample {sample_id}, prefix {prefix}: {num_out_of_bounds} bonds out of bounds (N={N}), filtering them out.", flush=True)

            valid_idxs = valid_idxs[in_bounds]

            if valid_idxs.shape[0] == 0:
                # No valid bonds, return empty
                # print(f"Warning: Sample {sample_id}, prefix {prefix}: No valid bonds remaining after boundary check.", flush=True)
                features_tensor[idxs_key] = torch.zeros((num_pairs_orig, 2), dtype=gather_idxs.dtype,
                                                        device=gather_idxs.device)
                features_tensor[mask_key] = torch.zeros((num_pairs_orig, 2), dtype=gather_mask.dtype,
                                                        device=gather_mask.device)
                features_tensor[shape_key] = torch.tensor([int(len(selected))], dtype=input_shape_old.dtype,
                                                          device=torch.device('cpu'))
                return

            remapped0 = token_idx_map[valid_idxs[:, 0]]
            remapped1 = token_idx_map[valid_idxs[:, 1]]
            remapped = torch.stack([remapped0, remapped1], dim=-1)  # [num_valid, 2]
            keep = (remapped[:, 0] >= 0) & (remapped[:, 1] >= 0)
            remapped = remapped[keep]
            out_dtype = gather_idxs.dtype
            out_device = gather_idxs.device
            out_idxs = torch.zeros((num_pairs_orig, 2), dtype=out_dtype, device=out_device)
            out_mask = torch.zeros((num_pairs_orig, 2), dtype=gather_mask.dtype, device=gather_mask.device)
            num_keep = int(remapped.shape[0])
            if num_keep > 0:
                fill_n = min(num_keep, num_pairs_orig)
                out_idxs[:fill_n] = remapped[:fill_n].to(dtype=out_dtype, device=out_device)
                out_mask[:fill_n] = 1.0
            features_tensor[idxs_key] = out_idxs
            features_tensor[mask_key] = out_mask
        # Update `input_shape` to match the cropped token count; keep it on CPU
        features_tensor[shape_key] = torch.tensor([int(len(selected))], dtype=input_shape_old.dtype,
                                                  device=torch.device('cpu'))

    _remap_filter_pad_bond_gather('tokens_to_polymer_ligand_bonds')
    _remap_filter_pad_bond_gather('tokens_to_ligand_ligand_bonds')

    # `token_atoms` GatherInfo: remap token indices while keeping atom indices
    def _remap_token_atoms_gather(prefix: str):
        """
        Crop `token_atoms` GatherInfo entries (for example `token_atoms_to_pseudo_beta`).

        Logic:
        - `input_shape = [num_tokens, max_atoms]`
        - `gather_idxs` has shape `[num_tokens]`, each value points to the flattened
          index: `token_idx * max_atoms + atom_idx`
        - Cropping keeps only tokens from `selected` and remaps `gather_idxs` to new
          flattened indices

        Example:
        Original: 6 tokens, `max_atoms=14`, `selected=[0, 2, 5]`
        - token 0: `gather_idxs[0] = 4` (0*14+4 → atom 4)
        - token 2: `gather_idxs[2] = 35` (2*14+7 → atom 7)
        - token 5: `gather_idxs[5] = 74` (5*14+4 → atom 4)

        After cropping: 3 tokens, `max_atoms=14`
        - new token 0 (old token 0): `gather_idxs_new[0] = 0*14+4 = 4`
        - new token 1 (old token 2): `gather_idxs_new[1] = 1*14+7 = 21`
        - new token 2 (old token 5): `gather_idxs_new[2] = 2*14+4 = 32`
        """
        idxs_key = f'{prefix}:gather_idxs'
        mask_key = f'{prefix}:gather_mask'
        shape_key = f'{prefix}:input_shape'
        if (idxs_key not in features_tensor) or (mask_key not in features_tensor) or (shape_key not in features_tensor):
            return

        gather_idxs_old = features_tensor[idxs_key]  # [N_old]
        gather_mask_old = features_tensor[mask_key]  # [N_old]
        input_shape_old = features_tensor[shape_key]  # [2] = [N_old, max_atoms]

        # Parse `input_shape`
        if input_shape_old.ndim == 0 or len(input_shape_old) < 2:
            return  # Invalid format, skip
        N_old = int(input_shape_old[0])
        max_atoms = int(input_shape_old[1])
        N_new = len(selected)

        # Validate shapes for consistency
        if gather_idxs_old.shape[0] != N_old:
            return  # `gather_idxs` length mismatched with `input_shape`, skip
        if len(selected) > 0 and (selected.max() >= N_old or selected.min() < 0):
            return  # `selected` indices out of range, skip

        # Build old->new token mapping (based on `N_old`, not the outer `N`)
        token_idx_map = torch.full((N_old,), -1, dtype=torch.long, device=selected.device)
        token_idx_map.index_copy_(0, selected, torch.arange(N_new, device=selected.device, dtype=torch.long))

        # Keep selected tokens and remap `gather_idxs`
        gather_idxs_selected = gather_idxs_old[selected]  # [N_new]
        gather_mask_selected = gather_mask_old[selected]  # [N_new]

        # Compute the new flattened index per kept token
        #   flat_old = old_token_idx * max_atoms + atom_idx
        #   flat_new = new_token_idx * max_atoms + atom_idx
        # where `atom_idx = flat_old % max_atoms` and
        # `new_token_idx = token_idx_map[old_token_idx]`
        atom_indices = gather_idxs_selected % max_atoms  # [N_new], per-token atom index
        new_token_indices = token_idx_map[selected]  # [N_new], new token positions
        gather_idxs_new = new_token_indices * max_atoms + atom_indices  # [N_new]

        # Update `features_tensor`
        features_tensor[idxs_key] = gather_idxs_new
        features_tensor[mask_key] = gather_mask_selected
        features_tensor[shape_key] = torch.tensor([N_new, max_atoms], dtype=input_shape_old.dtype,
                                                  device=torch.device('cpu'))

    _remap_token_atoms_gather('token_atoms_to_pseudo_beta')

    def _remap_token_atoms_to_polymer_ligand_bonds():
        """
        Crop `token_atoms_to_polymer_ligand_bonds` GatherInfo entries.

        Logic:
        - `input_shape = [num_tokens, max_atoms]`
        - `gather_idxs` has shape `[num_tokens, 2]`, each row represents a bond with two atom indices
          (flattened indices: `token_idx * max_atoms + atom_idx`)
        - `gather_mask` has shape `[num_tokens, 2]`, each row indicates if a bond is valid
        - Only bonds where both atoms' tokens are in `selected` are kept
        - Atom indices are remapped to the new token layout

        Example:
        Original: 10 tokens, `max_atoms=24`, `selected=[1, 3, 5, 7]`
        - bond 0: `gather_idxs[0] = [29, 80]` (token 1 atom 5, token 3 atom 8) → kept
        - bond 1: `gather_idxs[1] = [80, 108]` (token 3 atom 8, token 4 atom 12) → discarded (token 4 not in selected)

        After cropping: 4 tokens, `max_atoms=24`
        - new bond 0: `gather_idxs_new[0] = [5, 32]` (new token 0 atom 5, new token 1 atom 8)
        """
        idxs_key = 'token_atoms_to_polymer_ligand_bonds:gather_idxs'
        mask_key = 'token_atoms_to_polymer_ligand_bonds:gather_mask'
        shape_key = 'token_atoms_to_polymer_ligand_bonds:input_shape'

        # Check if keys exist
        if (idxs_key not in features_tensor) or (mask_key not in features_tensor) or (shape_key not in features_tensor):
            return

        gather_idxs_old = features_tensor[idxs_key]  # [N_old, 2]
        gather_mask_old = features_tensor[mask_key]  # [N_old, 2]
        input_shape_old = features_tensor[shape_key]  # [2] = [N_old, max_atoms]

        # Check input_shape format
        if input_shape_old.ndim == 0 or len(input_shape_old) < 2:
            return

        N_old = int(input_shape_old[0])
        max_atoms = int(input_shape_old[1])
        N_new = len(selected)

        # Boundary check: ensure selected indices are within valid range
        if N_new == 0:
            # print(f"Warning: Sample {sample_id}, _remap_token_atoms_to_polymer_ligand_bonds: N_new is 0.", flush=True)
            return
        if selected.max() >= N_old or selected.min() < 0:
            # selected indices out of bounds, skip this function
            # print(f"Error: Sample {sample_id}, _remap_token_atoms_to_polymer_ligand_bonds: selected indices out of bounds (N_old={N_old}, min={selected.min().item()}, max={selected.max().item()}). Skipping bond remapping.", flush=True)
            return

        # Build old->new token mapping
        token_idx_map = torch.full((N_old,), -1, dtype=torch.long, device=selected.device)
        token_idx_map.index_copy_(0, selected, torch.arange(N_new, device=selected.device, dtype=torch.long))

        # Identify valid bonds (rows where both mask values are True)
        valid_bonds_mask = gather_mask_old.to(torch.bool).all(dim=1)  # [N_old]
        valid_bond_indices = torch.where(valid_bonds_mask)[0]  # [num_valid_bonds]

        # Filter and remap bonds
        kept_bonds = []
        gather_idxs_long = gather_idxs_old.to(torch.long)

        for bond_idx_tensor in valid_bond_indices:
            bond_idx = int(bond_idx_tensor.item())
            atom_idx_0 = gather_idxs_long[bond_idx, 0]
            atom_idx_1 = gather_idxs_long[bond_idx, 1]

            # Extract token indices from flattened atom indices
            token_0 = atom_idx_0 // max_atoms
            token_1 = atom_idx_1 // max_atoms

            # Boundary check: skip bonds with indices out of range
            if token_0 < 0 or token_0 >= N_old or token_1 < 0 or token_1 >= N_old:
                if verbose:
                    print(
                        f"Warning: Sample {sample_id}, _remap_token_atoms_to_polymer_ligand_bonds: Bond atom indices point to out-of-bounds tokens (token_0={token_0}, token_1={token_1}, N_old={N_old}). Skipping this bond.",
                        flush=True)
                continue

            # Map to new token indices
            new_token_0 = token_idx_map[token_0]
            new_token_1 = token_idx_map[token_1]

            # Only keep bonds where both tokens are in selected
            if new_token_0 >= 0 and new_token_1 >= 0:
                # Extract local atom indices within their tokens
                atom_0_local = atom_idx_0 % max_atoms
                atom_1_local = atom_idx_1 % max_atoms

                # Compute new flattened atom indices
                new_atom_idx_0 = new_token_0 * max_atoms + atom_0_local
                new_atom_idx_1 = new_token_1 * max_atoms + atom_1_local

                kept_bonds.append([int(new_atom_idx_0.item()), int(new_atom_idx_1.item())])

        # Reassemble and pad to [N_new, 2]
        num_kept_bonds = len(kept_bonds)
        out_dtype = gather_idxs_old.dtype
        out_device = gather_idxs_old.device

        gather_idxs_new = torch.zeros((N_new, 2), dtype=out_dtype, device=out_device)
        gather_mask_new = torch.zeros((N_new, 2), dtype=gather_mask_old.dtype, device=gather_mask_old.device)

        if num_kept_bonds > 0:
            kept_bonds_tensor = torch.tensor(kept_bonds, dtype=out_dtype, device=out_device)
            fill_n = min(num_kept_bonds, N_new)
            gather_idxs_new[:fill_n] = kept_bonds_tensor[:fill_n]
            gather_mask_new[:fill_n] = 1.0

        # Update features
        features_tensor[idxs_key] = gather_idxs_new
        features_tensor[mask_key] = gather_mask_new
        features_tensor[shape_key] = torch.tensor([N_new, max_atoms], dtype=input_shape_old.dtype,
                                                  device=torch.device('cpu'))

    _remap_token_atoms_to_polymer_ligand_bonds()

    # Keys that require special handling
    token_feature_keys = {
        'residue_index', 'token_index', 'aatype', 'seq_mask',
        'entity_id', 'asym_id', 'sym_id',
        'is_protein', 'is_rna', 'is_dna', 'is_ligand',
        'is_nonstandard_polymer_chain', 'is_water',
    }
    msa_keys_2d = {'msa', 'msa_mask', 'deletion_matrix'}
    msa_keys_1d = {'deletion_mean'}
    msa_keys_profile = {'profile'}
    template_keys = {
        'template_aatype',
        'template_atom_positions',
        'template_atom_mask',
    }
    ref_structure_keys = {
        'ref_pos',
        'ref_mask',
        'ref_element',
        'ref_charge',
        'ref_atom_name_chars',
        'ref_space_uid',
        'true_positions',
        'true_positions_atom_mask',
        'pred_dense_atom_mask',
        'residue_center_index',
    }
    frames_keys = {'frames_mask'}

    def _crop_first_dim(t: torch.Tensor) -> torch.Tensor:
        if t.ndim >= 1 and t.shape[0] == N:
            return t.index_select(0, selected)
        return t

    def _crop_second_dim(t: torch.Tensor) -> torch.Tensor:
        if t.ndim >= 2 and t.shape[1] == N:
            return t.index_select(1, selected)
        return t

    # ========== Token dimension cropping ==========
    # Helper to crop template count
    def _crop_template_count(t: torch.Tensor, max_count: int) -> torch.Tensor:
        if t.ndim >= 1 and t.shape[0] > max_count:
            return t[:max_count]
        return t

    for key in list(features_tensor.keys()):
        value = features_tensor[key]
        if key in token_feature_keys:
            features_tensor[key] = _crop_first_dim(value)
        elif key in msa_keys_2d:
            features_tensor[key] = _crop_second_dim(value)
        elif key in msa_keys_1d:
            features_tensor[key] = _crop_first_dim(value)
        elif key in msa_keys_profile:
            features_tensor[key] = _crop_first_dim(value)
        elif key in template_keys:
            cropped = _crop_second_dim(value)
            if max_templates is not None:
                cropped = _crop_template_count(cropped, max_templates)
            features_tensor[key] = cropped
        elif key in ref_structure_keys:
            features_tensor[key] = _crop_first_dim(value)
        elif key in frames_keys:
            features_tensor[key] = _crop_first_dim(value)
        elif key == 'seq_length':
            features_tensor[key] = torch.tensor(
                len(selected),
                dtype=value.dtype,
                device=value.device
            )
        elif key == 'num_alignments':
            pass

    features_tensor['crop_indices'] = selected

    # ========== Recompute AtomCrossAtt features ==========
    # 1. Rebuild `all_token_atoms_layout` from `features_tensor`
    #    Note: `token_atoms_layout` should have been stored as an object
    if token_atoms_layout:
        # Extract from the stored object
        token_atoms_layout_obj = token_atoms_layout
        if isinstance(token_atoms_layout_obj, np.ndarray) and token_atoms_layout_obj.dtype == object:
            all_token_atoms_layout_full = token_atoms_layout_obj.item()
        else:
            all_token_atoms_layout_full = token_atoms_layout_obj

        # Crop `all_token_atoms_layout` to the selected tokens
        # Shape: (num_tokens, max_atoms_per_token); keep only `selected`
        selected_np = selected.cpu().numpy()
        all_token_atoms_layout_cropped = all_token_atoms_layout_full[selected_np, :]
    else:
        # Without `token_atoms_layout`, we have to rebuild it from ref_* features,
        # which requires atom names per token and related metadata.
        if verbose:
            print("Warning: 'token_atoms_layout' missing in features_tensor; cannot recompute AtomCrossAtt features.",
                  flush=True)
            print("Ensure the pickle file stores the 'token_atoms_layout' field.", flush=True)
        return
    # 2. Create new `PaddingShapes`
    N_cropped = len(selected)
    # Compute `num_atoms` based on ACTUAL atom count from cropped ref_mask
    # This prevents GatherInfo indices from exceeding tensor dimensions
    actual_atom_count = int(features_tensor['ref_mask'].sum().item())
    # Also consider the estimate as a fallback
    estimated_atom_count = N_cropped * average_num_atoms_per_token
    # Use the maximum to ensure enough space
    num_atoms = max(actual_atom_count, estimated_atom_count)
    num_atoms = int(np.ceil(num_atoms / queries_subset_size) * queries_subset_size)

    # Infer the remaining padding parameters from existing features (after cropping)
    msa_size = features_tensor['msa'].shape[0] if 'msa' in features_tensor else 16384
    num_templates = features_tensor['template_aatype'].shape[0] if 'template_aatype' in features_tensor else 4

    # Apply limits if specified
    if max_templates is not None and num_templates > max_templates:
        num_templates = max_templates

    padding_shapes = torch_features.PaddingShapes(
        num_tokens=N_cropped,
        msa_size=msa_size,
        num_chains=1000,  # Default value
        num_templates=num_templates,
        num_atoms=num_atoms,
    )
    # 3. Recompute via `AtomCrossAtt.compute_features`
    atom_cross_att = torch_features.AtomCrossAtt.compute_features(
        all_token_atoms_layout=all_token_atoms_layout_cropped,
        queries_subset_size=queries_subset_size,
        keys_subset_size=keys_subset_size,
        padding_shapes=padding_shapes,
    )

    # 4. Convert back to a dict and write into `features_tensor`
    atom_cross_att_dict = atom_cross_att.as_data_dict()

    # Convert numpy arrays to torch tensors
    for key, value in atom_cross_att_dict.items():
        if isinstance(value, np.ndarray):
            features_tensor[key] = torch.from_numpy(value)
        else:
            features_tensor[key] = value

    # 5. Validate GatherInfo indices to prevent index out of bounds
    def _validate_gather_info(prefix: str, max_valid_idx: int):
        """Check that gather_idxs values don't exceed max_valid_idx."""
        idxs_key = f'{prefix}:gather_idxs'
        mask_key = f'{prefix}:gather_mask'
        if idxs_key not in features_tensor or mask_key not in features_tensor:
            return
        gather_idxs = features_tensor[idxs_key]
        gather_mask = features_tensor[mask_key]
        if gather_idxs.numel() == 0:
            return
        # Only check valid (masked) entries
        valid_mask = gather_mask.to(torch.bool)
        if valid_mask.any():
            valid_idxs = gather_idxs[valid_mask]
            max_idx = int(valid_idxs.max().item()) if valid_idxs.numel() > 0 else -1
            min_idx = int(valid_idxs.min().item()) if valid_idxs.numel() > 0 else 0
            if max_idx >= max_valid_idx or min_idx < 0:
                if verbose:
                    print(f"Warning: Sample {sample_id}, {prefix}: gather_idxs out of bounds! "
                          f"range=[{min_idx}, {max_idx}], valid=[0, {max_valid_idx - 1}]", flush=True)

    # Validate key GatherInfo entries
    _validate_gather_info('token_atoms_to_queries', N_cropped * 24)  # max atoms per token = 24
    _validate_gather_info('queries_to_token_atoms', N_cropped * 24)

    # print(f"Sample: {sample_id}, Crop method: {crop_method}, cropped tokens: {N_cropped}, num_atoms: {num_atoms}", flush=True)
