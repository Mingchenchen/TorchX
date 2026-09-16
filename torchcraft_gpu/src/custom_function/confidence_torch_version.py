from typing import Dict, List, Union

import torch


def compute_confidence_metrics(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    featurised_example: Dict[str, torch.Tensor],
    device: torch.device,
    hotspot_indices: List[int] = None,
    negative_hotspot_indices: List[int] = None,
    design_positions_in_chain: List[int] = None,
    framework_negative_binder_indices: List[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Calculates four key confidence metrics: predicted_distance_error, ptm, iptm, plddt, while preserving gradient flow
    Also includes PAE loss, contact loss, and helix loss
    
    Args:
        result: Output from the model forward pass
        featurised_example: Featurized input example
        device: Computation device
    
    Returns:
        Loss dictionary containing:
        - ptm, iptm: Protein folding quality metrics
        - predicted_distance_error: Predicted distance error
        - plddt: Local confidence score
        - intra_binder_contact_loss: Intra-binder contact loss
        - interface_contact_loss: Target-binder interface contact loss
        - pae_loss: Predicted Aligned Error loss
        - helix_loss: Helix structure loss (encourages binder to form alpha-helices)
    """
    # Get necessary inputs
    num_tokens = featurised_example['seq_length'].item()
    asym_id = featurised_example['asym_id'][:num_tokens]
    
    # Calculate target_len and binder_len from featurised_example
    unique_asym_ids = torch.unique(asym_id)
    max_asym_id = unique_asym_ids.max()
    binder_len = (asym_id == max_asym_id).sum().item()
    target_len = num_tokens - binder_len
    
    # get residue_index
    if 'residue_index' in featurised_example:
        residue_index = featurised_example['residue_index'][:num_tokens]
    else:
        residue_index = torch.arange(num_tokens, device=device)
    
    # Calculate pae_single_mask
    frames_mask = featurised_example['frames_mask']
    pae_single_mask = torch.tile(
        frames_mask.unsqueeze(1),
        [1, frames_mask.shape[0]],
    )
    pae_single_mask = pae_single_mask.to(device=device)
    
    # Calculate ptm and iptm(total)
    ptm = compute_ptm_torch(
        result=result,
        num_tokens=num_tokens,
        asym_id=asym_id,
        pae_single_mask=pae_single_mask,
        interface=False,
    )
    iptm = compute_ptm_torch(
        result=result,
        num_tokens=num_tokens,
        asym_id=asym_id,
        pae_single_mask=pae_single_mask,
        interface=True,
    )


    chain_pair_iptm = compute_chain_pair_iptm_torch(
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
        asym_id=asym_id,
        pair_mask=pae_single_mask,
        num_tokens=num_tokens,
    )


    # Calculate pLDDT
    plddt_results = compute_plddt_torch(
        result=result,
        num_tokens=num_tokens,
        asym_id=asym_id
    )
    predicted_distance_errors = result['average_pde']
    
    # Calculate contact loss
    contact_losses = get_contact_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device,
        hotspot_indices=hotspot_indices,  # Pass parameters
        design_positions_in_chain=design_positions_in_chain  # Pass parameters
    )
    
    # Calculate negative contact loss
    negative_contact_loss = get_negative_contact_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device,
        negative_hotspot_indices=negative_hotspot_indices,  # Pass parameters
    )

    # Calculate negative_framework_contact_loss
    negative_framework_contact_loss = get_negative_framework_contact_loss_torch(
 	    result=result,
 	    num_tokens=num_tokens,
 	    target_len=target_len,
 	    binder_len=binder_len,
 	    residue_index=residue_index,
 	    device=device,
 	    design_positions_in_chain=design_positions_in_chain,  # Pass parameters
 	    framework_negative_binder_indices=framework_negative_binder_indices,
 	)

    paratope_loss, l_cdr, l_fw, l_cdr_target = get_paratope_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device,
        hotspot_indices=hotspot_indices,
        design_positions_in_chain=design_positions_in_chain,
    )

    # Calculate three types of PAE loss
    global_pae_loss = get_pae_loss_torch(
        result=result,
        num_tokens=num_tokens,
        use_interface=False
    )
    
    binder_pae_loss = get_binder_pae_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        device=device
    )
    
    binder_target_interface_pae_loss = get_binder_target_interface_pae_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        device=device,
        hotspot_indices=hotspot_indices
    )
    
    # Calculate helix loss (for binder chain)
    helix_loss = get_helix_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device
    )
    
    metrics_to_return = {
        'ptm': ptm,
        'iptm': iptm,
        'predicted_distance_error': predicted_distance_errors,
        'plddt': plddt_results['total_mean'],
        'intra_binder_contact_loss': contact_losses['binder_loss'],
        'interface_contact_loss': contact_losses['interface_loss'],
        'pae_loss': global_pae_loss, 
        'binder_pae_loss': binder_pae_loss, 
        'binder_target_interface_pae_loss': binder_target_interface_pae_loss,
        'helix_loss': helix_loss,
        'negative_contact_loss': negative_contact_loss,
        'negative_framework_contact_loss': negative_framework_contact_loss,
        'paratope_loss': paratope_loss,
        'paratope_cdr_loss': l_cdr,
        'paratope_fw_loss': l_fw,
        'paratope_cdr_target_loss': l_cdr_target,
    }
    

    chain_ids = torch.unique(asym_id[:num_tokens])
    # chain_pair_iptm shap: [num_samples, num_chains, num_chains]
    # After taking diagonal: [num_samples, num_chains]
    chain_ptm_diag = chain_pair_iptm.diagonal(dim1=-2, dim2=-1)  # [num_samples, num_chains]
    
    # Keep metrics only for the last chain (binder)
    binder_chain_id = chain_ids[-1]  # ID of the last chain
    binder_idx = len(chain_ids) - 1  # Index of binder in chain_ids
    metrics_to_return['ptm_binder'] = chain_ptm_diag[:, binder_idx]
    
    # Add binder-specific plddt metric only
    binder_chain_id_int = int(binder_chain_id)
    if binder_chain_id_int in plddt_results['per_chain']:
        metrics_to_return['plddt_binder'] = plddt_results['per_chain'][binder_chain_id_int]

    return metrics_to_return


def chain_pairwise_predicted_tm_scores_torch(
    tm_adjusted_pae: torch.Tensor,
    pair_mask: torch.Tensor,
    asym_id: torch.Tensor,
) -> torch.Tensor:
    """
    Computes pTM/ipTM between each pair of chains (pure PyTorch implementation, maintains differentiability).
    Returns a matrix with shape [num_chains, num_chains].
    """
    device = tm_adjusted_pae.device
    tm_adjusted_pae = tm_adjusted_pae.to(dtype=torch.float32)
    pair_mask = pair_mask.to(dtype=torch.bool, device=device)
    asym_id = asym_id.to(device=device)

    unique_chains, inverse_indices = torch.unique(asym_id, return_inverse=True)
    num_chains = unique_chains.shape[0]
    chain_scores = tm_adjusted_pae.new_zeros((num_chains, num_chains))

    for i in range(num_chains):
        chain_i_mask = inverse_indices == i
        if not torch.any(chain_i_mask):
            continue
        for j in range(i, num_chains):
            chain_j_mask = inverse_indices == j
            if not torch.any(chain_j_mask):
                continue

            combined_mask = chain_i_mask | chain_j_mask
            indices = torch.nonzero(combined_mask, as_tuple=False).squeeze(-1)

            sub_tm = tm_adjusted_pae.index_select(0, indices).index_select(1, indices)
            sub_pair_mask = pair_mask.index_select(0, indices).index_select(1, indices)
            sub_asym = asym_id.index_select(0, indices)

            score = predicted_tm_score_torch(
                tm_adjusted_pae=sub_tm,
                pair_mask=sub_pair_mask,
                asym_id=sub_asym,
                interface=i != j,
            )

            chain_scores[i, j] = score
            chain_scores[j, i] = score

    return chain_scores


def compute_chain_pair_iptm_torch(
    tm_adjusted_pae: Union[torch.Tensor, List[torch.Tensor]],
    asym_id: torch.Tensor,
    pair_mask: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """
    PyTorch-based chain pair ipTM calculation, preserving gradients for each sample.
    """
    if isinstance(tm_adjusted_pae, torch.Tensor):
        if tm_adjusted_pae.dim() == 2:
            pae_samples = (tm_adjusted_pae,)
        elif tm_adjusted_pae.dim() == 3:
            pae_samples = tuple(tm_adjusted_pae.unbind(0))
        else:
            raise ValueError(
                f'Unsupported tmscore_adjusted_pae tensor shape: {tm_adjusted_pae.shape}'
            )
    elif isinstance(tm_adjusted_pae, list):
        pae_samples = tuple(tm_adjusted_pae)
    else:
        raise TypeError(
            'tm_adjusted_pae must be a torch.Tensor or a list of torch.Tensor.'
        )

    trimmed_pair_mask = pair_mask[:num_tokens, :num_tokens]
    trimmed_asym = asym_id[:num_tokens]

    return torch.stack(
        [
            chain_pairwise_predicted_tm_scores_torch(
                tm_adjusted_pae=sample_tm.to(dtype=torch.float32)[
                    :num_tokens, :num_tokens
                ],
                pair_mask=trimmed_pair_mask,
                asym_id=trimmed_asym,
            )
            for sample_tm in pae_samples
        ],
        dim=0,
    )
    
def compute_confidence_metrics_simple(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    featurised_example: Dict[str, torch.Tensor],
    device: torch.device,
    hotspot_indices: List[int] = None,
    negative_hotspot_indices: List[int] = None,
    design_positions_in_chain: List[int] = None,
    framework_negative_binder_indices: List[int] = None,
) -> torch.Tensor:
    """
    Calculates four key confidence metrics: predicted_distance_error, ptm, iptm, plddt, while preserving gradient flow.
    Also includes PAE loss, contact loss, and helix loss.
    
    Args:
        result: Output from the model forward pass
        featurised_example: Featurized input example
        device: Computation device
    
    Returns:
        Loss dictionary containing:
        - ptm, iptm: Protein folding quality metrics
        - predicted_distance_error: Predicted distance error
        - plddt: Local confidence score
        - intro_binder_contact_loss: Intra-binder contact loss
        - interface_contact_loss: Target-binder interface contact loss
        - pae_loss: Predicted Aligned Error loss
        - helix_loss: Helix structure loss (encourages binder to form alpha-helices)
    """
    # Get necessary inputs
    num_tokens = featurised_example['seq_length'].item()
    asym_id = featurised_example['asym_id'][:num_tokens]
    
    # Calculate target_len and binder_len from featurised_example
    unique_asym_ids = torch.unique(asym_id)
    max_asym_id = unique_asym_ids.max()
    binder_len = (asym_id == max_asym_id).sum().item()
    target_len = num_tokens - binder_len
    
    # Get residue_index
    if 'residue_index' in featurised_example:
        residue_index = featurised_example['residue_index'][:num_tokens]
    else:
        residue_index = torch.arange(num_tokens, device=device)
    
    
    # Calculate contact loss
    contact_losses = get_contact_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device,
        hotspot_indices=hotspot_indices,  # Pass parameters
        design_positions_in_chain=design_positions_in_chain  # Pass parameters
    )

    # Calculate negative contact loss
    negative_contact_loss = get_negative_contact_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device,
        negative_hotspot_indices=negative_hotspot_indices,  # Pass parameters
    )
    
    # Calculate negative_framework_contact_loss
    negative_framework_contact_loss = get_negative_framework_contact_loss_torch(
 	    result=result,
 	    num_tokens=num_tokens,
 	    target_len=target_len,
 	    binder_len=binder_len,
 	    residue_index=residue_index,
 	    device=device,
 	    design_positions_in_chain=design_positions_in_chain,  # Pass parameters
 	    framework_negative_binder_indices=framework_negative_binder_indices,
 	)

    paratope_loss, l_cdr, l_fw, l_cdr_target = get_paratope_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device,
        hotspot_indices=hotspot_indices,
        design_positions_in_chain=design_positions_in_chain,
    )
    

    
    # Calculate helix loss (for binder chain)
    helix_loss = get_helix_loss_torch(
        result=result,
        num_tokens=num_tokens,
        target_len=target_len,
        binder_len=binder_len,
        residue_index=residue_index,
        device=device
    )
    
    return {
        'intra_binder_contact_loss': contact_losses['binder_loss'],
        'interface_contact_loss': contact_losses['interface_loss'],
        'helix_loss': helix_loss,
        'negative_contact_loss': negative_contact_loss,
        'negative_framework_contact_loss': negative_framework_contact_loss,
        'paratope_loss': paratope_loss,
        'paratope_cdr_loss': l_cdr,
        'paratope_fw_loss': l_fw,
        'paratope_cdr_target_loss': l_cdr_target,
    }

def compute_ptm_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    asym_id: torch.Tensor,
    pae_single_mask: torch.Tensor,
    interface: bool,
) -> torch.Tensor:
    """
    PyTorch version: Computes the pTM metric
    
    Args:
        result: Output from the model forward pass
        num_tokens: Number of tokens
        asym_id: Chain ID
        pae_single_mask: PAE single mask
        interface: Whether to compute interface pTM
    
    Returns:
        ptm: Calculated pTM metric
    """
    return torch.stack(
        [
            predicted_tm_score_torch(
                tm_adjusted_pae=tm_adjusted_pae[:num_tokens, :num_tokens],
                asym_id=asym_id,
                pair_mask=pae_single_mask[:num_tokens, :num_tokens],
                interface=interface,
            )
            for tm_adjusted_pae in result['tmscore_adjusted_pae_global']
        ],
        dim=0,
    )

def predicted_tm_score_torch(
    tm_adjusted_pae: torch.Tensor,
    pair_mask: torch.Tensor,
    asym_id: torch.Tensor,
    interface: bool = False,
) -> torch.Tensor:
    """
    PyTorch version: Computes the predicted TM score
    
    Args:
        tm_adjusted_pae: Adjusted PAE matrix
        pair_mask: Pair mask
        asym_id: chain ID
        interface: Whether to compute interface TM score
    
    Returns:
        tm_score: Calculated TM score
    """
    # Ensure consistent data type and device
    device = tm_adjusted_pae.device
    tm_adjusted_pae = tm_adjusted_pae.to(dtype=torch.float32)
    pair_mask = pair_mask.to(dtype=torch.bool, device=device)
    asym_id = asym_id.to(device=device)
    
    # Create pair mask
    if interface:
        pair_mask = pair_mask * (asym_id.unsqueeze(1) != asym_id.unsqueeze(0))
    
    # Handle case with no valid frames
    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    # Calculate normalized residue mask
    normed_residue_mask = pair_mask.float() / (
        1e-8 + torch.sum(pair_mask.float(), dim=-1, keepdim=True)
    )
    
    # Calculate score per alignment
    per_alignment = torch.sum(tm_adjusted_pae * normed_residue_mask, dim=-1)
    
    # Return maximum value
    return per_alignment.max()

def compute_plddt_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    asym_id: torch.Tensor
) -> Dict[str, Union[torch.Tensor, Dict[int, torch.Tensor]]]:
    """
    PyTorch version: Computes the pLDDT metric, including global average and per-chain averages.
    
    Args:
        result: Output from the model forward pass
        num_tokens: Number of tokens
        asym_id: Chain ID tensor
    
    Returns:
        A dictionary containing:
        - 'total_mean': Global average pLDDT score
        - 'per_chain': A dictionary containing average pLDDT score for each chain
    """
    try:
        # Get predicted_lddt from result
        predicted_lddt = result['predicted_lddt']
        if isinstance(predicted_lddt, list):
            predicted_lddt = predicted_lddt[0]

        # Get mask for filtering invalid atoms and padding
        mask = result['diffusion_samples']['mask']
        if isinstance(mask, list):
            mask = mask[0]

        # Ensure consistent data type and device
        predicted_lddt = predicted_lddt.to(dtype=torch.float32)
        mask = mask.to(dtype=torch.bool, device=predicted_lddt.device)
        asym_id = asym_id.to(device=predicted_lddt.device)

        # Truncate to the valid number of tokens (assuming batch size is the first dimension)
        predicted_lddt = predicted_lddt[:, :num_tokens]
        mask = mask[:, :num_tokens]

        # --- 1. Compute global pLDDT mean (calculated for each batch separately) ---
        # predicted_lddt shape: [batch_size, num_tokens, num_atoms]
        # Need to sum over token and atom dimensions, keeping the batch dimension
        masked_plddt_total = torch.where(mask, predicted_lddt, 0.0)
        # Sum over the last two dimensions (token and atom), keeping the batch dimension
        # If predicted_lddt is [2, 45, 24], sum results in [2]
        if len(predicted_lddt.shape) == 3:
            # 3D: [batch, tokens, atoms] -> sum over dim 1 and 2
            total_sum = masked_plddt_total.sum(dim=(1, 2))  # [batch_size]
            mask_sum = mask.sum(dim=(1, 2))  # [batch_size]
        else:
            # 2D: [batch, tokens] -> sum over dim 1
            total_sum = masked_plddt_total.sum(dim=1)  # [batch_size]
            mask_sum = mask.sum(dim=1)  # [batch_size]
        total_mean = total_sum / (mask_sum + 1e-8)  # [batch_size]

        # --- 2. Compute per-chain pLDDT mean (calculated for each batch separately) ---
        per_chain_means = {}
        unique_chain_ids = torch.unique(asym_id)

        for chain_id in unique_chain_ids:
            # Create mask for the current chain, shape must align with the token dimension
            # asym_id: [num_tokens] -> chain_mask needs to be broadcast to [batch_size, num_tokens, num_atoms]
            if len(predicted_lddt.shape) == 3:
                # 3D case: chain_mask needs to be [batch_size, num_tokens, num_atoms]
                chain_mask = (asym_id == chain_id).unsqueeze(0).unsqueeze(-1)  # [1, num_tokens, 1]
                chain_mask = chain_mask.expand(predicted_lddt.shape[0], -1, predicted_lddt.shape[2])  # [batch_size, num_tokens, num_atoms]
            else:
                # 2D case: chain_mask needs to be [batch_size, num_tokens]
                chain_mask = (asym_id == chain_id).unsqueeze(0)  # [1, num_tokens]
                chain_mask = chain_mask.expand(predicted_lddt.shape[0], -1)  # [batch_size, num_tokens]
            
            # Combine atom mask and chain mask
            final_mask_for_chain = mask & chain_mask
            
            # Calculate plddt using the final mask
            masked_plddt_chain = torch.where(final_mask_for_chain, predicted_lddt, 0.0)
            # Calculate the mean for each batch separately
            if len(predicted_lddt.shape) == 3:
                chain_sum = masked_plddt_chain.sum(dim=(1, 2))  # [batch_size]
                chain_mask_sum = final_mask_for_chain.sum(dim=(1, 2))  # [batch_size]
            else:
                chain_sum = masked_plddt_chain.sum(dim=1)  # [batch_size]
                chain_mask_sum = final_mask_for_chain.sum(dim=1)  # [batch_size]
            chain_mean = chain_sum / (chain_mask_sum + 1e-8)  # [batch_size]
            per_chain_means[int(chain_id)] = chain_mean

        return {
            'total_mean': total_mean,
            'per_chain': per_chain_means
        }
        
    except Exception as e:
        print(f"Error computing pLDDT: {e}")
        # Return a dictionary with the new structure containing error values
        return {
            'total_mean': torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True),
            'per_chain': {}
        }


def get_pae_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    mask_1d: torch.Tensor = None,
    mask_1b: torch.Tensor = None,
    mask_2d: torch.Tensor = None,
    use_interface: bool = False
) -> torch.Tensor:
    """
    PyTorch version: Computes PAE loss based on calculated PAE values, referencing get_pae_loss from ColabDesign
    
    Args:
        result: Output from the model forward pass
        num_tokens: Number of tokens
        mask_1d: 1D mask
        mask_1b: 1B mask
        mask_2d: 2D mask
        use_interface: Whether to use interface PAE (for binder design)
    
    Returns:
        pae_loss: PAE loss
    """
    try:
        # Select which PAE to use
        if use_interface and 'tmscore_adjusted_pae_interface' in result:
            # Use interface PAE (more suitable for binder design)
            pae_values = result['tmscore_adjusted_pae_interface']
        elif 'full_pae' in result:
            # Use full PAE
            pae_values = result['full_pae']
        else:
            print("Warning: No PAE data found in result")
            return torch.tensor(0.0, device='cuda', requires_grad=True)
        
        # If it's a list, take the first element
        if isinstance(pae_values, list):
            pae_values = pae_values[0]
        
        if pae_values.dim() == 3 and pae_values.shape[0] == 1:
            pae_values = pae_values.squeeze(0)
        
        # Truncate to the valid number of tokens
        pae_values = pae_values[:num_tokens, :num_tokens]
        
        # Ensure data type and device consistency
        device = pae_values.device
        pae_values = pae_values.to(dtype=torch.float32)
        
        # Normalize PAE values (similar to ColabDesign, divide by 31.0)
        # PAE values typically range from 0-31, so we use the same normalization
        p = pae_values / 31.0
        
        # Symmetrize (operation in ColabDesign)
        p = (p + p.T) / 2
        
        L = p.shape[0]
        
        # Set default masks
        if mask_1d is None:
            mask_1d = torch.ones(L, device=device, dtype=torch.bool)
        if mask_1b is None:
            mask_1b = torch.ones(L, device=device, dtype=torch.bool)
        if mask_2d is None:
            mask_2d = torch.ones((L, L), device=device, dtype=torch.bool)
        
        # Apply mask (referencing mask_2d calculation from ColabDesign)
        mask_2d = mask_2d * mask_1d.unsqueeze(1) * mask_1b.unsqueeze(0)
        
        # Calculate masked loss (referencing mask_loss function from ColabDesign)
        masked_pae = p * mask_2d.float()
        pae_loss = masked_pae.sum() / (mask_2d.float().sum() + 1e-8)
        
        return pae_loss
        
    except Exception as e:
        print(f"Error computing PAE loss: {e}")
        return torch.tensor(0.0, device='cuda', requires_grad=True)


def get_contact_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    target_len: int,
    binder_len: int,
    residue_index: torch.Tensor,
    device: torch.device,
    hotspot_indices: List[int] = None,
    design_positions_in_chain: List[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Contact Loss calculation function based on ColabDesign logic (PyTorch version)
    
    Args:
        result: Model prediction results containing distogram logits
        num_tokens: Total sequence length
        target_len: Length of the target protein
        binder_len: Length of the binder protein
        residue_index: Residue index tensor
        device: Computation device
        hotspot_indices: List of hotspot residue indices, None means using all target residues
    Returns:
        Dict containing:
            - binder_loss: Intra-binder contact loss
            - interface_loss: interface contact loss
    """
    from config.config import CON_OPT, I_CON_OPT
    
    # Use default configurations from the config file
    con_opt = CON_OPT
    i_con_opt = I_CON_OPT
    
    # Get distogram logits
    
    dgram_logits = result['distogram']['distogram_logits'][:num_tokens, :num_tokens]
    
    
    # Get distance bins
    dgram_bins = get_dgram_bins_torch(dgram_logits, device)
    
    # Calculate residue offset matrix
    residue_index_float = residue_index.float().to(device)
    offset = residue_index_float.unsqueeze(1) - residue_index_float.unsqueeze(0)
    
    # Create sequence mask
    seq_mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    
    # Create target and binder masks
    target_mask = torch.zeros_like(seq_mask)
    target_mask[:target_len] = True
    
    binder_mask = torch.zeros_like(seq_mask)
    binder_mask[-binder_len:] = True
    
    # Initialize loss variables
    binder_loss = torch.tensor(0.0, device=device)
    interface_loss = torch.tensor(0.0, device=device)
    

    # 1. Binder internal contact loss (if binder exists)
    if binder_len > 0:
        con_loss = get_con_loss_torch(
            dgram_logits, dgram_bins, offset, con_opt,
            mask_1d=binder_mask, mask_1b=binder_mask, device=device
        )
        binder_loss = con_loss
    
    # 2. Interface contact loss (target-binder)
    if target_len > 0 and binder_len > 0:

        # Use full-length binder by default
        binder_interface_mask = binder_mask 

        # If design sites (CDR) are provided, only calculate Loss for these sites
        if design_positions_in_chain is not None and len(design_positions_in_chain) > 0:
            # Create a mask with all False values
            temp_mask = torch.zeros_like(binder_mask)

            # Convert indices: design_positions_in_chain is relative to the start of Binder (0, 1...)
            valid_indices = [
                target_len + pos 
                for pos in design_positions_in_chain 
                if 0 <= (target_len + pos) < num_tokens
            ]

            if valid_indices:
                temp_mask[valid_indices] = True
                binder_interface_mask = temp_mask
            else:
                print(f"[WARNING] Design positions are empty after mapping! Fallback to using the entire Binder for Loss calculation.")

            print(f"\n[DEBUG Contact Loss] Calculating Interface Loss Mask...")
            print(f"  - Target Length: {target_len}, Binder Length: {binder_len}")
            print(f"  - Input Design Positions (relative indices, first 50): {design_positions_in_chain[:50]} ...")
            print(f"  - Converted Global Indices (absolute indices, first 50): {valid_indices[:50]} ...")
            print(f"  - Number of Binder residues used for final Loss calculation: {len(valid_indices)}\n")


        if hotspot_indices is not None and len(hotspot_indices) > 0:  # Modified here
            # Use hotspot residues
            hotspot_mask = torch.zeros_like(seq_mask)
            valid_hotspots = [i for i in hotspot_indices if 0 <= i < target_len]  # Use parameter
            if valid_hotspots:
                hotspot_mask[valid_hotspots] = True
                i_con_loss = get_con_loss_torch(
                    dgram_logits, dgram_bins, offset, i_con_opt,
                    mask_1d=hotspot_mask, mask_1b=binder_interface_mask, device=device
                )
                interface_loss = i_con_loss
        else:
            # Use all target residues
            i_con_loss = get_con_loss_torch(
                dgram_logits, dgram_bins, offset, i_con_opt,
                mask_1d=binder_interface_mask, mask_1b=target_mask, device=device
            )
            interface_loss = i_con_loss
    
    return {
        'binder_loss': binder_loss,
        'interface_loss': interface_loss
    }

def get_negative_contact_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    target_len: int,
    binder_len: int,
    residue_index: torch.Tensor,
    device: torch.device,
    negative_hotspot_indices: List[int] = None,
) -> torch.Tensor:
    """
    This function calculates the contact reward between the binder and specified residues (negative hotspots) on the target.
    It penalizes contacts between the binder and negative hotspots.

    Args:
        result: Model prediction dictionary, must contain 'distogram']['distogram_logits'.
        num_tokens: Number of tokens in the full sequence (target + binder).
        target_len: Sequence length of the target protein.
        binder_len: Sequence length of the binder protein.
        residue_index: Residue index tensor with shape (num_tokens,), used for offset calculation.
        device: Computation device.
        negative_hotspot_indices: List of residue indices on the target to penalize for contact with the binder (0-based).
                                  If None, empty list, or containing invalid indices, loss = 0.0.

    Returns:
        torch.Tensor: Scalar tensor, -interface_contact_loss (negative contact loss used as a reward term).
    """
    # Basic parameter validation
    if target_len < 0 or binder_len < 0 or target_len + binder_len != num_tokens:
        raise ValueError("target_len + binder_len must equal num_tokens, and both must be non-negative integers")

    if num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")

    if residue_index.shape[0] != num_tokens:
        raise ValueError("The length of residue_index must equal num_tokens")

    if "distogram" not in result or "distogram_logits" not in result["distogram"]:
        raise ValueError("result must contain 'distogram']['distogram_logits']")

    from config.config import NEG_CON_OPT

    # Use default configurations from the config file
    i_con_opt = NEG_CON_OPT

    # Get distogram logits
    dgram_logits = result["distogram"]["distogram_logits"][:num_tokens, :num_tokens]

    # Get distance bins
    dgram_bins = get_dgram_bins_torch(dgram_logits, device)

    # Calculate residue offset matrix
    residue_index_float = residue_index.float().to(device)
    offset = residue_index_float.unsqueeze(1) - residue_index_float.unsqueeze(0)

    # Create sequence mask
    seq_mask = torch.ones(num_tokens, device=device, dtype=torch.bool)

    # Create target and binder masks
    target_mask = torch.zeros_like(seq_mask)
    target_mask[:target_len] = True

    binder_mask = torch.zeros_like(seq_mask)
    binder_mask[-binder_len:] = True

    # Initialize loss variables
    negative_contact_loss = torch.tensor(0.0, device=device)

    # 2. Interface contact loss (target-binder)
    if target_len > 0 and binder_len > 0:
        if negative_hotspot_indices is not None and len(negative_hotspot_indices) > 0:  # Modified here
            # Use hotspot residues
            hotspot_mask = torch.zeros_like(seq_mask)
            valid_hotspots = [i for i in negative_hotspot_indices if 0 <= i < target_len]  # Use parameter
            if valid_hotspots:
                hotspot_mask[valid_hotspots] = True
                negative_contact_loss = get_con_loss_torch(
                    dgram_logits, dgram_bins, offset, i_con_opt, mask_1d=hotspot_mask, mask_1b=binder_mask, device=device
                )

    return -negative_contact_loss

def get_negative_framework_contact_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    target_len: int,
    binder_len: int,
    residue_index: torch.Tensor,
    device: torch.device,
    design_positions_in_chain: List[int] = None,
    framework_negative_binder_indices: List[int] = None,
) -> torch.Tensor:
    """
    This function calculates the contact penalty between the binder framework region (non-CDR region) and the entire target.
    The purpose is to prevent non-specific binding between the non-designed region (framework) and the target.

    Logic:
    Framework Region = Entire Binder Region - design_positions_in_chain (CDR)
    Loss = - get_con_loss(Target, Framework)
    (Since a smaller get_con_loss indicates better contact, negating it penalizes contact when minimized)

    Args:
        result: Model prediction dictionary.
        num_tokens: Number of tokens in the full sequence.
        target_len: Sequence length of the target protein.
        binder_len: Sequence length of the binder protein.
        residue_index: Residue index tensor.
        device: Computation device.
        design_positions_in_chain: Indices of design positions (CDR regions) within the binder chain, relative to the start of the binder (0-based).

    Returns:
        torch.Tensor: Scalar tensor, negative contact loss used as a penalty term.
    """
    # Basic parameter validation
    if target_len < 0 or binder_len < 0:
        return torch.tensor(0.0, device=device)
    
    if num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")

    if residue_index.shape[0] != num_tokens:
        raise ValueError("The length of residue_index must equal num_tokens")

    if "distogram" not in result or "distogram_logits" not in result["distogram"]:
        raise ValueError("result must contain 'distogram']['distogram_logits']")
    
    # Import negative contact configuration (usually includes a large cutoff or specific weight settings)
    from config.config import NEG_FRAMEWORK_CON_OPT
    neg_con_opt = NEG_FRAMEWORK_CON_OPT

    # Get distogram logits and bins
    # [L, L, num_bins]
    dgram_logits = result['distogram']['distogram_logits'][:num_tokens, :num_tokens]
    dgram_bins = get_dgram_bins_torch(dgram_logits, device)

    # Calculate residue offset matrix
    residue_index_float = residue_index.float().to(device)
    offset = residue_index_float.unsqueeze(1) - residue_index_float.unsqueeze(0)

    # Create sequence mask
    seq_mask = torch.ones(num_tokens, device=device, dtype=torch.bool)

    # 1. Create Target Mask (use the entire Target)
    target_mask = torch.zeros_like(seq_mask)
    target_mask[:target_len] = True

    # 2. Create Framework Mask (Binder - CDR)
    # Logic: First select the entire Binder, then exclude Design Positions (CDR)
    framework_mask = torch.zeros_like(seq_mask)
    # framework_mask[-binder_len:] = True  # Initialize as full-length Binder
    framework_mask[-binder_len:] = False  # Initialize as full-length Binder

    # Prioritize check: Are user-specified Framework exclusion sites provided?
    if framework_negative_binder_indices is not None and len(framework_negative_binder_indices) > 0:
        # [Branch A]: User specified explicit exclusion sites (e.g., face opposite to Target)
        # Directly set these specific positions to True
        for pos in framework_negative_binder_indices:
            global_idx = target_len + pos
            if 0 <= global_idx < num_tokens:
                framework_mask[global_idx] = True

    else:
        # [Branch B]: Not specified (fallback to original logic: full-length Binder minus CDR)
        framework_mask[-binder_len:] = True  # Initialize as full-length Binder

        if design_positions_in_chain is not None and len(design_positions_in_chain) > 0:
            # Convert CDR relative indices to global indices
            cdr_global_indices = []
            for pos in design_positions_in_chain:
                global_idx = target_len + pos
                if 0 <= global_idx < num_tokens:
                    cdr_global_indices.append(global_idx)

            # Set CDR regions to False in Framework Mask (exclude them)
            if cdr_global_indices:
                framework_mask[cdr_global_indices] = False

            else:
                print("[WARNING] Invalid indices after mapping design_positions_in_chain, Framework Loss will be calculated for the entire Binder.")

    # Initialize loss
    neg_fw_loss = torch.tensor(0.0, device=device)

    # 3. Calculate Loss
    # Only calculate if both Target and Framework have residues
    if target_len > 0 and framework_mask.sum() > 0:
        # Calculate contact degree using get_con_loss_torch
        # mask_1d = target_mask (Target)
        # mask_1b = framework_mask (Binder Framework)
        con_loss = get_con_loss_torch(
            dgram_logits, dgram_bins, offset, neg_con_opt,
            mask_1d=target_mask,
            mask_1b=framework_mask,
            device=device
        )
        
        # Negate the value: We want to penalize contact, so we prefer smaller con_loss (representing contact probability)
        # Original function: con_loss = -log(prob), small con_loss for good contact, large con_loss for poor contact.
        # See _get_con_loss_torch: con_loss = -torch.log(contact_prob)
        # Contact probability P -> 1 (contact), Loss -> 0
        # Contact probability P -> 0 (no contact), Loss -> inf
        # In get_negative_contact_loss_torch, returns -get_con_loss_torch(...)
        # Returns negative contact loss.
        neg_fw_loss = -con_loss

    return neg_fw_loss

def get_paratope_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    target_len: int,
    binder_len: int,
    residue_index: torch.Tensor,
    device: torch.device,
    hotspot_indices: List[int] = None,
    design_positions_in_chain: List[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements Paratope Loss from the Germinal paper, reusing get_con_loss_torch.
    Formula: L_paratope = (L_CDR / (L_framework - lambda)) * L_CDR_Target
    """
    device_tensor = lambda: torch.tensor(0.0, device=device, requires_grad=True)

    # 1. Basic checks
    if binder_len == 0 or target_len == 0:
        return device_tensor(), device_tensor(), device_tensor(), device_tensor()
        
    # Return 0 if CDR design positions are not defined (cannot calculate)
    if design_positions_in_chain is None or len(design_positions_in_chain) == 0:
        return device_tensor(), device_tensor(), device_tensor(), device_tensor()

    from config.config import PARATOPE_CDR_OPT, PARATOPE_FW_OPT, PARATOPE_SCALAR_OPT

    # 2. Prepare data (Distogram, Bins, Offset)
    # [L, L, num_bins]
    dgram_logits = result['distogram']['distogram_logits'][:num_tokens, :num_tokens]
    dgram_bins = get_dgram_bins_torch(dgram_logits, device)
    
    residue_index_float = residue_index.float().to(device)
    offset = residue_index_float.unsqueeze(1) - residue_index_float.unsqueeze(0)

    # 3. Create Mask
    seq_mask = torch.ones(num_tokens, device=device, dtype=torch.bool)

    # 3.1 CDR Mask (used for mask_1b in L_CDR and exclusion in L_framework)
    cdr_mask = torch.zeros_like(seq_mask)
    # Convert relative indices to global indices
    valid_cdr_indices = []
    for pos in design_positions_in_chain:
        global_idx = target_len + pos
        if 0 <= global_idx < num_tokens:
            valid_cdr_indices.append(global_idx)
            
    if not valid_cdr_indices:
        return device_tensor(), device_tensor(), device_tensor(), device_tensor()

    cdr_mask[valid_cdr_indices] = True

    # 3.2 Framework Mask (used for mask_1b in L_framework)
    # Framework = Binder - CDR
    binder_mask = torch.zeros_like(seq_mask)
    binder_mask[-binder_len:] = True
    
    # Logic: is Binder but not CDR
    framework_mask = binder_mask & (~cdr_mask)

    # 3.3 Target Hotspot Mask (used for mask_1d in L_CDR)
    target_hotspot_mask = torch.zeros_like(seq_mask)
    if hotspot_indices is not None and len(hotspot_indices) > 0:
        valid_hotspots = [i for i in hotspot_indices if 0 <= i < target_len]
        if valid_hotspots:
            target_hotspot_mask[valid_hotspots] = True
        else:
            # Fallback to all target if hotspot indices are invalid
            target_hotspot_mask[:target_len] = True
    else:
        # Default to all target
        target_hotspot_mask[:target_len] = True

    # 3.4 Target All Mask (used for mask_1d in L_framework)
    target_all_mask = torch.zeros_like(seq_mask)
    target_all_mask[:target_len] = True

    # 4. Calculate L_CDR
    # Objective: CDR (Cols) -> Hotspot (Rows)
    l_cdr = get_con_loss_torch(
        dgram_logits=dgram_logits,
        dgram_bins=dgram_bins,
        offset=offset,
        con_opt=PARATOPE_CDR_OPT,
        # mask_1d=target_hotspot_mask, # Only count Hotspot rows
        # mask_1b=cdr_mask,            # Only look at contacts with CDR columns
        mask_1d=cdr_mask,
        mask_1b=target_hotspot_mask,
        device=device
    )
     
    # 4. Calculate L_CDR_TARGET
    l_cdr_target = get_con_loss_torch(
        dgram_logits=dgram_logits,
        dgram_bins=dgram_bins,
        offset=offset,
        con_opt=PARATOPE_CDR_OPT, # Reuse configuration parameters for CDR
        mask_1d=cdr_mask,
        mask_1b=target_all_mask,
        device=device
    )

    # 5. Calculate L_framework
    # Objective: Framework (Rows) -> All Target (Cols)
    if framework_mask.sum() > 0:
        l_framework = get_con_loss_torch(
            dgram_logits=dgram_logits,
            dgram_bins=dgram_bins,
            offset=offset,
            con_opt=PARATOPE_FW_OPT,
            mask_1d=framework_mask,   # Only Framework          
            mask_1b=target_all_mask,  # Entire Target
            device=device
        )
    else:
        # If no framework exists (e.g., full-length CDR design), assign a large loss value representing "no contact"
        # -log(low_prob) -> Large value
        l_framework = torch.tensor(20.0, device=device)

    # 6. Combine Paratope Loss
    # Formula: L_cdr / (L_fw - lambda)
    # Note: Prevent denominator from being <= 0.
    # Smaller L_fw means tighter contact (bad), we want it to be large.
    # If L_fw < lambda, framework binding is too strong, denominator approaches 0 or negative; apply heavy penalty in this case.
    
    lambda_val = PARATOPE_SCALAR_OPT["lambda_offset"]
    eps = PARATOPE_SCALAR_OPT["eps"]
    rebalance_val = PARATOPE_SCALAR_OPT.get("rebalance", 1.0)

    # Use ReLU to ensure non-negative denominator.
    # Logic: If L_fw (2.0) <= lambda (2.0), denominator = 0+eps, Loss explodes -> heavy penalty for framework binding
    # If L_fw (5.0) > lambda (2.0), denominator = 3.0, scale CDR Loss normally
    denominator = torch.nn.functional.relu(l_framework - lambda_val) + eps
    rebalance_denominator = rebalance_val * denominator
    
    final_loss = l_cdr / rebalance_denominator

    final_loss = final_loss * l_cdr_target
    
    return final_loss, l_cdr, l_framework, l_cdr_target    

def get_helix_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    target_len: int,
    binder_len: int,
    residue_index: torch.Tensor,
    device: torch.device,
    cutoff: float = 6.0,
    binary: bool = True
) -> torch.Tensor:
    """
    Helix Loss calculation function based on BoltzDesign1 logic (PyTorch version)
    Calculates contact loss for i, i+3 residue pairs within the binder chain to encourage helix structure formation.
    
    Args:
        result: Model prediction results containing distogram logits
        num_tokens: Total sequence length
        target_len: Length of the target protein
        binder_len: Length of the binder protein
        residue_index: Residue index tensor
        device: Computation device
        cutoff: Distance cutoff value, default 6.0Å (suitable for i,i+3 contacts in alpha-helices)
        binary: Whether to use binary loss
    
    Returns:
        helix_loss: Scalar loss to encourage helix formation in the binder chain
    """
    # Return 0 if no binder exists
    if binder_len <= 3:
        return torch.tensor(0.0, device=device)
    
    # Get distogram logits
    dgram_logits = result['distogram']['distogram_logits'][:num_tokens, :num_tokens]
    
    # Get distance bins
    dgram_bins = get_dgram_bins_torch(dgram_logits, device)
    
    # Calculate residue offset matrix
    residue_index_float = residue_index.float().to(device)
    offset = residue_index_float.unsqueeze(1) - residue_index_float.unsqueeze(0)
    
    # Create mask for the binder chain (only consider within binder)
    binder_mask = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    binder_mask[-binder_len:] = True
    
    # Create 2D mask, keep only binder × binder region
    mask_2d = binder_mask.unsqueeze(1) * binder_mask.unsqueeze(0)
    
    # Call core helix loss calculation function
    helix_loss = _get_helix_loss_torch(
        dgram_logits=dgram_logits,
        dgram_bins=dgram_bins,
        offset=offset,
        mask_2d=mask_2d,
        cutoff=cutoff,
        binary=binary,
        device=device
    )
    
    return helix_loss


def _get_helix_loss_torch(
    dgram_logits: torch.Tensor,
    dgram_bins: torch.Tensor,
    offset: torch.Tensor = None,
    mask_2d: torch.Tensor = None,
    cutoff: float = 6.0,
    binary: bool = True,
    device: torch.device = None
) -> torch.Tensor:
    """
    Core helix loss calculation function based on BoltzDesign1
    
    Args:
        dgram_logits: Distance distribution logits [L, L, num_bins]
        dgram_bins: Distance bins [num_bins]
        offset: Residue offset matrix [L, L], automatically compute i,i+3 diagonals if None
        mask_2d: 2D mask [L, L]
        cutoff: Distance cutoff value
        binary: Whether to use binary loss
        device: Computation device
    
    Returns:
        helix_loss: Scalar loss value
    """
    if device is None:
        device = dgram_logits.device
    
    # Calculate basic contact loss matrix
    x = _get_con_loss_torch(dgram_logits, dgram_bins, cutoff, binary)
    
    if offset is None:
        # Offset=None branch in BoltzDesign1
        if mask_2d is None:
            # No mask, directly take the i,i+3 diagonal
            return x.diagonal(offset=3).mean()
        else:
            # With mask, only consider i,i+3 diagonal within the masked region
            mask_2d = mask_2d.float()
            
            # Extract i,i+3 diagonal after applying mask
            masked_x = x * mask_2d
            i_i3_losses = torch.diagonal(masked_x, offset=3, dim1=-2, dim2=-1)
            i_i3_mask = torch.diagonal(mask_2d, offset=3, dim1=-2, dim2=-1)
            
            # Compute weighted average
            return i_i3_losses.sum() / (i_i3_mask.sum() + 1e-8)
    else:
        # Offset!=None branch in BoltzDesign1 (used for specifying specific offsets)
        mask = (offset == 3).float()
        if mask_2d is not None:
            mask = mask * mask_2d.float()
        return (x * mask).sum() / (mask.sum() + 1e-8)


def get_con_loss_torch(
    dgram_logits: torch.Tensor,
    dgram_bins: torch.Tensor,
    offset: torch.Tensor,
    con_opt: Dict,
    mask_1d: torch.Tensor = None,
    mask_1b: torch.Tensor = None,
    mask_2d: torch.Tensor = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    PyTorch version of contact loss calculation (based on ColabDesign get_con_loss)
    
    Args:
        dgram_logits: Distance distribution logits [L, L, num_bins]
        dgram_bins: Distance bins [num_bins]
        offset: Residue offset matrix [L, L]
        con_opt: Contact configuration options
        mask_1d: 1D mask [L]
        mask_1b: 1D mask [L] 
        mask_2d: 2D mask [L, L]
        device: Computation device
    
    Returns:
        contact_loss: Scalar loss value
    """
    if device is None:
        device = dgram_logits.device
    
    L = dgram_logits.shape[0]
    
    # Top-k selection function
    def min_k_torch(x, k=1, mask=None):
        if mask is not None:
            x = torch.where(mask, x, float('inf'))
        
        # Sort along the last dimension
        y_sorted, _ = torch.sort(x, dim=-1)
        
        # Create k_mask
        k_mask = torch.arange(y_sorted.shape[-1], device=device) < k
        k_mask = k_mask & (y_sorted < float('inf'))  # Exclude inf values
        
        # Calculate mean
        numerator = torch.where(k_mask, y_sorted, 0.0).sum(-1)
        denominator = k_mask.sum(-1).float() + 1e-8
        
        return numerator / denominator
    
    # Calculate basic contact loss
    p = _get_con_loss_torch(dgram_logits, dgram_bins, con_opt["cutoff"], con_opt["binary"])
    
    # Apply sequence separation constraint
    if "seqsep" in con_opt and con_opt["seqsep"] > 0:
        seqsep_mask = torch.abs(offset) >= con_opt["seqsep"]
    else:
        seqsep_mask = torch.ones_like(offset, dtype=torch.bool)
    
    # Apply masks
    if mask_1d is None:
        mask_1d = torch.ones(L, device=device, dtype=torch.bool)
    if mask_1b is None:
        mask_1b = torch.ones(L, device=device, dtype=torch.bool)
    
    # Process mask according to ColabDesign logic
    if mask_2d is None:
        # Asymmetric mask: apply mask_1b only on column dimension for flexible contact calculation
        final_mask = seqsep_mask & mask_1b.unsqueeze(0)
    else:
        final_mask = seqsep_mask & mask_2d
    
    # First layer min_k: select top-k contacts for each residue
    p_topk = min_k_torch(p, con_opt["num"], final_mask)
    
    # Second layer min_k: select top positions
    if "num_pos" in con_opt and con_opt["num_pos"] != float("inf"):
        loss = min_k_torch(p_topk, con_opt["num_pos"], mask_1d)
    else:
        # Calculate average over all valid residues
        loss = torch.where(mask_1d, p_topk, 0.0).sum() / (mask_1d.sum().float() + 1e-8)
    
    return loss


def _get_con_loss_torch(
    dgram_logits: torch.Tensor,
    dgram_bins: torch.Tensor,
    cutoff: float = 8.0,
    binary: bool = True
) -> torch.Tensor:
    """
    Low-level contact loss calculation in PyTorch
    
    Args:
        dgram_logits: Distance distribution logits [L, L, num_bins]
        dgram_bins: Distance bins [num_bins]
        cutoff: Distance cutoff value
        binary: Whether to use binary cross-entropy
    
    Returns:
        contact_loss: [L, L] contact loss matrix
    """
    # Define contact bins (bins with distance less than cutoff)
    bins_mask = dgram_bins < cutoff  # [num_bins]
    
    # Calculate probability distribution
    px = torch.softmax(dgram_logits, dim=-1)  # [L, L, num_bins]
    
    # Only consider probabilities from contact bins
    masked_logits = dgram_logits - 1e7 * (~bins_mask).float()
    px_contact = torch.softmax(masked_logits, dim=-1)  # [L, L, num_bins]
    
    if binary:
        # Binary cross-entropy: -log(P(contact))
        contact_prob = (bins_mask.float() * px).sum(-1)  # [L, L]
        con_loss = -torch.log(contact_prob + 1e-8)
    else:
        # Categorical cross-entropy
        log_softmax_dgram = torch.log_softmax(dgram_logits, dim=-1)
        con_loss = -(px_contact * log_softmax_dgram).sum(-1)  # [L, L]
    
    return con_loss


def get_dgram_bins_torch(dgram_logits: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Get distance bins (based on ColabDesign get_dgram_bins)
    
    Args:
        dgram_logits: Distance distribution logits
        device: Computation device
    
    Returns:
        dgram_bins: Distance bins
    """
    num_bins = dgram_logits.shape[-1]
    
    if num_bins == 64:
        # One style: 64 bins
        bins = torch.linspace(2.3125, 21.6875, 63, device=device)
        dgram_bins = torch.cat([torch.tensor([0.0], device=device), bins])
    elif num_bins == 39:
        # Alternative style: 39 bins
        dgram_bins = torch.linspace(3.25, 50.75, 39, device=device) + 1.25
    else:
        # Default: uniform distribution from 2 to 22 Å
        dgram_bins = torch.linspace(2.0, 22.0, num_bins, device=device)
    
    return dgram_bins


def get_binder_pae_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    target_len: int,
    binder_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Calculate PAE loss within the binder
    """
    if binder_len <= 1:  # No internal PAE if binder length <= 1
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Create binder mask
    binder_mask = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    binder_mask[-binder_len:] = True  # Binder is at the end of the sequence
    
    # Call general PAE loss function, restrict calculation with binder mask
    binder_pae_loss = get_pae_loss_torch(
        result=result,
        num_tokens=num_tokens,
        mask_1d=binder_mask,  # Only consider binder residues
        mask_1b=binder_mask,  # Only consider binder residues
        use_interface=False   # Use full_pae
    )
    
    return binder_pae_loss

# Ignore hotspot parameter and always use the entire target region.
def get_binder_target_interface_pae_loss_torch(
    result: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    num_tokens: int,
    target_len: int,
    binder_len: int,
    device: torch.device,
    hotspot_indices: List[int] = None
) -> torch.Tensor:
    """
    Calculate interface PAE loss between binder and target
    """
    if binder_len == 0 or target_len == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Create target mask - ignore hotspot temporarily, always use the entire target region
    target_mask = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    target_mask[:target_len] = True  # Always consider the entire target region
    
    # Create binder mask
    binder_mask = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    binder_mask[-binder_len:] = True  # Binder is at the end of the sequence
    
    # Call general PAE loss function to calculate PAE between binder and target
    interface_pae_loss = get_pae_loss_torch(
        result=result,
        num_tokens=num_tokens,
        mask_1d=binder_mask,   # Rows: binder residues
        mask_1b=target_mask,   # Columns: entire target region
        use_interface=False    # Set to False to force using full_pae
    )
    
    return interface_pae_loss

    