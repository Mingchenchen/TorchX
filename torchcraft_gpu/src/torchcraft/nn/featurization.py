


from typing import Sequence

import torch
from torchx.constants import residue_names

from custom_function.straight_through import four_stage_sequence_optimization
from torchcraft import feat_batch, features
from torchcraft.nn import utils

CYCLIC_OFFSET_ENABLED = False

def gumbel_noise(  # NOTE: remove generator to fixate the seed
    shape: Sequence[int],
    device: torch.device,
    eps=1e-6,
) -> torch.Tensor:
    """Generate Gumbel Noise of given Shape.

    This generates samples from Gumbel(0, 1).

    Args:
        shape: Shape of noise to return.

    Returns:
        Gumbel noise of given shape.
    """
    uniform_noise = torch.rand(
        # shape, dtype=torch.bfloat16, device=device # use the global generator to make it deterministic
        shape, dtype=torch.float32, device=device # use the global generator to make it deterministic
    )
    gumbel = -torch.log(-torch.log(uniform_noise + eps) + eps)
    return gumbel


def gumbel_argsort_sample_idx(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Samples with replacement from a distribution given by 'logits'.

    This uses Gumbel trick to implement the sampling an efficient manner. For a
    distribution over k items this samples k times without replacement, so this
    is effectively sampling a random permutation with probabilities over the
    permutations derived from the logprobs.

    Args:
      logits: logarithm of probabilities to sample from, probabilities can be
        unnormalized.

    Returns:
      Sample from logprobs in one-hot form.
    """
    z = gumbel_noise(logits.shape, device=logits.device)
    return torch.argsort(logits + z, dim=-1, descending=True)


def create_msa_feat(msa: features.MSA) -> torch.Tensor:
    """Create and concatenate MSA features."""
    msa_1hot = torch.nn.functional.one_hot(
        msa.rows.to(
            dtype=torch.int64), residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1
    )
    deletion_matrix = msa.deletion_matrix
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0).unsqueeze(-1)
    deletion_value = ((2.0 / torch.pi) * torch.arctan(deletion_matrix / 3.0)).unsqueeze(-1)

    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
    ]

    return torch.concatenate(msa_feat, dim=-1)

def truncate_msa_batch(msa: features.MSA, num_msa: int) -> features.MSA:
    indices = torch.arange(num_msa, device=msa.rows.device, dtype=torch.int64)
    return msa.index_msa_rows(indices)


def create_target_feat(
    batch: feat_batch.Batch,
    binder_logits_20,
    append_per_atom_features: bool,
    current_epoch: int = 0,
    stage_epochs: list = [50, 50, 50, 10],  # Use stage_epochs directly
) -> torch.Tensor:
    """Make target feat."""
    token_features = batch.token_features
    target_features = []
    
    # First save the one-hot encoding result to a variable
    aatype_one_hot = torch.nn.functional.one_hot(
        token_features.aatype.to(dtype=torch.int64),
        residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
    )
    
    # Get sequence length N
    seq_length = binder_logits_20.shape[0]
    
    # Generate pseudo sequence (using four-stage optimization)
    binder_probs, stage_info = four_stage_sequence_optimization(
        binder_logits_20, 
        current_epoch, 
        sum(stage_epochs),  # Total epochs calculated from stage_epochs
        stage_epochs=stage_epochs
    )
    
    # Print current stage information
    print(f"Current stage: {stage_info['description']}")
    
    # Create zero tensor with shape (N, 11)
    zeros_tensor = torch.zeros((seq_length, 11), device=binder_logits_20.device)
    
    # Concatenate binder_probs and zeros_tensor along the last dimension to get tensor with shape (N_design, 31)
    binder_full_probs = torch.cat([binder_probs, zeros_tensor], dim=-1)
    
    # Find the maximum value in batch.token_features.asym_id, treat it as the binder chain
    max_asym_id = torch.max(batch.token_features.asym_id)

    # Create a mask to identify positions where asym_id equals the maximum value
    binder_mask = (batch.token_features.asym_id == max_asym_id)
    
    # Global token indices of the binder_chain in aatype_one_hot
    binder_indices = torch.nonzero(binder_mask, as_tuple=True)[0]  # [L_binder]

    # === Design Mode Selection ===
    # Default mode: old logic, binder_logits_20 covers the entire binder chain
    design_indices_in_chain = getattr(
        create_target_feat, "design_positions_in_chain", None
    )

    if design_indices_in_chain is None or len(design_indices_in_chain) == 0:
        # Compatibility with old behavior: require logits length == binder chain length
        assert binder_indices.shape[0] == binder_full_probs.shape[0], (
            f"In default mode, the length of binder_logits_20 ({binder_full_probs.shape[0]})"
            f"must equal the length of the binder chain ({binder_indices.shape[0]})"
        )
        design_token_indices = binder_indices
    else:
        # nanobody / sub-region design mode: only replace at specified positions within the chain
        device = binder_logits_20.device
        design_pos_tensor = torch.as_tensor(
            design_indices_in_chain, device=device, dtype=torch.long
        )
        # Check length consistency
        assert design_pos_tensor.shape[0] == binder_full_probs.shape[0], (
            f"The number of design positions ({design_pos_tensor.shape[0]})"
            f"and logits seq length ({binder_full_probs.shape[0]}) mismatch"
        )
        # Map "intra-chain indices" to global token indices
        design_token_indices = binder_indices[design_pos_tensor]

    # Expand binder_full_probs to full size in advance
    full_replacement = torch.zeros_like(aatype_one_hot, dtype=torch.float32)

    # Construct an all-zero replacement tensor, then write binder_full_probs at design positions
    # Use index_copy_ for copying
    full_replacement.index_copy_(0, design_token_indices, binder_full_probs)

    # Use replacement only at design positions, keep original one-hot for others
    design_mask_global = torch.zeros_like(
        batch.token_features.asym_id, dtype=torch.bool
    )
    design_mask_global.index_fill_(0, design_token_indices, True)
    design_mask_global = design_mask_global.view(-1, 1)

    aatype_one_hot = torch.where(design_mask_global, full_replacement, aatype_one_hot)
    target_features.append(aatype_one_hot)

    target_features.append(aatype_one_hot)
    target_features.append(batch.msa.deletion_mean.unsqueeze(-1))
    # Reference structure features
    if append_per_atom_features:
        print('append_per_atom_features')
        ref_mask = batch.ref_structure.mask
        element_feat = torch.nn.functional.one_hot(
            batch.ref_structure.element, 128)
        element_feat = utils.mask_mean(
            mask=ref_mask.unsqueeze(-1), value=element_feat, axis=-2, eps=1e-6
        )
        target_features.append(element_feat)
        pos_feat = batch.ref_structure.positions
        pos_feat = pos_feat.reshape([pos_feat.shape[0], -1])
        target_features.append(pos_feat)
        target_features.append(ref_mask)
    return torch.concatenate(target_features, dim=-1)


def create_relative_encoding(
    seq_features: features.TokenFeatures,
    max_relative_idx: int,
    max_relative_chain: int
) -> torch.Tensor:
    """Add relative position encodings."""
    rel_feats = []
    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    left_asym_id = asym_id.unsqueeze(1)
    right_asym_id = asym_id.unsqueeze(0)

    left_residue_index = residue_index.unsqueeze(1)
    right_residue_index = residue_index.unsqueeze(0)

    left_token_index = token_index.unsqueeze(1)
    right_token_index = token_index.unsqueeze(0)

    left_entity_id = entity_id.unsqueeze(1)
    right_entity_id = entity_id.unsqueeze(0)

    left_sym_id = sym_id.unsqueeze(1)
    right_sym_id = sym_id.unsqueeze(0)

    # Embed relative positions using a one-hot embedding of distance along chain
    offset = left_residue_index - right_residue_index

    if CYCLIC_OFFSET_ENABLED:
        max_asym = torch.max(asym_id)
        is_binder = (left_asym_id == max_asym) & (right_asym_id == max_asym)
        # Get the length of the binder
        binder_mask = (asym_id == max_asym)
        binder_len = binder_mask.sum()
        if binder_len > 0:
            curr_offset = offset
            L = binder_len.item()
            if L > 1: # No need for cyclization when length is 1
                cyclic_diff = (curr_offset + L // 2) % L - L // 2
                offset = torch.where(is_binder, cyclic_diff, offset)

    clipped_offset = torch.clip(
        offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = torch.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
    )
    rel_pos = torch.nn.functional.one_hot(final_offset.to(
        dtype=torch.int64), 2 * max_relative_idx + 2)
    rel_feats.append(rel_pos)

    # Embed relative token index as a one-hot embedding of distance along residue
    token_offset = left_token_index - right_token_index
    clipped_token_offset = torch.clip(
        token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    residue_same = (left_asym_id == right_asym_id) & (
        left_residue_index == right_residue_index
    )
    final_token_offset = torch.where(
        residue_same,
        clipped_token_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_token_offset),
    )
    rel_token = torch.nn.functional.one_hot(
        final_token_offset.to(dtype=torch.int64), 2 * max_relative_idx + 2)
    rel_feats.append(rel_token)

    # Embed same entity ID
    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype).unsqueeze(-1))

    # Embed relative chain ID inside each symmetry class
    rel_sym_id = left_sym_id - right_sym_id

    max_rel_chain = max_relative_chain

    clipped_rel_chain = torch.clip(
        rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
    )

    final_rel_chain = torch.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
    )
    rel_chain = torch.nn.functional.one_hot(final_rel_chain.to(
        dtype=torch.int64), 2 * max_relative_chain + 2)

    rel_feats.append(rel_chain)

    return torch.concatenate(rel_feats, dim=-1)


def shuffle_msa(
    msa: features.MSA
) -> features.MSA:
    """Shuffle MSA randomly, return batch with shuffled MSA.

    Args:
      msa: MSA object to sample msa from.

    Returns:
      Protein with sampled msa.
    """
    # Sample uniformly among sequences with at least one non-masked position.
    logits = (torch.clip(torch.sum(msa.mask, dim=-1), 0.0, 1.0) - 1.0) * 1e6
    index_order = gumbel_argsort_sample_idx(logits)

    return msa.index_msa_rows(index_order)
