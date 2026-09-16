from torchfold.runtime_policy import relative_encoding_row_chunk_size
from typing import Sequence

import torch

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from torchfold import feat_batch, features
from . import utils
from torchx.constants import residue_names
from ...parallel_ops import ParallelSpec, pad_to_length


def gumbel_noise(
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
        shape,
        dtype=torch.float32,
        device=device,
    )
    gumbel = -torch.log(-torch.log(uniform_noise + eps) + eps)
    return gumbel


def gumbel_argsort_sample_idx(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Return indices ordered by logits plus independent Gumbel noise.

    This generates a without-replacement ordering along the final axis.
    Logits may be unnormalized log probabilities. The returned integer tensor
    has the same shape as logits and contains indices, not one-hot values.
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
    deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / torch.pi)).unsqueeze(-1)

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
    append_per_atom_features: bool,
) -> torch.Tensor:
    token_features = batch.token_features
    target_features = []

    target_features.append(
        torch.nn.functional.one_hot(
            token_features.aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
        )
    )
    target_features.append(batch.msa.profile)
    target_features.append(batch.msa.deletion_mean.unsqueeze(-1))

    if append_per_atom_features:
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

    offset = left_residue_index - right_residue_index
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

    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype).unsqueeze(-1))

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


def create_relative_encoding_row_shard(
    seq_features: features.TokenFeatures,
    max_relative_idx: int,
    max_relative_chain: int,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """
    Returns [L_local, L_padded, F_rel].
    """
    rel_feats = []
    seq_mask = pad_to_length(seq_features.mask, 0, parallel_spec.n_padded, value=0)

    token_index = pad_to_length(seq_features.token_index, 0, parallel_spec.n_padded, value=0)
    residue_index = pad_to_length(seq_features.residue_index, 0, parallel_spec.n_padded, value=0)
    asym_id = pad_to_length(seq_features.asym_id, 0, parallel_spec.n_padded, value=0)
    entity_id = pad_to_length(seq_features.entity_id, 0, parallel_spec.n_padded, value=0)
    sym_id = pad_to_length(seq_features.sym_id, 0, parallel_spec.n_padded, value=0)

    left_asym_id = asym_id[parallel_spec.start:parallel_spec.end].unsqueeze(1)
    right_asym_id = asym_id.unsqueeze(0)

    left_residue_index = residue_index[parallel_spec.start:parallel_spec.end].unsqueeze(1)
    right_residue_index = residue_index.unsqueeze(0)

    left_token_index = token_index[parallel_spec.start:parallel_spec.end].unsqueeze(1)
    right_token_index = token_index.unsqueeze(0)

    left_entity_id = entity_id[parallel_spec.start:parallel_spec.end].unsqueeze(1)
    right_entity_id = entity_id.unsqueeze(0)

    left_sym_id = sym_id[parallel_spec.start:parallel_spec.end].unsqueeze(1)
    right_sym_id = sym_id.unsqueeze(0)

    offset = left_residue_index - right_residue_index
    clipped_offset = torch.clip(
        offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = torch.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
    )
    rel_pos = torch.nn.functional.one_hot(
        final_offset.to(dtype=torch.int64),
        2 * max_relative_idx + 2
    )
    rel_feats.append(rel_pos)

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
        final_token_offset.to(dtype=torch.int64),
        2 * max_relative_idx + 2
    )
    rel_feats.append(rel_token)

    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype).unsqueeze(-1))

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
    rel_chain = torch.nn.functional.one_hot(
        final_rel_chain.to(dtype=torch.int64),
        2 * max_relative_chain + 2
    )
    rel_feats.append(rel_chain)

    rel = torch.concatenate(rel_feats, dim=-1)
    valid_row = seq_mask[parallel_spec.start:parallel_spec.end].unsqueeze(1)
    valid_col = seq_mask.unsqueeze(0)
    valid_pair = (valid_row * valid_col).unsqueeze(-1).to(dtype=rel.dtype)
    return rel * valid_pair


def _create_relative_encoding_row_slice(
    seq_features: features.TokenFeatures,
    max_relative_idx: int,
    max_relative_chain: int,
    parallel_spec: ParallelSpec,
    row_start: int,
    row_end: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create row-sharded relative encoding for a global row slice."""
    seq_mask = pad_to_length(seq_features.mask, 0, parallel_spec.n_padded, value=0)

    token_index = pad_to_length(seq_features.token_index, 0, parallel_spec.n_padded, value=0)
    residue_index = pad_to_length(seq_features.residue_index, 0, parallel_spec.n_padded, value=0)
    asym_id = pad_to_length(seq_features.asym_id, 0, parallel_spec.n_padded, value=0)
    entity_id = pad_to_length(seq_features.entity_id, 0, parallel_spec.n_padded, value=0)
    sym_id = pad_to_length(seq_features.sym_id, 0, parallel_spec.n_padded, value=0)

    left_asym_id = asym_id[row_start:row_end].unsqueeze(1)
    right_asym_id = asym_id.unsqueeze(0)

    left_residue_index = residue_index[row_start:row_end].unsqueeze(1)
    right_residue_index = residue_index.unsqueeze(0)

    left_token_index = token_index[row_start:row_end].unsqueeze(1)
    right_token_index = token_index.unsqueeze(0)

    left_entity_id = entity_id[row_start:row_end].unsqueeze(1)
    right_entity_id = entity_id.unsqueeze(0)

    left_sym_id = sym_id[row_start:row_end].unsqueeze(1)
    right_sym_id = sym_id.unsqueeze(0)

    offset = left_residue_index - right_residue_index
    clipped_offset = torch.clip(
        offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = torch.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
    )
    rel_pos = torch.nn.functional.one_hot(
        final_offset.to(dtype=torch.int64),
        2 * max_relative_idx + 2,
    ).to(dtype=dtype)

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
        final_token_offset.to(dtype=torch.int64),
        2 * max_relative_idx + 2,
    ).to(dtype=dtype)

    entity_id_same = left_entity_id == right_entity_id
    rel_entity = entity_id_same.to(dtype=dtype).unsqueeze(-1)

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
    rel_chain = torch.nn.functional.one_hot(
        final_rel_chain.to(dtype=torch.int64),
        2 * max_relative_chain + 2,
    ).to(dtype=dtype)

    rel = torch.concatenate([rel_pos, rel_token, rel_entity, rel_chain], dim=-1)
    valid_row = seq_mask[row_start:row_end].unsqueeze(1)
    valid_col = seq_mask.unsqueeze(0)
    valid_pair = (valid_row * valid_col).unsqueeze(-1).to(dtype=dtype)
    return rel * valid_pair


def add_relative_encoding_to_pair_row_shard(
    pair_activations_row: torch.Tensor,
    projection: torch.nn.Module,
    seq_features: features.TokenFeatures,
    max_relative_idx: int,
    max_relative_chain: int,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """Add projected row-sharded relative encoding without materialising it all."""
    chunk_size = relative_encoding_row_chunk_size(
        local_rows=pair_activations_row.shape[0],
    )

    local_start = parallel_spec.start
    local_end = parallel_spec.end
    for row_start in range(local_start, local_end, chunk_size):
        row_end = min(row_start + chunk_size, local_end)
        local_slice = slice(row_start - local_start, row_end - local_start)
        rel_feat = _create_relative_encoding_row_slice(
            seq_features=seq_features,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
            parallel_spec=parallel_spec,
            row_start=row_start,
            row_end=row_end,
            dtype=pair_activations_row.dtype,
        )
        pair_activations_row[local_slice] += projection(rel_feat)

    return pair_activations_row


def shuffle_msa(
    msa: features.MSA,
    synchronize_across_ranks: bool = False,
) -> features.MSA:
    """Shuffle MSA randomly, return batch with shuffled MSA."""
    logits = (torch.clip(torch.sum(msa.mask, dim=-1), 0.0, 1.0) - 1.0) * 1e6

    if (
        synchronize_across_ranks
        and dist is not None
        and dist.is_available()
        and dist.is_initialized()
    ):
        # Sample once on every rank from the current global RNG and device.
        # This preserves the original reshuffle-on-call behavior while keeping
        # all ranks' RNG streams aligned by consuming the same number of values.
        index_order = gumbel_argsort_sample_idx(logits)
        dist.broadcast(index_order, src=0)
    else:
        index_order = gumbel_argsort_sample_idx(logits)

    return msa.index_msa_rows(index_order)
