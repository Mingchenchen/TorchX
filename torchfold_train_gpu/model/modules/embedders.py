"""torchfold.model.modules.embedders
"""

from typing import Sequence

import torch

from torchfold.model import feat_batch, features  # noqa: F401
from torchfold.model.constants import residue_names  # noqa: F401

def _mask_mean(
    mask: torch.Tensor,
    value: torch.Tensor,
    dim: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Masked mean along a single dimension; device-portable."""
    return torch.sum(mask * value, dim=dim) / torch.clamp(
        torch.sum(mask, dim=dim), min=eps
    )


# ---------------------------------------------------------------------------
# Gumbel sampling helpers
# ---------------------------------------------------------------------------

def gumbel_noise(
    shape: Sequence[int],
    device: torch.device,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Generate Gumbel Noise of given Shape.

    This generates samples from Gumbel(0, 1).

    Args:
        shape: Shape of noise to return.
        device: Device to generate noise on.
        eps: Small value for numerical stability.

    Returns:
        Gumbel noise of given shape.
    """
    uniform_noise = torch.rand(
        shape, dtype=torch.bfloat16, device=device  # use the global generator to make it deterministic
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


# ---------------------------------------------------------------------------
# MSA feature helpers
# ---------------------------------------------------------------------------

def create_msa_feat(msa: "features.MSA") -> torch.Tensor:
    """Create and concatenate MSA features.

    Returns tensor of shape [num_msa, num_tokens, POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP+3].
    Last dimension: one-hot(32) + has_deletion(1) + deletion_value(1) = 34.
    """
    msa_1hot = torch.nn.functional.one_hot(
        msa.rows.to(dtype=torch.int64),
        residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1,
    )
    deletion_matrix = msa.deletion_matrix
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / torch.pi))[
        ..., None
    ]

    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
    ]

    return torch.concatenate(msa_feat, dim=-1)


def truncate_msa_batch(msa: "features.MSA", num_msa: int) -> "features.MSA":
    """Sample up to num_msa MSA rows, capped at the available depth.

    """
    num_msa = min(int(num_msa), int(msa.rows.shape[0]))
    indices = torch.arange(num_msa, device=msa.rows.device, dtype=torch.int64)
    return msa.index_msa_rows(indices)


def shuffle_msa(
    msa: "features.MSA",
) -> "features.MSA":
    """Shuffle MSA randomly, return batch with shuffled MSA.

    Args:
      msa: MSA object to sample msa from.

    Returns:
      MSA with shuffled rows.
    """
    # Sample uniformly among sequences with at least one non-masked position.
    logits = (torch.clip(torch.sum(msa.mask, dim=-1), 0.0, 1.0) - 1.0) * 1e6
    index_order = gumbel_argsort_sample_idx(logits)
    return msa.index_msa_rows(index_order)


# ---------------------------------------------------------------------------
# Target feature creation
# ---------------------------------------------------------------------------

def create_target_feat(
    batch: "feat_batch.Batch",
    append_per_atom_features: bool,
) -> torch.Tensor:
    """Make target feat.

    Output dim (append_per_atom_features=False): 31 (one-hot) + profile(31) + deletion_mean(1) = 63
    Output dim (append_per_atom_features=True):  63 + element(128) + pos(max_atoms*3) + mask(max_atoms)
    """
    token_features = batch.token_features
    target_features = []

    target_features.append(
        torch.nn.functional.one_hot(
            token_features.aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
        )
    )
    target_features.append(batch.msa.profile)
    target_features.append(batch.msa.deletion_mean[..., None])

    # Reference structure features
    if append_per_atom_features:
        ref_mask = batch.ref_structure.mask
        element_feat = torch.nn.functional.one_hot(
            batch.ref_structure.element, 128)
        element_feat = _mask_mean(
            mask=ref_mask[..., None], value=element_feat.float(), dim=-2, eps=1e-6
        )
        target_features.append(element_feat)
        pos_feat = batch.ref_structure.positions
        pos_feat = pos_feat.reshape([pos_feat.shape[0], -1])
        target_features.append(pos_feat)
        target_features.append(ref_mask)

    return torch.concatenate(target_features, dim=-1)


# ---------------------------------------------------------------------------
# Relative position encoding
# ---------------------------------------------------------------------------

def create_relative_encoding(
    seq_features: "features.TokenFeatures",
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Add relative position encodings.

    Output dim: (2*max_relative_idx+2) * 2 + 1 + (2*max_relative_chain+2)
                = 4*(max_relative_idx+1) + 1 + 2*max_relative_chain + 2
    For max_relative_idx=32, max_relative_chain=2: 66+66+1+6 = 139
    """
    rel_feats = []
    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    left_asym_id = asym_id[:, None]
    right_asym_id = asym_id[None, :]

    left_residue_index = residue_index[:, None]
    right_residue_index = residue_index[None, :]

    left_token_index = token_index[:, None]
    right_token_index = token_index[None, :]

    left_entity_id = entity_id[:, None]
    right_entity_id = entity_id[None, :]

    left_sym_id = sym_id[:, None]
    right_sym_id = sym_id[None, :]

    # Embed relative positions using a one-hot embedding of distance along chain
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
    rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype)[..., None])

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


import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _original_checkpoint

from torchfold.utils import geometry  # noqa: F401
from torchfold.model import scoring  # noqa: F401
from torchfold.model.constants import protein_data_processing  # noqa: F401
from torchfold.model.constants import residue_names  # noqa: F401
# features already imported at the top of this file

# Ported PairformerBlock (no fastnn)
from torchfold.model.modules.pairformer import PairformerBlock
# Vendored LayerNorm (replaces fastnn.LayerNorm)
from torchfold.model.triangular.layers import LayerNorm


@dataclass
class DistogramFeaturesConfig:
    """Distogram bin configuration"""
    # The left edge of the first bin.
    min_bin: float = 3.25
    # The left edge of the final bin. The final bin catches everything larger than
    # `max_bin`.
    max_bin: float = 50.75
    # The number of bins in the distogram.
    num_bins: int = 39


def dgram_from_positions(positions: torch.Tensor, config: DistogramFeaturesConfig) -> torch.Tensor:
    """Compute distogram from amino acid positions.

    Args:
      positions: (num_res, 3) Position coordinates.
      config: Distogram bin configuration.

    Returns:
      Distogram with the specified number of bins.
    """
    lower_breaks = torch.linspace(
        config.min_bin, config.max_bin, config.num_bins, device=positions.device)
    lower_breaks = torch.square(lower_breaks)
    upper_breaks_last = torch.ones(1, device=lower_breaks.device) * 1e8
    upper_breaks = torch.concatenate(
        [lower_breaks[1:], upper_breaks_last], dim=-1
    )
    dist2 = torch.sum(
        torch.square(
            torch.unsqueeze(positions, dim=-2)
            - torch.unsqueeze(positions, dim=-3)
        ),
        dim=-1,
        keepdims=True,
    )
    dgram = (dist2 > lower_breaks).to(dtype=torch.bfloat16) * (
        dist2 < upper_breaks
    ).to(dtype=torch.bfloat16)
    return dgram


def make_backbone_rigid(
    positions: "geometry.Vec3Array",
    mask: torch.Tensor,
    group_indices: torch.Tensor,
) -> tuple["geometry.Rigid3Array", torch.Tensor]:
    """Make backbone Rigid3Array and mask.

    Args:
      positions: (num_res, num_atoms) of atom positions as Vec3Array.
      mask: (num_res, num_atoms) for atom mask.
      group_indices: (num_res, num_group, 3) for atom indices forming groups.

    Returns:
      tuple of backbone Rigid3Array and mask (num_res,).
    """
    backbone_indices = group_indices[:, 0]
    c, b, a = [backbone_indices[..., i] for i in range(3)]
    c, b, a = [x.to(dtype=torch.int64).unsqueeze(1) for x in [c, b, a]]

    rigid_mask = torch.gather(mask, 1, a).squeeze(1) \
        * torch.gather(mask, 1, b).squeeze(1) \
        * torch.gather(mask, 1, c).squeeze(1)

    frame_positions = []
    for indices in [a, b, c]:
        frame_positions.append(
            geometry.Vec3Array(
                x=torch.gather(positions.x, 1, indices).squeeze(1),
                y=torch.gather(positions.y, 1, indices).squeeze(1),
                z=torch.gather(positions.z, 1, indices).squeeze(1),
            )
        )

    rotation = geometry.Rot3Array.from_two_vectors(
        frame_positions[2] - frame_positions[1],
        frame_positions[0] - frame_positions[1],
    )
    rigid = geometry.Rigid3Array(rotation, frame_positions[1])
    return rigid, rigid_mask.to(dtype=torch.float32)


def safe_template_checkpoint(function, *args, **kwargs):
    """Non-reentrant checkpoint with autocast cache disabled to prevent metadata mismatch."""
    is_autocast = torch.is_autocast_enabled()
    autocast_dtype = torch.get_autocast_dtype("cuda") if is_autocast else torch.float32

    def _no_cache_autocast_ctx():
        if is_autocast:
            return torch.amp.autocast('cuda', enabled=True, dtype=autocast_dtype, cache_enabled=False)
        return contextlib.nullcontext()

    kwargs['use_reentrant'] = False
    kwargs['context_fn'] = lambda: (_no_cache_autocast_ctx(), _no_cache_autocast_ctx())
    return _original_checkpoint(function, *args, **kwargs)


class SingleTemplateEmbedding(nn.Module):
    """Embed a single template.

    All 9 template_pair_embedding_* attributes are preserved for af3.bin.zst
    strict state_dict load.
    """

    def __init__(self, num_channels: int = 64):
        super(SingleTemplateEmbedding, self).__init__()

        self.num_channels = num_channels
        self.template_stack_num_layer = 2

        self.dgram_features_config = DistogramFeaturesConfig()

        # Replaces fastnn.LayerNorm(128) — same API, pure-torch, eps=1e-5
        self.query_embedding_norm = LayerNorm(128)

        # All 9 separate linear embeddings — MUST match af3.bin.zst key names exactly
        self.template_pair_embedding_0 = nn.Linear(
            39, self.num_channels, bias=False)
        self.template_pair_embedding_1 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_2 = nn.Linear(
            31, self.num_channels, bias=False)
        self.template_pair_embedding_3 = nn.Linear(
            31, self.num_channels, bias=False)
        self.template_pair_embedding_4 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_5 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_6 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_7 = nn.Linear(
            1, self.num_channels, bias=False)
        self.template_pair_embedding_8 = nn.Linear(
            128, self.num_channels, bias=False)

        # 2× PairformerBlock (pair-only, num_intermediate_factor=2)
        self.template_embedding_iteration = nn.ModuleList(
            [PairformerBlock(c_pair=self.num_channels, num_intermediate_factor=2, with_single=False)
             for _ in range(self.template_stack_num_layer)]
        )

        # Replaces fastnn.LayerNorm(num_channels)
        self.output_layer_norm = LayerNorm(self.num_channels)

    def construct_input(
        self,
        query_embedding: torch.Tensor,
        templates: "features.Templates",
        multichain_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Build pair-wise template features for one template."""
        dtype = query_embedding.dtype

        aatype = templates.aatype
        dense_atom_mask = templates.atom_mask

        dense_atom_positions = templates.atom_positions
        dense_atom_positions = dense_atom_positions * dense_atom_mask[..., None]

        pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn(
            templates.aatype, dense_atom_positions, dense_atom_mask
        )
        pseudo_beta_mask_2d = (
            pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        )
        pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask_2d
        dgram = dgram_from_positions(
            pseudo_beta_positions, self.dgram_features_config
        )
        dgram = dgram * pseudo_beta_mask_2d[..., None]
        dgram = dgram.to(dtype=dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

        aatype = torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        ).to(dtype=dtype)
        to_concat.append((aatype[None, :, :], 1))
        to_concat.append((aatype[:, None, :], 1))

        template_group_indices = torch.take_along_dim(
            protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(device=templates.aatype.device),
            templates.aatype.to(dtype=torch.int64)[..., None, None],
            dim=0
        )

        rigid, backbone_mask = make_backbone_rigid(
            geometry.Vec3Array.from_array(dense_atom_positions),
            dense_atom_mask,
            template_group_indices.to(dtype=torch.int32),
        )

        points = rigid.translation

        rigid.rotation = geometry.Rot3Array(rigid.rotation.xx[:, None],
                                            rigid.rotation.xy[:, None],
                                            rigid.rotation.xz[:, None],
                                            rigid.rotation.yx[:, None],
                                            rigid.rotation.yy[:, None],
                                            rigid.rotation.yz[:, None],
                                            rigid.rotation.zx[:, None],
                                            rigid.rotation.zy[:, None],
                                            rigid.rotation.zz[:, None])
        rigid.translation = geometry.Vec3Array(rigid.translation.x[:, None],
                                               rigid.translation.y[:, None],
                                               rigid.translation.z[:, None])

        rigid_vec = rigid.inverse().apply_to_point(points)
        unit_vector = rigid_vec.normalized()
        unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

        unit_vector = [x.to(dtype=dtype) for x in unit_vector]
        backbone_mask = backbone_mask.to(dtype=dtype)

        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        backbone_mask_2d = backbone_mask_2d * multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))

        query_embedding = self.query_embedding_norm(query_embedding)

        to_concat.append((query_embedding, 1))

        act = 0

        for i, (x, n_input_dims) in enumerate(to_concat):
            if n_input_dims == 0:
                x = x[..., None]
            act = act + self.__getattr__(f'template_pair_embedding_{i}')(x)

        return act

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: "features.Templates",
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        is_checkpointed: bool = False,
    ) -> torch.Tensor:

        act = self.construct_input(
            query_embedding, templates, multichain_mask_2d)

        for pairformer_block in self.template_embedding_iteration:
            if is_checkpointed:
                def template_pairformer_forward(
                    pair_act: torch.Tensor,
                    pair_mask: torch.Tensor,
                    _block=pairformer_block,
                ) -> torch.Tensor:
                    return _block(pair_act, pair_mask=pair_mask)

                act = safe_template_checkpoint(
                    template_pairformer_forward,
                    act,
                    padding_mask_2d,
                )
            else:
                act = pairformer_block(act, pair_mask=padding_mask_2d)

        act = self.output_layer_norm(act)

        return act


class TemplateEmbedding(nn.Module):
    """Embed a set of templates.

    Uses SingleTemplateEmbedding above; output_linear attribute preserved for
    af3.bin.zst strict state_dict load.
    """

    def __init__(self, pair_channel: int = 128, num_channels: int = 64, disable_internal_progress: bool = False):
        super(TemplateEmbedding, self).__init__()

        self.pair_channel = pair_channel
        self.num_channels = num_channels
        self.disable_internal_progress = disable_internal_progress

        self.single_template_embedding = SingleTemplateEmbedding()

        self.output_linear = nn.Linear(
            self.num_channels, self.pair_channel, bias=False)

    def _embed_single_template(
        self,
        query_embedding: torch.Tensor,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        template_aatype: torch.Tensor,
        template_positions: torch.Tensor,
        template_mask: torch.Tensor,
        is_checkpointed: bool = False,
    ) -> torch.Tensor:
        template_i = features.Templates(
            aatype=template_aatype,
            atom_positions=template_positions,
            atom_mask=template_mask,
        )
        return self.single_template_embedding(
            query_embedding,
            template_i,
            padding_mask_2d,
            multichain_mask_2d,
            is_checkpointed=is_checkpointed,
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: "features.Templates",
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        is_checkpointed: bool = False,
    ) -> torch.Tensor:
        # embed EVERY template the pipeline passes, sum, divide by (1e-7 + num_templates).
        # No random selection / no per-template 0/1 weights (padded templates are zeroed
        # by the per-template mask inside SingleTemplateEmbedding).
        num_templates = templates.aatype.shape[0]

        summed_template_embeddings = query_embedding.new_zeros(
            query_embedding.shape[0], query_embedding.shape[1], self.num_channels
        )

        for template_idx in range(num_templates):
            template_embedding = self._embed_single_template(
                query_embedding,
                padding_mask_2d,
                multichain_mask_2d,
                templates.aatype[template_idx],
                templates.atom_positions[template_idx],
                templates.atom_mask[template_idx],
                is_checkpointed=is_checkpointed,
            )
            summed_template_embeddings = summed_template_embeddings + template_embedding

        embedding = summed_template_embeddings / (1e-7 + num_templates)

        embedding = torch.relu(embedding)

        embedding = self.output_linear(embedding)

        return embedding
