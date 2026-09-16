from dataclasses import dataclass

import mx_driving
import torch
import torch.nn as nn
import torch_npu
import contextlib
from torch.utils.checkpoint import checkpoint as _original_checkpoint

from torchfold import fastnn
from torchfold import features, scoring
from torchfold import geometry
from torchfold.constants import residue_names
from torchfold.nn import pairformer
from torchfold.processing.atom_layout import rigid_utils


@dataclass
class DistogramFeaturesConfig:
    # The left edge of the first bin.
    min_bin: float = 3.25
    # The left edge of the final bin. The final bin catches everything larger than
    # `max_bin`.
    max_bin: float = 50.75
    # The number of bins in the distogram.
    num_bins: int = 39


def dgram_from_positions(positions, config: DistogramFeaturesConfig):
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
        positions: geometry.Vec3Array,
        mask: torch.Tensor,
        group_indices: torch.Tensor,
) -> tuple[geometry.Rigid3Array, torch.Tensor]:
    """Make backbone Rigid3Array and mask.

    Args:
      positions: (num_res, num_atoms) of atom positions as Vec3Array.
      mask: (num_res, num_atoms) for atom mask.
      group_indices: (num_res, num_group, 3) for atom indices forming groups.

    Returns:
      tuple of backbone Rigid3Array and mask (num_res,).
    """
    backbone_indices = group_indices[:, 0]

    # main backbone frames differ in sidechain frame convention.
    # for sidechain it's (C, CA, N), for backbone it's (N, CA, C)
    # Hence using c, b, a, each of shape (num_res,).
    backbone_indices = backbone_indices.to(dtype=torch.int64)
    c, b, a = torch.unbind(backbone_indices, dim=1)
    mask = mask.permute(1, 0)
    rigid_mask = (mx_driving.npu_index_select(mask, dim=0, index=a)
                  * mx_driving.npu_index_select(mask, dim=0, index=b)
                  * mx_driving.npu_index_select(mask, dim=0, index=c)).diag()
    frame_positions = []
    for indices in [a, b, c]:
        frame_positions.append(
            geometry.Vec3Array(
                x=mx_driving.npu_index_select(positions.x.permute(1, 0), dim=0, index=indices).diag(),
                y=mx_driving.npu_index_select(positions.y.permute(1, 0), dim=0, index=indices).diag(),
                z=mx_driving.npu_index_select(positions.z.permute(1, 0), dim=0, index=indices).diag(),
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
    is_autocast = torch_npu.npu.is_autocast_enabled()
    autocast_dtype = torch_npu.npu.get_autocast_dtype() if is_autocast else torch.float32

    def _no_cache_autocast_ctx():
        if is_autocast:
            return torch_npu.npu.amp.autocast(enabled=True, dtype=autocast_dtype, cache_enabled=False)
        return contextlib.nullcontext()

    kwargs['use_reentrant'] = False
    kwargs['context_fn'] = lambda: (_no_cache_autocast_ctx(), _no_cache_autocast_ctx())
    return _original_checkpoint(function, *args, **kwargs)


class TemplateEmbedding(nn.Module):
    """Embed a set of templates."""

    def __init__(self, pair_channel: int = 128, num_channels: int = 64, disable_internal_progress: bool = False):
        super(TemplateEmbedding, self).__init__()

        self.pair_channel = pair_channel
        self.num_channels = num_channels
        self.disable_internal_progress = disable_internal_progress

        self.single_template_embedding = SingleTemplateEmbedding()

        self.output_linear = nn.Linear(
            self.num_channels, self.pair_channel, bias=False)

    def _select_template_batch(
            self,
            templates: features.Templates,
            num_res: int,
            max_templates: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        total_templates = templates.aatype.shape[0]
        num_templates = min(total_templates, max_templates)

        selected_aatype = templates.aatype.new_zeros((max_templates, num_res))
        selected_positions = templates.atom_positions.new_zeros(
            (max_templates, num_res, 24, 3)
        )
        selected_mask = templates.atom_mask.new_zeros((max_templates, num_res, 24))
        template_weights = torch.zeros(
            (max_templates,), dtype=dtype, device=device
        )

        if num_templates > 0:
            perm = torch.randperm(total_templates, device=templates.aatype.device)
            selected_idx = perm[:num_templates]
            selected_aatype[:num_templates] = templates.aatype[selected_idx]
            selected_positions[:num_templates] = templates.atom_positions[selected_idx]
            selected_mask[:num_templates] = templates.atom_mask[selected_idx]
            template_weights[:num_templates] = 1.0

        return (
            selected_aatype,
            selected_positions,
            selected_mask,
            template_weights,
            num_templates,
        )

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
            templates: features.Templates,
            padding_mask_2d: torch.Tensor,
            multichain_mask_2d: torch.Tensor,
            is_checkpointed: bool = False,
    ) -> torch.Tensor:
        max_templates = 4
        num_res, _, _ = query_embedding.shape

        (
            template_aatype,
            template_positions,
            template_mask,
            template_weights,
            num_templates,
        ) = self._select_template_batch(
            templates=templates,
            num_res=num_res,
            max_templates=max_templates,
            dtype=query_embedding.dtype,
            device=query_embedding.device,
        )

        summed_template_embeddings = query_embedding.new_zeros(
            query_embedding.shape[0], query_embedding.shape[1], self.num_channels
        )

        for template_idx in range(template_aatype.shape[0]):
            template_embedding = self._embed_single_template(
                query_embedding,
                padding_mask_2d,
                multichain_mask_2d,
                template_aatype[template_idx],
                template_positions[template_idx],
                template_mask[template_idx],
                is_checkpointed=is_checkpointed,
            )

            summed_template_embeddings = (
                    summed_template_embeddings
                    + template_embedding * template_weights[template_idx]
            )

        embedding = summed_template_embeddings / max(num_templates, 1)

        embedding = torch.relu(embedding)

        embedding = self.output_linear(embedding)

        return embedding


class SingleTemplateEmbedding(nn.Module):
    """Embed a single template."""

    def __init__(self, num_channels: int = 64):
        super(SingleTemplateEmbedding, self).__init__()

        self.num_channels = num_channels
        self.template_stack_num_layer = 2

        self.dgram_features_config = DistogramFeaturesConfig()

        self.query_embedding_norm = fastnn.LayerNorm(128)
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

        self.template_embedding_iteration = nn.ModuleList(
            [pairformer.PairformerBlock(c_pair=self.num_channels, num_intermediate_factor=2, with_single=False)
             for _ in range(self.template_stack_num_layer)]
        )

        self.output_layer_norm = fastnn.LayerNorm(self.num_channels)

    def construct_input(
            self, query_embedding, templates: features.Templates, multichain_mask_2d
    ) -> torch.Tensor:
        # Compute distogram feature for the template.

        dtype = query_embedding.dtype

        aatype = templates.aatype
        dense_atom_mask = templates.atom_mask

        dense_atom_positions = templates.atom_positions
        dense_atom_positions = dense_atom_positions * dense_atom_mask.unsqueeze(-1)

        pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn(
            templates.aatype, dense_atom_positions, dense_atom_mask
        )
        pseudo_beta_mask_2d = (
                pseudo_beta_mask.unsqueeze(1) * pseudo_beta_mask.unsqueeze(0)
        )
        pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask_2d
        dgram = dgram_from_positions(
            pseudo_beta_positions, self.dgram_features_config
        )
        dgram = dgram * pseudo_beta_mask_2d.unsqueeze(-1)
        dgram = dgram.to(dtype=dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

        aatype = torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        ).to(dtype=dtype)
        to_concat.append((aatype.unsqueeze(0), 1))
        to_concat.append((aatype.unsqueeze(1), 1))

        # Compute a feature representing the normalized vector between each
        # backbone affine - i.e. in each residues local frame, what direction are
        # each of the other residues.

        template_group_indices = torch.take_along_dim(
            rigid_utils.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(device=templates.aatype.device),
            templates.aatype.to(dtype=torch.int64).unsqueeze(-1).unsqueeze(-1),
            dim=0
        )

        rigid, backbone_mask = make_backbone_rigid(
            geometry.Vec3Array.from_array(dense_atom_positions),
            dense_atom_mask,
            template_group_indices.to(dtype=torch.int32),
        )

        points = rigid.translation

        rigid.rotation = geometry.Rot3Array(rigid.rotation.xx.unsqueeze(-1),
                                            rigid.rotation.xy.unsqueeze(-1),
                                            rigid.rotation.xz.unsqueeze(-1),
                                            rigid.rotation.yx.unsqueeze(-1),
                                            rigid.rotation.yy.unsqueeze(-1),
                                            rigid.rotation.yz.unsqueeze(-1),
                                            rigid.rotation.zx.unsqueeze(-1),
                                            rigid.rotation.zy.unsqueeze(-1),
                                            rigid.rotation.zz.unsqueeze(-1))
        rigid.translation = geometry.Vec3Array(rigid.translation.x.unsqueeze(-1),
                                               rigid.translation.y.unsqueeze(-1),
                                               rigid.translation.z.unsqueeze(-1))

        rigid_vec = rigid.inverse().apply_to_point(points)
        unit_vector = rigid_vec.normalized()
        unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

        unit_vector = [x.to(dtype=dtype) for x in unit_vector]
        backbone_mask = backbone_mask.to(dtype=dtype)

        backbone_mask_2d = backbone_mask.unsqueeze(-1) * backbone_mask.unsqueeze(0)
        backbone_mask_2d = backbone_mask_2d * multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        # Note that the backbone_mask takes into account C, CA and N (unlike
        # pseudo beta mask which just needs CB) so we add both masks as features.
        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))

        query_embedding = self.query_embedding_norm(query_embedding)

        to_concat.append((query_embedding, 1))

        act = 0

        for i, (x, n_input_dims) in enumerate(to_concat):
            if n_input_dims == 0:
                x = x.unsqueeze(-1)
            act = act + self.__getattr__(f'template_pair_embedding_{i}')(x)

        return act

    def forward(
            self,
            query_embedding: torch.Tensor,
            templates: features.Templates,
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
