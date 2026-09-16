from dataclasses import dataclass

import torch
import torch.nn as nn

from torchfold import features, scoring, protein_data_processing
from torchfold import geometry
from torchx.constants import residue_names
from . import pairformer

from .. import fastnn
from tqdm import trange
import mx_driving

from ...parallel_ops import (
    ParallelSpec,
    pad_to_length,
)


@dataclass
class DistogramFeaturesConfig:
    min_bin: float = 3.25
    max_bin: float = 50.75
    num_bins: int = 39


def dgram_from_positions(positions, config: DistogramFeaturesConfig):
    """Compute distogram from amino acid positions."""
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
        keepdim=True,
    )

    dgram = (dist2 > lower_breaks).to(dtype=torch.float32) * (
        dist2 < upper_breaks
    ).to(dtype=torch.float32)
    return dgram


def dgram_from_positions_row(
    positions: torch.Tensor,
    parallel_spec: ParallelSpec,
    config: DistogramFeaturesConfig,
) -> torch.Tensor:
    """
    positions: [L, 3]
    return   : [Ls, Lp, num_bins]
    """
    positions_full = pad_to_length(positions, dim=0, length=parallel_spec.n_padded, value=0.0)
    positions_local = positions_full[parallel_spec.start:parallel_spec.end]

    lower_breaks = torch.linspace(
        config.min_bin, config.max_bin, config.num_bins, device=positions.device)
    lower_breaks = torch.square(lower_breaks)
    upper_breaks_last = torch.ones(1, device=lower_breaks.device) * 1e8
    upper_breaks = torch.concatenate(
        [lower_breaks[1:], upper_breaks_last], dim=-1
    )

    dist2 = torch.sum(
        torch.square(
            positions_local.unsqueeze(-2) - positions_full.unsqueeze(-3)
        ),
        dim=-1,
        keepdim=True,
    )

    dgram = (dist2 > lower_breaks).to(dtype=torch.float32) * (
        dist2 < upper_breaks
    ).to(dtype=torch.float32)
    return dgram


def make_backbone_rigid(
    positions: geometry.Vec3Array,
    mask: torch.Tensor,
    group_indices: torch.Tensor,
) -> tuple[geometry.Rigid3Array, torch.Tensor]:
    """Make backbone Rigid3Array and mask."""
    backbone_indices = group_indices[:, 0]

    backbone_indices = backbone_indices.to(dtype=torch.int64)
    c, b, a = torch.unbind(backbone_indices, dim=1)

    mask = mask.transpose(0, 1)
    rigid_mask = (mx_driving.npu_index_select(mask, dim=0, index=a)
        * mx_driving.npu_index_select(mask, dim=0, index=b)
        * mx_driving.npu_index_select(mask, dim=0, index=c)).diag()

    frame_positions = []
    for indices in [a, b, c]:
        frame_positions.append(
            geometry.Vec3Array(
                x=mx_driving.npu_index_select(positions.x.transpose(0, 1), dim=0, index=indices).diag(),
                y=mx_driving.npu_index_select(positions.y.transpose(0, 1), dim=0, index=indices).diag(),
                z=mx_driving.npu_index_select(positions.z.transpose(0, 1), dim=0, index=indices).diag(),
            )
        )

    rotation = geometry.Rot3Array.from_two_vectors(
        frame_positions[2] - frame_positions[1],
        frame_positions[0] - frame_positions[1],
    )
    rigid = geometry.Rigid3Array(rotation, frame_positions[1])

    return rigid, rigid_mask.to(dtype=torch.float32)


class TemplateEmbedding(nn.Module):

    def __init__(self, pair_channel: int = 128, num_channels: int = 64):
        super(TemplateEmbedding, self).__init__()

        self.pair_channel = pair_channel
        self.num_channels = num_channels

        self.single_template_embedding = SingleTemplateEmbedding(
            num_channels=self.num_channels
        )

        self.output_linear = nn.Linear(
            self.num_channels, self.pair_channel, bias=False)

    def _forward_full(
        self,
        query_embedding: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        num_templates = templates.aatype.shape[0]
        num_res, _, _ = query_embedding.shape

        summed_template_embeddings = query_embedding.new_zeros(
            num_res, num_res, self.num_channels
        )

        for template_idx in trange(num_templates, desc="Temp Embed"):
            template_embedding = self.single_template_embedding(
                query_embedding,
                templates[template_idx],
                padding_mask_2d,
                multichain_mask_2d,
            )
            summed_template_embeddings += template_embedding

        embedding = summed_template_embeddings / (1e-7 + num_templates)
        embedding = torch.relu(embedding)
        embedding = self.output_linear(embedding)
        return embedding

    def forward(
        self,
        query_embedding_row: torch.Tensor,
        templates: features.Templates,
        padding_mask_row: torch.Tensor,
        multichain_mask_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        if parallel_spec is None or parallel_spec.world_size == 1:
            return self._forward_full(
                query_embedding=query_embedding_row,
                templates=templates,
                padding_mask_2d=padding_mask_row,
                multichain_mask_2d=multichain_mask_row,
            )

        total_templates = templates.aatype.shape[0]
        num_templates = total_templates
        num_res_local, num_res_full, _ = query_embedding_row.shape
        summed_template_embeddings = query_embedding_row.new_zeros(
            num_res_local, num_res_full, self.num_channels
        )

        for template_idx in trange(num_templates, desc="Temp Embed"):
            template_embedding = self.single_template_embedding(
                query_embedding_row,
                templates[template_idx],
                padding_mask_row,
                multichain_mask_row,
                parallel_spec=parallel_spec,
            )
            summed_template_embeddings += template_embedding

        embedding = summed_template_embeddings / (1e-7 + num_templates)
        embedding = torch.relu(embedding)
        embedding = self.output_linear(embedding)
        return embedding


class SingleTemplateEmbedding(nn.Module):
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
            [pairformer.PairformerBlock(
                c_pair=self.num_channels,
                num_intermediate_factor=2,
                with_single=False,
            ) for _ in range(self.template_stack_num_layer)]
        )

        self.output_layer_norm = fastnn.LayerNorm(self.num_channels)

    def _construct_input_full(
        self, query_embedding, templates: features.Templates, multichain_mask_2d
    ) -> torch.Tensor:
        dtype = query_embedding.dtype

        aatype = templates.aatype
        dense_atom_mask = templates.atom_mask

        dense_atom_positions = templates.atom_positions
        dense_atom_positions *= dense_atom_mask.unsqueeze(-1)

        pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn(
            templates.aatype, dense_atom_positions, dense_atom_mask
        )
        pseudo_beta_mask_2d = (
            pseudo_beta_mask.unsqueeze(1) * pseudo_beta_mask.unsqueeze(0)
        )
        pseudo_beta_mask_2d *= multichain_mask_2d
        dgram = dgram_from_positions(
            pseudo_beta_positions, self.dgram_features_config
        )
        dgram *= pseudo_beta_mask_2d.unsqueeze(-1)
        dgram = dgram.to(dtype=dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

        aatype = torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        ).to(dtype=dtype)
        to_concat.append((aatype.unsqueeze(0), 1))
        to_concat.append((aatype.unsqueeze(1), 1))

        template_group_indices = torch.take_along_dim(
            protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(
                device=templates.aatype.device
            ),
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

        backbone_mask_2d = backbone_mask.unsqueeze(1) * backbone_mask.unsqueeze(0)
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))

        query_embedding = self.query_embedding_norm(query_embedding)

        to_concat.append((query_embedding, 1))

        act = 0

        for i, (x, n_input_dims) in enumerate(to_concat):
            if n_input_dims == 0:
                x = x.unsqueeze(-1)
            act += self.__getattr__(f'template_pair_embedding_{i}')(x)

        return act

    def _construct_input_row(
        self,
        query_embedding_row: torch.Tensor,
        templates: features.Templates,
        multichain_mask_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        dtype = query_embedding_row.dtype
        ls = query_embedding_row.shape[0]
        lp = query_embedding_row.shape[1]
        l = templates.aatype.shape[0]
        local_valid = max(0, min(parallel_spec.end, l) - parallel_spec.start)

        aatype = templates.aatype
        dense_atom_mask = templates.atom_mask
        dense_atom_positions = templates.atom_positions
        dense_atom_positions = dense_atom_positions * dense_atom_mask.unsqueeze(-1)

        pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn(
            templates.aatype, dense_atom_positions, dense_atom_mask
        )

        pseudo_beta_mask_full = pad_to_length(
            pseudo_beta_mask.to(dtype=dtype), dim=0, length=lp, value=0.0
        )
        pseudo_beta_mask_local = pseudo_beta_mask_full[parallel_spec.start:parallel_spec.end]
        pseudo_beta_mask_row = pseudo_beta_mask_local.unsqueeze(1) * pseudo_beta_mask_full.unsqueeze(0)
        pseudo_beta_mask_row *= multichain_mask_row

        dgram = dgram_from_positions_row(
            pseudo_beta_positions, parallel_spec, self.dgram_features_config
        ).to(dtype=dtype)
        dgram *= pseudo_beta_mask_row.unsqueeze(-1)
        pseudo_beta_mask_row = pseudo_beta_mask_row.to(dtype=dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_row, 0)]

        aatype_1hot = torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        ).to(dtype=dtype)
        aatype_full = pad_to_length(aatype_1hot, dim=0, length=lp, value=0)
        aatype_local = aatype_full[parallel_spec.start:parallel_spec.end]
        to_concat.append((aatype_full.unsqueeze(0), 1))
        to_concat.append((aatype_local.unsqueeze(1), 1))

        template_group_indices = torch.take_along_dim(
            protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(
                device=templates.aatype.device
            ),
            templates.aatype.to(dtype=torch.int64).unsqueeze(-1).unsqueeze(-1),
            dim=0
        )

        rigid, backbone_mask = make_backbone_rigid(
            geometry.Vec3Array.from_array(dense_atom_positions),
            dense_atom_mask,
            template_group_indices.to(dtype=torch.int32),
        )

        backbone_mask_full = query_embedding_row.new_zeros(lp)
        backbone_mask_full[:backbone_mask.shape[0]] = backbone_mask.to(dtype=dtype)
        backbone_mask_local = backbone_mask_full[parallel_spec.start:parallel_spec.end]
        backbone_mask_row = backbone_mask_local.unsqueeze(1) * backbone_mask_full.unsqueeze(0)
        backbone_mask_row *= multichain_mask_row
        backbone_mask_row = backbone_mask_row.to(dtype=dtype)

        unit_x = query_embedding_row.new_zeros((ls, lp))
        unit_y = query_embedding_row.new_zeros((ls, lp))
        unit_z = query_embedding_row.new_zeros((ls, lp))

        if local_valid > 0:
            sl = slice(parallel_spec.start, parallel_spec.start + local_valid)
            rigid_local = geometry.Rigid3Array(
                geometry.Rot3Array(
                    rigid.rotation.xx[sl].unsqueeze(-1),
                    rigid.rotation.xy[sl].unsqueeze(-1),
                    rigid.rotation.xz[sl].unsqueeze(-1),
                    rigid.rotation.yx[sl].unsqueeze(-1),
                    rigid.rotation.yy[sl].unsqueeze(-1),
                    rigid.rotation.yz[sl].unsqueeze(-1),
                    rigid.rotation.zx[sl].unsqueeze(-1),
                    rigid.rotation.zy[sl].unsqueeze(-1),
                    rigid.rotation.zz[sl].unsqueeze(-1),
                ),
                geometry.Vec3Array(
                    rigid.translation.x[sl].unsqueeze(-1),
                    rigid.translation.y[sl].unsqueeze(-1),
                    rigid.translation.z[sl].unsqueeze(-1),
                )
            )
            points_full = geometry.Vec3Array(
                rigid.translation.x.unsqueeze(0),
                rigid.translation.y.unsqueeze(0),
                rigid.translation.z.unsqueeze(0),
            )
            rigid_vec = rigid_local.inverse().apply_to_point(points_full)
            unit_vector = rigid_vec.normalized()
            unit_x[:local_valid, :l] = unit_vector.x.to(dtype=dtype)
            unit_y[:local_valid, :l] = unit_vector.y.to(dtype=dtype)
            unit_z[:local_valid, :l] = unit_vector.z.to(dtype=dtype)

        unit_x *= backbone_mask_row
        unit_y *= backbone_mask_row
        unit_z *= backbone_mask_row

        to_concat.extend([(unit_x, 0), (unit_y, 0), (unit_z, 0)])
        to_concat.append((backbone_mask_row, 0))

        query_embedding_row = self.query_embedding_norm(query_embedding_row)
        to_concat.append((query_embedding_row, 1))

        act = 0
        for i, (x, n_input_dims) in enumerate(to_concat):
            if n_input_dims == 0:
                x = x.unsqueeze(-1)
            act += self.__getattr__(f'template_pair_embedding_{i}')(x)

        return act

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        parallel_spec: ParallelSpec = None,
    ) -> torch.Tensor:
        if parallel_spec is None or parallel_spec.world_size == 1:
            act = self._construct_input_full(
                query_embedding, templates, multichain_mask_2d
            )
            for pairformer_block in self.template_embedding_iteration:
                act = pairformer_block(act, pair_mask_row=padding_mask_2d)
        else:
            act = self._construct_input_row(
                query_embedding,
                templates,
                multichain_mask_2d,
                parallel_spec,
            )
            for pairformer_block in self.template_embedding_iteration:
                act = pairformer_block(
                    act,
                    pair_mask_row=padding_mask_2d,
                    parallel_spec=parallel_spec,
                )

        act = self.output_layer_norm(act)

        return act
