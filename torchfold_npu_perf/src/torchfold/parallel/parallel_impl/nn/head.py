from typing import Optional

import torch
import torch.nn as nn
import einops

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from torchfold import feat_batch
from torchfold.nn.confidence_utils import inference_output_offload_enabled
from torchx.constants import atom_types
from . import template, atom_layout, pairformer

from .. import fastnn
from ...parallel_ops import (
    pad_to_length,
    trim_to_global_length,
    make_pair_mask_row,
    all_gather_first_axis,
    gather_first_axis_to_rank0,
    transpose_col,
    transpose_col_local_chunk,
)

_CONTACT_THRESHOLD = 8.0
_CONTACT_EPSILON = 1e-3
from torchfold.runtime_policy import (
    DISTOGRAM_ROW_CHUNK_THRESHOLD as _DISTOGRAM_ROW_CHUNK_THRESHOLD,
    DISTOGRAM_ROW_CHUNK_SIZE as _DISTOGRAM_ROW_CHUNK_SIZE,
    distogram_row_chunk_size,
)


def _transpose_distance_logits_row(
    left_distance_logits_row: torch.Tensor,
    parallel_spec,
) -> torch.Tensor:
    """Return the local rows of the transposed distance logits."""
    return transpose_col(left_distance_logits_row.contiguous(), parallel_spec)


def _resolve_distogram_row_chunk_size(
    parallel_spec,
    *,
    original_sequence_length: int,
) -> int | None:
    """Use fixed Distogram chunks only for original lengths at least 8K."""
    return distogram_row_chunk_size(
        original_sequence_length=original_sequence_length,
        shard_size=parallel_spec.shard_size,
    )


class DistogramHead(nn.Module):
    def __init__(self,
                 c_pair: int = 128,
                 num_bins: int = 64,
                 first_break: float = 2.3125,
                 last_break: float = 21.6875) -> None:
        super(DistogramHead, self).__init__()

        self.c_pair = c_pair
        self.num_bins = num_bins
        self.first_break = first_break
        self.last_break = last_break

        self.half_logits = nn.Linear(self.c_pair, self.num_bins, bias=False)

        breaks = torch.linspace(
            self.first_break,
            self.last_break,
            self.num_bins - 1,
        )

        self.register_buffer('breaks', breaks)

        bin_tops = torch.cat(
            (breaks, (breaks[-1] + (breaks[-1] - breaks[-2])).reshape(1)))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)

        self.register_buffer('is_contact_bin', is_contact_bin)

    def forward(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pair_row = embeddings['pair']
        parallel_spec = embeddings.get('parallel_spec')
        if parallel_spec is None:
            raise ValueError(
                "Parallel DistogramHead requires embeddings['parallel_spec']."
            )

        seq_mask = batch.token_features.mask.to(dtype=torch.bool)
        pair_mask_row = make_pair_mask_row(seq_mask, parallel_spec)

        # Keep the 128-channel Pair row-sharded; exchange only 64-bin half logits.
        left_half_logits_row = self.half_logits(pair_row)
        seq_length = batch.token_features.seq_length
        if seq_length.numel() != 1:
            raise ValueError(
                "Distogram seq_length must be a scalar tensor, got shape "
                f"{tuple(seq_length.shape)}"
            )
        original_sequence_length = int(seq_length.item())
        chunk_size = _resolve_distogram_row_chunk_size(
            parallel_spec,
            original_sequence_length=original_sequence_length,
        )
        if chunk_size is None:
            right_half_logits_row = transpose_col(
                left_half_logits_row,
                parallel_spec,
            )
            logits_row = left_half_logits_row + right_half_logits_row
            del left_half_logits_row, right_half_logits_row

            probs_row = torch.softmax(logits_row, dim=-1)
            del logits_row
            contact_probs_row = torch.einsum(
                'ijb,b->ij',
                probs_row,
                self.is_contact_bin,
            )
            del probs_row
            contact_probs_row = pair_mask_row * contact_probs_row
        else:
            contact_probs_row = torch.empty(
                pair_mask_row.shape,
                dtype=left_half_logits_row.dtype,
                device=left_half_logits_row.device,
            )
            for row_start in range(
                0,
                parallel_spec.shard_size,
                chunk_size,
            ):
                row_end = min(
                    row_start + chunk_size,
                    parallel_spec.shard_size,
                )
                logits_chunk = transpose_col_local_chunk(
                    left_half_logits_row,
                    parallel_spec,
                    row_start,
                    row_end,
                )
                logits_chunk.add_(left_half_logits_row[row_start:row_end])
                probs_chunk = torch.softmax(logits_chunk, dim=-1)
                del logits_chunk

                contact_chunk = torch.einsum(
                    'ijb,b->ij',
                    probs_chunk,
                    self.is_contact_bin,
                )
                del probs_chunk
                contact_probs_row[row_start:row_end] = (
                    pair_mask_row[row_start:row_end] * contact_chunk
                )
                del contact_chunk
            del left_half_logits_row

        # Offloaded inference gathers [L, L] on rank 0; training/grad keeps all ranks.
        if inference_output_offload_enabled(training=self.training, parallel_spec=parallel_spec):
            contact_probs = gather_first_axis_to_rank0(
                contact_probs_row.contiguous(),
                parallel_spec,
            )
        else:
            contact_probs = all_gather_first_axis(
                contact_probs_row.contiguous(),
                parallel_spec,
            )
            contact_probs = trim_to_global_length(
                contact_probs, parallel_spec, dim=0
            )
            contact_probs = trim_to_global_length(
                contact_probs, parallel_spec, dim=1
            ).contiguous()

        return {
            'bin_edges': self.breaks,
            'contact_probs': contact_probs,
        }


class ConfidenceHead(nn.Module):
    def __init__(self, c_single: int = 384, c_pair: int = 128, c_target_feat: int = 447, n_pairformer_layers=4):
        super(ConfidenceHead, self).__init__()

        self.c_single = c_single
        self.c_pair = c_pair
        self.c_target_feat = c_target_feat

        self.dgram_features_config = template.DistogramFeaturesConfig()

        self.num_bins = 64
        self.max_error_bin = 31.0

        self.pae_num_bins = 64
        self.pae_max_error_bin = 31.0

        self.num_plddt_bins = 50
        self.num_atom = atom_types.DENSE_ATOM_NUM
        self.bin_width = 1.0 / self.num_plddt_bins

        self.left_target_feat_project = nn.Linear(
            self.c_target_feat, self.c_pair, bias=False)
        self.right_target_feat_project = nn.Linear(
            self.c_target_feat, self.c_pair, bias=False)
        self.distogram_feat_project = nn.Linear(
            self.dgram_features_config.num_bins, self.c_pair, bias=False)

        # Confidence shards Pair rows and replicates single features.
        self.confidence_pairformer = nn.ModuleList([
            pairformer.PairformerBlock(
                c_single=self.c_single,
                c_pair=self.c_pair,
                with_single=True,
                parallel_group=None,
            ) for _ in range(n_pairformer_layers)
        ])

        self.logits_ln = fastnn.LayerNorm(self.c_pair)
        self.left_half_distance_logits = nn.Linear(
            self.c_pair, self.num_bins, bias=False)

        self.register_buffer('distance_breaks', torch.linspace(
            0.0, self.max_error_bin, self.num_bins - 1))
        self.register_buffer(
            'step', self.distance_breaks[1] - self.distance_breaks[0])
        self.register_buffer(
            'bin_centers', self.distance_breaks + self.step / 2)
        self.bin_centers = torch.concatenate(
            [self.bin_centers, self.bin_centers[-1:] + self.step], dim=0
        )

        self.pae_logits_ln = fastnn.LayerNorm(self.c_pair)
        self.pae_logits = nn.Linear(self.c_pair, self.pae_num_bins, bias=False)

        self.register_buffer('pae_breaks', torch.linspace(
            0.0, self.pae_max_error_bin, self.pae_num_bins - 1))
        self.register_buffer(
            'pae_step', self.pae_breaks[1] - self.pae_breaks[0])

        pae_bin_centers_ = self.pae_breaks + self.pae_step / 2
        self.register_buffer(
            'pae_bin_centers', torch.concatenate(
                [pae_bin_centers_, pae_bin_centers_[-1:] + self.pae_step], dim=0
            )
        )

        self.register_buffer('plddt_bin_centers', torch.arange(
            0.5 * self.bin_width, 1.0, self.bin_width))

        self.plddt_logits_ln = fastnn.LayerNorm(self.c_single)
        self.plddt_logits = nn.Linear(
            self.c_single, self.num_atom * self.num_plddt_bins, bias=False)

        self.experimentally_resolved_ln = fastnn.LayerNorm(self.c_single)
        self.experimentally_resolved_logits = nn.Linear(
            self.c_single, self.num_atom * 2, bias=False)

    def _embed_features(
        self,
        dense_atom_positions: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:

        out = self.left_target_feat_project(target_feat).unsqueeze(-3) \
            + self.right_target_feat_project(target_feat).unsqueeze(-2)

        positions = atom_layout.convert(
            token_atoms_to_pseudo_beta,
            dense_atom_positions,
            layout_axes=(-3, -2),
        )

        dgram = template.dgram_from_positions(
            positions, self.dgram_features_config
        )

        dgram *= pair_mask.unsqueeze(-1)

        out += self.distogram_feat_project(dgram.to(torch.float32))

        return out

    def _embed_features_row_shard(
        self,
        dense_atom_positions: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        pair_mask_row: torch.Tensor,
        target_feat_full: torch.Tensor,
        parallel_spec,
    ) -> torch.Tensor:
        """
        Mathematically equivalent to the full _embed_features path, but only
        materializes local query rows.
        Output: [Ls, Lp, C]
        """
        target_feat_local = target_feat_full[parallel_spec.start:parallel_spec.end]

        # Match the full path: left(target_feat)[col] + right(target_feat)[row].
        out = self.left_target_feat_project(target_feat_full).unsqueeze(0) \
            + self.right_target_feat_project(target_feat_local).unsqueeze(1)

        positions = atom_layout.convert(
            token_atoms_to_pseudo_beta,
            dense_atom_positions,
            layout_axes=(-3, -2),
        )  # [L, 3]

        dgram_row = template.dgram_from_positions_row(
            positions, parallel_spec, self.dgram_features_config
        )
        dgram_row *= pair_mask_row.unsqueeze(-1)

        out += self.distogram_feat_project(dgram_row.to(torch.float32)).to(dtype=out.dtype)

        return out

    def _gather_square_output(
        self,
        x_row: torch.Tensor,
        parallel_spec,
    ) -> Optional[torch.Tensor]:
        if not inference_output_offload_enabled(training=self.training, parallel_spec=parallel_spec):
            x_full = all_gather_first_axis(x_row.contiguous(), parallel_spec)
            x_full = trim_to_global_length(x_full, parallel_spec, dim=0)
            x_full = trim_to_global_length(x_full, parallel_spec, dim=1)
            return x_full.contiguous()
        return gather_first_axis_to_rank0(
            x_row.contiguous(),
            parallel_spec,
        )

    def _all_reduce_scalar_in_group(self, x: torch.Tensor, parallel_spec) -> torch.Tensor:
        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and parallel_spec is not None
            and parallel_spec.world_size > 1
        ):
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=parallel_spec.group)
        return x

    def _get_tmscore_adjusted_pae(
        self,
        asym_id: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        bin_centers: torch.Tensor,
        pae_probs: torch.Tensor,
    ):

        def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
            clipped_num_res = torch.clamp(num_interface_tokens, min=19)

            d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

            d0 = d0.unsqueeze(2)
            bin_centers = bin_centers.unsqueeze(0).unsqueeze(0)

            tm_per_bin = 1.0 / \
                (1 + torch.square(bin_centers) / torch.square(d0))
            predicted_tm_term = torch.sum(pae_probs * tm_per_bin, dim=-1)
            return predicted_tm_term

        x = asym_id.unsqueeze(0) == asym_id.unsqueeze(1)
        num_chain_tokens = torch.sum(x * pair_mask, dim=-1, dtype=torch.int32)
        num_interface_tokens = num_chain_tokens.unsqueeze(0) + num_chain_tokens.unsqueeze(1)
        num_interface_tokens -= x * (num_interface_tokens // 2)
        num_interface_tokens = num_interface_tokens * pair_mask

        num_global_tokens = torch.ones(
            size=pair_mask.shape, dtype=torch.int32, device=x.device
        )
        num_global_tokens *= seq_mask.sum()

        assert num_global_tokens.dtype == torch.int32
        assert num_interface_tokens.dtype == torch.int32
        global_apae = get_tmscore_adjusted_pae(
            num_global_tokens, bin_centers, pae_probs
        )
        interface_apae = get_tmscore_adjusted_pae(
            num_interface_tokens, bin_centers, pae_probs
        )
        return global_apae, interface_apae

    def _get_tmscore_adjusted_pae_row(
        self,
        asym_id: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask_row: torch.Tensor,
        bin_centers: torch.Tensor,
        pae_probs_row: torch.Tensor,
        parallel_spec,
    ):
        """
        Mathematically equivalent to full _get_tmscore_adjusted_pae, restricted
        to local rows.
        Returns:
          global_apae_row:    [Ls, Lp]
          interface_apae_row: [Ls, Lp]
        """

        def get_tmscore_adjusted_pae_row(num_tokens_row, bin_centers, pae_probs_row):
            clipped_num_res = torch.clamp(num_tokens_row, min=19).to(dtype=pae_probs_row.dtype)
            d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

            d0 = d0.unsqueeze(-1)  # [Ls, Lp, 1]
            bin_centers = bin_centers.to(dtype=pae_probs_row.dtype).view(1, 1, -1)

            tm_per_bin = 1.0 / (1 + torch.square(bin_centers) / torch.square(d0))
            predicted_tm_term = torch.sum(pae_probs_row * tm_per_bin, dim=-1)
            return predicted_tm_term

        asym_id_full = pad_to_length(
            asym_id, dim=0, length=parallel_spec.n_padded, value=0
        )
        asym_id_local = asym_id_full[parallel_spec.start:parallel_spec.end]

        pair_mask_row_int = pair_mask_row.to(dtype=torch.int32)

        x_row = (asym_id_local.unsqueeze(1) == asym_id_full.unsqueeze(0))  # [Ls, Lp]

        num_chain_tokens_local = torch.sum(
            x_row.to(dtype=torch.int32) * pair_mask_row_int,
            dim=-1,
            dtype=torch.int32,
        )  # [Ls]

        num_chain_tokens_full = all_gather_first_axis(
            num_chain_tokens_local.contiguous(), parallel_spec
        )  # [Lp]

        num_interface_tokens_row = (
            num_chain_tokens_local.unsqueeze(1) + num_chain_tokens_full.unsqueeze(0)
        )
        num_interface_tokens_row -= x_row.to(dtype=torch.int32) * (num_interface_tokens_row // 2)
        num_interface_tokens_row = num_interface_tokens_row * pair_mask_row_int

        num_global_tokens_scalar = seq_mask.to(dtype=torch.int32).sum()
        num_global_tokens_row = torch.ones(
            size=pair_mask_row.shape, dtype=torch.int32, device=pair_mask_row.device
        ) * num_global_tokens_scalar

        global_apae_row = get_tmscore_adjusted_pae_row(
            num_global_tokens_row, bin_centers, pae_probs_row
        )
        interface_apae_row = get_tmscore_adjusted_pae_row(
            num_interface_tokens_row, bin_centers, pae_probs_row
        )

        return global_apae_row, interface_apae_row

    def forward(
        self,
        dense_atom_positions: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        asym_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        dtype = dense_atom_positions.dtype
        parallel_spec = embeddings.get('parallel_spec', None)

        if parallel_spec is None or parallel_spec.world_size == 1:
            seq_mask_cast = seq_mask.to(dtype=dtype)
            pair_mask = seq_mask_cast.unsqueeze(1) * seq_mask_cast.unsqueeze(0)
            pair_mask = pair_mask.to(dtype=dtype)

            pair_act = embeddings['pair'].clone().to(dtype=dtype)
            single_act = embeddings['single'].clone().to(dtype=dtype)
            target_feat = embeddings['target_feat'].clone().to(dtype=dtype)

            pair_act += self._embed_features(
                dense_atom_positions, token_atoms_to_pseudo_beta, pair_mask, target_feat)

            for layer in self.confidence_pairformer:
                pair_act, single_act = layer(
                    pair_row=pair_act,
                    pair_mask_row=pair_mask,
                    single=single_act,
                    seq_mask=seq_mask,
                    parallel_spec=None,
                )

            left_distance_logits = self.left_half_distance_logits(
                self.logits_ln(pair_act))
            right_distance_logits = left_distance_logits
            distance_logits = left_distance_logits + \
                torch.transpose(right_distance_logits, -2, -3).contiguous()

            distance_probs = torch.softmax(distance_logits, dim=-1)
            pred_distance_error = (
                torch.sum(distance_probs * self.bin_centers, dim=-1) * pair_mask
            )
            average_pred_distance_error = torch.sum(
                pred_distance_error, dim=[-2, -1]
            ) / torch.sum(pair_mask, dim=[-2, -1])

            pae_outputs = {}
            pae_logits = self.pae_logits(self.pae_logits_ln(pair_act))
            pae_probs = torch.softmax(pae_logits, dim=-1)

            pair_mask_bool = pair_mask.to(dtype=torch.bool)

            pae = torch.sum(pae_probs * self.pae_bin_centers,
                            dim=-1) * pair_mask_bool
            pae_outputs.update({
                'full_pae': pae,
            })

            tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = (
                self._get_tmscore_adjusted_pae(
                    asym_id=asym_id,
                    seq_mask=seq_mask,
                    pair_mask=pair_mask_bool,
                    bin_centers=self.pae_bin_centers,
                    pae_probs=pae_probs,
                )
            )

            pae_outputs.update({
                'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global,
                'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface,
            })

            plddt_logits = self.plddt_logits(self.plddt_logits_ln(single_act))
            plddt_logits = einops.rearrange(
                plddt_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=self.num_plddt_bins)
            predicted_lddt = torch.sum(
                torch.softmax(plddt_logits, dim=-1) * self.plddt_bin_centers, dim=-1
            )
            predicted_lddt = predicted_lddt * 100.0

            experimentally_resolved_logits = self.experimentally_resolved_logits(
                self.experimentally_resolved_ln(single_act))
            experimentally_resolved_logits = einops.rearrange(
                experimentally_resolved_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=2)

            predicted_experimentally_resolved = torch.softmax(
                experimentally_resolved_logits, dim=-1
            )[..., 1]

            return {
                'predicted_lddt': predicted_lddt,
                'predicted_experimentally_resolved': predicted_experimentally_resolved,
                'full_pde': pred_distance_error,
                'average_pde': average_pred_distance_error,
                **pae_outputs,
            }

        seq_mask_full = pad_to_length(
            seq_mask.to(dtype=dtype),
            dim=0,
            length=parallel_spec.n_padded,
            value=0.0,
        )
        pair_mask_row = make_pair_mask_row(seq_mask_full, parallel_spec).to(dtype=dtype)
        pair_mask_row_bool = pair_mask_row.to(dtype=torch.bool)

        # Always create an independent working shard for in-place updates.
        pair_act = embeddings['pair'].to(
            device=dense_atom_positions.device, dtype=dtype, copy=True,
        )  # [Ls, Lp, C]
        single_act = pad_to_length(
            embeddings['single'].clone().to(dtype=dtype),
            dim=0,
            length=parallel_spec.n_padded,
            value=0.0,
        )                                                            # Replicated single and target features both use [Lp, C].
        target_feat = pad_to_length(
            embeddings['target_feat'].clone().to(dtype=dtype),
            dim=0,
            length=parallel_spec.n_padded,
            value=0.0,
        )

        pair_act += self._embed_features_row_shard(
            dense_atom_positions=dense_atom_positions,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            pair_mask_row=pair_mask_row,
            target_feat_full=target_feat,
            parallel_spec=parallel_spec,
        )

        for layer in self.confidence_pairformer:
            if not self.training and not torch.is_grad_enabled():
                # Transfer only this sample's copy; preserve the original for later heads.
                pair_owner = [pair_act]
                del pair_act
                pair_act, single_act = layer(
                    pair_row=None, _pair_owner=pair_owner,
                    pair_mask_row=pair_mask_row,
                    single=single_act, seq_mask=seq_mask_full,
                    parallel_spec=parallel_spec,
                )
            else:
                pair_act, single_act = layer(
                    pair_row=pair_act,
                    pair_mask_row=pair_mask_row,
                    single=single_act,
                    seq_mask=seq_mask_full,
                    parallel_spec=parallel_spec,
                )

        left_distance_logits_row = self.left_half_distance_logits(
            self.logits_ln(pair_act)
        )  # [Ls, Lp, B]

        right_distance_logits_row = _transpose_distance_logits_row(
            left_distance_logits_row,
            parallel_spec,
        )  # [Ls, Lp, B]

        distance_logits_row = left_distance_logits_row + right_distance_logits_row
        distance_probs_row = torch.softmax(distance_logits_row, dim=-1)

        pred_distance_error_row = (
            torch.sum(distance_probs_row * self.bin_centers.to(dtype=distance_probs_row.dtype), dim=-1)
            * pair_mask_row
        )

        numerator = pred_distance_error_row.sum()
        denominator = pair_mask_row.sum()
        numerator = self._all_reduce_scalar_in_group(numerator, parallel_spec)
        denominator = self._all_reduce_scalar_in_group(denominator, parallel_spec)
        average_pred_distance_error = numerator / denominator

        pae_logits_row = self.pae_logits(self.pae_logits_ln(pair_act))
        pae_probs_row = torch.softmax(pae_logits_row, dim=-1)

        full_pae_row = torch.sum(
            pae_probs_row * self.pae_bin_centers.to(dtype=pae_probs_row.dtype),
            dim=-1,
        ) * pair_mask_row_bool

        tmscore_adjusted_pae_global_row, tmscore_adjusted_pae_interface_row = (
            self._get_tmscore_adjusted_pae_row(
                asym_id=asym_id,
                seq_mask=seq_mask,
                pair_mask_row=pair_mask_row_bool,
                bin_centers=self.pae_bin_centers,
                pae_probs_row=pae_probs_row,
                parallel_spec=parallel_spec,
            )
        )

        # Pairwise outputs are gathered to preserve the unsharded public interface.
        full_pde = self._gather_square_output(pred_distance_error_row, parallel_spec)
        full_pae = self._gather_square_output(full_pae_row.to(dtype=pred_distance_error_row.dtype), parallel_spec)
        tmscore_adjusted_pae_global = self._gather_square_output(
            tmscore_adjusted_pae_global_row, parallel_spec
        )
        tmscore_adjusted_pae_interface = self._gather_square_output(
            tmscore_adjusted_pae_interface_row, parallel_spec
        )

        # Single outputs are replicated, so only padding needs trimming.
        single_act = trim_to_global_length(single_act, parallel_spec, dim=0)

        plddt_logits = self.plddt_logits(self.plddt_logits_ln(single_act))
        plddt_logits = einops.rearrange(
            plddt_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=self.num_plddt_bins)
        predicted_lddt = torch.sum(
            torch.softmax(plddt_logits, dim=-1) * self.plddt_bin_centers, dim=-1
        )
        predicted_lddt = predicted_lddt * 100.0

        experimentally_resolved_logits = self.experimentally_resolved_logits(
            self.experimentally_resolved_ln(single_act))
        experimentally_resolved_logits = einops.rearrange(
            experimentally_resolved_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=2)

        predicted_experimentally_resolved = torch.softmax(
            experimentally_resolved_logits, dim=-1
        )[..., 1]

        return {
            'predicted_lddt': predicted_lddt,
            'predicted_experimentally_resolved': predicted_experimentally_resolved,
            'full_pde': full_pde,
            'average_pde': average_pred_distance_error,
            'full_pae': full_pae,
            'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global,
            'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface,
        }
