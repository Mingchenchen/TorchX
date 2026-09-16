import dataclasses
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchfold.model.modules import atom_layout
from torchfold.model import utils

from torchfold.model.triangular.layers import LayerNorm
from torchfold.model.modules.pairformer import (
    AdaptiveLayerNorm,
    AdaLNZero,
)


# ---------------------------------------------------------------------------
# Pure-torch gated linear unit — exact match to fastnn.gated_linear_unit
# (torch backend):  y = x @ weight_T;  a, b = chunk(y, 2);  silu(a) * b
# ---------------------------------------------------------------------------

def _gated_linear_unit_torch(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    """SiLU-gated linear unit.  Matches fastnn.gated_linear_unit (torch backend).

    Args:
        x: [..., K]
        weight_t: [K, 2*N] — pre-transposed weight from nn.Linear.weight.T
    Returns: [..., N]
    """
    y = torch.matmul(x, weight_t)
    a, b = torch.chunk(y, 2, dim=-1)
    return F.silu(a) * b


class DiffusionTransition(nn.Module):
    """Diffusion transition block with optional AdaLN conditioning.

    """

    def __init__(
        self,
        c_x: int,
        c_single_cond: int,
        num_intermediate_factor: int = 2,
        use_single_cond: bool = False,
    ) -> None:
        super(DiffusionTransition, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_intermediate_factor = num_intermediate_factor
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond
        )
        self.transition1 = nn.Linear(
            self.c_x, 2 * self.c_x * self.num_intermediate_factor, bias=False
        )
        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
        )

    def forward(
        self, x: torch.Tensor, single_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.adaptive_layernorm(x, single_cond)
        transition1_weight_t = self.transition1.weight.T.contiguous()
        c = _gated_linear_unit_torch(x, transition1_weight_t)
        return self.adaptive_zero_init(c, single_cond)


class CrossAttention(nn.Module):
    """Cross-attention with AdaLN conditioning on both queries and keys.

    """

    def __init__(
        self,
        key_dim: int = 128,
        value_dim: int = 128,
        c_single_cond: int = 128,
        num_head: int = 4,
    ) -> None:
        super(CrossAttention, self).__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        # q_scale buffer — kept for state_dict key compatibility
        self.register_buffer(
            "q_scale", torch.tensor([self.key_dim_per_head ** (-0.5)])
        )

        self.q_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True
        )
        self.k_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True
        )

        self.q_projection = nn.Linear(self.key_dim, self.key_dim, bias=True)
        self.k_projection = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_projection = nn.Linear(self.value_dim, self.value_dim, bias=False)

        self.gating_query = nn.Linear(self.key_dim, self.value_dim, bias=False)
        self.adaptive_zero_init = AdaLNZero(
            self.value_dim, self.value_dim, self.key_dim, use_single_cond=True
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        mask_q: torch.Tensor,
        mask_k: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
        single_cond_q: Optional[torch.Tensor] = None,
        single_cond_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(mask_q.shape) == len(x_q.shape) - 1, (
            f"{mask_q.shape}, {x_q.shape}"
        )
        assert len(mask_k.shape) == len(x_k.shape) - 1, (
            f"{mask_k.shape}, {x_k.shape}"
        )

        bias = (
            torch.tensor([1e9], dtype=x_q.dtype, device=x_q.device)
            * mask_q.logical_not()[..., None, :, None]
            * mask_k.logical_not()[..., None, None, :]
        )

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)

        q = self.q_projection(x_q)
        k = self.k_projection(x_k)
        q = torch.reshape(q, q.shape[:-1] + (self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, k.shape[:-1] + (self.num_head, self.key_dim_per_head))

        logits = torch.einsum("...qhc,...khc->...hqk", q * self.q_scale, k) + bias
        if pair_logits is not None:
            logits = logits + pair_logits
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(x_k)
        v = torch.reshape(v, v.shape[:-1] + (self.num_head, self.value_dim_per_head))
        weighted_avg = torch.einsum("...hqk,...khc->...qhc", weights, v)
        weighted_avg = torch.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))

        gate_logits = self.gating_query(x_q)
        weighted_avg = weighted_avg * torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond_q)


class DiffusionCrossAttTransformer(nn.Module):
    """Cross-attention transformer for atom-level features.

    """

    def __init__(
        self,
        c_query: int = 128,
        c_single_cond: int = 128,
        c_pair_cond: int = 16,
        num_blocks: int = 3,
        num_head: int = 4,
    ) -> None:
        super(DiffusionCrossAttTransformer, self).__init__()

        self.c_query = c_query
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_blocks = num_blocks
        self.num_head = num_head

        self.pair_input_layer_norm = LayerNorm(self.c_pair_cond, bias=False)
        self.pair_logits_projection = nn.Linear(
            self.c_pair_cond, self.num_blocks * self.num_head, bias=False
        )

        self.cross_attention = nn.ModuleList(
            [CrossAttention(num_head=self.num_head) for _ in range(self.num_blocks)]
        )
        self.transition_block = nn.ModuleList(
            [
                DiffusionTransition(
                    c_x=self.c_query,
                    c_single_cond=self.c_single_cond,
                    use_single_cond=True,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(
        self,
        queries_act: torch.Tensor,        # (num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,        # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,
        keys_mask: torch.Tensor,           # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor, # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,    # (num_subsets, num_keys, ch)
        pair_cond: torch.Tensor,           # (num_subsets, num_queries, num_keys, ch)
    ) -> torch.Tensor:

        pair_act = self.pair_input_layer_norm(pair_cond)
        pair_logits = self.pair_logits_projection(pair_act)
        pair_logits = einops.rearrange(
            pair_logits, "n q k (b h) -> b n h q k", h=self.num_head
        )

        if len(queries_mask.shape) == len(queries_act.shape) - 2:
            queries_mask = queries_mask.unsqueeze(0)
            keys_mask = keys_mask.unsqueeze(0)

        for block_idx in range(self.num_blocks):
            keys_act = atom_layout.convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )
            queries_act = queries_act + self.cross_attention[block_idx](
                x_q=queries_act,
                x_k=keys_act,
                mask_q=queries_mask,
                mask_k=keys_mask,
                pair_logits=pair_logits[block_idx, ...],
                single_cond_q=queries_single_cond,
                single_cond_k=keys_single_cond,
            )
            queries_act = queries_act + self.transition_block[block_idx](
                queries_act,
                queries_single_cond,
            )

        return queries_act


@dataclasses.dataclass(frozen=True)
class AtomCrossAttEncoderOutput:
    token_act: torch.Tensor             # (num_tokens, ch)
    skip_connection: torch.Tensor       # (num_subsets, num_queries, ch)
    queries_mask: torch.Tensor          # (num_subsets, num_queries)
    queries_single_cond: torch.Tensor   # (num_subsets, num_queries, ch)
    keys_mask: torch.Tensor             # (num_subsets, num_keys)
    keys_single_cond: torch.Tensor      # (num_subsets, num_keys, ch)
    pair_cond: torch.Tensor             # (num_subsets, num_queries, num_keys, ch)


class AtomCrossAttEncoder(nn.Module):
    """Atom cross-attention encoder.

    The legacy placeholder layers (single_to_pair_cond_row/col, embed_pair_offsets,
    embed_pair_distances — no _1 suffix) are kept frozen for checkpoint compatibility;
    they are never executed in forward().
    """

    def __init__(
        self,
        per_token_channels: int = 384,
        per_atom_channels: int = 128,
        per_atom_pair_channels: int = 16,
        with_token_atoms_act: bool = False,
        with_trunk_single_cond: bool = False,
        with_trunk_pair_cond: bool = False,
    ) -> None:
        super(AtomCrossAttEncoder, self).__init__()

        self.with_token_atoms_act = with_token_atoms_act
        self.with_trunk_single_cond = with_trunk_single_cond
        self.with_trunk_pair_cond = with_trunk_pair_cond

        self.c_positions = 3
        self.c_mask = 1
        self.c_element = 128
        self.c_charge = 1
        self.c_atom_name = 256
        self.c_pair_distance = 1
        self.per_token_channels = per_token_channels
        self.per_atom_channels = per_atom_channels
        self.per_atom_pair_channels = per_atom_pair_channels

        self.embed_ref_pos = nn.Linear(
            self.c_positions, self.per_atom_channels, bias=False
        )
        self.embed_ref_mask = nn.Linear(
            self.c_mask, self.per_atom_channels, bias=False
        )
        self.embed_ref_element = nn.Linear(
            self.c_element, self.per_atom_channels, bias=False
        )
        self.embed_ref_charge = nn.Linear(
            self.c_charge, self.per_atom_channels, bias=False
        )
        self.embed_ref_atom_name = nn.Linear(
            self.c_atom_name, self.per_atom_channels, bias=False
        )

        # ---- Legacy placeholder layers (checkpoint compat; never executed) ----
        self.single_to_pair_cond_row = nn.Linear(
            self.per_atom_channels, self.per_atom_pair_channels, bias=False
        )
        self.single_to_pair_cond_col = nn.Linear(
            self.per_atom_channels, self.per_atom_pair_channels, bias=False
        )
        self.embed_pair_offsets = nn.Linear(
            self.c_positions, self.per_atom_pair_channels, bias=False
        )
        self.embed_pair_distances = nn.Linear(
            self.c_pair_distance, self.per_atom_pair_channels, bias=False
        )
        for p in [
            self.single_to_pair_cond_row,
            self.single_to_pair_cond_col,
            self.embed_pair_offsets,
            self.embed_pair_distances,
        ]:
            p.weight.requires_grad_(False)
        # ---- End legacy layers ------------------------------------------------

        self.single_to_pair_cond_row_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False
        )
        self.single_to_pair_cond_col_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False
        )
        self.embed_pair_offsets_1 = nn.Linear(
            self.c_positions, self.per_atom_pair_channels, bias=False
        )
        self.embed_pair_distances_1 = nn.Linear(
            1, self.per_atom_pair_channels, bias=False
        )
        self.embed_pair_offsets_valid = nn.Linear(
            1, self.per_atom_pair_channels, bias=False
        )

        self.pair_mlp_1 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False
        )
        self.pair_mlp_2 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False
        )
        self.pair_mlp_3 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False
        )

        self.c_query = 128
        self.atom_transformer_encoder = DiffusionCrossAttTransformer(
            c_query=self.c_query
        )

        self.project_atom_features_for_aggr = nn.Linear(
            self.c_query, self.per_token_channels, bias=False
        )

        if self.with_trunk_single_cond is True:
            self.c_trunk_single_cond = 384
            self.lnorm_trunk_single_cond = LayerNorm(
                self.c_trunk_single_cond, bias=False
            )
            self.embed_trunk_single_cond = nn.Linear(
                self.c_trunk_single_cond, self.per_atom_channels, bias=False
            )

        if self.with_token_atoms_act is True:
            self.atom_positions_to_features = nn.Linear(
                self.c_positions, self.per_atom_channels, bias=False
            )

        if self.with_trunk_pair_cond is True:
            self.c_trunk_pair_cond = 128
            self.lnorm_trunk_pair_cond = LayerNorm(
                self.c_trunk_pair_cond, bias=False
            )
            self.embed_trunk_pair_cond = nn.Linear(
                self.c_trunk_pair_cond, self.per_atom_pair_channels, bias=False
            )

    def _per_atom_conditioning(self, batch) -> torch.Tensor:
        """Compute per-atom single conditioning features."""
        ref_mask = batch.ref_structure.mask
        ref_pos = batch.ref_structure.positions
        ref_pos = torch.nan_to_num(ref_pos, nan=0.0, posinf=0.0, neginf=0.0)
        ref_pos = torch.clamp(ref_pos, min=-1e4, max=1e4)
        ref_pos = ref_pos * ref_mask[:, :, None]
        act = self.embed_ref_pos(ref_pos)
        act = act + self.embed_ref_mask(
            ref_mask[:, :, None].to(dtype=self.embed_ref_mask.weight.dtype)
        )
        act = act + self.embed_ref_element(
            F.one_hot(
                batch.ref_structure.element.to(dtype=torch.int64), 128
            ).to(dtype=self.embed_ref_element.weight.dtype)
        )
        ref_charge = batch.ref_structure.charge
        ref_charge = torch.nan_to_num(ref_charge, nan=0.0, posinf=0.0, neginf=0.0)
        ref_charge = torch.clamp(ref_charge, min=-10.0, max=10.0)
        ref_charge = ref_charge * ref_mask
        act = act + self.embed_ref_charge(torch.arcsinh(ref_charge)[:, :, None])

        atom_name_chars_1hot = F.one_hot(
            batch.ref_structure.atom_name_chars.to(dtype=torch.int64), 64
        ).to(dtype=self.embed_ref_atom_name.weight.dtype)
        num_token, num_dense, _ = act.shape
        act = act + self.embed_ref_atom_name(
            atom_name_chars_1hot.reshape(num_token, num_dense, -1)
        )

        act = act * ref_mask[:, :, None]
        return act

    def forward(
        self,
        batch,
        token_atoms_act: Optional[torch.Tensor] = None,
        trunk_single_cond: Optional[torch.Tensor] = None,
        trunk_pair_cond: Optional[torch.Tensor] = None,
    ) -> AtomCrossAttEncoderOutput:

        assert (token_atoms_act is not None) == self.with_token_atoms_act
        assert (trunk_single_cond is not None) == self.with_trunk_single_cond
        assert (trunk_pair_cond is not None) == self.with_trunk_pair_cond

        token_atoms_mask = batch.predicted_structure_info.atom_mask
        token_atoms_single_cond = self._per_atom_conditioning(batch)

        queries_single_cond = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )
        queries_mask = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_mask,
            layout_axes=(-2, -1),
        )

        if trunk_single_cond is not None:
            trunk_single_cond = self.embed_trunk_single_cond(
                self.lnorm_trunk_single_cond(trunk_single_cond)
            )
            queries_single_cond = queries_single_cond + atom_layout.convert(
                batch.atom_cross_att.tokens_to_queries,
                trunk_single_cond,
                layout_axes=(-2,),
            )

        keys_single_cond = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        )
        keys_mask = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_mask,
            layout_axes=(-2, -1),
        )

        # Embed single features into pair conditioning
        row_act = self.single_to_pair_cond_row_1(torch.relu(queries_single_cond))
        pair_cond_keys_input = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        )
        col_act = self.single_to_pair_cond_col_1(torch.relu(pair_cond_keys_input))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        if trunk_pair_cond is not None:
            trunk_pair_cond = self.embed_trunk_pair_cond(
                self.lnorm_trunk_pair_cond(trunk_pair_cond)
            )
            num_tokens = trunk_pair_cond.shape[0]
            tokens_to_queries = batch.atom_cross_att.tokens_to_queries
            tokens_to_keys = batch.atom_cross_att.tokens_to_keys
            trunk_pair_to_atom_pair = atom_layout.GatherInfo(
                gather_idxs=(
                    num_tokens * tokens_to_queries.gather_idxs.unsqueeze(2)
                    + tokens_to_keys.gather_idxs.unsqueeze(1)
                ),
                gather_mask=(
                    tokens_to_queries.gather_mask.unsqueeze(2)
                    & tokens_to_keys.gather_mask.unsqueeze(1)
                ),
                input_shape=torch.tensor(
                    (num_tokens, num_tokens), device=torch.device("cpu")
                ),
            )
            pair_act = pair_act + atom_layout.convert(
                trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
            )

        # Embed pairwise offsets
        queries_ref_pos = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.positions,
            layout_axes=(-3, -2),
        )
        queries_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )
        keys_ref_pos = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_pos,
            layout_axes=(-3, -2),
        )
        keys_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_space_uid,
            layout_axes=(-2, -1),
        )
        offsets_valid = (
            queries_ref_space_uid.unsqueeze(-1) == keys_ref_space_uid.unsqueeze(1)
        )
        offsets_valid = (
            offsets_valid
            & queries_mask.unsqueeze(2).bool()
            & keys_mask.unsqueeze(1).bool()
        )
        offsets = queries_ref_pos.unsqueeze(2) - keys_ref_pos.unsqueeze(1)
        offsets = offsets * offsets_valid.unsqueeze(-1)

        pair_act = pair_act + (
            self.embed_pair_offsets_1(offsets) * offsets_valid.unsqueeze(-1)
        )
        sq_dists = torch.sum(torch.square(offsets), dim=-1)
        pair_act = pair_act + self.embed_pair_distances_1(
            1.0 / (1 + sq_dists.unsqueeze(-1))
        ) * offsets_valid.unsqueeze(-1)
        pair_act = pair_act + self.embed_pair_offsets_valid(
            offsets_valid.unsqueeze(-1).to(
                dtype=self.embed_pair_offsets_valid.weight.dtype
            )
        )

        # Small MLP on pair activations
        pair_act2 = self.pair_mlp_1(torch.relu(pair_act))
        pair_act2 = self.pair_mlp_2(torch.relu(pair_act2))
        pair_act = pair_act + self.pair_mlp_3(torch.relu(pair_act2))

        if token_atoms_act is None:
            queries_act = queries_single_cond.clone()
        else:
            queries_act = atom_layout.convert(
                batch.atom_cross_att.token_atoms_to_queries,
                token_atoms_act,
                layout_axes=(-3, -2),
            )
            queries_act = self.atom_positions_to_features(queries_act)
            queries_act = queries_act * queries_mask.unsqueeze(-1)
            queries_act = queries_act + queries_single_cond

        queries_act = self.atom_transformer_encoder(
            queries_act=queries_act,
            queries_mask=queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=keys_mask,
            queries_single_cond=queries_single_cond,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )

        queries_act = queries_act * queries_mask.unsqueeze(-1)
        skip_connection = queries_act.clone()

        queries_act = self.project_atom_features_for_aggr(queries_act)

        token_atoms_act = atom_layout.convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_act,
            layout_axes=(-3, -2),
        )

        if len(token_atoms_mask.shape) == (len(token_atoms_act.shape) - 2):
            token_atoms_mask = token_atoms_mask.unsqueeze(0)

        token_act = utils.mask_mean(
            token_atoms_mask.unsqueeze(-1),
            torch.relu(token_atoms_act),
            dim=-2,
        )

        return AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=skip_connection,
            queries_mask=queries_mask,
            queries_single_cond=queries_single_cond,
            keys_mask=keys_mask,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )


class AtomCrossAttDecoder(nn.Module):
    """Atom cross-attention decoder.

    """

    def __init__(self) -> None:
        super(AtomCrossAttDecoder, self).__init__()

        self.per_atom_channels = 128

        self.project_token_features_for_broadcast = nn.Linear(
            768, self.per_atom_channels, bias=False
        )
        self.atom_transformer_decoder = DiffusionCrossAttTransformer(
            c_query=self.per_atom_channels
        )
        self.atom_features_layer_norm = LayerNorm(self.per_atom_channels, bias=False)
        self.atom_features_to_position_update = nn.Linear(
            self.per_atom_channels, 3, bias=False
        )

    def forward(
        self,
        batch,
        token_act: torch.Tensor,   # (num_tokens, ch)
        enc: AtomCrossAttEncoderOutput,
    ) -> torch.Tensor:
        token_act = self.project_token_features_for_broadcast(token_act)
        num_token, max_atoms_per_token = (
            batch.atom_cross_att.queries_to_token_atoms.shape
        )
        if len(token_act.shape) == 3:
            token_atom_act = torch.broadcast_to(
                token_act.unsqueeze(2),
                (
                    token_act.size(0),
                    num_token,
                    max_atoms_per_token,
                    self.per_atom_channels,
                ),
            )
        else:
            token_atom_act = torch.broadcast_to(
                token_act.unsqueeze(1),
                (num_token, max_atoms_per_token, self.per_atom_channels),
            )
        queries_act = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atom_act,
            layout_axes=(-3, -2),
        )
        queries_mask_unsq = enc.queries_mask.unsqueeze(-1)
        queries_act = queries_act + enc.skip_connection
        queries_act = queries_act * queries_mask_unsq

        queries_act = self.atom_transformer_decoder(
            queries_act=queries_act,
            queries_mask=enc.queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=enc.keys_mask,
            queries_single_cond=enc.queries_single_cond,
            keys_single_cond=enc.keys_single_cond,
            pair_cond=enc.pair_cond,
        )

        queries_act = queries_act * queries_mask_unsq
        queries_act = self.atom_features_layer_norm(queries_act)
        queries_position_update = self.atom_features_to_position_update(queries_act)
        position_update = atom_layout.convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_position_update,
            layout_axes=(-3, -2),
        )
        return position_update
