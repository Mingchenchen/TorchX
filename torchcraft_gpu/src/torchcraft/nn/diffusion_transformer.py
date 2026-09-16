from typing import Optional

import einops
import torch
import torch.nn as nn

from torchcraft.nn import atom_layout
from torchcraft.nn.gated_linear import gated_linear_unit
from torchcraft.nn.layer_norm import LayerNorm
from torchcraft.nn.product_attention import dot_product_attention


class AdaptiveLayerNorm(nn.Module):
    def __init__(self,
                 c_x: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:

        super(AdaptiveLayerNorm, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        if self.use_single_cond is True:
            self.layer_norm = LayerNorm(
                self.c_x, elementwise_affine=False, bias=False)
            self.single_cond_layer_norm = LayerNorm(
                self.c_single_cond, bias=False)
            self.single_cond_scale = nn.Linear(
                self.c_single_cond, self.c_x, bias=True)
            self.single_cond_bias = nn.Linear(
                self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = LayerNorm(self.c_x)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None) == (self.use_single_cond is False)

        if self.use_single_cond is True:
            x = self.layer_norm(x)
            single_cond = self.single_cond_layer_norm(single_cond)
            single_scale = self.single_cond_scale(single_cond)
            single_bias = self.single_cond_bias(single_cond)
            return torch.sigmoid(single_scale) * x + single_bias
        else:
            return self.layer_norm(x)


class AdaLNZero(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:
        super(AdaLNZero, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        self.transition2 = nn.Linear(self.c_in, self.c_out, bias=False)
        if self.use_single_cond is True:
            self.adaptive_zero_cond = nn.Linear(
                self.c_single_cond, self.c_out, bias=True)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None) == (self.use_single_cond is False)

        output = self.transition2(x)
        if self.use_single_cond is True:
            cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output
        return output


class DiffusionTransition(nn.Module):
    def __init__(self,
                 c_x: int,
                 c_single_cond: int,
                 num_intermediate_factor: int = 2,
                 use_single_cond: bool = False) -> None:
        super(DiffusionTransition, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_intermediate_factor = num_intermediate_factor
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)
        self.transition1 = nn.Linear(
            self.c_x, 2 * self.c_x * self.num_intermediate_factor, bias=False)
        self.transition1_weight_t = None  # cache transposed weight

        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond
        )

    def forward(self, x: torch.Tensor, single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.adaptive_layernorm(x, single_cond)
        if self.transition1_weight_t is None:
            self.transition1_weight_t = self.transition1.weight.T.contiguous()
        c = gated_linear_unit(x, self.transition1_weight_t)
        return self.adaptive_zero_init(c, single_cond)


class SelfAttentionTorch(nn.Module):
    def __init__(self,
                 c_x: int = 768,
                 c_single_cond: int = 384,
                 num_head: int = 16,
                 use_single_cond: bool = False,
                 is_decoder: bool = True) -> None:

        super(SelfAttentionTorch, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head
        self.is_decoder = is_decoder

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond)

        self.first_run = True
        self.padded_pair_logits = None

    def forward(self,
                x: torch.Tensor,  # variable
                mask: torch.Tensor,  # constant
                pair_logits: Optional[torch.Tensor] = None,  # constant for each SelfAttention
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """

        assert (single_cond is None) == (self.use_single_cond is False)

        BLK_SZ = 128 if x.shape[0] > 1536 else 64

        if self.first_run:
            self.first_run = False
            self.padded_pair_logits = pair_logits

        if not self.is_decoder:
            self.padded_pair_logits = pair_logits

        x = self.adaptive_layernorm(x, single_cond)

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)
        q, k, v = map(lambda t: einops.rearrange(
            t, 'n (h c) -> h n c', h=self.num_head).unsqueeze(0), [q, k, v])

        weighted_avg = dot_product_attention(
            q, k, v, mask=mask, bias=self.padded_pair_logits
        )

        weighted_avg = weighted_avg.squeeze(0)
        weighted_avg = einops.rearrange(weighted_avg, 'h q c -> q (h c)')

        gate_logits = self.gating_query(x)
        weighted_avg *= torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond)


SelfAttention = SelfAttentionTorch


class DiffusionTransformer(nn.Module):
    def __init__(self,
                 c_act: int = 768,
                 c_single_cond: int = 384,
                 c_pair_cond: int = 128,
                 num_head: int = 16,
                 num_blocks: int = 24,
                 super_block_size: int = 4) -> None:

        super(DiffusionTransformer, self).__init__()

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        self.pair_input_layer_norm = LayerNorm(self.c_pair_cond)
        self.pair_logits_projection = nn.ModuleList(
            [nn.Linear(self.c_pair_cond, self.super_block_size * self.num_head, bias=False) for _ in
             range(self.num_super_blocks)])

        self.self_attention = nn.ModuleList(
            [SelfAttention(self.c_act, self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])
        self.transition_block = nn.ModuleList(
            [DiffusionTransition(self.c_act, self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])

        self.first_run = True
        self.pair_logits_list = []

    def forward(self,
                act: torch.Tensor,  # variable
                mask: torch.Tensor,  # constant
                single_cond: torch.Tensor,  # variable
                pair_cond: torch.Tensor,  # constant
                ) -> torch.Tensor:
        if self.first_run:
            self.first_run = False
            pair_act = self.pair_input_layer_norm(pair_cond)

            for super_block_i in range(self.num_super_blocks):
                pair_logits = self.pair_logits_projection[super_block_i](pair_act)
                pair_logits = einops.rearrange(
                    pair_logits, 'n s (b h) -> b h n s', h=self.num_head)
                self.pair_logits_list.append(pair_logits.clone())

        for super_block_i in range(self.num_super_blocks):
            for j in range(self.super_block_size):
                act = act + self.self_attention[super_block_i * self.super_block_size + j](
                    act, mask, self.pair_logits_list[super_block_i][j, ...], single_cond)
                act = act + self.transition_block[super_block_i *
                                                  self.super_block_size + j](act, single_cond)

        return act


class CrossAttention(nn.Module):
    def __init__(self, key_dim: int = 128, value_dim: int = 128, c_single_cond: int = 128, num_head: int = 4) -> None:
        super(CrossAttention, self).__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.c_single_cond = c_single_cond
        self.num_head = num_head
        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        self.q_scale = nn.Parameter(torch.Tensor([self.key_dim_per_head ** (-0.5)]).to(torch.float32).cuda())

        self.q_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)
        self.k_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)

        self.q_projection = nn.Linear(self.key_dim, self.key_dim, bias=True)
        self.k_projection = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_projection = nn.Linear(
            self.value_dim, self.value_dim, bias=False)

        self.gating_query = nn.Linear(self.key_dim, self.value_dim, bias=False)
        self.adaptive_zero_init = AdaLNZero(
            self.value_dim, self.value_dim, self.key_dim, use_single_cond=True)

    def forward(
            self,
            x_q: torch.Tensor,
            x_k: torch.Tensor,
            mask_q: torch.Tensor,
            mask_k: torch.Tensor,
            pair_logits: Optional[torch.Tensor] = None,
            single_cond_q: Optional[torch.Tensor] = None,
            single_cond_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert len(mask_q.shape) == len(x_q.shape) - \
               1, f'{mask_q.shape}, {x_q.shape}'
        assert len(mask_k.shape) == len(x_k.shape) - \
               1, f'{mask_k.shape}, {x_k.shape}'

        combined_mask = mask_q.unsqueeze(-1).unsqueeze(-3) | mask_k.unsqueeze(-2).unsqueeze(-2)
        bias = 1e9 * combined_mask.logical_not()

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)

        q = self.q_projection(x_q)
        k = self.k_projection(x_k)
        q = torch.reshape(q, q.shape[:-1] +
                          (self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, k.shape[:-1] +
                          (self.num_head, self.key_dim_per_head))
        logits = torch.einsum('...qhc,...khc->...hqk',
                              q * self.q_scale, k) + bias
        if pair_logits is not None:
            logits = logits + pair_logits
        weights = torch.softmax(logits, axis=-1)

        v = self.v_projection(x_k)
        v = torch.reshape(v, v.shape[:-1] +
                          (self.num_head, self.value_dim_per_head))
        weighted_avg = torch.einsum('...hqk,...khc->...qhc', weights, v)
        weighted_avg = torch.reshape(
            weighted_avg, weighted_avg.shape[:-2] + (-1,))

        gate_logits = self.gating_query(x_q)
        weighted_avg *= torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond_q)


class DiffusionCrossAttTransformer(nn.Module):
    def __init__(self, c_query: int = 128, c_single_cond: int = 128, c_pair_cond: int = 16, num_blocks: int = 3,
                 num_head: int = 4) -> None:
        super(DiffusionCrossAttTransformer, self).__init__()

        self.c_query = c_query
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond

        self.num_blocks = num_blocks
        self.num_head = num_head

        self.pair_input_layer_norm = LayerNorm(self.c_pair_cond, bias=False)
        self.pair_logits_projection = nn.Linear(
            self.c_pair_cond, self.num_blocks * self.num_head, bias=False)

        self.cross_attention = nn.ModuleList(
            [CrossAttention(num_head=self.num_head) for _ in range(self.num_blocks)])

        self.transition_block = nn.ModuleList(
            [DiffusionTransition(c_x=self.c_query, c_single_cond=self.c_single_cond, use_single_cond=True) for _ in
             range(self.num_blocks)])

        self.first_run = True
        self.pair_logits = None

    def forward(
            self,
            queries_act: torch.Tensor,  # (num_subsets, num_queries, ch) # The only variable
            queries_mask: torch.Tensor,  # (num_subsets, num_queries)
            queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys)
            keys_mask: torch.Tensor,  # (num_subsets, num_keys)
            queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
            keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
            pair_cond: torch.Tensor,  # (num_subsets, num_queries, num_keys, ch)
    ) -> torch.Tensor:

        if self.first_run:
            self.first_run = False
            pair_act = self.pair_input_layer_norm(pair_cond)
            pair_logits = self.pair_logits_projection(pair_act)

            pair_logits = einops.rearrange(
                pair_logits, 'n q k (b h) -> b n h q k', h=self.num_head)

            self.pair_logits = pair_logits

        for block_idx in range(self.num_blocks):
            keys_act = atom_layout.convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )

            queries_act = queries_act + self.cross_attention[block_idx](
                x_q=queries_act,
                x_k=keys_act,
                mask_q=queries_mask,
                mask_k=keys_mask,
                pair_logits=self.pair_logits[block_idx, ...],
                single_cond_q=queries_single_cond,
                single_cond_k=keys_single_cond,
            )
            queries_act = queries_act + self.transition_block[block_idx](
                queries_act,
                queries_single_cond,
            )

        return queries_act
