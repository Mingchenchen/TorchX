from typing import Optional

import einops
import torch
import torch.nn as nn

from torchfold.model.triangular.layers import LayerNorm, OuterProductMean, DropoutRowwise, DropoutColumnwise
from torchfold.model.modules.primitives import LinearNoBias, Transition
from torchfold.model.triangular.triangular import TriangleMultiplication


def _dot_product_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-torch attention: softmax((q*scale) @ k^T + bias + mask_fill) @ v.

    """
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    logits = torch.matmul(q, k.transpose(-1, -2))  # [B, H, N, N]

    if bias is not None:
        logits = logits + bias  # bias broadcasts [H, N, N] over [B, H, N, N]

    if mask is not None:
        if mask.dim() == 1:
            bool_mask = mask[None, None, None, :].to(dtype=torch.bool)
        elif mask.dim() == 2:
            bool_mask = mask[:, None, None, :].to(dtype=torch.bool)
        else:
            bool_mask = mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~bool_mask, -1e9)

    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, v)


def _dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    qkv_dims = q.dim()
    if qkv_dims == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    out = _dot_product_attention_torch(q, k, v, mask, bias)

    if qkv_dims == 3:
        out = out.squeeze(0)
    return out


import os as _os_pf
_USE_CUEQ_TRIATT = _os_pf.environ.get("TRIANGLE_ATTENTION", "").lower() == "cuequivariance"
_CUEQ_TRIATT_FN = None
def _cueq_triatt_fn():
    global _CUEQ_TRIATT_FN
    if _CUEQ_TRIATT_FN is None:
        from cuequivariance_torch.primitives.triangle import triangle_attention
        _CUEQ_TRIATT_FN = triangle_attention
    return _CUEQ_TRIATT_FN


class GridSelfAttention(nn.Module):
    """Grid (pair) self-attention with pair bias and gating.

    Args:
        c_pair: Pair representation channel dim. Default 128.
        num_head: Number of attention heads. Default 4.
        transpose: If True, transpose pair before attending (column attention).
    """

    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttention, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = LayerNorm(self.c_pair)
        self.pair_bias_projection = LinearNoBias(self.c_pair, self.num_head)

        self.q_projection = LinearNoBias(self.c_pair, self.c_pair)
        self.k_projection = LinearNoBias(self.c_pair, self.c_pair)
        self.v_projection = LinearNoBias(self.c_pair, self.c_pair)

        self.gating_query = LinearNoBias(self.c_pair, self.c_pair)
        self.output_projection = LinearNoBias(self.c_pair, self.c_pair)

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])

        if _USE_CUEQ_TRIATT and q.is_cuda:
            # cueq triangle_attention: q/k/v already [*, H, seq, C]; bias [H,N,N]
            # -> [1,H,N,N] (broadcast over row-batch); mask [N,N] -> bool key-mask
            # [N,1,1,N] (matches torch masked_fill on keys); scale = d**-0.5.
            scale = q.size(-1) ** -0.5
            cu_mask = None
            if mask is not None:
                m = mask.to(torch.bool)
                if m.dim() == 1:
                    cu_mask = m[None, None, None, :]
                elif m.dim() == 2:
                    cu_mask = m[:, None, None, :]
                else:
                    cu_mask = m
            weighted_avg = _cueq_triatt_fn()(
                q.contiguous(), k.contiguous(), v.contiguous(),
                bias[None].to(torch.float32).contiguous(), mask=cu_mask, scale=scale,
            )[0]
        else:
            weighted_avg = _dot_product_attention(q, k, v,
                                                  mask=mask,
                                                  bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        gate_values = self.gating_query(pair)

        weighted_avg = weighted_avg * torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        pair = self._attention(pair, mask, nonbatched_bias)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class MSAAttention(nn.Module):
    """MSA attention with pair-bias gating (no q/k, column softmax over seq).

    Args:
        c_msa: MSA channel dim. Default 64.
        c_pair: Pair channel dim. Default 128.
        num_head: Number of attention heads. Default 8.
    """

    def __init__(self, c_msa: int = 64, c_pair: int = 128, num_head: int = 8):
        super(MSAAttention, self).__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = LayerNorm(self.c_msa)
        self.pair_norm = LayerNorm(self.c_pair)
        self.pair_logits = LinearNoBias(self.c_pair, self.num_head)
        self.v_projection = LinearNoBias(
            self.c_msa, self.num_head * self.value_dim)
        self.gating_query = LinearNoBias(self.c_msa, self.c_msa)
        self.output_projection = LinearNoBias(self.c_msa, self.c_msa)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa (torch.Tensor): [N_seq, N_res, c_msa]
            msa_mask (torch.Tensor): [N_seq, N_res]
            pair (torch.Tensor): [N_res, N_res, c_pair]
        Returns:
            torch.Tensor: [N_seq, N_res, c_msa]
        """
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1)

        logits = logits + 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head)

        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        gate_values = self.gating_query(msa)
        v_avg = v_avg * torch.sigmoid(gate_values)

        return self.output_projection(v_avg)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm

    When use_single_cond=False: reduces to a plain LayerNorm (fastnn → vendored).
    When use_single_cond=True: conditions scale/bias on single_cond.
    """

    def __init__(self, c_x: int, c_single_cond: int, use_single_cond: bool = False) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        if self.use_single_cond:
            self.layer_norm = LayerNorm(self.c_x, elementwise_affine=False, bias=False)
            self.single_cond_layer_norm = LayerNorm(self.c_single_cond, bias=False)
            self.single_cond_scale = nn.Linear(self.c_single_cond, self.c_x, bias=True)
            self.single_cond_bias = nn.Linear(self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = LayerNorm(self.c_x)

    def forward(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (single_cond is None) == (not self.use_single_cond)
        if self.use_single_cond:
            x = self.layer_norm(x)
            single_cond = self.single_cond_layer_norm(single_cond)
            single_scale = self.single_cond_scale(single_cond)
            single_bias = self.single_cond_bias(single_cond)
            return torch.sigmoid(single_scale) * x + single_bias
        return self.layer_norm(x)


class AdaLNZero(nn.Module):
    """AdaLN-Zero output projection

    When use_single_cond=False: reduces to a plain linear (no adaptive gating).
    """

    def __init__(self, c_in: int, c_out: int, c_single_cond: int, use_single_cond: bool = False) -> None:
        super(AdaLNZero, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        self.transition2 = nn.Linear(self.c_in, self.c_out, bias=False)
        if self.use_single_cond:
            self.adaptive_zero_cond = nn.Linear(self.c_single_cond, self.c_out, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (single_cond is None) == (not self.use_single_cond)
        output = self.transition2(x)
        if self.use_single_cond:
            cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output
        return output


class SelfAttention(nn.Module):
    """Single self-attention with pair logit bias (Algorithm 24 in AF3).

    Args:
        c_x: Input/output channel dim. Default 768.
        c_single_cond: Conditioning dim (only used when use_single_cond=True). Default 384.
        num_head: Number of attention heads. Default 16.
        use_single_cond: If True, enable AdaLN conditioning. Default False.
        is_decoder: Passed through (not used in pure-torch path). Default True.
    """

    def __init__(
        self,
        c_x: int = 768,
        c_single_cond: int = 384,
        num_head: int = 16,
        use_single_cond: bool = False,
        is_decoder: bool = True,
    ) -> None:
        super(SelfAttention, self).__init__()
        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head
        self.is_decoder = is_decoder
        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond
        )
        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)
        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [N_token, c_x] (2-D) or [B, N_token, c_x] (3-D)
            mask (torch.Tensor): [N_token] (1-D)
            pair_logits (torch.Tensor, optional): [num_heads, N_token, N_token]
            single_cond (torch.Tensor, optional): conditioning; only when use_single_cond=True.
        Returns:
            torch.Tensor: same shape as x
        """
        assert (single_cond is None) == (not self.use_single_cond)

        x = self.adaptive_layernorm(x, single_cond)

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        if q.dim() == 2:
            # [N, C] → unsqueeze batch dim for _dot_product_attention
            q, k, v = [
                einops.rearrange(t, 'n (h c) -> h n c', h=self.num_head).unsqueeze(0)
                for t in (q, k, v)
            ]
            weighted_avg = _dot_product_attention(q, k, v, mask=mask, bias=pair_logits)
            weighted_avg = weighted_avg.squeeze(0)
        else:
            q, k, v = [
                einops.rearrange(t, '... n (h c) -> ... h n c', h=self.num_head)
                for t in (q, k, v)
            ]
            weighted_avg = _dot_product_attention(q, k, v, mask=mask, bias=pair_logits)

        weighted_avg = einops.rearrange(weighted_avg, '... h q c -> ... q (h c)')
        gate_logits = self.gating_query(x)
        weighted_avg = weighted_avg * torch.sigmoid(gate_logits)
        return self.adaptive_zero_init(weighted_avg, single_cond)


class PairformerBlock(nn.Module):
    """Pairformer block (Algorithm 17 lines 2-8 in AF3).

    Uses already-ported torchfold sub-modules (GridSelfAttention, TriangleMultiplication,
    Transition from primitives, SelfAttention above).
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_pair: int = 128,
        c_single: int = 384,
        c_hidden_mul: int = 128,
        n_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        with_single: bool = True,
        pair_dropout: float = 0.0,
    ) -> None:
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        self.with_single = with_single
        self.num_intermediate_factor = num_intermediate_factor

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair, _outgoing=True
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair, _outgoing=False
        )
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
        self.pair_transition = Transition(
            c_x=c_pair, num_intermediate_factor=self.num_intermediate_factor
        )
        self.drop_row = DropoutRowwise(pair_dropout)
        self.drop_col = DropoutColumnwise(pair_dropout)
        self.c_single = c_single
        if self.with_single:
            self.single_pair_logits_norm = LayerNorm(c_pair)
            self.single_pair_logits_projection = LinearNoBias(c_pair, n_heads)
            self.single_attention_ = SelfAttention(
                c_x=c_single, num_head=n_heads, use_single_cond=False, is_decoder=False
            )
            self.single_transition = Transition(c_x=self.c_single)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            pair (torch.Tensor): [..., N_token, N_token, c_pair]
            pair_mask (torch.Tensor): [..., N_token, N_token]
            single (torch.Tensor, optional): [..., N_token, c_single]
            seq_mask (torch.Tensor, optional): [..., N_token]

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: pair, single  (or just pair)
        """
        # AF3 SI Algorithm 17: every triangle op is a residual update with shared (row/col) dropout
        pair = pair + self.drop_row(self.triangle_multiplication_outgoing(pair, mask=pair_mask))
        pair = pair + self.drop_row(self.triangle_multiplication_incoming(pair, mask=pair_mask))
        pair = pair + self.drop_row(self.pair_attention1(pair, mask=pair_mask))
        pair = pair + self.drop_col(self.pair_attention2(pair, mask=pair_mask))
        pair = pair + self.pair_transition(pair)

        if self.with_single:
            pair_logits = self.single_pair_logits_projection(
                self.single_pair_logits_norm(pair)
            )
            pair_logits = pair_logits.permute(2, 0, 1)
            single = single + self.single_attention_(single, seq_mask, pair_logits=pair_logits)
            single = single + self.single_transition(single)
            return pair, single
        return pair


class EvoformerBlock(nn.Module):
    """Evoformer block (AF3 MSA stack).

    """

    def __init__(self, c_msa: int = 64, c_pair: int = 128, n_heads_pair: int = 4, pair_dropout: float = 0.0, msa_dropout: float = 0.0) -> None:
        super(EvoformerBlock, self).__init__()

        self.outer_product_mean = OuterProductMean(c_msa=c_msa, num_output_channel=c_pair)
        self.msa_attention1 = MSAAttention(c_msa=c_msa, c_pair=c_pair)
        self.msa_transition = Transition(c_x=c_msa)

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair, _outgoing=True
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair, _outgoing=False
        )
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
        self.pair_transition = Transition(c_x=c_pair)
        self.drop_row = DropoutRowwise(pair_dropout)
        self.drop_col = DropoutColumnwise(pair_dropout)
        self.drop_msa = DropoutRowwise(msa_dropout)

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple:
        """
        Args:
            msa (torch.Tensor): [N_seq, N_res, c_msa]
            pair (torch.Tensor): [N_res, N_res, c_pair]
            msa_mask (torch.Tensor): [N_seq, N_res]
            pair_mask (torch.Tensor): [N_res, N_res]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: msa, pair
        """
        pair = pair + self.outer_product_mean(msa, msa_mask)

        msa = msa + self.drop_msa(self.msa_attention1(msa, msa_mask, pair))
        msa = msa + self.msa_transition(msa)

        # AF3 SI Algorithm 17: every triangle op is a residual update with shared (row/col) dropout
        pair = pair + self.drop_row(self.triangle_multiplication_outgoing(pair, mask=pair_mask))
        pair = pair + self.drop_row(self.triangle_multiplication_incoming(pair, mask=pair_mask))
        pair = pair + self.drop_row(self.pair_attention1(pair, mask=pair_mask))
        pair = pair + self.drop_col(self.pair_attention2(pair, mask=pair_mask))
        pair = pair + self.pair_transition(pair)

        return msa, pair
