import einops
import torch
from torchfold.runtime_policy import TRIANGLE_ATTENTION_CHUNK_SIZE
import torch.nn as nn
import torch.nn.functional as F

from torchfold.nn import fastnn_config
from torchfold.nn.layer_norm import LayerNorm
from torchfold.nn.product_attention import (
    FAMaskPlan,
    dot_product_attention,
    dot_product_attention_FA_sequence_major,
)


class GridSelfAttention(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttention, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)
        self.qkv_projection = nn.Linear(self.c_pair, self.c_pair*3, bias=False)
        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)

    def _project_fusion_attention_inputs(
        self,
        pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project pre-scaled BF16 Q/K/V in sequence-major layout."""
        if pair.dim() != 3 or pair.shape[-1] != self.c_pair:
            raise ValueError(
                "Triangle Attention input must have shape [B,S,c_pair]; "
                f"got input={tuple(pair.shape)}, c_pair={self.c_pair}"
            )

        weight = self.qkv_projection.weight
        expected_weight_shape = (3 * self.c_pair, self.c_pair)
        if tuple(weight.shape) != expected_weight_shape:
            raise RuntimeError(
                "qkv_projection weight has an incompatible shape: "
                f"expected={expected_weight_shape}, got={tuple(weight.shape)}"
            )
        q_weight, k_weight, v_weight = weight.split(self.c_pair, dim=0)

        bias = self.qkv_projection.bias
        if bias is None:
            q_bias = k_bias = v_bias = None
        else:
            expected_bias_shape = (3 * self.c_pair,)
            if tuple(bias.shape) != expected_bias_shape:
                raise RuntimeError(
                    "qkv_projection bias has an incompatible shape: "
                    f"expected={expected_bias_shape}, got={tuple(bias.shape)}"
                )
            q_bias, k_bias, v_bias = bias.split(self.c_pair, dim=0)

        # Project Q/K/V sequentially to limit FP32 temporaries; scale Q before casting.
        q = F.linear(pair, q_weight, q_bias)
        q.mul_(self.qkv_dim ** -0.5)
        q = q.to(dtype=torch.bfloat16).contiguous()

        k = F.linear(pair, k_weight, k_bias)
        k = k.to(dtype=torch.bfloat16).contiguous()

        v = F.linear(pair, v_weight, v_bias)
        v = v.to(dtype=torch.bfloat16).contiguous()
        return q, k, v

    def _attention(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor,
        bias: torch.Tensor,
        fa_mask_plan: FAMaskPlan | None = None,
    ) -> torch.Tensor:
        if (
            fastnn_config.dot_product_attention_implementation
            == "Fusion_Attention"
        ):
            q, k, v = self._project_fusion_attention_inputs(pair)
            weighted_avg = dot_product_attention_FA_sequence_major(
                q,
                k,
                v,
                self.num_head,
                mask=mask,
                bias=bias,
                mask_plan=fa_mask_plan,
            )
            del q, k, v
        else:
            # Preserve head-first layout and general masks for the torch backend.
            bsz, tgt_len, _ = pair.size()
            qkv = self.qkv_projection(pair)
            qkv = (
                qkv.view(bsz, tgt_len, 3, -1)
                .permute(2, 0, 1, 3)
                .contiguous()
            )
            q, k, v = qkv.unbind(0)
            q, k, v = map(
                lambda tensor: einops.rearrange(
                    tensor,
                    "b n (h d) -> b h n d",
                    h=self.num_head,
                ),
                (q, k, v),
            )
            weighted_avg = dot_product_attention(
                q,
                k,
                v,
                self.num_head,
                mask=mask,
                bias=bias,
                mask_plan=fa_mask_plan,
            )
            weighted_avg = einops.rearrange(
                weighted_avg,
                "b h n d -> b n (h d)",
            )

        gate_values = self.gating_query(pair)
        weighted_avg.mul_(torch.sigmoid(gate_values))
        return self.output_projection(weighted_avg)

    def add_residual_batch_chunked_(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
        chunk_size: int = TRIANGLE_ATTENTION_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Apply inference Triangle Attention in independent batch chunks."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "Batch-chunked Triangle Attention is inference-only; "
                "use eval() together with no_grad() or inference_mode()"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if pair.ndim != 3:
            raise ValueError(
                f"pair must have shape [N, N, C], got {tuple(pair.shape)}"
            )

        n_token, n_token_col, c_pair = pair.shape
        if n_token != n_token_col or c_pair != self.c_pair:
            raise ValueError(
                f"pair must have shape [N, N, {self.c_pair}], "
                f"got {tuple(pair.shape)}"
            )
        if mask is not None:
            if mask.dim() == 1:
                if mask.shape[0] != n_token:
                    raise ValueError(
                        f"one-dimensional mask must have shape [{n_token}], "
                        f"got {tuple(mask.shape)}"
                    )
            elif mask.dim() == 2:
                if tuple(mask.shape) != (n_token, n_token):
                    raise ValueError(
                        "two-dimensional mask must have shape "
                        f"[{n_token}, {n_token}], got {tuple(mask.shape)}"
                    )
            else:
                raise ValueError(
                    "mask must be one- or two-dimensional, "
                    f"got {tuple(mask.shape)}"
                )

        if n_token == 0:
            return pair
        # Share normalized input across bias and Q/K/V; keep residual updates separate.
        normalized_pair = self.act_norm(pair)
        pair_bias = self.pair_bias_projection(normalized_pair).permute(2, 0, 1).contiguous()
        use_fusion_attention = (
            fastnn_config.dot_product_attention_implementation
            == "Fusion_Attention"
        )
        if use_fusion_attention:
            pair_bias = pair_bias.to(dtype=torch.bfloat16).contiguous()
        valid_len = None
        if use_fusion_attention and mask is not None and mask.dim() == 2:
            valid_len = fastnn_config.single_card_valid_token_count()
            if valid_len is not None and not 0 <= valid_len <= n_token:
                raise ValueError(
                    "Single-card Grid static FA contract mismatch: "
                    f"valid_len={valid_len}, pair={tuple(pair.shape)}, "
                    f"mask={tuple(mask.shape)}"
                )

        for batch0 in range(0, n_token, chunk_size):
            batch1 = min(batch0 + chunk_size, n_token)
            active1 = batch1 if valid_len is None else min(batch1, valid_len)
            if active1 <= batch0:
                continue

            if self.transpose:
                pair_compute = normalized_pair[:, batch0:active1].permute(1, 0, 2).contiguous()
            else:
                pair_compute = normalized_pair[batch0:active1]

            if mask is None or mask.dim() == 1:
                mask_chunk = mask
            else:
                mask_chunk = mask[batch0:active1].contiguous()

            fa_mask_plan = None
            if valid_len is not None:
                fa_mask_plan = FAMaskPlan(
                    expected_batch=active1 - batch0,
                    expected_key_len=n_token,
                    key_valid_len=valid_len,
                    expected_mask_dim=2,
                    uniform_valid_rows=True,
                    all_query_rows_valid=True,
                )
            pair_update = self._attention(
                pair_compute,
                mask_chunk,
                pair_bias,
                fa_mask_plan=fa_mask_plan,
            )
            if self.transpose:
                pair[:, batch0:active1].add_(
                    pair_update.permute(1, 0, 2)
                )
            else:
                pair[batch0:active1].add_(pair_update)
        return pair

    def forward(self, pair, mask):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        pair_compute = pair
        nonbatched_bias = (
            self.pair_bias_projection(pair_compute)
            .permute(2, 0, 1)
            .contiguous()
        )

        if self.transpose:
            pair_compute = pair_compute.permute(1, 0, 2).contiguous()

        valid_len = None
        if (
            fastnn_config.dot_product_attention_implementation
            == "Fusion_Attention"
            and mask is not None
            and mask.dim() == 2
        ):
            valid_len = fastnn_config.single_card_valid_token_count()

        if valid_len is None:
            pair = self._attention(pair_compute, mask, nonbatched_bias)
        else:
            tgt_len = pair_compute.shape[1]
            if (
                not 0 <= valid_len <= pair_compute.shape[0]
                or tgt_len != mask.shape[1]
            ):
                raise ValueError(
                    "Single-card Grid static FA contract mismatch: "
                    f"valid_len={valid_len}, pair={tuple(pair_compute.shape)}, "
                    f"mask={tuple(mask.shape)}"
                )
            fa_mask_plan = FAMaskPlan(
                expected_batch=valid_len,
                expected_key_len=tgt_len,
                key_valid_len=valid_len,
                expected_mask_dim=2,
                uniform_valid_rows=True,
                all_query_rows_valid=True,
            )
            if valid_len == pair_compute.shape[0]:
                # The common no-padding case needs no output staging Tensor.
                pair = self._attention(
                    pair_compute,
                    mask,
                    nonbatched_bias,
                    fa_mask_plan=fa_mask_plan,
                )
            else:
                pair_out = torch.zeros_like(pair)
                if valid_len > 0:
                    pair_out[:valid_len] = self._attention(
                        pair_compute[:valid_len].contiguous(),
                        mask[:valid_len].contiguous(),
                        nonbatched_bias,
                        fa_mask_plan=fa_mask_plan,
                    )
                pair = pair_out

        if self.transpose:
            pair = pair.permute(1, 0, 2).contiguous()

        return pair


class MSAAttention(nn.Module):
    def __init__(self, c_msa=64, c_pair=128, num_head=8):
        super(MSAAttention, self).__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = LayerNorm(self.c_msa)
        self.pair_norm = LayerNorm(self.c_pair)
        self.pair_logits = nn.Linear(self.c_pair, self.num_head, bias=False)
        self.v_projection = nn.Linear(
            self.c_msa, self.num_head * self.value_dim, bias=False)
        self.gating_query = nn.Linear(self.c_msa, self.c_msa, bias=False)
        self.output_projection = nn.Linear(self.c_msa, self.c_msa, bias=False)

    def forward(self, msa, msa_mask, pair):
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1).contiguous()

        logits += 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head)

        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        gate_values = self.gating_query(msa)
        v_avg *= torch.sigmoid(gate_values)

        return self.output_projection(v_avg)
