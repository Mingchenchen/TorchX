from typing import Optional

import einops
import torch
import torch.nn as nn

from torchfold.nn import atom_layout
from torchfold.nn import fastnn_config
from torchfold.nn.gated_linear import gated_linear_unit
from torchfold.nn.layer_norm import LayerNorm
from torchfold.nn.product_attention import (
    FAMaskPlan,
    dot_product_attention,
    dot_product_attention_FA_sequence_major,
)
from torchfold.runtime_policy import (
    DIFFUSION_PAIR_TRANSITION_ROW_STREAM_CHUNK_SIZE,
    DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
    DIFFUSION_PAIR_LOGITS_ROW_STREAM_CHUNK_SIZE,
    diffusion_pair_logits_row_streaming,
    native_pair_logits_cache_enabled,
)


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
        self.transition1_weight_t = None

        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond
        )

    def _forward_chunk(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.adaptive_layernorm(x, single_cond)
        if self.transition1_weight_t is None:
            self.transition1_weight_t = self.transition1.weight.T.contiguous()
        c = gated_linear_unit(x, self.transition1_weight_t)
        return self.adaptive_zero_init(c, single_cond)

    def add_pair_residual_row_streaming_(
        self,
        x: torch.Tensor,
        chunk_size: int = DIFFUSION_PAIR_TRANSITION_ROW_STREAM_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Apply an inference-only pair-conditioning residual by row."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "diffusion pair residual row streaming is inference-only"
            )
        if self.use_single_cond:
            raise RuntimeError(
                "diffusion pair residual row streaming does not accept "
                "single conditioning"
            )
        if x.ndim != 3:
            raise ValueError(
                "diffusion pair conditioning expects [rows, columns, channels], "
                f"got shape={tuple(x.shape)}"
            )
        if chunk_size <= 0:
            raise ValueError(
                "diffusion pair row-stream chunk size must be positive"
            )

        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            residual_chunk = x[start:end]
            compute_chunk = residual_chunk
            if not compute_chunk.is_contiguous():
                compute_chunk = compute_chunk.contiguous()
            update_chunk = self._forward_chunk(compute_chunk)
            residual_chunk.add_(update_chunk)
            del residual_chunk, compute_chunk, update_chunk
        return x

    def forward(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_chunk(x, single_cond)


class SelfAttention(nn.Module):
    def __init__(self,
                 c_x: int = 768,
                 c_single_cond: int = 384,
                 num_head: int = 16,
                 use_single_cond: bool = False,
                 is_decoder: bool = True) -> None:

        super(SelfAttention, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head
        self.is_decoder = is_decoder

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)

        self.qkv_projection = nn.Linear(self.c_x, self.c_x*3, bias=True)
        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond)

    def _attention_fusion_bsh(
        self,
        qkv: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: Optional[torch.Tensor],
        fa_mask_plan: Optional[FAMaskPlan] = None,
    ) -> torch.Tensor:
        """Run Diffusion SelfAttention without materializing BNSD tensors.

        The packed projection is kept unchanged for checkpoint and kernel
        compatibility.  Its Q/K/V row blocks are consumed as sequence-major
        ``[B,S,C]`` views, scaled/cast into final BF16 buffers, and passed to
        Fusion Attention with ``input_layout=\"BSH\"``.
        """
        input_was_unbatched = qkv.dim() == 2
        if input_was_unbatched:
            qkv = qkv.unsqueeze(0)
        elif qkv.dim() != 3:
            raise ValueError(
                "Diffusion BSH SelfAttention expects [N,3C] or [B,N,3C], "
                f"got shape={tuple(qkv.shape)}"
            )
        if qkv.size(-1) != 3 * self.c_x:
            raise ValueError(
                "Diffusion BSH packed projection has incompatible hidden size: "
                f"expected={3 * self.c_x}, got={qkv.size(-1)}"
            )

        q, k, v = torch.split(qkv, self.c_x, dim=-1)
        # Scale Q once in FP32; the FA helper uses scale=1.0.
        q = (q * (self.qkv_dim ** -0.5)).to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()
        weighted_avg = dot_product_attention_FA_sequence_major(
            q,
            k,
            v,
            self.num_head,
            mask=mask,
            bias=pair_logits,
            mask_plan=fa_mask_plan,
        )
        if input_was_unbatched:
            weighted_avg = weighted_avg.squeeze(0)
        return weighted_avg

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """

        assert (single_cond is None) == (self.use_single_cond is False)

        x = self.adaptive_layernorm(x, single_cond)
        qkv = self.qkv_projection(x)
        fa_mask_plan = None
        if (
            fastnn_config.dot_product_attention_implementation
            == "Fusion_Attention"
            and mask is not None
            and mask.dim() == 1
        ):
            valid_len = fastnn_config.single_card_valid_token_count()
            if valid_len is not None:
                token_length = qkv.shape[-2]
                if not 0 <= valid_len <= token_length:
                    raise ValueError(
                        "Single-card SelfAttention static FA contract mismatch: "
                        f"valid_len={valid_len}, token_length={token_length}"
                    )
                fa_mask_plan = FAMaskPlan(
                    expected_batch=qkv.shape[0] if qkv.ndim == 3 else 1,
                    expected_key_len=token_length,
                    key_valid_len=valid_len,
                    expected_mask_dim=1,
                )
        use_bsh = (
            fastnn_config.dot_product_attention_implementation
            == "Fusion_Attention"
        )
        if use_bsh:
            weighted_avg = self._attention_fusion_bsh(
                qkv,
                mask,
                pair_logits,
                fa_mask_plan=fa_mask_plan,
            )
        elif qkv.ndim == 3:
            batch_size, num_tokens = qkv.shape[:2]
            qkv = qkv.reshape(batch_size, num_tokens, 3, -1)
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            q, k, v = qkv.unbind(dim=1)
            q, k, v = map(
                lambda tensor: einops.rearrange(
                    tensor,
                    'b n (h c) -> b h n c',
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
                bias=pair_logits,
                mask_plan=fa_mask_plan,
            )
            weighted_avg = einops.rearrange(
                weighted_avg,
                'b h q c -> b q (h c)',
            )
        elif qkv.ndim == 2:
            qkv = qkv.view(x.shape[0], 3, -1).permute(1, 0, 2).contiguous()
            q, k, v = qkv.unbind(0)
            q, k, v = map(
                lambda tensor: einops.rearrange(
                    tensor,
                    'n (h c) -> h n c',
                    h=self.num_head,
                ).unsqueeze(0),
                (q, k, v),
            )
            weighted_avg = dot_product_attention(
                q,
                k,
                v,
                self.num_head,
                mask=mask,
                bias=pair_logits,
                mask_plan=fa_mask_plan,
            )
            weighted_avg = einops.rearrange(
                weighted_avg.squeeze(0),
                'h q c -> q (h c)',
            )
        else:
            raise ValueError(
                "Diffusion SelfAttention expects [N,C] or [B,N,C], "
                f"got shape={tuple(x.shape)}"
            )
        gate_logits = self.gating_query(x)
        weighted_avg *= torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond)


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
            [
                nn.Linear(
                    self.c_pair_cond,
                    self.super_block_size * self.num_head,
                    bias=False,
                )
                for _ in range(self.num_super_blocks)
            ]
        )

        self.self_attention = nn.ModuleList(
            [
                SelfAttention(
                    self.c_act,
                    self.c_single_cond,
                    use_single_cond=True,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.transition_block = nn.ModuleList(
            [
                DiffusionTransition(
                    self.c_act,
                    self.c_single_cond,
                    use_single_cond=True,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Prediction-only cache, excluded from state_dict and cleared after denoising.
        self._pair_logits_cache: Optional[list[torch.Tensor]] = None
        self._pair_logits_cache_key = None
        self._pair_logits_cache_source: Optional[torch.Tensor] = None
        self._pair_logits_cache_row_stream = False

    def _make_pair_logits_cache_key(
        self,
        pair_cond: torch.Tensor,
        cache_layout: str,
        cache_dtype: str,
    ) -> tuple:
        return (
            pair_cond.data_ptr(),
            tuple(pair_cond.shape),
            tuple(pair_cond.stride()),
            pair_cond.dtype,
            pair_cond.device,
            cache_layout,
            fastnn_config.dot_product_attention_implementation,
            cache_dtype,
        )

    def uses_row_stream_pair_logits_cache(
        self,
        pair_cond: torch.Tensor,
    ) -> bool:
        return diffusion_pair_logits_row_streaming(
            feature_length=pair_cond.shape[-2], training=self.training,
            grad_enabled=torch.is_grad_enabled(),
            attention_implementation=fastnn_config.dot_product_attention_implementation,
            threshold=DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
        )

    def clear_pair_logits_cache(self) -> None:
        self._pair_logits_cache = None
        self._pair_logits_cache_key = None
        self._pair_logits_cache_source = None
        self._pair_logits_cache_row_stream = False

    def has_pair_logits_cache(self) -> bool:
        return self._pair_logits_cache is not None

    def release_pair_logits_cache_source(self) -> None:
        self._pair_logits_cache_source = None

    def _project_pair_logits(
        self,
        pair_act: torch.Tensor,
        super_block_i: int,
        cache_dtype: str = "native",
    ) -> torch.Tensor:
        pair_logits = self.pair_logits_projection[super_block_i](pair_act)
        pair_logits = einops.rearrange(
            pair_logits,
            'n s (b h) -> b h n s',
            h=self.num_head,
        )
        if cache_dtype == "bf16" and pair_logits.dtype != torch.bfloat16:
            pair_logits = pair_logits.to(torch.bfloat16)
        return pair_logits

    def _build_row_stream_pair_logits_cache(
        self,
        pair_cond: torch.Tensor,
        chunk_size: int = DIFFUSION_PAIR_LOGITS_ROW_STREAM_CHUNK_SIZE,
    ) -> list[torch.Tensor]:
        """Build one contiguous BF16 attention bias per diffusion block."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "row-stream pair-logits caching is inference-only"
            )
        if (
            fastnn_config.dot_product_attention_implementation
            != "Fusion_Attention"
        ):
            raise RuntimeError(
                "row-stream BF16 pair-logits caching requires Fusion Attention"
            )
        if pair_cond.ndim != 3:
            raise ValueError(
                "diffusion pair conditioning expects [rows, columns, channels], "
                f"got shape={tuple(pair_cond.shape)}"
            )
        num_rows, num_columns, num_channels = pair_cond.shape
        if num_rows != num_columns:
            raise ValueError(
                "diffusion pair conditioning must be square, "
                f"got shape={tuple(pair_cond.shape)}"
            )
        if num_channels != self.c_pair_cond:
            raise ValueError(
                "diffusion pair conditioning has an incompatible channel size: "
                f"expected={self.c_pair_cond}, got={num_channels}"
            )
        if chunk_size <= 0:
            raise ValueError(
                "diffusion pair-logits row chunk size must be positive"
            )

        # Allocate per attention block to avoid a large super-block allocation.
        cache = [
            torch.empty(
                (self.num_head, num_rows, num_columns),
                dtype=torch.bfloat16,
                device=pair_cond.device,
            )
            for _ in range(self.num_blocks)
        ]

        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            pair_act_chunk = self.pair_input_layer_norm(
                pair_cond[start:end]
            )
            rows_in_chunk = end - start
            for super_block_i, projection in enumerate(
                self.pair_logits_projection
            ):
                projected_chunk = projection(pair_act_chunk).reshape(
                    rows_in_chunk,
                    num_columns,
                    self.super_block_size,
                    self.num_head,
                )
                block_base = super_block_i * self.super_block_size
                for block_offset in range(self.super_block_size):
                    prepared_bias = projected_chunk[
                        :, :, block_offset, :
                    ].permute(2, 0, 1).to(torch.bfloat16).contiguous()
                    cache[block_base + block_offset][
                        :, start:end, :
                    ].copy_(prepared_bias)
                    del prepared_bias
                del projected_chunk
            del pair_act_chunk
            # Drain row workspaces retained by TASK_QUEUE_ENABLE=2.
            if pair_cond.device.type == "npu":
                torch.npu.synchronize()
        return cache

    def _get_or_build_pair_logits_cache(
        self,
        pair_cond: torch.Tensor,
        row_stream: bool = False,
    ) -> list[torch.Tensor]:
        cache_dtype = (
            "bf16"
            if fastnn_config.dot_product_attention_implementation
            == "Fusion_Attention"
            else "native"
        )
        cache_layout = (
            "per_block_contiguous_bf16_v1"
            if row_stream
            else "superblock_v1"
        )
        cache_key = self._make_pair_logits_cache_key(
            pair_cond,
            cache_layout,
            cache_dtype,
        )
        if (
            self._pair_logits_cache is not None
            and self._pair_logits_cache_source is pair_cond
            and self._pair_logits_cache_key == cache_key
        ):
            return self._pair_logits_cache

        self.clear_pair_logits_cache()
        if row_stream:
            cache = self._build_row_stream_pair_logits_cache(pair_cond)
        else:
            pair_act = self.pair_input_layer_norm(pair_cond)
            cache = [
                self._project_pair_logits(
                    pair_act,
                    super_block_i,
                    cache_dtype=cache_dtype,
                )
                for super_block_i in range(self.num_super_blocks)
            ]
            del pair_act
        self._pair_logits_cache = cache
        self._pair_logits_cache_key = cache_key
        self._pair_logits_cache_source = pair_cond
        self._pair_logits_cache_row_stream = row_stream
        return cache

    def forward(self,
                act: torch.Tensor,  # Changes with each denoising step.
                mask: torch.Tensor,  # Fixed across denoising steps for this sample.
                single_cond: torch.Tensor,  # Includes step-specific noise conditioning.
                pair_cond: Optional[torch.Tensor],  # Fixed across steps; None only when an inference cache is ready.
                clear_flag: bool = False,
            ) -> torch.Tensor:
        pair_logits_cache = None
        pair_act = None
        if pair_cond is None:
            if self.training or torch.is_grad_enabled() or not self.has_pair_logits_cache():
                raise ValueError('pair_cond may be omitted only with an active inference cache')
            pair_logits_cache = self._pair_logits_cache
            row_stream_cache_enabled = self._pair_logits_cache_row_stream
        else:
            row_stream_cache_enabled = self.uses_row_stream_pair_logits_cache(pair_cond)
            cache_enabled = native_pair_logits_cache_enabled(
                feature_length=pair_cond.shape[-2], training=self.training,
                grad_enabled=torch.is_grad_enabled(),
                row_stream_enabled=row_stream_cache_enabled,
                threshold=DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
            )
            if not cache_enabled and self._pair_logits_cache is not None:
                self.clear_pair_logits_cache()
            if cache_enabled:
                pair_logits_cache = self._get_or_build_pair_logits_cache(
                    pair_cond, row_stream=row_stream_cache_enabled,
                )
            else:
                pair_act = self.pair_input_layer_norm(pair_cond)

        for super_block_i in range(self.num_super_blocks):
            pair_logits = None
            if not row_stream_cache_enabled:
                pair_logits = (
                    pair_logits_cache[super_block_i]
                    if pair_logits_cache is not None
                    else self._project_pair_logits(pair_act, super_block_i)
                )
            for j in range(self.super_block_size):
                if row_stream_cache_enabled:
                    pair_bias = pair_logits_cache[
                        super_block_i * self.super_block_size + j
                    ]
                else:
                    pair_bias = pair_logits[j, ...]
                act += self.self_attention[super_block_i * self.super_block_size + j](
                    act, mask, pair_bias, single_cond)
                act += self.transition_block[super_block_i *
                                             self.super_block_size + j](act, single_cond)
                del pair_bias
            # Release this projection before allocating the next super-block.
            del pair_logits
        del pair_act
        if clear_flag:
            self.clear_pair_logits_cache()
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

        self.q_scale = self.key_dim_per_head ** (-0.5)

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
        if mask_q.ndim == x_q.ndim - 2:
            mask_q = mask_q.unsqueeze(0).expand(x_q.shape[0], *mask_q.shape)
        if mask_k.ndim == x_k.ndim - 2:
            mask_k = mask_k.unsqueeze(0).expand(x_k.shape[0], *mask_k.shape)

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
            logits += pair_logits
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
            self.c_pair_cond, self.num_blocks * self.num_head, bias=False)

        self.cross_attention = nn.ModuleList(
            [CrossAttention(num_head=self.num_head) for _ in range(self.num_blocks)])

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

        self.first_run = True
        self.pair_logits = None

    def forward(
        self,
        queries_act: torch.Tensor,  # (num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,  # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys)
        keys_mask: torch.Tensor,  # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
        pair_cond: torch.Tensor,  # (num_subsets, num_queries, num_keys, ch)
        clear_flag: bool
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

            queries_act += self.cross_attention[block_idx](
                x_q=queries_act,
                x_k=keys_act,
                mask_q=queries_mask,
                mask_k=keys_mask,
                pair_logits=self.pair_logits[block_idx,...],
                single_cond_q=queries_single_cond,
                single_cond_k=keys_single_cond,
            )
            queries_act += self.transition_block[block_idx](
                queries_act,
                queries_single_cond,
            )

        if clear_flag:
            self.first_run = True
            self.pair_logits = None

        return queries_act
