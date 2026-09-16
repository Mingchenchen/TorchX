from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from . import atom_layout
from .. import fastnn
from ..fastnn import attention as fastnn_attention_impl
from ...parallel_ops import (
    pad_to_length,
    slice_1d_local,
    all_gather_first_axis,
    launch_all_gather_first_axis_async,
    wait_all_gather_first_axis,
    trim_to_global_length,
    zero_local_padding,
    zero_global_padding,
)
from torchfold.runtime_policy import (
    DIFFUSION_PAIR_LOGITS_ROW_STREAM_CHUNK_SIZE,
    PARALLEL_DIFFUSION_PAIR_TRANSITION_ROW_STREAM_CHUNK_SIZE as DIFFUSION_PAIR_TRANSITION_ROW_STREAM_CHUNK_SIZE,
    PARALLEL_DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD as DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
    diffusion_pair_logits_row_streaming,
    parallel_pair_logits_cache_enabled,
)


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        c_x: int,
        c_single_cond: int,
        use_single_cond: bool = False,
    ) -> None:
        super(AdaptiveLayerNorm, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        if self.use_single_cond is True:
            self.layer_norm = fastnn.LayerNorm(
                self.c_x, elementwise_affine=False, bias=False)
            self.single_cond_layer_norm = fastnn.LayerNorm(
                self.c_single_cond, bias=False)
            self.single_cond_scale = nn.Linear(
                self.c_single_cond, self.c_x, bias=True)
            self.single_cond_bias = nn.Linear(
                self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = fastnn.LayerNorm(self.c_x)

    def forward(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
    def __init__(
        self,
        c_in: int,
        c_out: int,
        c_single_cond: int,
        use_single_cond: bool = False,
    ) -> None:
        super(AdaLNZero, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        self.transition2 = nn.Linear(self.c_in, self.c_out, bias=False)
        if self.use_single_cond is True:
            self.adaptive_zero_cond = nn.Linear(
                self.c_single_cond, self.c_out, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (single_cond is None) == (self.use_single_cond is False)

        output = self.transition2(x)
        if self.use_single_cond is True:
            cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output
        return output


class DiffusionTransition(nn.Module):
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
            self.c_x, self.c_single_cond, self.use_single_cond)
        self.transition1 = nn.Linear(
            self.c_x,
            2 * self.c_x * self.num_intermediate_factor,
            bias=False,
        )
        # Derived cache: exclude from state_dict, DDP broadcasts and Module._apply.
        self.transition1_weight_t = None
        self._transition1_weight_t_signature = None

        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
        )

    def _apply(self, fn, recurse=True):
        self.invalidate_transition1_weight_cache_()
        return super()._apply(fn, recurse)

    @staticmethod
    def _weight_signature(weight: torch.Tensor) -> tuple:
        if torch.is_inference(weight):
            raise RuntimeError(
                "Diffusion transition parameters must be created, loaded, "
                "and moved outside torch.inference_mode(); enter "
                "inference_mode only for forward calls"
            )
        try:
            version = weight._version
        except RuntimeError:
            version = None
        return (
            weight.data_ptr(),
            version,
            weight.device,
            weight.dtype,
            tuple(weight.shape),
            tuple(weight.stride()),
        )

    @staticmethod
    def _backend_signature(backend) -> tuple:
        return (
            getattr(backend, "__module__", None),
            getattr(backend, "__qualname__", None),
            id(backend),
        )

    def invalidate_transition1_weight_cache_(self) -> "DiffusionTransition":
        self.transition1_weight_t = None
        self._transition1_weight_t_signature = None
        return self

    def _build_transition1_weight_t(self, detach: bool) -> torch.Tensor:
        weight = self.transition1.weight.detach() if detach else self.transition1.weight
        return weight.T.contiguous()

    @torch.no_grad()
    def pack_transition1_weight_(self) -> "DiffusionTransition":
        weight_signature = self._weight_signature(self.transition1.weight)
        signature = (
            weight_signature,
            self._backend_signature(fastnn.gated_linear_unit),
        )
        if (
            self.transition1_weight_t is None
            or weight_signature[1] is None
            or signature != self._transition1_weight_t_signature
        ):
            self.transition1_weight_t = self._build_transition1_weight_t(
                detach=True
            )
            self._transition1_weight_t_signature = signature
        return self

    def _transition1_weight_for_glu(self) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            return self._build_transition1_weight_t(detach=False)

        self.pack_transition1_weight_()
        return self.transition1_weight_t

    def _forward_chunk(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.adaptive_layernorm(x, single_cond)
        transition1_weight_t = self._transition1_weight_for_glu()
        c = fastnn.gated_linear_unit(x, transition1_weight_t)
        return self.adaptive_zero_init(c, single_cond)

    def add_pair_residual_row_streaming_(
        self,
        x: torch.Tensor,
        chunk_size: int = DIFFUSION_PAIR_TRANSITION_ROW_STREAM_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Apply pair conditioning in local-row chunks without collectives."""
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


class DiffusionTransitionParallel(DiffusionTransition):
    """Diffusion transition wrapper kept on the dense path for DAP row-shard."""

    def __init__(
        self,
        c_x: int,
        c_single_cond: int,
        num_intermediate_factor: int = 2,
        use_single_cond: bool = False,
        parallel_group=None,
    ) -> None:
        super().__init__(
            c_x=c_x,
            c_single_cond=c_single_cond,
            num_intermediate_factor=num_intermediate_factor,
            use_single_cond=use_single_cond,
        )

    def forward(
        self,
        x: torch.Tensor,
        single_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(x, single_cond)


@dataclass
class _PreparedSelfAttentionLocalQuery:
    """Next-block local query state produced while Act AllGather is in flight."""

    q: torch.Tensor
    mask_q: torch.Tensor
    gate_weights: torch.Tensor
    single_cond_q: Optional[torch.Tensor]


class SelfAttention(nn.Module):
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
        self.q_scale = self.qkv_dim ** (-0.5)

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)

        self.qkv_projection = nn.Linear(self.c_x, self.c_x * 3, bias=True)
        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond)

    def _qkv_weight_bias(
        self,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = self.qkv_projection.weight
        bias = self.qkv_projection.bias
        expected_weight_shape = (3 * self.c_x, self.c_x)
        if tuple(weight.shape) != expected_weight_shape:
            raise ValueError(
                "SelfAttention qkv_projection weight requires shape "
                f"{expected_weight_shape}, got {tuple(weight.shape)}"
            )
        if bias is not None and tuple(bias.shape) != (3 * self.c_x,):
            raise ValueError(
                "SelfAttention qkv_projection bias requires shape "
                f"{(3 * self.c_x,)}, got {tuple(bias.shape)}"
            )
        return weight, bias

    def _project_qkv(
        self,
        x_q_norm: torch.Tensor,
        x_kv_norm: torch.Tensor,
        shared_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project exactly the Q/K/V components consumed by attention.

        The checkpoint keeps one packed ``[Q; K; V]`` linear layer.  For
        shared Q/K/V input it is evaluated once.  For local-Q/global-KV input
        the packed weight is sliced so local rows produce Q only and global
        rows produce K/V only, without changing parameter names or layout.
        """
        if shared_input:
            qkv_proj = self.qkv_projection(x_q_norm)
            return torch.split(qkv_proj, self.c_x, dim=-1)

        # Project Q for local rows and K/V for global rows only.
        weight, bias = self._qkv_weight_bias()
        q_bias = None if bias is None else bias[:self.c_x]
        kv_bias = None if bias is None else bias[self.c_x:]
        q_proj = F.linear(x_q_norm, weight[:self.c_x], q_bias)
        kv_proj = F.linear(x_kv_norm, weight[self.c_x:], kv_bias)
        k_proj, v_proj = torch.split(kv_proj, self.c_x, dim=-1)
        return q_proj, k_proj, v_proj

    def _project_local_query(self, x_q_norm: torch.Tensor) -> torch.Tensor:
        weight, bias = self._qkv_weight_bias()
        q_bias = None if bias is None else bias[:self.c_x]
        return F.linear(x_q_norm, weight[:self.c_x], q_bias)

    def _project_global_key_value(
        self,
        x_kv_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight, bias = self._qkv_weight_bias()
        kv_bias = None if bias is None else bias[self.c_x:]
        kv_proj = F.linear(x_kv_norm, weight[self.c_x:], kv_bias)
        return torch.split(kv_proj, self.c_x, dim=-1)

    def prepare_local_query(
        self,
        x_q: torch.Tensor,
        mask_q: torch.Tensor,
        single_cond_q: Optional[torch.Tensor] = None,
    ) -> _PreparedSelfAttentionLocalQuery:
        """Prepare next-block local-only Q and gate work before global K/V exists."""
        assert (single_cond_q is None) == (self.use_single_cond is False)

        x_q_norm = self.adaptive_layernorm(x_q, single_cond_q)
        q_proj = self._project_local_query(x_q_norm)
        q = q_proj.view(q_proj.shape[0], self.num_head, self.qkv_dim)
        gate_weights = torch.sigmoid(self.gating_query(x_q_norm))
        return _PreparedSelfAttentionLocalQuery(
            q=q,
            mask_q=mask_q,
            gate_weights=gate_weights,
            single_cond_q=single_cond_q,
        )

    def _prepare_fusion_attention_bsh_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert projected ``[S,H,D]`` tensors to FA ``[1,S,C]`` inputs.

        Diffusion sample parallelism uses local query rows and global key/value
        rows, so Q and K/V may have different sequence lengths.  BSH supports
        that rectangular contract.  Query scaling remains in FP32 and is
        applied exactly once before all three inputs are converted to the BF16
        dtype consumed by Fusion Attention.
        """
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                "Diffusion BSH projection requires [S,H,D] Q/K/V; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}, "
                f"v={tuple(v.shape)}"
            )
        expected_head_shape = (self.num_head, self.qkv_dim)
        if tuple(q.shape[1:]) != expected_head_shape:
            raise ValueError(
                "Diffusion BSH query has incompatible head dimensions: "
                f"expected={expected_head_shape}, got={tuple(q.shape[1:])}"
            )
        if tuple(k.shape[1:]) != expected_head_shape or k.shape != v.shape:
            raise ValueError(
                "Diffusion BSH requires matching K/V head dimensions; "
                f"expected={expected_head_shape}, k={tuple(k.shape)}, "
                f"v={tuple(v.shape)}"
            )

        # Build sequence-major FA buffers directly, without head-first intermediates.
        q_bsh = (q.reshape(1, q.shape[0], self.c_x) * self.q_scale)
        q_bsh = q_bsh.to(dtype=torch.bfloat16).contiguous()
        k_bsh = k.reshape(1, k.shape[0], self.c_x)
        k_bsh = k_bsh.to(dtype=torch.bfloat16).contiguous()
        v_bsh = v.reshape(1, v.shape[0], self.c_x)
        v_bsh = v_bsh.to(dtype=torch.bfloat16).contiguous()
        return q_bsh, k_bsh, v_bsh

    def _finish_projected_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_q: torch.Tensor,
        pair_logits: Optional[torch.Tensor],
        single_cond_q: Optional[torch.Tensor],
        mask_kv: Optional[torch.Tensor],
        fa_mask_contract: Optional[str],
        key_valid_len: Optional[int],
        *,
        x_q_norm: Optional[torch.Tensor] = None,
        gate_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask_for_attention = mask_kv if mask_kv is not None else mask_q
        mask_for_attention = None if mask_for_attention is None else mask_for_attention.to(torch.bool)

        # Known valid-key prefixes allow a static FA mask without device synchronization.
        fa_mask_plan = None
        if fa_mask_contract is not None:
            if fa_mask_contract != "sequence_padding":
                raise ValueError(
                    f"Unsupported SelfAttention FA mask contract: {fa_mask_contract}"
                )
            if key_valid_len is None:
                raise ValueError(
                    "sequence_padding FA mask contract requires key_valid_len"
                )
            if mask_for_attention is None:
                raise ValueError(
                    "sequence_padding FA mask contract requires a sequence mask"
                )
            fa_mask_plan = fastnn_attention_impl.FAMaskPlan(
                # BNSD/BSH wrappers add a singleton batch to SelfAttention.
                expected_batch=1,
                expected_key_len=k.shape[0],
                key_valid_len=int(key_valid_len),
                expected_mask_dim=1,
            )

        if pair_logits is None:
            pair_logits_bias = None
        elif (
            pair_logits.dtype == torch.bfloat16
            and fastnn.config.dot_product_attention_implementation
            == "Fusion_Attention"
        ):
            # Preserve cached BF16 PSE to avoid a round-trip through FP32.
            pair_logits_bias = pair_logits
        else:
            pair_logits_bias = pair_logits.to(dtype=q.dtype)

        attention_impl = fastnn.config.dot_product_attention_implementation
        use_bsh = attention_impl == "Fusion_Attention"
        if use_bsh:
            q_bsh, k_bsh, v_bsh = self._prepare_fusion_attention_bsh_inputs(
                q,
                k,
                v,
            )
            weighted_avg = (
                fastnn_attention_impl.dot_product_attention_FA_sequence_major(
                    q_bsh,
                    k_bsh,
                    v_bsh,
                    self.num_head,
                    mask=mask_for_attention,
                    bias=pair_logits_bias,
                    mask_plan=fa_mask_plan,
                )
            ).squeeze(0)
            del q_bsh, k_bsh, v_bsh
        else:
            # The torch backend retains its established head-first layout.
            q_bnsd = einops.rearrange(q, 'n h c -> 1 h n c')
            k_bnsd = einops.rearrange(k, 'n h c -> 1 h n c')
            v_bnsd = einops.rearrange(v, 'n h c -> 1 h n c')
            weighted_avg = fastnn.dot_product_attention(
                q_bnsd,
                k_bnsd,
                v_bnsd,
                self.num_head,
                mask=mask_for_attention,
                bias=pair_logits_bias,
                mask_plan=fa_mask_plan,
            )
            weighted_avg = weighted_avg.squeeze(0)
            weighted_avg = einops.rearrange(
                weighted_avg,
                'h q c -> q (h c)',
            )

        if gate_weights is None:
            if x_q_norm is None:
                raise ValueError(
                    "SelfAttention requires x_q_norm or prepared gate weights."
                )
            gate_weights = torch.sigmoid(self.gating_query(x_q_norm))
        weighted_avg *= gate_weights

        return self.adaptive_zero_init(weighted_avg, single_cond_q)

    def finish_prepared_local_query(
        self,
        prepared: _PreparedSelfAttentionLocalQuery,
        x_kv: torch.Tensor,
        mask_kv: Optional[torch.Tensor],
        single_cond_kv: Optional[torch.Tensor],
        pair_logits: Optional[torch.Tensor] = None,
        fa_mask_contract: Optional[str] = None,
        key_valid_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Finish global K/V and attention after the matching AllGather wait."""
        assert (single_cond_kv is None) == (self.use_single_cond is False)

        x_kv_norm = self.adaptive_layernorm(x_kv, single_cond_kv)
        k_proj, v_proj = self._project_global_key_value(x_kv_norm)
        k = k_proj.view(k_proj.shape[0], self.num_head, self.qkv_dim)
        v = v_proj.view(v_proj.shape[0], self.num_head, self.qkv_dim)
        return self._finish_projected_attention(
            prepared.q,
            k,
            v,
            prepared.mask_q,
            pair_logits,
            prepared.single_cond_q,
            mask_kv,
            fa_mask_contract,
            key_valid_len,
            gate_weights=prepared.gate_weights,
        )

    def _forward_sample_batch(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: Optional[torch.Tensor],
        single_cond: Optional[torch.Tensor],
        fa_mask_contract: Optional[str],
        key_valid_len: Optional[int],
    ) -> torch.Tensor:
        """Run the reference-compatible world-size-one [B, N, C] path."""
        x_norm = self.adaptive_layernorm(x, single_cond)
        qkv = self.qkv_projection(x_norm)

        fa_mask_plan = None
        if (
            fastnn.config.dot_product_attention_implementation
            == "Fusion_Attention"
            and mask is not None
            and mask.dim() == 1
        ):
            if fa_mask_contract is not None:
                if fa_mask_contract != "sequence_padding":
                    raise ValueError(
                        "Unsupported sample-batch FA mask contract: "
                        f"{fa_mask_contract}"
                    )
                if key_valid_len is None:
                    raise ValueError(
                        "sequence_padding FA mask contract requires "
                        "key_valid_len"
                    )
                valid_len = int(key_valid_len)
            else:
                valid_len = fastnn.config.single_card_valid_token_count()
            if valid_len is not None:
                token_length = qkv.shape[-2]
                if not 0 <= valid_len <= token_length:
                    raise ValueError(
                        "Sample-batch SelfAttention static FA contract "
                        f"mismatch: valid_len={valid_len}, "
                        f"token_length={token_length}"
                    )
                fa_mask_plan = fastnn_attention_impl.FAMaskPlan(
                    expected_batch=qkv.shape[0],
                    expected_key_len=token_length,
                    key_valid_len=valid_len,
                    expected_mask_dim=1,
                )

        if (
            fastnn.config.dot_product_attention_implementation
            == "Fusion_Attention"
        ):
            q, k, v = torch.split(qkv, self.c_x, dim=-1)
            q = (q * self.q_scale).to(torch.bfloat16).contiguous()
            k = k.to(torch.bfloat16).contiguous()
            v = v.to(torch.bfloat16).contiguous()
            weighted_avg = (
                fastnn_attention_impl.dot_product_attention_FA_sequence_major(
                    q,
                    k,
                    v,
                    self.num_head,
                    mask=mask,
                    bias=pair_logits,
                    mask_plan=fa_mask_plan,
                )
            )
        else:
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
            weighted_avg = fastnn.dot_product_attention(
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

        weighted_avg *= torch.sigmoid(self.gating_query(x_norm))
        return self.adaptive_zero_init(weighted_avg, single_cond)

    def forward(
        self,
        x_q: torch.Tensor,
        mask_q: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
        single_cond_q: Optional[torch.Tensor] = None,
        x_kv: Optional[torch.Tensor] = None,
        mask_kv: Optional[torch.Tensor] = None,
        single_cond_kv: Optional[torch.Tensor] = None,
        fa_mask_contract: Optional[str] = None,
        key_valid_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Supports local-query/global-key-value attention.

        Args:
            x_q: [Lq, C]
            mask_q: [Lq]
            pair_logits: [H, Lq, Lk]
            single_cond_q: [Lq, Cc]
            x_kv: [Lk, C], defaults to x_q.
            mask_kv: [Lk], defaults to mask_q.
            single_cond_kv: [Lk, Cc], defaults to single_cond_q.
            fa_mask_contract: Explicit source contract for a static FA mask plan.
            key_valid_len: Host-known valid key prefix length for sequence padding.
        """
        assert (single_cond_q is None) == (self.use_single_cond is False)

        if x_kv is None:
            x_kv = x_q
        if mask_kv is None:
            mask_kv = mask_q
        if single_cond_kv is None:
            single_cond_kv = single_cond_q

        if x_q.ndim == 3:
            if x_kv is not x_q or single_cond_kv is not single_cond_q:
                raise ValueError(
                    "Sample-batch SelfAttention requires shared Q/K/V input"
                )
            return self._forward_sample_batch(
                x_q,
                mask_q,
                pair_logits,
                single_cond_q,
                fa_mask_contract,
                key_valid_len,
            )

        x_q_norm = self.adaptive_layernorm(x_q, single_cond_q)
        if x_kv is x_q and single_cond_kv is single_cond_q:
            x_kv_norm = x_q_norm
        else:
            x_kv_norm = self.adaptive_layernorm(x_kv, single_cond_kv)

        q_proj, k_proj, v_proj = self._project_qkv(
            x_q_norm,
            x_kv_norm,
            shared_input=(x_kv is x_q and x_kv_norm is x_q_norm),
        )

        q = q_proj.view(q_proj.shape[0], self.num_head, self.qkv_dim)
        k = k_proj.view(k_proj.shape[0], self.num_head, self.qkv_dim)
        v = v_proj.view(v_proj.shape[0], self.num_head, self.qkv_dim)

        return self._finish_projected_attention(
            q,
            k,
            v,
            mask_q,
            pair_logits,
            single_cond_q,
            mask_kv,
            fa_mask_contract,
            key_valid_len,
            x_q_norm=x_q_norm,
        )


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        c_act: int = 768,
        c_single_cond: int = 384,
        c_pair_cond: int = 128,
        num_head: int = 16,
        num_blocks: int = 24,
        super_block_size: int = 4,
    ) -> None:
        super(DiffusionTransformer, self).__init__()

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        if self.num_blocks % self.super_block_size != 0:
            raise ValueError(
                f"num_blocks={self.num_blocks} must be divisible by "
                f"super_block_size={self.super_block_size}"
            )

        self.num_super_blocks = self.num_blocks // self.super_block_size

        self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond)
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

        # Transition blocks remain unsharded.
        self.transition_block = nn.ModuleList(
            [
                DiffusionTransitionParallel(
                    c_x=self.c_act,
                    c_single_cond=self.c_single_cond,
                    num_intermediate_factor=2,
                    use_single_cond=True,
                    parallel_group=None,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Prediction-only cache, excluded from state_dict and cleared after denoising.
        self._pair_logits_cache: Optional[list[torch.Tensor]] = None
        self._pair_logits_cache_key = None

    @staticmethod
    def _parallel_pair_cache_key(parallel_spec) -> Optional[tuple]:
        if parallel_spec is None:
            return None
        return (
            parallel_spec.rank,
            parallel_spec.world_size,
            parallel_spec.n_global,
            parallel_spec.n_padded,
            parallel_spec.shard_size,
            parallel_spec.start,
            parallel_spec.end,
        )

    def _make_pair_logits_cache_key(
        self,
        pair_cond: torch.Tensor,
        parallel_spec,
        cache_dtype: str,
    ) -> tuple:
        return (
            pair_cond.data_ptr(),
            tuple(pair_cond.shape),
            tuple(pair_cond.stride()),
            pair_cond.dtype,
            pair_cond.device,
            self._parallel_pair_cache_key(parallel_spec),
            fastnn.config.dot_product_attention_implementation,
            cache_dtype,
        )

    def clear_pair_logits_cache(self) -> None:
        """Release prediction-lifetime pair-logits tensors immediately."""
        self._pair_logits_cache = None
        self._pair_logits_cache_key = None

    def has_pair_logits_cache(self) -> bool:
        return self._pair_logits_cache is not None

    def release_pair_logits_cache_source(self) -> None:
        """Parallel cache keys contain metadata only; kept for head parity."""

    def _prepare_pair_act(
        self,
        pair_cond: torch.Tensor,
        parallel_spec,
    ) -> torch.Tensor:
        pair_act = self.pair_input_layer_norm(pair_cond)
        if parallel_spec is not None:
            pair_act = zero_local_padding(
                pair_act,
                parallel_spec,
                dim=0,
                value=0.0,
            )
            pair_act = zero_global_padding(
                pair_act,
                parallel_spec,
                dim=1,
                value=0.0,
            )
        return pair_act

    def _project_pair_logits(
        self,
        pair_act: torch.Tensor,
        super_block_i: int,
        cache_dtype: str = "native",
    ) -> torch.Tensor:
        pair_logits = self.pair_logits_projection[super_block_i](pair_act)
        pair_logits = einops.rearrange(
            pair_logits,
            'q k (b h) -> b h q k',
            h=self.num_head,
        )
        if cache_dtype == "bf16" and pair_logits.dtype != torch.bfloat16:
            pair_logits = pair_logits.to(torch.bfloat16)
        return pair_logits

    def _build_pair_logits_cache_row_streaming(
        self,
        pair_cond: torch.Tensor,
        parallel_spec,
        chunk_size: int = DIFFUSION_PAIR_LOGITS_ROW_STREAM_CHUNK_SIZE,
    ) -> list[torch.Tensor]:
        """Normalize/project local rows directly into the persistent BF16 cache."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("diffusion pair-logits row streaming is inference-only")
        if chunk_size <= 0:
            raise ValueError("diffusion pair-logits row chunk size must be positive")
        rows, columns = pair_cond.shape[:2]
        cache = []
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            pair_act = self.pair_input_layer_norm(pair_cond[start:end])
            valid_rows = max(0, min(end, parallel_spec.local_valid_size) - start)
            pair_act[valid_rows:].zero_()
            zero_global_padding(pair_act, parallel_spec, dim=1, value=0.0)
            for super_block_i in range(self.num_super_blocks):
                logits = self._project_pair_logits(pair_act, super_block_i, "bf16")
                if start == 0:
                    # Match storage after casting: NPU may use b,h,q,k; CPU/already-BF16
                    # projections retain q,k,(b,h).
                    if logits.is_contiguous():
                        destination = logits.new_empty(
                            (*logits.shape[:2], rows, columns)
                        )
                    else:
                        destination = einops.rearrange(
                            logits.new_empty((rows, columns, logits.shape[0] * self.num_head)),
                            'q k (b h) -> b h q k', h=self.num_head,
                        )
                    cache.append(destination)
                cache[super_block_i][:, :, start:end, :].copy_(logits)
                del logits
            if pair_cond.device.type == "npu":
                torch.npu.synchronize()
            del pair_act
        return cache

    def _get_or_build_pair_logits_cache(
        self,
        pair_cond: torch.Tensor,
        parallel_spec,
    ) -> list[torch.Tensor]:
        cache_dtype = (
            "bf16"
            if fastnn.config.dot_product_attention_implementation
            == "Fusion_Attention"
            else "native"
        )
        cache_key = self._make_pair_logits_cache_key(
            pair_cond,
            parallel_spec,
            cache_dtype,
        )
        if (
            self._pair_logits_cache is not None
            and self._pair_logits_cache_key == cache_key
        ):
            return self._pair_logits_cache

        self.clear_pair_logits_cache()
        if (
            parallel_spec is not None
            and parallel_spec.is_distributed
            and diffusion_pair_logits_row_streaming(
                feature_length=parallel_spec.n_global,
                training=self.training,
                grad_enabled=torch.is_grad_enabled(),
                attention_implementation=fastnn.config.dot_product_attention_implementation,
            )
        ):
            cache = self._build_pair_logits_cache_row_streaming(
                pair_cond, parallel_spec,
                chunk_size=DIFFUSION_PAIR_LOGITS_ROW_STREAM_CHUNK_SIZE,
            )
        else:
            pair_act = self._prepare_pair_act(pair_cond, parallel_spec)
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
        return cache

    def forward(
        self,
        act: torch.Tensor,
        mask: torch.Tensor,
        single_cond: torch.Tensor,
        pair_cond: Optional[torch.Tensor],
        parallel_spec=None,
        clear_flag: bool = False,
    ) -> torch.Tensor:
        pair_logits_cache = None
        pair_act = None
        if pair_cond is None:
            if self.training or torch.is_grad_enabled() or self._pair_logits_cache is None:
                raise ValueError(
                    "pair_cond may be omitted only while an inference "
                    "pair-logits cache is active"
                )
            pair_logits_cache = self._pair_logits_cache
        else:
            is_distributed = (
                parallel_spec is not None and parallel_spec.is_distributed
            )
            padded_sequence_length = pair_cond.shape[-2]
            cache_enabled = parallel_pair_logits_cache_enabled(
                feature_length=padded_sequence_length,
                training=self.training, is_distributed=is_distributed,
                threshold=DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
            )
            if not cache_enabled and self._pair_logits_cache is not None:
                self.clear_pair_logits_cache()

            if cache_enabled:
                pair_logits_cache = self._get_or_build_pair_logits_cache(
                    pair_cond,
                    parallel_spec,
                )
            else:
                pair_act = self._prepare_pair_act(pair_cond, parallel_spec)

        sample_batch = act.ndim == 3
        if (
            sample_batch
            and parallel_spec is not None
            and parallel_spec.is_distributed
        ):
            raise ValueError(
                "Sample-batch DiffusionTransformer is world-size-one only"
            )

        if parallel_spec is None or sample_batch:
            for super_block_i in range(self.num_super_blocks):
                pair_logits = (
                    pair_logits_cache[super_block_i]
                    if pair_logits_cache is not None
                    else self._project_pair_logits(
                        pair_act,
                        super_block_i,
                    )
                )

                for j in range(self.super_block_size):
                    block_idx = super_block_i * self.super_block_size + j
                    act = act + self.self_attention[block_idx](
                        act,
                        mask,
                        pair_logits[j, ...],
                        single_cond,
                    )
                    act = act + self.transition_block[block_idx](
                        act,
                        single_cond,
                    )
            if clear_flag:
                self.clear_pair_logits_cache()
            return act

        act_full = pad_to_length(
            act,
            dim=0,
            length=parallel_spec.n_padded,
            value=0.0,
        )
        act_full = zero_global_padding(act_full, parallel_spec, dim=0, value=0.0)
        single_cond_full = pad_to_length(
            single_cond,
            dim=0,
            length=parallel_spec.n_padded,
            value=0.0,
        )
        single_cond_full = zero_global_padding(single_cond_full, parallel_spec, dim=0, value=0.0)
        mask_full = pad_to_length(
            mask,
            dim=0,
            length=parallel_spec.n_padded,
            value=0,
        )
        mask_full = zero_global_padding(mask_full, parallel_spec, dim=0, value=0)

        overlap_local_q_prep = parallel_spec.is_distributed
        prepared_local_query = None

        for super_block_i in range(self.num_super_blocks):
            pair_logits_row = (
                pair_logits_cache[super_block_i]
                if pair_logits_cache is not None
                else self._project_pair_logits(
                    pair_act,
                    super_block_i,
                )
            )

            for j in range(self.super_block_size):
                block_idx = super_block_i * self.super_block_size + j

                act_local = slice_1d_local(act_full, parallel_spec, pad_value=0.0)
                single_cond_local = slice_1d_local(single_cond_full, parallel_spec, pad_value=0.0)
                mask_local = slice_1d_local(mask_full, parallel_spec, pad_value=0)

                if prepared_local_query is None:
                    attention_update = self.self_attention[block_idx](
                        x_q=act_local,
                        mask_q=mask_local,
                        pair_logits=pair_logits_row[j, ...],
                        single_cond_q=single_cond_local,
                        x_kv=act_full,
                        mask_kv=mask_full,
                        single_cond_kv=single_cond_full,
                        fa_mask_contract="sequence_padding",
                        key_valid_len=parallel_spec.n_global,
                    )
                else:
                    attention_update = self.self_attention[
                        block_idx
                    ].finish_prepared_local_query(
                        prepared_local_query,
                        x_kv=act_full,
                        mask_kv=mask_full,
                        single_cond_kv=single_cond_full,
                        pair_logits=pair_logits_row[j, ...],
                        fa_mask_contract="sequence_padding",
                        key_valid_len=parallel_spec.n_global,
                    )
                    prepared_local_query = None

                act_local = act_local + attention_update
                act_local = act_local + self.transition_block[block_idx](
                    act_local,
                    single_cond_local,
                )

                gather_input = act_local.contiguous()
                if overlap_local_q_prep and block_idx + 1 < self.num_blocks:
                    pending_gather = launch_all_gather_first_axis_async(
                        gather_input,
                        parallel_spec,
                    )
                    try:
                        prepared_local_query = self.self_attention[
                            block_idx + 1
                        ].prepare_local_query(
                            gather_input,
                            mask_local,
                            single_cond_local,
                        )
                    finally:
                        act_full = wait_all_gather_first_axis(
                            pending_gather,
                            parallel_spec,
                        )
                else:
                    act_full = all_gather_first_axis(
                        gather_input,
                        parallel_spec,
                    )
                act_full = zero_global_padding(act_full, parallel_spec, dim=0, value=0.0)

        act_full = trim_to_global_length(act_full, parallel_spec, dim=0)
        if clear_flag:
            self.clear_pair_logits_cache()
        return act_full.contiguous()


class CrossAttention(nn.Module):
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

        self.q_scale = self.key_dim_per_head ** (-0.5)

        self.q_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)
        self.k_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)

        self.q_projection = nn.Linear(self.key_dim, self.key_dim, bias=True)
        self.k_projection = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_projection = nn.Linear(self.value_dim, self.value_dim, bias=False)

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
        single_cond_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask_q.ndim == x_q.ndim - 2:
            mask_q = mask_q.unsqueeze(0).expand(x_q.shape[0], *mask_q.shape)
        if mask_k.ndim == x_k.ndim - 2:
            mask_k = mask_k.unsqueeze(0).expand(x_k.shape[0], *mask_k.shape)

        assert len(mask_q.shape) == len(x_q.shape) - 1, f'{mask_q.shape}, {x_q.shape}'
        assert len(mask_k.shape) == len(x_k.shape) - 1, f'{mask_k.shape}, {x_k.shape}'

        combined_mask = mask_q.unsqueeze(-1).unsqueeze(-3) | mask_k.unsqueeze(-2).unsqueeze(-2)
        bias = 1e9 * combined_mask.logical_not()

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)

        q = self.q_projection(x_q)
        k = self.k_projection(x_k)
        q = torch.reshape(q, q.shape[:-1] + (self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, k.shape[:-1] + (self.num_head, self.key_dim_per_head))

        logits = torch.einsum('...qhc,...khc->...hqk', q * self.q_scale, k) + bias
        if pair_logits is not None:
            logits += pair_logits
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(x_k)
        v = torch.reshape(v, v.shape[:-1] + (self.num_head, self.value_dim_per_head))
        weighted_avg = torch.einsum('...hqk,...khc->...qhc', weights, v)
        weighted_avg = torch.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))

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

        self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond, bias=False)
        self.pair_logits_projection = nn.Linear(
            self.c_pair_cond, self.num_blocks * self.num_head, bias=False)

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

        self.first_run = True
        self.pair_logits = None

    def forward(
        self,
        queries_act: torch.Tensor,
        queries_mask: torch.Tensor,
        queries_to_keys: atom_layout.GatherInfo,
        keys_mask: torch.Tensor,
        queries_single_cond: torch.Tensor,
        keys_single_cond: torch.Tensor,
        pair_cond: torch.Tensor,
        clear_flag: bool,
    ) -> torch.Tensor:
        if self.first_run:
            self.first_run = False
            pair_act = self.pair_input_layer_norm(pair_cond)
            pair_logits = self.pair_logits_projection(pair_act)
            pair_logits = einops.rearrange(
                pair_logits,
                'n q k (b h) -> b n h q k',
                h=self.num_head,
            )
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
                pair_logits=self.pair_logits[block_idx, ...],
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
