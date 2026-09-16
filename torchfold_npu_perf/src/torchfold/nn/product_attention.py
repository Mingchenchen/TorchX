from typing import Optional

import torch
import torch_npu

from torchfold.nn import fastnn_config
from torchfold.parallel.parallel_impl.fastnn.attention import (
    FAMaskPlan,
    _prepare_fa_mask_for_shape,
)


def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                num_head: int,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    logits = torch.matmul(q, k.transpose(-1, -2).contiguous())

    if bias is not None:
        logits += bias

    if mask is not None:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(dtype=torch.bool)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1).to(dtype=torch.bool)
        logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)

    return torch.matmul(weights, v)


def _prepare_fusion_attention_mask(
    mask: Optional[torch.Tensor],
    *,
    mask_plan: Optional[FAMaskPlan] = None,
    batch_size: Optional[int] = None,
    query_length: Optional[int] = None,
    key_length: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Convert a validity mask to the fused operator's masking convention."""
    if mask_plan is not None:
        if batch_size is None or query_length is None or key_length is None:
            raise ValueError("Static FA mask plan requires explicit B/Q/K shapes")
        if (
            mask_plan.expected_mask_dim == 2
            and mask_plan.key_valid_len == key_length
        ):
            expected_shape = (batch_size, key_length)
            if (
                batch_size != mask_plan.expected_batch
                or key_length != mask_plan.expected_key_len
                or mask is None
                or tuple(mask.shape) != expected_shape
            ):
                raise ValueError(
                    "Static FA all-valid Pair mask contract mismatch: "
                    f"mask={None if mask is None else tuple(mask.shape)}, "
                    f"runtime={(batch_size, key_length)}, "
                    f"plan={(mask_plan.expected_batch, mask_plan.expected_key_len)}"
                )
            return None
        atten_mask, query_batch_keep = _prepare_fa_mask_for_shape(
            mask,
            B=batch_size,
            Q=query_length,
            K=key_length,
            mask_plan=mask_plan,
        )
        if query_batch_keep is not None:
            raise ValueError(
                "Single-card static FA plans require all query rows to be valid"
            )
        return atten_mask
    if mask is None or mask.all().item():
        return None

    sequence_length = mask.size(-1)
    if mask.dim() == 1:
        valid_mask = mask.expand(sequence_length, sequence_length)
    elif mask.dim() == 2:
        batch_size = mask.size(0)
        valid_mask = mask.unsqueeze(1).unsqueeze(2).expand(
            batch_size,
            1,
            sequence_length,
            sequence_length,
        )
    else:
        raise ValueError(
            "Fusion Attention mask must be one- or two-dimensional; "
            f"got shape={tuple(mask.shape)}"
        )

    return (~valid_mask.to(dtype=torch.bool)).contiguous()


def dot_product_attention_FA(q: torch.Tensor,
                             k: torch.Tensor,
                             v: torch.Tensor,
                             num_head: int,
                             mask: Optional[torch.Tensor] = None,
                             bias: Optional[torch.Tensor] = None,
                             mask_plan: Optional[FAMaskPlan] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    bias = bias.unsqueeze(0)
    atten_mask = _prepare_fusion_attention_mask(
        mask,
        mask_plan=mask_plan,
        batch_size=q.size(0),
        query_length=q.size(-2),
        key_length=k.size(-2),
    )

    q_in = q.to(torch.bfloat16)
    k_in = k.to(torch.bfloat16)
    v_in = v.to(torch.bfloat16)
    pse_in = bias.to(torch.bfloat16) if bias is not None else None

    return torch_npu.npu_fusion_attention(
        q_in, k_in, v_in, head_num=num_head, input_layout="BNSD", pse=pse_in,
        atten_mask=atten_mask, scale=1.0)[0]


def dot_product_attention_FA_sequence_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_head: int,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    mask_plan: Optional[FAMaskPlan] = None,
) -> torch.Tensor:
    """Apply Fusion Attention to pre-scaled BF16 [B,S,C] inputs."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "Sequence-major Fusion Attention requires three-dimensional "
            f"Q/K/V; got q={tuple(q.shape)}, k={tuple(k.shape)}, "
            f"v={tuple(v.shape)}"
        )
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "Sequence-major Fusion Attention requires matching "
            f"self-attention Q/K/V; got q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if num_head <= 0 or q.size(-1) % num_head != 0:
        raise ValueError(
            "Hidden size must be divisible by a positive num_head; "
            f"hidden={q.size(-1)}, num_head={num_head}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            "Fusion Attention requires Q/K/V on the same device; "
            f"got q={q.device}, k={k.device}, v={v.device}"
        )
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise TypeError("Fusion Attention requires prepared BF16 Q/K/V")
    if not all(tensor.is_contiguous() for tensor in (q, k, v)):
        raise ValueError("Fusion Attention requires contiguous Q/K/V")

    sequence_length = q.size(1)
    pse_in = None
    if bias is not None:
        expected_bias_shape = (num_head, sequence_length, sequence_length)
        if tuple(bias.shape) != expected_bias_shape:
            raise ValueError(
                "Triangle Attention bias has an incompatible shape: "
                f"expected={expected_bias_shape}, got={tuple(bias.shape)}"
            )
        pse_in = bias.unsqueeze(0).to(torch.bfloat16).contiguous()

    atten_mask = _prepare_fusion_attention_mask(
        mask,
        mask_plan=mask_plan,
        batch_size=q.size(0),
        query_length=q.size(1),
        key_length=k.size(1),
    )
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=num_head,
        input_layout="BSH",
        pse=pse_in,
        atten_mask=atten_mask,
        scale=1.0,
    )[0]


def dot_product_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          num_head: int,
                          mask: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None,
                          mask_plan: Optional[FAMaskPlan] = None):
    if fastnn_config.dot_product_attention_implementation == "torch":
        out = dot_product_attention_torch(q, k, v, num_head, mask, bias)
    elif fastnn_config.dot_product_attention_implementation == "Fusion_Attention":
        out = dot_product_attention_FA(
            q,
            k,
            v,
            num_head,
            mask,
            bias,
            mask_plan=mask_plan,
        )
    else:
        raise ValueError(
            f"Unknown dot_product_attention_implementation: {fastnn_config.dot_product_attention_implementation}"
        )

    return out
