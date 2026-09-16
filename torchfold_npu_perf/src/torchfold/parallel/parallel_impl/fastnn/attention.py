from dataclasses import dataclass
from typing import Optional

import torch
import torch_npu

from torchfold.nn import fastnn_config

# A caller-provided padding contract may bypass device-side mask inspection.
# Calls without that contract retain runtime validation and fallback behavior.
@dataclass(frozen=True)
class FAMaskPlan:
    expected_batch: int
    expected_key_len: int
    key_valid_len: int
    expected_mask_dim: int = 2
    uniform_valid_rows: bool = True
    all_query_rows_valid: bool = True


def dot_product_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_head: int,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
):
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
        else:
            raise ValueError(f"Unsupported mask dim for torch attention: {mask.dim()}")

        logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, v)


def _prepare_fa_bias(
    bias: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
) -> Optional[torch.Tensor]:
    if bias is None:
        return None

    B, H, Q, _ = q.shape
    K = k.shape[-2]
    return _prepare_fa_bias_for_shape(bias, B=B, H=H, Q=Q, K=K)


def _prepare_fa_bias_for_shape(
    bias: Optional[torch.Tensor],
    *,
    B: int,
    H: int,
    Q: int,
    K: int,
) -> Optional[torch.Tensor]:
    """Normalize attention bias without assuming a Q/K tensor layout."""
    if bias is None:
        return None

    shape = tuple(bias.shape)

    if bias.dim() == 3 and shape == (H, Q, K):
        return bias.unsqueeze(0).contiguous()

    if bias.dim() == 3 and shape == (B, H, K):
        return bias.unsqueeze(-2).contiguous()

    if bias.dim() == 4 and shape in {
        (1, H, Q, K),
        (B, H, Q, K),
        (B, H, 1, K),
    }:
        return bias.contiguous()

    raise ValueError(
        f"Unsupported FA bias shape {shape}; expected [H,Q,K], [B,H,K], "
        f"[1,H,Q,K], [B,H,Q,K], or [B,H,1,K]"
    )


def _prepare_fa_mask(
    mask: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    mask_plan: Optional[FAMaskPlan] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    B = q.shape[0]
    Q = q.shape[-2]
    K = k.shape[-2]
    return _prepare_fa_mask_for_shape(
        mask,
        B=B,
        Q=Q,
        K=K,
        mask_plan=mask_plan,
    )


def _prepare_fa_mask_for_shape(
    mask: Optional[torch.Tensor],
    *,
    B: int,
    Q: int,
    K: int,
    mask_plan: Optional[FAMaskPlan] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build a shared Q/K mask and an optional valid-query mask."""
    if mask is None:
        return None, None

    # Validate host-known metadata without reading NPU scalars on the hot path.
    if mask_plan is not None:
        if B != mask_plan.expected_batch or K != mask_plan.expected_key_len:
            raise ValueError(
                "Static FA mask plan mismatch: "
                f"runtime={(B, K)}, "
                f"plan={(mask_plan.expected_batch, mask_plan.expected_key_len)}"
            )
        if not 0 <= mask_plan.key_valid_len <= K:
            raise ValueError(
                "Static FA mask key_valid_len must be in [0, K], "
                f"got key_valid_len={mask_plan.key_valid_len}, K={K}"
            )
        if mask_plan.expected_mask_dim not in (1, 2):
            raise ValueError(
                "Static FA mask expected_mask_dim must be 1 or 2, "
                f"got {mask_plan.expected_mask_dim}"
            )
        if mask.dim() != mask_plan.expected_mask_dim:
            raise ValueError(
                "Static FA mask dim mismatch: "
                f"got {mask.dim()}, expected {mask_plan.expected_mask_dim}"
            )

        if mask_plan.expected_mask_dim == 1:
            if tuple(mask.shape) != (K,):
                raise ValueError(
                    "Static FA sequence mask shape mismatch: "
                    f"got {tuple(mask.shape)}, expected {(K,)}"
                )

            # Valid keys form a prefix; omit the mask when no padding is present.
            if mask_plan.key_valid_len == K:
                return None, None
            reference = mask.to(dtype=torch.bool)
            valid_mask = reference.view(1, K).expand(Q, K)
            return (~valid_mask).contiguous(), None

        if tuple(mask.shape) != (B, K):
            raise ValueError(
                "Static FA Pair mask shape mismatch: "
                f"got {tuple(mask.shape)}, expected {(B, K)}"
            )
        if not mask_plan.uniform_valid_rows or not mask_plan.all_query_rows_valid:
            raise ValueError(
                "Static FA Pair mask plan requires uniform valid rows and valid queries"
            )
        if B <= 0:
            raise ValueError("Static FA Pair mask plan requires at least one valid query row")

        reference = mask[0].to(dtype=torch.bool)
        valid_mask = reference.view(1, K).expand(Q, K)
        return (~valid_mask).contiguous(), None

    if mask.all().item():
        return None, None

    if mask.dim() == 1:
        if mask.shape[0] != K:
            raise ValueError(
                f"Unsupported FA 1D mask shape {tuple(mask.shape)}; expected [{K}]"
            )

        # Fusion Attention consumes one shared [Q,K] mask for these logits.
        valid_mask = mask.to(dtype=torch.bool).view(1, K).expand(Q, K)
        return (~valid_mask).contiguous(), None

    if mask.dim() == 2:
        if tuple(mask.shape) != (B, K):
            raise ValueError(
                f"Unsupported FA 2D mask shape {tuple(mask.shape)}; expected [{B},{K}]"
            )

        mask_bool = mask.to(dtype=torch.bool)
        row_keep = mask_bool.any(dim=-1)
        if not row_keep.any().item():
            return None, row_keep

        valid_rows = mask_bool[row_keep]
        reference = valid_rows[0]
        if not torch.equal(valid_rows, reference.expand_as(valid_rows)):
            raise ValueError("Unsupported FA 2D mask: valid rows are not identical")

        query_batch_keep = None if row_keep.all().item() else row_keep
        if reference.all().item():
            return None, query_batch_keep

        # Compact [B,K] to one shared [Q,K] mask. Invalid query rows are
        # cleared after Fusion Attention returns.
        valid_mask = reference.view(1, K).expand(Q, K)
        return (~valid_mask).contiguous(), query_batch_keep

    raise ValueError(f"Unsupported FA mask dim: {mask.dim()}")


def dot_product_attention_FA_sequence_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_head: int,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    mask_plan: Optional[FAMaskPlan] = None,
) -> torch.Tensor:
    """Run Fusion Attention on pre-scaled BF16 [B,S,C] inputs.

    Projection and query scaling belong to the caller.  This helper accepts
    both square self-attention and rectangular local-Q/global-KV attention,
    normalizes mask/PSE shapes, and invokes the NPU operator without another
    layout conversion.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "Sequence-major Fusion Attention requires three-dimensional Q/K/V; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if k.shape != v.shape:
        raise ValueError(
            "Sequence-major Fusion Attention requires matching K/V; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.size(0) != k.size(0) or q.size(-1) != k.size(-1):
        raise ValueError(
            "Sequence-major Fusion Attention requires matching batch and "
            "hidden dimensions; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
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

    B, Q, _ = q.shape
    K = k.shape[1]
    pse = _prepare_fa_bias_for_shape(
        bias,
        B=B,
        H=num_head,
        Q=Q,
        K=K,
    )
    atten_mask, query_batch_keep = _prepare_fa_mask_for_shape(
        mask,
        B=B,
        Q=Q,
        K=K,
        mask_plan=mask_plan,
    )
    if query_batch_keep is not None and not query_batch_keep.any().item():
        return torch.zeros_like(q)

    pse_in = None if pse is None else pse.to(torch.bfloat16).contiguous()
    out = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=num_head,
        input_layout="BSH",
        pse=pse_in,
        atten_mask=atten_mask,
        scale=1.0,
    )[0]

    if query_batch_keep is not None:
        keep = query_batch_keep.to(device=out.device, dtype=out.dtype)
        out = out * keep.view(-1, 1, 1)
    return out


def dot_product_attention_FA(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_head: int,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    mask_plan: Optional[FAMaskPlan] = None,
):
    scaling = q.size(-1) ** -0.5
    q = (q * scaling).contiguous()
    k = k.contiguous()
    v = v.contiguous()

    pse = _prepare_fa_bias(bias, q, k)
    atten_mask, query_batch_keep = _prepare_fa_mask(
        mask,
        q,
        k,
        mask_plan=mask_plan,
    )

    if query_batch_keep is not None and not query_batch_keep.any().item():
        return torch.zeros_like(q)

    pse_in = pse
    if pse_in is not None:
        # A caller-cached BF16 bias makes this conversion a no-op across chunks.
        pse_in = pse_in.to(torch.bfloat16)
    q_in = q.to(torch.bfloat16)
    k_in = k.to(torch.bfloat16)
    v_in = v.to(torch.bfloat16)

    out = torch_npu.npu_fusion_attention(
        q_in,
        k_in,
        v_in,
        head_num=num_head,
        input_layout="BNSD",
        pse=pse_in,
        atten_mask=atten_mask,
        scale=1.0,
    )[0]

    if query_batch_keep is not None:
        keep = query_batch_keep.to(device=out.device, dtype=out.dtype)
        out = out * keep.view(-1, 1, 1, 1)

    return out


def dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_head: int,
    mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    mask_plan: Optional[FAMaskPlan] = None,
):
    impl = fastnn_config.dot_product_attention_implementation

    if impl == "torch":
        return dot_product_attention_torch(q, k, v, num_head, mask, bias)

    if impl == "Fusion_Attention":
        return dot_product_attention_FA(
            q,
            k,
            v,
            num_head,
            mask,
            bias,
            mask_plan=mask_plan,
        )

    raise ValueError(
        f"Unknown dot_product_attention_implementation: {impl}"
    )
