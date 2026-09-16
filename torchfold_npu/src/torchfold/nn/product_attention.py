from typing import Optional

import torch
import torch_npu

from torchfold.nn import fastnn_config


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


def dot_product_attention_FA(q: torch.Tensor,
                             k: torch.Tensor,
                             v: torch.Tensor,
                             num_head: int,
                             mask: Optional[torch.Tensor] = None,
                             bias: Optional[torch.Tensor] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    bias = bias.unsqueeze(0)

    if mask is not None:
        if mask.all().item():
            mask = None
        else:
            S = mask.size(-1)
            if mask.dim() == 1:
                optimized_mask = mask.expand(S, S).to(dtype=torch.bool)
            elif mask.dim() == 2:
                B = mask.size(0)
                mask = mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)
                optimized_mask = mask.expand(B, 1, S, S)
            mask = ~optimized_mask.contiguous()

    return torch_npu.npu_fusion_attention(
        q, k, v, head_num=num_head, input_layout="BNSD", pse=bias,
        atten_mask=mask, scale=1.0)[0]


def dot_product_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          num_head: int,
                          mask: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None):
    if fastnn_config.dot_product_attention_implementation == "torch":
        out = dot_product_attention_torch(q, k, v, num_head, mask, bias)
    elif fastnn_config.dot_product_attention_implementation == "Fusion_Attention":
        out = dot_product_attention_FA(q, k, v, num_head, mask, bias)
    else:
        raise ValueError(
            f"Unknown dot_product_attention_implementation: {fastnn_config.dot_product_attention_implementation}"
        )

    return out
