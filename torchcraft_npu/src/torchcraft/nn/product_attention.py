from typing import Optional

import torch
import torch_npu

from torchcraft.nn import fastnn_config


def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
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


def dot_product_attention_torch_FA(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    bias = bias.unsqueeze(0)

    return torch_npu.npu_fusion_attention(
            q, k, v, head_num=q.size(1), input_layout="BNSD", pse=bias, scale=1.0)[0]


def dot_product_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None):
    if fastnn_config.dot_product_attention_implementations == "Fusion_Attention":
        out = dot_product_attention_torch_FA(q, k, v, mask, bias)
    else:
        out = dot_product_attention_torch(q, k, v, mask, bias)

    return out
