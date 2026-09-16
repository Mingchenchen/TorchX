from functools import partial
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size


# ---------------------------------------------------------------------------
# LayerNorm
# Pure-torch path; no Triton/fused kernel compilation.
# ---------------------------------------------------------------------------

_shape_t = Union[int, List[int], Size]

import os as _os
_USE_FAST_LN = _os.environ.get("LAYERNORM_TYPE", "").lower() == "fast_layernorm"
_FUSED_LN_FN = None
def _fused_ln_fn():
    global _FUSED_LN_FN
    if _FUSED_LN_FN is None:
        from torchfold.model.layer_norm.layer_norm import FusedLayerNormAffineFunction
        _FUSED_LN_FN = FusedLayerNormAffineFunction
    return _FUSED_LN_FN


class LayerNorm(nn.Module):
    """Pure-torch LayerNorm with weight/bias parameters.

    forward that never triggers CUDA/Triton kernel compilation.
    """

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Fused CUDA LayerNorm when enabled. Numerically
        # equivalent to F.layer_norm (eps, affine); verified bit-parity in tests.
        if _USE_FAST_LN and input.is_cuda and self.elementwise_affine and self.weight is not None:
            return _fused_ln_fn().apply(
                input, self.weight, self.bias, self.normalized_shape, self.eps
            )
        d = input.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast("cuda", enabled=False):
                w = self.weight.to(dtype=d) if self.weight is not None else None
                b = self.bias.to(dtype=d) if self.bias is not None else None
                return F.layer_norm(input, self.normalized_shape, w, b, self.eps)
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class OuterProductMean(nn.Module):
    """Outer product mean (Algorithm 9 in AF3).

    """

    def __init__(
        self,
        c_msa: int = 64,
        num_output_channel: int = 128,
        num_outer_channel: int = 32,
    ) -> None:
        super(OuterProductMean, self).__init__()

        self.c_msa = c_msa
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.epsilon = 1e-3

        self.layer_norm_input = LayerNorm(self.c_msa)
        self.left_projection = nn.Linear(self.c_msa, self.num_outer_channel, bias=False)
        self.right_projection = nn.Linear(self.c_msa, self.num_outer_channel, bias=False)

        self.output_w = nn.Parameter(
            torch.randn(
                self.num_outer_channel, self.num_outer_channel, self.num_output_channel
            )
        )
        self.output_b = nn.Parameter(torch.randn(self.num_output_channel))

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left_act = mask * self.left_projection(msa)
        right_act = mask * self.right_projection(msa)

        left_act = left_act.permute(0, 2, 1)
        act = torch.einsum("acb,ade->dceb", left_act, right_act)
        act = torch.einsum("dceb,cef->dbf", act, self.output_w) + self.output_b
        act = act.permute(1, 0, 2)

        norm = torch.einsum("abc,adc->bdc", mask, mask)
        return act / (self.epsilon + norm)


from functools import partialmethod as _partialmethod


class Dropout(nn.Module):
    """Dropout with the mask shared along the given dim(s)."""

    def __init__(self, r: float, batch_dim) -> None:
        super().__init__()
        self.r = float(r)
        self.batch_dim = [batch_dim] if isinstance(batch_dim, int) else list(batch_dim)
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.r == 0.0:
            return x
        shape = list(x.shape)
        for bd in self.batch_dim:
            shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))


class DropoutRowwise(Dropout):
    __init__ = _partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    __init__ = _partialmethod(Dropout.__init__, batch_dim=-2)
