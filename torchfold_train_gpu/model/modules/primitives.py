from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm lives in triangular/layers.py
from torchfold.model.triangular.layers import LayerNorm  # noqa: F401


class Linear(nn.Linear):
    """Linear layer with customized initialization.

    Subclasses nn.Linear so .weight has shape [out_features, in_features],
    matching the AF3 checkpoint layout exactly.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to use bias. Defaults to True.
        precision: Optional dtype override for forward computation.
        initializer: Weight init scheme ('default', 'relu', 'zeros').
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        precision: Optional[torch.dtype] = None,
        initializer: str = "default",
    ) -> None:
        self.use_bias = bias
        self.precision = precision
        self.initializer = initializer
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.precision is not None:
            input_dtype = input.dtype
            with torch.amp.autocast("cuda", enabled=False):
                bias = (
                    self.bias.to(dtype=self.precision)
                    if self.bias is not None
                    else None
                )
                return F.linear(
                    input.to(dtype=self.precision),
                    self.weight.to(dtype=self.precision),
                    bias,
                ).to(dtype=input_dtype)
        return F.linear(input, self.weight, self.bias)


LinearNoBias = partial(Linear, bias=False)


class Transition(nn.Module):
    """Transition block (Algorithm 11 in AF3).

    Uses vendored LayerNorm from torchfold.model.triangular.layers;
    Parameter layout (state_dict keys + shapes) is identical to
    """

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = LayerNorm(c_x)
        self.transition1 = LinearNoBias(c_x, self.num_intermediate_factor * c_x * 2)
        self.transition2 = LinearNoBias(self.num_intermediate_factor * c_x, c_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        y = self.transition1(x)
        a, b = y.chunk(2, dim=-1)
        c = F.silu(a) * b
        return self.transition2(c)
