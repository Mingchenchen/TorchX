import torch

from torchfold.nn import fastnn_config


def gated_linear_unit_torch(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out


def gated_linear_unit(x, weight):
    if fastnn_config.gated_linear_unit_implementation == "torch":
        out = gated_linear_unit_torch(x, weight)
    return out
