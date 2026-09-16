import torch


def gated_linear_unit_torch(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out


def gated_linear_unit(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = gated_linear_unit_torch(x, weight)
    return out
