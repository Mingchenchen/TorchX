import torch
import torch_npu

def gated_linear_unit_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of ``SiLU(gate) * value``."""
    projection = torch.matmul(x, weight)
    gate, value = torch.chunk(projection, 2, dim=-1)
    return torch.nn.functional.silu(gate) * value


def gated_linear_unit_npu_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Project once and fuse the split, SiLU, and multiply operations."""
    projection = torch.matmul(x, weight)
    if projection.size(-1) % 2 != 0:
        raise ValueError(
            "SwiGLU projection width must be even; "
            f"got shape={tuple(projection.shape)}"
        )
    if not projection.is_contiguous():
        projection = projection.contiguous()

    output = torch_npu.npu_swiglu(projection, dim=-1)
    del projection
    return output


def gated_linear_unit(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed production SwiGLU backend."""
    return gated_linear_unit_npu_swiglu(x, weight)
