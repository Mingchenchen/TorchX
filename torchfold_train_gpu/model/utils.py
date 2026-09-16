from collections import abc
import numbers

import torch
from typing import Union
import torch.nn as nn


def mask_mean(mask, value, dim=None, keepdim=False, eps=1e-10):
    """Masked mean."""

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(
        value_shape
    ), 'Shapes are not compatible, shapes: {}, {}'.format(mask_shape, value_shape)

    if isinstance(dim, numbers.Integral):
        dim = [dim]
    elif dim is None:
        dim = list(range(len(mask_shape)))
    assert isinstance(
        dim, abc.Iterable
    ), 'axis needs to be either an iterable, integer or "None"'

    broadcast_factor = torch.Tensor([1.0]).to(value.dtype).cuda()
    for dim_ in dim:
        value_size = value_shape[dim_]
        mask_size = mask_shape[dim_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            error = f'Shapes are not compatible, shapes: {mask_shape}, {value_shape}'
            assert mask_size == value_size, error

    return torch.sum(mask * value, keepdim=keepdim, dim=dim) / (
        torch.clamp(torch.sum(mask, keepdim=keepdim, dim=dim) * broadcast_factor, min=eps)
    )


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times."""
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1:]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad x at dimension dim with pad_length[0] left and pad_length[1] right."""
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)
