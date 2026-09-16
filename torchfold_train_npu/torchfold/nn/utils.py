


import numbers
from collections import abc

import torch


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

    broadcast_factor = torch.Tensor([1.0]).to(value.dtype).npu()
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
