import dataclasses

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class GatherInfo:
    """Tensor indices and masks for conversion between atom layouts.

    Attributes:
      gather_idxs: Integer torch.Tensor indexing the flattened source layout.
      gather_mask: Boolean torch.Tensor with the same shape as gather_idxs.
      input_shape: Integer torch.Tensor describing the unflattened source shape;
        its shape need not match gather_idxs or gather_mask.
    """

    gather_idxs: torch.Tensor
    gather_mask: torch.Tensor
    input_shape: torch.Tensor

    def __post_init__(self):
        if self.gather_mask.shape != self.gather_idxs.shape:
            raise ValueError(
                'All arrays must have the same shape. Got\n'
                f'gather_idxs.shape = {self.gather_idxs.shape}\n'
                f'gather_mask.shape = {self.gather_mask.shape}\n'
            )

def convert(
    gather_info: GatherInfo,
    arr: torch.Tensor,
    *,
    layout_axes: tuple[int, ...] = (0,),
) -> torch.Tensor:
    """Convert an array from one atom layout to another."""
    layout_axes = tuple(i if i >= 0 else i + arr.ndim for i in layout_axes)

    # Ensure that layout_axes are continuous.
    layout_axes_begin = layout_axes[0]
    layout_axes_end = layout_axes[-1] + 1

    if layout_axes != tuple(range(layout_axes_begin, layout_axes_end)):
        raise ValueError(f'layout_axes must be continuous. Got {layout_axes}.')
    layout_shape = arr.shape[layout_axes_begin:layout_axes_end]

    # we force the input_shape to be on the CPU
    assert gather_info.input_shape.device == torch.device("cpu")
    gather_info_input_shape = gather_info.input_shape.numpy()

    if (len(layout_shape) != gather_info_input_shape.size) or (
        isinstance(gather_info_input_shape, torch.Tensor)
        and (
            (layout_shape[0] < gather_info_input_shape[0])
            or (np.any(layout_shape[1:] != gather_info_input_shape[1:]))
        )
    ):
        raise ValueError(
            'Input array layout axes are incompatible. You specified layout '
            f'axes {layout_axes} with an input array of shape {arr.shape}, but '
            f'the gather info expects shape {gather_info.input_shape}. '
            'Your first axis size must be equal or greater than the '
            'gather_info.input_shape, and all subsequent axes sizes must '
            'match.'
        )

    batch_shape = arr.shape[:layout_axes_begin]
    features_shape = arr.shape[layout_axes_end:]
    arr_flattened_shape = batch_shape + \
        (np.prod(layout_shape),) + features_shape

    arr_flattened = arr.reshape(arr_flattened_shape)
    if layout_axes_begin == 0:
        out_arr = arr_flattened[gather_info.gather_idxs, ...]
    elif layout_axes_begin == 1:
        out_arr = arr_flattened[:, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 2:
        out_arr = arr_flattened[:, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 3:
        out_arr = arr_flattened[:, :, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 4:
        out_arr = arr_flattened[:, :, :, :, gather_info.gather_idxs, ...]
    else:
        raise ValueError(
            'Only 4 batch axes supported. If you need more, the code '
            'is easy to extend.'
        )

    broadcasted_mask_shape = (
        (1,) * len(batch_shape)
        + gather_info.gather_mask.shape
        + (1,) * len(features_shape)
    )
    out_arr *= gather_info.gather_mask.reshape(broadcasted_mask_shape)
    return out_arr
