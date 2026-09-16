from __future__ import annotations

import torch
from torchfold.runtime_policy import large_pair_offload


def inference_output_offload_enabled(*, training: bool, parallel_spec=None) -> bool:
    """Return whether inference-only output offload is safe in this context."""
    return large_pair_offload(
        training=training,
        num_tokens=getattr(parallel_spec, 'n_global', None),
        world_size=getattr(parallel_spec, 'world_size', 1),
    )


def offload_confidence_output(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Move a completed sample's outputs to CPU and drop device references."""
    offloaded = {}
    for key in list(output):
        offloaded[key] = output.pop(key).cpu()
    return offloaded


def stack_confidence_outputs(
    outputs_per_sample: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack sample outputs and release their superseded references."""
    if not outputs_per_sample:
        raise ValueError("Confidence outputs must contain at least one sample")
    stacked = {
        key: torch.stack(
            [sample[key] for sample in outputs_per_sample],
            dim=0,
        )
        for key in outputs_per_sample[0]
    }
    outputs_per_sample.clear()
    return stacked
