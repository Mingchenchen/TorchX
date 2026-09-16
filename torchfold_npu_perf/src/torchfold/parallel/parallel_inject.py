from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .parallel_config import ParallelConfig


@dataclass(frozen=True)
class ParallelInjectReport:
    enabled: bool = False
    model_replaced: bool = False
    missing_keys: int = 0
    unexpected_keys: int = 0

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "model_replaced": self.model_replaced,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
        }


def inject_parallel_model(
    model: nn.Module,
    cfg: ParallelConfig,
) -> tuple[nn.Module, ParallelInjectReport]:
    if not cfg.enabled:
        return model, ParallelInjectReport(enabled=False)

    from .parallel_impl.torchfold_model import TorchFold as ParallelTorchFold

    new_model = ParallelTorchFold(
        num_recycles=getattr(model, "num_recycles", 10),
        num_samples=getattr(model, "num_samples", 5),
        diffusion_steps=getattr(model, "diffusion_steps", 200),
        diffusion_sample_parallel=getattr(
            model,
            "diffusion_sample_parallel",
            True,
        ),
    )

    incompatible = new_model.load_state_dict(model.state_dict(), strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if cfg.strict and (missing or unexpected):
        raise RuntimeError(
            "Parallel injection state transfer failed: "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )

    new_model.eval()
    return new_model, ParallelInjectReport(
        enabled=True,
        model_replaced=True,
        missing_keys=len(missing),
        unexpected_keys=len(unexpected),
    )
