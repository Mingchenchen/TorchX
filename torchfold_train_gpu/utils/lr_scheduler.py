"""torchfold.utils.lr_scheduler
"""

from __future__ import annotations

import math
import warnings

import torch
from torch.optim.lr_scheduler import LRScheduler


class AlphaFold3LRScheduler(LRScheduler):
    """AF3 schedule: linear warmup then step-decay ( ``af3``)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        warmup_steps: int = 1000,
        lr: float = 1.8e-3,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_every_n_steps
        self.lr = lr
        self.decay_factor = decay_factor
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def _get_step_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            lr = step / self.warmup_steps * self.lr
        else:
            decay_count = step // self.decay_steps
            lr = self.lr * (self.decay_factor ** decay_count)
        return lr

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        return [
            self._get_step_lr(self.last_epoch) for _ in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float]:
        return [self._get_step_lr(self.last_epoch) for _ in self.base_lrs]


class CosineAnnealingWithWarmup(LRScheduler):
    """Linear warmup then cosine decay to ``min_lr`` """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        decay_steps: int,
        lr: float,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.lr = lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def _get_step_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return (step + 1) / (self.warmup_steps + 1) * self.lr
        elif step >= self.decay_steps:
            return self.min_lr
        else:
            decay_ratio = (step - self.warmup_steps) / (
                self.decay_steps - self.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.min_lr + coff * (self.lr - self.min_lr)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        return [
            self._get_step_lr(self.last_epoch) for _ in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float]:
        return [self._get_step_lr(self.last_epoch) for _ in self.base_lrs]


def get_lr_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    *,
    lr: float = 1.8e-3,
    warmup_steps: int = 1000,
    decay_every_n_steps: int = 50000,
    decay_factor: float = 0.95,
    max_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    **kwargs,
) -> LRScheduler:
    """Build an LR scheduler by name (``"af3"`` or ``"cosine_annealing"``)."""
    if name == "af3":
        return AlphaFold3LRScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            lr=lr,
            decay_every_n_steps=decay_every_n_steps,
            decay_factor=decay_factor,
            **kwargs,
        )
    elif name == "cosine_annealing":
        return CosineAnnealingWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            lr=lr,
            min_lr=lr * min_lr_ratio,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid lr scheduler: [{name}]")
