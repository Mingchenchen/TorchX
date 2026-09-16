"""torchfold.runner.ema.


"""

from __future__ import annotations

from typing import List, Optional

import torch


class EMAWrapper(object):
    """Exponential moving average of model parameters (fp32 shadow)."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        mutable_param_keywords: Optional[List[str]] = None,
    ):
        self.model = model
        self.decay = decay
        if mutable_param_keywords is not None:
            self.mutable_param_keywords = [
                s.strip() for s in mutable_param_keywords if s.strip()
            ]
        else:
            self.mutable_param_keywords = None
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

    def register(self) -> None:
        """Init shadow copies (fp32) from the current model parameters."""
        self.shadow = {}
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.detach().clone().to(torch.float32)

    @torch.no_grad()
    def update(self) -> None:
        """shadow <- decay * shadow + (1 - decay) * param (fp32)."""
        for name, param in self.model.named_parameters():
            if self.mutable_param_keywords and not any(
                kw in name for kw in self.mutable_param_keywords
            ):
                continue
            assert name in self.shadow, f"param {name} not registered in EMA"
            p32 = param.detach().to(torch.float32)
            new_average = (1.0 - self.decay) * p32 + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Swap shadow (EMA) weights into the model, backing up the live ones."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            assert name in self.shadow, f"param {name} not registered in EMA"
            self.backup[name] = param.data
            param.data = self.shadow[name].to(dtype=param.dtype, device=param.device)

    def restore(self) -> None:
        """Restore the live model weights backed up by ``apply_shadow``."""
        for name, param in self.model.named_parameters():
            assert name in self.backup, f"param {name} has no backup to restore"
            param.data = self.backup[name]
        self.backup = {}

    # -- convenience aliases ------------------------------------------------
    def apply(self) -> None:
        self.apply_shadow()

    # -- checkpoint ---------------------------------------------------------
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state.get("decay", self.decay)
        shadow = state["shadow"]
        self.shadow = {
            k: v.detach().clone().to(torch.float32) for k, v in shadow.items()
        }
