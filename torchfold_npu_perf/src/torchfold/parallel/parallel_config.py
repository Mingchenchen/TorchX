from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None


# Exclusive threshold: multi-card sample/step parallelism supports up to 3110.
from torchfold.runtime_policy import SHORT_DIFFUSION_THRESHOLD


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class ParallelConfig:
    enabled: bool = False
    strict: bool = True
    verbose: bool = True


def load_parallel_config(verbose: bool = True) -> ParallelConfig:
    enabled = (
        dist is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    strict = _env_bool("PARALLEL_STRICT", True)
    return ParallelConfig(
        enabled=enabled,
        strict=strict,
        verbose=verbose,
    )
