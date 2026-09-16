"""Single- and multi-card padding rules, without tensor dependencies.

Buckets are selected before featurisation; parallel alignment uses feature
length. Tensor padding and sharding are implemented in parallel_ops.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from torchfold.runtime_policy import (
    SINGLE_CARD_PADDING_THRESHOLD,
    SINGLE_CARD_PADDING_SHORT_ALIGNMENT,
    SINGLE_CARD_PADDING_LONG_ALIGNMENT,
    PARALLEL_PADDING_ALIGNMENT,
)


def single_card_padding_alignment(num_tokens: int) -> int:
    """Choose alignment from the original total length, before padding."""
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
    if num_tokens < SINGLE_CARD_PADDING_THRESHOLD:
        return SINGLE_CARD_PADDING_SHORT_ALIGNMENT
    return SINGLE_CARD_PADDING_LONG_ALIGNMENT


def resolve_inference_buckets(
    *,
    chain_lengths: Iterable[int],
    world_size: int,
    buckets: Sequence[int] | None,
) -> Sequence[int] | None:
    """Preserve explicit buckets; otherwise align single-card features only."""
    if buckets is not None:
        return buckets
    if world_size != 1:
        return None

    num_tokens = sum(chain_lengths)
    alignment = single_card_padding_alignment(num_tokens)
    padded_tokens = ((num_tokens + alignment - 1) // alignment) * alignment
    return (padded_tokens,)


def parallel_padding_multiple(world_size: int) -> int:
    """Return the global alignment multiple, reading the override per call."""
    align_to = int(os.environ.get(
        "TORCHFOLD_PARALLEL_PADDING_ALIGNMENT", str(PARALLEL_PADDING_ALIGNMENT)))
    if align_to < 1:
        raise ValueError(
            "TORCHFOLD_PARALLEL_PADDING_ALIGNMENT must be >= 1, "
            f"got {align_to}"
        )
    return world_size * align_to


def parallel_padded_length(feature_length: int, world_size: int) -> int:
    """Round feature length up so every rank has an aligned, equal shard."""
    multiple = parallel_padding_multiple(world_size)
    return ((feature_length + multiple - 1) // multiple) * multiple
