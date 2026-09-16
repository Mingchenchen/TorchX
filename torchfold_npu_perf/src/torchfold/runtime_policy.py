"""Runtime defaults, branch decisions and request-scoped offload helpers.

Padding length rules live in padding. Diffusion environment overrides are
read at import time; padding and relative-encoding overrides are read per call.
PyTorch is imported only when offload operations execute, so configuration
and padding remain usable without tensor dependencies. Restart after edits.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps


# Single-card alignment uses total unpadded chain length; explicit buckets win.
# The short alignment applies strictly below the threshold.
SINGLE_CARD_PADDING_THRESHOLD = 2000
SINGLE_CARD_PADDING_SHORT_ALIGNMENT = 64
SINGLE_CARD_PADDING_LONG_ALIGNMENT = 128
PARALLEL_PADDING_ALIGNMENT = 64  # Per rank; global multiple = world_size * alignment.

# Sample parallelism uses feature length with exclusive upper thresholds.
SINGLE_CARD_SAMPLE_PARALLEL_THRESHOLD = 3251
SHORT_DIFFUSION_THRESHOLD = 3111
SINGLE_CARD_DIFFUSION_SAMPLE_GROUP_SIZE = 5

# Offload: valid tokens > threshold; all world_size > 1 share one threshold.
SINGLE_CARD_OFFLOAD_MIN_TOKENS = 4000
MULTI_CARD_OFFLOAD_MIN_TOKENS = 4100

# Inference: Pairformer chunks above the feature-length threshold; Evoformer always chunks.
PAIRFORMER_TRIANGLE_FULL_PATH_MAX_TOKENS = 3500
TRIANGLE_MULTIPLICATION_CHUNK_SIZE = 256
TRIANGLE_ATTENTION_CHUNK_SIZE = 256

# Distributed Multiplication chunks when spec.n_padded > threshold.
PARALLEL_TRIANGLE_FULL_PATH_MAX_TOKENS = 4100
PARALLEL_TRIANGLE_MULTIPLICATION_CHUNK_SIZE = 256
# Distributed Grid Attention always caps local batches, without a length threshold.
PARALLEL_GRID_ATTENTION_BATCH_CHUNK_SIZE = 256

# OPM/Transition inference residual streaming has no length threshold.
OUTER_PRODUCT_MEAN_D_CHUNK_SIZE = 256
TRANSITION_ROW_STREAM_CHUNK_SIZE = 256
PARALLEL_OUTER_PRODUCT_MEAN_D_CHUNK_SIZE = 256
PARALLEL_TRANSITION_ROW_STREAM_CHUNK_SIZE = 256
PARALLEL_PAIR_TRANSITION_CHUNK_SIZE = 16  # Non-streaming parallel forward_chunked path.
PARALLEL_REL_ENCODING_ROW_CHUNK_SIZE = 64
RECYCLE_PAIR_CPU_OFFLOAD_ROW_CHUNK_SIZE = 64  # Chunk size only; does not enable offload.
PREVIOUS_PAIR_ROW_CHUNK_SIZE = 64

# Native conditioning/cache streaming: no-grad inference, length >= threshold.
DIFFUSION_CONDITIONING_ROW_STREAM_THRESHOLD = int(
    os.environ.get("TORCHFOLD_LONG_SEQUENCE_THRESHOLD", "3700")
)
DIFFUSION_CONDITIONING_ROW_STREAM_CHUNK_SIZE = int(
    os.environ.get("TORCHFOLD_DIFFUSION_CONDITIONING_ROW_CHUNK_SIZE", "128")
)
DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD = DIFFUSION_CONDITIONING_ROW_STREAM_THRESHOLD
DIFFUSION_PAIR_LOGITS_ROW_STREAM_CHUNK_SIZE = int(
    os.environ.get("TORCHFOLD_DIFFUSION_PAIR_CACHE_ROW_CHUNK_SIZE", "64")
)
DIFFUSION_PAIR_TRANSITION_ROW_STREAM_CHUNK_SIZE = 256
PARALLEL_DIFFUSION_PAIR_TRANSITION_ROW_STREAM_CHUNK_SIZE = 256
# Parallel single-card fallback ignores the native threshold override.
PARALLEL_DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD = 3700

# Distogram starts chunking at the inclusive original-sequence-length threshold.
DISTOGRAM_ROW_CHUNK_THRESHOLD = 8000
DISTOGRAM_ROW_CHUNK_SIZE = 256


def single_card_sample_parallel(
    *, feature_length: int, enabled: bool,
    threshold: int = SINGLE_CARD_SAMPLE_PARALLEL_THRESHOLD,
) -> bool:
    return enabled and feature_length < threshold


def multi_card_sample_parallel(
    *, feature_length: int, enabled: bool, is_distributed: bool,
    threshold: int = SHORT_DIFFUSION_THRESHOLD,
) -> bool:
    return enabled and is_distributed and feature_length < threshold


def inference_chunking(*, training: bool, grad_enabled: bool) -> bool:
    """OPM, Transition and Evoformer residual chunks: inference only."""
    return not training and not grad_enabled


def pairformer_triangle_chunked(
    *, feature_length: int, training: bool, grad_enabled: bool,
) -> bool:
    return (
        inference_chunking(training=training, grad_enabled=grad_enabled)
        and feature_length > PAIRFORMER_TRIANGLE_FULL_PATH_MAX_TOKENS
    )




def parallel_triangle_chunked(*, padded_length: int, world_size: int) -> bool:
    """Enable extra Multiplication chunks above the padded-global upper bound."""
    return world_size > 1 and padded_length > PARALLEL_TRIANGLE_FULL_PATH_MAX_TOKENS


def diffusion_conditioning_row_streaming(
    *, feature_length: int, training: bool, grad_enabled: bool,
    threshold: int = DIFFUSION_CONDITIONING_ROW_STREAM_THRESHOLD,
) -> bool:
    return (
        inference_chunking(training=training, grad_enabled=grad_enabled)
        and feature_length >= threshold
    )


def diffusion_pair_logits_row_streaming(
    *, feature_length: int, training: bool, grad_enabled: bool,
    attention_implementation: str,
    threshold: int = DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
) -> bool:
    return (
        inference_chunking(training=training, grad_enabled=grad_enabled)
        and feature_length >= threshold
        and attention_implementation == "Fusion_Attention"
    )


def native_pair_logits_cache_enabled(
    *, feature_length: int, training: bool, grad_enabled: bool,
    row_stream_enabled: bool,
    threshold: int = DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
) -> bool:
    return (
        inference_chunking(training=training, grad_enabled=grad_enabled)
        and feature_length < threshold
    ) or row_stream_enabled


def parallel_pair_logits_cache_enabled(
    *, feature_length: int, training: bool, is_distributed: bool,
    threshold: int = PARALLEL_DIFFUSION_PAIR_LOGITS_SINGLE_CARD_CACHE_THRESHOLD,
) -> bool:
    # This path intentionally checks training mode, not gradient mode.
    return not training and (is_distributed or feature_length < threshold)


def grid_attention_batch_chunk_size(
    *, shard_size: int, padded_length: int, world_size: int,
) -> int:
    """Cap local Grid Attention batches independently of global length.

    Keep the layout arguments for caller compatibility. Native single-card
    attention has a separate policy.
    """
    return min(PARALLEL_GRID_ATTENTION_BATCH_CHUNK_SIZE, shard_size)


def relative_encoding_row_chunk_size(*, local_rows: int) -> int:
    chunk_size = int(os.environ.get(
        "PARALLEL_REL_ENCODING_CHUNK_SIZE", str(PARALLEL_REL_ENCODING_ROW_CHUNK_SIZE)))
    return local_rows if chunk_size <= 0 else chunk_size


def distogram_row_chunk_size(
    *, original_sequence_length: int, shard_size: int,
) -> int | None:
    if original_sequence_length < 1:
        raise ValueError(
            "Distogram original_sequence_length must be >= 1, got "
            f"{original_sequence_length}"
        )
    if original_sequence_length < DISTOGRAM_ROW_CHUNK_THRESHOLD:
        return None
    if shard_size < 1:
        raise ValueError(f"Distogram shard_size must be >= 1, got {shard_size}")
    return min(DISTOGRAM_ROW_CHUNK_SIZE, shard_size)


@dataclass(frozen=True)
class OffloadPolicy:
    num_tokens: int
    world_size: int
    inference: bool

    @property
    def implicit_pair(self):
        return self.inference and self.num_tokens > 0

    @property
    def large_pair(self):
        threshold = (
            SINGLE_CARD_OFFLOAD_MIN_TOKENS if self.world_size == 1
            else MULTI_CARD_OFFLOAD_MIN_TOKENS
        )
        return self.inference and self.num_tokens > threshold


_policy = ContextVar('torchfold_offload_policy', default=None)


@contextmanager
def policy_context(batch, *, training, world_size=1):
    import torch

    if world_size < 1:
        raise ValueError('offload world_size must be positive')
    mask = batch['seq_mask'] if isinstance(batch, dict) else batch.token_features.mask
    # One scalar read per forward, never a padded or rank-local dimension.
    length = int(torch.count_nonzero(mask).item())
    value = OffloadPolicy(length, world_size, not training and not torch.is_grad_enabled())
    token = _policy.set(value)
    try:
        yield value
    finally:
        _policy.reset(token)


def large_pair_offload(*, training, num_tokens=None, world_size=1):
    import torch

    if training or torch.is_grad_enabled():
        return False
    value = _policy.get()
    if value is not None:
        return value.large_pair
    return num_tokens is not None and OffloadPolicy(num_tokens, world_size, True).large_pair


def confidence_pair_offload(*, training, num_tokens, world_size):
    """Keep the immutable trunk Pair on CPU while Confidence uses working copies."""
    import torch

    if training or torch.is_grad_enabled() or world_size <= 1:
        return False
    mode = os.environ.get('TORCHFOLD_CONFIDENCE_PAIR_CPU_OFFLOAD', 'on').lower()
    if mode not in ('on', 'off', 'auto'):
        raise ValueError('TORCHFOLD_CONFIDENCE_PAIR_CPU_OFFLOAD must be on, off, or auto')
    if mode == 'auto':
        return large_pair_offload(
            training=training, num_tokens=num_tokens, world_size=world_size,
        )
    return mode == 'on'


def implicit_pair_enabled(*, training):
    import torch

    value = _policy.get()
    return not training and not torch.is_grad_enabled() and value is not None and value.implicit_pair


def clear_diffusion_state(model):
    """End prediction caches, including after an interrupted forward."""
    head = getattr(model, 'diffusion_head', None)
    if head is None:
        return
    for module in head.modules():
        if hasattr(module, 'clear_pair_logits_cache'):
            module.clear_pair_logits_cache()
        if hasattr(module, 'first_run'):
            module.first_run = True
            for name in ('single_cond', 'pair_cond', 'pair_cond_for_atom',
                         'queries_mask', 'keys_mask', 'queries_single_cond',
                         'keys_single_cond', 'pair_act', 'pair_logits'):
                if hasattr(module, name):
                    setattr(module, name, None)


def with_offload_policy(*, distributed=False):
    def decorate(forward):
        @wraps(forward)
        def wrapped(self, batch, *args, **kwargs):
            import torch

            world_size, rank = 1, 0
            if distributed and torch.distributed.is_initialized():
                from torchfold.parallel.parallel_ops import get_parallel_groups
                group = get_parallel_groups().parallel_group
                world_size = torch.distributed.get_world_size(group)
                rank = torch.distributed.get_rank(group)
            clear_diffusion_state(self)
            try:
                with policy_context(batch, training=self.training, world_size=world_size) as value:
                    if rank == 0:
                        print(f'[offload] policy=auto_v1 original_tokens={value.num_tokens} '
                              f'world_size={world_size} implicit_pair={value.implicit_pair} '
                              f'large_pair={value.large_pair} '
                              f'thresholds={SINGLE_CARD_OFFLOAD_MIN_TOKENS}/'
                              f'{MULTI_CARD_OFFLOAD_MIN_TOKENS}', flush=True)
                    return forward(self, batch, *args, **kwargs)
            finally:
                clear_diffusion_state(self)
        return wrapped
    return decorate


def add_previous_pair(
    module, pair, previous, *, row_chunk_size=PREVIOUS_PAIR_ROW_CHUNK_SIZE, stream=False,
):
    import torch

    if previous is None:
        zero = pair.new_zeros((1, 1, pair.shape[-1]))
        return pair.add_(module.prev_embedding(module.prev_embedding_layer_norm(zero)))
    if tuple(previous.shape) != tuple(pair.shape):
        raise ValueError('previous Pair shape does not match current Pair')
    if not stream and previous.device == pair.device:
        return pair.add_(module.prev_embedding(module.prev_embedding_layer_norm(previous)))
    if previous.device.type != 'cpu' or row_chunk_size < 1:
        raise ValueError('streaming requires a CPU Pair and positive row chunk size')
    for start in range(0, pair.shape[0], row_chunk_size):
        end = min(start + row_chunk_size, pair.shape[0])
        chunk = previous[start:end].to(device=pair.device)
        update = module.prev_embedding(module.prev_embedding_layer_norm(chunk))
        pair[start:end].add_(update)
        if pair.device.type == 'npu':
            torch.npu.synchronize()
        del chunk, update
    return pair


def release_pair_conditioning(head):
    import torch

    if not large_pair_offload(training=head.training):
        return
    if not head.transformer.has_pair_logits_cache() or head.atom_cross_att_encoder.first_run:
        return
    if head.pair_cond is not None and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        print('[offload] release_conditioning_applied=True (derived caches ready)', flush=True)
    head.pair_cond = None
    if hasattr(head, 'pair_cond_for_atom'):
        head.pair_cond_for_atom = None
    head.transformer.release_pair_logits_cache_source()
