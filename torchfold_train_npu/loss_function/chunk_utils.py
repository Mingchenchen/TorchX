#!/usr/bin/env python3
"""
Chunk helper utilities – simplified (inner chunks only).
Used to split large tensors such as distance matrices inside loss functions to
avoid running out of memory.
"""

from functools import partial
from typing import Callable, Optional

import torch
import torch.utils.checkpoint


def get_checkpoint_fn():
    """Return a checkpointing helper (non-reentrant) to save memory."""
    return partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)


def chunk_forward(
    forward_fn: Callable,
    data: torch.Tensor,
    chunk_size: int,
    dim: int = -3,
    **kwargs
) -> torch.Tensor:
    """
    Generic chunked forward helper.
    
    Args:
        forward_fn: Callable applied to each chunk.
        data: Input tensor [..., N_sample, ...].
        chunk_size: Number of samples per chunk.
        dim: Dimension to chunk (default -3, typically the N_sample axis).
        **kwargs: Extra keyword arguments for forward_fn.
    
    Returns:
        Concatenated result over all chunks.
    
    Example:
        >>> def my_loss(pred, true):
        >>>     return (pred - true) ** 2
        >>> 
        >>> pred = torch.randn(100, 1000, 3)  # [N_sample=100, N_atoms, 3]
        >>> true = torch.randn(1000, 3)
        >>> 
        >>> # Without chunking (may OOM)
        >>> loss = my_loss(pred, true)
        >>> 
        >>> # Chunked execution
        >>> loss = chunk_forward(my_loss, pred, chunk_size=10, true=true)
    """
    N_sample = data.shape[dim]
    
    if chunk_size is None or chunk_size <= 0 or N_sample <= chunk_size:
        # No chunking required
        return forward_fn(data, **kwargs)
    
    # Iterate over chunks
    results = []
    for i in range(0, N_sample, chunk_size):
        end_idx = min(i + chunk_size, N_sample)
        
        # Slice the current chunk
        if dim == -3:
            chunk_data = data[..., i:end_idx, :, :]
        elif dim == 0:
            chunk_data = data[i:end_idx]
        else:
            # Generic slicing (slower path)
            slices = [slice(None)] * data.ndim
            slices[dim] = slice(i, end_idx)
            chunk_data = data[tuple(slices)]
        
        # Run the forward pass on this chunk
        chunk_result = forward_fn(chunk_data, **kwargs)
        results.append(chunk_result)
    
    # Concatenate chunk outputs
    return torch.cat(results, dim=dim)


def chunk_mean(
    forward_fn: Callable,
    data: torch.Tensor,
    chunk_size: int,
    dim: int = -3,
    **kwargs
) -> torch.Tensor:
    """
    Chunk a tensor, compute losses per chunk, and average them.
    
    Useful for losses that must evaluate per chunk before averaging.
    
    Args:
        forward_fn: Forward function returning a scalar or [N_sample] tensor.
        data: Input tensor [..., N_sample, ...].
        chunk_size: Number of samples per chunk.
        dim: Dimension to chunk (default -3).
        **kwargs: Extra keyword arguments for forward_fn.
    
    Returns:
        Scalar average loss.
    
    Example:
        >>> def compute_loss(pred):
        >>>     return pred.mean(dim=(-2, -1))  # [N_sample]
        >>> 
        >>> pred = torch.randn(100, 1000, 3)
        >>> loss = chunk_mean(compute_loss, pred, chunk_size=10)
    """
    N_sample = data.shape[dim]
    
    if chunk_size is None or chunk_size <= 0 or N_sample <= chunk_size:
        # No chunking needed
        result = forward_fn(data, **kwargs)
        if result.ndim == 0:
            return result
        else:
            return result.mean()
    
    # Iterate through chunks
    losses = []
    for i in range(0, N_sample, chunk_size):
        end_idx = min(i + chunk_size, N_sample)
        
        # Slice the chunk
        if dim == -3:
            chunk_data = data[..., i:end_idx, :, :]
        elif dim == 0:
            chunk_data = data[i:end_idx]
        else:
            slices = [slice(None)] * data.ndim
            slices[dim] = slice(i, end_idx)
            chunk_data = data[tuple(slices)]
        
        # Compute chunk loss
        chunk_loss = forward_fn(chunk_data, **kwargs)
        
        # Reduce [N_sample] tensors to scalars if needed
        if chunk_loss.ndim > 0:
            chunk_loss = chunk_loss.mean()
        
        losses.append(chunk_loss)
    
    # Average over chunk losses
    return torch.stack(losses).mean()


class ChunkProcessor:
    """
    Convenience wrapper around chunk_forward/chunk_mean for loss functions.
    
    Example:
        >>> processor = ChunkProcessor(chunk_size=10)
        >>> 
        >>> # Option 1: run chunked forward
        >>> result = processor.forward(my_fn, data, arg1=val1)
        >>> 
        >>> # Option 2: chunked mean
        >>> loss = processor.mean(loss_fn, data, arg1=val1)
    """
    
    def __init__(self, chunk_size: Optional[int] = None):
        """
        Args:
            chunk_size: Chunk length; None disables chunking.
        """
        self.chunk_size = chunk_size
    
    def forward(
        self,
        forward_fn: Callable,
        data: torch.Tensor,
        dim: int = -3,
        **kwargs
    ) -> torch.Tensor:
        """
        Run a forward pass with chunked inputs.
        
        Args:
            forward_fn: Callable applied per chunk.
            data: Input tensor.
            dim: Dimension to chunk.
            **kwargs: Extra arguments.
        
        Returns:
            Concatenated tensor over all chunks.
        """
        return chunk_forward(forward_fn, data, self.chunk_size, dim=dim, **kwargs)
    
    def mean(
        self,
        forward_fn: Callable,
        data: torch.Tensor,
        dim: int = -3,
        **kwargs
    ) -> torch.Tensor:
        """
        Chunk inputs, compute per-chunk losses, and average them.
        
        Args:
            forward_fn: Callable returning scalars or vectors.
            data: Input tensor.
            dim: Dimension along which to chunk.
            **kwargs: Extra arguments.
        
        Returns:
            Scalar mean across chunk losses.
        """
        return chunk_mean(forward_fn, data, self.chunk_size, dim=dim, **kwargs)
    
    def process_mean(self, *args, **kwargs):
        """Alias for mean (kept for backward compatibility)."""
        return self.mean(*args, **kwargs)




# Export public API
__all__ = [
    'get_checkpoint_fn',
    'chunk_forward',
    'chunk_mean',
    'ChunkProcessor',
]
