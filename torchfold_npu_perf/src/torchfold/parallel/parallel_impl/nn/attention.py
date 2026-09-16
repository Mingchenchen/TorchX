from dataclasses import dataclass
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchfold.nn import fastnn_config

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from .. import fastnn
from ..fastnn import attention as fastnn_attention_impl
from ...parallel_ops import (
    ParallelSpec,
    all_gather_first_axis,
    pair_padding_mask_col_from_row,
    pad_to_length,
    zero_local_padding,
    zero_global_padding,
)


from torchfold.runtime_policy import (
    PARALLEL_GRID_ATTENTION_BATCH_CHUNK_SIZE as _GRID_ATTENTION_BATCH_CHUNK_SIZE,
    grid_attention_batch_chunk_size,
)

def _resolve_grid_attention_batch_chunk_size(
    parallel_spec: ParallelSpec,
) -> int:
    """Use at most 256 local rows at every global sequence length."""
    return grid_attention_batch_chunk_size(
        shard_size=parallel_spec.shard_size,
        padded_length=parallel_spec.n_padded,
        world_size=parallel_spec.world_size,
    )

@dataclass
class _PendingGridTransposeColChunk:
    """In-flight row-to-column AllToAll chunk.

    Buffers must live through ``work.wait()``; the range keeps collective order
    aligned across ranks.
    """

    local_col_start: int
    local_col_end: int
    send_buffer: Optional[torch.Tensor]
    recv_buffer: Optional[torch.Tensor]
    work: object


def _zero_parallel_pair_padding(
    x: torch.Tensor,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """Clear padded local rows and padded global columns in pair tensors."""
    local_valid = max(
        0,
        min(parallel_spec.end, parallel_spec.n_global) - parallel_spec.start,
    )
    if local_valid < x.shape[0]:
        x[local_valid:] = 0
    if x.ndim >= 2 and parallel_spec.n_global < x.shape[1]:
        x[:, parallel_spec.n_global:] = 0
    return x


def _parallel_local_valid_size(
    parallel_spec: ParallelSpec,
    local_size: int,
) -> int:
    return max(
        0,
        min(
            local_size,
            min(parallel_spec.end, parallel_spec.n_global) - parallel_spec.start,
        ),
    )


def _parallel_transpose_col_local_chunk(
    x_row: torch.Tensor,
    parallel_spec: ParallelSpec,
    local_col_start: int,
    local_col_end: int,
) -> torch.Tensor:
    """Convert one local column window from row layout to column layout."""
    if dist is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed GridSelfAttention chunking requires torch.distributed.")

    x_row = pad_to_length(x_row, dim=1, length=parallel_spec.n_padded, value=0)
    _zero_parallel_pair_padding(x_row, parallel_spec)

    send_buffer = _pack_parallel_transpose_col_local_chunk(
        x_row,
        parallel_spec,
        local_col_start,
        local_col_end,
    )
    recv_buffer = torch.empty_like(send_buffer)
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        group=parallel_spec.group,
    )
    out = _materialize_parallel_transpose_col_local_chunk(
        recv_buffer,
        parallel_spec,
    )
    if parallel_spec.n_global < out.shape[1]:
        out[:, parallel_spec.n_global:] = 0
    return out


def _pack_parallel_transpose_col_local_chunk(
    x_row: torch.Tensor,
    parallel_spec: ParallelSpec,
    local_col_start: int,
    local_col_end: int,
) -> torch.Tensor:
    """Pack Grid row->column sends as [destination, chunk, local_row, ...]."""
    shard = parallel_spec.shard_size
    world_size = parallel_spec.world_size
    tail = list(range(3, x_row.ndim + 1))
    send_view = x_row.reshape(
        shard,
        world_size,
        shard,
        *x_row.shape[2:],
    )[:, :, local_col_start:local_col_end, ...]
    return send_view.permute(1, 2, 0, *tail).contiguous()


def _materialize_parallel_transpose_col_local_chunk(
    recv_buffer: torch.Tensor,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """Restore list-A2A ``cat(dim=1)`` layout from a direct receive buffer."""
    tail = list(range(3, recv_buffer.ndim))
    ordered = recv_buffer.permute(1, 0, 2, *tail).contiguous()
    return ordered.reshape(
        recv_buffer.shape[1],
        parallel_spec.n_padded,
        *recv_buffer.shape[3:],
    )


def _launch_parallel_transpose_col_local_chunk_async(
    x_row: torch.Tensor,
    parallel_spec: ParallelSpec,
    local_col_start: int,
    local_col_end: int,
) -> _PendingGridTransposeColChunk:
    """Launch one row-to-column AllToAll chunk without waiting.

    The caller overlaps this input transfer with attention on the preceding
    chunk.
    """
    if dist is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed GridSelfAttention overlap requires torch.distributed.")

    x_row = pad_to_length(x_row, dim=1, length=parallel_spec.n_padded, value=0)
    _zero_parallel_pair_padding(x_row, parallel_spec)

    send_buffer = _pack_parallel_transpose_col_local_chunk(
        x_row,
        parallel_spec,
        local_col_start,
        local_col_end,
    )
    recv_buffer = torch.empty_like(send_buffer)
    work = dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        group=parallel_spec.group,
        async_op=True,
    )
    if work is None:
        raise RuntimeError("Grid Attention async AllToAll returned no Work handle.")

    return _PendingGridTransposeColChunk(
        local_col_start=local_col_start,
        local_col_end=local_col_end,
        send_buffer=send_buffer,
        recv_buffer=recv_buffer,
        work=work,
    )


def _wait_parallel_transpose_col_local_chunk(
    pending: _PendingGridTransposeColChunk,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """Wait for an input chunk, restore global order, and release buffers."""
    pending.work.wait()
    if pending.recv_buffer is None:
        raise RuntimeError("Grid input A2A pending state has no receive buffer.")
    out = _materialize_parallel_transpose_col_local_chunk(
        pending.recv_buffer,
        parallel_spec,
    )
    pending.send_buffer = None
    pending.recv_buffer = None
    if parallel_spec.n_global < out.shape[1]:
        out[:, parallel_spec.n_global:] = 0
    return out


def _grid_attention_a2a_overlap_eligible(
    mask_local: Optional[torch.Tensor],
    fa_mask_contract: Optional[str],
    parallel_spec: ParallelSpec,
) -> bool:
    """Return whether one-chunk input overlap preserves collective order.

    Overlap needs at least two chunks and either no mask or a symmetric
    pair-padding mask that can be transposed locally.
    """
    batch_chunk_size = _resolve_grid_attention_batch_chunk_size(
        parallel_spec,
    )
    if parallel_spec.shard_size <= batch_chunk_size:
        return False
    if mask_local is None:
        return True
    return fa_mask_contract == "pair_padding"


def _parallel_write_col_chunk_to_row(
    x_col_chunk: torch.Tensor,
    out_row: torch.Tensor,
    parallel_spec: ParallelSpec,
    local_col_start: int,
    local_col_end: int,
    *,
    add: bool = False,
) -> None:
    """Scatter a column-layout chunk back into a row-layout output tensor."""
    if dist is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed GridSelfAttention chunking requires torch.distributed.")

    shard = parallel_spec.shard_size
    tail = list(range(2, x_col_chunk.ndim))
    perm_back = [1, 0] + tail

    # A contiguous output permits one direct receive-buffer view update.
    if out_row.is_contiguous():
        chunk_size = local_col_end - local_col_start
        tail_dims = list(range(3, x_col_chunk.ndim + 1))
        send_buffer = x_col_chunk.reshape(
            chunk_size,
            parallel_spec.world_size,
            shard,
            *x_col_chunk.shape[2:],
        ).permute(1, 0, 2, *tail_dims).contiguous()
        recv_buffer = torch.empty_like(send_buffer)
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            group=parallel_spec.group,
        )

        recv_view = recv_buffer.permute(2, 0, 1, *tail_dims)
        out_view = out_row.reshape(
            shard,
            parallel_spec.world_size,
            shard,
            *out_row.shape[2:],
        )[:, :, local_col_start:local_col_end, ...]
        if add:
            out_view.add_(recv_view)
        else:
            out_view.copy_(recv_view)
        return

    # Non-contiguous outputs use source-ordered receive chunks.
    send_chunks = []
    for dst in range(parallel_spec.world_size):
        row0 = dst * shard
        row1 = row0 + shard
        send_chunks.append(x_col_chunk[:, row0:row1, ...].contiguous())

    recv_chunks = [
        torch.empty_like(send_chunks[0])
        for _ in range(parallel_spec.world_size)
    ]
    dist.all_to_all(recv_chunks, send_chunks, group=parallel_spec.group)

    for src, chunk in enumerate(recv_chunks):
        col0 = src * shard + local_col_start
        col1 = src * shard + local_col_end
        row_chunk = chunk.permute(*perm_back).contiguous()
        if add:
            out_row[:, col0:col1, ...].add_(row_chunk)
        else:
            out_row[:, col0:col1, ...].copy_(row_chunk)


def _build_grid_fa_pair_bias_cache(
    pair_bias_fp32: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build one BF16 pair bias reused by all Fusion Attention chunks."""
    if (
        fastnn_attention_impl.fastnn_config.dot_product_attention_implementation
        != "Fusion_Attention"
    ):
        return None
    return pair_bias_fp32.to(torch.bfloat16)


def _all_gather_grid_pair_bias(
    pair_bias_local: torch.Tensor,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """Gather Grid bias in BF16 while preserving rank order and padding."""
    if parallel_spec.is_distributed:
        pair_bias_local = pair_bias_local.to(torch.bfloat16)
    pair_bias_local = pair_bias_local.contiguous()
    return all_gather_first_axis(pair_bias_local, parallel_spec)


class GridSelfAttention(nn.Module):
    def __init__(
        self,
        c_pair: int = 128,
        num_head: int = 4,
        transpose: bool = False,
    ):
        super(GridSelfAttention, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)
        self.qkv_projection = nn.Linear(self.c_pair, self.c_pair * 3, bias=False)
        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)

    def _resolve_full_attention_impl(
        self,
        batch_size: int,
        tgt_len: int,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        num_head: int,
        fa_mask_plan: Optional[fastnn_attention_impl.FAMaskPlan] = None,
    ) -> str:
        impl = fastnn_attention_impl.fastnn_config.dot_product_attention_implementation
        if impl == "torch":
            return "torch"
        if impl != "Fusion_Attention":
            raise ValueError(
                f"Unknown dot_product_attention_implementation: {impl}"
            )

        if fa_mask_plan is not None and (mask is None or mask.dim() != 2):
            return "torch"

        if mask is not None:
            if mask.dim() == 1:
                if mask.shape[0] != tgt_len:
                    return "torch"
            elif mask.dim() == 2:
                if tuple(mask.shape) != (batch_size, tgt_len):
                    return "torch"
                if fa_mask_plan is None:
                    mask_bool = mask.to(dtype=torch.bool)
                    if (not mask_bool.all().item()) and batch_size > 1:
                        if not torch.equal(mask_bool, mask_bool[:1].expand_as(mask_bool)):
                            return "torch"
                else:
                    # The pair-padding contract guarantees uniform valid rows.
                    if (
                        batch_size != fa_mask_plan.expected_batch
                        or tgt_len != fa_mask_plan.expected_key_len
                        or not 0 <= fa_mask_plan.key_valid_len <= tgt_len
                        or not fa_mask_plan.uniform_valid_rows
                        or not fa_mask_plan.all_query_rows_valid
                    ):
                        return "torch"
            else:
                return "torch"

        if bias is not None:
            if bias.dim() != 3 or tuple(bias.shape) != (num_head, tgt_len, tgt_len):
                return "torch"

        return "fa"

    def _apply_attention_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_head: int,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        impl: str,
        fa_mask_plan: Optional[fastnn_attention_impl.FAMaskPlan] = None,
    ) -> torch.Tensor:
        if impl == "torch":
            return fastnn_attention_impl.dot_product_attention_torch(
                q, k, v, num_head, mask=mask, bias=bias
            )
        if impl == "fa":
            return fastnn_attention_impl.dot_product_attention_FA(
                q,
                k,
                v,
                num_head,
                mask=mask,
                bias=bias,
                mask_plan=fa_mask_plan,
            )
        raise ValueError(f"Unknown forced attention impl: {impl}")

    def _project_fusion_attention_inputs(
        self,
        raw_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project pre-scaled BF16 Q/K/V in sequence-major layout."""
        if raw_chunk.dim() != 3 or raw_chunk.shape[-1] != self.c_pair:
            raise ValueError(
                "Triangle Attention input must have shape [B,S,c_pair]; "
                f"got input={tuple(raw_chunk.shape)}, c_pair={self.c_pair}"
            )

        weight = self.qkv_projection.weight
        expected_weight_shape = (3 * self.c_pair, self.c_pair)
        if tuple(weight.shape) != expected_weight_shape:
            raise RuntimeError(
                "qkv_projection weight has an incompatible shape: "
                f"expected={expected_weight_shape}, got={tuple(weight.shape)}"
            )
        q_weight, k_weight, v_weight = weight.split(self.c_pair, dim=0)

        bias = self.qkv_projection.bias
        if bias is None:
            q_bias = k_bias = v_bias = None
        else:
            expected_bias_shape = (3 * self.c_pair,)
            if tuple(bias.shape) != expected_bias_shape:
                raise RuntimeError(
                    "qkv_projection bias has an incompatible shape: "
                    f"expected={expected_bias_shape}, got={tuple(bias.shape)}"
                )
            q_bias, k_bias, v_bias = bias.split(self.c_pair, dim=0)

        # Project Q/K/V sequentially to limit FP32 temporaries; scale Q before casting.
        q = F.linear(raw_chunk, q_weight, q_bias)
        q.mul_(self.qkv_dim ** -0.5)
        q = q.to(dtype=torch.bfloat16).contiguous()

        k = F.linear(raw_chunk, k_weight, k_bias)
        k = k.to(dtype=torch.bfloat16).contiguous()

        v = F.linear(raw_chunk, v_weight, v_bias)
        v = v.to(dtype=torch.bfloat16).contiguous()
        return q, k, v

    def _grid_attention_project_chunk(
        self,
        raw_chunk: torch.Tensor,
        mask_chunk: Optional[torch.Tensor],
        pair_bias: torch.Tensor,
        tgt_len: int,
        fa_mask_plan: Optional[fastnn_attention_impl.FAMaskPlan] = None,
        pair_bias_fa_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask_bool = None if mask_chunk is None else mask_chunk.to(dtype=torch.bool)
        attention_impl = self._resolve_full_attention_impl(
            batch_size=raw_chunk.shape[0],
            tgt_len=tgt_len,
            mask=mask_bool,
            bias=pair_bias,
            num_head=self.num_head,
            fa_mask_plan=fa_mask_plan,
        )
        attention_bias = (
            pair_bias_fa_cache
            if attention_impl == "fa" and pair_bias_fa_cache is not None
            else pair_bias
        )
        if attention_impl == "fa":
            q, k, v = self._project_fusion_attention_inputs(raw_chunk)
            weighted_avg = (
                fastnn_attention_impl.dot_product_attention_FA_sequence_major(
                    q,
                    k,
                    v,
                    self.num_head,
                    mask=mask_bool,
                    bias=attention_bias,
                    mask_plan=fa_mask_plan,
                )
            )
            del q, k, v
        elif attention_impl == "torch":
            # The torch path supports masks outside the shared FA mask contract.
            qkv = self.qkv_projection(raw_chunk)
            qkv = (
                qkv.view(raw_chunk.shape[0], tgt_len, 3, -1)
                .permute(2, 0, 1, 3)
                .contiguous()
            )
            q, k, v = qkv.unbind(0)
            q, k, v = map(
                lambda t: einops.rearrange(
                    t,
                    "b n (h d) -> b h n d",
                    h=self.num_head,
                ),
                [q, k, v],
            )
            weighted_avg = fastnn_attention_impl.dot_product_attention_torch(
                q,
                k,
                v,
                self.num_head,
                mask=mask_bool,
                bias=attention_bias,
            )
            weighted_avg = einops.rearrange(
                weighted_avg,
                "b h n d -> b n (h d)",
            )
        else:
            raise RuntimeError(
                f"Unexpected Triangle Attention implementation: {attention_impl}"
            )
        gate_values = self.gating_query(raw_chunk)
        weighted_avg.mul_(torch.sigmoid(gate_values))
        return self.output_projection(weighted_avg)

    def _grid_attention_project_valid_rows(
        self,
        raw_chunk: torch.Tensor,
        mask_chunk: Optional[torch.Tensor],
        pair_bias: torch.Tensor,
        tgt_len: int,
        valid_len: int,
        *,
        fa_mask_contract: Optional[str] = None,
        key_valid_len: Optional[int] = None,
        pair_bias_fa_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_chunk = raw_chunk.new_zeros(raw_chunk.shape)
        if valid_len <= 0:
            return out_chunk

        raw_valid = raw_chunk[:valid_len].contiguous()
        mask_valid = None if mask_chunk is None else mask_chunk[:valid_len].contiguous()

        fa_mask_plan = None
        if fa_mask_contract is not None:
            if fa_mask_contract != "pair_padding":
                raise ValueError(
                    f"Unsupported FA mask contract: {fa_mask_contract}"
                )
            if mask_valid is None:
                raise ValueError("pair_padding FA mask contract requires mask_local")
            if key_valid_len is None:
                raise ValueError("pair_padding FA mask contract requires key_valid_len")

            # Valid query rows share one key mask under pair-padding.
            fa_mask_plan = fastnn_attention_impl.FAMaskPlan(
                expected_batch=valid_len,
                expected_key_len=tgt_len,
                key_valid_len=key_valid_len,
                uniform_valid_rows=True,
                all_query_rows_valid=True,
            )

        out_valid = self._grid_attention_project_chunk(
            raw_valid,
            mask_valid,
            pair_bias,
            tgt_len,
            fa_mask_plan=fa_mask_plan,
            pair_bias_fa_cache=pair_bias_fa_cache,
        )
        out_chunk[:valid_len].copy_(out_valid)
        return out_chunk

    def _forward_parallel_batch_chunked(
        self,
        pair_local: torch.Tensor,
        mask_local: Optional[torch.Tensor],
        parallel_spec: ParallelSpec,
        fa_mask_contract: Optional[str] = None,
    ) -> torch.Tensor:
        pair_input = self.act_norm(pair_local)
        pair_compute = pair_input
        pair_bias_local = self.pair_bias_projection(pair_compute)
        pair_bias = _all_gather_grid_pair_bias(
            pair_bias_local, parallel_spec
        )
        pair_bias = pair_bias.permute(2, 0, 1).contiguous()
        # Build the BF16 bias once outside the batch-chunk loop.
        pair_bias_fa_cache = _build_grid_fa_pair_bias_cache(pair_bias)

        out = torch.zeros_like(pair_input)
        local_valid = _parallel_local_valid_size(parallel_spec, pair_input.shape[0])
        batch_chunk_size = _resolve_grid_attention_batch_chunk_size(
            parallel_spec,
        )
        for b0 in range(0, pair_input.shape[0], batch_chunk_size):
            b1 = min(b0 + batch_chunk_size, pair_input.shape[0])
            valid_len = max(0, min(b1, local_valid) - b0)
            raw_chunk = pair_compute[b0:b1].contiguous()
            mask_chunk = None if mask_local is None else mask_local[b0:b1].contiguous()

            out_chunk = self._grid_attention_project_valid_rows(
                raw_chunk,
                mask_chunk,
                pair_bias,
                pair_input.shape[1],
                valid_len,
                fa_mask_contract=fa_mask_contract,
                key_valid_len=parallel_spec.n_global,
                pair_bias_fa_cache=pair_bias_fa_cache,
            )
            out[b0:b1].copy_(out_chunk)

        _zero_parallel_pair_padding(out, parallel_spec)
        return out

    def _forward_parallel_transpose_chunked(
        self,
        pair_local: torch.Tensor,
        mask_local: Optional[torch.Tensor],
        parallel_spec: ParallelSpec,
        *,
        add_to: Optional[torch.Tensor] = None,
        fa_mask_contract: Optional[str] = None,
    ) -> torch.Tensor:
        pair_input = self.act_norm(pair_local)
        pair_compute = pair_input
        pair_bias_local = self.pair_bias_projection(pair_compute)
        out_row = torch.zeros_like(pair_input) if add_to is None else add_to
        local_batch = parallel_spec.shard_size
        local_valid = _parallel_local_valid_size(parallel_spec, local_batch)
        local_mask_col = None
        if (
            mask_local is not None
            and fa_mask_contract == "pair_padding"
        ):
            # Transpose symmetric pair-padding masks locally; communicate other masks.
            local_mask_col = pair_padding_mask_col_from_row(mask_local, parallel_spec)

        pair_bias = _all_gather_grid_pair_bias(
            pair_bias_local, parallel_spec
        )
        pair_bias = pair_bias.permute(2, 0, 1).contiguous()
        pair_bias_fa_cache = _build_grid_fa_pair_bias_cache(pair_bias)

        batch_chunk_size = _resolve_grid_attention_batch_chunk_size(
            parallel_spec,
        )
        for b0 in range(0, local_batch, batch_chunk_size):
            b1 = min(b0 + batch_chunk_size, local_batch)
            valid_b1 = min(b1, local_valid)
            valid_len = max(0, valid_b1 - b0)

            raw_chunk = _parallel_transpose_col_local_chunk(
                pair_compute, parallel_spec, b0, b1
            )
            mask_chunk = None
            if local_mask_col is not None:
                mask_chunk = local_mask_col[b0:b1].contiguous()
            elif mask_local is not None:
                mask_chunk = _parallel_transpose_col_local_chunk(
                    mask_local, parallel_spec, b0, b1
                )
            out_chunk = self._grid_attention_project_valid_rows(
                raw_chunk,
                mask_chunk,
                pair_bias,
                pair_input.shape[1],
                valid_len,
                fa_mask_contract=fa_mask_contract,
                key_valid_len=parallel_spec.n_global,
                pair_bias_fa_cache=pair_bias_fa_cache,
            )
            _parallel_write_col_chunk_to_row(
                out_chunk,
                out_row,
                parallel_spec,
                b0,
                b1,
                add=add_to is not None,
            )

        _zero_parallel_pair_padding(out_row, parallel_spec)
        return out_row

    def _forward_parallel_transpose_overlap_chunked(
        self,
        pair_local: torch.Tensor,
        mask_local: Optional[torch.Tensor],
        parallel_spec: ParallelSpec,
        *,
        add_to: Optional[torch.Tensor] = None,
        fa_mask_contract: Optional[str] = None,
    ) -> torch.Tensor:
        """Overlap input AllToAll for chunk ``i+1`` with attention on chunk ``i``.

        The first input remains pipeline warm-up. Output AllToAll stays
        synchronous so only one pending input buffer is live and collective
        ordering stays explicit.
        """
        pair_input = self.act_norm(pair_local)
        pair_compute = pair_input
        pair_bias_local = self.pair_bias_projection(pair_compute) # [L_local, L_full, H]
        out_row = torch.zeros_like(pair_input) if add_to is None else add_to
        local_batch = parallel_spec.shard_size
        local_valid = _parallel_local_valid_size(parallel_spec, local_batch)
        local_mask_col = None
        if mask_local is not None:
            local_mask_col = pair_padding_mask_col_from_row(mask_local, parallel_spec)

        batch_chunk_size = _resolve_grid_attention_batch_chunk_size(
            parallel_spec,
        )
        chunk_ranges = [
            (b0, min(b0 + batch_chunk_size, local_batch))
            for b0 in range(0, local_batch, batch_chunk_size)
        ]
        if not chunk_ranges:
            return out_row

        def _launch_input_chunk(
            chunk_index: int,
        ) -> _PendingGridTransposeColChunk:
            chunk_b0, chunk_b1 = chunk_ranges[chunk_index]
            return _launch_parallel_transpose_col_local_chunk_async(
                pair_compute,
                parallel_spec,
                chunk_b0,
                chunk_b1,
            )

        pair_bias = _all_gather_grid_pair_bias(
            pair_bias_local, parallel_spec
        )
        pending = _launch_input_chunk(0)
        pair_bias = pair_bias.permute(2, 0, 1).contiguous()
        pair_bias_fa_cache = _build_grid_fa_pair_bias_cache(pair_bias)

        for index, (b0, b1) in enumerate(chunk_ranges):
            # All ranks must use identical ranges and collective order, even
            # when a rank has no valid query rows in the current chunk.
            if pending is None or (
                pending.local_col_start != b0
                or pending.local_col_end != b1
            ):
                raise RuntimeError(
                    "Grid Attention overlap chunk schedule became inconsistent."
                )

            raw_chunk = _wait_parallel_transpose_col_local_chunk(
                pending,
                parallel_spec,
            )

            next_pending = None
            if index + 1 < len(chunk_ranges):
                # Prefetch the next input to overlap transfer and computation.
                next_pending = _launch_input_chunk(index + 1)

            mask_chunk = (
                None
                if local_mask_col is None
                else local_mask_col[b0:b1].contiguous()
            )
            valid_b1 = min(b1, local_valid)
            valid_len = max(0, valid_b1 - b0)
            out_chunk = self._grid_attention_project_valid_rows(
                raw_chunk,
                mask_chunk,
                pair_bias,
                pair_input.shape[1],
                valid_len,
                fa_mask_contract=fa_mask_contract,
                key_valid_len=parallel_spec.n_global,
                pair_bias_fa_cache=pair_bias_fa_cache,
            )

            _parallel_write_col_chunk_to_row(
                out_chunk,
                out_row,
                parallel_spec,
                b0,
                b1,
                add=add_to is not None,
            )

            if next_pending is not None:
                pending = next_pending

        _zero_parallel_pair_padding(out_row, parallel_spec)
        return out_row

    def _attention(self, pair: torch.Tensor, mask: Optional[torch.Tensor], bias: torch.Tensor):
        bsz, tgt_len, _ = pair.size()
        qkv = self.qkv_projection(pair)

        qkv = qkv.view(bsz, tgt_len, 3, -1).permute(2, 0, 1, 3).contiguous()
        q, k, v = qkv.unbind(0)
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.num_head),
            [q, k, v]
        )

        mask_flag = None if mask is None else mask.to(dtype=torch.bool)
        weighted_avg = fastnn.dot_product_attention(
            q, k, v, self.num_head,
            mask=mask_flag,
            bias=bias
        )
        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        gate_values = self.gating_query(pair)
        weighted_avg *= torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    def forward(
        self,
        pair_local, # [L_local, L_full, C]
        mask_local=None, # [L_local, L_full]
        parallel_spec: Optional[ParallelSpec] = None,
        fa_mask_contract: Optional[str] = None,
        **kwargs,
    ):
        """
        pair_local: [B, S, C]
            - full mode: [L, L, C]
            - Parallel mode: [L_local, L_full, C]

        mask_local:
            - full mode: [L, L]
            - Parallel mode: [L_local, L_full]
        """
        if "mask" in kwargs:
            if mask_local is not None:
                raise TypeError("GridSelfAttention.forward got both mask and mask_local")
            mask_local = kwargs.pop("mask")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"GridSelfAttention.forward got unexpected keyword argument(s): {unexpected}")

        if (
            pair_local is not None
            and pair_local.ndim == 3
            and parallel_spec is not None
            and parallel_spec.world_size > 1
        ):
            if self.transpose:
                # Use overlap only when eligibility preserves collective order.
                if _grid_attention_a2a_overlap_eligible(
                    mask_local,
                    fa_mask_contract,
                    parallel_spec,
                ):
                    return self._forward_parallel_transpose_overlap_chunked(
                        pair_local,
                        mask_local,
                        parallel_spec,
                        fa_mask_contract=fa_mask_contract,
                    )
                return self._forward_parallel_transpose_chunked(
                    pair_local,
                    mask_local,
                    parallel_spec,
                    fa_mask_contract=fa_mask_contract,
                )
            return self._forward_parallel_batch_chunked(
                pair_local,
                mask_local,
                parallel_spec,
                fa_mask_contract=fa_mask_contract,
            )

        pair_local = self.act_norm(pair_local)
        pair_compute = pair_local
        pair_bias_local = self.pair_bias_projection(pair_compute)
        pair_bias = pair_bias_local.permute(2, 0, 1).contiguous()

        pair_work = pair_compute
        mask_work = mask_local
        if self.transpose:
            pair_work = pair_work.permute(1, 0, 2).contiguous()
            # The base transpose=True path does not additionally transpose the mask.

        mask_bool = None if mask_work is None else mask_work.to(dtype=torch.bool)
        out = self._attention(pair_work, mask_bool, pair_bias)

        if self.transpose:
            out = out.permute(1, 0, 2).contiguous()

        return out

    def forward_add_(
        self,
        pair_local,
        mask_local=None,
        parallel_spec: Optional[ParallelSpec] = None,
        fa_mask_contract: Optional[str] = None,
        **kwargs,
    ):
        if "mask" in kwargs:
            if mask_local is not None:
                raise TypeError("GridSelfAttention.forward_add_ got both mask and mask_local")
            mask_local = kwargs.pop("mask")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"GridSelfAttention.forward_add_ got unexpected keyword argument(s): {unexpected}")

        if (
            pair_local is not None
            and pair_local.ndim == 3
            and parallel_spec is not None
            and parallel_spec.world_size > 1
            and self.transpose
            and pair_local.shape[0]
            > _resolve_grid_attention_batch_chunk_size(
                parallel_spec,
            )
        ):
            if _grid_attention_a2a_overlap_eligible(
                mask_local,
                fa_mask_contract,
                parallel_spec,
            ):
                return self._forward_parallel_transpose_overlap_chunked(
                    pair_local,
                    mask_local,
                    parallel_spec,
                    add_to=pair_local,
                    fa_mask_contract=fa_mask_contract,
                )
            return self._forward_parallel_transpose_chunked(
                pair_local,
                mask_local,
                parallel_spec,
                add_to=pair_local,
                fa_mask_contract=fa_mask_contract,
            )

        pair_local.add_(self.forward(
            pair_local,
            mask_local=mask_local,
            parallel_spec=parallel_spec,
            fa_mask_contract=fa_mask_contract,
        ))
        return pair_local


class MSAAttention(nn.Module):
    def __init__(self, c_msa=64, c_pair=128, num_head=8):
        super(MSAAttention, self).__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = fastnn.LayerNorm(self.c_msa)
        self.pair_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_logits = nn.Linear(self.c_pair, self.num_head, bias=False)
        self.v_projection = nn.Linear(
            self.c_msa, self.num_head * self.value_dim, bias=False)
        self.gating_query = nn.Linear(self.c_msa, self.c_msa, bias=False)
        self.output_projection = nn.Linear(self.c_msa, self.c_msa, bias=False)

    def forward(
        self,
        msa: torch.Tensor,
        msa_mask: torch.Tensor,
        pair: torch.Tensor,
        parallel_spec: Optional[ParallelSpec] = None,
    ):
        """
        msa:
            [N_msa, Lp, c_msa]   (full replicated)

        msa_mask:
            [N_msa, Lp]          (full replicated)

        pair:
            - full mode: [Lp, Lp, c_pair]
            - Parallel mode: [Ls, Lp, c_pair]
        """
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)

        logits = self.pair_logits(pair)  # full: [Lp, Lp, H], parallel: [Ls, Lp, H]

        if parallel_spec is not None and parallel_spec.world_size > 1:
            logits = all_gather_first_axis(logits, parallel_spec)  # [Lp, Lp, H]

        logits = logits.permute(2, 0, 1).contiguous()  # [H, Lp, Lp]

        seq_mask = torch.max(msa_mask, dim=0).values.to(dtype=logits.dtype)
        logits += 1e9 * (seq_mask - 1.0)

        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head)

        v_avg = torch.einsum('hqk,bkhc->bqhc', weights, v)
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        gate_values = self.gating_query(msa)
        v_avg *= torch.sigmoid(gate_values)

        return self.output_projection(v_avg)
