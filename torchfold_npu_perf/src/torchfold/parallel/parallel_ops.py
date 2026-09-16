from torchfold import padding
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None


@dataclass
class ParallelSpec:
    group: Optional[object]
    rank: int
    world_size: int
    n_global: int
    n_padded: int
    shard_size: int
    start: int
    end: int

    @property
    def local_valid_size(self) -> int:
        return max(0, min(self.end, self.n_global) - self.start)

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


@dataclass
class PendingAllGatherFirstAxis:
    """Buffers and Work retained until an async first-axis AllGather finishes."""

    input_tensor: Optional[torch.Tensor]
    recv_chunks: list[torch.Tensor]
    work: Optional[object]


@dataclass(frozen=True)
class ParallelGroups:
    parallel_group: Optional[object]


def _parallel_padding_multiple(world_size: int) -> int:
    """Return the global alignment multiple from the shared padding rules."""
    return padding.parallel_padding_multiple(world_size)


def get_parallel_groups() -> ParallelGroups:
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return ParallelGroups(parallel_group=None)
    return ParallelGroups(parallel_group=dist.group.WORLD)


def build_parallel_spec(length: int, group=None) -> ParallelSpec:
    if dist is None or not dist.is_available() or not dist.is_initialized():
        world_size = 1
        rank = 0
        group = None
    else:
        if group is None:
            group = dist.group.WORLD
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

    n_padded = padding.parallel_padded_length(length, world_size)
    shard_size = n_padded // world_size
    start = rank * shard_size
    end = start + shard_size

    return ParallelSpec(
        group=group,
        rank=rank,
        world_size=world_size,
        n_global=length,
        n_padded=n_padded,
        shard_size=shard_size,
        start=start,
        end=end,
    )


def _zero_tail(x: torch.Tensor, dim: int, cutoff: int, value=0) -> torch.Tensor:
    if cutoff >= x.shape[dim]:
        return x

    index = [slice(None)] * x.ndim
    index[dim] = slice(max(cutoff, 0), None)
    x[tuple(index)] = value
    return x


def zero_local_padding(
    x: torch.Tensor,
    parallel_spec: ParallelSpec,
    dim: int = 0,
    value=0,
) -> torch.Tensor:
    return _zero_tail(x, dim=dim, cutoff=parallel_spec.local_valid_size, value=value)


def zero_global_padding(
    x: torch.Tensor,
    parallel_spec: ParallelSpec,
    dim: int = 0,
    value=0,
) -> torch.Tensor:
    return _zero_tail(x, dim=dim, cutoff=parallel_spec.n_global, value=value)


def pad_to_length(
    x: torch.Tensor,
    dim: int,
    length: int,
    value=0,
) -> torch.Tensor:
    cur = x.shape[dim]
    if cur >= length:
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = length - cur
    pad = x.new_full(pad_shape, value)
    return torch.cat([x, pad], dim=dim)


def slice_1d_local(x: torch.Tensor, parallel_spec: ParallelSpec, pad_value=0) -> torch.Tensor:
    x = pad_to_length(x, dim=0, length=parallel_spec.n_padded, value=pad_value)
    local = x[parallel_spec.start:parallel_spec.end]
    zero_local_padding(local, parallel_spec, dim=0, value=pad_value)
    return local


def make_pair_mask_row(seq_mask: torch.Tensor, parallel_spec: ParallelSpec) -> torch.Tensor:
    seq_mask = pad_to_length(seq_mask, dim=0, length=parallel_spec.n_padded, value=0)
    local = seq_mask[parallel_spec.start:parallel_spec.end]
    zero_local_padding(local, parallel_spec, dim=0, value=0)
    return local.unsqueeze(1) * seq_mask.unsqueeze(0)


def pair_padding_mask_col_from_row(
    pair_mask_row: torch.Tensor,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """Reuse a row-sharded pair-padding mask as its column-sharded transpose.

    ``make_pair_mask_row`` constructs the symmetric mask ``M[i, j] = s[i] * s[j]``.
    Row and column layouts therefore share the same local values. Validate the
    shard shape before skipping communication; arbitrary masks are not supported.
    """
    expected_shape = (parallel_spec.shard_size, parallel_spec.n_padded)
    if pair_mask_row.ndim != 2 or tuple(pair_mask_row.shape) != expected_shape:
        raise ValueError(
            "pair-padding mask local transpose requires shape "
            f"{expected_shape}, got {tuple(pair_mask_row.shape)}"
        )

    # Re-apply host-known tail bounds without reading a device scalar.
    zero_local_padding(pair_mask_row, parallel_spec, dim=0, value=0)
    zero_global_padding(pair_mask_row, parallel_spec, dim=1, value=0)
    return pair_mask_row.contiguous()


def all_gather_first_axis(x_local: torch.Tensor, parallel_spec: ParallelSpec) -> torch.Tensor:
    if not parallel_spec.is_distributed:
        return x_local

    zero_local_padding(x_local, parallel_spec, dim=0, value=0)
    chunks = [torch.empty_like(x_local) for _ in range(parallel_spec.world_size)]
    dist.all_gather(chunks, x_local, group=parallel_spec.group)
    full = torch.cat(chunks, dim=0)
    zero_global_padding(full, parallel_spec, dim=0, value=0)
    return full


def _group_rank_to_global(group, group_rank: int) -> int:
    if group is None or dist is None:
        return group_rank

    get_global_rank = getattr(dist, "get_global_rank", None)
    if get_global_rank is None:
        return group_rank

    try:
        return get_global_rank(group, group_rank)
    except (RuntimeError, ValueError):
        return group_rank


def gather_first_axis_to_rank0(
    x_local: torch.Tensor,
    parallel_spec: ParallelSpec,
) -> Optional[torch.Tensor]:
    """Gather a row-sharded square tensor only onto parallel rank zero."""
    if x_local.ndim < 2:
        raise ValueError(
            "Rank-zero square gather requires at least two dimensions, "
            f"got shape={tuple(x_local.shape)}"
        )
    if x_local.shape[0] < parallel_spec.shard_size:
        raise ValueError(
            "Rank-zero square gather received too few local rows: "
            f"shape={tuple(x_local.shape)} "
            f"shard_size={parallel_spec.shard_size}"
        )
    if x_local.shape[1] < parallel_spec.n_global:
        raise ValueError(
            "Rank-zero square gather received too few global columns: "
            f"shape={tuple(x_local.shape)} n_global={parallel_spec.n_global}"
        )

    n_global = parallel_spec.n_global
    local_valid = parallel_spec.local_valid_size
    local_unpadded = x_local[:local_valid, :n_global].contiguous()

    if not parallel_spec.is_distributed:
        return local_unpadded

    if dist is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Rank-zero square gather requires initialized torch.distributed."
        )

    root_global_rank = _group_rank_to_global(parallel_spec.group, 0)
    if parallel_spec.rank != 0:
        if local_valid > 0:
            dist.send(
                local_unpadded,
                dst=root_global_rank,
                group=parallel_spec.group,
            )
        return None

    full = x_local.new_empty(
        (n_global, n_global, *x_local.shape[2:])
    )
    if local_valid > 0:
        full[:local_valid].copy_(local_unpadded)

    for group_rank in range(1, parallel_spec.world_size):
        start = group_rank * parallel_spec.shard_size
        end = min(start + parallel_spec.shard_size, n_global)
        if start >= end:
            continue
        source_global_rank = _group_rank_to_global(
            parallel_spec.group,
            group_rank,
        )
        dist.recv(
            full[start:end],
            src=source_global_rank,
            group=parallel_spec.group,
        )
    return full


def launch_all_gather_first_axis_async(
    x_local: torch.Tensor,
    parallel_spec: ParallelSpec,
) -> PendingAllGatherFirstAxis:
    """Launch the distributed form of ``all_gather_first_axis`` asynchronously.

    The caller must keep the returned object alive and consume it exactly once
    with ``wait_all_gather_first_axis`` before reusing the input or outputs.
    Padding, rank order, dtype, and output layout match the synchronous helper.
    """
    if (
        dist is None
        or not dist.is_available()
        or not dist.is_initialized()
        or not parallel_spec.is_distributed
    ):
        raise RuntimeError(
            "Async first-axis AllGather requires initialized distributed mode."
        )

    zero_local_padding(x_local, parallel_spec, dim=0, value=0)
    recv_chunks = [
        torch.empty_like(x_local)
        for _ in range(parallel_spec.world_size)
    ]
    work = dist.all_gather(
        recv_chunks,
        x_local,
        group=parallel_spec.group,
        async_op=True,
    )
    if work is None:
        raise RuntimeError("Async first-axis AllGather returned no Work handle.")
    return PendingAllGatherFirstAxis(
        input_tensor=x_local,
        recv_chunks=recv_chunks,
        work=work,
    )


def wait_all_gather_first_axis(
    pending: PendingAllGatherFirstAxis,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    """Wait for an async first-axis AllGather and materialize its full tensor."""
    if pending.work is None:
        raise RuntimeError("Async first-axis AllGather has already been consumed.")
    pending.work.wait()
    full = torch.cat(pending.recv_chunks, dim=0)
    pending.input_tensor = None
    pending.recv_chunks.clear()
    pending.work = None
    zero_global_padding(full, parallel_spec, dim=0, value=0)
    return full


def _all_to_all_equal_chunks(x: torch.Tensor, parallel_spec: ParallelSpec) -> torch.Tensor:
    if not parallel_spec.is_distributed:
        return x

    # ``x`` is laid out as [destination_rank, ...].  With equal splits,
    # all_to_all_single writes source-rank i directly into output[i].
    if not x.is_contiguous():
        raise ValueError(
            "Pair A2A direct receive requires a contiguous send Tensor."
        )
    output = torch.empty_like(x)
    dist.all_to_all_single(
        output,
        x,
        group=parallel_spec.group,
    )
    return output


def transpose_col(x_row: torch.Tensor, parallel_spec: ParallelSpec) -> torch.Tensor:
    zero_local_padding(x_row, parallel_spec, dim=0, value=0)
    if not parallel_spec.is_distributed:
        out = x_row.transpose(0, 1).contiguous()
        zero_global_padding(out, parallel_spec, dim=0, value=0)
        return out

    x_row = pad_to_length(x_row, dim=1, length=parallel_spec.n_padded, value=0)
    zero_global_padding(x_row, parallel_spec, dim=1, value=0)
    ls = parallel_spec.shard_size
    ws = parallel_spec.world_size
    tail = x_row.shape[2:]

    send = x_row.reshape(ls, ws, ls, *tail)
    perm = [1, 0, 2] + list(range(3, send.ndim))
    send = send.permute(*perm).contiguous()
    recv = _all_to_all_equal_chunks(send, parallel_spec)

    perm_back = [2, 0, 1] + list(range(3, recv.ndim))
    out = recv.permute(*perm_back).contiguous()
    out = out.reshape(ls, parallel_spec.n_padded, *tail)
    zero_local_padding(out, parallel_spec, dim=0, value=0)
    zero_global_padding(out, parallel_spec, dim=1, value=0)
    return out


def transpose_col_local_chunk(
    x_row: torch.Tensor,
    parallel_spec: ParallelSpec,
    local_row_start: int,
    local_row_end: int,
) -> torch.Tensor:
    """Transpose one local output-row window without materializing all rows."""
    if not 0 <= local_row_start < local_row_end <= parallel_spec.shard_size:
        raise ValueError(
            "local transpose chunk must satisfy "
            "0 <= start < end <= shard_size, got "
            f"start={local_row_start}, end={local_row_end}, "
            f"shard_size={parallel_spec.shard_size}"
        )

    zero_local_padding(x_row, parallel_spec, dim=0, value=0)
    x_row = pad_to_length(
        x_row,
        dim=1,
        length=parallel_spec.n_padded,
        value=0,
    )
    zero_global_padding(x_row, parallel_spec, dim=1, value=0)

    if not parallel_spec.is_distributed:
        out = x_row[:, local_row_start:local_row_end, ...]
        out = out.transpose(0, 1).contiguous()
    else:
        if dist is None or not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Distributed local transpose chunk requires torch.distributed."
            )

        shard = parallel_spec.shard_size
        world_size = parallel_spec.world_size
        chunk_size = local_row_end - local_row_start
        tail_dims = list(range(3, x_row.ndim + 1))

        send_view = x_row.reshape(
            shard,
            world_size,
            shard,
            *x_row.shape[2:],
        )[:, :, local_row_start:local_row_end, ...]
        send_buffer = send_view.permute(1, 2, 0, *tail_dims).contiguous()
        recv_buffer = torch.empty_like(send_buffer)
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            group=parallel_spec.group,
        )

        ordered = recv_buffer.permute(1, 0, 2, *tail_dims).contiguous()
        out = ordered.reshape(
            chunk_size,
            parallel_spec.n_padded,
            *x_row.shape[2:],
        )

    valid_rows = max(
        0,
        min(local_row_end, parallel_spec.local_valid_size) - local_row_start,
    )
    _zero_tail(out, dim=0, cutoff=valid_rows, value=0)
    zero_global_padding(out, parallel_spec, dim=1, value=0)
    return out


def transpose_row(x_col: torch.Tensor, parallel_spec: ParallelSpec) -> torch.Tensor:
    zero_local_padding(x_col, parallel_spec, dim=0, value=0)
    if not parallel_spec.is_distributed:
        out = x_col.transpose(0, 1).contiguous()
        zero_global_padding(out, parallel_spec, dim=0, value=0)
        return out

    x_col = pad_to_length(x_col, dim=1, length=parallel_spec.n_padded, value=0)
    zero_global_padding(x_col, parallel_spec, dim=1, value=0)
    ls = parallel_spec.shard_size
    ws = parallel_spec.world_size
    tail = x_col.shape[2:]

    send = x_col.reshape(ls, ws, ls, *tail)
    perm = [1, 2, 0] + list(range(3, send.ndim))
    send = send.permute(*perm).contiguous()
    recv = _all_to_all_equal_chunks(send, parallel_spec)

    perm_back = [1, 0, 2] + list(range(3, recv.ndim))
    out = recv.permute(*perm_back).contiguous()
    out = out.reshape(ls, parallel_spec.n_padded, *tail)
    zero_local_padding(out, parallel_spec, dim=0, value=0)
    zero_global_padding(out, parallel_spec, dim=1, value=0)
    return out


def trim_to_global_length(
    x: torch.Tensor,
    parallel_spec: ParallelSpec,
    dim: int = 0,
) -> torch.Tensor:
    if x.shape[dim] <= parallel_spec.n_global:
        return x
    return x.narrow(dim, 0, parallel_spec.n_global).contiguous()


def gather_row_sharded_square(
    x_local: torch.Tensor,
    parallel_spec: ParallelSpec,
) -> torch.Tensor:
    x_full = all_gather_first_axis(x_local, parallel_spec)
    x_full = trim_to_global_length(x_full, parallel_spec, dim=0)
    x_full = trim_to_global_length(x_full, parallel_spec, dim=1)
    return x_full.contiguous()
