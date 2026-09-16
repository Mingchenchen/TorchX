from typing import Optional

import torch
import torch.nn as nn

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from .. import fastnn
from ...parallel_ops import ParallelSpec
from torchfold.runtime_policy import (
    PARALLEL_TRIANGLE_MULTIPLICATION_CHUNK_SIZE,
    parallel_triangle_chunked,
)


class TriangleMultiplication(nn.Module):
    """Triangle multiplication for row- and column-sharded pair tensors.

    Every rank owns ``pair[L_local, L_full, C]``.  Outgoing multiplication
    receives row layout; incoming receives the column layout produced by
    Pairformer.  Consequently, both distributed directions contract the local
    A block with broadcast B row shards.  The collective order and layouts are
    unchanged; only rank-local projection and contraction use the v3 path.
    """

    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super().__init__()

        self.c_pair = c_pair
        self._outgoing = _outgoing

        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair,
            self.c_pair,
            bias=False,
        )
        self.gating_linear = nn.Linear(
            self.c_pair,
            self.c_pair,
            bias=False,
        )

        # Derived inference weights are deliberately excluded from checkpoints.
        self.register_buffer(
            "_projection_gate_weight_grouped",
            torch.empty(0),
            persistent=False,
        )
        self._projection_gate_weight_signature = None

    @staticmethod
    def _weight_signature(weight: torch.Tensor) -> tuple:
        try:
            version = weight._version
        except RuntimeError:
            version = None
        return (
            weight.data_ptr(),
            version,
            weight.device,
            weight.dtype,
            tuple(weight.shape),
            tuple(weight.stride()),
        )

    @staticmethod
    def _pack_ab_weight(
        weight: torch.Tensor,
        c_pair: int,
    ) -> torch.Tensor:
        """Group rows ``[a0,b0,a1,b1,...]`` as ``[all A, all B]``."""
        return (
            weight.reshape(c_pair, 2, weight.shape[-1])
            .transpose(0, 1)
            .reshape(2 * c_pair, weight.shape[-1])
            .contiguous()
        )

    def invalidate_weight_cache_(self) -> "TriangleMultiplication":
        self._projection_gate_weight_signature = None
        return self

    @torch.no_grad()
    def pack_weights_(self) -> "TriangleMultiplication":
        """Refresh the grouped inference weights when canonical weights change."""
        signature = (
            self._weight_signature(self.projection.weight),
            self._weight_signature(self.gate.weight),
        )
        if (
            signature[0][1] is None
            or signature[1][1] is None
            or signature != self._projection_gate_weight_signature
        ):
            self._projection_gate_weight_grouped = torch.cat(
                (
                    self._pack_ab_weight(
                        self.projection.weight.detach(),
                        self.c_pair,
                    ),
                    self._pack_ab_weight(
                        self.gate.weight.detach(),
                        self.c_pair,
                    ),
                ),
                dim=0,
            ).contiguous()
            self._projection_gate_weight_signature = signature
        return self

    def _grouped_weight(self) -> torch.Tensor:
        if torch.is_grad_enabled():
            return torch.cat(
                (
                    self._pack_ab_weight(
                        self.projection.weight,
                        self.c_pair,
                    ),
                    self._pack_ab_weight(
                        self.gate.weight,
                        self.c_pair,
                    ),
                ),
                dim=0,
            )

        self.pack_weights_()
        return self._projection_gate_weight_grouped

    @staticmethod
    def _apply_mask(
        projection: torch.Tensor,
        mask: torch.Tensor,
        n_row: int,
        n_col: int,
    ) -> None:
        """Apply any mask accepted by the original broadcast semantics."""
        if tuple(mask.shape) == (n_row, n_col):
            projection.mul_(mask.reshape(1, n_row * n_col))
            return

        try:
            broadcast_mask = mask.expand(n_row, n_col)
        except RuntimeError as exc:
            raise ValueError(
                f"mask must be broadcastable to [{n_row}, {n_col}], "
                f"got {tuple(mask.shape)}"
            ) from exc
        projection.view(-1, n_row, n_col).mul_(
            broadcast_mask.unsqueeze(0)
        )

    @staticmethod
    def _group_src_rank(
        parallel_spec: ParallelSpec,
        src_rank_in_group: int,
    ) -> int:
        if dist is None or parallel_spec.group is None:
            return src_rank_in_group
        if parallel_spec.group == dist.group.WORLD:
            return src_rank_in_group
        if hasattr(dist, "get_global_rank"):
            return dist.get_global_rank(
                parallel_spec.group,
                src_rank_in_group,
            )
        return src_rank_in_group

    def _stream_triangle_mul(
        self,
        a_local: torch.Tensor,
        b_local: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        """Build all global output columns one broadcast/BMM block at a time."""
        out = a_local.new_empty(
            (self.c_pair, a_local.shape[1], parallel_spec.n_padded)
        )
        for src in range(parallel_spec.world_size):
            if src == parallel_spec.rank:
                b_block = b_local
            else:
                b_block = torch.empty_like(b_local)

            dist.broadcast(
                b_block,
                src=self._group_src_rank(parallel_spec, src),
                group=parallel_spec.group,
            )

            j0 = src * parallel_spec.shard_size
            j1 = j0 + parallel_spec.shard_size
            out[:, :, j0:j1].copy_(
                torch.bmm(a_local, b_block.transpose(1, 2))
            )
            del b_block
        return out


    def _add_distributed_residual_chunked_(
        self,
        residual: torch.Tensor,
        normalized_pair: torch.Tensor,
        a_local: torch.Tensor,
        b_local: torch.Tensor,
        parallel_spec: ParallelSpec,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        """Finish local-row tiles without materializing a full pair update.

        A/B are projected before any residual writes, so all broadcasts use
        the original operands. Both distributed directions use this contraction
        in their existing row/column layouts. Each contraction still reduces
        over the complete global token axis; only independent output rows split.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("distributed Triangle chunking is inference-only")
        if chunk_size < 1:
            raise ValueError("distributed Triangle chunk size must be positive")
        for src in range(parallel_spec.world_size):
            b_block = b_local if src == parallel_spec.rank else torch.empty_like(b_local)
            # Same count, shape and rank order as the unchunked path.
            dist.broadcast(
                b_block,
                src=self._group_src_rank(parallel_spec, src),
                group=parallel_spec.group,
            )
            j0 = src * parallel_spec.shard_size
            j1 = j0 + parallel_spec.shard_size
            for i0 in range(0, parallel_spec.shard_size, chunk_size):
                i1 = min(i0 + chunk_size, parallel_spec.shard_size)
                pair_update = torch.bmm(
                    a_local[:, i0:i1, :], b_block.transpose(1, 2),
                )
                self._finish_update(
                    residual[i0:i1, j0:j1],
                    normalized_pair[i0:i1, j0:j1],
                    pair_update,
                )
                del pair_update
            del b_block
        return residual

    def _finish_update(
        self,
        residual: torch.Tensor,
        normalized_pair: torch.Tensor,
        pair_update: torch.Tensor,
    ) -> torch.Tensor:
        pair_update = pair_update.permute(1, 2, 0).contiguous()
        pair_update = self.center_norm(pair_update)
        pair_update = self.output_projection(pair_update)

        output_gate = self.gating_linear(normalized_pair)
        if torch.is_grad_enabled():
            return torch.addcmul(
                residual,
                pair_update,
                torch.sigmoid(output_gate),
            )

        output_gate.sigmoid_()
        residual.addcmul_(pair_update, output_gate)
        return residual

    def forward(
        self,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor],
        parallel_spec: Optional[ParallelSpec] = None,
    ) -> torch.Tensor:
        """Update ``pair[L_local, L_full, C]`` in full or sharded layout.

        Distributed callers must retain their real padding mask.  In
        particular, world-size alignment can add padding even when the
        unsharded input had none.
        """
        if pair.ndim != 3 or pair.shape[-1] != self.c_pair:
            raise ValueError(
                "pair must have shape [L_local, L_full, "
                f"{self.c_pair}], got {tuple(pair.shape)}"
            )

        local_size, full_size, _ = pair.shape
        is_distributed = (
            parallel_spec is not None and parallel_spec.world_size > 1
        )
        if not is_distributed and local_size != full_size:
            raise ValueError(
                "non-distributed pair must be square, got "
                f"{tuple(pair.shape)}"
            )
        if is_distributed and (
            local_size != parallel_spec.shard_size
            or full_size != parallel_spec.n_padded
        ):
            raise ValueError(
                "pair shape does not match ParallelSpec: "
                f"shape={tuple(pair.shape)}, "
                f"shard_size={parallel_spec.shard_size}, "
                f"n_padded={parallel_spec.n_padded}"
            )
        if is_distributed and (dist is None or not dist.is_initialized()):
            raise RuntimeError(
                "parallel_spec.world_size > 1 requires initialized "
                "torch.distributed"
            )
        if is_distributed and torch.is_grad_enabled():
            # Broadcast is not autograd-aware; remote B gradients would be lost.
            raise RuntimeError(
                "distributed TriangleMultiplication is inference-only; use "
                "torch.inference_mode() or torch.no_grad()"
            )

        residual = pair
        normalized_pair = self.left_norm_input(pair)

        # Create channel-first projections directly from a transpose view.
        n_pair = local_size * full_size
        pair_2d = normalized_pair.reshape(n_pair, self.c_pair)
        pair_2d_t = pair_2d.transpose(0, 1)
        grouped_weight = self._grouped_weight()
        projection_weight, gate_weight = grouped_weight.split(
            2 * self.c_pair,
            dim=0,
        )
        projection = torch.matmul(projection_weight, pair_2d_t)
        gate = torch.matmul(gate_weight, pair_2d_t)
        if mask is not None:
            # Preserve (projection * mask) * sigmoid(gate).
            self._apply_mask(projection, mask, local_size, full_size)
        gate.sigmoid_()
        projection.mul_(gate)
        del gate, projection_weight, gate_weight, grouped_weight

        projection = projection.view(
            2,
            self.c_pair,
            local_size,
            full_size,
        )
        a_local, b_local = projection.unbind(dim=0)

        if not is_distributed:
            # Non-distributed fallback uses one complete update.
            if self._outgoing:
                pair_update = torch.bmm(
                    a_local,
                    b_local.transpose(1, 2),
                )
            else:
                pair_update = torch.bmm(
                    b_local.transpose(1, 2),
                    a_local,
                )
            result = self._finish_update(
                residual,
                normalized_pair,
                pair_update,
            )
            del pair_update
            del a_local, b_local, projection, pair_2d, pair_2d_t
            return result

        if parallel_triangle_chunked(
            padded_length=parallel_spec.n_padded,
            world_size=parallel_spec.world_size,
        ):
            result = self._add_distributed_residual_chunked_(
                residual, normalized_pair, a_local, b_local, parallel_spec,
                chunk_size=PARALLEL_TRIANGLE_MULTIPLICATION_CHUNK_SIZE,
            )
            del a_local, b_local, projection, pair_2d, pair_2d_t
            return result

        pair_update = self._stream_triangle_mul(
            a_local,
            b_local,
            parallel_spec,
        )
        del a_local, b_local, projection, pair_2d, pair_2d_t
        result = self._finish_update(
            residual,
            normalized_pair,
            pair_update,
        )
        return result
