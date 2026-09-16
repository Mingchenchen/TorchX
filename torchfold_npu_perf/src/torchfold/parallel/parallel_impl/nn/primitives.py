import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import fastnn
from ...parallel_ops import ParallelSpec
from ...partition import balanced_partition_range

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from torchfold.runtime_policy import (
    PARALLEL_PAIR_TRANSITION_CHUNK_SIZE as PAIR_TRANSITION_CHUNK_SIZE,
    PARALLEL_OUTER_PRODUCT_MEAN_D_CHUNK_SIZE as OUTER_PRODUCT_MEAN_D_CHUNK_SIZE,
    PARALLEL_TRANSITION_ROW_STREAM_CHUNK_SIZE as TRANSITION_ROW_STREAM_CHUNK_SIZE,
    inference_chunking,
)


class Transition(nn.Module):

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = fastnn.LayerNorm(c_x)
        self.transition1 = nn.Linear(
            c_x, self.num_intermediate_factor * c_x * 2, bias=False)
        self.transition1_weight_t = None
        self.transition2 = nn.Linear(
            self.num_intermediate_factor * c_x, c_x, bias=False)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        if self.transition1_weight_t is None:
            self.transition1_weight_t = self.transition1.weight.T.contiguous()
        c = fastnn.gated_linear_unit(x, self.transition1_weight_t)
        return self.transition2(c)

    def forward_chunked(
        self,
        x: torch.Tensor,
        chunk_size: int = PAIR_TRANSITION_CHUNK_SIZE,
        force_contiguous: bool = True,
    ) -> torch.Tensor:
        """Run pair-transition rows in fixed chunks to cap activation peaks."""
        if x.ndim >= 3:
            dim = x.ndim - 3
        elif x.ndim == 2:
            dim = 0
        else:
            return self._forward_chunk(x)

        out = torch.empty_like(x)
        length_total = x.shape[dim]
        for start in range(0, length_total, chunk_size):
            length = min(chunk_size, length_total - start)
            x_slice = torch.narrow(x, dim=dim, start=start, length=length)
            if force_contiguous and not x_slice.is_contiguous():
                x_slice = x_slice.contiguous()
            y_slice = self._forward_chunk(x_slice)
            torch.narrow(out, dim=dim, start=start, length=length).copy_(y_slice)
        return out

    def add_residual_row_streaming_(
        self,
        x: torch.Tensor,
        chunk_size: int = TRANSITION_ROW_STREAM_CHUNK_SIZE,
        force_contiguous: bool = True,
    ) -> torch.Tensor:
        """Apply full-weight row chunks without retaining a full update."""
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            raise RuntimeError(
                "transition residual row streaming is inference-only"
            )
        if chunk_size <= 0:
            raise ValueError("transition row-stream chunk size must be positive")

        if x.ndim >= 3:
            dim = x.ndim - 3
        elif x.ndim == 2:
            dim = 0
        else:
            raise ValueError(
                "transition residual streaming expects at least two "
                f"dimensions, got shape={tuple(x.shape)}"
            )

        length_total = x.shape[dim]
        for start in range(0, length_total, chunk_size):
            length = min(chunk_size, length_total - start)
            residual_chunk = torch.narrow(
                x,
                dim=dim,
                start=start,
                length=length,
            )
            compute_chunk = residual_chunk
            if force_contiguous and not compute_chunk.is_contiguous():
                compute_chunk = compute_chunk.contiguous()
            update_chunk = self._forward_chunk(compute_chunk)
            residual_chunk.add_(update_chunk)
            del residual_chunk, compute_chunk, update_chunk
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_chunk(x)

class OuterProductMean(nn.Module):
    def __init__(self, c_msa: int = 64, num_output_channel: int = 128, num_outer_channel: int = 32) -> None:
        super(OuterProductMean, self).__init__()

        self.c_msa = c_msa
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.epsilon = 1e-3

        self.layer_norm_input = fastnn.LayerNorm(self.c_msa)
        self.left_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)
        self.right_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)

        self.output_w = nn.Parameter(
            torch.randn(self.num_outer_channel, self.num_outer_channel, self.num_output_channel))
        self.output_b = nn.Parameter(
            torch.randn(self.num_output_channel))

    def _forward_full(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        msa:  [N_msa, L, c_msa]
        mask: [N_msa, L]
        out:  [L, L, c_pair]
        """
        mask = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)

        left_act = mask * self.left_projection(msa)    # [N_msa, L, C]
        right_act = mask * self.right_projection(msa)

        left_act = left_act.permute(0, 2, 1).contiguous()  # [N_msa, C, L]
        act = torch.einsum('acb,ade,cef->dbf', left_act, right_act, self.output_w)
        act = act.permute(1, 0, 2).contiguous()  # [L, L, c_pair]
        act = act + self.output_b

        norm = torch.einsum('abc,adc->bdc', mask, mask)  # [L, L, 1]
        return act / (self.epsilon + norm)

    def _forward_row_shard(
        self,
        msa: torch.Tensor,
        mask: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        """
        Parallel row-shard path without sequence chunking.

        Inputs:
            msa:  [N_msa, Lp, c_msa]  (full replicated, padded)
            mask: [N_msa, Lp]         (full replicated, padded)

        Output:
            act_local: [Ls, Lp, c_pair]
        """
        mask = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)

        msa_local = msa[:, parallel_spec.start:parallel_spec.end, :].contiguous()      # [N_msa, Ls, c_msa]
        mask_local = mask[:, parallel_spec.start:parallel_spec.end, :].contiguous()    # [N_msa, Ls, 1]

        left_act_local = mask_local * self.left_projection(msa_local)         # [N_msa, Ls, C]
        right_act = mask * self.right_projection(msa)                         # [N_msa, Lp, C]

        left_act_local = left_act_local.permute(0, 2, 1).contiguous()        # [N_msa, C, Ls]
        act_local = torch.einsum('acb,ade,cef->dbf', left_act_local, right_act, self.output_w)
        act_local = act_local.permute(1, 0, 2).contiguous()                  # [Ls, Lp, c_pair]
        act_local = act_local + self.output_b

        norm_local = torch.einsum('abc,adc->bdc', mask_local, mask)          # [Ls, Lp, 1]
        return act_local / (self.epsilon + norm_local)

    def _forward_stage12_d_chunked(
        self,
        msa: torch.Tensor,
        mask: torch.Tensor,
        parallel_spec: ParallelSpec = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stream both OPM contractions over full or row-sharded columns."""
        mask = mask.unsqueeze(-1)
        normalized_msa = self.layer_norm_input(msa)

        row_sharded = (
            parallel_spec is not None and parallel_spec.world_size > 1
        )
        if row_sharded:
            left_msa = normalized_msa[
                :, parallel_spec.start:parallel_spec.end, :
            ].contiguous()
            left_mask = mask[
                :, parallel_spec.start:parallel_spec.end, :
            ].contiguous()
        else:
            left_msa = normalized_msa
            left_mask = mask

        left_act_local = left_mask * self.left_projection(left_msa)
        right_act = mask * self.right_projection(normalized_msa)
        del normalized_msa, left_msa
        left_act_local = left_act_local.permute(0, 2, 1).contiguous()
        num_rows = left_act_local.shape[-1]
        num_columns = right_act.shape[1]
        expected_shape = (
            num_rows,
            num_columns,
            self.num_output_channel,
        )

        if residual is None:
            out = torch.empty(
                expected_shape,
                dtype=msa.dtype,
                device=msa.device,
            )
        else:
            if tuple(residual.shape) != expected_shape:
                raise ValueError(
                    "OPM residual shape does not match local row ownership: "
                    f"expected={expected_shape} actual={tuple(residual.shape)}"
                )
            out = residual

        # Tile global columns d for local rows only; no collective is needed.
        for start in range(0, num_columns, OUTER_PRODUCT_MEAN_D_CHUNK_SIZE):
            end = min(start + OUTER_PRODUCT_MEAN_D_CHUNK_SIZE, num_columns)
            right_chunk = right_act[:, start:end, :].contiguous()
            outer_chunk = torch.einsum(
                'acb,ake->kceb',
                left_act_local,
                right_chunk,
            )
            act = (
                torch.einsum('kceb,cef->kbf', outer_chunk, self.output_w)
                + self.output_b
            )
            act = act.permute(1, 0, 2).contiguous()
            norm_chunk = torch.einsum(
                'abc,akc->bkc',
                left_mask,
                mask[:, start:end, :],
            )
            act.div_(self.epsilon + norm_chunk)

            output_chunk = out[:, start:end, :]
            if residual is None:
                output_chunk.copy_(act)
            else:
                output_chunk.add_(act)
            del right_chunk, outer_chunk, act, norm_chunk, output_chunk

        return out

    def add_residual_stage12_d_chunked_(
        self,
        msa: torch.Tensor,
        mask: torch.Tensor,
        residual: torch.Tensor,
        parallel_spec: ParallelSpec = None,
    ) -> torch.Tensor:
        """Add inference OPM tiles directly to the owned pair rows."""
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            raise RuntimeError(
                "OPM residual D streaming is inference-only"
            )
        return self._forward_stage12_d_chunked(
            msa,
            mask,
            parallel_spec=parallel_spec,
            residual=residual,
        )

    def forward(
        self,
        msa: torch.Tensor,
        mask: torch.Tensor,
        parallel_spec: ParallelSpec = None,
    ) -> torch.Tensor:
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            if parallel_spec is not None and parallel_spec.world_size > 1:
                return self._forward_row_shard(msa, mask, parallel_spec)
            return self._forward_full(msa, mask)
        return self._forward_stage12_d_chunked(
            msa,
            mask,
            parallel_spec=parallel_spec,
        )


class TransitionParallel(Transition):
    def __init__(self, c_x: int, num_intermediate_factor: int = 4, parallel_group=None) -> None:
        super().__init__(c_x=c_x, num_intermediate_factor=num_intermediate_factor)
        self.parallel_group = parallel_group
        self._parallel_cache = None  # (key, w1t_local, w2_local, s, e, H)

    def _parallel_enabled(self) -> bool:
        if dist is None:
            return False
        if self.parallel_group is None:
            return False
        return dist.is_initialized() and dist.get_world_size(self.parallel_group) > 1

    def _get_local_weights(self):
        parallel_rank = dist.get_rank(self.parallel_group)
        parallel_size = dist.get_world_size(self.parallel_group)

        H = self.num_intermediate_factor * self.c_in
        s, e = balanced_partition_range(H, parallel_size, parallel_rank)

        w1 = self.transition1.weight  # [2H, C]
        w2 = self.transition2.weight  # [C, H]
        key = (parallel_rank, parallel_size, w1.device, w1.dtype, w1.data_ptr(), w2.data_ptr())

        if self._parallel_cache is not None and self._parallel_cache[0] == key:
            _, w1t_local, w2_local, s0, e0, H0 = self._parallel_cache
            return w1t_local, w2_local, s0, e0, H0

        # Slice local gate/value rows, then transpose for fastnn.gated_linear_unit.
        w1_local = torch.cat([w1[s:e], w1[H + s:H + e]], dim=0).contiguous()  # [2*h_local, C]
        w1t_local = w1_local.T.contiguous()  # [C, 2*h_local]

        w2_local = w2[:, s:e].contiguous()  # [C, h_local]

        self._parallel_cache = (key, w1t_local, w2_local, s, e, H)
        return w1t_local, w2_local, s, e, H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._parallel_enabled():
            return self._forward_chunk(x)

        w1t_local, w2_local, *_ = self._get_local_weights()
        x = self.input_layer_norm(x)
        c_local = fastnn.gated_linear_unit(x, w1t_local)  # [..., h_local]
        out = F.linear(c_local, w2_local)  # [..., C]

        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.parallel_group)
        return out

    def add_parallel_residual_row_streaming_(
        self,
        x: torch.Tensor,
        chunk_size: int = TRANSITION_ROW_STREAM_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Row-stream local TP work while preserving one full all-reduce."""
        if not self._parallel_enabled():
            return self.add_residual_row_streaming_(x, chunk_size=chunk_size)
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            raise RuntimeError(
                "transition residual row streaming is inference-only"
            )
        if chunk_size <= 0:
            raise ValueError("transition row-stream chunk size must be positive")

        if x.ndim >= 3:
            dim = x.ndim - 3
        elif x.ndim == 2:
            dim = 0
        else:
            raise ValueError(
                "transition residual streaming expects at least two "
                f"dimensions, got shape={tuple(x.shape)}"
            )

        w1t_local, w2_local, *_ = self._get_local_weights()
        local_update = None
        length_total = x.shape[dim]
        for start in range(0, length_total, chunk_size):
            length = min(chunk_size, length_total - start)
            input_chunk = torch.narrow(
                x,
                dim=dim,
                start=start,
                length=length,
            )
            if not input_chunk.is_contiguous():
                input_chunk = input_chunk.contiguous()
            normalized_chunk = self.input_layer_norm(input_chunk)
            hidden_chunk = fastnn.gated_linear_unit(
                normalized_chunk,
                w1t_local,
            )
            update_chunk = F.linear(hidden_chunk, w2_local)
            if local_update is None:
                # Preserve F.linear dtype/layout for the single full-shape collective.
                local_update = torch.empty(
                    x.shape,
                    dtype=update_chunk.dtype,
                    device=update_chunk.device,
                )
            torch.narrow(
                local_update,
                dim=dim,
                start=start,
                length=length,
            ).copy_(update_chunk)
            del input_chunk, normalized_chunk, hidden_chunk, update_chunk

        if local_update is None:
            local_update = torch.empty(
                x.shape,
                dtype=x.dtype,
                device=x.device,
            )
        dist.all_reduce(
            local_update,
            op=dist.ReduceOp.SUM,
            group=self.parallel_group,
        )
        x.add_(local_update)
        del local_update
        return x


class OuterProductMeanParallel(OuterProductMean):
    def __init__(
        self,
        c_msa: int = 64,
        num_output_channel: int = 128,
        num_outer_channel: int = 32,
        parallel_group=None
    ) -> None:
        super().__init__(
            c_msa=c_msa,
            num_output_channel=num_output_channel,
            num_outer_channel=num_outer_channel,
        )
        self.parallel_group = parallel_group
