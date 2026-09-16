import torch
import torch.nn as nn

from torchfold.nn.gated_linear import gated_linear_unit
from torchfold.nn.layer_norm import LayerNorm


from torchfold.runtime_policy import (
    OUTER_PRODUCT_MEAN_D_CHUNK_SIZE,
    TRANSITION_ROW_STREAM_CHUNK_SIZE,
    inference_chunking,
)


class Transition(nn.Module):

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = LayerNorm(c_x)
        self.transition1 = nn.Linear(
            c_x, self.num_intermediate_factor * c_x * 2, bias=False)
        self.transition1_weight_t = None
        self.transition2 = nn.Linear(
            self.num_intermediate_factor * c_x, c_x, bias=False)

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        if self.transition1_weight_t is None:
            self.transition1_weight_t = self.transition1.weight.T.contiguous()
        c = gated_linear_unit(x, self.transition1_weight_t)
        return self.transition2(c)

    @staticmethod
    def _row_stream_dim(x: torch.Tensor) -> int:
        if x.ndim >= 3:
            return x.ndim - 3
        if x.ndim == 2:
            return 0
        raise ValueError(
            "transition residual streaming expects at least two dimensions, "
            f"got shape={tuple(x.shape)}"
        )

    def add_residual_row_streaming_(
        self,
        x: torch.Tensor,
        chunk_size: int = TRANSITION_ROW_STREAM_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Apply an inference-only transition residual in independent rows."""
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            raise RuntimeError(
                "transition residual row streaming is inference-only"
            )
        if chunk_size <= 0:
            raise ValueError("transition row-stream chunk size must be positive")

        dim = self._row_stream_dim(x)
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
            if not compute_chunk.is_contiguous():
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

        self.layer_norm_input = LayerNorm(self.c_msa)
        self.left_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)
        self.right_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)

        self.output_w = nn.Parameter(
            torch.randn(self.num_outer_channel, self.num_outer_channel, self.num_output_channel))
        self.output_b = nn.Parameter(
            torch.randn(self.num_output_channel))

    def _forward_full(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left_act = mask * self.left_projection(msa)
        right_act = mask * self.right_projection(msa)

        left_act = left_act.permute(0, 2, 1).contiguous()
        act = torch.einsum('acb,ade->dceb', left_act, right_act)
        act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
        act = act.permute(1, 0, 2).contiguous()

        norm = torch.einsum('abc,adc->bdc', mask, mask)
        return act / (self.epsilon + norm)

    def _forward_stage12_d_chunked(
        self,
        msa: torch.Tensor,
        mask: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stream both OPM contractions along the output-column axis."""
        mask = mask.unsqueeze(-1)
        normalized_msa = self.layer_norm_input(msa)
        left_act = mask * self.left_projection(normalized_msa)
        right_act = mask * self.right_projection(normalized_msa)
        del normalized_msa

        left_act = left_act.permute(0, 2, 1).contiguous()
        num_rows = left_act.shape[-1]
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
                    "OPM residual shape does not match projected rows: "
                    f"expected={expected_shape} actual={tuple(residual.shape)}"
                )
            out = residual

        # Chunking d changes only independent output columns.  Each tile runs
        # Stage 1 and Stage 2 to completion before it is copied or accumulated.
        for start in range(0, num_columns, OUTER_PRODUCT_MEAN_D_CHUNK_SIZE):
            end = min(start + OUTER_PRODUCT_MEAN_D_CHUNK_SIZE, num_columns)
            right_chunk = right_act[:, start:end, :].contiguous()
            outer_chunk = torch.einsum(
                'acb,ake->kceb',
                left_act,
                right_chunk,
            )
            act = (
                torch.einsum('kceb,cef->kbf', outer_chunk, self.output_w)
                + self.output_b
            )
            act = act.permute(1, 0, 2).contiguous()
            norm_chunk = torch.einsum(
                'abc,akc->bkc',
                mask,
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
    ) -> torch.Tensor:
        """Add inference OPM tiles directly to the pair residual."""
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            raise RuntimeError(
                "OPM residual D streaming is inference-only"
            )
        return self._forward_stage12_d_chunked(msa, mask, residual=residual)

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # In-place residual assembly is not autograd-safe.  Training and any
        # grad-enabled call retain the original full computation graph.
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            return self._forward_full(msa, mask)
        return self._forward_stage12_d_chunked(msa, mask)
