from typing import Optional

import torch
from torchfold.runtime_policy import TRIANGLE_MULTIPLICATION_CHUNK_SIZE
import torch.nn as nn

from torchfold.nn.layer_norm import LayerNorm


class TriangleMultiplication(nn.Module):
    """Triangle multiplication with channel-first projected activations.

    Checkpoint parameters keep their original names and interleaved output-row
    layout.  A non-persistent cache groups those rows as all A channels followed
    by all B channels, allowing forward to create ``[C, N, N]`` activations
    directly instead of materializing a large channel-last transpose.
    """

    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super().__init__()

        self.c_pair = c_pair
        self._outgoing = _outgoing

        self.left_norm_input = LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = LayerNorm(self.c_pair)
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

        # Canonical parameters remain the only persistent state.  This cache is
        # an inference layout derived from projection/gate weights on demand.
        self.register_buffer(
            "_projection_gate_weight_grouped",
            torch.empty(0),
            persistent=False,
        )
        self._projection_gate_weight_signature = None

    @staticmethod
    def _weight_signature(weight: torch.Tensor) -> tuple:
        """Return metadata that changes whenever a packed weight can be stale."""
        try:
            version = weight._version
        except RuntimeError:
            # Inference tensors may omit version counters.  Repacking the small
            # weights on every call is safer than reusing an unverifiable cache.
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
        """Invalidate packed weights after an external storage-level update."""
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
        # A detached cache is correct only for inference.  Build from live
        # parameters when gradients are enabled so autograd keeps both links.
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

    def _prepare_operands(
        self,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize and project full A/B once before any residual writes."""
        n_token = pair.shape[0]
        normalized_pair = self.left_norm_input(pair)
        pair_t = normalized_pair.reshape(-1, self.c_pair).transpose(0, 1)
        projection_weight, gate_weight = self._grouped_weight().split(
            2 * self.c_pair, dim=0,
        )
        projection = torch.matmul(projection_weight, pair_t)
        gate = torch.matmul(gate_weight, pair_t)
        if mask is not None:
            # Retain (projection * mask) * sigmoid(gate).
            self._apply_mask(projection, mask, n_token, n_token)
        gate.sigmoid_()
        projection.mul_(gate)
        del gate
        a, b = projection.view(2, self.c_pair, n_token, n_token).unbind(dim=0)
        return normalized_pair, a, b

    def forward_residual_axis_chunked_(
        self,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor],
        chunk_size: int = TRIANGLE_MULTIPLICATION_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Apply inference Triangle updates in independent residual-axis chunks."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "Residual-axis Triangle chunking is inference-only; "
                "use eval() together with no_grad() or inference_mode()"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if pair.ndim != 3:
            raise ValueError(
                f"pair must have shape [N, N, C], got {tuple(pair.shape)}"
            )

        n_token, n_token_col, c_pair = pair.shape
        if n_token != n_token_col or c_pair != self.c_pair:
            raise ValueError(
                f"pair must have shape [N, N, {self.c_pair}], "
                f"got {tuple(pair.shape)}"
            )

        if mask is not None:
            try:
                pair_mask = mask.expand(n_token, n_token)
            except RuntimeError as exc:
                raise ValueError(
                    f"mask must be broadcastable to [{n_token}, {n_token}], "
                    f"got {tuple(mask.shape)}"
                ) from exc
        else:
            pair_mask = None

        if n_token == 0:
            return pair
        normalized_pair, a, b = self._prepare_operands(pair, pair_mask)
        b_transposed = b.transpose(1, 2)

        # All operands and normalization come from the original full input.
        # Only independent output rows/columns and finishing operations split.
        if self._outgoing:
            for row0 in range(0, n_token, chunk_size):
                row1 = min(row0 + chunk_size, n_token)
                pair_update = torch.bmm(a[:, row0:row1], b_transposed)
                self._finish_update(
                    pair[row0:row1], normalized_pair[row0:row1], pair_update,
                )
                del pair_update
        else:
            for col0 in range(0, n_token, chunk_size):
                col1 = min(col0 + chunk_size, n_token)
                pair_update = torch.bmm(b_transposed, a[:, :, col0:col1])
                self._finish_update(
                    pair[:, col0:col1], normalized_pair[:, col0:col1], pair_update,
                )
                del pair_update
        return pair


    def forward(
        self,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply outgoing or incoming multiplication to ``pair[N, N, C]``.

        ``mask=None`` is valid only when the caller has already established
        that no pair entry is padded.  The check deliberately stays outside
        this hot path to avoid a device-to-host synchronization.
        """
        if pair.ndim != 3:
            raise ValueError(
                f"pair must have shape [N, N, C], got {tuple(pair.shape)}"
            )

        n_token, n_token_col, c_pair = pair.shape
        if n_token != n_token_col or c_pair != self.c_pair:
            raise ValueError(
                f"pair must have shape [N, N, {self.c_pair}], "
                f"got {tuple(pair.shape)}"
            )

        residual = pair
        normalized_pair, a, b = self._prepare_operands(pair, mask)
        # Plain forward computes the complete update.  Inference chunking is
        # selected explicitly by Pairformer/Evoformer via the residual API.
        if self._outgoing:
            # u[c,i,j] = sum_k a[c,i,k] * b[c,j,k]
            pair_update = torch.bmm(a, b.transpose(1, 2))
        else:
            # u[c,i,j] = sum_k b[c,k,i] * a[c,k,j]
            pair_update = torch.bmm(b.transpose(1, 2), a)
        result = self._finish_update(
            residual,
            normalized_pair,
            pair_update,
        )
        del pair_update
        del a, b
        return result
