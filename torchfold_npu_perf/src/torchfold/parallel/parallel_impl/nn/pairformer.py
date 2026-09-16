from typing import Optional

import torch
from torchfold.runtime_policy import inference_chunking
import torch.nn as nn

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from .primitives import (
    Transition,
    TransitionParallel,
    OuterProductMeanParallel,
)
from .triangle_multiplication import TriangleMultiplication
from .attention import GridSelfAttention, MSAAttention
from .diffusion_transformer import SelfAttention

from .. import fastnn
from ...parallel_ops import (
    ParallelSpec,
    transpose_col,
    transpose_row,
    all_gather_first_axis,
    pad_to_length,
    pair_padding_mask_col_from_row,
    slice_1d_local,
    zero_global_padding,
    zero_local_padding,
)


def _add_full_pair_transition_residual_(
    transition: Transition,
    pair: torch.Tensor,
) -> torch.Tensor:
    add_parallel = getattr(
        transition,
        "add_parallel_residual_row_streaming_",
        None,
    )
    if add_parallel is not None:
        return add_parallel(pair)
    return transition.add_residual_row_streaming_(pair)


class PairformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int = 16,
        c_pair: int = 128,
        c_single: int = 384,
        c_hidden_mul: int = 128,
        n_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        with_single: bool = True,
        parallel_group="auto",
    ) -> None:
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        self.with_single = with_single
        self.num_intermediate_factor = num_intermediate_factor

        if parallel_group == "auto":
            parallel_group = dist.group.WORLD if (dist is not None and dist.is_initialized()) else None

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair, _outgoing=True
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair, _outgoing=False
        )
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )

        self.pair_transition = Transition(
            c_x=c_pair,
            num_intermediate_factor=self.num_intermediate_factor,
        )

        self.c_single = c_single
        if self.with_single is True:
            self.single_pair_logits_norm = fastnn.LayerNorm(c_pair)
            self.single_pair_logits_projection = nn.Linear(
                c_pair, n_heads, bias=False
            )
            self.single_attention_ = SelfAttention(
                c_x=c_single,
                num_head=n_heads,
                use_single_cond=False,
                is_decoder=False,
            )

            # Replicated single features must not use WORLD tensor parallelism.
            if parallel_group is None:
                self.single_transition = Transition(
                    c_x=self.c_single,
                    num_intermediate_factor=self.num_intermediate_factor,
                )
            else:
                self.single_transition = TransitionParallel(
                    c_x=self.c_single,
                    parallel_group=parallel_group,
                )

    def _forward_pair_transition_chunked(self, pair: torch.Tensor) -> torch.Tensor:
        forward_chunked = getattr(self.pair_transition, "forward_chunked", None)
        if forward_chunked is None:
            return self.pair_transition(pair)
        return forward_chunked(pair)

    def _forward_single_transition_full(self, single_local: torch.Tensor) -> torch.Tensor:
        """Apply complete Transition weights independently to local tokens.

        ``single_transition`` is a ``TransitionParallel`` in the distributed
        Pairformer today.  Calling its override on different token shards would
        all-reduce unrelated rows.  Invoking the base implementation preserves
        the checkpoint/state-dict layout while using the complete weights and
        deliberately avoids that hidden-dimension collective.
        """
        return Transition.forward(self.single_transition, single_local)

    def _forward_single_local_q(
        self,
        pair_logits_local: torch.Tensor,
        single: torch.Tensor,
        seq_mask: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        """Run local-Q/global-KV single attention and gather only ``[Ls, C]``.

        ``SelfAttention`` slices the packed QKV projection weights in this
        rectangular path, so local rows compute Q only and replicated global
        rows compute K/V only.

        Args:
            pair_logits_local: Row-sharded pair bias in ``[H, Ls, Lp]``.
            single: Replicated single representation in ``[L, C]`` or
                ``[Lp, C]``.
            seq_mask: Sequence mask in ``[L]`` or ``[Lp]``.
            parallel_spec: Row-sharding and padding contract.
        """
        expected_bias_shape = (
            self.n_heads,
            parallel_spec.shard_size,
            parallel_spec.n_padded,
        )
        if tuple(pair_logits_local.shape) != expected_bias_shape:
            raise ValueError(
                "Pairformer local-Q pair bias requires shape "
                f"{expected_bias_shape}, got {tuple(pair_logits_local.shape)}"
            )
        if single.ndim != 2 or single.shape[-1] != self.c_single:
            raise ValueError(
                "Pairformer single representation requires shape [L, "
                f"{self.c_single}], got {tuple(single.shape)}"
            )
        if single.shape[0] not in {parallel_spec.n_global, parallel_spec.n_padded}:
            raise ValueError(
                "Pairformer single length must equal n_global or n_padded, got "
                f"{single.shape[0]} for ({parallel_spec.n_global}, "
                f"{parallel_spec.n_padded})"
            )
        if seq_mask.ndim != 1 or seq_mask.shape[0] not in {
            parallel_spec.n_global,
            parallel_spec.n_padded,
        }:
            raise ValueError(
                "Pairformer sequence mask length must equal n_global or "
                f"n_padded, got {tuple(seq_mask.shape)}"
            )

        single_full = pad_to_length(
            single,
            dim=0,
            length=parallel_spec.n_padded,
            value=0.0,
        )
        seq_mask_full = pad_to_length(
            seq_mask,
            dim=0,
            length=parallel_spec.n_padded,
            value=0,
        )
        zero_global_padding(single_full, parallel_spec, dim=0, value=0.0)
        zero_global_padding(seq_mask_full, parallel_spec, dim=0, value=0)

        # Clone local rows to preserve replicated K/V until attention consumes it.
        single_local = slice_1d_local(
            single_full,
            parallel_spec,
            pad_value=0.0,
        ).clone()
        seq_mask_local = slice_1d_local(
            seq_mask_full,
            parallel_spec,
            pad_value=0,
        )

        single_local = single_local + self.single_attention_(
            x_q=single_local,
            mask_q=seq_mask_local,
            pair_logits=pair_logits_local,
            x_kv=single_full,
            mask_kv=seq_mask_full,
            fa_mask_contract="sequence_padding",
            key_valid_len=parallel_spec.n_global,
        )
        single_local = single_local + self._forward_single_transition_full(
            single_local
        )
        zero_local_padding(single_local, parallel_spec, dim=0, value=0.0)

        single_full = all_gather_first_axis(
            single_local.contiguous(),
            parallel_spec,
        )
        return zero_global_padding(
            single_full,
            parallel_spec,
            dim=0,
            value=0.0,
        )

    def forward(
        self,
        pair_row: Optional[torch.Tensor],
        pair_mask_row: torch.Tensor,
        single: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        parallel_spec: Optional[ParallelSpec] = None,
        _pair_owner: Optional[list[torch.Tensor]] = None,
    ):
        # Consume a container so caller frames and Module.__call__ cannot retain
        # the old row Tensor after redistribution.
        if _pair_owner is not None:
            if (parallel_spec is None or parallel_spec.world_size <= 1 or self.training
                    or torch.is_grad_enabled() or pair_row is not None or len(_pair_owner) != 1):
                raise ValueError("Confidence Pair handoff requires one owned Pair in distributed inference")
            pair_row = _pair_owner.pop()
        if parallel_spec is None or parallel_spec.world_size == 1:
            pair = self.triangle_multiplication_outgoing(
                pair_row, mask=pair_mask_row
            )
            pair = self.triangle_multiplication_incoming(
                pair, mask=pair_mask_row
            )
            pair = pair + self.pair_attention1(pair, mask_local=pair_mask_row, parallel_spec=parallel_spec)
            pair = pair + self.pair_attention2(pair, mask_local=pair_mask_row, parallel_spec=parallel_spec)
            if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
                pair = pair + self.pair_transition(pair)
            else:
                _add_full_pair_transition_residual_(
                    self.pair_transition,
                    pair,
                )

            if self.with_single is True:
                if single is None or seq_mask is None:
                    raise ValueError("single / seq_mask must be provided when with_single=True")

                pair_logits = self.single_pair_logits_projection(
                    self.single_pair_logits_norm(pair)
                )  # [L, L, H]
                pair_logits = pair_logits.permute(2, 0, 1).contiguous()  # [H, L, L]

                single = single + self.single_attention_(
                    single,
                    seq_mask,
                    pair_logits=pair_logits,
                )
                single = single + self.single_transition(single)
                return pair, single

            return pair

        pair_row = self.triangle_multiplication_outgoing(
            pair_row, mask=pair_mask_row, parallel_spec=parallel_spec
        )

        pair_col = transpose_col(pair_row, parallel_spec)
        if _pair_owner is not None:
            # transpose_row below creates the successor row.
            del pair_row
        # Symmetric pair-padding masks need no AllToAllV.
        pair_mask_col = pair_padding_mask_col_from_row(
            pair_mask_row,
            parallel_spec,
        )
        pair_col = self.triangle_multiplication_incoming(
            pair_col, mask=pair_mask_col, parallel_spec=parallel_spec
        )

        pair_row = transpose_row(pair_col, parallel_spec)
        # The pair-padding outer-product contract enables a static FA mask plan.
        pair_row += self.pair_attention1(
            pair_row,
            mask_local=pair_mask_row,
            parallel_spec=parallel_spec,
            fa_mask_contract="pair_padding",
        )

        if hasattr(self.pair_attention2, "forward_add_"):
            self.pair_attention2.forward_add_(
                pair_row,
                mask_local=pair_mask_row,
                parallel_spec=parallel_spec,
                fa_mask_contract="pair_padding",
            )
        else:
            pair_row += self.pair_attention2(
                pair_row,
                mask_local=pair_mask_row,
                parallel_spec=parallel_spec,
                fa_mask_contract="pair_padding",
            )
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            pair_row += self._forward_pair_transition_chunked(pair_row)
        else:
            # Replicated weights make the row-sharded transition communication-free.
            self.pair_transition.add_residual_row_streaming_(pair_row)

        if self.with_single is True:
            if single is None or seq_mask is None:
                raise ValueError("single / seq_mask must be provided when with_single=True")

            pair_logits_local = self.single_pair_logits_projection(
                self.single_pair_logits_norm(pair_row)
            )  # [Ls, Lp, H]

            # Shard Q and pair bias, replicate K/V, and gather only [Ls, C] outputs.
            pair_logits_local = pair_logits_local.permute(
                2, 0, 1
            ).contiguous()  # [H, Ls, Lp]
            single = self._forward_single_local_q(
                pair_logits_local,
                single,
                seq_mask,
                parallel_spec,
            )
            return pair_row, single

        return pair_row


class EvoformerBlock(nn.Module):
    def __init__(self, c_msa: int = 64, c_pair: int = 128, n_heads_pair: int = 4) -> None:
        super(EvoformerBlock, self).__init__()

        parallel_group = dist.group.WORLD if (dist is not None and dist.is_initialized()) else None

        self.outer_product_mean = OuterProductMeanParallel(
            c_msa=c_msa,
            num_output_channel=c_pair,
            parallel_group=parallel_group,
        )

        self.msa_attention1 = MSAAttention(c_msa=c_msa, c_pair=c_pair)

        self.msa_transition = TransitionParallel(c_x=c_msa, parallel_group=parallel_group)

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair, _outgoing=True
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair, _outgoing=False
        )
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )

        self.pair_transition = Transition(c_x=c_pair)

    def _forward_pair_transition_chunked(self, pair: torch.Tensor) -> torch.Tensor:
        forward_chunked = getattr(self.pair_transition, "forward_chunked", None)
        if forward_chunked is None:
            return self.pair_transition(pair)
        return forward_chunked(pair)

    def forward(
        self,
        msa: torch.Tensor,
        pair: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        parallel_spec: Optional[ParallelSpec] = None,
        _pair_owner: Optional[list[torch.Tensor]] = None,
    ):
        # Use the same ownership handoff as PairformerBlock.
        if _pair_owner is not None:
            if (parallel_spec is None or parallel_spec.world_size <= 1 or self.training
                    or torch.is_grad_enabled() or pair is not None or len(_pair_owner) != 1):
                raise ValueError("MSA Pair handoff requires one owned Pair in distributed inference")
            pair = _pair_owner.pop()
        if parallel_spec is None or parallel_spec.world_size == 1:
            if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
                pair += self.outer_product_mean(
                    msa,
                    msa_mask,
                    parallel_spec=None,
                )
            else:
                self.outer_product_mean.add_residual_stage12_d_chunked_(
                    msa,
                    msa_mask,
                    pair,
                    parallel_spec=None,
                )

            msa += self.msa_attention1(msa, msa_mask, pair, parallel_spec=None)
            msa += self.msa_transition(msa)

            pair = self.triangle_multiplication_outgoing(pair, mask=pair_mask)
            pair = self.triangle_multiplication_incoming(pair, mask=pair_mask)
            pair += self.pair_attention1(pair, mask_local=pair_mask, parallel_spec=parallel_spec)
            pair += self.pair_attention2(pair, mask_local=pair_mask, parallel_spec=parallel_spec)
            if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
                pair += self.pair_transition(pair)
            else:
                _add_full_pair_transition_residual_(
                    self.pair_transition,
                    pair,
                )

            return msa, pair

        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            pair += self.outer_product_mean(
                msa,
                msa_mask,
                parallel_spec=parallel_spec,
            )
        else:
            self.outer_product_mean.add_residual_stage12_d_chunked_(
                msa,
                msa_mask,
                pair,
                parallel_spec=parallel_spec,
            )

        msa += self.msa_attention1(msa, msa_mask, pair, parallel_spec=parallel_spec)
        msa += self.msa_transition(msa)

        pair = self.triangle_multiplication_outgoing(
            pair, mask=pair_mask, parallel_spec=parallel_spec
        )

        pair_col = transpose_col(pair, parallel_spec)
        # Drop only the old reference; stream tracking protects pending reads.
        del pair
        pair_mask_col = pair_padding_mask_col_from_row(
            pair_mask,
            parallel_spec,
        )
        pair_col = self.triangle_multiplication_incoming(
            pair_col, mask=pair_mask_col, parallel_spec=parallel_spec
        )

        pair = transpose_row(pair_col, parallel_spec)
        # Only pair-padding contracts use a static FA plan; other masks are checked.
        pair += self.pair_attention1(
            pair,
            mask_local=pair_mask,
            parallel_spec=parallel_spec,
            fa_mask_contract="pair_padding",
        )

        if hasattr(self.pair_attention2, "forward_add_"):
            self.pair_attention2.forward_add_(
                pair,
                mask_local=pair_mask,
                parallel_spec=parallel_spec,
                fa_mask_contract="pair_padding",
            )
        else:
            pair += self.pair_attention2(
                pair,
                mask_local=pair_mask,
                parallel_spec=parallel_spec,
                fa_mask_contract="pair_padding",
            )
        if not inference_chunking(training=self.training, grad_enabled=torch.is_grad_enabled()):
            pair += self._forward_pair_transition_chunked(pair)
        else:
            self.pair_transition.add_residual_row_streaming_(pair)

        return msa, pair
