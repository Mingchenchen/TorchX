import math
from typing import Any, Optional

import torch
import torch.nn as nn

from torchfold import feat_batch, features
from torchfold.runtime_policy import (
    implicit_pair_enabled,
    large_pair_offload,
    confidence_pair_offload,
    with_offload_policy,
    RECYCLE_PAIR_CPU_OFFLOAD_ROW_CHUNK_SIZE,
    multi_card_sample_parallel,
)
from torchfold.nn.confidence_utils import (
    inference_output_offload_enabled,
    offload_confidence_output,
    stack_confidence_outputs,
)
from . import fastnn
from .nn import featurization
from .nn.pairformer import EvoformerBlock, PairformerBlock
from .nn.head import DistogramHead, ConfidenceHead
from .nn.template import TemplateEmbedding
from .nn import atom_cross_attention
from .nn import diffusion_head
from tqdm import tqdm, trange

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from ..parallel_ops import (
    ParallelSpec,
    build_parallel_spec,
    get_parallel_groups,
    pad_to_length,
    make_pair_mask_row,
    trim_to_global_length,
    gather_row_sharded_square,
)
from ..parallel_config import (
    SHORT_DIFFUSION_THRESHOLD,
)


def _sample_batched_denoising_random_inputs(
    *,
    num_steps: int,
    positions_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample diffusion randomness in the reference batched call order."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if len(positions_shape) != 4 or positions_shape[-1] != 3:
        raise ValueError(
            "positions_shape must be [samples, residues, atoms, 3], got "
            f"{positions_shape}"
        )

    num_samples = positions_shape[0]
    rotations = []
    translations = []
    noises = []
    for _ in range(num_steps):
        rotations.append(
            diffusion_head.random_rotation(
                device=device,
                dtype=dtype,
                batch_size=num_samples,
            )
        )
        translations.append(
            torch.randn(
                size=(num_samples, 1, 1, 3),
                dtype=dtype,
                device=device,
            )
        )
        noises.append(
            torch.randn(
                size=positions_shape,
                dtype=dtype,
                device=device,
            )
        )

    return (
        torch.stack(rotations, dim=0),
        torch.stack(translations, dim=0),
        torch.stack(noises, dim=0),
    )


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


def _group_rank_zero_src(group) -> int:
    return _group_rank_to_global(group, 0)


STICKY_SCHEDULE_STATE_BUDGET = 250_000


class _StickyScheduleBudgetExceeded(RuntimeError):
    pass


def _build_cyclic_relay_matrix(
    *,
    relay_start: int,
    relay_sample_count: int,
    total_steps: int,
    chunk_steps: int,
    num_ranks: int,
) -> list[list[Any]]:
    """Build a deterministic, slot-aligned relay schedule."""
    chunks_per_sample = total_steps // chunk_steps
    next_chunk = [0] * relay_sample_count
    relay_matrix: list[list[Any]] = [[] for _ in range(num_ranks)]
    sample_cursor = 0
    slot = 0
    while any(chunk_idx < chunks_per_sample for chunk_idx in next_chunk):
        available = [
            (sample_cursor + offset) % relay_sample_count
            for offset in range(relay_sample_count)
            if next_chunk[(sample_cursor + offset) % relay_sample_count]
            < chunks_per_sample
        ]
        selected = available[:num_ranks]
        if not selected:
            raise RuntimeError("Sample-parallel scheduler made no progress")

        slot_tasks: list[Any] = [None] * num_ranks
        for task_idx, relay_sample in enumerate(selected):
            group_rank = (slot + task_idx) % num_ranks
            chunk_idx = next_chunk[relay_sample]
            slot_tasks[group_rank] = (
                relay_start + relay_sample,
                chunk_idx * chunk_steps,
                (chunk_idx + 1) * chunk_steps,
            )
            next_chunk[relay_sample] += 1

        for group_rank in range(num_ranks):
            relay_matrix[group_rank].append(slot_tasks[group_rank])
        sample_cursor = (selected[-1] + 1) % relay_sample_count
        slot += 1

    return relay_matrix


def generate_balanced_sample_pipeline(
    num_samples: int,
    num_ranks: int,
    total_steps: int,
) -> list[list[Any]]:
    """Build a slot-aligned sample/step schedule for dense Diffusion.

    Entries ``0..num_ranks-1`` contain one task per slot for each group rank.
    The final entry contains the position broadcasts required after each slot.
    A task is ``(sample_id, begin_step, end_step)`` or ``None`` for an idle rank.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1, got {num_ranks}")
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")

    schedule: list[list[Any]] = [[] for _ in range(num_ranks + 1)]

    if num_ranks == 1:
        schedule[0] = [
            (sample_id, 0, total_steps) for sample_id in range(num_samples)
        ]
        schedule[1] = [((0, sample_id),) for sample_id in range(num_samples)]
        return schedule

    if num_samples < num_ranks:
        for group_rank in range(num_ranks):
            task = (
                (group_rank, 0, total_steps)
                if group_rank < num_samples
                else None
            )
            schedule[group_rank].append(task)
        schedule[num_ranks].append(
            tuple((group_rank, group_rank) for group_rank in range(num_samples))
        )
        return schedule

    if num_samples % num_ranks == 0:
        samples_per_rank = num_samples // num_ranks
        for group_rank in range(num_ranks):
            for local_idx in range(samples_per_rank):
                sample_id = group_rank * samples_per_rank + local_idx
                schedule[group_rank].append((sample_id, 0, total_steps))

        for slot in range(samples_per_rank):
            schedule[num_ranks].append(
                tuple(
                    (group_rank, schedule[group_rank][slot][0])
                    for group_rank in range(num_ranks)
                )
            )
        return schedule

    # Assign whole samples, then balance remaining steps with a sticky schedule
    # to minimize sample transfers.
    complete_samples_per_rank = max(0, num_samples // num_ranks - 1)
    complete_sample_count = complete_samples_per_rank * num_ranks
    for group_rank in range(num_ranks):
        for local_idx in range(complete_samples_per_rank):
            sample_id = group_rank * complete_samples_per_rank + local_idx
            schedule[group_rank].append((sample_id, 0, total_steps))

    relay_sample_count = num_samples - complete_sample_count
    relay_start = complete_sample_count
    relay_work = total_steps * relay_sample_count
    relay_matrix: list[list[Any]]
    if relay_work % num_ranks == 0:
        steps_per_rank = relay_work // num_ranks
        chunk_steps = math.gcd(total_steps, steps_per_rank)
        chunks_per_sample = total_steps // chunk_steps
        relay_slots = relay_sample_count * chunks_per_sample // num_ranks
        relay_matrix = [[None] * relay_slots for _ in range(num_ranks)]
        next_chunk = [0] * relay_sample_count
        sample_last_slot = [-1] * relay_sample_count
        sticky_state_count = 0

        def place_sticky(slot: int, group_rank: int) -> bool:
            nonlocal sticky_state_count
            sticky_state_count += 1
            if sticky_state_count > STICKY_SCHEDULE_STATE_BUDGET:
                raise _StickyScheduleBudgetExceeded
            if slot == relay_slots:
                return True
            if group_rank == num_ranks:
                return place_sticky(slot + 1, 0)

            for relay_sample in range(relay_sample_count):
                if (
                    next_chunk[relay_sample] >= chunks_per_sample
                    or sample_last_slot[relay_sample] >= slot
                ):
                    continue
                chunk_idx = next_chunk[relay_sample]
                relay_matrix[group_rank][slot] = (
                    relay_start + relay_sample,
                    chunk_idx * chunk_steps,
                    (chunk_idx + 1) * chunk_steps,
                )
                previous_slot = sample_last_slot[relay_sample]
                next_chunk[relay_sample] += 1
                sample_last_slot[relay_sample] = slot

                if place_sticky(slot, group_rank + 1):
                    return True

                next_chunk[relay_sample] -= 1
                sample_last_slot[relay_sample] = previous_slot
                relay_matrix[group_rank][slot] = None
            return False

        try:
            if not place_sticky(0, 0):
                raise RuntimeError(
                    "Unable to construct sticky sample-parallel schedule: "
                    f"samples={num_samples} ranks={num_ranks} "
                    f"steps={total_steps}"
                )
        except _StickyScheduleBudgetExceeded:
            relay_matrix = _build_cyclic_relay_matrix(
                relay_start=relay_start,
                relay_sample_count=relay_sample_count,
                total_steps=total_steps,
                chunk_steps=chunk_steps,
                num_ranks=num_ranks,
            )
    else:
        # Fall back to rotation when equal per-rank step counts are impossible.
        steps_per_rank = relay_work // num_ranks
        chunk_steps = math.gcd(total_steps, steps_per_rank)
        if chunk_steps < 1:
            raise RuntimeError(
                "Unable to construct sample-parallel schedule: "
                f"samples={num_samples} ranks={num_ranks} steps={total_steps}"
            )
        relay_matrix = _build_cyclic_relay_matrix(
            relay_start=relay_start,
            relay_sample_count=relay_sample_count,
            total_steps=total_steps,
            chunk_steps=chunk_steps,
            num_ranks=num_ranks,
        )

    for group_rank in range(num_ranks):
        schedule[group_rank].extend(relay_matrix[group_rank])

    total_slots = len(schedule[0])
    broadcasts: list[list[tuple[int, int]]] = [
        [] for _ in range(total_slots)
    ]
    for sample_id in range(num_samples):
        history: list[tuple[int, int]] = []
        for slot in range(total_slots):
            for group_rank in range(num_ranks):
                task = schedule[group_rank][slot]
                if task is not None and task[0] == sample_id:
                    history.append((slot, group_rank))

        if not history:
            raise RuntimeError(f"Sample {sample_id} has no scheduled Diffusion work")

        segments: list[list[tuple[int, int]]] = [[history[0]]]
        for task_slot, owner in history[1:]:
            previous_slot, previous_owner = segments[-1][-1]
            if owner == previous_owner and task_slot == previous_slot + 1:
                segments[-1].append((task_slot, owner))
            else:
                segments.append([(task_slot, owner)])

        for segment_idx, segment in enumerate(segments):
            owner = segment[0][1]
            end_slot = segment[-1][0]
            is_final_segment = segment_idx == len(segments) - 1
            if is_final_segment:
                if end_slot == total_slots - 1 or sample_id < complete_sample_count:
                    broadcast_slot = total_slots - 1
                else:
                    broadcast_slot = (
                        end_slot if end_slot % 2 == 1 else total_slots - 1
                    )
            else:
                next_start_slot = segments[segment_idx + 1][0][0]
                broadcast_slot = (
                    end_slot if end_slot % 2 == 1 else next_start_slot - 1
                )
            broadcasts[broadcast_slot].append((owner, sample_id))

    schedule[num_ranks] = [
        tuple(sorted(row, key=lambda item: item[1])) for row in broadcasts
    ]
    if any(len(rank_schedule) != total_slots for rank_schedule in schedule):
        raise RuntimeError("Sample-parallel schedule has inconsistent slot counts")
    return schedule


class Evoformer(nn.Module):
    def __init__(self, msa_channel: int = 64):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024

        self.seq_channel = 384
        self.pair_channel = 128
        self.c_target_feat = 447

        self.left_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)
        self.right_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)

        self.prev_embedding_layer_norm = fastnn.LayerNorm(self.pair_channel)
        self.prev_embedding = nn.Linear(
            self.pair_channel, self.pair_channel, bias=False)

        self.c_rel_feat = 139
        self.position_activations = nn.Linear(
            self.c_rel_feat, self.pair_channel, bias=False)

        self.bond_embedding = nn.Linear(
            1, self.pair_channel, bias=False)

        self.template_embedding = TemplateEmbedding(
            pair_channel=self.pair_channel)

        self.msa_activations = nn.Linear(34, self.msa_channel, bias=False)
        self.extra_msa_target_feat = nn.Linear(
            self.c_target_feat, self.msa_channel, bias=False)
        self.msa_stack = nn.ModuleList(
            [EvoformerBlock() for _ in range(self.msa_stack_num_layer)])

        self.single_activations = nn.Linear(
            self.c_target_feat, self.seq_channel, bias=False)

        self.prev_single_embedding_layer_norm = fastnn.LayerNorm(self.seq_channel)
        self.prev_single_embedding = nn.Linear(
            self.seq_channel, self.seq_channel, bias=False)

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock(with_single=True) for _ in range(self.pairformer_num_layer)])

    def _relative_encoding(
        self,
        batch: feat_batch.Batch,
        pair_activations_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        return featurization.add_relative_encoding_to_pair_row_shard(
            pair_activations_row=pair_activations_row,
            projection=self.position_activations,
            seq_features=batch.token_features,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
            parallel_spec=parallel_spec,
        )

    def _seq_pair_embedding(
        self,
        token_features: features.TokenFeatures,
        target_feat: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate row-sharded pair embedding from sequence."""
        left_single = self.left_single(target_feat)   # [Lp, C]
        right_single = self.right_single(target_feat) # [Lp, C]

        left_single_local = left_single[parallel_spec.start:parallel_spec.end]  # [Ls, C]
        pair_activations = left_single_local.unsqueeze(1) + right_single.unsqueeze(0)

        pair_mask = make_pair_mask_row(token_features.mask.to(dtype=left_single.dtype), parallel_spec)
        return pair_activations, pair_mask

    def _embed_bonds(
        self,
        batch: feat_batch.Batch,
        pair_activations_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        """Embeds local-row bond features and merges into pair activations."""
        contact_matrix_row = torch.zeros(
            (parallel_spec.shard_size, parallel_spec.n_padded),
            dtype=pair_activations_row.dtype,
            device=pair_activations_row.device,
        )

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).to(
                dtype=gather_idxs_polymer_ligand.dtype).unsqueeze(-1)
        )
        gather_idxs_polymer_ligand = (
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        )

        tokens_to_ligand_ligand_bonds = (
            batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
        )
        gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
            dim=1
        ).to(dtype=gather_idxs_ligand_ligand.dtype).unsqueeze(-1)
        gather_idxs_ligand_ligand = (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )

        gather_idxs = torch.concatenate(
            [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
        )

        if gather_idxs.numel() > 0:
            row_idx = gather_idxs[:, 0]
            col_idx = gather_idxs[:, 1]

            valid = (
                (row_idx >= parallel_spec.start)
                & (row_idx < parallel_spec.end)
                & (col_idx >= 0)
                & (col_idx < parallel_spec.n_global)
            )

            if torch.any(valid):
                row_local = row_idx[valid] - parallel_spec.start
                col_global = col_idx[valid]
                contact_matrix_row[row_local, col_global] = 1.0

        if parallel_spec.start == 0:
            contact_matrix_row[0, 0] = 0.0

        bonds_act = self.bond_embedding(contact_matrix_row.unsqueeze(-1))
        return pair_activations_row + bonds_act

    def _embed_template_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations_row: torch.Tensor,
        pair_mask_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        """Embeds templates and merges into row-sharded pair activations."""
        templates = batch.templates
        asym_id_full = pad_to_length(
            batch.token_features.asym_id, dim=0, length=parallel_spec.n_padded, value=0
        )

        dtype = pair_activations_row.dtype
        asym_id_local = asym_id_full[parallel_spec.start:parallel_spec.end]
        multichain_mask_row = (asym_id_local.unsqueeze(1) == asym_id_full.unsqueeze(0)).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding_row=pair_activations_row,
            templates=templates,
            multichain_mask_row=multichain_mask_row,
            padding_mask_row=pair_mask_row,
            parallel_spec=parallel_spec,
        )

        return pair_activations_row + template_act

    def _embed_process_msa(
        self,
        msa_batch: features.MSA,
        pair_activations_row: Optional[torch.Tensor],
        pair_mask_row: torch.Tensor,
        target_feat: torch.Tensor,
        parallel_spec: ParallelSpec,
        _pair_owner: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Batch-3 layout contract:
        - MSA activations stay fully replicated on every rank.
        - Pair activations stay row-sharded as [Ls, Lp, C].
        - No extra sequence chunking is introduced in this path.
        """
        # Consume ownership so the caller does not retain the entry Pair.
        consume_pair = parallel_spec.world_size > 1 and not self.training and not torch.is_grad_enabled()
        if _pair_owner is not None:
            if not consume_pair or pair_activations_row is not None or len(_pair_owner) != 1:
                raise ValueError("MSA Pair handoff requires one owned Pair in distributed inference")
            pair_activations_row = _pair_owner.pop()
        dtype = pair_activations_row.dtype

        msa_batch = featurization.shuffle_msa(
            msa_batch,
            synchronize_across_ranks=(parallel_spec.world_size > 1),
        )

        msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_mask = pad_to_length(
            msa_mask,
            dim=1,
            length=parallel_spec.n_padded,
            value=0.0,
        )

        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)
        msa_feat = pad_to_length(
            msa_feat,
            dim=1,
            length=parallel_spec.n_padded,
            value=0.0,
        )

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat).unsqueeze(0)

        for msa_block in tqdm(self.msa_stack, desc="MSA stack"):
            if consume_pair:
                pair_owner = [pair_activations_row]
                del pair_activations_row
                msa_activations, pair_activations_row = msa_block(
                    msa=msa_activations, pair=None, _pair_owner=pair_owner,
                    msa_mask=msa_mask, pair_mask=pair_mask_row,
                    parallel_spec=parallel_spec,
                )
                continue
            msa_activations, pair_activations_row = msa_block(
                msa=msa_activations,
                pair=pair_activations_row,
                msa_mask=msa_mask,
                pair_mask=pair_mask_row,
                parallel_spec=parallel_spec,
            )

        return pair_activations_row

    def forward(
        self,
        batch: feat_batch.Batch,
        prev: dict[str, torch.Tensor],
        target_feat: torch.Tensor,
        idx: int = 0,
        parallel_spec: ParallelSpec = None,
    ) -> dict[str, torch.Tensor]:
        if parallel_spec is None:
            raise ValueError("Evoformer.forward requires parallel_spec in Parallel version.")

        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat, parallel_spec
        )

        pair_activations = self._add_prev_pair_embedding(
            pair_activations,
            prev['pair'],
        )
        # Release the previous Pair after enqueueing its embedding.
        del prev['pair']

        pair_activations = self._relative_encoding(batch, pair_activations, parallel_spec)

        pair_activations = self._embed_bonds(
            batch=batch,
            pair_activations_row=pair_activations,
            parallel_spec=parallel_spec,
        )

        pair_activations = self._embed_template_pair(
            batch=batch,
            pair_activations_row=pair_activations,
            pair_mask_row=pair_mask,
            parallel_spec=parallel_spec,
        )

        if parallel_spec.world_size > 1 and not self.training and not torch.is_grad_enabled():
            # Transfer ownership to avoid retaining the entry Pair across MSA blocks.
            pair_owner = [pair_activations]
            del pair_activations
            pair_activations = self._embed_process_msa(
                msa_batch=batch.msa, pair_activations_row=None, _pair_owner=pair_owner,
                pair_mask_row=pair_mask, target_feat=target_feat,
                parallel_spec=parallel_spec,
            )
        else:
            pair_activations = self._embed_process_msa(
                msa_batch=batch.msa,
                pair_activations_row=pair_activations,
                pair_mask_row=pair_mask,
                target_feat=target_feat,
                parallel_spec=parallel_spec,
            )

        single_activations = self.single_activations(target_feat)
        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single'])
        )

        seq_mask = pad_to_length(
            batch.token_features.mask,
            dim=0,
            length=parallel_spec.n_padded,
            value=0,
        )

        for pairformer_b in tqdm(self.trunk_pairformer, desc=f"Pairformer {idx}"):
            if parallel_spec.world_size > 1 and not self.training and not torch.is_grad_enabled():
                # Release the old row before incoming; Single uses the restored row.
                pair_owner = [pair_activations]
                del pair_activations
                pair_activations, single_activations = pairformer_b(
                    pair_row=None,
                    pair_mask_row=pair_mask,
                    single=single_activations,
                    seq_mask=seq_mask,
                    parallel_spec=parallel_spec,
                    _pair_owner=pair_owner,
                )
            else:
                pair_activations, single_activations = pairformer_b(
                    pair_row=pair_activations,
                    pair_mask_row=pair_mask,
                    single=single_activations,
                    seq_mask=seq_mask,
                    parallel_spec=parallel_spec,
                )

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
            'parallel_spec': parallel_spec,
        }

        return output

    def _add_prev_pair_embedding_streaming(
        self,
        pair_activations: torch.Tensor,
        previous_pair_cpu: torch.Tensor,
        row_chunk_size: int = RECYCLE_PAIR_CPU_OFFLOAD_ROW_CHUNK_SIZE,
    ) -> torch.Tensor:
        if row_chunk_size < 1:
            raise ValueError(
                f"recycle Pair row chunk size must be positive, got {row_chunk_size}"
            )
        if tuple(previous_pair_cpu.shape) != tuple(pair_activations.shape):
            raise ValueError(
                "recycle Pair shapes do not match: "
                f"previous={tuple(previous_pair_cpu.shape)} "
                f"current={tuple(pair_activations.shape)}"
            )

        for row_start in range(0, pair_activations.shape[0], row_chunk_size):
            row_end = min(
                row_start + row_chunk_size,
                pair_activations.shape[0],
            )
            previous_chunk = previous_pair_cpu[row_start:row_end].to(
                device=pair_activations.device,
            )
            previous_embedding = self.prev_embedding(
                self.prev_embedding_layer_norm(previous_chunk)
            )
            pair_activations[row_start:row_end].add_(previous_embedding)
            if pair_activations.device.type == "npu":
                torch.npu.synchronize()
            del previous_chunk, previous_embedding
        return pair_activations

    def _add_prev_pair_embedding(
        self,
        pair_activations: torch.Tensor,
        previous_pair: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if previous_pair is None:
            implicit_zero = pair_activations.new_zeros(
                (1, 1, pair_activations.shape[-1])
            )
            initial_embedding = self.prev_embedding(
                self.prev_embedding_layer_norm(implicit_zero)
            )
            pair_activations.add_(initial_embedding)
            return pair_activations

        if previous_pair.device == pair_activations.device:
            pair_activations.add_(
                self.prev_embedding(
                    self.prev_embedding_layer_norm(previous_pair)
                )
            )
            return pair_activations

        if previous_pair.device.type != "cpu":
            raise ValueError(
                "offloaded recycle Pair must be on CPU, got "
                f"device={previous_pair.device}"
            )
        return self._add_prev_pair_embedding_streaming(
            pair_activations,
            previous_pair,
        )


class TorchFold(nn.Module):
    def __init__(
        self,
        num_recycles: int = 10,
        num_samples: int = 5,
        diffusion_steps: int = 200,
        diffusion_sample_parallel: bool = True,
    ):
        super().__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.diffusion_sample_parallel = diffusion_sample_parallel

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()

        self.evoformer = Evoformer()

        self.diffusion_head = diffusion_head.DiffusionHead()

        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()

    def create_target_feat_embedding(self, batch: feat_batch.Batch) -> torch.Tensor:
        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
        )

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
            clear_flag=True,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat

    def _gather_full_pair(
        self,
        pair_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        pair_full = gather_row_sharded_square(pair_row, parallel_spec)
        pair_full = pair_full[:parallel_spec.n_global, :parallel_spec.n_global]
        return pair_full.contiguous()

    def _gather_full_pair_to_rank0(
        self,
        pair_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor | None:
        """Gather row-sharded pair activations only onto parallel rank 0.

        HCCL implements ``dist.gather`` with ``all_gather``, which would still
        materialize the full pair tensor on every rank. Use blocking point-to-
        point transfers instead. Rank 0 writes each received shard directly
        into the final unpadded tensor, avoiding per-rank full tensors and a
        temporary chunks-plus-cat copy on rank 0.
        """
        if not parallel_spec.is_distributed:
            return self._gather_full_pair(pair_row, parallel_spec)

        if dist is None or not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Rank0-only pair gather requires initialized torch.distributed."
            )

        if pair_row.ndim < 2:
            raise ValueError(
                "Row-sharded pair tensor must have at least 2 dimensions, "
                f"got shape={tuple(pair_row.shape)}"
            )
        if pair_row.shape[0] < parallel_spec.shard_size:
            raise ValueError(
                "Row-sharded pair tensor has fewer local rows than expected: "
                f"shape={tuple(pair_row.shape)} shard_size={parallel_spec.shard_size}"
            )
        if pair_row.shape[1] < parallel_spec.n_global:
            raise ValueError(
                "Row-sharded pair tensor has fewer global columns than expected: "
                f"shape={tuple(pair_row.shape)} n_global={parallel_spec.n_global}"
            )

        n_global = parallel_spec.n_global
        local_valid = parallel_spec.local_valid_size
        dst_global_rank = _group_rank_zero_src(parallel_spec.group)

        if parallel_spec.rank != 0:
            if local_valid > 0:
                local_pair = pair_row[:local_valid, :n_global].contiguous()
                dist.send(
                    local_pair,
                    dst=dst_global_rank,
                    group=parallel_spec.group,
                )
            return None

        pair_full = pair_row.new_empty(
            (n_global, n_global, *pair_row.shape[2:])
        )
        if local_valid > 0:
            pair_full[
                parallel_spec.start:parallel_spec.start + local_valid
            ].copy_(pair_row[:local_valid, :n_global])

        for group_rank in range(1, parallel_spec.world_size):
            start = group_rank * parallel_spec.shard_size
            end = min(start + parallel_spec.shard_size, n_global)
            if start >= end:
                continue

            src_global_rank = _group_rank_to_global(
                parallel_spec.group,
                group_rank,
            )
            dist.recv(
                pair_full[start:end],
                src=src_global_rank,
                group=parallel_spec.group,
            )

        return pair_full

    def _broadcast_full_pair_from_rank0(
        self,
        pair_full_rank0: torch.Tensor | None,
        pair_row: torch.Tensor,
        parallel_spec: ParallelSpec,
    ) -> torch.Tensor:
        """Broadcast rank0's dense Pair tensor to the sample-parallel ranks."""
        if not parallel_spec.is_distributed:
            if pair_full_rank0 is None:
                raise RuntimeError("Dense Pair tensor is missing in single-rank mode")
            return pair_full_rank0.contiguous()

        if dist is None or not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Short-sequence sample parallelism requires initialized "
                "torch.distributed."
            )

        n_global = parallel_spec.n_global
        expected_shape = (n_global, n_global, *pair_row.shape[2:])
        if parallel_spec.rank == 0:
            if pair_full_rank0 is None:
                raise RuntimeError("Rank0 has no dense Pair tensor to broadcast")
            if tuple(pair_full_rank0.shape) != expected_shape:
                raise RuntimeError(
                    "Unexpected dense Pair shape on rank0: "
                    f"actual={tuple(pair_full_rank0.shape)} "
                    f"expected={expected_shape}"
                )
            pair_full = pair_full_rank0.contiguous()
        else:
            if pair_full_rank0 is not None:
                raise RuntimeError(
                    "Non-root rank unexpectedly materialized a dense Pair tensor"
                )
            pair_full = pair_row.new_empty(expected_shape)

        src = _group_rank_zero_src(parallel_spec.group)
        # Chunk broadcasts to limit individual HCCL operation sizes.
        chunk_rows = max(1, parallel_spec.shard_size)
        for start in range(0, n_global, chunk_rows):
            end = min(start + chunk_rows, n_global)
            dist.broadcast(
                pair_full[start:end],
                src=src,
                group=parallel_spec.group,
            )
        return pair_full

    def _materialize_diffusion_embeddings(
        self,
        embeddings: dict[str, torch.Tensor],
        parallel_spec: ParallelSpec,
    ) -> dict[str, torch.Tensor]:
        single_full = trim_to_global_length(embeddings['single'], parallel_spec, dim=0)
        target_feat_full = trim_to_global_length(embeddings['target_feat'], parallel_spec, dim=0)

        return {
            'pair': embeddings['pair'].contiguous(),
            'single': single_full.contiguous(),
            'target_feat': target_feat_full.contiguous(),
            'parallel_spec': parallel_spec,
        }

    @staticmethod
    def _stage_distributed_pair_cpu_offload(
        owner_embeddings: dict[str, torch.Tensor],
        diffusion_embeddings: dict[str, torch.Tensor],
    ) -> None:
        pair_cpu = diffusion_embeddings['pair'].cpu()
        owner_embeddings['pair'] = pair_cpu
        diffusion_embeddings[diffusion_head.PAIR_CPU_OFFLOAD_KEY] = pair_cpu

    @staticmethod
    def _restore_distributed_pair_after_diffusion(
        embeddings: dict[str, torch.Tensor],
    ) -> None:
        pair_cpu = embeddings['pair']
        if pair_cpu.device.type != 'cpu':
            return

        target_device = embeddings['single'].device
        if target_device.type == 'npu':
            torch.npu.synchronize()
            torch.npu.empty_cache()
        embeddings['pair'] = pair_cpu.to(device=target_device)

    def _materialize_full_embeddings_for_heads(
        self,
        embeddings: dict[str, torch.Tensor],
        parallel_spec: ParallelSpec,
        pair_full: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pair_full is None:
            pair_full = self._gather_full_pair(embeddings['pair'], parallel_spec)

        single_full = trim_to_global_length(embeddings['single'], parallel_spec, dim=0)
        target_feat_full = trim_to_global_length(embeddings['target_feat'], parallel_spec, dim=0)

        head_parallel_spec = ParallelSpec(
            group=None,
            rank=0,
            world_size=1,
            n_global=parallel_spec.n_global,
            n_padded=parallel_spec.n_global,
            shard_size=parallel_spec.n_global,
            start=0,
            end=parallel_spec.n_global,
        )

        return {
            'pair': pair_full.contiguous(),
            'single': single_full.contiguous(),
            'target_feat': target_feat_full.contiguous(),
            'parallel_spec': head_parallel_spec,
        }

    def _apply_denoising_step(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_level_prev: torch.Tensor,
        mask: torch.Tensor,
        noise_level: torch.Tensor,
        clear_flag: bool,
        augmentation_rotation: torch.Tensor | None = None,
        augmentation_translation: torch.Tensor | None = None,
        diffusion_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if augmentation_rotation is None and augmentation_translation is None:
            positions = diffusion_head.random_augmentation(
                positions=positions,
                mask=mask,
            )
        else:
            positions = diffusion_head.random_augmentation(
                positions=positions,
                mask=mask,
                rotation=augmentation_rotation,
                translation=augmentation_translation,
            )

        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        if diffusion_noise is None:
            diffusion_noise = torch.randn(
                size=positions.shape,
                dtype=positions.dtype,
                device=noise_scale.device,
            )
        elif tuple(diffusion_noise.shape) != tuple(positions.shape):
            raise ValueError(
                "Unexpected diffusion noise shape: "
                f"actual={tuple(diffusion_noise.shape)} "
                f"expected={tuple(positions.shape)}"
            )
        noise = noise_scale * diffusion_noise
        positions_noisy = positions + noise

        positions_denoised = self.diffusion_head(
            positions_noisy=positions_noisy,
            noise_level=t_hat,
            batch=batch,
            embeddings=embeddings,
            use_conditioning=True,
            clear_flag=clear_flag,
        )
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad

        return positions_out

    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        mask = batch.predicted_structure_info.atom_mask
        num_samples = self.num_samples

        device = mask.device

        diffusion_steps = self.diffusion_steps

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, diffusion_steps + 1, device=device))

        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device)
        positions *= noise_levels[0]

        clear_flag = False
        for sample_idx in range(num_samples):
            for step_idx in trange(
                diffusion_steps,
                desc=f"Diffusion {sample_idx}",
            ):
                if (
                    sample_idx == num_samples - 1
                    and step_idx == diffusion_steps - 1
                ):
                    clear_flag = True
                positions[sample_idx] = self._apply_denoising_step(
                    batch,
                    embeddings,
                    positions[sample_idx],
                    noise_levels[step_idx],
                    mask,
                    noise_levels[step_idx + 1],
                    clear_flag,
                )

        final_dense_atom_mask = torch.tile(
            mask.unsqueeze(0),
            (num_samples, 1, 1),
        )

        return {'atom_positions': positions, 'mask': final_dense_atom_mask}

    def _sample_diffusion_parallel(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        parallel_spec: ParallelSpec,
    ) -> dict[str, torch.Tensor]:
        """Run configurable dense Diffusion samples across ranks."""
        if not parallel_spec.is_distributed:
            return self._sample_diffusion(batch, embeddings)
        if dist is None or not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Sample-parallel Diffusion requires initialized torch.distributed."
            )

        mask = batch.predicted_structure_info.atom_mask
        device = mask.device
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device)
        )

        positions_shape = (self.num_samples,) + tuple(mask.shape) + (3,)
        if parallel_spec.rank == 0:
            positions = torch.randn(positions_shape, device=device)
            positions *= noise_levels[0]
        else:
            positions = torch.empty(positions_shape, device=device)

        root_global_rank = _group_rank_zero_src(parallel_spec.group)
        dist.broadcast(
            positions,
            src=root_global_rank,
            group=parallel_spec.group,
        )

        if parallel_spec.rank == 0:
            rotations, translations, noises = (
                _sample_batched_denoising_random_inputs(
                    num_steps=self.diffusion_steps,
                    positions_shape=positions_shape,
                    device=device,
                    dtype=positions.dtype,
                )
            )
        else:
            rotations = positions.new_empty(
                (self.diffusion_steps, self.num_samples, 3, 3)
            )
            translations = positions.new_empty(
                (self.diffusion_steps, self.num_samples, 1, 1, 3)
            )
            noises = positions.new_empty(
                (self.diffusion_steps,) + positions_shape
            )

        for step_idx in range(self.diffusion_steps):
            dist.broadcast(
                rotations[step_idx],
                src=root_global_rank,
                group=parallel_spec.group,
            )
            dist.broadcast(
                translations[step_idx],
                src=root_global_rank,
                group=parallel_spec.group,
            )
            dist.broadcast(
                noises[step_idx],
                src=root_global_rank,
                group=parallel_spec.group,
            )

        schedule = generate_balanced_sample_pipeline(
            num_samples=self.num_samples,
            num_ranks=parallel_spec.world_size,
            total_steps=self.diffusion_steps,
        )
        rank_schedule = schedule[parallel_spec.rank]
        broadcast_schedule = schedule[parallel_spec.world_size]
        task_slots = [
            slot for slot, task in enumerate(rank_schedule) if task is not None
        ]
        last_task_slot = task_slots[-1] if task_slots else None

        for slot, task in enumerate(rank_schedule):
            if task is not None:
                sample_idx, begin_step, end_step = task
                for step_idx in trange(
                    begin_step,
                    end_step,
                    desc=f"Diffusion sample={sample_idx} rank={parallel_spec.rank}",
                ):
                    clear_flag = (
                        slot == last_task_slot and step_idx == end_step - 1
                    )
                    positions_sample = self._apply_denoising_step(
                        batch=batch,
                        embeddings=embeddings,
                        positions=positions[sample_idx:sample_idx + 1],
                        noise_level_prev=noise_levels[step_idx],
                        mask=mask.unsqueeze(0),
                        noise_level=noise_levels[step_idx + 1],
                        clear_flag=clear_flag,
                        augmentation_rotation=(
                            rotations[step_idx, sample_idx:sample_idx + 1]
                        ),
                        augmentation_translation=(
                            translations[step_idx, sample_idx:sample_idx + 1]
                        ),
                        diffusion_noise=(
                            noises[step_idx, sample_idx:sample_idx + 1]
                        ),
                    )
                    positions[sample_idx] = positions_sample[0]

            for owner_group_rank, sample_idx in broadcast_schedule[slot]:
                owner_global_rank = _group_rank_to_global(
                    parallel_spec.group,
                    owner_group_rank,
                )
                dist.broadcast(
                    positions[sample_idx],
                    src=owner_global_rank,
                    group=parallel_spec.group,
                )

        dist.barrier(group=parallel_spec.group)
        final_dense_atom_mask = torch.tile(
            mask.unsqueeze(0),
            (self.num_samples, 1, 1),
        )
        return {
            'atom_positions': positions,
            'mask': final_dense_atom_mask,
        }

    def _short_sample_parallel_mode(
        self,
        parallel_spec: ParallelSpec,
    ) -> tuple[bool, int]:
        return (
            multi_card_sample_parallel(
                enabled=self.diffusion_sample_parallel,
                is_distributed=parallel_spec.is_distributed,
                feature_length=parallel_spec.n_global,
                threshold=SHORT_DIFFUSION_THRESHOLD,
            ),
            SHORT_DIFFUSION_THRESHOLD,
        )

    def _distributed_pair_cpu_offload_enabled(
        self,
        parallel_spec: ParallelSpec,
    ) -> bool:
        return large_pair_offload(
            training=self.training, num_tokens=parallel_spec.n_global,
            world_size=parallel_spec.world_size,
        )

    def _recycle_pair_cpu_offload_enabled(
        self,
        parallel_spec: ParallelSpec,
    ) -> bool:
        return large_pair_offload(
            training=self.training, num_tokens=parallel_spec.n_global,
            world_size=parallel_spec.world_size,
        )

    def _offload_pair_for_next_recycle(
        self,
        embeddings: dict[str, torch.Tensor],
        parallel_spec: ParallelSpec,
        recycle_index: int,
    ) -> None:
        if recycle_index + 1 >= self.num_recycles:
            return
        if not self._recycle_pair_cpu_offload_enabled(parallel_spec):
            return
        pair = embeddings['pair']
        if pair.device.type == 'cpu':
            return
        embeddings['pair'] = pair.cpu()

    @with_offload_policy(distributed=True)
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res

        parallel_group = get_parallel_groups().parallel_group
        parallel_spec = build_parallel_spec(num_res, parallel_group)

        target_feat = self.create_target_feat_embedding(batch)
        target_feat = pad_to_length(
            target_feat,
            dim=0,
            length=parallel_spec.n_padded,
            value=0.0,
        )

        use_implicit_initial_pair = (
            implicit_pair_enabled(training=self.training)
            and self.num_recycles > 0
        )
        embeddings = {
            'pair': (
                None
                if use_implicit_initial_pair
                else torch.zeros(
                    [parallel_spec.shard_size, parallel_spec.n_padded, self.evoformer_pair_channel],
                    device=target_feat.device,
                    dtype=torch.float32,
                )
            ),
            'single': torch.zeros(
                [parallel_spec.n_padded, self.evoformer_seq_channel],
                dtype=torch.float32,
                device=target_feat.device,
            ),
            'target_feat': target_feat,
            'parallel_spec': parallel_spec,
        }

        num_recycles = self.num_recycles

        for i in range(num_recycles):
            embeddings = self.evoformer(
                batch=batch,
                prev=embeddings,
                target_feat=target_feat,
                idx=i,
                parallel_spec=parallel_spec,
            )
            self._offload_pair_for_next_recycle(
                embeddings,
                parallel_spec,
                recycle_index=i,
            )

        offload_confidence_pair = confidence_pair_offload(
            training=self.training, num_tokens=parallel_spec.n_global,
            world_size=parallel_spec.world_size,
        )

        short_sample_parallel, _ = (
            self._short_sample_parallel_mode(
                parallel_spec
            )
        )
        if short_sample_parallel:
            pair_full_rank0 = self._gather_full_pair_to_rank0(
                embeddings['pair'],
                parallel_spec,
            )
            diffusion_pair_full = self._broadcast_full_pair_from_rank0(
                pair_full_rank0=pair_full_rank0,
                pair_row=embeddings['pair'],
                parallel_spec=parallel_spec,
            )
            diffusion_embeddings = self._materialize_full_embeddings_for_heads(
                embeddings=embeddings,
                parallel_spec=parallel_spec,
                pair_full=diffusion_pair_full,
            )
            samples = self._sample_diffusion_parallel(
                batch=batch,
                embeddings=diffusion_embeddings,
                parallel_spec=parallel_spec,
            )
            del diffusion_embeddings
            del diffusion_pair_full
            del pair_full_rank0
        else:
            # Use row-sharded Diffusion outside sample-parallel mode.
            diffusion_embeddings = self._materialize_diffusion_embeddings(
                embeddings=embeddings,
                parallel_spec=parallel_spec,
            )
            pair_cpu_offload_enabled = (
                self._distributed_pair_cpu_offload_enabled(parallel_spec)
            )
            if pair_cpu_offload_enabled:
                self._stage_distributed_pair_cpu_offload(
                    owner_embeddings=embeddings,
                    diffusion_embeddings=diffusion_embeddings,
                )
            samples = self._sample_diffusion(batch, diffusion_embeddings)
            del diffusion_embeddings
            if pair_cpu_offload_enabled and not offload_confidence_pair:
                self._restore_distributed_pair_after_diffusion(embeddings)

        # Reuse the immutable CPU Pair; each sample creates its own working shard.
        if offload_confidence_pair:
            embeddings['pair'] = embeddings['pair'].cpu()

        confidence_output_per_sample = []
        # Offload completed samples regardless of the multi-card length threshold.
        offload_confidence_outputs = inference_output_offload_enabled(
            training=self.training
        ) or (
            parallel_spec.is_distributed
            and not self.training
            and not torch.is_grad_enabled()
        )
        materialize_outputs = (
            not offload_confidence_outputs
            or not parallel_spec.is_distributed
            or parallel_spec.rank == 0
        )
        for sample_dense_atom_position in tqdm(samples['atom_positions'], desc="Confidence"):
            sample_confidence_output = self.confidence_head(
                dense_atom_positions=sample_dense_atom_position,
                embeddings=embeddings,
                seq_mask=batch.token_features.mask,
                token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch.token_features.asym_id
            )
            if materialize_outputs:
                if offload_confidence_outputs:
                    sample_confidence_output = offload_confidence_output(
                        sample_confidence_output
                    )
                confidence_output_per_sample.append(sample_confidence_output)
            else:
                sample_confidence_output.clear()

        confidence_output = (
            stack_confidence_outputs(confidence_output_per_sample)
            if materialize_outputs
            else {}
        )

        # Restore the original Pair for Distogram.
        if offload_confidence_pair:
            embeddings['pair'] = embeddings['pair'].to(
                device=embeddings['single'].device,
            )
        distogram = self.distogram_head(batch, embeddings)

        return {
            'diffusion_samples': samples,
            'distogram': distogram,
            **confidence_output,
        }
