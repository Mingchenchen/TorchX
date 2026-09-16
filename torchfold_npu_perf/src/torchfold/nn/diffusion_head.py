
import torch
import torch.nn as nn

from torchfold import feat_batch
from torchfold.runtime_policy import (
    large_pair_offload,
    release_pair_conditioning,
    DIFFUSION_CONDITIONING_ROW_STREAM_THRESHOLD,
    DIFFUSION_CONDITIONING_ROW_STREAM_CHUNK_SIZE,
    diffusion_conditioning_row_streaming,
    inference_chunking,
)
from torchfold.nn import featurization, utils
from torchfold.nn.atom_cross_attention import AtomCrossAttEncoder, AtomCrossAttDecoder
from torchfold.nn.diffusion_transformer import DiffusionTransformer, DiffusionTransition
from torchfold.nn.layer_norm import LayerNorm

# Reference data scale for diffusion noise and coordinate normalization.
SIGMA_DATA = 16.0
SIGMA_DATA_2 = 256.0


def _create_relative_encoding_rows(
    seq_features,
    row_start: int,
    row_end: int,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Create an exact row slice of ``create_relative_encoding``."""
    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    num_tokens = token_index.shape[0]
    if not 0 <= row_start < row_end <= num_tokens:
        raise ValueError(
            "relative-encoding row range must satisfy "
            f"0 <= start < end <= {num_tokens}, got "
            f"start={row_start}, end={row_end}"
        )

    left_asym_id = asym_id[row_start:row_end].unsqueeze(1)
    right_asym_id = asym_id.unsqueeze(0)
    left_residue_index = residue_index[row_start:row_end].unsqueeze(1)
    right_residue_index = residue_index.unsqueeze(0)
    left_token_index = token_index[row_start:row_end].unsqueeze(1)
    right_token_index = token_index.unsqueeze(0)
    left_entity_id = entity_id[row_start:row_end].unsqueeze(1)
    right_entity_id = entity_id.unsqueeze(0)
    left_sym_id = sym_id[row_start:row_end].unsqueeze(1)
    right_sym_id = sym_id.unsqueeze(0)

    offset = left_residue_index - right_residue_index
    clipped_offset = torch.clip(
        offset + max_relative_idx,
        min=0,
        max=2 * max_relative_idx,
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = torch.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
    )
    rel_pos = torch.nn.functional.one_hot(
        final_offset.to(dtype=torch.int64),
        2 * max_relative_idx + 2,
    )

    token_offset = left_token_index - right_token_index
    clipped_token_offset = torch.clip(
        token_offset + max_relative_idx,
        min=0,
        max=2 * max_relative_idx,
    )
    residue_same = asym_id_same & (
        left_residue_index == right_residue_index
    )
    final_token_offset = torch.where(
        residue_same,
        clipped_token_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_token_offset),
    )
    rel_token = torch.nn.functional.one_hot(
        final_token_offset.to(dtype=torch.int64),
        2 * max_relative_idx + 2,
    )

    entity_id_same = left_entity_id == right_entity_id
    rel_entity = entity_id_same.to(dtype=rel_pos.dtype).unsqueeze(-1)

    rel_sym_id = left_sym_id - right_sym_id
    clipped_rel_chain = torch.clip(
        rel_sym_id + max_relative_chain,
        min=0,
        max=2 * max_relative_chain,
    )
    final_rel_chain = torch.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_relative_chain + 1)
        * torch.ones_like(clipped_rel_chain),
    )
    rel_chain = torch.nn.functional.one_hot(
        final_rel_chain.to(dtype=torch.int64),
        2 * max_relative_chain + 2,
    )

    return torch.concatenate(
        [rel_pos, rel_token, rel_entity, rel_chain],
        dim=-1,
    )


class FourierEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super(FourierEmbeddings, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(dim) * 0.02)
        self.bias = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not hasattr(self, "weight") or not hasattr(self, "bias"):
            raise RuntimeError("FourierEmbeddings not initialized")

        return torch.cos(2 * torch.pi * (x.unsqueeze(-1) * self.weight + self.bias))


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
    return (
        SIGMA_DATA
        * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
    )


def random_rotation(device, dtype, batch_size: int | None = None):
    # Create random rotations with Gram-Schmidt orthogonalization. The batched
    # path gives each Diffusion sample an independent rigid augmentation.
    if batch_size is None:
        v0, v1 = torch.randn(size=(2, 3), dtype=dtype, device=device)
        e0 = v0 / torch.clamp(torch.linalg.norm(v0), min=1e-10)
        v1 = v1 - e0 * torch.dot(v1, e0)
        e1 = v1 / torch.clamp(torch.linalg.norm(v1), min=1e-10)
        e2 = torch.cross(e0, e1, dim=-1)
        return torch.stack([e0, e1, e2])

    vectors = torch.randn(
        size=(batch_size, 2, 3),
        dtype=dtype,
        device=device,
    )
    v0, v1 = vectors.unbind(dim=1)
    e0 = v0 / torch.clamp(
        torch.linalg.norm(v0, dim=-1, keepdim=True),
        min=1e-10,
    )
    v1 = v1 - e0 * torch.sum(v1 * e0, dim=-1, keepdim=True)
    e1 = v1 / torch.clamp(
        torch.linalg.norm(v1, dim=-1, keepdim=True),
        min=1e-10,
    )
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=1)


def random_augmentation(
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply random rigid augmentation.

    Args:
      positions: atom positions of shape (<common_axes>, 3)
      mask: per-atom mask of shape (<common_axes>,)

    Returns:
      Transformed positions with the same shape as input positions.
    """

    center = utils.mask_mean(
        mask.unsqueeze(-1),
        positions,
        dim=(-2, -3),
        keepdim=True,
        eps=1e-6,
    )
    if positions.ndim == 4:
        if mask.ndim != 3 or positions.shape[:-1] != mask.shape:
            raise ValueError(
                "Batched positions and mask shapes are inconsistent: "
                f"positions={tuple(positions.shape)} mask={tuple(mask.shape)}"
            )
        batch_size = positions.shape[0]
        rotation = random_rotation(
            device=positions.device,
            dtype=positions.dtype,
            batch_size=batch_size,
        )
        shifted = positions - center
        shifted_flat = shifted.reshape(batch_size, -1, 3)
        augmented_positions = torch.bmm(shifted_flat, rotation).reshape_as(
            positions
        )
        translation = torch.randn(
            size=(batch_size, 1, 1, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        augmented_positions += translation
    else:
        rotation = random_rotation(
            device=positions.device,
            dtype=positions.dtype,
        )
        translation = torch.randn(
            size=(3,),
            dtype=positions.dtype,
            device=positions.device,
        )
        augmented_positions = (
            torch.einsum(
                '...i,ij->...j',
                positions - center,
                rotation,
            )
            + translation
        )
    return augmented_positions * mask.unsqueeze(-1)


class DiffusionHead(nn.Module):
    def __init__(self):
        super(DiffusionHead, self).__init__()

        self.c_act = 768
        self.pair_channel = 128
        self.seq_channel = 384

        self.c_pair_cond_initial = 267
        self.pair_cond_initial_norm = LayerNorm(
            self.c_pair_cond_initial, bias=False)
        self.pair_cond_initial_projection = nn.Linear(
            self.c_pair_cond_initial, self.pair_channel, bias=False)

        self.pair_transition_0 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)
        self.pair_transition_1 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)

        self.c_single_cond_initial = 831
        self.single_cond_initial_norm = LayerNorm(
            self.c_single_cond_initial, bias=False)
        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial, self.seq_channel, bias=False)

        self.c_noise_embedding = 256
        self.noise_embedding_initial_norm = LayerNorm(
            self.c_noise_embedding, bias=False)
        self.noise_embedding_initial_projection = nn.Linear(
            self.c_noise_embedding, self.seq_channel, bias=False)

        self.single_transition_0 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)
        self.single_transition_1 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)

        self.atom_cross_att_encoder = AtomCrossAttEncoder(per_token_channels=self.c_act,
                                                          with_token_atoms_act=True,
                                                          with_trunk_pair_cond=True,
                                                          with_trunk_single_cond=True)

        self.single_cond_embedding_norm = LayerNorm(
            self.seq_channel, bias=False)
        self.single_cond_embedding_projection = nn.Linear(
            self.seq_channel, self.c_act, bias=False)

        self.transformer = DiffusionTransformer()

        self.output_norm = LayerNorm(self.c_act, bias=False)

        self.atom_cross_att_decoder = AtomCrossAttDecoder()

        self.fourier_embeddings = FourierEmbeddings(dim=256)

        self.first_run = True
        self.single_cond = None
        self.pair_cond = None

    @staticmethod
    def _flush_conditioning_npu_queue(
        tensor: torch.Tensor,
        empty_cache: bool = False,
    ) -> None:
        """Finish one-off conditioning work before releasing its buffers."""
        if tensor.device.type != "npu":
            return
        torch.npu.synchronize()
        if empty_cache:
            torch.npu.empty_cache()

    def _build_pair_conditioning_row_streaming(
        self,
        token_features,
        pair_embedding: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int = DIFFUSION_CONDITIONING_ROW_STREAM_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Build long-sequence pair conditioning with bounded workspace."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "diffusion conditioning row streaming is inference-only"
            )
        if pair_embedding.ndim != 3:
            raise ValueError(
                "pair embedding must have shape [rows, columns, channels], "
                f"got shape={tuple(pair_embedding.shape)}"
            )
        num_rows, num_columns, pair_channels = pair_embedding.shape
        if num_rows != num_columns:
            raise ValueError(
                "single-card diffusion conditioning expects a square pair "
                f"embedding, got shape={tuple(pair_embedding.shape)}"
            )
        if pair_channels + 139 != self.c_pair_cond_initial:
            raise ValueError(
                "pair embedding has an incompatible channel count: "
                f"pair_channels={pair_channels}, "
                f"expected={self.c_pair_cond_initial - 139}"
            )
        if chunk_size <= 0:
            raise ValueError(
                "diffusion conditioning row chunk size must be positive"
            )

        # Drain pairformer work before allocating the persistent result. With
        # TASK_QUEUE_ENABLE=2, deleting Python references alone is insufficient.
        self._flush_conditioning_npu_queue(
            pair_embedding,
            empty_cache=True,
        )
        pair_cond = torch.empty(
            (num_rows, num_columns, self.pair_channel),
            dtype=pair_embedding.dtype,
            device=pair_embedding.device,
        )

        for row_start in range(0, num_rows, chunk_size):
            row_end = min(row_start + chunk_size, num_rows)
            pair_embedding_chunk = pair_embedding[row_start:row_end]
            if not use_conditioning:
                pair_embedding_chunk = pair_embedding_chunk * 0

            rel_features = _create_relative_encoding_rows(
                token_features,
                row_start=row_start,
                row_end=row_end,
                max_relative_idx=32,
                max_relative_chain=2,
            ).to(
                device=pair_embedding.device,
                dtype=pair_embedding.dtype,
            )
            features_2d = torch.concatenate(
                [pair_embedding_chunk, rel_features],
                dim=-1,
            )
            pair_cond_chunk = self.pair_cond_initial_projection(
                self.pair_cond_initial_norm(features_2d)
            )
            self.pair_transition_0.add_pair_residual_row_streaming_(
                pair_cond_chunk,
                chunk_size=row_end - row_start,
            )
            self.pair_transition_1.add_pair_residual_row_streaming_(
                pair_cond_chunk,
                chunk_size=row_end - row_start,
            )
            pair_cond[row_start:row_end].copy_(pair_cond_chunk)

            self._flush_conditioning_npu_queue(pair_cond)
            del (
                pair_embedding_chunk,
                rel_features,
                features_2d,
                pair_cond_chunk,
            )

        self._flush_conditioning_npu_queue(pair_cond, empty_cache=True)
        return pair_cond

    def _conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
        clear_flag: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.first_run:
            self.first_run = False
            single_embedding = use_conditioning * embeddings['single']
            use_row_streaming = diffusion_conditioning_row_streaming(
                feature_length=embeddings['pair'].shape[0],
                training=self.training, grad_enabled=torch.is_grad_enabled(),
                threshold=DIFFUSION_CONDITIONING_ROW_STREAM_THRESHOLD,
            )
            if use_row_streaming:
                pair_cond = self._build_pair_conditioning_row_streaming(
                    token_features=batch.token_features,
                    pair_embedding=embeddings['pair'],
                    use_conditioning=use_conditioning,
                )
            else:
                # Reuse the conditioned trunk pair; retain the multiplied-zero
                # branch so training keeps the original connected gradient.
                pair_embedding = (
                    embeddings['pair']
                    if use_conditioning
                    else embeddings['pair'] * 0
                )

                rel_features = featurization.create_relative_encoding(
                    batch.token_features,
                    max_relative_idx=32,
                    max_relative_chain=2,
                    dtype=pair_embedding.dtype,
                )
                features_2d = torch.concatenate(
                    [pair_embedding, rel_features],
                    dim=-1,
                )
                pair_cond = self.pair_cond_initial_projection(
                    self.pair_cond_initial_norm(features_2d)
                )
                del features_2d, rel_features, pair_embedding

                if not inference_chunking(
                    training=self.training, grad_enabled=torch.is_grad_enabled(),
                ):
                    pair_cond += self.pair_transition_0(pair_cond)
                    pair_cond += self.pair_transition_1(pair_cond)
                else:
                    self.pair_transition_0.add_pair_residual_row_streaming_(
                        pair_cond
                    )
                    self.pair_transition_1.add_pair_residual_row_streaming_(
                        pair_cond
                    )

            # Confidence and Distogram consume the trunk pair only after the
            # derived long-sequence FA cache has finished using NPU memory.
            if large_pair_offload(training=self.training):
                embeddings['pair'] = embeddings['pair'].cpu()
                print('[offload] diffusion_pair_to_cpu=True', flush=True)
                self._flush_conditioning_npu_queue(
                    pair_cond,
                    empty_cache=True,
                )

            target_feat = embeddings['target_feat']
            features_1d = torch.concatenate(
                [single_embedding, target_feat], dim=-1)
            single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d))

            self.single_cond = single_cond
            self.pair_cond = pair_cond

        single_cond = self.single_cond.clone()  # Clone cached conditioning before step-specific in-place updates.
        pair_cond = self.pair_cond
        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA)
        )

        single_cond += self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )

        single_cond += self.single_transition_0(single_cond)
        single_cond += self.single_transition_1(single_cond)

        if clear_flag:
            self.first_run = True
            self.single_cond = None
            self.pair_cond = None

        return single_cond, pair_cond

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
        clear_flag: bool,
    ) -> torch.Tensor:
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
            clear_flag=clear_flag,
        )
        level = noise_level**2 + SIGMA_DATA_2
        level_sqrt = torch.sqrt(level)
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        act = positions_noisy * atom_mask.unsqueeze(-1)
        act = act / level_sqrt

        enc = self.atom_cross_att_encoder(
            batch=batch,
            token_atoms_act=act,
            trunk_single_cond=embeddings['single'],
            trunk_pair_cond=trunk_pair_cond,
            clear_flag=clear_flag,
        )
        act = enc.token_act

        act += self.single_cond_embedding_projection(
            self.single_cond_embedding_norm(trunk_single_cond)
        )

        act = self.transformer(
            act=act,
            single_cond=trunk_single_cond,
            mask=sequence_mask,
            pair_cond=trunk_pair_cond,
            clear_flag=clear_flag,
        )
        release_pair_conditioning(self)
        del trunk_pair_cond
        act = self.output_norm(act)

        position_update = self.atom_cross_att_decoder(
            batch=batch,
            token_act=act,
            enc=enc,
            clear_flag=clear_flag,
        )

        skip_scaling = SIGMA_DATA_2 / level
        out_scaling = (
            noise_level * SIGMA_DATA / level_sqrt
        )

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask.unsqueeze(-1)
