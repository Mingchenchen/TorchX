import torch
import torch.nn as nn

from torchfold import feat_batch
from torchfold.runtime_policy import (
    DIFFUSION_CONDITIONING_ROW_STREAM_CHUNK_SIZE,
    diffusion_conditioning_row_streaming,
    inference_chunking,
    release_pair_conditioning,
)
from . import featurization, utils
from .diffusion_transformer import DiffusionTransformer, DiffusionTransition
from .atom_cross_attention import AtomCrossAttEncoder, AtomCrossAttDecoder
from ...parallel_ops import (
    pad_to_length,
    slice_1d_local,
)

from .. import fastnn

# Reference data scale for diffusion noise and coordinate normalization.
SIGMA_DATA = 16.0
SIGMA_DATA_2 = 256.0
PAIR_CPU_OFFLOAD_KEY = "_diffusion_pair_cpu_offload"


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
    rotation: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply random rigid augmentation.

    Args:
      positions: atom positions of shape (<common_axes>, 3)
      mask: per-atom mask of shape (<common_axes>,)
    Returns:
      Transformed positions with the same shape as input positions.
    """
    if (rotation is None) != (translation is None):
        raise ValueError(
            "rotation and translation must either both be provided or both be omitted"
        )

    center = utils.mask_mean(
        mask.unsqueeze(-1), positions, dim=(-2, -3), keepdim=True, eps=1e-6
    )

    if positions.ndim == 4:
        if mask.ndim != 3 or positions.shape[:-1] != mask.shape:
            raise ValueError(
                "Batched positions and mask shapes are inconsistent: "
                f"positions={tuple(positions.shape)} mask={tuple(mask.shape)}"
            )
        batch_size = positions.shape[0]
        if rotation is None:
            rotation = random_rotation(
                device=positions.device,
                dtype=positions.dtype,
                batch_size=batch_size,
            )
            translation = torch.randn(
                size=(batch_size, 1, 1, 3),
                dtype=positions.dtype,
                device=positions.device,
            )
        elif tuple(rotation.shape) != (batch_size, 3, 3):
            raise ValueError(
                "Unexpected batched rotation shape: "
                f"actual={tuple(rotation.shape)} expected={(batch_size, 3, 3)}"
            )
        if tuple(translation.shape) != (batch_size, 1, 1, 3):
            raise ValueError(
                "Unexpected batched translation shape: "
                "actual="
                f"{tuple(translation.shape)} expected={(batch_size, 1, 1, 3)}"
            )
        shifted = positions - center
        shifted_flat = shifted.reshape(batch_size, -1, 3)
        augmented_positions = torch.bmm(shifted_flat, rotation).reshape_as(
            positions
        )
        augmented_positions += translation
    else:
        if rotation is None:
            rotation = random_rotation(
                device=positions.device,
                dtype=positions.dtype,
            )
            translation = torch.randn(
                size=(3,),
                dtype=positions.dtype,
                device=positions.device,
            )
        elif tuple(rotation.shape) != (3, 3):
            raise ValueError(
                "Unexpected rotation shape: "
                f"actual={tuple(rotation.shape)} expected={(3, 3)}"
            )
        if tuple(translation.shape) != (3,):
            raise ValueError(
                "Unexpected translation shape: "
                f"actual={tuple(translation.shape)} expected={(3,)}"
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


def _create_relative_encoding_row_shard(
    batch: feat_batch.Batch,
    parallel_spec,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Prefer row-sharded relative encoding when available.
    Older checkouts may only expose the full relative encoding helper; in that
    case build the full tensor and then slice the local shard.
    """
    if hasattr(featurization, "create_relative_encoding_row_shard"):
        return featurization.create_relative_encoding_row_shard(
            batch.token_features,
            max_relative_idx=32,
            max_relative_chain=2,
            parallel_spec=parallel_spec,
        ).to(dtype=dtype)

    rel_full = featurization.create_relative_encoding(
        batch.token_features,
        max_relative_idx=32,
        max_relative_chain=2,
    ).to(dtype=dtype)

    rel_full = pad_to_length(rel_full, dim=0, length=parallel_spec.n_padded, value=0.0)
    rel_full = pad_to_length(rel_full, dim=1, length=parallel_spec.n_padded, value=0.0)
    rel_row = slice_1d_local(rel_full, parallel_spec, pad_value=0.0)
    return rel_row.contiguous()


class DiffusionHead(nn.Module):
    def __init__(self):
        super(DiffusionHead, self).__init__()

        self.c_act = 768
        self.pair_channel = 128
        self.seq_channel = 384

        self.c_pair_cond_initial = 267
        self.pair_cond_initial_norm = fastnn.LayerNorm(
            self.c_pair_cond_initial, bias=False)
        self.pair_cond_initial_projection = nn.Linear(
            self.c_pair_cond_initial, self.pair_channel, bias=False)

        self.pair_transition_0 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)
        self.pair_transition_1 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)

        self.c_single_cond_initial = 831
        self.single_cond_initial_norm = fastnn.LayerNorm(
            self.c_single_cond_initial, bias=False)
        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial, self.seq_channel, bias=False)

        self.c_noise_embedding = 256
        self.noise_embedding_initial_norm = fastnn.LayerNorm(
            self.c_noise_embedding, bias=False)
        self.noise_embedding_initial_projection = nn.Linear(
            self.c_noise_embedding, self.seq_channel, bias=False)

        self.single_transition_0 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)
        self.single_transition_1 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)

        self.atom_cross_att_encoder = AtomCrossAttEncoder(
            per_token_channels=self.c_act,
            with_token_atoms_act=True,
            with_trunk_pair_cond=True,
            with_trunk_single_cond=True,
        )

        self.single_cond_embedding_norm = fastnn.LayerNorm(
            self.seq_channel, bias=False)
        self.single_cond_embedding_projection = nn.Linear(
            self.seq_channel, self.c_act, bias=False)

        self.transformer = DiffusionTransformer()

        self.output_norm = fastnn.LayerNorm(self.c_act, bias=False)
        self.atom_cross_att_decoder = AtomCrossAttDecoder()
        self.fourier_embeddings = FourierEmbeddings(dim=256)

        self.first_run = True
        self.single_cond = None
        self.pair_cond = None
        self.pair_cond_for_atom = None

    @staticmethod
    def _flush_conditioning_npu_queue(
        tensor: torch.Tensor,
        empty_cache: bool = False,
    ) -> None:
        if tensor.device.type != "npu":
            return
        torch.npu.synchronize()
        if empty_cache:
            torch.npu.empty_cache()

    def _release_staged_pair_to_cpu(
        self,
        embeddings: dict[str, torch.Tensor],
        pair_cond: torch.Tensor,
    ) -> None:
        pair_cpu = embeddings.pop(PAIR_CPU_OFFLOAD_KEY, None)
        if pair_cpu is None:
            return
        embeddings['pair'] = pair_cpu
        self._flush_conditioning_npu_queue(pair_cond, empty_cache=True)

    def _release_pair_conditioning_if_cached(self) -> None:
        release_pair_conditioning(self)

    def _build_pair_conditioning_row_streaming(
        self,
        token_features,
        pair_embedding: torch.Tensor,
        parallel_spec,
        use_conditioning: bool,
        chunk_size: int = DIFFUSION_CONDITIONING_ROW_STREAM_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Build a local pair shard without full-shard expanded intermediates."""
        if not inference_chunking(
            training=self.training, grad_enabled=torch.is_grad_enabled(),
        ):
            raise RuntimeError("diffusion conditioning row streaming is inference-only")
        if chunk_size <= 0:
            raise ValueError("diffusion conditioning row chunk size must be positive")
        if pair_embedding.shape != (
            parallel_spec.shard_size, parallel_spec.n_padded, self.pair_channel,
        ):
            raise ValueError("diffusion conditioning expects a local pair row shard")

        self._flush_conditioning_npu_queue(pair_embedding, empty_cache=True)
        pair_cond = pair_embedding.new_empty(pair_embedding.shape)
        for start in range(0, pair_embedding.shape[0], chunk_size):
            end = min(start + chunk_size, pair_embedding.shape[0])
            pair_chunk = pair_embedding[start:end]
            if not use_conditioning:
                pair_chunk = pair_chunk * 0
            relative_chunk = featurization._create_relative_encoding_row_slice(
                token_features,
                max_relative_idx=32,
                max_relative_chain=2,
                parallel_spec=parallel_spec,
                row_start=parallel_spec.start + start,
                row_end=parallel_spec.start + end,
                dtype=pair_embedding.dtype,
            )
            features_chunk = torch.cat((pair_chunk, relative_chunk), dim=-1)
            cond_chunk = self.pair_cond_initial_projection(
                self.pair_cond_initial_norm(features_chunk)
            )
            self.pair_transition_0.add_pair_residual_row_streaming_(
                cond_chunk, chunk_size=end - start,
            )
            self.pair_transition_1.add_pair_residual_row_streaming_(
                cond_chunk, chunk_size=end - start,
            )
            pair_cond[start:end].copy_(cond_chunk)
            # Bound queued NPU temporaries as well as Python tensor lifetimes.
            self._flush_conditioning_npu_queue(pair_cond)
            del pair_chunk, relative_chunk, features_chunk, cond_chunk
        self._flush_conditioning_npu_queue(pair_cond, empty_cache=True)
        return pair_cond

    def _conditioning(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
        clear_flag: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parallel_spec = embeddings.get('parallel_spec', None)

        if self.first_run:
            self.first_run = False

            cond_scale = 1.0 if use_conditioning else 0.0
            single_embedding = cond_scale * embeddings['single']   # [L, C]
            if (
                parallel_spec is not None
                and parallel_spec.is_distributed
                and diffusion_conditioning_row_streaming(
                    feature_length=parallel_spec.n_global,
                    training=self.training,
                    grad_enabled=torch.is_grad_enabled(),
                )
            ):
                pair_cond = self._build_pair_conditioning_row_streaming(
                    batch.token_features, embeddings['pair'], parallel_spec,
                    use_conditioning,
                )
            else:
                # Reuse the read-only row shard; keep the zero branch connected for gradients.
                pair_embedding = (
                    embeddings['pair']
                    if use_conditioning
                    else embeddings['pair'] * 0
                )  # [Ls, Lp, C]

                if parallel_spec is not None:
                    rel_features = _create_relative_encoding_row_shard(
                        batch=batch,
                        parallel_spec=parallel_spec,
                        dtype=pair_embedding.dtype,
                    )
                else:
                    rel_features = featurization.create_relative_encoding(
                        batch.token_features,
                        max_relative_idx=32,
                        max_relative_chain=2,
                    ).to(dtype=pair_embedding.dtype)

                features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)

                pair_cond = self.pair_cond_initial_projection(
                    self.pair_cond_initial_norm(features_2d)
                )
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

                del features_2d, rel_features, pair_embedding

            if parallel_spec is not None:
                pair_cond_for_atom = pair_cond
            else:
                pair_cond_for_atom = pair_cond.contiguous()

            self._release_staged_pair_to_cpu(
                embeddings=embeddings,
                pair_cond=pair_cond,
            )

            target_feat = embeddings['target_feat']  # [L, C]
            features_1d = torch.concatenate([single_embedding, target_feat], dim=-1)
            single_cond = self.single_cond_initial_projection(
                self.single_cond_initial_norm(features_1d)
            )

            self.single_cond = single_cond
            self.pair_cond = pair_cond
            self.pair_cond_for_atom = pair_cond_for_atom

        single_cond = self.single_cond.clone()
        pair_cond = self.pair_cond
        pair_cond_for_atom = self.pair_cond_for_atom

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
            self.pair_cond_for_atom = None

        return single_cond, pair_cond, pair_cond_for_atom

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
        clear_flag: bool,
    ) -> torch.Tensor:
        parallel_spec = embeddings.get('parallel_spec', None)

        trunk_single_cond, trunk_pair_cond, trunk_pair_cond_for_atom = self._conditioning(
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
            trunk_pair_cond=trunk_pair_cond_for_atom,
            clear_flag=clear_flag,
            parallel_spec=parallel_spec,
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
            parallel_spec=parallel_spec,
            clear_flag=clear_flag,
        )
        self._release_pair_conditioning_if_cached()
        del trunk_pair_cond, trunk_pair_cond_for_atom
        act = self.output_norm(act)

        position_update = self.atom_cross_att_decoder(
            batch=batch,
            token_act=act,
            enc=enc,
            clear_flag=clear_flag,
        )

        skip_scaling = SIGMA_DATA_2 / level
        out_scaling = noise_level * SIGMA_DATA / level_sqrt

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask.unsqueeze(-1)
