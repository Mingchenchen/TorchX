import torch
import torch.nn as nn
import einops
from functools import partial
from typing import Callable, List, Optional

from torchfold.model import feat_batch
from torchfold.model.modules import embedders as featurization
from torchfold.model import utils

from torchfold.model.triangular.layers import LayerNorm
from torchfold.model.modules.pairformer import SelfAttention
from torchfold.model.modules.transformer import (
    DiffusionTransition,
    AtomCrossAttEncoder,
    AtomCrossAttDecoder,
)

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


# ========== Training noise sampler ==========
class TrainingNoiseSampler:
    """
    Noise sampler used during training; draws noise levels from a log-normal
    distribution, following the AlphaFold3 paper.

    Optional step curriculum (``p_mean_schedule`` + ``update_every``):
    every ``update_every`` optimizer steps, advance to the next ``p_mean`` in
    the schedule and hold the last value thereafter. Example::

        TrainingNoiseSampler(
            p_mean_schedule=[0.5, 0.0, -0.5, -1.2],
            update_every=10000,
        )
        # steps [0, 10k): 0.5; [10k, 20k): 0.0; [20k, 30k): -0.5; >=30k: -1.2
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,
        p_mean_schedule: Optional[List[float]] = None,
        update_every: Optional[int] = None,
    ):
        """
        Args:
            p_mean: Mean of the underlying log-normal (used when no schedule).
            p_std: Std-dev of the log-normal.
            sigma_data: Data standard deviation (typically 16.0).
            p_mean_schedule: Optional piecewise ``p_mean`` curriculum.
            update_every: Optimizer steps per schedule stage (required if schedule set).
        """
        self.sigma_data = sigma_data
        self.p_std = p_std
        self.p_mean_schedule = (
            [float(x) for x in p_mean_schedule] if p_mean_schedule else None
        )
        self.update_every = int(update_every) if update_every else None
        if self.p_mean_schedule is not None:
            if not self.p_mean_schedule:
                raise ValueError("p_mean_schedule must be non-empty when provided")
            if self.update_every is None or self.update_every <= 0:
                raise ValueError(
                    "update_every must be a positive int when p_mean_schedule is set"
                )
            self.p_mean = float(self.p_mean_schedule[0])
        else:
            self.p_mean = float(p_mean)

    def set_step(self, step: int) -> float:
        """Apply curriculum for optimizer ``step``; returns the active ``p_mean``."""
        if self.p_mean_schedule is not None and self.update_every:
            idx = min(int(step) // self.update_every, len(self.p_mean_schedule) - 1)
            self.p_mean = float(self.p_mean_schedule[idx])
        return self.p_mean

    def __call__(
        self,
        size: tuple,
        device: torch.device = torch.device("cpu"),
        return_t: bool = True,
    ) -> torch.Tensor:
        """
        Sample noise levels.

        Args:
            size: Target shape, e.g. (batch_size, N_sample).
            device: Device hosting the tensor.

        Returns:
            Noise-level tensor of shape [..., N_sample].
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        if not return_t:
            return noise_level
        t_values = 0.5 * (1.0 + torch.erf(rnd_normal / (2.0 ** 0.5)))
        return noise_level, t_values

    def sample_with_params(
        self,
        size: tuple,
        device: torch.device,
        p_mean: float = None,
        p_std: float = None,
        return_t: bool = False,
    ) -> torch.Tensor:
        """
        Sample noise levels with custom parameters (Used to distinguish noise distribution by data source).

        Args:
            size: Target shape, e.g. (N_sample,).
            device: Device hosting the tensor.
            p_mean: Override mean of log-normal (None = use default).
            p_std: Override std of log-normal (None = use default).

        Returns:
            Noise-level tensor of shape [..., N_sample].
        """
        p_mean = p_mean if p_mean is not None else self.p_mean
        p_std = p_std if p_std is not None else self.p_std

        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * p_std + p_mean).exp() * self.sigma_data
        if not return_t:
            return noise_level
        t_values = 0.5 * (1.0 + torch.erf(rnd_normal / (2.0 ** 0.5)))
        return noise_level, t_values


# ========== Centre-aware random augmentation ==========
def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Apply centre-preserving random augmentation (rotation + translation).

    Args:
        x_input_coords: Coordinates [..., N_token, 24, 3] (token-dense format).
        N_sample: Number of augmented samples to generate.
        mask: Optional atom mask [..., N_token, 24].

    Returns:
        Augmented coordinates [..., N_sample, N_token, 24, 3].
    """
    dtype = x_input_coords.dtype
    device = x_input_coords.device

    # Flatten token+atom axes to compute a centroid
    # x_input_coords: [..., N_token, 24, 3]
    original_shape = x_input_coords.shape
    x_flat = x_input_coords.reshape(-1, 3)  # [N_token * 24, 3]

    # Compute centroid (respect the mask if provided)
    if mask is not None:
        mask_flat = mask.reshape(-1)  # [N_token * 24]
        mask_expanded = mask_flat[:, None]  # [N_token * 24, 1]
        center = (x_flat * mask_expanded).sum(dim=0, keepdim=True) / \
            (mask_expanded.sum(dim=0, keepdim=True) + 1e-6)  # [1, 3]
    else:
        center = x_flat.mean(dim=0, keepdim=True)  # [1, 3]

    # Centre the coordinates
    x_centered_flat = x_flat - center  # [N_token * 24, 3]
    x_centered = x_centered_flat.reshape(original_shape)  # [..., N_token, 24, 3]

    # Generate N_sample augmented variants
    augmented_samples = []

    for _ in range(N_sample):
        # Random rotation matrix
        rot = random_rotation(device=device, dtype=dtype)  # [3, 3]

        # Random translation vector
        translation = torch.randn(size=(3,), dtype=dtype, device=device)

        # Apply transform: R * x + t
        # x_centered: [..., N_token, 24, 3]
        # Multiply along the last axis
        x_rotated = torch.einsum('...i,ij->...j', x_centered, rot)
        x_augmented = x_rotated + translation

        # Reapply mask if provided
        if mask is not None:
            x_augmented = x_augmented * mask[..., None]

        augmented_samples.append(x_augmented)

    # Stack all samples into [..., N_sample, N_token, 24, 3]
    return torch.stack(augmented_samples, dim=-4)


class FourierEmbeddings(nn.Module):
    """Fourier feature embeddings with FIXED (frozen) weight/bias.

    AF3 treats these as fixed random Fourier features (never trained); kept as
    nn.Parameter(requires_grad=False) so af3.bin still loads them but the optimizer
    never drifts them.
    """

    def __init__(self, dim: int):
        super(FourierEmbeddings, self).__init__()
        self.dim = dim
        # Fixed random Fourier features (overwritten by af3.bin constants on load);
        # frozen to match AF3 -- never trained.
        self.weight = nn.Parameter(torch.randn(dim) * 0.02, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(dim) * 0.02, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(2 * torch.pi * (x[..., None] * self.weight + self.bias))


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
    return (
        SIGMA_DATA
        * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
    )


def random_rotation(device, dtype, leading_shape=()):
    # Create random rotation(s) via Gram-Schmidt of two random normal vectors.
    # leading_shape=() -> a single [3, 3] matrix (original behaviour);
    # leading_shape=(S,) -> [S, 3, 3] INDEPENDENT rotations (one per sample), so
    # a batched chunk of diffusion samples is augmented exactly as if each sample
    # had been processed on its own.
    v = torch.randn(size=(*leading_shape, 2, 3), dtype=dtype, device=device)
    v0, v1 = v[..., 0, :], v[..., 1, :]
    e0 = v0 / torch.clamp(torch.linalg.norm(v0, dim=-1, keepdim=True), min=1e-10)
    v1 = v1 - e0 * (v1 * e0).sum(dim=-1, keepdim=True)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1, dim=-1, keepdim=True), min=1e-10)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-2)


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

    # positions is (<sample_axes>, N_token, N_dense_atom, 3): the last two axes
    # are the atom axes (centre reduces over dim=(-2, -3)), so any leading axes
    # are independent samples. Broadcast the mask over those sample axes so
    # mask_mean's ndim check matches the batched positions, and draw one
    # rotation+translation per sample -> a batched chunk is numerically
    # equivalent to the old per-sample loop; an empty sample_shape reproduces
    # the original single-sample behaviour exactly.
    sample_shape = positions.shape[:-3]
    mask_b = mask.expand(*sample_shape, *mask.shape)
    center = utils.mask_mean(
        mask_b[..., None], positions, dim=(-2, -3), keepdim=True, eps=1e-6
    )
    rot = random_rotation(
        device=positions.device, dtype=positions.dtype, leading_shape=sample_shape
    )
    rot = rot.reshape(*sample_shape, 1, 1, 3, 3)
    translation = torch.randn(
        size=(*sample_shape, 1, 1, 3), dtype=positions.dtype, device=positions.device)

    augmented_positions = (
        torch.einsum(
            '...i,...ij->...j',
            positions - center,
            rot,
        )
        + translation
    )
    return augmented_positions * mask_b[..., None]


class DiffusionTransformer(nn.Module):
    """AF3 super-block diffusion transformer.

    Blocks are grouped into super-blocks; pair_logits_projection is a ModuleList
    of one Linear per super-block (6 for the AF3 default 24/4 config).
    """

    def __init__(
        self,
        c_act: int = 768,
        c_single_cond: int = 384,
        c_pair_cond: int = 128,
        num_head: int = 16,
        num_blocks: int = 24,
        super_block_size: int = 4,
    ) -> None:
        super(DiffusionTransformer, self).__init__()

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        self.pair_input_layer_norm = LayerNorm(self.c_pair_cond)
        self.pair_logits_projection = nn.ModuleList(
            [
                nn.Linear(
                    self.c_pair_cond,
                    self.super_block_size * self.num_head,
                    bias=False,
                )
                for _ in range(self.num_super_blocks)
            ]
        )

        self.self_attention = nn.ModuleList(
            [
                SelfAttention(self.c_act, self.c_single_cond, use_single_cond=True)
                for _ in range(self.num_blocks)
            ]
        )
        self.transition_block = nn.ModuleList(
            [
                DiffusionTransition(
                    self.c_act, self.c_single_cond, use_single_cond=True
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(
        self,
        act: torch.Tensor,          # variable
        mask: torch.Tensor,         # constant
        single_cond: torch.Tensor,  # variable
        pair_cond: torch.Tensor,    # constant
    ) -> torch.Tensor:

        pair_act = self.pair_input_layer_norm(pair_cond)

        # Work with local scratch buffers
        pair_logits_list = []

        for super_block_i in range(self.num_super_blocks):
            pair_logits = self.pair_logits_projection[super_block_i](pair_act)
            pair_logits = einops.rearrange(
                pair_logits, 'n s (b h) -> b h n s', h=self.num_head)
            pair_logits_list.append(pair_logits)

        for super_block_i in range(self.num_super_blocks):
            for j in range(self.super_block_size):
                act = act + self.self_attention[super_block_i * self.super_block_size + j](
                    act, mask, pair_logits_list[super_block_i][j, ...], single_cond)
                act = act + self.transition_block[
                    super_block_i * self.super_block_size + j](act, single_cond)

        return act


class DiffusionHead(nn.Module):
    """AF3 diffusion head with inlined conditioning.
    """

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

        self.atom_cross_att_encoder = AtomCrossAttEncoder(
            per_token_channels=self.c_act,
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

    def _conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
        epitope_pair_bias: torch.Tensor = None,  # [N_token, N_token, pair_channel] or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_embedding = use_conditioning * embeddings['single']
        pair_embedding = use_conditioning * embeddings['pair']
        rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=32, max_relative_chain=2
        ).to(dtype=pair_embedding.dtype)
        features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)

        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )

        pair_cond = pair_cond + self.pair_transition_0(pair_cond)
        pair_cond = pair_cond + self.pair_transition_1(pair_cond)

        # === Epitope Guidance: inject epitope-paratope contact bias into pair_cond ===
        if epitope_pair_bias is not None:
            pair_cond = pair_cond + epitope_pair_bias

        target_feat = embeddings['target_feat']
        features_1d = torch.concatenate(
            [single_embedding, target_feat], dim=-1)
        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d))

        single_cond_out = single_cond
        pair_cond_out = pair_cond

        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(torch.clamp(noise_level, min=1e-8) / SIGMA_DATA)
        )

        noise_embedding = self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )
        if noise_level.dim() > 0:
            single_cond_out = single_cond_out + noise_embedding.unsqueeze(1)
        else:
            single_cond_out = single_cond_out + noise_embedding

        single_cond_out = single_cond_out + self.single_transition_0(single_cond_out)
        single_cond_out = single_cond_out + self.single_transition_1(single_cond_out)

        return single_cond_out, pair_cond_out

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
        epitope_pair_bias: torch.Tensor = None,  # [N_token, N_token, pair_channel] or None
    ) -> torch.Tensor:
        # Get conditioning
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
            epitope_pair_bias=epitope_pair_bias,
        )

        # Extract features
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        # Position features
        act = positions_noisy * atom_mask.unsqueeze(-1)
        if noise_level.dim() > 0:
            noise_level = noise_level.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        level = noise_level**2 + SIGMA_DATA**2
        level_sqrt = torch.sqrt(level)
        act = act / level_sqrt

        enc = self.atom_cross_att_encoder(
            batch=batch,
            token_atoms_act=act,
            trunk_single_cond=embeddings['single'],
            trunk_pair_cond=trunk_pair_cond,
        )
        act = enc.token_act

        act = act + self.single_cond_embedding_projection(
            self.single_cond_embedding_norm(trunk_single_cond)
        )

        act = self.transformer(
            act=act,
            single_cond=trunk_single_cond,
            mask=sequence_mask,
            pair_cond=trunk_pair_cond,
        )
        act = self.output_norm(act)
        # (Possibly) atom-granularity decoder
        position_update = self.atom_cross_att_decoder(
            batch=batch,
            token_act=act,
            enc=enc,
        )

        skip_scaling = SIGMA_DATA**2 / level
        out_scaling = (
            noise_level * SIGMA_DATA / level_sqrt
        )

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask.unsqueeze(-1)
