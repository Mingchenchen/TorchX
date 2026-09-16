


import torch
import torch.nn as nn

from torchcraft import feat_batch
from torchcraft.nn import featurization, utils
from torchcraft.nn.atom_cross_attention import AtomCrossAttEncoder, AtomCrossAttDecoder
from torchcraft.nn.diffusion_transformer import DiffusionTransformer, DiffusionTransition
from torchcraft.nn.layer_norm import LayerNorm

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0
SIGMA_DATA_2 = 256.0


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
    return (SIGMA_DATA * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p)


def random_rotation(device, dtype):
    # Create a random rotation (Gram-Schmidt orthogonalization of two
    # random normal vectors)
    v0, v1 = torch.randn(size=(2, 3), dtype=dtype, device=device)
    e0 = v0 / torch.clamp(torch.linalg.norm(v0), min=1e-10)
    v1 = v1 - e0 * torch.dot(v1, e0)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1), min=1e-10)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2])


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
        mask.unsqueeze(-1), positions, dim=(-2, -3), keepdim=True, eps=1e-6
    )
    rot = random_rotation(device=positions.device, dtype=positions.dtype)
    translation = torch.randn(
        size=(3,), dtype=positions.dtype, device=positions.device)

    augmented_positions = (
        torch.einsum(
            '...i,ij->...j',
            positions - center,
            rot,
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

    def _conditioning(
        self,
        batch,  # constant
        embeddings: dict[str, torch.Tensor],  # constant
        noise_level: torch.Tensor,  # variable
        use_conditioning: bool,  # True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.first_run:
            self.first_run = False
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

            target_feat = embeddings['target_feat']
            features_1d = torch.concatenate(
                [single_embedding, target_feat], dim=-1)
            single_cond = self.single_cond_initial_projection(
                self.single_cond_initial_norm(features_1d))

            self.single_cond = single_cond
            self.pair_cond = pair_cond

        single_cond = self.single_cond.clone()  # will be changed
        pair_cond = self.pair_cond
        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA)
        )

        single_cond = single_cond + self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )

        single_cond = single_cond + self.single_transition_0(single_cond)
        single_cond = single_cond + self.single_transition_1(single_cond)

        return single_cond, pair_cond

    def forward(
        self,
        positions_noisy: torch.Tensor,  # variable
        noise_level: torch.Tensor,  # variable
        batch: feat_batch.Batch,  # constant
        embeddings: dict[str, torch.Tensor],  # constant
        use_conditioning: bool,  # True
    ) -> torch.Tensor:
        # Get conditioning
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # Extract features
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        # Position features
        act = positions_noisy * atom_mask.unsqueeze(-1)
        level = noise_level**2 + SIGMA_DATA_2
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

        skip_scaling = SIGMA_DATA_2 / level
        out_scaling = (
            noise_level * SIGMA_DATA / level_sqrt
        )

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask.unsqueeze(-1)
