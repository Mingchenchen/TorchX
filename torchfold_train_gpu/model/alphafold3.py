"""torchfold.model.alphafold3 — top-level AlphaFold3 model, built from the already-ported torchfold modules.

This is the capstone of the torchfold port.  The class tree (``self.evoformer``,
``self.evoformer_conditioning``, ``self.diffusion_head``, ``self.distogram_head``,
``self.confidence_head`` and all of their sub-modules) is byte-for-byte identical
to ``alphafold3.AlphaFold3`` so that the existing
``params.import_jax_weights_`` loads ``af3.bin.zst`` into this NEW model with
ZERO missing / unexpected keys.

The recycling loop, conditioning wiring and forward computation are preserved
exactly (recycle ``range(num_recycles + 1)``; embed → recycle → diffusion → heads).
"""

import contextlib
import os
import random
import time
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm, trange

# Pure data structures
from torchfold.model import feat_batch, features

# Ported torchfold featurization helpers
from torchfold.model.modules import embedders as featurization

# Ported torchfold sub-modules (reuse — do NOT re-implement).
from torchfold.model.modules.embedders import TemplateEmbedding
from torchfold.model.modules.pairformer import EvoformerBlock, PairformerBlock
from torchfold.model.modules.head import DistogramHead
from torchfold.model.modules.confidence import ConfidenceHead
from torchfold.model.modules.transformer import AtomCrossAttEncoder
from torchfold.model.modules import diffusion as diffusion_head

from torchfold.model.triangular.layers import LayerNorm


USE_DIST = 0
_DIFFUSION_PARALLEL_ATOM_THRE_TRAIN0 = 320
_DIFFUSION_PARALLEL_ATOM_THRE_TRAIN1 = 541


def _make_no_cache_autocast_context():
    """Mirror the current autocast state but disable its cache.

    Used as ``context_fn`` for ``torch.utils.checkpoint.checkpoint`` with
    ``use_reentrant=False`` so forward and recompute produce the same set of
    intermediate tensors.
    """
    if torch.is_autocast_enabled():
        return torch.amp.autocast(
            'cuda',
            enabled=True,
            dtype=torch.get_autocast_dtype("cuda"),
            cache_enabled=False,
        )
    return contextlib.nullcontext()


def safe_checkpoint(function, *args, **kwargs):
    kwargs['use_reentrant'] = False
    kwargs['context_fn'] = lambda: (_make_no_cache_autocast_context(),
                                    _make_no_cache_autocast_context())
    return checkpoint.checkpoint(function, *args, **kwargs)


def safe_diffusion_checkpoint(function, *args, **kwargs):
    kwargs['use_reentrant'] = False
    kwargs['context_fn'] = lambda: (_make_no_cache_autocast_context(),
                                    _make_no_cache_autocast_context())
    return checkpoint.checkpoint(function, *args, **kwargs)


class Evoformer(nn.Module):
    """Evoformer trunk (mirrors ``alphafold3.Evoformer``).

    Bare ModuleList trunk (no standalone PairformerStack class).  Attribute names
    are identical so the af3.bin.zst state_dict loads strict.  Uses the
    ported torchfold ``EvoformerBlock`` / ``PairformerBlock`` / ``TemplateEmbedding``
    and the vendored ``LayerNorm`` (no fastnn).
    """

    def __init__(
        self,
        msa_channel: int = 64,
        checkpoint_config: dict = None,
        disable_internal_progress: bool = False,
        pair_dropout: float = 0.0,
        msa_dropout: float = 0.0,
    ):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024

        # Parse gradient-checkpointing configuration
        if checkpoint_config is None:
            checkpoint_config = {}

        pairformer_config = checkpoint_config.get('pairformer', {})
        self.pairformer_checkpoint_enabled = pairformer_config.get('enabled', False)
        self.pairformer_checkpoint_group_size = pairformer_config.get('group_size', 8)

        msa_config = checkpoint_config.get('msa', {})
        self.msa_checkpoint_enabled = msa_config.get('enabled', False)
        self.msa_checkpoint_group_size = msa_config.get('group_size', 4)

        template_config = checkpoint_config.get('template', {})
        self.template_checkpoint_enabled = template_config.get('enabled', False)

        self.disable_internal_progress = disable_internal_progress

        self.seq_channel = 384
        self.pair_channel = 128
        self.c_target_feat = 447

        self.left_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)
        self.right_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)

        self.prev_embedding_layer_norm = LayerNorm(self.pair_channel)
        self.prev_embedding = nn.Linear(
            self.pair_channel, self.pair_channel, bias=False)

        self.c_rel_feat = 139
        self.position_activations = nn.Linear(
            self.c_rel_feat, self.pair_channel, bias=False)

        self.bond_embedding = nn.Linear(
            1, self.pair_channel, bias=False)

        self.template_embedding = TemplateEmbedding(
            pair_channel=self.pair_channel,
            disable_internal_progress=disable_internal_progress)

        self.msa_activations = nn.Linear(34, self.msa_channel, bias=False)
        self.extra_msa_target_feat = nn.Linear(
            self.c_target_feat, self.msa_channel, bias=False)
        self.msa_stack = nn.ModuleList(
            [EvoformerBlock(pair_dropout=pair_dropout, msa_dropout=msa_dropout) for _ in range(self.msa_stack_num_layer)])

        self.single_activations = nn.Linear(
            self.c_target_feat, self.seq_channel, bias=False)

        self.prev_single_embedding_layer_norm = LayerNorm(self.seq_channel)
        self.prev_single_embedding = nn.Linear(
            self.seq_channel, self.seq_channel, bias=False)

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock(with_single=True, pair_dropout=pair_dropout) for _ in range(self.pairformer_num_layer)])

    def _relative_encoding(
        self, batch: "feat_batch.Batch", pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            max_relative_idx,
            max_relative_chain,
        ).to(dtype=pair_activations.dtype)
        rel_feat = self.position_activations(rel_feat)
        pair_activations = pair_activations + rel_feat
        return pair_activations

    def _seq_pair_embedding(
        self, token_features: "features.TokenFeatures", target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated Pair embedding from sequence."""
        left_single = self.left_single(target_feat).unsqueeze(1)
        right_single = self.right_single(target_feat).unsqueeze(0)
        pair_activations = left_single + right_single

        mask = token_features.mask
        pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(0)).to(dtype=left_single.dtype)

        return pair_activations, pair_mask

    def _embed_bonds(
        self, batch: "feat_batch.Batch", pair_activations: torch.Tensor
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations."""
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device
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
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0

        contact_matrix[0, 0] = 0.0
        bonds_act = self.bond_embedding(contact_matrix.unsqueeze(-1))
        pair_activations = pair_activations + bonds_act
        return pair_activations

    def _embed_template_pair(
            self,
            templates: "features.Templates",
            asym_id: torch.Tensor,
            pair_activations: torch.Tensor,
            pair_mask: torch.Tensor,
            is_checkpointed: bool = False,
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        dtype = pair_activations.dtype

        multichain_mask = (
                asym_id.unsqueeze(-1) == asym_id.unsqueeze(0)
        ).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask,
            is_checkpointed=is_checkpointed,
        )

        # Template dropout for diffusion training.
        # Entire template condition is randomly removed.
        # if self.training:
        #     template_dropout_p = 0.25
        #
        #     keep_template = (
        #             torch.rand(1, device=template_act.device)
        #             > template_dropout_p
        #     )
        #
        #     if not keep_template:
        #         template_act = torch.zeros_like(template_act)

        pair_activations = pair_activations + template_act

        return pair_activations

    def _embed_process_msa(
        self,
        batch: "feat_batch.Batch",
        msa_batch: "features.MSA",
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype

        msa_batch = featurization.shuffle_msa(msa_batch)
        msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations = msa_activations + self.extra_msa_target_feat(target_feat).unsqueeze(0)

        if self.msa_checkpoint_enabled and self.training and torch.is_grad_enabled():
            for i in tqdm(range(0, len(self.msa_stack), self.msa_checkpoint_group_size), desc="MSA Layers (checkpointed)", disable=self.disable_internal_progress):
                group_end = min(i + self.msa_checkpoint_group_size, len(self.msa_stack))
                msa_group = tuple(self.msa_stack[i:group_end])

                def msa_group_forward(msa_act, pair_act, m_mask, p_mask, _layers=msa_group):
                    for msa_block in _layers:
                        msa_act, pair_act = msa_block(
                            msa=msa_act,
                            pair=pair_act,
                            msa_mask=m_mask,
                            pair_mask=p_mask,
                        )
                    return msa_act, pair_act

                msa_activations, pair_activations = safe_checkpoint(
                    msa_group_forward, msa_activations, pair_activations, msa_mask, pair_mask
                )
        else:
            for msa_block in tqdm(self.msa_stack, desc="MSA Layers", disable=self.disable_internal_progress):
                msa_activations, pair_activations = msa_block(
                    msa=msa_activations,
                    pair=pair_activations,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )

        return pair_activations

    def _compute_pair_single_init(
        self,
        batch: dict[str, torch.Tensor],
        target_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pair_init and single_init once before recycle loop (AF3 Alg 1)."""
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )
        pair_activations = self._relative_encoding(batch, pair_activations)
        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )
        single_init = self.single_activations(target_feat)
        return pair_activations, pair_mask, single_init

    def forward(
        self,
        batch: dict[str, torch.Tensor],  # constant
        prev: dict[str, torch.Tensor],  # variable
        target_feat: torch.Tensor,  # constant
        idx: int = 0,
        pair_init: torch.Tensor = None,
        single_init: torch.Tensor = None,
        pair_init_mask: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )
        pair_activations = pair_activations + self.prev_embedding(
            self.prev_embedding_layer_norm(prev['pair']))
        pair_activations = self._relative_encoding(batch, pair_activations)
        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )
        pair_activations = self._embed_template_pair(
            templates=batch.templates,
            asym_id=batch.token_features.asym_id,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            is_checkpointed=self.template_checkpoint_enabled and self.training and torch.is_grad_enabled(),
        )

        pair_activations = self._embed_process_msa(
            batch=batch,
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
        )

        single_activations = self.single_activations(target_feat)
        single_activations = single_activations + self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single']))

        if self.pairformer_checkpoint_enabled and self.training and torch.is_grad_enabled():
            group_size = self.pairformer_checkpoint_group_size
            num_layers = len(self.trunk_pairformer)

            for i in tqdm(range(0, num_layers, group_size), desc="Pairformer Layers (checkpointed)", disable=self.disable_internal_progress):
                end_idx = min(i + group_size, num_layers)
                group_layers = tuple(self.trunk_pairformer[i:end_idx])

                def pairformer_group_forward(pair_act, single_act, p_mask, seq_mask, _layers=group_layers):
                    for layer in _layers:
                        pair_act, single_act = layer(
                            pair_act,
                            p_mask,
                            single_act,
                            seq_mask,
                        )
                    return pair_act, single_act

                pair_activations, single_activations = safe_checkpoint(
                    pairformer_group_forward,
                    pair_activations,
                    single_activations,
                    pair_mask,
                    batch.token_features.mask,
                )
        else:
            for layer in tqdm(self.trunk_pairformer, desc="Pairformer Layers", disable=self.disable_internal_progress):
                pair_activations, single_activations = layer(
                    pair=pair_activations,
                    pair_mask=pair_mask,
                    single=single_activations,
                    seq_mask=batch.token_features.mask,
                )

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output


class AlphaFold3(nn.Module):
    """Top-level AlphaFold3 model (mirrors ``alphafold3.AlphaFold3``).

    Built entirely from ported torchfold sub-modules; the attribute tree is identical so the whole
    af3.bin.zst state_dict loads strict.
    """

    def __init__(
        self,
        num_recycles: int = 10,
        num_samples: int = 5,
        diffusion_steps: int = 200,
        mini_rollout_steps: int = 20,
        num_diffusion_samples_training: int = 48,
        sample_diffusion_chunk_size: int = 5,
        checkpoint_config: dict = None,
        save_diffusion_debug: bool = False,
        disable_internal_progress: bool = False,
        noise_configs: dict = None,
        noise_p_mean: float = -1.2,
        noise_p_std: float = 1.5,
        noise_p_mean_schedule: list = None,
        noise_p_mean_update_every: int = None,
        condition_embedding_drop_rate: float = 0.0,
        pair_dropout: float = 0.0,
        msa_dropout: float = 0.0,
        train_mode: str = "full",  # "full" | "structure_only" | "confidence_only"
        randomize_num_recycles: bool = False,
    ):
        super(AlphaFold3, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.mini_rollout_steps = mini_rollout_steps
        self.sample_diffusion_chunk_size = sample_diffusion_chunk_size
        assert train_mode in ("full", "structure_only", "confidence_only"), train_mode
        self.train_mode = train_mode
        self._confidence_only = train_mode == "confidence_only"        # freeze trunk+diffusion
        self._trains_structure = train_mode != "confidence_only"       # trunk/diffusion get grad
        self._trains_confidence_head = train_mode != "structure_only"  # confidence head trained

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        if checkpoint_config is None:
            checkpoint_config = {}

        diffusion_config = checkpoint_config.get('diffusion', {})
        self.diffusion_checkpoint_enabled = diffusion_config.get('enabled', False)
        self.diffusion_checkpoint_group_size = diffusion_config.get('group_size', 8)

        confidence_config = checkpoint_config.get('confidence', {})
        self.confidence_checkpoint_enabled = confidence_config.get('enabled', False)

        self.disable_internal_progress = disable_internal_progress

        self.evoformer_conditioning = AtomCrossAttEncoder()

        self.evoformer = Evoformer(
            checkpoint_config=checkpoint_config,
            disable_internal_progress=disable_internal_progress,
            pair_dropout=pair_dropout,
            msa_dropout=msa_dropout,
        )

        self.diffusion_head = diffusion_head.DiffusionHead()

        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()
        self.save_diffusion_debug = save_diffusion_debug
        self.condition_embedding_drop_rate = condition_embedding_drop_rate

        self.train_noise_sampler = diffusion_head.TrainingNoiseSampler(
            p_mean=noise_p_mean,
            p_std=noise_p_std,
            sigma_data=16.0,
            p_mean_schedule=noise_p_mean_schedule,
            update_every=noise_p_mean_update_every,
        )
        self.num_diffusion_samples_training = num_diffusion_samples_training

        self.noise_configs = noise_configs if noise_configs is not None else {}
        self.randomize_num_recycles = randomize_num_recycles
        self.confidence_chunk_size = 1
        self._confidence_output_keys = (
            'predicted_lddt',
            'predicted_experimentally_resolved',
            'full_pde',
            'average_pde',
            'full_pae',
            'tmscore_adjusted_pae_global',
            'tmscore_adjusted_pae_interface',
            # Per-bin confidence logits (ADDITIVE — emitted by ConfidenceHead
            # alongside the expectations above). Threaded through the same
            # stack / checkpoint-tuple machinery and splatted into the final
            # output so the strict-AF3 cross-entropy confidence loss can read
            # them. They do not affect any pre-existing output value.
            'plddt_logits',
            'pae_logits',
            'pde_logits',
            'experimentally_resolved_logits',
        )

        self._apply_param_freezing()

    def _reset_module_states(self):
        pass

    def _sample_training_num_recycles(self, device: torch.device) -> int:
        if (
            not self.training
            or self._confidence_only
            or not self.randomize_num_recycles
            or self.num_recycles <= 0
        ):
            return self.num_recycles

        sampled = torch.zeros(1, device=device, dtype=torch.int64)
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            sampled = torch.randint(
                low=0,
                high=self.num_recycles + 1,
                size=(1,),
                device=device,
            )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(sampled, src=0)
        return int(sampled.item())

    def _run_diffusion_training_chunks(
        self,
        batch: "feat_batch.Batch",
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
        chunk_size: int,
        checkpoint_chunks: bool = False,
    ) -> torch.Tensor:
        x_denoised_chunks = []
        num_samples = positions_noisy.shape[0]
        for i in range(0, num_samples, chunk_size):
            i_end = min(i + chunk_size, num_samples)

            if checkpoint_chunks:
                def diffusion_chunk_forward(
                    positions_noisy_chunk: torch.Tensor,
                    noise_level_chunk: torch.Tensor,
                    single_cond: torch.Tensor,
                    pair_cond: torch.Tensor,
                    target_feat: torch.Tensor,
                ) -> torch.Tensor:
                    return self.diffusion_head(
                        positions_noisy=positions_noisy_chunk,
                        noise_level=noise_level_chunk,
                        batch=batch,
                        embeddings={
                            'single': single_cond,
                            'pair': pair_cond,
                            'target_feat': target_feat,
                        },
                        use_conditioning=use_conditioning,
                    )

                x_denoised_i = safe_diffusion_checkpoint(
                    diffusion_chunk_forward,
                    positions_noisy[i:i_end],
                    noise_level[i:i_end],
                    embeddings['single'],
                    embeddings['pair'],
                    embeddings['target_feat'],
                )
            else:
                x_denoised_i = self.diffusion_head(
                    positions_noisy=positions_noisy[i:i_end],
                    noise_level=noise_level[i:i_end],
                    batch=batch,
                    embeddings=embeddings,
                    use_conditioning=use_conditioning,
                )
            x_denoised_chunks.append(x_denoised_i)
        return torch.cat(x_denoised_chunks, dim=0)

    def _run_confidence_head_chunks(
        self,
        dense_atom_positions: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        batch: "feat_batch.Batch",
    ) -> dict[str, torch.Tensor]:
        outputs = []
        num_samples = dense_atom_positions.shape[0]
        for i in range(num_samples):
            outputs.append(
                self.confidence_head(
                    dense_atom_positions=dense_atom_positions[i],
                    embeddings=embeddings,
                    seq_mask=batch.token_features.mask,
                    token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                    asym_id=batch.token_features.asym_id,
                )
            )

        merged = {}
        for key in self._confidence_output_keys:
            merged[key] = torch.stack([sample[key] for sample in outputs], dim=0)
        return merged

    def _run_confidence_head_chunks_tuple(
        self,
        dense_atom_positions: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        batch: "feat_batch.Batch",
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._run_confidence_head_chunks(
            dense_atom_positions=dense_atom_positions,
            embeddings=embeddings,
            batch=batch,
        )
        return tuple(outputs[key] for key in self._confidence_output_keys)

    def _apply_param_freezing(self) -> None:
        """One-time freeze from train_mode (single source of truth).

        requires_grad is NOT reset by model.train()/eval(), so one build-time pass
        suffices -- no per-forward toggle. af3.bin params are unaffected (only the
        requires_grad flags change); DDP cleanly skips the frozen params.
          confidence_only : freeze everything EXCEPT the confidence head
          structure_only  : freeze ONLY the confidence head
          full            : everything trainable
        """
        if self._confidence_only:
            for name, param in self.named_parameters():
                param.requires_grad = name.startswith("confidence_head.")
        elif not self._trains_confidence_head:
            for name, param in self.named_parameters():
                if name.startswith("confidence_head."):
                    param.requires_grad = False

    def create_target_feat_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:

        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
        )

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat

    def _apply_denoising_step(
        self,
        batch: "feat_batch.Batch",
        embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_level_prev: torch.Tensor,
        mask: torch.Tensor,
        noise_level: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        positions = diffusion_head.random_augmentation(
            positions=positions, mask=mask
        )
        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        noise = noise_scale * \
            torch.randn(size=positions.shape, device=noise_scale.device)
        positions_noisy = positions + noise

        positions_denoised = self.diffusion_head(positions_noisy=positions_noisy,
                                                 noise_level=t_hat,
                                                 batch=batch,
                                                 embeddings=embeddings,
                                                 use_conditioning=True)
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad

        return positions_out

    def _apply_denoising_steps_group(
        self,
        batch: "feat_batch.Batch",
        embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_levels: torch.Tensor,
        mask: torch.Tensor,
        step_start: int,
        step_end: int,
    ) -> torch.Tensor:
        """Apply a group of denoising steps with gradient checkpointing."""
        current_positions = positions
        for step_idx in range(step_start, step_end):
            current_positions = self._apply_denoising_step(
                batch,
                embeddings,
                current_positions,
                noise_levels[step_idx],
                mask,
                noise_levels[1 + step_idx],
            )
        return current_positions

    def _sample_diffusion(
        self,
        batch: "feat_batch.Batch",
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample using denoiser on batch."""

        mask = batch.predicted_structure_info.atom_mask

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        num_samples = self.num_samples
        device = mask.device

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device, dtype=torch.float32))

        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device, dtype=torch.float32)
        positions *= noise_levels[0]

        chunk_size = self.sample_diffusion_chunk_size or num_samples
        per_step_all = [] if self.save_diffusion_debug else None
        for c_start in range(0, num_samples, chunk_size):
            c_end = min(c_start + chunk_size, num_samples)
            chunk = positions[c_start:c_end]
            per_step_positions = [] if self.save_diffusion_debug else None
            for step_idx in trange(
                self.diffusion_steps,
                desc=f"Diffusion samples [{c_start}:{c_end}]",
                disable=self.disable_internal_progress,
            ):
                chunk = self._apply_denoising_step(
                    batch,
                    embeddings,
                    chunk,
                    noise_levels[step_idx],
                    mask,
                    noise_levels[1 + step_idx],
                )
                if per_step_positions is not None:
                    per_step_positions.append(chunk.detach().cpu())
            positions[c_start:c_end] = chunk
            if per_step_all is not None:
                per_step_all.append(per_step_positions)
        final_dense_atom_mask = torch.tile(mask[None], (num_samples, 1, 1))
        out = {'atom_positions': positions, 'mask': final_dense_atom_mask}
        if self.save_diffusion_debug and 'per_step_all' in locals() and per_step_all is not None:
            out['per_step_positions'] = per_step_all
        return out

    def _sample_diffusion_training(
            self,
            batch: "feat_batch.Batch",
            embeddings: dict[str, torch.Tensor],
            gt_positions: torch.Tensor,
            gt_mask: torch.Tensor,
            N_sample: int = None,
            data_source_idx: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-time diffusion sampling with single-step denoising."""
        device = gt_positions.device
        dtype = gt_positions.dtype

        x_gt_augment = diffusion_head.centre_random_augmentation(
            x_input_coords=gt_positions,
            N_sample=N_sample,
            mask=gt_mask,
        ).to(dtype)

        if data_source_idx is not None and data_source_idx in self.noise_configs:
            noise_config = self.noise_configs[data_source_idx]
            p_mean = noise_config.get('p_mean', None)
            p_std = noise_config.get('p_std', None)
            sigma, t_values = self.train_noise_sampler.sample_with_params(
                size=(N_sample,),
                device=device,
                p_mean=p_mean,
                p_std=p_std,
                return_t=True,
            )
            sigma = sigma.to(dtype)
            t_values = t_values.to(dtype)
        else:
            sigma, t_values = self.train_noise_sampler(
                size=(N_sample,),
                device=device,
                return_t=True,
            )
            sigma = sigma.to(dtype)
            t_values = t_values.to(dtype)

        noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_noisy = x_gt_augment + noise

        drop_conditioning = False
        if self.training and self.condition_embedding_drop_rate > 0.0:
            drop_conditioning = random.random() < self.condition_embedding_drop_rate
        use_conditioning = True  # not drop_conditioning

        diffusion_chunk_size = min(
            max(1, self.diffusion_checkpoint_group_size),
            N_sample,
        )

        x_denoised = self._run_diffusion_training_chunks(
            batch=batch,
            positions_noisy=x_noisy,
            noise_level=sigma,
            embeddings={
                "single": embeddings["single"],
                "pair": embeddings["pair"],
                "target_feat": embeddings["target_feat"],
            },
            use_conditioning=use_conditioning,
            chunk_size=diffusion_chunk_size,
            checkpoint_chunks=self.diffusion_checkpoint_enabled,
        )

        return x_gt_augment, x_noisy, x_denoised, sigma, t_values

    def _sample_diffusion_mini_rollout(
        self,
        batch: "feat_batch.Batch",
        embeddings: dict[str, torch.Tensor],
        num_steps: int = None,
    ) -> dict[str, torch.Tensor]:
        """Mini-rollout: short denoising trajectories for confidence-head training."""
        if num_steps is None:
            num_steps = self.mini_rollout_steps

        mask = batch.predicted_structure_info.atom_mask
        num_samples = 1
        device = mask.device

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, num_steps + 1, device=device)
        )

        positions = torch.randn((num_samples,) + mask.shape + (3,), device=device)
        positions = positions * noise_levels[0]

        with torch.no_grad():
            for sample_idx in range(num_samples):
                for step_idx in range(num_steps):
                    positions[sample_idx] = self._apply_denoising_step(
                        batch,
                        embeddings,
                        positions[sample_idx],
                        noise_levels[step_idx],
                        mask,
                        noise_levels[1 + step_idx],
                    )

        final_dense_atom_mask = torch.tile(mask.unsqueeze(0), (num_samples, 1, 1))

        return {'atom_positions': positions, 'mask': final_dense_atom_mask}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        permute_label_cb=None,
    ) -> dict[str, torch.Tensor]:
        """AlphaFold3 forward pass.

        Args:
            batch: input feature dict.
            permute_label_cb: optional callback (Approach B). When provided in the
                structure-training branch it receives the mini-rollout dense coords
                and returns (gt_positions, gt_mask) already permuted to the
                mini-rollout frame; these are then used as the diffusion-training GT
                so the diffusion pred is born in the permuted frame.
        """
        phase = "train" if self.training else "eval"

        self._reset_module_states()
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        target_feat = self.create_target_feat_embedding(batch)

        embeddings = {
            'pair': torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel],
                device=target_feat.device,
                dtype=target_feat.dtype,
            ),
            'single': torch.zeros(
                [num_res, self.evoformer_seq_channel],
                dtype=target_feat.dtype,
                device=target_feat.device,
            ),
            'target_feat': target_feat,
        }

        actual_num_recycles = self._sample_training_num_recycles(target_feat.device)
        recycle_grad_enabled = self.training and self._trains_structure

        if self.training and self._confidence_only:
            self.evoformer.eval()

        for i in range(actual_num_recycles + 1):
            with torch.set_grad_enabled(recycle_grad_enabled and i == actual_num_recycles):
                embeddings = self.evoformer(
                    batch=batch, prev=embeddings, target_feat=target_feat, idx=i,
                )

        if self.training and self._confidence_only:
            self.evoformer.train()

        if self.training:
            mini_rollout_steps = 20 if self._confidence_only else self.mini_rollout_steps
            if self._confidence_only:
                samples = {}
                if mini_rollout_steps > 0:
                    samples_for_confidence = self._sample_diffusion_mini_rollout(
                        batch, embeddings, num_steps=mini_rollout_steps
                    )
                else:
                    gt_positions = batch.ground_truth_structure.true_positions
                    gt_mask = batch.ground_truth_structure.true_positions_atom_mask
                    samples_for_confidence = {
                        'atom_positions': gt_positions.unsqueeze(0),
                        'mask': gt_mask.unsqueeze(0),
                    }
            else:
                if mini_rollout_steps > 0:
                    samples_for_confidence = self._sample_diffusion_mini_rollout(
                        batch, embeddings, num_steps=mini_rollout_steps
                    )
                else:
                    samples_for_confidence = None

                gt_positions = batch.ground_truth_structure.true_positions
                gt_mask = batch.ground_truth_structure.true_positions_atom_mask

                if permute_label_cb is not None and samples_for_confidence is not None:
                    gt_positions, gt_mask = permute_label_cb(
                        samples_for_confidence['atom_positions']
                    )

                data_source_idx = None
                if hasattr(batch, 'data_source_idx') and batch.data_source_idx is not None:
                    data_source_idx = batch.data_source_idx.item() if isinstance(batch.data_source_idx, torch.Tensor) else batch.data_source_idx

                x_gt_augment, x_noisy, x_denoised, noise_levels, t_values = self._sample_diffusion_training(
                    batch=batch,
                    embeddings=embeddings,
                    gt_positions=gt_positions,
                    gt_mask=gt_mask,
                    N_sample=self.num_diffusion_samples_training,
                    data_source_idx=data_source_idx,
                )

                samples = {
                    'atom_positions': x_denoised,
                    'mask': gt_mask[None, :, :].expand(self.num_diffusion_samples_training, -1, -1),
                    'gt_positions': x_gt_augment,
                    'noise_levels': noise_levels,
                    't_values': t_values,
                }

                if samples_for_confidence is None:
                    # mini_rollout_steps == 0 fallback: feed the confidence head from
                    # the diffusion-training denoised output (current code's fallback).
                    samples_for_confidence = {
                        'atom_positions': x_denoised,
                        'mask': gt_mask[None, :, :].expand(self.num_diffusion_samples_training, -1, -1),
                    }
        else:
            samples = self._sample_diffusion(batch, embeddings)
            samples_for_confidence = samples

        t3 = time.time()

        if (self.training and self._trains_confidence_head) or not self.training:
            if self.training:
                torch.cuda.empty_cache()
            if self.confidence_checkpoint_enabled and torch.is_grad_enabled():
                def confidence_forward(
                    atom_pos: torch.Tensor,
                    single_act: torch.Tensor,
                    pair_act: torch.Tensor,
                    target_feat: torch.Tensor,
                ) -> tuple[torch.Tensor, ...]:
                    return self._run_confidence_head_chunks_tuple(
                        dense_atom_positions=atom_pos,
                        embeddings={
                            'single': single_act,
                            'pair': pair_act,
                            'target_feat': target_feat,
                        },
                        batch=batch,
                    )

                confidence_output_values = safe_checkpoint(
                    confidence_forward,
                    samples_for_confidence['atom_positions'],
                    embeddings['single'],
                    embeddings['pair'],
                    embeddings['target_feat'],
                )
                confidence_output = {
                    key: value
                    for key, value in zip(self._confidence_output_keys, confidence_output_values)
                }
            else:
                confidence_output = self._run_confidence_head_chunks(
                    dense_atom_positions=samples_for_confidence['atom_positions'],
                    embeddings=embeddings,
                    batch=batch,
                )
        else:
            confidence_output = {}

        t4 = time.time()

        distogram = self.distogram_head(batch, embeddings)

        output = {
            'distogram': distogram,
            **confidence_output,
        }
        if self.training and self._confidence_only:
            output['pred_coords'] = samples_for_confidence['atom_positions']
            output['confidence_only'] = True
        else:
            output['diffusion_samples'] = samples

        output['confidence_atom_positions'] = samples_for_confidence['atom_positions']

        return output
