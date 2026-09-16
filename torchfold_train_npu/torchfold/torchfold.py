import contextlib
import os
import einops
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch_npu
from tqdm import tqdm, trange

from torchfold import fastnn
from torchfold import feat_batch, features
from torchfold.fastnn import config as fastnn_config
from torchfold.nn import atom_cross_attention
from torchfold.nn import diffusion_head
from torchfold.nn import featurization
from torchfold.nn.head import DistogramHead, ConfidenceHead
from torchfold.nn.pairformer import EvoformerBlock, PairformerBlock
from torchfold.nn.template import TemplateEmbedding

_Checkpoint_Combine_Seq_Thre = 512  # if seq_len > 512aa, Double-layer gradient checkpointing for memory optimization.
_Diffusion_Parallel_Atom_Thre_train0 = 320  # 320aa
_Diffusion_Parallel_Atom_Thre_train1 = 541  # 541aa


def _make_no_cache_autocast_context():
    """Return a context manager that mirrors the current autocast state but disables its cache.

    When used as ``context_fn`` for ``torch.utils.checkpoint.checkpoint`` with
    ``use_reentrant=False``, this ensures the forward pass and the recompute
    pass produce exactly the same set of intermediate tensors (same order,
    same metadata).  Without this, the AMP autocast *cache* may satisfy a
    cache-hit during recompute that was a cache-miss during forward, shifting
    the saved-tensor positions and triggering ``CheckpointError``.
    """
    if torch_npu.npu.is_autocast_enabled():
        return torch_npu.npu.amp.autocast(
            enabled=True,
            dtype=torch_npu.npu.get_autocast_dtype(),
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
    def __init__(
            self,
            msa_channel: int = 64,
            checkpoint_config: dict = None,
            disable_internal_progress: bool = False,
            antibody_msa: bool = True,
            antibody_msa_drop_prob: float = 0.5,
            cross_chain_pair_dropout: bool = False,
            cross_chain_pair_dropout_prob: float = 0.1,
            cross_chain_pair_dropout_mode: str = "zero",  # "zero", "noise", "scale"
            cross_chain_pair_dropout_noise_scale: float = 1.0,
            cross_chain_pair_dropout_scale_factor: float = 0.1,
    ):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024  # //0401

        # Cross-chain pair dropout config
        self.cross_chain_pair_dropout = cross_chain_pair_dropout
        self.cross_chain_pair_dropout_prob = cross_chain_pair_dropout_prob
        self.cross_chain_pair_dropout_mode = cross_chain_pair_dropout_mode
        self.cross_chain_pair_dropout_noise_scale = cross_chain_pair_dropout_noise_scale
        self.cross_chain_pair_dropout_scale_factor = cross_chain_pair_dropout_scale_factor

        # Parse gradient-checkpointing configuration
        if checkpoint_config is None:
            checkpoint_config = {}

        # Pairformer checkpoint configuration
        pairformer_config = checkpoint_config.get('pairformer', {})
        self.pairformer_checkpoint_enabled = pairformer_config.get('enabled', False)
        self.pairformer_checkpoint_group_size = pairformer_config.get('group_size', 8)

        # MSA checkpoint configuration
        msa_config = checkpoint_config.get('msa', {})
        self.msa_checkpoint_enabled = msa_config.get('enabled', False)
        self.msa_checkpoint_group_size = msa_config.get('group_size', 4)

        # Template checkpoint configuration (default False to ensure template/prev_embedding gradients are correct)
        template_config = checkpoint_config.get('template', {})
        self.template_checkpoint_enabled = template_config.get('enabled', False)

        # Internal progress bar control
        self.disable_internal_progress = disable_internal_progress
        self.antibody_msa = antibody_msa
        self.antibody_msa_drop_prob = antibody_msa_drop_prob

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
            pair_channel=self.pair_channel, disable_internal_progress=disable_internal_progress)

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
        self.debug_grad_chain_enabled = os.environ.get("TROCHFOLD_DEBUG_GRAD_CHAIN", "0") == "1"
        self._debug_grad_report = {}

    def reset_debug_grad_report(self) -> None:
        self._debug_grad_report = {}

    def consume_debug_grad_report(self) -> dict[str, dict]:
        report = self._debug_grad_report
        self._debug_grad_report = {}
        return report

    def _probe_grad_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not self.debug_grad_chain_enabled or not torch.is_grad_enabled():
            return tensor

        self._debug_grad_report[name] = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'requires_grad': bool(tensor.requires_grad),
            'grad_seen': False,
            'grad_is_none': False,
            'grad_fn': type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None,
        }

        if tensor.requires_grad:
            def _hook(grad: torch.Tensor | None, _name: str = name) -> torch.Tensor | None:
                if grad is None:
                    self._debug_grad_report[_name].update({
                        'grad_seen': False,
                        'grad_is_none': True,
                        'grad_abs_mean': None,
                        'grad_norm': None,
                    })
                    return grad

                self._debug_grad_report[_name].update({
                    'grad_seen': True,
                    'grad_is_none': False,
                    'grad_abs_mean': float(grad.detach().abs().mean().cpu()),
                    'grad_norm': float(grad.detach().norm().cpu()),
                })
                return grad

            tensor.register_hook(_hook)

        return tensor

    def _relative_encoding(
            self, batch: feat_batch.Batch, pair_activations: torch.Tensor
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
            self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Pair embedding from sequence."""
        left_single = self.left_single(target_feat).unsqueeze(1)
        right_single = self.right_single(target_feat).unsqueeze(0)
        pair_activations = left_single + right_single

        mask = token_features.mask

        pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(0)).to(dtype=left_single.dtype)

        return pair_activations, pair_mask

    def _embed_bonds(
            self, batch: feat_batch.Batch, pair_activations: torch.Tensor
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
            templates: features.Templates,
            asym_id: torch.Tensor,
            pair_activations: torch.Tensor,
            pair_mask: torch.Tensor,
            is_checkpointed: bool = False,
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        dtype = pair_activations.dtype
        multichain_mask = (asym_id.unsqueeze(-1) ==
                           asym_id.unsqueeze(0)).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask,
            is_checkpointed=is_checkpointed,
        )
        pair_activations = pair_activations + template_act
        return pair_activations

    def _embed_process_msa(
            self,
            batch: feat_batch.Batch,
            msa_batch: features.MSA,
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

        # Evoformer MSA stack — optionally apply gradient checkpointing
        # Torchfold: checkpoint only when grad is enabled, to avoid redundancy and potential gradient issues in no-grad recycle
        if self.msa_checkpoint_enabled and self.training and torch.is_grad_enabled():
            for i in tqdm(range(0, len(self.msa_stack), self.msa_checkpoint_group_size),
                          desc="MSA Layers (checkpointed)", disable=self.disable_internal_progress):
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
            # No checkpointing (or inference mode): iterate layer by layer
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
        """
        Compute pair_init (z_ij^init) and single_init (s_i^init) ONCE before recycle loop.
        Returns: (pair_init, pair_mask, single_init)
        """
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )
        pair_activations = self._relative_encoding(batch, pair_activations)
        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )
        single_init = self.single_activations(target_feat)
        return pair_activations, pair_mask, single_init

    def _apply_cross_chain_pair_dropout(
            self,
            pair_activations: torch.Tensor,  # [N_token, N_token, C_pair]
            asym_id: torch.Tensor,  # [N_token]
    ) -> torch.Tensor:
        """
        Cross-chain Pair Representation Dropout.

        During training, with probability cross_chain_pair_dropout_prob,
        apply dropout to cross-chain pair representations, forcing the diffusion head
        cross-chain pair information is missing.

        During inference, dropout can be disabled (full information) or used with

        Three modes:
        - "zero": set cross-chain pair to zero (similar to conditioning dropout)
        - "noise": add Gaussian noise to cross-chain pair
        - "scale": scale cross-chain pair to a very small value
        """
        if not self.training or not self.cross_chain_pair_dropout:
            return pair_activations

        # Decide with probability whether to apply dropout for this sample
        if torch.rand((), device=pair_activations.device).item() > self.cross_chain_pair_dropout_prob:
            return pair_activations

        # Build cross-chain mask: [N_token, N_token], True = cross-chain positions
        cross_chain_mask = (asym_id.unsqueeze(1) != asym_id.unsqueeze(0))  # [N, N]
        cross_chain_mask = cross_chain_mask.unsqueeze(-1).to(dtype=pair_activations.dtype)  # [N, N, 1]

        mode = self.cross_chain_pair_dropout_mode
        if mode == "zero":
            # Set cross-chain positions to zero, intra-chain unchanged
            pair_activations = pair_activations * (1.0 - cross_chain_mask)
        elif mode == "noise":
            # Add Gaussian noise to cross-chain positions
            noise = torch.randn_like(pair_activations) * self.cross_chain_pair_dropout_noise_scale
            pair_activations = pair_activations + noise * cross_chain_mask
        elif mode == "scale":
            # Scale cross-chain positions
            scale = self.cross_chain_pair_dropout_scale_factor
            pair_activations = pair_activations * (1.0 - cross_chain_mask * (1.0 - scale))
        else:
            raise ValueError(f"Unknown cross_chain_pair_dropout_mode: {mode}")

        return pair_activations

    def forward(
            self,
            batch: dict[str, torch.Tensor],  # constant
            prev: dict[str, torch.Tensor],  # variable
            target_feat: torch.Tensor,  # constant
            idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        # pair_init (seq_pair + relpos + bonds) and single_init are computed
        # ONCE outside the recycle loop. When provided, use them; otherwise fallback to legacy.
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )
        pair_activations = self._probe_grad_tensor(
            f"recycle_{idx}.pair_after_seq_pair", pair_activations)
        pair_activations = pair_activations + self.prev_embedding(
            self.prev_embedding_layer_norm(prev['pair']))
        pair_activations = self._probe_grad_tensor(
            f"recycle_{idx}.pair_after_prev", pair_activations)
        pair_activations = self._relative_encoding(batch, pair_activations)
        pair_activations = self._probe_grad_tensor(
            f"recycle_{idx}.pair_after_relpos", pair_activations)
        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )
        pair_activations = self._probe_grad_tensor(
            f"recycle_{idx}.pair_after_bonds", pair_activations)
        pair_activations = self._embed_template_pair(
            templates=batch.templates,
            asym_id=batch.token_features.asym_id,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            is_checkpointed=self.template_checkpoint_enabled and self.training and torch.is_grad_enabled(),
        )
        pair_activations = self._probe_grad_tensor(
            f"recycle_{idx}.pair_after_template", pair_activations)

        pair_activations = self._embed_process_msa(
            batch=batch,
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
        )
        pair_activations = self._probe_grad_tensor(
            f"recycle_{idx}.pair_after_msa", pair_activations)

        # if single_init is not None:
        #     single_activations = single_init + self.prev_single_embedding(
        #         self.prev_single_embedding_layer_norm(prev['single']))
        # else:
        single_activations = self.single_activations(target_feat)
        single_activations = single_activations + self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single']))
        single_activations = self._probe_grad_tensor(
            f"recycle_{idx}.single_after_input", single_activations)
        # Pairformer trunk — enable checkpointing based on config
        # Torchfold: checkpoint only when grad is enabled (PairformerStack: if not torch.is_grad_enabled(): blocks_per_ckpt=None)
        if self.pairformer_checkpoint_enabled and self.training and torch.is_grad_enabled():
            # Use gradient checkpointing over grouped PairformerBlocks
            if pair_activations.size(
                    0) > _Checkpoint_Combine_Seq_Thre and fastnn_config.dot_product_attention_implementations == "torch":
                self.pairformer_checkpoint_group_size = 24
            else:
                self.pairformer_checkpoint_group_size = 2
            group_size = self.pairformer_checkpoint_group_size
            num_layers = len(self.trunk_pairformer)
            for i in tqdm(range(0, num_layers, group_size), desc="Pairformer Layers (checkpointed)",
                          disable=self.disable_internal_progress):
                end_idx = min(i + group_size, num_layers)
                group_layers = tuple(self.trunk_pairformer[i:end_idx])

                def pairformer_group_forward(pair_act, single_act, p_mask, seq_mask, _layers=group_layers):
                    if pair_act.size(0) > _Checkpoint_Combine_Seq_Thre and len(
                            _layers) > 1 and fastnn_config.dot_product_attention_implementations == "torch":
                        for layer in _layers:
                            pair_act, single_act = safe_checkpoint(
                                layer,
                                pair_act,
                                pair_mask,
                                single_act,
                                seq_mask,
                            )
                    else:
                        for layer in _layers:
                            pair_act, single_act = layer(
                                pair_act,
                                p_mask,
                                single_act,
                                seq_mask,
                            )
                    return pair_act, single_act

                # pair_activations, single_activations = checkpoint.checkpoint(pairformer_group_forward,
                #     pair_activations,
                #     single_activations,
                #     pair_mask,
                #     batch.token_features.mask,
                #     use_reentrant=False)

                pair_activations, single_activations = safe_checkpoint(
                    pairformer_group_forward,
                    pair_activations,
                    single_activations,
                    pair_mask,
                    batch.token_features.mask,
                )
        else:
            # No checkpointing (or inference): iterate through layers
            for layer in tqdm(self.trunk_pairformer, desc="Pairformer Layers", disable=self.disable_internal_progress):
                pair_activations, single_activations = layer(
                    pair=pair_activations,
                    pair_mask=pair_mask,
                    single=single_activations,
                    seq_mask=batch.token_features.mask,
                )

        # === Cross-chain Pair Representation Dropout ===
        # Apply dropout to cross-chain pair after Pairformer output, before passing to diffusion head
        # if self.cross_chain_pair_dropout:
        #     pair_activations = self._apply_cross_chain_pair_dropout(
        #         pair_activations=pair_activations,
        #         asym_id=batch.token_features.asym_id,
        #     )

        pair_activations = self._probe_grad_tensor(
            f"recycle_{idx}.pair_after_trunk", pair_activations)
        single_activations = self._probe_grad_tensor(
            f"recycle_{idx}.single_after_trunk", single_activations)

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output


class TorchFold(nn.Module):
    def __init__(
            self,
            num_recycles: int = 10,
            num_samples: int = 5,
            diffusion_steps: int = 200,
            mini_rollout_steps: int = 20,
            num_diffusion_samples_training: int = 48,
            checkpoint_config: dict = None,
            train_confidence: bool = True,
            save_diffusion_debug: bool = False,
            disable_internal_progress: bool = False,
            noise_configs: dict = None,
            noise_p_mean: float = -1.2,
            noise_p_std: float = 1.5,
            noise_p_mean_schedule: list = None,
            noise_p_mean_update_every: int = None,
            condition_embedding_drop_rate: float = 0.0,
            # Conditional dropout probability during training
            train_only_confidence: bool = False,  # Train only confidence head, skip diffusion training steps
            antibody_msa: bool = True,  # Provide MSA for antibody chains
            antibody_msa_drop_prob: float = 0.5,  # Probability of dropping MSA for antibody chains
            randomize_num_recycles: bool = False,
            # === Cross-chain Pair Representation Dropout ===
            cross_chain_pair_dropout: bool = False,
            cross_chain_pair_dropout_prob: float = 0.1,
            cross_chain_pair_dropout_mode: str = "zero",
            cross_chain_pair_dropout_noise_scale: float = 1.0,
            cross_chain_pair_dropout_scale_factor: float = 0.1,
    ):
        super(TorchFold, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.mini_rollout_steps = mini_rollout_steps
        self.train_only_confidence = train_only_confidence
        self._confidence_only_mode = False

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        # Parse gradient-checkpointing configuration
        if checkpoint_config is None:
            checkpoint_config = {}

        # Diffusion checkpoint configuration
        diffusion_config = checkpoint_config.get('diffusion', {})
        self.diffusion_checkpoint_enabled = diffusion_config.get('enabled', False)
        self.diffusion_checkpoint_group_size = diffusion_config.get('group_size', 8)

        # Confidence head checkpoint configuration
        confidence_config = checkpoint_config.get('confidence', {})
        self.confidence_checkpoint_enabled = confidence_config.get('enabled', False)
        self.train_confidence = train_confidence

        # Internal progress bar control
        self.disable_internal_progress = disable_internal_progress

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()

        self.evoformer = Evoformer(
            checkpoint_config=checkpoint_config,
            disable_internal_progress=disable_internal_progress,
            antibody_msa=antibody_msa,
            antibody_msa_drop_prob=antibody_msa_drop_prob,
            cross_chain_pair_dropout=cross_chain_pair_dropout,
            cross_chain_pair_dropout_prob=cross_chain_pair_dropout_prob,
            cross_chain_pair_dropout_mode=cross_chain_pair_dropout_mode,
            cross_chain_pair_dropout_noise_scale=cross_chain_pair_dropout_noise_scale,
            cross_chain_pair_dropout_scale_factor=cross_chain_pair_dropout_scale_factor,
        )

        self.diffusion_head = diffusion_head.DiffusionHead()

        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()
        # Optional debug flag to collect diffusion per-step coordinates
        self.save_diffusion_debug = save_diffusion_debug
        self.condition_embedding_drop_rate = condition_embedding_drop_rate

        # Training-time noise sampler
        self.train_noise_sampler = diffusion_head.TrainingNoiseSampler(
            p_mean=noise_p_mean,
            p_std=noise_p_std,
            sigma_data=16.0,
            p_mean_schedule=noise_p_mean_schedule,
            update_every=noise_p_mean_update_every,
        )
        self.num_diffusion_samples_training = num_diffusion_samples_training

        # Per-data-source noise parameters
        # noise_configs format: {data_source_idx: {'p_mean': float, 'p_std': float}}
        # Example: {0: {'p_mean': -2.5, 'p_std': 1.2}}  # data source 0 uses low noise
        self.noise_configs = noise_configs if noise_configs is not None else {}
        self.antibody_msa = antibody_msa
        self.antibody_msa_drop_prob = antibody_msa_drop_prob
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
        )
        self.debug_grad_chain_enabled = os.environ.get("TROCHFOLD_DEBUG_GRAD_CHAIN", "0") == "1"
        self._debug_grad_report = {}

    # Helper: reset submodule state before each batch
    def _reset_module_states(self):
        pass

    def _reset_debug_grad_report(self) -> None:
        self._debug_grad_report = {}
        if hasattr(self.evoformer, 'reset_debug_grad_report'):
            self.evoformer.reset_debug_grad_report()

    def _probe_grad_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not self.debug_grad_chain_enabled or not torch.is_grad_enabled():
            return tensor

        self._debug_grad_report[name] = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'requires_grad': bool(tensor.requires_grad),
            'grad_seen': False,
            'grad_is_none': False,
            'grad_fn': type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None,
        }

        if tensor.requires_grad:
            def _hook(grad: torch.Tensor | None, _name: str = name) -> torch.Tensor | None:
                if grad is None:
                    self._debug_grad_report[_name].update({
                        'grad_seen': False,
                        'grad_is_none': True,
                        'grad_abs_mean': None,
                        'grad_norm': None,
                    })
                    return grad

                self._debug_grad_report[_name].update({
                    'grad_seen': True,
                    'grad_is_none': False,
                    'grad_abs_mean': float(grad.detach().abs().mean().cpu()),
                    'grad_norm': float(grad.detach().norm().cpu()),
                })
                return grad

            tensor.register_hook(_hook)

        return tensor

    def consume_debug_grad_report(self) -> dict[str, dict]:
        report = dict(self._debug_grad_report)
        self._debug_grad_report = {}
        if hasattr(self.evoformer, 'consume_debug_grad_report'):
            evoformer_report = self.evoformer.consume_debug_grad_report()
            for key, value in evoformer_report.items():
                report[f"evoformer.{key}"] = value
        return report

    def _sample_training_num_recycles(self, device: torch.device) -> int:
        if (
                not self.training
                or self.train_only_confidence
                or not self.randomize_num_recycles
                or self.num_recycles <= 0
        ):
            return self.num_recycles

        sampled = torch.zeros(1, device=device, dtype=torch.int64)
        # All ranks must execute the same recycle depth in a DDP step.
        # Loop over range(actual_num_recycles+1), 0 means 1 iteration; Torchfold's N_cycle is number of iterations without +1
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

    def _run_confidence_head_chunks(
            self,
            dense_atom_positions: torch.Tensor,
            embeddings: dict[str, torch.Tensor],
            batch: feat_batch.Batch,
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

    def _set_confidence_only_mode(self, enabled: bool) -> None:
        """Freeze/unfreeze all params except confidence head when enabled."""
        if self._confidence_only_mode == enabled:
            return
        self._confidence_only_mode = enabled
        if enabled:
            for name, param in self.named_parameters():
                if name.startswith("confidence_head."):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True

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
            batch: feat_batch.Batch,
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
                      torch.sqrt(t_hat ** 2 - noise_level_prev ** 2)
        noise = noise_scale * \
                torch.randn(size=positions.shape, device=noise_scale.device)
        # noise = noise_scale
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
            batch: feat_batch.Batch,
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
            batch: feat_batch.Batch,
            embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample using denoiser on batch."""

        mask = batch.predicted_structure_info.atom_mask

        # === Fix 1: Remove extra batch dimension ===
        # When validation batch_size=1, mask may be (1, N, 24), need to become (N, 24)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        num_samples = self.num_samples
        device = mask.device

        # === Fix 2: Restore dtype=torch.float32 (consistent with inference) ===
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device, dtype=torch.float32))  ###(201)

        # === Fix 3: Initial noise positions also use float32 ===
        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device, dtype=torch.float32)  ###[num_samples, N_token, 24, 3]
        positions *= noise_levels[0]  ###[num_samples, N_token, 24, 3]

        per_step_all = [] if self.save_diffusion_debug else None
        for sample_idx in range(num_samples):
            per_step_positions = [] if self.save_diffusion_debug else None
            for step_idx in trange(self.diffusion_steps, desc=f"Diffusion sample-{sample_idx}",
                                   disable=self.disable_internal_progress):
                positions[sample_idx] = self._apply_denoising_step(
                    batch,
                    embeddings,
                    positions[sample_idx],
                    noise_levels[step_idx],
                    mask,
                    noise_levels[1 + step_idx],
                )
                if per_step_positions is not None:
                    per_step_positions.append(positions[sample_idx].detach().cpu())
            if per_step_positions is not None:
                per_step_all.append(
                    per_step_positions)  ###per_step_positions [[N_token, 24, 3],[N_token, 24, 3]........]
        final_dense_atom_mask = torch.tile(mask[None], (num_samples, 1, 1))
        out = {'atom_positions': positions, 'mask': final_dense_atom_mask}
        if self.save_diffusion_debug and 'per_step_all' in locals() and per_step_all is not None:
            out['per_step_positions'] = per_step_all
        return out

    def _sample_diffusion_training(
            self,
            batch: feat_batch.Batch,
            embeddings: dict[str, torch.Tensor],
            gt_positions: torch.Tensor,  # [N_token, 24, 3]
            gt_mask: torch.Tensor,  # [N_token, 24]
            N_sample: int = None,
            data_source_idx: int = None,  # Data source index, used to select noise distribution
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training-time diffusion sampling with single-step denoising.

        Args:
            batch: Input batch.
            embeddings: Evoformer outputs.
            gt_positions: Ground-truth coordinates [N_token, 24, 3].
            gt_mask: Ground-truth mask [N_token, 24].
            N_sample: Number of training samples (defaults to `self.num_diffusion_samples_training`).
            data_source_idx: Index of the data source (used for per-source noise distribution).

        Returns:
            tuple of:
                - x_gt_augment: Augmented ground truth [N_sample, N_token, 24, 3]
                - x_noisy: Noisy coordinates [N_sample, N_token, 24, 3]
                - x_denoised: Denoised coordinates [N_sample, N_token, 24, 3]
                - sigma: Noise levels [N_sample]
                - t_values: Sampled t in (0, 1) [N_sample]
        """
        # if N_sample is None:
        #     N_sample = self.num_diffusion_samples_training

        device = gt_positions.device
        dtype = gt_positions.dtype

        # 1. Create `N_sample` augmented versions (random rotation + translation)
        # gt_positions: [N_token, 24, 3] -> x_gt_augment: [N_sample, N_token, 24, 3]
        x_gt_augment = diffusion_head.centre_random_augmentation(
            x_input_coords=gt_positions,
            N_sample=N_sample,
            mask=gt_mask,
        ).to(dtype)  ###[N_sample, N_token, 24, 3]

        # 2. Sample independent noise levels per example (log-normal distribution)
        # Select different noise distribution parameters according to data source
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
            # Use default noise distribution
            sigma, t_values = self.train_noise_sampler(
                size=(N_sample,),
                device=device,
                return_t=True,
            )
            sigma = sigma.to(dtype)
            t_values = t_values.to(dtype)  ### [N_sample]

        # 3. Add independent Gaussian noise
        # noise: [N_sample, N_token, 24, 3]
        noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_noisy = x_gt_augment + noise  ###[N_sample, N_token, 24, 3]

        # 4. Single-step denoising via the diffusion head
        # Conditioning dropout: drop all conditioning for this batch with prob p

        use_conditioning = True  # not drop_conditioning

        # Note: diffusion head expects [N_token, 24, 3], so iterate per sample
        x_denoised_list = []
        if fastnn_config.dot_product_attention_implementations == "Fusion_Attention":
            if batch.num_res <= _Diffusion_Parallel_Atom_Thre_train0:
                N_sample_step = min(16, N_sample)
            else:
                N_sample_step = min(8, N_sample)
        else:
            if batch.num_res <= _Diffusion_Parallel_Atom_Thre_train0:
                N_sample_step = min(8, N_sample)
            elif _Diffusion_Parallel_Atom_Thre_train0 < batch.num_res < _Diffusion_Parallel_Atom_Thre_train1:
                N_sample_step = min(4, N_sample)
            else:
                N_sample_step = min(2, N_sample)
        if self.diffusion_checkpoint_enabled:
            for i in range(0, N_sample, N_sample_step):
                i_end = min(N_sample, i + N_sample_step)
                x_denoised_i = safe_diffusion_checkpoint(
                    self.diffusion_head,
                    positions_noisy=x_noisy[i:i_end],  # [N_token, 24, 3]
                    noise_level=sigma[i:i_end],  # scalar for every sample
                    batch=batch,
                    embeddings=embeddings,
                    use_conditioning=use_conditioning,
                )
                x_denoised_list.append(x_denoised_i)
        else:
            for i in range(0, N_sample, N_sample_step):
                i_end = i + N_sample_step
                x_denoised_i = self.diffusion_head(
                    positions_noisy=x_noisy[i:i_end],  # [N_token, 24, 3]
                    noise_level=sigma[i:i_end],  # scalar for every sample
                    batch=batch,
                    embeddings=embeddings,
                    use_conditioning=use_conditioning
                )
                x_denoised_list.append(x_denoised_i)

        x_denoised = einops.rearrange(torch.stack(x_denoised_list, dim=0),
                                      'a b h w c -> (a b) h w c')  # [N_sample, N_token, 24, 3]

        return x_gt_augment, x_noisy, x_denoised, sigma, t_values

    def _sample_diffusion_mini_rollout(
            self,
            batch: feat_batch.Batch,
            embeddings: dict[str, torch.Tensor],
            num_steps: int = None,
    ) -> dict[str, torch.Tensor]:
        """
        Mini-rollout: short denoising trajectories from pure noise for training the
        confidence head. Runs under `torch.no_grad()`.

        Args:
            batch: Input batch.
            embeddings: Evoformer outputs.
            num_steps: Number of mini-rollout steps (defaults to `self.mini_rollout_steps`).

        Returns:
            Dictionary containing `atom_positions` and `mask`.
        """
        if num_steps is None:
            num_steps = self.mini_rollout_steps

        mask = batch.predicted_structure_info.atom_mask
        # Typical mini-rollout produces one sample for the confidence loss, but we reuse
        # `self.num_diffusion_samples_training` instead.
        num_samples = 1
        # num_samples = self.num_diffusion_samples_training
        device = mask.device

        # Build a noise schedule (only need `num_steps+1` points)
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, num_steps + 1, device=device)
        )

        # Start from pure noise
        positions = torch.randn((num_samples,) + mask.shape + (3,), device=device)
        positions = positions * noise_levels[0]

        # Run mini-rollout without gradients
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
    ) -> dict[str, torch.Tensor]:
        """
        forward pass.
        """
        # Important: reset state before processing a new batch

        self._reset_module_states()
        self._reset_debug_grad_report()
        if self.training and self.train_only_confidence:
            self._set_confidence_only_mode(True)
        else:
            self._set_confidence_only_mode(False)

        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res

        target_feat = self.create_target_feat_embedding(batch)  ###batch_size is not supported

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
        recycle_grad_enabled = self.training and (not self.train_only_confidence)
        if self.debug_grad_chain_enabled:
            self._debug_grad_report['meta.actual_num_recycles'] = {
                'value': int(actual_num_recycles),
            }
            self._debug_grad_report['meta.recycle_loop_count'] = {
                'value': int(actual_num_recycles),
            }

        for i in range(actual_num_recycles + 1):
            with torch.set_grad_enabled(recycle_grad_enabled and i == actual_num_recycles):
                embeddings = self.evoformer(
                    batch=batch,
                    prev=embeddings,
                    target_feat=target_feat,
                    idx=i,
                )

        embeddings['pair'] = self._probe_grad_tensor(
            'torchfold.embeddings_pair_pre_heads', embeddings['pair'])
        embeddings['single'] = self._probe_grad_tensor(
            'torchfold.embeddings_single_pre_heads', embeddings['single'])

        # t2 = time.time()
        # _t_diff_start = _t_diff_end = _t_mini_start = _t_mini_end = time.time()  # 0310_test_time: defaults
        # ========== Diffusion pipeline for training vs inference ==========
        if self.training:
            # Force mini-rollout steps when training confidence
            mini_rollout_steps = 20 if self.train_only_confidence else self.mini_rollout_steps
            if self.train_only_confidence:
                # When training confidence only, skip expensive diffusion training sampling (single-step denoising)
                # In this case, diffusion loss is not computed
                samples = {}

                # If mini-rollout is not enabled, confidence head still needs some structures as input
                # Here we default to using ground truth (with a dummy dimension) to save memory and computation
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
                # Normal diffusion + confidence training pipeline
                gt_positions = batch.ground_truth_structure.true_positions  ### [N_token,24, 3], N_atom=24
                gt_mask = batch.ground_truth_structure.true_positions_atom_mask  ### [N_token,24]

                # Get data source index (if exists)
                data_source_idx = None
                if hasattr(batch, 'data_source_idx') and batch.data_source_idx is not None:
                    data_source_idx = batch.data_source_idx.item() if isinstance(batch.data_source_idx,
                                                                                 torch.Tensor) else batch.data_source_idx

                # Training diffusion (single-step denoising, multiple samples)
                x_gt_augment, x_noisy, x_denoised, noise_levels, t_values = self._sample_diffusion_training(
                    batch=batch,
                    embeddings=embeddings,
                    gt_positions=gt_positions,
                    gt_mask=gt_mask,
                    N_sample=self.num_diffusion_samples_training,
                    data_source_idx=data_source_idx,
                )
                x_denoised = self._probe_grad_tensor(
                    'torchfold.x_denoised_for_loss', x_denoised)
                # torch.cuda.synchronize()  # 0310_test_time
                # _t_diff_end = time.time()  # 0310_test_time

                # Samples used for the diffusion loss
                samples = {
                    'atom_positions': x_denoised,  ### [N_sample, N_token, 24, 3]
                    'mask': gt_mask.unsqueeze(0).expand(self.num_diffusion_samples_training, -1, -1),
                    ### [N_sample, N_token, 24]
                    'gt_positions': x_gt_augment,  ### Augmented ground truth  [N_sample, N_token, 24, 3]
                    'noise_levels': noise_levels,  ### Noise levels [N_sample]
                    't_values': t_values,  ### Sampled t in (0, 1) [N_sample]
                }

                # debug dump: keep noisy/augmented samples for one-off debug dump
                if os.environ.get("TROCHFOLD_DEBUG_0116", "0") == "1":
                    samples['x_noisy'] = x_noisy

                # Only add debug fields when save_diffusion_debug is enabled
                # This avoids creating unnecessary references that prevent memory release
                # Commented out to avoid DDP errors: these fields are duplicates of atom_positions/gt_positions
                # and don't participate in loss calculation, causing DDP to complain
                # if self.save_diffusion_debug:
                #     samples['x_gt_augment'] = x_gt_augment
                #     samples['x_noisy'] = x_noisy
                #     samples['x_denoised'] = x_denoised

                # Mini-rollout for the confidence head
                if mini_rollout_steps > 0:
                    samples_for_confidence = self._sample_diffusion_mini_rollout(
                        batch, embeddings, num_steps=mini_rollout_steps
                    )
                else:
                    # Without mini-rollout, reuse all training samples
                    samples_for_confidence = {
                        'atom_positions': x_denoised,  # Use every sample
                        'mask': gt_mask.unsqueeze(0).expand(self.num_diffusion_samples_training, -1, -1),
                    }
        else:

            samples = self._sample_diffusion(batch, embeddings)
            samples_for_confidence = samples

        # t3 = time.time()
        # logger.info(f"Time taken for Diffusion: {t3 - t2:.4f}s")

        # Confidence head: mini-rollout outputs during training, full diffusion otherwise
        if (self.training and self.train_confidence) or not self.training:
            confidence_output_per_sample = []
            desc = "Confidence (checkpointed)" if (
                    self.confidence_checkpoint_enabled and self.training) else "Confidence"

            for sample_dense_atom_position in tqdm(samples_for_confidence['atom_positions'], desc=desc,
                                                   disable=self.disable_internal_progress):
                if self.confidence_checkpoint_enabled and self.training:
                    def confidence_forward(atom_pos):
                        return self.confidence_head(
                            dense_atom_positions=atom_pos,
                            embeddings=embeddings,
                            seq_mask=batch.token_features.mask,
                            token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                            asym_id=batch.token_features.asym_id
                        )

                    confidence_output = checkpoint.checkpoint(
                        confidence_forward, sample_dense_atom_position, use_reentrant=True
                    )
                else:
                    confidence_output = self.confidence_head(
                        dense_atom_positions=sample_dense_atom_position,
                        embeddings=embeddings,
                        seq_mask=batch.token_features.mask,
                        token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                        asym_id=batch.token_features.asym_id
                    )

                confidence_output_per_sample.append(confidence_output)

            confidence_output = {}
            if confidence_output_per_sample:
                for key in confidence_output_per_sample[0]:
                    confidence_output[key] = torch.stack(
                        [sample[key] for sample in confidence_output_per_sample], dim=0)
        else:
            # Skip confidence head if disabled during training
            confidence_output = {}

        # t4 = time.time()
        # logger.info(f"Time taken for Confidence: {t4 - t3:.4f}s")
        distogram = self.distogram_head(batch, embeddings)
        if isinstance(distogram, dict):
            if 'logits' in distogram:
                distogram['logits'] = self._probe_grad_tensor(
                    'torchfold.distogram_logits', distogram['logits'])
            if 'contact_probs' in distogram:
                distogram['contact_probs'] = self._probe_grad_tensor(
                    'torchfold.distogram_contact_probs', distogram['contact_probs'])

        output = {
            'distogram': distogram,
            **confidence_output,
        }
        if self.training and self.train_only_confidence:
            output['pred_coords'] = samples_for_confidence['atom_positions']
            output['confidence_only'] = True
        else:
            output['diffusion_samples'] = samples  # For the diffusion loss

        output['confidence_atom_positions'] = samples_for_confidence['atom_positions']

        # DDP static_graph + randomize_num_recycles: force all trainable parameters to participate in backward,
        # making the gradient graph consistent each step, thus safely enabling static_graph (avoiding "marked twice" and "graph changed")
        # if self.training and self.randomize_num_recycles:
        #     anchor = sum((p * 0.0).sum() for p in self.parameters() if p.requires_grad)
        #     output['ddp_static_anchor'] = anchor

        return output
