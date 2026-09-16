import time

import torch
import torch.distributed as dist
import torch.nn as nn
from checkpoint_reentrant_function import CheckpointReentrantFunction
from loguru import logger
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm, trange

from torchcraft import feat_batch, features
from torchcraft.nn import atom_cross_attention
from torchcraft.nn import diffusion_head
from torchcraft.nn import featurization
from torchcraft.nn.head import DistogramHead, ConfidenceHead
from torchcraft.nn.layer_norm import LayerNorm
from torchcraft.nn.pairformer import EvoformerBlock, PairformerBlock
from torchcraft.nn.template import TemplateEmbedding

USE_DIST=0


class PairformerTmp(nn.Module):
    def __init__(self, pairformer_checkpoint_interval: int = 4):
        super().__init__()
        self.pairformer_checkpoint_interval = pairformer_checkpoint_interval  # Checkpoint every N Pairformer layers
        self.pair_activations = None
        self.pair_mask = None
    
    def forward(self, trunk_pairformer, pair_activations, pair_mask, single_activations, token_mask):
        """Helper function that wraps the Pairformer stack with checkpointing, supports grouped fine-grained control"""
        # Use grouped checkpoint
        num_layers = len(trunk_pairformer)
        
        def pairformer_group_checkpoint_fn(pair_act, single_act, pair_m, token_m, layer_group):
            """Checkpoint function for processing a group of Pairformer layers"""
            current_p = pair_act
            current_s = single_act
            for layer in layer_group:
                current_p, current_s = layer(current_p, pair_m, current_s, token_m)
            return current_p, current_s

        for group_start in range(0, num_layers, self.pairformer_checkpoint_interval):
            group_end = min(group_start + self.pairformer_checkpoint_interval, num_layers)
            group_layers = trunk_pairformer[group_start:group_end]

            pair_activations, single_activations =  CheckpointReentrantFunction.apply(
                pairformer_group_checkpoint_fn,
                pair_activations, 
                single_activations, 
                pair_mask, 
                token_mask,
                group_layers
            )
        
        return pair_activations, single_activations


class Evoformer(nn.Module):
    def __init__(self, msa_channel: int = 64, use_gradient_checkpointing: bool = True, 
                 pairformer_checkpoint_interval: int = 4):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Fine-grained checkpoint control parameters
        self.pairformer_checkpoint_interval = pairformer_checkpoint_interval  # Checkpoint every N Pairformer layers

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
            )

        self.msa_activations = nn.Linear(34, self.msa_channel, bias=False)
        self.extra_msa_target_feat = nn.Linear(
            self.c_target_feat, self.msa_channel, bias=False)
        self.msa_stack = nn.ModuleList(
            [EvoformerBlock() for _ in range(self.msa_stack_num_layer)])

        self.single_activations = nn.Linear(
            self.c_target_feat, self.seq_channel, bias=False)

        self.prev_single_embedding_layer_norm = LayerNorm(self.seq_channel)
        self.prev_single_embedding = nn.Linear(
            self.seq_channel, self.seq_channel, bias=False)

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock(with_single=True) for _ in range(self.pairformer_num_layer)])
        self.pairformer_stack = PairformerTmp(pairformer_checkpoint_interval)
        self.first_run = True
        self.pair_activations = None
        self.pair_mask = None
        self.rel_feat = None
        self.bonds_act = None

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        if self.first_run:
            self.rel_feat = featurization.create_relative_encoding(
                batch.token_features,
                max_relative_idx,
                max_relative_chain,
            ).to(dtype=pair_activations.dtype)
            self.rel_feat = self.position_activations(self.rel_feat)

        pair_activations = pair_activations + self.rel_feat
        return pair_activations

    def _seq_pair_embedding(
        self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated Pair embedding from sequence."""
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
        # Construct contact matrix.
        if self.first_run:
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
                    dtype=gather_idxs_polymer_ligand.dtype).unsqueeze(1)
            )
            # If valid mask then it will be all 1's, so idxs should be unchanged.
            gather_idxs_polymer_ligand = (
                gather_idxs_polymer_ligand * gather_mask_polymer_ligand
            )

            tokens_to_ligand_ligand_bonds = (
                batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
            )
            gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
            gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
                dim=1
            ).to(dtype=gather_idxs_ligand_ligand.dtype).unsqueeze(1)
            gather_idxs_ligand_ligand = (
                gather_idxs_ligand_ligand * gather_mask_ligand_ligand
            )

            gather_idxs = torch.concatenate(
                [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
            )
            contact_matrix[
                gather_idxs[:, 0], gather_idxs[:, 1]
            ] = 1.0

            # Because all the padded index's are 0's.
            contact_matrix[0, 0] = 0.0

            bonds_act = self.bond_embedding(contact_matrix.unsqueeze(2))
            self.bonds_act = bonds_act

        return pair_activations + self.bonds_act


    def _embed_template_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        use_gradient_checkpointing: int
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        templates = batch.templates
        asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        multichain_mask = (asym_id.unsqueeze(1) ==
                           asym_id.unsqueeze(0)).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        return pair_activations + template_act

    def _embed_process_msa(
        self, msa_batch: features.MSA,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
        use_gradient_checkpointing: int
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype

        msa_batch = featurization.shuffle_msa(msa_batch)
        msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations = msa_activations + self.extra_msa_target_feat(target_feat).unsqueeze(0)

        # Evoformer MSA stack - use coarse-grained mode, no checkpoint inside
        for msa_block in tqdm(self.msa_stack, desc="MSA stack"): 
           
            if use_gradient_checkpointing:
                msa_activations, pair_activations = checkpoint(msa_block,
                    msa=msa_activations,
                    pair=pair_activations,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                    use_reentrant=False
                )
            else:
                msa_activations, pair_activations = msa_block(
                    msa=msa_activations,
                    pair=pair_activations,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )

        return pair_activations

    def forward(
        self,
        batch: dict[str, torch.Tensor],  # constant
        prev: dict[str, torch.Tensor],  # variable
        target_feat: torch.Tensor,  # constant
        idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        self.first_run = idx==0
        if self.first_run:
            self.pair_activations, self.pair_mask = self._seq_pair_embedding(batch.token_features, target_feat)
        pair_activations = self.pair_activations.clone()

        pair_activations = pair_activations + self.prev_embedding(
            self.prev_embedding_layer_norm(prev['pair']))

        pair_activations = self._relative_encoding(batch, pair_activations)

        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )

        # Template embedding - Uses the same coarse-grained checkpoint structure as MSA
        pair_activations = self._embed_template_pair(
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=self.pair_mask,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        # MSA processing with optional checkpoint
        pair_activations = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=self.pair_mask,
            target_feat=target_feat,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        single_activations = self.single_activations(target_feat)
        single_activations = single_activations + self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single']))

        # Pairformer stack with fine-grained checkpoint control
        if self.use_gradient_checkpointing:
            pair_activations, single_activations = self.pairformer_stack(
                    self.trunk_pairformer,
                    pair_activations, 
                    self.pair_mask, 
                    single_activations, 
                    batch.token_features.mask
            )
        else:
            # Original loop when checkpoint is not used
            for _, pairformer_b in enumerate(tqdm(self.trunk_pairformer, desc=f"Pairformer {idx}")):
                pair_activations, single_activations = pairformer_b(
                    pair_activations, self.pair_mask, single_activations, batch.token_features.mask)

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }
        return output


class TorchFold(nn.Module):
    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200,
                 use_gradient_checkpointing: bool = True, pairformer_checkpoint_interval: int = 4,
                 turn_off_diffusion_confidence: bool = True,
                 stage3_diffusion_steps: int = 20):
        super(TorchFold, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.pairformer_checkpoint_interval = pairformer_checkpoint_interval
        self.turn_off_diffusion_confidence = turn_off_diffusion_confidence

        self.diffusion_steps = diffusion_steps
        self.stage3_diffusion_steps = stage3_diffusion_steps

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()

        self.evoformer = Evoformer(
            use_gradient_checkpointing=use_gradient_checkpointing,
            pairformer_checkpoint_interval=pairformer_checkpoint_interval
        )

        self.diffusion_head = diffusion_head.DiffusionHead()

        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()


        # Add attribute for tracking epoch
        self.current_epoch = 0
        self.stage_epochs = [0, 0, 0, 100]  # Default configuration

    def _checkpoint_evoformer_step(self, batch, prev_embeddings, target_feat, idx):
        """Checkpointable evoformer step"""
        return self.evoformer(
            batch=batch,
            prev=prev_embeddings,
            target_feat=target_feat,
            idx=idx,
        )

    def create_target_feat_embedding(self, batch: dict[str, torch.Tensor], binder_logits_20) -> torch.Tensor:
        target_feat = featurization.create_target_feat(
            batch,
            binder_logits_20,
            append_per_atom_features=False,
            current_epoch=self.current_epoch,  # Pass epoch information
            stage_epochs=self.stage_epochs,  # Pass stage configuration
        )

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat

    def set_epoch_info(self, current_epoch: int, stage_epochs: list):
        """Set current epoch information for four-stage optimization"""
        self.current_epoch = current_epoch
        self.stage_epochs = stage_epochs

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
        # gamma = gamma.to(torch.bfloat16) # self.gamma_0 is float32
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
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

    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample using denoiser on batch."""

        mask = batch.predicted_structure_info.atom_mask
        num_samples = self.num_samples

        device = mask.device

        ############################################################
        # Dynamically determine diffusion steps: judge the current stage based on epoch
        stage1_end = self.stage_epochs[0]
        stage2_end = stage1_end + self.stage_epochs[1]
        stage3_end = stage2_end + self.stage_epochs[2]
        # stage4 is after stage3_end
        
        if self.current_epoch < stage3_end:
            # Use fewer steps for stages 1/2/3
            current_diffusion_steps = self.stage3_diffusion_steps
        else:
            # Use full steps for stage 4
            current_diffusion_steps = self.diffusion_steps
        
        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, current_diffusion_steps + 1, device=device, dtype=torch.float32))

        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device, dtype=torch.float32) #bfloat16
        positions = positions * noise_levels[0]

        # assert self.diffusion_steps == 200

        if USE_DIST:
            assert num_samples == 5, "Distributed mode only supports 5 samples" 
            rk, ws = dist.get_rank(), dist.get_world_size()

            assert ws <= 5, f"World size {ws} is currently not supported. Please set world size to 5 or less."

            # Scheme description:
            # 1. solution[ws][rk] is a list of tuples (sample_idx, begin, end), means that
            #    `rk` will run `positions[sample_idx]` for [begin, end) when world size is `ws`.
            # 2. solution[ws][ws] is a list of tuples (from_rk, sample_id), means that
            #    `positions[sample_id]` be broadcasted from `from_rk` to all other ranks.
            # Note that currently the workflow is handcrafted for at most 5 ranks.
            # The main purpose is to reduce the number of broadcast operations.

            solution = [[] for _ in range(6)]

            solution[1].append([(i, 0, 200) for i in range(5)])
            solution[1].append([() for _ in range(5)])

            solution[2].append([(0, 0, 200), (2, 0, 100), (4, 0, 100), (4, 100, 200)])
            solution[2].append([(1, 0, 200), (3, 0, 100), (3, 100, 200), (2, 100, 200)])
            solution[2].append([(), (), ((0, 2),), ((1, 1), (1, 2), (1, 3), (0, 0), (0, 4))])

            solution[3].append([(0, 0, 66), (3, 0, 66), (4, 0, 66), (4, 66, 200)])
            solution[3].append([(1, 0, 66), (1, 66, 134), (1, 134, 200), (3, 66, 200)])
            solution[3].append([(2, 0, 66), (2, 66, 134), (2, 134, 200), (0, 66, 200)])
            solution[3].append([(), (), ((0, 3), (0, 0)), ((1, 3), (1, 1), (2, 0), (2, 2), (0, 4))])

            solution[4].append([(0, 0, 50), (4, 0, 50), (4, 50, 100), (4, 100, 150), (4, 150, 200)])
            solution[4].append([(1, 0, 50), (1, 50, 100), (0, 50, 100), (0, 100, 150), (0, 150, 200)])
            solution[4].append([(2, 0, 50), (2, 50, 100), (2, 100, 150), (1, 100, 150), (1, 150, 200)])
            solution[4].append([(3, 0, 50), (3, 50, 100), (3, 100, 150), (3, 150, 200), (2, 150, 200)])
            solution[4].append([(), ((0, 0), (1, 1)), (), ((2, 2), (3, 3)), ((0, 4), (1, 0), (2, 1), (3, 2))])

            solution[5].append([(0, 0, 200)])
            solution[5].append([(1, 0, 200)])
            solution[5].append([(2, 0, 200)])
            solution[5].append([(3, 0, 200)])
            solution[5].append([(4, 0, 200)])
            solution[5].append([((0, 0), (1, 1), (2, 2), (3, 3), (4, 4))])

            for idx, (sample_idx, begin, end) in enumerate(solution[ws][rk]):
                logger.info(f"Diffusion {idx}, running for {sample_idx} from {begin} to {end}")
                for step_idx in trange(begin, end):
                    positions[sample_idx] = self._apply_denoising_step(
                        batch,
                        embeddings,
                        positions[sample_idx],
                        noise_levels[step_idx],
                        mask,
                        noise_levels[1 + step_idx],
                    )
                for from_rk, sample_id in solution[ws][ws][idx]:
                    logger.info(f"Diffusion {idx}, broadcasting {sample_id} from {from_rk}")
                    dist.broadcast(positions[sample_id], src=from_rk)
        else:
            for sample_idx in range(num_samples):
                for step_idx in trange(current_diffusion_steps, desc=f"Diffusion {sample_idx}"):
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

    def forward(self, batch: dict[str, torch.Tensor],binder_logits_20) -> dict[str, torch.Tensor]:
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res
        # Manually initialize first_run on each forward pass
        self.evoformer_conditioning.first_run = True
        self.evoformer_conditioning.atom_transformer_encoder.first_run = True
        for pair in self.evoformer.trunk_pairformer:
            pair.single_attention_.first_run = True
            
        target_feat = self.create_target_feat_embedding(batch, binder_logits_20)

        embeddings = {
            'pair': torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
            ),
            'single': torch.zeros(
                [num_res, self.evoformer_seq_channel], device=target_feat.device, 
            ),
            'target_feat': target_feat,  # type: ignore
        }

        t1 = time.time()
        rk = dist.get_rank() if USE_DIST else 0
        logger.info(f"rank {rk}: Start running Evoformer with gradient checkpointing: {self.use_gradient_checkpointing}")

        for i in range(self.num_recycles):
            embeddings = self.evoformer(
                batch=batch,
                prev=embeddings,
                target_feat=target_feat,
                idx=i,
            )
        if self.turn_off_diffusion_confidence:
            t2 = time.time()
            logger.info(f"Time taken for Evoformer: {t2 - t1:.4f}s")
            
            distogram = self.distogram_head(batch, embeddings)
            logger.info(f"Time taken for Distogram: {time.time() - t2:.4f}s")
            return {
                'distogram': distogram,
            }
        else:
            t2 = time.time()

            # Manually initialize first_run before each diffusion round
            self.diffusion_head.first_run = True
            self.diffusion_head.atom_cross_att_encoder.first_run = True
            self.diffusion_head.atom_cross_att_encoder.atom_transformer_encoder.first_run = True
            self.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder.first_run = True
            self.diffusion_head.transformer.first_run = True
            for sa in self.diffusion_head.transformer.self_attention:
                sa.first_run = True
            self.diffusion_head.transformer.pair_logits_list = []
            for pair in self.confidence_head.confidence_pairformer:
                pair.single_attention_.first_run = True
            

            logger.info(f"Time taken for Evoformer: {t2 - t1:.4f}s")
            samples = self._sample_diffusion(batch, embeddings)
            t3 = time.time()
            logger.info(f"Time taken for Diffusion: {t3 - t2:.4f}s")

            confidence_output_per_sample = []
            for sample_dense_atom_position in tqdm(samples['atom_positions'], desc="Confidence"):
                confidence_output_per_sample.append(self.confidence_head(
                    dense_atom_positions=sample_dense_atom_position,
                    embeddings=embeddings,
                    seq_mask=batch.token_features.mask,
                    token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                    asym_id=batch.token_features.asym_id,
                    use_gradient_checkpointing=self.use_gradient_checkpointing
                ))

            confidence_output = {}
            for key in confidence_output_per_sample[0]:
                confidence_output[key] = torch.stack(
                    [sample[key] for sample in confidence_output_per_sample], dim=0)

            t4 = time.time()
            logger.info(f"Time taken for Confidence: {t4 - t3:.4f}s")

            if rk != 0:
                logger.info(f"Skip distogram head for rank {rk}")
                return None

            distogram = self.distogram_head(batch, embeddings)
            logger.info(f"Time taken for Distogram: {time.time() - t4:.4f}s")

            return {
                'diffusion_samples': samples,
                'distogram': distogram,
                **confidence_output,
            }
