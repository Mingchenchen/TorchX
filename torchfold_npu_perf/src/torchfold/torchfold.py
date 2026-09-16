import torch
import torch.nn as nn
from tqdm import tqdm, trange

from torchfold import feat_batch, features
from torchfold.nn import atom_cross_attention
from torchfold.nn import diffusion_head
from torchfold.nn import featurization
from torchfold.nn.confidence_utils import stack_confidence_outputs, offload_confidence_output
from torchfold.runtime_policy import (
    with_offload_policy,
    implicit_pair_enabled,
    large_pair_offload,
    add_previous_pair,
    SINGLE_CARD_SAMPLE_PARALLEL_THRESHOLD,
    SINGLE_CARD_DIFFUSION_SAMPLE_GROUP_SIZE,
    single_card_sample_parallel,
)
from torchfold.nn.head import DistogramHead, ConfidenceHead
from torchfold.nn.layer_norm import LayerNorm
from torchfold.nn.pairformer import EvoformerBlock, PairformerBlock
from torchfold.nn.template import TemplateEmbedding


# Exclusive threshold: single-card sample batching supports up to 3250.


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

        self.prev_embedding_layer_norm = LayerNorm(self.pair_channel)
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

        self.prev_single_embedding_layer_norm = LayerNorm(self.seq_channel)
        self.prev_single_embedding = nn.Linear(
            self.seq_channel, self.seq_channel, bias=False)

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock(with_single=True) for _ in range(self.pairformer_num_layer)])

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            max_relative_idx,
            max_relative_chain,
            dtype=pair_activations.dtype,
        )

        pair_activations += self.position_activations(rel_feat)
        return pair_activations

    def _seq_pair_embedding(
        self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_single = self.left_single(target_feat).unsqueeze(1)
        right_single = self.right_single(target_feat).unsqueeze(0)
        pair_activations = left_single + right_single

        mask = token_features.mask

        pair_mask = (mask.unsqueeze(-1) * mask.unsqueeze(0)).to(dtype=left_single.dtype)

        return pair_activations, pair_mask

    def _embed_bonds(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations."""
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device)

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).to(
                dtype=gather_idxs_polymer_ligand.dtype).unsqueeze(-1)
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

        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0

        bonds_act = self.bond_embedding(contact_matrix.unsqueeze(-1))

        return pair_activations + bonds_act

    def _embed_template_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        templates = batch.templates
        asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        multichain_mask = (asym_id.unsqueeze(-1) ==
                           asym_id.unsqueeze(0)).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask
        )

        return pair_activations + template_act

    def _embed_process_msa(
        self, msa_batch: features.MSA,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype

        msa_batch = featurization.shuffle_msa(msa_batch)
        msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat).unsqueeze(0)

        for msa_block in tqdm(self.msa_stack, desc="MSA stack"):
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )

        return pair_activations

    def forward(
        self,
        batch: dict[str, torch.Tensor],  # Input features shared across recycles.
        prev: dict[str, torch.Tensor],  # Activations from the previous recycle.
        target_feat: torch.Tensor,  # Target features shared across recycles.
        idx: int = 0,
    ) -> dict[str, torch.Tensor]:

        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )

        pair_activations = self._add_prev_pair_embedding(pair_activations, prev['pair'])
        # The previous recycle pair is not read again; release the caller's
        # full [N, N, C] reference after its embedding has been enqueued.
        del prev['pair']

        pair_activations = self._relative_encoding(batch, pair_activations)

        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )

        pair_activations = self._embed_template_pair(
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
        )

        pair_activations = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
        )

        single_activations = self.single_activations(target_feat)
        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single']))

        for pairformer_b in tqdm(self.trunk_pairformer, desc=f"Pairformer {idx}"):
            pair_activations, single_activations = pairformer_b(
                pair_activations, pair_mask, single_activations, batch.token_features.mask)

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output


    def _add_prev_pair_embedding(self, pair_activations, previous_pair):
        return add_previous_pair(self, pair_activations, previous_pair)


class TorchFold(nn.Module):
    def __init__(
        self,
        num_recycles: int = 10,
        num_samples: int = 5,
        diffusion_steps: int = 200,
        diffusion_sample_parallel: bool = True,
    ):
        super(TorchFold, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps
        self.diffusion_sample_parallel = diffusion_sample_parallel
        self.single_card_sample_parallel_threshold = (
            SINGLE_CARD_SAMPLE_PARALLEL_THRESHOLD
        )

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
            clear_flag=True
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
        noise_level: torch.Tensor,
        clear_flag: bool
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
                                                 use_conditioning=True,
                                                 clear_flag=clear_flag)
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

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device)
        positions *= noise_levels[0]

        use_sample_batch = self._single_card_sample_parallel_mode(batch.num_res)

        if use_sample_batch:
            group_size = SINGLE_CARD_DIFFUSION_SAMPLE_GROUP_SIZE
            mask_expanded = mask.unsqueeze(0).expand(num_samples, -1, -1)

            num_groups = (num_samples + group_size - 1) // group_size
            for group_idx in range(num_groups):
                group_start = group_idx * group_size
                group_end = min(group_start + group_size, num_samples)

                positions_group = positions[group_start:group_end]
                mask_group = mask_expanded[group_start:group_end]

                is_last_group = (group_idx == num_groups - 1)
                for step_idx in trange(
                    self.diffusion_steps,
                    desc=f"Diffusion group {group_idx}/{num_groups}",
                ):
                    clear_flag = is_last_group and (step_idx == self.diffusion_steps - 1)
                    positions_group = self._apply_denoising_step(
                        batch,
                        embeddings,
                        positions_group,
                        noise_levels[step_idx],
                        mask_group,
                        noise_levels[1 + step_idx],
                        clear_flag,
                    )

                positions[group_start:group_end] = positions_group

            final_dense_atom_mask = mask_expanded

        else:
            clear_flag = False
            for sample_idx in range(num_samples):
                for step_idx in trange(
                    self.diffusion_steps,
                    desc=f"Diffusion {sample_idx}",
                ):
                    if (
                        sample_idx == num_samples - 1
                        and step_idx == self.diffusion_steps - 1
                    ):
                        clear_flag = True
                    positions[sample_idx] = self._apply_denoising_step(
                        batch,
                        embeddings,
                        positions[sample_idx],
                        noise_levels[step_idx],
                        mask,
                        noise_levels[1 + step_idx],
                        clear_flag,
                    )

            final_dense_atom_mask = torch.tile(
                mask.unsqueeze(0),
                (num_samples, 1, 1),
            )

        return {'atom_positions': positions, 'mask': final_dense_atom_mask}

    def _single_card_sample_parallel_mode(self, num_res: int) -> bool:
        return single_card_sample_parallel(
            feature_length=num_res,
            enabled=self.diffusion_sample_parallel,
            threshold=self.single_card_sample_parallel_threshold,
        )

    @with_offload_policy()
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = feat_batch.Batch.from_data_dict(batch)
        num_res = batch.num_res

        target_feat = self.create_target_feat_embedding(batch)

        embeddings = {
            'pair': None if implicit_pair_enabled(training=self.training) and self.num_recycles > 0 else torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=torch.float32,
            ),
            'single': torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
            ),
            'target_feat': target_feat,  # type: ignore
        }

        for i in range(self.num_recycles):
            embeddings = self.evoformer(
                batch=batch,
                prev=embeddings,
                target_feat=target_feat,
                idx=i,
            )
            if large_pair_offload(training=self.training) and i + 1 < self.num_recycles:
                embeddings['pair'] = embeddings['pair'].cpu()
                print(f'[offload] recycle_pair_to_cpu recycle={i}', flush=True)

        samples = self._sample_diffusion(batch, embeddings)

        # Long single-card diffusion keeps the trunk pair bit-for-bit on CPU.
        # Drain the final denoise tail before Confidence reloads that pair.
        if embeddings['pair'].device.type == 'cpu':
            torch.npu.synchronize()
            torch.npu.empty_cache()

        confidence_output_per_sample = []
        for sample_dense_atom_position in tqdm(samples['atom_positions'], desc="Confidence"):
            sample_output = self.confidence_head(
                dense_atom_positions=sample_dense_atom_position,
                embeddings=embeddings,
                seq_mask=batch.token_features.mask,
                token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch.token_features.asym_id
            )
            if large_pair_offload(training=self.training):
                sample_output = offload_confidence_output(sample_output)
                print('[offload] confidence_output_to_cpu=True', flush=True)
            confidence_output_per_sample.append(sample_output)

        confidence_output = stack_confidence_outputs(
            confidence_output_per_sample,
        )
        del confidence_output_per_sample

        distogram = self.distogram_head(batch, embeddings)

        return {
            'diffusion_samples': samples,
            'distogram': distogram,
            **confidence_output,
        }
