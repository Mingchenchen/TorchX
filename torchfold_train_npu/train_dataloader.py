from patch_imports import patch_pkl_remapping
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset, Sampler

from torchfold.processing import features as features_utils

from torchfold.processing.cropping_applier import apply_token_cropping


class LengthGroupedDistributedSampler(Sampler):
    """
    Distributed sampler: ensures that samples processed by different NPUs in the same step have similar lengths
    Avoids the problem of fast NPUs waiting for slow NPUs

    Core idea:
    1. Sort samples by length
    2. Divide adjacent-length samples into groups of size world_size
    3. Samples in the same group are processed by different NPUs in the same step
    4. Shuffle between groups to ensure randomness
    """

    def __init__(
            self,
            dataset: Dataset,
            lengths: List[int],  # Length of each sample
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = True,
            # ===== Optional enhancement: sample by data source ratio =====
            enable_source_ratio_sampling: bool = False,
            source_sizes: Optional[List[int]] = None,
            source_sampling_ratios: Optional[List[float]] = None,
    ):
        if num_replicas is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.lengths = lengths
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.enable_source_ratio_sampling = enable_source_ratio_sampling
        self.source_sizes = source_sizes
        self.source_sampling_ratios = source_sampling_ratios

        use_ratio = (
                self.enable_source_ratio_sampling
                and (self.source_sizes is not None)
                and (self.source_sampling_ratios is not None)
                and (len(self.source_sizes) == len(self.source_sampling_ratios))
                and (sum(self.source_sizes) == len(self.dataset))
        )
        self.enable_source_ratio_sampling = bool(use_ratio)

        if self.enable_source_ratio_sampling:
            source_sizes = self.source_sizes
            source_sampling_ratios = self.source_sampling_ratios

            # Target sample count for each data source in this epoch (determined only by sizes/ratios, not changing across epochs)
            self._source_offsets: List[int] = []
            offset = 0
            for sz in source_sizes:
                self._source_offsets.append(offset)
                offset += int(sz)

            self._source_target_counts: List[int] = []
            total_selected = 0
            for sz, ratio in zip(source_sizes, source_sampling_ratios):
                r = float(ratio)
                if r <= 0.0:
                    k = 0
                elif r >= 1.0:
                    k = int(sz)
                else:
                    k = int(int(sz) * r)  # floor
                k = max(0, min(int(sz), k))
                self._source_target_counts.append(k)
                total_selected += k

            self._base_total_size = total_selected
        else:
            self._base_total_size = len(self.dataset)

        # Calculate number of samples per rank (based on base_total_size, not necessarily equal to len(dataset))
        if self.drop_last:
            self.num_samples = self._base_total_size // self.num_replicas
        else:
            self.num_samples = (self._base_total_size + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def _build_epoch_indices(self) -> List[int]:
        """Build the global index list used in the current epoch (must be consistent across all ranks)."""
        if not self.enable_source_ratio_sampling:
            return list(range(len(self.dataset)))

        source_sizes = self.source_sizes or []
        source_sampling_ratios = self.source_sampling_ratios or []

        selected: List[int] = []
        for src_idx, (src_size, ratio, src_offset, k) in enumerate(
                zip(
                    source_sizes,
                    source_sampling_ratios,
                    self._source_offsets,
                    self._source_target_counts,
                )
        ):
            if k <= 0:
                continue
            if float(ratio) >= 1.0:
                selected.extend(range(src_offset, src_offset + int(src_size)))
                continue

            # Sample without replacement within each data source; each source uses its own seed for stable reproducibility
            g = torch.Generator()
            num_sources = max(1, len(source_sizes))
            g.manual_seed(self.seed + self.epoch * num_sources + int(src_idx))
            perm = torch.randperm(int(src_size), generator=g).tolist()
            chosen_local = perm[:k]
            selected.extend([src_offset + i for i in chosen_local])
        return selected

    def __iter__(self):
        # Build the index set for this epoch, then sort by length to get indices
        epoch_indices = self._build_epoch_indices()
        indices = sorted(epoch_indices, key=lambda i: self.lengths[i])

        # Divide into groups of size num_replicas; samples in the same group are processed by different NPUs in the same step
        num_groups = len(indices) // self.num_replicas
        groups = [
            indices[i * self.num_replicas: (i + 1) * self.num_replicas]
            for i in range(num_groups)
        ]

        # Handle remaining samples
        remainder = len(indices) % self.num_replicas
        if remainder > 0:
            if not self.drop_last:
                # Pad the last group with the first few samples to reach num_replicas
                last_group = indices[-remainder:] + indices[:self.num_replicas - remainder]
                groups.append(last_group)

        # Shuffle the order of groups (not within groups)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            group_perm = torch.randperm(len(groups), generator=g).tolist()
            groups = [groups[i] for i in group_perm]

        # Each rank takes the sample at its position within each group
        result = [group[self.rank] for group in groups if len(group) > self.rank]

        return iter(result)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch, used for random seed in shuffling"""
        self.epoch = epoch


class MixedSubsetDistributedSampler(Sampler):
    """
    Mixed sampler: samples a fixed number of samples from two data types according to a ratio in each epoch.

    Suitable for mixed training of antigen-antibody and PPI data.
    """

    def __init__(
            self,
            ab_size: int,
            ppi_size: int,
            epoch_samples: int = 5000,
            ab_ratio: float = 1.0,
            ppi_ratio: float = 1.0,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = True,
    ):
        if num_replicas is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.ab_size = ab_size
        self.ppi_size = ppi_size
        self.epoch_samples = epoch_samples
        self.ab_ratio = ab_ratio
        self.ppi_ratio = ppi_ratio
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Number of samples per rank (based on epoch_samples)
        if self.drop_last:
            self.num_samples = self.epoch_samples // self.num_replicas
        else:
            self.num_samples = (self.epoch_samples + self.num_replicas - 1) // self.num_replicas

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        total_ratio = self.ab_ratio + self.ppi_ratio
        ab_count = int(round(self.epoch_samples * (self.ab_ratio / total_ratio)))
        ppi_count = self.epoch_samples - ab_count

        # Sample antigen-antibody data
        if self.ab_size <= 0 or ab_count <= 0:
            ab_indices = torch.empty((0,), dtype=torch.long)
        elif ab_count <= self.ab_size:
            ab_indices = torch.randperm(self.ab_size, generator=g)[:ab_count]
        else:
            ab_indices = torch.randint(0, self.ab_size, (ab_count,), generator=g)

        # Sample PPI data (offset to global indices)
        if self.ppi_size <= 0 or ppi_count <= 0:
            ppi_indices = torch.empty((0,), dtype=torch.long)
        elif ppi_count <= self.ppi_size:
            ppi_indices = torch.randperm(self.ppi_size, generator=g)[:ppi_count]
        else:
            ppi_indices = torch.randint(0, self.ppi_size, (ppi_count,), generator=g)
        if ppi_indices.numel() > 0:
            ppi_indices = ppi_indices + self.ab_size

        indices = torch.cat([ab_indices, ppi_indices], dim=0)
        if self.shuffle:
            perm = torch.randperm(indices.numel(), generator=g)
            indices = indices[perm]

        # Distribute to each rank
        if self.drop_last:
            total_size = (indices.numel() // self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        else:
            total_size = ((indices.numel() + self.num_replicas - 1) // self.num_replicas) * self.num_replicas
            if total_size > indices.numel():
                padding = indices[:(total_size - indices.numel())]
                indices = torch.cat([indices, padding], dim=0)

        indices = indices[self.rank:total_size:self.num_replicas]
        return iter(indices.tolist())


class TorchFoldDataset(Dataset):
    """
    training dataset

    Assumes data is already preprocessed, including:
    - MSA computation
    - Template search
    - Ground-truth structure coordinates

    Data format: each sample contains
    1. featurized data (*.pkl): precalculated features
    2. ground-truth structure (*.pkl): reference coordinates
    """

    def __init__(
            self,
            data_dir: str,
            split: str = 'train',
            max_num_res: Optional[int] = None,
            sample_weight_file: Optional[str] = None,
            enable_cropping: bool = True,
            crop_size: Optional[int] = 50,
            crop_complete_ligand_unstdRes: bool = False,
            spatial_crop_complete_ligand_unstdRes: bool = False,
            drop_last: bool = False,
            remove_metal: bool = False,
            crop_method_weights: Optional[List[float]] = None,
            interface_minimal_distance: int = 15,
            max_templates: Optional[int] = None,
            data_source_idx: int = 0,
            # Data source index, used to distinguish noise distributions from different sources during training
            remove_unresolved_tokens: bool = False,  # Whether to filter unresolved tokens before cropping
            verbose: bool = False,  # Whether to print details for each sample
    ):
        """
        Args:
            data_dir: Directory that contains the precomputed feature files
            split: 'train', 'val', or 'test'
            max_num_res: Maximum residues allowed (used for validation filtering)
            sample_weight_file: Optional sample-weight JSON
            enable_cropping: Whether to enable contiguous cropping
            crop_size: Number of tokens after cropping (None keeps all tokens)
            crop_complete_ligand_unstdRes: Preserve complete ligands/unstandard residues during contiguous cropping
            spatial_crop_complete_ligand_unstdRes: Preserve complete ligands/unstandard residues during spatial cropping
            drop_last: Whether to drop the final partial ligand/unstandard residue when exceeding limits
            remove_metal: Whether to remove metals/ions
            crop_method_weights: Sampling weights for contiguous/spatial/spatial-interface cropping
            interface_minimal_distance: Minimal distance that defines an interface in spatial-interface cropping
            max_templates: Maximum number of templates to keep (None = no limit, recommended: 4)
            data_source_idx: Index of the data source directory (used for per-source noise distribution in training)
            remove_unresolved_tokens: Whether to filter out unresolved tokens before cropping (default False)
            verbose: Whether to print per-sample info during __getitem__ (default False)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_num_res = max_num_res
        self.enable_cropping = enable_cropping
        self.crop_size = crop_size
        self.crop_complete_ligand_unstdRes = crop_complete_ligand_unstdRes
        self.spatial_crop_complete_ligand_unstdRes = spatial_crop_complete_ligand_unstdRes
        self.drop_last = drop_last
        self.remove_metal = remove_metal
        self.crop_method_weights = (
            crop_method_weights if crop_method_weights is not None else [0.34, 0.33, 0.33]
        )
        self.interface_minimal_distance = interface_minimal_distance
        self.max_templates = max_templates
        self.data_source_idx = data_source_idx  # Data source index
        self.remove_unresolved_tokens = remove_unresolved_tokens  # Whether to filter unresolved tokens

        self.verbose = verbose

        # Load sample list
        self.samples = self._load_sample_list()

        # Load sample weights when provided
        self.sample_weights = None
        if sample_weight_file and os.path.exists(sample_weight_file):
            self.sample_weights = self._load_sample_weights(sample_weight_file)

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_sample_list(self) -> List[Dict]:
        """Load the sample list from disk"""
        list_file = self.data_dir / f"{self.split}_list.json"

        if not list_file.exists():
            raise FileNotFoundError(
                f"Sample list file not found: {list_file}\n"
                f"Please create a JSON file with list of sample IDs"
            )

        with open(list_file, 'r') as f:
            samples = json.load(f)

        # Filter overly long samples (validation only)
        filtered_samples = []
        for sample in samples:
            if self.split == 'val' and self.max_num_res is not None:
                num_res = sample.get('num_res', None)
                if num_res is not None and num_res > self.max_num_res:
                    continue
            filtered_samples.append(sample)

        return filtered_samples

    def _load_sample_weights(self, weight_file: str) -> Dict[str, float]:
        """Load sample weights"""
        with open(weight_file, 'r') as f:
            weights = json.load(f)
        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, _retry_count: int = 0) -> Dict[str, torch.Tensor]:
        """
        Load a single sample

        Return format:
        {
            # Input features (used for forward pass)
            'msa': ...,
            'msa_mask': ...,
            'template_*': ...,
            'seq_mask': ...,
            ...

            # Ground-truth labels (used for loss computation)
            'ref_pos': [num_res, max_atoms, 3],
            'ref_mask': [num_res, max_atoms],
            'ref_element': [num_res, max_atoms],
            ...
        }
        """
        # Maximum retry count to prevent infinite recursion
        MAX_RETRIES = 10

        sample_info = self.samples[idx]
        sample_id = sample_info['id']

        try:
            # Load precomputed features
            # print(f"Loading sample: {sample_id}", flush=True)
            feature_file = self.data_dir / self.split / f"{sample_id}_features.pkl"  ###dict
            # feature_file = self.data_dir / 'train' / f"{sample_id}_features.pkl"

            with open(feature_file, 'rb') as f:
                features = pickle.load(f)
            token_atoms_layout = features['token_atoms_layout']

            # Convert to tensors

            features_tensor = self._convert_to_tensor(features)  ###dict
            # Normalize dtype
            if 'deletion_mean' in features_tensor:
                features_tensor['deletion_mean'] = features_tensor['deletion_mean'].to(
                    dtype=torch.float32
                )
            # Apply cropping when needed

            if self.enable_cropping and self.crop_size is not None:
                # Dynamically select up to 2 reference chains for spatial cropping
                unique_chain_ids = torch.unique(features_tensor['asym_id']).tolist()
                dynamic_ref_chain_ids = None
                if unique_chain_ids:
                    num_chains_to_select = min(len(unique_chain_ids), random.randint(1, 2))
                    dynamic_ref_chain_ids = random.sample(unique_chain_ids, k=num_chains_to_select)

                apply_token_cropping(
                    features_tensor,
                    int(self.crop_size),
                    sample_id=sample_id,
                    crop_method_weights=self.crop_method_weights,
                    contiguous_crop_complete_lig=self.crop_complete_ligand_unstdRes,
                    spatial_crop_complete_lig=self.spatial_crop_complete_ligand_unstdRes,
                    drop_last=self.drop_last,
                    remove_metal=self.remove_metal,
                    interface_minimal_distance=self.interface_minimal_distance,
                    reference_chain_ids=dynamic_ref_chain_ids,  # list
                    token_atoms_layout=token_atoms_layout,
                    max_templates=self.max_templates,
                    remove_unresolved_tokens=self.remove_unresolved_tokens,
                    verbose=self.verbose,
                )
        except Exception as e:
            # If loading or cropping fails, randomly select another sample
            if _retry_count >= MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to load sample after {MAX_RETRIES} retries. "
                    f"Last failed sample: {sample_id}, error: {e}"
                )
            if self.verbose:
                print(f"Warning: Failed to load/crop sample {sample_id} (retry {_retry_count + 1}/{MAX_RETRIES}): {e}",
                      flush=True)
            # Randomly select another sample (avoid picking the same one)
            new_idx = random.randint(0, len(self.samples) - 1)
            while new_idx == idx and len(self.samples) > 1:
                new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx, _retry_count=_retry_count + 1)

        # Attach sample weight when available
        if self.sample_weights and sample_id in self.sample_weights:
            features_tensor['sample_weight'] = torch.tensor(
                self.sample_weights[sample_id], dtype=torch.float32
            )
        else:
            features_tensor['sample_weight'] = torch.tensor(1.0, dtype=torch.float32)
        # Keep sample id for downstream (e.g., per-sample file naming)
        features_tensor['sample_id'] = sample_id
        # Add data source index to distinguish noise distributions from different sources during training
        features_tensor['data_source_idx'] = torch.tensor(self.data_source_idx, dtype=torch.long)

        features_tensor['paratope_mask'] = torch.zeros(
            features_tensor['aatype'].shape[0], dtype=torch.bool
        )

        return features_tensor

    def _convert_to_tensor(self, features: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays in the feature tree into PyTorch tensors"""
        cleaned = features_utils.remove_invalidly_typed_feats(features)
        tensor_tree = pytree.tree_map(torch.from_numpy, cleaned)
        return tensor_tree

    def get_lengths(self) -> List[int]:
        """Return the length list of all samples for length-grouped sampling"""
        return [sample.get('num_res', 0) for sample in self.samples]


class CollateFn:
    """Custom collate function to handle variable-length sequences"""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Convert the batch into a unified format

        typically uses batch_size=1, so this mostly handles format conversion
        """
        if len(batch) == 0:  ###list
            raise ValueError("Empty batch")

        # usually processes a single sample at a time
        if len(batch) == 1:
            return batch[0]

        # For true batching we need padding; implement a simple version here
        collated = {}

        # Gather all keys
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]

            # Check whether we can stack directly
            if all(isinstance(v, torch.Tensor) for v in values):
                # Verify consistent shapes
                if all(v.shape == values[0].shape for v in values):
                    # Same shape, stack directly
                    collated[key] = torch.stack(values, dim=0)
                else:
                    # Different shapes require padding
                    collated[key] = self._pad_and_stack(values, key)
            else:
                collated[key] = values

        return collated

    def _pad_and_stack(
            self,
            tensors: List[torch.Tensor],
            key: str
    ) -> torch.Tensor:
        """Pad tensors to same shape and stack"""
        # Find the maximum extent for each dimension
        max_shape = []
        ndim = tensors[0].ndim

        for dim in range(ndim):
            max_dim = max(t.shape[dim] for t in tensors)
            max_shape.append(max_dim)

        # Pad each tensor
        padded = []
        for tensor in tensors:
            pad_width = []
            for dim in range(ndim - 1, -1, -1):  # iterate from the last dimension
                pad_dim = max_shape[dim] - tensor.shape[dim]
                pad_width.extend([0, pad_dim])

            padded_tensor = torch.nn.functional.pad(
                tensor, pad_width, value=self.pad_value
            )
            padded.append(padded_tensor)

        return torch.stack(padded, dim=0)


def create_dataloaders(
        data_dir: str | List[str],
        ppi_data_dir: Optional[List[str]] = None,
        ppi_epoch_samples: int = 5000,
        ppi_ratio: float = 1.0,
        batch_size: int = 1,
        num_workers: int = 4,
        max_num_res: int = 512,
        val_max_num_res: int = 512,
        sample_weight_file: Optional[str] = None,
        enable_cropping: bool = True,
        crop_size: Optional[int] = 50,
        crop_complete_ligand_unstdRes: bool = False,
        spatial_crop_complete_ligand_unstdRes: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
        crop_method_weights: Optional[List[float]] = None,
        interface_minimal_distance: int = 15,
        use_ddp: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        use_length_grouped_sampler: bool = False,
        # ===== Optional enhancement: sample by data_dir ratios (only affects training set sampler) =====
        enable_source_ratio_sampling: bool = False,
        data_source_sampling_ratios: Optional[List[float]] = None,
        max_templates: Optional[int] = None,
        remove_unresolved_tokens: bool = False,
        val_enable_cropping: bool = False,
        verbose: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """
    Create training and validation dataloaders.

    Args:
        data_dir: Data directory (str) or list of data directories (List[str]) to merge.
        ppi_data_dir: Optional PPI data directories to mix into training.
        ppi_epoch_samples: Total samples per epoch when mixing PPI (default 5000).
        ppi_ratio: PPI sampling ratio relative to AB data (default 1.0 -> 1:1).
        batch_size: Batch size (typically uses 1).
        num_workers: Number of data-loading workers.
        # max_num_res: Maximum residues for training (placeholder).
        val_max_num_res: Maximum residues for validation (filter out longer samples).
        sample_weight_file: Optional sample-weight file path.
        enable_cropping: Enable contiguous token cropping.
        crop_size: Number of tokens after cropping.
        crop_complete_ligand_unstdRes: Keep ligands/unusual residues intact during contiguous cropping.
        spatial_crop_complete_ligand_unstdRes: Keep ligands/unusual residues intact during spatial cropping.
        drop_last: Drop the final incomplete ligand/unusual residue when exceeding limits.
        remove_metal: Remove metals/ions.
        crop_method_weights: Sampling probability for contiguous/spatial/spatial-interface cropping.
        interface_minimal_distance: Minimum distance to qualify as an interface in spatial-interface cropping.
        use_ddp: Whether to use distributed training.
        rank: Process rank for distributed training (required if use_ddp=True).
        world_size: Total number of processes (required if use_ddp=True).
        use_length_grouped_sampler: Use length-grouped sampler to ensure similar sample lengths
            across NPUs in each step, avoiding fast-NPU-waits-for-slow-NPU problem.
        max_templates: Maximum number of templates to keep (None = no limit, recommended: 4).
        remove_unresolved_tokens: Whether to filter out unresolved tokens before cropping (default False).

    Returns:
        train_loader, val_loader, (train_sampler, val_sampler) if use_ddp else (None, None)
    """
    # Handle multiple data directories (antigen-antibody data)
    if isinstance(data_dir, str):
        data_dirs = [data_dir]
    else:
        data_dirs = data_dir

    # Optional PPI data directories
    if ppi_data_dir is not None:
        if isinstance(ppi_data_dir, str):
            ppi_data_dirs = [ppi_data_dir]
        else:
            ppi_data_dirs = ppi_data_dir
        if len(ppi_data_dirs) == 0:
            ppi_data_dirs = None
    else:
        ppi_data_dirs = None

    # Build datasets for each data directory
    train_datasets = []
    val_datasets = []
    ppi_train_datasets = []

    for data_source_idx, data_dir_path in enumerate(data_dirs):
        train_dataset = TorchFoldDataset(
            data_dir=data_dir_path,
            split='train',
            max_num_res=max_num_res,
            sample_weight_file=sample_weight_file,
            enable_cropping=enable_cropping,
            crop_size=crop_size,
            crop_complete_ligand_unstdRes=crop_complete_ligand_unstdRes,
            spatial_crop_complete_ligand_unstdRes=spatial_crop_complete_ligand_unstdRes,
            drop_last=drop_last,
            remove_metal=remove_metal,
            crop_method_weights=crop_method_weights,
            interface_minimal_distance=interface_minimal_distance,
            max_templates=max_templates,
            data_source_idx=data_source_idx,  # Pass data source index
            remove_unresolved_tokens=remove_unresolved_tokens,  # Pass whether to filter unresolved tokens
            verbose=verbose,
        )
        train_datasets.append(train_dataset)

        val_dataset = TorchFoldDataset(
            data_dir=data_dir_path,
            split='val',
            max_num_res=val_max_num_res,
            enable_cropping=val_enable_cropping,
            crop_size=crop_size if val_enable_cropping else None,
            crop_complete_ligand_unstdRes=crop_complete_ligand_unstdRes,
            spatial_crop_complete_ligand_unstdRes=spatial_crop_complete_ligand_unstdRes,
            drop_last=drop_last,
            remove_metal=remove_metal,
            crop_method_weights=crop_method_weights,
            interface_minimal_distance=interface_minimal_distance,
            max_templates=max_templates,
            data_source_idx=data_source_idx,  # Pass data source index
            remove_unresolved_tokens=remove_unresolved_tokens,  # Pass whether to filter unresolved tokens
            verbose=verbose,
        )
        val_datasets.append(val_dataset)

    # Build PPI training datasets (no validation for PPI)
    if ppi_data_dirs:
        ppi_data_source_offset = len(data_dirs)
        for ppi_idx, data_dir_path in enumerate(ppi_data_dirs):
            train_dataset = TorchFoldDataset(
                data_dir=data_dir_path,
                split='train',
                max_num_res=max_num_res,
                sample_weight_file=sample_weight_file,
                enable_cropping=enable_cropping,
                crop_size=crop_size,
                crop_complete_ligand_unstdRes=crop_complete_ligand_unstdRes,
                spatial_crop_complete_ligand_unstdRes=spatial_crop_complete_ligand_unstdRes,
                drop_last=drop_last,
                remove_metal=remove_metal,
                crop_method_weights=crop_method_weights,
                interface_minimal_distance=interface_minimal_distance,
                max_templates=max_templates,
                data_source_idx=ppi_data_source_offset + ppi_idx,
                remove_unresolved_tokens=remove_unresolved_tokens,
                verbose=verbose,
            )
            ppi_train_datasets.append(train_dataset)

    # Merge datasets if multiple directories provided
    if len(train_datasets) > 1:
        ab_train_dataset = ConcatDataset(train_datasets)
    else:
        ab_train_dataset = train_datasets[0]

    if len(val_datasets) > 1:
        val_dataset = ConcatDataset(val_datasets)
    else:
        val_dataset = val_datasets[0]

    # Merge PPI datasets if provided
    if ppi_train_datasets:
        if len(ppi_train_datasets) > 1:
            ppi_train_dataset = ConcatDataset(ppi_train_datasets)
        else:
            ppi_train_dataset = ppi_train_datasets[0]
        train_dataset = ConcatDataset([ab_train_dataset, ppi_train_dataset])
        if (rank is None) or (rank == 0):
            print(
                f"Merged AB+PPI datasets: AB={len(ab_train_dataset)} "
                f"PPI={len(ppi_train_dataset)} total={len(train_dataset)}"
            )
    else:
        train_dataset = ab_train_dataset
        if (rank is None) or (rank == 0):
            print(
                f"Merged {len(data_dirs)} data directories: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # Build collate function
    collate_fn = CollateFn()

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if use_ddp:
        if rank is None or world_size is None:
            raise ValueError("rank and world_size must be provided when use_ddp=True")

        if ppi_train_datasets:
            if use_length_grouped_sampler and ((rank is None) or (rank == 0)):
                print("Warning: PPI mixing enabled; LengthGroupedDistributedSampler is ignored.")
            train_sampler = MixedSubsetDistributedSampler(
                ab_size=len(ab_train_dataset),
                ppi_size=len(ppi_train_dataset),
                epoch_samples=ppi_epoch_samples,
                ab_ratio=1.0,
                ppi_ratio=ppi_ratio,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if (rank is None) or (rank == 0):
                print(
                    f"Using MixedSubsetDistributedSampler: epoch_samples={ppi_epoch_samples}, "
                    f"AB:PPI={1.0}:{ppi_ratio}"
                )
        elif use_length_grouped_sampler:
            # Use length-grouped sampler to ensure similar sample lengths across different GPUs in the same step
            # Get length information
            if isinstance(train_dataset, ConcatDataset):
                # Merge lengths from multiple datasets
                train_lengths = []
                for ds in train_dataset.datasets:
                    train_lengths.extend(ds.get_lengths())
                val_lengths = []
                for ds in val_dataset.datasets:
                    val_lengths.extend(ds.get_lengths())
            else:
                train_lengths = train_dataset.get_lengths()
                val_lengths = val_dataset.get_lengths()

            # Optional: sample by data_dir source ratios (only for train_sampler)
            source_sizes = None
            if enable_source_ratio_sampling and (data_source_sampling_ratios is not None):
                if isinstance(train_dataset, ConcatDataset):
                    source_sizes = [len(ds) for ds in train_dataset.datasets]
                else:
                    source_sizes = [len(train_dataset)]

            train_sampler = LengthGroupedDistributedSampler(
                train_dataset,
                lengths=train_lengths,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                enable_source_ratio_sampling=enable_source_ratio_sampling,
                source_sizes=source_sizes,
                source_sampling_ratios=data_source_sampling_ratios,
            )
            val_sampler = LengthGroupedDistributedSampler(
                val_dataset,
                lengths=val_lengths,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if (rank is None) or (rank == 0):
                print("Using LengthGroupedDistributedSampler for balanced workload across GPUs")
        else:
            # Use the original DistributedSampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,  # Don't drop last for validation
            )
    elif ppi_train_datasets:
        train_sampler = MixedSubsetDistributedSampler(
            ab_size=len(ab_train_dataset),
            ppi_size=len(ppi_train_dataset),
            epoch_samples=ppi_epoch_samples,
            ab_ratio=1.0,
            ppi_ratio=ppi_ratio,
            num_replicas=1,
            rank=0,
            shuffle=True,
            drop_last=False,
        )
        if (rank is None) or (rank == 0):
            print(
                f"Using MixedSubsetDistributedSampler (single process): "
                f"epoch_samples={ppi_epoch_samples}, AB:PPI={1.0}:{ppi_ratio}"
            )

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True if (not use_ddp and train_sampler is None) else None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False if not use_ddp else None,  # shuffle must be None when using sampler
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def prepare_sample_list(
        json_dir: str,
        output_dir: str,
        train_ratio: float = 0.9,
        max_num_res: int = 512,
        random_seed: int = 42,
):
    """
    Prepare train/validation sample lists.

    Helper utility that scans a JSON directory and writes train_list.json / val_list.json.

    Args:
        json_dir: Directory containing JSON fold-input files.
        output_dir: Destination directory for the split lists.
        train_ratio: Ratio assigned to the training split.
        # max_num_res: Maximum number of residues (currently unused placeholder).
        random_seed: Random seed for reproducible splits.
    """
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Scan every JSON file
    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    samples = []
    for json_file in json_files:
        try:
            # Read JSON to approximate sequence lengths
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Estimate residue/token count
            num_res = 0
            if 'sequences' in data:
                for seq in data['sequences']:
                    if 'protein' in seq:
                        num_res += len(seq['protein']['sequence'])
                    if 'dna' in seq:
                        num_res += len(seq['dna']['sequence'])
                    if 'rna' in seq:
                        num_res += len(seq['rna']['sequence'])
                    if 'ligand' in seq:
                        # For ligands, count SMILES string length as approximation
                        num_res += len(seq['ligand']['smiles'])

            if num_res > 0:  # and num_res <= max_num_res:
                samples.append({
                    'id': json_file.stem,
                    'num_res': num_res,
                    'path': str(json_file),
                })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print(f"Valid samples: {len(samples)}")

    # Fix random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle samples
    np.random.shuffle(samples)

    # Split into train/validation sets
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Persist metadata
    with open(output_dir / 'train_list.json', 'w') as f:
        json.dump(train_samples, f, indent=2)

    with open(output_dir / 'val_list.json', 'w') as f:
        json.dump(val_samples, f, indent=2)

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Saved to {output_dir}")


def setup_pkl_remapping(pkl_name):
    patch_pkl_remapping(pkl_name)

if __name__ == '__main__':
    # Example usage: create train/val lists
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str,
                        default='',
                        help='Directory containing JSON fold input files (required)')
    parser.add_argument('--output_dir', type=str,
                        default='',
                        help='Output directory for train/val lists (required)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    # parser.add_argument('--max_num_res', type=int, default=512,
    #                     help='Maximum number of residues')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducible data splitting')

    args = parser.parse_args()

    if not args.json_dir or not args.output_dir:
        parser.error('--json_dir and --output_dir are required')

    prepare_sample_list(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        # max_num_res=args.max_num_res,
        random_seed=args.random_seed,
    )
