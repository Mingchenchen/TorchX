import argparse
import logging
import csv
import json
import os
import pathlib
import pickle
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
import torch_npu
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss_function.torchfold_loss_no_chunk import get_default_config
from loss_function.train_loss import loss_fn
from loss_function.train_loss_no_chunk import WeightedRigidAlign
from torchfold.processing import post_processing
from torchfold.torchfold import TorchFold
from train_dataloader import create_dataloaders, setup_pkl_remapping
# For structure extraction and saving
from torchfold.fastnn import config as fastnn_config

torch_npu.npu.matmul.allow_hf32 = True
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


# ==================================
# YAML configuration loading and parsing
# ==================================


def maybe_print_ddp_param_index_mapping(model: nn.Module, rank: int) -> None:
    raw = os.environ.get("TROCHFOLD_DEBUG_DDP_PARAM_INDICES", "").strip()
    if not raw or rank != 0:
        return

    wanted_indices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            wanted_indices.append(int(item))
        except ValueError:
            print(f"[ddp_param_map] skip invalid index: {item}", flush=True)

    if not wanted_indices:
        return

    named_params = list(model.named_parameters())
    print(f"[ddp_param_map] total params in order: {len(named_params)}", flush=True)
    for idx in wanted_indices:
        if 0 <= idx < len(named_params):
            name, param = named_params[idx]
            print(
                f"[ddp_param_map] {idx}: {name} shape={tuple(param.shape)} "
                f"requires_grad={param.requires_grad}",
                flush=True,
            )
        else:
            print(f"[ddp_param_map] {idx}: <out of range>", flush=True)


def parse_debug_param_names_env(var_name: str) -> List[str]:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return []
    names = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            names.append(item)
    return names


@dataclass
class TrainingConfig:
    """Data class for training configuration"""
    # Data configuration
    data_dirs: List[str] = field(default_factory=list)
    ppi_data_dirs: Optional[List[str]] = None
    ppi_epoch_samples: int = 5000
    ppi_ratio: float = 1.0
    output_dir: str = ""
    pretrained_model_dir: Optional[str] = None
    resume_from: Optional[str] = None
    resume_optimizer: bool = True  # False = only load model weights, reset optimizer/scheduler
    max_num_res: int = 512
    val_max_num_res: int = 512

    # Cropping configuration
    enable_cropping: bool = False
    crop_size: int = 512
    crop_complete_ligand_unstdRes: bool = False
    spatial_crop_complete_ligand_unstdRes: bool = False
    drop_last: bool = False
    remove_metal: bool = False
    crop_method_weights: List[float] = field(default_factory=lambda: [0.34, 0.33, 0.33])
    interface_minimal_distance: int = 15
    max_templates: Optional[int] = None
    remove_unresolved_tokens: bool = False

    # Model configuration
    num_recycles: int = 3
    randomize_num_recycles: bool = False
    recycle_grad_all: bool = True  # True ensures prev_embedding/template/msa receive gradients; False saves memory but may lack gradients
    num_diffusion_samples: int = 1
    diffusion_steps: int = 200
    mini_rollout_steps: int = 0
    num_diffusion_samples_training: int = 48
    condition_embedding_drop_rate: float = 0.0
    dot_product_attention: str = "Fusion_Attention"
    antibody_msa: bool = True
    antibody_msa_drop_prob: float = 0.5

    # Training configuration
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    train_only_confidence: bool = False
    train_only_diffusion: bool = False
    no_train_confidence: bool = False
    use_scheduler: bool = False
    scheduler_t0: int = 1000
    use_amp: bool = True
    amp_dtype: str = "bf16"

    # Noise configuration
    enable_per_source_noise: bool = False
    noise_configs: Optional[Dict[int, Dict[str, float]]] = None

    # Checkpointing configuration
    checkpoint_pairformer: bool = False
    pairformer_group_size: int = 8
    checkpoint_msa: bool = False
    msa_group_size: int = 2
    checkpoint_template: bool = False
    checkpoint_diffusion: bool = False
    checkpoint_confidence: bool = False

    # Logging and saving configuration
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1
    eval_interval_steps: Optional[int] = None  # If set, validate every N steps (takes precedence over eval_interval)
    epoch_save_interval: int = 10
    save_val_structures: bool = False
    save_val_structures_epoch_interval: int = 1
    max_diffusion_steps_to_save: Optional[int] = None
    max_batches_to_save: Optional[int] = None
    save_structures_interval: int = 1
    initial_val_max_batches: Optional[int] = None

    # Distributed training configuration
    use_length_grouped_sampler: bool = False
    enable_source_ratio_sampling: bool = False
    data_source_sampling_ratios: Optional[List[float]] = None
    num_workers: int = 1

    # Loss function weights
    fape_weight: float = 1.0
    distogram_weight: float = 0.3
    confidence_weight: float = 0.01

    # Interface loss weighting
    use_interface_loss_weight: bool = False
    interface_loss_weight: float = 5.0
    interface_distance_threshold: float = 10.0

    # Per‑module learning rates
    # e.g. {"evoformer": 5e-5, "diffusion_head": 1e-4, "confidence_head": 1e-5}
    layer_lr: Optional[Dict[str, float]] = None

    # List of modules to freeze
    # e.g. ["confidence_head", "distogram_head"]
    freeze_modules: Optional[List[str]] = None

    # Cross-chain Pair Representation Dropout
    cross_chain_pair_dropout: bool = False
    cross_chain_pair_dropout_prob: float = 0.1
    cross_chain_pair_dropout_mode: str = "zero"  # "zero", "noise", "scale"
    cross_chain_pair_dropout_noise_scale: float = 1.0
    cross_chain_pair_dropout_scale_factor: float = 0.1

    # Online Self-Distillation
    use_online_distillation: bool = False
    distill_iptm_threshold: float = 0.6
    distill_keep_epochs: int = 3
    val_crop: bool = False

    # Other configuration
    seed: int = 42
    disable_internal_progress: bool = False
    enable_exception_handling: bool = False

    # Training noise schedule curriculum
    noise_p_mean: float = -1.2
    p_std: float = 1.5
    sigma_data: float = 16.0
    # Optional curriculum (also via CLI --noise_p_mean_schedule /
    # --noise_p_mean_update_every). Example: every 10k steps
    # p_mean goes 0.5 -> 0.0 -> -0.5 -> -1.2.
    p_mean_schedule: str = "0.5,0.0,-0.5,-1.2"
    update_every: int = 10000


def load_config_from_yaml(yaml_path: str) -> TrainingConfig:
    """Load configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = TrainingConfig()

    # Data configuration
    if 'data' in yaml_config:
        data_cfg = yaml_config['data']
        config.data_dirs = data_cfg.get('data_dirs', [])
        config.ppi_data_dirs = data_cfg.get('ppi_data_dir')
        config.ppi_epoch_samples = data_cfg.get('ppi_epoch_samples', 5000)
        config.ppi_ratio = float(data_cfg.get('ppi_ratio', 1.0))
        config.output_dir = data_cfg.get('output_dir', '')
        config.pretrained_model_dir = data_cfg.get('pretrained_model_dir')
        config.resume_from = data_cfg.get('resume_from')
        config.resume_optimizer = data_cfg.get('resume_optimizer', True)
        config.max_num_res = data_cfg.get('max_num_res', 512)
        config.val_max_num_res = data_cfg.get('val_max_num_res', 512)

    # Cropping configuration
    if 'cropping' in yaml_config:
        crop_cfg = yaml_config['cropping']
        config.enable_cropping = crop_cfg.get('enable_cropping', False)
        config.crop_size = crop_cfg.get('crop_size', 512)
        config.crop_complete_ligand_unstdRes = crop_cfg.get('crop_complete_ligand_unstdRes', False)
        config.spatial_crop_complete_ligand_unstdRes = crop_cfg.get('spatial_crop_complete_ligand_unstdRes', False)
        config.drop_last = crop_cfg.get('drop_last', False)
        config.remove_metal = crop_cfg.get('remove_metal', False)
        config.crop_method_weights = crop_cfg.get('crop_method_weights', [0.34, 0.33, 0.33])
        config.interface_minimal_distance = crop_cfg.get('interface_minimal_distance', 15)
        config.max_templates = crop_cfg.get('max_templates')
        config.remove_unresolved_tokens = crop_cfg.get('remove_unresolved_tokens', False)

    # Model configuration
    if 'model' in yaml_config:
        model_cfg = yaml_config['model']
        config.num_recycles = model_cfg.get('num_recycles', 3)
        config.randomize_num_recycles = model_cfg.get('randomize_num_recycles', True)
        config.recycle_grad_all = model_cfg.get('recycle_grad_all', True)
        config.num_diffusion_samples = model_cfg.get('num_diffusion_samples', 1)
        config.diffusion_steps = model_cfg.get('diffusion_steps', 200)
        config.mini_rollout_steps = model_cfg.get('mini_rollout_steps', 0)
        config.num_diffusion_samples_training = model_cfg.get('num_diffusion_samples_training', 48)
        config.condition_embedding_drop_rate = model_cfg.get('condition_embedding_drop_rate', 0.0)
        config.antibody_msa = model_cfg.get('antibody_msa', True)
        config.antibody_msa_drop_prob = model_cfg.get('antibody_msa_drop_prob', 0.5)

    # Training configuration
    if 'training' in yaml_config:
        train_cfg = yaml_config['training']
        config.num_epochs = train_cfg.get('num_epochs', 100)
        config.batch_size = train_cfg.get('batch_size', 1)
        config.learning_rate = float(train_cfg.get('learning_rate', 1e-4))
        config.min_lr = float(train_cfg.get('min_lr', 1e-6))
        config.weight_decay = float(train_cfg.get('weight_decay', 0.0))
        config.gradient_clip = float(train_cfg.get('gradient_clip', 1.0))
        config.accumulation_steps = train_cfg.get('accumulation_steps', 1)
        config.train_only_confidence = train_cfg.get('train_only_confidence', False)
        config.train_only_diffusion = train_cfg.get('train_only_diffusion', False)
        config.no_train_confidence = train_cfg.get('no_train_confidence', False)
        config.use_scheduler = train_cfg.get('use_scheduler', False)
        config.scheduler_t0 = train_cfg.get('scheduler_t0', 1000)
        config.use_amp = train_cfg.get('use_amp', True)
        config.amp_dtype = train_cfg.get('amp_dtype', 'bf16')
        pkl_name = train_cfg.get('pkl_name', '')
        if pkl_name and len(pkl_name) > 0:
            setup_pkl_remapping(pkl_name)

    # Noise configuration
    if 'noise' in yaml_config:
        noise_cfg = yaml_config['noise']
        config.enable_per_source_noise = noise_cfg.get('enable_per_source_noise', False)
        if 'configs' in noise_cfg and noise_cfg['configs']:
            # Convert keys to integers
            config.noise_configs = {int(k): v for k, v in noise_cfg['configs'].items()}

    # Checkpointing configuration
    if 'checkpointing' in yaml_config:
        ckpt_cfg = yaml_config['checkpointing']
        config.checkpoint_pairformer = ckpt_cfg.get('checkpoint_pairformer', False)
        config.pairformer_group_size = ckpt_cfg.get('pairformer_group_size', 8)
        config.checkpoint_msa = ckpt_cfg.get('checkpoint_msa', False)
        config.msa_group_size = ckpt_cfg.get('msa_group_size', 2)
        config.checkpoint_template = ckpt_cfg.get('checkpoint_template', False)
        config.checkpoint_diffusion = ckpt_cfg.get('checkpoint_diffusion', False)
        config.checkpoint_confidence = ckpt_cfg.get('checkpoint_confidence', False)

    # Logging and saving configuration
    if 'logging' in yaml_config:
        log_cfg = yaml_config['logging']
        config.log_interval = log_cfg.get('log_interval', 100)
        config.save_interval = log_cfg.get('save_interval', 1000)
        config.eval_interval = log_cfg.get('eval_interval', 1)
        config.eval_interval_steps = log_cfg.get('eval_interval_steps')
        config.epoch_save_interval = log_cfg.get('epoch_save_interval', 10)
        config.save_val_structures = log_cfg.get('save_val_structures', False)
        config.save_val_structures_epoch_interval = log_cfg.get('save_val_structures_epoch_interval', 1)
        config.max_diffusion_steps_to_save = log_cfg.get('max_diffusion_steps_to_save')
        config.max_batches_to_save = log_cfg.get('max_batches_to_save')
        config.save_structures_interval = log_cfg.get('save_structures_interval', 1)
        config.initial_val_max_batches = log_cfg.get('initial_val_max_batches', 1)

    # Distributed training configuration
    if 'distributed' in yaml_config:
        dist_cfg = yaml_config['distributed']
        config.use_length_grouped_sampler = dist_cfg.get('use_length_grouped_sampler', False)
        config.enable_source_ratio_sampling = dist_cfg.get('enable_source_ratio_sampling', False)
        ratios = dist_cfg.get('data_source_sampling_ratios', None)
        if ratios is not None:
            config.data_source_sampling_ratios = [float(x) for x in ratios]
        config.num_workers = dist_cfg.get('num_workers', 1)

    # Loss function weights
    if 'loss_weights' in yaml_config:
        loss_cfg = yaml_config['loss_weights']
        config.fape_weight = float(loss_cfg.get('fape_weight', 1.0))
        config.distogram_weight = float(loss_cfg.get('distogram_weight', 0.3))
        config.confidence_weight = float(loss_cfg.get('confidence_weight', 0.01))

    # Interface loss weighting
    if 'interface_loss' in yaml_config:
        iface_cfg = yaml_config['interface_loss']
        config.use_interface_loss_weight = iface_cfg.get('enabled', False)
        config.interface_loss_weight = float(iface_cfg.get('weight', 5.0))
        config.interface_distance_threshold = float(iface_cfg.get('distance_threshold', 10.0))

    # Per-module learning rates
    if 'layer_lr' in yaml_config:
        layer_lr_cfg = yaml_config['layer_lr']
        if isinstance(layer_lr_cfg, dict):
            config.layer_lr = {k: float(v) for k, v in layer_lr_cfg.items()}

    # Freeze modules
    if 'freeze_modules' in yaml_config:
        freeze_cfg = yaml_config['freeze_modules']
        if isinstance(freeze_cfg, list):
            config.freeze_modules = freeze_cfg

    # Online Self-Distillation
    if 'online_distillation' in yaml_config:
        distill_cfg = yaml_config['online_distillation']
        if isinstance(distill_cfg, dict):
            config.use_online_distillation = distill_cfg.get('enabled', False)
            config.distill_iptm_threshold = float(distill_cfg.get('iptm_threshold', 0.6))
            config.distill_keep_epochs = int(distill_cfg.get('keep_epochs', 3))
            config.val_crop = distill_cfg.get('val_crop', False)

    # Cross-chain Pair Representation Dropout
    if 'cross_chain_pair_dropout' in yaml_config:
        ccpd_cfg = yaml_config['cross_chain_pair_dropout']
        if isinstance(ccpd_cfg, dict):
            config.cross_chain_pair_dropout = ccpd_cfg.get('enabled', False)
            config.cross_chain_pair_dropout_prob = float(ccpd_cfg.get('prob', 0.1))
            config.cross_chain_pair_dropout_mode = ccpd_cfg.get('mode', 'zero')
            config.cross_chain_pair_dropout_noise_scale = float(ccpd_cfg.get('noise_scale', 1.0))
            config.cross_chain_pair_dropout_scale_factor = float(ccpd_cfg.get('scale_factor', 0.1))

    # Other configuration
    if 'misc' in yaml_config:
        misc_cfg = yaml_config['misc']
        config.seed = misc_cfg.get('seed', 42)
        config.disable_internal_progress = misc_cfg.get('disable_internal_progress', False)
        config.enable_exception_handling = misc_cfg.get('enable_exception_handling', False)

    return config


class Trainer:
    """trainer"""

    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler._LRScheduler],
            device: torch.device,
            output_dir: str,
            dot_product_attention: str,
            data_dir: Union[str, List[str]] = None,
            # Data directory (or list of directories) for reloading pickle files
            gradient_clip: float = 1.0,
            accumulation_steps: int = 1,
            use_amp: bool = True,
            amp_dtype: str = "fp16",  # Added support for "fp16" or "bf16"
            log_interval: int = 10,
            save_interval: int = 1000,
            eval_interval: int = 100,
            eval_interval_steps: Optional[int] = None,
            epoch_save_interval: int = 10,
            save_structures_interval: int = 1,  # Save structures every N validations (1 = every validation)
            max_batches_to_save: int = None,  # Max batches to save per validation (None = save all)
            initial_val_max_batches: Optional[int] = None,  # Max batches for initial validation
            save_diffusion_debug: bool = False,  # Save diffusion coords during training
            max_diffusion_steps_to_save: int = None,
            # Max diffusion steps to save (None = save all, e.g., 100 = save last 100 steps)
            save_val_structures_epoch_interval: int = 1,  # Save val structures every N epochs (1 = every epoch)
            use_ddp: bool = False,
            local_rank: int = 0,
            global_rank: int = 0,
            catch_exceptions: bool = False,  # Enable exception handling
            # Online Self-Distillation
            use_online_distillation: bool = False,
            distill_iptm_threshold: float = 0.6,
            distill_keep_epochs: int = 3,
            config: Optional['TrainingConfig'] = None,  # stored for rebuilding dataloaders
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = pathlib.Path(output_dir)
        # Handle multiple data directories
        if data_dir is None:
            self.data_dir = None
            self.data_dirs = []
        elif isinstance(data_dir, str):
            self.data_dir = pathlib.Path(data_dir)
            self.data_dirs = [pathlib.Path(data_dir)]
        else:
            self.data_dir = pathlib.Path(data_dir[0]) if data_dir else None  # Keep first for backward compatibility
            self.data_dirs = [pathlib.Path(d) for d in data_dir]
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_interval_steps = eval_interval_steps
        self.epoch_save_interval = epoch_save_interval
        self.save_structures_interval = save_structures_interval
        self.max_batches_to_save = max_batches_to_save
        self.initial_val_max_batches = initial_val_max_batches
        self.save_diffusion_debug = save_diffusion_debug
        self.max_diffusion_steps_to_save = max_diffusion_steps_to_save
        self.save_val_structures_epoch_interval = save_val_structures_epoch_interval
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.catch_exceptions = catch_exceptions
        # In multi-node setup, only global rank 0 is the true "main" process.
        # For non-DDP runs, the single process is also treated as main.
        self.is_main_process = (not self.use_ddp) or (self.global_rank == 0)
        fastnn_config.dot_product_attention_implementations = dot_product_attention

        # Get world size for logging
        if self.use_ddp:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        # Counter for validation runs (to control structure saving frequency)
        self.eval_count = 0

        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        # self.structures_dir = self.output_dir / 'structures'
        # self.structures_dir.mkdir(exist_ok=True)

        # TensorBoard
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        else:
            self.writer = None

        # AMP scaler
        if use_amp:
            if self.amp_dtype == 'bf16' and not torch.npu.is_bf16_supported():
                print("Warning: bf16 is not supported on this device, falling back to fp16.", flush=True)
                self.amp_dtype = 'fp16'

            if self.amp_dtype == 'fp16':
                self.scaler = torch.npu.amp.GradScaler()
            else:  # bf16
                self.scaler = None
        else:
            self.scaler = None

        # Training state trackers
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Metric storage
        self.train_stats = {
            'loss': [],
            'fape_loss': [],
            'distogram_loss': [],
        }

        # Error tracking
        # Each process writes to its own error log file to avoid conflicts
        self.error_log_file = self.output_dir / f'error_log_rank{self.global_rank}.txt'
        self.error_batches_dir = self.output_dir / 'error_batches'
        if self.catch_exceptions:
            self.error_batches_dir.mkdir(exist_ok=True, parents=True)

        # debug dump: one-off debug dump controls
        self.debug_0116_enabled = os.environ.get("TROCHFOLD_DEBUG_0116", "0") == "1"
        self.debug_0116_max_steps = int(os.environ.get("TROCHFOLD_DEBUG_0116_MAX_STEPS", "1"))
        # 0 or negative => save all diffusion samples
        self.debug_0116_max_samples = int(os.environ.get("TROCHFOLD_DEBUG_0116_MAX_SAMPLES", "0"))
        self.debug_0116_count = 0
        self.debug_grad_chain_enabled = os.environ.get("TROCHFOLD_DEBUG_GRAD_CHAIN", "0") == "1"
        self.debug_grad_chain_max_steps = int(os.environ.get("TROCHFOLD_DEBUG_GRAD_CHAIN_MAX_STEPS", "3"))
        self.debug_grad_chain_rank = int(os.environ.get("TROCHFOLD_DEBUG_GRAD_CHAIN_RANK", "0"))
        self.debug_grad_chain_count = 0
        self.debug_param_names = parse_debug_param_names_env("TROCHFOLD_DEBUG_PARAM_GRADS")
        self.debug_param_grads_max_steps = int(os.environ.get("TROCHFOLD_DEBUG_PARAM_GRADS_MAX_STEPS", "10"))
        self.debug_param_grads_rank = int(os.environ.get("TROCHFOLD_DEBUG_PARAM_GRADS_RANK", "0"))
        self.debug_param_grads_count = 0
        self.debug_loss_term_grads_enabled = os.environ.get("TROCHFOLD_DEBUG_LOSS_TERM_GRADS", "0") == "1"
        self.debug_loss_term_grads_max_steps = int(os.environ.get("TROCHFOLD_DEBUG_LOSS_TERM_GRADS_MAX_STEPS", "2"))
        self.debug_loss_term_grads_rank = int(os.environ.get("TROCHFOLD_DEBUG_LOSS_TERM_GRADS_RANK", "0"))
        self.debug_loss_term_grads_count = 0

        # High MSE dump controls
        self.high_mse_threshold = float(os.environ.get("TROCHFOLD_DUMP_MSE_THRESHOLD", "0.8"))
        self.high_mse_max_steps = int(os.environ.get("TROCHFOLD_DUMP_MSE_MAX_STEPS", "5"))
        self.high_mse_dump_count = 0

        # Online Self-Distillation state
        self.use_online_distillation = use_online_distillation
        self.distill_iptm_threshold = distill_iptm_threshold
        self.distill_keep_epochs = distill_keep_epochs
        self._config = config  # Store for rebuilding dataloaders
        self._current_epoch_distill_samples = []  # accumulate during evaluate()
        self._distill_dir = self.output_dir / 'distill_data'
        if self.use_online_distillation:
            self._distill_dir.mkdir(parents=True, exist_ok=True)
            (self._distill_dir / 'train').mkdir(parents=True, exist_ok=True)

        # effective optimizer-step counter (one per accumulation window).
        self.step = 0
        # micro-batch counter (for accumulation boundaries).
        self.micro_step = 0
        # epoch counter (incremented in train() each time the dataloader wraps).
        self.epoch = 0

    def _apply_noise_curriculum(self) -> float:
        """Sync TrainingNoiseSampler.p_mean to the current optimizer step."""
        sampler = self.model.train_noise_sampler
        p_mean = sampler.set_step(self.step)
        if p_mean != getattr(self, "_noise_p_mean_logged", None):
            print(
                "[noise] step=%d -> p_mean=%.4g (schedule=%s every=%s)"
                % (
                    self.step,
                    p_mean,
                    getattr(sampler, "p_mean_schedule", None),
                    getattr(sampler, "update_every", None),
                )
            , flush=True)
            self._noise_p_mean_logged = p_mean
        return p_mean

    def _resolve_named_param(self, name: str) -> tuple[str, Optional[nn.Parameter]]:
        named_params = dict(self.model.named_parameters())
        param = named_params.get(name)
        resolved_name = name
        if param is None and name.startswith("module."):
            resolved_name = name[len("module."):]
            param = named_params.get(resolved_name)
        elif param is None and self.use_ddp:
            resolved_name = f"module.{name}"
            param = named_params.get(resolved_name)
            if param is None:
                resolved_name = name
        return resolved_name, param

    def _maybe_print_selected_param_grads(self) -> None:
        if not self.debug_param_names:
            return
        if self.debug_param_grads_count >= self.debug_param_grads_max_steps:
            return
        if 0 <= self.debug_param_grads_rank != self.global_rank:
            return

        print(f"[param_grads][rank {self.global_rank}] step {self.global_step}", flush=True)
        for name in self.debug_param_names:
            _, param = self._resolve_named_param(name)
            if param is None:
                print(f"  {name}: <missing>", flush=True)
                continue
            grad = param.grad
            if grad is None:
                print(
                    f"  {name}: grad=None requires_grad={param.requires_grad} "
                    f"shape={tuple(param.shape)}",
                    flush=True,
                )
            else:
                grad_detached = grad.detach()
                print(
                    f"  {name}: grad=None? False requires_grad={param.requires_grad} "
                    f"shape={tuple(param.shape)} grad_abs_mean={float(grad_detached.abs().mean().cpu())} "
                    f"grad_norm={float(grad_detached.norm().cpu())}",
                    flush=True,
                )
        self.debug_param_grads_count += 1

    def _maybe_print_selected_param_loss_term_grads(
            self,
            output: Dict,
            loss_dict: Dict,
    ) -> None:
        if not self.debug_loss_term_grads_enabled:
            return
        if not self.debug_param_names:
            return
        if self.debug_loss_term_grads_count >= self.debug_loss_term_grads_max_steps:
            return
        if 0 <= self.debug_loss_term_grads_rank != self.global_rank:
            return

        selected_params = []
        for name in self.debug_param_names:
            resolved_name, param = self._resolve_named_param(name)
            if param is None:
                continue
            selected_params.append((name, resolved_name, param))

        if not selected_params:
            print(
                f"[loss_term_grads][rank {self.global_rank}] step {self.global_step} no selected params resolved",
                flush=True,
            )
            self.debug_loss_term_grads_count += 1
            return

        debug_terms = []
        for key in (
                'distogram_loss',
                'mse_loss',
                'smooth_lddt_loss',
                'bond_loss',
                'total_confidence_loss',
        ):
            value = loss_dict.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == 1 and value.requires_grad:
                debug_terms.append((key, value))
        if isinstance(output, dict):
            anchor = output.get('ddp_static_anchor')
            if isinstance(anchor, torch.Tensor) and anchor.numel() == 1 and anchor.requires_grad:
                debug_terms.append(('ddp_static_anchor', anchor))

        if not debug_terms:
            print(
                f"[loss_term_grads][rank {self.global_rank}] step {self.global_step} no grad-carrying terms",
                flush=True,
            )
            self.debug_loss_term_grads_count += 1
            return

        print(f"[loss_term_grads][rank {self.global_rank}] step {self.global_step}", flush=True)
        params_only = [param for _, _, param in selected_params]
        for term_name, term_value in debug_terms:
            grads = torch.autograd.grad(
                term_value,
                params_only,
                retain_graph=True,
                allow_unused=True,
            )
            print(f"  term={term_name}", flush=True)
            for (requested_name, resolved_name, param), grad in zip(selected_params, grads):
                display_name = requested_name if requested_name == resolved_name else f"{requested_name} -> {resolved_name}"
                if grad is None:
                    print(
                        f"    {display_name}: grad=None requires_grad={param.requires_grad} shape={tuple(param.shape)}",
                        flush=True,
                    )
                else:
                    grad_detached = grad.detach()
                    print(
                        f"    {display_name}: grad=None? False requires_grad={param.requires_grad} "
                        f"shape={tuple(param.shape)} grad_abs_mean={float(grad_detached.abs().mean().cpu())} "
                        f"grad_norm={float(grad_detached.norm().cpu())}",
                        flush=True,
                    )

        self.debug_loss_term_grads_count += 1

    def _check_nan_in_loss(self, loss: torch.Tensor, loss_dict: Dict, batch: Dict) -> None:
        """Check if loss contains NaN and log error if found"""
        if torch.isnan(loss) or torch.isinf(loss):
            sample_ids = batch.get('sample_id', ['unknown'])
            error_msg = (
                f"[Rank {self.global_rank}, Local {self.local_rank}] "
                f"Step {self.global_step}: "
                f"Loss is {'NaN' if torch.isnan(loss) else 'Inf'}: {loss.item()}, "
                f"Sample IDs: {sample_ids}\n"
            )
            print(error_msg, flush=True)

            # Log detailed loss breakdown
            if loss_dict:
                error_msg += "Loss breakdown:\n"
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any() or torch.isinf(value).any():
                            error_msg += f"  {key}: {'NaN' if torch.isnan(value).any() else 'Inf'}\n"

            # Write to error log file
            with open(self.error_log_file, 'a') as f:
                f.write(error_msg + "\n")

    def _log_exception(self, exception: Exception, batch: Dict, batch_idx: int,
                       epoch: Optional[int] = None, step: Optional[int] = None,
                       phase: str = 'train') -> None:
        """
        Log exception to error log file and save problematic batch as pkl

        Args:
            exception: The caught exception object
            batch: The batch that caused the exception
            batch_idx: Batch index in the current epoch/validation
            epoch: Epoch number (for training phase)
            step: Global step number (for training phase)
            phase: 'train' or 'val'
        """
        # Extract sample_id from batch
        sample_ids = batch.get('sample_id', ['unknown'])
        if isinstance(sample_ids, list) and len(sample_ids) > 0:
            sample_id = sample_ids[0]
        elif isinstance(sample_ids, str):
            sample_id = sample_ids
        else:
            sample_id = 'unknown'

        # Get batch size
        batch_size = 1
        if 'aatype' in batch:
            if isinstance(batch['aatype'], torch.Tensor):
                batch_size = batch['aatype'].shape[0] if len(batch['aatype'].shape) > 0 else 1
        elif 'sample_id' in batch:
            if isinstance(batch['sample_id'], list):
                batch_size = len(batch['sample_id'])

        # Generate pkl filename
        if phase == 'train':
            pkl_filename = f"epoch_{epoch}_step_{step}_batch_{batch_idx}_rank{self.global_rank}_{sample_id}.pkl"
        else:  # val
            pkl_filename = f"val_batch_{batch_idx}_rank{self.global_rank}_{sample_id}.pkl"

        pkl_filepath = self.error_batches_dir / pkl_filename

        def _detach_to_cpu(obj):
            """Recursively detach tensors to CPU to avoid NPU errors when saving."""
            if isinstance(obj, torch.Tensor):
                try:
                    return obj.detach().cpu()
                except Exception:
                    return obj
            if isinstance(obj, dict):
                return {k: _detach_to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                converted = [_detach_to_cpu(v) for v in obj]
                return converted if isinstance(obj, list) else tuple(converted)
            return obj

        # Save batch as pkl
        try:
            cpu_batch = _detach_to_cpu(batch)
            with open(pkl_filepath, 'wb') as f:
                pickle.dump(cpu_batch, f)
        except Exception as save_error:
            print(f"Warning: Failed to save batch to {pkl_filepath}: {save_error}", flush=True)
            pkl_filepath = None

        # Prepare error message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
        error_msg = f"""
{'=' * 80}
[{type(exception).__name__}] [{timestamp}]
[Rank {self.global_rank}/{self.world_size}, Local Rank {self.local_rank}]
[Phase: {phase}]
"""
        if phase == 'train':
            error_msg += f"[Epoch {epoch}, Global Step {step}] "
        error_msg += f"[Batch Index: {batch_idx}]\n"
        error_msg += f"[Sample ID: {sample_id}] [Batch Size: {batch_size}]\n"
        error_msg += f"Exception: {str(exception)}\n"
        if pkl_filepath:
            error_msg += f"Batch saved to: {pkl_filepath}\n"
        else:
            error_msg += "Batch save failed\n"
        error_msg += f"Traceback:\n{''.join(traceback.format_exc())}\n"
        error_msg += f"{'=' * 80}\n\n"

        # Write to error log file (each process writes to its own file)
        try:
            with open(self.error_log_file, 'a') as f:
                f.write(error_msg)
                f.flush()  # Ensure immediate write
        except Exception as log_error:
            print(f"Warning: Failed to write to error log {self.error_log_file}: {log_error}", flush=True)

        # Also print to console
        print(
            f"[Rank {self.global_rank}] Exception caught in {phase} phase, batch {batch_idx}, sample {sample_id}: {exception}",
            flush=True)

    # debug dump: dump diffusion samples + per-sample metrics for one-off validation
    def _debug_dump_training_samples(
            self,
            output: Dict,
            batch: Dict,
            loss_dict: Dict,
            *,
            force: bool = False,
            out_dir_suffix: str = "",
    ) -> None:
        if not self.is_main_process:
            return
        if not force and not self.debug_0116_enabled:
            return
        if not force and self.debug_0116_count >= self.debug_0116_max_steps:
            return
        if output is None or 'diffusion_samples' not in output:
            return
        ds = output['diffusion_samples']
        if 'atom_positions' not in ds or 'gt_positions' not in ds or 'x_noisy' not in ds:
            logging.debug("Missing diffusion samples (atom_positions/gt_positions/x_noisy).")
            return

        # Extract sample_id
        sample_ids = batch.get('sample_id', ['unknown'])
        if isinstance(sample_ids, list) and len(sample_ids) > 0:
            sample_id = sample_ids[0]
        elif isinstance(sample_ids, str):
            sample_id = sample_ids
        else:
            sample_id = 'unknown'

        # Find feature file (train split)
        feature_file = None
        for data_dir_path in self.data_dirs:
            candidate_file = data_dir_path / 'train' / f"{sample_id}_features.pkl"
            if candidate_file.exists():
                feature_file = candidate_file
                break
        if feature_file is None:
            logging.debug(f"Feature file not found for sample {sample_id}, skip dump.")
            return

        with open(feature_file, 'rb') as f:
            features = pickle.load(f)

        # Prepare GT (original) coords
        true_pos = features.get('true_positions')
        if isinstance(true_pos, np.ndarray):
            true_pos = torch.from_numpy(true_pos).to(self.device)
        elif isinstance(true_pos, torch.Tensor):
            true_pos = true_pos.to(self.device)
        else:
            logging.debug(f"true_positions type unsupported: {type(true_pos)}")
            return

        crop_idx = batch.get('crop_indices')
        if crop_idx is not None:
            if isinstance(crop_idx, torch.Tensor):
                crop_idx = crop_idx.to(true_pos.device).long()
            true_pos = true_pos[crop_idx]

        # debug dump: compute aligned GT (align GT -> pred for each diffusion sample)
        aligned_coords = None
        try:
            coord_mask = batch.get('true_positions_atom_mask', None)
            if coord_mask is not None:
                coord_mask = coord_mask.to(self.device)
                atoms_per_token = coord_mask.shape[1]
                valid_mask = coord_mask.reshape(-1).bool()

                is_dna = batch.get('is_dna', None)
                is_rna = batch.get('is_rna', None)
                is_ligand = batch.get('is_ligand', None)
                if is_ligand is None and is_dna is not None:
                    is_ligand = torch.zeros_like(is_dna)

                cfg = get_default_config()  # debug dump: keep in sync with loss defaults
                w_dna, w_rna, w_lig = cfg['alpha_dna'], cfg['alpha_rna'], cfg['alpha_ligand']
                weights = torch.ones_like(coord_mask, dtype=torch.float32)
                if is_dna is not None:
                    weights = weights + is_dna.unsqueeze(-1).expand(-1, atoms_per_token).float() * w_dna
                if is_rna is not None:
                    weights = weights + is_rna.unsqueeze(-1).expand(-1, atoms_per_token).float() * w_rna
                if is_ligand is not None:
                    weights = weights + is_ligand.unsqueeze(-1).expand(-1, atoms_per_token).float() * w_lig
                weights = weights * coord_mask.float()

                weights_valid = weights.reshape(-1)[valid_mask]
                pred_coords = ds['atom_positions'].to(self.device).reshape(ds['atom_positions'].shape[0], -1, 3)
                pred_coords_valid = pred_coords[:, valid_mask, :]

                true_coords = true_pos.to(self.device).reshape(-1, 3)
                true_coords_valid = true_coords[valid_mask]

                rigid_align = WeightedRigidAlign()
                # Ensure batched shapes match for alignment: align GT -> pred frame per sample
                gt_valid_b = true_coords_valid
                pred_valid_b = pred_coords_valid
                if gt_valid_b.dim() == 2 and pred_valid_b.dim() == 3:
                    gt_valid_b = gt_valid_b.unsqueeze(0).expand(pred_valid_b.shape[0], -1, -1)
                elif gt_valid_b.dim() == 3 and pred_valid_b.dim() == 2:
                    pred_valid_b = pred_valid_b.unsqueeze(0).expand(gt_valid_b.shape[0], -1, -1)

                aligned_valid = rigid_align(
                    gt_valid_b,  # [N_sample, N_valid, 3] - GT (source)
                    pred_valid_b,  # [N_sample, N_valid, 3] - pred (target)
                    weights_valid
                )  # [N_sample, N_valid, 3] aligned GT in pred frame

                aligned_full = pred_coords.new_zeros(pred_coords.shape)
                aligned_full[:, valid_mask, :] = aligned_valid
                aligned_full = aligned_full.reshape(pred_coords.shape[0], -1, atoms_per_token, 3)
                aligned_full = aligned_full * coord_mask.unsqueeze(0).unsqueeze(-1)
                aligned_coords = aligned_full
        except Exception as e:
            logging.debug(f"Failed to compute aligned GT coords: {e}")

        coords_dict = {
            'gt_true': true_pos,
            'gt_aug': ds.get('gt_positions'),
            'x_noisy': ds.get('x_noisy'),
            'x_denoised': ds.get('atom_positions'),
            'gt_aligned': aligned_coords,
        }
        # debug dump: drop missing entries to avoid save failures
        coords_dict = {k: v for k, v in coords_dict.items() if v is not None}

        suffix = f"_{out_dir_suffix}" if out_dir_suffix else ""
        out_dir = self.output_dir / 'debug_0116' / f"step_{self.global_step}_{sample_id}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.save_diffusion_samples_to_cif(
                features=features,
                sample_id=sample_id,
                sample_offset=0,
                coords_dict=coords_dict,
                output_dir=str(out_dir),
                crop_indices=crop_idx,
                max_samples=self.debug_0116_max_samples,
            )
        except Exception as e:
            logging.debug(f"Failed to save CIFs: {e}")

        # Save per-sample metrics
        per_sample_mse = loss_dict.get('debug_0116_per_sample_mse', None)
        per_sample_weighted = loss_dict.get('debug_0116_per_sample_weighted_mse', None)
        noise_levels = loss_dict.get('debug_0116_noise_levels', None)
        t_values = loss_dict.get('debug_0116_t_values', None)

        if per_sample_mse is not None:
            csv_path = out_dir / 'per_sample_metrics.csv'
            mse = per_sample_mse.detach().cpu().numpy()
            w_mse = per_sample_weighted.detach().cpu().numpy() if per_sample_weighted is not None else None
            noise = noise_levels.detach().cpu().numpy() if noise_levels is not None else None
            tvals = t_values.detach().cpu().numpy() if t_values is not None else None

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'sample_id', 'sample_idx', 'mse_aligned', 'mse_weighted', 'noise_level', 't_value'
                ])
                for i in range(len(mse)):
                    writer.writerow([
                        sample_id,
                        i,
                        float(mse[i]),
                        float(w_mse[i]) if w_mse is not None else '',
                        float(noise[i]) if noise is not None else '',
                        float(tvals[i]) if tvals is not None else '',
                    ])

        if not force:
            self.debug_0116_count += 1

    def _maybe_dump_high_mse(self, output: Dict, batch: Dict, loss_dict: Dict) -> None:
        """Dump structures when mse_loss exceeds threshold."""
        if not self.is_main_process:
            return
        if self.high_mse_dump_count >= self.high_mse_max_steps:
            return
        mse_loss = loss_dict.get('mse_loss', None)
        if mse_loss is None:
            return
        if isinstance(mse_loss, torch.Tensor):
            if mse_loss.numel() != 1:
                return
            mse_value = float(mse_loss.item())
        else:
            mse_value = float(mse_loss)
        if mse_value < self.high_mse_threshold:
            return

        self.high_mse_dump_count += 1
        print(f"[High MSE Dump] mse_loss={mse_value:.4f} >= {self.high_mse_threshold}", flush=True)
        self._debug_dump_training_samples(
            output=output,
            batch=batch,
            loss_dict=loss_dict,
            force=True,
            out_dir_suffix=f"high_mse_{mse_value:.3f}",
        )

    def _check_nan_in_gradients(self, batch: Dict) -> bool:
        """Check if any gradient contains NaN and log error if found.

        Returns:
            True if NaN/Inf gradients are found, False otherwise.
        """
        nan_params = []
        inf_params = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                if torch.isinf(param.grad).any():
                    inf_params.append(name)

        if nan_params or inf_params:
            sample_ids = batch.get('sample_id', ['unknown'])
            error_msg = (
                f"[Rank {self.global_rank}, Local {self.local_rank}] "
                f"Step {self.global_step}: "
                f"Gradient contains NaN/Inf, Sample IDs: {sample_ids}\n"
            )
            if nan_params:
                error_msg += f"  Parameters with NaN gradients ({len(nan_params)} total): {nan_params[:20]}\n"
            if inf_params:
                error_msg += f"  Parameters with Inf gradients ({len(inf_params)} total): {inf_params[:20]}\n"

            # Print gradient statistics for NaN parameters to help diagnose
            for name in (nan_params + inf_params)[:5]:
                for pname, param in self.model.named_parameters():
                    if pname == name and param.grad is not None:
                        grad = param.grad
                        finite_mask = torch.isfinite(grad)
                        if finite_mask.any():
                            finite_grad = grad[finite_mask]
                            error_msg += (f"    {name}: grad shape={list(grad.shape)}, "
                                          f"finite_max={finite_grad.max().item():.6e}, "
                                          f"finite_min={finite_grad.min().item():.6e}, "
                                          f"nan_count={torch.isnan(grad).sum().item()}, "
                                          f"inf_count={torch.isinf(grad).sum().item()}\n")
                        break

            print(error_msg, flush=True)

            # Write to error log file
            with open(self.error_log_file, 'a') as f:
                f.write(error_msg + "\n")
            return True
        return False

    def train_epoch(self, train_loader, epoch: int, train_sampler=None, val_loader=None):
        """Train for one epoch"""
        self.model.train()

        # Set epoch for distributed sampler to ensure proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=(not self.is_main_process)
        )
        for batch_idx, batch in enumerate(pbar):
            try:
                # ============ DDP synchronization skip logic ============
                # Check if it is an empty batch (dummy batch returned by dataloader)
                # is_empty = batch is None or batch.get('sample_id') == '__empty_batch__'
                # skip_flag = torch.tensor([1.0 if is_empty else 0.0], device=self.device)

                # In DDP mode, ensure all ranks synchronize the skip decision
                # if self.use_ddp and dist.is_initialized():
                #     dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)

                # if skip_flag.item() > 0:
                #     if self.is_main_process:
                #         sample_ids = [batch.get('sample_id', 'unknown')] if batch else ['unknown']
                #         print(f"Warning: Non-finite input features, Sample IDs: {sample_ids}, keys={list(batch.keys()) if batch else []}. Skipping step.")
                #     continue
                # ==========================================

                step_start = time.perf_counter()
                if torch.npu.is_available():
                    torch.npu.synchronize()

                # Move data to the target device
                batch = self._to_device(batch)
                if torch.npu.is_available():
                    torch.npu.synchronize()
                data_load_time = time.perf_counter() - step_start

                # Forward pass
                loss, loss_dict, timing = self._train_step(batch)

                # Accumulate stats for averaged logging
                self._accumulate_train_stats(loss, loss_dict)

                # Track running loss
                epoch_losses.append(loss.item())

                # Update progress bar with current loss and lr (every step)
                if self.is_main_process:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

                # Detailed logging to TensorBoard (only at log_interval)
                if self.global_step % self.log_interval == 0 and self.is_main_process:
                    self._log_training(loss_dict, epoch, batch_idx)

                # Periodic checkpointing (only on global rank 0)
                if self.global_step % self.save_interval == 0 and self.is_main_process:
                    self._save_checkpoint('latest')

                # Periodically clear cache to reduce fragmentation (every 50 batches)
                if self.global_step % 50 == 0:
                    torch.npu.empty_cache()

                if torch.npu.is_available():
                    torch.npu.synchronize()
                step_time = time.perf_counter() - step_start
                if self.is_main_process:
                    print(
                        f"[Step {self.global_step}] data_load={data_load_time:.4f}s "
                        f"forward={timing['forward']:.4f}s backward={timing['backward']:.4f}s "
                        f"opt={timing['opt']:.4f}s step={step_time:.4f}s",
                        flush=True,
                    )
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        'step_s': f"{step_time:.3f}",
                        'fw_s': f"{timing['forward']:.3f}",
                        'bw_s': f"{timing['backward']:.3f}",
                    })

                self.global_step += 1

                # Validate every eval_interval_steps
                if (
                        self.eval_interval_steps is not None
                        and self.eval_interval_steps > 0
                        and val_loader is not None
                        and self.global_step % self.eval_interval_steps == 0
                ):
                    val_loss, val_loss_dict = self.evaluate(val_loader)
                    if self.is_main_process:
                        print(f"Step {self.global_step}: Val Loss = {val_loss:.4f}", flush=True)
                        self._log_validation(val_loss_dict, epoch + 1)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint('best')
                            print(f"New best model! Val Loss = {val_loss:.4f}", flush=True)
                    self.model.train()

                if (self.use_online_distillation
                        and self.eval_interval_steps is not None
                        and self.eval_interval_steps > 0
                        and self.global_step % self.eval_interval_steps == 0):
                    self._finalize_distill_epoch()
                    if self.is_main_process:
                        self._save_checkpoint(f'distill_step_{self.global_step}')
            except Exception as e:
                if self.catch_exceptions:
                    # Log exception and save batch
                    self._log_exception(e, batch, batch_idx, epoch=epoch, step=self.global_step, phase='train')
                    # Skip this batch and continue
                    continue
                else:
                    # Re-raise exception if exception handling is disabled
                    raise

        # Aggregate and synchronize loss across all processes
        avg_loss = np.mean(epoch_losses)
        if self.use_ddp:
            # Convert to tensor for all_reduce
            # loss_tensor = torch.tensor(avg_loss, device=self.device)
            loss_tensor = torch.tensor(avg_loss, dtype=torch.float, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor.item() / dist.get_world_size())

        return avg_loss

    def _train_step(self, batch: Dict) -> tuple[torch.Tensor, Dict, Dict]:
        """Single training step"""

        def _sync():
            if torch.npu.is_available():
                torch.npu.synchronize()

        # Mixed-precision training
        if self.use_amp:
            # Select dtype based on the amp_dtype flag
            dtype = torch.bfloat16 if self.amp_dtype == 'bf16' else torch.float16
            with torch_npu.npu.amp.autocast(dtype=dtype):
                _sync()
                t0 = time.perf_counter()
                # Forward pass
                output = self.model(batch)

                # Compute losses
                loss, loss_dict = self.loss_fn(output, batch)
                if isinstance(output, dict) and 'ddp_static_anchor' in output:
                    loss = loss + output['ddp_static_anchor']
                self._maybe_dump_high_mse(output, batch, loss_dict)

                # Apply gradient accumulation
                loss = loss / self.accumulation_steps
                _sync()
                t1 = time.perf_counter()

            # debug dump: one-off dump before backward (keeps tensors intact)
            if self.debug_0116_enabled:
                self._debug_dump_training_samples(output, batch, loss_dict)

            # Check for NaN/Inf in loss (will raise exception if found)
            # self._check_nan_in_loss(loss, loss_dict, batch)

            # Backward
            _sync()
            t2 = time.perf_counter()
            self._maybe_print_selected_param_loss_term_grads(output, loss_dict)
            if self.scaler is not None:  # fp16
                self.scaler.scale(loss).backward()
            else:  # bf16
                loss.backward()
            _sync()
            if (
                    self.debug_grad_chain_enabled
                    and self.debug_grad_chain_count < self.debug_grad_chain_max_steps
                    and (self.debug_grad_chain_rank < 0 or self.global_rank == self.debug_grad_chain_rank)
            ):
                model_for_debug = self.model.module if self.use_ddp else self.model
                if hasattr(model_for_debug, 'consume_debug_grad_report'):
                    debug_report = model_for_debug.consume_debug_grad_report()
                    if debug_report:
                        print(
                            f"[grad_chain][rank {self.global_rank}] step {self.global_step}",
                            flush=True,
                        )
                        for name in sorted(debug_report.keys()):
                            info = debug_report[name]
                            if 'value' in info:
                                print(f"  {name}: {info['value']}", flush=True)
                                continue
                            print(
                                "  "
                                f"{name}: shape={info.get('shape')} "
                                f"dtype={info.get('dtype')} "
                                f"requires_grad={info.get('requires_grad')} "
                                f"grad_seen={info.get('grad_seen')} "
                                f"grad_fn={info.get('grad_fn')} "
                                f"grad_abs_mean={info.get('grad_abs_mean')} "
                                f"grad_norm={info.get('grad_norm')}",
                                flush=True,
                            )
                        self.debug_grad_chain_count += 1
            self._maybe_print_selected_param_grads()
            # unused = []
            # for name, param in self.model.named_parameters():
            #     if param.grad is None and param.requires_grad:
            #         unused.append(name)
            #         # print(f"{self.global_step}: [unused] {name}")
            # print(f"{self.global_step}: unused parameters: {unused}")

            t3 = time.perf_counter()

            # Check for NaN/Inf in gradients after backward
            has_nan_grad = False  # self._check_nan_in_gradients(batch)

            # Update parameters when accumulation completes
            opt_time = 0.0
            if (self.global_step + 1) % self.accumulation_steps == 0:
                _sync()
                t4 = time.perf_counter()

                # Skip optimizer step if NaN gradients detected - prevents model weight corruption
                if has_nan_grad:
                    print(f"[Rank {self.global_rank}] Step {self.global_step}: "
                          f"Skipping optimizer step due to NaN/Inf gradients", flush=True)
                    self.optimizer.zero_grad()
                    # Still update scaler to avoid scale getting stuck
                    if self.scaler is not None:
                        self.scaler.update()
                else:
                    if self.scaler is not None:  # fp16
                        # Gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        # Check gradients again after unscaling (before clipping)
                        has_nan_after_unscale = False  # self._check_nan_in_gradients(batch)

                        if has_nan_after_unscale:
                            print(f"[Rank {self.global_rank}] Step {self.global_step}: "
                                  f"Skipping optimizer step due to NaN/Inf after unscaling", flush=True)
                            self.optimizer.zero_grad()
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.gradient_clip
                            )

                            # Optimizer step
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()

                            # Scheduler step
                            if self.scheduler is not None:
                                self.scheduler.step()
                    else:  # bf16
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip
                        )

                        # Optimizer step
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Scheduler step
                        if self.scheduler is not None:
                            self.scheduler.step()

                _sync()
                opt_time = time.perf_counter() - t4
            timing = {
                'forward': t1 - t0,
                'backward': t3 - t2,
                'opt': opt_time,
            }

        else:
            # Non-AMP path
            _sync()
            t0 = time.perf_counter()
            output = self.model(batch)
            loss, loss_dict = self.loss_fn(output, batch)
            if isinstance(output, dict) and 'ddp_static_anchor' in output:
                loss = loss + output['ddp_static_anchor']
            self._maybe_dump_high_mse(output, batch, loss_dict)

            loss = loss / self.accumulation_steps
            _sync()
            t1 = time.perf_counter()

            # debug dump: one-off dump before backward (keeps tensors intact)
            if self.debug_0116_enabled:
                self._debug_dump_training_samples(output, batch, loss_dict)

            # Check for NaN/Inf in loss (will raise exception if found)
            self._check_nan_in_loss(loss, loss_dict, batch)

            _sync()
            t2 = time.perf_counter()
            self._maybe_print_selected_param_loss_term_grads(output, loss_dict)
            loss.backward()
            _sync()
            t3 = time.perf_counter()
            self._maybe_print_selected_param_grads()

            # Check for NaN/Inf in gradients after backward
            has_nan_grad = False  # self._check_nan_in_gradients(batch)

            opt_time = 0.0
            if (self.global_step + 1) % self.accumulation_steps == 0:
                _sync()
                t4 = time.perf_counter()

                # Skip optimizer step if NaN gradients detected - prevents model weight corruption
                if has_nan_grad:
                    print(f"[Rank {self.global_rank}] Step {self.global_step}: "
                          f"Skipping optimizer step due to NaN/Inf gradients (non-AMP)", flush=True)
                    self.optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        self.scheduler.step()

                _sync()
                opt_time = time.perf_counter() - t4
            timing = {
                'forward': t1 - t0,
                'backward': t3 - t2,
                'opt': opt_time,
            }
        # if self.save_diffusion_debug and isinstance(output, dict) and 'diffusion_samples' in output:
        #     # Detach diffusion_samples to avoid keeping them in computation graph
        #     # This prevents memory accumulation during backward pass while still allowing debug saving
        #     diffusion_samples = output['diffusion_samples']
        #     if isinstance(diffusion_samples, dict):
        #         detached_samples = {}
        #         for key, value in diffusion_samples.items():
        #             if isinstance(value, torch.Tensor):
        #                 detached_samples[key] = value.detach()  # Disconnect computation graph, does not affect gradients
        #             else:
        #                 detached_samples[key] = value
        #         loss_dict['diffusion_samples'] = detached_samples
        #     else:
        #         loss_dict['diffusion_samples'] = diffusion_samples.detach() if isinstance(diffusion_samples, torch.Tensor) else diffusion_samples

        return loss * self.accumulation_steps, loss_dict, timing

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: Optional[int] = None):
        """Evaluate the model on the validation loader"""
        self.model.eval()

        all_losses = []
        all_loss_dicts = []
        val_rows = []

        pbar = tqdm(
            val_loader,
            desc="Evaluating",
            disable=(not self.is_main_process)
        )

        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            try:

                batch = self._to_device(batch)

                # Forward
                if self.use_amp:
                    # Choose dtype based on amp_dtype
                    dtype = torch.bfloat16 if self.amp_dtype == 'bf16' else torch.float16
                    with torch_npu.npu.amp.autocast(dtype=dtype):
                        output = self.model(batch)
                        loss, loss_dict = self.loss_fn(output, batch)
                else:
                    output = self.model(batch)
                    loss, loss_dict = self.loss_fn(output, batch)

                all_losses.append(loss.item())
                # Record per-sample MSE for this validation batch
                sample_ids = batch.get('sample_id', ['unknown'])
                sample_id = sample_ids[0] if isinstance(sample_ids, list) and sample_ids else sample_ids
                mse_loss = loss_dict.get('mse_loss', None)
                print(f"batc: {batch['sample_id']} mse_loss: {mse_loss}", flush=True)
                if isinstance(mse_loss, torch.Tensor):
                    if mse_loss.numel() == 1:
                        mse_value = float(mse_loss.item())
                    else:
                        mse_value = None
                elif mse_loss is None:
                    mse_value = None
                else:
                    mse_value = float(mse_loss)
                # === Compute MSE per diffusion sample ===
                per_sample_mse_list = self._compute_per_sample_mse(output, batch)
                min_mse = min(per_sample_mse_list) if per_sample_mse_list else None
                max_mse = max(per_sample_mse_list) if per_sample_mse_list else None

                # === Compute antigen iPTM (for CSV statistics and distillation) ===
                iptm_results = self._compute_antigen_xchain_iptm(output, batch)
                best_iptm = None
                best_iptm_sample_idx = None
                best_iptm_mse = None
                if iptm_results and len(iptm_results) > 0:
                    best_iptm, best_iptm_sample_idx = iptm_results[0]
                    if per_sample_mse_list and best_iptm_sample_idx < len(per_sample_mse_list):
                        best_iptm_mse = per_sample_mse_list[best_iptm_sample_idx]

                if mse_value is not None:
                    val_rows.append((
                        sample_id, mse_value,
                        min_mse if min_mse is not None else '',
                        max_mse if max_mse is not None else '',
                        best_iptm if best_iptm is not None else '',
                        best_iptm_mse if best_iptm_mse is not None else '',
                        best_iptm_sample_idx if best_iptm_sample_idx is not None else '',
                    ))

                # === Online Self-Distillation: collect high iPTM predictions (reuse iptm_results computed above) ===
                if self.use_online_distillation:
                    if iptm_results is not None and len(iptm_results) > 0:
                        distill_best_iptm, distill_best_idx = iptm_results[0]
                        if distill_best_iptm >= self.distill_iptm_threshold:
                            sample_dict = self._save_distill_sample(output, batch, distill_best_idx)
                            if sample_dict is not None:
                                self._current_epoch_distill_samples.append(sample_dict)
                                print(f"[Distill] Rank {self.global_rank}: "
                                      f"{sample_id} iPTM={distill_best_iptm:.4f} >= {self.distill_iptm_threshold} -> saved",
                                      flush=True)

                scalar_loss_dict = {}
                for k, v in loss_dict.items():
                    if isinstance(v, dict):
                        continue
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            scalar_loss_dict[k] = v.item()
                        else:
                            # Skip non-scalar tensors (e.g., debug per-sample vectors)
                            continue
                    elif v is None:
                        continue
                    else:
                        scalar_loss_dict[k] = float(v)
                all_loss_dicts.append(scalar_loss_dict)

                # Immediately save diffusion debug structures for this batch (all ranks)
                # Each rank saves to its own subdirectory to avoid conflicts
                if (self.save_diffusion_debug and
                        self.data_dirs is not None and
                        self.save_val_structures_epoch_interval > 0 and
                        self.epoch % self.save_val_structures_epoch_interval == 0):
                    # Check if we should save this batch based on max_batches_to_save limit
                    if self.max_batches_to_save is None or batch_idx < self.max_batches_to_save:
                        self._save_val_diffusion_debug_for_batch(output, batch, 'val')

            except Exception as e:
                if self.catch_exceptions:
                    # Log exception and save batch
                    self._log_exception(e, batch, batch_idx, phase='val')
                    # Skip this batch and continue
                    continue
                else:
                    # Re-raise exception if exception handling is disabled
                    raise

        # Save per-sample MSE to CSV (each rank saves its own file)
        if val_rows:
            val_metrics_dir = self.output_dir / 'val_metrics'
            val_metrics_dir.mkdir(parents=True, exist_ok=True)
            csv_path = val_metrics_dir / f'epoch_{self.epoch}_rank_{self.global_rank}_mse.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'mse_loss', 'min_mse', 'max_mse', 'best_iptm', 'best_iptm_mse',
                                 'best_iptm_sample_idx'])
                writer.writerows(val_rows)

        # Wait for all ranks to finish writing their CSV files, then merge on rank 0
        if self.use_ddp:
            dist.barrier()
        if self.is_main_process and self.use_ddp:
            merged_rows = []
            world_size = dist.get_world_size()
            for r in range(world_size):
                rank_csv = val_metrics_dir / f'epoch_{self.epoch}_rank_{r}_mse.csv'
                if rank_csv.exists():
                    with open(rank_csv, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)  # skip header
                        for row in reader:
                            merged_rows.append(row)
            if merged_rows:
                merged_csv_path = val_metrics_dir / f'epoch_{self.epoch}_mse.csv'
                with open(merged_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['sample_id', 'mse_loss', 'min_mse', 'max_mse', 'best_iptm', 'best_iptm_mse',
                                     'best_iptm_sample_idx'])
                    writer.writerows(merged_rows)
                print(f"[Rank 0] Merged {len(merged_rows)} val MSE rows from {world_size} ranks -> {merged_csv_path}", flush=True)
        elif not self.use_ddp and val_rows:
            # Non-DDP: rename the single-rank file to the standard name
            standard_csv = val_metrics_dir / f'epoch_{self.epoch}_mse.csv'
            rank0_csv = val_metrics_dir / f'epoch_{self.epoch}_rank_0_mse.csv'
            if rank0_csv.exists() and not standard_csv.exists():
                rank0_csv.rename(standard_csv)

        # Aggregate validation metrics
        if len(all_losses) == 0:
            avg_loss = 0.0
            avg_loss_dict = {}
        else:
            avg_loss = np.mean(all_losses)
            if len(all_loss_dicts) > 0:
                avg_loss_dict = {}
                all_keys = set().union(*(d.keys() for d in all_loss_dicts))
                for key in all_keys:
                    values = [d[key] for d in all_loss_dicts if key in d]
                    if len(values) == 0:
                        continue
                    avg_loss_dict[key] = float(np.mean(values))
            else:
                avg_loss_dict = {}

        # Synchronize validation metrics across all processes
        print(f"avg_loss before: {avg_loss}", flush=True)
        print(f"avg_loss_dict before: {avg_loss_dict}", flush=True)
        if self.use_ddp:
            # Synchronize average loss
            # loss_tensor = torch.tensor(avg_loss, device=self.device)
            loss_tensor = torch.tensor(avg_loss, dtype=torch.float, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor.item() / dist.get_world_size())

            # Synchronize loss dict components
            if len(avg_loss_dict) > 0:
                synchronized_loss_dict = {}
                for key, value in avg_loss_dict.items():
                    # value_tensor = torch.tensor(value, device=self.device)
                    value_tensor = torch.tensor(value, dtype=torch.float, device=self.device)
                    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                    synchronized_loss_dict[key] = (value_tensor.item() / dist.get_world_size())
                avg_loss_dict = synchronized_loss_dict

                # return avg_loss, avg_loss_dict
        print(f"avg_loss after: {avg_loss}", flush=True)
        print(f"avg_loss_dict after: {avg_loss_dict}", flush=True)
        return avg_loss, avg_loss_dict

    @torch.no_grad()
    def _compute_per_sample_mse(
            self,
            output: dict,
            batch: dict,
    ) -> Optional[list]:
        """
        Compute aligned MSE for each diffusion sample, used for validation statistics.

        Returns:
            List of per-sample MSE values (aligned), or None if cannot compute.
        """
        atom_positions = output.get('diffusion_samples', {}).get('atom_positions', None)
        if atom_positions is None or atom_positions.ndim != 4:
            return None

        true_pos = batch.get('true_positions')
        atom_mask = batch.get('true_positions_atom_mask')
        if true_pos is None or atom_mask is None:
            return None

        try:
            atoms_per_token = atom_mask.shape[1]
            mask_float = atom_mask.float()
            valid_mask = mask_float.reshape(-1).bool()

            if not valid_mask.any():
                return None

            # Build alignment weights (consistent with loss)
            weights = mask_float.clone()
            is_dna = batch.get('is_dna')
            is_rna = batch.get('is_rna')
            is_ligand = batch.get('is_ligand')
            if is_ligand is None and is_dna is not None:
                is_ligand = torch.zeros_like(is_dna)
            cfg = get_default_config()
            if is_dna is not None:
                weights = weights + is_dna.unsqueeze(-1).expand(-1, atoms_per_token).float() * cfg['alpha_dna']
            if is_rna is not None:
                weights = weights + is_rna.unsqueeze(-1).expand(-1, atoms_per_token).float() * cfg['alpha_rna']
            if is_ligand is not None:
                weights = weights + is_ligand.unsqueeze(-1).expand(-1, atoms_per_token).float() * cfg['alpha_ligand']
            weights = weights * mask_float
            weights_valid = weights.reshape(-1)[valid_mask]

            gt_coords_valid = true_pos.reshape(-1, 3)[valid_mask]  # [N_valid, 3]
            rigid_align = WeightedRigidAlign()
            n_samples = atom_positions.shape[0]
            per_sample_mse = []

            for s in range(n_samples):
                pred_valid = atom_positions[s].reshape(-1, 3)[valid_mask]  # [N_valid, 3]
                # Align GT -> pred
                aligned_gt = rigid_align(
                    gt_coords_valid.unsqueeze(0),  # [1, N_valid, 3]
                    pred_valid.unsqueeze(0),  # [1, N_valid, 3]
                    weights_valid,
                )  # [1, N_valid, 3]
                diff = (pred_valid.unsqueeze(0) - aligned_gt) ** 2  # [1, N_valid, 3]
                mse = (diff.sum(dim=-1) * weights_valid.unsqueeze(0)).sum() / weights_valid.sum().clamp(min=1)
                per_sample_mse.append(mse.item())

            return per_sample_mse
        except Exception as e:
            print(f"Warning: Failed to compute per-sample MSE: {e}", flush=True)
            return None

    # ================================================================
    # Online Self-Distillation methods
    # ================================================================

    @torch.no_grad()
    def _compute_antigen_xchain_iptm(
            self,
            output: dict,
            batch: dict,
    ) -> Optional[list]:
        """
        Compute cross-chain iPTM for the antigen chain (the last chain).

        Compute from the tmscore_adjusted_pae_interface output of the confidence head,
        extract the antigen chain's average iPTM against antibody chains.

        Returns:
            List of (xchain_iptm, sample_idx) sorted by iPTM descending,
            or None if cannot compute.
        """
        tm_interface = output.get('tmscore_adjusted_pae_interface', None)
        if tm_interface is None:
            return None

        asym_id = batch.get('asym_id', None)
        if asym_id is None:
            return None
        if isinstance(asym_id, torch.Tensor):
            if asym_id.dim() > 1:
                asym_id = asym_id[0]
            asym_id_cpu = asym_id.cpu()
        else:
            asym_id_cpu = torch.tensor(asym_id)

        unique_chains = torch.unique(asym_id_cpu)
        n_chains = len(unique_chains)

        if n_chains < 2:
            return None  # Single chain, no interface

        n_samples = tm_interface.shape[0]

        results = []
        for s in range(n_samples):
            tm_s = tm_interface[s].cpu()  # [N_token, N_token]

            # Build chain pair iPTM matrix
            chain_pair = torch.zeros(n_chains, n_chains)
            for i, ci in enumerate(unique_chains):
                for j, cj in enumerate(unique_chains):
                    mask_i = asym_id_cpu == ci
                    mask_j = asym_id_cpu == cj
                    block = tm_s[mask_i][:, mask_j]
                    if block.numel() > 0:
                        chain_pair[i, j] = block.mean()

            # Antigen chain cross-chain iPTM: average with all non-self chains
            ant_idx = n_chains - 1
            off_diag = [chain_pair[ant_idx, j].item()
                        for j in range(n_chains) if j != ant_idx]
            xchain_iptm = sum(off_diag) / len(off_diag) if off_diag else 0.0
            results.append((xchain_iptm, s))

        # Sort by iPTM descending
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def _save_distill_sample(
            self,
            output: dict,
            batch: dict,
            best_sample_idx: int,
    ) -> Optional[dict]:
        """
        Save predicted coordinates with high iPTM as pseudo ground truth.

        Returns:
            sample dict for train_list.json, or None if save failed.
        """
        # Get sample_id
        sample_ids = batch.get('sample_id', ['unknown'])
        if isinstance(sample_ids, list) and len(sample_ids) > 0:
            sample_id = sample_ids[0]
        elif isinstance(sample_ids, str):
            sample_id = sample_ids
        else:
            return None

        # Get predicted coordinates
        atom_positions = output.get('diffusion_samples', {}).get('atom_positions', None)
        if atom_positions is None:
            return None

        pred_coords = atom_positions[best_sample_idx]  # [N_token, 24, 3]

        # Find original feature file
        feature_file = None
        for data_dir_path in self.data_dirs:
            candidate = data_dir_path / 'val' / f"{sample_id}_features.pkl"
            if candidate.exists():
                feature_file = candidate
                break

        if feature_file is None:
            print(f"[Distill] Feature file not found for {sample_id}, skip.", flush=True)
            return None

        # Load original features
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)

        # Handle cropping: if crop_indices exist, scatter back to full length
        crop_idx = batch.get('crop_indices')
        pred_np = pred_coords.detach().cpu().numpy()

        if crop_idx is not None:
            true_pos = features['true_positions']
            if isinstance(true_pos, torch.Tensor):
                true_pos = true_pos.numpy()
            new_pos = true_pos.copy()
            crop_idx_np = crop_idx.cpu().numpy() if isinstance(crop_idx, torch.Tensor) else crop_idx
            new_pos[crop_idx_np] = pred_np
            features['true_positions'] = new_pos
        else:
            features['true_positions'] = pred_np

        # Save to distillation directory
        train_dir = self._distill_dir / 'train'
        train_dir.mkdir(parents=True, exist_ok=True)

        new_id = f"{sample_id}_distill_ep{self.epoch}"
        out_path = train_dir / f"{new_id}_features.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(features, f)

        num_res = len(features.get('aatype', []))
        return {
            'id': new_id,
            'num_res': num_res,
            'source': 'distillation',
            'original_id': sample_id,
            'distill_epoch': self.epoch,
        }

    def _finalize_distill_epoch(self):
        """
        After validation, aggregate distillation data, clean old epochs, and rebuild train_list.json.
        """

        # Each rank saves its own collected sample list
        rank_list_file = self._distill_dir / f'train_list_rank_{self.global_rank}_ep_{self.epoch}.json'
        with open(rank_list_file, 'w') as f:
            json.dump(self._current_epoch_distill_samples, f)

        # Wait for all ranks to finish writing
        if self.use_ddp:
            dist.barrier()

        # Rank 0 merges lists from all ranks
        if self.is_main_process:
            # Merge all rank data for this epoch
            epoch_samples = []
            for r in range(self.world_size):
                rfile = self._distill_dir / f'train_list_rank_{r}_ep_{self.epoch}.json'
                if rfile.exists():
                    with open(rfile, 'r') as f:
                        epoch_samples.extend(json.load(f))

            # Save merged list for this epoch
            epoch_list = self._distill_dir / f'train_list_ep_{self.epoch}.json'
            with open(epoch_list, 'w') as f:
                json.dump(epoch_samples, f)

            # Combine data from the last keep_epochs rounds
            oldest_keep = max(0, self.epoch - self.distill_keep_epochs + 1)
            combined = []
            for ep in range(oldest_keep, self.epoch + 1):
                ep_file = self._distill_dir / f'train_list_ep_{ep}.json'
                if ep_file.exists():
                    with open(ep_file, 'r') as f:
                        combined.extend(json.load(f))

            # Deduplicate by original_id, keeping the sample with the largest distill_epoch
            dedup_map = {}  # original_id -> sample_dict (keep largest epoch)
            for sample in combined:
                oid = sample.get('original_id', sample.get('id'))
                ep = sample.get('distill_epoch', -1)
                if oid not in dedup_map or ep > dedup_map[oid].get('distill_epoch', -1):
                    dedup_map[oid] = sample
            n_before_dedup = len(combined)
            combined = list(dedup_map.values())

            # Write final train_list.json
            train_list_path = self._distill_dir / 'train_list.json'
            with open(train_list_path, 'w') as f:
                json.dump(combined, f)

            # Clean old epoch data
            self._cleanup_old_distill_epochs(oldest_keep)

            print(f"[Distill] Epoch {self.epoch}: "
                  f"{len(epoch_samples)} new samples this epoch, "
                  f"{n_before_dedup} total before dedup, "
                  f"{len(combined)} unique samples after dedup "
                  f"(keep_epochs={self.distill_keep_epochs})",
                  flush=True)

        # Wait for rank 0 to finish
        if self.use_ddp:
            dist.barrier()

        # Reset the current epoch's collection list
        self._current_epoch_distill_samples = []

    def _cleanup_old_distill_epochs(self, oldest_keep: int):
        """Clean old epoch distillation data (called only on rank 0)."""

        # Delete old epoch list files and rank list files
        for f in self._distill_dir.iterdir():
            if not f.is_file():
                continue
            name = f.name
            # train_list_ep_N.json
            if name.startswith('train_list_ep_'):
                try:
                    ep = int(name.split('_ep_')[1].replace('.json', ''))
                    if ep < oldest_keep:
                        f.unlink()
                except (ValueError, IndexError):
                    pass
            # train_list_rank_R_ep_N.json
            elif name.startswith('train_list_rank_'):
                try:
                    ep = int(name.split('_ep_')[1].replace('.json', ''))
                    if ep < oldest_keep:
                        f.unlink()
                except (ValueError, IndexError):
                    pass

        # Delete old pkl files
        train_dir = self._distill_dir / 'train'
        if train_dir.exists():
            for f in train_dir.iterdir():
                if not f.name.endswith('_features.pkl'):
                    continue
                # Filename format: {sample_id}_distill_ep{N}_features.pkl
                try:
                    ep_part = f.stem.split('_distill_ep')[1].split('_')[0]
                    ep = int(ep_part)
                    if ep < oldest_keep:
                        f.unlink()
                except (ValueError, IndexError):
                    pass

    def _rebuild_train_loader_with_distill(self, original_dataset):
        """
        Rebuild training DataLoader with distillation data + original data.

        Returns:
            (new_loader, new_sampler) or (None, None) if no distill data.
        """
        from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
        from train_dataloader import TorchFoldDataset, CollateFn

        train_list_path = self._distill_dir / 'train_list.json'
        if not train_list_path.exists():
            return None, None

        with open(train_list_path, 'r') as f:
            distill_samples = json.load(f)
        if not distill_samples:
            return None, None

        cfg = self._config
        if cfg is None:
            print("[Distill] Warning: config not stored, cannot rebuild loader.", flush=True)
            return None, None

        # Create distillation dataset
        distill_dataset = TorchFoldDataset(
            data_dir=str(self._distill_dir),
            split='train',
            max_num_res=cfg.max_num_res,
            enable_cropping=cfg.enable_cropping,
            crop_size=cfg.crop_size,
            crop_complete_ligand_unstdRes=cfg.crop_complete_ligand_unstdRes,
            spatial_crop_complete_ligand_unstdRes=cfg.spatial_crop_complete_ligand_unstdRes,
            drop_last=cfg.drop_last,
            remove_metal=cfg.remove_metal,
            crop_method_weights=cfg.crop_method_weights,
            interface_minimal_distance=cfg.interface_minimal_distance,
            max_templates=cfg.max_templates,
            data_source_idx=len(cfg.data_dirs),  # New data source index
            remove_unresolved_tokens=cfg.remove_unresolved_tokens,
        )

        if len(distill_dataset) == 0:
            return None, None

        combined = ConcatDataset([original_dataset, distill_dataset])
        collate_fn = CollateFn()

        if self.use_ddp:
            sampler = DistributedSampler(
                combined,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=True,
            )
            loader = DataLoader(
                combined,
                batch_size=self._config.batch_size,
                sampler=sampler,
                num_workers=self._config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
            )
            if self.is_main_process:
                print(f"[Distill] Rebuilt train loader: {len(original_dataset)} original + "
                      f"{len(distill_dataset)} distill = {len(combined)} total", flush=True)
            return loader, sampler
        else:
            loader = DataLoader(
                combined,
                batch_size=self._config.batch_size,
                shuffle=True,
                num_workers=self._config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
            )
            print(f"[Distill] Rebuilt train loader: {len(original_dataset)} original + "
                  f"{len(distill_dataset)} distill = {len(combined)} total", flush=True)
            return loader, None

    def _save_val_diffusion_debug_for_batch(
            self,
            output: dict,
            batch: dict,
            split
    ):
        """
        Save diffusion debug structures for a single batch immediately after processing.
        This allows viewing results without waiting for the entire validation epoch to complete.

        Args:
            output: Model output dictionary for a single batch
            batch: Input batch dictionary
            val_loader: Validation dataloader (to get dataset split info)
        """

        ds = output.get('diffusion_samples', {})
        per_step = ds.get('per_step_positions', None)

        # Extract sample_id directly from batch
        sample_ids = batch.get('sample_id', ['unknown'])
        if isinstance(sample_ids, list) and len(sample_ids) > 0:
            sample_id = sample_ids[0]
        elif isinstance(sample_ids, str):
            sample_id = sample_ids
        else:
            sample_id = 'unknown'

        # Get split from dataset (needed for feature file path)
        # dataset = val_loader.dataset
        # split = dataset.split

        for sample_offset, per_sample_steps in enumerate(per_step):
            if per_sample_steps is None:
                continue

            coords_dict = {}
            total_steps = len(per_sample_steps)
            # Only save the last N steps if max_diffusion_steps_to_save is set
            start_step = 0
            if self.max_diffusion_steps_to_save is not None and total_steps > self.max_diffusion_steps_to_save:
                start_step = total_steps - self.max_diffusion_steps_to_save

            for step_i, arr in enumerate(per_sample_steps):
                if arr is None:
                    continue
                # Only save steps from start_step onwards
                if step_i >= start_step:
                    coords_dict[f"per_step_{step_i}"] = arr

            if not coords_dict:
                continue

            # Create a directory per epoch and per rank to avoid multi‑card write conflicts
            out_dir = (
                    self.output_dir
                    / 'save_val_structures'
                    / f'epoch_{self.epoch}'
                    / f'rank_{self.global_rank}'
            )

            # Find the correct data directory for this sample
            feature_file = None
            for data_dir_path in self.data_dirs:
                candidate_file = data_dir_path / split / f"{sample_id}_features.pkl"
                if candidate_file.exists():
                    feature_file = candidate_file
                    break

            if feature_file is None:
                print(f"Warning: Feature file not found for sample {sample_id}, skipping CIF save.", flush=True)
                continue

            with open(feature_file, 'rb') as f:
                features = pickle.load(f)

            # Handle both numpy array and torch.Tensor cases
            true_pos = features['true_positions']
            if isinstance(true_pos, np.ndarray):
                true_pos = torch.from_numpy(true_pos).to(self.device)
            elif isinstance(true_pos, torch.Tensor):
                true_pos = true_pos.to(self.device)
            else:
                raise TypeError(f"true_positions must be numpy.ndarray or torch.Tensor, got {type(true_pos)}")
            crop_idx = batch.get('crop_indices')
            if crop_idx is not None:
                # Ensure indices are on the same device and integer type
                if isinstance(crop_idx, torch.Tensor):
                    crop_idx = crop_idx.to(true_pos.device)
                # Assume crop indices are along the token dimension
                true_pos = true_pos[crop_idx]
            # Save the ground truth only for the first sample
            if sample_offset == 0:
                coords_dict['true_positions'] = true_pos

            # Pass the already loaded features to avoid re-reading in save_diffusion_samples_to_cif
            self.save_diffusion_samples_to_cif(
                features=features,
                sample_id=sample_id,
                sample_offset=sample_offset,
                coords_dict=coords_dict,  ### Ensure keys are names, values are tensors with shape [N_token, 24, 3]
                output_dir=str(out_dir),
                crop_indices=crop_idx,
            )

    def _save_structures(
            self,
            outputs: list,
            batches: list,
            batch_indices: list,
            val_loader,
            epoch: int
    ):
        """Extract and save predicted structures from model outputs.

        Args:
            outputs: List of model output dictionaries
            batches: List of input batch dictionaries (not used if reloading pickle)
            batch_indices: List of batch indices for reloading pickle files
            val_loader: Validation dataloader to get sample IDs
            epoch: Current epoch number
        """
        try:
            import pickle

            for batch_idx, (output, batch, batch_idx_in_loader) in enumerate(zip(outputs, batches, batch_indices)):
                # Check if cropping is enabled - if yes, must use current batch (cropped)
                # If no cropping, can reload original pickle file
                dataset = val_loader.dataset
                use_reload = (self.data_dir is not None and
                              not (dataset.enable_cropping and dataset.crop_size is not None))

                if use_reload:
                    # Reload original pickle file to get clean numpy BatchDict
                    # Similar to extract_structures in run_torchfold.py (lines 388-390)
                    # Get sample_id from dataloader dataset
                    # Calculate actual sample index (accounting for batch_size and drop_last)
                    sample_idx = batch_idx_in_loader * val_loader.batch_size
                    if sample_idx >= len(dataset):
                        print(f"Warning: sample_idx {sample_idx} >= dataset length {len(dataset)}, skipping...", flush=True)
                        continue

                    sample_info = dataset.samples[sample_idx]
                    sample_id = sample_info['id']
                    split = dataset.split  # 'val' or 'train'

                    # Load original pickle file - search in all data directories
                    feature_file = None
                    for data_dir_path in self.data_dirs:
                        candidate_file = data_dir_path / split / f"{sample_id}_features.pkl"
                        if candidate_file.exists():
                            feature_file = candidate_file
                            break

                    if feature_file is None:
                        print(
                            f"Warning: Pickle file not found for sample {sample_id} in any data directory, using batch conversion instead.", flush=True)
                        use_reload = False

                if use_reload:
                    with open(feature_file, 'rb') as f:
                        features = pickle.load(f)

                    # Remove ground truth fields (similar to what extract_structures expects)
                    # Keep only featurised_example fields, remove true_feats
                    ground_truth_fields = [
                        'true_positions', 'true_positions_atom_mask'

                    ]
                    batch_dict = {k: v for k, v in features.items() if k not in ground_truth_fields}

                    # Ensure all values are numpy arrays (they should be already)
                    # But convert any torch.Tensor if present
                    batch_dict = pytree.tree_map_only(
                        torch.Tensor,
                        lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x,
                        batch_dict
                    )
                else:
                    # Fallback: Convert batch from torch.Tensor to numpy BatchDict format
                    # This is used when cropping is enabled or pickle reload fails
                    batch_dict = pytree.tree_map_only(
                        torch.Tensor,
                        lambda x: x.cpu().detach().numpy(),
                        batch
                    )

                # Process model output: convert to ModelResult format
                # Similar to run_inference in run_torchfold.py (lines 296-311)
                result = output.copy()

                # Add __identifier__ (model identifier)
                # Handle DDP case: if model is wrapped, access module
                model_to_check = self.model.module if self.use_ddp else self.model

                if hasattr(model_to_check, '__identifier__'):
                    identifier = model_to_check.__identifier__
                    if isinstance(identifier, torch.Tensor):
                        result['__identifier__'] = identifier.cpu().numpy()
                    elif isinstance(identifier, np.ndarray):
                        result['__identifier__'] = identifier
                    else:
                        result['__identifier__'] = np.array(identifier)
                else:
                    # Use a default identifier if not available
                    result['__identifier__'] = np.array([0], dtype=np.uint8)

                # Convert tensors to float32 and then to numpy (same as run_inference)
                result = pytree.tree_map_only(
                    torch.Tensor,
                    lambda x: x.to(dtype=torch.float32) if x.dtype == torch.bfloat16 else x,
                    result,
                )
                result = pytree.tree_map_only(
                    torch.Tensor,
                    lambda x: x.cpu().detach().numpy(),
                    result
                )

                # Convert __identifier__ to bytes format (same as run_inference line 309)
                if isinstance(result['__identifier__'], np.ndarray):
                    result['__identifier__'] = result['__identifier__'].tobytes()

                # Extract structures using model.Model.get_inference_result
                # Similar to extract_structures in run_torchfold.py (lines 313-328)
                inference_results = list(
                    post_processing.get_inference_result(
                        batch=batch_dict,
                        result=result,
                        target_name=f"epoch_{epoch}_batch_{batch_idx}"
                    )
                )

                # Save structures (similar to write_outputs in run_torchfold.py, lines 427-470)
                ranking_scores = []
                max_ranking_score = None
                max_ranking_result = None

                for sample_idx, inference_result in enumerate(inference_results):
                    sample_dir = self.structures_dir / f"epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}"
                    sample_dir.mkdir(exist_ok=True, parents=True)

                    post_processing.write_output(
                        inference_result=inference_result,
                        output_dir=str(sample_dir)
                    )

                    # Track ranking scores (similar to write_outputs)
                    ranking_score = float(inference_result.metadata['ranking_score'])
                    ranking_scores.append((batch_idx, sample_idx, ranking_score))
                    if max_ranking_score is None or ranking_score > max_ranking_score:
                        max_ranking_score = ranking_score
                        max_ranking_result = inference_result

                # Save best structure to parent directory (similar to write_outputs)
                if max_ranking_result is not None:
                    batch_dir = self.structures_dir / f"epoch_{epoch}_batch_{batch_idx}"
                    batch_dir.mkdir(exist_ok=True, parents=True)
                    post_processing.write_output(
                        inference_result=max_ranking_result,
                        output_dir=str(batch_dir),
                        name=f"epoch_{epoch}_batch_{batch_idx}_best"
                    )

                    # Save ranking scores CSV
                    import csv
                    csv_path = batch_dir / 'ranking_scores.csv'
                    with open(csv_path, 'wt') as f:
                        writer = csv.writer(f)
                        writer.writerow(['batch_idx', 'sample', 'ranking_score'])
                        writer.writerows(ranking_scores)

                if self.is_main_process:
                    print(
                        f"Saved {len(inference_results)} structure(s) for batch {batch_idx} (best score: {max_ranking_score:.4f})", flush=True)

        except Exception as e:
            print(f"Error saving structures: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # --------------------------------------------------------------------------- #
    # Utility: save diffusion samples (gt / noisy / denoised) to CIF using pickle #
    # --------------------------------------------------------------------------- #

    def save_diffusion_samples_to_cif(
            self,
            features: Dict,
            sample_id: str,
            sample_offset: int,
            coords_dict: Dict[str, torch.Tensor],
            output_dir: str,
            crop_indices: Optional[torch.Tensor] = None,
            max_samples: int = 0,
    ):
        """
        Save diffusion samples (e.g., x_gt_augment / x_noisy / x_denoised) to CIF files.

        Args:
            features: Feature dict loaded from <data_dir>/<split>/{sample_id}_features.pkl
            sample_id: Sample identifier
            coords_dict: Dict of name -> coords tensor/array with shape
                        [N_sample, N_token, atoms_per_token, 3] or [N_token, atoms_per_token, 3]
            output_dir: Destination directory to write CIFs
        """
        from pathlib import Path
        # Correctly import AtomLayout helpers
        from torchfold.processing.atom_layout.atom_layout import (
            AtomLayout,
            compute_gather_idxs,
            convert,
        )

        output_dir = Path(output_dir)
        required_keys = ['token_atoms_layout', 'flat_output_layout', 'empty_output_struc']
        for k in required_keys:
            if k not in features:
                raise KeyError(f"Missing key '{k}' in features for sample_id={sample_id}")

        token_atoms_layout = features['token_atoms_layout']
        flat_output_layout = features['flat_output_layout']
        empty_output_struc = features['empty_output_struc']

        def _ensure_atom_layout(obj):
            # Already correct
            if isinstance(obj, AtomLayout):
                return obj
            # Packed in object array
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                obj = obj.item()
                if isinstance(obj, AtomLayout):
                    return obj
            # Dict-like with required fields
            required = ('atom_name', 'res_id', 'chain_id')
            if isinstance(obj, dict) and all(k in obj for k in required):
                return AtomLayout(
                    atom_name=obj['atom_name'],
                    res_id=obj['res_id'],
                    chain_id=obj['chain_id'],
                    atom_element=obj.get('atom_element', None),
                    res_name=obj.get('res_name', None),
                    chain_type=obj.get('chain_type', None),
                )
            raise TypeError(f"Cannot convert layout object of type {type(obj)} to AtomLayout")

        def _unwrap(obj):
            # If it's an object array with single element, unwrap
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                obj = obj.item()
            return obj

        token_atoms_layout = _unwrap(token_atoms_layout)
        flat_output_layout = _unwrap(flat_output_layout)
        empty_output_struc = _unwrap(empty_output_struc)

        token_atoms_layout = _ensure_atom_layout(token_atoms_layout)
        flat_output_layout = _ensure_atom_layout(flat_output_layout)
        if not hasattr(empty_output_struc, 'copy_and_update_atoms'):
            raise TypeError(f"empty_output_struc is not a Structure (type={type(empty_output_struc)})")

        gather = compute_gather_idxs(
            source_layout=token_atoms_layout,
            target_layout=flat_output_layout,
        )

        sentinel_value = 1e9  # Placeholder coordinates for uncropped parts
        n_full = token_atoms_layout.atom_name.shape[0]

        # If cropping is used, scatter cropped coords back to full length before convert
        def _scatter_to_full(coords, crop_idx_np: np.ndarray):
            # coords: [N_crop, A, 3], crop_idx_np: [N_crop]
            if coords.ndim != 3:
                raise ValueError(f"Expected cropped coords dim=3, got {coords.shape}")
            # Use token count (first dim of token_atoms_layout) for full length
            atoms_per_token_local = coords.shape[1]
            full = np.full((n_full, atoms_per_token_local, 3), sentinel_value, dtype=coords.dtype)
            full[crop_idx_np] = coords
            return full

        def _to_struct(arr):
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            # Handle [N_sample, N_token, atoms, 3] or [N_token, atoms, 3]
            if arr.ndim == 4:
                arr_list = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 3:
                arr_list = [arr]
            else:
                raise ValueError(f"Unexpected coords shape: {arr.shape}")

            structs = []
            for coords in arr_list:
                idx_np = crop_indices.detach().cpu().numpy() if crop_indices is not None else None
                if crop_indices is not None:
                    coords = _scatter_to_full(coords, idx_np)
                coords_flat = convert(
                    gather_info=gather,
                    arr=coords,
                    layout_axes=(-3, -2),
                )  # [N_atoms, 3]
                # If cropping used, filter out placeholder atoms that are all sentinel
                if crop_indices is not None:
                    mask = ~(coords_flat == sentinel_value).all(axis=-1)
                    keep_idx = np.nonzero(mask)[0].astype(np.int64)
                    coords_flat = coords_flat[keep_idx]
                    bfactor_flat = np.zeros_like(coords_flat[..., 0], dtype=np.float32)
                    occ = np.ones_like(coords_flat[..., 0], dtype=np.float32)
                    struct = empty_output_struc.copy_and_update_atoms(
                        atom_x=coords_flat[..., 0],
                        atom_y=coords_flat[..., 1],
                        atom_z=coords_flat[..., 2],
                        atom_b_factor=bfactor_flat,
                        atom_occupancy=occ,
                    )
                else:
                    bfactor_flat = np.zeros_like(coords_flat[..., 0])
                    occ = np.ones_like(coords_flat[..., 0])
                    struct = empty_output_struc.copy_and_update_atoms(
                        atom_x=coords_flat[..., 0],
                        atom_y=coords_flat[..., 1],
                        atom_z=coords_flat[..., 2],
                        atom_b_factor=bfactor_flat,
                        atom_occupancy=occ,
                    )
                structs.append(struct)
            return structs

        # Simple CIF writer using Structure built-in (avoids post_processing expectations)
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, arr in coords_dict.items():
            structs = _to_struct(arr)
            if max_samples is not None and max_samples > 0:
                structs = structs[:max_samples]
            for idx, struct in enumerate(structs):
                # Directly save .cif file, naming format: {sample_id}_sample_{sample_offset+idx}_{name}.cif
                cif_filename = f"{sample_id}_sample_{sample_offset + idx}_{name}.cif"
                cif_path = output_dir / cif_filename
                # Use Structure's to_mmCIF (expects numpy arrays)
                with open(cif_path, "w") as f:
                    f.write(struct.to_mmcif())

    def _to_device(self, batch: Dict) -> Dict:
        """Move every tensor in the batch onto the target device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _log_training(self, loss_dict: Dict, epoch: int, batch_idx: int):
        """Log training metrics to TensorBoard"""
        if self.writer is None:
            return

        # MSE loss partitioned by t interval, grouped under train/mse_by_t/
        mse_by_t_keys = {'mse_t_', 'mse_weighted_t_'}

        # Record averaged losses over log_interval steps
        for key, values in self.train_stats.items():
            if not values:
                continue
            value = float(np.mean(values))
            if any(key.startswith(prefix) for prefix in mse_by_t_keys):
                self.writer.add_scalar(f'train/mse_by_t/{key}', value, self.global_step)
            else:
                self.writer.add_scalar(f'train/{key}', value, self.global_step)

        # Track learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/lr', lr, self.global_step)

        # Clear stats after logging to start next window
        for key in list(self.train_stats.keys()):
            self.train_stats[key].clear()

    def _accumulate_train_stats(self, loss: torch.Tensor, loss_dict: Dict) -> None:
        """Accumulate scalar metrics for averaged logging."""
        # Always track total loss
        if 'loss' not in self.train_stats:
            self.train_stats['loss'] = []
        self.train_stats['loss'].append(float(loss.item()))

        for key, value in loss_dict.items():
            # Skip complex entries (e.g., diffusion_samples dict)
            if isinstance(value, dict):
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    # Non-scalar tensors are skipped
                    continue
            elif value is None:
                continue
            else:
                value = float(value)

            if key not in self.train_stats:
                self.train_stats[key] = []
            self.train_stats[key].append(float(value))

    def _log_validation(self, loss_dict: Dict, epoch: int):
        """Log validation metrics to TensorBoard"""
        if self.writer is None:
            return

        for key, value in loss_dict.items():
            print(f"key: {key}, value: {value}", flush=True)
            self.writer.add_scalar(f'val/{key}', value, epoch)

    def _save_checkpoint(self, name: str = 'latest'):
        """Persist a checkpoint to disk"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict() if not self.use_ddp
            else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save fixed-name checkpoint (will be overwritten)
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}", flush=True)

        # Also save a timestamped version (won't be overwritten) for latest and best
        if name in ['latest', 'best']:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if name == 'latest':
                timestamped_name = f'{name}_step_{self.global_step}_epoch_{self.epoch}_{timestamp}.pt'
            else:  # best
                timestamped_name = f'{name}_val_loss_{self.best_val_loss:.6f}_epoch_{self.epoch}_{timestamp}.pt'

            timestamped_path = self.checkpoint_dir / timestamped_name
            torch.save(checkpoint, timestamped_path)
            print(f"Saved timestamped checkpoint to {timestamped_path}", flush=True)

    def load_checkpoint(self, checkpoint_path: str, resume_optimizer: bool = True):
        """Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file.
            resume_optimizer: If False, only load model weights and reset
                optimizer/scheduler/scaler state from scratch. Use this when
                resuming from a checkpoint that was trained with buggy gradients.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded checkpoint weight success", flush=True)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded checkpoint weight success", flush=True)

        if resume_optimizer:
            try:
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                print(f"[load_checkpoint] Skip optimizer state due to mismatch: {e}", flush=True)

            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"[load_checkpoint] Skip scheduler state due to mismatch: {e}", flush=True)

            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except Exception as e:
                    print(f"[load_checkpoint] Skip scaler state due to mismatch: {e}", flush=True)
        else:
            print("[load_checkpoint] resume_optimizer=False: optimizer/scheduler/scaler state NOT loaded (fresh start)", flush=True)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from {checkpoint_path}", flush=True)
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}", flush=True)

    def train(self, train_loader, val_loader, num_epochs: int):
        """Main training loop"""
        if self.is_main_process:
            print(f"Starting training for {num_epochs} epochs", flush=True)
            print(f"Total steps per epoch: {len(train_loader)}", flush=True)
            if self.use_online_distillation:
                print(f"Online self-distillation enabled: "
                      f"iPTM threshold={self.distill_iptm_threshold}, "
                      f"keep_epochs={self.distill_keep_epochs}", flush=True)

        # Record start time
        start_time = time.time()

        # Save original training dataset (for later merging with distillation data)
        original_train_dataset = train_loader.dataset

        # # Run a validation pass before training starts
        # if self.is_main_process:
        #     print("Running initial validation before training...")
        # val_loss, val_loss_dict = self.evaluate(val_loader, max_batches=self.initial_val_max_batches)
        # if self.is_main_process:
        #     print(f"Initial Val Loss = {val_loss:.4f}")
        #     self._log_validation(val_loss_dict, self.epoch)
        #     if val_loss < self.best_val_loss:
        #         self.best_val_loss = val_loss
        #         self._save_checkpoint('best')
        #         print(f"New best model! Val Loss = {val_loss:.4f}")

        # Distillation handling after initial validation
        if self.use_online_distillation:
            self._finalize_distill_epoch()

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # === Online Distillation: rebuild DataLoader with distillation data ===
            if self.use_online_distillation:
                rebuilt = self._rebuild_train_loader_with_distill(original_train_dataset)
                if rebuilt is not None and rebuilt[0] is not None:
                    train_loader, new_sampler = rebuilt
                    if new_sampler is not None:
                        self.train_sampler = new_sampler

            # Train for one epoch
            train_loss = self.train_epoch(
                train_loader, epoch,
                train_sampler=getattr(self, 'train_sampler', None),
                val_loader=val_loader,
            )

            if self.is_main_process:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}", flush=True)

            # Validate periodically (only when step-based validation is not used, validate by epoch)
            if (self.eval_interval_steps is None or self.eval_interval_steps <= 0) and (
                    (epoch + 1) % self.eval_interval == 0 or epoch == num_epochs - 1
            ):
                val_loss, val_loss_dict = self.evaluate(val_loader)

                if self.is_main_process:
                    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}", flush=True)
                    self._log_validation(val_loss_dict, epoch + 1)

                    # Save best-performing model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint('best')
                        print(f"New best model! Val Loss = {val_loss:.4f}", flush=True)

                # === Online Distillation: aggregate distillation data, save checkpoint ===
                if self.use_online_distillation:
                    self._finalize_distill_epoch()
                    # Save checkpoint after collecting distillation data
                    if self.is_main_process:
                        self._save_checkpoint(f'distill_epoch_{epoch}')

            # Save checkpoints at epoch granularity
            if self.is_main_process:
                if (epoch + 1) % self.epoch_save_interval == 0 or epoch == num_epochs - 1:
                    self._save_checkpoint(f'epoch_{epoch}')

        if self.is_main_process:
            print("Training completed!", flush=True)
            if self.writer is not None:
                self.writer.close()

            # Calculate and output total runtime
            end_time = time.time()
            total_time = end_time - start_time
            total_minutes = total_time / 60
            print(f"Program end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}", flush=True)
            print(f"Total runtime: {total_time:.2f} seconds ({total_minutes:.2f} minutes)", flush=True)


# def setup_ddp():
#     """Configure distributed training via torch.distributed"""
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ['RANK'])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])
#     else:
#         rank = 0
#         world_size = 1
#         local_rank = 0

#     if world_size > 1:
#         # Set device BEFORE initializing process group to avoid warnings
#         torch.npu.set_device(local_rank)
#         dist.init_process_group(
#             backend='hccl',
#             init_method='env://',
#             world_size=world_size,
#             rank=rank
#         )
#         return True, local_rank

#     return False, 0
def setup_ddp():
    """Configure distributed training via torch.distributed"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        # Must set device before any NPU tensor creation / any HCCL call
        torch.npu.set_device(local_rank)

        # Explicitly tell HCCL which card this rank uses
        # Also set longer timeout to reduce false timeouts due to network jitter / checkpoint recomputation
        import datetime as _dt
        timeout_s = int(os.environ.get("TORCH_DIST_TIMEOUT_SEC", "1800"))
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=_dt.timedelta(seconds=timeout_s),
        )
        return True, local_rank

    return False, 0


def main_with_config(config: TrainingConfig):
    """Entry point for launching training with TrainingConfig object"""

    # Record start time
    start_time = time.time()

    # Check for conflicting options
    if config.train_only_confidence and config.no_train_confidence:
        raise ValueError("Cannot use both train_only_confidence and no_train_confidence at the same time")
    if config.train_only_confidence and config.train_only_diffusion:
        raise ValueError("Cannot use both train_only_confidence and train_only_diffusion at the same time")

    # Configure distributed execution
    use_ddp, local_rank = setup_ddp()

    # Determine global rank
    if use_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        print(f"Program start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", flush=True)

    # Select compute device
    device = torch.device(f'npu:{local_rank}' if torch.npu.is_available() else 'cpu')

    if rank == 0:
        print("=" * 80, flush=True)
        print("Training TrochFold", flush=True)
        print(f"Device: {device}", flush=True)
        print(f"Use DDP: {use_ddp}", flush=True)
        print(f"Output dir: {config.output_dir}", flush=True)
        print(f"Online Self-Distillation: {config.use_online_distillation}", flush=True)
        if config.use_online_distillation:
            print(f"  iPTM threshold: {config.distill_iptm_threshold}", flush=True)
            print(f"  Keep epochs: {config.distill_keep_epochs}", flush=True)
            print(f"  Val crop: {config.val_crop}", flush=True)
        if config.train_only_confidence:
            print("Mode: Training ONLY confidence head (all other parameters frozen)", flush=True)
        elif config.train_only_diffusion:
            print("Mode: Training ONLY diffusion head (all other parameters frozen)", flush=True)
        print("=" * 80, flush=True)

    noise_p_mean = float(config.noise_p_mean)
    noise_p_std = float(config.p_std)
    noise_p_mean_schedule = (
        [float(x) for x in config.p_mean_schedule]
        if config.p_mean_schedule
        else None
    )
    noise_p_mean_update_every = int(config.update_every)

    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        data_dir=config.data_dirs,
        ppi_data_dir=config.ppi_data_dirs,
        ppi_epoch_samples=config.ppi_epoch_samples,
        ppi_ratio=config.ppi_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_num_res=config.max_num_res,
        val_max_num_res=config.val_max_num_res,
        enable_cropping=config.enable_cropping,
        crop_size=config.crop_size,
        crop_complete_ligand_unstdRes=config.crop_complete_ligand_unstdRes,
        spatial_crop_complete_ligand_unstdRes=config.spatial_crop_complete_ligand_unstdRes,
        drop_last=config.drop_last,
        remove_metal=config.remove_metal,
        crop_method_weights=config.crop_method_weights,
        interface_minimal_distance=config.interface_minimal_distance,
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        use_length_grouped_sampler=config.use_length_grouped_sampler,
        enable_source_ratio_sampling=config.enable_source_ratio_sampling,
        data_source_sampling_ratios=config.data_source_sampling_ratios,
        max_templates=config.max_templates,
        remove_unresolved_tokens=config.remove_unresolved_tokens,
        val_enable_cropping=config.val_crop,
    )

    if rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}", flush=True)
        print(f"Val samples: {len(val_loader.dataset)}", flush=True)

    # Build gradient-checkpoint configuration
    checkpoint_config = {
        'pairformer': {
            'enabled': config.checkpoint_pairformer,
            'group_size': config.pairformer_group_size
        },
        'msa': {
            'enabled': config.checkpoint_msa,
            'group_size': config.msa_group_size
        },
        'template': {
            'enabled': config.checkpoint_template
        },
        'diffusion': {
            'enabled': config.checkpoint_diffusion
        },
        'confidence': {
            'enabled': config.checkpoint_confidence
        }
    }

    # Noise configs
    noise_configs = None
    if config.enable_per_source_noise:
        noise_configs = config.noise_configs
        if rank == 0 and noise_configs:
            print(f"Per-data-source noise enabled with configs: {noise_configs}", flush=True)

    # Instantiate the model
    model = TorchFold(
        num_recycles=config.num_recycles,
        randomize_num_recycles=config.randomize_num_recycles,
        num_samples=config.num_diffusion_samples,
        diffusion_steps=config.diffusion_steps,
        checkpoint_config=checkpoint_config,
        mini_rollout_steps=config.mini_rollout_steps,
        num_diffusion_samples_training=config.num_diffusion_samples_training,
        train_confidence=not config.no_train_confidence,
        save_diffusion_debug=config.save_val_structures,
        disable_internal_progress=config.disable_internal_progress,
        noise_configs=noise_configs,
        condition_embedding_drop_rate=config.condition_embedding_drop_rate,
        train_only_confidence=config.train_only_confidence,
        antibody_msa=config.antibody_msa,
        antibody_msa_drop_prob=config.antibody_msa_drop_prob,
        # Cross-chain Pair Representation Dropout
        cross_chain_pair_dropout=config.cross_chain_pair_dropout,
        cross_chain_pair_dropout_prob=config.cross_chain_pair_dropout_prob,
        cross_chain_pair_dropout_mode=config.cross_chain_pair_dropout_mode,
        cross_chain_pair_dropout_noise_scale=config.cross_chain_pair_dropout_noise_scale,
        cross_chain_pair_dropout_scale_factor=config.cross_chain_pair_dropout_scale_factor,
        noise_p_mean=noise_p_mean,
        noise_p_std=noise_p_std,
        noise_p_mean_schedule=noise_p_mean_schedule,
        noise_p_mean_update_every=(
            noise_p_mean_update_every if noise_p_mean_schedule else None
        ),
    )

    # Load pretrained weights
    if config.pretrained_model_dir:
        from torchfold.params import import_jax_weights_
        if rank == 0:
            print(f"Loading pretrained weights from {Path(config.pretrained_model_dir)}", flush=True)
        import_jax_weights_(model, Path(config.pretrained_model_dir))
        if use_ddp:
            dist.barrier()

    model = model.to(device)

    # Freeze/unfreeze parameters based on training mode (before DDP wrapping)
    # This must be done on all ranks to ensure consistent parameter states
    if config.train_only_confidence:
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only confidence head parameters
        for param in model.confidence_head.parameters():
            param.requires_grad = True

        # Count trainable parameters after freezing
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        confidence_params = sum(p.numel() for p in model.confidence_head.parameters())

        # Synchronize parameter states across all ranks before DDP wrapping
        if use_ddp:
            dist.barrier()

        # Print summary only on global rank 0
        if rank == 0:
            print("Parameter freezing summary:", flush=True)
            print(f"  Total parameters: {total_params:,}", flush=True)
            print(f"  Trainable before: {trainable_before:,}", flush=True)
            print(f"  Trainable after (confidence head only): {trainable_after:,}", flush=True)
            print(f"  Confidence head parameters: {confidence_params:,}", flush=True)
            print(f"  Frozen parameters: {total_params - trainable_after:,}", flush=True)

    elif config.train_only_diffusion:
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only diffusion head parameters
        for param in model.diffusion_head.parameters():
            param.requires_grad = True

        # Count trainable parameters after freezing
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diffusion_params = sum(p.numel() for p in model.diffusion_head.parameters())

        # Synchronize parameter states across all ranks before DDP wrapping
        if use_ddp:
            dist.barrier()

        # Print summary only on global rank 0
        if rank == 0:
            print("Parameter freezing summary:", flush=True)
            print(f"  Total parameters: {total_params:,}", flush=True)
            print(f"  Trainable before: {trainable_before:,}", flush=True)
            print(f"  Trainable after (diffusion head only): {trainable_after:,}", flush=True)
            print(f"  Diffusion head parameters: {diffusion_params:,}", flush=True)
            print(f"  Frozen parameters: {total_params - trainable_after:,}", flush=True)

    elif config.no_train_confidence:
        confidence_frozen = 0
        for param in model.confidence_head.parameters():
            if param.requires_grad:
                param.requires_grad = False
                confidence_frozen += param.numel()
        if rank == 0 and confidence_frozen > 0:
            print(f"Frozen confidence_head (no_train_confidence=True): {confidence_frozen:,} params", flush=True)

    # === Custom module freezing (from YAML freeze_modules list) ===
    if config.freeze_modules and not config.train_only_confidence and not config.train_only_diffusion:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_modules_info = []
        for module_name in config.freeze_modules:
            # Support nested module names like "evoformer.msa_module"
            parts = module_name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None:
                frozen_count = 0
                for param in module.parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_count += param.numel()
                frozen_modules_info.append((module_name, frozen_count))
            else:
                if rank == 0:
                    print(f"Warning: Module '{module_name}' not found in model, skipping freeze.", flush=True)

        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            print("Custom module freezing summary:", flush=True)
            print(f"  Total parameters: {total_params:,}", flush=True)
            print(f"  Trainable before: {trainable_before:,}", flush=True)
            print(f"  Trainable after: {trainable_after:,}", flush=True)
            for mname, mcount in frozen_modules_info:
                print(f"  Frozen '{mname}': {mcount:,} params", flush=True)

    # Wrap model with DDP
    if use_ddp:
        # When randomize_num_recycles is used, ddp_static_anchor forces all parameters to participate in backward, so find_unused is not needed
        # model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused, gradient_as_bucket_view=True)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # static_graph must be enabled for reentrant checkpoint, otherwise DDP will report "marked twice".
        # When randomize_num_recycles is used, ddp_static_anchor forces all parameters to participate in backward, making the graph consistent.
        # model._set_static_graph()
        # maybe_print_ddp_param_index_mapping(model, rank)

    # Configure loss (with interface loss weighting)
    from loss_function.train_loss import configure_loss_fn
    configure_loss_fn(
        use_confidence=not config.no_train_confidence,
        use_interface_loss_weight=config.use_interface_loss_weight,
        interface_loss_weight=config.interface_loss_weight,
        interface_distance_threshold=config.interface_distance_threshold,
        alpha_diffusion=config.fape_weight,
        alpha_distogram=config.distogram_weight,
        alpha_confidence=config.confidence_weight,
    )
    if rank == 0:
        print(f"Loss weights: alpha_diffusion={config.fape_weight}, "
              f"alpha_distogram={config.distogram_weight}, "
              f"alpha_confidence={config.confidence_weight}", flush=True)
    if rank == 0 and config.use_interface_loss_weight:
        print(f"Interface loss weighting enabled: weight={config.interface_loss_weight}, "
              f"distance_threshold={config.interface_distance_threshold}Å", flush=True)
    if rank == 0 and config.cross_chain_pair_dropout:
        print(f"Cross-chain pair dropout enabled: prob={config.cross_chain_pair_dropout_prob}, "
              f"mode={config.cross_chain_pair_dropout_mode}", flush=True)

    # Create optimizer (with optional per-module learning rates)
    model_for_params = model.module if use_ddp else model

    if config.train_only_confidence or config.train_only_diffusion:
        trainable_params = [p for p in model_for_params.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay,
            foreach=True,
        )
    elif config.layer_lr:
        # === Layered learning rates: per-module LR ===
        param_groups = []
        assigned_param_ids = set()

        for module_name, module_lr in config.layer_lr.items():
            # Support nested module names like "evoformer.msa_module"
            parts = module_name.split('.')
            module = model_for_params
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None:
                params = [p for p in module.parameters() if p.requires_grad and id(p) not in assigned_param_ids]
                for p in params:
                    assigned_param_ids.add(id(p))
                if params:
                    param_groups.append({
                        'params': params,
                        'lr': module_lr,
                        'name': module_name,  # for logging
                    })
            else:
                if rank == 0:
                    print(f"Warning: Module '{module_name}' not found for layer_lr, skipping.", flush=True)

        # Default group for remaining trainable params
        remaining_params = [
            p for p in model_for_params.parameters()
            if p.requires_grad and id(p) not in assigned_param_ids
        ]
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': config.learning_rate,
                'name': 'default',
            })

        optimizer = optim.Adam(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay,
            foreach=True,
        )

        if rank == 0:
            print("Layered learning rate configuration:", flush=True)
            for pg in param_groups:
                n_params = sum(p.numel() for p in pg['params'])
                print(f"  {pg.get('name', '?')}: lr={pg['lr']:.2e}, params={n_params:,}", flush=True)
    else:
        optimizer = optim.Adam(
            model_for_params.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay,
            foreach=True,
        )

    # Create scheduler
    scheduler = None
    if config.use_scheduler:
        def lr_lambda(step):
            warmup_steps = 1000
            decay_steps = 5000
            decay_rate = 0.95
            if step < warmup_steps:
                # Linear warmup from 0 to 1
                return step / warmup_steps
            else:
                # Exponential decay: apply decay_rate every decay_steps
                decay_count = (step - warmup_steps) // decay_steps
                return decay_rate ** decay_count

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Build trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=config.output_dir,
        dot_product_attention=config.dot_product_attention,
        data_dir=config.data_dirs,
        gradient_clip=config.gradient_clip,
        accumulation_steps=config.accumulation_steps,
        use_amp=config.use_amp,
        amp_dtype=config.amp_dtype,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        eval_interval=config.eval_interval,
        eval_interval_steps=config.eval_interval_steps,
        epoch_save_interval=config.epoch_save_interval,
        save_structures_interval=config.save_structures_interval,
        max_batches_to_save=config.max_batches_to_save,
        initial_val_max_batches=config.initial_val_max_batches,
        save_diffusion_debug=config.save_val_structures,
        max_diffusion_steps_to_save=config.max_diffusion_steps_to_save,
        save_val_structures_epoch_interval=config.save_val_structures_epoch_interval,
        use_ddp=use_ddp,
        local_rank=local_rank,
        global_rank=rank,
        catch_exceptions=config.enable_exception_handling,
        # Online Self-Distillation
        use_online_distillation=config.use_online_distillation,
        distill_iptm_threshold=config.distill_iptm_threshold,
        distill_keep_epochs=config.distill_keep_epochs,
        config=config,
    )

    trainer.train_sampler = train_sampler
    trainer.val_sampler = val_sampler

    if config.resume_from:
        trainer.load_checkpoint(config.resume_from, resume_optimizer=config.resume_optimizer)

    trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)

    if use_ddp:
        dist.destroy_process_group()

    if rank == 0:
        end_time = time.time()
        total_time = end_time - start_time
        total_minutes = total_time / 60
        total_hours = total_time / 3600
        print(f"\nProgram end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}", flush=True)
        print(f"Total runtime: {total_time:.2f} seconds ({total_minutes:.2f} minutes, {total_hours:.2f} hours)", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TorchFold')

    # YAML configuration file (recommended)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (recommended)')

    # The following command-line arguments are kept for backward compatibility (will be overridden by YAML config)
    # Data options
    parser.add_argument('--data_dir', type=str, action='append', default=None,
                        help='Directory containing preprocessed training data (can be specified multiple times)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to store logs and checkpoints')
    parser.add_argument('--max_num_res', type=int, default=760,
                        help='Maximum residue count filter (train placeholder)')
    parser.add_argument('--val_max_num_res', type=int, default=760,
                        help='Maximum residue count filter for validation')

    # Cropping options
    parser.add_argument('--enable_cropping', action='store_true', default=False,
                        help='Enable contiguous cropping')
    parser.add_argument('--crop_size', type=int, default=50,
                        help='Token count after cropping')
    parser.add_argument('--crop_complete_ligand_unstdRes', action='store_true', default=False,
                        help='Keep ligands/unstandard residues intact during contiguous cropping')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Drop the final partial ligand/unstandard residue')
    parser.add_argument('--remove_metal', action='store_true', default=False,
                        help='Remove metal/ion entries')
    parser.add_argument('--spatial_crop_complete_ligand_unstdRes', action='store_true', default=False,
                        help='Keep ligands/unstandard residues intact during spatial cropping')
    parser.add_argument('--crop_method_weights', type=float, nargs=3, default=[0.34, 0.33, 0.33],
                        help='Sampling probabilities for contiguous/spatial/spatial-interface cropping')
    parser.add_argument('--interface_minimal_distance', type=int, default=15,
                        help='Distance threshold for interface spatial cropping')
    parser.add_argument('--use_length_grouped_sampler', action='store_true', default=False,
                        help='Use length-grouped sampler for DDP training')
    parser.add_argument('--max_templates', type=int, default=None,
                        help='Maximum number of templates')
    parser.add_argument('--remove_unresolved_tokens', action='store_true', default=False,
                        help='Remove unresolved tokens before cropping')

    # Noise options
    parser.add_argument('--enable_per_source_noise', action='store_true', default=False,
                        help='Enable per-data-source noise distribution')
    parser.add_argument('--noise_configs', type=str, default=None,
                        help='JSON string for per-data-source noise configs')

    # Model parameters
    parser.add_argument('--num_recycles', type=int, default=3,
                        help='Number of Evoformer recycles')
    parser.add_argument('--randomize_num_recycles', action='store_true', default=False,
                        help='Sample recycle count uniformly from [0, num_recycles] during training')
    parser.add_argument('--num_diffusion_samples', type=int, default=1,
                        help='Number of diffusion samples')
    parser.add_argument('--diffusion_steps', type=int, default=200,
                        help='Number of diffusion steps')
    parser.add_argument('--pretrained_model_dir', type=str, default=None,
                        help='Optional JAX parameter dump directory (unused when --resume_from is set)')
    parser.add_argument('--mini_rollout_steps', type=int, default=0,
                        help='Mini-rollout steps for confidence head')
    parser.add_argument('--antibody_msa', action=argparse.BooleanOptionalAction, default=True,
                        help='Provide MSA for antibody chains')
    parser.add_argument('--antibody_msa_drop_prob', type=float, default=0.5,
                        help='Probability to drop antibody MSA')

    # Checkpoint settings
    parser.add_argument('--checkpoint_pairformer', action='store_true', default=False,
                        help='Enable Pairformer gradient checkpointing')
    parser.add_argument('--pairformer_group_size', type=int, default=8,
                        help='Group size for Pairformer checkpointing')
    parser.add_argument('--checkpoint_msa', action='store_true', default=False,
                        help='Enable MSA gradient checkpointing')
    parser.add_argument('--msa_group_size', type=int, default=2,
                        help='Group size for MSA checkpointing')
    parser.add_argument('--checkpoint_template', action='store_true', default=False,
                        help='Enable Template checkpointing')
    parser.add_argument('--checkpoint_diffusion', action='store_true', default=False,
                        help='Enable diffusion checkpointing')
    parser.add_argument('--checkpoint_confidence', action='store_true', default=False,
                        help='Enable confidence checkpointing')
    parser.add_argument('--disable-internal-progress', action='store_true', default=False,
                        help='Disable internal progress bars')
    parser.add_argument('--no-train-confidence', action='store_true', default=False,
                        help='Disable training of confidence head')
    parser.add_argument('--train-only-confidence', action='store_true', default=False,
                        help='Only train confidence head')
    parser.add_argument('--train-only-diffusion', action='store_true', default=False,
                        help='Only train diffusion head')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Lower bound for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_diffusion_samples_training', type=int, default=48,
                        help='Number of diffusion samples during training')
    parser.add_argument('--condition_embedding_drop_rate', type=float, default=0.1,
                        help='Conditioning dropout rate')
    parser.add_argument('--dot_product_attention', type=str, default='Fusion_Attention',
                        help='Choose the implementation of dot product attention, options: ["torch", "Fusion_Attention"]')

    # Loss weights
    parser.add_argument('--fape_weight', type=float, default=1.0,
                        help='FAPE loss weight')
    parser.add_argument('--distogram_weight', type=float, default=0.3,
                        help='Distogram loss weight')
    parser.add_argument('--confidence_weight', type=float, default=0.01,
                        help='Confidence loss weight')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Enable mixed-precision training')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['fp16', 'bf16'],
                        help='Precision type for AMP')
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                        help='Enable learning-rate scheduler')
    parser.add_argument('--scheduler_t0', type=int, default=1000,
                        help='Scheduler cycle length')

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval in steps')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Checkpoint save interval in steps')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluation interval in epochs')
    parser.add_argument('--epoch_save_interval', type=int, default=200,
                        help='Epoch checkpoint save interval')
    parser.add_argument('--max_batches_to_save', type=int, default=None,
                        help='Max batches to save per validation')
    parser.add_argument('--save_structures_interval', type=int, default=0,
                        help='Save structures every N validations')
    parser.add_argument('--save_val_structures', action='store_true', default=False,
                        help='Save validation structures')
    parser.add_argument('--save_val_structures_epoch_interval', type=int, default=1,
                        help='Save val structures every N epochs')
    parser.add_argument('--max_diffusion_steps_to_save', type=int, default=None,
                        help='Max diffusion steps to save')
    parser.add_argument('--enable_exception_handling', action='store_true', default=False,
                        help='Enable exception handling')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data-loading workers')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Prefer loading from YAML configuration file
    if args.config:
        print(f'Loading configuration from: {args.config}', flush=True)
        config = load_config_from_yaml(args.config)
    else:
        # Build configuration from command-line arguments
        if args.data_dir is None or args.output_dir is None:
            raise ValueError("Either --config or both --data_dir and --output_dir are required")

        config = TrainingConfig(
            data_dirs=args.data_dir,
            output_dir=args.output_dir,
            pretrained_model_dir=args.pretrained_model_dir,
            resume_from=args.resume_from,
            max_num_res=args.max_num_res,
            val_max_num_res=args.val_max_num_res,
            enable_cropping=args.enable_cropping,
            crop_size=args.crop_size,
            crop_complete_ligand_unstdRes=args.crop_complete_ligand_unstdRes,
            spatial_crop_complete_ligand_unstdRes=args.spatial_crop_complete_ligand_unstdRes,
            drop_last=args.drop_last,
            remove_metal=args.remove_metal,
            crop_method_weights=args.crop_method_weights,
            interface_minimal_distance=args.interface_minimal_distance,
            max_templates=args.max_templates,
            remove_unresolved_tokens=args.remove_unresolved_tokens,
            num_recycles=args.num_recycles,
            randomize_num_recycles=args.randomize_num_recycles,
            num_diffusion_samples=args.num_diffusion_samples,
            diffusion_steps=args.diffusion_steps,
            mini_rollout_steps=args.mini_rollout_steps,
            num_diffusion_samples_training=args.num_diffusion_samples_training,
            condition_embedding_drop_rate=args.condition_embedding_drop_rate,
            dot_product_attention=args.dot_product_attention,
            antibody_msa=args.antibody_msa,
            antibody_msa_drop_prob=args.antibody_msa_drop_prob,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            accumulation_steps=args.accumulation_steps,
            train_only_confidence=getattr(args, 'train_only_confidence', False),
            train_only_diffusion=getattr(args, 'train_only_diffusion', False),
            no_train_confidence=getattr(args, 'no_train_confidence', False),
            use_scheduler=args.use_scheduler,
            scheduler_t0=args.scheduler_t0,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            enable_per_source_noise=args.enable_per_source_noise,
            checkpoint_pairformer=args.checkpoint_pairformer,
            checkpoint_msa=args.checkpoint_msa,
            msa_group_size=args.msa_group_size,
            checkpoint_template=args.checkpoint_template,
            checkpoint_diffusion=args.checkpoint_diffusion,
            checkpoint_confidence=args.checkpoint_confidence,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            epoch_save_interval=args.epoch_save_interval,
            save_val_structures=args.save_val_structures,
            save_val_structures_epoch_interval=args.save_val_structures_epoch_interval,
            max_diffusion_steps_to_save=args.max_diffusion_steps_to_save,
            max_batches_to_save=args.max_batches_to_save,
            save_structures_interval=args.save_structures_interval,
            use_length_grouped_sampler=args.use_length_grouped_sampler,
            num_workers=args.num_workers,
            fape_weight=args.fape_weight,
            distogram_weight=args.distogram_weight,
            confidence_weight=args.confidence_weight,
            seed=args.seed,
            disable_internal_progress=getattr(args, 'disable_internal_progress', False),
            enable_exception_handling=args.enable_exception_handling,
        )

        # Parse noise configuration
        if args.noise_configs:
            try:
                noise_configs_raw = json.loads(args.noise_configs)
                config.noise_configs = {int(k): v for k, v in noise_configs_raw.items()}
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --noise_configs: {e}")

    print('Configuration loaded:', flush=True)
    print(f'  data_dirs: {config.data_dirs}', flush=True)
    print(f'  output_dir: {config.output_dir}', flush=True)

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    print(f'  seed: {config.seed}', flush=True)

    # Call main function
    main_with_config(config)
