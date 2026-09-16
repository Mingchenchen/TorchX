"""torchfold.runner.train — full Trainer (GPU + DDP) for the torchfold AF3 model.

"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from typing import Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# GT dense coords for the training forward (true_positions / *_atom_mask).
# ---------------------------------------------------------------------------
def _build_gt_dense(
    batch: dict,
    af3_feats: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter flat GT coords into AF3 dense [N_token, max_dense, 3].

    Uses the SAME (atom_to_token_idx, atom_to_tokatom_idx) maps the input
    adapter uses to scatter ref_pos, so true_positions lines up slot-for-slot
    with pred_dense_atom_mask / ref_pos.
    """
    feats = batch["input_feature_dict"]
    label = batch["label_dict"]

    a2t = feats["atom_to_token_idx"].to(torch.int64).to(device)      # [N_atom]
    a2ta = feats["atom_to_tokatom_idx"].to(torch.int64).to(device)   # [N_atom]

    # dense grid shape comes straight from the adapter output (already built).
    n_token, max_dense = af3_feats["pred_dense_atom_mask"].shape

    coord = label["coordinate"].to(torch.float32).to(device)         # [N_atom, 3]
    cmask = label["coordinate_mask"].to(torch.float32).to(device)    # [N_atom]

    true_positions = coord.new_zeros((n_token, max_dense, 3))
    true_positions[a2t, a2ta] = coord

    true_mask = cmask.new_zeros((n_token, max_dense))
    true_mask[a2t, a2ta] = cmask

    return true_positions, true_mask.to(torch.bool)


def _dense_to_flat(
    dense: torch.Tensor,
    a2t: torch.Tensor,
    a2ta: torch.Tensor,
) -> torch.Tensor:
    """Gather AF3 dense coords [..., N_token, max_dense, 3] -> flat [..., N_atom, 3].

    Inverse of the scatter used by ``_build_gt_dense`` / the adapter ref_pos.
    """
    return dense[..., a2t, a2ta, :]


class Trainer:
    """AF3 trainer (single-GPU + DDP): scheduler + AMP bf16 + grad-accum + EMA + ckpt + eval."""

    def __init__(
        self,
        af3_params_dir,
        *,
        model_name: str = "torchfold_af3_default",
        num_recycles: int = 4,
        randomize_num_recycles: bool = False,
        pair_dropout: float = 0.0,
        msa_dropout: float = 0.0,
        train_mode: str = "full",
        grad_checkpoint: bool = False,
        num_diffusion_samples_training: int = 48,
        mini_rollout_steps: int = 20,
        diffusion_sparse_loss: bool = True,
        diffusion_lddt_loss_dense: bool = True,
        loss_alpha_diffusion: float = 4.0,
        loss_alpha_bond: float = 1.0,
        loss_smooth_lddt: float = 1.0,
        loss_alpha_distogram: float = 3e-2,
        loss_alpha_confidence: float = 1e-4,
        loss_alpha_pae: float = 1.0,
        loss_smooth_lddt_interface_weight: float = 0.0,
        loss_smooth_lddt_interface_distance_threshold: float = 5.0,
        loss_interface_mse_weight: float = 0.0,
        loss_interface_mse_distance_threshold: float = 10.0,
        loss_interface_mse_mode: str = "ca",
        train_crop_size: int = 64,
        config_overrides=None,
        num_dl_workers: int = 2,
        epoch_size: int = 16,
        lr: float = 1.8e-3,
        grad_clip: float = 10.0,
        seed: int = 42,
        load_jax_params: bool = True,
        # -- new knobs --
        accumulation_steps: int = 1,
        amp_dtype: str = "bf16",            # "bf16" | "fp32"
        warmup_steps: int = 1000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
        lr_scheduler_name: str = "af3",
        ema_decay: float = 0.999,
        eval_diffusion_steps: int = 200,
        eval_num_samples: int = 5,
        sample_diffusion_chunk_size: int = 5,
        eval_ema_only: bool = False,
        log_interval: int = 50,
        test_max_n_token: int = -1,
        run_dir: Optional[str] = None,
        wandb_dir: Optional[str] = None,
        wandb_project: str = "torchfold",
        device: Optional[torch.device] = None,
        # -- DDP knobs --
        ddp_find_unused_parameters: bool = True,
        # -- training noise curriculum --
        noise_p_mean: float = -1.2,
        noise_p_std: float = 1.5,
        noise_p_mean_schedule: Optional[list] = None,
        noise_p_mean_update_every: int = 10000,
    ):
        # import_jax_weights_ does `model_path / "af3.bin"`, so this must be a
        # pathlib.Path. Coerce here so str args (CLI default / scripts) also work;
        # the single-GPU test already passes a Path, which is unchanged.
        import pathlib as _pathlib

        self.af3_params_dir = (
            af3_params_dir
            if isinstance(af3_params_dir, _pathlib.Path)
            else _pathlib.Path(af3_params_dir)
        )
        self.num_recycles = num_recycles
        self.model_name = model_name
        self.randomize_num_recycles = bool(randomize_num_recycles)
        self.pair_dropout = float(pair_dropout)
        self.msa_dropout = float(msa_dropout)
        self.train_mode = str(train_mode)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.num_diffusion_samples_training = num_diffusion_samples_training
        self.mini_rollout_steps = mini_rollout_steps
        self.diffusion_sparse_loss = bool(diffusion_sparse_loss)
        self.diffusion_lddt_loss_dense = bool(diffusion_lddt_loss_dense)
        self.loss_alpha_diffusion = float(loss_alpha_diffusion)
        self.loss_alpha_bond = float(loss_alpha_bond)
        self.loss_smooth_lddt = float(loss_smooth_lddt)
        self.loss_alpha_distogram = float(loss_alpha_distogram)
        self.loss_alpha_confidence = float(loss_alpha_confidence)
        self.loss_alpha_pae = float(loss_alpha_pae)
        self.loss_smooth_lddt_interface_weight = float(loss_smooth_lddt_interface_weight)
        self.loss_smooth_lddt_interface_distance_threshold = float(loss_smooth_lddt_interface_distance_threshold)
        self.loss_interface_mse_weight = float(loss_interface_mse_weight)
        self.loss_interface_mse_distance_threshold = float(loss_interface_mse_distance_threshold)
        self.loss_interface_mse_mode = str(loss_interface_mse_mode)
        self.train_crop_size = train_crop_size
        self.config_overrides = list(config_overrides) if config_overrides else []
        self.num_dl_workers = num_dl_workers
        self.epoch_size = epoch_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.seed = seed
        self.load_jax_params = bool(load_jax_params)

        self.accumulation_steps = max(1, int(accumulation_steps))
        self.amp_dtype = amp_dtype
        self.warmup_steps = warmup_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor
        self.lr_scheduler_name = lr_scheduler_name
        self.ema_decay = ema_decay
        self.eval_diffusion_steps = eval_diffusion_steps
        self.eval_num_samples = eval_num_samples
        self.sample_diffusion_chunk_size = sample_diffusion_chunk_size
        self.eval_ema_only = eval_ema_only
        self.log_interval = log_interval
        self.test_max_n_token = test_max_n_token

        self.noise_p_mean = float(noise_p_mean)
        self.noise_p_std = float(noise_p_std)
        self.noise_p_mean_schedule = (
            [float(x) for x in noise_p_mean_schedule]
            if noise_p_mean_schedule
            else None
        )
        self.noise_p_mean_update_every = int(noise_p_mean_update_every)

        self.run_dir = run_dir
        self.wandb_dir = wandb_dir
        self.wandb_project = wandb_project

        # -- DDP / distributed topology (resolved in init_env) --
        self.ddp_find_unused_parameters = bool(ddp_find_unused_parameters)
        self.is_distributed = False          # WORLD_SIZE > 1 ?
        self.rank = 0                        # global rank
        self.world_size = 1                  # number of processes
        self.local_rank = 0                  # local (per-node) rank == cuda index
        # base_seed is the rank-INDEPENDENT seed handed to the data sampler so the
        # weighted partition is consistent; rank_seed is the per-rank seed used to
        # seed torch/numpy so noise differs across ranks (set in init_env).
        self.base_seed = seed
        self.rank_seed = seed

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # self.model       -> the *unwrapped* AF3 module (EMA / ckpt / metrics)
        # self.model_ddp   -> the forward/backward handle (== model in single-GPU,
        #                     == DDP(model) when distributed)
        self.model = None
        self.model_ddp = None
        self.loss_fn = None
        self.train_dl = None
        self.val_dl = None   # kept for backward compat; points to first test_dl
        self.test_dls = {}   # {name: dl} — all configured test dataloaders
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.wandb_run = None

        self._train_metric_acc: dict[str, list[float]] = {}

        # metrics (lazy-built in init_metrics)
        self._lddt_metric = None
        self._clash_metric = None

        # effective optimizer-step counter (one per accumulation window).
        self.step = 0
        # micro-batch counter (for accumulation boundaries).
        self.micro_step = 0
        # epoch counter (incremented in train() each time the dataloader wraps).
        self.epoch = 0
        # best_val_loss is kept static at 0.0 for checkpoint format compatibility.
        # Best step is chosen manually from logged wandb metric curves
        # DO NOT update this field; there is no auto best-selection.
        self.best_val_loss = 0.0

    # -- distributed helpers ------------------------------------------------
    @property
    def is_main(self) -> bool:
        """True on rank 0 (or in single-process mode). Gate side effects on this."""
        return self.rank == 0

    def init_env(self):
        """Detect torchrun env, init NCCL process group, pin device + rank seed.

        Single-process path (no RANK/WORLD_SIZE in env, or WORLD_SIZE==1) leaves
        the trainer in its single-GPU configuration: rank 0, world_size 1, no
        process group. This keeps the existing single-GPU tests working unchanged.
        """
        env = os.environ
        has_env = "RANK" in env and "WORLD_SIZE" in env
        world_size = int(env.get("WORLD_SIZE", "1"))

        if has_env and world_size > 1:
            self.rank = int(env["RANK"])
            self.world_size = world_size
            self.local_rank = int(env.get("LOCAL_RANK", env["RANK"]))
            self.is_distributed = True

            # set device BEFORE any NCCL call / CUDA tensor allocation.
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")

            if not dist.is_initialized():
                import datetime as _dt

                timeout_s = int(env.get("TORCH_DIST_TIMEOUT_SEC", "1800"))
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=self.world_size,
                    rank=self.rank,
                    device_id=torch.device(f"cuda:{self.local_rank}"),
                    timeout=_dt.timedelta(seconds=timeout_s),
                )
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_distributed = False

        # rank-aware seed: different ranks see different
        # sampler draws / diffusion noise even though base_seed partitions data.
        self.base_seed = self.seed
        self.rank_seed = (
            int(hashlib.sha256(f"{self.seed}_{self.rank}".encode()).hexdigest(), 16)
            % (2 ** 31)
        )
        torch.manual_seed(self.rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.rank_seed)
        try:
            import numpy as _np

            _np.random.seed(self.rank_seed)
        except Exception:
            pass
        return self.is_distributed

    def barrier(self):
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def shutdown(self):
        """Tear down the NCCL process group (safe to call when not distributed)."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _log_main(self, msg: str):
        if self.is_main:
            print(msg, flush=True)

    # -- autocast context ---------------------------------------------------
    def _autocast(self):
        if self.amp_dtype == "bf16" and self.device.type == "cuda":
            return torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
            )
        return contextlib.nullcontext()

    # -- builders -----------------------------------------------------------
    def init_model(self):
        from torchfold.model.build import build_model
        from torchfold.configs.configs_model_type import model_configs

        # Gradient (activation) checkpointing: AF3 uses it to fit large
        # crops (256/384/640+). group_size=1 = checkpoint every block (max memory
        # savings, more recompute). Only active in train()+grad mode (the model
        # guards each site with `self.training and torch.is_grad_enabled()`), so
        # eval/inference is unaffected.
        ckpt_cfg = None
        if self.grad_checkpoint:
            ckpt_cfg = {
                "pairformer": {"enabled": True, "group_size": 1},
                "msa": {"enabled": True, "group_size": 1},
                "template": {"enabled": True},
                # diffusion group_size = N_sample chunk for the 48-sample diffusion
                # training pass. chunk=1 under-batches:
                # 48 sequential checkpointed diffusion-head passes => ~4x slower fwd+bwd.
                "diffusion": {"enabled": True, "group_size": 4},
                "confidence": {"enabled": True},
            }

        runtime_kwargs = dict(
            num_recycles=self.num_recycles,
            num_samples=self.eval_num_samples,   # inference sampler (eval)
            diffusion_steps=self.eval_diffusion_steps,  # inference sampler (eval)
            sample_diffusion_chunk_size=self.sample_diffusion_chunk_size,
            mini_rollout_steps=self.mini_rollout_steps,
            num_diffusion_samples_training=self.num_diffusion_samples_training,
            train_mode=self.train_mode,
            randomize_num_recycles=self.randomize_num_recycles,
            pair_dropout=self.pair_dropout,
            msa_dropout=self.msa_dropout,
            checkpoint_config=ckpt_cfg,
            save_diffusion_debug=False,
            disable_internal_progress=True,
            noise_p_mean=self.noise_p_mean,
            noise_p_std=self.noise_p_std,
            noise_p_mean_schedule=self.noise_p_mean_schedule,
            noise_p_mean_update_every=(
                self.noise_p_mean_update_every if self.noise_p_mean_schedule else None
            ),
        )
        # model selected by --model_name (registry in configs/configs_model_type.py).
        # model = build_model(
        #     self.model_name, model_configs, runtime_kwargs=runtime_kwargs,
        #     af3_params_dir=self.af3_params_dir, log=self._log_main,
        # )
        model = build_model(
            self.model_name, model_configs, runtime_kwargs=runtime_kwargs,
            af3_params_dir=self.af3_params_dir, load_params=self.load_jax_params,
            log=self._log_main,
        )
        model = model.to(self.device)
        model.train()
        # self.model is ALWAYS the unwrapped AF3 module (EMA / ckpt / metrics use it).
        self.model = model

        # DDP wrap (only when distributed). The wrapped handle is used only for
        # the training forward/backward; EMA / checkpoint / eval keep using the
        # raw module so param names carry no "module." prefix.
        if self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model_ddp = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.ddp_find_unused_parameters,
            )
        else:
            self.model_ddp = model
        sampler = model.train_noise_sampler
        if getattr(sampler, "p_mean_schedule", None):
            self._log_main(
                "[noise] p_mean curriculum schedule=%s update_every=%s (start p_mean=%s)"
                % (sampler.p_mean_schedule, sampler.update_every, sampler.p_mean)
            )
        else:
            self._log_main(
                "[noise] fixed p_mean=%s p_std=%s"
                % (sampler.p_mean, sampler.p_std)
            )
        self._noise_p_mean_logged = sampler.p_mean
        return model

    def _apply_noise_curriculum(self) -> float:
        """Sync TrainingNoiseSampler.p_mean to the current optimizer step."""
        sampler = self.model.train_noise_sampler
        p_mean = sampler.set_step(self.step)
        if p_mean != getattr(self, "_noise_p_mean_logged", None):
            self._log_main(
                "[noise] step=%d -> p_mean=%.4g (schedule=%s every=%s)"
                % (
                    self.step,
                    p_mean,
                    getattr(sampler, "p_mean_schedule", None),
                    getattr(sampler, "update_every", None),
                )
            )
            self._noise_p_mean_logged = p_mean
        return p_mean

    def init_loss(self):
        from torchfold.model.loss import Loss
        from types import SimpleNamespace

        # sparse-loss switches threaded into the Loss config (Loss reads
        # them via getattr; all other loss hyperparams keep their AF3 defaults).
        loss_cfg = SimpleNamespace(
            diffusion_sparse_loss_enable=self.diffusion_sparse_loss,
            diffusion_lddt_loss_dense=self.diffusion_lddt_loss_dense,
            # configurable loss weights
            alpha_diffusion=self.loss_alpha_diffusion,
            alpha_bond=self.loss_alpha_bond,
            weight_smooth_lddt=self.loss_smooth_lddt,
            alpha_distogram=self.loss_alpha_distogram,
            alpha_confidence=self.loss_alpha_confidence,
            alpha_pae=self.loss_alpha_pae,
            weight_smooth_lddt_interface=self.loss_smooth_lddt_interface_weight,
            interface_radius=self.loss_smooth_lddt_interface_distance_threshold,
            interface_mse_weight=self.loss_interface_mse_weight,
            interface_mse_distance_threshold=self.loss_interface_mse_distance_threshold,
            interface_mse_mode=self.loss_interface_mse_mode,
        )
        self.loss_fn = Loss(loss_cfg).to(self.device)
        self._log_main(
            "[loss] sparse=%s lddt_dense=%s | diffusion=%s bond=%s smooth_lddt=%s "
            "distogram=%s confidence=%s pae=%s"
            % (self.diffusion_sparse_loss, self.diffusion_lddt_loss_dense,
               self.loss_alpha_diffusion, self.loss_alpha_bond, self.loss_smooth_lddt,
               self.loss_alpha_distogram, self.loss_alpha_confidence, self.loss_alpha_pae)
        )
        return self.loss_fn

    def _build_configs(self):
        from torchfold.configs.configs_data import data_configs
        from torchfold.config import parse_configs

        minimal_base = {
            "seed": self.seed, "train_crop_size": self.train_crop_size,
            "test_max_n_token": self.test_max_n_token,
            "train_lig_atom_rename": False, "train_shuffle_mols": False,
            "train_shuffle_sym_ids": False, "test_lig_atom_rename": False,
            "test_shuffle_mols": False, "test_shuffle_sym_ids": False,
            "project": "x", "run_name": "x", "base_dir": "/tmp",
            "eval_interval": 1, "log_interval": 1, "max_steps": 1,
            "data": data_configs,
        }
        arg_str = " ".join(self.config_overrides) if self.config_overrides else None
        configs = parse_configs(minimal_base, arg_str=arg_str, fill_required_with_null=True)
        if arg_str:
            self._log_main("[config] CLI dotted overrides applied: %s" % arg_str)
        configs.data.num_dl_workers = self.num_dl_workers
        configs.data.epoch_size = self.epoch_size
        return configs

    def init_data(self):
        configs = self._build_configs()
        from torchfold.data.pipeline.dataloader import get_dataloaders

        train_dl, test_dls = get_dataloaders(
            configs, world_size=self.world_size, seed=self.base_seed
        )
        self.train_dl = train_dl
        # Store all test dataloaders as a dict.
        if isinstance(test_dls, dict):
            self.test_dls = test_dls
        elif test_dls is not None:
            self.test_dls = {"default": test_dls}
        else:
            self.test_dls = {}
        # val_dl: keep for backward compat; points to first test loader.
        if self.test_dls:
            self.val_dl = next(iter(self.test_dls.values()))
        else:
            self.val_dl = train_dl
        return train_dl

    def init_optim(self):
        assert self.model is not None, "call init_model() first"
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8,
        )
        return self.optimizer

    def init_scheduler(self, last_epoch: int = -1):
        assert self.optimizer is not None, "call init_optim() first"
        from torchfold.utils.lr_scheduler import get_lr_scheduler

        self.scheduler = get_lr_scheduler(
            self.lr_scheduler_name,
            self.optimizer,
            lr=self.lr,
            warmup_steps=self.warmup_steps,
            decay_every_n_steps=self.decay_every_n_steps,
            decay_factor=self.decay_factor,
            last_epoch=last_epoch,
        )
        return self.scheduler

    def init_ema(self):
        assert self.model is not None, "call init_model() first"
        from torchfold.runner.ema import EMAWrapper

        self.ema = EMAWrapper(self.model, decay=self.ema_decay)
        self.ema.register()
        return self.ema

    def init_metrics(self):
        from torchfold.metrics.lddt_metrics import LDDT
        from torchfold.metrics.clash import Clash

        self._lddt_metric = LDDT().to(self.device)
        # vdw clash needs mol_id + elements_one_hot; keep af3-clash only (geometric).
        self._clash_metric = Clash(
            compute_af3_clash=True, compute_vdw_clash=False
        )
        return self._lddt_metric, self._clash_metric

    def init_wandb(self):
        # rank0-only: only the main process owns the wandb run.
        if not self.is_main:
            self.wandb_run = None
            return None

        import wandb

        os.environ.setdefault("WANDB_MODE", "offline")
        kwargs = dict(
            project=self.wandb_project,
            mode="offline",
            config={
                "num_recycles": self.num_recycles,
                "num_diffusion_samples_training": self.num_diffusion_samples_training,
                "mini_rollout_steps": self.mini_rollout_steps,
                "train_crop_size": self.train_crop_size,
                "lr": self.lr,
                "grad_clip": self.grad_clip,
                "accumulation_steps": self.accumulation_steps,
                "amp_dtype": self.amp_dtype,
                "ema_decay": self.ema_decay,
                "lr_scheduler": self.lr_scheduler_name,
                "warmup_steps": self.warmup_steps,
            },
        )
        if self.wandb_dir is not None:
            os.makedirs(self.wandb_dir, exist_ok=True)
            kwargs["dir"] = self.wandb_dir

        # Resume into the SAME wandb run when WANDB_RUN_ID is set (typical
        # use: re-launch reads the run id of the original job from env and
        # appends to it). New metrics attach to the existing time series
        # under matching keys; keys present only in the new code (e.g.
        # after a metric-layout refactor) appear as fresh curves starting
        # at the resume step, with the old curves frozen at the resume
        # point -- a single wandb run shows the whole training history,
        # with the format transition visible. WANDB_RESUME overrides the
        # default "allow"; set "must" to fail if the id is unknown, or
        # "never" to disable resume entirely.
        resume_id = os.environ.get("WANDB_RUN_ID") or None
        if resume_id:
            kwargs["id"] = resume_id
            kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
            print(
                f"[wandb] resuming run id={resume_id} "
                f"resume_mode={kwargs['resume']}",
                flush=True,
            )
        self.wandb_run = wandb.init(**kwargs)
        return self.wandb_run

    def init_all(self):
        # init_env() first: resolves rank/world_size, inits the NCCL group and
        # pins self.device, so init_model (DDP wrap) and init_data (distributed
        # sampler) see the right topology. Single-process path is a no-op-ish.
        self.init_env()
        self.init_model()
        self.init_loss()
        self.init_permutation()
        self.init_data()
        self.init_optim()
        self.init_scheduler()
        self.init_ema()
        self.init_metrics()
        self.init_wandb()

    def init_permutation(self):
        """Symmetric (chain + atom) permutation of the GT label to match the
        mini-rollout prediction (AF3 permute_label_to_match_mini_rollout)."""
        self.symmetric_permutation = None
        try:
            import ml_collections
            from torchfold.utils.permutation.permutation import SymmetricPermutation
            perm_cfg = ml_collections.ConfigDict({
                "chain_permutation": {
                    "train": {"mini_rollout": True, "diffusion_sample": False},
                    "test": {"diffusion_sample": True},
                    "permute_by_pocket": True,
                    "configs": {
                        "use_center_rmsd": False,
                        "find_gt_anchor_first": False,
                        "accept_it_as_it_is": False,
                        "enumerate_all_anchor_pairs": False,
                        "selection_metric": "aligned_rmsd",
                    },
                },
                "atom_permutation": {
                    "train": {"mini_rollout": True, "diffusion_sample": False},
                    "test": {"diffusion_sample": True},
                    "permute_by_pocket": True,
                    "global_align_wo_symmetric_atom": False,
                },
            })
            err_dir = os.path.join(self.run_dir, "perm_errors") if self.run_dir else None
            self.symmetric_permutation = SymmetricPermutation(perm_cfg, error_dir=err_dir)
            self._log_main("[perm] SymmetricPermutation enabled (mini-rollout label permutation)")
        except Exception as e:
            self._log_main(f"[perm] DISABLED (unavailable): {type(e).__name__}: {e}")

    def _make_permute_label_cb(self, batch: dict, af3_feats: dict):
        """Approach B: build the callback the model forward uses to permute the GT
        label to the mini-rollout frame BEFORE diffusion-training, so the diffusion
        pred is born in the permuted frame (no post-forward pred gather needed).

        Returns None (caller falls back to plain dense GT) when permutation is
        unavailable / disabled. Otherwise returns a closure cb(mini_dense_coords) ->
        (gt_positions, gt_mask). FIX (B): the whole permutation runs on a DEEP COPY
        of the label dict and batch['label_dict'] is committed ONLY on full success;
        on ANY exception nothing is mutated and the ORIGINAL dense GT is returned, so
        the end state stays fully-original-consistent.
        """
        import os as _os
        import copy as _copy
        if getattr(self, "symmetric_permutation", None) is None:
            return None
        if _os.environ.get("TORCHFOLD_DISABLE_PERM", "0") == "1":
            return None

        sp = self.symmetric_permutation
        device = self.device
        # original dense GT the forward already had (failure fallback).
        orig_gt_positions = af3_feats["true_positions"]
        orig_gt_mask = af3_feats["true_positions_atom_mask"]

        feats0 = batch["input_feature_dict"]
        a2t0 = feats0["atom_to_token_idx"].to(torch.int64).to(device)
        a2ta0 = feats0["atom_to_tokatom_idx"].to(torch.int64).to(device)
        n_token, max_dense = af3_feats["pred_dense_atom_mask"].shape

        def cb(mini_dense_coords):
            with torch.autocast(device_type="cuda", enabled=False):
                try:
                    # 1) flatten mini-rollout dense coords -> flat [1, N_atom, 3].
                    mini = mini_dense_coords.to(torch.float32)
                    mini_flat = _dense_to_flat(mini, a2t0, a2ta0).detach()
                    # 2) permute a DEEP COPY of the label dict (the method mutates its
                    #    label_dict in place via .update() -- never touch the live dict).
                    label_copy = _copy.deepcopy(batch["label_dict"])
                    label_full = batch.get("label_full_dict", label_copy)
                    dev = label_copy["coordinate"].device
                    new_label, _log = sp.permute_label_to_match_mini_rollout(
                        mini_flat.to(dev), feats0, label_copy, label_full,
                    )
                    # 3) rebuild dense GT from the PERMUTED-COPY coordinate using the
                    #    SAME scatter pattern as _build_gt_dense.
                    coord = new_label["coordinate"].to(torch.float32).to(device)
                    cmask = new_label["coordinate_mask"].to(torch.float32).to(device)
                    gt_positions = coord.new_zeros((n_token, max_dense, 3))
                    gt_positions[a2t0, a2ta0] = coord
                    gt_mask = cmask.new_zeros((n_token, max_dense))
                    gt_mask[a2t0, a2ta0] = cmask
                    gt_mask = gt_mask.to(torch.bool)
                    # 4) commit the permuted label LAST, only on full success.
                    batch["label_dict"] = new_label
                    return gt_positions, gt_mask
                except Exception as e:
                    self._log_main(
                        f"[perm] WARNING: label perm skipped (kept original GT): "
                        f"{type(e).__name__}: {e}"
                    )
                    return orig_gt_positions, orig_gt_mask

        return cb

    def _maybe_permute_label(self, batch: dict, out: dict) -> None:
        """In-place permute batch['label_dict'] to match the mini-rollout prediction.
        No-op (with a logged warning) if permutation is unavailable or errors."""
        import os as _os
        if _os.environ.get("TORCHFOLD_DISABLE_PERM", "0") == "1":
            return
        sp = getattr(self, "symmetric_permutation", None)
        if sp is None or "confidence_atom_positions" not in out:
            return
        try:
            feats0 = batch["input_feature_dict"]
            a2t0 = feats0["atom_to_token_idx"].to(torch.int64).to(self.device)
            a2ta0 = feats0["atom_to_tokatom_idx"].to(torch.int64).to(self.device)
            mini = out["confidence_atom_positions"].to(torch.float32)        # [1,N_tok,max_dense,3]
            mini_flat = _dense_to_flat(mini, a2t0, a2ta0).detach()           # [1, N_atom, 3]
            label_dict = batch["label_dict"]
            label_full = batch.get("label_full_dict", label_dict)
            # match devices: run permutation on the label tensors' device.
            dev = label_dict["coordinate"].device
            new_label, _log = sp.permute_label_to_match_mini_rollout(
                mini_flat.to(dev), feats0, label_dict, label_full,
            )
            batch["label_dict"] = new_label
        except Exception as e:
            self._log_main(f"[perm] WARNING: skipped (kept original GT): {type(e).__name__}: {e}")

    def _maybe_permute_diffusion_pred(self, batch: dict, out: dict) -> None:
        import os as _os
        if _os.environ.get("TORCHFOLD_DISABLE_PERM", "0") == "1":
            return
        sp = getattr(self, "symmetric_permutation", None)
        diff = out.get("diffusion_samples") if isinstance(out, dict) else None
        if sp is None or not isinstance(diff, dict) or "atom_positions" not in diff:
            return
        try:
            feats0 = batch["input_feature_dict"]
            a2t0 = feats0["atom_to_token_idx"].to(torch.int64).to(self.device)
            a2ta0 = feats0["atom_to_tokatom_idx"].to(torch.int64).to(self.device)
            flat_pred = _dense_to_flat(
                diff["atom_positions"].to(torch.float32), a2t0, a2ta0
            )                                                                # [S, N_atom, 3] (grad-bearing)
            label_dict = batch["label_dict"]
            dev = label_dict["coordinate"].device
            pred_dict = {
                "coordinate": flat_pred.detach().to(dev),
                "coordinate_mask": label_dict["coordinate_mask"].to(dev),
            }
            # stage="test" is the only stage where torchfold's method enables CHAIN perm
            # (it skips chain perm when stage=="train"); test.diffusion_sample=True so the
            # search returns per-sample chain+atom indices. label_dict is the cropped,
            # mini-rollout-permuted match target (atom counts match the cropped pred).
            _new_pred, _log, pidx, _ = sp.permute_diffusion_sample_to_match_label(
                feats0, pred_dict, label_dict, stage="test", permute_by_pocket=False,
            )
            if pidx:
                # apply per-sample [N_atom] index perm to the NON-detached flat_pred (pure
                # gather -> diffusion gradient preserved); detached search coords are unused.
                perm = torch.stack([p.to(flat_pred.device) for p in pidx], dim=0)   # [S, N_atom]
                permuted = torch.gather(flat_pred, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
                out["diffusion_samples"]["pred_coord_flat"] = permuted              # [S, N_atom, 3]
        except Exception as e:
            self._log_main(
                f"[perm] WARNING: diffusion-pred perm skipped (unpermuted pred): "
                f"{type(e).__name__}: {e}"
            )

    # -- forward + loss (shared train/eval helper) --------------------------
    def _adapt_with_gt(self, batch: dict):
        from torchfold.data.pipeline.adapter import standard_batch_to_af3

        device = self.device
        af3_feats = standard_batch_to_af3(batch, device=device)
        true_positions, true_mask = _build_gt_dense(batch, af3_feats, device)
        af3_feats["true_positions"] = true_positions
        af3_feats["true_positions_atom_mask"] = true_mask
        return af3_feats

    def _label_on_device(self, batch: dict):
        device = self.device

        def _to_dev(d):
            return {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in d.items()
            }

        return {
            "input_feature_dict": _to_dev(batch["input_feature_dict"]),
            "label_dict": _to_dev(batch["label_dict"]),
        }

    # -- one micro-batch (forward/backward; optimizer step on accum boundary) -
    def train_micro_step(self, batch: dict) -> dict:
        assert self.model.training, "model must be in train() for the training forward"

        # Apply p_mean curriculum for the optimizer step about to be taken.
        noise_p_mean = self._apply_noise_curriculum()

        import time as _time
        def _psync():
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        _pt = {}
        _t0 = _time.perf_counter()

        af3_feats = self._adapt_with_gt(batch)
        _psync(); _pt["adapt"] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()

        # Is this micro-step the accumulation boundary (the one that will fire the
        # optimizer step)? Under DDP, NON-boundary micro-steps run inside
        # no_sync() so the gradient all-reduce only happens on the boundary,
        # avoiding premature/extra syncs during accumulation.
        is_boundary = ((self.micro_step + 1) % self.accumulation_steps == 0)
        if self.is_distributed and not is_boundary:
            sync_ctx = self.model_ddp.no_sync()
        else:
            sync_ctx = contextlib.nullcontext()

        with sync_ctx:
            # ---- AMP bf16 forward (model only); loss in fp32 (autocast disabled) ----
            # Approach B: build the label-permute callback; the structure-training
            # branch runs the mini-rollout first then permutes the GT label to that
            # frame, so the diffusion pred is born permuted (no post-forward gather).
            permute_label_cb = self._make_permute_label_cb(batch, af3_feats)
            with self._autocast():
                out = self.model_ddp(af3_feats, permute_label_cb=permute_label_cb)
                # confidence_only:
                # trunk+diffusion frozen; the model emits confidence_atom_positions
                # (mini-rollout) instead of diffusion_samples and only the confidence
                # loss is computed -- skip the structure-path asserts.
                if not out.get("confidence_only", False):
                    assert "diffusion_samples" in out, (
                        "training forward did not emit diffusion_samples"
                    )
                    assert "noise_levels" in out["diffusion_samples"], (
                        "training forward did not expose noise_levels (per-sample noise level)"
                    )
            _psync(); _pt["fwd"] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()

            # Approach B post-forward routing:
            #  - confidence_only: still permute the label to the mini-rollout (UNCHANGED).
            #  - structure branch: do NOTHING -- the GT label was already permuted inside
            #    the forward (via permute_label_cb) and the diffusion pred is born in that
            #    permuted frame, so FIX (A) drops the post-forward chain+atom pred gather.
            #    loss.py falls back to the plain dense gather when pred_coord_flat is absent.
            if out.get("confidence_only", False):
                self._maybe_permute_label(batch, out)
            _psync(); _pt["perm"] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()

            label = self._label_on_device(batch)
            # loss with autocast explicitly DISABLED.
            with torch.autocast(device_type="cuda", enabled=False) if self.device.type == "cuda" else contextlib.nullcontext():
                total, loss_dict = self.loss_fn(pred=out, label=label, mode="train")
            _psync(); _pt["loss"] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()

            # ---- backward (scaled for accumulation) ----
            scaled = total / self.accumulation_steps
            scaled.backward()
            _psync(); _pt["bwd"] = _time.perf_counter() - _t0

        self._log_main(
            "[timing] micro=%d adapt=%.1f fwd=%.1f perm=%.1f loss=%.1f bwd=%.1f (s)"
            % (self.micro_step, _pt.get("adapt", -1.0), _pt.get("fwd", -1.0),
               _pt.get("perm", -1.0), _pt.get("loss", -1.0), _pt.get("bwd", -1.0))
        )

        self.micro_step += 1
        did_opt_step = False
        grad_norm = None
        cur_lr = self.optimizer.param_groups[0]["lr"]

        if self.micro_step % self.accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            did_opt_step = True
            cur_lr = self.optimizer.param_groups[0]["lr"]

        log = {
            "loss": float(total.detach().item()),
            "lr": float(cur_lr),
            "step": self.step,
            "did_opt_step": did_opt_step,
            "noise_p_mean": float(noise_p_mean),   # 0907
        }
        if grad_norm is not None:
            log["grad_norm"] = float(grad_norm)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                log[k] = float(v.detach().item())
        # wandb
        if did_opt_step:
            for k, v in log.items():
                if k in ("did_opt_step", "step"):
                    continue
                self._train_metric_acc.setdefault(k, []).append(float(v))

            if (
                self.wandb_run is not None
                and self.log_interval > 0
                and self.step % self.log_interval == 0
            ):
                payload = {
                    f"train/{k}.avg": float(sum(vs) / len(vs))
                    for k, vs in self._train_metric_acc.items()
                    if vs and k not in ("lr", "noise_p_mean")
                }
                if "lr" in self._train_metric_acc and self._train_metric_acc["lr"]:
                    payload["train/lr"] = float(self._train_metric_acc["lr"][-1])
                self.wandb_run.log(payload, step=self.step)
                self._train_metric_acc.clear()
        return log

    # Backwards-compatible single-step alias (no accumulation): one full step.
    def train_step(self, batch: dict) -> dict:
        # process exactly `accumulation_steps` copies of the same batch so that a
        # real optimizer step happens; primarily for ad-hoc use.
        log = None
        for _ in range(self.accumulation_steps):
            log = self.train_micro_step(batch)
        return log

    # -- eval / metrics -----------------------------------------------------
    @torch.no_grad()
    def evaluate(self, max_batches_per_set: int = 0) -> dict:
        """Evaluate on all configured test datasets.

        Runs inference over each test set in self.test_dls, computes per-dataset
        LDDT (+ RMSD + clash) metrics over each of the N_sample diffusion
        samples, aggregated with three AF3-paper confidence
        rankers (plddt.rank1 / gpde.rank1 / ranking_score.rank1) alongside
        the GT-driven rankers (best/worst/mean/median/random).

        Args:
            max_batches_per_set: cap number of batches per dataset (0 = no cap).
        Returns:
            dict with all logged metrics (raw + ema) for all test sets.
        """
        if self._lddt_metric is None:
            self.init_metrics()

        all_metrics: dict = {}

        if not self.eval_ema_only:
            raw_metrics = self._evaluate(ema_prefix="", max_batches=max_batches_per_set)
            all_metrics.update(raw_metrics)

        if self.ema is not None:
            self.ema.apply_shadow()
            try:
                ema_metrics = self._evaluate(
                    ema_prefix=f"ema{self.ema_decay}_",
                    max_batches=max_batches_per_set,
                )
                all_metrics.update(ema_metrics)
            finally:
                self.ema.restore()

        # rank0-only wandb log (aggregated over all datasets)
        if self.is_main and self.wandb_run is not None and all_metrics:
            self.wandb_run.log(all_metrics, step=self.step)
            self.wandb_run.summary.update(all_metrics)

        return all_metrics

    @torch.no_grad()
    def _evaluate(self, ema_prefix: str = "", max_batches: int = 0) -> dict:
        """Internal: run eval loop over all test sets; return flat metric dict.

        Metric keys:
          - LDDT/RMSD/clash rankers:
              ``{dataset}/{ema_prefix}{metric}/complex/{ranker}.avg``
              ``ranker`` ∈ best/worst/mean/median/random/plddt.rank1/
              gpde.rank1/ranking_score.rank1.
          - Loss terms:
              ``{dataset}/{ema_prefix}{loss_name}.avg`` for each term in
              the loss aggregator output (loss, mse_loss, weighted_mse_loss,
              bond_loss, smooth_lddt_loss, distogram_loss, plddt_loss,
              pde_loss, resolved_loss, pae_loss, ... + weighted_* variants).
        Raw and EMA share the per-dataset wandb panel.

        Args:
            ema_prefix: inserted between the dataset and metric segments of every
                key
            max_batches: cap batches per set (0 = no cap; use full test set).
        """
        device = self.device
        was_training = self.model.training
        self.model.eval()

        test_dls = self.test_dls
        if not test_dls:
            # Fallback: no test sets configured; use val_dl under the name "val"
            test_dls = {"val": self.val_dl} if self.val_dl is not None else {}

        all_metrics: dict = {}

        try:
            for test_name, dl in test_dls.items():
                # populated dynamically with per-ranker keys like
                # "lddt/complex/best.avg" (see N_sample aggregation below).
                agg: dict[str, list] = {}
                n_done = 0
                n_attempted = 0
                # max_attempts: cap total attempts to 3x the batch limit to avoid
                # spending forever on a broken CUDA context (device-side assert propagates).
                max_attempts = (max_batches * 3 + 10) if max_batches > 0 else 10000
                it = iter(dl)
                while True:
                    if max_batches > 0 and n_done >= max_batches:
                        break
                    if n_attempted >= max_attempts:
                        self._log_main(
                            f"[eval] WARNING: {test_name} hit max_attempts={max_attempts}; "
                            f"only {n_done}/{n_attempted} batches succeeded."
                        )
                        break
                    try:
                        batch = next(it)
                    except StopIteration:
                        break
                    n_attempted += 1

                    try:
                        from torchfold.data.pipeline.adapter import standard_batch_to_af3

                        af3_feats = standard_batch_to_af3(batch, device=device)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _t_b0 = time.perf_counter()
                        with self._autocast():
                            out = self.model(af3_feats)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _t_fwd = time.perf_counter() - _t_b0

                        # inference output: diffusion_samples['atom_positions']
                        # has shape [N_sample, N_token, max_dense, 3] (or
                        # [N_token, max_dense, 3] when N_sample == 1).
                        samples = out["diffusion_samples"]
                        pred_dense_all = samples["atom_positions"].to(torch.float32)
                        if pred_dense_all.dim() == 3:
                            pred_dense_all = pred_dense_all.unsqueeze(0)
                        N_sample = pred_dense_all.shape[0]

                        feats = batch["input_feature_dict"]
                        a2t = feats["atom_to_token_idx"].to(torch.int64).to(device)
                        a2ta = feats["atom_to_tokatom_idx"].to(torch.int64).to(device)

                        label = batch["label_dict"]
                        true_flat = label["coordinate"].to(torch.float32).to(device)
                        true_mask = label["coordinate_mask"].to(torch.float32).to(device)

                        best_plddt_idx = None  # filled after summary below
                        best_gpde_idx = None
                        best_ranking_idx = None

                        sample_plddt_masked = None
                        if "predicted_lddt" in out:
                            plddt_dense = out["predicted_lddt"].to(torch.float32)
                            if plddt_dense.dim() == 2:
                                plddt_dense = plddt_dense.unsqueeze(0)
                            atom_plddt = plddt_dense[..., a2t, a2ta]  # [N_sample, N_atom]
                            mask_w = true_mask.unsqueeze(0).expand_as(atom_plddt)
                            denom = mask_w.sum(dim=-1).clamp(min=1.0)
                            sample_plddt_masked = (atom_plddt * mask_w).sum(dim=-1) / denom
                            best_plddt_idx = int(sample_plddt_masked.argmax().item())

                        # Per-sample metrics for this batch (one entry per
                        # diffusion sample; later rank-aggregated below).
                        batch_per_sample: dict[str, list[float]] = {
                            "lddt": [], "rmsd": [], "clash_frac": [],
                        }
                        per_sample_has_clash: list[float] = []  # 0/1 per sample
                        for s_idx in range(N_sample):
                            pred_flat = _dense_to_flat(
                                pred_dense_all[s_idx], a2t, a2ta
                            )  # [N_atom, 3]
                            m = self._compute_batch_metrics(
                                pred_flat, true_flat, true_mask, feats, a2t, device
                            )
                            for src_k, dst_k in (
                                ("val_lddt", "lddt"),
                                ("val_rmsd", "rmsd"),
                                ("val_clash_frac", "clash_frac"),
                            ):
                                if src_k in m and m[src_k] is not None:
                                    batch_per_sample[dst_k].append(float(m[src_k]))
                            cf = m.get("val_clash_frac")
                            per_sample_has_clash.append(
                                1.0 if (cf is not None and cf > 0.0) else 0.0
                            )

                        # Per-sample summary confidence (gpde / ptm / iptm /
                        # ranking_score). Falls back gracefully if the model
                        # output does not carry the necessary logits (e.g.
                        # confidence head off in some smoke configs).
                        try:
                            from torchfold.runner.sample_confidence import (
                                compute_summary_confidence,
                            )

                            has_clash_t = torch.tensor(
                                per_sample_has_clash,
                                dtype=torch.float32,
                                device=device,
                            )
                            summary = compute_summary_confidence(
                                out, feats, has_clash_t
                            )
                            if summary.get("gpde") is not None and summary["gpde"].numel() > 0:
                                # gpde: lower is better -> argmin.
                                best_gpde_idx = int(summary["gpde"].argmin().item())
                            if (
                                summary.get("ranking_score") is not None
                                and summary["ranking_score"].numel() > 0
                            ):
                                best_ranking_idx = int(
                                    summary["ranking_score"].argmax().item()
                                )
                        except (KeyError, RuntimeError) as _e:
                            # confidence head missing or shape mismatch ->
                            # silently skip gpde / ranking_score rankers; the
                            # other rankers (best/worst/mean/median/random/
                            # plddt.rank1) are unaffected.
                            self._log_main(
                                f"[eval] summary_confidence skipped: "
                                f"{type(_e).__name__}: {_e}"
                            )

                        # Aggregate over N_sample
                        for metric_name, vals in batch_per_sample.items():
                            if not vals:
                                continue
                            arr = torch.tensor(vals, dtype=torch.float32)
                            rankers = {
                                "best":   float(arr.max()),
                                "worst":  float(arr.min()),
                                "mean":   float(arr.mean()),
                                "median": float(arr.median()),
                                "random": float(arr[0]),  # sample 0
                            }
                            if best_plddt_idx is not None and best_plddt_idx < arr.shape[0]:
                                rankers["plddt.rank1"] = float(arr[best_plddt_idx])
                            if best_gpde_idx is not None and best_gpde_idx < arr.shape[0]:
                                rankers["gpde.rank1"] = float(arr[best_gpde_idx])
                            if best_ranking_idx is not None and best_ranking_idx < arr.shape[0]:
                                rankers["ranking_score.rank1"] = float(arr[best_ranking_idx])
                            diffs = {
                                "diff/best_worst":  rankers["best"] - rankers["worst"],
                                "diff/best_random": rankers["best"] - rankers["random"],
                                "diff/best_median": rankers["best"] - rankers["median"],
                            }
                            for rk in ("plddt", "gpde", "ranking_score"):
                                if f"{rk}.rank1" in rankers:
                                    diffs[f"diff/best_{rk}"] = (
                                        rankers["best"] - rankers[f"{rk}.rank1"]
                                    )
                                    diffs[f"diff/{rk}_median"] = (
                                        rankers[f"{rk}.rank1"] - rankers["median"]
                                    )
                            for ranker, v in rankers.items():
                                # Per-ranker key under each metric. Final
                                # wandb key composed below as
                                # "{dataset}/{ema_prefix}{this}".
                                agg.setdefault(
                                    f"{metric_name}/complex/{ranker}.avg", []
                                ).append(v)
                            del diffs  # explicit: we computed but discard

                        # Eval loss on this batch
                        try:
                            _, eval_loss_dict = self.loss_fn(
                                pred=out, label=batch, mode="eval"
                            )
                            for lk, lv in eval_loss_dict.items():
                                if isinstance(lv, torch.Tensor):
                                    agg.setdefault(f"{lk}.avg", []).append(
                                        float(lv.detach().item())
                                    )
                        except (KeyError, RuntimeError, AssertionError) as _le:
                            # Loss skipped (likely missing label key or
                            # confidence-head output): keep the lddt-side
                            # metrics, log once per batch for visibility.
                            self._log_main(
                                f"[eval] loss skipped: "
                                f"{type(_le).__name__}: {_le}"
                            )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _t_total = time.perf_counter() - _t_b0
                        self._log_main(
                            f"[eval-prof] {test_name} #{n_done}: "
                            f"fwd={_t_fwd:.1f}s metrics+loss={_t_total - _t_fwd:.1f}s "
                            f"total={_t_total:.1f}s"
                        )
                        n_done += 1
                    except torch.cuda.OutOfMemoryError as _e:
                        # Only OOM (huge uncropped test structures) is skipped, with a
                        # loud warning + cache reset. Every other exception propagates:
                        # silently masking real errors here once hid the feature-adapter
                        # bugs (LDDT 0). A skipped batch does NOT pollute the average
                        # (we `continue` without appending), it only lowers num_batches.
                        self._log_main(
                            f"[eval] WARNING: OOM on {test_name} batch, skipping: {_e}"
                        )
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        continue

                # Distributed merge before averaging. Each rank holds only
                # its KeySumBalancedSampler shard's per-sample values in `agg`.
                merged_agg = agg
                total_done = n_done
                if self.is_distributed and dist.is_initialized():
                    gathered: list = [None] * self.world_size
                    dist.all_gather_object(
                        gathered, {"agg": agg, "n_done": n_done}
                    )
                    merged_agg = {}
                    total_done = 0
                    for part in gathered:
                        if not part:
                            continue
                        total_done += part["n_done"]
                        for mk, mv in part["agg"].items():
                            merged_agg.setdefault(mk, []).extend(mv)

                # aggregate (avg) and add to result dict with per-dataset prefix
                for metric_name, vals in merged_agg.items():
                    if vals:
                        avg = float(sum(vals) / len(vals))
                        key = f"{test_name}/{ema_prefix}{metric_name}"
                        all_metrics[key] = avg

                all_metrics[f"{test_name}/{ema_prefix}num_batches"] = total_done
                self._log_main(
                    f"[eval] step={self.step} {ema_prefix}{test_name}: "
                    + ", ".join(
                        # console: print only the `mean` ranker per metric to
                        # keep the line short; full per-ranker breakdown is in
                        # wandb.
                        f"{k}={v:.4f}" for k, v in all_metrics.items()
                        if k.startswith(f"{test_name}/{ema_prefix}") and k.endswith("/mean.avg")
                    )
                )
        finally:
            if was_training:
                self.model.train()

        return all_metrics

    @torch.no_grad()
    def validate(self, num_batches: int = 1, use_ema: bool = True) -> dict:
        """Backward-compat alias for evaluate().

        Calls evaluate(max_batches_per_set=num_batches) and returns the flat
        metric dict.
        """
        return self.evaluate(max_batches_per_set=num_batches)

    @torch.no_grad()
    def _compute_batch_metrics(
        self, pred_flat, true_flat, true_mask, feats, a2t, device
    ) -> dict:
        from torchfold.metrics.lddt_metrics import LDDT

        out = {}
        mask_bool = true_mask > 0.5
        if mask_bool.sum() < 2:
            return out

        # restrict to atoms with valid GT coords.
        idx = torch.nonzero(mask_bool, as_tuple=True)[0]
        pred_v = pred_flat.index_select(0, idx)      # [N_valid, 3]
        true_v = true_flat.index_select(0, idx)      # [N_valid, 3]

        # ---- LDDT (cap atom count to keep the NxN mask cheap) ----
        cap = 1500
        if pred_v.shape[0] > cap:
            sel = torch.randperm(pred_v.shape[0], device=device)[:cap]
            pred_l, true_l = pred_v.index_select(0, sel), true_v.index_select(0, sel)
        else:
            pred_l, true_l = pred_v, true_v
        coord_mask = torch.ones(true_l.shape[0], device=device)
        lddt_mask = LDDT.compute_lddt_mask(
            true_coordinate=true_l, true_coordinate_mask=coord_mask, threshold=15.0
        )
        lddt = self._lddt_metric(pred_l.unsqueeze(0), true_l, lddt_mask)  # [1]
        out["val_lddt"] = float(lddt.mean().item())

        # ---- self-aligned RMSD ----
        from torchfold.metrics.rmsd import self_aligned_rmsd

        atom_mask = torch.ones(1, pred_v.shape[0], device=device)
        r, _, _, _ = self_aligned_rmsd(
            pred_v.unsqueeze(0), true_v.unsqueeze(0), atom_mask, reduce=True
        )
        out["val_rmsd"] = float(r.item())

        # ---- AF3 clash (geometric; needs >=2 chains) ----
        try:
            asym_id = feats["asym_id"].to(torch.int64).to(device)        # [N_token]
            is_protein = feats["is_protein"].to(torch.float32).to(device)
            is_ligand = feats["is_ligand"].to(torch.float32).to(device)
            is_dna = feats["is_dna"].to(torch.float32).to(device)
            is_rna = feats["is_rna"].to(torch.float32).to(device)
            n_atom = pred_flat.shape[0]
            if is_protein.shape[0] == n_atom and asym_id.shape[0] != n_atom:
                pass  # token-level asym_id, atom-level type flags: ok for clash
            n_chains = int(torch.unique(asym_id).numel())
            if n_chains >= 2:
                res = self._clash_metric(
                    pred_coordinate=pred_flat.unsqueeze(0),
                    asym_id=asym_id,
                    atom_to_token_idx=a2t,
                    is_ligand=is_ligand,
                    is_protein=is_protein,
                    is_dna=is_dna,
                    is_rna=is_rna,
                )
                af3_clash = res["summary"]["af3_clash"].to(torch.float32)  # [S,C,C]
                # fraction of off-diagonal chain pairs flagged as clashing.
                C = af3_clash.shape[-1]
                eye = torch.eye(C, device=af3_clash.device)
                off = (1.0 - eye)
                denom = off.sum().clamp(min=1.0)
                out["val_clash_frac"] = float(
                    (af3_clash[0] * off).sum().item() / denom.item()
                )
            else:
                out["val_clash_frac"] = 0.0
        except Exception as e:  # clash is best-effort; never break eval
            out["val_clash_frac"] = float("nan")
        return out

    # -- checkpoint ---------------------------------------------------------
    def _default_ckpt_dir(self):
        if self.run_dir is not None:
            d = os.path.join(self.run_dir, "checkpoints")
        else:
            d = os.path.join("/tmp", "torchfold_runs", time.strftime("%Y%m%d_%H%M%S"),
                             "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def save_checkpoint(self, ema_suffix: str = "", path: Optional[str] = None) -> Optional[str]:
        """Save a checkpoint to ``{ckpt_dir}/{step}{ema_suffix}.pt`` (rank-0 only).

        Parameters
        ----------
        ema_suffix:
            Appended to the step number in the auto-generated filename.
            Use ``""`` for the raw-weights file and ``"_ema"`` for the EMA
            file (caller must call ``ema.apply_shadow()`` before and
            ``ema.restore()`` after).  Ignored when *path* is given explicitly.
        path:
            Explicit destination path (overrides the auto-generated name).
            Kept for backward compatibility with test code that passes a path
            directly.
        """
        # rank0-only write: only the main process persists the checkpoint. The
        # state dict comes from the *unwrapped* module (self.model), so the file
        # format is identical to the single-GPU checkpoint (no "module." prefix).
        if not self.is_main:
            return None
        if path is None:
            path = os.path.join(self._default_ckpt_dir(), f"{self.step}{ema_suffix}.pt")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "micro_step": self.micro_step,
            "ema": self.ema.state_dict() if self.ema else None,
        }
        torch.save(ckpt, path)
        return path

    def try_load_checkpoint(self, path: str, load_params_only: bool = False) -> bool:
        if not os.path.isfile(path):
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        assert self.model is not None, "build the model before loading a checkpoint"

        if "model_state_dict" in ckpt and "model" not in ckpt:
            ckpt["model"] = ckpt["model_state_dict"]
        if "optimizer_state_dict" in ckpt and "optimizer" not in ckpt:
            ckpt["optimizer"] = ckpt["optimizer_state_dict"]
        if "scheduler_state_dict" in ckpt and "scheduler" not in ckpt:
            ckpt["scheduler"] = ckpt["scheduler_state_dict"]
        if "global_step" in ckpt and "step" not in ckpt:
            ckpt["step"] = ckpt["global_step"]

        # Strip a leading "module." prefix from model state-dict keys if present.
        raw_sd = ckpt["model"]
        if any(k.startswith("module.") for k in raw_sd):
            raw_sd = {k[len("module."):]: v for k, v in raw_sd.items()}
            ckpt["model"] = raw_sd

        self.model.load_state_dict(ckpt["model"], strict=False)

        if load_params_only:
            # also refresh EMA shadow from the loaded params.
            if self.ema is not None:
                self.ema.register()
            return True

        self.step = int(ckpt.get("step", 0))
        self.micro_step = int(ckpt.get("micro_step", self.step * self.accumulation_steps))
        self.epoch = int(ckpt.get("epoch", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", 0.0))

        if self.optimizer is not None and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # rebuild scheduler at the loaded step so its internal counter matches.
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.init_scheduler(last_epoch=self.step - 1)
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        # Re-apply noise curriculum for the resumed step.
        if hasattr(self.model, "train_noise_sampler"):
            self._apply_noise_curriculum()   # 0907
        return True

    # -- top-level loop -----------------------------------------------------
    def train(
        self,
        num_steps: int,
        *,
        eval_interval: int = 0,
        ckpt_interval: int = 0,
        eval_batches: int = 1,
    ) -> list[dict]:
        """Run ``num_steps`` *optimizer* steps with periodic eval + checkpoint.

        Each optimizer step consumes ``accumulation_steps`` micro-batches.
        """
        assert self.train_dl is not None, "call init_data() first"
        logs = []
        it = iter(self.train_dl)

        def _next():
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(self.train_dl)
                return next(it)

        start_step = self.step
        while self.step < start_step + num_steps:
            for _ in range(self.accumulation_steps):
                batch = _next()
                log = self.train_micro_step(batch)
            logs.append(log)
            self._log_main(
                f"[rank{self.rank}] step {self.step} loss={log['loss']:.5f} "
                f"lr={log['lr']:.3e} grad_norm={log.get('grad_norm')}"
            )

            if eval_interval and self.step % eval_interval == 0:
                self.evaluate(max_batches_per_set=eval_batches)
                self.barrier()  # cheap resync; ranks already synced in _evaluate
            if ckpt_interval and self.step % ckpt_interval == 0:
                # ----- two-file save: raw + EMA-----
                # 1. Raw weights.
                self.save_checkpoint("")
                # 2. EMA weights (apply shadow, save with _ema suffix, restore).
                if self.ema is not None:
                    self.ema.apply_shadow()
                    self.save_checkpoint("_ema")
                    self.ema.restore()
                self.barrier()  # keep ranks aligned around the rank0 write
        return logs


# ---------------------------------------------------------------------------
# torchrun entry point.
#
#   python -m torchfold.runner.train --af3_params_dir <dir> [--num_steps ...] ...
# ---------------------------------------------------------------------------
def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="torchfold trainer (single-GPU + DDP) torchrun entry."
    )
    p.add_argument(
        "--af3_params_dir",
        type=str,
        default="./Alphafold3params",
        help="JAX parameter dump directory (used unless --resume_ckpt).",
    )
    p.add_argument("--num_steps", type=int, default=4, help="optimizer steps")
    # resume / fine-tune from a torchfold {step}.pt checkpoint. When set, the
    # JAX dump is not imported. Full resume (model+optim+scheduler+step+ema)
    # unless --load_params_only (then only model params; fresh optimizer + LR).
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--load_params_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--model_name", type=str, default="torchfold_af3_default")  # registry: configs/configs_model_type.py
    p.add_argument("--num_recycles", type=int, default=4)
    p.add_argument("--pair_dropout", type=float, default=0.0, help="AF3 pairformer rowwise pair dropout (0.25=AF3).")
    p.add_argument("--msa_dropout", type=float, default=0.0, help="AF3 MSA rowwise dropout (0.15=AF3).")
    p.add_argument(
        "--randomize_num_recycles", action="store_true", default=False,
        help="variable recycle count (handled under DDP via find_unused_parameters).",
    )
    p.add_argument(
        "--train_mode", choices=["full", "structure_only", "confidence_only"], default="full",
        help="full = structure+distogram+confidence; structure_only = structure only "
             "(freeze confidence head); confidence_only = freeze trunk+diffusion, train only "
             "the confidence head.",
    )
    p.add_argument(
        "--grad_checkpoint", action="store_true", default=False,
        help="enable activation/gradient checkpointing (needed for large crops, "
             "e.g. 256/384/640+); slower per step, much lower memory.",
    )
    p.add_argument("--num_diffusion_samples_training", type=int, default=48)
    p.add_argument("--mini_rollout_steps", type=int, default=20)
    p.add_argument("--train_crop_size", type=int, default=64)
    p.add_argument("--num_dl_workers", type=int, default=16)
    p.add_argument("--epoch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.8e-3)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--decay_every_n_steps", type=int, default=50000)
    p.add_argument("--decay_factor", type=float, default=0.95)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--eval_diffusion_steps", type=int, default=200)
    p.add_argument("--eval_num_samples", type=int, default=5)
    p.add_argument("--sample_diffusion_chunk_size", type=int, default=5)
    p.add_argument("--eval_ema_only", action="store_true", default=False)
    p.add_argument("--diffusion_sparse_loss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--diffusion_lddt_loss_dense", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loss_alpha_diffusion", type=float, default=4.0)
    p.add_argument("--loss_alpha_bond", type=float, default=1.0)
    p.add_argument("--loss_smooth_lddt", type=float, default=1.0)
    p.add_argument("--loss_alpha_distogram", type=float, default=3e-2)
    p.add_argument("--loss_alpha_confidence", type=float, default=1e-4)
    p.add_argument("--loss_alpha_pae", type=float, default=1.0)
    p.add_argument("--loss_smooth_lddt_interface_weight", type=float, default=0.0)
    p.add_argument("--loss_smooth_lddt_interface_distance_threshold", type=float, default=5.0)
    p.add_argument("--loss_interface_mse_weight", type=float, default=0.0)  # interface-weighted MSE (off by default; e.g. 10.0)
    p.add_argument("--loss_interface_mse_distance_threshold", type=float, default=10.0)  # A; CA-CA (mode=ca) or atom-atom (mode=atom)
    p.add_argument("--loss_interface_mse_mode", type=str, default="ca", choices=["ca", "atom"])
    p.add_argument("--test_max_n_token", type=int, default=-1,
                   help="max tokens per test sample (-1 = no limit; set small to skip huge inputs)")
    p.add_argument("--eval_interval", type=int, default=0)
    p.add_argument("--ckpt_interval", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=50,
                   help="optimizer steps between train-metric .avg flushes to wandb")
    p.add_argument("--eval_batches", type=int, default=1)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--wandb_dir", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="torchfold")
    p.add_argument(
        "--save_final_ckpt", action="store_true", default=False,
        help="save a checkpoint at the end of training (rank0 only).",
    )
    p.add_argument(
        "--noise_p_mean", type=float, default=-1.2,
        help="fixed TrainingNoiseSampler p_mean when --noise_p_mean_schedule is empty",
    )
    p.add_argument(
        "--noise_p_std", type=float, default=1.5,
        help="TrainingNoiseSampler p_std (log-normal std)",
    )
    p.add_argument(
        "--noise_p_mean_schedule", type=str, default="",
        help="comma-separated p_mean curriculum, e.g. '0.5,0,-0.5,-1.2'; "
             "empty disables schedule and uses --noise_p_mean",
    )
    p.add_argument(
        "--noise_p_mean_update_every", type=int, default=10000,
        help="optimizer steps per p_mean stage when schedule is set",
    )
    return p


def _parse_float_list(s: str):
    s = (s or "").strip()
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args, config_overrides = _build_arg_parser().parse_known_args()

    trainer = Trainer(
        af3_params_dir=args.af3_params_dir,
        model_name=args.model_name,
        num_recycles=args.num_recycles,
        randomize_num_recycles=args.randomize_num_recycles,
        pair_dropout=args.pair_dropout,
        msa_dropout=args.msa_dropout,
        train_mode=args.train_mode,
        grad_checkpoint=args.grad_checkpoint,
        num_diffusion_samples_training=args.num_diffusion_samples_training,
        mini_rollout_steps=args.mini_rollout_steps,
        diffusion_sparse_loss=args.diffusion_sparse_loss,
        diffusion_lddt_loss_dense=args.diffusion_lddt_loss_dense,
        loss_alpha_diffusion=args.loss_alpha_diffusion,
        loss_alpha_bond=args.loss_alpha_bond,
        loss_smooth_lddt=args.loss_smooth_lddt,
        loss_alpha_distogram=args.loss_alpha_distogram,
        loss_alpha_confidence=args.loss_alpha_confidence,
        loss_alpha_pae=args.loss_alpha_pae,
        loss_smooth_lddt_interface_weight=args.loss_smooth_lddt_interface_weight,
        loss_smooth_lddt_interface_distance_threshold=args.loss_smooth_lddt_interface_distance_threshold,
        loss_interface_mse_weight=args.loss_interface_mse_weight,
        loss_interface_mse_distance_threshold=args.loss_interface_mse_distance_threshold,
        loss_interface_mse_mode=args.loss_interface_mse_mode,
        train_crop_size=args.train_crop_size,
        num_dl_workers=args.num_dl_workers,
        epoch_size=args.epoch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        seed=args.seed,
        load_jax_params=not bool(args.resume_ckpt),
        accumulation_steps=args.accumulation_steps,
        amp_dtype=args.amp_dtype,
        warmup_steps=args.warmup_steps,
        decay_every_n_steps=args.decay_every_n_steps,
        decay_factor=args.decay_factor,
        ema_decay=args.ema_decay,
        eval_diffusion_steps=args.eval_diffusion_steps,
        eval_num_samples=args.eval_num_samples,
        sample_diffusion_chunk_size=args.sample_diffusion_chunk_size,
        eval_ema_only=args.eval_ema_only,
        log_interval=args.log_interval,
        test_max_n_token=args.test_max_n_token,
        run_dir=args.run_dir,
        wandb_dir=args.wandb_dir,
        wandb_project=args.wandb_project,
        config_overrides=config_overrides,
        noise_p_mean=args.noise_p_mean,
        noise_p_std=args.noise_p_std,
        noise_p_mean_schedule=_parse_float_list(args.noise_p_mean_schedule),
        noise_p_mean_update_every=args.noise_p_mean_update_every,
    )
    trainer.init_all()
    trainer._log_main(
        f"[rank{trainer.rank}/{trainer.world_size}] init done "
        f"(distributed={trainer.is_distributed}, device={trainer.device}, "
        f"rank_seed={trainer.rank_seed})"
    )
    # Snapshot the resolved run config into run_dir (rank0 only) for reproducibility.
    if trainer.is_main and getattr(args, "run_dir", None):
        import json as _json, yaml as _yaml
        os.makedirs(args.run_dir, exist_ok=True)
        _cfg_path = os.path.join(args.run_dir, "config.yaml")
        _cfg = {"args": vars(args)}
        try:
            _cfg["config_overrides"] = config_overrides
        except Exception:
            pass
        # json round-trip (default=str) coerces any non-primitive to a YAML-safe form.
        _safe = _json.loads(_json.dumps(_cfg, default=str))
        with open(_cfg_path, "w") as _f:
            _yaml.safe_dump(_safe, _f, default_flow_style=False, sort_keys=True, allow_unicode=True)
        trainer._log_main(f"[config] wrote run config -> {_cfg_path}")
    if args.resume_ckpt:
        ok = trainer.try_load_checkpoint(args.resume_ckpt, load_params_only=args.load_params_only)
        trainer._log_main(
            f"[resume] {args.resume_ckpt} loaded={ok} "
            f"params_only={args.load_params_only} -> step {trainer.step}"
        )
        if not ok:
            raise FileNotFoundError(f"--resume_ckpt not found: {args.resume_ckpt}")
    try:
        trainer.train(
            args.num_steps,
            eval_interval=args.eval_interval,
            ckpt_interval=args.ckpt_interval,
            eval_batches=args.eval_batches,
        )
        if args.save_final_ckpt:
            path = trainer.save_checkpoint("")
            if path is not None:
                trainer._log_main(f"[rank{trainer.rank}] final ckpt -> {path}")
            if trainer.ema is not None:
                trainer.ema.apply_shadow()
                ema_path = trainer.save_checkpoint("_ema")
                trainer.ema.restore()
                if ema_path is not None:
                    trainer._log_main(f"[rank{trainer.rank}] final ema ckpt -> {ema_path}")
        trainer.barrier()
    finally:
        if trainer.wandb_run is not None:
            trainer.wandb_run.finish()
        trainer.shutdown()


if __name__ == "__main__":
    main()
