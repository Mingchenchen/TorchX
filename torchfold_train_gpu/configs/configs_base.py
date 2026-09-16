# torchfold base config.

from torchfold.config.extend_types import (
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
)

basic_configs = {
    "project": RequiredValue(str),
    "run_name": RequiredValue(str),
    "base_dir": RequiredValue(str),
    "eval_interval": RequiredValue(int),
    "log_interval": RequiredValue(int),
    "checkpoint_interval": -1,
    "eval_first": False,
    "iters_to_accumulate": 1,
    "train_confidence_only": False,
    "load_checkpoint_path": "",          # trained {step}.pt to resume/finetune from
    "load_ema_checkpoint_path": "",
    "load_params_only": True,
    "load_strict": True,
    "use_wandb": False,
    "seed": 42,
    "ema_decay": 0.999,
    "eval_ema_only": False,
    # which model architecture to build (key into configs_model_type.model_configs).
    "model_name": "torchfold_af3_default",
    # JAX dump directory (loaded by torchfold.weights.import_jax_weights_).
    "af3_params_dir": "./Alphafold3params",
    # Kernel backends (also overridable via env LAYERNORM_TYPE / TRIANGLE_*).
    # Default to portable torch path; set cuequivariance only if that package is installed.
    "layernorm_type": "fast_layernorm",
    "triangle_multiplicative": "torch",
    "triangle_attention": "torch",
}

data_configs = {
    "train_crop_size": 640,
    "test_max_n_token": 1024,
    "num_dl_workers": 4,
    "epoch_size": 10000,
    "train_lig_atom_rename": False,
    "train_shuffle_mols": False,
    "train_shuffle_sym_ids": False,
    "test_lig_atom_rename": False,
    "test_shuffle_mols": False,
    "test_shuffle_sym_ids": False,
}

optim_configs = {
    "lr": 1.8e-3,
    "lr_scheduler": "af3",
    "warmup_steps": 1000,
    "max_steps": RequiredValue(int),
    "decay_every_n_steps": 50000,
    "decay_factor": 0.95,
    "grad_clip_norm": 10.0,
    "adam": {"beta1": 0.9, "beta2": 0.95, "weight_decay": 1e-8,
             "lr": GlobalConfigValue("lr"), "use_adamw": True},
}

# ---- model architecture. configs["model"].* ----
model_configs = {
    "c_s": 384,
    "c_z": 128,
    "c_s_inputs": 449,
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 384,
    "n_blocks": 48,                      # pairformer blocks
    "max_atoms_per_token": 24,
    "sigma_data": 16.0,
    "diffusion_batch_size": 48,          # N noised samples / training step
    "diffusion_chunk_size": 4,           # sample-chunk for the diffusion fwd/bwd
    "model": {
        "N_cycle": 4,                    # recycles
        "num_msa": 1024,                 # MSA depth used by the trunk
        "esm": {"enable": False},
        "template_embedder": {"c": 64, "c_z": GlobalConfigValue("c_z"), "n_blocks": 2},
        "msa_module": {"c_m": 64, "c_z": GlobalConfigValue("c_z"),
                       "n_blocks": 4, "msa_chunk_size": ValueMaybeNone(2048),
                       "msa_max_size": 16384},
        "pairformer": {"n_blocks": GlobalConfigValue("n_blocks"),
                       "c_z": GlobalConfigValue("c_z"), "c_s": GlobalConfigValue("c_s"),
                       "n_heads": 16},
        "diffusion_module": {
            "c_token": 768, "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"), "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"), "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "atom_encoder": {"n_blocks": 3, "n_heads": 4},
            "transformer": {"n_blocks": 24, "n_heads": 16},
            "atom_decoder": {"n_blocks": 3, "n_heads": 4},
        },
        "confidence_head": {"c_z": GlobalConfigValue("c_z"), "c_s": GlobalConfigValue("c_s"),
                            "n_blocks": 4, "max_atoms_per_token": GlobalConfigValue("max_atoms_per_token")},
        "distogram_head": {"c_z": GlobalConfigValue("c_z"), "no_bins": 64},
    },
    "mini_rollout": {"N_step": 20, "N_sample": 1},     # confidence-head training rollout
    "sample_diffusion": {"N_step": 200, "N_sample": 5, "gamma0": 0.8, "gamma_min": 1.0,
                         "noise_scale_lambda": 1.003, "step_scale_eta": 1.5},
    "train_noise_sampler": {
        "p_mean": -1.2,
        "p_std": 1.5,
        "sigma_data": GlobalConfigValue("sigma_data"),
        # Optional curriculum (also via CLI --noise_p_mean_schedule /
        # --noise_p_mean_update_every). Example: every 10k steps
        # p_mean goes 0.5 -> 0.0 -> -0.5 -> -1.2.
        # "p_mean_schedule": [0.5, 0.0, -0.5, -1.2],
        # "update_every": 10000,
    },
}

# ---- loss weights (AF3 defaults; Trainer also exposes --loss_* CLI) ----
loss_configs = {
    "loss": {
        "diffusion_lddt_loss_dense": True,
        "diffusion_sparse_loss_enable": True,
        "weight": {
            "alpha_confidence": 1e-4,
            "alpha_pae": 1.0,
            "alpha_except_pae": 1.0,
            "alpha_diffusion": 4.0,
            "alpha_distogram": 3e-2,
            "alpha_bond": 1.0,
            "smooth_lddt": 1.0,
            "smooth_lddt_interface": 0.0,   # ft1 Ab-Ag sets 6.0
        },
        "diffusion": {"smooth_lddt_interface": {"radius": 5.0}},
    },
}

configs = {
    **basic_configs,
    **data_configs,
    **optim_configs,
    **model_configs,
    **loss_configs,
}
