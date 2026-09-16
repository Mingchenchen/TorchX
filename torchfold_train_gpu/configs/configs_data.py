import os
from copy import deepcopy

from torchfold.config.extend_types import GlobalConfigValue, ListValue

# Training / eval data root. Override with TORCHFOLD_ROOT_DIR.
TORCHFOLD_ROOT_DIR = os.environ.get("TORCHFOLD_ROOT_DIR", "train_data")

default_test_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
    },
    "cropping_configs": {
        "method_weights": [
            0.0,  # ContiguousCropping
            0.0,  # SpatialCropping
            1.0,  # SpatialInterfaceCropping
        ],
        "crop_size": -1,
    },
    "lig_atom_rename": GlobalConfigValue("test_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("test_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("test_shuffle_sym_ids"),
    "constraint": {
        "enable": False,
        "fix_seed": False,
    },
}

default_weighted_pdb_configs = {
    "sampler_configs": {
        "sampler_type": "weighted",
        "beta_dict": {
            "chain": 0.5,
            "interface": 1,
        },
        "alpha_dict": {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        },
        "force_recompute_weight": True,
    },
    "cropping_configs": {
        "method_weights": ListValue([0.2, 0.4, 0.4]),
        "crop_size": GlobalConfigValue("train_crop_size"),
    },
    "sample_weight": 0.5,
    "limits": -1,
    "lig_atom_rename": GlobalConfigValue("train_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("train_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("train_shuffle_sym_ids"),
    "constraint": {
        "enable": False,
        "fix_seed": False,
        "pocket": {
            "prob": 0.0,
            "size": 1 / 3,
            "spec_binder_chain": False,
            "max_distance_range": {"PP": ListValue([6, 20]), "LP": ListValue([6, 20])},
            "group": "complex",
            "distance_type": "center_atom",
        },
        "contact": {
            "prob": 0.0,
            "size": 1 / 3,
            "max_distance_range": {
                "PP": ListValue([6, 30]),
                "PL": ListValue([4, 10]),
            },
            "group": "complex",
            "distance_type": "center_atom",
        },
        "substructure": {
            "prob": 0.0,
            "size": 0.8,
            "mol_type_pairs": {
                "PP": 15,
                "PL": 10,
                "LP": 10,
            },
            "feature_type": "one_hot",
            "ratios": {
                "full": [0.0, 0.5, 1.0],
                "partial": 0.3,
            },
            "coord_noise_scale": 0.05,
            "spec_asym_id": False,
        },
        "contact_atom": {
            "prob": 0.0,
            "size": 1 / 3,
            "max_distance_range": {
                "PP": ListValue([2, 12]),
                "PL": ListValue([2, 8]),
            },
            "min_distance": -1,
            "group": "complex",
            "distance_type": "atom",
            "feature_type": "continuous",
        },
    },
}

data_configs = {
    "num_dl_workers": 8,
    "epoch_size": 10000,
    "train_ref_pos_augment": True,
    "test_ref_pos_augment": True,
    # Default train set; override via --data.train_sets
    "train_sets": ListValue(["sabdab"]),
    "train_sampler": {
        "train_sample_weights": ListValue([1.0]),
        "sampler_type": "weighted",
    },
    "test_sets": ListValue(["recentPDB_1536_sample384_0925", "posebusters_0925", "abag_2025"]),



    # -----------------------------------------------------------------------
    # Training dataset: weightedPDB pre-2021-09-30 (2026.01.01 data version)
    # -----------------------------------------------------------------------
    "weightedPDB_before210930_v20260101": {
        "base_info": {
            "mmcif_dir": os.path.join(TORCHFOLD_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                TORCHFOLD_ROOT_DIR, "mmcif_bioassembly"
            ),
            "indices_fpath": os.path.join(
                TORCHFOLD_ROOT_DIR,
                "indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,
            "max_release_date": "2021-09-30",
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # -----------------------------------------------------------------------
    # Eval / test datasets: evaluated by Trainer.evaluate() via test_sets list.
    # -----------------------------------------------------------------------

    "sabdab": {
        "base_info": {
            "mmcif_dir": os.path.join(TORCHFOLD_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(TORCHFOLD_ROOT_DIR, "mmcif_bioassembly"),
            "indices_fpath": os.path.join(
                TORCHFOLD_ROOT_DIR, "indices/sabdab_from_torchfold_indices.csv"
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,
            "max_release_date": "2025-01-01",
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        "template": {
            "enable_prot_template": True,
            "template_dropout_rate": 0.0,
            "prot_template_mmcif_dir": os.path.join(TORCHFOLD_ROOT_DIR, "mmcif"),
            "prot_template_cache_dir": "",
            "prot_template_raw_paths": ListValue(
                [os.path.join(TORCHFOLD_ROOT_DIR, "mmcif_msa_template")]
            ),
            "prot_seq_or_filename_to_templatedir_jsons": ListValue(
                [os.path.join(TORCHFOLD_ROOT_DIR, "common/seq_to_pdb_index.json")]
            ),
            "prot_indexing_methods": ListValue(["sequence"]),
            "release_dates_path": os.path.join(
                TORCHFOLD_ROOT_DIR, "common/release_date_cache.json"
            ),
            "obsolete_pdbs_path": os.path.join(
                TORCHFOLD_ROOT_DIR, "common/obsolete_to_successor.json"
            ),
            "kalign_binary_path": "bin/kalign",
        },
        "msa": {
            "prot_seq_or_filename_to_msadir_jsons": ListValue(
                [os.path.join(TORCHFOLD_ROOT_DIR, "common/seq_to_pdb_index.json")]
            ),
            "prot_msadir_raw_paths": ListValue(
                [os.path.join(TORCHFOLD_ROOT_DIR, "mmcif_msa_template")]
            ),
            "prot_pairing_dbs": ListValue(["pairing"]),
            "prot_non_pairing_dbs": ListValue(["non_pairing"]),
            "prot_indexing_methods": ListValue(["sequence"]),
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    # abag_2025 Ab-Ag eval set
    "abag_2025": {
        "base_info": {
            "mmcif_dir": "",
            "bioassembly_dict_dir": os.path.join(
                TORCHFOLD_ROOT_DIR, "benchmark/abag_2025/bioassembly"
            ),
            "indices_fpath": os.path.join(
                TORCHFOLD_ROOT_DIR, "benchmark/abag_2025/indices/abag_2025_indices.csv"
            ),
            "pdb_list": "",
            "max_n_token": GlobalConfigValue("test_max_n_token"),
            "sort_by_n_token": False,
            "group_by_pdb_id": True,
            "find_eval_chain_interface": True,
        },
        "msa": {
            "prot_seq_or_filename_to_msadir_jsons": ListValue(
                [os.path.join(
                    TORCHFOLD_ROOT_DIR, "benchmark/abag_2025/common/seq_to_pdb_index.json"
                )]
            ),
            "prot_msadir_raw_paths": ListValue(
                [os.path.join(
                    TORCHFOLD_ROOT_DIR, "benchmark/abag_2025/mmcif_msa_template"
                )]
            ),
            "prot_pairing_dbs": ListValue(["pairing"]),
            "prot_non_pairing_dbs": ListValue(["non_pairing-uniref100_hits-mmseqs_other_hits"]),
            "prot_indexing_methods": ListValue(["sequence"]),
        },
        "template": {"enable_prot_template": True},
        **deepcopy(default_test_configs),
    },
    # -----------------------------------------------------------------------
    # MSA config
    # -----------------------------------------------------------------------
    "msa": {
        "enable_prot_msa": True,
        "prot_seq_or_filename_to_msadir_jsons": ListValue(
            [os.path.join(TORCHFOLD_ROOT_DIR, "common/seq_to_pdb_index.json")]
        ),
        "prot_msadir_raw_paths": ListValue(
            [os.path.join(TORCHFOLD_ROOT_DIR, "mmcif_msa_template")]
        ),
        "prot_pairing_dbs": ListValue(["pairing"]),
        "prot_non_pairing_dbs": ListValue(["pairing-non_pairing"]),
        "prot_indexing_methods": ListValue(["sequence"]),
        "enable_rna_msa": False,
        "rna_seq_or_filename_to_msadir_jsons": ListValue(
            [os.path.join(TORCHFOLD_ROOT_DIR, "rna_msa/rna_sequence_to_pdb_chains.json")]
        ),
        "rna_msadir_raw_paths": ListValue(
            [os.path.join(TORCHFOLD_ROOT_DIR, "rna_msa/msas")]
        ),
        "rna_indexing_methods": ListValue(["sequence"]),
        "min_size": {
            "train": 1,
            "test": 1,
        },
        "max_size": {
            "train": 16384,
            "test": 16384,
        },
        "sample_cutoff": {
            "train": 16384,
            "test": 16384,
        },
    },
    # -----------------------------------------------------------------------
    # Template config
    # -----------------------------------------------------------------------
    "template": {
        "enable_prot_template": True,
        "template_dropout_rate": 0.0,
        "prot_template_mmcif_dir": os.path.join(TORCHFOLD_ROOT_DIR, "mmcif"),
        "prot_template_cache_dir": "",
        "prot_template_raw_paths": ListValue(
            [os.path.join(TORCHFOLD_ROOT_DIR, "mmcif_msa_template")]
        ),
        "prot_seq_or_filename_to_templatedir_jsons": ListValue(
            [os.path.join(TORCHFOLD_ROOT_DIR, "common/seq_to_pdb_index.json")]
        ),
        "prot_indexing_methods": ListValue(["sequence"]),
        "release_dates_path": os.path.join(
            TORCHFOLD_ROOT_DIR, "common/release_date_cache.json"
        ),
        "obsolete_pdbs_path": os.path.join(
            TORCHFOLD_ROOT_DIR, "common/obsolete_to_successor.json"
        ),
        "kalign_binary_path": "bin/kalign",
    },
    # -----------------------------------------------------------------------
    # CCD / PDB cluster files under common/
    # -----------------------------------------------------------------------
    "ccd_components_file": os.path.join(TORCHFOLD_ROOT_DIR, "common/components.cif"),
    "ccd_components_rdkit_mol_file": os.path.join(
        TORCHFOLD_ROOT_DIR, "common/components.cif.rdkit_mol.pkl"
    ),
    "obsolete_release_data_csv": os.path.join(
        TORCHFOLD_ROOT_DIR, "common/obsolete_release_date.csv"
    ),
    "pdb_cluster_file": os.path.join(
        TORCHFOLD_ROOT_DIR, "common/clusters-by-entity-40.txt"
    ),
}
