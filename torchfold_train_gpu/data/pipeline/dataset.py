import json
import os
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import biotite.structure as struc
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from torchfold.data.constants import (
    EvaluationChainInterface,
    PRO_STD_RESIDUES,
    RES_ATOMS_DICT,
    RNA_STD_RESIDUES,
    DNA_STD_RESIDUES,
    STD_RESIDUES,
    mmcif_restype_3to1,
)
from torchfold.data.constraint.constraint_featurizer import ConstraintFeatureGenerator
from torchfold.data.core.featurizer import Featurizer
from torchfold.data.msa.msa_featurizer import MSAFeaturizer
from torchfold.data.pipeline.data_pipeline import DataPipeline
from torchfold.data.template.template_featurizer import TemplateFeaturizer
from torchfold.data.tokenizer import TokenArray
from torchfold.data.utils import (
    data_type_transform,
    get_antibody_clusters,
    make_dummy_feature,
)
from torchfold.utils.cropping import CropData
from torchfold.utils.file_io import read_indices_csv
from torchfold.utils.logger import get_logger
from torchfold.utils.torch_utils import dict_to_tensor

logger = get_logger(__name__)


class DesignIncompatibleSampleError(ValueError):
    """Raised when an index row is valid generally but unsuitable for design pretraining."""


class BaseSingleDataset(Dataset):
    """
    dataset for a single data source
    data = self.__item__(idx)
    return a dict of features and labels
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        template_featurizer: Optional[TemplateFeaturizer] = None,
        name: str = None,
        **kwargs,
    ) -> None:
        super(BaseSingleDataset, self).__init__()

        # Configs
        self.mmcif_dir = mmcif_dir
        self.bioassembly_dict_dir = bioassembly_dict_dir
        self.indices_fpath = indices_fpath
        self.cropping_configs = cropping_configs
        self.name = name
        # General dataset configs
        self.ref_pos_augment = kwargs.get("ref_pos_augment", True)
        self.lig_atom_rename = kwargs.get("lig_atom_rename", False)
        self.reassign_continuous_chain_ids = kwargs.get(
            "reassign_continuous_chain_ids", False
        )
        self.shuffle_mols = kwargs.get("shuffle_mols", False)
        self.shuffle_sym_ids = kwargs.get("shuffle_sym_ids", False)

        # Typically used for test sets
        self.find_pocket = kwargs.get("find_pocket", False)
        self.find_all_pockets = kwargs.get("find_all_pockets", False)  # for dev
        self.find_eval_chain_interface = kwargs.get("find_eval_chain_interface", False)
        self.group_by_pdb_id = kwargs.get("group_by_pdb_id", False)  # for test set
        self.sort_by_n_token = kwargs.get("sort_by_n_token", False)

        # Typically used for training set
        self.random_sample_if_failed = kwargs.get("random_sample_if_failed", False)
        self.use_reference_chains_only = kwargs.get("use_reference_chains_only", False)
        self.is_distillation = kwargs.get("is_distillation", False)
        # 0710: WeightedPDB design pretraining needs crop samples to look like two-chain target/binder tasks.
        self.require_exact_two_chains_after_crop = kwargs.get(
            "require_exact_two_chains_after_crop", False
        )
        self.require_standard_protein_tokens_after_crop = kwargs.get(
            "require_standard_protein_tokens_after_crop", False
        )
        self.design_interface_contact_cutoff = float(
            kwargs.get("design_interface_contact_cutoff", 10.0)
        )
        # 0711: Generic target-binder pretraining should skip homo-oligomer symmetric interfaces.
        self.require_distinct_interface_entities = kwargs.get(
            "require_distinct_interface_entities", False
        )

        # Configs for data filters
        self.max_n_token = kwargs.get("max_n_token", -1)
        self.min_n_token = kwargs.get("min_n_token", -1)
        # release_date upper bound (lexicographic YYYY-MM-DD; "" disables).
        # Rows whose release_date is >= this date are dropped. Distilled datasets
        # use "9999-12-31" as a placeholder, so leaving the filter unset keeps them.
        self.max_release_date = kwargs.get("max_release_date", "")
        self.pdb_list = kwargs.get("pdb_list", None)
        if len(self.pdb_list) == 0:
            self.pdb_list = None
        # Used for removing rows in the indices list. Column names and excluded values are specified in this dict.
        self.exclusion_dict = kwargs.get("exclusion", {})
        # 0710: Include filters let design pretraining keep only interface/prot_prot rows without hardcoding a dataset.
        self.inclusion_dict = kwargs.get("inclusion", {})
        self.limits = kwargs.get(
            "limits", -1
        )  # Limit number of indices rows, mainly for test
        # 0713: Exhausted resampling should raise the real data issue instead of returning an undefined local.
        self.max_sample_retries = int(kwargs.get("max_sample_retries", 10))
        # Configs for constraint
        self.constraint = kwargs.get("constraint", {})
        if self.constraint.get("enable", False):
            logger.info(f"[{self.name}] constraint config: {self.constraint}")
            # Do not rely on new files for users who do not use constraint feature
            self.ab_top2_clusters = get_antibody_clusters()
            self.constraint_generator = ConstraintFeatureGenerator(
                self.constraint, self.ab_top2_clusters
            )

        self.error_dir = kwargs.get("error_dir", None)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

        self.msa_featurizer = msa_featurizer
        self.template_featurizer = template_featurizer

        # Read data
        self.indices_list = self.read_indices_list(indices_fpath)

    @staticmethod
    def read_pdb_list(pdb_list: Union[list, str]) -> Optional[list]:
        """
        Reads a list of PDB IDs from a file or directly from a list.

        Args:
            pdb_list: A list of PDB IDs or a file path containing PDB IDs.

        Returns:
            A list of PDB IDs if the input is valid, otherwise None.
        """
        if pdb_list is None:
            return None

        if isinstance(pdb_list, list):
            return pdb_list

        with open(pdb_list, "r") as f:
            pdb_filter_list = []
            for l in f.readlines():
                l = l.strip()
                if l:
                    pdb_filter_list.append(l)
        return pdb_filter_list

    def read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        """
        Reads and processes a list of indices from a CSV file.

        Args:
            indices_fpath: Path to the CSV file containing the indices.

        Returns:
            A DataFrame containing the processed indices.
        """
        indices_list = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        self.check_indices_list(indices_list, "initial loading")
        logger.info(f"#Rows in indices list: {num_data}")
        # Filter by pdb_list
        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")
            self.check_indices_list(indices_list, "pdb_list filtering")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"max_n_token ({self.max_n_token}) filtering"
            )

        if self.min_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) >= self.min_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by min_n_token ({self.min_n_token})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"min_n_token ({self.min_n_token}) filtering"
            )

        # Filter by release_date upper bound (e.g. "2025-01-01" keeps rows
        # released strictly before 2025-01-01). Lexicographic compare is correct
        # for the canonical YYYY-MM-DD string format used in indices.csv.
        if self.max_release_date:
            valid_mask = indices_list["release_date"].astype(str) < self.max_release_date
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed by release_date] #Rows: {len(removed_list)}")
            logger.info(f"[removed by release_date] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by max_release_date (<{self.max_release_date})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"max_release_date (<{self.max_release_date}) filtering"
            )

        # Filter by inclusion_dict
        for col_name, inclusion_list in self.inclusion_dict.items():
            if len(inclusion_list) == 0:
                continue
            cols = col_name.split("|")
            inclusion_set = {tuple(incl.split("|")) for incl in inclusion_list}

            def is_included(row):
                return tuple(str(row[col]) for col in cols) in inclusion_set

            valid_mask = indices_list.apply(is_included, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Included by {col_name} -- {inclusion_list}] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"inclusion_dict ({col_name}) filtering"
            )

        if self.require_distinct_interface_entities:
            interface_mask = indices_list["type"].astype(str) == "interface"
            ent1 = pd.to_numeric(indices_list["entity_1_id"], errors="coerce")
            ent2 = pd.to_numeric(indices_list["entity_2_id"], errors="coerce")
            distinct_entity_mask = ~interface_mask | (
                ent1.notna() & ent2.notna() & (ent1 != ent2)
            )
            indices_list = indices_list[distinct_entity_mask].reset_index(drop=True)
            logger.info(
                "[filtered by distinct interface entities] "
                f"#Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, "require_distinct_interface_entities filtering"
            )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"exclusion_dict ({col_name}) filtering"
            )
        self.print_data_stats(indices_list)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            indices_list = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

        if self.sort_by_n_token:
            # Sort the dataset in a descending order, so that if OOM it will raise Error at an early stage.
            if self.group_by_pdb_id:
                indices_list = sorted(
                    indices_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )
            else:
                indices_list = indices_list.sort_values(
                    by="num_tokens", key=lambda x: x.astype(int), ascending=False
                ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            if self.group_by_pdb_id:
                indices_list = [
                    df
                    for df in indices_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]
            else:
                # Vectorized equivalent of:
                #   indices_list[indices_list["eval_type"].apply(lambda x: x in EvaluationChainInterface)]
                # .isin() checks each element of the Series against the set, same semantics as the lambda.
                indices_list = indices_list[
                    indices_list["eval_type"].isin(EvaluationChainInterface)
                ]
            self.check_indices_list(indices_list, "find_eval_chain_interface filtering")
        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
            self.check_indices_list(indices_list, "limits filtering")
        return indices_list

    def check_indices_list(self, indices_list, step_name: str = ""):
        if len(indices_list) == 0:
            msg = "After filtering, the dataset is empty."
            if step_name:
                msg = f"After {step_name}, the dataset is empty."
            raise ValueError(msg)

    def print_data_stats(self, df: pd.DataFrame) -> None:
        """
        Prints statistics about the dataset, including the distribution of molecular group types.

        Args:
            df: A DataFrame containing the indices list.
        """
        if self.name:
            logger.info("-" * 10 + f" Dataset {self.name}" + "-" * 10)
        col1 = df["mol_1_type"].astype(str)
        col2 = df["mol_2_type"].astype(str).str.replace("nan", "intra", regex=False)
        lo = np.where(col1.values <= col2.values, col1.values, col2.values)
        hi = np.where(col1.values <= col2.values, col2.values, col1.values)
        df["mol_group_type"] = np.char.add(np.char.add(lo, "_"), hi)

        group_size_dict = dict(df["mol_group_type"].value_counts())
        for i, n_i in group_size_dict.items():
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i*100/len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(f"{i}: {n_i}/{n_cluster}({round(n_i*100/n_cluster, 2)}%)")
            logger.info("-" * 30)

        logger.info(f"Final pdb ids: {len(set(df.pdb_id.tolist()))}")
        logger.info("-" * 30)

    def __len__(self) -> int:
        return len(self.indices_list)

    def save_error_data(self, idx: int, error_message: str) -> None:
        """
        Saves the error data for a specific index to a JSON file in the error directory.

        Args:
            idx: The index of the data sample that caused the error.
            error_message: The error message to be saved.
        """
        if self.error_dir is not None:
            sample_indice = self._get_sample_indice(idx=idx)
            data = sample_indice.to_dict()
            data["error"] = error_message

            filename = f"{sample_indice.pdb_id}-{sample_indice.chain_1_id}-{sample_indice.chain_2_id}.json"
            fpath = os.path.join(self.error_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    json.dump(data, f)

    @staticmethod
    def _clean_chain_id(chain_id: Any) -> str:
        # 0713: Role-aware AbAg rows may store optional chains as blank/NaN values.
        if chain_id is None or pd.isna(chain_id):
            return ""
        chain_id = str(chain_id).strip()
        if chain_id.lower() in {"", "nan", "none", "<na>"}:
            return ""
        return chain_id

    def _get_reference_chain_ids(self, sample_indice: pd.Series) -> list[str]:
        # 0713: Keep the default two-chain index behavior while allowing dataset subclasses to override roles.
        ref_chain_ids = [
            self._clean_chain_id(sample_indice.chain_1_id),
            self._clean_chain_id(sample_indice.chain_2_id),
        ]
        if sample_indice.type == "chain":
            ref_chain_ids.pop(-1)
        return [chain_id for chain_id in ref_chain_ids if chain_id]

    @staticmethod
    def _has_atom_annotation(atom_array: AtomArray, annotation_name: str) -> bool:
        try:
            if annotation_name not in atom_array.get_annotation_categories():
                return False
        except Exception:
            if not hasattr(atom_array, annotation_name):
                return False
        try:
            return len(getattr(atom_array, annotation_name)) == len(atom_array)
        except Exception:
            return True

    def _set_atom_annotation_if_missing(
        self, atom_array: AtomArray, annotation_name: str, values: Any
    ) -> bool:
        if self._has_atom_annotation(atom_array, annotation_name):
            return False
        atom_array.set_annotation(annotation_name, values)
        return True

    def _ensure_bonds(self, atom_array: AtomArray) -> None:
        if atom_array.bonds is not None:
            return
        # 0713: External AbAg pkls may omit bonds; an empty BondList keeps feature code schema-compatible.
        atom_array.bonds = struc.BondList(
            len(atom_array), np.empty((0, 3), dtype=np.uint32)
        )

    def _ensure_label_annotations(self, atom_array: AtomArray) -> None:
        chain_ids = np.asarray([str(chain_id) for chain_id in atom_array.chain_id])
        if not self._has_atom_annotation(atom_array, "label_asym_id"):
            # 0713: MSA/template role metadata needs label asym IDs; chain_id is the safest fallback.
            atom_array.set_annotation("label_asym_id", chain_ids.copy())
        if not self._has_atom_annotation(atom_array, "label_entity_id"):
            chain_order = list(dict.fromkeys(chain_ids.tolist()))
            chain_to_entity = {
                chain_id: str(idx + 1) for idx, chain_id in enumerate(chain_order)
            }
            atom_array.set_annotation(
                "label_entity_id",
                np.asarray(
                    [chain_to_entity[chain_id] for chain_id in chain_ids],
                    dtype=object,
                ),
            )

    def _ensure_integer_chain_annotations(self, atom_array: AtomArray) -> None:
        if (
            self._has_atom_annotation(atom_array, "asym_id_int")
            and self._has_atom_annotation(atom_array, "entity_id_int")
            and self._has_atom_annotation(atom_array, "sym_id_int")
        ):
            return
        chain_ids = np.asarray([str(chain_id) for chain_id in atom_array.chain_id])
        if self._has_atom_annotation(atom_array, "label_entity_id"):
            entity_keys = np.asarray(
                [str(entity_id) for entity_id in atom_array.label_entity_id]
            )
        else:
            entity_keys = chain_ids
        chain_order = list(dict.fromkeys(chain_ids.tolist()))
        entity_order = list(dict.fromkeys(entity_keys.tolist()))
        entity_to_int = {entity_id: idx for idx, entity_id in enumerate(entity_order)}
        entity_seen_count: dict[str, int] = {}
        asym_ids = np.zeros(len(atom_array), dtype=np.int64)
        entity_ids = np.zeros(len(atom_array), dtype=np.int64)
        sym_ids = np.zeros(len(atom_array), dtype=np.int64)
        for asym_id, chain_id in enumerate(chain_order):
            mask = chain_ids == chain_id
            entity_id = entity_keys[np.nonzero(mask)[0][0]]
            asym_ids[mask] = asym_id
            entity_ids[mask] = entity_to_int[entity_id]
            sym_ids[mask] = entity_seen_count.get(entity_id, 0)
            entity_seen_count[entity_id] = entity_seen_count.get(entity_id, 0) + 1
        # 0713: Custom AbAg pkls may skip TorchFold parser integer chain IDs used by crop/features.
        self._set_atom_annotation_if_missing(atom_array, "asym_id_int", asym_ids)
        self._set_atom_annotation_if_missing(atom_array, "entity_id_int", entity_ids)
        self._set_atom_annotation_if_missing(atom_array, "sym_id_int", sym_ids)

    def _ensure_molecule_annotations(self, atom_array: AtomArray) -> None:
        if (
            self._has_atom_annotation(atom_array, "mol_id")
            and self._has_atom_annotation(atom_array, "entity_mol_id")
            and self._has_atom_annotation(atom_array, "mol_atom_index")
        ):
            return
        if self._has_atom_annotation(atom_array, "mol_id"):
            mol_ids = np.asarray(atom_array.mol_id, dtype=np.int64)
        else:
            chain_ids = np.asarray([str(chain_id) for chain_id in atom_array.chain_id])
            chain_order = list(dict.fromkeys(chain_ids.tolist()))
            chain_to_mol_id = {chain_id: idx for idx, chain_id in enumerate(chain_order)}
            mol_ids = np.asarray(
                [chain_to_mol_id[chain_id] for chain_id in chain_ids], dtype=np.int64
            )
        if self._has_atom_annotation(atom_array, "label_entity_id"):
            entity_keys = np.asarray(
                [str(entity_id) for entity_id in atom_array.label_entity_id]
            )
        else:
            entity_keys = np.asarray([str(mol_id) for mol_id in mol_ids])
        entity_order = list(dict.fromkeys(entity_keys.tolist()))
        entity_to_mol_id = {entity_id: idx for idx, entity_id in enumerate(entity_order)}
        entity_mol_ids = np.asarray(
            [entity_to_mol_id[entity_id] for entity_id in entity_keys], dtype=np.int64
        )
        mol_atom_index = np.zeros(len(atom_array), dtype=np.int64)
        for mol_id in np.unique(mol_ids):
            mask = mol_ids == mol_id
            mol_atom_index[mask] = np.arange(int(mask.sum()), dtype=np.int64)
        # 0713: Chain permutation requires molecule IDs even for parser-external pkl files.
        self._set_atom_annotation_if_missing(atom_array, "mol_id", mol_ids)
        self._set_atom_annotation_if_missing(atom_array, "entity_mol_id", entity_mol_ids)
        self._set_atom_annotation_if_missing(atom_array, "mol_atom_index", mol_atom_index)

    def _ensure_mol_type_annotations(self, atom_array: AtomArray) -> None:
        if not self._has_atom_annotation(atom_array, "mol_type"):
            res_names = np.asarray([str(res_name) for res_name in atom_array.res_name])
            mol_types = np.full(len(atom_array), "ligand", dtype=object)
            mol_types[np.isin(res_names, list(PRO_STD_RESIDUES.keys()))] = "protein"
            mol_types[np.isin(res_names, list(RNA_STD_RESIDUES.keys()))] = "rna"
            mol_types[np.isin(res_names, list(DNA_STD_RESIDUES.keys()))] = "dna"
            # 0713: Infer minimal molecule type labels when external pkls lack parser annotations.
            atom_array.set_annotation("mol_type", mol_types)

        if not self._has_atom_annotation(atom_array, "chain_mol_type"):
            chain_mol_types = np.empty(len(atom_array), dtype=object)
            chain_ids = np.asarray([str(chain_id) for chain_id in atom_array.chain_id])
            for chain_id in dict.fromkeys(chain_ids.tolist()):
                mask = chain_ids == chain_id
                values, counts = np.unique(atom_array.mol_type[mask], return_counts=True)
                chain_mol_types[mask] = values[np.argmax(counts)]
            atom_array.set_annotation("chain_mol_type", chain_mol_types)

        for mol_type in ("protein", "ligand", "dna", "rna"):
            annotation_name = f"is_{mol_type}"
            if not self._has_atom_annotation(atom_array, annotation_name):
                atom_array.set_annotation(
                    annotation_name,
                    (atom_array.chain_mol_type == mol_type).astype(np.int64),
                )

    def _ensure_token_atom_annotations(
        self, atom_array: AtomArray, token_array: Optional[TokenArray]
    ) -> None:
        if token_array is not None and not self._has_atom_annotation(
            atom_array, "centre_atom_mask"
        ):
            centre_atom_mask = np.zeros(len(atom_array), dtype=np.int64)
            centre_atom_indices = token_array.get_annotation("centre_atom_index")
            centre_atom_mask[np.asarray(centre_atom_indices, dtype=np.int64)] = 1
            # 0713: Tokenized external pkls already know centers; reuse them to restore atom masks.
            atom_array.set_annotation("centre_atom_mask", centre_atom_mask)

        if not self._has_atom_annotation(atom_array, "centre_atom_mask"):
            centre_atom_mask = np.zeros(len(atom_array), dtype=np.int64)
            residue_starts = struc.get_residue_starts(
                atom_array, add_exclusive_stop=True
            )
            for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
                atom_names = atom_array.atom_name[start:stop].tolist()
                if "CA" in atom_names:
                    centre_atom_mask[start + atom_names.index("CA")] = 1
                elif "C1'" in atom_names:
                    centre_atom_mask[start + atom_names.index("C1'")] = 1
                else:
                    centre_atom_mask[start] = 1
            atom_array.set_annotation("centre_atom_mask", centre_atom_mask)

        if not self._has_atom_annotation(atom_array, "plddt_m_rep_atom_mask"):
            atom_array.set_annotation(
                "plddt_m_rep_atom_mask", atom_array.centre_atom_mask.copy()
            )
        if not self._has_atom_annotation(atom_array, "distogram_rep_atom_mask"):
            distogram_mask = np.zeros(len(atom_array), dtype=np.int64)
            residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
            for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
                atom_names = atom_array.atom_name[start:stop].tolist()
                res_name = str(atom_array.res_name[start])
                chosen = None
                if res_name == "GLY" and "CA" in atom_names:
                    chosen = "CA"
                elif res_name in PRO_STD_RESIDUES and "CB" in atom_names:
                    chosen = "CB"
                elif "CA" in atom_names:
                    chosen = "CA"
                elif "C1'" in atom_names:
                    chosen = "C1'"
                if chosen is not None:
                    distogram_mask[start + atom_names.index(chosen)] = 1
                else:
                    centre_indices = np.nonzero(atom_array.centre_atom_mask[start:stop])[0]
                    fallback_idx = (
                        start + int(centre_indices[0])
                        if len(centre_indices)
                        else start
                    )
                    distogram_mask[fallback_idx] = 1
            atom_array.set_annotation("distogram_rep_atom_mask", distogram_mask)

        if not self._has_atom_annotation(atom_array, "modified_res_mask"):
            modified_res_mask = (
                (~np.isin(atom_array.res_name, list(STD_RESIDUES.keys())))
                & (atom_array.mol_type != "ligand")
            ).astype(np.int64)
            atom_array.set_annotation("modified_res_mask", modified_res_mask)

        if not self._has_atom_annotation(atom_array, "tokatom_idx"):
            tokatom_idx = []
            for res_name, atom_name, mol_type in zip(
                atom_array.res_name, atom_array.atom_name, atom_array.mol_type
            ):
                atom_name_position = RES_ATOMS_DICT.get(str(res_name), {})
                if str(mol_type) == "ligand":
                    tokatom_idx.append(0)
                else:
                    tokatom_idx.append(atom_name_position.get(str(atom_name), 0))
            atom_array.set_annotation(
                "tokatom_idx", np.asarray(tokatom_idx, dtype=np.int64)
            )

    def _ensure_canonical_residue_names(self, atom_array: AtomArray) -> None:
        if self._has_atom_annotation(atom_array, "cano_seq_resname"):
            return
        cano_seq_resname = []
        for res_name, mol_type in zip(atom_array.res_name, atom_array.mol_type):
            res_name = str(res_name)
            mol_type = str(mol_type)
            if res_name in STD_RESIDUES:
                cano_seq_resname.append(res_name)
            elif mol_type == "protein":
                cano_seq_resname.append("UNK")
            elif mol_type == "dna":
                cano_seq_resname.append("DN")
            elif mol_type == "rna":
                cano_seq_resname.append("N")
            else:
                cano_seq_resname.append("UNK")
        # 0713: Featurizer consumes canonical residue names, while external pkls may only have res_name.
        atom_array.set_annotation(
            "cano_seq_resname", np.asarray(cano_seq_resname, dtype=object)
        )

    def _ensure_ref_space_uid(self, atom_array: AtomArray) -> None:
        if self._has_atom_annotation(atom_array, "ref_space_uid"):
            return
        keys = list(zip(atom_array.chain_id.astype(str), atom_array.res_id.astype(str)))
        mapping: dict[tuple[str, str], int] = {}
        ref_space_uid = []
        for key in keys:
            if key not in mapping:
                mapping[key] = len(mapping)
            ref_space_uid.append(mapping[key])
        # 0713: Reference features and atom permutation group atoms by residue via ref_space_uid.
        atom_array.set_annotation("ref_space_uid", np.asarray(ref_space_uid, dtype=np.int64))

    def _ensure_reference_features(self, atom_array: AtomArray) -> None:
        has_ref_pos = self._has_atom_annotation(atom_array, "ref_pos")
        if not has_ref_pos:
            # 0713: Avoid leaking native coordinates when parser-style ideal reference conformers are absent.
            atom_array.set_annotation("ref_pos", np.zeros((len(atom_array), 3), dtype=np.float32))
        if not self._has_atom_annotation(atom_array, "ref_mask"):
            ref_mask = (
                np.isfinite(atom_array.ref_pos).all(axis=-1).astype(np.int64)
                if has_ref_pos
                else np.zeros(len(atom_array), dtype=np.int64)
            )
            atom_array.set_annotation("ref_mask", ref_mask)
        if not self._has_atom_annotation(atom_array, "ref_charge"):
            atom_array.set_annotation(
                "ref_charge", np.zeros(len(atom_array), dtype=np.int64)
            )

    def _ensure_res_perm(self, atom_array: AtomArray) -> None:
        if self._has_atom_annotation(atom_array, "res_perm"):
            return
        residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        res_perm = []
        for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
            res_perm.extend([str(i) for i in range(stop - start)])
        # 0713: Missing CCD symmetry info falls back to identity atom permutation per residue.
        atom_array.set_annotation("res_perm", np.asarray(res_perm, dtype=object))

    def _ensure_required_atom_annotations(
        self, bioassembly_dict: dict[str, Any]
    ) -> None:
        atom_array = bioassembly_dict["atom_array"]
        token_array = bioassembly_dict.get("token_array")
        self._ensure_bonds(atom_array)
        self._ensure_label_annotations(atom_array)
        if not self._has_atom_annotation(atom_array, "is_resolved"):
            coord = np.asarray(atom_array.coord)
            is_resolved = np.isfinite(coord).all(axis=-1)
            if is_resolved.shape[0] != len(atom_array):
                is_resolved = np.ones(len(atom_array), dtype=bool)
            # 0713: External AbAg pkls may omit parser-added is_resolved; downstream labels/crop require it.
            atom_array.set_annotation("is_resolved", is_resolved.astype(bool))
            atom_array.coord[~is_resolved] = 0.0
        self._ensure_integer_chain_annotations(atom_array)
        self._ensure_mol_type_annotations(atom_array)
        self._ensure_molecule_annotations(atom_array)
        self._ensure_token_atom_annotations(atom_array, token_array)
        self._ensure_canonical_residue_names(atom_array)
        self._ensure_ref_space_uid(atom_array)
        self._ensure_reference_features(atom_array)
        self._ensure_res_perm(atom_array)

    def _filter_bioassembly_to_chains(
        self, bioassembly_dict: dict[str, Any], ref_chain_ids: list[str]
    ) -> None:
        # 0713: Centralize reference-chain filtering so AbAgDesignDataset can keep antigen+binder role chains.
        if len(ref_chain_ids) == 0:
            raise DesignIncompatibleSampleError("No reference chains were provided")

        token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
            "centre_atom_index"
        )
        centre_atoms = bioassembly_dict["atom_array"][token_centre_atom_indices]
        token_chain_id = np.asarray([str(chain_id) for chain_id in centre_atoms.chain_id])
        missing_chains = sorted(set(ref_chain_ids) - set(token_chain_id.tolist()))
        if missing_chains:
            raise DesignIncompatibleSampleError(
                f"Reference chains not found in atom_array: {missing_chains}"
            )

        is_ref_chain = np.isin(token_chain_id, ref_chain_ids)
        (
            bioassembly_dict["token_array"],
            bioassembly_dict["atom_array"],
        ) = CropData.select_by_token_indices(
            token_array=bioassembly_dict["token_array"],
            atom_array=bioassembly_dict["atom_array"],
            selected_token_indices=np.arange(len(is_ref_chain))[is_ref_chain],
        )

    def __getitem__(self, idx: int):
        """
        Retrieves a data sample by processing the given index.
        If an error occurs, it attempts to handle it by either saving the error data or randomly sampling another index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A dictionary containing the processed data sample.
        """
        last_error_message = ""
        for _ in range(self.max_sample_retries):
            try:
                data = self.process_one(idx)
                return data
            except Exception as e:
                error_message = f"{e} at idx {idx}:\n{traceback.format_exc()}"
                last_error_message = error_message
                self.save_error_data(idx, error_message)

                if self.random_sample_if_failed:
                    if isinstance(e, DesignIncompatibleSampleError):
                        # 0710: Expected weightedPDB design filtering misses should not flood logs with tracebacks.
                        logger.warning(f"[skip design-incompatible data {idx}] {e}")
                    else:
                        logger.exception(f"[skip data {idx}] {error_message}")
                    # Random sample an index
                    idx = random.choice(range(len(self.indices_list)))
                    continue
                else:
                    raise Exception(e)
        # 0713: Make repeated data failures actionable for DataLoader workers.
        raise RuntimeError(
            f"Failed to fetch a valid sample after {self.max_sample_retries} attempts. "
            f"Last error:\n{last_error_message}"
        )

    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sample_indice = self._get_sample_indice(idx=idx)
        if self.bioassembly_dict_dir is not None:
            bioassembly_dict_fpath = os.path.join(
                self.bioassembly_dict_dir, sample_indice.pdb_id + ".pkl.gz"
            )
        else:
            bioassembly_dict_fpath = None

        bioassembly_dict = DataPipeline.get_data_bioassembly(
            bioassembly_dict_fpath=bioassembly_dict_fpath
        )
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id

        entity_pair_to_chain_pairs = bioassembly_dict.get("entity_pair_to_chain_pairs")
        if sample_indice["type"] == "interface" and entity_pair_to_chain_pairs:
            entity1 = sample_indice.entity_1_id
            entity2 = sample_indice.entity_2_id
            chain_pairs = entity_pair_to_chain_pairs.get((entity1, entity2))
            chain1, chain2 = random.choice(chain_pairs)
            new_sample_indice = deepcopy(sample_indice)
            new_sample_indice["chain_1_id"] = chain1
            new_sample_indice["chain_2_id"] = chain2
            return new_sample_indice, bioassembly_dict, bioassembly_dict_fpath
        else:
            return sample_indice, bioassembly_dict, bioassembly_dict_fpath

    @staticmethod
    def _reassign_atom_array_chain_id(atom_array: AtomArray):
        """
        In experiments conducted to observe overfitting effects using training sets,
        the pre-stored AtomArray in the training set may experience issues with discontinuous chain IDs due to filtering.
        Consequently, a temporary patch has been implemented to resolve this issue.

        e.g. 3x6u asym_id_int = [0, 1, 2, ... 18, 20] -> reassigned_asym_id_int [0, 1, 2, ..., 18, 19]
        """

        def _get_contiguous_array(array):
            array_uniq = np.sort(np.unique(array))
            return np.searchsorted(array_uniq, array)

        atom_array.asym_id_int = _get_contiguous_array(atom_array.asym_id_int)
        atom_array.entity_id_int = _get_contiguous_array(atom_array.entity_id_int)
        atom_array.sym_id_int = _get_contiguous_array(atom_array.sym_id_int)
        return atom_array

    @staticmethod
    def _shuffle_array_based_on_mol_id(token_array: TokenArray, atom_array: AtomArray):
        """
        Shuffle both token_array and atom_array.
        Atoms/tokens with the same mol_id will be shuffled as a integrated component.
        """

        # Get token mol_id
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        token_mol_id = atom_array[centre_atom_indices].mol_id

        # Get unique molecule IDs and shuffle them in place
        shuffled_mol_ids = np.unique(token_mol_id).copy()
        np.random.shuffle(shuffled_mol_ids)

        # Get shuffled token indices
        original_token_indices = np.arange(len(token_mol_id))
        shuffled_token_indices = []
        for mol_id in shuffled_mol_ids:
            mol_token_indices = original_token_indices[token_mol_id == mol_id]
            shuffled_token_indices.append(mol_token_indices)
        shuffled_token_indices = np.concatenate(shuffled_token_indices)

        # Get shuffled token and atom array
        # Use `CropData.select_by_token_indices` to shuffle safely
        (
            token_array,
            atom_array,
        ) = CropData.select_by_token_indices(
            token_array=token_array,
            atom_array=atom_array,
            selected_token_indices=shuffled_token_indices,
        )

        return token_array, atom_array

    @staticmethod
    def _assign_random_sym_id(atom_array: AtomArray):
        """
        Assign random sym_id for chains of the same entity_id
        e.g.
        when entity_id = 0
            sym_id_int = [0, 1, 2] -> random_sym_id_int = [2, 0, 1]
        when entity_id = 1
            sym_id_int = [0, 1, 2, 3] -> random_sym_id_int = [3, 0, 1, 2]
        """

        def _shuffle(x):
            x_unique = np.sort(np.unique(x))
            x_shuffled = x_unique.copy()
            np.random.shuffle(x_shuffled)  # shuffle in-place
            indices = np.searchsorted(x_unique, x)
            return x_shuffled[indices].copy()

        for entity_id in np.unique(atom_array.label_entity_id):
            mask = atom_array.label_entity_id == entity_id
            atom_array.sym_id_int[mask] = _shuffle(atom_array.sym_id_int[mask])
        return atom_array

    def process_one(
        self, idx: int, return_atom_token_array: bool = False
    ) -> dict[str, dict]:
        """
        Processes a single data sample by retrieving bioassembly data, applying various transformations, and cropping the data.
        It then extracts features and labels, and optionally returns the processed atom and token arrays.

        Args:
            idx: The index of the data sample to process.
            return_atom_token_array: Whether to return the processed atom and token arrays.

        Returns:
            A dict containing the input features, labels, basic_info and optionally the processed atom and token arrays.
        """

        (
            sample_indice,
            bioassembly_dict,
            bioassembly_dict_fpath,
        ) = self._get_bioassembly_data(idx=idx)

        self._ensure_required_atom_annotations(bioassembly_dict)

        if self.use_reference_chains_only:
            # 0713: Use the dataset-specific reference-chain hook for AbAg role-chain filtering.
            sample_indice, ref_chain_ids = self._resolve_reference_chain_ids(
                sample_indice=sample_indice,
                bioassembly_dict=bioassembly_dict,
                ref_chain_ids=self._get_reference_chain_ids(sample_indice),
            )
            self._filter_bioassembly_to_chains(
                bioassembly_dict=bioassembly_dict,
                ref_chain_ids=ref_chain_ids,
            )

        if self.shuffle_mols:
            (
                bioassembly_dict["token_array"],
                bioassembly_dict["atom_array"],
            ) = self._shuffle_array_based_on_mol_id(
                token_array=bioassembly_dict["token_array"],
                atom_array=bioassembly_dict["atom_array"],
            )

        if self.shuffle_sym_ids:
            bioassembly_dict["atom_array"] = self._assign_random_sym_id(
                bioassembly_dict["atom_array"]
            )

        if self.reassign_continuous_chain_ids:
            bioassembly_dict["atom_array"] = self._reassign_atom_array_chain_id(
                bioassembly_dict["atom_array"]
            )

        if self.require_standard_protein_tokens_after_crop:
            standard_token_mask = self._standard_protein_token_mask(
                bioassembly_dict["token_array"],
                bioassembly_dict["atom_array"],
            )
            if not bool(np.all(standard_token_mask)):
                # 0710: Drop non-design tokens before crop instead of noisy resampling on UNK/modified residues.
                (
                    bioassembly_dict["token_array"],
                    bioassembly_dict["atom_array"],
                ) = CropData.select_by_token_indices(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                    selected_token_indices=np.arange(len(standard_token_mask))[
                        standard_token_mask
                    ],
                )
            if len(bioassembly_dict["token_array"]) == 0:
                raise DesignIncompatibleSampleError(
                    "No standard protein tokens remain after filtering"
                )

        if self.require_exact_two_chains_after_crop:
            self._assert_reference_interface_ready(sample_indice, bioassembly_dict)

        max_entity_mol_id = bioassembly_dict["atom_array"].entity_mol_id.max()

        # Crop
        (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
        ) = self.crop(
            sample_indice=sample_indice,
            bioassembly_dict=bioassembly_dict,
            **self.cropping_configs,
        )
        if (
            self.require_exact_two_chains_after_crop
            or self.require_standard_protein_tokens_after_crop
        ):
            # 0710: Fail fast so random_sample_if_failed can resample non-design-like crops.
            self._assert_design_compatible_crop(cropped_token_array, cropped_atom_array)

        feat, label, label_full = self.get_feature_and_label(
            idx=idx,
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            msa_features=cropped_msa_features,
            template_features=cropped_template_features,
            full_atom_array=bioassembly_dict["atom_array"],
            is_spatial_crop="spatial" in crop_method.lower(),
            max_entity_mol_id=max_entity_mol_id,
            sample_indice=sample_indice,
        )

        # Basic info, e.g. dimension related items
        basic_info = {
            "pdb_id": (
                bioassembly_dict["pdb_id"]
                if self.is_distillation is False
                else sample_indice["pdb_id"]
            ),
            "N_asym": torch.tensor([len(torch.unique(feat["asym_id"]))]),
            "N_token": torch.tensor([feat["token_index"].shape[0]]),
            "N_atom": torch.tensor([feat["atom_to_token_idx"].shape[0]]),
            "N_msa": torch.tensor([feat["msa"].shape[0]]),
            "bioassembly_dict_fpath": bioassembly_dict_fpath,
            "N_msa_prot_pair": torch.tensor([feat["prot_pair_num_alignments"]]),
            "N_msa_prot_unpair": torch.tensor([feat["prot_unpair_num_alignments"]]),
            "N_msa_rna_pair": torch.tensor([feat["rna_pair_num_alignments"]]),
            "N_msa_rna_unpair": torch.tensor([feat["rna_unpair_num_alignments"]]),
        }

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))
            basic_info[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            basic_info[f"N_{abbr_type}_token"] = torch.tensor([n_token])

        # Add chain level chain_id
        asymn_id_to_chain_id = {
            atom.asym_id_int: atom.chain_id for atom in cropped_atom_array
        }
        chain_id_list = [
            asymn_id_to_chain_id[asymn_id_int]
            for asymn_id_int in sorted(asymn_id_to_chain_id.keys())
        ]
        basic_info["chain_id"] = chain_id_list

        data = {
            "input_feature_dict": feat,
            "label_dict": label,
            "label_full_dict": label_full,
            "basic": basic_info,
        }

        if return_atom_token_array:
            data["cropped_atom_array"] = cropped_atom_array
            data["cropped_token_array"] = cropped_token_array
        return data

    def _assert_design_compatible_crop(
        self, token_array: TokenArray, atom_array: AtomArray
    ) -> None:
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms = atom_array[centre_atom_indices]
        if self.require_exact_two_chains_after_crop:
            chain_ids = {str(chain_id) for chain_id in centre_atoms.chain_id}
            if len(chain_ids) != 2:
                raise DesignIncompatibleSampleError(
                    f"Expected exactly two chains after crop, found {sorted(chain_ids)}"
                )
        if not self.require_standard_protein_tokens_after_crop:
            return

        if "chain_mol_type" in centre_atoms.get_annotation_categories():
            mol_types = {str(mol_type) for mol_type in centre_atoms.chain_mol_type}
            if mol_types != {"protein"}:
                raise DesignIncompatibleSampleError(
                    f"Expected protein-only crop, found chain_mol_type={sorted(mol_types)}"
                )

        allowed_resnames = set(PRO_STD_RESIDUES) - {"UNK"}
        res_names = {str(res_name) for res_name in centre_atoms.res_name}
        nonstandard = sorted(res_names - allowed_resnames)
        if nonstandard:
            raise DesignIncompatibleSampleError(
                f"Expected standard amino-acid crop, found res_name={nonstandard}"
            )

    def _standard_protein_token_mask(
        self, token_array: TokenArray, atom_array: AtomArray
    ) -> np.ndarray:
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms = atom_array[centre_atom_indices]
        allowed_resnames = set(PRO_STD_RESIDUES) - {"UNK"}
        is_standard_residue = np.isin(centre_atoms.res_name, list(allowed_resnames))
        if "chain_mol_type" not in centre_atoms.get_annotation_categories():
            return is_standard_residue
        is_protein = np.asarray(centre_atoms.chain_mol_type) == "protein"
        return is_standard_residue & is_protein

    def _assert_reference_interface_ready(
        self, sample_indice: pd.Series, bioassembly_dict: dict[str, Any]
    ) -> None:
        if sample_indice.type != "interface":
            return
        ref_chain_ids = [
            str(sample_indice.chain_1_id).strip(),
            str(sample_indice.chain_2_id).strip(),
        ]
        token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
            "centre_atom_index"
        )
        centre_atoms = bioassembly_dict["atom_array"][token_centre_atom_indices]
        chain_ids = np.asarray([str(chain_id) for chain_id in centre_atoms.chain_id])
        resolved = (
            np.asarray(centre_atoms.is_resolved).astype(bool)
            if "is_resolved" in centre_atoms.get_annotation_categories()
            else np.ones(len(centre_atoms), dtype=bool)
        )

        coords_by_chain = {}
        for chain_id in ref_chain_ids:
            chain_mask = chain_ids == chain_id
            if not bool(chain_mask.any()):
                raise DesignIncompatibleSampleError(
                    f"Reference chain {chain_id} has no standard protein tokens"
                )
            coords = centre_atoms.coord[chain_mask & resolved]
            if coords.shape[0] == 0:
                raise DesignIncompatibleSampleError(
                    f"Reference chain {chain_id} has no resolved standard protein tokens"
                )
            coords_by_chain[chain_id] = coords

        # 0710: SpatialInterfaceCropping needs at least one resolved target-binder contact.
        min_dist = self._min_inter_chain_center_distance(
            coords_by_chain[ref_chain_ids[0]], coords_by_chain[ref_chain_ids[1]]
        )
        if not np.isfinite(min_dist) or min_dist >= self.design_interface_contact_cutoff:
            raise DesignIncompatibleSampleError(
                "Reference chains have no resolved standard-token contact "
                f"< {self.design_interface_contact_cutoff:g}A after filtering"
            )

    @staticmethod
    def _min_inter_chain_center_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
        min_dist = np.inf
        for start in range(0, coords_a.shape[0], 512):
            chunk = coords_a[start : start + 512]
            diff = chunk[:, None, :] - coords_b[None, :, :]
            chunk_min = float(np.sqrt(np.sum(diff * diff, axis=-1)).min())
            min_dist = min(min_dist, chunk_min)
        return min_dist

    def _resolve_reference_chain_ids(
        self,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        ref_chain_ids: list[str],
    ) -> tuple[pd.Series, list[str]]:
        # 0713: Base datasets already store chain IDs in the same namespace as atom_array.
        return sample_indice, ref_chain_ids

    def crop(
        self,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = True,
        drop_last: bool = True,
        remove_metal: bool = True,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crops the bioassembly data based on the specified configurations.

        Returns:
            A tuple containing the cropping method, cropped token array, cropped atom array,
                cropped MSA features, and cropped template features.
        """
        return DataPipeline.crop(
            one_sample=sample_indice,
            bioassembly_dict=bioassembly_dict,
            crop_size=crop_size,
            msa_featurizer=self.msa_featurizer,
            template_featurizer=self.template_featurizer,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

    def _get_sample_indice(self, idx: int) -> pd.Series:
        """
        Retrieves the sample indice for a given index. If the dataset is grouped by PDB ID, it returns the first row of the PDB-idx.
        Otherwise, it returns the row at the specified index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A pandas Series containing the sample indice.
        """
        if self.group_by_pdb_id:
            # Row-0 of PDB-idx
            sample_indice = self.indices_list[idx].iloc[0]
        else:
            sample_indice = self.indices_list.iloc[idx]
        return sample_indice

    def _get_pdb_indice(self, idx: int) -> pd.core.series.Series:
        if self.group_by_pdb_id:
            pdb_indice = self.indices_list[idx].copy()
        else:
            pdb_indice = self.indices_list.iloc[idx : idx + 1].copy()
        return pdb_indice

    def _get_eval_chain_interface_mask(
        self, idx: int, atom_array_chain_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Retrieves the evaluation chain/interface mask for a given index.

        Args:
            idx: The index of the data sample.
            atom_array_chain_id: An array containing the chain IDs of the atom array.

        Returns:
            A tuple containing the evaluation type, cluster ID, chain 1 mask, and chain 2 mask.
        """
        if self.group_by_pdb_id:
            df = self.indices_list[idx]
        else:
            df = self.indices_list.iloc[idx : idx + 1]

        # Only consider chain/interfaces defined in EvaluationChainInterface
        df = df[df["eval_type"].isin(EvaluationChainInterface)].copy()
        if len(df) < 1:
            raise ValueError("Cannot find a chain/interface for evaluation in the PDB.")

        def get_atom_mask(row):
            chain_1_mask = atom_array_chain_id == row["chain_1_id"]
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_mask = atom_array_chain_id == row["chain_2_id"]
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        if len(df) < 1:
            raise ValueError(
                "Cannot find a chain/interface for evaluation in the atom_array."
            )

        eval_type = np.array(df["eval_type"].tolist())
        cluster_id = np.array(df["cluster_id"].tolist())
        # [N_eval, N_atom]
        chain_1_mask = torch.stack(df["chain_1_mask"].tolist())
        # [N_eval, N_atom]
        chain_2_mask = torch.stack(df["chain_2_mask"].tolist())

        return eval_type, cluster_id, chain_1_mask, chain_2_mask

    def get_constraint_feature(
        self,
        idx,
        atom_array,
        token_array,
        msa_features,
        max_entity_mol_id,
        full_atom_array,
    ):
        sample_indice = self._get_sample_indice(idx=idx)
        pdb_indice = self._get_pdb_indice(idx=idx)
        features_dict = {}
        (
            token_array,
            atom_array,
            msa_features,
            constraint_feature_dict,
            feature_info,
            log_dict,
            full_atom_array,
        ) = self.constraint_generator.generate(
            atom_array,
            token_array,
            sample_indice,
            pdb_indice,
            msa_features,
            max_entity_mol_id,
            full_atom_array,
        )
        features_dict["constraint_feature"] = constraint_feature_dict
        features_dict.update(feature_info)
        features_dict["constraint_log_info"] = log_dict
        return token_array, atom_array, features_dict, msa_features, full_atom_array

    def get_feature_and_label(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        msa_features: dict[str, Any],
        template_features: dict[str, Any],
        full_atom_array: AtomArray,
        is_spatial_crop: bool = True,
        max_entity_mol_id: int = None,
        sample_indice: Optional[pd.Series] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get feature and label information for a given data point.
        It uses a Featurizer object to obtain input features and labels, and applies several
        steps to add other features and labels. Finally, it returns the feature dictionary, label
        dictionary, and a full label dictionary.

        Args:
            idx: Index of the data point.
            token_array: Token array representing the amino acid sequence.
            atom_array: Atom array containing atomic information.
            msa_features: Dictionary of MSA features.
            template_features: Dictionary of template features.
            full_atom_array: Full atom array containing all atoms.
            is_spatial_crop: Flag indicating whether spatial cropping is applied, by default True.
            max_entity_mol_id: Maximum entity mol ID in the full atom array.
        Returns:
            A tuple containing the feature dictionary and the label dictionary.

        Raises:
            ValueError: If the ligand cannot be found in the data point.
        """
        features_dict = {}
        if self.constraint.get("enable", False):
            (
                token_array,
                atom_array,
                features_dict,
                msa_features,
                full_atom_array,
            ) = self.get_constraint_feature(
                idx,
                atom_array,
                token_array,
                msa_features,
                max_entity_mol_id,
                full_atom_array,
            )

        # Get feature and labels from Featurizer
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
        )
        features_dict.update(feat.get_all_input_features())
        labels_dict = feat.get_labels()

        # Permutation list for atom permutation
        features_dict["atom_perm_list"] = feat.get_atom_permutation_list()
        self._add_prom_design_role_features(
            idx, token_array, atom_array, features_dict, sample_indice
        )

        # Labels for multi-chain permutation
        # Note: the returned full_atom_array may contain fewer atoms than the input
        label_full_dict, full_atom_array = Featurizer.get_gt_full_complex_features(
            atom_array=full_atom_array,
            cropped_atom_array=atom_array,
            get_cropped_asym_only=is_spatial_crop,
        )

        # Masks for Pocket Metrics
        if self.find_pocket:
            # Get entity_id of the interested ligand
            sample_indice = self._get_sample_indice(idx=idx)
            if sample_indice.mol_1_type == "ligand":
                lig_entity_id = str(sample_indice.entity_1_id)
                lig_chain_id = str(sample_indice.chain_1_id)
            elif sample_indice.mol_2_type == "ligand":
                lig_entity_id = str(sample_indice.entity_2_id)
                lig_chain_id = str(sample_indice.chain_2_id)
            else:
                raise ValueError("Cannot find ligand from this data point.")
            # Make sure the cropped array contains interested ligand
            assert lig_entity_id in set(atom_array.label_entity_id)
            assert lig_chain_id in set(atom_array.chain_id)

            # Get asym ID of the specific ligand in the `main` pocket
            lig_asym_id = atom_array.label_asym_id[atom_array.chain_id == lig_chain_id]
            assert len(np.unique(lig_asym_id)) == 1
            lig_asym_id = lig_asym_id[0]
            ligands = [lig_asym_id]

            if self.find_all_pockets:
                # Get asym ID of other ligands with the same entity_id
                all_lig_asym_ids = set(
                    full_atom_array[
                        full_atom_array.label_entity_id == lig_entity_id
                    ].label_asym_id
                )
                ligands.extend(list(all_lig_asym_ids - set([lig_asym_id])))

            # Note: the `main` pocket is the 0-indexed one.
            # [N_pocket, N_atom], [N_pocket, N_atom].
            # If not find_all_pockets, then N_pocket = 1.
            interested_ligand_mask, pocket_mask = feat.get_lig_pocket_mask(
                atom_array=full_atom_array, lig_label_asym_id=ligands
            )

            label_full_dict["pocket_mask"] = pocket_mask
            label_full_dict["interested_ligand_mask"] = interested_ligand_mask

        # Masks for Chain/Interface Metrics
        if self.find_eval_chain_interface:
            (
                eval_type,
                cluster_id,
                chain_1_mask,
                chain_2_mask,
            ) = self._get_eval_chain_interface_mask(
                idx=idx, atom_array_chain_id=full_atom_array.chain_id
            )
            labels_dict["eval_type"] = eval_type  # [N_eval]
            labels_dict["cluster_id"] = cluster_id  # [N_eval]
            labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
            labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        # Make dummy features for not implemented features
        dummy_feats = []
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)

        features_dict = make_dummy_feature(
            features_dict=features_dict, dummy_feats=dummy_feats
        )
        # Transform to right data type
        features_dict = data_type_transform(feat_or_label_dict=features_dict)
        labels_dict = data_type_transform(feat_or_label_dict=labels_dict)

        # Is_distillation
        features_dict["is_distillation"] = torch.tensor([self.is_distillation])
        if self.is_distillation is True:
            features_dict["resolution"] = torch.tensor([-1.0])
        return features_dict, labels_dict, label_full_dict

    def _add_prom_design_role_features(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        features_dict: dict[str, torch.Tensor],
        sample_indice: Optional[pd.Series] = None,
    ) -> None:
        # 0713: Base datasets do not have target/binder roles, so Promera conditioning falls back to old sampling.
        return


# 0713: Use a dedicated AbAg/Nb/TCR dataset before CDR-level metadata is available.
class AbAgDesignDataset(BaseSingleDataset):
    """Role-aware dataset for antibody/nanobody/TCR design fine-tuning CSVs."""

    target_chain_columns = (
        "antigen_chain_1",
        "antigen_chain_2",
        "antigen_chain_3",
        "antigen_chain_4",
    )
    binder_chain_columns = ("heavy_chain", "light_chain")

    def __init__(self, *args, **kwargs) -> None:
        # 0713: AbAg design rows define roles directly, so always keep only those role chains.
        kwargs["use_reference_chains_only"] = True
        kwargs.setdefault("require_standard_protein_tokens_after_crop", True)
        kwargs.setdefault("require_exact_two_chains_after_crop", False)
        super().__init__(*args, **kwargs)

    @classmethod
    def _collect_chain_ids(cls, row: pd.Series, columns: tuple[str, ...]) -> list[str]:
        chain_ids = []
        for col in columns:
            if col not in row:
                continue
            chain_id = BaseSingleDataset._clean_chain_id(row[col])
            if chain_id:
                chain_ids.append(chain_id)
        return chain_ids

    @staticmethod
    def _join_chain_ids(chain_ids: list[str]) -> str:
        return "|".join(chain_ids)

    def _normalize_abag_design_indices(self, indices_list: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"pdb_id", "heavy_chain", "antigen_chain_1", "num_tokens"}
        missing_cols = sorted(required_cols - set(indices_list.columns))
        if missing_cols:
            raise ValueError(
                f"AbAgDesignDataset indices missing required columns: {missing_cols}"
            )

        df = indices_list.copy()
        target_chain_ids = []
        binder_chain_ids = []
        keep_rows = []
        for _, row in df.iterrows():
            targets = self._collect_chain_ids(row, self.target_chain_columns)
            binders = self._collect_chain_ids(row, self.binder_chain_columns)
            target_chain_ids.append(targets)
            binder_chain_ids.append(binders)
            keep_rows.append(bool(targets) and bool(binders))

        df = df[np.asarray(keep_rows, dtype=bool)].reset_index(drop=True)
        target_chain_ids = [
            target_chain_ids[i] for i, keep in enumerate(keep_rows) if keep
        ]
        binder_chain_ids = [
            binder_chain_ids[i] for i, keep in enumerate(keep_rows) if keep
        ]
        self.check_indices_list(df, "AbAg role-chain normalization")

        # 0713: Fill legacy index columns so existing samplers, logging, and error dumps remain compatible.
        df["target_chain_ids"] = [
            self._join_chain_ids(chain_ids) for chain_ids in target_chain_ids
        ]
        df["binder_chain_ids"] = [
            self._join_chain_ids(chain_ids) for chain_ids in binder_chain_ids
        ]
        df["chain_1_id"] = [chain_ids[0] for chain_ids in target_chain_ids]
        df["chain_2_id"] = [chain_ids[0] for chain_ids in binder_chain_ids]
        df["type"] = df.get("type", "interface")
        df["mol_1_type"] = df.get("mol_1_type", "prot")
        df["mol_2_type"] = df.get("mol_2_type", "prot")
        df["mol_type_group"] = df.get("mol_type_group", "prot_prot")
        df["sub_mol_1_type"] = df.get("sub_mol_1_type", "prot")
        df["sub_mol_2_type"] = df.get("sub_mol_2_type", "prot")
        df["eval_type"] = df.get("eval_type", "prot_prot")
        df["entity_1_id"] = df.get("entity_1_id", df["target_chain_ids"])
        df["entity_2_id"] = df.get("entity_2_id", df["binder_chain_ids"])
        df["cluster_id"] = df.get("cluster_id", df["pdb_id"])

        logger.info(
            "[AbAgDesignDataset] normalized %d/%d rows with role chains",
            len(df),
            len(indices_list),
        )
        logger.info(
            "[AbAgDesignDataset] binder chain count distribution: %s",
            dict(pd.Series([len(x) for x in binder_chain_ids]).value_counts().sort_index()),
        )
        logger.info(
            "[AbAgDesignDataset] target chain count distribution: %s",
            dict(pd.Series([len(x) for x in target_chain_ids]).value_counts().sort_index()),
        )
        return df

    def read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        indices_list = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        self.check_indices_list(indices_list, "initial loading")
        logger.info(f"#Rows in indices list: {num_data}")
        indices_list = self._normalize_abag_design_indices(indices_list)

        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")
            self.check_indices_list(indices_list, "pdb_list filtering")

        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"max_n_token ({self.max_n_token}) filtering"
            )

        if self.min_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) >= self.min_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by min_n_token ({self.min_n_token})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"min_n_token ({self.min_n_token}) filtering"
            )

        if self.max_release_date and "release_date" in indices_list.columns:
            valid_mask = indices_list["release_date"].astype(str) < self.max_release_date
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed by release_date] #Rows: {len(removed_list)}")
            logger.info(
                f"[removed by release_date] #PDB: {removed_list['pdb_id'].nunique()}"
            )
            logger.info(
                f"[filtered by max_release_date (<{self.max_release_date})] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"max_release_date (<{self.max_release_date}) filtering"
            )

        for col_name, inclusion_list in self.inclusion_dict.items():
            if len(inclusion_list) == 0:
                continue
            cols = col_name.split("|")
            inclusion_set = {tuple(incl.split("|")) for incl in inclusion_list}

            def is_included(row):
                return tuple(str(row[col]) for col in cols) in inclusion_set

            valid_mask = indices_list.apply(is_included, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Included by {col_name} -- {inclusion_list}] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"inclusion_dict ({col_name}) filtering"
            )

        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
            self.check_indices_list(
                indices_list, f"exclusion_dict ({col_name}) filtering"
            )

        self.print_data_stats(indices_list)

        if self.sort_by_n_token:
            indices_list = indices_list.sort_values(
                by="num_tokens", key=lambda x: x.astype(int), ascending=False
            ).reset_index(drop=True)

        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
            self.check_indices_list(indices_list, "limits filtering")
        return indices_list

    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # 0713: AbAg rows already specify concrete chain roles; do not randomize via entity_pair_to_chain_pairs.
        sample_indice = self._get_sample_indice(idx=idx)
        if self.bioassembly_dict_dir is not None:
            bioassembly_dict_fpath = os.path.join(
                self.bioassembly_dict_dir, sample_indice.pdb_id + ".pkl.gz"
            )
        else:
            bioassembly_dict_fpath = None

        bioassembly_dict = DataPipeline.get_data_bioassembly(
            bioassembly_dict_fpath=bioassembly_dict_fpath
        )
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id
        return sample_indice, bioassembly_dict, bioassembly_dict_fpath

    def _get_reference_chain_ids(self, sample_indice: pd.Series) -> list[str]:
        # 0713: Use all antigen chains plus heavy/light binder chains for design training.
        chain_ids = []
        for field_name in ("target_chain_ids", "binder_chain_ids"):
            chain_ids.extend(
                [
                    self._clean_chain_id(chain_id)
                    for chain_id in str(sample_indice.get(field_name, "")).split("|")
                ]
            )
        return [chain_id for chain_id in chain_ids if chain_id]

    def _expected_role_chain_specs(
        self, sample_indice: pd.Series
    ) -> list[tuple[str, str, Optional[int]]]:
        specs = []
        for col, len_col in (
            ("antigen_chain_1", "antigen_1_len"),
            ("antigen_chain_2", "antigen_2_len"),
            ("antigen_chain_3", "antigen_3_len"),
            ("antigen_chain_4", "antigen_4_len"),
        ):
            chain_id = self._clean_chain_id(sample_indice.get(col, ""))
            if chain_id:
                specs.append((chain_id, "target", self._safe_int(sample_indice.get(len_col))))
        for col, len_col in (("heavy_chain", "heavy_len"), ("light_chain", "light_len")):
            chain_id = self._clean_chain_id(sample_indice.get(col, ""))
            if chain_id:
                specs.append((chain_id, "binder", self._safe_int(sample_indice.get(len_col))))
        return specs

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None or pd.isna(value):
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _chain_token_counts(
        bioassembly_dict: dict[str, Any],
    ) -> tuple[dict[str, int], list[str]]:
        token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
            "centre_atom_index"
        )
        centre_atoms = bioassembly_dict["atom_array"][token_centre_atom_indices]
        token_chain_ids = [str(chain_id) for chain_id in centre_atoms.chain_id]
        chain_order = list(dict.fromkeys(token_chain_ids))
        chain_counts = {
            chain_id: int(np.sum(np.asarray(token_chain_ids) == chain_id))
            for chain_id in chain_order
        }
        return chain_counts, chain_order

    @staticmethod
    def _annotation_values(array: AtomArray, annotation_name: str) -> Optional[np.ndarray]:
        try:
            annotation_names = set(array.get_annotation_categories())
        except Exception:
            annotation_names = set()
        if annotation_name not in annotation_names and not hasattr(array, annotation_name):
            return None
        try:
            return np.asarray(getattr(array, annotation_name))
        except Exception:
            try:
                return np.asarray(array.get_annotation(annotation_name))
            except Exception:
                return None

    @classmethod
    def _chain_alias_to_actual_chain_id(
        cls, bioassembly_dict: dict[str, Any]
    ) -> dict[str, str]:
        # 0713: Custom AbAg pkls may keep CSV chain names in auth/label annotations while chain_id gets suffixed.
        token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
            "centre_atom_index"
        )
        centre_atoms = bioassembly_dict["atom_array"][token_centre_atom_indices]
        actual_chain_ids = np.asarray([str(chain_id) for chain_id in centre_atoms.chain_id])
        alias_to_actuals: dict[str, set[str]] = {}
        for annotation_name in (
            "auth_asym_id",
            "label_asym_id",
            "auth_chain_id",
            "label_chain_id",
            "orig_chain_id",
            "chain_id",
        ):
            values = (
                actual_chain_ids
                if annotation_name == "chain_id"
                else cls._annotation_values(centre_atoms, annotation_name)
            )
            if values is None or len(values) != len(actual_chain_ids):
                continue
            for alias, actual in zip(values, actual_chain_ids):
                alias = cls._clean_chain_id(alias)
                if alias:
                    alias_to_actuals.setdefault(alias, set()).add(str(actual))
        return {
            alias: next(iter(actuals))
            for alias, actuals in alias_to_actuals.items()
            if len(actuals) == 1
        }

    @staticmethod
    def _sequence_text(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("sequence", "seq", "protein_sequence"):
                if key in value:
                    value = value[key]
                    break
        if value is None:
            return ""
        return "".join(str(value).split()).upper()

    @classmethod
    def _chain_sequences_from_tokens(
        cls, bioassembly_dict: dict[str, Any]
    ) -> dict[str, str]:
        token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
            "centre_atom_index"
        )
        centre_atoms = bioassembly_dict["atom_array"][token_centre_atom_indices]
        actual_chain_ids = [str(chain_id) for chain_id in centre_atoms.chain_id]
        res_names = (
            centre_atoms.cano_seq_resname
            if hasattr(centre_atoms, "cano_seq_resname")
            else centre_atoms.res_name
        )
        seq_chars_by_chain: dict[str, list[str]] = {}
        for chain_id, res_name in zip(actual_chain_ids, res_names):
            text = str(res_name).strip().upper()
            char = mmcif_restype_3to1.get(text, text if len(text) == 1 else "X")
            seq_chars_by_chain.setdefault(chain_id, []).append(char)
        return {
            chain_id: "".join(chars)
            for chain_id, chars in seq_chars_by_chain.items()
            if chars
        }

    @classmethod
    def _sequence_alias_to_actual_chain_id(
        cls, bioassembly_dict: dict[str, Any]
    ) -> dict[str, str]:
        # 0713: Use sequence keys as role aliases when CSV names match sequences but not atom_array.chain_id.
        sequences = bioassembly_dict.get("sequences", {})
        if not isinstance(sequences, dict):
            return {}
        actual_sequences = cls._chain_sequences_from_tokens(bioassembly_dict)
        sequence_to_actuals: dict[str, set[str]] = {}
        for actual_chain_id, sequence in actual_sequences.items():
            if sequence:
                sequence_to_actuals.setdefault(sequence, set()).add(actual_chain_id)
        alias_to_actual = {}
        for alias, sequence in sequences.items():
            alias = cls._clean_chain_id(alias)
            sequence = cls._sequence_text(sequence)
            actuals = sequence_to_actuals.get(sequence, set())
            if alias and len(actuals) == 1:
                alias_to_actual[alias] = next(iter(actuals))
        return alias_to_actual

    def _resolve_reference_chain_ids(
        self,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        ref_chain_ids: list[str],
    ) -> tuple[pd.Series, list[str]]:
        chain_counts, chain_order = self._chain_token_counts(bioassembly_dict)
        if set(ref_chain_ids).issubset(set(chain_counts)):
            return sample_indice, ref_chain_ids

        specs = self._expected_role_chain_specs(sample_indice)
        alias_to_actual = self._chain_alias_to_actual_chain_id(bioassembly_dict)
        sequence_alias_to_actual = self._sequence_alias_to_actual_chain_id(bioassembly_dict)
        used_actual = set()
        mapped_by_original = {}
        for original_chain_id, _role, expected_len in specs:
            if original_chain_id in chain_counts and original_chain_id not in used_actual:
                mapped_by_original[original_chain_id] = original_chain_id
                used_actual.add(original_chain_id)
                continue

            alias_actual = alias_to_actual.get(original_chain_id)
            if alias_actual in chain_counts and alias_actual not in used_actual:
                mapped_by_original[original_chain_id] = alias_actual
                used_actual.add(alias_actual)
                continue

            sequence_actual = sequence_alias_to_actual.get(original_chain_id)
            if sequence_actual in chain_counts and sequence_actual not in used_actual:
                mapped_by_original[original_chain_id] = sequence_actual
                used_actual.add(sequence_actual)
                continue

            candidates = [chain_id for chain_id in chain_order if chain_id not in used_actual]
            if not candidates:
                continue
            if expected_len is not None:
                exact_candidates = [
                    chain_id
                    for chain_id in candidates
                    if chain_counts.get(chain_id) == expected_len
                ]
                if len(exact_candidates) == 1:
                    chosen = exact_candidates[0]
                else:
                    chosen = min(
                        candidates,
                        key=lambda chain_id: (
                            abs(chain_counts.get(chain_id, 0) - expected_len),
                            chain_order.index(chain_id),
                        ),
                    )
            else:
                chosen = candidates[0]
            mapped_by_original[original_chain_id] = chosen
            used_actual.add(chosen)

        target_actual = [
            mapped_by_original[original]
            for original, role, _length in specs
            if role == "target" and original in mapped_by_original
        ]
        binder_actual = [
            mapped_by_original[original]
            for original, role, _length in specs
            if role == "binder" and original in mapped_by_original
        ]
        mapped_ref_chain_ids = target_actual + binder_actual
        if (
            bool(target_actual)
            and bool(binder_actual)
            and set(mapped_ref_chain_ids).issubset(set(chain_counts))
        ):
            # 0713: Some AbAg pkl files rewrite chain IDs; map CSV roles to actual atom_array chains before filtering.
            mapped_sample_indice = sample_indice.copy()
            mapped_sample_indice["target_chain_ids"] = self._join_chain_ids(target_actual)
            mapped_sample_indice["binder_chain_ids"] = self._join_chain_ids(binder_actual)
            if target_actual:
                mapped_sample_indice["chain_1_id"] = target_actual[0]
            if binder_actual:
                mapped_sample_indice["chain_2_id"] = binder_actual[0]
            if mapped_ref_chain_ids != ref_chain_ids:
                logger.debug(
                    "[AbAgDesignDataset] mapped CSV chains to atom_array chains for %s: %s -> %s",
                    sample_indice.pdb_id,
                    ref_chain_ids,
                    mapped_ref_chain_ids,
                )
            return mapped_sample_indice, mapped_ref_chain_ids

        return sample_indice, ref_chain_ids

    def _add_prom_design_role_features(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        features_dict: dict[str, torch.Tensor],
        sample_indice: Optional[pd.Series] = None,
    ) -> None:
        # 0713: Role masks let train-time masking/epitope conditioning avoid antigen-antigen contacts.
        sample_indice = sample_indice if sample_indice is not None else self._get_sample_indice(idx=idx)
        target_chain_ids = set(
            self._clean_chain_id(chain_id)
            for chain_id in str(sample_indice.get("target_chain_ids", "")).split("|")
        )
        binder_chain_ids = set(
            self._clean_chain_id(chain_id)
            for chain_id in str(sample_indice.get("binder_chain_ids", "")).split("|")
        )
        target_chain_ids.discard("")
        binder_chain_ids.discard("")

        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms = atom_array[centre_atom_indices]
        token_chain_ids = np.asarray([str(chain_id) for chain_id in centre_atoms.chain_id])
        is_target = np.isin(token_chain_ids, list(target_chain_ids))
        is_binder = np.isin(token_chain_ids, list(binder_chain_ids))

        if not bool(is_target.any()) or not bool(is_binder.any()):
            raise DesignIncompatibleSampleError(
                "AbAg role masks are empty after crop "
                f"(target={sorted(target_chain_ids)}, binder={sorted(binder_chain_ids)})"
            )

        features_dict["prom_is_target"] = torch.from_numpy(is_target.astype(bool))
        features_dict["prom_is_binder"] = torch.from_numpy(is_binder.astype(bool))


def get_msa_featurizer(configs, dataset_name: str, stage: str) -> Optional[Callable]:
    """
    Creates and returns an MSAFeaturizer object based on the provided configurations.

    Args:
        configs: A dictionary containing the configurations for the MSAFeaturizer.
        dataset_name: The name of the dataset.
        stage: The stage of the dataset (e.g., 'train', 'test').

    Returns:
        An MSAFeaturizer object if MSA is enabled in the configurations, otherwise None.
    """
    msa_info = configs["data"]["msa"]
    msa_args = deepcopy(msa_info)
    msa_args["dataset_name"] = dataset_name
    # If the dataset has special MSA settings, then overwrite the default MSA settings
    if "msa" in (dataset_config := configs["data"][dataset_name]):
        for k, v in dataset_config["msa"].items():
            msa_args[k] = v
    return MSAFeaturizer(
        dataset_name=msa_args.dataset_name,
        prot_seq_or_filename_to_msadir_jsons=msa_args.prot_seq_or_filename_to_msadir_jsons,
        prot_msadir_raw_paths=msa_args.prot_msadir_raw_paths,
        rna_seq_or_filename_to_msadir_jsons=msa_args.rna_seq_or_filename_to_msadir_jsons,
        rna_msadir_raw_paths=msa_args.rna_msadir_raw_paths,
        prot_pairing_dbs=msa_args.prot_pairing_dbs,
        prot_non_pairing_dbs=msa_args.prot_non_pairing_dbs,
        prot_indexing_methods=msa_args.prot_indexing_methods,
        rna_indexing_methods=msa_args.rna_indexing_methods,
        enable_prot_msa=msa_args.enable_prot_msa,
        enable_rna_msa=msa_args.enable_rna_msa,
    )


def get_template_featurizer(
    configs: ConfigDict, dataset_name: str, stage: str
) -> Union[Callable, None]:
    template_info = configs["data"]["template"]
    template_args = deepcopy(template_info)
    template_args["dataset_name"] = dataset_name
    # If the dataset has special template settings, then overwrite the default template settings
    if "template" in (dataset_config := configs["data"][dataset_name]):
        for k, v in dataset_config["template"].items():
            template_args[k] = v
    template_args.update(
        {
            "stage": stage,
        }
    )
    return TemplateFeaturizer(**template_args)


class WeightedMultiDataset(Dataset):
    """
    A weighted dataset composed of multiple datasets with weights.

    Args:
        datasets: A list of Dataset objects.
        dataset_names: A list of dataset names corresponding to the datasets.
        datapoint_weights: A list of lists containing sampling weights for each datapoint in the datasets.
        dataset_sample_weights: A list of torch tensors containing sampling weights for each dataset.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        datapoint_weights: list[list[float]],
        dataset_sample_weights: list[torch.tensor],
    ):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = torch.Tensor(dataset_sample_weights)
        self.iteration = 0
        self.offset = 0
        self.init_datasets()

    def init_datasets(self):
        """Calculate global weights of each datapoint in datasets for future sampling."""
        self.merged_datapoint_weights = []
        self.weight = 0.0
        self.dataset_indices = []
        self.within_dataset_indices = []
        for dataset_index, (
            dataset,
            datapoint_weight_list,
            dataset_weight,
        ) in enumerate(
            zip(self.datasets, self.datapoint_weights, self.dataset_sample_weights)
        ):
            # normalize each dataset weights
            weight_sum = sum(datapoint_weight_list)
            datapoint_weight_list = [
                dataset_weight * w / weight_sum for w in datapoint_weight_list
            ]
            self.merged_datapoint_weights.extend(datapoint_weight_list)
            self.weight += dataset_weight
            self.dataset_indices.extend([dataset_index] * len(datapoint_weight_list))
            self.within_dataset_indices.extend(list(range(len(datapoint_weight_list))))
        self.merged_datapoint_weights = torch.tensor(
            self.merged_datapoint_weights, dtype=torch.float64
        )

    def __len__(self) -> int:
        return len(self.merged_datapoint_weights)

    def __getitem__(self, index: int) -> dict[str, dict]:
        return self.datasets[self.dataset_indices[index]][
            self.within_dataset_indices[index]
        ]


def get_weighted_pdb_weight(
    data_type: str,
    cluster_size: int,
    chain_count: dict,
    eps: float = 1e-9,
    beta_dict: Optional[dict] = None,
    alpha_dict: Optional[dict] = None,
) -> float:
    """
    Get sample weight for each example in a weighted PDB dataset.

        data_type (str): Type of data, either 'chain' or 'interface'.
        cluster_size (int): Cluster size of this chain/interface.
        chain_count (dict): Count of each kind of chains, e.g., {"prot": int, "nuc": int, "ligand": int}.
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-9.
        beta_dict (Optional[dict], optional): Dictionary containing beta values for 'chain' and 'interface'.
        alpha_dict (Optional[dict], optional): Dictionary containing alpha values for different chain types.

    Returns:
         float: Calculated weight for the given chain/interface.
    """
    if not beta_dict:
        beta_dict = {
            "chain": 0.5,
            "interface": 1,
        }
    if not alpha_dict:
        alpha_dict = {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        }

    assert cluster_size > 0
    assert data_type in ["chain", "interface"]
    beta = beta_dict[data_type]
    assert set(chain_count.keys()).issubset(set(alpha_dict.keys()))
    weight = (
        beta
        * sum(
            [alpha * chain_count[data_mode] for data_mode, alpha in alpha_dict.items()]
        )
        / (cluster_size + eps)
    )
    return weight


def calc_weights_for_df(
    indices_df: pd.DataFrame,
    beta_dict: dict[str, Any],
    alpha_dict: dict[str, Any],
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Calculate weights for each example in the dataframe.

    Args:
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.

    Returns:
        A pandas DataFrame with an column 'weights' containing the calculated weights.
    """
    # Specific to assembly, and entities (chain or interface)
    df = indices_df.copy()

    df[["entity_1_id", "entity_2_id"]] = (
        df[["entity_1_id", "entity_2_id"]].astype(object).fillna("None")
    )

    s = df["pdb_id"].astype(str)
    if "assembly_id" in df.columns:
        s = s.str.cat(df["assembly_id"].astype(str), sep="_")

    e1s = df["entity_1_id"].astype(str)
    e2s = df["entity_2_id"].astype(str)
    emin = e1s.where(e1s <= e2s, e2s)
    emax = e2s.where(e1s <= e2s, e1s)
    df["pdb_sorted_entity_id"] = s.str.cat(emin, sep="_").str.cat(emax, sep="_")

    df["pdb_sorted_entity_id_member_num"] = df.groupby("pdb_sorted_entity_id")[
        "pdb_id"
    ].transform("size")

    df["cluster_size"] = df.groupby("cluster_id")["pdb_sorted_entity_id"].transform(
        "nunique"
    )
    beta_dict_bytes = {k.encode(): v for k, v in beta_dict.items()}
    df["beta"] = df["type"].map({**beta_dict, **beta_dict_bytes}).astype(float)

    weighted_count = np.zeros(len(df), dtype=np.float64)
    mol1 = df["mol_1_type"]
    mol2 = df["mol_2_type"]
    for t, alpha in alpha_dict.items():
        c_t = mol1.eq(t).to_numpy(dtype=np.int8) + mol2.eq(t).to_numpy(dtype=np.int8)
        weighted_count += float(alpha) * c_t

    tmp_weights = (
        df["beta"].to_numpy() * weighted_count / (df["cluster_size"].to_numpy() + eps)
    )

    df["tmp_weights"] = tmp_weights  # do not use this
    df["weights"] = df["tmp_weights"] / df["pdb_sorted_entity_id_member_num"]
    return df


def get_sample_weights(
    sampler_type: str,
    indices_df: pd.DataFrame = None,
    beta_dict: dict = {
        "chain": 0.5,
        "interface": 1,
    },
    alpha_dict: dict = {
        "prot": 3,
        "nuc": 3,
        "ligand": 1,
    },
    force_recompute_weight: bool = False,
) -> Union[pd.Series, list[float]]:
    """
    Computes sample weights based on the specified sampler type.

    Args:
        sampler_type: The type of sampler to use ('weighted' or 'uniform').
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.
        force_recompute_weight: Whether to force recomputation of weights even if they already exist.

    Returns:
        A list of sample weights.

    Raises:
        ValueError: If an unknown sampler type is provided.
    """
    if sampler_type == "weighted":
        assert indices_df is not None
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(
                indices_df=indices_df,
                beta_dict=beta_dict,
                alpha_dict=alpha_dict,
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "uniform":
        assert indices_df is not None
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def get_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> tuple[WeightedMultiDataset, dict[str, BaseSingleDataset]]:
    """
    Get training and testing datasets given configs

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A tuple containing the training dataset and a dictionary of testing datasets.
    """

    def _get_dataset_param(config_dict, dataset_name: str, stage: str):
        # Template_featurizer is under development
        # Lig_atom_rename/shuffle_mols/shuffle_sym_ids do not affect the performance very much
        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage),
            "template_featurizer": get_template_featurizer(
                configs, dataset_name, stage
            ),
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "constraint": config_dict.get("constraint", {}),
        }

    def _get_dataset_class(config_dict):
        # 0713: Let role-aware AbAg design data opt into its own Dataset from config.
        dataset_class_name = config_dict.get("dataset_class", "base")
        dataset_class_map = {
            "base": BaseSingleDataset,
            "BaseSingleDataset": BaseSingleDataset,
            "abag_design": AbAgDesignDataset,
            "AbAgDesignDataset": AbAgDesignDataset,
        }
        if dataset_class_name not in dataset_class_map:
            raise ValueError(f"Unknown dataset_class: {dataset_class_name}")
        return dataset_class_map[dataset_class_name]

    data_config = configs.data
    logger.info(f"Using train sets {data_config.train_sets}")
    assert len(data_config.train_sets) == len(
        data_config.train_sampler.train_sample_weights
    )
    train_datasets = []
    datapoint_weights = []
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=train_name, stage="train"
        )
        dataset_param["ref_pos_augment"] = data_config.get(
            "train_ref_pos_augment", True
        )
        dataset_param["limits"] = data_config.get("limits", -1)
        train_dataset = _get_dataset_class(config_dict)(**dataset_param)
        train_datasets.append(train_dataset)
        datapoint_weights.append(
            get_sample_weights(
                **data_config[train_name]["sampler_configs"],
                indices_df=train_dataset.indices_list,
            )
        )
    train_dataset = WeightedMultiDataset(
        datasets=train_datasets,
        dataset_names=data_config.train_sets,
        datapoint_weights=datapoint_weights,
        dataset_sample_weights=data_config.train_sampler.train_sample_weights,
    )

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        test_dataset = _get_dataset_class(config_dict)(**dataset_param)
        test_datasets[test_name] = test_dataset
    return train_dataset, test_datasets
