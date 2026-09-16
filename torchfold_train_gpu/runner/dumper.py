import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import biotite.structure.io.pdbx as pdbx


def _map_values_to_list(data, recursive=True):
    """Recursively convert torch tensors / numpy arrays to plain lists.

    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = v.cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, dict) and recursive:
            data[k] = _map_values_to_list(v, recursive)
    return data


def save_json(data, output_fpath, indent=4):
    data_json = data.copy()
    data_json = _map_values_to_list(data_json)
    with open(output_fpath, "w") as f:
        if indent is not None:
            json.dump(data_json, f, indent=indent)
        else:
            json.dump(data_json, f)


def round_values(data, recursive=True):
    """Recursively round floats to 2 decimals; tensors / arrays -> rounded arrays.

    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = np.round(v.cpu().numpy(), 2)
        elif isinstance(v, np.ndarray):
            data[k] = np.round(v, 2)
        elif isinstance(v, list):
            data[k] = list(np.round(np.array(v), 2))
        elif isinstance(v, dict) and recursive:
            data[k] = round_values(v, recursive)
    return data


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """Clean and round the full confidence dict.

    """
    # Remove heavy / non-serializable keys if present.
    full_confidence_dict.pop("atom_coordinate", None)
    full_confidence_dict.pop("atom_is_polymer", None)
    # Keep two decimal places.
    full_confidence_dict = round_values(full_confidence_dict)
    return full_confidence_dict


# --------------------------------------------------------------------------- #
# DataDumper
# --------------------------------------------------------------------------- #
class DataDumper:
    """Dump prediction data (ranked structures + confidence scores).

    Args:
        base_dir (str): Base directory for saving dumped data.
        need_atom_confidence (bool): Whether to save detailed atom-level
            confidence data (the `*_full_data_sample_{rank}.json` files).
        sorted_by_ranking_score (bool): Whether to name output files by their
            rank (best == 0) according to `ranking_score`. If False, files are
            named by original sample index.
    """

    def __init__(
        self,
        base_dir: str,
        need_atom_confidence: bool = False,
        sorted_by_ranking_score: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence
        self.sorted_by_ranking_score = sorted_by_ranking_score

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array,
        entity_poly_type: Optional[dict] = None,
    ):
        """Dump predictions and related data to the output layout.

        Args:
            dataset_name (str): Name of the dataset.
            pdb_id (str): PDB ID / sample name.
            seed (int): Random seed used for the prediction.
            pred_dict (dict): {
                "coordinate": [N_sample, N_atom, 3] tensor/ndarray,
                "summary_confidence": list[dict] (len N_sample, has
                    "ranking_score"),
                "full_data": list[dict] (len N_sample; may be [{}] * N),
            }
            atom_array (biotite AtomArray): Template (N_atom); coords are
                overwritten per sample.

        """
        dump_dir = self._get_dump_dir(dataset_name, pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            seed=seed,
        )

    def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int) -> str:
        return os.path.join(self.base_dir, dataset_name, sample_name, f"seed_{seed}")

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array,
        seed: int,
    ):
        prediction_save_dir = os.path.join(dump_dir, "predictions")
        os.makedirs(prediction_save_dir, exist_ok=True)

        # Build per-sample b_factor (atom_plddt * 100) from full_data, if present.
        b_factor = None
        if "full_data" in pred_dict:
            all_atom_plddt = []
            for each_sample_dict in pred_dict["full_data"]:
                if "atom_plddt" in each_sample_dict:
                    atom_plddt = each_sample_dict["atom_plddt"]
                    if isinstance(atom_plddt, torch.Tensor):
                        if atom_plddt.dtype == torch.bfloat16:
                            atom_plddt = atom_plddt.to(torch.float32)
                        atom_plddt = atom_plddt.cpu().numpy()
                    else:
                        atom_plddt = np.asarray(atom_plddt)
                    all_atom_plddt.append(atom_plddt * 100.0)
            # Only use b_factor if EVERY sample contributed an atom_plddt.
            if len(all_atom_plddt) == len(pred_dict["full_data"]):
                b_factor = all_atom_plddt

        sorted_indices = self._get_ranker_indices(data=pred_dict)
        self._save_structure(
            pred_coordinates=pred_dict["coordinate"],
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            atom_array=atom_array,
            sorted_indices=sorted_indices,
            b_factor=b_factor,
        )
        self._save_confidence(
            data=pred_dict,
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            sorted_indices=sorted_indices,
        )

    def _get_ranker_indices(self, data: dict) -> List[int]:
        """Per-sample rank by ranking_score (best == 0).

        `torch.argsort(torch.argsort(value, descending=True))` yields, for each
        original sample index i, its rank position. When
        `sorted_by_ranking_score` is False, returns identity (name by original
        index).
        """
        N_sample = len(data["summary_confidence"])
        if self.sorted_by_ranking_score:
            value = torch.tensor(
                [
                    float(data["summary_confidence"][i]["ranking_score"])
                    for i in range(N_sample)
                ]
            )
            sorted_indices = [
                int(i) for i in torch.argsort(torch.argsort(value, descending=True))
            ]
        else:
            sorted_indices = [i for i in range(N_sample)]
        return sorted_indices

    def _save_structure(
        self,
        pred_coordinates,
        prediction_save_dir: str,
        sample_name: str,
        atom_array,
        sorted_indices: Optional[List[int]],
        b_factor: Optional[List[np.ndarray]] = None,
    ):
        """Write one CIF per sample, named by rank.

        Biotite write pattern copied from torchfold/runner/inference.py::dump_cif:
            arr = atom_array.copy(); arr.coord = coords[idx]
            arr.set_annotation("b_factor", np.round(...))
            cif = pdbx.CIFFile(); pdbx.set_structure(cif, arr); cif.write(fpath)
        """
        assert atom_array is not None
        if isinstance(pred_coordinates, torch.Tensor):
            if pred_coordinates.dtype == torch.bfloat16:
                pred_coordinates = pred_coordinates.to(torch.float32)
            pred_coordinates = pred_coordinates.cpu().numpy()
        else:
            pred_coordinates = np.asarray(pred_coordinates)

        N_sample = pred_coordinates.shape[0]
        if sorted_indices is None:
            sorted_indices = range(N_sample)  # do not rank the output file

        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir, f"{sample_name}_sample_{rank}.cif"
            )
            arr = atom_array.copy()
            arr.coord = np.asarray(pred_coordinates[idx], dtype=np.float32)
            if b_factor is not None:
                # b_factor[idx].shape == [N_atom]
                arr.set_annotation("b_factor", np.round(b_factor[idx], 2))
            cif = pdbx.CIFFile()
            pdbx.set_structure(cif, arr)
            cif.write(output_fpath)

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        sample_name: str,
        sorted_indices: Optional[List[int]],
    ):
        """Write summary JSON for every sample (always) and full_data JSON
        (only if need_atom_confidence). Files named by rank.
        """
        N_sample = len(data["summary_confidence"])
        if self.need_atom_confidence:
            for idx in range(N_sample):
                data["full_data"][idx] = get_clean_full_confidence(
                    data["full_data"][idx]
                )
        if sorted_indices is None:
            sorted_indices = range(N_sample)
        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_summary_confidence_sample_{rank}.json",
            )
            save_json(data["summary_confidence"][idx], output_fpath, indent=4)
            if self.need_atom_confidence:
                output_fpath = os.path.join(
                    prediction_save_dir,
                    f"{sample_name}_full_data_sample_{rank}.json",
                )
                save_json(data["full_data"][idx], output_fpath, indent=None)
