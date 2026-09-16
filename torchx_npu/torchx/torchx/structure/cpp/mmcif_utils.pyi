

from collections.abc import Sequence

import numpy as np
from torchx.cpp import cif_dict
from torchx.structure.python import mmcif_layout


def filter(
    mmcif: cif_dict.CifDict,
    include_nucleotides: bool,
    include_ligands: bool = ...,
    include_water: bool = ...,
    include_other: bool = ...,
    model_id: str = ...,
) -> tuple[np.ndarray[int], mmcif_layout.MmcifLayout]: ...


def fix_residues(
    layout: mmcif_layout.MmcifLayout,
    comp_id: Sequence[str],
    atom_id: Sequence[str],
    atom_x: Sequence[float],
    atom_y: Sequence[float],
    atom_z: Sequence[float],
    fix_arg: bool = ...,
) -> None: ...


def read_layout(
    mmcif: cif_dict.CifDict, model_id: str = ...
) -> mmcif_layout.MmcifLayout: ...


def selected_ligand_residue_mask(
    layout: mmcif_layout.MmcifLayout,
    atom_site_label_asym_ids: list[str],
    atom_site_label_seq_ids: list[str],
    atom_site_auth_seq_ids: list[str],
    atom_site_label_comp_ids: list[str],
    atom_site_pdbx_pdb_ins_codes: list[str],
    nonpoly_asym_ids: list[str],
    nonpoly_auth_seq_ids: list[str],
    nonpoly_pdb_ins_codes: list[str],
    nonpoly_mon_ids: list[str],
    branch_asym_ids: list[str],
    branch_auth_seq_ids: list[str],
    branch_pdb_ins_codes: list[str],
    branch_mon_ids: list[str],
) -> tuple[list[bool], list[bool]]: ...


def selected_polymer_residue_mask(
    layout: mmcif_layout.MmcifLayout,
    atom_site_label_asym_ids: list[str],
    atom_site_label_seq_ids: list[str],
    atom_site_label_comp_ids: list[str],
    poly_seq_asym_ids: list[str],
    poly_seq_seq_ids: list[str],
    poly_seq_mon_ids: list[str],
) -> list[bool]: ...
