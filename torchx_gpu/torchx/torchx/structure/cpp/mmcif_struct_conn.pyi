

from torchx.cpp import cif_dict

def get_bond_atom_indices(mmcif_dict: cif_dict.CifDict, model_id: str) -> tuple[list[int],list[int]]: ...
