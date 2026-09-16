

from collections.abc import Callable

from torchx.cpp import cif_dict


def get_internal_to_author_chain_id_map(
    mmcif: cif_dict.CifDict
) -> dict[str,str]: ...


def get_or_infer_type_symbol(
    mmcif: cif_dict.CifDict,
    atom_id_to_type_symbol: Callable[[str, str], str],
) -> list[str]: ...
