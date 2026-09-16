import enum
import functools
from collections.abc import Collection, Mapping, MutableMapping, Sequence

from torchfold.constants import chemical_components, residue_names, mmcif_names
from torchfold.structure import chemical_components as struc_chem_comps, mmcif
from torchfold.structure.parsing.parsing_constants import (
    BondAtomId, BondIndices, SequenceFormat, ModelID, NoAtomsError
)


def _create_bond_lookup(
        bonded_atom_pairs: Sequence[tuple[BondAtomId, BondAtomId]],
) -> Mapping[tuple[int, int], Mapping[str, BondIndices]]:
    """Creates maps to help find bonds during a loop over residues."""
    bond_lookup = {}
    for bond_i, (from_atom_id, dest_atom_id) in enumerate(bonded_atom_pairs):
        from_chain_i, from_res_i, from_atom_name = from_atom_id
        dest_chain_i, dest_res_i, dest_atom_name = dest_atom_id
        bonds_by_from_atom_name = bond_lookup.setdefault(
            (from_chain_i, from_res_i), {}
        )
        bonds_by_dest_atom_name = bond_lookup.setdefault(
            (dest_chain_i, dest_res_i), {}
        )
        bonds_by_from_atom_name.setdefault(
            from_atom_name, BondIndices(from_indices=[], dest_indices=[])
        ).from_indices.append(bond_i)
        bonds_by_dest_atom_name.setdefault(
            dest_atom_name, BondIndices(from_indices=[], dest_indices=[])
        ).dest_indices.append(bond_i)
    return bond_lookup


def _get_atom_element(
        ccd: chemical_components.Ccd, res_name: str, atom_name: str
) -> str:
    type_symbol = chemical_components.type_symbol(
        ccd, res_name=res_name, atom_name=atom_name
    )
    return type_symbol or '?'


def _get_representative_atom(
        ccd: chemical_components.Ccd,
        res_name: str,
        chain_type: str,
        sequence_format: SequenceFormat,
) -> tuple[str, str]:
    match sequence_format:
        case SequenceFormat.CCD_CODES:
            atom_name = _get_first_non_leaving_atom(ccd=ccd, res_name=res_name)
            atom_element = _get_atom_element(
                ccd=ccd, res_name=res_name, atom_name=atom_name
            )
            return atom_name, atom_element
        case SequenceFormat.LIGAND_SMILES:
            return '', '?'
        case SequenceFormat.FASTA:
            if chain_type in mmcif_names.PEPTIDE_CHAIN_TYPES:
                return 'CA', 'C'
            if chain_type in mmcif_names.NUCLEIC_ACID_CHAIN_TYPES:
                return "C1'", 'C'
            else:
                raise ValueError(chain_type)
        case _:
            raise ValueError(sequence_format)


@functools.lru_cache(maxsize=128)
def _get_first_non_leaving_atom(
        ccd: chemical_components.Ccd, res_name: str
) -> str:
    """Returns first definitely non-leaving atom if exists, as a stand-in."""
    all_atoms = struc_chem_comps.get_all_atoms_in_entry(ccd, res_name=res_name)[
        '_chem_comp_atom.atom_id'
    ]
    representative_atom = all_atoms[0]
    if representative_atom == 'O1' and len(all_atoms) > 1:
        representative_atom = all_atoms[1]
    return representative_atom


def _add_ligand_to_chem_comp(
        chem_comp: MutableMapping[str, struc_chem_comps.ChemCompEntry],
        ligand_id: str,
        ligand_smiles: str,
):
    """Adds a ligand to chemical components. Raises ValueError on mismatch."""
    new_entry = struc_chem_comps.ChemCompEntry(
        type='non-polymer', pdbx_smiles=ligand_smiles
    )
    existing_entry = chem_comp.get(ligand_id)
    if existing_entry is None:
        chem_comp[ligand_id] = new_entry
    elif existing_entry != new_entry:
        raise ValueError(
            f'Mismatching data for ligand {ligand_id}: '
            f'{new_entry} != {existing_entry}'
        )


def _get_first_model_id(cif: mmcif.Mmcif) -> str:
    """Returns cheaply the first model ID from the mmCIF."""
    return cif.get_array(
        '_atom_site.pdbx_PDB_model_num', dtype=object, gather=slice(1)
    )[0]


def _get_str_model_id(
        cif: mmcif.Mmcif,
        model_id: ModelID | int,
) -> str:
    """Converts a user-specified model_id argument into a string."""
    match model_id:
        case int():
            str_model_id = str(model_id)
        case enum.Enum():
            match model_id.value:
                case ModelID.FIRST.value:
                    try:
                        str_model_id = _get_first_model_id(cif)
                    except IndexError as e:
                        raise NoAtomsError(
                            'The mmCIF does not have any atoms or'
                            ' _atom_site.pdbx_PDB_model_num is missing.'
                        ) from e
                case ModelID.ALL.value:
                    str_model_id = ''
                case _:
                    raise ValueError(
                        f'Model ID {model_id} with value {model_id.value} not recognized.'
                    )
        case _:
            raise ValueError(
                f'Model ID {model_id} with type {type(model_id)} not recognized.'
            )
    return str_model_id


def _guess_entity_type(
        chain_residues: Collection[str], atom_types: Collection[str]
) -> str:
    """Guess the entity type (polymer/non-polymer/water) based on residues/atoms.

    We treat both arguments as unordered collections since we care only whether
    all elements satisfy come conditions. The chain_residues can be either
    grouped by residue (length num_res), or it can be raw (length num_atoms).
    Atom type is unique for each atom in a residue, so don't group atom_types.

    Args:
      chain_residues: A sequence of full residue name (1-letter for DNA, 2-letters
        for RNA, 3 for protein). The _atom_site.label_comp_id column in mmCIF.
      atom_types: Atom type: ATOM or HETATM. The _atom_site.group_PDB column in
        mmCIF.

    Returns:
      One of polymer/non-polymer/water based on the following criteria:
      * If all atoms are HETATMs and all residues are water -> water.
      * If all atoms are HETATMs and not all residues are water -> non-polymer.
      * Otherwise -> polymer.
    """
    if not chain_residues or not atom_types:
        raise ValueError(
            f'chain_residues (len {len(chain_residues)}) and atom_types (len '
            f'{len(atom_types)}) must be both non-empty. Got: {chain_residues=} '
            f'and {atom_types=}'
        )

    if all(a == 'HETATM' for a in atom_types):
        if all(c in residue_names.WATER_TYPES for c in chain_residues):
            return mmcif_names.WATER
        return mmcif_names.NON_POLYMER_CHAIN
    return mmcif_names.POLYMER_CHAIN
