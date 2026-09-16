from collections.abc import Sequence

import numpy as np

from torchfold.constants import chemical_components, mmcif_names, residue_names
from torchfold.structure import (
    bonds, mmcif, structure,
    chemical_components as struc_chem_comps
)
from torchfold.structure.tables import Chains, Residues, Atoms, Bonds
from torchfold.structure.parsing.parsing_constants import SequenceFormat, BondAtomId
from torchfold.structure.parsing.parsing_utils import _create_bond_lookup, _get_atom_element, _get_representative_atom, \
    _add_ligand_to_chem_comp


def from_sequences_and_bonds(
        *,
        sequences: Sequence[str],
        chain_types: Sequence[str],
        sequence_formats: Sequence[SequenceFormat],
        bonded_atom_pairs: Sequence[tuple[BondAtomId, BondAtomId]] | None,
        ccd: chemical_components.Ccd,
        chain_ids: Sequence[str] | None = None,
        name: str = 'from_sequences_and_bonds',
        bond_type: str | None = None,
        **constructor_args,
) -> structure.Structure:
    """Returns a minimal structure for the input sequences and bonds.

    The returned structure will have at least one atom per residue. If the
    residue has any bonded atoms, according to `bonded_atom_pairs`, then
    all (and only) those atoms will be present for that residue. If the residue
    is not involved in any bond then an arbitrary atom will be created.

    Args:
      sequences: A sequence of strings, each one representing a single chain.
      chain_types: The types of each chain, e.g. polypeptide(L). The n-th element
        describes the n-th sequence in `sequences`.
      sequence_formats: The format of each sequence. The n-th element describes
        the n-th sequence in `sequences`.
      bonded_atom_pairs: A sequence of bonded atom pairs. Each atom is described
        as a tuple of (chain_index, res_index, atom_name), where the first two
        values are 0-based indices. The chain_index is the index of the chain in
        the `sequences` argument, and the res_index is the index of the residue in
        that sequence. The atom_name is the name of the atom in the residue, e.g.
        CA. If the atom is not found in the standard atoms for that residue
        (according to the CCD) then an error is raised.
      ccd: The chemical components dictionary.
      chain_ids: A sequence of chain IDs, one for each chain in `sequences`. If
        not provided, then the chain IDs will be generated automatically based on
        sequence indices.
      name: A name for the returned structure.
      bond_type: This type will be used for all bonds in the structure, where type
        follows PDB scheme, e.g. unknown (?), hydrog, metalc, covale, disulf.
      **constructor_args: These arguments are passed directly to the
        structure.Structure constructor.
    """
    chain_id = []
    chain_type = []
    chain_res_count = []
    res_id = []
    res_name = []
    res_atom_count = []
    atom_name = []
    atom_element = []
    chem_comp = {}

    num_bonds = len(bonded_atom_pairs or ())
    from_atom_key = np.full((num_bonds,), -1, dtype=np.int64)
    dest_atom_key = np.full((num_bonds,), -1, dtype=np.int64)

    # Create map (chain_i, res_i) -> {atom_name -> (from_idxs dest_idxs)}.
    # This allows quick lookup of whether a residue has any bonded atoms, and
    # which bonds those atoms participate in.
    bond_lookup = _create_bond_lookup(bonded_atom_pairs or ())

    current_atom_key = 0
    for chain_i, (sequence, curr_chain_type, sequence_format) in enumerate(
            zip(sequences, chain_types, sequence_formats, strict=True)
    ):
        if chain_ids is not None:
            current_chain_id = chain_ids[chain_i]
        else:
            current_chain_id = mmcif.int_id_to_str_id(chain_i + 1)
        num_chain_residues = 0
        for res_i, full_res_name in enumerate(
                expand_sequence(sequence, curr_chain_type, sequence_format)
        ):
            current_res_id = res_i + 1
            num_res_atoms = 0

            # Look for bonded atoms in the bond lookup and if any are found, add
            # their atom keys to the bond atom_key columns.
            if bond_indices_by_atom_name := bond_lookup.get((chain_i, res_i)):
                comp_atoms = None
                if sequence_format != SequenceFormat.LIGAND_SMILES:
                    comp_atoms = set(ccd.get(full_res_name)['_chem_comp_atom.atom_id'])
                for bond_atom_name, bond_indices in bond_indices_by_atom_name.items():
                    if comp_atoms is not None and bond_atom_name not in comp_atoms:
                        raise ValueError(
                            f'Bonded atom "{bond_atom_name}" was not found in the list of'
                            f' atoms of the chemical component {full_res_name}. Valid atom'
                            f' names for {full_res_name} are: {sorted(comp_atoms)}.'
                            ' This is likely caused by an invalid atom name in the bonded'
                            f' atom (chain_id={current_chain_id}, res_id={current_res_id},'
                            f' atom_name={bond_atom_name}) specified in `bondedAtomPairs`'
                            ' in the input JSON.'
                        )
                    atom_name.append(bond_atom_name)
                    atom_element.append(
                        _get_atom_element(
                            ccd=ccd, res_name=full_res_name, atom_name=bond_atom_name
                        )
                    )
                    for from_bond_i in bond_indices.from_indices:
                        from_atom_key[from_bond_i] = current_atom_key
                    for dest_bond_i in bond_indices.dest_indices:
                        dest_atom_key[dest_bond_i] = current_atom_key
                    current_atom_key += 1
                    num_res_atoms += 1
            else:
                # If this residue has no bonded atoms then we need to add one atom
                # like in from_sequences.
                assert num_res_atoms == 0
                rep_atom_name, rep_atom_element = _get_representative_atom(
                    ccd=ccd,
                    res_name=full_res_name,
                    chain_type=curr_chain_type,
                    sequence_format=sequence_format,
                )
                atom_name.append(rep_atom_name)
                atom_element.append(rep_atom_element)
                num_res_atoms += 1
                current_atom_key += 1

            if sequence_format == SequenceFormat.LIGAND_SMILES:
                # Sequence expect to be in the format <ligand_id>:<ligand_smiles>,
                # which always corresponds to a single-residue chain.
                ligand_id, ligand_smiles = sequence.split(':', maxsplit=1)
                if ccd.get(ligand_id) is not None:
                    raise ValueError(
                        f'Ligand name {ligand_id} is in CCD - it is not supported to give'
                        ' ligands created from SMILES the same name as CCD components.'
                    )
                # We need to provide additional chemical components metadata for
                # ligands specified via SMILES strings since they might not be in CCD.
                _add_ligand_to_chem_comp(chem_comp, ligand_id, ligand_smiles)

            assert num_res_atoms >= 1
            res_atom_count.append(num_res_atoms)
            num_chain_residues += 1
            res_id.append(current_res_id)
            res_name.append(full_res_name)

        chain_id.append(current_chain_id)
        chain_type.append(curr_chain_type)
        chain_res_count.append(num_chain_residues)

    chem_comp_data = struc_chem_comps.ChemicalComponentsData(chem_comp)
    chem_comp_data = struc_chem_comps.populate_missing_ccd_data(
        ccd=ccd,
        chemical_components_data=chem_comp_data,
        chemical_component_ids=set(res_name),
    )

    if bonded_atom_pairs is not None:
        unknown_bond_col = np.full((num_bonds,), '?', dtype=object)
        if bond_type is None:
            bond_type_col = unknown_bond_col
        else:
            bond_type_col = np.full((num_bonds,), bond_type, dtype=object)
        bonds_table = bonds.Bonds(
            key=np.arange(num_bonds, dtype=np.int64),
            type=bond_type_col,
            role=unknown_bond_col,
            from_atom_key=from_atom_key,
            dest_atom_key=dest_atom_key,
        )
    else:
        bonds_table = Bonds.make_empty()

    chain_key = np.arange(len(sequences), dtype=np.int64)  # 1 chain per sequence.
    chain_id = np.array(chain_id, dtype=object)
    chains_table = Chains(
        key=chain_key,
        id=chain_id,
        type=np.array(chain_type, dtype=object),
        auth_asym_id=chain_id,
        entity_id=np.char.mod('%d', chain_key + 1).astype(object),
        entity_desc=np.array(['.'] * len(chain_key), dtype=object),
    )

    res_key = np.arange(len(res_name), dtype=np.int64)
    res_chain_key = np.repeat(chain_key, chain_res_count)
    residues_table = Residues(
        key=res_key,
        chain_key=res_chain_key,
        id=np.array(res_id, dtype=np.int32),
        name=np.array(res_name, dtype=object),
        auth_seq_id=np.char.mod('%d', res_id).astype(object),
        insertion_code=np.full(len(res_name), '?', dtype=object),
    )

    num_atoms = current_atom_key
    atom_float32_zeros = np.zeros(num_atoms, dtype=np.float32)
    atoms_table = Atoms(
        key=np.arange(num_atoms, dtype=np.int64),
        chain_key=np.repeat(res_chain_key, res_atom_count),
        res_key=np.repeat(res_key, res_atom_count),
        name=np.array(atom_name, dtype=object),
        element=np.array(atom_element, dtype=object),
        x=atom_float32_zeros,
        y=atom_float32_zeros,
        z=atom_float32_zeros,
        b_factor=atom_float32_zeros,
        occupancy=np.ones(num_atoms, np.float32),
    )

    return structure.Structure(
        name=name,
        atoms=atoms_table,
        residues=residues_table,
        chains=chains_table,
        bonds=bonds_table,
        chemical_components_data=chem_comp_data,
        **constructor_args,
    )


def expand_sequence(
        sequence: str, chain_type: str, sequence_format: SequenceFormat
) -> Sequence[str]:
    """Returns full residue names based on a sequence string.

    Args:
      sequence: A string representing the sequence.
      chain_type: The chain type of the sequence.
      sequence_format: The format of the sequence argument.
    """
    match sequence_format:
        case SequenceFormat.FASTA:
            if not all(c.isalpha() for c in sequence):
                raise ValueError(f'Sequence "{sequence}" has non-alphabetic characters')
            match chain_type:
                case mmcif_names.PROTEIN_CHAIN:
                    res_name_map = residue_names.PROTEIN_COMMON_ONE_TO_THREE
                    default_res_name = residue_names.UNK
                case mmcif_names.RNA_CHAIN:
                    res_name_map = {r: r for r in residue_names.RNA_TYPES}
                    default_res_name = residue_names.UNK_RNA
                case mmcif_names.DNA_CHAIN:
                    res_name_map = residue_names.DNA_COMMON_ONE_TO_TWO
                    default_res_name = residue_names.UNK_DNA
                case _:
                    raise ValueError(f'{chain_type=} not supported for FASTA format.')
            return [
                res_name_map.get(one_letter_res, default_res_name)
                for one_letter_res in sequence
            ]
        case SequenceFormat.CCD_CODES:
            return sequence.strip('()').split(')(')
        case SequenceFormat.LIGAND_SMILES:
            ligand_id, _ = sequence.split(':', maxsplit=1)
            return [ligand_id]
