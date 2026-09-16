import numpy as np
from typing import Mapping, Sequence
from torchfold.cpp import string_array
from torchfold.constants import residue_names
from torchfold.constants import mmcif_names


def get_change_indices(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=np.int32)
    else:
        changing_idxs = np.where(arr[1:] != arr[:-1])[0] + 1
        return np.concatenate(([0], changing_idxs), axis=0)


def fix_non_standard_polymer_residues(
        res_names: np.ndarray, chain_type: str
) -> np.ndarray:
    """Remaps residue names to the closest standard protein/RNA/DNA residue.

    If residue name is already a standard type, it is not altered.
    If a match cannot be found, returns 'UNK' for protein chainresidues and 'N'
      for RNA/DNA chain residue.

    Args:
       res_names: A numpy array of string residue names (CCD monomer codes). E.g.
         'ARG' (protein), 'DT' (DNA), 'N' (RNA).
       chain_type: The type of the chain, must be PROTEIN_CHAIN, RNA_CHAIN or
         DNA_CHAIN.

    Returns:
      An array remapped so that its elements are all from
      PROTEIN_TYPES_WITH_UNKNOWN | RNA_TYPES | DNA_TYPES | {'N'}.

    Raises:
      ValueError: If chain_type not in PEPTIDE_CHAIN_TYPES or
        {OTHER_CHAIN, RNA_CHAIN, DNA_CHAIN, DNA_RNA_HYBRID_CHAIN}.
    """
    # Map to one letter code, then back to common res_names.
    one_letter_codes = string_array.remap(
        res_names, mapping=residue_names.CCD_NAME_TO_ONE_LETTER, default_value='X'
    )

    if (
            chain_type in mmcif_names.PEPTIDE_CHAIN_TYPES
            or chain_type == mmcif_names.OTHER_CHAIN
    ):
        mapping = residue_names.PROTEIN_COMMON_ONE_TO_THREE
        default_value = 'UNK'
    elif chain_type == mmcif_names.RNA_CHAIN:
        # RNA has single-letter CCD monomer codes.
        mapping = {r: r for r in residue_names.RNA_TYPES}
        default_value = 'N'
    elif chain_type == mmcif_names.DNA_CHAIN:
        mapping = residue_names.DNA_COMMON_ONE_TO_TWO
        default_value = 'N'
    elif chain_type == mmcif_names.DNA_RNA_HYBRID_CHAIN:
        mapping = {r: r for r in residue_names.NUCLEIC_TYPES_WITH_UNKNOWN}
        default_value = 'N'
    else:
        raise ValueError(f'Expected a protein/DNA/RNA chain but got {chain_type}')

    return string_array.remap(
        one_letter_codes, mapping=mapping, default_value=default_value
    )


def get_chain_res_name_sequence(
        struc,
        *,
        include_missing_residues: bool = True,
        fix_non_standard_polymer_res: bool = False,
) -> Mapping[str, Sequence[str]]:
    """A mapping from internal chain ID to a sequence of residue names.

           The residue names are the full residue names rather than single letter
           codes. For instance, for proteins these are the 3 letter CCD codes.

           Args:
             struc:self
             include_missing_residues: Whether to include residues with no atoms in the
               returned sequences.
             fix_non_standard_polymer_res: Whether to map non standard residues in
               protein / RNA / DNA chains to standard residues (e.g. MSE -> MET) or UNK
               / N if a match is not found.

           Returns:
             A mapping from (internal) chain IDs to a sequence of residue names.
           """
    res_table = (
        struc.residues_table if include_missing_residues else struc.present_residues
    )
    residue_chain_boundaries = get_change_indices(res_table.chain_key)
    boundaries = struc.iter_residue_ranges(
        residue_chain_boundaries, count_unresolved=include_missing_residues
    )
    chain_keys = res_table.chain_key[residue_chain_boundaries]
    chain_ids = struc.chains_table.apply_array_to_column('id', chain_keys)
    chain_types = struc.chains_table.apply_array_to_column('type', chain_keys)
    chain_seqs = {}
    for idx, (start, end) in enumerate(boundaries):
        chain_id = chain_ids[idx]
        chain_type = chain_types[idx]
        chain_res = res_table.name[start:end]
        if (
                fix_non_standard_polymer_res
                and chain_type in mmcif_names.POLYMER_CHAIN_TYPES
        ):
            chain_seqs[chain_id] = tuple(
                fix_non_standard_polymer_residues(
                    res_names=chain_res, chain_type=chain_type
                )
            )
        else:
            chain_seqs[chain_id] = tuple(chain_res)

    return chain_seqs


def structure_rename_chain_ids(struc, new_id_by_old_id: Mapping[str, str]):
    """Returns a new structure with renamed chain IDs (label_asym_ids).

    The chains' auth_asym_ids will be updated to be identical to the chain ID
    since there isn't one unambiguous way to maintain the auth_asym_ids after
    renaming the chain IDs (depending on whether you view the auth_asym_id as
    more strongly associated with a given physical chain, or with a given
    chain ID).

    The residues' auth_seq_id will be updated to be identical to the residue ID
    since they are strongly tied to the original author chain naming and keeping
    them would be misleading.

    Args:
      struc:self
      new_id_by_old_id: A mapping from original chain ID to their new values.
        Any chain IDs in this structure that are not in this mapping will remain
        unchanged.

    Returns:
      A new structure with renamed chains (and bioassembly data if it is
      present).

    Raises:
      ValueError: If any two previously distinct chains do not have unique names
        anymore after the rename.
    """
    new_chain_id = string_array.remap(struc.chains_table.id, new_id_by_old_id)
    if len(new_chain_id) != len(set(new_chain_id)):
        raise ValueError(f"New chain names aren't unique: {sorted(new_chain_id)}")

    # Map label_asym_ids in the bioassembly data.
    if struc.bioassembly_data is None:
        new_bioassembly_data = None
    else:
        new_bioassembly_data = struc.bioassembly_data.rename_label_asym_ids(
            new_id_by_old_id, present_chains=set(struc.present_chains.id)
        )

    # Set author residue IDs to be the string version of internal residue IDs.
    new_residues = struc.residues_table.copy_and_update(
        auth_seq_id=struc.residues_table.id.astype(str).astype(object)
    )

    new_chains = struc.chains_table.copy_and_update(
        id=new_chain_id, auth_asym_id=new_chain_id
    )

    return struc.copy_and_update(
        bioassembly_data=new_bioassembly_data,
        chains=new_chains,
        residues=new_residues,
        skip_validation=True,
    )


def structure_reorder_chains(struc, new_order: Sequence[str]):
    """Reorders tables so that the label_asym_ids are in the given order.

    This method changes the order of the chains, residues, and atoms tables so
    that they are all consistent with each other. Moreover, it remaps chain keys
    so that they stay monotonically increasing in chains/residues/atoms tables.

    Args:
      struc:self
      new_order: The order in which the chain IDs (label_asym_id) should be.
        This must be a permutation of the current chain IDs.

    Returns:
      A structure with chains reorded.
    """
    if len(new_order) != len(struc.chains_table):
        raise ValueError(f'Chain count mismatch: {len(new_order)} vs {len(struc.chains_table)}')

    new_chain_set = set(new_order)
    if len(new_chain_set) != len(new_order):
        raise ValueError(f'The new order {new_order} contains non-unique IDs.')
    if new_chain_set.symmetric_difference(set(struc.chains_table)):
        raise ValueError(f'New chain IDs do not match the old ones.')

    if struc.chains_table == tuple(new_order):
        return struc  # Shortcut: the new order is the same as the current one.

    desired_chain_id_pos = {chain_id: i for i, chain_id in enumerate(new_order)}
    current_chain_index_order = np.empty(struc.num_chains, dtype=np.int64)
    for index, old_chain_id in enumerate(struc.chains_table.id):
        current_chain_index_order[index] = desired_chain_id_pos[old_chain_id]

    chain_reorder = np.argsort(current_chain_index_order, kind='stable')
    chain_key_map = dict(zip(struc.chains_table.key[chain_reorder], range(struc.num_chains)))

    # The stable sort keeps the original residue ordering within each chain.
    chains = struc.chains_table.apply_index(chain_reorder).copy_and_remap(key=chain_key_map)
    residues = struc.residues_table.copy_and_remap(chain_key=chain_key_map)
    residue_reorder = np.argsort(residues.chain_key, kind='stable')
    residues = residues.apply_index(residue_reorder)

    # The stable sort keeps the original atom ordering within each chain.
    atoms = struc.atoms_table.copy_and_remap(chain_key=chain_key_map)
    atoms_reorder = np.argsort(atoms.chain_key, kind='stable')
    atoms = atoms.apply_index(atoms_reorder)

    # Bonds unchanged - each references 2 atom keys, hence ordering not defined.
    return struc.copy_and_update(chains=chains, residues=residues, atoms=atoms)


def get_chain_single_letter_sequence(
        struc, include_missing_residues: bool = True
) -> Mapping[str, str]:
    """Returns a mapping from chain ID to a single letter residue sequence.

    Args:
      struc:self
      include_missing_residues: Whether to include residues that have no atoms.
    """
    res_table = (
        struc.residues_table if include_missing_residues else struc.present_residues
    )
    residue_chain_boundaries = get_change_indices(res_table.chain_key)
    boundaries = struc.iter_residue_ranges(
        residue_chain_boundaries,
        count_unresolved=include_missing_residues,
    )
    chain_keys = res_table.chain_key[residue_chain_boundaries]
    chain_ids = struc.chains_table.apply_array_to_column('id', chain_keys)
    chain_types = struc.chains_table.apply_array_to_column('type', chain_keys)
    chain_seqs = {}
    for idx, (start, end) in enumerate(boundaries):
        chain_id = chain_ids[idx]
        chain_type = chain_types[idx]
        chain_res = res_table.name[start:end]
        if chain_type in mmcif_names.PEPTIDE_CHAIN_TYPES:
            unknown_default = 'X'
        elif chain_type in mmcif_names.NUCLEIC_ACID_CHAIN_TYPES:
            unknown_default = 'N'
        else:
            chain_seqs[chain_id] = 'X' * chain_res.size
            continue

        chain_res = string_array.remap(
            chain_res,
            mapping=residue_names.CCD_NAME_TO_ONE_LETTER,
            inplace=False,
            default_value=unknown_default,
        )
        chain_seqs[chain_id] = ''.join(chain_res.tolist())

    return chain_seqs
