import numpy as np
import typing
from typing import Mapping
from torchfold.cpp import membership
from torchfold.structure.internal import table
from torchfold.structure.internal.constants import (
    CHAIN_FIELDS, RESIDUE_FIELDS, ATOM_FIELDS, CascadeDelete
)
from torchfold.structure import tables as structure_tables
from torchfold.constants import mmcif_names


def _unpack_filter_predicates(
        predicate_by_field_name: Mapping[str, table.FilterPredicate],
) -> tuple[
    Mapping[str, table.FilterPredicate],
    Mapping[str, table.FilterPredicate],
    Mapping[str, table.FilterPredicate],
]:
    """Unpacks filter kwargs into predicates for each table."""
    chain_predicates = {}
    res_predicates = {}
    atom_predicates = {}
    for k, pred in predicate_by_field_name.items():
        if col := CHAIN_FIELDS.get(k):
            chain_predicates[col] = pred
        elif col := RESIDUE_FIELDS.get(k):
            res_predicates[col] = pred
        elif col := ATOM_FIELDS.get(k):
            atom_predicates[col] = pred
        else:
            raise ValueError(k)
    return chain_predicates, res_predicates, atom_predicates


def cascade_delete(
        struc,
        *,
        chains: structure_tables.Chains | None = None,
        residues: structure_tables.Residues | None = None,
        atoms: structure_tables.Atoms | None = None,
        bonds: structure_tables.Bonds | None = None,
):
    """Performs a cascade delete operation on the structure's tables.

    Cascade delete ensures all the tables are consistent after any table fields
    are being updated by cascading any deletions down the hierarchy of tables:
    chains > residues > atoms > bonds.

    E.g.: if a row from residues table is removed then all the atoms in that
    residue will also be removed from the atoms table. In turn this cascades
    also to the bond table, by removing any bond row which involves any of those
    removed atoms. However the chains table will not be modified, even if
    that was the only residue in its chain, because the chains table is above
    the residues table in the hierarchy.

    Args:
      chains: An optional new chains table.
      residues: An optional new residues table.
      atoms: An optional new atoms table.
      bonds: An optional new bonds table.

    Returns:
      A StructureTables object with the updated tables.
    """
    from torchfold.structure.models import StructureTables
    if chains_unchanged := chains is None:
        chains = struc.chains_table
    if residues_unchanged := residues is None:
        residues = struc.residues_table
    if atoms_unchanged := atoms is None:
        atoms = struc.atoms_table
    if bonds is None:
        bonds = struc.bonds_table

    if not chains_unchanged:
        residues_mask = membership.isin(residues.chain_key, set(chains.key))  # pylint:disable=attribute-error
        if not np.all(residues_mask):  # Only apply if this is not a no-op.
            residues = residues[residues_mask]
            residues_unchanged = False
    if not residues_unchanged:
        atoms_mask = membership.isin(atoms.res_key, set(residues.key))  # pylint:disable=attribute-error
        if not np.all(atoms_mask):  # Only apply if this is not a no-op.
            atoms = atoms[atoms_mask]
            atoms_unchanged = False
    if not atoms_unchanged:
        bonds = bonds.restrict_to_atoms(atoms.key)
    return StructureTables(
        chains=chains, residues=residues, atoms=atoms, bonds=bonds
    )


def structure_filter(
        struc,
        mask: np.ndarray | None = None,
        *,
        apply_per_element: bool = False,
        invert: bool = False,
        cascade_delete_behavior: CascadeDelete = CascadeDelete.CHAINS,
        **predicate_by_field_name: table.FilterPredicate,
):
    """Filters the structure by field values and returns a new structure.

    Predicates are specified as keyword arguments, with names following the
    pattern: <table_name>_<col_name>, where table_name := (chain|res|atom).
    For instance the auth_seq_id column in the residues table can be filtered
    by passing `res_auth_seq_id=pred_value`. The full list of valid options
    are defined in the `col_by_field_name` fields on the different Table
    dataclasses.

    Predicate values can be either:
      1. A constant value, e.g. 'CA'. In this case then only rows that match
        this value for the given field are retained.
      2. A (non-string) iterable e.g. ('A', 'B'). In this
        case then rows are retained if they match any of the provided values for
        the given field.
      3. A boolean function e.g. lambda b_fac: b_fac < 100.0.
        In this case then only rows that evaluate to True are retained. By
        default this function's parameter is expected to be an array, unless
        apply_per_element=True.

    Example usage:
      # Filter to backbone atoms in residues up to 100 in chain B.
      filtered_struc = struc.filter(
          chain_id='B',
          atom_name=('N', 'CA', 'C'),
          res_id=lambda res_id: res_id < 100)

    Example usage where predicate must be applied per-element:
      # Filter to residues with IDs in either [1, 100) or [300, 400).
      ranges = ((1, 100), (300, 400))
      filtered_struc = struc.filter(
          res_id=lambda i: np.any([start <= i < end for start, end in ranges]),
          apply_per_element=True)

    Example usage of providing a raw mask:
      filtered_struc = struc.filter(struc.atom_b_factor < 10.0)

    Args:
      mask: An optional boolean NumPy array with length equal to num_atoms. If
        provided then this will be combined with the other predicates so that an
        atom is included if it is masked-in *and* matches all the predicates.
      apply_per_element: Whether apply predicates to each element individually,
        or to pass the whole column array to the predicate.
      invert: Whether to remove, rather than retain, the entities which match
        the specified predicates.
      cascade_delete: Whether to remove residues and chains which are left
        unresolved in a cascade. filter operates on the atoms table, removing
        atoms which match the predicate. If all atoms in a residue are removed,
        the residue is "unresolved". The value of this argument then determines
        whether such residues and their parent chains should be deleted. FULL
        implies that all unresolved residues should be deleted, and any chains
        which are left with no resolved residues should be deleted. CHAINS is
        the default behaviour - only chains with no resolved residues, and their
        child residues are deleted. Unresolved residues in partially resolved
        chains remain. NONE implies that no unresolved residues or chains should
        be deleted.
      **predicate_by_field_name: A mapping from field name to a predicate.
        Filtered columns must be 1D arrays. If multiple fields are provided as
        keyword arguments then each predicate is applied and the results are
        combined using a boolean AND operation, so an atom is only retained if
        it passes all predicates.

    Returns:
      A new structure representing a filtered version of the current structure.

    Raises:
      ValueError: If mask is provided and is not a bool array with shape
        (num_atoms,).
    """
    chain_predicates, res_predicates, atom_predicates = (
        _unpack_filter_predicates(predicate_by_field_name)
    )
    # Get boolean masks for each table. These are None if none of the filter
    # parameters affect the table in question.
    chain_mask = struc.chains_table.make_filter_mask(
        **chain_predicates, apply_per_element=apply_per_element
    )
    res_mask = struc.residues_table.make_filter_mask(
        **res_predicates, apply_per_element=apply_per_element
    )
    atom_mask = struc.atoms_table.make_filter_mask(
        mask, **atom_predicates, apply_per_element=apply_per_element
    )
    if atom_mask is None:
        atom_mask = np.ones((struc.atoms_table.size,), dtype=bool)

    # Remove atoms that belong to filtered out chains.
    if chain_mask is not None:
        atom_chain_mask = membership.isin(
            struc.atoms_table.chain_key, set(struc.chains_table.key[chain_mask])
        )
        np.logical_and(atom_mask, atom_chain_mask, out=atom_mask)

    # Remove atoms that belong to filtered out residues.
    if res_mask is not None:
        atom_res_mask = membership.isin(
            struc.atoms_table.res_key, set(struc.residues_table.key[res_mask])
        )
        np.logical_and(atom_mask, atom_res_mask, out=atom_mask)

    final_atom_mask = ~atom_mask if invert else atom_mask

    if cascade_delete_behavior == CascadeDelete.NONE and np.all(final_atom_mask):
        return struc  # Shortcut: The filter is a no-op, so just return itself.

    filtered_atoms = typing.cast(
        structure_tables.Atoms, struc.atoms_table[final_atom_mask]
    )

    match cascade_delete_behavior:
        case CascadeDelete.FULL:
            nonempty_residues_mask = np.isin(struc.residues_table.key, filtered_atoms.res_key)
            filtered_residues = struc.residues_table[nonempty_residues_mask]
            nonempty_chain_mask = np.isin(struc.chains_table.key, filtered_atoms.chain_key)
            filtered_chains = struc.chains_table[nonempty_chain_mask]
            updated_tables = cascade_delete(
                struc,
                chains=filtered_chains,
                residues=filtered_residues,
                atoms=filtered_atoms,
            )
        case CascadeDelete.CHAINS:
            # To match v1 behavior we remove chains that have no atoms remaining,
            # and we remove residues in those chains.
            # NB we do not remove empty residues.
            nonempty_chain_mask = membership.isin(
                struc.chains_table.key, set(filtered_atoms.chain_key)
            )
            filtered_chains = struc.chains_table[nonempty_chain_mask]
            updated_tables = cascade_delete(struc, chains=filtered_chains, atoms=filtered_atoms)
        case CascadeDelete.NONE:
            updated_tables = cascade_delete(struc, atoms=filtered_atoms)
        case _:
            raise ValueError(f'Unknown cascade_delete behaviour: {cascade_delete_behavior}')
    return struc.copy_and_update(
        chains=updated_tables.chains,
        residues=updated_tables.residues,
        atoms=updated_tables.atoms,
        bonds=updated_tables.bonds,
        skip_validation=True,
    )


def filter_to_entity_type(
        struc,
        *,
        protein: bool = False,
        rna: bool = False,
        dna: bool = False,
        dna_rna_hybrid: bool = False,
        ligand: bool = False,
        water: bool = False,
):
    """Filters the structure to only include the selected entity types.

    This convenience method abstracts away the specifics of mmCIF entity
    type names which, especially for ligands, are non-trivial.

    Args:
      protein: Whether to include protein (polypeptide(L)) chains.
      rna: Whether to include RNA chains.
      dna: Whether to include DNA chains.
      dna_rna_hybrid: Whether to include DNA RNA hybrid chains.
      ligand: Whether to include ligand (i.e. not polymer) chains.
      water: Whether to include water chains.

    Returns:
      The filtered structure.
    """
    include_types = []
    if protein:
        include_types.append(mmcif_names.PROTEIN_CHAIN)
    if rna:
        include_types.append(mmcif_names.RNA_CHAIN)
    if dna:
        include_types.append(mmcif_names.DNA_CHAIN)
    if dna_rna_hybrid:
        include_types.append(mmcif_names.DNA_RNA_HYBRID_CHAIN)
    if ligand:
        include_types.extend(mmcif_names.LIGAND_CHAIN_TYPES)
    if water:
        include_types.append(mmcif_names.WATER)
    return structure_filter(struc, chain_type=include_types)
