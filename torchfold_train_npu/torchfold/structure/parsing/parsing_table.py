import itertools
from collections.abc import Mapping, Sequence

import numpy as np
from torchfold.cpp import mmcif_utils, string_array

from torchfold.constants import mmcif_names
from torchfold.structure import bonds, mmcif
from torchfold.structure.tables import Chains, Residues, Atoms
from torchfold.structure.parsing.parsing_builder import _ChainResBuilder
from torchfold.structure.parsing.parsing_constants import _INSERTION_CODE_REMAP
from torchfold.structure.parsing.parsing_utils import _guess_entity_type


def get_tables(
        cif: mmcif.Mmcif,
        fix_mse_residues: bool,
        fix_arginines: bool,
        fix_unknown_dna: bool,
        include_water: bool,
        include_other: bool,
        model_id: str,
) -> tuple[
    Chains, Residues, Atoms
]:
    """Returns chain, residue, and atom tables from a parsed mmcif.

    Args:
      cif: A parsed mmcif.Mmcif.
      fix_mse_residues: See from_mmcif.
      fix_arginines: See from_mmcif.
      fix_unknown_dna: See from_mmcif.
      include_water: See from_mmcif.
      include_other: See from_mmcif.
      model_id: A string defining which model ID to use. If set, only coordinates,
        b-factors and occupancies for the given model are returned. If empty,
        coordinates, b-factors and occupanciesall for models are returned with a
        leading dimension of num_models. Note that the model_id argument in
        from_mmcif is an integer and has slightly different use (see from_mmcif).
    """
    # Add any missing tables and columns we require for parsing.
    if cif_update := _generate_required_tables_if_missing(cif):
        cif = cif.copy_and_update(cif_update)

    # Resolve alt-locs, selecting only a single option for each residue. Also
    # computes the layout, which defines where chain and residue boundaries are.
    atom_site_all_models, layout = mmcif_utils.filter(
        cif,
        include_nucleotides=True,
        include_ligands=True,
        include_water=include_water,
        include_other=include_other,
        model_id=model_id,
    )
    atom_site_first_model = atom_site_all_models[0]

    # Get atom information from the _atom_site table.
    def _first_model_string_array(col: str) -> np.ndarray:
        return cif.get_array(col, dtype=object, gather=atom_site_first_model)

    def _requested_models_float_array(col: str) -> np.ndarray:
        if not model_id:
            # Return data for all models with a leading dimension of num_models.
            return cif.get_array(col, dtype=np.float32, gather=atom_site_all_models)
        else:
            # Return data only for the single requested model.
            return cif.get_array(col, dtype=np.float32, gather=atom_site_first_model)

    # These columns are the same for all models, fetch them just for the 1st one.
    label_comp_ids = _first_model_string_array('_atom_site.label_comp_id')
    label_asym_ids = _first_model_string_array('_atom_site.label_asym_id')
    label_seq_ids = _first_model_string_array('_atom_site.label_seq_id')
    label_atom_ids = _first_model_string_array('_atom_site.label_atom_id')
    if '_atom_site.auth_seq_id' in cif:
        auth_seq_ids = _first_model_string_array('_atom_site.auth_seq_id')
    else:
        auth_seq_ids = label_seq_ids  # auth_seq_id unset, fallback to label_seq_id.
    type_symbols = _first_model_string_array('_atom_site.type_symbol')
    pdbx_pdb_ins_codes = _first_model_string_array('_atom_site.pdbx_PDB_ins_code')

    # These columns are different for all models, fetch them as requested.
    atom_x = _requested_models_float_array('_atom_site.Cartn_x')
    atom_y = _requested_models_float_array('_atom_site.Cartn_y')
    atom_z = _requested_models_float_array('_atom_site.Cartn_z')
    atom_b_factor = _requested_models_float_array('_atom_site.B_iso_or_equiv')
    atom_occupancy = _requested_models_float_array('_atom_site.occupancy')

    # Make sure the scheme (residue) tables exist in case they are not present.
    if cif_update := _maybe_add_missing_scheme_tables(
            cif,
            res_starts=layout.residue_starts(),
            label_asym_ids=label_asym_ids,
            label_seq_ids=label_seq_ids,
            label_comp_ids=label_comp_ids,
            auth_seq_ids=auth_seq_ids,
            pdb_ins_codes=pdbx_pdb_ins_codes,
    ):
        cif = cif.copy_and_update(cif_update)

    # Fix common issues found in mmCIF files, like swapped arginine NH atoms.
    mmcif_utils.fix_residues(
        layout,
        comp_id=label_comp_ids,
        atom_id=label_atom_ids,
        atom_x=atom_x[0] if not model_id else atom_x,
        atom_y=atom_y[0] if not model_id else atom_y,
        atom_z=atom_z[0] if not model_id else atom_z,
        fix_arg=fix_arginines,
    )

    # Get keys for chains in the order they appear in _atom_site while also
    # dealing with empty chains.
    resolved_chain_ids = label_asym_ids[layout.chain_starts()]
    struct_asym_chain_ids = cif.get_array('_struct_asym.id', dtype=object)

    chain_key_by_chain_id = _get_chain_key_by_chain_id(
        resolved_chain_ids=resolved_chain_ids,
        struct_asym_chain_ids=struct_asym_chain_ids,
    )
    entity_id_by_chain_id = dict(
        zip(struct_asym_chain_ids, cif['_struct_asym.entity_id'], strict=True)
    )
    entity_description = cif.get(
        '_entity.pdbx_description', ['?'] * len(cif['_entity.id'])
    )
    entity_desc_by_entity_id = dict(
        zip(cif['_entity.id'], entity_description, strict=True)
    )
    chain_type_by_entity_id = mmcif.get_chain_type_by_entity_id(cif)
    auth_asym_id_by_chain_id = mmcif.get_internal_to_author_chain_id_map(cif)

    chain_res_builder = _ChainResBuilder(
        chain_key_by_chain_id=chain_key_by_chain_id,
        entity_id_by_chain_id=entity_id_by_chain_id,
        chain_type_by_entity_id=chain_type_by_entity_id,
        entity_desc_by_entity_id=entity_desc_by_entity_id,
        fix_mse_residues=fix_mse_residues,
        fix_unknown_dna=fix_unknown_dna,
    )

    # Collect data for polymer chain and residue tables. _pdbx_poly_seq_scheme is
    # guaranteed to be present thanks to _maybe_add_missing_scheme_tables.
    def _get_poly_seq_scheme_col(col: str) -> np.ndarray:
        return cif.get_array(key=f'_pdbx_poly_seq_scheme.{col}', dtype=object)

    poly_seq_asym_ids = _get_poly_seq_scheme_col('asym_id')
    poly_seq_pdb_seq_nums = _get_poly_seq_scheme_col('pdb_seq_num')
    poly_seq_seq_ids = _get_poly_seq_scheme_col('seq_id')
    poly_seq_mon_ids = _get_poly_seq_scheme_col('mon_id')
    poly_seq_pdb_strand_ids = _get_poly_seq_scheme_col('pdb_strand_id')
    poly_seq_pdb_ins_codes = _get_poly_seq_scheme_col('pdb_ins_code')
    string_array.remap(
        poly_seq_pdb_ins_codes, mapping=_INSERTION_CODE_REMAP, inplace=True
    )

    # We resolved alt-locs earlier for the atoms table. In cases of heterogeneous
    # residues (a residue with an alt-loc that is of different residue type), we
    # need to also do the same resolution in the residues table. Compute a mask
    # for the residues that were selected in the atoms table.
    poly_seq_mask = mmcif_utils.selected_polymer_residue_mask(
        layout=layout,
        atom_site_label_asym_ids=label_asym_ids[layout.residue_starts()],
        atom_site_label_seq_ids=label_seq_ids[layout.residue_starts()],
        atom_site_label_comp_ids=label_comp_ids[layout.residue_starts()],
        poly_seq_asym_ids=poly_seq_asym_ids,
        poly_seq_seq_ids=poly_seq_seq_ids,
        poly_seq_mon_ids=poly_seq_mon_ids,
    )

    if not include_other and poly_seq_mask:
        # Mask filtered-out residues so that they are not treated as missing.
        # Instead, we don't want them included in the chains/residues tables at all.
        keep_mask = string_array.remap(
            poly_seq_asym_ids,
            mapping={cid: True for cid in resolved_chain_ids},
            default_value=False,
            inplace=False,
        ).astype(bool)
        poly_seq_mask &= keep_mask

    chain_res_builder.add_residues(
        chain_ids=poly_seq_asym_ids[poly_seq_mask],
        chain_auth_asym_ids=poly_seq_pdb_strand_ids[poly_seq_mask],
        res_ids=poly_seq_seq_ids[poly_seq_mask].astype(np.int32),
        res_names=poly_seq_mon_ids[poly_seq_mask],
        res_auth_seq_ids=poly_seq_pdb_seq_nums[poly_seq_mask],
        res_ins_codes=poly_seq_pdb_ins_codes[poly_seq_mask],
    )

    # Collect data for ligand chain and residue tables. _pdbx_nonpoly_scheme
    # could be empty/unset if there are only branched ligands.
    def _get_nonpoly_scheme_col(col: str) -> np.ndarray:
        key = f'_pdbx_nonpoly_scheme.{col}'
        if f'_pdbx_nonpoly_scheme.{col}' in cif:
            return cif.get_array(key=key, dtype=object)
        else:
            return np.array([], dtype=object)

    nonpoly_asym_ids = _get_nonpoly_scheme_col('asym_id')
    nonpoly_auth_seq_ids = _get_nonpoly_scheme_col('pdb_seq_num')
    nonpoly_pdb_ins_codes = _get_nonpoly_scheme_col('pdb_ins_code')
    nonpoly_mon_ids = _get_nonpoly_scheme_col('mon_id')
    nonpoly_auth_asym_id = string_array.remap(
        nonpoly_asym_ids, mapping=auth_asym_id_by_chain_id, inplace=False
    )

    def _get_branch_scheme_col(col: str) -> np.ndarray:
        key = f'_pdbx_branch_scheme.{col}'
        if f'_pdbx_branch_scheme.{col}' in cif:
            return cif.get_array(key=key, dtype=object)
        else:
            return np.array([], dtype=object)

    branch_asym_ids = _get_branch_scheme_col('asym_id')
    branch_auth_seq_ids = _get_branch_scheme_col('pdb_seq_num')
    branch_pdb_ins_codes = _get_branch_scheme_col('pdb_ins_code')
    branch_mon_ids = _get_branch_scheme_col('mon_id')
    branch_auth_asym_id = string_array.remap(
        branch_asym_ids, mapping=auth_asym_id_by_chain_id, inplace=False
    )

    if branch_asym_ids.size > 0 and branch_pdb_ins_codes.size == 0:
        branch_pdb_ins_codes = np.array(['.'] * branch_asym_ids.size, dtype=object)

    # Compute the heterogeneous residue masks as above, this time for ligands.
    nonpoly_mask, branch_mask = mmcif_utils.selected_ligand_residue_mask(
        layout=layout,
        atom_site_label_asym_ids=label_asym_ids[layout.residue_starts()],
        atom_site_label_seq_ids=label_seq_ids[layout.residue_starts()],
        atom_site_auth_seq_ids=auth_seq_ids[layout.residue_starts()],
        atom_site_label_comp_ids=label_comp_ids[layout.residue_starts()],
        atom_site_pdbx_pdb_ins_codes=pdbx_pdb_ins_codes[layout.residue_starts()],
        nonpoly_asym_ids=nonpoly_asym_ids,
        nonpoly_auth_seq_ids=nonpoly_auth_seq_ids,
        nonpoly_pdb_ins_codes=nonpoly_pdb_ins_codes,
        nonpoly_mon_ids=nonpoly_mon_ids,
        branch_asym_ids=branch_asym_ids,
        branch_auth_seq_ids=branch_auth_seq_ids,
        branch_pdb_ins_codes=branch_pdb_ins_codes,
        branch_mon_ids=branch_mon_ids,
    )

    if not include_water:
        if nonpoly_mask:
            nonpoly_mask &= (nonpoly_mon_ids != 'HOH') & (nonpoly_mon_ids != 'DOD')
        if branch_mask:
            # Fix for bad mmCIFs that have water in the branch scheme table.
            branch_mask &= (branch_mon_ids != 'HOH') & (branch_mon_ids != 'DOD')

    string_array.remap(
        pdbx_pdb_ins_codes, mapping=_INSERTION_CODE_REMAP, inplace=True
    )
    string_array.remap(
        nonpoly_pdb_ins_codes, mapping=_INSERTION_CODE_REMAP, inplace=True
    )
    string_array.remap(
        branch_pdb_ins_codes, mapping=_INSERTION_CODE_REMAP, inplace=True
    )

    def _ligand_residue_ids(chain_ids: np.ndarray) -> np.ndarray:
        """Computes internal residue ID for ligand residues that don't have it."""

        # E.g. chain_ids=[A, A, A, B, C, C, D, D, D] -> [1, 2, 3, 1, 1, 2, 1, 2, 3].
        indices = np.arange(chain_ids.size, dtype=np.int32)
        return (indices + 1) - np.maximum.accumulate(
            indices * (chain_ids != np.roll(chain_ids, 1))
        )

    branch_residue_ids = _ligand_residue_ids(branch_asym_ids[branch_mask])
    nonpoly_residue_ids = _ligand_residue_ids(nonpoly_asym_ids[nonpoly_mask])

    chain_res_builder.add_residues(
        chain_ids=branch_asym_ids[branch_mask],
        chain_auth_asym_ids=branch_auth_asym_id[branch_mask],
        res_ids=branch_residue_ids,
        res_names=branch_mon_ids[branch_mask],
        res_auth_seq_ids=branch_auth_seq_ids[branch_mask],
        res_ins_codes=branch_pdb_ins_codes[branch_mask],
    )

    chain_res_builder.add_residues(
        chain_ids=nonpoly_asym_ids[nonpoly_mask],
        chain_auth_asym_ids=nonpoly_auth_asym_id[nonpoly_mask],
        res_ids=nonpoly_residue_ids,
        res_names=nonpoly_mon_ids[nonpoly_mask],
        res_auth_seq_ids=nonpoly_auth_seq_ids[nonpoly_mask],
        res_ins_codes=nonpoly_pdb_ins_codes[nonpoly_mask],
    )

    chains = chain_res_builder.make_chains_table()
    residues = chain_res_builder.make_residues_table()

    # Construct foreign residue keys for the atoms table.
    res_ends = np.array(layout.residues(), dtype=np.int32)
    res_starts = np.array(layout.residue_starts(), dtype=np.int32)
    res_lengths = res_ends - res_starts

    # Check just for HOH, DOD can be part e.g. of hydroxycysteine.
    if include_water:
        res_chain_types = chains.apply_array_to_column(
            column_name='type', arr=residues.chain_key
        )
        water_mask = res_chain_types != mmcif_names.WATER
        if 'HOH' in set(residues.name[water_mask]):
            raise ValueError('Bad mmCIF file: non-water entity has water molecules.')
    else:
        # Include resolved and unresolved residues.
        if 'HOH' in set(residues.name) | set(label_comp_ids[res_starts]):
            raise ValueError('Bad mmCIF file: non-water entity has water molecules.')

    atom_chain_key = string_array.remap(
        label_asym_ids, mapping=chain_res_builder.chain_key_by_chain_id
    ).astype(int)

    # If any of the residue lookups failed, the mmCIF is corrupted.
    try:
        atom_res_key_per_res = string_array.remap_multiple(
            (
                label_asym_ids[res_starts],
                auth_seq_ids[res_starts],
                label_comp_ids[res_starts],
                pdbx_pdb_ins_codes[res_starts],
            ),
            mapping=chain_res_builder.key_for_res,
        )
    except KeyError as e:
        raise ValueError(
            'Lookup for the following atom from the _atom_site table failed: '
            f'(atom_id, auth_seq_id, res_name, ins_code)={e}. This is '
            'likely due to a known issue with some multi-model mmCIFs that only '
            'match the first model in _atom_site table to the _pdbx_poly_scheme, '
            '_pdbx_nonpoly_scheme, or _pdbx_branch_scheme tables.'
        ) from e

    # The residue ID will be shared for all atoms within that residue.
    atom_res_key = np.repeat(atom_res_key_per_res, repeats=res_lengths)

    if fix_mse_residues:
        met_residues_mask = (residues.name == 'MET')[atom_res_key]
        unfixed_mse_selenium_mask = met_residues_mask & (label_atom_ids == 'SE')
        label_atom_ids[unfixed_mse_selenium_mask] = 'SD'
        type_symbols[unfixed_mse_selenium_mask] = 'S'

    atoms = Atoms(
        key=atom_site_first_model,
        chain_key=atom_chain_key,
        res_key=atom_res_key,
        name=label_atom_ids,
        element=type_symbols,
        x=atom_x,
        y=atom_y,
        z=atom_z,
        b_factor=atom_b_factor,
        occupancy=atom_occupancy,
    )

    return chains, residues, atoms


def _generate_required_tables_if_missing(
        cif: mmcif.Mmcif,
) -> Mapping[str, Sequence[str]]:
    """Generates all required tables and columns if missing."""
    update = {}

    atom_site_entities = _get_string_array_default(
        cif, '_atom_site.label_entity_id', []
    )

    # OpenMM produces files that don't have any of the tables and also have
    # _atom_site.label_entity_id set to '?' for all atoms. We infer the entities
    # based on the _atom_site.label_asym_id column. We start with cheaper O(1)
    # checks to prevent running the expensive O(n) check on most files.
    if (
            len(atom_site_entities) > 0  # pylint: disable=g-explicit-length-test
            and '_entity.id' not in cif  # Ignore if the _entity table exists.
            and atom_site_entities[0] == '?'  # Cheap check.
            and set(atom_site_entities) == {'?'}  # Expensive check.
    ):
        label_asym_ids = cif.get_array('_atom_site.label_asym_id', dtype=object)
        atom_site_entities = [
            str(mmcif.str_id_to_int_id(cid)) for cid in label_asym_ids
        ]
        # Update _atom_site.label_entity_id to be consistent with the new tables.
        update['_atom_site.label_entity_id'] = atom_site_entities

    # Check table existence by checking the presence of its primary key.
    if '_struct_asym.id' not in cif:
        # Infer the _struct_asym table using the _atom_site table.
        asym_ids = _get_string_array_default(cif, '_atom_site.label_asym_id', [])

        if len(atom_site_entities) == 0 or len(asym_ids) == 0:  # pylint: disable=g-explicit-length-test
            raise ValueError(
                'Could not parse an mmCIF with no _struct_asym table and also no '
                '_atom_site.label_entity_id or _atom_site.label_asym_id columns.'
            )

        # Deduplicate, but keep the order intact - dict.fromkeys maintains order.
        entity_id_chain_id_pairs = list(
            dict.fromkeys(zip(atom_site_entities, asym_ids, strict=True))
        )
        update['_struct_asym.entity_id'] = [e for e, _ in entity_id_chain_id_pairs]
        update['_struct_asym.id'] = [c for _, c in entity_id_chain_id_pairs]

    if '_entity.id' not in cif:
        # Infer the _entity_poly and _entity tables using the _atom_site table.
        residues = _get_string_array_default(cif, '_atom_site.label_comp_id', [])
        group_pdb = _get_string_array_default(cif, '_atom_site.group_PDB', [])
        if '_atom_site.label_entity_id' in cif:
            entities = atom_site_entities
        else:
            # If _atom_site.label_entity_id not set, use the asym_id -> entity_id map.
            asym_to_entity = dict(
                zip(
                    cif['_struct_asym.id'], cif['_struct_asym.entity_id'], strict=True
                )
            )
            entities = string_array.remap(
                cif.get_array('_atom_site.label_asym_id', dtype=object),
                mapping=asym_to_entity,
            )

        entity_ids = []
        entity_types = []
        entity_poly_entity_ids = []
        entity_poly_types = []
        entity_poly_table_missing = '_entity_poly.entity_id' not in cif
        for entity_id, group in itertools.groupby(
                zip(entities, residues, group_pdb, strict=True), key=lambda e: e[0]
        ):
            _, entity_residues, entity_group_pdb = zip(*group, strict=True)
            entity_type = _guess_entity_type(
                chain_residues=entity_residues, atom_types=entity_group_pdb
            )
            entity_ids.append(entity_id)
            entity_types.append(entity_type)

            if entity_poly_table_missing and entity_type == mmcif_names.POLYMER_CHAIN:
                polymer_type = mmcif_names.guess_polymer_type(entity_residues)
                entity_poly_entity_ids.append(entity_id)
                entity_poly_types.append(polymer_type)

        update['_entity.id'] = entity_ids
        update['_entity.type'] = entity_types
        if entity_poly_table_missing:
            update['_entity_poly.entity_id'] = entity_poly_entity_ids
            update['_entity_poly.type'] = entity_poly_types

    if '_atom_site.type_symbol' not in cif:
        update['_atom_site.type_symbol'] = mmcif.get_or_infer_type_symbol(cif)

    return update


def _get_string_array_default(cif: mmcif.Mmcif, key: str, default: list[str]):
    try:
        return cif.get_array(key, dtype=object)
    except KeyError:
        return default


def _maybe_add_missing_scheme_tables(
        cif: mmcif.Mmcif,
        res_starts: Sequence[int],
        label_asym_ids: np.ndarray,
        label_seq_ids: np.ndarray,
        label_comp_ids: np.ndarray,
        auth_seq_ids: np.ndarray,
        pdb_ins_codes: np.ndarray,
) -> Mapping[str, Sequence[str]]:
    """If missing, infers the scheme tables from the _atom_site table."""
    update = {}

    required_poly_seq_scheme_cols = (
        '_pdbx_poly_seq_scheme.asym_id',
        '_pdbx_poly_seq_scheme.pdb_seq_num',
        '_pdbx_poly_seq_scheme.pdb_ins_code',
        '_pdbx_poly_seq_scheme.seq_id',
        '_pdbx_poly_seq_scheme.mon_id',
        '_pdbx_poly_seq_scheme.pdb_strand_id',
    )
    if not all(col in cif for col in required_poly_seq_scheme_cols):
        # Create a mask for atoms where each polymer residue start.
        entity_id_by_chain_id = dict(
            zip(cif['_struct_asym.id'], cif['_struct_asym.entity_id'], strict=True)
        )
        chain_type_by_entity_id = dict(
            zip(cif['_entity.id'], cif['_entity.type'], strict=True)
        )
        # Remap asym ID -> entity ID.
        chain_type = string_array.remap(
            label_asym_ids, mapping=entity_id_by_chain_id, inplace=False
        )
        # Remap entity ID -> chain type.
        string_array.remap(
            chain_type, mapping=chain_type_by_entity_id, inplace=True
        )
        res_mask = np.zeros_like(label_seq_ids, dtype=bool)
        res_mask[res_starts] = True
        res_mask &= chain_type == mmcif_names.POLYMER_CHAIN

        entity_poly_seq_cols = (
            '_entity_poly_seq.entity_id',
            '_entity_poly_seq.num',
            '_entity_poly_seq.mon_id',
        )
        if all(col in cif for col in entity_poly_seq_cols):
            # Use _entity_poly_seq if available.
            poly_seq_num = cif.get_array('_entity_poly_seq.num', dtype=object)
            poly_seq_mon_id = cif.get_array('_entity_poly_seq.mon_id', dtype=object)
            poly_seq_entity_id = cif.get_array(
                '_entity_poly_seq.entity_id', dtype=object
            )
            label_seq_id_to_auth_seq_id = dict(
                zip(label_seq_ids[res_mask], auth_seq_ids[res_mask], strict=True)
            )
            scheme_pdb_seq_num = string_array.remap(
                poly_seq_num, mapping=label_seq_id_to_auth_seq_id, default_value='.'
            )
            label_seq_id_to_ins_code = dict(
                zip(label_seq_ids[res_mask], pdb_ins_codes[res_mask], strict=True)
            )
            scheme_pdb_ins_code = string_array.remap(
                poly_seq_num, mapping=label_seq_id_to_ins_code, default_value='.'
            )

            # The _entity_poly_seq table is entity-based, while _pdbx_poly_seq_scheme
            # is chain-based. A single entity could mean multiple chains (asym_ids),
            # we therefore need to replicate each entity for all of the chains.
            scheme_asym_id: np.ndarray = np.array([])
            select = []
            indices = np.arange(len(poly_seq_entity_id), dtype=np.int32)
            for asym_id, entity_id in zip(
                    cif['_struct_asym.id'], cif['_struct_asym.entity_id'], strict=True
            ):
                entity_mask = poly_seq_entity_id == entity_id
                select.extend(indices[entity_mask])
                scheme_asym_id.extend([asym_id] * sum(entity_mask))

            scheme_pdb_strand_id = string_array.remap(
                np.array(scheme_asym_id, dtype=object),
                mapping=mmcif.get_internal_to_author_chain_id_map(cif),
                inplace=False,
            )

            update['_pdbx_poly_seq_scheme.asym_id'] = scheme_asym_id
            update['_pdbx_poly_seq_scheme.pdb_strand_id'] = scheme_pdb_strand_id
            update['_pdbx_poly_seq_scheme.pdb_seq_num'] = scheme_pdb_seq_num[select]
            update['_pdbx_poly_seq_scheme.pdb_ins_code'] = scheme_pdb_ins_code[select]
            update['_pdbx_poly_seq_scheme.seq_id'] = poly_seq_num[select]
            update['_pdbx_poly_seq_scheme.mon_id'] = poly_seq_mon_id[select]
        else:
            # _entity_poly_seq not available, fallback to _atom_site.
            res_asym_ids = label_asym_ids[res_mask]
            res_strand_ids = string_array.remap(
                array=res_asym_ids,
                mapping=mmcif.get_internal_to_author_chain_id_map(cif),
                inplace=False,
            )
            update['_pdbx_poly_seq_scheme.asym_id'] = res_asym_ids
            update['_pdbx_poly_seq_scheme.pdb_seq_num'] = auth_seq_ids[res_mask]
            update['_pdbx_poly_seq_scheme.pdb_ins_code'] = pdb_ins_codes[res_mask]
            update['_pdbx_poly_seq_scheme.seq_id'] = label_seq_ids[res_mask]
            update['_pdbx_poly_seq_scheme.mon_id'] = label_comp_ids[res_mask]
            update['_pdbx_poly_seq_scheme.pdb_strand_id'] = res_strand_ids

    required_nonpoly_scheme_cols = (
        '_pdbx_nonpoly_scheme.mon_id',
        '_pdbx_nonpoly_scheme.asym_id',
        '_pdbx_nonpoly_scheme.pdb_seq_num',
        '_pdbx_nonpoly_scheme.pdb_ins_code',
    )
    required_branch_scheme_cols = (
        '_pdbx_branch_scheme.mon_id',
        '_pdbx_branch_scheme.asym_id',
        '_pdbx_branch_scheme.pdb_seq_num',
    )

    # Generate _pdbx_nonpoly_scheme only if both tables are missing.
    if not (
            all(col in cif for col in required_nonpoly_scheme_cols)
            or all(col in cif for col in required_branch_scheme_cols)
    ):
        # To be strictly semantically correct, multi-residue ligands should be
        # written in _pdbx_branch_scheme. However, Structure parsing handles
        # correctly multi-residue ligands in _pdbx_nonpoly_scheme and the tables
        # constructed here live only while parsing, hence this is unnecessary.
        entity_id_by_chain_id = dict(
            zip(cif['_struct_asym.id'], cif['_struct_asym.entity_id'], strict=True)
        )
        chain_type_by_entity_id = dict(
            zip(cif['_entity.id'], cif['_entity.type'], strict=True)
        )
        # Remap asym ID -> entity ID.
        chain_type = string_array.remap(
            label_asym_ids, mapping=entity_id_by_chain_id, inplace=False
        )
        # Remap entity ID -> chain type.
        string_array.remap(
            chain_type, mapping=chain_type_by_entity_id, inplace=True
        )
        res_mask = np.zeros_like(label_seq_ids, dtype=bool)
        res_mask[res_starts] = True
        res_mask &= chain_type != mmcif_names.POLYMER_CHAIN

        if not np.any(res_mask):
            return update  # Shortcut: no non-polymer residues.

        ins_codes = string_array.remap(
            pdb_ins_codes[res_mask], mapping={'?': '.'}, inplace=False
        )

        update['_pdbx_nonpoly_scheme.asym_id'] = label_asym_ids[res_mask]
        update['_pdbx_nonpoly_scheme.pdb_seq_num'] = auth_seq_ids[res_mask]
        update['_pdbx_nonpoly_scheme.pdb_ins_code'] = ins_codes
        update['_pdbx_nonpoly_scheme.mon_id'] = label_comp_ids[res_mask]

    return update


def _get_chain_key_by_chain_id(
        resolved_chain_ids: np.ndarray, struct_asym_chain_ids: np.ndarray
) -> Mapping[str, int]:
    """Returns chain key for each chain ID respecting resolved chain ordering."""
    # Check that all chain IDs found in the (potentially filtered) _atom_site
    # table are present in the _struct_asym table.
    unique_resolved_chain_ids = set(resolved_chain_ids)
    if not unique_resolved_chain_ids.issubset(set(struct_asym_chain_ids)):
        unique_resolved_chain_ids = sorted(unique_resolved_chain_ids)
        unique_struct_asym_chain_ids = sorted(set(struct_asym_chain_ids))
        raise ValueError(
            'Bad mmCIF: chain IDs in _atom_site.label_asym_id '
            f'{unique_resolved_chain_ids} is not a subset of chain IDs in '
            f'_struct_asym.id {unique_struct_asym_chain_ids}.'
        )

    resolved_mask = string_array.isin(
        struct_asym_chain_ids, unique_resolved_chain_ids
    )
    # For all resolved chains, use the _atom_site order they appear in. E.g.
    # resolved_chain_ids     = [B A   E D F]
    # struct_asym_chain_ids  = [A B C D E F]
    # consistent_chain_order = [B A C E D F]
    # chain_keys             = [0 1 2 3 4 5]
    consistent_chain_order = struct_asym_chain_ids.copy()
    consistent_chain_order[resolved_mask] = resolved_chain_ids
    return dict(zip(consistent_chain_order, range(len(struct_asym_chain_ids))))


def _parse_bonds(
        cif: mmcif.Mmcif,
        atom_key: np.ndarray,
        model_id: str,
) -> bonds.Bonds:
    """Returns the bonds table extracted from the mmCIF.

    Args:
      cif: The raw mmCIF to extract the bond information from.
      atom_key: A numpy array defining atom key for each atom in _atom_site. Note
        that the atom key must be computed before resolving alt-locs since this
        function operates on the raw mmCIF!
      model_id: The ID of the model to get bonds for.
    """
    if '_struct_conn.id' not in cif:
        # This is the category key item for the _struct_conn table, therefore
        # we use it to determine whether to parse bond info.
        return bonds.Bonds.make_empty()
    from_atom, dest_atom = mmcif.get_bond_atom_indices(cif, model_id)
    from_atom = np.array(from_atom, dtype=np.int64)
    dest_atom = np.array(dest_atom, dtype=np.int64)
    num_bonds = from_atom.shape[0]
    bond_key = np.arange(num_bonds, dtype=np.int64)
    bond_type = cif.get_array('_struct_conn.conn_type_id', dtype=object)
    if '_struct_conn.pdbx_role' in cif:  # This column isn't always present.
        bond_role = cif.get_array('_struct_conn.pdbx_role', dtype=object)
    else:
        bond_role = np.full((num_bonds,), '?', dtype=object)

    bonds_mask = np.ones((num_bonds,), dtype=bool)
    # Symmetries other than 1_555 imply the atom is not part of the asymmetric
    # unit, and therefore this is a bond that only exists in the expanded
    # bioassembly.
    # We do not currently support parsing these types of bonds.
    if '_struct_conn.ptnr1_symmetry' in cif:
        ptnr1_symmetry = cif.get_array('_struct_conn.ptnr1_symmetry', dtype=object)
        np.logical_and(bonds_mask, ptnr1_symmetry == '1_555', out=bonds_mask)
    if '_struct_conn.ptnr2_symmetry' in cif:
        ptnr2_symmetry = cif.get_array('_struct_conn.ptnr2_symmetry', dtype=object)
        np.logical_and(bonds_mask, ptnr2_symmetry == '1_555', out=bonds_mask)
    # Remove bonds that involve atoms that are not part of the structure,
    # e.g. waters if include_water=False. In a rare case this also removes invalid
    # bonds that are indicated by a key that is set to _atom_site size.
    np.logical_and(bonds_mask, np.isin(from_atom, atom_key), out=bonds_mask)
    np.logical_and(bonds_mask, np.isin(dest_atom, atom_key), out=bonds_mask)
    return bonds.Bonds(
        key=bond_key[bonds_mask],
        type=bond_type[bonds_mask],
        role=bond_role[bonds_mask],
        from_atom_key=from_atom[bonds_mask],
        dest_atom_key=dest_atom[bonds_mask],
    )
