"""Module for parsing various data sources and producing Structures."""

import dataclasses
import datetime
from collections.abc import Mapping, Sequence

import numpy as np

from torchfold.structure import (
    bioassemblies, bonds, mmcif, structure,
    chemical_components as struc_chem_comps
)
from torchfold.structure.internal.constants import ATOM_FIELDS, CHAIN_FIELDS, RESIDUE_FIELDS, TABLE_FIELDS
from torchfold.structure.tables import Chains, Residues, Atoms, Bonds
from torchfold.structure.parsing.parsing_constants import ModelID
from torchfold.structure.parsing.parsing_table import get_tables, _parse_bonds
from torchfold.structure.parsing.parsing_utils import _get_str_model_id, _get_first_model_id
from torchfold.structure.parsing.table_builder import tables_from_atom_arrays


@dataclasses.dataclass(frozen=True, slots=True)
class _MmcifHeader:
    name: str
    resolution: float | None
    release_date: datetime.date | None
    structure_method: str | None
    bioassembly_data: bioassemblies.BioassemblyData | None
    chemical_components_data: struc_chem_comps.ChemicalComponentsData | None


def _get_mmcif_header(
        cif: mmcif.Mmcif,
        fix_mse: bool,
        fix_unknown_dna: bool,
) -> _MmcifHeader:
    """Extract header fields from an mmCIF object."""
    entry_id = cif.get('_entry.id')
    name = entry_id[0] if entry_id else cif.get_data_name()
    resolution = mmcif.get_resolution(cif)

    release_date = mmcif.get_release_date(cif)
    if release_date is not None:
        release_date = datetime.date.fromisoformat(release_date)

    experiments = cif.get('_exptl.method')
    structure_method = ','.join(experiments) if experiments else None

    try:
        bioassembly_data = bioassemblies.BioassemblyData.from_mmcif(cif)
    except bioassemblies.MissingBioassemblyDataError:
        bioassembly_data = None

    try:
        chemical_components_data = (
            struc_chem_comps.ChemicalComponentsData.from_mmcif(
                cif, fix_mse=fix_mse, fix_unknown_dna=fix_unknown_dna
            )
        )
    except struc_chem_comps.MissingChemicalComponentsDataError:
        chemical_components_data = None

    return _MmcifHeader(
        name=name,
        resolution=resolution,
        release_date=release_date,
        structure_method=structure_method,
        bioassembly_data=bioassembly_data,
        chemical_components_data=chemical_components_data,
    )


def from_parsed_mmcif(
        mmcif_object: mmcif.Mmcif,
        *,
        name: str | None = None,
        fix_mse_residues: bool = False,
        fix_arginines: bool = False,
        fix_unknown_dna: bool = False,
        include_water: bool = False,
        include_other: bool = False,
        include_bonds: bool = False,
        model_id: int | ModelID = ModelID.FIRST,
) -> structure.Structure:
    """Construct a Structure from a parsed mmCIF object.

    This function is called by `from_mmcif` but can be useful when an mmCIF has
    already been parsed e.g. to extract extra information from the header before
    then converting to Structure for further manipulation.

    Args:
      mmcif_object: A parsed mmcif.Mmcif object.
      name: Optional name for the structure. If not provided, the name will be
        taken from the mmCIF data_ field.
      fix_mse_residues: If True, selenium atom sites (SE) in selenomethionine
        (MSE) residues will be changed to sulphur atom sites (SD). This is because
        methionine (MET) residues are often replaced with MSE to aid X-Ray
        crystallography. If False, the SE MSE atom sites won't be modified.
      fix_arginines: If True, NH1 and NH2 in arginine will be swapped if needed so
        that NH1 is always closer to CD than NH2. If False, no atom sites in
        arginine will be touched. Note that HH11, HH12, HH21, HH22 are fixed too.
      fix_unknown_dna: If True, residues with name N in DNA chains will have their
        res_name replaced with DN. Atoms are not changed.
      include_water: If True, water (HOH) molecules will be parsed. Water
        molecules may be grouped into chains, where number of residues > 1. Water
        molecules are usually grouped into chains but do not necessarily all share
        the same chain ID.
      include_other: If True, all other atoms that are not included by any of the
        above parameters will be included. This covers e.g. "polypeptide(D)" and
        "macrolide" entities, as well as all other non-standard types.
      include_bonds: If True, bond information will be parsed from the mmCIF and
        stored in the Structure.
      model_id: Either the integer model ID to parse, or one of ModelID.FIRST to
        parse the first model, or ModelID.ALL to parse all models.

    Returns:
      A Structure representation of the mmCIF object.
    """
    str_model_id = _get_str_model_id(cif=mmcif_object, model_id=model_id)
    header = _get_mmcif_header(
        mmcif_object, fix_mse=fix_mse_residues, fix_unknown_dna=fix_unknown_dna
    )

    chains, residues, atoms = get_tables(
        cif=mmcif_object,
        fix_mse_residues=fix_mse_residues,
        fix_arginines=fix_arginines,
        fix_unknown_dna=fix_unknown_dna,
        include_water=include_water,
        include_other=include_other,
        model_id=str_model_id,
    )

    if include_bonds and atoms.size > 0:
        # NB: parsing the atom table before the bonds table allows for a more
        # informative error message when dealing with bad multi-model mmCIFs.
        # Also always use a specific model ID, even when parsing all models.
        if str_model_id == '':  # pylint: disable=g-explicit-bool-comparison
            bonds_model_id = _get_first_model_id(mmcif_object)
        else:
            bonds_model_id = str_model_id

        bonds_table = _parse_bonds(
            mmcif_object,
            atom_key=atoms.key,
            model_id=bonds_model_id,
        )
    else:
        bonds_table = bonds.Bonds.make_empty()

    return structure.Structure(
        name=name if name is not None else header.name,
        resolution=header.resolution,
        release_date=header.release_date,
        structure_method=header.structure_method,
        bioassembly_data=header.bioassembly_data,
        chemical_components_data=header.chemical_components_data,
        bonds=bonds_table,
        chains=chains,
        residues=residues,
        atoms=atoms,
    )


def from_mmcif(
        mmcif_string: str | bytes,
        *,
        name: str | None = None,
        fix_mse_residues: bool = False,
        fix_arginines: bool = False,
        fix_unknown_dna: bool = False,
        include_water: bool = False,
        include_other: bool = False,
        include_bonds: bool = False,
        model_id: int | ModelID = ModelID.FIRST,
) -> structure.Structure:
    """Construct a Structure from a mmCIF string.

    Args:
      mmcif_string: The string contents of an mmCIF file.
      name: Optional name for the structure. If not provided, the name will be
        taken from the mmCIF data_ field.
      fix_mse_residues: If True, selenium atom sites (SE) in selenomethionine
        (MSE) residues will be changed to sulphur atom sites (SD). This is because
        methionine (MET) residues are often replaced with MSE to aid X-Ray
        crystallography. If False, the SE MSE atom sites won't be modified.
      fix_arginines: If True, NH1 and NH2 in arginine will be swapped if needed so
        that NH1 is always closer to CD than NH2. If False, no atom sites in
        arginine will be touched. Note that HH11, HH12, HH21, HH22 are fixed too.
      fix_unknown_dna: If True, residues with name N in DNA chains will have their
        res_name replaced with DN. Atoms are not changed.
      include_water: If True, water (HOH) molecules will be parsed. Water
        molecules may be grouped into chains, where number of residues > 1. Water
        molecules are usually grouped into chains but do not necessarily all share
        the same chain ID.
      include_other: If True, all other atoms that are not included by any of the
        above parameters will be included. This covers e.g. "polypeptide(D)" and
        "macrolide" entities, as well as all other non-standard types.
      include_bonds: If True, bond information will be parsed from the mmCIF and
        stored in the Structure.
      model_id: Either the integer model ID to parse, or one of ModelID.FIRST to
        parse the first model, or ModelID.ALL to parse all models.

    Returns:
      A Structure representation of the mmCIF string.
    """
    mmcif_object = mmcif.from_string(mmcif_string)

    return from_parsed_mmcif(
        mmcif_object,
        name=name,
        fix_mse_residues=fix_mse_residues,
        fix_arginines=fix_arginines,
        fix_unknown_dna=fix_unknown_dna,
        include_water=include_water,
        include_other=include_other,
        include_bonds=include_bonds,
        model_id=model_id,
    )


def from_res_arrays(atom_mask: np.ndarray, **kwargs) -> structure.Structure:
    """Returns Structure created from from arrays with a residue dimension.

    All unset fields are filled with defaults (e.g. 1.0 for occupancy) or
    unset/unknown values (e.g. UNK for residue type, or '.' for atom element).

    Args:
      atom_mask: A array with shape (num_res, num_atom). This is used to decide
        which atoms in the atom dimension are present in a given residue. Present
        atoms should have a nonzero value, e.g. 1.0 or True.
      **kwargs: A mapping from field name to values. For all array-valued fields
        these arrays must have a dimension of length num_res. Chain and residue
        fields should have this as their only dimension and atom fields should be
        shaped (num_res, num_atom). Coordinate fields may also have arbitrary
        leading dimensions (they must be the same across all coordinate fields).
        See structure.{CHAIN,RESIDUE,ATOM}_FIELDS for a list of allowed fields.
    """
    num_res, num_atom = atom_mask.shape
    included_indices = np.flatnonzero(atom_mask)

    array_fields = (CHAIN_FIELDS.keys() | RESIDUE_FIELDS.keys() | ATOM_FIELDS.keys())
    initializer_kwargs = {}
    fields = {}
    for k, val in kwargs.items():
        if k not in array_fields:
            # The kwarg key isn't an array field name. Such kwargs are forwarded as-is
            # to the constructor. They are expected to be global fields (e.g. name).
            # Other values will raise an error when the constructor is called.
            if k in TABLE_FIELDS:
                raise ValueError(f'Table fields must not be set. Got {k}.')
            initializer_kwargs[k] = val
            continue
        elif val is None:
            raise ValueError(f'{k} must be non-None.')

        if not isinstance(val, np.ndarray):
            raise TypeError(f'Value for {k} must be a NumPy array. Got {type(val)}.')
        if k in CHAIN_FIELDS or k in RESIDUE_FIELDS:
            if val.shape != (num_res,):
                raise ValueError(
                    f'{k} must have shape ({num_res=},). Got {val.shape=}.'
                )
            # Do not reshape the chain/residue arrays, they have the shape we need.
            fields[k] = val
        else:
            assert k in ATOM_FIELDS
            if val.shape[-2:] != (num_res, num_atom):
                raise ValueError(
                    f'{k} must have final two dimensions of length '
                    f'{(num_res, num_atom)=}. Got {val.shape=}.'
                )
            leading_dims = val.shape[:-2]
            flat_val = val.reshape(leading_dims + (-1,), order='C')
            masked_val = flat_val[..., included_indices]
            fields[k] = masked_val

    # Get chain IDs or assume this is a single-chain structure.
    chain_id = kwargs.get('chain_id', np.array(['A'] * num_res, dtype=object))
    # Find chain starts in res-sized arrays, use these to make chain-sized arrays.
    chain_start = np.concatenate(
        ([0], np.where(chain_id[1:] != chain_id[:-1])[0] + 1)
    )
    if len(set(chain_id)) != len(chain_start):
        raise ValueError(f'Chain IDs must be contiguous, but got {chain_id}')

    chain_lengths = np.diff(chain_start, append=len(chain_id))
    chain_key = np.repeat(np.arange(len(chain_start)), chain_lengths)

    chain_entity_id = fields.get('chain_entity_id')
    if chain_entity_id is not None:
        entity_id = chain_entity_id[chain_start]
    else:
        entity_id = np.array(
            [str(mmcif.str_id_to_int_id(cid)) for cid in chain_id[chain_start]],
            dtype=object,
        )
    chain_str_empty = np.full((num_res,), '.', dtype=object)
    chains_table = Chains(
        key=chain_key[chain_start],
        id=chain_id[chain_start],
        type=fields.get('chain_type', chain_str_empty)[chain_start],
        auth_asym_id=fields.get('chain_auth_asym_id', chain_id)[chain_start],
        entity_id=entity_id,
        entity_desc=fields.get('chain_entity_desc', chain_str_empty)[chain_start],
    )

    # Since all arrays are residue-shaped, we can use them directly.
    res_key = np.arange(num_res, dtype=np.int64)
    res_id = fields.get('res_id', res_key + 1).astype(np.int32)
    residues_table = Residues(
        key=res_key,
        chain_key=chain_key,
        id=res_id,
        name=fields.get('res_name', np.full(num_res, 'UNK', dtype=object)),
        auth_seq_id=fields.get(
            'res_auth_seq_id', np.char.mod('%d', res_id).astype(object)
        ),
        insertion_code=fields.get(
            'res_insertion_code', np.full(num_res, '?', dtype=object)
        ),
    )

    # The atom-sized arrays have already been masked and reshaped.
    num_atoms_per_res = np.sum(atom_mask, axis=1, dtype=np.int32)
    num_atoms_total = np.sum(num_atoms_per_res, dtype=np.int32)
    # Structure is immutable, so use the same array multiple times to save RAM.
    atom_str_empty = np.full(num_atoms_total, '.', dtype=object)
    atom_float32_zeros = np.zeros(num_atoms_total, dtype=np.float32)
    atom_float32_ones = np.ones(num_atoms_total, dtype=np.float32)
    atoms_table = Atoms(
        key=np.arange(num_atoms_total, dtype=np.int64),
        chain_key=np.repeat(chain_key, num_atoms_per_res),
        res_key=np.repeat(res_key, num_atoms_per_res),
        name=fields.get('atom_name', atom_str_empty),
        element=fields.get('atom_element', atom_str_empty),
        x=fields.get('atom_x', atom_float32_zeros),
        y=fields.get('atom_y', atom_float32_zeros),
        z=fields.get('atom_z', atom_float32_zeros),
        b_factor=fields.get('atom_b_factor', atom_float32_zeros),
        occupancy=fields.get('atom_occupancy', atom_float32_ones),
    )

    return structure.Structure(
        chains=chains_table,
        residues=residues_table,
        atoms=atoms_table,
        bonds=Bonds.make_empty(),  # Currently not set.
        **initializer_kwargs,
    )


def from_atom_arrays(
        *,
        res_id: np.ndarray,
        name: str = 'unset',
        release_date: datetime.date | None = None,
        resolution: float | None = None,
        structure_method: str | None = None,
        all_residues: Mapping[str, Sequence[tuple[str, int]]] | None = None,
        bioassembly_data: bioassemblies.BioassemblyData | None = None,
        chemical_components_data: (
                struc_chem_comps.ChemicalComponentsData | None
        ) = None,
        bond_table: Bonds | None = None,
        chain_id: np.ndarray | None = None,
        chain_type: np.ndarray | None = None,
        res_name: np.ndarray | None = None,
        atom_key: np.ndarray | None = None,
        atom_name: np.ndarray | None = None,
        atom_element: np.ndarray | None = None,
        atom_x: np.ndarray | None = None,
        atom_y: np.ndarray | None = None,
        atom_z: np.ndarray | None = None,
        atom_b_factor: np.ndarray | None = None,
        atom_occupancy: np.ndarray | None = None,
) -> structure.Structure:
    """Returns a Structure constructed from atom array level data.

    All fields except name and, res_id are optional, all array fields consist of a
    value for each atom in the structure - so residue and chain values should hold
    the same value for each atom in the chain or residue. Fields which are not
    defined are filled with default values.

    Validation is performed by the Structure constructor where possible - but
    author_naming scheme and all_residues must be checked in this function.

    It is not possible to construct structures with chains that do not contain
    any resolved residues using this function. If this is necessary, use the
    structure.Structure constructor directly.

    Args:
      res_id: Integer array of shape [num_atom]. The unique residue identifier for
        each residue. mmCIF field - _atom_site.label_seq_id.
      name: The name of the structure. E.g. a PDB ID.
      release_date: The release date of the structure as a `datetime.date`.
      resolution: The resolution of the structure in Angstroms.
      structure_method: The method used to solve this structure's coordinates.
      all_residues: An optional mapping from each chain ID (i.e. label_asym_id) to
        a sequence of (label_comp_id, label_seq_id) tuples, one per residue. This
        can contain residues that aren't present in the atom arrays. This is
        common in experimental data where some residues are not resolved but are
        known to be present.
      bioassembly_data: An optional instance of bioassembly.BioassemblyData. If
        present then a new Structure representing a specific bioassembly can be
        extracted using `Structure.generate_bioassembly(assembly_id)`.
      chemical_components_data: An optional instance of ChemicalComponentsData.
        Its content will be used for providing metadata about chemical components
        in this Structure instance. If not specified information will be retrieved
        from the standard chemical component dictionary (CCD, for more details see
        https://www.wwpdb.org/data/ccd).
      bond_table: A table representing manually-specified bonds. This corresponds
        to the _struct_conn table in an mmCIF. Atoms are identified by their key,
        as specified by the atom_key column. If this table is provided then the
        atom_key column must also be defined.
      chain_id: String array of shape [num_atom] of unique chain identifiers.
        mmCIF field - _atom_site.label_asym_id.
      chain_type: String array of shape [num_atom]. The molecular type of the
        current chain (e.g. polyribonucleotide). mmCIF field - _entity_poly.type
        OR _entity.type (for non-polymers).
      res_name: String array of shape [num_atom].. The name of each residue,
        typically a 3 letter string for polypeptides or 1-2 letter strings for
        polynucleotides. mmCIF field - _atom_site.label_comp_id.
      atom_key: A unique sorted integer array, used only by the bonds table to
        identify the atoms participating in each bond. If the bonds table is
        specified then this column must be non-None.
      atom_name: String array of shape [num_atom]. The name of each atom (e.g CA,
        O2', etc.). mmCIF field - _atom_site.label_atom_id.
      atom_element: String array of shape [num_atom]. The element type of each
        atom (e.g. C, O, N, etc.). mmCIF field - _atom_site.type_symbol.
      atom_x: Float array of shape [..., num_atom] of atom x coordinates. May have
        arbitrary leading dimensions, provided that these are consistent across
        all coordinate fields.
      atom_y: Float array of shape [..., num_atom] of atom y coordinates. May have
        arbitrary leading dimensions, provided that these are consistent across
        all coordinate fields.
      atom_z: Float array of shape [..., num_atom] of atom z coordinates. May have
        arbitrary leading dimensions, provided that these are consistent across
        all coordinate fields.
      atom_b_factor: Float array of shape [..., num_atom] or [num_atom] of atom
        b-factors or equivalent. If there are no extra leading dimensions then
        these values are assumed to apply to all coordinates for a given atom. If
        there are leading dimensions then these must match those used by the
        coordinate fields.
      atom_occupancy: Float array of shape [..., num_atom] or [num_atom] of atom
        occupancies or equivalent. If there are no extra leading dimensions then
        these values are assumed to apply to all coordinates for a given atom. If
        there are leading dimensions then these must match those used by the
        coordinate fields.
    """

    atoms, residues, chains = tables_from_atom_arrays(
        res_id=res_id,
        all_residues=all_residues,
        chain_id=chain_id,
        chain_type=chain_type,
        res_name=res_name,
        atom_key=atom_key,
        atom_name=atom_name,
        atom_element=atom_element,
        atom_x=atom_x,
        atom_y=atom_y,
        atom_z=atom_z,
        atom_b_factor=atom_b_factor,
        atom_occupancy=atom_occupancy,
    )

    return structure.Structure(
        name=name,
        release_date=release_date,
        resolution=resolution,
        structure_method=structure_method,
        bioassembly_data=bioassembly_data,
        chemical_components_data=chemical_components_data,
        atoms=atoms,
        chains=chains,
        residues=residues,
        bonds=bond_table or Bonds.make_empty(),
    )
