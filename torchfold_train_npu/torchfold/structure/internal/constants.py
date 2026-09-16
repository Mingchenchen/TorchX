import enum
from collections.abc import Collection, Mapping
from typing import Final

# Controls the default number of decimal places for coordinates when writing to
# mmCIF.
COORDS_DECIMAL_PLACES: Final[int] = 3

# External residue ID given to missing residues that don't have an ID
# already provided. In mmCIFs this data is found in _pdbx_poly_seq_scheme.
MISSING_AUTH_SEQ_ID: Final[str] = '.'

# Maps from structure fields to column names in the relevant table.
CHAIN_FIELDS: Final[Mapping[str, str]] = {
    'chain_id': 'id',
    'chain_type': 'type',
    'chain_auth_asym_id': 'auth_asym_id',
    'chain_entity_id': 'entity_id',
    'chain_entity_desc': 'entity_desc',
}

RESIDUE_FIELDS: Final[Mapping[str, str]] = {
    'res_id': 'id',
    'res_name': 'name',
    'res_auth_seq_id': 'auth_seq_id',
    'res_insertion_code': 'insertion_code',
}

ATOM_FIELDS: Final[Mapping[str, str]] = {
    'atom_name': 'name',
    'atom_element': 'element',
    'atom_x': 'x',
    'atom_y': 'y',
    'atom_z': 'z',
    'atom_b_factor': 'b_factor',
    'atom_occupancy': 'occupancy',
    'atom_key': 'key',
}

# Fields in structure.
ARRAY_FIELDS = frozenset({
    'atom_b_factor',
    'atom_element',
    'atom_key',
    'atom_name',
    'atom_occupancy',
    'atom_x',
    'atom_y',
    'atom_z',
    'chain_id',
    'chain_type',
    'res_id',
    'res_name',
})

TABLE_FIELDS: Final[Collection[str]] = frozenset(
    {'chains', 'residues', 'atoms', 'bonds'}
)


@enum.unique
class CascadeDelete(enum.Enum):
    NONE = 0
    FULL = 1
    CHAINS = 2
