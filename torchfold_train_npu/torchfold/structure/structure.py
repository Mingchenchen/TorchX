import enum
import collections
import datetime
import functools
import itertools
import typing
from collections.abc import Collection, Iterator, Mapping, Sequence, Set
from typing import Any, ClassVar, Self, TypeAlias, TypeVar

import numpy as np
from torchfold.cpp import membership

from torchfold.constants import mmcif_names
from torchfold.structure import bioassemblies
from torchfold.structure import chemical_components as struc_chem_comps
from torchfold.structure import mmcif
from torchfold.structure import tables as structure_tables
from torchfold.structure.internal import table
from torchfold.structure.internal.constants import CHAIN_FIELDS, RESIDUE_FIELDS, MISSING_AUTH_SEQ_ID, \
    CascadeDelete, COORDS_DECIMAL_PLACES, TABLE_FIELDS
from torchfold.structure.models import StructureTables

# AllResidues is a mapping from label_asym_id to a sequence of (label_comp_id,
# label_seq_id) pairs. These represent the full sequence including residues
# that might be missing (e.g. unresolved residues in X-ray data).
AllResidues: TypeAlias = Mapping[str, Sequence[tuple[str, int]]]
AuthorNamingScheme: TypeAlias = structure_tables.AuthorNamingScheme

_T = TypeVar('_T')


class _UnsetSentinel(enum.Enum):
    UNSET = object()


_UNSET = _UnsetSentinel.UNSET
UnsetOr: TypeAlias = _T | _UnsetSentinel


def _get_change_indices(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=np.int32)
    else:
        changing_idxs = np.where(arr[1:] != arr[:-1])[0] + 1
        return np.concatenate(([0], changing_idxs), axis=0)


class Structure(table.Database):
    """Structure class for representing and processing molecular structures."""

    tables: ClassVar[Collection[str]] = TABLE_FIELDS

    foreign_keys: ClassVar[Mapping[str, Collection[tuple[str, str]]]] = {
        'residues': (('chain_key', 'chains'),),
        'atoms': (('chain_key', 'chains'), ('res_key', 'residues')),
        'bonds': (('from_atom_key', 'atoms'), ('dest_atom_key', 'atoms')),
    }

    def __init__(
            self,
            *,
            name: str = 'unset',
            release_date: datetime.date | None = None,
            resolution: float | None = None,
            structure_method: str | None = None,
            bioassembly_data: bioassemblies.BioassemblyData | None = None,
            chemical_components_data: (
                    struc_chem_comps.ChemicalComponentsData | None
            ) = None,
            chains: structure_tables.Chains,
            residues: structure_tables.Residues,
            atoms: structure_tables.Atoms,
            bonds: structure_tables.Bonds,
            skip_validation: bool = False,
    ):
        # Version number is written to mmCIF and should be incremented when changes
        # are made to mmCIF writing or internals that affect this.
        # b/345221494 Rename this variable when structure_v1 compatibility code
        # is removed.
        self._version = '2.0.0'  # pylint: disable=invalid-name
        self._name = name
        self._release_date = release_date
        self._resolution = resolution
        self._structure_method = structure_method
        self._bioassembly_data = bioassembly_data
        self._chemical_components_data = chemical_components_data

        self._chains = chains
        self._residues = residues
        self._atoms = atoms
        self._bonds = bonds

        if not skip_validation:
            self._validate_table_foreign_keys()
            self._validate_consistent_table_ordering()

    def _validate_table_foreign_keys(self):
        """Validates that all foreign keys are present in the referred tables."""
        residue_keys = set(self._residues.key)
        chain_keys = set(self._chains.key)
        if np.any(membership.isin(self._atoms.res_key, residue_keys, invert=True)):
            raise ValueError(
                'Atom residue keys not in the residues table: '
                f'{set(self._atoms.res_key).difference(self._residues.key)}'
            )
        if np.any(membership.isin(self._atoms.chain_key, chain_keys, invert=True)):
            raise ValueError(
                'Atom chain keys not in the chains table: '
                f'{set(self._atoms.chain_key).difference(self._chains.key)}'
            )
        if np.any(
                membership.isin(self._residues.chain_key, chain_keys, invert=True)
        ):
            raise ValueError(
                'Residue chain keys not in the chains table: '
                f'{set(self._residues.chain_key).difference(self._chains.key)}'
            )

    def _validate_consistent_table_ordering(self):
        """Validates that all tables have the same ordering."""
        atom_chain_keys = self._atoms.chain_key[self.chain_boundaries]
        atom_res_keys = self._atoms.res_key[self.res_boundaries]

        if not np.array_equal(self.present_chains.key, atom_chain_keys):
            raise ValueError(
                f'Atom table chain order\n{atom_chain_keys}\ndoes not match the '
                f'chain table order\n{self._chains.key}'
            )
        if not np.array_equal(self.present_residues.key, atom_res_keys):
            raise ValueError(
                f'Atom table residue order\n{atom_res_keys}\ndoes not match the '
                f'present residue table order\n{self.present_residues.key}'
            )

    def get_table(self, table_name: str) -> table.Table:
        match table_name:
            case 'chains':
                return self.chains_table
            case 'residues':
                return self.residues_table
            case 'atoms':
                return self.atoms_table
            case 'bonds':
                return self.bonds_table
            case _:
                raise ValueError(table_name)

    @property
    def chains_table(self) -> structure_tables.Chains:
        """Chains table."""
        return self._chains

    @property
    def residues_table(self) -> structure_tables.Residues:
        """Residues table."""
        return self._residues

    @property
    def atoms_table(self) -> structure_tables.Atoms:
        """Atoms table."""
        return self._atoms

    @property
    def bonds_table(self) -> structure_tables.Bonds:
        """Bonds table."""
        return self._bonds

    @property
    def version(self) -> str:
        return self._version

    @property
    def name(self) -> str:
        return self._name

    @property
    def release_date(self) -> datetime.date | None:
        return self._release_date

    @property
    def resolution(self) -> float | None:
        return self._resolution

    @property
    def structure_method(self) -> str | None:
        return self._structure_method

    @property
    def bioassembly_data(self) -> bioassemblies.BioassemblyData | None:
        return self._bioassembly_data

    @property
    def chemical_components_data(
            self,
    ) -> struc_chem_comps.ChemicalComponentsData | None:
        return self._chemical_components_data

    @property
    def bonds(self) -> structure_tables.Bonds:
        return self._bonds

    @functools.cached_property
    def author_naming_scheme(self) -> AuthorNamingScheme:
        auth_asym_id = {}
        entity_id = {}
        entity_desc = {}
        auth_seq_id = collections.defaultdict(dict)
        insertion_code = collections.defaultdict(dict)

        for chain_i in range(self._chains.size):
            chain_id = self._chains.id[chain_i]
            auth_asym_id[chain_id] = self._chains.auth_asym_id[chain_i]
            chain_entity_id = self._chains.entity_id[chain_i]
            entity_id[chain_id] = chain_entity_id
            entity_desc[chain_entity_id] = self._chains.entity_desc[chain_i]

        chain_index_by_key = self._chains.index_by_key
        for res_i in range(self._residues.size):
            chain_key = self._residues.chain_key[res_i]
            chain_id = self._chains.id[chain_index_by_key[chain_key]]
            res_id = self._residues.id[res_i]
            res_auth_seq_id = self._residues.auth_seq_id[res_i]
            if res_auth_seq_id == MISSING_AUTH_SEQ_ID:
                continue
            auth_seq_id[chain_id][res_id] = res_auth_seq_id
            ins_code = self._residues.insertion_code[res_i]
            # Compatibility with Structure v1 which used None to represent . or ?.
            insertion_code[chain_id][res_id] = (
                ins_code if ins_code not in {'.', '?'} else None
            )

        return AuthorNamingScheme(
            auth_asym_id=auth_asym_id,
            entity_id=entity_id,
            entity_desc=entity_desc,
            auth_seq_id=dict(auth_seq_id),
            insertion_code=dict(insertion_code),
        )

    @functools.cached_property
    def all_residues(self) -> AllResidues:
        chain_id_by_key = dict(zip(self._chains.key, self._chains.id))
        residue_chain_boundaries = _get_change_indices(self._residues.chain_key)
        boundaries = self.iter_residue_ranges(
            residue_chain_boundaries, count_unresolved=True
        )
        return {
            chain_id_by_key[self._residues.chain_key[start]]: list(
                zip(self._residues.name[start:end], self._residues.id[start:end])
            )
            for start, end in boundaries
        }

    @functools.cached_property
    def label_asym_id_to_entity_id(self) -> Mapping[str, str]:
        return dict(zip(self._chains.id, self._chains.entity_id))

    @functools.cached_property
    def chain_entity_id(self) -> np.ndarray:
        """Returns the entity ID for each atom in the structure."""
        return self.chains_table.apply_array_to_column(
            'entity_id', self._atoms.chain_key
        )

    @functools.cached_property
    def chain_entity_desc(self) -> np.ndarray:
        """Returns the entity description for each atom in the structure."""
        return self.chains_table.apply_array_to_column(
            'entity_desc', self._atoms.chain_key
        )

    @functools.cached_property
    def chain_auth_asym_id(self) -> np.ndarray:
        """Returns the chain auth asym ID for each atom in the structure."""
        return self.chains_table.apply_array_to_column(
            'auth_asym_id', self._atoms.chain_key
        )

    @functools.cached_property
    def chain_id(self) -> np.ndarray:
        chain_index_by_key = self._chains.index_by_key
        return self._chains.id[chain_index_by_key[self._atoms.chain_key]]

    @functools.cached_property
    def chain_type(self) -> np.ndarray:
        chain_index_by_key = self._chains.index_by_key
        return self._chains.type[chain_index_by_key[self._atoms.chain_key]]

    @functools.cached_property
    def res_id(self) -> np.ndarray:
        return self._residues['id', self._atoms.res_key]

    @functools.cached_property
    def res_name(self) -> np.ndarray:
        return self._residues['name', self._atoms.res_key]

    @functools.cached_property
    def res_auth_seq_id(self) -> np.ndarray:
        """Returns the residue auth seq ID for each atom in the structure."""
        return self.residues_table.apply_array_to_column(
            'auth_seq_id', self._atoms.res_key
        )

    @functools.cached_property
    def res_insertion_code(self) -> np.ndarray:
        """Returns the residue insertion code for each atom in the structure."""
        return self.residues_table.apply_array_to_column(
            'insertion_code', self._atoms.res_key
        )

    @property
    def atom_key(self) -> np.ndarray:
        return self._atoms.key

    @property
    def atom_name(self) -> np.ndarray:
        return self._atoms.name

    @property
    def atom_element(self) -> np.ndarray:
        return self._atoms.element

    @property
    def atom_x(self) -> np.ndarray:
        return self._atoms.x

    @property
    def atom_y(self) -> np.ndarray:
        return self._atoms.y

    @property
    def atom_z(self) -> np.ndarray:
        return self._atoms.z

    @property
    def atom_b_factor(self) -> np.ndarray:
        return self._atoms.b_factor

    @property
    def atom_occupancy(self) -> np.ndarray:
        return self._atoms.occupancy

    @functools.cached_property
    def chain_boundaries(self) -> np.ndarray:
        """The indices in the atom fields where each present chain begins."""
        return _get_change_indices(self._atoms.chain_key)

    @functools.cached_property
    def res_boundaries(self) -> np.ndarray:
        """The indices in the atom fields where each present residue begins."""
        return _get_change_indices(self._atoms.res_key)

    @functools.cached_property
    def present_chains(self) -> structure_tables.Chains:
        """Returns table of chains which have at least 1 resolved atom."""
        is_present_mask = np.isin(self._chains.key, self._atoms.chain_key)
        return typing.cast(structure_tables.Chains, self._chains[is_present_mask])

    @functools.cached_property
    def present_residues(self) -> structure_tables.Residues:
        """Returns table of residues which have at least 1 resolved atom."""
        is_present_mask = np.isin(self._residues.key, self._atoms.res_key)
        return typing.cast(
            structure_tables.Residues, self._residues[is_present_mask]
        )

    @functools.cached_property
    def unresolved_residues(self) -> structure_tables.Residues:
        """Returns table of residues which have at least 1 resolved atom."""
        is_unresolved_mask = np.isin(
            self._residues.key, self._atoms.res_key, invert=True
        )
        return typing.cast(
            structure_tables.Residues, self._residues[is_unresolved_mask]
        )

    def __getitem__(self, field: str) -> Any:
        """Gets raw field data using field name as a string."""
        if field in TABLE_FIELDS:
            return self.get_table(field)
        else:
            return getattr(self, field)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle calls this on dump.

        Returns:
          Members with cached properties removed.
        """
        cached_props = {
            k
            for k, v in self.__class__.__dict__.items()
            if isinstance(v, functools.cached_property)
        }
        return {k: v for k, v in self.__dict__.items() if k not in cached_props}

    def __repr__(self):
        return (
            f'Structure({self._name}: {self.num_chains} chains, '
            f'{self.num_residues(count_unresolved=False)} residues, '
            f'{self.num_atoms} atoms)'
        )

    @property
    def num_atoms(self) -> int:
        return self._atoms.size

    def num_residues(self, *, count_unresolved: bool) -> int:
        """Returns the number of residues in this Structure.

        Args:
          count_unresolved: Whether to include unresolved (empty) residues.

        Returns:
          Number of residues in the Structure.
        """
        if count_unresolved:
            return self._residues.size
        else:
            return self.present_residues.size

    @property
    def num_chains(self) -> int:
        return self._chains.size

    @property
    def num_models(self) -> int:
        """The number of models of this Structure."""
        return self._atoms.num_models

    def _atom_mask(self, entities: Set[str]) -> np.ndarray:
        """Boolean label indicating if each atom is from entities or not."""
        mask = np.zeros(self.num_atoms, dtype=bool)
        chain_index_by_key = self._chains.index_by_key
        for start, end in self.iter_chain_ranges():
            chain_index = chain_index_by_key[self._atoms.chain_key[start]]
            chain_type = self._chains.type[chain_index]
            mask[start:end] = chain_type in entities
        return mask

    @functools.cached_property
    def is_protein_mask(self) -> np.ndarray:
        """Boolean label indicating if each atom is from protein or not."""
        return self._atom_mask(entities={mmcif_names.PROTEIN_CHAIN})

    @functools.cached_property
    def is_dna_mask(self) -> np.ndarray:
        """Boolean label indicating if each atom is from DNA or not."""
        return self._atom_mask(entities={mmcif_names.DNA_CHAIN})

    @functools.cached_property
    def is_rna_mask(self) -> np.ndarray:
        """Boolean label indicating if each atom is from RNA or not."""
        return self._atom_mask(entities={mmcif_names.RNA_CHAIN})

    @functools.cached_property
    def is_nucleic_mask(self) -> np.ndarray:
        """Boolean label indicating if each atom is a nucleic acid or not."""
        return self._atom_mask(entities=mmcif_names.NUCLEIC_ACID_CHAIN_TYPES)

    @functools.cached_property
    def is_ligand_mask(self) -> np.ndarray:
        """Boolean label indicating if each atom is a ligand or not."""
        return self._atom_mask(entities=mmcif_names.LIGAND_CHAIN_TYPES)

    @functools.cached_property
    def is_water_mask(self) -> np.ndarray:
        """Boolean label indicating if each atom is from water or not."""
        return self._atom_mask(entities={mmcif_names.WATER})

    def iter_atoms(self) -> Iterator[Mapping[str, Any]]:
        """Iterates over the atoms in the structure."""
        if self._atoms.size == 0:
            return

        current_chain = self._chains.get_row_by_key(
            column_name_map=CHAIN_FIELDS, key=self._atoms.chain_key[0]
        )
        current_chain_key = self._atoms.chain_key[0]
        current_res = self._residues.get_row_by_key(
            column_name_map=RESIDUE_FIELDS, key=self._atoms.res_key[0]
        )
        current_res_key = self._atoms.res_key[0]
        for atom_i in range(self._atoms.size):
            atom_chain_key = self._atoms.chain_key[atom_i]
            atom_res_key = self._atoms.res_key[atom_i]

            if atom_chain_key != current_chain_key:
                chain_index = self._chains.index_by_key[atom_chain_key]
                current_chain = {
                    'chain_id': self._chains.id[chain_index],
                    'chain_type': self._chains.type[chain_index],
                    'chain_auth_asym_id': self._chains.auth_asym_id[chain_index],
                    'chain_entity_id': self._chains.entity_id[chain_index],
                    'chain_entity_desc': self._chains.entity_desc[chain_index],
                }
                current_chain_key = atom_chain_key
            if atom_res_key != current_res_key:
                res_index = self._residues.index_by_key[atom_res_key]
                current_res = {
                    'res_id': self._residues.id[res_index],
                    'res_name': self._residues.name[res_index],
                    'res_auth_seq_id': self._residues.auth_seq_id[res_index],
                    'res_insertion_code': self._residues.insertion_code[res_index],
                }
                current_res_key = atom_res_key

            yield {
                'atom_name': self._atoms.name[atom_i],
                'atom_element': self._atoms.element[atom_i],
                'atom_x': self._atoms.x[..., atom_i],
                'atom_y': self._atoms.y[..., atom_i],
                'atom_z': self._atoms.z[..., atom_i],
                'atom_b_factor': self._atoms.b_factor[..., atom_i],
                'atom_occupancy': self._atoms.occupancy[..., atom_i],
                'atom_key': self._atoms.key[atom_i],
                **current_res,
                **current_chain,
            }

    def _iter_atom_ranges(
            self, boundaries: Sequence[int]
    ) -> Iterator[tuple[int, int]]:
        """Iterator for (start, end) pairs from an array of start indices."""
        yield from itertools.pairwise(boundaries)
        # Use explicit length test as boundaries can be a NumPy array.
        if len(boundaries) > 0:  # pylint: disable=g-explicit-length-test
            yield boundaries[-1], self.num_atoms

    def iter_residue_ranges(
            self,
            boundaries: Sequence[int],
            *,
            count_unresolved: bool,
    ) -> Iterator[tuple[int, int]]:
        """Iterator for (start, end) pairs from an array of start indices."""
        yield from itertools.pairwise(boundaries)
        # Use explicit length test as boundaries can be a NumPy array.
        if len(boundaries) > 0:  # pylint: disable=g-explicit-length-test
            yield boundaries[-1], self.num_residues(count_unresolved=count_unresolved)

    def iter_chain_ranges(self) -> Iterator[tuple[int, int]]:
        """Iterates pairs of (chain_start, chain_end) indices.

        Yields:
          Pairs of (start, end) indices for each chain, where end is not inclusive.
          i.e. struc.chain_id[start:end] would be a constant array with length
          equal to the number of atoms in the chain.
        """
        yield from self._iter_atom_ranges(self.chain_boundaries)

    def _apply_atom_index_array(
            self,
            index_arr: np.ndarray,
            chain_boundaries: np.ndarray | None = None,
            res_boundaries: np.ndarray | None = None,
            skip_validation: bool = False,
    ) -> Self:
        """Applies index_arr to the atom table using NumPy-style array indexing.

        Args:
          index_arr: A 1D NumPy array that will be used to index into the atoms
            table. This can either be a boolean array to act as a mask, or an
            integer array to perform a gather operation.
          chain_boundaries: Unused in structure v2.
          res_boundaries: Unused in structure v2.
          skip_validation: Whether to skip the validation step that checks internal
            consistency after applying atom index array. Do not set to True unless
            you are certain the transform is safe, e.g. when the order of atoms is
            guaranteed to not change.

        Returns:
          A new Structure with an updated atoms table.
        """
        del chain_boundaries, res_boundaries

        if index_arr.ndim != 1:
            raise ValueError(
                f'index_arr must be a 1D NumPy array, but has shape {index_arr.shape}'
            )

        if index_arr.dtype == bool and np.all(index_arr):
            return self  # Shortcut: The operation is a no-op, so just return itself.

        atoms = structure_tables.Atoms(
            **{col: self._atoms[col][..., index_arr] for col in self._atoms.columns}
        )
        updated_tables = self._cascade_delete(atoms=atoms)
        return self.copy_and_update(
            atoms=updated_tables.atoms,
            bonds=updated_tables.bonds,
            skip_validation=skip_validation,
        )

    @property
    def group_by_residue(self) -> Self:
        """Returns a Structure with one atom per residue.

        e.g. restypes = struc.group_by_residue['res_id']

        Returns:
          A new Structure with one atom per residue such that per-atom arrays
          such as res_name (i.e. Structure v1 fields) have one element per residue.
        """
        # This use of _apply_atom_index_array is safe because the chain/residue/atom
        # ordering won't change (essentially applying a residue start mask).
        return self._apply_atom_index_array(
            self.res_boundaries, skip_validation=True
        )

    @property
    def group_by_chain(self) -> Self:
        """Returns a Structure where all fields are per-chain.

        e.g. chains = struc.group_by_chain['chain_id']

        Returns:
          A new Structure with one atom per chain such that per-atom arrays
          such as res_name (i.e. Structure v1 fields) have one element per chain.
        """
        # This use of _apply_atom_index_array is safe because the chain/residue/atom
        # ordering won't change (essentially applying a chain start mask).
        return self._apply_atom_index_array(
            self.chain_boundaries, skip_validation=True
        )

    @property
    def with_sorted_chains(self) -> Self:
        """Returns a new structure with the chains are in reverse spreadsheet style.

        This is the usual order to write chains in an mmCIF:
        (A < B < ... < AA < BA < CA < ... < AB < BB < CB ...)

        NB: this method will fail if chains do not conform to this mmCIF naming
        convention.

        Only to be used for third party metrics that rely on the chain order.
        Elsewhere chains should be identified by name and code should be agnostic to
        the order.
        """
        sorted_chains = sorted(self.chains, key=mmcif.str_id_to_int_id)
        return self.reorder_chains(new_order=sorted_chains)

    @functools.cached_property
    def atom_ids(self) -> Sequence[tuple[str, str, None, str]]:
        """Gets a list of atom ID tuples from Structure class arrays.

        Returns:
          A list of tuples of (chain_id, res_id, insertion_code, atom_name) where
          insertion code is always None. There is one element per atom, and the
          list is ordered according to the order of atoms in the input arrays.
        """
        # Convert to Numpy strings, then to Python strings (dtype=object).
        res_ids = self.residues_table.id.astype(str).astype(object)
        res_ids = res_ids[
            self.residues_table.index_by_key[self.atoms_table.res_key]
        ]
        ins_codes = [None] * self.num_atoms
        return list(
            zip(self.chain_id, res_ids, ins_codes, self.atom_name, strict=True)
        )

    def copy_and_update(
            self,
            *,
            name: UnsetOr[str] = _UNSET,
            release_date: UnsetOr[datetime.date | None] = _UNSET,
            resolution: UnsetOr[float | None] = _UNSET,
            structure_method: UnsetOr[str | None] = _UNSET,
            bioassembly_data: UnsetOr[bioassemblies.BioassemblyData | None] = _UNSET,
            chemical_components_data: UnsetOr[struc_chem_comps.ChemicalComponentsData | None] = _UNSET,
            chains: UnsetOr[structure_tables.Chains] = _UNSET,
            residues: UnsetOr[structure_tables.Residues] = _UNSET,
            atoms: UnsetOr[structure_tables.Atoms] = _UNSET,
            bonds: UnsetOr[structure_tables.Bonds] = _UNSET,
            skip_validation: bool = False,
    ) -> Self:
        """Performs a shallow copy but with specified fields updated."""

        def all_unset(fields):
            return all(field == _UNSET for field in fields)

        if all_unset((chains, residues, atoms, bonds)):
            if all_unset((
                    name,
                    release_date,
                    resolution,
                    structure_method,
                    bioassembly_data,
                    chemical_components_data,
            )):
                raise ValueError(
                    'Unnecessary call to copy_and_update with no changes. As Structure'
                    ' and its component tables are immutable, there is no need to copy'
                    ' it. Any subsequent operation that modifies structure will return'
                    ' a new object.'
                )
            else:
                raise ValueError(
                    'When only changing global fields, prefer to use the specialised '
                    'copy_and_update_globals.'
                )

        def select(field, default):
            return field if field != _UNSET else default

        return Structure(
            name=select(name, self.name),
            release_date=select(release_date, self.release_date),
            resolution=select(resolution, self.resolution),
            structure_method=select(structure_method, self.structure_method),
            bioassembly_data=select(bioassembly_data, self.bioassembly_data),
            chemical_components_data=select(
                chemical_components_data, self.chemical_components_data
            ),
            chains=select(chains, self._chains),
            residues=select(residues, self._residues),
            atoms=select(atoms, self._atoms),
            bonds=select(bonds, self._bonds),
            skip_validation=skip_validation,
        )

    def copy_and_update_globals(
            self,
            *,
            name: UnsetOr[str] = _UNSET,
            release_date: UnsetOr[datetime.date | None] = _UNSET,
            resolution: UnsetOr[float | None] = _UNSET,
            structure_method: UnsetOr[str | None] = _UNSET,
            bioassembly_data: UnsetOr[bioassemblies.BioassemblyData | None] = _UNSET,
            chemical_components_data: UnsetOr[struc_chem_comps.ChemicalComponentsData | None] = _UNSET,
    ) -> Self:
        """Returns a shallow copy with the global columns updated."""

        def select(field, default):
            return field if field != _UNSET else default

        name = select(name, self.name)
        release_date = select(release_date, self.release_date)
        resolution = select(resolution, self.resolution)
        structure_method = select(structure_method, self.structure_method)
        bioassembly_data = select(bioassembly_data, self.bioassembly_data)
        chem_data = select(chemical_components_data, self.chemical_components_data)

        return Structure(
            name=name,
            release_date=release_date,
            resolution=resolution,
            structure_method=structure_method,
            bioassembly_data=bioassembly_data,
            chemical_components_data=chem_data,
            atoms=self._atoms,
            residues=self._residues,
            chains=self._chains,
            bonds=self._bonds,
        )

    def copy_and_update_atoms(
            self,
            *,
            atom_name: np.ndarray | None = None,
            atom_element: np.ndarray | None = None,
            atom_x: np.ndarray | None = None,
            atom_y: np.ndarray | None = None,
            atom_z: np.ndarray | None = None,
            atom_b_factor: np.ndarray | None = None,
            atom_occupancy: np.ndarray | None = None,
    ) -> Self:
        """Returns a shallow copy with the atoms table updated."""
        new_atoms = structure_tables.Atoms(
            key=self._atoms.key,
            res_key=self._atoms.res_key,
            chain_key=self._atoms.chain_key,
            name=atom_name if atom_name is not None else self.atom_name,
            element=atom_element if atom_element is not None else self.atom_element,
            x=atom_x if atom_x is not None else self.atom_x,
            y=atom_y if atom_y is not None else self.atom_y,
            z=atom_z if atom_z is not None else self.atom_z,
            b_factor=(
                atom_b_factor if atom_b_factor is not None else self.atom_b_factor
            ),
            occupancy=(
                atom_occupancy
                if atom_occupancy is not None
                else self.atom_occupancy
            ),
        )
        return self.copy_and_update(atoms=new_atoms)

    def _cascade_delete(
            self,
            *,
            chains: structure_tables.Chains | None = None,
            residues: structure_tables.Residues | None = None,
            atoms: structure_tables.Atoms | None = None,
            bonds: structure_tables.Bonds | None = None,
    ) -> StructureTables:
        from torchfold.structure.structure_ops import cascade_delete
        return cascade_delete(
            self,
            chains=chains,
            residues=residues,
            atoms=atoms,
            bonds=bonds
        )

    def filter(
            self,
            mask: np.ndarray | None = None,
            *,
            apply_per_element: bool = False,
            invert: bool = False,
            cascade_delete: CascadeDelete = CascadeDelete.CHAINS,
            **predicate_by_field_name: table.FilterPredicate,
    ) -> Self:
        from torchfold.structure.structure_ops import structure_filter
        return structure_filter(
            self,
            mask=mask,
            apply_per_element=apply_per_element,
            invert=invert,
            cascade_delete=cascade_delete,
            **predicate_by_field_name
        )

    def filter_to_entity_type(
            self,
            *,
            protein: bool = False,
            rna: bool = False,
            dna: bool = False,
            dna_rna_hybrid: bool = False,
            ligand: bool = False,
            water: bool = False,
    ) -> Self:
        from torchfold.structure.structure_ops import filter_to_entity_type
        return filter_to_entity_type(
            self,
            protein=protein,
            rna=rna,
            dna=dna,
            dna_rna_hybrid=dna_rna_hybrid,
            ligand=ligand,
            water=water
        )

    @property
    def coords(self) -> np.ndarray:
        """A [..., num_atom, 3] shaped array of atom coordinates."""
        return np.stack([self._atoms.x, self._atoms.y, self._atoms.z], axis=-1)

    def chain_single_letter_sequence(
            self, include_missing_residues: bool = True
    ) -> Mapping[str, str]:
        from torchfold.structure.chain_manager import get_chain_single_letter_sequence
        return get_chain_single_letter_sequence(self, include_missing_residues=include_missing_residues)

    def chain_res_name_sequence(
            self,
            *,
            include_missing_residues: bool = True,
            fix_non_standard_polymer_res: bool = False,
    ) -> Mapping[str, Sequence[str]]:
        from torchfold.structure.chain_manager import get_chain_res_name_sequence
        return get_chain_res_name_sequence(
            self,
            include_missing_residues=include_missing_residues,
            fix_non_standard_polymer_res=fix_non_standard_polymer_res
        )

    @property
    def slice_leading_dims(self) -> '_LeadingDimSlice':
        """Used to create a new Structure by slicing into the leading dimensions.

        Example usage 1:

        ```
        final_state = multi_state_struc.slice_leading_dims[-1]
        ```

        Example usage 2:

        ```
        # Structure has leading batch and time dimensions.
        # Get final 3 time frames from first two batch elements.
        sliced_strucs = batched_trajectories.slice_leading_dims[:2, -3:]
        ```
        """
        return _LeadingDimSlice(self)

    def unstack(self, axis: int = 0) -> Sequence[Self]:
        """Unstacks a multi-model structure into a list of Structures.

        This method is the inverse of `stack`.

        Example usage:
        ```
        strucs = multi_dim_struc.unstack(axis=0)
        ```

        Args:
          axis: The axis to unstack over. The structures in the returned list won't
            have this axis in their coordinate of b-factor fields.

        Returns:
          A list of `Structure`s with length equal to the size of the specified
          axis in the coorinate field arrays.

        Raises:
          IndexError: If axis does not refer to one of the leading dimensions of
            `self.atoms_table.size`.
        """
        ndim = self._atoms.ndim
        if not (-ndim <= axis < ndim):
            raise IndexError(
                f'{axis=} is out of range for atom coordinate fields with {ndim=}.'
            )
        elif axis < 0:
            axis += ndim
        if axis == ndim - 1:
            raise IndexError(
                'axis must refer to one of the leading dimensions, not the final '
                f'dimension. The atom fields have {ndim=} and {axis=} was specified.'
            )
        unstacked = []
        leading_dim_slice = self.slice_leading_dims  # Compute once here.
        for i in range(self._atoms.shape[axis]):
            slice_i = (slice(None),) * axis + (i,)
            unstacked.append(leading_dim_slice[slice_i])
        return unstacked

    def reorder_chains(self, new_order: Sequence[str]) -> Self:
        from torchfold.structure.chain_manager import structure_reorder_chains
        return structure_reorder_chains(self, new_order)

    def rename_chain_ids(self, new_id_by_old_id: Mapping[str, str]) -> Self:
        from torchfold.structure.chain_manager import structure_rename_chain_ids
        return structure_rename_chain_ids(self, new_id_by_old_id)

    @functools.cached_property
    def chains(self) -> tuple[str, ...]:
        """Ordered internal chain IDs (label_asym_id) present in the Structure."""
        return tuple(self._chains.id)

    def to_mmcif_dict(self, *, coords_decimal_places: int = COORDS_DECIMAL_PLACES) -> mmcif.Mmcif:
        from torchfold.structure.io_mmcif import structure_to_mmcif_dict
        return structure_to_mmcif_dict(self, coords_decimal_places=coords_decimal_places)

    def to_mmcif(self, *, coords_decimal_places: int = COORDS_DECIMAL_PLACES) -> str:
        from torchfold.structure.io_mmcif import structure_to_mmcif_string
        return structure_to_mmcif_string(self, coords_decimal_places=coords_decimal_places)


class _LeadingDimSlice:
    """Helper class for slicing the leading dimensions of a `Structure`.

    Wraps a `Structure` instance and applies a slice operation to the coordinate
    fields and other fields that may have leading dimensions (e.g. b_factor).

    Example usage:
      t0_struc = multi_state_struc.slice_leading_dims[0]
    """

    def __init__(self, struc: Structure):
        self._struc = struc

    def __getitem__(self, *args, **kwargs) -> Structure:
        sliced_atom_cols = {}
        for col_name in structure_tables.Atoms.multimodel_cols:
            if (col := self._struc.atoms_table.get_column(col_name)).ndim > 1:
                sliced_col = col.__getitem__(*args, **kwargs)
                if (
                        not sliced_col.shape
                        or sliced_col.shape[-1] != self._struc.num_atoms
                ):
                    raise ValueError(
                        'Coordinate slice cannot change final (atom) dimension.'
                    )
                sliced_atom_cols[col_name] = sliced_col
        sliced_atoms = self._struc.atoms_table.copy_and_update(**sliced_atom_cols)
        return self._struc.copy_and_update(atoms=sliced_atoms, skip_validation=True)
