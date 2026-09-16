"""Bond representation for structure module."""

import collections
import dataclasses
import typing
from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np

from torchfold.structure.internal import table


@dataclasses.dataclass(frozen=True, kw_only=True)
class Bonds(table.Table):
    """Table of atomic bonds."""

    # mmCIF column: _struct_conn.conn_type_id
    # mmCIF desc: This data item is a pointer to _struct_conn_type.id in the
    #             STRUCT_CONN_TYPE category.
    # E.g.: "covale", "disulf", "hydrog", "metalc".
    type: np.ndarray

    # mmCIF column: _struct_conn.pdbx_role
    # mmCIF desc: The chemical or structural role of the interaction.
    # E.g.: "N-Glycosylation", "O-Glycosylation".
    role: np.ndarray

    # mmCIF columns: _struct_conn.ptnr1_*
    from_atom_key: np.ndarray

    # mmCIF columns: _struct_conn.ptnr2_*
    dest_atom_key: np.ndarray

    @classmethod
    def make_empty(cls) -> Self:
        return cls(
            key=np.empty((0,), dtype=np.int64),
            from_atom_key=np.empty((0,), dtype=np.int64),
            dest_atom_key=np.empty((0,), dtype=np.int64),
            type=np.empty((0,), dtype=object),
            role=np.empty((0,), dtype=object),
        )

    def get_atom_indices(
            self,
            atom_key: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the indices of the from/dest atoms in the atom_key array."""
        from_atom_missing = ~np.isin(self.from_atom_key, atom_key)
        dest_atom_missing = ~np.isin(self.dest_atom_key, atom_key)
        if np.any(from_atom_missing):
            raise ValueError(
                f'No atoms for from_atom_key {self.from_atom_key[from_atom_missing]}'
            )
        if np.any(dest_atom_missing):
            raise ValueError(
                f'No atoms for dest_atom_key {self.dest_atom_key[dest_atom_missing]}'
            )
        sort_indices = np.argsort(atom_key)
        from_indices_sorted = np.searchsorted(
            atom_key, self.from_atom_key, sorter=sort_indices
        )
        dest_indices_sorted = np.searchsorted(
            atom_key, self.dest_atom_key, sorter=sort_indices
        )
        from_indices = sort_indices[from_indices_sorted]
        dest_indices = sort_indices[dest_indices_sorted]
        return from_indices, dest_indices

    def restrict_to_atoms(self, atom_key: np.ndarray) -> Self:
        if not self.size:  # Early-out for empty table.
            return self
        from_atom_mask = np.isin(self.from_atom_key, atom_key)
        dest_atom_mask = np.isin(self.dest_atom_key, atom_key)
        mask = np.logical_and(from_atom_mask, dest_atom_mask)
        return typing.cast(Bonds, self.filter(mask=mask))

    def to_mmcif_dict_from_atom_arrays(
            self,
            atom_key: np.ndarray,
            chain_id: np.ndarray,
            res_id: np.ndarray,
            res_name: np.ndarray,
            atom_name: np.ndarray,
            auth_asym_id: np.ndarray,
            auth_seq_id: np.ndarray,
            insertion_code: np.ndarray,
    ) -> Mapping[str, Sequence[str] | np.ndarray]:
        """Returns a dict suitable for building a CifDict, representing bonds.

        Args:
          atom_key: A (num_atom,) integer array of atom_keys.
          chain_id: A (num_atom,) array of label_asym_id strings.
          res_id: A (num_atom,) array of label_seq_id strings.
          res_name: A (num_atom,) array of label_comp_id strings.
          atom_name: A (num_atom,) array of label_atom_id strings.
          auth_asym_id: A (num_atom,) array of auth_asym_id strings.
          auth_seq_id: A (num_atom,) array of auth_seq_id strings.
          insertion_code: A (num_atom,) array of insertion code strings.
        """
        mmcif_dict = collections.defaultdict(list)
        ptnr1_indices, ptnr2_indices = self.get_atom_indices(atom_key)

        mmcif_dict['_struct_conn.ptnr1_label_asym_id'] = chain_id[ptnr1_indices]
        mmcif_dict['_struct_conn.ptnr2_label_asym_id'] = chain_id[ptnr2_indices]
        mmcif_dict['_struct_conn.ptnr1_label_comp_id'] = res_name[ptnr1_indices]
        mmcif_dict['_struct_conn.ptnr2_label_comp_id'] = res_name[ptnr2_indices]
        mmcif_dict['_struct_conn.ptnr1_label_seq_id'] = res_id[ptnr1_indices]
        mmcif_dict['_struct_conn.ptnr2_label_seq_id'] = res_id[ptnr2_indices]
        mmcif_dict['_struct_conn.ptnr1_label_atom_id'] = atom_name[ptnr1_indices]
        mmcif_dict['_struct_conn.ptnr2_label_atom_id'] = atom_name[ptnr2_indices]

        mmcif_dict['_struct_conn.ptnr1_auth_asym_id'] = auth_asym_id[ptnr1_indices]
        mmcif_dict['_struct_conn.ptnr2_auth_asym_id'] = auth_asym_id[ptnr2_indices]
        mmcif_dict['_struct_conn.ptnr1_auth_seq_id'] = auth_seq_id[ptnr1_indices]
        mmcif_dict['_struct_conn.ptnr2_auth_seq_id'] = auth_seq_id[ptnr2_indices]
        mmcif_dict['_struct_conn.pdbx_ptnr1_PDB_ins_code'] = insertion_code[
            ptnr1_indices
        ]
        mmcif_dict['_struct_conn.pdbx_ptnr2_PDB_ins_code'] = insertion_code[
            ptnr2_indices
        ]

        label_alt_id = ['?'] * self.size
        mmcif_dict['_struct_conn.pdbx_ptnr1_label_alt_id'] = label_alt_id
        mmcif_dict['_struct_conn.pdbx_ptnr2_label_alt_id'] = label_alt_id

        # We need to set this to make visualisation work in NGL/PyMOL.
        mmcif_dict['_struct_conn.pdbx_value_order'] = ['?'] * self.size

        # We use a symmetry of 1_555 which is the no-op transformation. Other
        # values are used when bonds involve atoms that only exist after expanding
        # the bioassembly, but we don't support this kind of bond at the moment.
        symmetry = ['1_555'] * self.size
        mmcif_dict['_struct_conn.ptnr1_symmetry'] = symmetry
        mmcif_dict['_struct_conn.ptnr2_symmetry'] = symmetry
        bond_type_counter = collections.Counter()
        for bond_row in self.iterrows():
            bond_type = bond_row['type']
            bond_type_counter[bond_type] += 1
            mmcif_dict['_struct_conn.id'].append(
                f'{bond_type}{bond_type_counter[bond_type]}'
            )
            mmcif_dict['_struct_conn.pdbx_role'].append(bond_row['role'])
            mmcif_dict['_struct_conn.conn_type_id'].append(bond_type)

        bond_types = np.unique(self.type)
        mmcif_dict['_struct_conn_type.id'] = bond_types
        unknown = ['?'] * len(bond_types)
        mmcif_dict['_struct_conn_type.criteria'] = unknown
        mmcif_dict['_struct_conn_type.reference'] = unknown

        return dict(mmcif_dict)
