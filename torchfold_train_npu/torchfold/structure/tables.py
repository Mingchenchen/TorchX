"""Table implementations for the Structure class."""

import dataclasses
import functools
import typing
from collections.abc import Mapping, Sequence
from typing import ClassVar, Self, Any
import numpy as np

from torchfold.constants import mmcif_names
from torchfold.structure import bonds as bonds_module
from torchfold.structure.internal import table

Bonds = bonds_module.Bonds


def _default(
        candidate_value: np.ndarray | None, default_value: Sequence[Any], dtype: Any
) -> np.ndarray:
    if candidate_value is None:
        return np.array(default_value, dtype=dtype)
    return np.array(candidate_value, dtype=dtype)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AuthorNamingScheme:
    """A mapping from internal values to author values in a mmCIF.

    Fields:
      auth_asym_id: A mapping from label_asym_id to auth_asym_id.
      auth_seq_id: A mapping from label_asym_id to a mapping from
        label_seq_id to auth_seq_id.
      insertion_code: A mapping from label_asym_id to a mapping from
        label_seq_id to insertion codes.
      entity_id: A mapping from label_asym_id to _entity.id.
      entity_desc: A mapping from _entity.id to _entity.pdbx_description.
    """

    auth_asym_id: Mapping[str, str]
    auth_seq_id: Mapping[str, Mapping[int, str]]
    insertion_code: Mapping[str, Mapping[int, str | None]]
    entity_id: Mapping[str, str]
    entity_desc: Mapping[str, str]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Atoms(table.Table):
    """Table of atoms in a Structure."""

    chain_key: np.ndarray
    res_key: np.ndarray
    name: np.ndarray
    element: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    b_factor: np.ndarray
    occupancy: np.ndarray
    multimodel_cols: ClassVar[tuple[str, ...]] = (
        'x',
        'y',
        'z',
        'b_factor',
        'occupancy',
    )

    def __post_init__(self):
        # Validates that the atom coordinates, b-factors and occupancies are finite.
        for column_name in ('x', 'y', 'z', 'b_factor', 'occupancy'):
            column = self.get_column(column_name)
            if not np.isfinite(column).all():
                raise ValueError(
                    f'Column {column_name} must not contain NaN/inf values.'
                )
        # super().__post_init__() can't be used as that causes the following error:
        # TypeError: super(type, obj): obj must be an instance or subtype of type
        super(Atoms, self).__post_init__()

    @classmethod
    def make_empty(cls) -> Self:
        return cls(
            key=np.array([], dtype=np.int64),
            chain_key=np.array([], dtype=np.int64),
            res_key=np.array([], dtype=np.int64),
            name=np.array([], dtype=object),
            element=np.array([], dtype=object),
            x=np.array([], dtype=np.float32),
            y=np.array([], dtype=np.float32),
            z=np.array([], dtype=np.float32),
            b_factor=np.array([], dtype=np.float32),
            occupancy=np.array([], dtype=np.float32),
        )

    @classmethod
    def from_defaults(
            cls,
            *,
            chain_key: np.ndarray,
            res_key: np.ndarray,
            key: np.ndarray | None = None,
            name: np.ndarray | None = None,
            element: np.ndarray | None = None,
            x: np.ndarray | None = None,
            y: np.ndarray | None = None,
            z: np.ndarray | None = None,
            b_factor: np.ndarray | None = None,
            occupancy: np.ndarray | None = None,
    ) -> Self:
        """Create an Atoms table with minimal user inputs."""
        num_atoms = len(chain_key)
        if not num_atoms:
            return cls.make_empty()
        return Atoms(
            chain_key=chain_key,
            res_key=res_key,
            key=_default(key, np.arange(num_atoms), np.int64),
            name=_default(name, ['?'] * num_atoms, object),
            element=_default(element, ['?'] * num_atoms, object),
            x=_default(x, [0.0] * num_atoms, np.float32),
            y=_default(y, [0.0] * num_atoms, np.float32),
            z=_default(z, [0.0] * num_atoms, np.float32),
            b_factor=_default(b_factor, [0.0] * num_atoms, np.float32),
            occupancy=_default(occupancy, [1.0] * num_atoms, np.float32),
        )

    def get_value_by_index(
            self, column_name: str, index: int
    ) -> table.TableEntry | np.ndarray:
        if column_name in self.multimodel_cols:
            return self.get_column(column_name)[..., index]
        else:
            return self.get_column(column_name)[index]

    def copy_and_update_coords(self, coords: np.ndarray) -> Self:
        """Returns a copy with the x, y and z columns updated."""
        if coords.shape[-1] != 3:
            raise ValueError(
                f'Expecting 3-dimensional coordinates, got {coords.shape}'
            )
        return typing.cast(
            Atoms,
            self.copy_and_update(
                x=coords[..., 0], y=coords[..., 1], z=coords[..., 2]
            ),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.x.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @functools.cached_property
    def num_models(self) -> int:
        """The number of models of this Structure."""
        leading_dims = self.shape[:-1]
        match leading_dims:
            case ():
                return 1
            case (single_leading_dim_size, ):
                return single_leading_dim_size
            case _:
                raise ValueError(
                    'num_models not defined for atom tables with more than one '
                    'leading dimension.'
                )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Residues(table.Table):
    """Table of residues in a Structure."""

    chain_key: np.ndarray
    id: np.ndarray
    name: np.ndarray
    auth_seq_id: np.ndarray
    insertion_code: np.ndarray

    @classmethod
    def make_empty(cls) -> Self:
        return cls(
            key=np.array([], dtype=np.int64),
            chain_key=np.array([], dtype=np.int64),
            id=np.array([], dtype=np.int32),
            name=np.array([], dtype=object),
            auth_seq_id=np.array([], dtype=object),
            insertion_code=np.array([], dtype=object),
        )

    @classmethod
    def from_defaults(
            cls,
            *,
            id: np.ndarray,  # pylint:disable=redefined-builtin
            chain_key: np.ndarray,
            key: np.ndarray | None = None,
            name: np.ndarray | None = None,
            auth_seq_id: np.ndarray | None = None,
            insertion_code: np.ndarray | None = None,
    ) -> Self:
        """Create a Residues table with minimal user inputs."""
        num_res = len(id)
        if not num_res:
            return cls.make_empty()
        return Residues(
            key=_default(key, np.arange(num_res), np.int64),
            id=id,
            chain_key=chain_key,
            name=_default(name, ['UNK'] * num_res, object),
            auth_seq_id=_default(auth_seq_id, id.astype(str), object),
            insertion_code=_default(insertion_code, ['?'] * num_res, object),
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Chains(table.Table):
    """Table of chains in a Structure."""

    id: np.ndarray
    type: np.ndarray
    auth_asym_id: np.ndarray
    entity_id: np.ndarray
    entity_desc: np.ndarray

    @classmethod
    def make_empty(cls) -> Self:
        return cls(
            key=np.array([], dtype=np.int64),
            id=np.array([], dtype=object),
            type=np.array([], dtype=object),
            auth_asym_id=np.array([], dtype=object),
            entity_id=np.array([], dtype=object),
            entity_desc=np.array([], dtype=object),
        )

    @classmethod
    def from_defaults(
            cls,
            *,
            id: np.ndarray,  # pylint:disable=redefined-builtin
            key: np.ndarray | None = None,
            type: np.ndarray | None = None,  # pylint:disable=redefined-builtin
            auth_asym_id: np.ndarray | None = None,
            entity_id: np.ndarray | None = None,
            entity_desc: np.ndarray | None = None,
    ) -> Self:
        """Create a Chains table with minimal user inputs."""
        num_chains = len(id)
        if not num_chains:
            return cls.make_empty()

        return Chains(
            key=_default(key, np.arange(num_chains), np.int64),
            id=id,
            type=_default(type, [mmcif_names.PROTEIN_CHAIN] * num_chains, object),
            auth_asym_id=_default(auth_asym_id, id, object),
            entity_id=_default(
                entity_id, np.arange(1, num_chains + 1).astype(str), object
            ),
            entity_desc=_default(entity_desc, ['.'] * num_chains, object),
        )
