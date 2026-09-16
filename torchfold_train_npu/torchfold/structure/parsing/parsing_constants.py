import dataclasses
import enum
from collections.abc import Mapping
from typing import TypeAlias

ChainIndex: TypeAlias = int
ResIndex: TypeAlias = int
AtomName: TypeAlias = str
BondAtomId: TypeAlias = tuple[ChainIndex, ResIndex, AtomName]

_INSERTION_CODE_REMAP: Mapping[str, str] = {'.': '?'}


class NoAtomsError(Exception):
    """Raise when the mmCIF does not have any atoms."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BondIndices:
    from_indices: list[int]
    dest_indices: list[int]


@enum.unique
class ModelID(enum.Enum):
    """Values for specifying model IDs when parsing."""
    FIRST = 1  # The first model in the file.
    ALL = 2  # All models in the file.


@enum.unique
class SequenceFormat(enum.Enum):
    """The possible formats for an input sequence."""
    FASTA = 'fasta'  # One-letter code used in FASTA.
    CCD_CODES = 'ccd_codes'  # Multiple-letter chemical components dictionary ids.
    LIGAND_SMILES = 'ligand_smiles'  # SMILES string defining a molecule.
