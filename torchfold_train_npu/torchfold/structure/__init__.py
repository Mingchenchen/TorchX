"""Structure module initialization."""

# pylint: disable=g-importing-member
from torchfold.structure.bioassemblies import BioassemblyData
from torchfold.structure.bonds import Bonds
from torchfold.structure.chemical_components import ChemCompEntry
from torchfold.structure.chemical_components import ChemicalComponentsData
from torchfold.structure.chemical_components import get_data_for_ccd_components
from torchfold.structure.chemical_components import populate_missing_ccd_data
from torchfold.structure.mmcif import BondParsingError
from torchfold.structure.internal.constants import ARRAY_FIELDS, ATOM_FIELDS, CHAIN_FIELDS, RESIDUE_FIELDS
from torchfold.structure.structure import AuthorNamingScheme
from torchfold.structure.structure import CascadeDelete
from torchfold.structure.structure import Structure
from torchfold.structure.tables import Atoms
from torchfold.structure.tables import Chains
from torchfold.structure.tables import Residues

__all__ = [
    "BioassemblyData",
    "Bonds",
    "ChemCompEntry",
    "ChemicalComponentsData",
    "get_data_for_ccd_components",
    "populate_missing_ccd_data",
    "BondParsingError",
    "ARRAY_FIELDS",
    "ATOM_FIELDS",
    "CHAIN_FIELDS",
    "RESIDUE_FIELDS",
    "AuthorNamingScheme",
    "CascadeDelete",
    "Structure",
    "Atoms",
    "Chains",
    "Residues",
]
