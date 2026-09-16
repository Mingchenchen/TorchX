"""Structure class for representing and processing molecular structures."""
import dataclasses

from torchfold.structure.tables import Atoms, Bonds, Chains, Residues


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StructureTables:
    chains: Chains
    residues: Residues
    atoms: Atoms
    bonds: Bonds
