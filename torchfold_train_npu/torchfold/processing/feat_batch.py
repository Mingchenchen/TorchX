"""Batch dataclass."""
import dataclasses
from typing import Self

import jax

from torchfold.processing import features


@dataclasses.dataclass(frozen=True)
class Batch:
    """Dataclass containing batch."""

    msa: features.MSA
    templates: features.Templates
    token_features: features.TokenFeatures
    ref_structure: features.RefStructure
    predicted_structure_info: features.PredictedStructureInfo
    polymer_ligand_bond_info: features.PolymerLigandBondInfo
    ligand_ligand_bond_info: features.LigandLigandBondInfo
    pseudo_beta_info: features.PseudoBetaInfo
    atom_cross_att: features.AtomCrossAtt
    convert_model_output: features.ConvertModelOutput
    frames: features.Frames

    @property
    def num_res(self) -> int:
        return self.token_features.aatype.shape[-1]

    @classmethod
    def from_data_dict(cls, batch: features.BatchDict) -> Self:
        """Construct batch object from dictionary."""
        return cls(
            msa=features.MSA.from_data_dict(batch),
            templates=features.Templates.from_data_dict(batch),
            token_features=features.TokenFeatures.from_data_dict(batch),
            ref_structure=features.RefStructure.from_data_dict(batch),
            predicted_structure_info=features.PredictedStructureInfo.from_data_dict(
                batch
            ),
            polymer_ligand_bond_info=features.PolymerLigandBondInfo.from_data_dict(
                batch
            ),
            ligand_ligand_bond_info=features.LigandLigandBondInfo.from_data_dict(
                batch
            ),
            pseudo_beta_info=features.PseudoBetaInfo.from_data_dict(batch),
            atom_cross_att=features.AtomCrossAtt.from_data_dict(batch),
            convert_model_output=features.ConvertModelOutput.from_data_dict(batch),
            frames=features.Frames.from_data_dict(batch),
        )


jax.tree_util.register_dataclass(
    Batch,
    data_fields=[f.name for f in dataclasses.fields(Batch)],
    meta_fields=[],
)
