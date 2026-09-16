"""Batch dataclass."""
from typing import Self, Optional
import dataclasses
import torch

from torchfold.model import features


@dataclasses.dataclass(frozen=True)
class Batch:
    """Dataclass containing batch."""

    msa: features.MSA
    templates: features.Templates
    token_features: features.TokenFeatures
    ref_structure: features.RefStructure
    ground_truth_structure: features.GroundTruthStructure
    predicted_structure_info: features.PredictedStructureInfo
    polymer_ligand_bond_info: features.PolymerLigandBondInfo
    ligand_ligand_bond_info: features.LigandLigandBondInfo
    pseudo_beta_info: features.PseudoBetaInfo
    atom_cross_att: features.AtomCrossAtt
    convert_model_output: features.ConvertModelOutput
    frames: features.Frames
    data_source_idx: Optional[torch.Tensor] = None
    # Epitope guidance / constraint related fields (optional)
    use_epitope_guidance_this_sample: Optional[torch.Tensor] = None
    sampled_epitope_mask: Optional[torch.Tensor] = None
    paratope_mask: Optional[torch.Tensor] = None
    epitope_mask: Optional[torch.Tensor] = None

    @property
    def num_res(self) -> int:
        return self.token_features.aatype.shape[-1]

    @classmethod
    def from_data_dict(cls, batch: features.BatchDict) -> Self:
        """Construct batch object from dictionary."""
        data_source_idx = batch.get('data_source_idx', None)
        use_epitope_guidance_this_sample = batch.get('use_epitope_guidance_this_sample', None)
        sampled_epitope_mask = batch.get('sampled_epitope_mask', None)
        paratope_mask = batch.get('paratope_mask', None)
        epitope_mask = batch.get('epitope_mask', None)
        
        return cls(
            msa=features.MSA.from_data_dict(batch),
            templates=features.Templates.from_data_dict(batch),
            token_features=features.TokenFeatures.from_data_dict(batch),
            ref_structure=features.RefStructure.from_data_dict(batch),
            ground_truth_structure=features.GroundTruthStructure.from_data_dict(batch),
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
            convert_model_output=features.ConvertModelOutput.from_data_dict(
                batch),
            frames=features.Frames.from_data_dict(batch),
            data_source_idx=data_source_idx,
            use_epitope_guidance_this_sample=use_epitope_guidance_this_sample,
            sampled_epitope_mask=sampled_epitope_mask,
            paratope_mask=paratope_mask,
            epitope_mask=epitope_mask,
        )

    def as_data_dict(self) -> features.BatchDict:
        """Converts batch object to dictionary."""
        output = {
            **self.msa.as_data_dict(),
            **self.templates.as_data_dict(),
            **self.token_features.as_data_dict(),
            **self.ref_structure.as_data_dict(),
            **self.ground_truth_structure.as_data_dict(),
            **self.predicted_structure_info.as_data_dict(),
            **self.polymer_ligand_bond_info.as_data_dict(),
            **self.ligand_ligand_bond_info.as_data_dict(),
            **self.pseudo_beta_info.as_data_dict(),
            **self.atom_cross_att.as_data_dict(),
            **self.convert_model_output.as_data_dict(),
            **self.frames.as_data_dict(),
        }
        if self.data_source_idx is not None:
            output['data_source_idx'] = self.data_source_idx
        if self.use_epitope_guidance_this_sample is not None:
            output['use_epitope_guidance_this_sample'] = self.use_epitope_guidance_this_sample
        if self.sampled_epitope_mask is not None:
            output['sampled_epitope_mask'] = self.sampled_epitope_mask
        if self.paratope_mask is not None:
            output['paratope_mask'] = self.paratope_mask
        if self.epitope_mask is not None:
            output['epitope_mask'] = self.epitope_mask
        return output
