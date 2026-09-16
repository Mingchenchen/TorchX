"""Post-processing utilities for TorchFold inference results."""

import concurrent.futures
import dataclasses
import datetime
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
from absl import logging

from torchx import version, structure
from torchx.processing import confidence_types
from torchx.processing import confidences
from torchx.processing import feat_batch
from torchx.processing import features
from torchx.processing import mmcif_metadata
from torchx.processing.atom_layout import atom_layout

ModelResult: TypeAlias = confidence_types.ModelResult
InferenceResult: TypeAlias = confidence_types.InferenceResult


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProcessedInferenceResult:
    """Stores attributes of a processed inference result.

    Attributes:
      cif: CIF file containing an inference result.
      mean_confidence_1d: Mean 1D confidence calculated from confidence_1d.
      ranking_score: Ranking score extracted from CIF metadata.
      structure_confidence_summary_json: Content of JSON file with structure
        confidences summary calculated from CIF file.
      structure_full_data_json: Content of JSON file with structure full
        confidences calculated from CIF file.
      model_id: Identifier of the model that produced the inference result.
    """

    cif: bytes
    mean_confidence_1d: float
    ranking_score: float
    structure_confidence_summary_json: bytes
    structure_full_data_json: bytes
    model_id: bytes


def get_predicted_structure(
        result: ModelResult, batch: feat_batch.Batch
) -> structure.Structure:
    """Creates the predicted structure and ion preditions.

    Args:
      result: model output in a model specific layout
      batch: model input batch

    Returns:
      Predicted structure.
    """
    model_output_coords = result['diffusion_samples']['atom_positions']

    # Rearrange model output coordinates to the flat output layout.
    model_output_to_flat = atom_layout.compute_gather_idxs(
        source_layout=batch.convert_model_output.token_atoms_layout,
        target_layout=batch.convert_model_output.flat_output_layout,
    )
    pred_flat_atom_coords = atom_layout.convert(
        gather_info=model_output_to_flat,
        arr=model_output_coords,
        layout_axes=(-3, -2),
    )

    predicted_lddt = result.get('predicted_lddt')

    if predicted_lddt is not None:
        pred_flat_b_factors = atom_layout.convert(
            gather_info=model_output_to_flat,
            arr=predicted_lddt,
            layout_axes=(-2, -1),
        )
    else:
        # Handle models which don't have predicted_lddt outputs.
        pred_flat_b_factors = np.zeros(pred_flat_atom_coords.shape[:-1])

    (missing_atoms_indices,) = np.nonzero(model_output_to_flat.gather_mask == 0)
    if missing_atoms_indices.shape[0] > 0:
        missing_atoms_flat_layout = batch.convert_model_output.flat_output_layout[
            missing_atoms_indices
        ]
        missing_atoms_uids = list(
            zip(
                missing_atoms_flat_layout.chain_id,
                missing_atoms_flat_layout.res_id,
                missing_atoms_flat_layout.res_name,
                missing_atoms_flat_layout.atom_name,
            )
        )
        logging.warning(
            'Target %s: warning: %s atoms were not predicted by the '
            'model, setting their coordinates to (0, 0, 0). '
            'Missing atoms: %s',
            batch.convert_model_output.empty_output_struc.name,
            missing_atoms_indices.shape[0],
            missing_atoms_uids,
        )

    # Put them into a structure
    pred_struc = batch.convert_model_output.empty_output_struc
    pred_struc = pred_struc.copy_and_update_atoms(
        atom_x=pred_flat_atom_coords[..., 0],
        atom_y=pred_flat_atom_coords[..., 1],
        atom_z=pred_flat_atom_coords[..., 2],
        atom_b_factor=pred_flat_b_factors,
        atom_occupancy=np.ones(pred_flat_atom_coords.shape[:-1]),  # Always 1.0.
    )
    # Set manually/differently when adding metadata.
    pred_struc = pred_struc.copy_and_update_globals(release_date=None)
    return pred_struc


def _compute_ptm(
        result: ModelResult,
        num_tokens: int,
        asym_id: np.ndarray,
        pae_single_mask: np.ndarray,
        interface: bool,
) -> np.ndarray:
    """Computes the pTM metrics from PAE."""
    return np.stack(
        [
            confidences.predicted_tm_score(
                tm_adjusted_pae=tm_adjusted_pae[:num_tokens, :num_tokens],
                asym_id=asym_id,
                pair_mask=pae_single_mask[:num_tokens, :num_tokens],
                interface=interface,
            )
            for tm_adjusted_pae in result['tmscore_adjusted_pae_global']
        ],
        axis=0,
    )


def _compute_chain_pair_iptm(
        num_tokens: int,
        asym_ids: np.ndarray,
        mask: np.ndarray,
        tm_adjusted_pae: np.ndarray,
) -> np.ndarray:
    """Computes the chain pair ipTM metrics from PAE."""
    return np.stack(
        [
            confidences.chain_pairwise_predicted_tm_scores(
                tm_adjusted_pae=sample_tm_adjusted_pae[:num_tokens],
                asym_id=asym_ids[:num_tokens],
                pair_mask=mask[:num_tokens, :num_tokens],
            )
            for sample_tm_adjusted_pae in tm_adjusted_pae
        ],
        axis=0,
    )


def get_inference_result(
        batch: features.BatchDict,
        result: ModelResult,
        target_name: str = '',
) -> Iterable[InferenceResult]:
    """Get the predicted structure, scalars, and arrays for inference."""
    del target_name
    batch = feat_batch.Batch.from_data_dict(batch)

    # Retrieve structure and construct a predicted structure.
    pred_structure = get_predicted_structure(result=result, batch=batch)

    num_tokens = batch.token_features.seq_length.item()

    pae_single_mask = np.tile(
        batch.frames.mask[:, None],
        [1, batch.frames.mask.shape[0]],
    )
    ptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=False,
    )
    iptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=True,
    )
    ptm_iptm_average = 0.8 * iptm + 0.2 * ptm

    asym_ids = batch.token_features.asym_id[:num_tokens]
    # Map asym IDs back to chain IDs.
    chain_ids = [pred_structure.chains[asym_id - 1] for asym_id in asym_ids]
    res_ids = batch.token_features.residue_index[:num_tokens]

    if len(np.unique(asym_ids[:num_tokens])) > 1:
        ranking_confidence = ptm_iptm_average
    else:
        ranking_confidence = ptm

    contact_probs = result['distogram']['contact_probs']
    # Compute PAE related summaries.
    _, chain_pair_pae_min, _ = confidences.chain_pair_pae(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
    )
    chain_pair_pde_mean, chain_pair_pde_min = confidences.chain_pair_pde(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pde=result['full_pde'],
    )
    intra_chain_single_pde, cross_chain_single_pde, _ = confidences.pde_single(
        num_tokens,
        batch.token_features.asym_id,
        result['full_pde'],
        contact_probs,
    )
    pae_metrics = confidences.pae_metrics(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
        contact_probs=contact_probs,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    ranking_confidence_pae = confidences.rank_metric(
        result['full_pae'],
        contact_probs * batch.frames.mask[:, None].astype(float),
    )
    chain_pair_iptm = _compute_chain_pair_iptm(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        mask=pae_single_mask,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    iptm_ichain = chain_pair_iptm.diagonal(axis1=-2, axis2=-1)
    iptm_xchain = confidences.get_iptm_xchain(chain_pair_iptm)

    predicted_distance_errors = result['average_pde']

    pred_structures = pred_structure.unstack()
    num_workers = len(pred_structures)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
    ) as executor:
        has_clash = list(executor.map(confidences.has_clash, pred_structures))
        fraction_disordered = list(
            executor.map(confidences.fraction_disordered, pred_structures)
        )

    for idx, pred_structure in enumerate(pred_structures):
        ranking_score = confidences.get_ranking_score(
            ptm=ptm[idx],
            iptm=iptm[idx],
            fraction_disordered_=fraction_disordered[idx],
            has_clash_=has_clash[idx],
        )
        yield InferenceResult(
            predicted_structure=pred_structure,
            numerical_data={
                'full_pde': result['full_pde'][idx, :num_tokens, :num_tokens],
                'full_pae': result['full_pae'][idx, :num_tokens, :num_tokens],
                'contact_probs': contact_probs[:num_tokens, :num_tokens],
            },
            metadata={
                'predicted_distance_error': predicted_distance_errors[idx],
                'ranking_score': ranking_score,
                'fraction_disordered': fraction_disordered[idx],
                'has_clash': has_clash[idx],
                'predicted_tm_score': ptm[idx],
                'interface_predicted_tm_score': iptm[idx],
                'chain_pair_pde_mean': chain_pair_pde_mean[idx],
                'chain_pair_pde_min': chain_pair_pde_min[idx],
                'chain_pair_pae_min': chain_pair_pae_min[idx],
                'ptm': ptm[idx],
                'iptm': iptm[idx],
                'ptm_iptm_average': ptm_iptm_average[idx],
                'intra_chain_single_pde': intra_chain_single_pde[idx],
                'cross_chain_single_pde': cross_chain_single_pde[idx],
                'pae_ichain': pae_metrics['pae_ichain'][idx],
                'pae_xchain': pae_metrics['pae_xchain'][idx],
                'ranking_confidence': ranking_confidence[idx],
                'ranking_confidence_pae': ranking_confidence_pae[idx],
                'chain_pair_iptm': chain_pair_iptm[idx],
                'iptm_ichain': iptm_ichain[idx],
                'iptm_xchain': iptm_xchain[idx],
                'token_chain_ids': chain_ids,
                'token_res_ids': res_ids,
            },
            model_id=result['__identifier__'],
            debug_outputs={},
        )


def _build_processed_result(
        inference_result: InferenceResult,
        confidence_1d: confidence_types.AtomConfidence,
        summary_json: str,
        full_json: str,
) -> ProcessedInferenceResult:
    timestamp = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    cif_with_metadata = mmcif_metadata.add_metadata_to_mmcif(
        old_cif=inference_result.predicted_structure.to_mmcif_dict(),
        version=f'{version.__version__} @ {timestamp}',
        model_id=inference_result.model_id,
    )
    cif = mmcif_metadata.add_legal_comment(cif_with_metadata.to_string())
    cif = cif.encode('utf-8')
    mean_confidence_1d = np.mean(confidence_1d.confidence)

    return ProcessedInferenceResult(
        cif=cif,
        mean_confidence_1d=mean_confidence_1d,
        ranking_score=float(inference_result.metadata['ranking_score']),
        structure_confidence_summary_json=summary_json.encode('utf-8'),
        structure_full_data_json=full_json.encode('utf-8'),
        model_id=inference_result.model_id,
    )


def post_process_inference_result(
        inference_result: InferenceResult,
) -> ProcessedInferenceResult:
    """Returns cif, confidence_1d_json, confidence_2d_json, mean_confidence_1d, and ranking confidence."""
    confidence_1d = confidence_types.AtomConfidence.from_inference_result(
        inference_result
    )
    summary_json = (
        confidence_types.StructureConfidenceSummary.from_inference_result(
            inference_result
        )
        .to_json()
    )
    full_json = (
        confidence_types.StructureConfidenceFull.from_inference_result(
            inference_result
        )
        .to_json()
    )
    return _build_processed_result(inference_result, confidence_1d, summary_json, full_json)


def post_process_inference_result_for_score(
        inference_result: InferenceResult,
) -> ProcessedInferenceResult:
    """Returns cif, confidence_1d_json, confidence_2d_json, mean_confidence_1d, and ranking confidence."""
    confidence_1d = confidence_types.AtomConfidence.from_inference_result_no_round(
        inference_result
    )
    summary_json = (
        confidence_types.StructureConfidenceSummary.from_inference_result(
            inference_result
        )
        .to_json_no_round()
    )
    full_json = (
        confidence_types.StructureConfidenceFull.from_inference_result(
            inference_result
        )
        .to_json_no_round()
    )
    return _build_processed_result(inference_result, confidence_1d, summary_json, full_json)
