import os
from typing import TypeAlias, Any

from torchx.processing import confidence_types
from torchx.processing.post_processing import (
    post_process_inference_result,
    post_process_inference_result_for_score,
)

InferenceResult: TypeAlias = confidence_types.InferenceResult


def _write_file(file_path: os.PathLike[str] | str, content: bytes | str, mode: str) -> None:
    """Helper function to handle generic file writing."""
    with open(file_path, mode) as f:
        f.write(content)


def _write_common_outputs(
    processed_result: Any,
    output_dir: os.PathLike[str] | str,
    prefix: str,
    terms_of_use: str | None = None,
    include_confidences: bool = True
) -> None:
    """Helper function to handle shared output writing logic across tasks."""
    # 1. Write the core CIF file
    _write_file(os.path.join(output_dir, f'{prefix}model.cif'), processed_result.cif, 'wb')

    # 2. Write confidence JSONs if required (used by fold and score)
    if include_confidences:
        _write_file(
            os.path.join(output_dir, f'{prefix}summary_confidences.json'),
            processed_result.structure_confidence_summary_json,
            'wb'
        )
        _write_file(
            os.path.join(output_dir, f'{prefix}confidences.json'),
            processed_result.structure_full_data_json,
            'wb'
        )

    # 3. Write terms of use if provided
    if terms_of_use is not None:
        _write_file(os.path.join(output_dir, 'TERMS_OF_USE.md'), terms_of_use, 'wt')


def write_output_for_design(
        inference_result: InferenceResult,
        output_dir: os.PathLike[str] | str | None = None,
        terms_of_use: str | None = None,
        name: str | None = None,
        last_stage_epoch_inf: str | None = None,
) -> None:
    """Writes processed inference result to a directory for the design task."""
    if output_dir is None:
        return

    processed_result = post_process_inference_result(inference_result)

    # Build design-specific prefix
    prefix = f'{name}_' if name is not None else ''
    if last_stage_epoch_inf is not None:
        prefix += f'{last_stage_epoch_inf}_'

    _write_common_outputs(
        processed_result=processed_result,
        output_dir=output_dir,
        prefix=prefix,
        terms_of_use=terms_of_use,
        include_confidences=False  # Design task does not output confidence files
    )


def write_output_for_fold(
        inference_result: InferenceResult,
        output_dir: os.PathLike[str] | str,
        terms_of_use: str | None = None,
        name: str | None = None,
) -> None:
    """Writes processed inference result to a directory for the fold task."""
    processed_result = post_process_inference_result(inference_result)
    prefix = f'{name}_' if name is not None else ''

    _write_common_outputs(processed_result, output_dir, prefix, terms_of_use)


def write_output_for_score(
        inference_result: InferenceResult,
        output_dir: os.PathLike[str] | str,
        terms_of_use: str | None = None,
        name: str | None = None,
) -> None:
    """Writes processed inference result to a directory for the score task."""
    # Note: Score task uses a distinct post-processing function
    processed_result = post_process_inference_result_for_score(inference_result)
    prefix = f'{name}_' if name is not None else ''

    _write_common_outputs(processed_result, output_dir, prefix, terms_of_use)