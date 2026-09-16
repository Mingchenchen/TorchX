import csv
import dataclasses
import multiprocessing
import os
import pathlib
import random
import shutil
import string
import time
from collections.abc import Mapping, Sequence
from typing import overload

import numpy as np
import torch
import torch.utils._pytree as pytree
import torchx.cpp
from absl import app
from absl import flags
from torchx.common import folding_input
from torchx.constants import chemical_components
from torchx.data import featurisation
from torchx.data import pipeline
from torchx.processing import features
from torchx.processing import output_handlers
from torchx.processing import post_processing

try:
    import torch.distributed as dist
except (ImportError, OSError):
    dist = None

from torchfold.nn import fastnn_config
from torchfold.parallel import (
    inject_parallel_model,
    load_parallel_config,
)
from torchfold.output_paths import safe_job_output_dir
from torchfold.params import import_jax_weights_
from torchfold.padding import (
    single_card_padding_alignment,
    resolve_inference_buckets,
)
from torchfold.torchfold import TorchFold

rank = 0
world_size = 1


def dist_is_initialized() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _host_valid_token_prefix(featurised_example) -> int | None:
    """Return a CPU-proven valid-token prefix without reading an NPU scalar."""
    seq_mask = featurised_example.get("seq_mask")
    if seq_mask is None:
        return None
    seq_mask = np.asarray(seq_mask, dtype=np.bool_)
    if seq_mask.ndim != 1:
        return None
    valid_len = int(np.count_nonzero(seq_mask))
    if not np.all(seq_mask[:valid_len]) or np.any(seq_mask[valid_len:]):
        return None
    return valid_len


def init_dist_from_env() -> torch.device:
    """Initialise HCCL when USE_DIST=1; otherwise keep the single-NPU path."""
    global rank, world_size

    use_dist = os.environ.get("USE_DIST", "0") == "1"
    if (not use_dist) or (dist is None) or (not dist.is_available()):
        rank, world_size = 0, 1
        return torch.device("npu")

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    return torch.device(f"npu:{local_rank}")


_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = pathlib.Path('/path/to/model_dir')
DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)

_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model (JAX params) for inference.',
)

_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    False,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

_USE_FASTNN = flags.DEFINE_bool(
    'fastnn',
    True,
    'Whether to run inference with fastnn.',
)

_DOT_PRODUCT_ATTENTION = flags.DEFINE_string(
    'dot_product_attention',
    'torch',
    'Choose the implementation of dot product attention, options: '
    '["torch", "Fusion_Attention"]',
)

_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

_DB_DIR = flags.DEFINE_string(
    'db_dir',
    DEFAULT_DB_DIR.as_posix(),
    'Path to the directory containing the databases.',
)
_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/pdb_2022_09_28_mmcif_files.tar',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of trunk recycle iterations.',
)
_DIFFUSION_STEPS = flags.DEFINE_integer(
    'diffusion_steps',
    200,
    'Number of denoising steps per diffusion sample.',
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
)
_DIFFUSION_SAMPLE_PARALLEL = flags.DEFINE_bool(
    'diffusion_sample_parallel',
    True,
    'Enable sample-level Diffusion parallelism. The actual single-card or '
    'multi-card path also depends on the fixed sequence-length thresholds.',
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Path to a PyTorch checkpoint (model_state_dict) produced by training.',
)
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2025-01-01',
    'Maximum template release date in YYYY-MM-DD format.',
)


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        model_dir: pathlib.Path,
        device: torch.device,
    ):
        self._model_dir = model_dir
        self._device = device

        self._model = TorchFold(
            num_recycles=_NUM_RECYCLES.value,
            num_samples=_NUM_DIFFUSION_SAMPLES.value,
            diffusion_steps=_DIFFUSION_STEPS.value,
            diffusion_sample_parallel=_DIFFUSION_SAMPLE_PARALLEL.value,
        )
        self._model.eval()
        if _CHECKPOINT_PATH.value:
            ckpt_path = pathlib.Path(_CHECKPOINT_PATH.value)
            print(f'Loading PyTorch checkpoint from {ckpt_path} ...')
            checkpoint = torch.load(
                ckpt_path,
                map_location=device,
                weights_only=True,
            )
            if not isinstance(checkpoint, Mapping):
                raise TypeError(
                    "PyTorch checkpoint must contain a mapping, "
                    f"got {type(checkpoint).__name__}"
                )

            # Training checkpoints wrap weights in ``model_state_dict``.
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if not isinstance(state_dict, Mapping):
                raise TypeError(
                    "model_state_dict must be a mapping, "
                    f"got {type(state_dict).__name__}"
                )
            params = dict(state_dict)
            for k in state_dict:
                if 'q_projection' in k:
                    k_name = k.replace('q_projection', 'k_projection')
                    v_name = k.replace('q_projection', 'v_projection')
                    key = k.replace('q_projection', 'qkv_projection')
                    if k_name in state_dict and v_name in state_dict:
                        params[key] = torch.cat(
                            (params[k], params[k_name], params[v_name]),
                            dim=0,
                        )
                    elif 'bias' in k:
                        bias_0 = torch.zeros_like(params[k])
                        params[key] = torch.cat(
                            (params[k], bias_0, bias_0), dim=0
                        )
            del state_dict
            # Strip the DDP wrapper prefix when present.
            if any(k.startswith('module.') for k in params):
                params = {
                    k.replace('module.', '', 1) if k.startswith('module.') else k: v
                    for k, v in params.items()
                }

            incompatible = self._model.load_state_dict(params, strict=False)
            missing = list(incompatible.missing_keys)
            legacy_projection_names = (
                'q_projection.',
                'k_projection.',
                'v_projection.',
            )
            unexpected = [
                key
                for key in incompatible.unexpected_keys
                if not any(name in key for name in legacy_projection_names)
            ]
            if missing or unexpected:
                raise RuntimeError(
                    "PyTorch checkpoint is incompatible with TorchFold: "
                    f"missing_keys={missing}, unexpected_keys={unexpected}"
                )
            print(f'Successfully loaded checkpoint from {ckpt_path}')
        else:
            print('Loading JAX params from model_dir...')
            import_jax_weights_(self._model, model_dir)

        parallel_cfg = load_parallel_config(verbose=True)
        self._model, parallel_report = inject_parallel_model(self._model, parallel_cfg)
        self._parallel_model_enabled = parallel_report.model_replaced

        self._model = self._model.to(device=self._device)

        if _USE_FASTNN.value is False:
            fastnn_config.layer_norm_implementation = 'torch'
            fastnn_config.dot_product_attention_implementation = 'torch'

        fastnn_config.dot_product_attention_implementation = _DOT_PRODUCT_ATTENTION.value

    @torch.inference_mode()
    def run_inference(
        self, featurised_example: features.BatchDict
    ) -> post_processing.ModelResult:
        """Computes a forward pass of the model on a featurised example."""
        valid_token_count = (
            None
            if self._parallel_model_enabled
            else _host_valid_token_prefix(featurised_example)
        )
        featurised_example = pytree.tree_map(
            torch.from_numpy,
            features.remove_invalidly_typed_feats(featurised_example),
        )
        featurised_example = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(device=self._device),
            featurised_example,
        )
        featurised_example['deletion_mean'] = featurised_example[
            'deletion_mean'
        ].to(dtype=torch.float32)

        with fastnn_config.single_card_fa_mask_context(valid_token_count):
            result = self._model(featurised_example)
        if rank != 0:
            return None
        # Preserve a checkpoint identifier, with a stable mmCIF fallback.
        identifier = getattr(self._model, "__identifier__", None)
        if identifier is None:
            identifier_str = "TorchFold"
            identifier_bytes = identifier_str.encode("ascii")
            identifier = np.frombuffer(identifier_bytes, dtype=np.uint8)
        else:
            identifier = identifier.numpy()
        result['__identifier__'] = identifier

        result = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(
                dtype=torch.float32) if x.dtype == torch.bfloat16 else x,
            result,
        )
        result = pytree.tree_map_only(
            torch.Tensor, lambda x: x.cpu().detach().numpy(), result)
        result['__identifier__'] = result['__identifier__'].tobytes()

        return result

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: post_processing.ModelResult,
        target_name: str,
    ) -> list[post_processing.InferenceResult]:
        """Generates structures from model outputs."""
        return list(
            post_processing.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
    """

    seed: int
    inference_results: Sequence[post_processing.InferenceResult]
    full_fold_input: folding_input.Input


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    print(f'Featurising data for seeds {fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    chain_lengths = tuple(len(chain) for chain in fold_input.chains)
    resolved_buckets = resolve_inference_buckets(
        chain_lengths=chain_lengths,
        world_size=world_size,
        buckets=buckets,
    )
    if rank == 0 and world_size == 1 and buckets is None:
        original_tokens = sum(chain_lengths)
        print(
            '[padding] single-card feature alignment='
            f'{single_card_padding_alignment(original_tokens)} original_tokens={original_tokens} '
            f'padded_tokens={resolved_buckets[0]}'
        )
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=resolved_buckets,
        ccd=ccd,
        verbose=True,
    )
    print(
        f'Featurising data for seeds {fold_input.rng_seeds} took '
        f' {time.time() - featurisation_start_time:.2f} seconds.'
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        print(f'[rank={rank}] Running model inference for seed {seed}...')
        torch.npu.synchronize()
        inference_start_time = time.time()

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = model_runner.run_inference(example)
        torch.npu.synchronize()
        print(
            f'[rank={rank}] Running model inference for seed {seed} took '
            f' {time.time() - inference_start_time:.2f} seconds.'
        )
        if dist_is_initialized():
            dist.barrier()

        if rank != 0:
            print(f'Skipping extracting output structures for rank = {rank}')
            if dist_is_initialized():
                dist.barrier()
            continue

        print(
            f'Extracting output structures (one per sample) for seed {seed}...')
        extract_structures = time.time()
        inference_results = model_runner.extract_structures(
            batch=example, result=result, target_name=fold_input.name
        )
        print(
            f'Extracting output structures (one per sample) for seed {seed} took '
            f' {time.time() - extract_structures:.2f} seconds.'
        )
        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
            )
        )
        print(
            'Running model inference and extracting output structures for seed'
            f' {seed} took  {time.time() - inference_start_time:.2f} seconds.'
        )

        if dist_is_initialized():
            dist.barrier()
    print(
        'Running model inference and extracting output structures for seeds'
        f' {fold_input.rng_seeds} took '
        f' {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(
            output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
    ) as f:
        f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    output_terms_path = (
        pathlib.Path(torchx.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    )
    output_terms = output_terms_path.read_text() if output_terms_path.exists() else None

    os.makedirs(output_dir, exist_ok=True)
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            sample_dir = os.path.join(
                output_dir, f'seed-{seed}_sample-{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            output_handlers.write_output_for_fold(
                inference_result=result, output_dir=sample_dir
            )
            ranking_score = float(result.metadata['ranking_score'])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

    if max_ranking_result is not None:
        output_handlers.write_output_for_fold(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=output_terms,
            name=job_name,
        )
        with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
    ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """
    print(f'[rank={rank}] Processing fold input {fold_input.name}')

    torch.npu.matmul.allow_hf32 = True

    if not fold_input.chains:
        raise ValueError('Fold input has no chains.')

    os.makedirs(output_dir, exist_ok=True)

    if data_pipeline_config is None:
        print(f'[rank={rank}] Skipping data pipeline...')
        if rank == 0:
            print(f'[rank=0] Writing model input JSON to {output_dir}')
            write_fold_input_json(fold_input, output_dir)
        if dist_is_initialized():
            dist.barrier()
    else:
        if dist_is_initialized() and rank != 0:
            print(f'[rank={rank}] Waiting for rank0 to run data pipeline...')
            dist.barrier()
            processed_json = os.path.join(
                output_dir, f'{fold_input.sanitised_name()}_data.json'
            )
            fold_input_list = list(
                folding_input.load_fold_inputs_from_path(pathlib.Path(processed_json))
            )
            if not fold_input_list:
                raise RuntimeError(
                    f'[rank={rank}] Failed to load processed fold input from {processed_json}'
                )
            fold_input = fold_input_list[0]
        else:
            print('[rank=0] Running data pipeline...')
            fold_input = pipeline.DataPipeline(
                data_pipeline_config).process(fold_input)
            print(f'[rank=0] Writing processed model input JSON to {output_dir}')
            write_fold_input_json(fold_input, output_dir)
            if dist_is_initialized():
                dist.barrier()

    if model_runner is None:
        print(f'[rank={rank}] Skipping inference...')
        output = fold_input
    else:
        print(
            f'[rank={rank}] Predicting 3D structure for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
        )
        if rank == 0:
            print(
                f'[rank=0] Writing outputs for {fold_input.name} for seed(s)'
                f' {fold_input.rng_seeds}...'
            )
            write_outputs(
                all_inference_results=all_inference_results,
                output_dir=output_dir,
                job_name=fold_input.sanitised_name(),
            )
        if dist_is_initialized():
            dist.barrier()
        output = all_inference_results

    print(f'[rank={rank}] Done processing fold input {fold_input.name}.')
    return output


def main(_):
    if _NUM_RECYCLES.value < 1:
        raise ValueError(
            '--num_recycles must be >= 1, '
            f'got {_NUM_RECYCLES.value}'
        )
    if _DIFFUSION_STEPS.value < 1:
        raise ValueError(
            '--diffusion_steps must be >= 1, '
            f'got {_DIFFUSION_STEPS.value}'
        )
    if _NUM_DIFFUSION_SAMPLES.value < 1:
        raise ValueError(
            '--num_diffusion_samples must be >= 1, '
            f'got {_NUM_DIFFUSION_SAMPLES.value}'
        )

    device = init_dist_from_env()
    if rank == 0:
        print(f'[dist] USE_DIST={os.environ.get("USE_DIST", "0")} rank={rank} world_size={world_size} device={device}')
    else:
        print(f'[dist] rank={rank} world_size={world_size} device={device}')

    if _JSON_PATH.value is None == _INPUT_DIR.value is None:
        raise ValueError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )

    if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
        raise ValueError(
            'At least one of --run_inference or --run_data_pipeline must be'
            ' set to true.'
        )

    if _INPUT_DIR.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_dir(
            pathlib.Path(_INPUT_DIR.value)
        )
    elif _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    else:
        raise AssertionError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )

    # Fail before model work if the output directory cannot be created.
    if rank == 0:
        try:
            os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
        except OSError as e:
            print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
            raise
    if dist_is_initialized():
        dist.barrier()

    if _RUN_DATA_PIPELINE.value:
        def replace_db_dir(x): return string.Template(x).substitute(
            DB_DIR=_DB_DIR.value
        )
        data_pipeline_config = pipeline.DataPipelineConfig(
            max_template_date=_MAX_TEMPLATE_DATE.value,
            jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
            nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
            hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
            hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
            hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
            small_bfd_database_path=replace_db_dir(
                _SMALL_BFD_DATABASE_PATH.value),
            mgnify_database_path=replace_db_dir(_MGNIFY_DATABASE_PATH.value),
            uniprot_cluster_annot_database_path=replace_db_dir(
                _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
            ),
            uniref90_database_path=replace_db_dir(
                _UNIREF90_DATABASE_PATH.value),
            ntrna_database_path=replace_db_dir(_NTRNA_DATABASE_PATH.value),
            rfam_database_path=replace_db_dir(_RFAM_DATABASE_PATH.value),
            rna_central_database_path=replace_db_dir(
                _RNA_CENTRAL_DATABASE_PATH.value
            ),
            pdb_database_path=replace_db_dir(_PDB_DATABASE_PATH.value),
            seqres_database_path=replace_db_dir(_SEQRES_DATABASE_PATH.value),
            jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
            nhmmer_n_cpu=_NHMMER_N_CPU.value,
        )
    else:
        if rank == 0:
            print('Skipping running the data pipeline.')
        data_pipeline_config = None

    if _RUN_INFERENCE.value:
        if rank == 0:
            print('Building model from scratch...')
        model_runner = ModelRunner(
            model_dir=pathlib.Path(_MODEL_DIR.value),
            device=device,
        )
    else:
        if rank == 0:
            print('Skipping running model inference.')
        model_runner = None

    fold_inputs = list(fold_inputs)
    if rank == 0:
        print(f'Processing {len(fold_inputs)} fold inputs.')
    if dist_is_initialized():
        dist.barrier()

    for fold_input in fold_inputs:
        job_output_dir = safe_job_output_dir(
            _OUTPUT_DIR.value,
            fold_input.sanitised_name(),
        )
        process_fold_input(
            fold_input=fold_input,
            data_pipeline_config=data_pipeline_config,
            model_runner=model_runner,
            output_dir=job_output_dir,
        )

    if rank == 0:
        print(f'Done processing {len(fold_inputs)} fold inputs.')


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'output_dir',
    ])
    app.run(main)
