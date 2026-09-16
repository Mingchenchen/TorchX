import csv
import dataclasses
import json
import multiprocessing
import os
import pathlib
import random
import shutil
import string
import sys
import textwrap
import time
import types
from collections.abc import Sequence, Mapping
from typing import overload, List, Any, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils._pytree as pytree
import torchx.cpp
from absl import app
from absl import flags
from torchx import structure
from torchx.common import folding_input
from torchx.constants import chemical_components
from torchx.data import featurisation
from torchx.data import pipeline
from torchx.processing import feat_batch
from torchx.processing import features
from torchx.processing import output_handlers
from torchx.processing import post_processing

import torchcraft.nn.template
from custom_function.iglm_model import CustomIgLM
from custom_function.nanobody_utils import (
    build_design_positions_from_cif_and_ranges,
    extract_framework_aa_from_cif,
    parse_range_list,
    parse_two_ranges,
    update_framework_template_in_json,
    extract_framework_auth_ids_from_cif,
)
from torchcraft.nn import fastnn_config
from torchcraft.nn import featurization
from torchcraft.params import import_jax_weights_
from torchcraft.torchfold import TorchFold


def _annotate_framework_with_brackets(
    sequence: str,
    framework_indices: Sequence[int] | list[int],
) -> str:
    """Return a human-readable sequence with framework positions wrapped in [].

    Consecutive framework segments are wrapped in a single pair of brackets, for example:
      Sequence:  ACDEFGHIK
      Framework indices: [0, 1, 5, 6]
      Result:  [AC]DEF[GH]IK

    For visualization/CSV output only, not used in calculations.
    """
    if not sequence or not framework_indices:
        return sequence

    fw_set = {int(i) for i in framework_indices}
    chars: list[str] = []
    in_fw = False

    for i, aa in enumerate(sequence):
        is_fw = i in fw_set
        if is_fw and not in_fw:
            chars.append("[")
            in_fw = True
        if not is_fw and in_fw:
            chars.append("]")
            in_fw = False
        chars.append(aa)

    if in_fw:
        chars.append("]")

    return "".join(chars)


USE_DIST=0

_BUCKETS: tuple[int, ...] = (
    64,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1408,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
)

os.environ["RANK"] = str(os.environ.get("RANK", 0))
os.environ["LOCAL_RANK"] = str(os.environ.get("LOCAL_RANK", 0))
os.environ["WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE", 1))

if USE_DIST:
    backend = os.environ.get("USE_BACKEND", "hccl")
    print(f"Using backend: {backend}")
    dist.init_process_group(
        backend,
        init_method="env://",
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank != 0:
        fnull = open(os.devnull, "w")
        os.dup2(fnull.fileno(), sys.stdout.fileno())
        os.dup2(fnull.fileno(), sys.stderr.fileno())
else:
    rank = 0
    world_size = 1

_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = pathlib.Path('/path/to/model_dir')
DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
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

# === Nanobody / partial-design related flags ===
_FRAMEWORK_CIF_PATH = flags.DEFINE_string(
    'framework_cif_path',
    '',
    (
        'Optional path to a framework-only mmCIF for binder nanobody design. '
        'Used together with --cdr_ranges and --side_range to automatically '
        'derive design positions in the binder chain.'
    ),
)

_CDR_RANGES = flags.DEFINE_string(
    'cdr_ranges',
    '',
    (
        'Per-CDR length ranges in the form "min1-max1,min2-max2,...". The '
        'number of ranges must equal the number of gaps inferred from the '
        'framework CIF.'
    ),
)

_SIDE_RANGE = flags.DEFINE_string(
    'side_range',
    '',
    (
        'Two ranges for N- and C-terminal designable side segments, in the '
        'form "left_min-left_max,right_min-right_max". Use "0-0,0-0" to '
        'disable side design.'
    ),
)

_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
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

# Choose the implementation of dot product attention
_DOT_PRODUCT_ATTENTION = flags.DEFINE_string(
    'dot_product_attention',
    'torch',
    'Choose the implementation of dot product attention, options: ["torch", "Fusion_Attention"]',
)

# Binary paths.
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

# Database paths.
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

# Number of CPUs to use for MSA tools.
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

_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
)

_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    5,
    'Number of recycling iterations in Evoformer.',
)

_USE_GRADIENT_CHECKPOINTING = flags.DEFINE_boolean(
    'use_gradient_checkpointing',
    True,
    'Whether to use gradient checkpointing to reduce memory usage.',
)

_TURN_OFF_DIFFUSION_CONFIDENCE = flags.DEFINE_bool(
    'turn_off_diffusion_confidence',
    False,
    (
        'Turn off diffusion module and confidence module, loss only includes '
        'the contact_loss and helix_loss'
    ),
)

_STAGE3_DIFFUSION_STEPS = flags.DEFINE_integer(
    'stage3_diffusion_steps',
    20,
    'Number of diffusion steps to use during Stage 3 optimization (default: 20).',
)

_CYCLIC_OFFSET = flags.DEFINE_boolean(
    'cyclic_offset',
    False,
    'Whether to apply cyclic offset encoding to the binder chain (assumed to be the chain with max asym_id).'
)

_PAIRFORMER_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'pairformer_checkpoint_interval',
    2,
    'Gradient checkpointing interval for pairformer module.',
)

_LEARNING_RATE_DESIGN = flags.DEFINE_float(
    'learning_rate_design',
    0.001,
    'Learning rate for sequence design',
)
_GROUND_TRUTH_SEQUENCE = flags.DEFINE_string(
    'ground_truth_sequence',
    '',
    'Ground truth sequence',
)
_DESIGN_LENGTH = flags.DEFINE_integer(
    'design_length',
    62,
    'Length of the designed sequence',
)
_DESIGN_INITIAL_SEQUENCE = flags.DEFINE_string(
    'design_initial_sequence',
    'random_sequence',
    'Initial sequence strategy, options: random_sequence, allA_sequence, Ground_truth_sequence, gumbel_sequence, from_file',
)

_PTM_WEIGHT = flags.DEFINE_float(
    'ptm_weight',
    0,
    'Weight of the pTM score in the loss function',
)
_IPTM_WEIGHT = flags.DEFINE_float(
    'iptm_weight',
    0,
    'Weight of the ipTM score in the loss function',
)
_PDE_WEIGHT = flags.DEFINE_float(
    'pde_weight',
    0,
    'Weight of the PDE score in the loss function',
)

_PLDDT_WEIGHT = flags.DEFINE_float(
    'plddt_weight',
    0,
    'Weight of the pLDDT score in the loss function',
)
_PAE_WEIGHT = flags.DEFINE_float(
    'pae_weight',
    0,
    'Weight of the PAE loss in the loss function',
)

_BINDER_PAE_WEIGHT = flags.DEFINE_float(
    'binder_pae_weight',
    0,
    'Weight of the intra-binder PAE loss in the loss function',
)
_BINDER_TARGET_INTERFACE_PAE_WEIGHT = flags.DEFINE_float(
    'binder_target_interface_pae_weight',
    0,
    'Weight of the Binder-Target interface PAE loss in the loss function',
)

_BINDER_CONTACT_WEIGHT = flags.DEFINE_float(
    'binder_contact_weight',
    0,
    'Weight of the Binder contact loss in the loss function',
)
_INTERFACE_CONTACT_WEIGHT = flags.DEFINE_float(
    'interface_contact_weight',
    0,
    'Weight of the Interface contact loss in the loss function',
)
_HELIX_WEIGHT = flags.DEFINE_float(
    'helix_weight',
    0,
    'Weight of the Helix loss in the loss function',
)
_STAGE_EPOCHS = flags.DEFINE_string(
    'stage_epochs',
    '0,0,80,5', 
    'Epoch configuration for four stages, format: pre-design,stage1,stage2,stage3',
)

_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed',
    1,
    'Random seed value for ensuring experiment reproducibility',
)

_HOTSPOT_INDICES = flags.DEFINE_string(
    'hotspot_indices',
    '', 
    'Hotspot residue indices separated by commas, e.g. "10,15,20,25,30". Empty string means using all target residues',
)

_ENTROPY_WEIGHT = flags.DEFINE_float(
    'entropy_weight',
    0,
    'Weight of the entropy loss in the loss function',
)

_NEGATIVE_HOTSPOT_INDICES = flags.DEFINE_string(
    'negative_hotspot_indices',
    '', 
    'Hotspot residue indices separated by commas, e.g. "10,15,20,25,30". Empty string means using all target residues',
)

_NEGATIVE_CONTACT_WEIGHT = flags.DEFINE_float(
    'negative_contact_weight',
    0,
    'Weight of the negative_contact loss in the loss function',
)

_NEGATIVE_FRAMEWORK_CONTACT_WEIGHT = flags.DEFINE_float(
 	'negative_framework_contact_weight',
 	0,
 	'Weight of the negative_framework_contact loss in the loss function',
)

_FRAMEWORK_NEGATIVE_BINDER_INDICES = flags.DEFINE_string(
    'framework_negative_binder_indices',
    '',
    'List of auth_seq_id in Framework CIF (e.g. "1-10,25,30-35"), these residues will be penalized for contact with the Target.'
)

_INITIAL_LOGITS_PATH = flags.DEFINE_string(
    'initial_logits_path',
    '',
    'Path to the initial .pt logits file. If provided, overrides --design_initial_sequence.',
)

_FIXED_RESIDUES = flags.DEFINE_string(
    'fixed_residues',
    '',
    'Comma-separated list of 0-indexed residue positions in the binder to keep fixed '
    '(e.g., "10,15,20"). Gradients for these positions will not be updated.',
)

_PARATOPE_LOSS_WEIGHT = flags.DEFINE_float(
    'paratope_weight',
    0, # Defaults to 0 unless explicitly enabled
    'Weight of the Paratope Loss in the loss function (Germinal Paper implementation)',
)

_AA_TYPE_LOSS_WEIGHT = flags.DEFINE_float(
    'aa_type_loss_weight',
    0.0,
    'Weight for the amino acid type cross-entropy loss. Default is 0.0 (disabled).',
)

_AA_TYPE_LOSS_DIST_NAME = flags.DEFINE_string(
    'aa_type_loss_dist_name',
    'ppiflow_v1',
    'The name of the reference amino acid distribution to use for the AA Type Loss. '
    'Options: "sabdab", "ppiflow_v1".'
)

# === IgLM Related Parameters ===
_USE_IGLM = flags.DEFINE_bool(
    'use_iglm',
    False,
    'Whether to use the IgLM language model to guide antibody sequence design',
)
_IGLM_WEIGHT = flags.DEFINE_float(
    'iglm_weight',
    0.0,
    'Weight scaling factor for IgLM gradients',
)
_IGLM_TEMP = flags.DEFINE_float(
    'iglm_temp',
    0.6,
    'IgLM softmax temperature parameter',
)
_IGLM_SPECIES = flags.DEFINE_string(
    'iglm_species',
    '[HUMAN]',
    'IgLM species token, options: [HUMAN], [MOUSE], [CAMEL], etc.',
)
_IGLM_CHAIN = flags.DEFINE_string(
    'iglm_chain', 
    '[HEAVY]', 
    'IgLM chain type token, options: [HEAVY], [LIGHT], etc. Use [HEAVY] for VHH/nanobody',
)
_IGLM_CDR1_WEIGHT = flags.DEFINE_float(
    'iglm_cdr1_weight',
    0.0,
    'Weight of IgLM CDR1 region log-likelihood in the loss function',
)
_IGLM_CDR2_WEIGHT = flags.DEFINE_float(
    'iglm_cdr2_weight',
    0.0,
    'Weight of IgLM CDR2 region log-likelihood in the loss function',
)
_IGLM_CDR3_WEIGHT = flags.DEFINE_float(
    'iglm_cdr3_weight',
    0.0,
    'Weight of IgLM CDR3 region log-likelihood in the loss function',
)

# Mutually exclusive with model_dir (validated by run.sh). If provided, load a trained
# checkpoint; otherwise import JAX weights from model_dir.
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Path to a PyTorch checkpoint (model_state_dict) produced by training.',
)

ModelResult: TypeAlias = Mapping[str, Any]
_ScalarNumberOrArray: TypeAlias = Mapping[str, float | int | np.ndarray]
@dataclasses.dataclass(frozen=True)
class InferenceResult:
    """Postprocessed model result.

    Attributes:
        predicted_structure: Predicted protein structure.
        numerical_data: Useful numerical data (scalars or arrays) to be saved at
        inference time.
        metadata: Smaller numerical data (usually scalar) to be saved as inference
        metadata.
        debug_outputs: Additional dict for debugging, e.g. raw outputs of a model
        forward pass.
        model_id: Model identifier.
    """

    predicted_structure: structure.Structure = dataclasses.field()
    numerical_data: _ScalarNumberOrArray = dataclasses.field(default_factory=dict)
    metadata: _ScalarNumberOrArray = dataclasses.field(default_factory=dict)
    debug_outputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    model_id: bytes = b''

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

    # Put them into a structure
    pred_struc = batch.convert_model_output.empty_output_struc
    pred_struc = pred_struc.copy_and_update_globals(release_date=None)
    return pred_struc

class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        model_dir: pathlib.Path,
        device: torch.device,
        hotspot_indices: List[int] = None,
        negative_hotspot_indices: List[int] = None,
        design_positions_in_chain: List[int] = None,
        framework_negative_binder_indices: List[int] = None,
        stage3_diffusion_steps: int = 20, 
    ):
        self._model_dir = model_dir
        self._device = device
        self.hotspot_indices = hotspot_indices
        self.negative_hotspot_indices = negative_hotspot_indices
        self.design_positions_in_chain = design_positions_in_chain
        self.framework_negative_binder_indices = framework_negative_binder_indices
        self._model = TorchFold(
            num_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            use_gradient_checkpointing=_USE_GRADIENT_CHECKPOINTING.value,
            # Remove the template_checkpoint_interval parameter
            pairformer_checkpoint_interval=_PAIRFORMER_CHECKPOINT_INTERVAL.value,
            turn_off_diffusion_confidence=_TURN_OFF_DIFFUSION_CONFIDENCE.value,
            diffusion_steps=200,           # Keep 200 steps for stage 4
            stage3_diffusion_steps=stage3_diffusion_steps,
        )
        self._model.eval()
        print('loading the model parameters...')
        # If a PyTorch checkpoint is provided, load it and skip importing JAX weights.
        # Otherwise, fall back to importing JAX weights from model_dir (default behavior).
        if _CHECKPOINT_PATH.value:
            ckpt_path = pathlib.Path(_CHECKPOINT_PATH.value)
            print(f'Loading PyTorch checkpoint from {ckpt_path} ...')
            
            # Load checkpoint (same as training code: weights_only=False)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Extract model state dict (training checkpoints use 'model_state_dict')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            params = state_dict.copy()
            for k in state_dict.keys():
                if 'q_projection' in k:
                    k_name = k.replace('q_projection', 'k_projection')
                    v_name = k.replace('q_projection', 'v_projection')
                    key = k.replace('q_projection', 'qkv_projection')
                    if k_name in state_dict.keys() and v_name in state_dict.keys():
                        params[key] = torch.cat((params[k], params[k_name], params[v_name]), dim=0)
                    elif 'bias' in k:
                        bias_0 = torch.zeros_like(params[k])
                        params[key] = torch.cat((params[k], bias_0, bias_0), dim=0)
            del state_dict
            # Handle DDP-trained checkpoints: remove 'module.' prefix if present
            # Training code saves as: model.module.state_dict() when use_ddp=True
            if any(k.startswith('module.') for k in params.keys()):
                params = {k.replace('module.', '', 1) if k.startswith('module.') else k: v 
                             for k, v in params.items()}
            
            # Load state dict (same approach as training code, but with strict=False for flexibility)
            missing, _ = self._model.load_state_dict(params, strict=False)
            
            if missing:
                print(f'Warning: {len(missing)} missing keys:')
                for k in missing:
                    print(f'  {k}')
            print(f'Successfully loaded checkpoint from {ckpt_path}')
        else:
            print('Loading JAX params from model_dir...')
            import_jax_weights_(self._model, model_dir)

        # Freeze all model parameters to ensure only binder_logits_20 is updated
        for param in self._model.parameters():
            param.requires_grad = False

        # Save the original _sample_diffusion method
        original_sample_diffusion = self._model._sample_diffusion

        # Patch the _sample_diffusion method with torch.no_grad() to reduce memory usage
        def no_grad_sample_diffusion(self, batch, embeddings):
            with torch.no_grad():
                return original_sample_diffusion(batch, embeddings)

        
        # Apply the patch
        self._model._sample_diffusion = types.MethodType(no_grad_sample_diffusion, self._model)

        self._model = self._model.to(device=self._device)
        
        fastnn_config.dot_product_attention_implementations = _DOT_PRODUCT_ATTENTION.value

    # @torch.inference_mode()  # Remove this decorator to enable gradient computation
    def run_inference(
        self, featurised_example: features.BatchDict
    ) -> post_processing.ModelResult:
        """Computes a forward pass of the model on a featurised example."""
        featurised_example = pytree.tree_map(
            torch.from_numpy, features.remove_invalidly_typed_feats(
                featurised_example)
        )
        featurised_example = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(device=self._device),
            featurised_example,
        )
        featurised_example['deletion_mean'] = featurised_example['deletion_mean'].to(
            dtype=torch.float32)

        result = self._model(featurised_example, self.binder_logits_20)
        if rank != 0:
            return None

        # Prefer the identifier loaded from the original checkpoint if present;
        # otherwise fall back to a fixed "TorchFold" ASCII identifier so that
        # mmCIF shows "TorchFold".
        identifier = getattr(self._model, "__identifier__", None)
        if identifier is None:
            identifier_str = "TorchFold"
            identifier_bytes = identifier_str.encode("ascii")
            identifier = np.frombuffer(identifier_bytes, dtype=np.uint8)
        else:
            identifier = identifier.numpy()
        result['__identifier__'] = identifier

        # Call the function to calculate confidence metrics here (PyTorch version)
        from custom_function.confidence_torch_version import compute_confidence_metrics, compute_confidence_metrics_simple
        if _TURN_OFF_DIFFUSION_CONFIDENCE.value:
            confidence_metrics_torch = compute_confidence_metrics_simple(
                result, featurised_example, self._device, self.hotspot_indices, self.negative_hotspot_indices, self.design_positions_in_chain, self.framework_negative_binder_indices
            )
        else:
            confidence_metrics_torch = compute_confidence_metrics(
                result, featurised_example, self._device, self.hotspot_indices, self.negative_hotspot_indices, self.design_positions_in_chain, self.framework_negative_binder_indices,
            )

        # Convert results to CPU and detach computation graph, but keep gradient connection for confidence_metrics_torch
        result = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.to(dtype=torch.float32).cpu().detach().numpy() if x.dtype == torch.bfloat16 else x.cpu().detach().numpy(),
            result,
        )
        result['__identifier__'] = result['__identifier__'].tobytes()

        return result, confidence_metrics_torch

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: post_processing.ModelResult,
        target_name: str,
    ) -> list[post_processing.InferenceResult]:
        """Generates structures from model outputs."""
        if _TURN_OFF_DIFFUSION_CONFIDENCE.value:
            batch = feat_batch.Batch.from_data_dict(batch)

            # Retrieve structure and construct a predicted structure.
            pred_structure = get_predicted_structure(result=result, batch=batch)

            num_tokens = batch.token_features.seq_length.item()
            asym_ids = batch.token_features.asym_id[:num_tokens]
            chain_ids = [pred_structure.chains[asym_id - 1] for asym_id in asym_ids]
            res_ids = batch.token_features.residue_index[:num_tokens]
            contact_probs = result['distogram']['contact_probs']
            import concurrent
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                return InferenceResult(
                    predicted_structure=pred_structure,
                    numerical_data={
                        'contact_probs': contact_probs[:num_tokens, :num_tokens],
                    },
                    metadata={
                        'token_chain_ids': chain_ids,
                        'token_res_ids': res_ids,
                    },
                    model_id=result['__identifier__'],
                    debug_outputs={},
                )
        else:
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
    epoch:int,
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    print(f'Featurising data for seeds {fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True
    )
    print(
        f'Featurising data for seeds {fold_input.rng_seeds} took '
        f' {time.time() - featurisation_start_time:.2f} seconds.'
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    all_confidence_metrics = []  # Create a list to store all confidence_metrics_torch
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        print(f'Running model inference for seed {seed}...')
        if torch.npu.is_available():
            torch.npu.synchronize()
        inference_start_time = time.time()

        # set the random seed for the model.
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        result, confidence_metrics_torch = model_runner.run_inference(example)
        all_confidence_metrics.append(confidence_metrics_torch)  # Stores the confidence_metrics_torch of the current seed

        if torch.npu.is_available():
            torch.npu.synchronize()
        print(
            f'Running model inference for seed {seed} took '
            f' {time.time() - inference_start_time:.2f} seconds.'
        )

        if rank != 0:
            print(f"Skipping extracting output structures for rank = {rank}")
            return None

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
    
    combined_confidence_metrics = {}
    for metrics in all_confidence_metrics:
        for key, value in metrics.items():
            if key not in combined_confidence_metrics:
                combined_confidence_metrics[key] = []
            combined_confidence_metrics[key].append(value)
    
    print("Combined confidence metrics across all seeds:")
    for key in list(combined_confidence_metrics.keys()):
        values = combined_confidence_metrics[key]
        # Skipping non-tensor types
        if not isinstance(values[0], torch.Tensor):
            # For dict type, only keep the value of the first seed
            combined_confidence_metrics[key] = values[0]
            print(f"  {key}: (non-tensor, kept first seed)")
            continue
        # Stack metrics from all seeds
        stacked_metrics = torch.stack(values)
        # Calculate the overall mean (single scalar)
        overall_mean = stacked_metrics.mean(dim=None)
        combined_confidence_metrics[key] = overall_mean
        print(f"  {key}: {overall_mean.item():.4f}")
    
    print(
        'Running model inference and extracting output structures for seeds'
        f' {fold_input.rng_seeds} took '
        f' {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results, combined_confidence_metrics


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
    output_dir: os.PathLike[str] | str | None,
    job_name: str,
    last_stage_epoch_inf: str | None = None,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    # output_terms = (
    #     pathlib.Path(torchfold.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    # ).read_text()
    
    output_terms_path = (
        pathlib.Path(torchx.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    )
    output_terms = output_terms_path.read_text() if output_terms_path.exists() else None

    # only save last stage epochs info, iff output_dir is not None
    if output_dir is not None:     
        os.makedirs(output_dir, exist_ok=True)
        for results_for_seed in all_inference_results:
            seed = results_for_seed.seed
            for sample_idx, result in enumerate(results_for_seed.inference_results):
                ranking_score = float(result.metadata['ranking_score'])
                ranking_scores.append((seed, sample_idx, ranking_score))
                if max_ranking_score is None or ranking_score > max_ranking_score:
                    max_ranking_score = ranking_score
                    max_ranking_result = result

        if max_ranking_result is not None:  # True iff ranking_scores non-empty.
            output_handlers.write_output_for_design(
                inference_result=max_ranking_result,
                output_dir=output_dir,
                # The output terms of use are the same for all seeds/samples.
                terms_of_use=output_terms,
                name=job_name,
                last_stage_epoch_inf=last_stage_epoch_inf,
            )
            # Save csv of ranking scores with seeds and sample indices, to allow easier
            # comparison of ranking scores across different runs.
            # with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(['seed', 'sample', 'ranking_score'])
            #     writer.writerows(ranking_scores)


@overload
def process_fold_input(
    epoch:int,
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str | None = None,
    last_stage_epoch_info: str | None = None,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
    ...


@overload
def process_fold_input(
    epoch:int,
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str | None = None,
    last_stage_epoch_info: str | None = None,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
    ...


def process_fold_input(
    epoch:int,
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str | None = None,
    last_stage_epoch_info: str | None = None,
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
    if model_runner is None:
        print('Skipping inference...')
        output = fold_input
    else:
        print(
            f'Predicting 3D structure for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        all_inference_results, combined_confidence_metrics = predict_structure(
            epoch=epoch,
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
        )

        if rank != 0:
            print(f"Skipping writing outputs for rank = {rank}")
            return None

        print(
            f'Writing outputs for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        
        if not _TURN_OFF_DIFFUSION_CONFIDENCE.value:
            write_outputs(
                all_inference_results=all_inference_results,
                output_dir=output_dir,
                job_name=fold_input.sanitised_name(),
                last_stage_epoch_inf=last_stage_epoch_info,
            )
        # Ensure the returned value is confidence metrics for loss calculation
        output = combined_confidence_metrics

    print(f'Done processing fold input {fold_input.name}.')
    return output


def main(_):
    # Ensure the os module is imported at the beginning of the main function
    import os
    
    # Record start time
    start_time = time.time()
    print(f"Program start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Enable PyTorch anomaly detection to help identify gradient calculation issues
    torch.autograd.set_detect_anomaly(False)
    
    if _JSON_PATH.value is None:
        raise ValueError(
            '--json_path must be specified.'
        )

    if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
        raise ValueError(
            'At least one of --run_inference or --run_data_pipeline must be'
            ' set to true.'
        )

    # Make sure we can create the output directory before running anything.
    try:
        os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
        raise

    notice = textwrap.wrap(
        'Running TorchCraft Process. '
        ' TorchFold model parameters are available.',
        break_long_words=False,
        break_on_hyphens=False,
        width=80,
    )
    print('\n'.join(notice))

    if _RUN_DATA_PIPELINE.value:
        def replace_db_dir(x): return string.Template(x).substitute(
            DB_DIR=_DB_DIR.value
        )
        data_pipeline_config = pipeline.DataPipelineConfig(
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
        print('Skipping running the data pipeline.')
        data_pipeline_config = None

    # Get sequence design parameters from the command line
    # Parse hotspot parameters
    hotspot_indices_str = _HOTSPOT_INDICES.value
    negative_hotspot_str = _NEGATIVE_HOTSPOT_INDICES.value
    if hotspot_indices_str.strip():
        try:
            hotspot_indices = [int(x.strip()) for x in hotspot_indices_str.split(',')]
            print(f"Using hotspot residues: {hotspot_indices}")
        except ValueError:
            raise ValueError(f"Invalid format for hotspot_indices: {hotspot_indices_str}")
    else:
        hotspot_indices = None
        print("No hotspot residues specified, will use all target residues")

    if negative_hotspot_str and negative_hotspot_str.strip():
        try:
            negative_hotspot_indices = set()

            for part in negative_hotspot_str.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-', 1)
                    start, end = int(start), int(end)
                    if start > end:
                        raise ValueError(f"Range start is greater than end: {part}")
                    negative_hotspot_indices.update(range(start, end + 1))
                else:
                    negative_hotspot_indices.add(int(part))

            negative_hotspot_indices = sorted(negative_hotspot_indices)
            print(f"Using negative_hotspot residues: {negative_hotspot_indices}")

        except ValueError as e:
            raise ValueError(
                f"Invalid format for negative_hotspot_str, should be like '1-22,25,26,32-38', current value:"
                f"{negative_hotspot_str}"
            ) from e
    else:
        negative_hotspot_indices = None
        print("No negative_hotspot specified, will use all target residues")
    # num_epochs = _NUM_EPOCHS.value # Remove this parameter definition

    # Parse the initial_logits_path parameter
    initial_logits_path = _INITIAL_LOGITS_PATH.value or None
    if initial_logits_path:
        print(f"Loading initial logits from file: {initial_logits_path}")

    # Parse fixed_residues parameter
    fixed_residues_str = _FIXED_RESIDUES.value
    fixed_residue_indices = None
    if fixed_residues_str and fixed_residues_str.strip():
        try:
            # Convert a string like "10,15,20" to a list of integers
            fixed_residue_indices = [int(x.strip()) for x in fixed_residues_str.split(',')]
            print(f"Fixing logits for the following binder residues (0-indexed): {fixed_residue_indices}")
        except ValueError:
            raise ValueError(f"Invalid format for fixed_residues: {fixed_residues_str}")
    else:
        print("No fixed_residues specified, all binder logits will be optimized.")
    learning_rate_design = _LEARNING_RATE_DESIGN.value
    Ground_truth_sequence = _GROUND_TRUTH_SEQUENCE.value
    design_length = _DESIGN_LENGTH.value
    design_Initial_sequence = _DESIGN_INITIAL_SEQUENCE.value
    random_seed = _RANDOM_SEED.value  
    featurization.CYCLIC_OFFSET_ENABLED = _CYCLIC_OFFSET.value

    ptm_weight = _PTM_WEIGHT.value
    iptm_weight = _IPTM_WEIGHT.value
    pde_weight = _PDE_WEIGHT.value
    plddt_weight = _PLDDT_WEIGHT.value
    pae_weight = _PAE_WEIGHT.value
    binder_contact_weight = _BINDER_CONTACT_WEIGHT.value
    interface_contact_weight = _INTERFACE_CONTACT_WEIGHT.value
    helix_weight = _HELIX_WEIGHT.value
    binder_pae_weight = _BINDER_PAE_WEIGHT.value
    binder_target_interface_pae_weight = _BINDER_TARGET_INTERFACE_PAE_WEIGHT.value
    entropy_weight = _ENTROPY_WEIGHT.value
    negative_contact_weight = _NEGATIVE_CONTACT_WEIGHT.value
    negative_framework_contact_weight = _NEGATIVE_FRAMEWORK_CONTACT_WEIGHT.value
    paratope_loss_weight = _PARATOPE_LOSS_WEIGHT.value
    aa_type_loss_weight = _AA_TYPE_LOSS_WEIGHT.value
    aa_type_loss_dist_name = _AA_TYPE_LOSS_DIST_NAME.value

    # === IgLM parameters ===
    use_iglm = _USE_IGLM.value
    iglm_weight = _IGLM_WEIGHT.value
    iglm_temp = _IGLM_TEMP.value
    iglm_species = _IGLM_SPECIES.value
    iglm_chain = _IGLM_CHAIN.value
    iglm_cdr1_weight = _IGLM_CDR1_WEIGHT.value
    iglm_cdr2_weight = _IGLM_CDR2_WEIGHT.value
    iglm_cdr3_weight = _IGLM_CDR3_WEIGHT.value

    # Parse stage configuration and calculate total epochs directly
    stage_epochs_str = _STAGE_EPOCHS.value
    stage_epochs = [int(x) for x in stage_epochs_str.split(',')]
    num_epochs = sum(stage_epochs)  # Calculate total directly from stage_epochs
    last_stage_epochs = stage_epochs[-1] # It is always a positive integer, only save the last stage info
    num_epochs_except_last = num_epochs - last_stage_epochs # total epochs except the last stage
    fold_input = None
    
    print(f"Four-stage configuration:")
    print(f"  Pre-design stage: {stage_epochs[0]} epochs")
    print(f"  Stage 1: {stage_epochs[1]} epochs") 
    print(f"  Stage 2: {stage_epochs[2]} epochs")
    print(f"  Stage 3: {stage_epochs[3]} epochs")
    print(f"  Total epochs: {num_epochs}")
    
    print(f"Sequence design parameters:")
    print(f"  Total iterations: {num_epochs}")  # Show calculated total
    print(f"  Learning rate: {learning_rate_design}")
    print(f"  Designed sequence length: {design_length}")
    print(f"  Initial sequence strategy: {design_Initial_sequence}")
    print(f"  Random seed: {random_seed}")  
    print(f"Loss function weights:")
    print(f"  pTM weight: {ptm_weight}")
    print(f"  ipTM weight: {iptm_weight}")
    print(f"  PDE weight: {pde_weight}")
    print(f"  pLDDT weight: {plddt_weight}")
    print(f"  PAE weight: {pae_weight}")
    print(f"  Binder_PAE weight: {binder_pae_weight}")
    print(f"  interface_PAE weight: {binder_target_interface_pae_weight}")
    print(f"  Binder Contact weight: {binder_contact_weight}")
    print(f"  Interface Contact weight: {interface_contact_weight}")
    print(f"  Helix weight: {helix_weight}")
    print(f"  Entropy weight: {entropy_weight}")
    print(f"  Negative Contact weight: {negative_contact_weight}")
    print(f"  Negative Framework Contact weight: {negative_framework_contact_weight}")
    print(f"  Paratope Loss weight: {paratope_loss_weight}")
    print(f"  AA Type Loss weight: {aa_type_loss_weight}")
    if aa_type_loss_weight > 0:
        print(f"  AA Type Loss distribution: {aa_type_loss_dist_name}")
    if _CYCLIC_OFFSET.value:
        print(f"Cyclic offset enabled: {featurization.CYCLIC_OFFSET_ENABLED}")
    # Print IgLM configuration
    if use_iglm:
        print(f"IgLM Configuration (Advanced Infilling Mode):")
        print(f"  IgLM Enabled: {use_iglm}")
        print(f"  IgLM Global Weight (Autoregressive): {iglm_weight}")
        print(f"  IgLM CDR1 Weight (Infilling): {iglm_cdr1_weight}")
        print(f"  IgLM CDR2 Weight (Infilling): {iglm_cdr2_weight}")
        print(f"  IgLM CDR3 Weight (Infilling): {iglm_cdr3_weight}")
        print(f"  IgLM Temperature: {iglm_temp}")
        print(f"  IgLM Species: {iglm_species}")
        print(f"  IgLM Chain Type: {iglm_chain}")

    # Only two modes are supported:
    # 1) Provide framework_cif_path + cdr_ranges (side_range optional) -> Nanobody automatic mode
    # 2) Otherwise, default to full binder chain design (legacy mode)
    fw_cif = _FRAMEWORK_CIF_PATH.value or ''
    cdr_ranges_str = _CDR_RANGES.value or ''
    side_range_str = _SIDE_RANGE.value or ''

    design_positions_in_chain = None
    framework_negative_binder_indices = None # Framework negative sample contact sites

    if fw_cif and cdr_ranges_str:
        # Automatic nanobody mode: Generate design positions based on framework CIF + cdr_ranges + side_range
        cdr_ranges = parse_range_list(cdr_ranges_str)
        if side_range_str.strip():
            side_ranges = parse_two_ranges(side_range_str)
        else:
            # Default: no design on both sides
            side_ranges = ((0, 0), (0, 0))

        design_positions_in_chain, meta = build_design_positions_from_cif_and_ranges(
            cif_path=fw_cif,
            binder_length=None,
            cdr_ranges=cdr_ranges,
            side_ranges=side_ranges,
            seed=random_seed,
        )
        print("Enabled nanobody automatic design mode:")
        print(f"  framework CIF: {fw_cif}")
        print(f"  Number of framework segments: {meta['num_framework_segments']} (gap={meta['num_gaps']})")
        print(f"  framework segment lengths: {meta['framework_segment_lengths']}")
        print(f"  Sampled CDR lengths: {meta['cdr_lengths']}")
        print(f"  Left and right lengths: left={meta['left_len']}, right={meta['right_len']}")
        print(f"  binder total length: {meta['total_length']}")
        print(f"  Number of design positions: {len(design_positions_in_chain)}")

        # Parse framework_negative_indices parameter
        fw_neg_str = _FRAMEWORK_NEGATIVE_BINDER_INDICES.value
        if fw_neg_str.strip():
            # 1. Parse user-input CIF auth_seq_id (e.g. "1-10,25")
            try:
                user_neg_ids = set()
                # Reuse parse_range_list or parse manually
                for part in fw_neg_str.split(','):
                    part = part.strip()
                    if '-' in part:
                        s, e = part.split('-', 1)
                        user_neg_ids.update(range(int(s), int(e) + 1))
                    else:
                        user_neg_ids.add(int(part))
                print(f"User-specified Framework exclusion IDs (CIF auth_id): {sorted(user_neg_ids)}")
            except ValueError as e:
                raise ValueError(f"framework_negative_indices invalid format: {e}")

            # 2. Extract the real auth_seq_id list from CIF (corresponding to templateIndices)
            cif_auth_ids = extract_framework_auth_ids_from_cif(fw_cif)

            # 3. Perform mapping: Template Index -> Binder Sequence Index
            framework_query_indices = meta["framework_query_indices"]
            framework_template_indices = meta["framework_template_indices"]

            framework_negative_binder_indices = []

            # Iterate over each Framework residue position in the Binder
            for q_idx, t_idx in zip(framework_query_indices, framework_template_indices):
                if t_idx < 0 or t_idx >= len(cif_auth_ids):
                    raise ValueError(
                        f"framework_template_index {t_idx} is out of the CIF auth_id range"
                    )
                # t_idx is the index of the CIF sequence (0, 1, 2...)
                # cif_auth_ids[t_idx] gets the real residue number in the CIF (e.g. 10, 11...)
                real_auth_id = cif_auth_ids[t_idx]
                
                if real_auth_id in user_neg_ids:
                    framework_negative_binder_indices.append(q_idx)
            
            framework_negative_binder_indices = sorted(list(set(framework_negative_binder_indices)))
            print(f"Mapped Framework exclusion sites (Binder Index): {framework_negative_binder_indices}")
            print(f"  (Total {len(framework_negative_binder_indices)} residues selected for exclusion)")
        else:
            print("No framework_negative_indices specified, Negative Framework Loss will apply to all non-CDR regions by default.")

        # Update the templates segment of the binder chain in JSON using the framework info from meta
        try:
            framework_query_indices = meta["framework_query_indices"]
            framework_template_indices = meta["framework_template_indices"]
            print(
                "  Writing framework CIF to JSON templates (last chain):\n"
                f"    framework_query_indices (len={len(framework_query_indices)})\n"
                f"    framework_template_indices (len={len(framework_template_indices)})"
            )
            update_framework_template_in_json(
                json_path=_JSON_PATH.value,
                framework_cif_path=fw_cif,
                framework_query_indices=framework_query_indices,
                framework_template_indices=framework_template_indices,
                target_index=-1,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate template indices from CIF/ranges and write to JSON: {e}"
            ) from e
    else:
        print(
            "No (framework_cif_path + cdr_ranges) specified,"
            "will design the entire binder chain"
        )

    # Create main output directory
    main_output_dir = _OUTPUT_DIR.value
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Create CSV file to record training process
    csv_file_path = os.path.join(main_output_dir, 'design_results.csv')


    if _TURN_OFF_DIFFUSION_CONFIDENCE.value:
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                'Epoch',
                'Sequence',  # For nanobody mode, this is the framework annotated sequence with brackets[]
                'Binder_Contact_Loss',
                'Interface_Contact_Loss',
                'Helix_Loss',
                'Negative_Contact_Loss',
                'Negative_Framework_Contact_Loss',
                'Paratope_Loss',
                'Paratope_CDR_Loss',
                'Paratope_FW_Loss',
                'Paratope_CDR_Target_Loss',
                'AA_Type_Loss',
                'IgLM_LL',
                'IgLM_LL_CDR1',
                'IgLM_LL_CDR2',
                'IgLM_LL_CDR3',
                'Loss',
                'Stage',
                'LR_Scale',
                'Effective_LR',
            ])
    else:
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                'Epoch',
                'Sequence',
                'pTM',
                'ipTM',
                'PredictedDistanceError',
                'pLDDT',
                'PAE_Loss',
                'Binder_PAE_Loss',
                'Binder_Target_Interface_PAE_Loss',
                'Binder_Contact_Loss',
                'Interface_Contact_Loss',
                'Helix_Loss',
                'Entropy_Loss',
                'Negative_Contact_Loss',
                'Negative_Framework_Contact_Loss',
                'Paratope_Loss',
                'Paratope_CDR_Loss',
                'Paratope_FW_Loss',
                'Paratope_CDR_Target_Loss',
                'AA_Type_Loss',
                'IgLM_LL',
                'IgLM_LL_CDR1',
                'IgLM_LL_CDR2',
                'IgLM_LL_CDR3',
                'Loss',
                'Stage',
                'LR_Scale',
                'Effective_LR',
            ])


    # === Initialize Model ===
    if _RUN_INFERENCE.value:
        device = torch.device("npu")
        print(f'Found local device: {device}')
        print('Initializing model...')  
        model_runner = ModelRunner(
            model_dir=pathlib.Path(_MODEL_DIR.value),
            device=device,
            hotspot_indices=hotspot_indices,
            negative_hotspot_indices=negative_hotspot_indices,
            design_positions_in_chain=design_positions_in_chain,
            framework_negative_binder_indices=framework_negative_binder_indices,
            stage3_diffusion_steps=_STAGE3_DIFFUSION_STEPS.value,
        )
    else:
        print('Skipping running model inference.')
        model_runner = None

    # === Initialize IgLM Model ===
    iglm_model = None
    if use_iglm:
        print("Initializing IgLM model...")
        try:
            iglm_model = CustomIgLM(
                model_name="IgLM",
                chain_token=iglm_chain,
                iglm_species=iglm_species,
                ablm_temp=iglm_temp,
                device=device,
                seed=random_seed,
            )
            print("IgLM model initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize IgLM model: {e}")
            print("Continuing without IgLM")
            use_iglm = False
            iglm_model = None

    # Training epoch
    epoch = 0
    # Maximum retry count
    max_retry_count = 15
    # Current retry count
    retry_count = 0

    # Main training loop
    while epoch < num_epochs:
        print(f"\n{'='*50}")
        print(f"Starting iteration {epoch} ...")
        print(f"{'='*50}")
        
        # === State Initialization/Loading ===
        if epoch == 0:
            # First epoch: initialize state
            print("Initializing state for the first epoch...")
            
            # Generate sequences using the fill_sequences script
            input_JSON_PATH = pathlib.Path(_JSON_PATH.value)

            if design_positions_in_chain is None:
                # Legacy mode: full binder chain design, keep the original initialization logic
                from custom_function.fill_sequences import process_json_file

                # Check if loading from file
                current_strategy = design_Initial_sequence
                if initial_logits_path:
                    print(f"Detected initial_logits_path, forcing 'from_file' strategy.")
                    current_strategy = 'from_file'

                print(f"Using strategy '{current_strategy}' and seed {random_seed} generating initial sequence and filling the json file...")
                sequence, binder_logits_20 = process_json_file(
                    input_JSON_PATH,
                    strategy=current_strategy,
                    design_length=design_length,
                    ground_truth_sequence=Ground_truth_sequence,
                    target_index=-1,
                    device=device,
                    seed=random_seed,
                    file_path=initial_logits_path,
                )
                print(f"Generated initial sequence: {sequence}")
                binder_logits_20.requires_grad_(True)
            else:
                # Nanobody design mode: Generate logits for design positions and fill the complete binder initial sequence in JSON
                from custom_function.fill_sequences import generate_logits
                from custom_function.update_sequences import logits_to_sequence

                print(
                    f"Nanobody mode: Initializing binder_logits_20 at {len(design_positions_in_chain)} positions"
                    f"using random seed ({random_seed})"
                )

                # Check if loading from a file
                current_strategy = design_Initial_sequence
                if initial_logits_path:
                    print(f"Detected initial_logits_path, forcing the 'from_file' strategy.")
                    current_strategy = 'from_file'

                logits = generate_logits(
                    strategy=current_strategy,
                    design_length=len(design_positions_in_chain),
                    ground_truth_sequence=Ground_truth_sequence,
                    device=device,
                    seed=random_seed,
                    file_path=initial_logits_path,
                )
                binder_logits_20 = logits.to(device)
                binder_logits_20.requires_grad_(True)

                # 1) Extract framework AA sequence from CIF (1-letter, template order)
                framework_aas = extract_framework_aa_from_cif(fw_cif)
                framework_query_indices = meta["framework_query_indices"]
                framework_template_indices = meta["framework_template_indices"]
                if len(framework_aas) != len(framework_template_indices):
                    raise ValueError(
                        "Length of framework_aas does not match framework_template_indices:"
                        f"{len(framework_aas)} != {len(framework_template_indices)}"
                    )

                total_len = meta["total_length"]
                full_seq_list = ["X"] * total_len

                # 2) First fill framework positions (keep consistent with CIF sequence)
                # framework_template_indices are consecutive indices 0..N_fw-1
                for q_idx, t_idx in zip(framework_query_indices, framework_template_indices):
                    if q_idx < 0 or q_idx >= total_len:
                        raise ValueError(
                            f"framework_query_index {q_idx} exceeds total length {total_len}"
                        )
                    full_seq_list[q_idx] = framework_aas[t_idx]

                # 3) Then fill amino acids generated from current logits at design positions
                designed_subseq = logits_to_sequence(binder_logits_20)
                if len(designed_subseq) != len(design_positions_in_chain):
                    raise ValueError(
                        "design_positions_in_chain length does not match logits generated sequence:"
                        f"{len(design_positions_in_chain)} != {len(designed_subseq)}"
                    )
                for idx, pos in enumerate(design_positions_in_chain):
                    if pos < 0 or pos >= total_len:
                        raise ValueError(
                            f"Design position {pos} exceeds total length {total_len}"
                        )
                    full_seq_list[pos] = designed_subseq[idx]

                full_sequence = "".join(full_seq_list)
                print(f"Nanobody initial full binder sequence length: {len(full_sequence)}")


                # 4) Save framework information for IgLM usage
                framework_positions_sorted = sorted(framework_query_indices)
                framework_sequence_for_iglm = "".join([full_seq_list[i] for i in framework_positions_sorted])
                
                # Calculate the absolute ranges of CDRs in the full sequence (for Infilling mode)
                cdr_ranges_in_full = []
                temp_pos = meta["left_len"]
                for i in range(meta["num_gaps"]):
                    temp_pos += meta["framework_segment_lengths"][i]
                    start = temp_pos
                    end = temp_pos + meta["cdr_lengths"][i]
                    cdr_ranges_in_full.append((start, end))
                    temp_pos = end

                iglm_context_info = {
                    "framework_sequence": framework_sequence_for_iglm,
                    "design_positions": list(design_positions_in_chain),
                    "total_length": total_len,
                    "cdr_ranges": cdr_ranges_in_full,
                }
                print(f"IgLM Context Information:")
                print(f"  Framework Sequence Length: {len(framework_sequence_for_iglm)}")
                print(f"  Number of Design Positions: {len(design_positions_in_chain)}")
                print(f"  Full Sequence Length: {total_len}")
                for i, (s, e) in enumerate(cdr_ranges_in_full):
                    print(f"  CDR{i+1} Range: {s}-{e}")

                # 4) Write the full sequence to the last chain in JSON (only update sequence, do not modify MSA)
                with open(input_JSON_PATH, 'r') as f:
                    json_data = json.load(f)
                if "sequences" not in json_data or not json_data["sequences"]:
                    raise ValueError(f"Missing or empty 'sequences' field in JSON:: {input_JSON_PATH}")
                target_index = len(json_data["sequences"]) - 1
                seq_item = json_data["sequences"][target_index]
                if not seq_item:
                    raise ValueError(f"In JSON at index={target_index} sequence is empty.")
                chain_data = next(iter(seq_item.values()))
                chain_data["sequence"] = full_sequence
                with open(input_JSON_PATH, 'w') as f:
                    json.dump(json_data, f, indent=2)
                sequence = full_sequence

            # Load fold_input (the JSON now contains the complete binder initial sequence)
            fold_inputs_generator = folding_input.load_fold_inputs_from_path(input_JSON_PATH)
            fold_input = next(fold_inputs_generator)

            
            # Run data pipeline if needed
            if data_pipeline_config is not None:
                print('Running data pipeline...')
                print('Note: The last chain (binder) will automatically skip MSA and template search regardless of run_data_pipeline setting')
                fold_input = pipeline.DataPipeline(
                    data_pipeline_config).process(fold_input)
                # Write the updated fold_input back to the original JSON file
                print(f'Write updated fold_input back to original JSON file: {input_JSON_PATH}')
                with open(input_JSON_PATH, 'wt') as f:
                    f.write(fold_input.to_json())
            
        else:
            # Subsequent epochs: Load state from file
            print(f"From epoch {epoch-1} saved states loading...")
            print(f"Loaded sequence: {fold_input.chains[-1].sequence}")
        
        # === Initialize model and optimizer, add learning rate scheduling ===
        if _RUN_INFERENCE.value:       
            # Set epoch information for the model, used for 4-stage optimization (to simplify calling)
            model_runner._model.set_epoch_info(epoch, stage_epochs)
            # Attach design positions (intra-chain indices) to global featurisation attributes for create_target_feat usage
            try:
                import torchcraft.nn.featurization as xfeat
                setattr(
                    xfeat.create_target_feat,
                    "design_positions_in_chain",
                    design_positions_in_chain,
                )
            except Exception as e:
                print(f"Warning: Failed to set design_positions_in_chain on featurization.create_target_feat: {e}")
        else:
            print('Skipping running model inference.')
            model_runner = None
        
        from custom_function.straight_through import four_stage_sequence_optimization, compute_entropy_loss

        # Calculate directly using the real binder_logits_20, while obtaining stage_info and pseudo_seq
        real_pseudo_seq, stage_info = four_stage_sequence_optimization(
            binder_logits_20, epoch, num_epochs, stage_epochs
        )

        from custom_function.aa_type_loss import compute_aa_type_loss
        aa_type_loss = torch.tensor(0.0, device=device)

        if aa_type_loss_weight > 0:
            aa_type_loss = compute_aa_type_loss(
                binder_logits_20=binder_logits_20,
                reference_aa_dist_name=aa_type_loss_dist_name,
                device=device,
            )   

        entropy_loss_value, _ = compute_entropy_loss(real_pseudo_seq)

        lr_scale = stage_info["step"] * (
            (1 - stage_info["soft"]) + (stage_info["soft"] * stage_info["temp"])
        )
        
        # Apply learning rate scaling
        effective_learning_rate = learning_rate_design * lr_scale


        print(f"Stage: {stage_info['stage']}")
        print(f"  - soft: {stage_info['soft']:.4f}")
        print(f"  - temp: {stage_info['temp']:.4f}")
        print(f"  - step: {stage_info['step']:.4f}")
        print(f"  - lr_scale: {lr_scale:.4f}")
        print(f"  - effective_lr: {effective_learning_rate:.6f}")
        print(f"  - description: {stage_info['description']}")
        
        # Initialize optimizer with adjusted learning rate
        optimizer = optim.SGD([binder_logits_20], lr=effective_learning_rate)
        
        # === Create output directory for current epoch ===
        # only save info in the last stage, when: epoch >= num_epochs_except_last (i.e. num_epochs - last_stage_epochs)
        if epoch >= num_epochs_except_last:
            # epoch_output_dir = os.path.join(main_output_dir, f'last_stage_epoch_{epoch - num_epochs_except_last}')
            epoch_output_dir = main_output_dir
            os.makedirs(epoch_output_dir, exist_ok=True)
        
        # === Run inference ===
        print("Performing forward pass...")

        if _RUN_INFERENCE.value:
            model_runner.binder_logits_20 = binder_logits_20
            
        results = process_fold_input(
            epoch,
            fold_input=fold_input,
            data_pipeline_config=None,  # Skip data pipeline, use processed data
            model_runner=model_runner,
            buckets=None,  # Disable buckets feature
            output_dir=epoch_output_dir if epoch >= num_epochs_except_last else None, # only save last stage epochs info.
            last_stage_epoch_info=f'last_stage_epoch_{epoch - num_epochs_except_last}' if epoch >= num_epochs_except_last else None
        )

        if _TURN_OFF_DIFFUSION_CONFIDENCE.value:
            intra_binder_contact_loss = results['intra_binder_contact_loss']
            interface_contact_loss = results['interface_contact_loss']
            helix_loss_score = results['helix_loss']
            negative_contact_loss_score = results['negative_contact_loss']
            negative_framework_contact_loss_score = results['negative_framework_contact_loss']
            paratope_loss_score = results['paratope_loss']
            paratope_cdr_loss_score = results['paratope_cdr_loss']
            paratope_fw_loss_score = results['paratope_fw_loss']
            paratope_cdr_target_loss_score = results['paratope_cdr_target_loss']


            # Modify loss function: Avoid using ptm and iptm to prevent CheckpointError, hotspot not implemented, todo
            # Focus on core losses: pde, plddt, pae_loss, contact_loss, helix_loss
            # Lower pde and pae_loss are better (positive sign), higher plddt is better (negative sign), lower contact_loss and helix_loss are better (positive sign)
            loss = (intra_binder_contact_loss * binder_contact_weight + 
                    interface_contact_loss * interface_contact_weight +
                    helix_loss_score * helix_weight +
                    negative_contact_loss_score * negative_contact_weight +
                    negative_framework_contact_loss_score * negative_framework_contact_weight +
                    paratope_loss_weight * paratope_loss_score + 
                    aa_type_loss_weight * aa_type_loss)
        else:
            ptm_score = results['ptm_binder']
            iptm_score = results['iptm']
            pde_score = results['predicted_distance_error']
            plddt_score = results['plddt_binder']
            pae_loss_score = results['pae_loss']
            intra_binder_contact_loss = results['intra_binder_contact_loss']
            interface_contact_loss = results['interface_contact_loss']
            helix_loss_score = results['helix_loss']
            binder_pae_loss_score = results['binder_pae_loss']
            binder_target_interface_pae_loss_score = results['binder_target_interface_pae_loss']
            negative_contact_loss_score = results['negative_contact_loss']
            negative_framework_contact_loss_score = results['negative_framework_contact_loss']
            paratope_loss_score = results['paratope_loss']
            paratope_cdr_loss_score = results['paratope_cdr_loss']
            paratope_fw_loss_score = results['paratope_fw_loss']
            paratope_cdr_target_loss_score = results['paratope_cdr_target_loss']

            # Modify loss function
            loss = (pde_score * pde_weight -
                    plddt_score * plddt_weight -
                    ptm_score * ptm_weight -
                    iptm_score * iptm_weight +
                    pae_loss_score * pae_weight +
                    binder_pae_loss_score * binder_pae_weight +  
                    binder_target_interface_pae_loss_score * binder_target_interface_pae_weight +  
                    intra_binder_contact_loss * binder_contact_weight + 
                    interface_contact_loss * interface_contact_weight +
                    helix_loss_score * helix_weight +
                    entropy_loss_value * entropy_weight +
                    negative_contact_loss_score * negative_contact_weight +
                    negative_framework_contact_loss_score * negative_framework_contact_weight +
                    paratope_loss_weight * paratope_loss_score +
                    aa_type_loss_weight * aa_type_loss)

        # Initialize IgLM log-likelihood (for CSV logging)
        iglm_ll = 0.0
        iglm_ll_cdr1 = 0.0
        iglm_ll_cdr2 = 0.0
        iglm_ll_cdr3 = 0.0
        
        # === Calculate IgLM Loss and Add to Total Loss ===
        if use_iglm and iglm_model is not None:
            print("Compute IgLM Losss (Decoupled Input/Target Mode)...")
            try:
                # 1. Prepare input for Prediction A (new version: constant temperature 1.0)
                from custom_function.straight_through import exponential_anneal, create_colabdesign_bias
                stage_epochs = [int(x) for x in _STAGE_EPOCHS.value.split(',')]
                total_iglm_anneal_steps = stage_epochs[1] + stage_epochs[2]
                current_iglm_step = max(0, epoch - stage_epochs[0])
                iglm_curr_temp = exponential_anneal(current_iglm_step, total_iglm_anneal_steps, T0=1.0, T_min=1.0)
                
                iglm_bias = create_colabdesign_bias(binder_logits_20.shape[0], rm_aa="C", device=device)
                iglm_input_probs = torch.softmax((binder_logits_20 + iglm_bias) / iglm_curr_temp, dim=-1)
                
                # 2. Prepare input for Target B (used for argmax, with hard annealing real_pseudo_seq T_min=0.01)
                # real_pseudo_seq has already been calculated by four_stage_sequence_optimization earlier
                target_probs = real_pseudo_seq
                
                # 3. Construct probability matrices for full sequence (build Input and Target versions separately)
                full_input_probs = torch.zeros((iglm_context_info["total_length"], 20), device=device)
                full_target_probs = torch.zeros((iglm_context_info["total_length"], 20), device=device)
                aa_to_idx = {aa: i for i, aa in enumerate(iglm_model.amino_acids)}
                
                # Fill framework regions (one-hot encoded)
                design_pos_set = set(iglm_context_info["design_positions"])
                fw_idx = 0
                for i in range(iglm_context_info["total_length"]):
                    if i not in design_pos_set:
                        aa = iglm_context_info["framework_sequence"][fw_idx]
                        if aa in aa_to_idx:
                            val = 1.0
                            full_input_probs[i, aa_to_idx[aa]] = val
                            full_target_probs[i, aa_to_idx[aa]] = val
                        fw_idx += 1
                
                # Fill design positions
                for i, pos in enumerate(iglm_context_info["design_positions"]):
                    full_input_probs[pos] = iglm_input_probs[i]
                    full_target_probs[pos] = target_probs[i]
                
                # 4. Calculate global autoregressive LL
                _, iglm_ll, pos_losses = iglm_model.forward_with_probs(
                    input_probs=full_input_probs, 
                    target_probs=full_target_probs, 
                    chain=iglm_chain
                )
                loss += iglm_weight * pos_losses.mean()
                
                # 5. Calculate Infilling LL for each of the three CDRs separately
                cdr_weights = [iglm_cdr1_weight, iglm_cdr2_weight, iglm_cdr3_weight]
                cdr_lls = []
                for i, cdr_range in enumerate(iglm_context_info["cdr_ranges"]):
                    cdr_ce_loss, cdr_ll = iglm_model.forward_with_infill_probs(
                        input_probs=full_input_probs, 
                        target_probs=full_target_probs, 
                        chain=iglm_chain, 
                        infill_range=cdr_range
                    )
                    loss += cdr_weights[i] * cdr_ce_loss
                    cdr_lls.append(cdr_ll)
                
                if len(cdr_lls) >= 1: iglm_ll_cdr1 = cdr_lls[0]
                if len(cdr_lls) >= 2: iglm_ll_cdr2 = cdr_lls[1]
                if len(cdr_lls) >= 3: iglm_ll_cdr3 = cdr_lls[2]
                
                print(f"  IgLM Input Temp: {iglm_curr_temp:.4f}")
                print(f"  IgLM LL (Full): {iglm_ll:.4f}")
                print(f"  IgLM LL (CDR1): {iglm_ll_cdr1:.4f}")
                print(f"  IgLM LL (CDR2): {iglm_ll_cdr2:.4f}")
                print(f"  IgLM LL (CDR3): {iglm_ll_cdr3:.4f}")
                
            except Exception as e:
                print(f"Warning: Failed to calculate IgLM loss: {e}")
                import traceback
                traceback.print_exc()

        try:
            # Backward pass to compute gradients
            loss.backward()


            # === Fix gradients at specified positions ===
            # Before the optimizer update, manually set the gradients of the residues you do not want to update to zero.
            if fixed_residue_indices and binder_logits_20.grad is not None:
                print(f"Zeroing gradients for fixed positions: {fixed_residue_indices}")
                with torch.no_grad():
                    for idx in fixed_residue_indices:
                        # Check if the index is valid
                        if 0 <= idx < binder_logits_20.grad.shape[0]:
                            binder_logits_20.grad[idx].zero_()
                        else:
                            print(f"Warning: fixed_residue index {idx} is out of bounds, ignored.")

        except RuntimeError as e:
            print("Error occurred during backward computation:",e)
            if retry_count >= max_retry_count:
                print("Maximum retry count reached, training stopped")
                sys.exit(0)
            
            retry_count +=1
            # if epoch >= num_epochs_except_last:
            # # Clear current directory, only for last stage epochs mkdir epoch_output_dir
            #     shutil.rmtree(epoch_output_dir) 
            print(f"\n{'='*50}")
            print(f"Restarting iteration {epoch} ...")
            print(f"{'='*50}")

            # Clear current gradients
            optimizer.zero_grad()
            continue

        # === Save gradient information ===
        # only save info in the last stage, i.e. when: epoch >= (num_epochs - last_stage_epochs)
        if binder_logits_20.grad is not None and epoch >= num_epochs_except_last:
            # # Gradient clipping
            # torch.nn.utils.clip_grad_norm_([binder_logits_20], max_norm=1.0)
            # Get gradients [sequence_length, 20]
            gradients = binder_logits_20.grad.detach().cpu()
            
            # Calculate average gradient magnitude for 20 amino acids at each position
            position_avg_gradients = torch.mean(torch.abs(gradients), dim=1)  # [sequence_length]
            
            # Save detailed gradient information
            gradient_info = {
                'epoch': epoch,
                'sequence_length': gradients.shape[0],
                'full_gradients': gradients,  # [sequence_length, 20] full gradient matrix
                'position_avg_gradients': position_avg_gradients,  # [sequence_length] average gradient magnitude per position
                'gradient_statistics': {
                    'mean_gradient': float(torch.mean(torch.abs(gradients))),
                    'max_gradient': float(torch.max(torch.abs(gradients))),
                    'min_gradient': float(torch.min(torch.abs(gradients))),
                    'std_gradient': float(torch.std(torch.abs(gradients))),
                }
            }            

            print(f"  - Average gradient magnitude: {gradient_info['gradient_statistics']['mean_gradient']:.6f}")
        
        if binder_logits_20.grad is None:
            print("Warning: No gradient information found")
            
        # Update parameters
        optimizer.step()
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # === Save updated state ===
        print("Saving updated state...")
           
        import tempfile
        temp_json = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        temp_json_path = temp_json.name
        temp_json.close()

        # Copy original file to temporary file
        shutil.copy(input_JSON_PATH, temp_json_path)

        # Update sequence in temporary file
        from custom_function.update_sequences import update_sequences

        from custom_function.straight_through import create_omit_C_bias
        bias_tensor = create_omit_C_bias(binder_logits_20.shape[0], device=binder_logits_20.device)
        logits_for_decoding = binder_logits_20 + bias_tensor

        updated_sequence = update_sequences(
            temp_json_path,
            logits_for_decoding,
            target_index=-1,
            design_positions_in_chain=design_positions_in_chain,
        )
        fold_input.chains[-1].sequence = updated_sequence

        # Clean up temporary file
        import os
        os.unlink(temp_json_path)

        if epoch >= num_epochs_except_last:
            print(f"Saved updated state for epoch {epoch} to: {epoch_output_dir}")
        
        # === Log results ===
        # If in nanobody mode and framework information exists, add [] visualization markers to sequences in CSV
        annotated_sequence = updated_sequence
        try:
            if design_positions_in_chain is not None and 'framework_query_indices' in meta:
                fw_indices = meta.get('framework_query_indices', [])
                if fw_indices:
                    annotated_sequence = _annotate_framework_with_brackets(
                        updated_sequence, fw_indices
                    )
        except NameError:
            # In non-nanobody mode, meta does not exist, use the original sequence directly
            annotated_sequence = updated_sequence

        with open(csv_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            if _TURN_OFF_DIFFUSION_CONFIDENCE.value:
                csv_writer.writerow([
                    epoch, 
                    annotated_sequence, 
                    float(intra_binder_contact_loss.detach().cpu().numpy()),  # Binder Contact loss
                    float(interface_contact_loss.detach().cpu().numpy()),     # Interface Contact loss
                    float(helix_loss_score.detach().cpu().numpy()),           # Helix loss
                    float(negative_contact_loss_score.detach().cpu().numpy()),           # Negative Binder Contact loss
                    float(negative_framework_contact_loss_score.detach().cpu().numpy()),  # add negative framework contact loss
                    float(paratope_loss_score.detach().cpu().numpy()),
                    float(paratope_cdr_loss_score.detach().cpu().numpy()),
                    float(paratope_fw_loss_score.detach().cpu().numpy()),
                    float(paratope_cdr_target_loss_score.detach().cpu().numpy()),
                    float(aa_type_loss.detach().cpu().numpy()),
                    float(iglm_ll),
                    float(iglm_ll_cdr1),
                    float(iglm_ll_cdr2),
                    float(iglm_ll_cdr3),
                    float(loss.detach().cpu().numpy()),
                    stage_info['stage'],                                       # Add stage information
                    float(lr_scale),                                          # Add learning rate scale
                    float(effective_learning_rate)                           # Add effective learning rate
                ])
            else:
                csv_writer.writerow([
                    epoch, 
                    annotated_sequence, 
                    float(ptm_score.detach().cpu().numpy()),
                    float(iptm_score.detach().cpu().numpy()),
                    float(pde_score.detach().cpu().numpy()),
                    float(plddt_score.detach().cpu().numpy()),
                    float(pae_loss_score.detach().cpu().numpy()),
                    float(binder_pae_loss_score.detach().cpu().numpy()),
                    float(binder_target_interface_pae_loss_score.detach().cpu().numpy()),
                    float(intra_binder_contact_loss.detach().cpu().numpy()),
                    float(interface_contact_loss.detach().cpu().numpy()),
                    float(helix_loss_score.detach().cpu().numpy()),
                    float(entropy_loss_value.detach().cpu().numpy()),  # Add entropy loss
                    float(negative_contact_loss_score.detach().cpu().numpy()),           # Negative Binder Contact loss
                    float(negative_framework_contact_loss_score.detach().cpu().numpy()),  # Add negative framework contact loss
                    float(paratope_loss_score.detach().cpu().numpy()),
                    float(paratope_cdr_loss_score.detach().cpu().numpy()),
                    float(paratope_fw_loss_score.detach().cpu().numpy()),
                    float(paratope_cdr_target_loss_score.detach().cpu().numpy()),
                    float(aa_type_loss.detach().cpu().numpy()),
                    float(iglm_ll),
                    float(iglm_ll_cdr1),
                    float(iglm_ll_cdr2),
                    float(iglm_ll_cdr3),
                    float(loss.detach().cpu().numpy()),
                    stage_info['stage'],
                    float(lr_scale),
                    float(effective_learning_rate)
                ])
        
        print(f"Epoch {epoch} finished - Sequence: {updated_sequence}")

        # === Cleanup ===
        # Clear NPU cache
        # torch.npu.empty_cache()
        epoch += 1
    
    print(f'\nBinder design for all epochs completed')
    print(f'Training log saved to: {csv_file_path}')

    # Calculate and print total runtime
    end_time = time.time()
    total_time = end_time - start_time
    total_minutes = total_time / 60
    print(f"Program finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total runtime: {total_time:.2f} seconds ({total_minutes:.2f} minutes)")

    if USE_DIST:
        torch.distributed.barrier()
        dist.destroy_process_group()
    print(f'Done processing sequence design.')


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'output_dir',
    ])
    app.run(main)
