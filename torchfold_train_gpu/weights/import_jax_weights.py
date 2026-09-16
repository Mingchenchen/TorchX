"""Model param loading."""

from enum import Enum
from dataclasses import dataclass
from functools import partial
from typing import Union, List

import bisect
import collections
from collections.abc import Iterator
import contextlib
import io
import os
import pathlib
import re
import struct
import sys
from typing import IO

import numpy as np
import zstandard
import torch


class RecordError(Exception):
    """Error reading a record."""


def encode_record(scope: str, name: str, arr: np.ndarray) -> bytes:
    """Encodes a single haiku param as bytes, preserving non-numpy dtypes."""
    scope = scope.encode('utf-8')
    name = name.encode('utf-8')
    shape = arr.shape
    dtype = str(arr.dtype).encode('utf-8')
    arr = np.ascontiguousarray(arr)
    if sys.byteorder == 'big':
        arr = arr.byteswap()
    arr_buffer = arr.tobytes('C')
    header = struct.pack(
        '<5i', len(scope), len(name), len(dtype), len(shape), len(arr_buffer)
    )
    return header + b''.join(
        (scope, name, dtype, struct.pack(f'{len(shape)}i', *shape), arr_buffer)
    )


_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "uint8": torch.uint8,
}


def _read_record(stream: IO[bytes]) -> tuple[str, str, np.ndarray] | None:
    """Reads a record encoded by `_encode_record` from a byte stream."""
    header_size = struct.calcsize('<5i')
    header = stream.read(header_size)
    if not header:
        return None
    if len(header) < header_size:
        raise RecordError(
            f'Incomplete header: {len(header)=} < {header_size=}')
    (scope_len, name_len, dtype_len, shape_len, arr_buffer_len) = struct.unpack(
        '<5i', header
    )
    fmt = f'<{scope_len}s{name_len}s{dtype_len}s{shape_len}i'
    payload_size = struct.calcsize(fmt) + arr_buffer_len
    payload = stream.read(payload_size)
    if len(payload) < payload_size:
        raise RecordError(
            f'Incomplete payload: {len(payload)=} < {payload_size=}')
    scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
    scope = scope.decode('utf-8')
    name = name.decode('utf-8')
    dtype = dtype.decode('utf-8')
    arr = torch.frombuffer(
        bytearray(payload[-arr_buffer_len:]), dtype=_DTYPE_MAP[dtype])
    arr = torch.reshape(arr, shape)
    return scope, name, arr


def read_records(stream: IO[bytes]) -> Iterator[tuple[str, str, np.ndarray]]:
    """Fully reads the contents of a byte stream."""
    while record := _read_record(stream):
        yield record


class _MultiFileIO(io.RawIOBase):
    """A file-like object that presents a concatenated view of multiple files."""

    def __init__(self, files: list[pathlib.Path]):
        self._files = files
        self._stack = contextlib.ExitStack()
        self._handles = [
            self._stack.enter_context(file.open('rb')) for file in files
        ]
        self._sizes = []
        for handle in self._handles:
            handle.seek(0, os.SEEK_END)
            self._sizes.append(handle.tell())
        self._length = sum(self._sizes)
        self._offsets = [0]
        for s in self._sizes[:-1]:
            self._offsets.append(self._offsets[-1] + s)
        self._abspos = 0
        self._relpos = (0, 0)

    def _abs_to_rel(self, pos: int) -> tuple[int, int]:
        idx = bisect.bisect_right(self._offsets, pos) - 1
        return idx, pos - self._offsets[idx]

    def close(self):
        self._stack.close()

    def closed(self) -> bool:
        return all(handle.closed for handle in self._handles)

    def fileno(self) -> int:
        return -1

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._abspos

    def seek(self, pos: int, whence: int = os.SEEK_SET, /):
        match whence:
            case os.SEEK_SET:
                pass
            case os.SEEK_CUR:
                pos += self._abspos
            case os.SEEK_END:
                pos = self._length - pos
            case _:
                raise ValueError(f'Invalid whence: {whence}')
        self._abspos = pos
        self._relpos = self._abs_to_rel(pos)

    def readinto(self, b: bytearray | memoryview) -> int:
        result = 0
        mem = memoryview(b)
        while mem:
            self._handles[self._relpos[0]].seek(self._relpos[1])
            count = self._handles[self._relpos[0]].readinto(mem)
            result += count
            self._abspos += count
            self._relpos = self._abs_to_rel(self._abspos)
            mem = mem[count:]
            if self._abspos == self._length:
                break
        return result


@contextlib.contextmanager
def open_for_reading(model_files: list[pathlib.Path], is_compressed: bool):
    with contextlib.closing(_MultiFileIO(model_files)) as f:
        if is_compressed:
            yield zstandard.ZstdDecompressor().stream_reader(f)
        else:
            yield f


def _match_model(
    paths: list[pathlib.Path], pattern: re.Pattern[str]
) -> dict[str, list[pathlib.Path]]:
    """Match files in a directory with a pattern, and group by model name."""
    models = collections.defaultdict(list)
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match:
            models[match.group('model_name')].append(path)
    return {k: sorted(v) for k, v in models.items()}


def select_model_files(
    model_dir: pathlib.Path, model_name: str | None = None
) -> tuple[list[pathlib.Path], bool]:
    """Select the model files from a model directory."""
    files = [file for file in model_dir.iterdir() if file.is_file()]

    for pattern, is_compressed in (
        (r'(?P<model_name>.*)\.[0-9]+\.bin\.zst$', True),
        (r'(?P<model_name>.*)\.bin\.zst\.[0-9]+$', True),
        (r'(?P<model_name>.*)\.[0-9]+\.bin$', False),
        (r'(?P<model_name>.*)\.bin]\.[0-9]+$', False),
        (r'(?P<model_name>.*)\.bin\.zst$', True),
        (r'(?P<model_name>.*)\.bin$', False),
    ):
        models = _match_model(files, re.compile(pattern))
        if model_name is not None:
            if model_name in models:
                return models[model_name], is_compressed
        else:
            if models:
                if len(models) > 1:
                    raise RuntimeError(
                        f'Multiple models matched in {model_dir}')
                _, model_files = models.popitem()
                return model_files, is_compressed
    raise FileNotFoundError(f'No models matched in {model_dir}')


def get_alphafold3_params(checkpoint_path: pathlib.Path):
    if not os.path.exists(checkpoint_path):
        raise Exception(
            f"Given checkpoint path not exist [{checkpoint_path}]")
    print(f"Loading from {checkpoint_path}")
    is_compressed = False
    if checkpoint_path.suffix == ".zst":
        is_compressed = True
    params = {}
    with open_for_reading([pathlib.Path(checkpoint_path)], is_compressed) as stream:
        for scope, name, arr in read_records(stream):
            params[f"{scope}/{name}"] = arr
    return params


def stacked(param_dict_list, out=None):
    """
    Args:
        param_dict_list:
            A list of (nested) Param dicts to stack. The structure of
            each dict must be the identical (down to the ParamTypes of
            "parallel" Params). There must be at least one dict
            in the list.
    """
    if out is None:
        out = {}
    template = param_dict_list[0]
    for k, _ in template.items():
        v = [d[k] for d in param_dict_list]
        if type(v[0]) is dict:
            out[k] = {}
            stacked(v, out=out[k])
        elif type(v[0]) is Param:
            stacked_param = Param(
                param=[param.param for param in v],
                param_type=v[0].param_type,
                stacked=True,
            )

            out[k] = stacked_param

    return out


def _process_translations_dict(d, _key_prefix, top_layer=True):
    flat = {}
    for k, v in d.items():
        if type(v) == dict:
            prefix = _key_prefix if top_layer else ""
            sub_flat = {
                (prefix + "/".join([k, k_prime])): v_prime
                for k_prime, v_prime in _process_translations_dict(
                    v, _key_prefix, top_layer=False
                ).items()
            }
            flat.update(sub_flat)
        else:
            flat[k] = v

    return flat


def assign(translation_dict, param_to_load):
    for k, param in translation_dict.items():
        with torch.no_grad():
            weights = torch.as_tensor(param_to_load[k])
            ref, param_type = param.param, param.param_type
            if param.stacked:
                if len(ref) == weights.shape[0]:
                    weights = torch.unbind(weights, 0)
                # fixme: this is a hack to handle the fact that the 2 level stacked
                elif len(ref) == weights.shape[0] * weights.shape[1]:
                    weights = torch.unbind(
                        weights.reshape(-1, *weights.shape[2:]), 0)
            else:
                weights = [weights]
                ref = [ref]

            try:
                weights = list(map(param_type.transformation, weights))
                for p, w in zip(ref, weights):
                    p.copy_(w)
            except:
                print(k)
                print(ref[0].shape)
                print(weights[0].shape)
                raise


# With Param, a poor man's enum with attributes (Rust-style)
class ParamType(Enum):
    LinearWeight = partial(  # hack: partial prevents fns from becoming methods
        lambda w: w.transpose(-1, -2)
    )
    LinearWeightMHA = partial(
        lambda w: w.reshape(*w.shape[:-2], -1).transpose(-1, -2)
    )
    LinearWeightNoTransposeMHA = partial(
        lambda w: w.reshape(-1, w.shape[-1])
    )
    LinearBiasMHA = partial(lambda w: w.reshape(*w.shape[:-2], -1))
    LinearFlat = partial(lambda w: w.unsqueeze(-1))
    Other = partial(lambda w: w)

    def __init__(self, fn):
        self.transformation = fn


def cat_params(params, prefix):
    return {
        f"{prefix}{k}": v
        for k, v in params.items()
    }


@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor]]
    param_type: ParamType = ParamType.Other
    stacked: bool = False


def LinearWeight(l, already_transpose_weights=False):
    if already_transpose_weights is True:
        return (Param(l))
    return (Param(l, param_type=ParamType.LinearWeight))


def LinearWeightMHA(l, already_transpose_weights=False):
    if already_transpose_weights is True:
        return (Param(l, param_type=ParamType.LinearWeightNoTransposeMHA))
    return (Param(l, param_type=ParamType.LinearWeightMHA))


def LinearBiasMHA(b): return (Param(b, param_type=ParamType.LinearBiasMHA))


def LinearParams(l, use_bias=False, already_transpose_weights=False):
    d = {"weights": LinearWeight(l.weight, already_transpose_weights)}

    if use_bias:
        d["bias"] = Param(l.bias)

    return d


def LinearfromFlatParams(l, use_bias=False):
    d = {"weights": Param(l.weight, param_type=ParamType.LinearFlat)}

    if use_bias:
        d["bias"] = Param(l.bias)

    return d


def LinearHMAParams(l, use_bias=False, already_transpose_weights=False):
    d = {"weights": LinearWeightMHA(l.weight, already_transpose_weights)}

    if use_bias:
        d["bias"] = LinearBiasMHA(l.bias)
    return d


def LayerNormParams(l, use_bias=True):
    d = {
        "scale": Param(l.weight),
    }
    if use_bias:
        d["offset"] = Param(l.bias)

    return d


def AdaptiveLayerNormParams(aln, use_single_cond=False):
    if use_single_cond is False:
        return {
            "layer_norm": LayerNormParams(aln.layer_norm),
        }
    else:
        return {
            "single_cond_layer_norm": LayerNormParams(aln.single_cond_layer_norm, use_bias=False),
            "single_cond_scale": LinearParams(aln.single_cond_scale, use_bias=True),
            "single_cond_bias": LinearParams(aln.single_cond_bias),
        }


def AdaLNZeroParams(ada_ln_zero, use_single_cond=False):
    d = {
        "transition2": LinearParams(ada_ln_zero.transition2),
    }

    if use_single_cond is True:
        d.update({
            "adaptive_zero_cond": LinearParams(ada_ln_zero.adaptive_zero_cond, use_bias=True),
        })

    return d


def TriMulParams(tri_mul): return {
    "left_norm_input": LayerNormParams(tri_mul.left_norm_input),
    "projection": LinearParams(tri_mul.projection),
    "gate": LinearParams(tri_mul.gate),
    "center_norm": LayerNormParams(tri_mul.center_norm),
    "output_projection": LinearParams(tri_mul.output_projection),
    "gating_linear": LinearParams(tri_mul.gating_linear)
}


def OuterProductMeanParams(outer_product_mean): return {
    "layer_norm_input": LayerNormParams(outer_product_mean.layer_norm_input),
    "left_projection": LinearParams(outer_product_mean.left_projection),
    "right_projection": LinearParams(outer_product_mean.right_projection),
    "output_w": Param(outer_product_mean.output_w),
    "output_b": Param(outer_product_mean.output_b),
}


def TransitionParams(transition): return {
    "input_layer_norm": LayerNormParams(transition.input_layer_norm),
    "transition1": LinearParams(transition.transition1),
    "transition2": LinearParams(transition.transition2),
}


def GridSelfAttentionParams(pair_attention): return {
    "act_norm": LayerNormParams(pair_attention.act_norm),
    "pair_bias_projection": LinearParams(pair_attention.pair_bias_projection),
    "q_projection": LinearHMAParams(pair_attention.q_projection, already_transpose_weights=True),
    "k_projection": LinearHMAParams(pair_attention.k_projection, already_transpose_weights=True),
    "v_projection": LinearHMAParams(pair_attention.v_projection),
    "gating_query": LinearParams(pair_attention.gating_query, already_transpose_weights=True),
    "output_projection": LinearParams(pair_attention.output_projection),
}


def SelfAttentionParams(self_attention, use_single_cond=False):
    return {
        "q_projection": LinearHMAParams(self_attention.q_projection, use_bias=True),
        "k_projection": LinearHMAParams(self_attention.k_projection),
        "v_projection": LinearHMAParams(self_attention.v_projection),
        "gating_query": LinearParams(self_attention.gating_query),
        "transition2": LinearParams(self_attention.adaptive_zero_init.transition2),
        **AdaptiveLayerNormParams(self_attention.adaptive_layernorm, use_single_cond),
        **AdaLNZeroParams(self_attention.adaptive_zero_init, use_single_cond),
    }


def CrossAttentionParams(cross_attention): return {
    **cat_params(AdaptiveLayerNormParams(cross_attention.q_adaptive_layernorm, use_single_cond=True), "q"),
    **cat_params(AdaptiveLayerNormParams(cross_attention.k_adaptive_layernorm, use_single_cond=True), "k"),
    "q_projection": LinearHMAParams(cross_attention.q_projection, use_bias=True),
    "k_projection": LinearHMAParams(cross_attention.k_projection),
    "v_projection": LinearHMAParams(cross_attention.v_projection),
    "gating_query": LinearParams(cross_attention.gating_query),
    **AdaLNZeroParams(cross_attention.adaptive_zero_init, use_single_cond=True),
}


def MSAAttentionParams(msa_attention): return {
    "act_norm": LayerNormParams(msa_attention.act_norm),
    "pair_norm": LayerNormParams(msa_attention.pair_norm),
    "pair_logits": LinearParams(msa_attention.pair_logits),
    "v_projection": LinearHMAParams(msa_attention.v_projection),
    "gating_query": LinearParams(msa_attention.gating_query),
    "output_projection": LinearParams(msa_attention.output_projection),
}


def DiffusionTransitionParams(transition, use_single_cond=False):
    return {
        **AdaptiveLayerNormParams(transition.adaptive_layernorm, use_single_cond),
        "transition1": LinearParams(transition.transition1),
        **AdaLNZeroParams(transition.adaptive_zero_init, use_single_cond),
    }


def DiffusionTransformerParams(transformer):

    self_attention_params = stacked([SelfAttentionParams(
        l, use_single_cond=True) for l in transformer.self_attention])
    transistion_params = stacked([DiffusionTransitionParams(
        l, use_single_cond=True) for l in transformer.transition_block])

    return {
        "pair_input_layer_norm": LayerNormParams(transformer.pair_input_layer_norm, use_bias=False),
        "__layer_stack_with_per_layer/pair_logits_projection": stacked([LinearHMAParams(l) for l in transformer.pair_logits_projection]),
        **cat_params(self_attention_params, "__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer"),
        **cat_params(transistion_params, "__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_"),
    }


def DiffusionCrossAttTransformerParams(transformer, prefix="diffusion_atom_transformer_encoder"):

    cross_attention_params = stacked([CrossAttentionParams(
        l) for l in transformer.cross_attention])
    transistion_params = stacked([DiffusionTransitionParams(
        l, use_single_cond=True) for l in transformer.transition_block])

    return {
        "pair_input_layer_norm": LayerNormParams(transformer.pair_input_layer_norm, use_bias=False),
        "pair_logits_projection": LinearHMAParams(transformer.pair_logits_projection),
        **cat_params(cross_attention_params, f"__layer_stack_with_per_layer/{prefix}"),
        **cat_params(transistion_params, f"__layer_stack_with_per_layer/{prefix}ffw_"),
    }


def AtomCrossAttEncoderParams(encoder,
                              with_token_atoms_act=False,
                              with_trunk_single_cond=False,
                              with_trunk_pair_cond=False,
                              prefix="evoformer_conditioning_atom_transformer_encoder"):
    d = {
        "embed_ref_pos": LinearParams(encoder.embed_ref_pos),
        "embed_ref_mask": LinearParams(encoder.embed_ref_mask),
        "embed_ref_element": LinearParams(encoder.embed_ref_element),
        "embed_ref_charge": LinearParams(encoder.embed_ref_charge),
        "embed_ref_atom_name": LinearParams(encoder.embed_ref_atom_name),
        "single_to_pair_cond_row": LinearParams(encoder.single_to_pair_cond_row),
        "single_to_pair_cond_col": LinearParams(encoder.single_to_pair_cond_col),
        "embed_pair_offsets": LinearParams(encoder.embed_pair_offsets),
        "embed_pair_distances": LinearParams(encoder.embed_pair_distances),
        "single_to_pair_cond_row_1": LinearParams(encoder.single_to_pair_cond_row_1),
        "single_to_pair_cond_col_1": LinearParams(encoder.single_to_pair_cond_col_1),
        "embed_pair_offsets_1": LinearParams(encoder.embed_pair_offsets_1),
        "embed_pair_distances_1": LinearParams(encoder.embed_pair_distances_1),
        "embed_pair_offsets_valid": LinearParams(encoder.embed_pair_offsets_valid),
        "pair_mlp_1": LinearParams(encoder.pair_mlp_1),
        "pair_mlp_2": LinearParams(encoder.pair_mlp_2),
        "pair_mlp_3": LinearParams(encoder.pair_mlp_3),
        "atom_transformer_encoder": DiffusionCrossAttTransformerParams(encoder.atom_transformer_encoder, prefix=prefix),
        "project_atom_features_for_aggr": LinearParams(encoder.project_atom_features_for_aggr),
    }

    if with_token_atoms_act is True:
        d.update({
            "atom_positions_to_features": LinearParams(encoder.atom_positions_to_features),
        })

    if with_trunk_single_cond is True:
        d.update({
            "lnorm_trunk_single_cond": LayerNormParams(encoder.lnorm_trunk_single_cond, use_bias=False),
            "embed_trunk_single_cond": LinearParams(encoder.embed_trunk_single_cond),
        })

    if with_trunk_pair_cond:
        d.update({
            "lnorm_trunk_pair_cond": LayerNormParams(encoder.lnorm_trunk_pair_cond, use_bias=False),
            "embed_trunk_pair_cond": LinearParams(encoder.embed_trunk_pair_cond),
        })

    return d


def AtomCrossAttDecoderParams(decoder): return {
    "project_token_features_for_broadcast": LinearParams(decoder.project_token_features_for_broadcast),
    "atom_transformer_decoder": DiffusionCrossAttTransformerParams(decoder.atom_transformer_decoder, prefix="diffusion_atom_transformer_decoder"),
    "atom_features_layer_norm": LayerNormParams(decoder.atom_features_layer_norm, use_bias=False),
    "atom_features_to_position_update": LinearParams(decoder.atom_features_to_position_update),
}


def TemplateEmbeddingParams(template_embedding):

    pairformer_params = stacked(
        [PairformerBlockParams(b, with_single=False) for b in template_embedding.single_template_embedding.template_embedding_iteration])

    return {
        "single_template_embedding/query_embedding_norm": LayerNormParams(template_embedding.single_template_embedding.query_embedding_norm),
        "single_template_embedding/template_pair_embedding_0": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_0),
        "single_template_embedding/template_pair_embedding_1": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_1),
        "single_template_embedding/template_pair_embedding_2": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_2),
        "single_template_embedding/template_pair_embedding_3": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_3),
        "single_template_embedding/template_pair_embedding_4": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_4),
        "single_template_embedding/template_pair_embedding_5": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_5),
        "single_template_embedding/template_pair_embedding_6": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_6),
        "single_template_embedding/template_pair_embedding_7": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_7),
        "single_template_embedding/template_pair_embedding_8": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_8),
        **cat_params(pairformer_params, f"single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/"),
        "single_template_embedding/output_layer_norm": LayerNormParams(template_embedding.single_template_embedding.output_layer_norm),
        "output_linear": LinearParams(template_embedding.output_linear),
    }


def PairformerBlockParams(b, with_single=False):
    d = {
        "triangle_multiplication_outgoing": TriMulParams(b.triangle_multiplication_outgoing),
        "triangle_multiplication_incoming": TriMulParams(b.triangle_multiplication_incoming),
        "pair_attention1": GridSelfAttentionParams(b.pair_attention1),
        "pair_attention2": GridSelfAttentionParams(b.pair_attention2),
        "pair_transition": TransitionParams(b.pair_transition),
    }

    if with_single is True:
        d.update({
            "single_pair_logits_norm": LayerNormParams(b.single_pair_logits_norm),
            "single_pair_logits_projection": LinearParams(b.single_pair_logits_projection),
            **cat_params(SelfAttentionParams(b.single_attention_), "single_attention_"),
            "single_transition": TransitionParams(b.single_transition),
        })

    return d


def EvoformerBlockParams(b): return {
    "outer_product_mean": OuterProductMeanParams(b.outer_product_mean),
    "msa_attention1": MSAAttentionParams(b.msa_attention1),
    "msa_transition": TransitionParams(b.msa_transition),
    "triangle_multiplication_outgoing": TriMulParams(b.triangle_multiplication_outgoing),
    "triangle_multiplication_incoming": TriMulParams(b.triangle_multiplication_incoming),
    "pair_attention1": GridSelfAttentionParams(b.pair_attention1),
    "pair_attention2": GridSelfAttentionParams(b.pair_attention2),
    "pair_transition": TransitionParams(b.pair_transition),
}


def DiffusionHeadParams(head):
    return {
        "pair_cond_initial_norm": LayerNormParams(head.pair_cond_initial_norm, use_bias=False),
        "pair_cond_initial_projection": LinearParams(head.pair_cond_initial_projection),
        **cat_params(DiffusionTransitionParams(head.pair_transition_0), "pair_transition_0ffw_"),
        **cat_params(DiffusionTransitionParams(head.pair_transition_1), "pair_transition_1ffw_"),
        "single_cond_initial_norm": LayerNormParams(head.single_cond_initial_norm, use_bias=False),
        "single_cond_initial_projection": LinearParams(head.single_cond_initial_projection),
        "noise_embedding_initial_norm": LayerNormParams(head.noise_embedding_initial_norm, use_bias=False),
        "noise_embedding_initial_projection": LinearParams(head.noise_embedding_initial_projection),
        **cat_params(DiffusionTransitionParams(head.single_transition_0), "single_transition_0ffw_"),
        **cat_params(DiffusionTransitionParams(head.single_transition_1), "single_transition_1ffw_"),
        **cat_params(AtomCrossAttEncoderParams(head.atom_cross_att_encoder,
                                               with_token_atoms_act=True,
                                               with_trunk_pair_cond=True,
                                               with_trunk_single_cond=True,
                                               prefix="diffusion_atom_transformer_encoder"), "diffusion_"),
        "single_cond_embedding_norm": LayerNormParams(head.single_cond_embedding_norm, use_bias=False),
        "single_cond_embedding_projection": LinearParams(head.single_cond_embedding_projection),
        "transformer": DiffusionTransformerParams(head.transformer),
        "output_norm": LayerNormParams(head.output_norm, use_bias=False),
        **cat_params(AtomCrossAttDecoderParams(head.atom_cross_att_decoder), "diffusion_")
    }


def ConfidenceHeadParams(head):

    pairformer_blocks_params = stacked(
        [PairformerBlockParams(b, with_single=True) for b in head.confidence_pairformer])

    return {
        "~_embed_features/left_target_feat_project": LinearParams(head.left_target_feat_project),
        "~_embed_features/right_target_feat_project": LinearParams(head.right_target_feat_project),
        "~_embed_features/distogram_feat_project": LinearParams(head.distogram_feat_project),
        "__layer_stack_no_per_layer/confidence_pairformer": pairformer_blocks_params,
        "logits_ln": LayerNormParams(head.logits_ln),
        "left_half_distance_logits": LinearParams(head.left_half_distance_logits),
        "pae_logits_ln": LayerNormParams(head.pae_logits_ln),
        "pae_logits": LinearParams(head.pae_logits),
        "plddt_logits_ln": LayerNormParams(head.plddt_logits_ln),
        "plddt_logits": LinearHMAParams(head.plddt_logits),
        "experimentally_resolved_ln": LayerNormParams(head.experimentally_resolved_ln),
        "experimentally_resolved_logits": LinearHMAParams(head.experimentally_resolved_logits),
    }


def EvoformerParams(evoformer):

    msa_stack_params = stacked(
        [EvoformerBlockParams(b) for b in evoformer.msa_stack])

    trunk_pairformer_params = stacked(
        [PairformerBlockParams(b, with_single=True) for b in evoformer.trunk_pairformer])

    return {
        "left_single": LinearParams(evoformer.left_single),
        "right_single": LinearParams(evoformer.right_single),
        "prev_embedding_layer_norm": LayerNormParams(evoformer.prev_embedding_layer_norm),
        "prev_embedding": LinearParams(evoformer.prev_embedding),
        "~_relative_encoding/position_activations": LinearParams(evoformer.position_activations),
        "bond_embedding": LinearParams(evoformer.bond_embedding),
        "template_embedding": TemplateEmbeddingParams(evoformer.template_embedding),
        "msa_activations": LinearParams(evoformer.msa_activations),
        "extra_msa_target_feat": LinearParams(evoformer.extra_msa_target_feat),
        **cat_params(msa_stack_params, "__layer_stack_no_per_layer/msa_stack/"),
        "single_activations": LinearParams(evoformer.single_activations),
        "prev_single_embedding_layer_norm": LayerNormParams(evoformer.prev_single_embedding_layer_norm),
        "prev_single_embedding": LinearParams(evoformer.prev_single_embedding),
        **cat_params(trunk_pairformer_params, "__layer_stack_no_per_layer_1/trunk_pairformer/"),
    }


def get_translation_dict(model):
    translations = {
        **cat_params(AtomCrossAttEncoderParams(model.evoformer_conditioning), "evoformer_conditioning_"),
        "evoformer": EvoformerParams(model.evoformer),
        "~/diffusion_head": DiffusionHeadParams(model.diffusion_head),
        "distogram_head/half_logits": LinearParams(model.distogram_head.half_logits),
        "confidence_head": ConfidenceHeadParams(model.confidence_head),
    }

    return translations


def import_jax_weights_(model, model_path: pathlib.Path):
    try:
        params = get_alphafold3_params(model_path / "af3.bin")
    except Exception:
        params = get_alphafold3_params(model_path / "af3.bin.zst")

    translations = get_translation_dict(model)

    flat = _process_translations_dict(translations, _key_prefix="diffuser/")

    for k in flat.keys():
        if k not in params:
            print(f"Key {k} not found in params")

    for k in params.keys():
        if k not in flat and k != "__meta__/__identifier__":
            print(f"Key {k} not found in torch module")

    assign(flat, params)

    model.__identifier__ = params['__meta__/__identifier__']

    fourier_embeddings = model.diffusion_head.fourier_embeddings
    _WEIGHT = [
        0.45873642,  0.06516238, -0.07278306, -0.26992258,  0.64292115,
        -0.40763968,  3.60116863,  0.54461384, -0.32644904,  2.10888267,
        1.30805349,  1.19838560, -1.37745857,  1.99475312, -1.64120293,
        1.07823789, -0.02288206,  0.88305283,  0.48099944,  0.17655374,
        0.30281949,  0.80646873,  0.62605333, -0.23965347, -1.02609432,
        0.75006109, -0.19913037,  0.07466396,  0.66431236, -0.60990530,
        -0.69709194, -0.44453633, -1.77656078,  0.02299878,  0.04095552,
        0.35485864, -0.47602659, -0.98820388, -0.24106771, -1.07254291,
        -0.99741757,  0.22697604,  1.41390419,  1.54984057, -0.12237291,
        0.20156337,  0.61767143,  0.23959029,  0.92454034,  1.84082258,
        0.89030224,  0.39598912, -1.52224910,  0.29669049,  1.52356744,
        -0.33968377,  0.24155144, -0.52308381, -0.23622665,  0.92825454,
        -0.63864607, -0.62169307,  0.78623551, -0.80352145, -0.45496067,
        1.30877995, -0.06686528,  1.00248849, -0.63593471,  0.16372502,
        -1.46133232,  1.10562658, -0.01693927,  0.28684548, -0.72843230,
        0.66133535, -1.92225552,  0.70241231, -0.96868867, -0.47309339,
        -1.66894221,  0.46018723, -0.56806105,  0.32694784, -0.46529883,
        1.02299964,  0.84688205,  1.19581807, -1.82454145,  0.05999713,
        -0.59530073,  1.44862521, -0.34933713, -0.46564487, -0.55005538,
        -1.61170268,  0.17502306,  0.38670063, -1.12133658, -0.29343036,
        -0.52527446, -1.26285112,  1.07982683,  0.51215219,  1.48963666,
        1.09847653, -0.01563358,  0.32574457,  1.94779706, -1.29198587,
        1.06249654, -0.86965990,  0.22975266, -0.27182648, -0.21130897,
        -0.41773933, -0.02329035,  1.31049252,  0.05579265, -1.23127055,
        -0.99691105,  0.27058721, -0.72509319, -0.14421797, -1.48605061,
        1.35041201,  1.29619241, -1.01022530, -0.79787987, -0.16166858,
        0.87210685,  1.69248152,  1.42469788, -0.72325104, -1.24823737,
        0.07051118,  0.71332991, -0.07360429, -0.91955227, -2.68856549,
        -0.44033936,  0.35482934, -0.57933813,  0.97468042, -0.31050494,
        -0.88454425, -2.08785224,  0.47322822, -0.02400172,  0.26644820,
        -0.19147627, -2.10538960, -1.27962470, -1.35999286,  2.09867334,
        0.65099514,  0.21604492, -0.45951018,  0.15994427, -0.31420693,
        -0.65202618, -0.61077976, -1.06100249, -1.47254968,  1.18165290,
        -0.78656220,  1.28182006,  1.80323684,  1.09196901,  0.26118696,
        -0.30168581,  0.39749333,  0.26812574, -1.51995814, -0.46909946,
        0.03874255, -1.36774313,  2.30143976,  2.06959820, -0.41647521,
        1.85624206,  0.49019700, -0.06726539,  0.00457313,  0.23915423,
        -1.84971249, -0.20482327, -0.34097880, -0.57933033, -1.10541213,
        -0.30269983, -0.16430426, -0.82371718,  0.10345812,  1.78753936,
        0.04786763,  1.86778629, -0.65214992,  0.81544143, -0.28214937,
        0.31187257,  0.57661986,  1.21938801, -1.56046617,  0.38046429,
        -0.18235965,  0.81794524, -0.40474343,  0.46538028, -1.15558851,
        0.59625793, -1.07801270,  0.07310858,  0.61526084,  0.55518496,
        -0.49787554,  0.92703879, -1.27780271, -0.83373469, -0.43015575,
        0.41877759, -1.03987372, -1.46055734,  0.61282396,  0.15590595,
        -0.34269521,  0.56509072, -1.17904210,  0.11374855, -1.83310866,
        0.38734794, -0.58623004,  0.77931106,  1.53930688, -0.70299625,
        -0.11389336, -1.14818096, -0.44400632,  1.21887410,  0.64066756,
        -0.70249403, -0.27244881,  0.38586098, -1.07925785,  0.12448707,
        -1.28286278,  0.37827531,  0.68812364,  1.65695465,  0.12440517,
        -0.03689830,  1.10224664, -0.28323629, -0.47939169,  0.70120829,
        -0.67204583
    ]

    _BIAS = [
        0.00465965, 0.21738243, 0.22277749, 0.68463874, 0.84596848, 0.17337036,
        0.39573753, 0.78153563, 0.86311185, 0.21782327, 0.24377882, 0.42310703,
        0.19887352, 0.10486019, 0.48707581, 0.22205460, 0.97263455, 0.29714966,
        0.11244559, 0.53020525, 0.36796236, 0.37294638, 0.80261672, 0.04669094,
        0.86319661, 0.75907171, 0.77297020, 0.01114798, 0.55850804, 0.91799915,
        0.23032320, 0.12154722, 0.26701927, 0.42934716, 0.47951782, 0.96782577,
        0.86785042, 0.61985648, 0.05743814, 0.41800117, 0.68881893, 0.60575199,
        0.21058667, 0.64412105, 0.63958526, 0.89390790, 0.69755554, 0.89345169,
        0.53330755, 0.56985939, 0.30724049, 0.00984561, 0.91407037, 0.92118979,
        0.94153070, 0.81097460, 0.70537627, 0.32810748, 0.47227263, 0.11821401,
        0.44983089, 0.30767226, 0.31756389, 0.62969446, 0.69892538, 0.16949117,
        0.06207097, 0.46717727, 0.95348179, 0.62363589, 0.49018729, 0.06920040,
        0.39333904, 0.41299903, 0.52514863, 0.61197245, 0.56871891, 0.65053988,
        0.22203422, 0.46748531, 0.86931503, 0.87050021, 0.40208721, 0.32084906,
        0.55084610, 0.94584596, 0.76279902, 0.36250532, 0.74272907, 0.66682065,
        0.96452832, 0.64768302, 0.88070846, 0.56995463, 0.06395614, 0.69499350,
        0.44494808, 0.39775658, 0.20280898, 0.33363521, 0.05999005, 0.44414878,
        0.65227020, 0.01199079, 0.71995056, 0.19045687, 0.48342144, 0.25127733,
        0.66515994, 0.22465158, 0.22313106, 0.06302810, 0.55783665, 0.93625581,
        0.58800840, 0.72525370, 0.52879298, 0.77195418, 0.15548682, 0.01028740,
        0.39325142, 0.45401239, 0.71494079, 0.33011997, 0.05050695, 0.26381660,
        0.63064706, 0.47604024, 0.08593416, 0.00383425, 0.06352687, 0.05510247,
        0.03552997, 0.35810637, 0.56094289, 0.60922170, 0.88599777, 0.45419788,
        0.40486634, 0.71297824, 0.34976673, 0.97825217, 0.12915993, 0.09566259,
        0.64318919, 0.16717327, 0.82308614, 0.32672071, 0.81688786, 0.84857118,
        0.99922776, 0.07551706, 0.18766022, 0.13051236, 0.39136350, 0.08768725,
        0.92048228, 0.87185788, 0.39158428, 0.79224777, 0.17492688, 0.68902445,
        0.81980729, 0.70458186, 0.59489477, 0.93324888, 0.49986637, 0.40705478,
        0.89202917, 0.20673239, 0.39339757, 0.20996964, 0.02923799, 0.53992438,
        0.40119815, 0.10366607, 0.08044600, 0.95551598, 0.20518017, 0.68826210,
        0.90159297, 0.69008791, 0.86880815, 0.16246438, 0.89628279, 0.11481643,
        0.61353648, 0.41545081, 0.92478311, 0.78212476, 0.48292696, 0.79621077,
        0.11947489, 0.01747024, 0.22928023, 0.87387264, 0.86349785, 0.89526737,
        0.58904779, 0.13896775, 0.68194926, 0.55824125, 0.44428205, 0.55422378,
        0.28189969, 0.27923775, 0.09979951, 0.66994715, 0.45943546, 0.71207762,
        0.17300689, 0.83434916, 0.02573085, 0.45858085, 0.55934799, 0.30676675,
        0.52219367, 0.34544575, 0.19280875, 0.26937950, 0.07147646, 0.06295013,
        0.76382887, 0.38737607, 0.58825982, 0.17423475, 0.05509448, 0.97228825,
        0.94380617, 0.91664016, 0.18800116, 0.41771865, 0.59420645, 0.77371931,
        0.64687788, 0.27284670, 0.22310913, 0.15663862, 0.45573199, 0.50386798,
        0.66712272, 0.71649647, 0.28475654, 0.83415413, 0.75261366, 0.61517799,
        0.93544555, 0.76141870, 0.85474241, 0.74766934, 0.33459592, 0.78477907,
        0.07250881, 0.10174239, 0.95332730, 0.80793905
    ]
    fourier_embeddings_weight = torch.from_numpy(np.array(_WEIGHT)).to(dtype=torch.float32)
    fourier_embeddings_bias = torch.from_numpy(np.array(_BIAS)).to(dtype=torch.float32)

    with torch.no_grad():
        if hasattr(fourier_embeddings, 'weight'):
            fourier_embeddings.weight.copy_(fourier_embeddings_weight)
        else:
            fourier_embeddings.register_buffer("weight", fourier_embeddings_weight)

        if hasattr(fourier_embeddings, 'bias'):
            fourier_embeddings.bias.copy_(fourier_embeddings_bias)
        else:
            fourier_embeddings.register_buffer("bias", fourier_embeddings_bias)
