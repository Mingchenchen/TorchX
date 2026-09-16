"""Data-side of the input features processing."""

import dataclasses
import datetime
from typing import Self, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from torchfold.cpp import cif_dict

from torchfold import structure
from torchfold.processing.atom_layout import atom_layout

xnp_ndarray: TypeAlias = np.ndarray | jnp.ndarray  # pylint: disable=invalid-name
BatchDict: TypeAlias = dict[str, xnp_ndarray]

VALID_DTYPES = [np.float32, np.float64, np.int8, np.int32, np.int64, bool]


def remove_invalidly_typed_feats(
        batch: BatchDict,
) -> BatchDict:
    """Remove features of types we don't want to send to the TPU e.g. strings."""
    return {
        k: v
        for k, v in batch.items()
        if hasattr(v, 'dtype') and v.dtype in VALID_DTYPES
    }


@dataclasses.dataclass(frozen=True)
class PaddingShapes:
    num_tokens: int
    msa_size: int
    num_chains: int
    num_templates: int
    num_atoms: int


def _pad_to(
        arr: np.ndarray, shape: tuple[int | None, ...], **kwargs
) -> np.ndarray:
    """Pads an array to a given shape. Wrapper around np.pad().

    Args:
      arr: numpy array to pad
      shape: target shape, use None for axes that should stay the same
      **kwargs: additional args for np.pad, e.g. constant_values=-1

    Returns:
      the padded array

    Raises:
      ValueError if arr and shape have a different number of axes.
    """
    if arr.ndim != len(shape):
        raise ValueError(
            f'arr and shape have different number of axes. {arr.shape=}, {shape=}'
        )

    num_pad = []
    for axis, width in enumerate(shape):
        if width is None:
            num_pad.append((0, 0))
        else:
            if width >= arr.shape[axis]:
                num_pad.append((0, width - arr.shape[axis]))
            else:
                raise ValueError(
                    f'Can not pad to a smaller shape. {arr.shape=}, {shape=}'
                )
    padded_arr = np.pad(arr, pad_width=num_pad, **kwargs)
    return padded_arr


def _unwrap(obj):
    """Unwrap an object from a zero-dim np.ndarray."""
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    else:
        return obj


@dataclasses.dataclass(frozen=True)
class Chains:
    chain_id: np.ndarray
    asym_id: np.ndarray
    entity_id: np.ndarray
    sym_id: np.ndarray


jax.tree_util.register_dataclass(
    Chains,
    data_fields=[f.name for f in dataclasses.fields(Chains)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class MSA:
    """Dataclass containing MSA."""

    rows: xnp_ndarray
    mask: xnp_ndarray
    deletion_matrix: xnp_ndarray
    # Occurrence of each residue type along the sequence, averaged over MSA rows.
    profile: xnp_ndarray
    # Occurrence of deletions along the sequence, averaged over MSA rows.
    deletion_mean: xnp_ndarray
    # Number of MSA alignments.
    num_alignments: xnp_ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        output = cls(
            rows=batch['msa'],
            mask=batch['msa_mask'],
            deletion_matrix=batch['deletion_matrix'],
            profile=batch['profile'],
            deletion_mean=batch['deletion_mean'],
            num_alignments=batch['num_alignments'],
        )
        return output

    def as_data_dict(self) -> BatchDict:
        return {
            'msa': self.rows,
            'msa_mask': self.mask,
            'deletion_matrix': self.deletion_matrix,
            'profile': self.profile,
            'deletion_mean': self.deletion_mean,
            'num_alignments': self.num_alignments,
        }


jax.tree_util.register_dataclass(
    MSA,
    data_fields=[f.name for f in dataclasses.fields(MSA)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class Templates:
    """Dataclass containing templates."""

    # aatype of templates, int32 w shape [num_templates, num_res]
    aatype: xnp_ndarray
    # atom positions of templates, float32 w shape [num_templates, num_res, 24, 3]
    atom_positions: xnp_ndarray
    # atom mask of templates, bool w shape [num_templates, num_res, 24]
    atom_mask: xnp_ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        """Make Template from batch dictionary."""
        return cls(
            aatype=batch['template_aatype'],
            atom_positions=batch['template_atom_positions'],
            atom_mask=batch['template_atom_mask'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'template_aatype': self.aatype,
            'template_atom_positions': self.atom_positions,
            'template_atom_mask': self.atom_mask,
        }


jax.tree_util.register_dataclass(
    Templates,
    data_fields=[f.name for f in dataclasses.fields(Templates)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class TokenFeatures:
    """Dataclass containing features for tokens."""

    residue_index: xnp_ndarray
    token_index: xnp_ndarray
    aatype: xnp_ndarray
    mask: xnp_ndarray
    seq_length: xnp_ndarray

    # Chain symmetry identifiers
    # for an A3B2 stoichiometry the meaning of these features is as follows:
    # asym_id:    1 2 3 4 5
    # entity_id:  1 1 1 2 2
    # sym_id:     1 2 3 1 2
    asym_id: xnp_ndarray
    entity_id: xnp_ndarray
    sym_id: xnp_ndarray

    # token type features
    is_protein: xnp_ndarray
    is_rna: xnp_ndarray
    is_dna: xnp_ndarray
    is_ligand: xnp_ndarray
    is_nonstandard_polymer_chain: xnp_ndarray
    is_water: xnp_ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            residue_index=batch['residue_index'],
            token_index=batch['token_index'],
            aatype=batch['aatype'],
            mask=batch['seq_mask'],
            entity_id=batch['entity_id'],
            asym_id=batch['asym_id'],
            sym_id=batch['sym_id'],
            seq_length=batch['seq_length'],
            is_protein=batch['is_protein'],
            is_rna=batch['is_rna'],
            is_dna=batch['is_dna'],
            is_ligand=batch['is_ligand'],
            is_nonstandard_polymer_chain=batch['is_nonstandard_polymer_chain'],
            is_water=batch['is_water'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'residue_index': self.residue_index,
            'token_index': self.token_index,
            'aatype': self.aatype,
            'seq_mask': self.mask,
            'entity_id': self.entity_id,
            'asym_id': self.asym_id,
            'sym_id': self.sym_id,
            'seq_length': self.seq_length,
            'is_protein': self.is_protein,
            'is_rna': self.is_rna,
            'is_dna': self.is_dna,
            'is_ligand': self.is_ligand,
            'is_nonstandard_polymer_chain': self.is_nonstandard_polymer_chain,
            'is_water': self.is_water,
        }


jax.tree_util.register_dataclass(
    TokenFeatures,
    data_fields=[f.name for f in dataclasses.fields(TokenFeatures)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class PredictedStructureInfo:
    """Contains information necessary to work with predicted structure."""

    atom_mask: xnp_ndarray
    residue_center_index: xnp_ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            atom_mask=batch['pred_dense_atom_mask'],
            residue_center_index=batch['residue_center_index'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'pred_dense_atom_mask': self.atom_mask,
            'residue_center_index': self.residue_center_index,
        }


jax.tree_util.register_dataclass(
    PredictedStructureInfo,
    data_fields=[f.name for f in dataclasses.fields(PredictedStructureInfo)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class PolymerLigandBondInfo:
    """Contains information about polymer-ligand bonds."""

    tokens_to_polymer_ligand_bonds: atom_layout.GatherInfo
    # Gather indices to convert from cropped dense atom layout to bonds layout
    # (num_tokens, 2)
    token_atoms_to_bonds: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            tokens_to_polymer_ligand_bonds=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_polymer_ligand_bonds'
            ),
            token_atoms_to_bonds=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='token_atoms_to_polymer_ligand_bonds'
            ),
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.tokens_to_polymer_ligand_bonds.as_dict(
                key_prefix='tokens_to_polymer_ligand_bonds'
            ),
            **self.token_atoms_to_bonds.as_dict(
                key_prefix='token_atoms_to_polymer_ligand_bonds'
            ),
        }


jax.tree_util.register_dataclass(
    PolymerLigandBondInfo,
    data_fields=[f.name for f in dataclasses.fields(PolymerLigandBondInfo)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class LigandLigandBondInfo:
    """Contains information about the location of ligand-ligand bonds."""

    tokens_to_ligand_ligand_bonds: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            tokens_to_ligand_ligand_bonds=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_ligand_ligand_bonds'
            )
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.tokens_to_ligand_ligand_bonds.as_dict(
                key_prefix='tokens_to_ligand_ligand_bonds'
            )
        }


jax.tree_util.register_dataclass(
    LigandLigandBondInfo,
    data_fields=[f.name for f in dataclasses.fields(LigandLigandBondInfo)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class PseudoBetaInfo:
    """Contains information for extracting pseudo-beta and equivalent atoms."""

    token_atoms_to_pseudo_beta: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            token_atoms_to_pseudo_beta=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='token_atoms_to_pseudo_beta'
            ),
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.token_atoms_to_pseudo_beta.as_dict(
                key_prefix='token_atoms_to_pseudo_beta'
            ),
        }


jax.tree_util.register_dataclass(
    PseudoBetaInfo,
    data_fields=[f.name for f in dataclasses.fields(PseudoBetaInfo)],
    meta_fields=[],
)


def random_rotation(random_state: np.random.RandomState) -> np.ndarray:
    # Create a random rotation (Gram-Schmidt orthogonalization of two
    # random normal vectors)
    v0, v1 = random_state.normal(size=(2, 3))
    e0 = v0 / np.maximum(1e-10, np.linalg.norm(v0))
    v1 = v1 - e0 * np.dot(v1, e0)
    e1 = v1 / np.maximum(1e-10, np.linalg.norm(v1))
    e2 = np.cross(e0, e1)
    return np.stack([e0, e1, e2])


def random_augmentation(
        positions: np.ndarray,
        random_state: np.random.RandomState,
) -> np.ndarray:
    """Center then apply random translation and rotation."""

    center = np.mean(positions, axis=0)
    rot = random_rotation(random_state)
    positions_target = np.einsum('ij,kj->ki', rot, positions - center)

    translation = random_state.normal(size=(3,))
    positions_target = positions_target + translation
    return positions_target


def _get_reference_positions_from_ccd_cif(
        ccd_cif: cif_dict.CifDict,
        ref_max_modified_date: datetime.date,
        logging_name: str,
) -> np.ndarray:
    """Creates reference positions from a CCD mmcif data block."""
    num_atoms = len(ccd_cif['_chem_comp_atom.atom_id'])
    if '_chem_comp_atom.pdbx_model_Cartn_x_ideal' in ccd_cif:
        atom_x = ccd_cif['_chem_comp_atom.pdbx_model_Cartn_x_ideal']
        atom_y = ccd_cif['_chem_comp_atom.pdbx_model_Cartn_y_ideal']
        atom_z = ccd_cif['_chem_comp_atom.pdbx_model_Cartn_z_ideal']
    else:
        atom_x = np.array(['?'] * num_atoms)
        atom_y = np.array(['?'] * num_atoms)
        atom_z = np.array(['?'] * num_atoms)
    pos = np.array([[x, y, z] for x, y, z in zip(atom_x, atom_y, atom_z)])
    # Unknown reference coordinates are specified by '?' in chem comp dict.
    # Replace unknown reference coords with 0.
    if '?' in pos and '_chem_comp.pdbx_modified_date' in ccd_cif:
        # Use reference coordinates if modifed date is before cutoff.
        modified_dates = [
            datetime.date.fromisoformat(date)
            for date in ccd_cif['_chem_comp.pdbx_modified_date']
        ]
        max_modified_date = max(modified_dates)
        if max_modified_date < ref_max_modified_date:
            atom_x = ccd_cif['_chem_comp_atom.model_Cartn_x']
            atom_y = ccd_cif['_chem_comp_atom.model_Cartn_y']
            atom_z = ccd_cif['_chem_comp_atom.model_Cartn_z']
            pos = np.array([[x, y, z] for x, y, z in zip(atom_x, atom_y, atom_z)])
    if '?' in pos:
        if np.all(pos == '?'):
            logging.warning('All ref positions unknown for: %s', logging_name)
        else:
            logging.warning('Some ref positions unknown for: %s', logging_name)
        pos[pos == '?'] = 0
    return np.array(pos, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class RefStructure:
    """Contains ref structure information."""

    # Array with positions, float32, shape [num_res, max_atoms_per_token, 3]
    positions: xnp_ndarray
    # Array with masks, bool, shape [num_res, max_atoms_per_token]
    mask: xnp_ndarray
    # Array with elements, int32, shape [num_res, max_atoms_per_token]
    element: xnp_ndarray
    # Array with charges, float32, shape [num_res, max_atoms_per_token]
    charge: xnp_ndarray
    # Array with atom name characters, int32, [num_res, max_atoms_per_token, 4]
    atom_name_chars: xnp_ndarray
    # Array with reference space uids, int32, [num_res, max_atoms_per_token]
    ref_space_uid: xnp_ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            positions=batch['ref_pos'],
            mask=batch['ref_mask'],
            element=batch['ref_element'],
            charge=batch['ref_charge'],
            atom_name_chars=batch['ref_atom_name_chars'],
            ref_space_uid=batch['ref_space_uid'],
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'ref_pos': self.positions,
            'ref_mask': self.mask,
            'ref_element': self.element,
            'ref_charge': self.charge,
            'ref_atom_name_chars': self.atom_name_chars,
            'ref_space_uid': self.ref_space_uid,
        }


jax.tree_util.register_dataclass(
    RefStructure,
    data_fields=[f.name for f in dataclasses.fields(RefStructure)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class ConvertModelOutput:
    """Contains atom layout info."""

    cleaned_struc: structure.Structure
    token_atoms_layout: atom_layout.AtomLayout
    flat_output_layout: atom_layout.AtomLayout
    empty_output_struc: structure.Structure
    polymer_ligand_bonds: atom_layout.AtomLayout
    ligand_ligand_bonds: atom_layout.AtomLayout

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        """Construct atom layout object from dictionary."""

        return cls(
            cleaned_struc=_unwrap(batch.get('cleaned_struc', None)),
            token_atoms_layout=_unwrap(batch.get('token_atoms_layout', None)),
            flat_output_layout=_unwrap(batch.get('flat_output_layout', None)),
            empty_output_struc=_unwrap(batch.get('empty_output_struc', None)),
            polymer_ligand_bonds=_unwrap(batch.get('polymer_ligand_bonds', None)),
            ligand_ligand_bonds=_unwrap(batch.get('ligand_ligand_bonds', None)),
        )

    def as_data_dict(self) -> BatchDict:
        return {
            'cleaned_struc': np.array(self.cleaned_struc, object),
            'token_atoms_layout': np.array(self.token_atoms_layout, object),
            'flat_output_layout': np.array(self.flat_output_layout, object),
            'empty_output_struc': np.array(self.empty_output_struc, object),
            'polymer_ligand_bonds': np.array(self.polymer_ligand_bonds, object),
            'ligand_ligand_bonds': np.array(self.ligand_ligand_bonds, object),
        }


jax.tree_util.register_dataclass(
    ConvertModelOutput,
    data_fields=[f.name for f in dataclasses.fields(ConvertModelOutput)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class AtomCrossAtt:
    """Operate on flat atoms."""

    token_atoms_to_queries: atom_layout.GatherInfo
    tokens_to_queries: atom_layout.GatherInfo
    tokens_to_keys: atom_layout.GatherInfo
    queries_to_keys: atom_layout.GatherInfo
    queries_to_token_atoms: atom_layout.GatherInfo

    @classmethod
    def compute_features(
            cls,
            all_token_atoms_layout: atom_layout.AtomLayout,  # (num_tokens, num_dense)
            queries_subset_size: int,
            keys_subset_size: int,
            padding_shapes: PaddingShapes,
    ) -> Self:
        """Computes gather indices and meta data to work with a flat atom list."""

        token_atoms_layout = all_token_atoms_layout.copy_and_pad_to(
            (padding_shapes.num_tokens, all_token_atoms_layout.shape[1])
        )
        token_atoms_mask = token_atoms_layout.atom_name.astype(bool)
        flat_layout = token_atoms_layout[token_atoms_mask]
        num_atoms = flat_layout.shape[0]

        padded_flat_layout = flat_layout.copy_and_pad_to((
            padding_shapes.num_atoms,
        ))

        # Create the layout for queries
        num_subsets = padding_shapes.num_atoms // queries_subset_size
        lay_arr = padded_flat_layout.to_array()
        queries_layout = atom_layout.AtomLayout.from_array(
            lay_arr.reshape((6, num_subsets, queries_subset_size))
        )

        # Create the layout for the keys (the key subsets are centered around the
        # query subsets)
        # Create initial gather indices (contain out-of-bound indices)
        subset_centers = np.arange(
            queries_subset_size / 2, padding_shapes.num_atoms, queries_subset_size
        )
        flat_to_key_gathers = (
                subset_centers[:, None]
                + np.arange(-keys_subset_size / 2, keys_subset_size / 2)[None, :]
        )
        flat_to_key_gathers = flat_to_key_gathers.astype(int)
        # Shift subsets with out-of-bound indices, such that they are fully within
        # the bounds.
        for row in range(flat_to_key_gathers.shape[0]):
            if flat_to_key_gathers[row, 0] < 0:
                flat_to_key_gathers[row, :] -= flat_to_key_gathers[row, 0]
            elif flat_to_key_gathers[row, -1] > num_atoms - 1:
                overflow = flat_to_key_gathers[row, -1] - (num_atoms - 1)
                flat_to_key_gathers[row, :] -= overflow
        # Create the keys layout.
        keys_layout = padded_flat_layout[flat_to_key_gathers]

        # Create gather indices for conversion between token atoms layout,
        # queries layout and keys layout.
        token_atoms_to_queries = atom_layout.compute_gather_idxs(
            source_layout=token_atoms_layout, target_layout=queries_layout
        )

        token_atoms_to_keys = atom_layout.compute_gather_idxs(
            source_layout=token_atoms_layout, target_layout=keys_layout
        )

        queries_to_keys = atom_layout.compute_gather_idxs(
            source_layout=queries_layout, target_layout=keys_layout
        )

        queries_to_token_atoms = atom_layout.compute_gather_idxs(
            source_layout=queries_layout, target_layout=token_atoms_layout
        )

        # Create gather indices for conversion of tokens layout to
        # queries and keys layout
        token_idxs = np.arange(padding_shapes.num_tokens).astype(np.int64)
        token_idxs = np.broadcast_to(token_idxs[:, None], token_atoms_layout.shape)
        tokens_to_queries = atom_layout.GatherInfo(
            gather_idxs=atom_layout.convert(
                token_atoms_to_queries, token_idxs, layout_axes=(0, 1)
            ),
            gather_mask=atom_layout.convert(
                token_atoms_to_queries, token_atoms_mask, layout_axes=(0, 1)
            ),
            input_shape=np.array((padding_shapes.num_tokens,)),
        )

        tokens_to_keys = atom_layout.GatherInfo(
            gather_idxs=atom_layout.convert(
                token_atoms_to_keys, token_idxs, layout_axes=(0, 1)
            ),
            gather_mask=atom_layout.convert(
                token_atoms_to_keys, token_atoms_mask, layout_axes=(0, 1)
            ),
            input_shape=np.array((padding_shapes.num_tokens,)),
        )

        return cls(
            token_atoms_to_queries=token_atoms_to_queries,
            tokens_to_queries=tokens_to_queries,
            tokens_to_keys=tokens_to_keys,
            queries_to_keys=queries_to_keys,
            queries_to_token_atoms=queries_to_token_atoms,
        )

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(
            token_atoms_to_queries=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='token_atoms_to_queries'
            ),
            tokens_to_queries=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_queries'
            ),
            tokens_to_keys=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='tokens_to_keys'
            ),
            queries_to_keys=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='queries_to_keys'
            ),
            queries_to_token_atoms=atom_layout.GatherInfo.from_dict(
                batch, key_prefix='queries_to_token_atoms'
            ),
        )

    def as_data_dict(self) -> BatchDict:
        return {
            **self.token_atoms_to_queries.as_dict(
                key_prefix='token_atoms_to_queries'
            ),
            **self.tokens_to_queries.as_dict(key_prefix='tokens_to_queries'),
            **self.tokens_to_keys.as_dict(key_prefix='tokens_to_keys'),
            **self.queries_to_keys.as_dict(key_prefix='queries_to_keys'),
            **self.queries_to_token_atoms.as_dict(
                key_prefix='queries_to_token_atoms'
            ),
        }


jax.tree_util.register_dataclass(
    AtomCrossAtt,
    data_fields=[f.name for f in dataclasses.fields(AtomCrossAtt)],
    meta_fields=[],
)


@dataclasses.dataclass(frozen=True)
class Frames:
    """Features for backbone frames."""

    mask: xnp_ndarray

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(mask=batch['frames_mask'])

    def as_data_dict(self) -> BatchDict:
        return {'frames_mask': self.mask}


jax.tree_util.register_dataclass(
    Frames,
    data_fields=[f.name for f in dataclasses.fields(Frames)],
    meta_fields=[],
)
