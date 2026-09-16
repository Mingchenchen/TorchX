import torch

from torchfold.constants import atom_types
from torchfold.constants import residue_names
from torchfold.constants import side_chains

NUM_DENSE = atom_types.DENSE_ATOM_NUM
NUM_AA = len(residue_names.PROTEIN_TYPES)
NUM_AA_WITH_UNK_AND_GAP = len(
    residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP
)
NUM_RESTYPES_WITH_UNK_AND_GAP = (
    residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
)


def _make_restype_rigidgroup_dense_atom_idx():
    """Create Mapping from rigid_groups to dense_atom indices."""
    # Create an array with the atom names.
    # shape (num_restypes, num_rigidgroups, 3_atoms):
    # (31, 8, 3)
    base_atom_indices = torch.zeros(
        (NUM_RESTYPES_WITH_UNK_AND_GAP, 8, 3), dtype=torch.int32
    )

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(
            residue_names.PROTEIN_TYPES_ONE_LETTER
    ):
        resname = residue_names.PROTEIN_COMMON_ONE_TO_THREE[restype_letter]

        dense_atom_names = atom_types.ATOM14[resname]
        # 0: backbone frame
        base_atom_indices[restype, 0, :] = torch.tensor([
            dense_atom_names.index(atom) for atom in ['C', 'CA', 'N']
        ], dtype=torch.int32)

        # 3: 'psi-group'
        base_atom_indices[restype, 3, :] = torch.tensor([
            dense_atom_names.index(atom) for atom in ['CA', 'C', 'O']
        ], dtype=torch.int32)

        for chi_idx in range(4):
            if side_chains.CHI_ANGLES_MASK[restype][chi_idx]:
                atom_names = side_chains.CHI_ANGLES_ATOMS[resname][chi_idx]
                base_atom_indices[restype, chi_idx + 4, :] = torch.tensor([
                    dense_atom_names.index(atom) for atom in atom_names[1:]
                ], dtype=torch.int32)
    dense_atom_names = atom_types.DENSE_ATOM['A']
    nucleic_rigid_atoms = torch.tensor([
        dense_atom_names.index(atom) for atom in ["C1'", "C3'", "C4'"]
    ], dtype=torch.int32)
    for nanum, _ in enumerate(residue_names.NUCLEIC_TYPES):
        # 0: backbone frame only.
        # we have aa + unk + gap, so we want to start after those
        resnum = nanum + NUM_AA_WITH_UNK_AND_GAP
        base_atom_indices[resnum, 0, :] = nucleic_rigid_atoms

    return base_atom_indices


RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX = _make_restype_rigidgroup_dense_atom_idx()


def _make_restype_pseudobeta_idx():
    """Returns indices of residue's pseudo-beta."""
    restype_pseudobeta_index = torch.zeros(
        (NUM_RESTYPES_WITH_UNK_AND_GAP,), dtype=torch.int32
    )
    for restype, restype_letter in enumerate(
            residue_names.PROTEIN_TYPES_ONE_LETTER
    ):
        restype_name = residue_names.PROTEIN_COMMON_ONE_TO_THREE[restype_letter]
        atom_names = list(atom_types.ATOM14[restype_name])
        if restype_name in {'GLY'}:
            restype_pseudobeta_index[restype] = atom_names.index('CA')
        else:
            restype_pseudobeta_index[restype] = atom_names.index('CB')
    for nanum, resname in enumerate(residue_names.NUCLEIC_TYPES):
        atom_names = list(atom_types.DENSE_ATOM[resname])
        # 0: backbone frame only.
        # we have aa + unk , so we want to start after those
        restype = nanum + NUM_AA_WITH_UNK_AND_GAP
        if resname in {'A', 'G', 'DA', 'DG'}:
            restype_pseudobeta_index[restype] = atom_names.index('C4')
        else:
            restype_pseudobeta_index[restype] = atom_names.index('C2')
    return restype_pseudobeta_index


RESTYPE_PSEUDOBETA_INDEX = _make_restype_pseudobeta_idx()


def _make_aatype_dense_atom_to_atom37():
    """Map from dense_atom to atom37 per residue type."""
    restype_dense_atom_to_atom37 = [
    ]  # mapping (restype, dense_atom) --> atom37
    for rt in residue_names.PROTEIN_TYPES_ONE_LETTER:
        atom_names = list(
            atom_types.ATOM14_PADDED[residue_names.PROTEIN_COMMON_ONE_TO_THREE[rt]]
        )
        atom_names.extend([''] * (NUM_DENSE - len(atom_names)))
        restype_dense_atom_to_atom37.append(
            [(atom_types.ATOM37_ORDER[name] if name else 0)
             for name in atom_names]
        )
    # Add dummy mapping for restype 'UNK', '-' (gap), and nucleics [but not DN].
    for _ in range(2 + len(residue_names.NUCLEIC_TYPES_WITH_UNKNOWN)):
        restype_dense_atom_to_atom37.append([0] * NUM_DENSE)

    restype_dense_atom_to_atom37 = torch.tensor(
        restype_dense_atom_to_atom37, dtype=torch.int32
    )
    return restype_dense_atom_to_atom37


PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37 = _make_aatype_dense_atom_to_atom37()


def identify_mol_type(
        ref_space_uid: torch.Tensor,
        atom_sums: torch.Tensor,
        chain_id: torch.Tensor,
        chain_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Generate mol_type masks based on the given rules.

    Args:
        ref_space_uid (torch.Tensor): A tensor of unique ids, shape (N,).
        atom_sums (torch.Tensor): A tensor of atom sums corresponding to each unique id, shape (N,).
        chain_id (torch.Tensor): A tensor of chain IDs corresponding to each unique id, shape (N,).
        chain_lengths (torch.Tensor): A tensor of chain lengths, shape (num_chains,).

    Returns:
        is_metal (torch.Tensor): A mask indicating metals.
        first_indices (torch.Tensor): A tensor of first indices for each unique id, shape (N,).
        last_indices (torch.Tensor): A tensor of last indices for each unique id, shape (N,).
    """

    assert (
            ref_space_uid.shape == atom_sums.shape
    ), "ref_space_uid and atom_sums must have the same shape."
    # Initialize masks
    is_metal = torch.zeros_like(ref_space_uid, dtype=torch.bool)
    first_indices = torch.zeros_like(ref_space_uid, dtype=torch.long)
    last_indices = torch.zeros_like(ref_space_uid, dtype=torch.long)

    # Count occurrences of each ref_space_uid
    unique_ids, counts = torch.unique(ref_space_uid, return_counts=True)
    for unique_id, count in zip(unique_ids, counts):
        mask = ref_space_uid == unique_id
        first_index = mask.nonzero(as_tuple=False)[0].item()
        last_index = mask.nonzero(as_tuple=False)[-1].item()
        first_indices[mask] = first_index
        last_indices[mask] = last_index
        atom_sum = atom_sums[mask]

        if count == 1 and chain_lengths[chain_id[mask].long()] == 1:
            is_metal[mask] = atom_sum == 1

    return (
        is_metal,
        first_indices,
        last_indices,
    )
