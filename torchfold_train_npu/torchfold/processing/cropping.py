"""Process Structure Data."""

from typing import Optional

import numpy as np
import torch

from torchfold.processing.atom_layout.rigid_utils import identify_mol_type


def get_continues_crop_index(
        tokens: torch.Tensor,
        chain_id: torch.Tensor,
        ref_space_uid_token: torch.Tensor,
        atom_sums: torch.Tensor,
        crop_size: int,
        crop_complete_ligand_unstdRes: Optional[bool] = False,
        drop_last: Optional[bool] = False,
        remove_metal: Optional[bool] = False,
) -> torch.Tensor:
    """
    Crop sequences continuesly across chains. Reference: AF-multimer Algorithm 1.
    Args:
        tokens:    [all_token_length,], flatten tokens
        chain_id:  [all_token_length,], all tokens' chain ID within an assembly
        atom_sums: [all_token_length,] sum of atoms within one ref_space_uid
        ref_space_uid_token: [all_atom_length,] unique chain-residue id
        crop_size: total crop size of the whole assembly
        crop_complete_ligand_unstdRes: Whether to crop the complete ligand or unstandard residues.
                              If False, the ligand is usually fragmented during sequential cropping.
        drop_last: whether to ensure all ligands or unstandard residues to be cropped completely,
                    if not, we will ignore the completion of the last one to meet the crop_size quota.
        remove_metal: whether remove all metal/ions
    Returns:
        selected_token_indices: torch.Tensor, shape=(crop_size)
    """
    # get chain counts info
    unique_chain_id = torch.unique(chain_id)
    chain_lengths = torch.bincount(chain_id.long())
    chain_offset_list = torch.tensor(
        [torch.where(chain_id == chain_idx)[0][0] for chain_idx in unique_chain_id],
    )

    # identify the mol type
    (
        is_metal,
        uid_first_indices,
        uid_last_indices,
    ) = identify_mol_type(ref_space_uid_token, atom_sums, chain_id, chain_lengths)

    def _qualify_crop_size(cur_crop_size, crop_size_min, n_added):
        if cur_crop_size < crop_size_min:
            return False
        if cur_crop_size + n_added > crop_size:
            return False
        return True

    def _determine_start_end_point(start_idx, end_idx, crop_size_min, n_added):
        if start_idx == end_idx:
            return start_idx, end_idx

        # determine the start_idx
        left_start_point = right_start_point = start_idx
        # if this is not the first time this uid occurants, then it must be a middle point
        if uid_first_indices[start_idx] != start_idx:
            start_in_middle = True
            left_start_point = uid_first_indices[start_idx]
            right_start_point = uid_last_indices[start_idx] + 1
        else:
            start_in_middle = False

        # determine the end_idx
        left_end_point = right_end_point = end_idx
        # if this is not the last time this uid occurants, then it must be a middle point
        if end_idx > 0 and uid_last_indices[end_idx - 1] != end_idx - 1:
            end_in_middle = True
            left_end_point = uid_first_indices[end_idx - 1]
            right_end_point = uid_last_indices[end_idx - 1] + 1
        else:
            end_in_middle = False

        if start_in_middle is False and end_in_middle is False:
            return start_idx, end_idx
        elif start_in_middle is True and end_in_middle is True:
            # always use left edge
            start_in_middle = False
            start_idx = left_start_point

        if start_in_middle is False and end_in_middle is True:
            # need to determine: use left end or right end
            left_crop_size = left_end_point - start_idx
            right_crop_size = right_end_point - start_idx
            is_left_ok = _qualify_crop_size(left_crop_size, crop_size_min, n_added)
            is_right_ok = _qualify_crop_size(right_crop_size, crop_size_min, n_added)
            if is_left_ok and is_right_ok:
                end_idx = (
                    left_end_point
                    if torch.randint(low=0, high=2, size=(1,)).item() == 0
                    else right_end_point
                )
                return start_idx, end_idx
            elif is_left_ok:
                return start_idx, left_end_point
            elif is_right_ok:
                return start_idx, right_end_point
            elif drop_last is True:
                end_point = left_end_point
                while end_point - start_idx + n_added > crop_size:
                    if end_point > start_idx:
                        end_point = uid_first_indices[end_point - 1]
                    else:
                        break
                return start_idx, end_point
            else:
                cur_crop_size = min(end_idx - start_idx, crop_size - n_added)
                return start_idx, start_idx + cur_crop_size
        elif start_in_middle is True and end_in_middle is False:
            # need to determine: use left start or right start
            left_crop_size = end_idx - left_start_point
            right_crop_size = end_idx - right_start_point
            is_left_ok = _qualify_crop_size(left_crop_size, crop_size_min, n_added)
            is_right_ok = _qualify_crop_size(right_crop_size, crop_size_min, n_added)
            if is_left_ok and is_right_ok:
                start_idx = (
                    left_start_point
                    if torch.randint(low=0, high=2, size=(1,)).item() == 0
                    else right_start_point
                )
                return start_idx, end_idx
            elif is_left_ok:
                return left_start_point, end_idx
            elif is_right_ok:
                return right_start_point, end_idx
            elif drop_last is True:
                return right_start_point, end_idx
            else:
                return start_idx, end_idx

    # shuffle the list of chains
    chain_shuffle_index = torch.randperm(len(unique_chain_id))

    # crop over chains iteratively
    selected_token_indices = []
    n_added = 0  # number of tokens already selected
    n_remaining = len(tokens)  # number of tokens in remaining chains
    if remove_metal is True:
        n_remaining -= sum(is_metal).item()
    for idx in chain_shuffle_index:
        if n_added >= crop_size:
            break

        # get chain type: whether it is metal/ions
        curr_is_metal = is_metal[chain_offset_list[idx]]
        # whether remove metal chain
        if remove_metal is True and curr_is_metal:
            # skip if it is metal/ions
            continue

        chain_length = chain_lengths[unique_chain_id[idx].int()]
        n_remaining -= chain_length

        # determine the crop size
        crop_size_min = min(chain_length, max(0, crop_size - (n_added + n_remaining)))
        crop_size_max = min(crop_size - n_added, chain_length)
        if crop_size_min > crop_size_max:
            print(f"error crop_size: {crop_size_min} > {crop_size_max}")

        chain_crop_size = torch.randint(
            low=crop_size_min,
            high=crop_size_max + 1,
            size=(1,),
            device=tokens.device,
        ).item()

        chain_crop_start = torch.randint(
            low=0,
            high=chain_length - chain_crop_size + 1,
            size=(1,),
            device=tokens.device,
        ).item()

        chain_offset = chain_offset_list[idx]
        start_token_index = chain_offset + chain_crop_start
        end_token_index = chain_offset + chain_crop_start + chain_crop_size
        if crop_complete_ligand_unstdRes is True:
            start_token_index, end_token_index = _determine_start_end_point(
                start_token_index, end_token_index, crop_size_min, n_added
            )
            assert (
                    end_token_index >= start_token_index
            ), f"invalid crop indices!! {start_token_index}, {end_token_index}"
            chain_crop_size = end_token_index - start_token_index

        selected_token_indices.append(
            torch.arange(
                start_token_index,
                end_token_index,
            )
        )
        n_added += chain_crop_size
        if crop_complete_ligand_unstdRes is True and drop_last is True:
            if start_token_index < end_token_index:
                assert uid_first_indices[start_token_index] == start_token_index
                assert uid_last_indices[end_token_index - 1] == end_token_index - 1

    selected_token_indices = torch.concat(selected_token_indices).sort().values
    selected_token_indices = torch.flatten(selected_token_indices)
    if drop_last is True:
        assert (
                selected_token_indices.shape[0] <= crop_size
        ), f"Continuous cropping crop {selected_token_indices.shape[0]}, more than {crop_size} tokens!!"
    return selected_token_indices


def get_interface_token(
        chain_id: torch.Tensor,
        reference_chain_id: torch.Tensor,
        token_distance: torch.Tensor,
        token_distance_mask: torch.Tensor,
        interface_minimal_distance: int = 15,
) -> torch.Tensor:
    """
    Get tokens in contact with the other chain.
    Args:
        chain_id:           [all_token_length, ], chain ID of each token
        reference_chain_id: [1] or [2], the reference atom is selected within the reference chains
        token_distance:     [chain/interface_token_length, all_token_length], distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask:[chain/interface_token_length, all_token_length], indicates valid distance
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        interface_token_indices: indices of tokens of interface
    """
    # expand reference_chain_id to chain_id shape
    expand_reference_chain_id = torch.zeros(chain_id.size(), dtype=torch.int)
    for _chain_id in reference_chain_id:
        expand_reference_chain_id += chain_id == _chain_id

    # get distance mask, difference chain mask
    mask_distance = token_distance < interface_minimal_distance
    mask_diff_chain = (chain_id.unsqueeze(0) != chain_id.unsqueeze(1))[
        expand_reference_chain_id.nonzero(as_tuple=True)[0]
    ]

    mask = mask_distance * mask_diff_chain * token_distance_mask
    mask_interface = torch.sum(mask, dim=-1)
    interface_token_indices = torch.nonzero(mask_interface, as_tuple=True)[0]
    return interface_token_indices


def get_spatial_crop_index(
        tokens: torch.Tensor,
        chain_id: torch.Tensor,
        token_distance: torch.Tensor,
        token_distance_mask: torch.Tensor,
        reference_chain_id: torch.Tensor,
        ref_space_uid_token: torch.Tensor,
        crop_size: int,
        crop_complete_ligand_unstdRes: bool = False,
        interface_crop: bool = False,
        interface_minimal_distance: int = 15,
) -> torch.Tensor:
    """
    Crop sequences continuesly across chains.
    Args:
        tokens:   [all_token_length,], all tokens within an assembly
        chain_id: [all_token_length,], all tokens' chain ID within an assembly
        token_distance: [chain/interface_token_length, all_token_length], distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask: [chain/interface_token_length, all_token_length], indicates valid distance
        reference_chain_id:  [1] or [2],the reference atom is selected within the reference_chains ID
        crop_size: total crop size of the whole assembly
        interface_crop: whether use interface tokens as referenced token
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        selected_token_indices: torch.Tensor, shape=(min(crop_size, tokens.shape[0]), )
    """

    # interface spatial cropping: select reference tokens with contact to the other
    if interface_crop and interface_minimal_distance is not None:
        reference_token_indices = get_interface_token(
            chain_id=chain_id,
            reference_chain_id=reference_chain_id,
            token_distance=token_distance,
            token_distance_mask=token_distance_mask,
            interface_minimal_distance=interface_minimal_distance,
        )
        if len(reference_token_indices) < 1 and len(reference_chain_id) == 1:
            # If a chain does not contain any interfacial atoms, use all resolved tokens.
            reference_token_indices = torch.nonzero(
                token_distance_mask.bool().any(-1), as_tuple=True
            )[0]
    else:
        # select reference tokens within the given chain or interface
        reference_token_indices = torch.nonzero(
            token_distance_mask.bool().any(-1), as_tuple=True
        )[0]

    # random select one token from reference_token_indices
    assert len(reference_token_indices) > 0, "No resolved atoms in reference tokens!"

    random_idx = torch.randint(0, reference_token_indices.shape[0], (1,)).item()
    reference_token_idx = reference_token_indices[random_idx].item()

    assert (
        token_distance_mask[reference_token_idx].bool().any()
    ), "Select a unresolved reference token"
    distance_to_reference = token_distance[reference_token_idx]
    # add noise to break tie
    noise_break_tie = torch.arange(0, distance_to_reference.shape[0]).float() * 1e-3

    distance_to_reference_mask = token_distance_mask[reference_token_idx]
    distance_to_reference = torch.where(
        distance_to_reference_mask.bool(), distance_to_reference, torch.inf
    )

    # find k nearest tokens
    nearest_k = min(crop_size, tokens.shape[0])
    selected_token_indices = (
        torch.topk(distance_to_reference + noise_break_tie, nearest_k, largest=False)
        .indices.sort()
        .values
    )

    def drop_uncompleted_mol(selected_token_indices):
        selected_uid = ref_space_uid_token[selected_token_indices]
        mask = torch.ones_like(ref_space_uid_token, dtype=torch.bool)
        mask[selected_token_indices] = False
        unselected_uid = ref_space_uid_token[mask]

        # Find overlap elements
        overlap_uid = torch.Tensor(np.intersect1d(selected_uid, unselected_uid))

        # Remove overlap elements from elements_B
        remain_indices = selected_token_indices[
            ~torch.isin(selected_uid, overlap_uid)
        ].long()
        return remain_indices

    selected_token_indices = torch.flatten(selected_token_indices)
    if crop_complete_ligand_unstdRes is True:
        selected_token_indices = drop_uncompleted_mol(selected_token_indices)
    assert (
            selected_token_indices.shape[0] <= crop_size
    ), f"Spatial cropping crop {selected_token_indices.shape[0]}, more than {crop_size} tokens!!"
    return selected_token_indices, reference_token_idx
