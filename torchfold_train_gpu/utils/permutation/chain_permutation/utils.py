import torch

from torchfold.metrics.rmsd import align_pred_to_true


def get_optimal_transform(
    src_atoms: torch.Tensor,
    tgt_atoms: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor]:
    """
    A function that obtain the transformation that optimally align
    src_atoms to tgt_atoms.

    Args:
        src_atoms: ground-truth centre atom positions, shape: [N, 3]
        tgt_atoms: predicted centre atom positions, shape: [N, 3]
        mask: a vector of boolean values, shape: [N]

    Returns:
        tuple[torch.Tensor]: A rotation matrix that records the optimal rotation
                             that will best align src_atoms to tgt_atoms.
                             A tanslation matrix records how the atoms should be shifted after applying r.
    """
    assert src_atoms.shape == tgt_atoms.shape, (src_atoms.shape, tgt_atoms.shape)
    assert src_atoms.shape[-1] == 3
    if mask is not None:
        mask = mask.bool()
        assert mask.dim() == 1, "mask should have the shape of [N]"
        assert mask.shape[-1] == src_atoms.shape[-2]
        src_atoms = src_atoms[mask, :]
        tgt_atoms = tgt_atoms[mask, :]

    with torch.amp.autocast("cuda", enabled=False):
        _, rot, trans = align_pred_to_true(
            pred_pose=src_atoms.to(dtype=torch.float32),
            true_pose=tgt_atoms.to(dtype=torch.float32),
            allowing_reflection=False,
        )  # svd alignment does not support BF16

    return rot, trans


def apply_transform(pose, rot, trans):
    return torch.matmul(pose, rot.transpose(-1, -2)) + trans


def num_unique_matches(match_list: list[dict]):
    return len({tuple(sorted(match.items())) for match in match_list})
