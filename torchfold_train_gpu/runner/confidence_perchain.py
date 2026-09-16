"""torchfold.runner.confidence_perchain
"""
from __future__ import annotations

import torch

from torchfold.runner.sample_confidence import (
    calculate_iptm,
    calculate_ptm,
    get_bin_centers,  # re-exported for callers; used by None of the helpers directly
)

__all__ = [
    "calculate_chain_based_ptm",
    "calculate_chain_based_plddt",
    "calculate_chain_pair_pae",
    "calculate_chain_based_gpde",
]


def _remap_asym_id_contiguous(asym_id: torch.Tensor) -> torch.Tensor:
    """Remap ``asym_id`` to contiguous 0..N_chain-1 if chains were filtered out.
    """
    asym_id = asym_id.long()
    unique_asym_ids = torch.unique(asym_id)
    if len(unique_asym_ids) != asym_id.max() + 1:
        remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
        asym_id = torch.tensor(
            [remap[x.item()] for x in asym_id],
            dtype=torch.long,
            device=asym_id.device,
        )
    return asym_id


def calculate_chain_based_ptm(
    pae_prob: torch.Tensor,
    has_frame: torch.BoolTensor,
    asym_id: torch.LongTensor,
    token_is_ligand: torch.BoolTensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
) -> dict[str, torch.Tensor]:
    """Compute chain-based pTM scores.

    Args:
        pae_prob (torch.Tensor): Predicted probability from PAE loss head.
            Shape: [..., N_token, N_token, N_bins]
        has_frame (torch.BoolTensor): Indicator for tokens having a frame.
            Shape: [N_token, ]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token, ]
        token_is_ligand (torch.BoolTensor): Indicator for tokens being ligands.
            Shape: [N_token, ]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.

    Returns:
        dict: Dictionary containing chain-based pTM scores.
            - chain_ptm (torch.Tensor): pTM scores for each chain.
            - chain_iptm (torch.Tensor): ipTM scores for chain interface.
            - chain_pair_iptm (torch.Tensor): Pairwise ipTM scores between chains.
            - chain_pair_iptm_global (torch.Tensor): Global pairwise ipTM scores
              between chains.
    """
    has_frame = has_frame.bool()
    asym_id = _remap_asym_id_contiguous(asym_id)
    asym_id_to_asym_mask = {
        aid.item(): asym_id == aid for aid in torch.unique(asym_id)
    }
    chain_is_ligand = {
        aid.item(): token_is_ligand[asym_id == aid].sum()
        >= (asym_id == aid).sum() // 2
        for aid in torch.unique(asym_id)
    }

    batch_shape = pae_prob.shape[:-3]

    # Chain_pair_iptm.
    # Dense tensor (not dict-of-dict) to avoid cross-device aggregation issues.
    N_chain = len(asym_id_to_asym_mask)
    chain_pair_iptm = torch.zeros(size=batch_shape + (N_chain, N_chain)).to(
        pae_prob.device
    )
    for aid_1 in range(N_chain):
        for aid_2 in range(N_chain):
            if aid_1 == aid_2:
                continue
            if aid_1 > aid_2:
                chain_pair_iptm[:, aid_1, aid_2] = chain_pair_iptm[
                    :, aid_2, aid_1
                ]
                continue
            pair_mask = (
                asym_id_to_asym_mask[aid_1] + asym_id_to_asym_mask[aid_2]
            )
            chain_pair_iptm[:, aid_1, aid_2] = calculate_iptm(
                pae_prob,
                has_frame,
                asym_id,
                min_bin,
                max_bin,
                no_bins,
                token_mask=pair_mask,
            )

    # chain_ptm
    chain_ptm = torch.zeros(size=batch_shape + (N_chain,)).to(pae_prob.device)
    for aid, asym_mask in asym_id_to_asym_mask.items():
        chain_ptm[:, aid] = calculate_ptm(
            pae_prob,
            has_frame,
            min_bin,
            max_bin,
            no_bins,
            token_mask=asym_mask,
        )

    # Chain iptm
    chain_has_frame = [
        (asym_id_to_asym_mask[i] * has_frame).any() for i in range(N_chain)
    ]

    chain_iptm = torch.zeros(size=batch_shape + (N_chain,)).to(pae_prob.device)
    for aid, asym_mask in asym_id_to_asym_mask.items():
        pairs = [
            (i, j)
            for i in range(N_chain)
            for j in range(N_chain)
            if (i == aid or j == aid) and (i != j) and chain_has_frame[i]
        ]
        vals = [chain_pair_iptm[:, i, j] for (i, j) in pairs]
        if len(vals) > 0:
            chain_iptm[:, aid] = torch.stack(vals, dim=-1).mean(dim=-1)

    # Chain_pair_iptm_global
    chain_pair_iptm_global = torch.zeros(
        size=batch_shape + (N_chain, N_chain)
    ).to(pae_prob.device)
    for aid_1 in range(N_chain):
        for aid_2 in range(N_chain):
            if aid_1 == aid_2:
                continue
            if chain_is_ligand[aid_1]:
                chain_pair_iptm_global[:, aid_1, aid_2] = chain_iptm[:, aid_1]
            elif chain_is_ligand[aid_2]:
                chain_pair_iptm_global[:, aid_1, aid_2] = chain_iptm[:, aid_2]
            else:
                chain_pair_iptm_global[:, aid_1, aid_2] = (
                    chain_iptm[:, aid_1] + chain_iptm[:, aid_2]
                ) * 0.5

    return {
        "chain_ptm": chain_ptm,
        "chain_iptm": chain_iptm,
        "chain_pair_iptm": chain_pair_iptm,
        "chain_pair_iptm_global": chain_pair_iptm_global,
    }


def calculate_chain_based_gpde(
    token_pair_pde: torch.Tensor,
    contact_probs: torch.Tensor,
    asym_id: torch.LongTensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Calculate chain-based gPDE values.

    Args:
        token_pair_pde (torch.Tensor): PDE (Predicted Distance Error) of
            token-token pairs. [..., N_token, N_token]
        contact_probs (torch.Tensor): Contact probabilities.
            [..., N_token, N_token]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing chain-based gPDE values.
            - chain_gpde (torch.Tensor): Intra-chain gPDE.
            - chain_pair_gpde (torch.Tensor): Interface gPDE.
    """
    asym_id = _remap_asym_id_contiguous(asym_id)
    N_chain = len(torch.unique(asym_id))

    batch_shape = token_pair_pde.shape[:-2]
    device = token_pair_pde.device

    def _cal_gpde(token_mask_1, token_mask_2):
        masked_contact_probs = contact_probs[..., token_mask_1, :][
            ..., token_mask_2
        ]
        masked_pde = token_pair_pde[..., token_mask_1, :][..., token_mask_2]
        return (masked_pde * masked_contact_probs).sum(dim=(-1, -2)) / (
            masked_contact_probs.sum(dim=(-1, -2)) + eps
        )

    # Chain_gpde
    chain_gpde = torch.zeros(size=batch_shape + (N_chain,), device=device)
    for aid in range(N_chain):
        chain_gpde[..., aid] = _cal_gpde(
            token_mask_1=asym_id == aid,
            token_mask_2=asym_id == aid,
        )

    # Chain_pair_gpde
    chain_pair_gpde = torch.zeros(
        size=batch_shape + (N_chain, N_chain), device=device
    )
    for aid_1 in range(N_chain):
        for aid_2 in range(N_chain):
            if aid_1 == aid_2:
                continue
            if aid_2 < aid_1:
                chain_pair_gpde[..., aid_1, aid_2] = chain_pair_gpde[
                    ..., aid_2, aid_1
                ]
                continue
            chain_pair_gpde[..., aid_1, aid_2] = _cal_gpde(
                token_mask_1=asym_id == aid_1,
                token_mask_2=asym_id == aid_2,
            )

    return {"chain_gpde": chain_gpde, "chain_pair_gpde": chain_pair_gpde}


def calculate_chain_pair_pae(
    token_pair_pae: torch.Tensor,
    asym_id: torch.LongTensor,
    token_has_frame: torch.BoolTensor,
    contact_probs: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Calculate chain-pair PAE values.

    Args:
        token_pair_pae (torch.Tensor): PAE (Predicted Aligned Error) of
            token-token pairs. [..., N_token, N_token]
        asym_id (torch.LongTensor): Asymmetric ID for tokens. [N_token]
        token_has_frame (torch.BoolTensor): Indicator for tokens having a frame.
            [N_token]
        contact_probs (torch.Tensor | None): Optional contact probabilities.
            [..., N_token, N_token]
        eps (float): Small value to avoid division by zero.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing chain-pair PAE values.
            - chain_pair_pae_mean (torch.Tensor): Mean PAE for chain pairs.
            - chain_pair_pae_min (torch.Tensor): Min PAE for chain pairs.

    """
    asym_id = _remap_asym_id_contiguous(asym_id)
    N_chain = len(torch.unique(asym_id))

    batch_shape = token_pair_pae.shape[:-2]
    device = token_pair_pae.device

    if contact_probs is None:
        contact_probs = torch.ones(token_pair_pae.shape[1:], dtype=float).to(
            device
        )

    mask = (
        token_has_frame[:, None] & token_has_frame[None, :]
    )  # [N_token, N_token]
    assert mask.shape == token_pair_pae.shape[1:]

    chain_pair_pae_mean = torch.zeros(
        size=batch_shape + (N_chain, N_chain), device=device
    )
    chain_pair_pae_min = torch.zeros(
        size=batch_shape + (N_chain, N_chain), device=device
    )

    for aid_1 in range(N_chain):
        mask_1 = asym_id == aid_1
        sub_pae = token_pair_pae[..., mask_1, :]
        sub_mask = mask[mask_1, :]
        sub_contact_probs = contact_probs[mask_1, :]
        for aid_2 in range(N_chain):
            mask_2 = asym_id == aid_2

            subsub_pae = sub_pae[..., mask_2]
            subsub_mask = sub_mask[..., mask_2]
            subsub_contact_probs = sub_contact_probs[..., mask_2]

            (flat_subsub_mask_idxs,) = torch.where(subsub_mask.flatten() > 0)
            flat_subsub_pae = subsub_pae.view(batch_shape[0], -1)
            flat_subsub_contact_probs = subsub_contact_probs.flatten()

            if not flat_subsub_mask_idxs.any():
                chain_pair_pae_mean[..., aid_1, aid_2] = torch.nan
                chain_pair_pae_min[..., aid_1, aid_2] = torch.nan
            else:
                valid_pae = flat_subsub_pae[:, flat_subsub_mask_idxs]
                valid_contact_probs = flat_subsub_contact_probs[
                    flat_subsub_mask_idxs
                ]

                # min
                chain_pair_pae_min[..., aid_1, aid_2] = valid_pae.min(
                    dim=-1
                ).values

                # weighted mean
                chain_pair_pae_mean[..., aid_1, aid_2] = (
                    valid_contact_probs * valid_pae
                ).mean(dim=-1) / (valid_contact_probs.mean(dim=-1) + eps)

    return {
        "chain_pair_pae_mean": chain_pair_pae_mean,
        "chain_pair_pae_min": chain_pair_pae_min,
    }


def calculate_chain_based_plddt(
    atom_plddt: torch.Tensor,
    asym_id: torch.LongTensor,
    atom_to_token_idx: torch.LongTensor,
) -> dict[str, torch.Tensor]:
    """Calculate chain-based pLDDT scores.

    Args:
        atom_plddt (torch.Tensor): Predicted pLDDT scores for atoms.
            Shape: [N_sample, N_atom]
        asym_id (torch.LongTensor): Asymmetric ID for tokens. Shape: [N_token]
        atom_to_token_idx (torch.LongTensor): Mapping from atoms to tokens.
            Shape: [N_atom]

    Returns:
        dict: Dictionary containing chain-based pLDDT scores.
            - chain_plddt (torch.Tensor): pLDDT scores for each chain.
            - chain_pair_plddt (torch.Tensor): Pairwise pLDDT scores between
              chains.
    """
    asym_id = _remap_asym_id_contiguous(asym_id)
    asym_id_to_asym_mask = {
        aid.item(): asym_id == aid for aid in torch.unique(asym_id)
    }
    N_chain = len(asym_id_to_asym_mask)
    assert N_chain == asym_id.max() + 1  # make sure it is from 0 to N_chain-1

    def _calculate_lddt_with_token_mask(token_mask):
        atom_mask = token_mask[atom_to_token_idx]
        sub_plddt = atom_plddt[:, atom_mask].mean(-1)
        return sub_plddt

    batch_shape = atom_plddt.shape[:-1]
    # Chain_plddt
    chain_plddt = torch.zeros(size=batch_shape + (N_chain,)).to(
        atom_plddt.device
    )
    for aid, asym_mask in asym_id_to_asym_mask.items():
        chain_plddt[:, aid] = _calculate_lddt_with_token_mask(
            token_mask=asym_mask
        )

    # Chain_pair_plddt
    chain_pair_plddt = torch.zeros(size=batch_shape + (N_chain, N_chain)).to(
        atom_plddt.device
    )
    for aid_1 in asym_id_to_asym_mask:
        for aid_2 in asym_id_to_asym_mask:
            if aid_1 == aid_2:
                continue
            pair_mask = (
                asym_id_to_asym_mask[aid_1] + asym_id_to_asym_mask[aid_2]
            )
            chain_pair_plddt[:, aid_1, aid_2] = _calculate_lddt_with_token_mask(
                token_mask=pair_mask
            )

    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}
