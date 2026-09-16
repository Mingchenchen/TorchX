import torch


def af3_has_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    is_polymer: torch.Tensor,
    threshold: float = 1.1,
) -> torch.Tensor:
    """AF3 'has_clash' per sample.

    pred_coordinate:  [N_sample, N_atom, 3] float
    asym_id:          [N_token] long (per-token chain id)
    atom_to_token_idx:[N_atom] long
    is_polymer:       [N_atom] bool (True=polymer atom)
    threshold:        AF3 clash distance threshold
    returns:          [N_sample] float tensor, 1.0 if the complex has an AF3 clash else 0.0
    """
    device = pred_coordinate.device
    N_sample = pred_coordinate.shape[0]

    asym_id = asym_id.long()
    atom_to_token_idx = atom_to_token_idx.long()

    # Per-atom chain id (broadcast token-level asym_id to atoms).
    atom_asym_id = asym_id[atom_to_token_idx]  # [N_atom]

    # Per-atom polymer flag as bool.
    is_polymer_atom = is_polymer.to(torch.bool)  # [N_atom]

    unique_chain_ids = torch.unique(atom_asym_id)

    polymer_chain_masks = []  # list of [N_atom] bool masks
    polymer_chain_sizes = []  # list of int atom counts
    for cid in unique_chain_ids.tolist():
        chain_mask = atom_asym_id == cid  # [N_atom]
        if not bool(is_polymer_atom[chain_mask].any()):
            # non-polymer ("lig") chain -> skipped in AF3 clash
            continue
        polymer_chain_masks.append(chain_mask)
        polymer_chain_sizes.append(int(chain_mask.sum().item()))

    has_clash = torch.zeros(N_sample, device=device, dtype=torch.float32)

    n_poly = len(polymer_chain_masks)
    if n_poly < 2:
        return has_clash

    for s in range(N_sample):
        coords = pred_coordinate[s]  # [N_atom, 3]
        clashed = False
        for i in range(n_poly):
            if clashed:
                break
            mask_i = polymer_chain_masks[i]
            n_i = polymer_chain_sizes[i]
            coords_i = coords[mask_i]  # [n_i, 3]
            for j in range(i + 1, n_poly):
                mask_j = polymer_chain_masks[j]
                n_j = polymer_chain_sizes[j]
                coords_j = coords[mask_j]  # [n_j, 3]

                pred_dist = torch.cdist(coords_i, coords_j)  # [n_i, n_j]
                total_clash = int((pred_dist < threshold).sum().item())
                relative_clash = total_clash / min(n_i, n_j)

                if total_clash > 100 or relative_clash > 0.5:
                    clashed = True
                    break
        if clashed:
            has_clash[s] = 1.0

    return has_clash
