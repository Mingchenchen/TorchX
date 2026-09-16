import torch
import torch.nn.functional as F

# Import required functions from the straight_through.py script in the same directory
from .straight_through import create_colabdesign_bias

# Critical: This amino acid order must exactly match the order implied by the PRO_STD_RESIDUES dictionary
# in scripts such as fill_sequences.py.
# The last dimension [seq_len, 20] of the model's logits tensor corresponds to this order.
# Index 0 = 'A', Index 1 = 'R', Index 2 = 'N', ...
PRO_STD_RESIDUES_ORDER = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", 
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]

# SReference amino acid frequencies from SAbDab or other sources.
# This is a standard Python dictionary; the order of its key-value pairs itself is irrelevant.
SABDAB_AA_FREQUENCIES = {
    'A': 0.06312278831489954, 
    'R': 0.04921230696338412,
    'N': 0.03918476731962273, 
    'D': 0.07701846340632819,
    'C': 0.0034091413849775686, 
    'Q': 0.006103880720768741,
    'E': 0.01636832052591836, 
    'G': 0.13745021395045826,
    'H': 0.011200935755637484, 
    'I': 0.05732243592590947,
    'L': 0.025799908201187462, 
    'K': 0.015298568234649609,
    'M': 0.01191533780482388, 
    'F': 0.07818445638816092,
    'P': 0.030115933016479367, 
    'S': 0.12155199218229468,
    'T': 0.07750706998919143, 
    'W': 0.020884229852381586,
    'Y': 0.12482047409644798, 
    'V': 0.03352877596647863
}

PPIFLOW_V1_AA_FREQUENCIES = {
    'A': 0.12323963576720771,
    'R': 0.049770940557660764,
    'N': 0.015327187376279623,
    'D': 0.03659295288728013,
    'C': 0.0,
    'Q': 0.014705050619308858,
    'E': 0.1800237543125389,
    'G': 0.033934732198405065,
    'H': 0.00509020982976076,
    'I': 0.06424975962898026,
    'L': 0.16045472541145864,
    'K': 0.12312651999321304,
    'M': 0.017363271308183922,
    'F': 0.01911656580510152,
    'P': 0.022170691702957976,
    'S': 0.032633900797466205,
    'T': 0.02471579661783836,
    'W': 0.002771336462869747,
    'Y': 0.01775917651716532,
    'V': 0.05695379220632317,
}

def _get_reference_distribution(
    name: str,
    device: torch.device,
    ) -> torch.Tensor:
    """
    Returns the SAbDab amino acid distribution as a PyTorch tensor.
    The order of this tensor strictly matches the internal residue index order of the model.

    Args:
        device: The torch device where the tensor is located (e.g., 'cpu' or 'npu').

    Returns:
        A torch.Tensor of shape [20] containing the reference frequencies.
    """
    if name == "sabdab":
        source_frequencies = SABDAB_AA_FREQUENCIES
    elif name == "ppiflow_v1":
        source_frequencies = PPIFLOW_V1_AA_FREQUENCIES
    else:
        raise ValueError(f"Unknown reference distribution name: '{name}'. Available options are 'sabdab' or 'ppiflow_v1'.")

    # Look up frequency values from the dictionary in the correct order defined by PRO_STD_RESIDUES_ORDER,
    # thereby constructing a frequency list with the right order to ensure the final tensor aligns with the model's logits.
    ordered_frequencies = [source_frequencies[aa] for aa in PRO_STD_RESIDUES_ORDER]
    
    # Create the tensor and move it to the specified device
    ref_dist = torch.tensor(ordered_frequencies, dtype=torch.float32, device=device)
    
    # Normalize to ensure the sum of all frequencies is 1, which is good practice.
    ref_dist = ref_dist / ref_dist.sum()
    
    return ref_dist


def _calculate_cross_entropy(
    predicted_distribution: torch.Tensor,
    reference_distribution: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the cross-entropy loss between the predicted amino acid distribution and the reference distribution.

    Args:
        predicted_distribution (torch.Tensor): Probability distribution predicted by the model for the designed sequence.
            Shape: [sequence_length, 20].
            The last dimension must be aligned with PRO_STD_RESIDUES_ORDER.
        reference_distribution (torch.Tensor): Target amino acid distribution.
            Shape: [20]. The correct order of this tensor has been ensured by the get_sabdab_distribution function.

    Returns:
        torch.Tensor: A scalar tensor representing the cross-entropy loss.
    """
    # 1. Aggregate the predicted distribution: take the mean along the sequence length dimension.
    # This represents the overall amino acid composition of the entire designed sequence.
    aggregated_predicted_dist = torch.mean(predicted_distribution, dim=0) # Shape becomes: [20]

    # 2. Add a small epsilon value to the predicted distribution to avoid NaN caused by log(0) during log calculation.
    epsilon = 1e-9
    aggregated_predicted_dist = aggregated_predicted_dist + epsilon
    
    # 3. Calculate cross-entropy: H(p, q) = -Σ [p(i) * log(q(i))]
    #    p = reference_distribution (ground truth distribution)
    #    q = aggregated_predicted_dist (distribution we want to optimize)
    # The element-wise calculation here is completely correct since the indices of both tensors are perfectly aligned.
    cross_entropy_loss = -torch.sum(
        reference_distribution * torch.log(aggregated_predicted_dist)
    )

    return cross_entropy_loss

# --- Public exposed main function ---
def compute_aa_type_loss(
    binder_logits_20: torch.Tensor,
    reference_aa_dist_name: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculates the cross-entropy loss for amino acid composition (AA Type).
    
    Assumes binder_logits_20 only contains residues to be designed (e.g., CDRs).
    Therefore, we directly compute the overall distribution without slicing.
    """
    
    # 1. Get reference distribution
    reference_aa_dist = _get_reference_distribution(
        name=reference_aa_dist_name,
        device=device
    )

    # 2. Prepare Logits
    # Create bias (mask out Cysteine)
    bias = create_colabdesign_bias(binder_logits_20.shape[0], rm_aa="C", device=device)
    
    # Center the logits for improved numerical stability
    # This is crucial to prevent vanishing or exploding gradients at initialization
    centered_logits = binder_logits_20 - binder_logits_20.mean(dim=-1, keepdim=True)
    
    # 3. Calculate predicted distribution
    # Force Softmax with T=1.0 to ensure smooth gradients
    temperature = 1.0 
    predicted_dist = F.softmax((centered_logits + bias) / temperature, dim=-1)

    # 4. Calculate loss
    # Compute the distribution using all input logits, as in Nanobody mode,
    # the binder_logits_20 has already been trimmed to contain only design sites.
    aa_type_loss = _calculate_cross_entropy(predicted_dist, reference_aa_dist)
    
    return aa_type_loss