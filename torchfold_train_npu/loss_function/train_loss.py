#!/usr/bin/env python3
"""
Loss function module for training.
"""
import sys

from torchfold_loss_no_chunk import TorchfoldLossNoChunk, get_default_config

_loss_calculator = None

def configure_loss_fn(
    use_confidence: bool = True,
    use_interface_loss_weight: bool = False,
    interface_loss_weight: float = 5.0,
    interface_distance_threshold: float = 10.0,
    alpha_diffusion: float = None,
    alpha_distogram: float = None,
    alpha_confidence: float = None,
):
    """
    Configures the loss function calculator.

    Args:
        use_confidence (bool): Whether to include confidence loss.
        use_interface_loss_weight (bool): Whether to apply higher weights to interface residues.
        interface_loss_weight (float): Multiplier for interface residue loss (e.g. 5.0).
        interface_distance_threshold (float): CA-CA distance threshold (Å) for interface definition.
        alpha_diffusion (float): Override diffusion loss weight (None = use default 4.0).
        alpha_distogram (float): Override distogram loss weight (None = use default 3e-2).
        alpha_confidence (float): Override confidence loss weight (None = use default 1e-4).
    """
    global _loss_calculator
    config = get_default_config()
    config['use_smooth_lddt_loss'] = True
    config['use_bond_loss'] = True
    config['use_confidence_loss'] = use_confidence
    config['use_distogram_loss'] = True
    
    # Interface loss weighting
    config['use_interface_loss_weight'] = use_interface_loss_weight
    config['interface_loss_weight'] = interface_loss_weight
    config['interface_distance_threshold'] = interface_distance_threshold

    if alpha_diffusion is not None:
        config['alpha_diffusion'] = alpha_diffusion
    if alpha_distogram is not None:
        config['alpha_distogram'] = alpha_distogram
    if alpha_confidence is not None:
        config['alpha_confidence'] = alpha_confidence

    _loss_calculator = TorchfoldLossNoChunk(config)

def loss_fn(output, batch):
    """
    Calculates the loss for TorchFold.

    Args:
        output: The output from the TorchFold model.
        batch: The input batch.

    Returns:
        A tuple containing:
        - total_loss (torch.Tensor): The total computed loss.
        - loss_dict (dict): A dictionary of all individual loss components.
    """
    try:


        loss_dict = _loss_calculator(output, batch)
        total_loss = loss_dict['total_loss']
        return total_loss, loss_dict
    except Exception as e:
        print(f"\n✗ Error during loss calculation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Re-raise the exception to stop the training process
        raise e
