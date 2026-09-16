#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Straight-Through Estimator for sequence design

Implement the pseudo-sequence mechanism similar to ColabDesign:
- Forward pass uses one-hot (discrete)
- Backward pass uses softmax (continuous)
"""

import torch
import torch.nn.functional as F


def compute_entropy_loss(pseudo_seq):
    """
    Calculate the entropy loss of the sequence, encouraging a more deterministic distribution (close to one-hot)

    Args:
        pseudo_seq: [seq_len, 20] Probability distribution after softmax

    Returns:
        entropy_loss: Scalar, the larger the entropy, the larger the loss
    """
    # Calculate entropy for each position: H = -sum(p * log(p))
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-8
    log_probs = torch.log(pseudo_seq + epsilon)
    entropy_per_position = -torch.sum(pseudo_seq * log_probs, dim=-1)  # [seq_len]

    # Return the average entropy as the loss
    entropy_loss = torch.mean(entropy_per_position)

    return entropy_loss, entropy_per_position

def exponential_anneal(step, total_steps, T0=1.0, T_min=0.01):
    """
    Exponential annealing function

    Args:
        step: Current step (0 to total_steps-1)
        total_steps: Total number of steps
        T0: Initial temperature
        T_min: Final temperature

    Returns:
        Current temperature value
    """
    if total_steps <= 1:
        return T_min

    # Calculate the decay factor so that the temperature drops to T_min after total_steps steps
    decay = (T_min / T0) ** (1 / (total_steps))
    current_temp = T0 * (decay ** step)

    # Ensure the temperature does not drop below T_min
    return max(current_temp, T_min)

def create_omit_C_bias(sequence_length, rm_aa="C", device='npu'):
    """
    Fully simulate the ColabDesign bias mechanism for masking amino acids
    """
    # Amino acid to index mapping (order used by ColabDesign)
    aa_order = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 
                'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 
                'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

    # Initialize bias matrix
    bias = torch.zeros(sequence_length, 20, device=device)

    # Mask the specified amino acids
    if rm_aa is not None:
        for aa in rm_aa.split(","):
            bias[:, aa_order[aa]] -= 1e6

    return bias

def create_colabdesign_bias(sequence_length, rm_aa="C", device='npu'):
    """
    Fully simulate the ColabDesign bias mechanism for masking amino acids
    """
    # Amino acid to index mapping (order used by ColabDesign)
    aa_order = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 
                'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 
                'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

    # Initialize bias matrix
    bias = torch.zeros(sequence_length, 20, device=device)

    # Mask the specified amino acids
    if rm_aa is not None:
        for aa in rm_aa.split(","):
            bias[:, aa_order[aa]] -= 1e6  # Fully simulate line 66 of ColabDesign

    return bias

def four_stage_sequence_optimization(logits, step, total_iterations, 
                                   stage_epochs=[50, 50, 50, 10],
                                   bias=None):
    """
    Four-stage sequence optimization using ColabDesign-style bias mechanism to mask cysteine
    Returns pseudo_seq, stage_info (contains learning rate scheduling information)
    """
    # 🔥 Create bias to mask cysteine if not provided
    if bias is None:
        bias = create_colabdesign_bias(logits.shape[0], rm_aa="C", device=logits.device)

    # 🔥 Define bias application function, fully simulate ColabDesign line 214
    def apply_bias_and_softmax(raw_logits, temp=1.0):
        logits_with_bias = raw_logits + bias  # Simulate ColabDesign: seq["logits"] = seq["input"] + bias
        return F.softmax(logits_with_bias / temp, dim=-1) # Original Softmax

    # Verify that total_iterations matches the sum of stage_epochs
    calculated_total = sum(stage_epochs)
    if total_iterations != calculated_total:
        print(f"Warning: total_iterations({total_iterations}) and stage_epochs calculated sum ({calculated_total}) mismatch, use stage_epochs")
        total_iterations = calculated_total

    # Calculate the boundaries for each stage
    pre_end = stage_epochs[0]
    stage1_end = pre_end + stage_epochs[1] 
    stage2_end = stage1_end + stage_epochs[2]
    stage3_end = stage2_end + stage_epochs[3]

    current_epoch = step

    if current_epoch < pre_end:
        # 🌟 Pre-design stage: apply bias
        stage_progress = current_epoch / pre_end if pre_end > 0 else 1.0
        soft = 0.0  # Use logits in Pre-design stage
        temp = 1.0

        stage_info = {
            "stage": "Pre-design",
            "soft": soft,
            "hard": 0.0, 
            "temp": temp,
            "step": 1.0,  # Step parameter, used for learning rate scheduling in ColabDesign
            "description": f"Global exploration with soft sequences (epoch {current_epoch}/{pre_end})"
        }

        if current_epoch == 0:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits + gumbel_noise * 0.1

        # Calculate softmax after applying bias
        pseudo_seq = apply_bias_and_softmax(logits, stage_info["temp"])

    elif current_epoch < stage1_end:
        # 🎯 Stage 1: Apply bias, simulate the soft transition of ColabDesign
        stage1_progress = (current_epoch - pre_end) / stage_epochs[1] if stage_epochs[1] > 0 else 1.0
        soft = stage1_progress  # Transition from 0 to 1
        temp = 1.0
        step = 1.0

        stage_info = {
            "stage": "Stage 1", 
            "soft": soft,
            "hard": 0.0,
            "temp": temp,
            "step": step,
            "description": f"Logits→Soft transition (epoch {current_epoch-pre_end}/{stage_epochs[1]}, soft={soft:.3f})"
        }

        # Calculate softmax after applying bias
        soft_seq = apply_bias_and_softmax(logits, stage_info["temp"])
        input_seq = logits  # Original logits (keep your hybrid logic)
        pseudo_seq = soft * soft_seq + (1 - soft) * input_seq

    elif current_epoch < stage2_end:
        # 🌡️ Stage 2: Temperature annealing, simulate ColabDesign temperature scheduling
        stage2_progress = (current_epoch - stage1_end) / stage_epochs[2] if stage_epochs[2] > 0 else 1.0
        soft = 1.0

        # Use exponential annealing: temperature decays exponentially from 1.0 to 0.01
        current_step = current_epoch - stage1_end  # Current step in Stage 2 (starts at 0)
        total_stage2_steps = stage_epochs[2]       # Total steps for Stage 2
        temp = exponential_anneal(current_step, total_stage2_steps, T0=1.0, T_min=0.01)

        step = 1.0

        stage_info = {
            "stage": "Stage 2",
            "soft": soft,
            "hard": 0.0,
            "temp": temp,
            "step": step,
            "description": f"Temperature annealing (epoch {current_epoch-stage1_end}/{stage_epochs[2]}, temp={temp:.4f})"
        }

        # Calculate softmax after applying bias
        pseudo_seq = apply_bias_and_softmax(logits, temp)

    else:
        # ⚡ Stage 3: Straight-Through optimization
        soft = 1.0
        hard = 1.0
        temp = 0.01
        step = 1.0

        stage_info = {
            "stage": "Stage 3",
            "soft": soft, 
            "hard": hard,
            "temp": temp,
            "step": step,
            "description": f"Straight-Through optimization (epoch {current_epoch-stage2_end}/{stage_epochs[3]})"
        }

        # Calculate softmax after applying bias
        soft_seq = apply_bias_and_softmax(logits, stage_info["temp"])

        # True hard sequence
        hard_indices = torch.argmax(soft_seq, dim=-1)
        hard_seq = F.one_hot(hard_indices, num_classes=logits.shape[-1]).float()

        # Straight-Through Estimator
        pseudo_seq = (hard_seq - soft_seq).detach() + soft_seq

        # Final blending
        temp_pseudo = soft * soft_seq + (1 - soft) * logits
        pseudo_seq = hard * pseudo_seq + (1 - hard) * temp_pseudo

    return pseudo_seq, stage_info  # Return only two values

# To maintain compatibility with PyTorch (since PyTorch does not have stop_gradient)
# We need to define a custom one
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_input, hard_input):
        """Return hard_input for forward pass, gradients flow to soft_input during backward pass"""
        return hard_input

    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradients directly to soft_input"""
        return grad_output, None


def torch_stop_gradient(tensor):
    """PyTorch implementation of stop_gradient"""
    return tensor.detach()