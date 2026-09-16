#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fill empty sequences in JSON files

This script is used to modify input JSON files, generate sequences according to different strategies and update the JSON files.
"""

import json
import os

import torch

# Standard amino acid dictionary
PRO_STD_RESIDUES = {
    0: "A",   # Alanine
    1: "R",   # Arginine
    2: "N",   # Asparagine
    3: "D",   # Aspartic Acid
    4: "C",   # Cysteine
    5: "Q",   # Glutamine
    6: "E",   # Glutamic Acid
    7: "G",   # Glycine
    8: "H",   # Histidine
    9: "I",   # Isoleucine
    10: "L",  # Leucine
    11: "K",  # Lysine
    12: "M",  # Methionine
    13: "F",  # Phenylalanine
    14: "P",  # Proline
    15: "S",  # Serine
    16: "T",  # Threonine
    17: "W",  # Tryptophan
    18: "Y",  # Tyrosine
    19: "V",  # Valine
}

# Mapping from amino acids to indices
AA_TO_INDEX = {aa: idx for idx, aa in PRO_STD_RESIDUES.items()}


def generate_logits(strategy, design_length, ground_truth_sequence=None, device='cuda', seed=None, file_path=None):
    """
    Generate logits matrix according to different strategies
    
    Args:
        strategy: Generation strategy ('random_sequence', 'allA_sequence', 'Ground_truth_sequence')
        design_length: Length of the designed sequence
        ground_truth_sequence: Ground truth sequence (used when strategy is 'Ground_truth_sequence')
        device: Computing device
        seed: Random seed value (optional). If provided, random seed will be set for reproducibility.
              If not provided, current random state will be used (non-reproducible).
    
    Returns:
        logits: Amino acid logits matrix [design_length, 20]
        binder_logits_matrix: Logits matrix concatenated with zero matrix [design_length, 32]
    """
    # Set random seed for reproducibility if seed is provided
    if seed is not None:
        torch.manual_seed(seed)
    
    if strategy == 'random_sequence':
        # Fully random logits
        logits = torch.randn(design_length, 20, device=device)

    elif strategy == 'gumbel_sequence':
        _logits = torch.distributions.Gumbel(loc=0.0, scale=1.0).sample((design_length, 20)).to(device)
        logits = 2*0.1*torch.softmax(_logits, dim=-1)
    
    elif strategy == 'allA_sequence':
        # A corresponds to index 0, assign a higher logits value to it
        logits = torch.randn(design_length, 20, device=device)
        logits[:, 0] += 5.0  # Increase the probability of A
    
    elif strategy == 'Ground_truth_sequence':
        # Use the provided ground truth sequence
        if ground_truth_sequence is None or len(ground_truth_sequence) < design_length:
            raise ValueError("Ground truth sequence is required and must be at least as long as design_length")
        
        # Initialize logits matrix
        logits = torch.randn(design_length, 20, device=device)
        
        # Increase the probability of the true amino acids
        for i in range(design_length):
            if i < len(ground_truth_sequence):
                aa = ground_truth_sequence[i]
                if aa in AA_TO_INDEX:
                    aa_idx = AA_TO_INDEX[aa]
                    logits[i, aa_idx] += 5.0
    
    elif strategy == 'from_file':
        # New strategy: Load logits directly from file
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"File path '{file_path}' is required and must exist for 'from_file' strategy.")
        
        # Load the file directly using torch.load()
        # map_location=device ensures the tensor is loaded onto the specified device
        logits = torch.load(file_path, map_location=device)

        # (Optional) Check if the loaded logits shape matches the required dimensions
        if logits.shape[0] != design_length:
            raise ValueError(f"The length of logits from file ({logits.shape[0]}) does not match design_length ({design_length}).")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    
    
    return logits


def logits_to_sequence(logits):
    """
    Convert logits to amino acid sequence
    
    Args:
        logits: Amino acid logits matrix [design_length, 20]
    
    Returns:
        sequence: Amino acid sequence
        probs: Softmax probabilities [design_length, 20]
        binder_probs: Probability matrix concatenated with zero matrix [design_length, 32]
    """
    # Apply softmax
    probs = torch.softmax(logits, dim=1)
    # Select the amino acid index with the highest probability
    aa_indices = torch.argmax(probs, dim=1)
    
    # Convert to amino acid sequence
    sequence = ''.join([PRO_STD_RESIDUES[idx.item()] for idx in aa_indices])
    
   
    
    return sequence


def fill_empty_sequences(json_path, fill_with="AAA", target_index=-1):
    """
    Modify sequences in JSON file
    
    Args:
        json_path: Path to the JSON file
        fill_with: Sequence used for filling
        target_index: Index of the sequence to replace, -1 means the last one
    
    Returns:
        modified: Whether modification was performed
        modified_chains: List of modified chain IDs
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get target sequence item
    if target_index == -1:
        target_index = len(data["sequences"]) - 1
    
    if target_index < 0 or target_index >= len(data["sequences"]):
        print(f"Error: Index {target_index} out of range [0, {len(data['sequences'])-1}]")
        return False, []
    
    seq_item = data["sequences"][target_index]
    
    # Modify sequences
    modified = False
    modified_chains = []
    
    for chain_type, chain_data in seq_item.items():
        if "sequence" in chain_data:
            chain_data["sequence"] = fill_with
            modified = True
            modified_chains.append(chain_data['id'])
            print(f"Modified sequence for chain {chain_data['id']} to {fill_with}")
    
    # Save the modified file
    if modified:
        with open(json_path, 'wt') as f:
            json.dump(data, f, indent=2)
        print(f"Saved modified file: {json_path}")
    else:
        print("No sequences found to modify")
    
    return modified, modified_chains


def generate_sequence(strategy, design_length, ground_truth_sequence=None, device='cuda', seed=1, file_path=None):
    """
    Main function for sequence generation
    
    Args:
        strategy: Generation strategy
        design_length: Length of the designed sequence
        ground_truth_sequence: Ground truth sequence
        device: Computing device
        seed: Random seed value
    
    Returns:
        sequence: Generated sequence
        logits: Raw logits
        binder_logits_matrix: Concatenated logits matrix
        probs: Softmax probabilities
    """
    # Generate logits (pass seed to generate_logits for unified random seed setting)
    logits= generate_logits(
        strategy, design_length, ground_truth_sequence, device, seed=seed, file_path=file_path)
    
    # Convert to sequence
    sequence= logits_to_sequence(logits)
    
    return sequence, logits


def process_json_file(json_path, strategy='random_sequence', design_length=19, 
                      ground_truth_sequence=None, target_index=-1, device='cuda', seed=1, file_path=None):
    """
    Process JSON file, generate sequence and update
    
    Args:
        json_path: Path to the JSON file
        strategy: Generation strategy
        design_length: Length of the designed sequence
        ground_truth_sequence: Ground truth sequence
        target_index: Target sequence index
        device: Computing device
        seed: Random seed value
    
    Returns:
        sequence: Generated sequence
        logits: Raw logits (with computation history detached)
    """
    # Generate sequence
    sequence, logits = generate_sequence(
        strategy, design_length, ground_truth_sequence, device, seed, file_path=file_path)
    
    # Update JSON file
    fill_empty_sequences(json_path, fill_with=sequence, target_index=target_index)
    
    # Detach logits to keep only the initial values without computation history
    return sequence, logits.detach()




