#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update sequences in JSON files

This script is used to convert logits to amino acid sequences and update the TorchFold input JSON file.
"""

import json

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


def logits_to_sequence(logits):
    """
    Convert logits to amino acid sequence
    
    Args:
        logits: Amino acid logits matrix [N, 20]
    
    Returns:
        sequence: Amino acid sequence
    """
    # Apply softmax
    probs = torch.softmax(logits, dim=1)
    
    # Select the amino acid index with the highest probability
    aa_indices = torch.argmax(probs, dim=1)
    
    # Convert to amino acid sequence
    sequence = ''.join([PRO_STD_RESIDUES[idx.item()] for idx in aa_indices])
    
    return sequence


def update_sequences(json_path, binder_logits_20, target_index=-1, design_positions_in_chain=None):
    """
    Update sequences in the JSON file
    
    Args:
        json_path: Path to the JSON file
        binder_logits_20: Amino acid logits matrix [N, 20]
        target_index: Index of the sequence to replace, -1 for the last one
    
    Returns:
        sequence: Generated complete sequence (target chain)
    """
    # Convert logits to amino acid sequence (only for designed positions)
    designed_subseq = logits_to_sequence(binder_logits_20)
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get the target sequence entry
    if target_index == -1:
        target_index = len(data["sequences"]) - 1
    
    if target_index < 0 or target_index >= len(data["sequences"]):
        print(f"Error: Index {target_index} out of range [0, {len(data['sequences'])-1}]")
        return sequence
    
    seq_item = data["sequences"][target_index]

    # Currently only support a single protein chain as binder
    if not seq_item:
        print(f"Error: Sequence item at index {target_index} is empty")
        return ""

    chain_type, chain_data = next(iter(seq_item.items()))
    if "sequence" not in chain_data:
        print(f"Error: chain {chain_data.get('id', '?')} does not have 'sequence' field")
        return ""

    full_sequence = list(chain_data.get("sequence", ""))

    # Fall back to "full chain design" if design_positions_in_chain is not provided (backward compatibility)
    if design_positions_in_chain is None:
        sequence = designed_subseq
        chain_data["sequence"] = sequence
        print(f"Modified sequence of chain {chain_data['id']} to {sequence}")
        modified = True
    else:
        # Only write new amino acids at designed positions, keep others (e.g. framework) unchanged
        if len(design_positions_in_chain) != len(designed_subseq):
            print(
                f"Error: design_positions_in_chain length ({len(design_positions_in_chain)}) "
                f"and logits generated seq length ({len(designed_subseq)}) mismatch"
            )
            return "".join(full_sequence)

        for idx, pos in enumerate(design_positions_in_chain):
            if pos < 0 or pos >= len(full_sequence):
                print(f"Warning: Designed position {pos} exceeds current chain length {len(full_sequence)}, skipping")
                continue
            full_sequence[pos] = designed_subseq[idx]

        new_seq = "".join(full_sequence)
        chain_data["sequence"] = new_seq
        sequence = new_seq
        modified = True
        print(
            f"Updated sequence of chain {chain_data['id']} only at designed positions, length {len(full_sequence)} "
            f"(number of designed positions={len(design_positions_in_chain)})"
        )
    
    # Save the modified file
    if modified:
        with open(json_path, 'wt') as f:
            json.dump(data, f, indent=2)
        print(f"Saved modified file: {json_path}")
    else:
        print("No sequences found to modify")
    
    return sequence


