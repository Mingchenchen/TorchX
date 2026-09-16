#!/usr/bin/env python3
"""Minimal test script: load sample data and evaluate the loss.

Evaluates both variants:
1. torchfold_loss_no_chunk.py - baseline without chunking
2. torchfold_loss_with_chunk_utils.py - version using inner chunking
"""

import io
import pickle
import sys

import torch

from torchfold_loss_no_chunk import TorchfoldLossNoChunk, get_default_config
from torchfold_loss_with_chunk_utils import TorchfoldLossWithChunkUtils


def load_pickle_cpu(filepath):
    """Load NPU pickles while remapping tensors to CPU."""
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)
    
    with open(filepath, 'rb') as f:
        return CPUUnpickler(f).load()


# Load sample data
print("Loading data...")
output = load_pickle_cpu('output.pkl')
batch = load_pickle_cpu('batch.pkl')
print("✓ Data loaded successfully")

# Configure losses (start from defaults and enable everything)
config = get_default_config()
config['use_smooth_lddt_loss'] = True
config['use_bond_loss'] = True
config['use_confidence_loss'] = True
config['use_distogram_loss'] = True  # Enable distogram loss

# Test 1: no-chunk version
print("\n" + "=" * 80)
print("Test 1: torchfold_loss_no_chunk.py (no chunk)")
print("=" * 80)
try:
    loss_fn = TorchfoldLossNoChunk(config)
    print("✓ Loss function instantiated")
    
    with torch.no_grad():
        losses = loss_fn(output, batch)
    
    print("\n✓ Loss computed successfully")
    print("\nIndividual loss terms:")
    print(f"  total_loss:            {losses['total_loss'].item():.6f}")
    print(f"  ├─ smooth_lddt_loss:   {losses['smooth_lddt_loss'].item():.6f}")
    print(f"  ├─ bond_loss:          {losses['bond_loss'].item():.6f}")
    print(f"  ├─ mse_loss:           {losses['mse_loss'].item():.6f}")
    print(f"  ├─ plddt_loss:         {losses['plddt_loss'].item():.6f}")
    print(f"  ├─ pae_loss:           {losses['pae_loss'].item():.6f}")
    print(f"  ├─ pde_loss:           {losses['pde_loss'].item():.6f}")
    print(f"  ├─ resolved_loss:      {losses['resolved_loss'].item():.6f}")
    print(f"  └─ distogram_loss:     {losses['distogram_loss'].item():.6f}")
    
    losses_no_chunk = losses
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: chunked version
print("\n" + "=" * 80)
print("Test 2: torchfold_loss_with_chunk_utils.py (chunked)")
print("=" * 80)
try:
    config['chunk_size'] = 1  # Configure chunk size
    loss_fn = TorchfoldLossWithChunkUtils(config)
    print("✓ Loss function instantiated")
    
    with torch.no_grad():
        losses = loss_fn(output, batch)
    
    print("\n✓ Loss computed successfully")
    print("\nIndividual loss terms:")
    print(f"  total_loss:            {losses['total_loss'].item():.6f}")
    print(f"  ├─ smooth_lddt_loss:   {losses['smooth_lddt_loss'].item():.6f}")
    print(f"  ├─ bond_loss:          {losses['bond_loss'].item():.6f}")
    print(f"  ├─ mse_loss:           {losses['mse_loss'].item():.6f}")
    print(f"  ├─ plddt_loss:         {losses['plddt_loss'].item():.6f}")
    print(f"  ├─ pae_loss:           {losses['pae_loss'].item():.6f}")
    print(f"  ├─ pde_loss:           {losses['pde_loss'].item():.6f}")
    print(f"  ├─ resolved_loss:      {losses['resolved_loss'].item():.6f}")
    print(f"  └─ distogram_loss:     {losses['distogram_loss'].item():.6f}")
    
    losses_with_chunk = losses
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare the two versions
print("\n" + "=" * 80)
print("Consistency check")
print("=" * 80)
print(f"\n{'Loss name':25s} {'No chunk':>12s} {'Chunked':>12s} {'Diff':>12s}")
print("-" * 70)

all_match = True
test_keys = ['total_loss', 'smooth_lddt_loss', 'bond_loss', 'mse_loss', 
             'plddt_loss', 'pae_loss', 'pde_loss', 'resolved_loss', 'distogram_loss']

for key in test_keys:
    v1 = losses_no_chunk[key].item()
    v2 = losses_with_chunk[key].item()
    diff = abs(v1 - v2)
    match = diff < 1e-4
    
    status = "✅" if match else "❌"
    print(f"{key:25s} {v1:12.6f} {v2:12.6f} {diff:12.8f} {status}")
    
    if not match:
        all_match = False

print("\n" + "=" * 80)
if all_match:
    print("✅ All tests passed, both versions agree.")
else:
    print("⚠️ Differences detected between the two versions.")
print("=" * 80)
