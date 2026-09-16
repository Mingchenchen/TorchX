#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics plotting utilities for TorchFold binder design

Plot the change curves of various metrics during training, including:
- Confidence metrics (pTM, ipTM, pLDDT)
- Prediction error (PDE)
- PAE loss (global, within binder, interface)
- Contact loss (within binder, interface)
- Helical loss
- Entropy loss
- Total loss
"""

import csv
import os
from typing import List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt

# Set matplotlib fonts and styles
matplotlib.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # Axis label font size
    'xtick.labelsize': 12,     # X-axis tick font size
    'ytick.labelsize': 12,     # Y-axis tick font size
    'legend.fontsize': 12,     # Legend font size
    'figure.titlesize': 18,    # Figure title font size
    'lines.linewidth': 2.5,    # Line width
    'grid.alpha': 0.3,         # Grid transparency
    'figure.dpi': 100,         # Display resolution
    'savefig.dpi': 300,        # Save resolution
    'savefig.bbox': 'tight'    # Compact save
})


def read_metrics_from_csv(csv_file_path: str,
                        turn_off_diffusion_confidence: bool = True) -> Dict[str, List[Any]]:
    """
    Read training metrics data from a CSV file
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary containing all metrics data
    """
    if turn_off_diffusion_confidence:
        metrics_data = {
            'epochs': [],
            'sequences': [],
            'binder_contact_loss_scores': [],
            'interface_contact_loss_scores': [],
            'helix_loss_scores': [],
            'negative_contact_loss_score': [],
            'negative_framework_contact_loss_score': [],
            'paratope_loss_scores': [],
            'paratope_cdr_loss_scores': [],
            'paratope_fw_loss_scores': [],
            'paratope_cdr_target_loss_scores': [],
            'aa_type_loss_scores': [],
            'iglm_ll_scores': [],
            'iglm_ll_cdr1_scores': [],
            'iglm_ll_cdr2_scores': [],
            'iglm_ll_cdr3_scores': [],
            'loss_scores': [],
            'stage_info': [],
            'lr_scale': [],
            'effective_lr': []
        }
    else:
        metrics_data = {
            'epochs': [],
            'sequences': [],
            'ptm_scores': [],
            'iptm_scores': [],
            'pde_scores': [],
            'plddt_scores': [],
            'pae_loss_scores': [],
            'binder_pae_loss_scores': [],
            'binder_target_interface_pae_loss_scores': [],
            'binder_contact_loss_scores': [],
            'interface_contact_loss_scores': [],
            'helix_loss_scores': [],
            'entropy_loss_scores': [], 
            'negative_contact_loss_score': [],
            'negative_framework_contact_loss_score': [],
            'paratope_loss_scores': [],
            'paratope_cdr_loss_scores': [],
            'paratope_fw_loss_scores': [],
            'paratope_cdr_target_loss_scores': [],
            'aa_type_loss_scores': [],
            'iglm_ll_scores': [],
            'iglm_ll_cdr1_scores': [],
            'iglm_ll_cdr2_scores': [],
            'iglm_ll_cdr3_scores': [],
            'loss_scores': [],
            'stage_info': [],
            'lr_scale': [],
            'effective_lr': []
        }
    
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        print(f"CSV header: {header}")
        print(f"Number of CSV columns: {len(header)}")
        
        for row in reader:
            if len(row) >= 25: # Added 3 new CDR columns
                try:
                    metrics_data['epochs'].append(int(row[0]))
                    metrics_data['sequences'].append(row[1])
                    metrics_data['ptm_scores'].append(float(row[2]))
                    metrics_data['iptm_scores'].append(float(row[3]))
                    metrics_data['pde_scores'].append(float(row[4]))
                    metrics_data['plddt_scores'].append(float(row[5]))
                    metrics_data['pae_loss_scores'].append(float(row[6]))
                    metrics_data['binder_pae_loss_scores'].append(float(row[7]))
                    metrics_data['binder_target_interface_pae_loss_scores'].append(float(row[8]))
                    metrics_data['binder_contact_loss_scores'].append(float(row[9]))
                    metrics_data['interface_contact_loss_scores'].append(float(row[10]))
                    metrics_data['helix_loss_scores'].append(float(row[11]))
                    metrics_data['entropy_loss_scores'].append(float(row[12]))
                    metrics_data['negative_contact_loss_score'].append(float(row[13]))
                    metrics_data['negative_framework_contact_loss_score'].append(float(row[14]))
                    metrics_data['paratope_loss_scores'].append(float(row[15]))
                    metrics_data['paratope_cdr_loss_scores'].append(float(row[16]))
                    metrics_data['paratope_fw_loss_scores'].append(float(row[17]))
                    metrics_data['paratope_cdr_target_loss_scores'].append(float(row[18]))
                    metrics_data['aa_type_loss_scores'].append(float(row[19]))
                    metrics_data['iglm_ll_scores'].append(float(row[20])) 
                    metrics_data['iglm_ll_cdr1_scores'].append(float(row[21]))
                    metrics_data['iglm_ll_cdr2_scores'].append(float(row[22]))
                    metrics_data['iglm_ll_cdr3_scores'].append(float(row[23]))
                    metrics_data['loss_scores'].append(float(row[24]))
                    
                    # Optional additional information
                    if len(row) >= 28:
                        metrics_data['stage_info'].append(row[25] if len(row) > 25 else '')
                        metrics_data['lr_scale'].append(float(row[26]) if len(row) > 26 else 0.0)
                        metrics_data['effective_lr'].append(float(row[27]) if len(row) > 27 else 0.0)
                    else:
                        metrics_data['stage_info'].append('')
                        metrics_data['lr_scale'].append(0.0)
                        metrics_data['effective_lr'].append(0.0)
                        
                except (ValueError, IndexError) as e:
                    print(f"Skipping invalid row: {row}, error: {e}")
                    continue
            elif len(row) >= 17: # Added 3 new CDR columns (turn_off_diffusion_confidence=True)
                try:
                    metrics_data['epochs'].append(int(row[0]))
                    metrics_data['sequences'].append(row[1])
                    metrics_data['binder_contact_loss_scores'].append(float(row[2]))
                    metrics_data['interface_contact_loss_scores'].append(float(row[3]))
                    metrics_data['helix_loss_scores'].append(float(row[4]))
                    metrics_data['negative_contact_loss_score'].append(float(row[5]))
                    metrics_data['negative_framework_contact_loss_score'].append(float(row[6]))
                    metrics_data['paratope_loss_scores'].append(float(row[7]))
                    metrics_data['paratope_cdr_loss_scores'].append(float(row[8]))
                    metrics_data['paratope_fw_loss_scores'].append(float(row[9]))
                    metrics_data['paratope_cdr_target_loss_scores'].append(float(row[10]))
                    metrics_data['aa_type_loss_scores'].append(float(row[11]))
                    metrics_data['iglm_ll_scores'].append(float(row[12]))
                    metrics_data['iglm_ll_cdr1_scores'].append(float(row[13]))
                    metrics_data['iglm_ll_cdr2_scores'].append(float(row[14]))
                    metrics_data['iglm_ll_cdr3_scores'].append(float(row[15]))
                    metrics_data['loss_scores'].append(float(row[16]))
                    
                    # Optional additional information
                    if len(row) >= 20:
                        metrics_data['stage_info'].append(row[17] if len(row) > 17 else '')
                        metrics_data['lr_scale'].append(float(row[18]) if len(row) > 18 else 0.0)
                        metrics_data['effective_lr'].append(float(row[19]) if len(row) > 19 else 0.0)
                    else:
                        metrics_data['stage_info'].append('')
                        metrics_data['lr_scale'].append(0.0)
                        metrics_data['effective_lr'].append(0.0)
                        
                except (ValueError, IndexError) as e:
                    print(f"Skipping invalid row: {row}, error: {e}")
                    continue
                
    print(f"Successfully read {len(metrics_data['epochs'])} rows of data")
    return metrics_data


def plot_design_metrics(csv_file_path: str, output_dir: str, 
                       include_new_pae_losses: bool = True,
                       figure_size: tuple = (16, 90),
                       turn_off_diffusion_confidence: bool = True) -> None:# Adjust height to fit 11 subplots

    """
    Plot metric change graphs during binder design training
    
    Args:
        csv_file_path: Path to the CSV file
        output_dir: Output directory
        include_new_pae_losses: Whether to include new PAE loss metrics
        figure_size: Figure size (width, height)
    """
    # Read data
    metrics_data = read_metrics_from_csv(csv_file_path,turn_off_diffusion_confidence)
    epochs = metrics_data['epochs']
    
    if not epochs:
        print("Warning: No valid data read")
        return

    plt.figure(figsize=figure_size)
    # Create figure - 6 subplots without diffusion, 13 subplots with diffusion enabled
    if turn_off_diffusion_confidence:
        num_subplots = 12
        
        # 1.Binder contact loss
        plt.subplot(num_subplots, 1, 1)
        plt.plot(epochs, metrics_data['binder_contact_loss_scores'], 'brown', 
                label='Binder Contact Loss', linewidth=2.5)
        plt.ylabel('Contact Loss', fontsize=14, fontweight='bold')
        plt.title('Binder Internal Contact Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 2. Interface contact loss
        plt.subplot(num_subplots, 1, 2)
        plt.plot(epochs, metrics_data['interface_contact_loss_scores'], 'teal', 
                label='Interface Contact Loss', linewidth=2.5)
        plt.ylabel('Contact Loss', fontsize=14, fontweight='bold')
        plt.title('Binder-Target Interface Contact Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 3. Helix Loss
        plt.subplot(num_subplots, 1, 3)
        plt.plot(epochs, metrics_data['helix_loss_scores'], 'cyan', label='Helix Loss', linewidth=2.5)
        plt.ylabel('Helix Loss', fontsize=14, fontweight='bold')
        plt.title('Secondary Structure (Helix) Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 4. Negative Binder Contact loss
        plt.subplot(num_subplots, 1, 4)  
        plt.plot(epochs, metrics_data['negative_contact_loss_score'], 'darkgreen',label='Negative Contact Loss', linewidth=2.5)  
        plt.ylabel('Negative Contact Loss', fontsize=14, fontweight='bold')  
        plt.title('Negative Binder Contact Loss', fontsize=16, fontweight='bold') 
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 5. Negative Framework Contact loss
        plt.subplot(num_subplots, 1, 5)
        plt.plot(epochs, metrics_data['negative_framework_contact_loss_score'], 'darkred',label='Negative Framework Contact Loss', linewidth=2.5)
        plt.ylabel('Negative Framework Contact Loss', fontsize=14, fontweight='bold')
        plt.title('Negative Framework Contact Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 6. Paratope Loss
        plt.subplot(num_subplots, 1, 6)
        plt.plot(epochs, metrics_data['paratope_loss_scores'], 'royalblue', label='Paratope Loss', linewidth=2.5)
        plt.ylabel('Paratope Loss', fontsize=14, fontweight='bold')
        plt.title('Paratope Loss (Germinal)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 7. Paratope CDR Loss
        plt.subplot(num_subplots, 1, 7)
        plt.plot(epochs, metrics_data['paratope_cdr_loss_scores'], 'dodgerblue', label='L_CDR (Minimize)', linewidth=2.5)
        plt.ylabel('L_CDR', fontsize=14, fontweight='bold')
        plt.title('Paratope CDR Loss (Encourage CDR Binding)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 8. Paratope FW Loss
        plt.subplot(num_subplots, 1, 8)
        plt.plot(epochs, metrics_data['paratope_fw_loss_scores'], 'crimson', label='L_Framework (Maximize)', linewidth=2.5)
        plt.ylabel('L_Framework', fontsize=14, fontweight='bold')
        plt.title('Paratope Framework Loss (Discourage FW Binding)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 9. Paratope CDR-Target Loss
        plt.subplot(num_subplots, 1, 9)
        plt.plot(epochs, metrics_data['paratope_cdr_target_loss_scores'], 'indigo', label='L_CDR_Target', linewidth=2.5)
        plt.ylabel('L_CDR_Target', fontsize=14, fontweight='bold')
        plt.title('Paratope CDR-Target Loss (CDR vs All Target)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 10. Amino Acid Type Loss
        plt.subplot(num_subplots, 1, 10)
        plt.plot(epochs, metrics_data['aa_type_loss_scores'], 'darkorange', label='Amino Acid Type Loss', linewidth=2.5)
        plt.ylabel('AA Type Loss', fontsize=14, fontweight='bold')
        plt.title('Amino Acid Type Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 11. IgLM Log-Likelihood
        plt.subplot(num_subplots, 1, 11)
        plt.plot(epochs, metrics_data['iglm_ll_scores'], 'olive', label='Full LL', linewidth=2.5)
        if 'iglm_ll_cdr1_scores' in metrics_data and metrics_data['iglm_ll_cdr1_scores']:
            plt.plot(epochs, metrics_data['iglm_ll_cdr1_scores'], 'r--', label='CDR1 LL', linewidth=1.5)
        if 'iglm_ll_cdr2_scores' in metrics_data and metrics_data['iglm_ll_cdr2_scores']:
            plt.plot(epochs, metrics_data['iglm_ll_cdr2_scores'], 'g--', label='CDR2 LL', linewidth=1.5)
        if 'iglm_ll_cdr3_scores' in metrics_data and metrics_data['iglm_ll_cdr3_scores']:
            plt.plot(epochs, metrics_data['iglm_ll_cdr3_scores'], 'b--', label='CDR3 LL', linewidth=1.5)
        plt.ylabel('Log-Likelihood', fontsize=14, fontweight='bold')
        plt.title('IgLM Log-Likelihood (Higher is Better)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 12. Total Loss
        plt.subplot(num_subplots, 1, 12)
        plt.plot(epochs, metrics_data['loss_scores'], 'black', label='Total Loss', linewidth=2.5)
        plt.ylabel('Total Loss', fontsize=14, fontweight='bold')
        plt.title('Total Training Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
    else:
        num_subplots = 19
        
        # 1. Confidence metrics (pTM, ipTM)
        plt.subplot(num_subplots, 1, 1)
        plt.plot(epochs, metrics_data['ptm_scores'], 'b-', label='binder pTM', linewidth=2.5)
        plt.plot(epochs, metrics_data['iptm_scores'], 'g-', label='ipTM', linewidth=2.5)
        plt.ylabel('Confidence Score', fontsize=14, fontweight='bold')
        plt.title('Protein Confidence Metrics vs Iterations', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 2. Binder pLDDT
        plt.subplot(num_subplots, 1, 2)
        plt.plot(epochs, metrics_data['plddt_scores'], 'orange', label='pLDDT', linewidth=2.5)
        plt.ylabel('pLDDT Score', fontsize=14, fontweight='bold')
        plt.title('Binder Local Confidence (pLDDT)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 3. Predicted Distance Error
        plt.subplot(num_subplots, 1, 3)
        plt.plot(epochs, metrics_data['pde_scores'], 'r-', label='PDE', linewidth=2.5)
        plt.ylabel('Distance Error (Å)', fontsize=14, fontweight='bold')
        plt.title('Predicted Distance Error', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 4. Global PAE Loss
        plt.subplot(num_subplots, 1, 4)
        plt.plot(epochs, metrics_data['pae_loss_scores'], 'purple', label='Global PAE Loss', linewidth=2.5)
        plt.ylabel('Global PAE Loss', fontsize=14, fontweight='bold')
        plt.title('Global Predicted Alignment Error Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 5. Binder Internal PAE Loss
        plt.subplot(num_subplots, 1, 5)
        plt.plot(epochs, metrics_data['binder_pae_loss_scores'], 'magenta', label='Binder PAE Loss', linewidth=2.5)
        plt.ylabel('Binder PAE Loss', fontsize=14, fontweight='bold')
        plt.title('Binder Internal PAE Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 6. Binder-Target Interface PAE Loss
        plt.subplot(num_subplots, 1, 6)
        plt.plot(epochs, metrics_data['binder_target_interface_pae_loss_scores'], 'darkviolet', 
                label='Interface PAE Loss', linewidth=2.5)
        plt.ylabel('Interface PAE Loss', fontsize=14, fontweight='bold')
        plt.title('Binder-Target Interface PAE Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 7. Binder Contact Loss
        plt.subplot(num_subplots, 1, 7)
        plt.plot(epochs, metrics_data['binder_contact_loss_scores'], 'brown', 
                label='Binder Contact Loss', linewidth=2.5)
        plt.ylabel('Contact Loss', fontsize=14, fontweight='bold')
        plt.title('Binder Internal Contact Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 8. Interface Contact Loss
        plt.subplot(num_subplots, 1, 8)
        plt.plot(epochs, metrics_data['interface_contact_loss_scores'], 'teal', 
                label='Interface Contact Loss', linewidth=2.5)
        plt.ylabel('Contact Loss', fontsize=14, fontweight='bold')
        plt.title('Binder-Target Interface Contact Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 9. Helix Loss
        plt.subplot(num_subplots, 1, 9)
        plt.plot(epochs, metrics_data['helix_loss_scores'], 'cyan', label='Helix Loss', linewidth=2.5)
        plt.ylabel('Helix Loss', fontsize=14, fontweight='bold')
        plt.title('Secondary Structure (Helix) Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 10. Entropy Loss
        plt.subplot(num_subplots, 1, 10)
        plt.plot(epochs, metrics_data['entropy_loss_scores'], 'gold', label='Entropy Loss', linewidth=2.5)
        plt.ylabel('Entropy Loss', fontsize=14, fontweight='bold')
        plt.title('Sequence Entropy Loss (All Stages)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 11. Negative Binder Contact loss
        plt.subplot(num_subplots, 1, 11)  
        plt.plot(epochs, metrics_data['negative_contact_loss_score'], 'darkgreen',label='Negative Contact Loss', linewidth=2.5)  
        plt.ylabel('Negative Contact Loss', fontsize=14, fontweight='bold')  
        plt.title('Negative Binder Contact Loss', fontsize=16, fontweight='bold') 
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 12. Negative Framework Contact loss
        plt.subplot(num_subplots, 1, 12)
        plt.plot(epochs, metrics_data['negative_framework_contact_loss_score'], 'darkred',label='Negative Framework Contact Loss', linewidth=2.5)
        plt.ylabel('Negative Framework Contact Loss', fontsize=14, fontweight='bold')
        plt.title('Negative Framework Contact Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 13. Paratope Loss
        plt.subplot(num_subplots, 1, 13)
        plt.plot(epochs, metrics_data['paratope_loss_scores'], 'royalblue', label='Paratope Loss', linewidth=2.5)
        plt.ylabel('Paratope Loss', fontsize=14, fontweight='bold')
        plt.title('Paratope Loss (Germinal)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 14. Paratope CDR Loss
        plt.subplot(num_subplots, 1, 14)
        plt.plot(epochs, metrics_data['paratope_cdr_loss_scores'], 'dodgerblue', label='L_CDR', linewidth=2.5)
        plt.ylabel('L_CDR', fontsize=14, fontweight='bold')
        plt.title('Paratope CDR Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 15. Paratope FW Loss
        plt.subplot(num_subplots, 1, 15)
        plt.plot(epochs, metrics_data['paratope_fw_loss_scores'], 'crimson', label='L_Framework', linewidth=2.5)
        plt.ylabel('L_Framework', fontsize=14, fontweight='bold')
        plt.title('Paratope Framework Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 16. Paratope CDR-Target Loss
        plt.subplot(num_subplots, 1, 16)
        plt.plot(epochs, metrics_data['paratope_cdr_target_loss_scores'], 'indigo', label='L_CDR_Target', linewidth=2.5)
        plt.ylabel('L_CDR_Target', fontsize=14, fontweight='bold')
        plt.title('Paratope CDR-Target Loss (CDR vs All Target)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        #17. Amino Acid Type Loss
        plt.subplot(num_subplots, 1, 17)
        plt.plot(epochs, metrics_data['aa_type_loss_scores'], 'darkorange', label='Amino Acid Type Loss', linewidth=2.5)
        plt.ylabel('AA Type Loss', fontsize=14, fontweight='bold')
        plt.title('Amino Acid Type Loss', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 18. IgLM Log-Likelihood
        plt.subplot(num_subplots, 1, 18)
        plt.plot(epochs, metrics_data['iglm_ll_scores'], 'olive', label='Full LL', linewidth=2.5)
        if 'iglm_ll_cdr1_scores' in metrics_data and metrics_data['iglm_ll_cdr1_scores']:
            plt.plot(epochs, metrics_data['iglm_ll_cdr1_scores'], 'r--', label='CDR1 LL', linewidth=1.5)
        if 'iglm_ll_cdr2_scores' in metrics_data and metrics_data['iglm_ll_cdr2_scores']:
            plt.plot(epochs, metrics_data['iglm_ll_cdr2_scores'], 'g--', label='CDR2 LL', linewidth=1.5)
        if 'iglm_ll_cdr3_scores' in metrics_data and metrics_data['iglm_ll_cdr3_scores']:
            plt.plot(epochs, metrics_data['iglm_ll_cdr3_scores'], 'b--', label='CDR3 LL', linewidth=1.5)
        plt.ylabel('Log-Likelihood', fontsize=14, fontweight='bold')
        plt.title('IgLM Log-Likelihood (Higher is Better)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 19. Total Loss
        plt.subplot(num_subplots, 1, 19)
        plt.plot(epochs, metrics_data['loss_scores'], 'black', label='Total Loss', linewidth=2.5)
        plt.ylabel('Total Loss', fontsize=14, fontweight='bold')
        plt.title('Total Training Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        
    output_path = os.path.join(output_dir, 'binder_design_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Training metric chart saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        csv_path = sys.argv[1]
        output_dir = sys.argv[2]
        plot_design_metrics(csv_path, output_dir)
    else:
        print("Usage: python plot_metrics.py <csv_file_path> <output_dir>")