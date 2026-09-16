"""
IgLM Language Model Wrapper for Antibody Sequence Design
Adapted from the Germinal Project (https://github.com/Graylab/IgLM)

IgLM: Infilling Language Model for Antibody Sequence Design
Shuai, R. W., Ruffolo, J. A., & Gray, J. J. (2023)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from iglm import IgLM

# IgLM Special Token Definitions
MASK_TOKEN = '[MASK]'
SEP_TOKEN = '[SEP]'


class CustomIgLM(nn.Module, IgLM):
    """
    Wrapper for the IgLM model, used to compute language model gradients for antibody sequences.
    Uses the Straight-Through Estimator (STE) to make discrete choices differentiable.
    
    Key Features:
    1. Evaluate the "naturalness" (log-likelihood) of antibody sequences
    2. Compute gradients to guide sequence optimization towards more natural antibody sequences
    """

    def __init__(
        self,
        model_name: str = "IgLM",
        chain_token: str = "[HEAVY]",
        iglm_species: str = "[HUMAN]",
        ablm_temp: float = 1.0,
        is_scfv: bool = False,
        vh_first: bool = True,
        vh_len: Optional[int] = None,
        vl_len: Optional[int] = None,
        device: torch.device = None,
        seed: int = 0,
    ):
        """
        Initialize the CustomIgLM model.

        Args:
            model_name: IgLM model name (e.g. "IgLM" or "IgLM-S")
            chain_token: Antibody chain type token
                - "[HEAVY]": Heavy chain (VH)
                - "[LIGHT]": Light chain (VL)
                - Use "[HEAVY]" for VHH/nanobody
            iglm_species: Species token
                - "[HUMAN]": Human
                - "[MOUSE]": Mouse
                - "[CAMEL]": Camel (used for VHH)
                - Others: "[RABBIT]", "[RHESUS]", "[RAT]"
            ablm_temp: Softmax temperature parameter, controls the sharpness of the probability distribution
                - Lower temperature (< 1.0): Sharper distribution, more confident selections
                - Higher temperature (> 1.0): Smoother distribution, more exploration
            is_scfv: Whether it is scFv (single-chain antibody containing both VH and VL)
            vh_first: Whether the heavy chain comes first in scFv (only valid when is_scfv=True)
            vh_len: Length of the heavy chain (required in scFv mode)
            vl_len: Length of the light chain (required in scFv mode)
            device: Computation device (cuda/npu/cpu)
            seed: Random seed
        """
        super().__init__()
        IgLM.__init__(self, model_name=model_name)

        if device is not None:
            self.device = device
        self.model.to(self.device)

        # Freeze IgLM model parameters, only used for inference and gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

        self.chain_token = chain_token
        self.species_token = iglm_species
        self.is_scfv = is_scfv
        self.vh_first = vh_first
        self.vh_len = vh_len
        self.vl_len = vl_len

        self.tau = ablm_temp

        # 20 standard amino acids (order matches that in TorchCraft)
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

        # Get amino acid IDs from the IgLM tokenizer
        aa_ids = []
        for aa in self.amino_acids:
            tid = self.tokenizer.convert_tokens_to_ids(aa)
            if tid == self.tokenizer.unk_token_id:
                raise ValueError(f"Unrecognized amino acid token: {aa}")
            aa_ids.append(tid)
        self.amino_acid_ids = torch.tensor(aa_ids, device=self.device)

        # Get IDs for special tokens
        self.chain_id = self.tokenizer.convert_tokens_to_ids(chain_token)
        self.species_id = self.tokenizer.convert_tokens_to_ids(self.species_token)
        self.suffix_id = self.tokenizer.sep_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.cls_id = self.tokenizer.cls_token_id

        # Set random seed
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)

    def forward(self, seq_logits: torch.Tensor, chain: str) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Forward pass to compute the language model loss for the sequence.

        Uses Straight-Through Estimator (STE):
        - Forward pass: Uses hard (argmax) selection, IgLM sees discrete sequences
        - Backward pass: Gradients propagate through soft (softmax) probabilities

        Args:
            seq_logits: Sequence logits, shape = (seq_length, 20)
            chain: Chain type token, e.g., "[HEAVY]" or "[LIGHT]"

        Returns:
            ce_loss: Cross-entropy loss (scalar tensor)
            log_likelihood: Log-likelihood of the sequence (float)
            position_losses: Loss for each position, shape = (seq_length,)
        """
        # Step 1: Logits -> Soft Probabilities -> Hard One-Hot
        soft_probs = F.softmax(seq_logits / self.tau, dim=-1)  # (L, 20)
        hard_indices = soft_probs.argmax(dim=-1)  # (L,)
        hard_one_hot = F.one_hot(hard_indices, num_classes=soft_probs.size(-1)).float()  # (L, 20)

        # Step 2: Straight-Through Estimator
        # Forward pass: ste_probs = hard_one_hot (since soft_probs - soft_probs.detach() = 0)
        # Backward pass: Gradients flow through soft_probs
        ste_probs = hard_one_hot + (soft_probs - soft_probs.detach())

        # Step 3: Compute sequence embeddings
        final_probs = ste_probs
        embed_layer = self.model.get_input_embeddings()
        amino_embeds = embed_layer(self.amino_acid_ids)  # (20, embed_dim)
        var_embeds = final_probs @ amino_embeds  # (L, embed_dim)

        # Step 4: Construct the full input: [CHAIN] [SPECIES] [SEQUENCE...] [SEP]
        chain_id = self.tokenizer.convert_tokens_to_ids(chain)
        prefix_ids = torch.tensor([chain_id, self.species_id], device=self.device)
        prefix_embeds = embed_layer(prefix_ids)  # (2, embed_dim)

        suffix_ids = torch.tensor([self.suffix_id], device=self.device)
        suffix_embeds = embed_layer(suffix_ids)  # (1, embed_dim)

        # Concatenate: [prefix] + [sequence] + [suffix]
        full_embeds = torch.cat([prefix_embeds, var_embeds, suffix_embeds], dim=0).unsqueeze(0)
        
        # Step 5: IgLM forward pass
        outputs = self.model(inputs_embeds=full_embeds)
        logits = outputs.logits  # (1, total_length, vocab_size)

        # Step 6: Construct target labels
        var_token_ids = self.amino_acid_ids[hard_indices]
        full_target_ids = torch.cat([prefix_ids, var_token_ids, suffix_ids], dim=0).unsqueeze(0)

        # Step 7: Compute autoregressive Cross-Entropy Loss
        # Predict token at position i+1 given positions 0~i
        shift_logits = logits[:, :-1, :]  # Predicted logits
        shift_labels = full_target_ids[:, 1:]  # Ground truth labels
        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        )

        # Only compute loss for the sequence region (remove prefix and suffix)
        position_losses = ce_loss.reshape(shift_labels.shape)
        position_losses = position_losses[:, 1:-1]  # Remove loss from chain_token and sep_token

        # Compute average loss
        total_loss = position_losses.mean()
        log_likelihood = -total_loss.item()

        return total_loss, log_likelihood, position_losses.squeeze(0)

    def forward_with_probs(
        self, 
        input_probs: torch.Tensor, 
        target_probs: torch.Tensor, 
        chain: str
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Forward pass using two probability distributions (New version: KL/Dist loss mode).
        input_probs: [L, 20] predicted distribution (carries gradients)
        target_probs: [L, 20] target probabilities (usually pseudo_sequence)
        """
        # 1. Dimension validation and reshape (handle optional batch dimension [B, L, 20])
        if input_probs.dim() == 3:
            B, L, D = input_probs.shape
            input_probs = input_probs.view(-1, D)
            target_probs = target_probs.view(-1, D)
        elif input_probs.dim() != 2:
            raise ValueError(f"Expected input_probs shape [L, 20] or [B, L, 20], got {input_probs.shape}")

        embed_layer = self.model.get_input_embeddings()
        amino_embeds = embed_layer(self.amino_acid_ids)
        
        # 2. Take argmax on target_probs to get discrete IDs (used as IgLM context)
        hard_indices = target_probs.argmax(dim=-1)
        
        # 3. Extract IgLM expected distribution
        with torch.no_grad():
            # Generate inputs_embeds using discrete sequence IDs
            var_embeds = amino_embeds[hard_indices] # [L, D]
            L = var_embeds.shape[0]
            
            chain_id = self.tokenizer.convert_tokens_to_ids(chain)
            prefix_ids = torch.tensor([chain_id, self.species_id], device=self.device)
            prefix_embeds = embed_layer(prefix_ids)
            suffix_ids = torch.tensor([self.suffix_id], device=self.device)
            suffix_embeds = embed_layer(suffix_ids)

            full_embeds = torch.cat([prefix_embeds, var_embeds, suffix_embeds], dim=0).unsqueeze(0)
            outputs = self.model(inputs_embeds=full_embeds)
            logits = outputs.logits # [1, Seq, Vocab]

            # Fix slicing logic:
            # full_target_ids: [prefix(2), var(L), suffix(1)]
            # logits[0, i] predicts token at position i+1
            # We need to predict the var segment (positions 2 to L+1)
            # Therefore i+1 ∈ [2, L+1] => i ∈ [1, L]
            var_logits = logits[0, 1:1+L, :] # [L, Vocab]
            iglm_expected_logits = var_logits[:, self.amino_acid_ids] # [L, 20]
            iglm_expected_dist = torch.softmax(iglm_expected_logits, dim=-1) # [L, 20]

        # 4. Compute Cross-Entropy: -sum(P_iglm * log(P_binder))
        eps = 1e-10
        # input_probs should already be a softmax-normalized distribution
        # Issues may occur if not; but caller ensures input_probs = softmax(logits)
        position_losses = -(iglm_expected_dist * torch.log(input_probs + eps)).sum(dim=-1)
        
        total_loss = position_losses.mean()
        log_likelihood = -total_loss.item()

        return total_loss, log_likelihood, position_losses

    def forward_with_infill(
        self,
        seq_logits: torch.Tensor,
        chain: str,
        infill_range: Tuple[int, int],
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute language model loss for specified CDR regions using infilling mode.
        [CHAIN] [SPECIES] [LEFT_SEQUENCE] [MASK] [RIGHT_SEQUENCE] [SEP] [CDR_SEQUENCE] [CLS]
        """
        infill_start, infill_end = infill_range
        
        soft_probs = F.softmax(seq_logits / self.tau, dim=-1)
        hard_indices = soft_probs.argmax(dim=-1)
        hard_one_hot = F.one_hot(hard_indices, num_classes=soft_probs.size(-1)).float()
        ste_probs = hard_one_hot + (soft_probs - soft_probs.detach())
        
        embed_layer = self.model.get_input_embeddings()
        amino_embeds = embed_layer(self.amino_acid_ids)
        full_seq_embeds = ste_probs @ amino_embeds
        
        chain_id = self.tokenizer.convert_tokens_to_ids(chain)
        prefix_ids = torch.tensor([chain_id, self.species_id], device=self.device)
        prefix_embeds = embed_layer(prefix_ids)
        
        left_embeds = full_seq_embeds[:infill_start]
        mask_ids = torch.tensor([self.mask_id], device=self.device)
        mask_embeds = embed_layer(mask_ids)
        right_embeds = full_seq_embeds[infill_end:]
        sep_ids = torch.tensor([self.suffix_id], device=self.device)
        sep_embeds = embed_layer(sep_ids)
        cdr_embeds = full_seq_embeds[infill_start:infill_end]
        cls_ids = torch.tensor([self.cls_id], device=self.device)
        cls_embeds = embed_layer(cls_ids)
        
        full_embeds = torch.cat([
            prefix_embeds, left_embeds, mask_embeds, right_embeds, sep_embeds, cdr_embeds, cls_embeds
        ], dim=0).unsqueeze(0)
        
        outputs = self.model(inputs_embeds=full_embeds)
        logits = outputs.logits
        
        left_token_ids = self.amino_acid_ids[hard_indices[:infill_start]]
        right_token_ids = self.amino_acid_ids[hard_indices[infill_end:]]
        cdr_token_ids = self.amino_acid_ids[hard_indices[infill_start:infill_end]]
        
        full_target_ids = torch.cat([
            prefix_ids, left_token_ids, mask_ids, right_token_ids, sep_ids, cdr_token_ids, cls_ids
        ], dim=0).unsqueeze(0)
        
        total_len = seq_logits.shape[0]
        sep_position = 2 + infill_start + 1 + (total_len - infill_end)
        
        shift_logits = logits[:, sep_position:-1, :]
        shift_labels = full_target_ids[:, sep_position + 1:].contiguous().long()
        
        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='mean'
        )
        log_likelihood = -ce_loss.item()
        return ce_loss, log_likelihood

    def forward_with_infill_probs(
        self,
        input_probs: torch.Tensor,
        target_probs: torch.Tensor,
        chain: str,
        infill_range: Tuple[int, int],
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform Infilling calculation using two probability distributions (new version: KL/Dist loss mode).
        input_probs: [Full_L, 20] predicted distribution (with gradients)
        target_probs: [Full_L, 20] target probabilities (usually pseudo_sequence)
        """
        # 1. Dimension check and reshape
        if input_probs.dim() == 3:
            B, L, D = input_probs.shape
            input_probs = input_probs.view(-1, D)
            target_probs = target_probs.view(-1, D)
        elif input_probs.dim() != 2:
            raise ValueError(f"Expected input_probs shape [L, 20] or [B, L, 20], got {input_probs.shape}")

        infill_start, infill_end = infill_range
        embed_layer = self.model.get_input_embeddings()
        amino_embeds = embed_layer(self.amino_acid_ids)
        
        # 2. Take argmax on target_probs to obtain discrete IDs (used as IGLM context)
        hard_indices = target_probs.argmax(dim=-1)
        
        # 3. Extract the expected distribution for IGLM
        with torch.no_grad():
            # Generate inputs_embeds using discrete sequence IDs
            full_seq_embeds = amino_embeds[hard_indices]
            
            chain_id = self.tokenizer.convert_tokens_to_ids(chain)
            prefix_ids = torch.tensor([chain_id, self.species_id], device=self.device)
            prefix_embeds = embed_layer(prefix_ids)
            
            left_embeds = full_seq_embeds[:infill_start]
            mask_ids = torch.tensor([self.mask_id], device=self.device)
            mask_embeds = embed_layer(mask_ids)
            right_embeds = full_seq_embeds[infill_end:]
            sep_ids = torch.tensor([self.suffix_id], device=self.device)
            sep_embeds = embed_layer(sep_ids)
            cdr_embeds = full_seq_embeds[infill_start:infill_end]
            cls_ids = torch.tensor([self.cls_id], device=self.device)
            cls_embeds = embed_layer(cls_ids)
            
            full_embeds = torch.cat([
                prefix_embeds, left_embeds, mask_embeds, right_embeds, sep_embeds, cdr_embeds, cls_embeds
            ], dim=0).unsqueeze(0)
            
            outputs = self.model(inputs_embeds=full_embeds)
            logits = outputs.logits # [1, Seq, Vocab]
            
            total_len = input_probs.shape[0]
            L_infill = infill_end - infill_start
            sep_position = 2 + infill_start + 1 + (total_len - infill_end)
            
            # Fix slicing logic:
            # End part of full_embeds: [SEP] [cdr] [CLS]
            # sep_position is the index of [SEP].
            # cdr positions: [sep_position + 1, sep_position + L_infill]
            # To predict cdr, use logits[0, i], where i+1 ∈ [sep_position + 1, sep_position + L_infill]
            # i.e. i ∈ [sep_position, sep_position + L_infill - 1]
            shift_logits = logits[0, sep_position : sep_position + L_infill, :] # [L_infill, Vocab]
            iglm_expected_logits = shift_logits[:, self.amino_acid_ids] # [L_infill, 20]
            iglm_expected_dist = torch.softmax(iglm_expected_logits, dim=-1) # [L_infill, 20]
        
        # 4. Calculate Cross-Entropy
        eps = 1e-10
        binder_probs = input_probs[infill_start:infill_end] # [L_infill, 20]
        
        # Check dimension matching
        if binder_probs.shape != iglm_expected_dist.shape:
            raise ValueError(f"Shape mismatch: binder_probs {binder_probs.shape} vs iglm_expected_dist {iglm_expected_dist.shape}")

        ce_loss = -(iglm_expected_dist * torch.log(binder_probs + eps)).sum(dim=-1).mean()
        log_likelihood = -ce_loss.item()
        
        return ce_loss, log_likelihood

    def _get_iglm_grad(self, seq_logits: torch.Tensor, chain: str = '[HEAVY]') -> Tuple[torch.Tensor, float]:
        """
        Compute IgLM gradient.

        Args:
            seq_logits: Sequence logits, shape = (seq_length, 20), requires_grad=True
            chain: Chain type marker

        Returns:
            grad: Gradient, shape = (seq_length, 20)
            log_likelihood: Log-likelihood of the sequence
        """
        ce_loss, ll = self.forward(seq_logits, chain=chain)
        # Compute gradient with respect to seq_logits
        full_grad = torch.autograd.grad(ce_loss, seq_logits)[0]
        return full_grad.detach(), ll

    def get_iglm_loss_and_grad(self, seq_logits) -> Tuple[np.ndarray, float]:
        """
        Compute IgLM loss and gradient (main interface).

        Args:
            seq_logits: Sequence logits, can be:
                - numpy array, shape = (seq_length, 20)
                - torch tensor, shape = (seq_length, 20)

        Returns:
            iglm_grad: Gradient, numpy array, shape = (seq_length, 20)
            log_likelihood: the sequence log-likelihood (float)
        """
        # Convert input to torch tensor
        if isinstance(seq_logits, np.ndarray):
            current_logits = torch.tensor(seq_logits, device=self.device, requires_grad=True)
        else:
            current_logits = seq_logits.clone().detach().to(self.device).requires_grad_(True)

        if self.is_scfv:
            # scFv mode: Compute gradients for heavy and light chains separately
            if self.vh_first:
                current_logits_h = current_logits[:self.vh_len, :]
                current_logits_l = current_logits[-self.vl_len:, :]
            else:
                current_logits_l = current_logits[:self.vl_len, :]
                current_logits_h = current_logits[-self.vh_len:, :]

            # Compute gradients for heavy and light chains separately
            iglm_grad_h, ll_h = self._get_iglm_grad(current_logits_h, chain='[HEAVY]')
            iglm_grad_l, ll_l = self._get_iglm_grad(current_logits_l, chain='[LIGHT]')

            # Calculate the length of the linker region (gradient is zero)
            linker_len = current_logits.shape[0] - current_logits_h.shape[0] - current_logits_l.shape[0]
            
            # Concatenate gradients: [VH gradient] + [linker zero gradient] + [VL gradient]
            iglm_grad = torch.cat([
                iglm_grad_h,
                torch.zeros((linker_len, 20), device=self.device),
                iglm_grad_l
            ], dim=0)

            ll = ll_h + ll_l
        else:
            # Single-chain mode (VHH, VH, VL)
            iglm_grad, ll = self._get_iglm_grad(current_logits, chain=self.chain_token)

        return iglm_grad.cpu().numpy(), ll

    def get_iglm_loss_and_grad_with_context(
        self, 
        design_logits,
        framework_sequence: str,
        design_positions: list,
        total_length: int
    ) -> Tuple[np.ndarray, float]:
        """
        Compute IgLM loss and gradient using full sequence context (for nanobody mode).
        
        This method combines the framework sequence and design site logits into a full sequence,
        then passes it to IgLM for calculation. IgLM can see the complete antibody sequence context,
        which more accurately evaluates the "naturalness" of the CDR regions.
        
        Only the gradients for the design positions are returned, matching the shape of input design_logits.

        Args:
            design_logits: Logits for design positions, shape = (num_design_positions, 20)
                Can be numpy array or torch tensor
            framework_sequence: Amino acid sequence of the framework region (in positional order)
            design_positions: List of position indices of design sites in the full sequence
            total_length: Length of the full sequence

        Returns:
            iglm_grad: Gradients for design positions, shape = (num_design_positions, 20)
            log_likelihood: the full sequence log-likelihood (float)
        """
        # Convert design position logits to torch tensor
        if isinstance(design_logits, np.ndarray):
            design_logits_tensor = torch.tensor(
                design_logits, device=self.device, dtype=torch.float32, requires_grad=True
            )
        else:
            design_logits_tensor = design_logits.clone().detach().to(self.device).requires_grad_(True)

        # Construct logits for the full sequence
        full_logits = torch.zeros((total_length, 20), device=self.device)
        
        # Calculate framework positions (non-design positions)
        design_positions_set = set(design_positions)
        framework_positions = sorted([i for i in range(total_length) if i not in design_positions_set])
        
        # Verify length matching
        if len(framework_positions) != len(framework_sequence):
            raise ValueError(
                f"Framework positions number ({len(framework_positions)}) and "
                f"framework sequence length ({len(framework_sequence)}) mismatch"
            )
        
        # Amino acid to index mapping
        aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        
        # Fill logits for framework positions (use large values to represent one-hot)
        for i, pos in enumerate(framework_positions):
            aa = framework_sequence[i]
            if aa in aa_to_idx:
                full_logits[pos, aa_to_idx[aa]] = 100.0  # Large value makes softmax close to 1
            # Unknown amino acids remain 0 (uniform distribution)
        
        # Fill logits for design positions (maintain gradient connection)
        for i, pos in enumerate(design_positions):
            full_logits[pos] = design_logits_tensor[i]
        
        # Calculate IgLM loss
        ce_loss, ll, _ = self.forward(full_logits, chain=self.chain_token)
        
        # Compute gradients (only for design_logits_tensor)
        grad = torch.autograd.grad(ce_loss, design_logits_tensor)[0]
        
        return grad.detach().cpu().numpy(), ll


    def get_iglm_loss_with_context(
        self, 
        design_logits: torch.Tensor,
        framework_sequence: str,
        design_positions: list,
        total_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IgLM loss using full sequence context (differentiable version).
        
        Args:
            design_logits: Logits for design positions, shape = (num_design_positions, 20)
            framework_sequence: Amino acid sequence of the framework region
            design_positions: List of position indices of design sites in the full sequence
            total_length: Length of the full sequence

        Returns:
            position_losses: Loss for each position in the full sequence, shape = (total_length,)
            ll: full sequence log-likelihood (float)
        """
        # Construct logits for the full sequence
        full_logits = torch.zeros((total_length, 20), device=self.device)
        
        # Calculate framework positions
        design_positions_set = set(design_positions)
        framework_positions = sorted([i for i in range(total_length) if i not in design_positions_set])
        
        # Amino acid to index mapping
        aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        
        # Fill logits for framework positions
        for i, pos in enumerate(framework_positions):
            aa = framework_sequence[i]
            if aa in aa_to_idx:
                full_logits[pos, aa_to_idx[aa]] = 100.0
        
        # Fill logits for design positions
        for i, pos in enumerate(design_positions):
            full_logits[pos] = design_logits[i]
        
        # Calculate IgLM loss
        _, ll, position_losses = self.forward(full_logits, chain=self.chain_token)
        
        return position_losses, ll


def normalize_iglm_grad(iglm_grad: np.ndarray, af3_grad: np.ndarray) -> np.ndarray:
    """
    Normalize the IgLM gradient to match the scale of the AF3 gradient.

    This is the method used in the Germinal paper to ensure both gradients
    have similar magnitudes when combined, preventing one from dominating the other.

    Args:
        iglm_grad: IgLM gradient, shape = (seq_length, 20)
        af3_grad: AF3 gradient, shape = (seq_length, 20)

    Returns:
        Normalized IgLM gradient, shape = (seq_length, 20)
    """
    af3_norm = np.linalg.norm(af3_grad)
    iglm_norm = np.linalg.norm(iglm_grad)
    scale_factor = af3_norm / (iglm_norm + 1e-7)
    return iglm_grad * scale_factor
