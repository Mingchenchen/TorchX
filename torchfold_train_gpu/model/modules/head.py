"""torchfold.model.modules.head
"""

import torch
import torch.nn as nn

from torchfold.model import feat_batch  # noqa: F401


_CONTACT_THRESHOLD = 8.0
_CONTACT_EPSILON = 1e-3


class DistogramHead(nn.Module):
    """Distogram head predicting pairwise distance distributions.

    """

    def __init__(self,
                 c_pair: int = 128,
                 num_bins: int = 64,
                 first_break: float = 2.3125,
                 last_break: float = 21.6875) -> None:
        super(DistogramHead, self).__init__()

        self.c_pair = c_pair
        self.num_bins = num_bins
        self.first_break = first_break
        self.last_break = last_break

        self.half_logits = nn.Linear(self.c_pair, self.num_bins, bias=False)

        breaks = torch.linspace(
            self.first_break,
            self.last_break,
            self.num_bins - 1,
        )

        self.register_buffer('breaks', breaks.detach())

        bin_tops = torch.cat(
            (breaks, (breaks[-1] + (breaks[-1] - breaks[-2])).reshape(1)))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)

        self.register_buffer('is_contact_bin', is_contact_bin.detach())

    def forward(
        self,
        batch: "feat_batch.Batch",
        embeddings: dict,
    ) -> dict:
        """
        Args:
            batch: feat_batch.Batch containing token_features.mask
            embeddings (dict): must contain key 'pair' with pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            dict with keys:
                bin_edges: distogram bin break values
                logits: raw logits [*, N_token, N_token, num_bins]
                contact_probs: contact probability [*, N_token, N_token]
        """

        pair_act = embeddings['pair']
        seq_mask = batch.token_features.mask.to(dtype=torch.bool)
        pair_mask = seq_mask[:, None] * seq_mask[None, :]

        left_half_logits = self.half_logits(pair_act)
        right_half_logits = left_half_logits
        logits = left_half_logits + right_half_logits.transpose(-2, -3)
        probs = torch.softmax(logits, dim=-1)
        contact_probs = torch.einsum('...ijb,b->...ij', probs, self.is_contact_bin)

        contact_probs = pair_mask * contact_probs

        return {
            'bin_edges': self.breaks,
            'logits': logits,
            'contact_probs': contact_probs,
        }
