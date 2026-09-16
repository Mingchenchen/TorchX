


import einops
import torch
import torch.nn as nn

from torchfold import fastnn


class GridSelfAttention(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttention, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])

        weighted_avg = fastnn.dot_product_attention(q, k, v,
                                                    mask=mask,
                                                    bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        gate_values = self.gating_query(pair)

        weighted_avg = weighted_avg * torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    def forward(self, pair, mask):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        if pair.dim() == 3:
            nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1).contiguous()

            if self.transpose:
                pair = pair.permute(1, 0, 2).contiguous()

            pair = self._attention(pair, mask, nonbatched_bias)

            if self.transpose:
                pair = pair.permute(1, 0, 2).contiguous()
        else:
            nonbatched_bias = self.pair_bias_projection(pair).permute(0, 3, 1, 2).contiguous()
            seq_len = nonbatched_bias.shape[-1]
            parallel_times = nonbatched_bias.shape[0]
            indices = torch.arange(parallel_times).repeat(seq_len)  # [0,0,...,0,1,1,...,1]
            nonbatched_bias = nonbatched_bias[indices]

            if self.transpose:
                pair = pair.permute(0, 2, 1, 3).contiguous()
            
            pair_dim0 = pair.shape[0]
            pair_dim1 = pair.shape[1]
            pair = pair.flatten(0, 1).contiguous()
            pair = self._attention(pair, mask, nonbatched_bias)
            pair = pair.view(pair_dim0, pair_dim1, *pair.shape[1:]).contiguous()
            if self.transpose:
                pair = pair.permute(0, 2, 1, 3).contiguous()

        return pair


class MSAAttention(nn.Module):
    def __init__(self, c_msa=64, c_pair=128, num_head=8):
        super(MSAAttention, self).__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = fastnn.LayerNorm(self.c_msa)
        self.pair_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_logits = nn.Linear(self.c_pair, self.num_head, bias=False)
        self.v_projection = nn.Linear(
            self.c_msa, self.num_head * self.value_dim, bias=False)
        self.gating_query = nn.Linear(self.c_msa, self.c_msa, bias=False)
        self.output_projection = nn.Linear(self.c_msa, self.c_msa, bias=False)

    def forward(self, msa, msa_mask, pair):
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1).contiguous()

        logits = logits + 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head)

        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        gate_values = self.gating_query(msa)
        v_avg = v_avg * torch.sigmoid(gate_values)

        return self.output_projection(v_avg)
