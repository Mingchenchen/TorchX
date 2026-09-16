

import os
import torch
import torch.nn as nn

from torchcraft.nn.layer_norm import LayerNorm

# Optional fused triangle multiplication via cuequivariance, enabled by env
# TRIANGLE_MULTIPLICATIVE=cuequivariance. cueq needs c_pair % 32 == 0 and CUDA.
# AF3 uses a single interleaved projection/gate (even ch = a, odd ch = b);
# cueq expects concatenated [a; b] -- we de-interleave below. xfold's forward
# INCLUDES the residual, and the kernel returns update only, so we add it back.
_USE_CUEQ_TRIMUL = os.environ.get("TRIANGLE_MULTIPLICATIVE", "").lower() == "cuequivariance"
_CUEQ_TRIMUL_FN = None
def _cueq_trimul_fn():
    global _CUEQ_TRIMUL_FN
    if _CUEQ_TRIMUL_FN is None:
        from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
        _CUEQ_TRIMUL_FN = triangle_multiplicative_update
    return _CUEQ_TRIMUL_FN

class TriangleMultiplicationTorch(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplicationTorch, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self._outgoing = _outgoing
        
        total_elements = 2 * c_pair
        self.a_indices = torch.arange(0, total_elements, 2, device='cuda')
        self.b_indices = torch.arange(1, total_elements, 2, device='cuda')

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        if _USE_CUEQ_TRIMUL and pair.is_cuda and self.c_pair % 32 == 0:
            proj_w = self.projection.weight           # [2c, c], interleaved (even=a, odd=b)
            gate_w = self.gate.weight                 # [2c, c], interleaved
            a_p, b_p = proj_w[0::2], proj_w[1::2]
            a_g, b_g = gate_w[0::2], gate_w[1::2]
            if self._outgoing:
                p_in = torch.cat([a_p, b_p], dim=0)
                g_in = torch.cat([a_g, b_g], dim=0)
            else:
                p_in = torch.cat([b_p, a_p], dim=0)
                g_in = torch.cat([b_g, a_g], dim=0)
            m = pair.new_ones(pair.shape[:-1]) if mask is None else mask
            update = _cueq_trimul_fn()(
                pair[None],
                direction="outgoing" if self._outgoing else "incoming",
                mask=m[None],
                norm_in_weight=self.left_norm_input.weight,
                norm_in_bias=self.left_norm_input.bias,
                p_in_weight=p_in,
                g_in_weight=g_in,
                norm_out_weight=self.center_norm.weight,
                norm_out_bias=self.center_norm.bias,
                p_out_weight=self.output_projection.weight,
                g_out_weight=self.gating_linear.weight,
                eps=self.left_norm_input.eps,
            )[0]
            return pair + update

        input = pair
        pair = self.left_norm_input(pair)
        input_pair = pair

        projection = self.projection(pair)
        if mask is not None:
            projection = projection * mask.unsqueeze(-1)

        gate = self.gate(pair)
        projection = (projection * torch.sigmoid(gate)).permute(2, 0, 1).contiguous()
        
        # Extract corresponding elements using index_select
        a = torch.index_select(projection, dim=0, index=self.a_indices)
        b = torch.index_select(projection, dim=0, index=self.b_indices)
        if self._outgoing is True:
            pair = torch.matmul(a, b.permute(0, 2, 1).contiguous())
        else:
            pair = torch.matmul(b.permute(0, 2, 1).contiguous(), a)

        pair = pair.permute(1, 2, 0).contiguous()
        pair = self.center_norm(pair)
        pair = self.output_projection(pair)

        gate_out = self.gating_linear(input_pair)
        pair = pair * torch.sigmoid(gate_out)
        input = input + pair
        return input


TriangleMultiplication = TriangleMultiplicationTorch
