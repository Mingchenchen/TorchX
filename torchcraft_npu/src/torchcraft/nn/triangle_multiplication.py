


import torch
import torch.nn as nn

from torchcraft.nn.layer_norm import LayerNorm


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
        self.a_indices = torch.arange(0, total_elements, 2, device='npu')
        self.b_indices = torch.arange(1, total_elements, 2, device='npu')

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
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
