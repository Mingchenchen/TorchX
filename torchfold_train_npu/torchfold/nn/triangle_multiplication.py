



import torch
import torch.nn as nn

from torchfold import fastnn


class TriangleMultiplication(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplication, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation='...ckj,...cki->...cij'
        if _outgoing is True:
            self.equation='...cik,...cjk->...cij'

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
        if projection.dim() == 3:
            projection = projection.permute(2, 0, 1).contiguous()
        else:
            projection = projection.permute(0, 3, 1, 2).contiguous()
        if mask is not None:
            projection = projection * mask.unsqueeze(0)

        gate = self.gate(pair)
        if gate.dim() == 3:
            gate = gate.permute(2, 0, 1).contiguous()
            projection = projection * torch.sigmoid(gate)
            projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])
            a, b = torch.chunk(projection, 2, dim=1)
            a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
            pair = torch.einsum(self.equation, a, b)
            pair = pair.permute(1, 2, 0).contiguous()
        else:
            gate = gate.permute(0, 3, 1, 2).contiguous()
            projection = projection * torch.sigmoid(gate)
            projection = projection.reshape(projection.shape[0], self.c_pair, 2, *projection.shape[2:])
            
            a, b = torch.chunk(projection, 2, dim=2)
            a, b = torch.squeeze(a, dim=2), torch.squeeze(b, dim=2)
            pair = torch.einsum(self.equation, a, b)
            pair = pair.permute(0, 2, 3, 1).contiguous()

        pair = self.center_norm(pair)
        pair = self.output_projection(pair)

        gate_out = self.gating_linear(input_pair)
        pair = pair * torch.sigmoid(gate_out)
        input = input + pair
        return input
