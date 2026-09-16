import torch
import torch.nn as nn

from torchscore.nn.layer_norm import LayerNorm


class TriangleMultiplication(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplication, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation = 'ckj,cki->cij'
        if _outgoing is True:
            self.equation = 'cik,cjk->cij'

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
        projection = projection.permute(2, 0, 1).contiguous()
        if mask is not None:
            projection *= mask.unsqueeze(0)

        gate = self.gate(pair)
        gate = gate.permute(2, 0, 1).contiguous()
        projection *= torch.sigmoid(gate)

        projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])

        a, b = torch.chunk(projection, 2, dim=1)
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
        pair = torch.einsum(self.equation, a, b)

        pair = pair.permute(1, 2, 0).contiguous()
        pair = self.center_norm(pair)
        pair = self.output_projection(pair)

        gate_out = self.gating_linear(input_pair)
        pair *= torch.sigmoid(gate_out)
        input += pair
        return input
