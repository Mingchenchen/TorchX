



import torch
import torch.nn as nn

from torchfold import fastnn


class Transition(nn.Module):

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = fastnn.LayerNorm(c_x)
        self.transition1 = nn.Linear(
            c_x, self.num_intermediate_factor * c_x * 2, bias=False)
        self.transition1_weight_t = None # cache transposed weight
        self.transition2 = nn.Linear(
            self.num_intermediate_factor * c_x, c_x, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        if self.transition1_weight_t is None:
            self.transition1_weight_t = self.transition1.weight.transpose(0,1)
        c = fastnn.gated_linear_unit(x, self.transition1_weight_t)
        return self.transition2(c)


class OuterProductMean(nn.Module):
    def __init__(self, c_msa: int = 64, num_output_channel: int = 128, num_outer_channel: int = 32) -> None:
        super(OuterProductMean, self).__init__()

        self.c_msa = c_msa
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.epsilon = 1e-3

        self.layer_norm_input = fastnn.LayerNorm(self.c_msa)
        self.left_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)
        self.right_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)

        self.output_w = nn.Parameter(
            torch.randn(self.num_outer_channel, self.num_outer_channel, self.num_output_channel))
        self.output_b = nn.Parameter(
            torch.randn(self.num_output_channel))

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask1 = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left_act = mask1 * self.left_projection(msa)
        right_act = mask1 * self.right_projection(msa)

        act = torch.einsum('abc,ade->dceb', left_act, right_act)
        act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
        act = act.permute(1, 0, 2)

        norm = torch.matmul(mask.permute(1, 0), mask).unsqueeze(-1)
        return act / (self.epsilon + norm)
