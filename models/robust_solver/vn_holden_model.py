import torch
import torch.nn as nn

from models.vn_layers import *
from models.robust_solver.utils import VNResidualBlock


class VNHoldenModel(nn.Module):
    """
    This is the VN implementation of Holden's model
    """
    def __init__(self, num_markers: int, num_joints: int, hidden_size: int, num_skip_layers: int, use_svd: bool = False) -> None:
        super(VNHoldenModel, self).__init__()

        self.use_svd = use_svd
        self.input_size = num_markers
        self.hidden_size = hidden_size // 3
        self.output_size = num_joints * 4

        self.first_layer = VNLinear(self.input_size, self.hidden_size)

        self.resnet_block = torch.nn.Sequential()
        for i in range(num_skip_layers):
            self.resnet_block.add_module(f"skip_{i}", VNResidualBlock(self.hidden_size))

        self.relu = VNLeakyReLU(self.hidden_size)
        self.last_layer = VNLinear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: point features of shape [B, N_feat, 3]
        """
        out = self.first_layer(x)

        out = self.resnet_block(out)

        out = self.relu(out)
        out = self.last_layer(out) # [B, N_joints * 4, 3]
        out = out.view(-1, self.output_size // 4, 4, 3)
        out = out.transpose(-1, -2) # [B, N_joints, 4, 3]

        # if self.use_svd:
        #     out = out.view(-1, 3, 4)
        #     out[:, :, :3] = symmetric_orthogonalization(out[:, :, :3].clone()).clone()
        #     out = out.view(-1, self.output_size)

        return out
