import torch
import torch.nn as nn

from models.mocap_solver.utils import ResidualBlock, DenseBlock


class MarkerReliability(nn.Module):
    """
    Marker Reliability model
    """
    def __init__(self, num_markers: int, num_ref_markers: int, hidden_size: int = 1024, num_res_layers: int = 4):
        super(MarkerReliability, self).__init__()
        self.input_size = num_markers * 3
        self.output_size = num_ref_markers
        
        self.dense1 = DenseBlock(self.input_size, hidden_size)
        self.lerelu =  nn.LeakyReLU(negative_slope=0.2)

        self.res_block = nn.Sequential()
        for i in range(num_skip_layers):
            self.res_block.add_module(f"resnet_{i}", ResidualBlock(hidden_size, hidden_size))
        
        self.dense2 = DenseBlock(hidden_size, self.output_size)

    def forward(self, x):
        x.view(x.shape[0], -1)
        out = self.dense1(x)
        out = self.lerelu(out)
        out = self.res_block(out)
        out = self.lerelu(out)
        out = self.dense2(out)
        return out
