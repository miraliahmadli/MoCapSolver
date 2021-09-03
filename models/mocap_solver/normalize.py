import torch
import torch.nn as nn
from models.mocap_solver.utils import ResidualBlock, DenseBlock


class MarkerReliability(nn.Module):
    """
    Marker Reliability model
    """
    def __init__(self, num_markers: int, num_ref_markers: int, window_size: int = 64,
                hidden_size: int = 1024, num_res_layers: int = 4):
        super(MarkerReliability, self).__init__()
        self.input_size = num_markers * 3 * window_size
        self.num_ref_markers = num_ref_markers
        self.window_size = window_size
        self.output_size = num_ref_markers * window_size
        
        self.dense1 = DenseBlock(self.input_size, hidden_size)
        self.lerelu =  nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

        self.res_block = nn.Sequential()
        for i in range(num_res_layers):
            self.res_block.add_module(f"resnet_{i}", ResidualBlock(hidden_size, hidden_size))
        
        self.dense2 = DenseBlock(hidden_size, self.output_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.dense1(x)
        out = self.lerelu(out)
        out = self.res_block(out)
        out = self.lerelu(out)
        out = self.dense2(out)
        out = self.sigmoid(out)
        out = out.view(-1, self.window_size, self.num_ref_markers)
        return out

def test():
    x = torch.rand(512, 64, 56, 3).to("cuda")
    model = MarkerReliability(56, 8).to("cuda")
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    test()