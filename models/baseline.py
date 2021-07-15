import torch
import torch.nn as nn
from collections import OrderedDict


class DenseBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.linear(out)
        out += x
        return out


class Baseline(nn.Module):
    """
    This is the baseline neural model
    """
    def __init__(self, num_markers: int, num_joints: int, num_pose: int, hidden_size: int):
        super(Baseline, self).__init__()

        self.input_size = num_markers * 3 * 2
        self.hidden_size = hidden_size
        self.output_size = num_joints * 3 * 4

        self.first_layer = nn.Linear(self.input_size, hidden_size)
        self.skip_block = nn.Sequential(OrderedDict([
          ('skip1', DenseBlock(hidden_size)),
          ('skip2', DenseBlock(hidden_size)),
          ('skip3', DenseBlock(hidden_size)),
          ('skip4', DenseBlock(hidden_size)),
          ('skip5', DenseBlock(hidden_size)),
        ]))
        self.relu = nn.ReLU()
        self.last_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.skip_block(x)
        x = self.relu(x)
        x = self.last_layer(x)
        return x


def test():
    num_markers = 41
    num_joints = 31
    hidden_size = 2048
    num_pose = 100

    model = Baseline(num_markers, num_joints, num_pose, hidden_size)
    print(model)
    x = torch.randn(num_pose, num_markers*3*2)

    x_out = model(x)
    print(x_out.shape)

if __name__ == "__main__":
    test()
