import torch
import torch.nn as nn

from tools.utils import symmetric_orthogonalization


class ResNetBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super(ResNetBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out += x
        out = self.relu(out)
        return out


class Baseline(nn.Module):
    """
    This is the baseline neural model
    """
    def __init__(self, num_markers: int, num_joints: int, hidden_size: int, num_skip_layers: int, use_svd: bool = False):
        super(Baseline, self).__init__()

        self.use_svd = use_svd
        self.input_size = num_markers * 3 * 2
        self.hidden_size = hidden_size
        self.output_size = num_joints * 3 * 4

        self.first_layer = nn.Linear(self.input_size, hidden_size)

        self.resnet_block = torch.nn.Sequential()
        for i in range(num_skip_layers):
            self.resnet_block.add_module(f"skip_{i}", ResNetBlock(hidden_size))

        self.relu = nn.ReLU()
        self.last_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, X, Z):
        x = torch.cat((X[..., None], Z[..., None]), axis=-1).view(-1, self.input_size)
        out = self.first_layer(x)
        out = self.relu(out)
        out = self.resnet_block(out)
        out = self.last_layer(out)

        if self.use_svd:
            out = out.view(-1, 3, 4)
            out[:, :, :3] = symmetric_orthogonalization(out[:, :, :3].clone()).clone()
            out = out.view(-1, self.output_size)

        return out


def test():
    num_markers = 41
    num_joints = 31
    hidden_size = 2048
    num_pose = 100
    num_skip_layers = 5

    model = Baseline(num_markers, num_joints, hidden_size, 5)
    print(model)
    x = torch.randn(num_pose, num_markers, 3)
    z = torch.randn(num_pose, num_markers, 3)

    x_out = model(x, z)
    print(x_out.shape)

if __name__ == "__main__":
    test()
