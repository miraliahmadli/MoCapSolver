import torch
import torch.nn as nn

from tools.utils import symmetric_orthogonalization


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
    def __init__(self, num_markers: int, num_joints: int, hidden_size: int, num_skip_layers: int, use_svd: bool = False):
        super(Baseline, self).__init__()

        self.use_svd = use_svd
        self.input_size = num_markers * 3 * 2
        self.hidden_size = hidden_size
        self.output_size = num_joints * 3 * 4

        self.first_layer = nn.Linear(self.input_size, hidden_size)

        self.skip_block = torch.nn.Sequential()
        for i in range(num_skip_layers):
            self.skip_block.add_module(f"skip_{i}", DenseBlock(hidden_size))

        self.relu = nn.ReLU()
        self.last_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, X, Z):
        x = torch.cat((X[..., None], Z[..., None]), axis=-1).view(-1, self.input_size)
        x = self.first_layer(x)
        x = self.skip_block(x)
        x = self.relu(x)
        x = self.last_layer(x)

        if self.use_svd:
            x = x.view(-1, 3, 4)
            x[:, :, :3] = symmetric_orthogonalization(x[:, :, :3].clone()).clone()
            x = x.view(-1, self.output_size)

        return x


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
