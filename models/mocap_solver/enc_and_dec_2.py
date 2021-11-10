import torch
import torch.nn as nn
from models.mocap_solver.skeleton2 import *


class MotionEncoder(nn.Module):
    def __init__(self, edges, num_layers=2, skeleton_dist=1,
                kernel_size=15, padding_mode="reflection",
                skeleton_info='concat', offset_channels=[], offset_joint_num=[]):
        super(MotionEncoder, self).__init__()
        self.topologies = [edges]
        self.channel_base = [4]
        self.channel_list = []
        self.edge_num = [len(edges) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.convs = []

        padding = (kernel_size - 1) // 2
        bias = True
        self.skeleton_info = skeleton_info
        if skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]# + 4*(i==0)
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                    padding=padding, padding_mode=padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))#,
                                    # first_conv=i==0))
            self.convs.append(seq[-1])
            last_pool = True if i == num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode="mean",
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        input = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1)

        for i, layer in enumerate(self.layers):
            if self.skeleton_info == 'concat' and offset[i] is not None:
                self.convs[i].set_offset(offset[i])
            input = layer(input)
        return input


class MotionDecoder(nn.Module):
    def __init__(self, enc, num_layers=2, skeleton_dist=1,
                kernel_size=15, skeleton_info='concat', 
                upsampling="linear", padding_mode="reflection"):
        super(MotionDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.enc = enc
        self.convs = []

        padding = (kernel_size - 1) // 2
        self.skeleton_info = skeleton_info
        if skeleton_info == 'concat': add_offset = True
        else: add_offset = False

        for i in range(num_layers):
            seq = []
            in_channels = enc.channel_list[num_layers - i]
            out_channels = in_channels // 2# + 4*(i==num_layers-1)
            neighbor_list = find_neighbor(enc.topologies[num_layers - i - 1], skeleton_dist)

            if i != 0 and i != num_layers - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[num_layers - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=upsampling, align_corners=False))
            seq.append(self.unpools[-1])

            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                    padding=padding, padding_mode=padding_mode, bias=bias, add_offset=add_offset,
                                    in_offset_channel=3 * enc.channel_base[num_layers - i - 1] // enc.channel_base[0]))#,
                                    # last_conv=i==num_layers-1))
            self.convs.append(seq[-1])
            if i != num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            if self.skeleton_info == 'concat' and offset[i] is not None:
                self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        # throw the padded rwo for global position
        input = input[:, :-1, :]

        return input


def test():
    # import argparse
    # args = argparse.ArgumentParser().parse_args()
    # args.kernel_size = 15
    # args.skeleton_info = ""
    # args.num_layers = 2
    # args.extra_conv = 1
    # args.skeleton_dist = 1
    # args.padding_mode = "reflection"
    # args.skeleton_pool = "mean"
    # args.rotation = "quaternion"
    # args.pos_repr = "3d"
    # args.upsampling = "linear"

    def get_topology():
        joint_topology = [-1] * 24
        with open("./dataset/hierarchy_synthetic_bfs.txt") as f:
            lines = f.readlines()
            for l in lines:
                lst = list(map(int, l.strip().split()))
                parent = lst[0]
                for j in lst[1:]:
                    joint_topology[j] = parent
        return joint_topology

    joint_topology = get_topology()
    # print(joint_topology)
    from models.mocap_solver.skeleton import build_edge_topology
    edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
    edges = [e[:2] for e in edges]
    # print(edges)
    enc = MotionEncoder(edges, skeleton_info="")
    dec = MotionDecoder(enc, skeleton_info="")

    x_t = torch.rand(5, len(joint_topology) * 3)
    l_t = torch.rand(5, 168)

    x_m = torch.rand(5, len(joint_topology)*4 + 3, 64)
    print(x_m.shape, x_t.shape, l_t.shape)
    latent = enc(x_m, [x_t, l_t])
    print("\nLatent", latent.shape)
    result = dec(latent, [l_t, x_t])
    print("\nX_m", result.shape)

if __name__ == "__main__":
    test()
