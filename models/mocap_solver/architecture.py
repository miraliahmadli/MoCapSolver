import torch
import torch.nn as nn

import sys
sys.path.append("../../")
from models.mocap_solver.skeleton import *


class ResidualBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ResidualBlock, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.dense = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.dense(out)
        out += x
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                add_offset=False, offset_dim=None):
        super(DenseBlock, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.add_offset = add_offset and (offset_dim is not None)
        if self.add_offset:
            self.offset_enc = nn.Linear(offset_dim, hidden_size)

    def set_offset(self, offset):
        if not self.add_offset: raise Exception('Wrong Combination of Parameters')
        self.offset = offset.reshape(offset.shape[-2], -1)

    def forward(self, x):
        out = self.dense(x)
        if self.add_offset:
            offset_res = self.offset_enc(self.offset)
            out += offset_res / 100
        return out


class SolverBlock(nn.Module):
    def __init__(self, hidden_size: int, out_size: int):
        super(SolverBlock, self).__init__()
        self.res_block = ResidualBlock(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.res_block(x)
        out = self.res_block(out)
        out = self.relu(out)
        out = self.dense(out)
        return out


# eoncoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, edges, num_layers=1, skeleton_dist=1):
        super(StaticEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.pooling_list = []
        self.topologies = [edges]

        neighbor_list = find_neighbor(self.topologies[-1], skeleton_dist)

        channels = 3
        in_channels = channels * len(neighbor_list)
        out_channels = channels * 8 * len(neighbor_list)
        self.channel_list = [in_channels, out_channels]

        seq = []
        linear = SkeletonLinear(neighbor_list, in_channels=in_channels,
                                out_channels=out_channels, extra_dim1=False)
        seq.append(linear)

        pool = SkeletonPool(self.topologies[-1], channels_per_edge=channels*8, 
                            pooling_mode='mean', last_pool=True)
        seq.append(pool)
        self.pooling_list.append(pool.pooling_list)
        self.topologies.append(pool.new_edges)

        activation = nn.ReLU()
        seq.append(activation)

        self.encoder = nn.Sequential(*seq)

    # input should have shape batch * (num_joints * 3)
    def forward(self, input: torch.Tensor):
        out = self.encoder(input)
        return out


# decoder for static part, i.e. offset part
class StaticDecoder(nn.Module):
    def __init__(self, enc, num_layers=1, skeleton_dist=1):
        super(StaticDecoder, self).__init__()
        self.layers = nn.ModuleList()

        in_channels = enc.channel_list[1]
        out_channels = enc.channel_list[0]
        neighbor_list = find_neighbor(enc.topologies[-2], skeleton_dist)

        seq = []
        unpool = SkeletonUnpool(enc.pooling_list[-1], in_channels // len(neighbor_list))
        seq.append(unpool)

        activation = nn.LeakyReLU(negative_slope=0.2)
        seq.append(activation)

        seq.append(SkeletonLinear(neighbor_list, in_channels=in_channels,
                                out_channels=out_channels, extra_dim1=False))

        self.decoder = nn.Sequential(*seq)

    # input should have shape batch x (num_joints * 3)
    def forward(self, input: torch.Tensor):
        out = self.decoder(input)
        return out


# eoncoder for dynamic part, i.e. motion + offset part
class DynamicEncoder(nn.Module):
    def __init__(self, edges, num_layers=2, skeleton_dist=1,
                kernel_size=15, skeleton_info='concat', padding_mode="zeros"):
        super(DynamicEncoder, self).__init__()
        self.topologies = [edges]
        self.channel_list = []
        self.edge_num = [len(edges) + 1]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.convs = []
        self.channel_base = [4]

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
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            conv = SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                padding=padding, padding_mode=padding_mode, bias=bias, add_offset=add_offset,
                                in_offset_channel=3 * self.channel_base[i] // self.channel_base[0])#, last_conv=(i == num_layers - 1))
            seq.append(conv)
            self.convs.append(seq[-1])

            last_pool = True if i == num_layers - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode="mean",
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.ReLU())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]) + 1)
            if i == num_layers - 1:
                self.last_channel = self.edge_num[-1] * self.channel_base[i + 1]

    def forward(self, input, offset=None):
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        out = torch.cat((input, torch.zeros_like(input[:, [0], :])), dim=1).permute(0, 2, 1)

        for i, layer in enumerate(self.layers):
            if self.skeleton_info == 'concat' and offset is not None:
                self.convs[i].set_offset(offset[i])
            out = layer(out)
        return out


# decoder for dynamic part, i.e. motion + offset part
class DynamicDecoder(nn.Module):
    def __init__(self, enc, num_layers=2, skeleton_dist=1,
                kernel_size=15, skeleton_info='concat', 
                upsampling="linear", padding_mode="zeros"):
        super(DynamicDecoder, self).__init__()
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
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[num_layers - i - 1], skeleton_dist)

            if i != 0 and i != num_layers - 1:
                bias = False
            else:
                bias = True

            unpool = SkeletonUnpool(enc.pooling_list[num_layers - i - 1], 
                                    in_channels // len(neighbor_list))
            self.unpools.append(unpool)

            self.layers.append(nn.Upsample(scale_factor=2, mode=upsampling, align_corners=False))
            seq.append(unpool)
            conv = SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                joint_num=enc.edge_num[num_layers - i - 1], kernel_size=kernel_size, stride=1,
                                padding=padding, padding_mode=padding_mode, bias=bias, add_offset=add_offset,
                                in_offset_channel=3 * enc.channel_base[num_layers - i - 1] // enc.channel_base[0])
            seq.append(conv)
            self.convs.append(conv)
            if i != num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Upsample):
                input = layer(input.transpose(-2, -1))
            else:
                if self.skeleton_info == 'concat' and offset is not None:
                    self.convs[i // 2].set_offset(offset[i // 2])
                input = layer(input)

        # throw the padded rwo for global position
        input = input[:, :, :-1]

        return input


# eoncoder for dynamic part, i.e. motion + offset part
class MarkerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, template_skeleton="add"):
        super(MarkerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.dense_blocks = []

        self.template_skeleton = template_skeleton
        if template_skeleton == 'add': add_offset = True
        else: add_offset = False

        for i in range(num_layers):
            seq = []
            if i == 0: 
                dense = DenseBlock(input_size, hidden_size, add_offset)
            else:
                dense = DenseBlock(hidden_size, hidden_size, add_offset)
            seq.append(dense)
            self.dense_blocks.append(dense)
            seq.append(nn.ReLU())
            res_block = ResidualBlock(hidden_size, hidden_size)
            seq.append(res_block)

            self.layers.append(nn.Sequential(*seq))

    def forward(self, x, offset=None):
        for i, layer in enumerate(self.layers):
            if self.template_skeleton == 'add' and offset is not None:
                self.dense_blocks[i].set_offset(offset[i])
            x = layer(x)
        return x


# decoder for dynamic part, i.e. motion + offset part
class MarkerDecoder(nn.Module):
    def __init__(self, enc, num_layers=2, template_skeleton="add"):
        super(MarkerDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.enc = enc
        self.hidden_size = enc.hidden_size
        self.output_size = enc.inp_size
        self.dense_blocks = []

        self.template_skeleton = template_skeleton
        if template_skeleton == 'add': add_offset = True
        else: add_offset = False

        for i in range(num_layers):
            seq = []
            if i == num_layers - 1:
                seq.append(nn.ReLU())
                dense = DenseBlock(hidden_size, self.output_size, False)
                seq.append(dense)
                self.layers.append(nn.Sequential(*seq))
                break

            dense = DenseBlock(hidden_size, hidden_size, add_offset)
            seq.append(dense)
            self.dense_blocks.append(dense)
            seq.append(nn.ReLU())
            res_block = ResidualBlock(hidden_size, hidden_size)
            seq.append(res_block)

            self.layers.append(nn.Sequential(*seq))

    def forward(self, x, offset=None):
        for i, layer in enumerate(self.layers):
            if i < len(self.dense_blocks) and self.template_skeleton == 'add' and offset is not None:
                self.dense_blocks[i].set_offset(offset[i])
            x = layer(x)
        return x


class MocapSolver(nn.Module):
    def __init__(self, num_markers, window_size, hidden_size, num_res_layers=3,
                use_marker_conf=False, marker_out_size:int=1024, 
                use_skeleton=False, skeleton_out_size:int=168, 
                use_motion=False, motion_out_size:int=1792):
        super(MocapSolver, self).__init__()
        self.use_marker_conf = use_marker_conf
        self.use_skeleton = use_skeleton
        self.use_motion = use_motion

        seq = []
        dense = nn.Linear(num_markers*window_size*3, hidden_size)
        seq.append(dense)
        lerelu = nn.LeakyReLU(negative_slope=0.2)
        seq.append(lerelu)
        for i in range(num_res_layers):
            res_block = ResidualBlock(hidden_size, hidden_size)
            seq.append(res_block)
        self.first_layer = nn.Sequential(*seq)

        if self.use_marker_conf:
            self.ms_c = SolverBlock(hidden_size, marker_out_size)
        if self.use_skeleton:
            self.ms_t = SolverBlock(hidden_size, skeleton_out_size)
        if self.use_motion:
            self.ms_m = SolverBlock(hidden_size, motion_out_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.first_layer(x)
        outputs = []
        if self.use_marker_conf:
            out_c = self.ms_c(out)
            outputs.append(out_c)
        if self.use_skeleton:
            out_t = self.ms_t(out)
            outputs.append(out_t)
        if self.use_motion:
            out_m = self.ms_m(out)
            outputs.append(out_m)
        return outputs


def test_models():
    def get_topology():
        joint_topology = [-1] * 25
        with open("../../dataset/hierarchy_no_hand.txt") as f:
            lines = f.readlines()
            for l in lines:
                lst = list(map(int, l.strip().split()))
                parent = lst[0]
                for j in lst[1:]:
                    joint_topology[j] = parent
        return joint_topology

    joint_topology = get_topology()
    print(joint_topology)
    print("-------------")
    edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
    for row in edges:
        print(row)
    print("-------------")
    print("Static encoder")
    x_t = torch.rand(1, len(joint_topology) * 3)
    print(x_t.shape)

    stat_enc = StaticEncoder(edges)
    lat_t = stat_enc(x_t).T
    print(lat_t.shape)

    stat_dec = StaticDecoder(stat_enc)
    res_t = stat_dec(lat_t)
    print(res_t.shape)
    print("\n----------------------------\n")

    print("Dynamic encodder")
    x_m = torch.rand(1, len(edges)*4 + 3, 64)
    print(x_m.shape)
    dynamic_enc = DynamicEncoder(edges, skeleton_info='')
    lat_m = dynamic_enc(x_m)
    print(lat_m.shape)
    dynamic_dec = DynamicDecoder(dynamic_enc, skeleton_info='')
    res_m = dynamic_dec(lat_m).transpose(-2, -1)
    print(res_m.shape)
    print("\n----------------------------\n")

    # print("Dynamic encodder with offset")
    # x_ = torch.rand(1, len(edges)*4 + 3, 64)
    # offsets_enc = [x_t, lat_t.T]
    # print(x_.shape)
    # print(offsets_enc[0].shape, offsets_enc[1].shape)
    # dynamic_enc = DynamicEncoder(edges)
    # lat_m = dynamic_enc(x_, offsets_enc)
    # offsets_dec = [lat_t, res_t]
    # print(lat_m.shape)
    # dynamic_dec = DynamicDecoder(dynamic_enc)
    # res_m = dynamic_dec(lat_m, offsets_dec).transpose(-2, -1)
    # print(res_m.shape)
    print("Mocap Solver")
    num_markers = 56
    window_size = 64
    x = torch.rand(5, 64, 56, 3)
    print(x.shape)
    ms = MocapSolver(num_markers, window_size, 1024, use_motion=True)
    res_ms = ms(x)
    print("Results")
    for res in res_ms:
        print(res.shape)
    enc = DynamicEncoder(edges, skeleton_info='')
    dec = DynamicDecoder(enc, skeleton_info='')
    lat_m_ms = res_ms[0]
    lat_m_ms = lat_m_ms.view(lat_m_ms.shape[0], 16, -1)
    print(lat_m_ms.shape)
    res_dec = dec(lat_m_ms)
    print(res_dec.shape)


if __name__ == "__main__":
    test_models()