import torch
import torch.nn as nn

from models.mocap_solver.skeleton import *
from models.mocap_solver.utils import ResidualBlock, DenseBlock


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

    def freeze_params(self):
        for param in self.decoder.parameters():
            param.requires_grad = False


# eoncoder for dynamic part, i.e. motion + offset part
class DynamicEncoder(nn.Module):
    def __init__(self, edges, num_layers=2, skeleton_dist=1,
                kernel_size=15, padding_mode="zeros",
                skeleton_info='concat', offset_channels=[], offset_joint_num=[]):
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
        if skeleton_info == 'concat':
            add_offset = True
            self.offset_joint_num = offset_joint_num
            self.offset_channels = offset_channels
        else:
            add_offset = False
            self.offset_joint_num = [None]*num_layers
            self.offset_channels = [0]*num_layers

        for i in range(num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(num_layers):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            if i == 0:
                in_channels += 4
                out_channels = in_channels * 2
                
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            conv = SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                joint_num=self.edge_num[i], kernel_size=kernel_size, stride=2,
                                padding=padding, padding_mode=padding_mode, bias=bias, add_offset=add_offset,
                                in_offset_channel=3 * self.offset_channels[i], offset_joint_num=self.offset_joint_num[i])
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
            self.layers.append(nn.Upsample(scale_factor=2, mode=upsampling, align_corners=False))

            seq = []
            in_channels = enc.channel_list[num_layers - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[num_layers - i - 1], skeleton_dist)

            unpool = SkeletonUnpool(enc.pooling_list[num_layers - i - 1], 
                                    in_channels // len(neighbor_list), last_unpool=(i==num_layers-1))
            self.unpools.append(unpool)
            seq.append(unpool)

            if i != 0 and i != num_layers - 1: bias = False
            else: bias = True
            conv = SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                            joint_num=enc.edge_num[num_layers - i - 1], kernel_size=kernel_size, stride=1,
                            padding=padding, padding_mode=padding_mode, bias=bias, add_offset=add_offset,
                            in_offset_channel=3 * enc.offset_channels[-i - 1], offset_joint_num=enc.offset_joint_num[-i-1])
            seq.append(conv)
            self.convs.append(conv)

            if i != num_layers - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Upsample): input = layer(input.transpose(-2, -1))
            else:
                if self.skeleton_info == 'concat' and offset is not None:
                    self.convs[i // 2].set_offset(offset[i // 2])
                input = layer(input)

        # throw the padded rwo for global position
        input = input[:, :, :-1]

        return input

    def freeze_params(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False


# eoncoder for dynamic part, i.e. motion + offset part
class MarkerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, skeleton_info="concat", offset_dims=None):
        super(MarkerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.dense_blocks = []

        self.skeleton_info = skeleton_info
        if skeleton_info == "concat":
            add_offset = True
            self.offset_dims = offset_dims
        else:
            add_offset = False
            self.offset_dims = [None]*num_layers

        for i in range(num_layers):
            seq = []
            if i == 0: dense = DenseBlock(self.input_size, self.hidden_size, add_offset, self.offset_dims[i])
            else: dense = DenseBlock(self.hidden_size, self.hidden_size, add_offset, self.offset_dims[i])
            seq.append(dense)
            self.dense_blocks.append(dense)

            seq.append(nn.ReLU())

            res_block = ResidualBlock(hidden_size, hidden_size)
            seq.append(res_block)

            self.layers.append(nn.Sequential(*seq))

    def forward(self, x, offset=None):
        out = x.reshape(x.shape[0], -1)
        for i, layer in enumerate(self.layers):
            if self.skeleton_info == "concat" and offset is not None:
                self.dense_blocks[i].set_offset(offset[i])
            out = layer(out)
        return out


# decoder for dynamic part, i.e. motion + offset part
class MarkerDecoder(nn.Module):
    def __init__(self, enc, num_layers=2):
        super(MarkerDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.enc = enc
        self.hidden_size = enc.hidden_size
        self.output_size = enc.input_size
        self.dense_blocks = []

        self.skeleton_info = enc.skeleton_info
        if self.skeleton_info == "concat": add_offset = True
        else: add_offset = False

        for i in range(num_layers):
            seq = []

            dense = DenseBlock(self.hidden_size, self.hidden_size, add_offset, enc.offset_dims[-i-1])
            seq.append(dense)
            self.dense_blocks.append(dense)
            seq.append(nn.ReLU())
            res_block = ResidualBlock(self.hidden_size, self.hidden_size)
            seq.append(res_block)

            self.layers.append(nn.Sequential(*seq))

            if i == num_layers - 1:
                seq.append(nn.ReLU())
                dense = DenseBlock(self.hidden_size, self.output_size, False)
                seq.append(dense)
                self.layers.append(nn.Sequential(*seq))

    def forward(self, x, offset=None):
        out = x.reshape(x.shape[0], -1)
        for i, layer in enumerate(self.layers):
            if i < len(self.dense_blocks) and self.skeleton_info == "concat" and offset is not None:
                self.dense_blocks[i].set_offset(offset[i])
            out = layer(out)
        return out

    def freeze_params(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, edges, input_size, hidden_size, 
                skeleton_info="concat", offset_dims=None,
                offset_channels=[], offset_joint_num=[]):
        super(Encoder, self).__init__()
        self.static_enc = StaticEncoder(edges)
        self.dynamic_enc = DynamicEncoder(edges, skeleton_info=skeleton_info, 
                                        offset_channels=offset_channels, 
                                        offset_joint_num=offset_joint_num)
        self.marker_enc = MarkerEncoder(input_size, hidden_size, 
                                        skeleton_info=skeleton_info, 
                                        offset_dims=offset_dims)

    def forward(self, x_c, x_t, x_m):
        # static encoder
        l_t = self.static_enc(x_t)

        # offsets
        offset_enc = [x_t, l_t]

        # dynamic encoder
        l_m = self.dynamic_enc(x_m, offset_enc)

        # marker encoder
        l_c = self.marker_enc(x_c, offset_enc)

        return l_c, l_t, l_m


class Decoder(nn.Module):
    def __init__(self, enc, skeleton_info="concat", offset_dims=None):
        super(Decoder, self).__init__()
        self.static_dec = StaticDecoder(enc.static_enc)
        self.dynamic_dec = DynamicDecoder(enc.dynamic_enc)
        self.marker_dec = MarkerDecoder(enc.marker_enc)

    def forward(self, l_c, l_t, l_m):
        # static decoder
        Y_t = self.static_dec(l_t.T)

        # offsets
        offset_dec = [l_t, Y_t]

        # dynamic decoder
        Y_m = self.dynamic_dec(l_m, offset_dec)

        # marker decoder
        Y_c = self.marker_dec(l_c, offset_dec)

        return Y_c, Y_t, Y_m
    
    def freeze_params(self):
        self.static_dec.freeze_params()
        self.dynamic_dec.freeze_params()
        self.marker_dec.freeze_params()


def test_models():
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
    print(joint_topology)
    edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
    print("-------------")
    print("Static encoder")
    x_t = torch.rand(5, len(joint_topology) * 3)
    print(x_t.shape)

    stat_enc = StaticEncoder(edges)
    lat_t = stat_enc(x_t).T
    print(lat_t.shape)

    stat_dec = StaticDecoder(stat_enc)
    res_t = stat_dec(lat_t)
    print(res_t.shape)
    print("\n----------------------------\n")

    print("Dynamic encodder")
    x_m = torch.rand(5, len(joint_topology)*4 + 3, 64)
    print(x_m.shape)
    dynamic_enc = DynamicEncoder(edges, skeleton_info='')
    lat_m = dynamic_enc(x_m)
    print(lat_m.shape)
    dynamic_dec = DynamicDecoder(dynamic_enc, skeleton_info='')
    res_m = dynamic_dec(lat_m).transpose(-2, -1)
    print(res_m.shape)
    print("\n----------------------------\n")

    print("Dynamic encodder with offset")
    x_ = torch.rand(5, len(edges)*4 + 3, 64)
    offsets_enc = [x_t, lat_t.T]
    print(x_.shape)
    print(offsets_enc[0].shape, offsets_enc[1].shape)
    dynamic_enc = DynamicEncoder(edges, offset_channels=[1, 8], offset_joint_num=[len(joint_topology), 7])
    lat_m = dynamic_enc(x_, offsets_enc)
    offsets_dec = [lat_t.T, res_t]
    print(lat_m.shape)
    print("Decoder")
    print(offsets_dec[0].shape, offsets_dec[1].shape)
    dynamic_dec = DynamicDecoder(dynamic_enc)
    res_m = dynamic_dec(lat_m, offsets_dec).transpose(-2, -1)
    print(res_m.shape)
    print("\n----------------------------\n")

    print("Marker encodder with offset")
    x_c = torch.rand(5, 56 * 24 * 3)
    offsets_enc = [x_t, lat_t.T]
    print(x_c.shape)
    print(offsets_enc[0].shape, offsets_enc[1].shape)
    marker_enc = MarkerEncoder(x_c.shape[1], 1024, offset_dims=[75, 168])
    lat_c = marker_enc(x_c, offsets_enc)
    offsets_dec = [lat_t.T, res_t]
    print(lat_c.shape)
    print("Decoder")
    print(offsets_dec[0].shape, offsets_dec[1].shape)
    marker_dec = MarkerDecoder(marker_enc)
    res_c = marker_dec(lat_c, offsets_dec).view(5, 56, 24, 3)
    print(res_c.shape)
    print("\n----------------------------\n")


if __name__ == "__main__":
    test_models()
