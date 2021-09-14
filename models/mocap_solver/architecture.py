import torch
import torch.nn as nn

from models.mocap_solver.enc_and_dec import Encoder, Decoder
from models.mocap_solver.utils import ResidualBlock, DenseBlock, MSBlock


class AE(nn.Module):
    def __init__(self, edges, num_markers, num_joints, 
                hidden_size, num_res_layers=3,
                skeleton_info="concat", offset_dims=None,
                offset_channels=[], offset_joint_num=[]):
        super(AE, self).__init__()
        self.marker_config_size = 3 * num_markers * num_joints
        self.encoder = Encoder(edges, self.marker_config_size, hidden_size,
                                skeleton_info=skeleton_info, 
                                offset_dims=offset_dims,
                                offset_channels=offset_channels, 
                                offset_joint_num=offset_joint_num)
        self.decoder = Decoder(self.encoder)

    def forward(self, x_c, x_t, x_m):
        lat_c, lat_t, lat_m = self.encoder(x_c, x_t, x_m)
        Y_c, Y_t, Y_m = self.decoder(lat_c, lat_t, lat_m)
        return Y_c, Y_t, Y_m


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
        self.ms = nn.Sequential(*seq)

        if self.use_marker_conf:
            self.ms_c = MSBlock(hidden_size, marker_out_size)
        if self.use_skeleton:
            self.ms_t = MSBlock(hidden_size, skeleton_out_size)
        if self.use_motion:
            self.ms_m = MSBlock(hidden_size, motion_out_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.ms(x)
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


class MSD(nn.Module):
    def __init__(self, edges, num_markers=56, num_joints=24, window_size=64):
        super(MSD, self).__init__()
        self.decoder = AE(edges, num_markers, num_joints, 1024, offset_dims=[72, 168], 
                               offset_channels=[1, 8], offset_joint_num=[num_joints, 7]).decoder
        self.mocap_solver = MocapSolver(num_markers, window_size, 1024,
                                use_motion=True, use_marker_conf=True, use_skeleton=True)

        self.num_joints = num_joints
        self.num_markers = num_markers

    def forward(self, X):
        l_c, l_t, l_m = self.mocap_solver(X)
        l_m = l_m.view(l_m.shape[0], 16, -1)
        Y_c, Y_t, Y_m = self.decoder(l_c, l_t, l_m)
        Y_c = Y_c.view(Y_c.shape[0], self.num_markers, self.num_joints, 3)
        Y_t = Y_t.view(Y_t.shape[0], self.num_joints, 3)
        return Y_c, Y_t, Y_m


def test_models():
    from models.mocap_solver.skeleton import build_edge_topology
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

    print("AutoEncoder")
    x_c = torch.rand(5, 56 * 24 * 3)
    x_m = torch.rand(5, len(joint_topology)*4 + 3, 64)
    x_t = torch.rand(5, len(joint_topology) * 3)
    print(x_c.shape, x_t.shape, x_m.shape)
    auto_encoder = AE(edges, 56, 24, 1024, offset_dims=[24*3, 168],
                    offset_channels=[1, 8], offset_joint_num=[len(joint_topology), 7])
    outs = auto_encoder(x_c, x_t, x_m)
    for out in outs:
        print("\t", out.shape)
    print("\n----------------------------\n")

    # print("Mocap Solver")
    # num_markers = 56
    # window_size = 64
    # x = torch.rand(5, 64, 56, 3)
    # print("Input\n\t", x.shape)
    # ms = MocapSolver(num_markers, window_size, 1024,
    #                 use_motion=True, use_marker_conf=True, use_skeleton=True)
    # res_ms = ms(x)
    # latent_c, latent_t, latent_m = res_ms

    # print("Results")
    # print("\tLatent vecotrs:")
    # for res in res_ms:
    #     print("\t", res.shape)

    # dec = Decoder(auto_encoder.encoder)
    # latent_m = latent_m.view(latent_m.shape[0], 16, -1)
    # outs = dec(latent_c, latent_t, latent_m)
    # print("\tPredictions")
    # for out in outs:
    #     print("\t", out.shape)
    # print("\n----------------------------\n")


if __name__ == "__main__":
    test_models()
