# Copyright (c) 2020, Kfir Aberman, Peizhuo Li, Yijia Weng, Dani Lischinski, Olga Sorkine-Hornung, Daniel Cohen-Or and Baoquan Chen.
# All rights reserved.
# 
# This source code is licensed under the BSD-style license 
# https://github.com/DeepMotionEditing/deep-motion-editing/blob/master/license.txt

import math
import torch
import torch.nn as nn
import numpy as np


class ForwardKinematics:
    def __init__(self, topology, edges, fk_world=False, pos_repr="3d", rotation='quaternion'):
        self.topology = topology # J
        self.rotation_map = [e[1] for e in edges] # J-1, all joints except root
        self.world = fk_world
        self.pos_repr = pos_repr
        self.quater = rotation == 'quaternion'

    def process_raw_repr(self, raw, quater=None):
        if self.pos_repr == '3d':
            # raw: bs x ((J-1) * 4 + 3) x T
            position = raw[:, -3:, :] # bs x 3 x T
            rotation = raw[:, :-3, :] # bs x ((J-1) * 4) x T
        elif self.pos_repr == '4d':
            raise Exception('Not support')

        if quater:
            rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1])) # bs x (J-1) x 4 x T
            identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
        else:
            rotation = rotation.reshape((rotation.shape[0], -1, 3, rotation.shape[-1]))
            identity = torch.zeros((3, ), dtype=torch.float, device=raw.device)
        
        identity = identity.view((1, 1, -1, 1))
        new_shape = list(rotation.shape)
        new_shape[1] += 1
        new_shape[2] = 1
        rotation_final = identity.repeat(new_shape) # bs x J x 3/4 x T

        # put joint's rotation in corresponding position in its mapping
        for i, j in enumerate(self.rotation_map):
            rotation_final[:, j, :, :] = rotation[:, i, :, :]
        
        return rotation_final, position, identity

    def forward_from_raw(self, raw, offset, world=None, quater=None):
        '''
        Separates raw rotation matrix into rotation matrices of joints and global position of root
        and then applies forward kinematics

        Input:
            raw should have shape batch_size * (Joint_num - 1) * 3/4 + 3) * Time
            offset should have shape batch_size * Joint_num * 3
        Output:
            rotation_final have shape batch_size * Joint_num * (3/4) * Time
            we then apply forward kinematics
        '''
        if world is None: world = self.world
        if quater is None: quater = self.quater

        rotation, position, identity = self.process_raw_repr(raw, quater)

        return self.forward(rotation, position, offset, world=world, quater=quater)

    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False, world=True):
        '''
        Computes local or global positions of each joint (depending on world argument)

        Input:
            rotation should have shape batch_size * Joint_num * (3/4) * Time
            position should have shape batch_size * 3 * Time
            offset should have shape batch_size * Joint_num * 3
        Output:
            output have shape batch_size * Time * Joint_num * 3
        '''
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2) # bs x T x J x 4
        position = position.permute(0, 2, 1) # bs x T x 3
        result = torch.empty(rotation.shape[:-1] + (3, ), device=position.device) # bs x T x J x 3

        # normalize rotation matrix
        norm = torch.norm(rotation, dim=-1, keepdim=True)
        rotation = rotation / norm

        if quater:
            transform = self.transform_from_quaternion(rotation) # bs x T x J x 3 x 3
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1)) # bs x 1 x J x 3 x 1

        result[..., 0, :] = position # root's result will be its global position
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :], transform[..., i, :, :])
            result[..., i, :] = torch.matmul(transform[..., i, :, :], offset[..., i, :, :]).squeeze()
            if world: result[..., i, :] += result[..., pi, :]
        return result

    def from_local_to_world(self, res: torch.Tensor):
        res = res.clone()
        for i, pi in enumerate(self.topology):
            if pi == 0 or pi == -1:
                continue
            res[..., i, :] += res[..., pi, :]
        return res

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m
