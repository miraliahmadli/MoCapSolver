# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license 
# https://github.com/facebookresearch/pytorch3d/blob/master/LICENSE

import math
import functools
from typing import Optional

import torch
import torch.nn.functional as F


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # clipping is not important here; if q_abs is small, the candidate won't be picked
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clip(0.1))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def transformation_diff(Y_hat, Y):
    Y_h = Y_hat.detach().clone()
    Y_ = Y.detach().clone()
    R_hat, t_hat = Y_h[...,:3].transpose(-2, -1), Y_[..., 3]
    t = Y_[..., 3]
    R = R_hat @ Y_[..., :3]
    translation_diff = torch.norm(t - t_hat, p=2, dim=2)
    axis_angle = matrix_to_axis_angle(R)
    angle_diff = axis_angle[..., -1]

    return angle_diff, translation_diff


def test_angle():
    pi_4 = torch.tensor(math.pi + math.pi / 4)
    pi_6 = torch.tensor(5*math.pi)
    deg_cos_1 = torch.cos(pi_4)
    deg_cos_2 = torch.cos(pi_6)
    deg_sin_1 = torch.sin(pi_4)
    deg_sin_2 = torch.sin(pi_6)

    matrix_1 = [[[
        [deg_cos_1, -deg_sin_1, 0, 1],
        [deg_sin_1, deg_cos_1, 0, 1],
        [0, 0, 1, 1]
    ]],
    [[
        [deg_cos_1, -deg_sin_1, 0, 1],
        [deg_sin_1, deg_cos_1, 0, 1],
        [0, 0, 1, 1]
    ]]]

    matrix_2 = [[[
        [deg_cos_2, -deg_sin_2, 0, 0],
        [deg_sin_2, deg_cos_2, 0, 0],
        [0, 0, 1, 0]
    ]],
    [[
        [deg_cos_2, -deg_sin_2, 0, 0],
        [deg_sin_2, deg_cos_2, 0, 0],
        [0, 0, 1, 0]
    ]]]
    a = torch.Tensor(matrix_1)
    b = torch.Tensor(matrix_2)
    print(a.shape)
    print(b.shape)
    ax_ang, translation_diff = transformation_diff(a, b)
    print(ax_ang / math.pi * 180)
    print(translation_diff)

if __name__ == "__main__":
    test_angle()
