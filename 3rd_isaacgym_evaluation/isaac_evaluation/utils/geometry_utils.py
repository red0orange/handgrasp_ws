from isaacgym import gymapi,gymtorch
import torch
import numpy as np

import pytorch3d.transforms as p3d_tf


def T2sevendof(T):
    wxyz = p3d_tf.matrix_to_quaternion(T[:3, :3][None, ...])[0]
    xyz = T[:3, 3]
    return xyz, wxyz


def sevendof2T(xyz, wxyz):
    xyz, wxyz = torch.tensor(xyz), torch.tensor(wxyz)
    rot_mat = p3d_tf.quaternion_to_matrix(wxyz[None, ...])[0]

    res = torch.eye(4)
    res[:3, :3] = rot_mat
    res[:3, 3] = xyz

    return res


def H_2_Transform(H):
    trans, quat = T2sevendof(H)

    p = trans
    q = quat

    p = gymapi.Vec3(x=p[0], y=p[1], z=p[2])
    q = gymapi.Quat(w=q[0], x=q[1], y=q[2], z=q[3])  # wxyz

    return gymapi.Transform(p, q)

def Transform_2_H(T):
    p = [T.p.x, T.p.y, T.p.z]
    q = [T.r.w, T.r.x, T.r.y, T.r.z]
    H = sevendof2T(p, q)
    return H

def pq_to_H(p, q):
    # expects as input: quaternion with convention [x y z w]
    # arrange quaternion with convention [w x y z] for theseus
    q = torch.tensor([q[3], q[0], q[1], q[2]])
    return sevendof2T(p, q)