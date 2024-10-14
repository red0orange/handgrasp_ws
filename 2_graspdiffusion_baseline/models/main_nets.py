import time
import functools

import trimesh.transformations as tra
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d

from .components import PointNetPlusPlus, UNetPoseNet, TransformerPoseNet
from .pointnet_util import PointNetSetAbstractionMsg
from utils.rotate_rep import *


def interpolate_3d_points(p1, p2, n):
    t = torch.linspace(0, 1, n, device=p1.device).float()
    points = (1 - t)[:, None] * p1 + t[:, None] * p2
    return points


def get_gripper_control_points():
    # panda?
    return np.array([
        [-0.10, 0, 0, 1],
        [-0.03, 0, 0, 1],
        [-0.03, 0.07, 0, 1],
        [0.03, 0.07, 0, 1],
        [-0.03, 0.07, 0, 1],
        [-0.03, -0.07, 0, 1],
        [0.03, -0.07, 0, 1]])
    # # robotiq_85
    # return np.array([
    #     [-0.08, 0, 0, 1],
    #     [-0.03, 0, 0, 1],
    #     [-0.03, 0.055, 0, 1],
    #     [0.03, 0.055, 0, 1],
    #     [-0.03, 0.055, 0, 1],
    #     [-0.03, -0.055, 0, 1],
    #     [0.03, -0.055, 0, 1]])
    

def get_gripper_keypoints_torch(grasps, scale=1):
    grasps = grasps.detach().clone()

    update_y_rot = tra.rotation_matrix(-np.pi / 2, [0, 1, 0])
    update_x_rot = tra.rotation_matrix(np.pi / 2, [1, 0, 0])
    T = tra.concatenate_matrices(update_y_rot, update_x_rot)
    T = torch.from_numpy(T).to(grasps.device).float()

    tmp_grasps = torch.einsum('bij,jk->bik', grasps, T)

    gripper_control_points = get_gripper_control_points()
    gripper_control_points = torch.from_numpy(gripper_control_points).to(grasps.device)
    gripper_control_points[:, :3] = scale * gripper_control_points[:, :3]
    gripper_control_points = gripper_control_points.float()

    grasp_keypoints = torch.einsum('bij,jk->bik', tmp_grasps, gripper_control_points.T).transpose(1, 2)[..., :3]
    return grasp_keypoints


def linear_diffusion_schedule(betas, T):
    """_summary_
    Linear cheduling for sampling in training.
    """
    beta_t = (betas[1] - betas[0]) * torch.arange(0, T + 1, dtype=torch.float32) / T + betas[0]
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


def marginal_prob_std(t, sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """    
    t = t.clone()
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
    
    Returns:
        The vector of diffusion coefficients.
    """
    return (sigma**t).clone()


def cal_grasp_refine_score(g_i, g):
    class GraspRefineScoreNet(nn.Module):
        def __init__(self, g_i, loss_rot_weight=1.0, loss_trans_weight=1.0):
            super(GraspRefineScoreNet, self).__init__()
            self.g_i = g_i
            self.loss_rot_weight = loss_rot_weight
            self.loss_trans_weight = loss_trans_weight

            self.loss_mse_rot = nn.MSELoss()
            self.loss_mse_trans = nn.MSELoss()
            pass

        def forward(self, g):
            g_rotation = g[:, :6]
            g_translation = g[:, 6:]

            g_i_rotation = self.g_i[:6][None, ...].repeat(g.shape[0], 1)
            g_i_translation = self.g_i[6:][None, ...].repeat(g.shape[0], 1)

            rot_loss = self.loss_mse_rot(g_rotation, g_i_rotation)
            trans_loss = self.loss_mse_trans(g_translation, g_i_translation)

            return self.loss_rot_weight * rot_loss + self.loss_trans_weight * trans_loss

    g = g.clone().detach()
    g.requires_grad_(True)
    net = GraspRefineScoreNet(g_i, 2.0, 1.0)
    loss = net(g)
    loss.backward()

    return -g.grad


def new_cal_grasp_refine_score(g_i, g):
    class GraspRefineScoreNet(nn.Module):
        def __init__(self, g_i, loss_rot_weight=1.0, loss_trans_weight=1.0):
            super(GraspRefineScoreNet, self).__init__()
            self.g_i = g_i
            self.loss_rot_weight = loss_rot_weight
            self.loss_trans_weight = loss_trans_weight

            self.loss_mse_rot = nn.MSELoss()
            self.loss_mse_trans = nn.MSELoss()
            pass

        def forward(self, g):

            g_rotation = g[:, :6]
            g_translation = g[:, 6:]

            g_i_rotation = self.g_i[:6][None, ...].repeat(g.shape[0], 1)
            g_i_translation = self.g_i[6:][None, ...].repeat(g.shape[0], 1)

            g_rotation_matrix = rotation_6d_to_matrix(g_rotation)
            g_T = torch.eye(4, device=g.device)[None, :, :].repeat(g.shape[0], 1, 1)
            g_T[:, :3, :3] = g_rotation_matrix
            g_T[:, :3, 3] = g_translation
            gripper_depth_T = torch.eye(4, device=g.device)
            # gripper_depth_T[2, 3] = -0.08
            g_T = torch.einsum('bij,jk->bik', g_T, gripper_depth_T)
            g_translation = g_T[:, :3, 3]

            g_i_rotation_matrix = rotation_6d_to_matrix(g_i_rotation)

            g_rotation_z_axis = g_rotation_matrix[:, :3, 2]
            g_i_rotation_z_axis = g_i_rotation_matrix[:, :3, 2]
            eps = 1e-5
            g_rotation_z_axis = g_rotation_z_axis / (torch.norm(g_rotation_z_axis, dim=1, keepdim=True) + eps)
            g_i_rotation_z_axis = g_i_rotation_z_axis / (torch.norm(g_i_rotation_z_axis, dim=1, keepdim=True) + eps)
            z_angles = torch.acos(torch.clamp(torch.sum(g_rotation_z_axis * g_i_rotation_z_axis, dim=1), -1 + eps, 1 - eps))

            # rot_loss = self.loss_mse_rot(g_rotation, g_i_rotation)
            rot_loss = torch.mean(z_angles)
            trans_loss = self.loss_mse_trans(g_translation, g_i_translation)
            print(trans_loss)

            return self.loss_rot_weight * rot_loss + self.loss_trans_weight * trans_loss

    g = g.clone().detach()
    g.requires_grad_(True)
    net = GraspRefineScoreNet(g_i, 0.1, 5.0)
    loss = net(g)
    loss.backward()

    return -g.grad


class GraspEvalNet(nn.Module):
    def __init__(self):
        super(GraspEvalNet, self).__init__()
        pass


class ScoreBasedGraspingDiffusion(nn.Module):
    def __init__(self, betas, n_T, device, drop_prob=0.1, rot6d_rep=True, unet_or_transformer="unet", training_method="ddpm", sigma=0.07, grasp_obj_joint_embed=True):
        super(ScoreBasedGraspingDiffusion, self).__init__()

        self.unet_or_transformer = unet_or_transformer
        self.training_method = training_method
        self.rot6d_rep = rot6d_rep
        self.action_dim = 9 if self.rot6d_rep else 7
        self.grasp_obj_joint_embed = grasp_obj_joint_embed

        self.n_T = n_T
        self.eps = 1e-5
        self.num_steps = 1000
        self.signal_to_noise_ratio = 0.16
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

        self.obj_pc_embed_dim = 1024
        self.grasp_pc_embed_dim = 64

        pointnet_dim = self.obj_pc_embed_dim
        if self.grasp_obj_joint_embed:
            pointnet_dim += self.grasp_pc_embed_dim
        if self.unet_or_transformer == "unet":
            self.posenet = UNetPoseNet(action_dim=self.action_dim, pointnet_dim=pointnet_dim)
        elif self.unet_or_transformer == "transformer":
            self.posenet = TransformerPoseNet(action_dim=self.action_dim, pointnet_dim=pointnet_dim)
        else:
            raise BaseException()

        self.pointnetplusplus = PointNetPlusPlus()
        if self.grasp_obj_joint_embed:
            self.grasp_point_encoder = torch.nn.Sequential(
                torch.nn.Linear(3, 64),
                torch.nn.GELU(),
            )
            # self.grasp_point_encoder = PointNetSetAbstractionMsg(16, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Register_buffer allows accessing dictionary, e.g. can access self.sqrtab later
        if self.training_method == "ddpm":
            for k, v in linear_diffusion_schedule(betas, n_T).items():
                self.register_buffer(k, v)
        elif self.training_method == "score_based":
            self.sigma = sigma
            self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=self.device)
            self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=self.device)
        else:
            raise BaseException()

    def forward(self, xyz, T, data_scale=1.0):
        """_summary_
        This method is used in training, so samples _ts and noise randomly.
        """
        B = T.shape[0]
        grasp_num = T.shape[1]

        # 转换 grasp pose 表征
        if self.rot6d_rep:
            flatten_T = T.reshape(-1, 4, 4)
            flatten_rotation = matrix_to_rotation_6d(flatten_T[:, :3, :3])
            rotation = flatten_rotation.reshape(B, -1, 6)
            translation = T[:, :, :3, 3]
        else:
            flatten_T = T.reshape(-1, 4, 4)
            flatten_rotation = matrix_to_quaternion(flatten_T[:, :3, :3])
            rotation = flatten_rotation.reshape(B, -1, 4)
            translation = T[:, :, :3, 3]

        g = torch.cat((rotation, translation), dim=2)
        g = g.float()

        # PointNet++ feature extraction
        T = T.float()
        xyz = xyz.float()
        obj_c = self.independent_obj_pc_embed(xyz, grasp_num)

        # @note 把 grasp_num 展平到 batch 维度统一处理
        B = B * grasp_num
        g = g.reshape(-1, self.action_dim)
        obj_c = obj_c.reshape(-1, obj_c.shape[-1])
            
        # 上面整理好 feature
        if self.training_method == "ddpm":
            _ts = torch.randint(1, self.n_T + 1, (B,)).to(self.device)
            noise = torch.randn_like(g)  # eps ~ N(0, 1), g size [B, action_dim]
            g_t = (
                self.sqrtab[_ts - 1, None] * g
                + self.sqrtmab[_ts - 1, None] * noise
            )  # This is the g_t, which is sqrt(alphabar) g_0 + sqrt(1-alphabar) * eps

            if self.grasp_obj_joint_embed:
                g_t = g_t.reshape(-1, grasp_num, g_t.shape[-1])
                grasp_c = self.independent_grasp_pc_embed(self.g2T(g_t), data_scale=data_scale)
                grasp_c = grasp_c.reshape(-1, grasp_c.shape[-1])
                c = torch.cat((obj_c, grasp_c), dim=1)
            else:
                c = obj_c

            # dropout context with some probability
            context_mask = torch.bernoulli(torch.zeros(B, 1) + 1 - self.drop_prob).to(self.device)
            # Loss for poseing is MSE between added noise, and our predicted noise
            pose_loss = self.loss_mse(noise, self.posenet(g_t, c, context_mask, _ts / self.n_T))
        elif self.training_method == "score_based":
            # score-based dropout 先不要
            context_mask = torch.ones(B, 1).float().to(self.device)
            random_t = torch.rand(g.shape[0], device=self.device) * (1. - self.eps) + self.eps
            z = torch.randn_like(g)
            std = self.marginal_prob_std_fn(random_t)
            perturbed_g = g + z * std[:, None]

            if self.grasp_obj_joint_embed:
                perturbed_g = perturbed_g.reshape(-1, grasp_num, perturbed_g.shape[-1])
                grasp_c = self.independent_grasp_pc_embed(self.g2T(perturbed_g), data_scale=data_scale)
                grasp_c = grasp_c.reshape(-1, grasp_c.shape[-1])
                perturbed_g = perturbed_g.reshape(-1, perturbed_g.shape[-1])
                c = torch.cat((obj_c, grasp_c), dim=1)
            else:
                c = obj_c

            score = self.posenet(perturbed_g, c, context_mask, random_t)
            pose_loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
        else:
            raise BaseException()

        return pose_loss
    
    def detect_and_sample(self, xyz, n_sample, guide_w, data_scale=1.0):
        """_summary_
        Detect affordance for one point cloud and sample [n_sample] poses that support the 'text' affordance task,
        following the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'.
        """
        if self.training_method == "ddpm":
            g_i = torch.randn(n_sample, (self.action_dim)).to(self.device) # start by sampling from Gaussian noise
            context_mask = torch.ones((n_sample, 1)).float().to(self.device)

            with torch.no_grad():
                obj_c = self.independent_obj_pc_embed(xyz, grasp_num=n_sample)
                for i in range(self.n_T, 0, -1):
                    if self.grasp_obj_joint_embed:
                        grasp_c = self.independent_grasp_pc_embed(self.g2T(g_i)[None, ...], data_scale=data_scale)
                        c = torch.cat((obj_c, grasp_c), dim=2)
                        c = c.reshape(-1, c.shape[-1])
                    else:
                        c = obj_c.reshape(-1, obj_c.shape[-1])
                    
                    # Double the batch
                    c_i = c_i.repeat(2, 1)
                    context_mask = context_mask.repeat(2, 1)
                    context_mask[n_sample:] = 0.    #@note make second half of the back context-free, 这是 classifer-free 的技巧，这样能增加平衡采样的多样性和有效性

                    _t_is = torch.tensor([i / self.n_T]).repeat(n_sample).repeat(2).to(self.device)
                    g_i = g_i.repeat(2, 1)
                    
                    z = torch.randn(n_sample, (self.action_dim)) if i > 1 else torch.zeros((n_sample, self.action_dim))
                    z = z.to(self.device)
                    eps = self.posenet(g_i, c_i, context_mask, _t_is)
                    eps1 = eps[:n_sample]
                    eps2 = eps[n_sample:]
                    eps = (1 + guide_w) * eps1 - guide_w * eps2
                    
                    g_i = g_i[:n_sample]
                    g_i = self.oneover_sqrta[i] * (g_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
                return g_i.detach().cpu().numpy()
        elif self.training_method == "score_based":
            t = torch.ones(n_sample, device=self.device)
            g_i = torch.randn(n_sample, (self.action_dim)).to(self.device) * self.marginal_prob_std_fn(t)[:, None]
            time_steps = torch.linspace(1., self.eps, self.num_steps, device=self.device)
            step_size = time_steps[0] - time_steps[1]
            g = g_i
            context_mask = torch.ones((n_sample, 1)).float().to(self.device)
            with torch.no_grad():
                obj_c = self.independent_obj_pc_embed(xyz, grasp_num=n_sample)
                for time_step in time_steps:
                    step_num = 1
                    if time_step < 0.2:  # for refine
                        step_num = 3

                    for i in range(step_num):
                        if self.grasp_obj_joint_embed:
                            T = self.g2T(g)[None, ...]
                            tmp_g = self.T2g(T, rot6d_rep=True)

                            grasp_c = self.independent_grasp_pc_embed(self.g2T(g)[None, ...], data_scale=data_scale)
                            c = torch.cat((obj_c, grasp_c), dim=2)
                            c = c.reshape(-1, c.shape[-1])
                        else:
                            c = obj_c.reshape(-1, obj_c.shape[-1])

                        batch_time_step = torch.ones(n_sample, device=self.device) * time_step
                        sde_g = self.diffusion_coeff_fn(batch_time_step)
                        # 下面是核心的采样公式
                        g_mean = g + (sde_g ** 2)[:, None] * self.posenet(g, c, context_mask, batch_time_step) * step_size
                        g = g_mean + torch.sqrt(step_size) * sde_g[:, None] * torch.randn_like(g)
                
                return g.detach().cpu().numpy()
            pass
        else:
            raise NotImplementedError()
    
    def batch_detect_and_sample(self, xyz, n_sample, guide_w, data_scale=1.0):
        """_summary_
        Detect affordance for one point cloud and sample [n_sample] poses that support the 'text' affordance task,
        following the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'.
        """
        if self.training_method == "ddpm":
            raise NotImplementedError()
        elif self.training_method == "score_based":
            batch_size = xyz.shape[0]
            t = torch.ones([batch_size * n_sample], device=self.device)
            g_i = torch.randn(batch_size * n_sample, (self.action_dim)).to(self.device) * self.marginal_prob_std_fn(t)[..., None]
            time_steps = torch.linspace(1., self.eps, self.num_steps, device=self.device)
            step_size = time_steps[0] - time_steps[1]
            g = g_i
            context_mask = torch.ones((batch_size * n_sample, 1)).float().to(self.device)
            with torch.no_grad():
                obj_c = self.independent_obj_pc_embed(xyz, grasp_num=n_sample)
                for time_step in time_steps:
                    step_num = 1
                    if time_step < 0.2:  # for refine
                        step_num = 3

                    for i in range(step_num):
                        if self.grasp_obj_joint_embed:
                            grasp_c = self.independent_grasp_pc_embed(self.g2T(g)[None, ...], data_scale=data_scale)
                            c = obj_c.reshape(-1, obj_c.shape[-1])
                            c = torch.cat((c, grasp_c.squeeze(0)), dim=1)
                        else:
                            c = obj_c.reshape(-1, obj_c.shape[-1])

                        batch_time_step = torch.ones(batch_size*n_sample, device=self.device) * time_step
                        sde_g = self.diffusion_coeff_fn(batch_time_step)
                        # 下面是核心的采样公式
                        g_mean = g + (sde_g ** 2)[:, None] * self.posenet(g, c, context_mask, batch_time_step) * step_size
                        g = g_mean + torch.sqrt(step_size) * sde_g[:, None] * torch.randn_like(g)
                
                res_g = g.reshape(batch_size, n_sample, -1)
                return res_g.detach().cpu().numpy()
            pass
        else:
            raise NotImplementedError()

    def refine_grasp_sample(self, xyz, n_sample, g_init, data_scale=1.0):
        if self.training_method == "ddpm":
            raise NotImplementedError()
        elif self.training_method == "score_based":
            t = torch.ones(n_sample, device=self.device)
            # g_i = torch.randn(n_sample, (self.action_dim)).to(self.device) * self.marginal_prob_std_fn(t)[:, None]
            g_i = g_init[None, :].repeat(n_sample, 1)
            time_steps = torch.linspace(1., self.eps, self.num_steps, device=self.device)
            step_size = time_steps[0] - time_steps[1]
            g = g_i
            context_mask = torch.ones((n_sample, 1)).float().to(self.device)

            with torch.no_grad():
                obj_c = self.independent_obj_pc_embed(xyz, grasp_num=n_sample)

            for time_step in time_steps:
                step_num = 1
                if time_step < 0.2:  # for refine
                    step_num = 3

                for i in range(step_num):
                    with torch.no_grad():
                        if self.grasp_obj_joint_embed:
                            grasp_c = self.independent_grasp_pc_embed(self.g2T(g)[None, ...], data_scale=data_scale)
                            c = torch.cat((obj_c, grasp_c), dim=2)
                            c = c.reshape(-1, c.shape[-1])
                        else:
                            c = obj_c.reshape(-1, obj_c.shape[-1])

                        batch_time_step = torch.ones(n_sample, device=self.device) * time_step
                        sde_g = self.diffusion_coeff_fn(batch_time_step)

                        score = self.posenet(g, c, context_mask, batch_time_step)

                    # refine cost
                    # grad = cal_grasp_refine_score(g_init, g)
                    grad = new_cal_grasp_refine_score(g_init, g)
                    refine_weight = 150.000
                    score = score + refine_weight * grad
                    
                    # 下面是核心的采样公式
                    g_mean = g + (sde_g ** 2)[:, None] * score  * step_size
                    g = g_mean + torch.sqrt(step_size) * sde_g[:, None] * torch.randn_like(g)
            
            return g.detach().cpu().numpy()
        else:
            raise NotImplementedError()

    def independent_grasp_pc_embed(self, grasp_T, data_scale=1.0):
        """
        grasp_pc: [B, grasp_num, gripper_kp_num, 3]

        return: [B, grasp_num, gripper_dim]
        """
        B = grasp_T.shape[0]
        grasp_num = grasp_T.shape[1]

        # @note 计算 gripper pc
        flatten_T = grasp_T.reshape(-1, 4, 4)
        flatten_grasp_pc = get_gripper_keypoints_torch(flatten_T, scale=data_scale).float()
        grasp_pc = flatten_grasp_pc.reshape(B, -1, flatten_grasp_pc.shape[-2], flatten_grasp_pc.shape[-1])

        # 
        gripper_top_p = flatten_grasp_pc[:, 0, :]
        gripper_center_p = flatten_grasp_pc[:, 1, :]
        gripper_left_top_p = flatten_grasp_pc[:, 2, :]
        gripper_left_down_p = flatten_grasp_pc[:, 3, :]
        gripper_right_top_p = flatten_grasp_pc[:, 5, :]
        gripper_right_down_p = flatten_grasp_pc[:, 6, :]

        t = torch.linspace(0, 1, 3, device=grasp_T.device).float()

        gripper_top_center_pc = (1 - t)[None, None, :] * gripper_top_p[..., None] + t[None, None, :] * gripper_center_p[..., None]
        gripper_center_left_pc = (1 - t)[None, None, :] * gripper_center_p[..., None] + t[None, None, :] * gripper_left_top_p[..., None]
        gripper_center_right_pc = (1 - t)[None, None, :] * gripper_center_p[..., None] + t[None, None, :] * gripper_right_top_p[..., None]
        gripper_left_pc = (1 - t)[None, None, :] * gripper_left_top_p[..., None] + t[None, None, :] * gripper_left_down_p[..., None]
        gripper_right_pc = (1 - t)[None, None, :] * gripper_right_top_p[..., None] + t[None, None, :] * gripper_right_down_p[..., None]
        gripper_top_center_pc = gripper_top_center_pc.transpose(1, 2)
        gripper_center_left_pc = gripper_center_left_pc.transpose(1, 2)
        gripper_center_right_pc = gripper_center_right_pc.transpose(1, 2)
        gripper_left_pc = gripper_left_pc.transpose(1, 2)
        gripper_right_pc = gripper_right_pc.transpose(1, 2)

        flatten_grasp_pc = torch.cat((gripper_top_center_pc, gripper_center_left_pc, gripper_center_right_pc, gripper_left_pc, gripper_right_pc), dim=1)

        # # debug vis
        # import open3d as o3d
        # from roboutils.vis.grasp import draw_scene
        # from utils.rotate_rep import rotation_6d_to_matrix_np

        # vis_pc = xyz[0].detach().cpu().numpy()
        # vis_Ts = T[0].detach().cpu().numpy()
        # vis_grasp_pc = flatten_grasp_pc[0].detach().cpu().numpy()

        # grasp_pc_o3d = o3d.geometry.PointCloud()
        # grasp_pc_o3d.points = o3d.utility.Vector3dVector(vis_grasp_pc)
        # # 0: gripper 顶点，2, 3: 左爪，5, 6: 右爪, 1, 4: 中间
        # # grasp_pc_o3d.colors = o3d.utility.Vector3dVector([[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], 
        #                                                 #   [0, 0, 1], [0, 0, 0], [0, 0, 0]])
        # # grasp_pc_o3d.paint_uniform_color([1, 0, 0])

        # vis_pc_o3d = o3d.geometry.PointCloud()
        # vis_pc_o3d.points = o3d.utility.Vector3dVector(vis_pc)
        # vis_pc_o3d.paint_uniform_color([0, 1, 0])

        # o3d.visualization.draw_geometries([grasp_pc_o3d, vis_pc_o3d])
        # draw_scene(vis_pc, vis_Ts, z_direction=True, scale=1.0/data_scale)


        # @note 开始 encode
        # 实现 1
        # flatten_grasp_pc = flatten_grasp_pc.contiguous().transpose(1, 2)
        # flatten_grasp_c = self.grasp_point_encoder(flatten_grasp_pc, flatten_grasp_pc)[1]   # [B * grasp_num, 7, 256]
        # flatten_grasp_c = flatten_grasp_c.transpose(1, 2)

        flatten_grasp_c = self.grasp_point_encoder(flatten_grasp_pc)   # [B * grasp_num, 7, 64]

        flatten_grasp_c = torch.max(flatten_grasp_c, dim=1)[0]  # [B * grasp_num, 256]
        grasp_c = flatten_grasp_c.reshape(B, grasp_num, flatten_grasp_c.shape[-1])
        
        return grasp_c
    
    def independent_obj_pc_embed(self, xyz, grasp_num):
        """
        xyz     : [B, obj_pc_num, 3]

        return: [B, grasp_num, pointnet_dim]
        """
        B = xyz.shape[0]
        _, obj_c = self.pointnetplusplus(xyz)                          # c'size [B, 1024]
        obj_c = obj_c.reshape(B, obj_c.shape[-1])
        obj_c = obj_c.unsqueeze(1).repeat(1, grasp_num, 1)
        return obj_c
    
    def joint_embed_grasp_obj(self, grasp_T, xyz, data_scale=1.0):
        """
        grasp_pc: [B, grasp_num, gripper_kp_num, 3]
        xyz     : [B, obj_pc_num, 3]

        return: [B, grasp_num, pointnet_dim+gripper_dim]
        """
        B = grasp_T.shape[0]
        grasp_num = grasp_T.shape[1]

        # @note 方法 1：分别独立编码，然后拼接。这样是能够保证训练速度，但是可能会导致 joint embed 的效果不好
        grasp_c = self.independent_grasp_pc_embed(grasp_T, data_scale=data_scale)
        obj_c   = self.independent_obj_pc_embed(xyz, grasp_num)
        c = torch.cat((obj_c, grasp_c), dim=2)
        return c

    @staticmethod
    def g2T(g):
        """_summary_
        Convert grasp pose representation to transformation matrix.
        g: [..., 7] or [..., 9]
        """
        ori_shape = g.shape
        g = g.reshape(-1, g.shape[-1])
        if g.shape[-1] == 7:
            rotation = quaternion_to_matrix(g[:, :4])
            translation = g[:, 4:]
        elif g.shape[-1] == 9:
            rotation = rotation_6d_to_matrix(g[:, :6])
            translation = g[:, 6:]
        else:
            raise NotImplementedError()
        T = torch.eye(4, device=g.device)[None, ...].repeat(g.shape[0], 1, 1)
        T[:, :3, :3] = rotation
        T[:, :3, 3] = translation
        T = T.reshape(ori_shape[:-1] + (4, 4))
        return T
    
    @staticmethod
    def T2g(T, rot6d_rep=True):
        """_summary_
        Convert transformation matrix to grasp pose representation.
        T: [..., 4, 4]
        """
        ori_shape = T.shape
        T = T.reshape(-1, 4, 4)
        if rot6d_rep:
            rotation = matrix_to_rotation_6d(T[:, :3, :3])
            translation = T[:, :3, 3]
            g = torch.cat((rotation, translation), dim=1)
        else:
            rotation = matrix_to_quaternion(T[:, :3, :3])
            translation = T[:, :3, 3]
            g = torch.cat((rotation, translation), dim=1)
        g = g.reshape(ori_shape[:-2] + (g.shape[-1],))
        return g