import os
import time
from os.path import join as opj
import pickle
import sys
import subprocess
import shutil
import datetime
from gorilla.config import Config

from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
import torch

from utils import *
from models.graspdiff.main_nets import load_graspdiff
from models.graspdiff.losses import get_loss_fn
from models.graspdiff.learning_rate import get_lr_scheduler
from models.graspdiff.sampler import Constrained_Grasp_AnnealedLD
from dataset._CONGDiffDataet import _CONGDiffDataset

from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose
from my_eval_oakink_only_s import OakinkGraspDataset


def eval_for_isaacgym(work_dir):
    save_name = "eval_oakink_direct_multigripper"
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")
    
    GUIDE_W = 0.5
    DEVICE=torch.device('cuda')

    cfg = Config.fromfile(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = load_graspdiff(cfg.training_cfg.feature_backbone)
    dataset = build_dataset(cfg)['test_set']
    
    print("Loading checkpoint....")
    _, exten = os.path.splitext(checkpoint_path)
    if exten == '.t7':
        model.load_state_dict(torch.load(checkpoint_path))
    elif exten == '.pth':
        model.load_state_dict(torch.load(checkpoint_path))
        # check = torch.load(checkpoint_path)
        # model.load_state_dict(check['model_state_dict'])

    print("Evaluating")
    viser_for_grasp = ViserForGrasp()
    model.eval()

    # @note 加载 oakink 数据集
    oakink_data_root = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_Oakink"
    oakink_dataset = OakinkGraspDataset(oakink_data_root, dataset)
    dataloader = torch.utils.data.DataLoader(oakink_dataset, batch_size=1, shuffle=False, num_workers=16)

    results = []
    # 采样 100 个，每 100 个作为一次预测，从每 10 个预测中选出最佳的 1 个
    # each_time_sample_num = 100
    each_time_sample_num = 10
    eval_time_num = 1
    sample_num = each_time_sample_num * eval_time_num
    device = "cuda"
    generator = Constrained_Grasp_AnnealedLD(model, batch=sample_num, T=70, T_fit=50, k_steps=2, device=device)
    for data in tqdm(dataloader, total=len(dataloader)):
        file_paths, mesh_paths, obj_verts, mesh_T, my_palm_Ts, hand_verts = data[0], data[1], data[2], data[3], data[4], data[5]
        file_paths = file_paths[0]
        mesh_paths = mesh_paths[0]
        xyz = obj_verts
        batch = xyz.shape[0]

        # 处理 palm_T
        refine_palm_Ts = []
        for i in range(len(my_palm_Ts)):
            cur_obj_verts = obj_verts[i].cpu().numpy()
            cur_hand_verts = hand_verts[i].cpu().numpy()
            my_palm_T = my_palm_Ts[i].cpu().numpy()

            obj_kdtree = KDTree(cur_obj_verts)
            palm_grasp_center = my_palm_T[:3, 3]
            distances, indices = obj_kdtree.query(palm_grasp_center, k=96)

            refine_palm_T = my_palm_T.copy()
            refine_palm_T[:3, 3] = cur_obj_verts[indices[0]]  # 距离最近的点作为抓取中心
            refine_palm_Ts.append(refine_palm_T)

            # # debug
            # closest_pcs = cur_obj_verts[indices]
            # viser_for_grasp.add_pcd(closest_pcs, colors=np.array([(255, 0, 0)]*closest_pcs.shape[0]))
            # viser_for_grasp.add_pcd(cur_hand_verts, colors=np.array([(0, 255, 0)]*cur_hand_verts.shape[0]))
            # # viser_for_grasp.add_pcd(obj_verts, colors=np.array([(0, 0, 255)]*obj_verts.shape[0]))
            # viser_for_grasp.add_grasp(my_palm_T, z_direction=True)
            # viser_for_grasp.add_grasp(refine_palm_T, z_direction=True)
            # print("wait for reset")
            # viser_for_grasp.wait_for_reset()
        refine_palm_Ts = np.array(refine_palm_Ts)

        # # @note 开始预测
        # start_time = time.time()
        # xyz = xyz.float().cuda()
        # model.set_latent(xyz, batch=sample_num)
        # input_constrained_H = torch.from_numpy(refine_palm_Ts).float().cuda()
        # grasp_Ts, vis_grasp_Ts = generator.constrained_batch_sample(batch=batch, sample_num=sample_num, constrained_H=input_constrained_H, return_vis_Hs=True)
        # end_time = time.time()
        # print("Time cost: {:.3f}s".format((end_time - start_time) / batch))

        for refine_i, refine_palm_T in enumerate(refine_palm_Ts):
            refine_palm_Ts[refine_i] = update_pose(refine_palm_T, rotate=-90, rotate_axis='y')
            refine_palm_Ts[refine_i] = update_pose(refine_palm_Ts[refine_i], rotate=-90, rotate_axis='z')
            refine_palm_Ts[refine_i] = update_pose(refine_palm_Ts[refine_i], translate=[0, 0, -0.08*dataset.scale])
            
        grasp_Ts = torch.from_numpy(refine_palm_Ts)[None, ...]
        grasp_Ts = grasp_Ts.repeat(1, each_time_sample_num, 1, 1)
        grasp_Ts[...,:3,3] /= dataset.scale

        for batch_i in range(batch):
            batch_i_filename = file_paths[batch_i]
            batch_i_mesh_path = mesh_paths[batch_i]
            batch_i_xyz = xyz[batch_i]
            batch_i_grasp_Ts = grasp_Ts[batch_i]
            batch_i_mesh_T = mesh_T[batch_i].cpu().numpy()
            batch_i_palm_T = my_palm_Ts[batch_i].cpu().numpy()

            batch_i_xyz = batch_i_xyz.cpu().numpy()
            batch_i_grasp_Ts = batch_i_grasp_Ts.cpu().numpy()
            vis_palm_T = batch_i_palm_T.copy()

            vis_xyz, vis_grasp_Ts = batch_i_xyz.copy(), batch_i_grasp_Ts.copy()
            vis_xyz /= dataset.scale
            vis_palm_T[:3, 3] /= dataset.scale
            batch_i_mesh_T[:3, 3] /= dataset.scale
            vis_hand_verts = hand_verts[batch_i].cpu().numpy()
            
            selected_grasp_Ts = vis_grasp_Ts

            # # debug vis
            # # viser_for_grasp.vis_grasp_scene(vis_grasp_Ts, pc=vis_xyz, max_grasp_num=1000)
            # # viser_for_grasp.wait_for_reset()
            # viser_for_grasp.vis_grasp_scene(selected_grasp_Ts, pc=vis_xyz, max_grasp_num=1000)
            # viser_for_grasp.add_pcd(vis_hand_verts, colors=np.array([(0, 255, 0)]*vis_hand_verts.shape[0]))
            # viser_for_grasp.add_crood(selected_grasp_Ts[0])
            # # viser_for_grasp.add_grasp(vis_palm_T, z_direction=True)
            # viser_for_grasp.wait_for_reset()

            data_dict = {
                'filename': batch_i_filename,
                'mesh_path': batch_i_mesh_path,
                'xyz': vis_xyz,
                'grasp_Ts': selected_grasp_Ts,
                'mesh_T': batch_i_mesh_T,
            }
            results.append(data_dict)
    with open(os.path.join(work_dir, '{}.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(results, f)
    # # exit()
    
    results = pickle.load(open(os.path.join(work_dir, '{}.pkl'.format(save_name)), 'rb'))
    
    # 将结果转换为 IsaacGym 格式
    isaacgym_eval_data_dict = {}
    for i, data_dict in tqdm(enumerate(results), total=len(results)):
        isaacgym_eval_data_dict[i] = {}

        obj_pc = data_dict['xyz']
        mesh_T = data_dict['mesh_T']
        mesh_path = data_dict['mesh_path']
        mesh_scale = 1.0

        isaacgym_eval_data_dict[i]['mesh_path'] = mesh_path
        isaacgym_eval_data_dict[i]['mesh_scale'] = mesh_scale
        isaacgym_eval_data_dict[i]['grasp_Ts'] = data_dict['grasp_Ts']
        isaacgym_eval_data_dict[i]['mesh_T'] = mesh_T

        # # for debug vis
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # mesh.scale(mesh_scale, center=np.zeros(3))
        # mesh.transform(mesh_T)
        # grasp_Ts = data_dict['grasp_Ts']
        # viser_for_grasp.vis_grasp_scene(grasp_Ts, mesh=mesh, pc=obj_pc, max_grasp_num=50)
        # viser_for_grasp.wait_for_reset()

    np.save(os.path.join(work_dir, '{}_isaacgym.npy'.format(save_name)), isaacgym_eval_data_dict)
    pass


if __name__ == '__main__':
    save_name = "graspdiff_eval_cong"
    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_499_20241029-103352_grasp_diffusion_baseline"
    eval_for_isaacgym(work_dir)