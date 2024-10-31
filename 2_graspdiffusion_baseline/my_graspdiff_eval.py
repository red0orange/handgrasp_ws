import os
from os.path import join as opj
import pickle
import sys
import subprocess
import shutil
import datetime
from gorilla.config import Config

from tqdm import tqdm
import numpy as np
import torch

from utils import *
from models.graspdiff.main_nets import load_graspdiff
from models.graspdiff.losses import get_loss_fn
from models.graspdiff.learning_rate import get_lr_scheduler
from models.graspdiff.sampler import Grasp_AnnealedLD
from dataset._CONGDiffDataet import _CONGDiffDataset

from roboutils.vis.viser_grasp import ViserForGrasp


if __name__ == '__main__':
    save_name = "graspdiff_eval_cong"
    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_499_20241029-190130_grasp_diffusion_baseline"
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")

    cfg = Config.fromfile(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    feature_backbone = cfg.training_cfg.feature_backbone
    model = load_graspdiff(feature_backbone=feature_backbone)
    dataset = build_dataset(cfg)['test_set']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    print("Loading checkpoint....")
    _, exten = os.path.splitext(checkpoint_path)
    if exten == '.t7':
        model.load_state_dict(torch.load(checkpoint_path))
    elif exten == '.pth':
        model.load_state_dict(torch.load(checkpoint_path))
    
    # 开始采样
    print("Evaluating")
    viser_for_grasp = ViserForGrasp()
    model.eval()
    results = []
    grasp_per_obj = 10
    device = "cuda"

    # 初始化采样器
    generator = Grasp_AnnealedLD(model, batch=grasp_per_obj, T=70, T_fit=50, k_steps=2, device=device)
    for data in tqdm(dataloader, total=len(dataloader)):
        model_input = data[0]
        gt = data[1]
        batch = model_input['visual_context'].shape[0]

        obj_pc = model_input['visual_context']
        # grasp_Ts = model_input['x_ene_pos']
        obj_sdf = gt['sdf']
        obj_sdf_xyz = model_input['x_sdf']
        mesh_T = model_input['mesh_T']
        filename = model_input['filename']

        obj_pc = obj_pc.to(device)
        # obj_pc = obj_pc[1][None, ...]
        model.set_latent(obj_pc, batch=grasp_per_obj)
        # sample_grasp_Ts = generator.sample()
        sample_grasp_Ts = generator.batch_sample(batch=batch, sample_num=grasp_per_obj)

        for batch_i in range(batch):
            batch_i_mesh_T = mesh_T[batch_i].squeeze().cpu().numpy()
            batch_i_filename = filename[batch_i]

            vis_obj_pc = obj_pc[batch_i].squeeze().cpu().numpy()
            sample_grasp_Ts_i = sample_grasp_Ts[batch_i].squeeze().cpu().numpy()
            vis_obj_pc /= dataset.scale
            sample_grasp_Ts_i[:, :3, 3] /= dataset.scale
            batch_i_mesh_T[:3, 3] /= dataset.scale

            # # debug
            # viser_for_grasp.vis_grasp_scene(sample_grasp_Ts_i, pc=vis_obj_pc, max_grasp_num=50, z_direction=True)
            # viser_for_grasp.wait_for_reset()
        
            data_dict = {
                'filename': batch_i_filename,
                'xyz': vis_obj_pc,
                'grasp_Ts': sample_grasp_Ts_i,
                'mesh_T': batch_i_mesh_T,
            }
            results.append(data_dict)
        pass

    with open(os.path.join(work_dir, '{}.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(results, f)
    # # exit()
    
    results = pickle.load(open(os.path.join(work_dir, '{}.pkl'.format(save_name)), 'rb'))
    
    # 将结果转换为 IsaacGym 格式
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_root = os.path.join(proj_dir, 'data/obj_ShapeNetSem/models-OBJ/models')
    isaacgym_eval_data_dict = {}
    for i, data_dict in tqdm(enumerate(results), total=len(results)):
        isaacgym_eval_data_dict[i] = {}

        pickle_data_path = data_dict['filename']
        obj_pc = data_dict['xyz']
        mesh_T = data_dict['mesh_T']
        pickle_data = pickle.load(open(pickle_data_path, 'rb'))
        mesh_fname = os.path.basename(pickle_data["mesh/file"])
        mesh_path = os.path.join(mesh_root, mesh_fname)
        mesh_scale = pickle_data["mesh/scale"]

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