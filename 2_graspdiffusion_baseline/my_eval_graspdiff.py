import os
from os.path import join as opj
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
    work_dir = "/home/huangdehao/Projects/handgrasp_ws/2_graspdiffusion_baseline/log/epoch_31_20241024-211612_grasp_diffusion_baseline copy"
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")

    cfg = Config.fromfile(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = load_graspdiff()
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

        obj_pc = model_input['visual_context']
        # grasp_Ts = model_input['x_ene_pos']
        obj_sdf = gt['sdf']
        obj_sdf_xyz = model_input['x_sdf']

        obj_pc = obj_pc.to(device)
        obj_pc = obj_pc[0][None, ...]
        model.set_latent(obj_pc, batch=grasp_per_obj)
        sample_grasp_Ts = generator.sample()

        # vis
        vis_obj_pc = obj_pc.squeeze().cpu().numpy()
        sample_grasp_Ts = sample_grasp_Ts.squeeze().cpu().numpy()
        vis_obj_pc /= 8.
        sample_grasp_Ts[:, :3, 3] /= 8.
        viser_for_grasp.vis_grasp_scene(sample_grasp_Ts, pc=vis_obj_pc, max_grasp_num=50, z_direction=True)
        viser_for_grasp.wait_for_reset()



        # scale = model_input['scale'].numpy()[0]
        # mesh_T = model_input['mesh_T'].numpy()

        pass

    pass


# def parse_args():
#     p = configargparse.ArgumentParser()
#     p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

#     p.add_argument('--obj_id', type=int, default=0)
#     p.add_argument('--n_grasps', type=str, default='200')
#     p.add_argument('--obj_class', type=str, default='Laptop')
#     p.add_argument('--device', type=str, default='cuda:0')
#     p.add_argument('--eval_sim', type=bool, default=False)
#     p.add_argument('--model', type=str, default='grasp_dif_multi')


#     opt = p.parse_args()
#     return opt


# def get_approximated_grasp_diffusion_field(p, args, device='cpu'):
#     model_params = args.model
#     batch = 50
#     ## Load model
#     model_args = {
#         'device': device,
#         'pretrained_model': model_params
#     }
#     model = load_model(model_args)

#     context = to_torch(p[None,...], device)
#     model.set_latent(context, batch=batch)

#     ########### 2. SET SAMPLING METHOD #############
#     generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

#     return generator, model


# def sample_pointcloud(obj_id=0, obj_class='Mug'):
#     grasps_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#     grasps_dir = os.path.join(grasps_dir, 'data', 'grasps')
#     acronym_grasps = AcronymGraspsDirectory(filename=grasps_dir, data_type=obj_class)
#     mesh = acronym_grasps.avail_obj[obj_id].load_mesh()

#     P = mesh.sample(1000)

#     sampled_rot = scipy.spatial.transform.Rotation.random()
#     rot = sampled_rot.as_matrix()
#     rot_quat = sampled_rot.as_quat()

#     P = np.einsum('mn,bn->bm', rot, P)
#     P *= 8.
#     P_mean = np.mean(P, 0)
#     P += -P_mean

#     H = np.eye(4)
#     H[:3,:3] = rot
#     mesh.apply_transform(H)
#     mesh.apply_scale(8.)
#     H = np.eye(4)
#     H[:3,-1] = -P_mean
#     mesh.apply_transform(H)
#     translational_shift = copy.deepcopy(H)



#     return P, mesh, translational_shift, rot_quat


# if __name__ == '__main__':
#     import copy
#     import configargparse
#     args = parse_args()

#     EVAL_SIMULATION = args.eval_sim
#     # isaac gym has to be imported here as it is supposed to be imported before torch
#     if (EVAL_SIMULATION):
#         # Alternatively: Evaluate Grasps in Simulation:
#         from isaac_evaluation.grasp_quality_evaluation import GraspSuccessEvaluator

#     import scipy.spatial.transform
#     import numpy as np
#     from se3dif.datasets import AcronymGraspsDirectory
#     from se3dif.models.loader import load_model
#     from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
#     from se3dif.utils import to_numpy, to_torch

#     import torch
#     import open3d as o3d



#     print('##########################################################')
#     print('Object Class: {}'.format(args.obj_class))
#     print(args.obj_id)
#     print('##########################################################')

#     n_grasps = int(args.n_grasps)
#     obj_id = int(args.obj_id)
#     obj_class = args.obj_class
#     n_envs = 30
#     device = args.device

#     ## Set Model and Sample Generator ##
#     P, mesh, trans, rot_quad = sample_pointcloud(obj_id, obj_class)
#     generator, model = get_approximated_grasp_diffusion_field(P, args, device)

#     o3d_pcd = o3d.geometry.PointCloud()
#     o3d_pcd.points = o3d.utility.Vector3dVector(P)
#     o3d.visualization.draw_geometries([o3d_pcd])    

#     H = generator.sample()

#     H_grasp = copy.deepcopy(H)
#     # counteract the translational shift of the pointcloud (as the spawned model in simulation will still have it)
#     H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()
#     H[..., :3, -1] *=1/8.
#     H_grasp[..., :3, -1] *=1/8.

#     ## Visualize results ##
#     from se3dif.visualization import grasp_visualization

#     vis_H = H.squeeze()
#     P *=1/8
#     mesh = mesh.apply_scale(1/8)
#     grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P, mesh=mesh)

#     if (EVAL_SIMULATION):
#         ## Evaluate Grasps in Simulation##
#         num_eval_envs = 10
#         evaluator = GraspSuccessEvaluator(obj_class, n_envs=num_eval_envs, idxs=[args.obj_id] * num_eval_envs, viewer=True, device=device, \
#                                           rotations=[rot_quad]*num_eval_envs, enable_rel_trafo=False)
#         succes_rate = evaluator.eval_set_of_grasps(H_grasp)
#         print('Success cases : {}'.format(succes_rate))
