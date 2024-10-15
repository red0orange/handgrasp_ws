import os
import pickle
from tqdm import tqdm

import open3d as o3d
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from gorilla.config import Config

from utils import *
from utils.rotate_rep import rotation_6d_to_matrix_np
from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


if __name__ == "__main__":
    GUIDE_W = 0.5
    DEVICE=torch.device('cuda')
    eval_save_name = "scorenet_with_energy_sort_eval"

    # @note 初始化 energynet 和 scorenet 模型
    energynet_work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241014-211722_detectiondiffusion"
    config_file_path = os.path.join(energynet_work_dir, "config.py")
    checkpoint_path = os.path.join(energynet_work_dir, "current_model.t7")
    cfg = Config.fromfile(config_file_path)
    energy_model = build_model(cfg).to(DEVICE)

    # score net
    scorenet_work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion"
    config_file_path = os.path.join(scorenet_work_dir, "config.py")
    checkpoint_path = os.path.join(scorenet_work_dir, "current_model.t7")
    # checkpoint_path = os.path.join(work_dir, "model_epoch_75.pth")
    
    cfg = Config.fromfile(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = build_model(cfg).to(DEVICE)
    dataset = build_dataset(cfg)['test_set']
    # dataset = build_dataset(cfg)['train_set']
    rot6d_rep = cfg.hyper_params.rot6d_rep
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    
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
    results = []
    grasp_per_obj = 40  # 采集 40 个，根据 energy 排序选择 top 10 个
    for data in tqdm(dataloader, total=len(dataloader)):
        # ori_xyz, xyz, gt_T, _, _ = data
        filename, xyz, gt_T, mesh_T = data[0], data[1], data[2], data[3]
        filename = filename[0]

        xyz = xyz.float().cuda()
        pred_g = model.batch_detect_and_sample(xyz, grasp_per_obj, guide_w=GUIDE_W, data_scale=dataset.scale)
        
        energy_pred_g = pred_g.copy()
        energy_pred_g = torch.from_numpy(energy_pred_g).float().cuda()
        energy = energy_model.batch_estimate_energy(xyz, energy_pred_g, data_scale=dataset.scale)

        xyz = xyz.cpu().numpy()

        # 根据 energy 排序选择 top 10 个
        # pred_g: (batch_size, num_grasp, 9)
        # energy: (batch_size, num_grasp)
        batch_size, grasp_num, _ = pred_g.shape
        sorted_idx = np.argsort(-energy, axis=1)[:, :10]
        selected_pred_g = pred_g[np.arange(batch_size)[:, None], sorted_idx]
        pred_g = selected_pred_g

        batch_size = pred_g.shape[0]
        num_grasp = pred_g.shape[1]
        pred_g = pred_g.reshape(-1, pred_g.shape[-1])  # 先展平
        num_pose = pred_g.shape[0]
        if not rot6d_rep:
            raise NotImplementedError
        else:
            rotation = rotation_6d_to_matrix_np(pred_g[:, :6])
            rotation = np.concatenate((rotation, np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
            translation = np.expand_dims(np.concatenate((pred_g[:, 6:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
        grasp_Ts = np.concatenate((rotation, translation), axis=2)
        grasp_Ts = grasp_Ts.reshape(batch_size, num_grasp, 4, 4)

        for batch_i in range(batch_size):
            bathc_i_filename = filename[batch_i]
            batch_i_xyz = xyz[batch_i]
            batch_i_grasp_Ts = grasp_Ts[batch_i]
            batch_i_mesh_T = mesh_T[batch_i].cpu().numpy()

            vis_xyz, vis_grasp_Ts = batch_i_xyz.copy(), batch_i_grasp_Ts.copy()
            vis_xyz /= dataset.scale
            vis_grasp_Ts[:, :3, 3] /= dataset.scale
            batch_i_mesh_T[:3, 3] /= dataset.scale

            # # debug vis
            # viser_for_grasp.vis_grasp_scene(vis_grasp_Ts, pc=vis_xyz, max_grasp_num=50)
            # viser_for_grasp.wait_for_reset()

            data_dict = {
                'filename': bathc_i_filename,
                'xyz': vis_xyz,
                'grasp_Ts': vis_grasp_Ts,
                'mesh_T': batch_i_mesh_T,
            }
            results.append(data_dict)
    with open(os.path.join(scorenet_work_dir, '{}.pkl'.format(eval_save_name)), 'wb') as f:
        pickle.dump(results, f)
    # exit()
    
    results = pickle.load(open(os.path.join(scorenet_work_dir, '{}.pkl'.format(eval_save_name)), 'rb'))
    
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

    np.save(os.path.join(scorenet_work_dir, 'isaacgym_{}_results.npy'.format(eval_save_name)), isaacgym_eval_data_dict)
