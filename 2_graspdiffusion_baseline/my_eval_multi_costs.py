import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from gorilla.config import Config

from utils import *
from utils.rotate_rep import rotation_6d_to_matrix_np, matrix_to_rotation_6d_np
from roboutils.vis.grasp import draw_scene
from roboutils.proj_llm_robot.pose_transform import update_pose


if __name__ == "__main__":
    work_dir = "/home/red0orange/Projects/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log/epoch_34_20240711-132025_detectiondiffusion_star_score_based_avail"
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")
    # checkpoint_path = os.path.join(work_dir, "model_epoch_30.pth")
    
    GUIDE_W = 0.5
    DEVICE=torch.device('cuda')

    cfg = Config.fromfile(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = build_model(cfg).to(DEVICE)
    dataset = build_dataset(cfg)['test_set']
    # dataset = build_dataset(cfg)['train_set']
    rot6d_rep = cfg.hyper_params.rot6d_rep
    
    print("Loading checkpoint....")
    _, exten = os.path.splitext(checkpoint_path)
    if exten == '.t7':
        model.load_state_dict(torch.load(checkpoint_path))
    elif exten == '.pth':
        check = torch.load(checkpoint_path)
        model.load_state_dict(check['model_state_dict'])

    # 加载一个 grasp 作为 coarse pose
    Ts = np.load("/home/red0orange/Projects/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log/0_data/predicted_poses.npy")
    T_init = Ts[0]
    T_init = update_pose(T_init, translate=[0, 0, -0.45])  # 给一个偏移，加大难度
    g_init = np.concatenate((matrix_to_rotation_6d_np(T_init[:3, :3]), T_init[:3, 3]), axis=0)
    g_init = torch.from_numpy(g_init).float().cuda()

    print("Evaluating")
    model.eval()
    for data in dataset:
        xyz, gt_T, gt_quat, gt_trans = data
        # draw_scene(xyz, gt_T, z_direction=True, scale=1.0/dataset.scale)

        xyz = torch.from_numpy(xyz).unsqueeze(0).float().cuda()
        pred = model.refine_grasp_sample(xyz, 40, g_init=g_init, data_scale=dataset.scale)
        # pred = model.detect_and_sample(xyz, 40, guide_w=GUIDE_W)
        xyz = xyz.squeeze(0).cpu().numpy()

        num_pose = pred.shape[0]
        if not rot6d_rep:
            rotation = np.concatenate((R.from_quat(pred[:, :4]).as_matrix(), np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
            translation = np.expand_dims(np.concatenate((pred[:, 4:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
        else:
            rotation = rotation_6d_to_matrix_np(pred[:, :6])
            rotation = np.concatenate((rotation, np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
            translation = np.expand_dims(np.concatenate((pred[:, 6:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
        T = np.concatenate((rotation, translation), axis=2)

        # # save the predicted poses
        # np.save("/home/red0orange/Projects/Language-Conditioned-Affordance-Pose-Detection-in-3D-Point-Clouds/log/0_data/predicted_poses.npy", T)

        show_T = np.concatenate([T_init[None, ...], T], axis=0)
        show_colors = np.zeros((show_T.shape[0], 3))
        show_colors[0] = np.array([1, 0, 0])
        draw_scene(xyz, show_T[:20], grasp_colors=show_colors.tolist()[:20], max_grasps=20, z_direction=True, scale=1.0/dataset.scale)

