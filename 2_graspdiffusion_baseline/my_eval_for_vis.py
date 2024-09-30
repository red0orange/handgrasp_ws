import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from gorilla.config import Config

from utils import *
from utils.rotate_rep import rotation_6d_to_matrix_np
from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


if __name__ == "__main__":
    # work_dir = "/home/huangdehao/Projects/handgrasp_ws/2_graspdiff_baseline/log/epoch_199_20240923-232338_detectiondiffusion"
    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_199_20240925-131342_detectiondiffusion"
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")
    # checkpoint_path = os.path.join(work_dir, "model_epoch_75.pth")
    
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
        model.load_state_dict(torch.load(checkpoint_path))
        # check = torch.load(checkpoint_path)
        # model.load_state_dict(check['model_state_dict'])

    print("Evaluating")
    viser_for_grasp = ViserForGrasp()
    model.eval()
    grasp_per_obj = 10
    for data in tqdm(dataset, total=len(dataset)):
        # ori_xyz, xyz, gt_T, _, _ = data
        # filename, xyz, gt_T = data
        filename, xyz, gt_T, mesh_T = data[0], data[1], data[2], data[3]

        xyz = torch.from_numpy(xyz).unsqueeze(0).float().cuda()
        pred = model.detect_and_sample(xyz, grasp_per_obj, guide_w=GUIDE_W, data_scale=dataset.scale)
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

        vis_xyz, vis_T = xyz.copy(), T.copy()
        vis_xyz /= dataset.scale
        vis_T[:, :3, 3] /= dataset.scale

        # debug vis
        print("Waiiting for viser visualization")
        viser_for_grasp.vis_grasp_scene(vis_T, pc=vis_xyz, max_grasp_num=50)
        viser_for_grasp.wait_for_reset()