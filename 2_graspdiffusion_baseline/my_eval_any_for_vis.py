import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from gorilla.config import Config

from utils import *
from utils.rotate_rep import rotation_6d_to_matrix_np, matrix_to_rotation_6d_np
from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


def transform_pcd(pcd, T):
    return (T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]


if __name__ == "__main__":
    # work_dir = "/home/huangdehao/Projects/handgrasp_ws/2_graspdiff_baseline/log/epoch_199_20240923-232338_detectiondiffusion"
    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_199_20240926-131702_detectiondiffusion"
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

    # oakink_data
    tmp_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/tmp_oakink_data.npy"
    item = np.load(tmp_data_path, allow_pickle=True).item()
    hand_verts = item['verts']
    hand_faces = item['hand_th_faces']
    obj_verts = item['obj_verts']
    obj_faces = item['obj_faces']
    my_palm_T = item['my_palm_T']

    my_palm_T = update_pose(my_palm_T, rotate=np.pi/2, rotate_axis="x")
    my_palm_T = update_pose(my_palm_T, rotate=-np.pi / 2, rotate_axis='x')
    my_palm_T = update_pose(my_palm_T, rotate=np.pi / 2, rotate_axis='y')

    # viser_for_grasp.add_pcd(obj_verts)
    # viser_for_grasp.add_grasp(my_palm_T)
    # viser_for_grasp.wait_for_reset()

    hand_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hand_verts), o3d.utility.Vector3iVector(hand_faces))
    hand_mesh.compute_vertex_normals()
    obj_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj_verts), o3d.utility.Vector3iVector(obj_faces))
    obj_mesh.compute_vertex_normals()
    
    xyz = dataset.preprocess_infer_data(obj_verts)

    # @note begin infer
    xyz = torch.from_numpy(xyz).unsqueeze(0).float().cuda()


    # @note 无限制直接预测
    # pred = model.detect_and_sample(xyz, grasp_per_obj, guide_w=GUIDE_W, data_scale=dataset.scale)

    # @note 限制预测
    g_init = np.concatenate((matrix_to_rotation_6d_np(my_palm_T[:3, :3]), my_palm_T[:3, 3]), axis=0)
    g_init = torch.from_numpy(g_init).float().cuda()
    pred = model.refine_grasp_sample(xyz, grasp_per_obj, g_init)


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


    rotate_T = update_pose(np.eye(4), rotate=np.pi/2, rotate_axis="y")
    vis_xyz = transform_pcd(vis_xyz, rotate_T)
    hand_mesh.transform(rotate_T)
    vis_T = [rotate_T @ t for t in vis_T]

    # debug vis
    print("Waiiting for viser visualization")
    viser_for_grasp.vis_grasp_scene(vis_T, pc=vis_xyz, max_grasp_num=50)
    # viser_for_grasp.add_mesh(hand_mesh)
    viser_for_grasp.wait_for_reset()