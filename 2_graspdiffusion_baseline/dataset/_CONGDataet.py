import os
import time
import glob
import pickle as pkl
import json

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group

import torch
from torch.utils.data import Dataset, DataLoader

from roboutils.proj_llm_robot.pose_transform import update_pose


class _CONGDataset(Dataset):
    def __init__(self, data_dir, mode='train', split_json_path=None, n_pointcloud=2048, n_grasps=80, augmented_rotation=True, center_pcl=True, partial=True):
        self.data_dir = data_dir
        self.type = mode

        if split_json_path is None:
            split_json_path = os.path.join(data_dir, 'split.json')
        self.split_json_file = split_json_path
        self.split = json.load(open(self.split_json_file, 'r'))
        self.data_files = self.split[mode]
        self.data_files = [os.path.join(data_dir, 'data', i) for i in self.data_files]

        # params
        self.n_pointcloud = n_pointcloud
        self.n_grasps = n_grasps
        self.augmented_rotation = augmented_rotation
        self.center_pcl = center_pcl
        self.partial = partial

        # other fixed params
        self.scale = 8.
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]

        pkl_data = pkl.load(open(data_file, 'rb'))

        obj_pc = pkl_data["sampled_pc_{}".format(self.n_pointcloud)]
        grasp_Ts = list(pkl_data["grasps/transformations"])
        grasp_successes = pkl_data["grasps/successes"]
        # @note TODO for partial
        # rendering_pcs = pickle_data["rendering/point_clouds"]
        # rendering_camera_Ts = pickle_data["rendering/camera_poses"]

        # deal grasps
        grasp_Ts = [i for e, i in enumerate(grasp_Ts) if grasp_successes[e]]
        grasp_Ts = np.array(grasp_Ts)
        if len(grasp_Ts) == 0:
            return self.__getitem__(np.random.randint(len(self)))
        while len(grasp_Ts) < self.n_grasps:
            grasp_Ts = np.concatenate([grasp_Ts, grasp_Ts], axis=0)
        random_idx = np.random.choice(len(grasp_Ts), self.n_grasps)
        grasp_Ts = grasp_Ts[random_idx]

        # @note pre grasp
        for i in range(grasp_Ts.shape[0]):
            grasp_Ts[i] = update_pose(grasp_Ts[i], translate=[0, 0, 0.09])

        # scale
        mesh_scale = self.scale
        obj_pc = obj_pc * self.scale
        grasp_Ts[..., :3, -1] = grasp_Ts[..., :3, -1] * self.scale

        mesh_T = np.eye(4)
        if self.center_pcl:
            ## translate ##
            mean = np.mean(obj_pc, 0)
            obj_pc -= mean
            grasp_Ts[..., :3, -1] = grasp_Ts[..., :3, -1] - mean
            mesh_T[:3, -1] = -mean

        if self.augmented_rotation and self.type == "train":
            ## Random rotation ##
            random_R = special_ortho_group.rvs(3)   # 看上去是想高斯采样，但是里面看不出来
            random_R_T = np.eye(4)
            random_R_T[:3, :3] = random_R
            obj_pc = np.einsum('mn,bn->bm', random_R, obj_pc)
            grasp_Ts = np.einsum('mn,bnk->bmk', random_R_T, grasp_Ts)   # 应用随机的旋转
            mesh_T[:3, :3] = random_R

        return [data_file], obj_pc, grasp_Ts, mesh_T, mesh_scale
    
    def preprocess_infer_data(self, obj_pc):
        obj_pc_num = obj_pc.shape[0]
        while obj_pc_num < self.n_pointcloud:
            obj_pc = np.concatenate([obj_pc, obj_pc], axis=0)
        random_idx = np.random.choice(obj_pc_num, self.n_pointcloud, replace=False)
        obj_pc = obj_pc[random_idx]

        # scale
        obj_pc = obj_pc * self.scale

        return obj_pc


if __name__ == "__main__":
    data_dir = "/home/huangdehao/Projects/handgrasp_ws/2_graspdiff_baseline/data/grasp_CONG_graspldm"
    dataset = _CONGDataset(data_dir=data_dir, mode="train")

    for data in dataset:
        file_path, obj_pc, grasp_Ts = data

        # vis debug
        from roboutils.vis.viser_grasp import ViserForGrasp
        viser = ViserForGrasp()
        viser.vis_grasp_scene(grasp_Ts, pc=obj_pc, max_grasp_num=50)
        viser.wait_for_reset()
        pass
    
    pass