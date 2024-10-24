import os
import time
import glob
import pickle as pkl
import json

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group

import torch
from torch.utils.data import Dataset, DataLoader

from roboutils.proj_llm_robot.pose_transform import update_pose


class AcronymGrasps():
    def __init__(self, filename, load_grasps=True):

        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = data["object/file"][()].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        if load_grasps:
            self.grasps, self.success = self.load_grasps(filename)
            good_idxs = np.argwhere(self.success==1)[:,0]
            bad_idxs  = np.argwhere(self.success==0)[:,0]
            self.good_grasps = self.grasps[good_idxs,...]
            self.bad_grasps  = self.grasps[bad_idxs,...]

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def load_mesh(self, data_dir):
        mesh_path_file = os.path.join(data_dir, self.mesh_fname)

        mesh = trimesh.load(mesh_path_file,  file_type='obj', force='mesh')

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh


# CONG dataset for grasp diffusion baseline 训练
class _CONGDiffDataset(Dataset):
    def __init__(self, data_dir, acronym_data_dir, mode='train', split_json_path=None, n_pointcloud=1024, n_grasps=80, augmented_rotation=True, center_pcl=True):
        self.data_dir = data_dir
        self.acronym_data_dir = acronym_data_dir
        self.type = mode

        if split_json_path is None:
            split_json_path = os.path.join(data_dir, 'split.json')
        self.split_json_file = split_json_path
        self.split = json.load(open(self.split_json_file, 'r'))
        self.data_files = self.split[mode]
        self.data_files = [os.path.join(data_dir, 'data', i) for i in self.data_files]

        self.acronym_grasp_data_dir = os.path.join(self.acronym_data_dir, 'grasps')
        self.acronym_sdf_data_dir = os.path.join(self.acronym_data_dir, 'sdf')
        self.acronym_data_files = glob.glob(os.path.join(self.acronym_grasp_data_dir, '*.h5'))
        self.acronym_data_names = [os.path.basename(i).rsplit('.', maxsplit=1)[0] for i in self.acronym_data_files]
        self.acronym_data_names = [i.rsplit('_', maxsplit=1)[0] for i in self.acronym_data_names]

        # params
        self.n_pointcloud = n_pointcloud
        self.n_occ = n_pointcloud
        self.n_grasps = n_grasps
        self.augmented_rotation = augmented_rotation
        self.center_pcl = center_pcl
        # self.partial = partial

        # other fixed params
        self.scale = 8.
    
    def __len__(self):
        return len(self.data_files)
    
    def get_sdf(self, grasp_obj):
        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_type = mesh_fname.split('/')[1]
        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.acronym_sdf_data_dir, mesh_type, filename+'.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pkl.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale
        return xyz, sdf

    def get_mesh_pcl(self, grasp_obj):
        # for debug
        mesh = grasp_obj.load_mesh(data_dir=self.acronym_data_dir)
        return mesh.sample(self.n_pointcloud)

    def __getitem__(self, idx):
        idx = 10

        data_file = self.data_files[idx]

        data_name = os.path.basename(data_file)
        acronym_data_name = data_name.split('_')[1] + '_' + data_name.split('_')[2]
        acronym_data_file_path = self.acronym_data_files[self.acronym_data_names.index(acronym_data_name)]

        pkl_data = pkl.load(open(data_file, 'rb'))
        grasp_obj = AcronymGrasps(acronym_data_file_path, load_grasps=False)

        obj_sdf_xyz, obj_sdf = self.get_sdf(grasp_obj)
        obj_pc = pkl_data["sampled_pc_{}".format(self.n_pointcloud)]
        grasp_Ts = list(pkl_data["grasps/transformations"])
        grasp_successes = pkl_data["grasps/successes"]

        # # Debug vis
        # obj_pcl = self.get_mesh_pcl(grasp_obj)  # for debug
        # from roboutils.vis.viser_grasp import ViserForGrasp
        # viser = ViserForGrasp()
        # # viser.vis_grasp_scene(grasp_Ts, pc=obj_pc, max_grasp_num=50)
        # # viser.add_pcd(obj_pc)
        # viser.add_pcd(obj_x_sdf)
        # viser.add_pcd(obj_pcl)
        # viser.add_pcd(obj_pc, colors=np.array([[255, 0, 0]]*obj_pc.shape[0]))
        # viser.wait_for_reset()

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

        # # @note pre grasp
        # for i in range(grasp_Ts.shape[0]):
        #     grasp_Ts[i] = update_pose(grasp_Ts[i], translate=[0, 0, 0.09])

        # scale
        mesh_scale = self.scale
        obj_pc = obj_pc * self.scale
        obj_sdf_xyz = obj_sdf_xyz * self.scale
        obj_sdf = obj_sdf * self.scale  # sdf 值也要乘以 scale
        grasp_Ts[..., :3, -1] = grasp_Ts[..., :3, -1] * self.scale

        mesh_T = np.eye(4)
        if self.center_pcl:
            ## translate ##
            mean = np.mean(obj_pc, 0)
            obj_pc -= mean
            obj_sdf_xyz -= mean
            grasp_Ts[..., :3, -1] = grasp_Ts[..., :3, -1] - mean
            mesh_T[:3, -1] = -mean

        if self.augmented_rotation and self.type == "train":
            ## Random rotation ##
            random_R = special_ortho_group.rvs(3)   # 看上去是想高斯采样，但是里面看不出来
            random_R_T = np.eye(4)
            random_R_T[:3, :3] = random_R
            obj_pc = np.einsum('mn,bn->bm', random_R, obj_pc)
            obj_sdf_xyz = np.einsum('mn,bn->bm', random_R, obj_sdf_xyz)
            grasp_Ts = np.einsum('mn,bnk->bmk', random_R_T, grasp_Ts)   # 应用随机的旋转
            mesh_T[:3, :3] = random_R
        
        res = {'visual_context': torch.from_numpy(obj_pc).float(),
                'x_sdf': torch.from_numpy(obj_sdf_xyz).float(),
                'x_ene_pos': torch.from_numpy(grasp_Ts).float(),
                'scale': torch.Tensor([self.scale]).float(),
                'mesh_T': torch.from_numpy(mesh_T).float(),
            }

        return res, {'sdf': torch.from_numpy(obj_sdf).float()}
    
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
    data_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm"
    acronym_data_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_Acronym"
    dataset = _CONGDiffDataset(data_dir=data_dir, acronym_data_dir=acronym_data_dir, mode="train")

    from roboutils.vis.viser_grasp import ViserForGrasp
    viser = ViserForGrasp()
    for data in dataset:
        (model_input, gt) = data

        obj_pc = model_input['visual_context'].numpy()
        grasp_Ts = model_input['x_ene_pos'].numpy()
        obj_sdf = gt['sdf'].numpy()
        obj_sdf_xyz = model_input['x_sdf'].numpy()

        scale = model_input['scale'].numpy()[0]
        mesh_T = model_input['mesh_T'].numpy()

        obj_pc = obj_pc / scale
        obj_sdf_xyz = obj_sdf_xyz / scale
        obj_sdf = obj_sdf / scale
        grasp_Ts[..., :3, -1] = grasp_Ts[..., :3, -1] / scale

        # vis debug
        viser.vis_grasp_scene(grasp_Ts, pc=obj_pc, max_grasp_num=80)
        viser.add_pcd(obj_sdf_xyz, colors=np.array([[255, 0, 0]]*obj_sdf_xyz.shape[0]))
        viser.wait_for_reset()
        pass
    
    pass