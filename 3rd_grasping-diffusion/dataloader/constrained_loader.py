"""
Dataloader for AcronymAndSDFDataset

output: translation, quaternion, sdf


"""

import glob
import copy
import time
# import configargparse
import numpy as np
import trimesh
from dataloader.voxel_utils import *
# from dataloader.directory_utils import get_data_src, get_root_src
from torch.utils.data import DataLoader, Dataset
from scipy.stats import special_ortho_group
import os
import torch


import json
import pickle
import h5py

class AcronymGrasps():
    def __init__(self, filename, data_dir):

        self.filename = filename
        self.data_dir = data_dir
        self.voxel_dir = os.path.join(data_dir, "voxel_grids")
        self.mask_dir = os.path.join(data_dir, "constrain_masks")
        self.grasps_dir = os.path.join(data_dir, "grasps")

        if filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = data["object/file"][()].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object/scale"][()]
            self.mesh_norm_scale = data["object/norm_scale"][()]
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.good_grasps = data['grasps/transforms'][()]
        # good_idxs = np.argwhere(self.success==1)[:,0]
        # bad_idxs  = np.argwhere(self.success==0)[:,0]
        # self.good_grasps = self.grasps[good_idxs,...]
        # self.bad_grasps  = self.grasps[bad_idxs,...]

    def load_voxel_grid(self):

        output_fname = os.path.join(self.voxel_dir, os.path.splitext(os.path.basename(self.filename))[0] + ".npz")

        return torch.from_numpy(np.load(output_fname)['arr_0']).float()
    
    def load_mask(self):
        output_fname = os.path.join(self.mask_dir, os.path.splitext(os.path.basename(self.filename))[0] + ".npz")

        return torch.from_numpy(np.load(output_fname)['arr_0']).float()


class AcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, grasp_dict, loader_type="train",
                 data_dir = "/home/username/data/constrained"):
        
        self.data_dir = data_dir
        self.grasp_dict = grasp_dict

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')
        self.grasps_data_point = []
        self.grasp_files = []

        cls_grasps_files = [os.path.join(self.grasps_dir, i) for i in self.grasp_dict[loader_type]]

        for grasp_file in cls_grasps_files:
            g_obj = AcronymGrasps(grasp_file, self.data_dir)

            if g_obj.good_grasps.shape[0] > 0:
                self.grasp_files.append(grasp_file)
                for grasp in g_obj.good_grasps:
                    translation, quaternion = decompose_matrix(grasp)
                    Septernion = np.concatenate((translation, quaternion))
                    self.grasps_data_point.append({'grasp': Septernion, 'grasp_file': grasp_file, 'norm_scale':g_obj.mesh_norm_scale})

        self.len = len(self.grasps_data_point)

    def __len__(self):
        return self.len
    
    def _get_item(self, index):

        grasps_obj = AcronymGrasps(self.grasps_data_point[index]["grasp_file"], self.data_dir)

        sdf = grasps_obj.load_voxel_grid()
        mask = grasps_obj.load_mask()
        combined_voxel_grid = np.stack((sdf, mask), axis=0)

        Septernion = self.grasps_data_point[index]["grasp"]
        norm_scale = self.grasps_data_point[index]['norm_scale']

        return Septernion, combined_voxel_grid, norm_scale
    
    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)
