import glob
import copy
import time
import numpy as np
import trimesh
from dataloader.voxel_utils import *
from torch.utils.data import DataLoader, Dataset
# from scipy.stats import special_ortho_group
import torch
import os
import json
import pickle
import h5py

class AcronymGrasps():
    def __init__(self, filename, data_dir, loader=False):

        self.data_dir = data_dir
        self.voxel_dirname = os.path.join(self.data_dir, "voxel_grids_jc_32")
        self.filename = filename

        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            # data = h5py.File(filename, "r")
            with h5py.File(filename, 'r') as data:
                self.mesh_fname = data["object/file"][()].decode('utf-8')
                self.mesh_type = self.mesh_fname.split('/')[1]
                self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
                self.mesh_scale = data["object/scale"][()] if scale is None else scale

                self.mesh_norm_scale = data["object/norm_scale"][()]
            # data.close()
        else:
            raise RuntimeError("Unknown file ending:", filename)

        if(loader == True):
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
            # data = h5py.File(filename, "r")
            with h5py.File(filename, 'r') as data:
                T = np.array(data["grasps/transforms"])
                success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def load_voxel_grid(self):

        output_fname = os.path.join(self.voxel_dirname, os.path.splitext(os.path.basename(self.filename))[0] + ".npy")

        return torch.from_numpy(np.load(output_fname)).float()

class AcronymAndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, grasp_dict, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut'],
                 data_dir = "/Users/username/Desktop/TUM/Courses/Sem 3/DL_Robotics/Project/acronym-full-dataset",
                 ):

        self.class_type = class_type
        self.data_dir = data_dir
        self.grasp_dict = grasp_dict

        self.grasps_dir = os.path.join(self.data_dir, 'grasps_updated_jc')
        self.grasps_data_point = []
        self.grasp_files = []

        cls_grasps_files = [os.path.join(self.grasps_dir, i) for i in self.grasp_dict['train']]

        for grasp_file in cls_grasps_files:
            g_obj = AcronymGrasps(grasp_file, self.data_dir, loader=True)

            if g_obj.good_grasps.shape[0] > 0:
                self.grasp_files.append(grasp_file)
                step_ = 0
                for grasp in g_obj.good_grasps:
                    translation, quaternion = decompose_matrix(grasp)
                    Septernion = np.concatenate((translation, quaternion))
                    self.grasps_data_point.append({'grasp': Septernion, 'grasp_file': grasp_file, 'object_dir': g_obj.mesh_fname, 'norm_scale':g_obj.mesh_norm_scale})
                        # step_ += 1
                        # if(step_ > 500):
                        #     break;


        self.len = len(self.grasps_data_point)

    def __len__(self):
        return self.len
    
    def _get_item(self, index):

        grasps_obj = AcronymGrasps(self.grasps_data_point[index]["grasp_file"], self.data_dir)

        ## SDF
        sdf = grasps_obj.load_voxel_grid()

        ## get grasp
        # translation, quaternion = decompose_matrix(self.grasps_data_point[index]["grasp"])

        # #concatenate
        # Septernion = np.concatenate((translation, quaternion))
        Septernion = self.grasps_data_point[index]["grasp"]

        norm_scale = self.grasps_data_point[index]['norm_scale']

        return Septernion, sdf, norm_scale
    
    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)