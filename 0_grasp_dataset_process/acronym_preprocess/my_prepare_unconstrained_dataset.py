import os
import json
import h5py
import trimesh
import trimesh.path
import trimesh.transformations as tra
import numpy as np
from acronym_tools import load_mesh, load_grasps, create_gripper_marker

from tqdm import tqdm
import os, json
import trimesh
import mesh2sdf
import numpy as np
from scipy.spatial.transform import Rotation as R


def create_voxel_grid(mesh, n=32, scale=1):
    """
    Uses the the mesh2sdf library to convert the mesh into a (n x n x n) voxel grid
    The values within the grid are sdf (signed distance function) values
    Input - 
        1. mesh_path --> Path to mesh file (.obj file)
        2. n --> size of voxel grid
    Output - 
        1. mesh --> mesh object after loading, normalizing and fixing mesh
        2. sdf --> (n x n x n) numpy array 
    """

    # Load mesh file
    # mesh = trimesh.load(mesh_path, force='mesh')

    mesh_scale = scale
    size = n
    level = 2 / size

    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)

    # output
    mesh.vertices = mesh.vertices / scale + center

    return mesh, sdf

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    grasp_path = "/home/huangdehao/Projects/grasping-diffusion/data/constrain_data/grasps/1Shelves_1e3df0ab57e8ca8587f357007f9e75d1_0_011099225885734912_1.h5"
    data = h5py.File(grasp_path, "r")
    print(data["object/scale"][()])
    print(data["object/norm_scale"][()])
    # exit()


    grasp_dir = "/home/huangdehao/Projects/grasping-diffusion/data/my_acronym/grasps"
    mesh_root = "/home/huangdehao/Projects/grasping-diffusion/data/my_acronym"
    voxel_dirname = "/home/huangdehao/Projects/grasping-diffusion/data/my_acronym/voxel_grids"

    grasp_paths = [os.path.join(grasp_dir, i) for i in os.listdir(grasp_dir)]
    print(len(grasp_paths))

    for i in tqdm(range(len(grasp_paths))):
        grasp_path = grasp_paths[i]
        obj_cat, obj_id, grasp_id = grasp_path.split("/")[-1].split("_")[:3]
        if obj_id != "1e3df0ab57e8ca8587f357007f9e75d1":
            continue

        data = h5py.File(grasp_path, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()]

        mesh_path = os.path.join(mesh_root, mesh_fname)
        if not os.path.exists(mesh_path):
            print("Skipping", mesh_path, "not found")
            continue

        temp = mesh_fname.split("/")
        converted_dirname = os.path.join(voxel_dirname, temp[1])
        makedirs(converted_dirname)
        output_fname = os.path.splitext(temp[2])[0] + ".npy"
        output_fname = os.path.join(converted_dirname, output_fname)
        # print(output_fname)

        obj_mesh = trimesh.load(mesh_path,  file_type='obj', force='mesh')
        mesh, sdf = create_voxel_grid(obj_mesh, scale=mesh_scale)
        # print(sdf.shape)

        np.save(output_fname, sdf)
        # break;