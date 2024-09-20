import os
import json
import h5py
import trimesh
import trimesh.path
import trimesh.transformations as tra
import numpy as np
# from acronym_tools import load_mesh, load_grasps, create_gripper_marker

import os, json
import trimesh
import mesh2sdf
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pickle

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def create_voxel_grid(mesh, n=32):
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

    # try:
    mesh_scale = 0.8
    size = n
    level = 2 / size

    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    # mesh_scaled = mesh.apply_scale(scale)

    sdf = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=False)

    mesh.vertices = vertices
    return sdf, scale, center, mesh

def point_to_voxel(point, grid_size):
    """Converts a point in the range [-1, 1] to voxel grid coordinates."""
    return np.clip(((point + 1) / 2) * (grid_size - 1), 0, grid_size - 1).astype(int)

def update_mask(mask, points):
    """Updates the mask for each point in the list of points."""
    grid_size = mask.shape[0]  # Assuming the mask is a cubic grid
    for point in points:
        voxel = point_to_voxel(point, grid_size)
        mask[voxel[0], voxel[1], voxel[2]] = 1

    return mask

def center_grasps(grasps, center):
    translation_T = np.zeros_like(np.eye(4))
    translation_T[0][3] = -center[0]
    translation_T[1][3] = -center[1]
    translation_T[2][3] = -center[2]
    g = grasps + translation_T
    return g

def get_n_query_points_and_grasps(data, T, center_scale, norm_scale, grasp_success_idxs, n=4):
    count_succ = 0

    num_pc = len(data['rendering/point_clouds'])

    rand_pc_ix = np.random.choice(num_pc, size=min(n*4, num_pc), replace=False)

    output = []
    for pc_ix in rand_pc_ix:
        obj = {}

        pc = data['rendering/point_clouds'][pc_ix]

        grasp_ix = data['query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud'][pc_ix]
        qp_ixes = data['query_points/points_with_grasps_on_each_rendered_point_cloud'][pc_ix]

        cam_pose = data['rendering/camera_poses'][pc_ix]
        if(len(qp_ixes) > 0):
            # Randomly picking one set of query points
            k = np.random.randint(0, len(qp_ixes)-1)
            qp_ixes_rand = qp_ixes[k]
            grasp_ix_rand = grasp_ix[k]
            intersection_ix = np.intersect1d(grasp_ix_rand, grasp_success_idxs)

            cam_pose_inv = np.linalg.inv(cam_pose)

            query_point_arr = pc[qp_ixes_rand]
            query_point_arr_added_dim = np.concatenate([query_point_arr, np.ones_like(query_point_arr[:, :1])], axis=1)
            query_point_arr_t = (cam_pose_inv @ query_point_arr_added_dim.T).T
            new_qp = query_point_arr_t[:,:3]
            
            # Now, we have correct query points (after applying camera pose inverse)

            if(new_qp.shape[0] > 10):
                count_succ += 1
                if(count_succ > n):
                    return output
                new_qp_norm = (new_qp - center_scale) * norm_scale
                mask = np.zeros((32, 32, 32))
                mask = update_mask(mask, new_qp_norm)

                obj['mask'] = mask
                obj['query_points_normalized'] = new_qp_norm
                # We center the grasps but we don't scale them, instead we will save this scale value and provide it as a input to the model
                obj['constrained_grasps'] = center_grasps(T[intersection_ix], center_scale)
                output.append(obj)


def cong_data_prepare(cong_path, mesh_root, masks_output_dirname, grasps_output_dirname, voxels_output_dirname):

    cong_fname = cong_path

    obj_type = os.path.basename(cong_fname).split("_")[1]

    # Read cong file
    with open(cong_fname, 'rb') as f:
        cong_data = pickle.load(f)

    # Get grasps and success from cong data (same as acronym)
    T = cong_data['grasps/transformations']
    success = cong_data['grasps/successes']

    mesh_scale = cong_data['mesh/scale'] # Scale from cong data (same as acronym)

    # Loading and applying initial scale to mesh
    mesh_fname = os.path.join(os.path.join(mesh_root, obj_type), os.path.basename(cong_data['mesh/file']))
    if not os.path.exists(mesh_fname):
        print("Skipping", mesh_fname, "not found")
        return 1
    mesh = trimesh.load(mesh_fname, force='mesh')
    mesh = mesh.apply_scale(mesh_scale)

    # Getting indices for all successful grasps
    good_idxs = np.argwhere(success==1)[:,0]

    # Normalizing mesh between -1 and 1, creating voxel grid
    sdf, norm_scale, center_scale, mesh = create_voxel_grid(mesh, n=32)

    num_pc = len(cong_data['rendering/point_clouds'])
    # num_pc
    rand_pc_ix = np.random.choice(num_pc, size=num_pc, replace=False)
    rand_pc_ix

    output = []
    for pc_ix in rand_pc_ix:
        obj = {}

        pc = cong_data['rendering/point_clouds'][pc_ix]
        grasp_ix = cong_data['query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud'][pc_ix]
        qp_ixes = cong_data['query_points/points_with_grasps_on_each_rendered_point_cloud'][pc_ix]
        cam_pose = cong_data['rendering/camera_poses'][pc_ix]

        if(len(qp_ixes) > 1):
            # Randomly picking one set of query points
            k = np.random.randint(0, len(qp_ixes)-1)
            qp_ixes_rand = qp_ixes[k]
            grasp_ix_rand = grasp_ix[k]
            intersection_ix = np.intersect1d(grasp_ix_rand, good_idxs)

            cam_pose_inv = np.linalg.inv(cam_pose)

            query_point_arr = pc[qp_ixes_rand]
            query_point_arr_added_dim = np.concatenate([query_point_arr, np.ones_like(query_point_arr[:, :1])], axis=1)
            query_point_arr_t = (cam_pose_inv @ query_point_arr_added_dim.T).T
            new_qp = query_point_arr_t[:,:3]
            
            # Now, we have correct query points (after applying camera pose inverse)

            if(new_qp.shape[0] > 10):
                if(len(output) > 1):
                    break
                new_qp_norm = (new_qp - center_scale) * norm_scale
                mask = np.zeros((32, 32, 32))
                mask = update_mask(mask, new_qp_norm)

                obj['mask'] = mask
                # obj['query_points_normalized'] = new_qp_norm
                obj['cam_pose'] = cam_pose
                obj['cam_pose_inv'] = cam_pose_inv
                # We center the grasps but we don't scale them, instead we will save this scale value and provide it as a input to the model
                obj['constrained_grasps'] = center_grasps(T[intersection_ix], center_scale)
                obj['new_qp'] = new_qp
                output.append(obj)
        # break;
    try:
        f = output[0]

        temp = os.path.basename(cong_fname)[12:]
        mask_output_fname = os.path.join(masks_output_dirname, os.path.splitext(temp)[0]+".npz")
        np.savez_compressed(mask_output_fname, f['mask'])

        voxel_grid_output_fname = os.path.join(voxels_output_dirname, os.path.splitext(temp)[0]+".npz")
        np.savez_compressed(voxel_grid_output_fname, sdf)

        grasp_output_fname = os.path.join(grasps_output_dirname, os.path.splitext(temp)[0]+".h5")

        with h5py.File(grasp_output_fname, 'w') as new_data:
            new_data.create_dataset("grasps/transforms", data=f['constrained_grasps'])
            new_data.create_dataset("object/file", data=mesh_fname)
            new_data.create_dataset("object/scale", data=mesh_scale)
            new_data.create_dataset("object/norm_scale", data=norm_scale)
            new_data.create_dataset("object/center_scale", data=center_scale)
            # These query points are already inverted by cam_pose_inv
            new_data.create_dataset("object/query_points", data=f['new_qp'])
            new_data.create_dataset("camera_pose", data=f['cam_pose'])
            new_data.create_dataset("camera_pose_inv", data=f['cam_pose_inv'])
    except Exception as e:
        print(e)
        return 1

    return 0


if __name__ == "__main__":
    cong_dir = "/home/huangdehao/Projects/grasping-diffusion/data/my_cong/ori_data"
    mesh_root = "/home/huangdehao/Projects/grasping-diffusion/data/my_acronym/meshes"
    cong_files = os.listdir(cong_dir)

    cong_files = [os.path.join(cong_dir, i) for i in cong_files]
    print(len(cong_files))

    masks_output_dirname = "/home/huangdehao/Projects/grasping-diffusion/data/my_cong/constrain_masks"
    grasps_output_dirname = "/home/huangdehao/Projects/grasping-diffusion/data/my_cong/grasps"
    voxels_output_dirname = "/home/huangdehao/Projects/grasping-diffusion/data/my_cong/voxel_grids"

    failed_counts = 0
    for i in tqdm(range(len(cong_files))):
        s = cong_data_prepare(cong_files[i], mesh_root, masks_output_dirname, grasps_output_dirname, voxels_output_dirname)
        failed_counts += s


    import multiprocessing

    with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(cong_data_prepare, cong_files)

    # voxel_grid_maker(grasp_paths[0])

    pass