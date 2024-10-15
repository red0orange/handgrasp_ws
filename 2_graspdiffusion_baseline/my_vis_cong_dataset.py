import os
import random
import pickle
import h5py
import time

import numpy as np
import open3d as o3d

from roboutils.vis.viser_grasp import ViserForGrasp


def transform_pcd(pcd, T):
    return (T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]


if __name__ == "__main__":
    import json
    # grasp_diff_cong_split_json_path = "/home/huangdehao/Projects/handgrasp_ws/2_graspnet_baseline/data/constrained_data_split.json"
    grasp_diff_cong_split_json_path = "/home/huangdehao/Projects/handgrasp_ws/2_graspnet_baseline/data/unconstrained_data_split.json"
    json_data = json.load(open(grasp_diff_cong_split_json_path, "r"))

    grasp_viser = ViserForGrasp()

    # mesh_root = "/home/huangdehao/Projects/grasping-diffusion/data/my_acronym/meshes"
    mesh_root = "/home/huangdehao/Projects/handgrasp_ws/2_graspnet_baseline/data/obj_ShapeNetSem/models-OBJ/models"
    cong_root = "/home/huangdehao/Projects/handgrasp_ws/2_graspnet_baseline/data/grasp_CONG"
    pickle_paths = [os.path.join(cong_root, i) for i in os.listdir(cong_root) if i.endswith(".pickle")]

    for i, pickle_path in enumerate(pickle_paths):
        pickle_fname = os.path.basename(pickle_path).split(".")[0]
        pickle_data = pickle.load(open(pickle_path, "rb"))

        obj_cat = pickle_fname.split("_")[1]
        mesh_fname = os.path.basename(pickle_data["mesh/file"])
        mesh_scale = pickle_data["mesh/scale"]
        # mesh_path = os.path.join(mesh_root, obj_cat, mesh_fname)
        mesh_path = os.path.join(mesh_root, mesh_fname)
        if not os.path.exists(mesh_path):
            print("Skipping", mesh_path, "not found")
            continue
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.scale(mesh_scale, center=np.zeros(3))

        # @note 关键的，partial pcd
        rendering_pcs = pickle_data["rendering/point_clouds"]
        rendering_camera_Ts = pickle_data["rendering/camera_poses"]
        grasp_Ts = list(pickle_data["grasps/transformations"])
        grasp_successes = pickle_data["grasps/successes"]
        success_grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if grasp_successes[i]]

        # vis
        for render_i in range(len(rendering_pcs)):
            rendering_pc = rendering_pcs[render_i]
            rendering_camera_T = rendering_camera_Ts[render_i]

            world_pc = transform_pcd(rendering_pc, np.linalg.inv(rendering_camera_T))

            # sample_grasp_Ts = random.sample(success_grasp_Ts, 50)
            sample_grasp_Ts = random.sample(grasp_Ts, 50)
            grasp_viser.vis_grasp_scene(sample_grasp_Ts, pc=world_pc, mesh=mesh, z_direction=True, max_grasp_num=50)
            # grasp_viser.vis_grasp_scene(success_grasp_Ts, pc=world_pc, mesh=None, z_direction=True)
            grasp_viser.wait_for_reset()
            break


    pass


