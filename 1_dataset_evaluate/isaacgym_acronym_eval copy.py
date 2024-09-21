import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(cur_dir), "3rd_isaacgym_evaluation"))

import torch
import numpy as np
import open3d as o3d

from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


if __name__ == '__main__':
    test_data_path = "/home/red0orange/Projects/handgrasp_ws/tmp.npy"
    data_dict = np.load(test_data_path, allow_pickle=True).item()


    viser_grasp = ViserForGrasp()

    device = "cuda:0"
    for data_id, data in data_dict.items():
        obj_mesh_path = data['mesh_path']
        mesh_scale = data['mesh_scale']
        mesh_T = data['mesh_T']
        grasp_Ts = data['grasp_Ts']
        grasp_success = data['grasp_success']

        grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if grasp_success[i]]

        if True:
        # if False:
            mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
            mesh = mesh.scale(mesh_scale, center=[0, 0, 0])
            mesh = mesh.transform(mesh_T)
            pc = np.array(mesh.vertices)

            # grasp_success = data['grasp_success']
            # success_grasp_Ts = np.where(grasp_success)[0]
            # success_grasp_Ts = [grasp_Ts[i] for i in success_grasp_Ts]
            success_grasp_Ts = grasp_Ts
            print(f"Success grasp num: {len(success_grasp_Ts)}")
            for i in range(len(success_grasp_Ts)):
                print("current grasp id: ", i)
                grasp_T = success_grasp_Ts[i]
                viser_grasp.vis_grasp_scene(max_grasp_num=1, pc=pc, grasp_Ts=[grasp_T], mesh=mesh)
                viser_grasp.wait_for_reset()