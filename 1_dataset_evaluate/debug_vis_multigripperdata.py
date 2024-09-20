import os 
import json

import numpy as np
import open3d as o3d

import viser.transforms as tf
from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


def is_valid_quaternion(q, tol=1e-6):
    # 检查是否是长度为 4 的四元数
    if len(q) != 4:
        return False
    
    # 计算四元数的模
    norm_q = np.linalg.norm(q)
    
    # 检查是否为单位四元数（模为 1）
    return np.abs(norm_q - 1.0) < tol


def sevendof2T(xyz, xyzw):
    tf_se3 = tf.SE3.from_rotation_and_translation(tf.SO3.from_quaternion_xyzw(xyzw), xyz)
    return tf_se3.as_matrix()


if __name__ == '__main__':
    viser = ViserForGrasp()

    gripper_mesh_path = "/home/red0orange/Projects/handgrasp_ws/3rd_urdf_to_obj/franka_hand.obj"
    gripper_mesh = o3d.io.read_triangle_mesh(gripper_mesh_path)
    obj_dir = "/home/red0orange/Projects/handgrasp_ws/0_Data/MultiGripperGrasp/Object_Models/YCB"
    # 读取文件
    with open("/home/red0orange/Projects/handgrasp_ws/0_Data/MultiGripperGrasp/franka_panda-003_cracker_box_success.json", "r") as f:
        data_dict = json.load(f)
        
        gripper = data_dict["gripper"]
        object_id = data_dict["object_id"]
        grasp_poses = np.array(data_dict["pose"])
        fall_time = np.array(data_dict["fall_time"])
        slip_time = np.array(data_dict["slip_time"])
        # grasp_poses = grasp_poses[slip_time > 1]

        grasp_Ts = [sevendof2T(pose[:3], np.array([pose[4], pose[5], pose[6], pose[3]])) for pose in grasp_poses]

        # obj_mesh_path = os.path.join(obj_dir, object_id, "meshes/model.obj")
        obj_mesh_path = os.path.join(obj_dir, object_id, "textured_simple.obj")
        mesh = o3d.io.read_triangle_mesh(obj_mesh_path)

        for Tog in grasp_Ts:
            Tgo = np.linalg.inv(Tog)

            mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
            # mesh.transform(Tgo)
            gripper_mesh.transform(Tog)

            viser.add_mesh(mesh)
            viser.add_mesh(gripper_mesh)

            test_T = grasp_Ts[0]
            if gripper == "h5_hand":
                test_T = update_pose(test_T, rotate=np.pi, rotate_axis="y")
                test_T = update_pose(test_T, rotate=np.pi/2, rotate_axis="z")
            elif gripper == "franka_panda":
                test_T = update_pose(test_T, rotate=np.pi/2, rotate_axis="z")
                test_T = update_pose(test_T, translate=[0, 0, 0.05])
            else:
                raise ValueError(f"Unknown gripper {gripper}")

            viser.vis_grasp_scene([test_T], mesh=mesh, max_grasp_num=40, z_direction=True)
            viser.wait_for_reset()

            pass
        
        # viser.vis_grasp_scene(grasp_Ts, mesh=mesh, max_grasp_num=40, z_direction=True)
        # viser.wait_for_reset()

    pass