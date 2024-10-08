import os
import numpy as np
import pickle
import open3d as o3d

from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


def transform_pcd(pcd, T):
    return (T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]


if __name__ == '__main__':
    data_path = "/home/red0orange/Projects/one_shot_il_ws/OakInk/processed_oakink_shape_all_data.pkl"
    data = pickle.load(open(data_path, "rb"))

    viser_grasp = ViserForGrasp()

    last_cat_id = None
    for item in data:
        obj_cat_id = item['cate_id']
        if last_cat_id is not None and last_cat_id == obj_cat_id:
            last_cat_id = obj_cat_id
            continue
        last_cat_id = obj_cat_id

        hand_verts = item['verts']
        hand_faces = item['hand_th_faces']
        obj_verts = item['obj_verts']
        obj_faces = item['obj_faces']
        my_palm_T = item['my_palm_T']

        random_pc_idx = np.random.choice(len(obj_verts), min(len(obj_verts), 1024), replace=False)
        obj_verts = obj_verts[random_pc_idx]
        
        cat_pc = np.concatenate([hand_verts, obj_verts], axis=0)
        cat_colors = np.array([[255, 0, 0]] * len(hand_verts) + [[0, 255, 0]] * len(obj_verts))

        viser_grasp.add_pcd(cat_pc, colors=cat_colors)
        my_palm_T = update_pose(my_palm_T, rotate=np.pi/2, rotate_axis="x")
        viser_grasp.add_grasp(my_palm_T, z_direction=False)
        break_flag = viser_grasp.wait_for_reset()
        if break_flag:
            np.save("tmp_oakink_data.npy", item)
            break
    
    # viser_grasp = ViserForGrasp()
    # tmp_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/tmp_oakink_data.npy"
    # item = np.load(tmp_data_path, allow_pickle=True).item()
    # hand_verts = item['verts']
    # hand_faces = item['hand_th_faces']
    # obj_verts = item['obj_verts']
    # obj_faces = item['obj_faces']
    # my_palm_T = item['my_palm_T']
    # my_palm_T = update_pose(my_palm_T, rotate=np.pi/2, rotate_axis="x")
    # hand_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hand_verts), o3d.utility.Vector3iVector(hand_faces))
    # hand_mesh.compute_vertex_normals()
    # obj_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj_verts), o3d.utility.Vector3iVector(obj_faces))
    # obj_mesh.compute_vertex_normals()

    # rotate_T = update_pose(np.eye(4), rotate=np.pi/2, rotate_axis="y")
    # obj_verts = transform_pcd(obj_verts, rotate_T)
    # hand_verts = transform_pcd(hand_verts, rotate_T)
    # my_palm_T = rotate_T @ my_palm_T
    # hand_mesh.transform(rotate_T)
    # obj_mesh.transform(rotate_T)

    # scale = 1.5
    # hand_verts = hand_verts * scale
    # obj_verts = obj_verts * scale
    # my_palm_T[:3, 3] = my_palm_T[:3, 3] * scale
    # hand_mesh.scale(scale, center=np.array([0, 0, 0]))
    # obj_mesh.scale(scale, center=np.array([0, 0, 0]))

    # random_pc_idx = np.random.choice(len(obj_verts), min(len(obj_verts), 1024), replace=False)
    # obj_verts = obj_verts[random_pc_idx]
    
    # cat_pc = np.concatenate([hand_verts, obj_verts], axis=0)
    # cat_colors = np.array([[255, 0, 0]] * len(hand_verts) + [[0, 255, 0]] * len(obj_verts))
    # obj_pc_colors = np.array([[0, 255, 0]] * len(obj_verts))
    # hand_pc_colors = np.array([[255, 0, 0]] * len(hand_verts))

    # viser_grasp.add_mesh(obj_mesh)
    # viser_grasp.wait_for_reset()

    # # viser_grasp.add_pcd(cat_pc, colors=cat_colors, point_size=0.001)
    # # viser_grasp.add_mesh(hand_mesh)
    # viser_grasp.add_grasp(my_palm_T, z_direction=False)
    # viser_grasp.wait_for_reset()

    # viser_grasp.add_pcd(obj_verts, colors=obj_pc_colors, point_size=0.001)
    # viser_grasp.wait_for_reset()
    # viser_grasp.add_pcd(hand_verts, colors=hand_pc_colors, point_size=0.001)
    # viser_grasp.add_mesh(hand_mesh)
    # viser_grasp.wait_for_reset()
    # # viser_grasp.add_grasp(my_palm_T, z_direction=False)
    # # viser_grasp.wait_for_reset()
    # pass

