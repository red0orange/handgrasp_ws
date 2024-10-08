import os
import numpy as np
import pickle
import open3d as o3d
from tqdm import tqdm

from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


def transform_pcd(pcd, T):
    return (T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]


if __name__ == '__main__':
    save_root = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_Oakink"
    save_mesh_path = os.path.join(save_root, "meshes")
    save_data_path = os.path.join(save_root, "data")

    # 导出好的 oakink_shape 数据
    data_path = "/home/red0orange/Projects/one_shot_il_ws/OakInk/processed_oakink_shape_all_data.pkl"
    data = pickle.load(open(data_path, "rb"))

    viser_grasp = ViserForGrasp()

    last_cat_id = None
    cat_cnt = 0
    for data_idx, item in tqdm(enumerate(data), total=len(data)):
        obj_cat_id = item['cate_id']
        if last_cat_id is not None and (last_cat_id == obj_cat_id) and (cat_cnt > 10):
            last_cat_id = obj_cat_id
            continue
        if last_cat_id != obj_cat_id:
            cat_cnt = 0
        last_cat_id = obj_cat_id
        cat_cnt += 1

        hand_verts = item['verts']
        hand_faces = item['hand_th_faces']
        obj_verts = item['obj_verts']
        obj_faces = item['obj_faces']
        my_palm_T = item['my_palm_T']

        # 生成 mesh
        obj_mesh = o3d.geometry.TriangleMesh()
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
        # obj_mesh.compute_vertex_normals()
        # obj_mesh.compute_triangle_normals()

        # 处理 palm_T
        my_palm_T = update_pose(my_palm_T, rotate=np.pi/2, rotate_axis="x")
        # 微调一下姿态
        my_palm_T = update_pose(my_palm_T, rotate=np.pi/7, rotate_axis="z")
        # 改为 z 轴朝前
        my_palm_T = update_pose(my_palm_T, rotate=-np.pi / 2, rotate_axis='x')
        my_palm_T = update_pose(my_palm_T, rotate=np.pi / 2, rotate_axis='y')

        random_pc_idx = np.random.choice(len(obj_verts), min(len(obj_verts), 1024), replace=False)
        obj_verts = obj_verts[random_pc_idx]
        
        cat_pc = np.concatenate([hand_verts, obj_verts], axis=0)
        cat_colors = np.array([[255, 0, 0]] * len(hand_verts) + [[0, 255, 0]] * len(obj_verts))

        # # @note debug vis
        # viser_grasp.add_pcd(cat_pc, colors=cat_colors)
        # viser_grasp.add_grasp(my_palm_T, z_direction=True)
        # break_flag = viser_grasp.wait_for_reset()
        # if break_flag:
        #     break

        mesh_save_path = os.path.join(save_mesh_path, f"{data_idx:06d}.obj")
        o3d.io.write_triangle_mesh(mesh_save_path, obj_mesh)
        data_save_path = os.path.join(save_data_path, f"{data_idx:06d}.pkl")
        with open(data_save_path, "wb") as f:
            pickle.dump({
                "hand_verts": hand_verts,
                "hand_faces": hand_faces,
                "palm_T": my_palm_T,
            }, f)