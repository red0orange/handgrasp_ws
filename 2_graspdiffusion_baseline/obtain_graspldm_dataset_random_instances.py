import os
import sys
import json
import shutil
import random
import pickle as pkl

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = cur_dir

from tqdm import tqdm
import numpy as np
import open3d as o3d

# 主要需要处理两个问题：
# 1. 在 pkl 里增加完整点云的数据，用于 Graspdiff 训练
# 2. 按照 GraspLDM 的数据量，train 是 63 个类别的 1100 个实例，验证是同样 63 个类别的任意其他 400 个实例（先按简单的来）

if __name__ == '__main__':
    cong_dataset_root_dir = os.path.join(project_dir, 'data', 'grasp_CONG')
    save_dataset_root_dir = os.path.join(project_dir, 'data', 'grasp_CONG_graspldm')
    mesh_root = os.path.join(project_dir, 'data/obj_ShapeNetSem/models-OBJ/models')
    os.makedirs(save_dataset_root_dir, exist_ok=True)

    split_json_path = os.path.join(save_dataset_root_dir, 'split.json')
    data_save_dir = os.path.join(save_dataset_root_dir, 'data')
    os.makedirs(data_save_dir, exist_ok=True)

    # 获得有效的数据
    pkl_data_paths = [os.path.join(cong_dataset_root_dir, i) for i in os.listdir(cong_dataset_root_dir)]
    if os.path.exists(os.path.join(save_dataset_root_dir, 'valid_pkl_data_paths.txt')):
        valid_pkl_data_paths = np.loadtxt(os.path.join(save_dataset_root_dir, 'valid_pkl_data_paths.txt'), dtype=str)
        valid_pkl_data_paths = [os.path.join(cong_dataset_root_dir, os.path.basename(i)) for i in valid_pkl_data_paths]
    else:
        valid_pkl_data_paths = []
        for i, pickle_path in tqdm(enumerate(pkl_data_paths), total=len(pkl_data_paths)):
            pickle_fname = os.path.basename(pickle_path).split(".")[0]
            pickle_data = pkl.load(open(pickle_path, "rb"))

            obj_cat = pickle_fname.split("_")[1]
            mesh_fname = os.path.basename(pickle_data["mesh/file"])
            mesh_scale = pickle_data["mesh/scale"]
            mesh_path = os.path.join(mesh_root, mesh_fname)

            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_vertices_num = np.array(mesh.vertices).shape[0]
            if (not os.path.exists(mesh_path)) or (mesh_vertices_num < 50):
                print("Skipping", mesh_path, "not found or too small")
                continue

            valid_pkl_data_paths.append(pickle_path)
        np.savetxt(os.path.join(save_dataset_root_dir, 'valid_pkl_data_paths.txt'), valid_pkl_data_paths, fmt='%s')
    print("Valid Ratio: {}/{}".format(len(valid_pkl_data_paths), len(pkl_data_paths)))

    # 
    data_dict = {}
    for i, pickle_path in enumerate(valid_pkl_data_paths):
        pickle_fname = os.path.basename(pickle_path).split(".")[0]
        obj_cat = pickle_fname.split("_")[1]
        instance_id = pickle_fname.split("_")[2]

        if obj_cat not in data_dict:
            data_dict[obj_cat] = []
        data_dict[obj_cat].append(pickle_path)
    
    # 
    obj_cats = list(data_dict.keys())
    per_cat_max_num = 60
    seed_num = 0
    while True:
        random.seed(seed_num)
        random.shuffle(obj_cats)
        selected_obj_cats = obj_cats[:63]
        selected_obj_instances = []
        for obj_cat in selected_obj_cats:
            selected_obj_instances.extend(random.sample(data_dict[obj_cat], min(len(data_dict[obj_cat]), per_cat_max_num)))
        if len(selected_obj_instances) >= 1500:
            random.shuffle(selected_obj_instances)
            selected_obj_instances = selected_obj_instances[:1500]
            break
        seed_num += 1

    # 划分数据集
    train_pkl_data_paths = selected_obj_instances[:1100]
    valid_pkl_data_paths = selected_obj_instances[1100:]
    split_json_data = {"train": [os.path.basename(i) for i in train_pkl_data_paths], "valid": [os.path.basename(i) for i in valid_pkl_data_paths]}
    with open(split_json_path, "w") as f:
        json.dump(split_json_data, f)

    # 增加完整点云的数据
    sampled_pc_num = [1024, 2048, 4096]
    for i, pickle_path in tqdm(enumerate(selected_obj_instances), total=len(selected_obj_instances)):
        pickle_fname = os.path.basename(pickle_path).split(".")[0]
        pickle_data = pkl.load(open(pickle_path, "rb"))

        mesh_fname = os.path.basename(pickle_data["mesh/file"])
        mesh_scale = pickle_data["mesh/scale"]
        mesh_path = os.path.join(mesh_root, mesh_fname)
        if not os.path.exists(mesh_path):
            print("Skipping", mesh_path, "not found")
            continue
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.scale(mesh_scale, center=np.zeros(3))
        
        # # @note debug vis
        # from roboutils.vis.viser_grasp import ViserForGrasp
        # grasp_viser = ViserForGrasp()
        # rendering_pcs = pickle_data["rendering/point_clouds"]
        # rendering_camera_Ts = pickle_data["rendering/camera_poses"]
        # grasp_Ts = list(pickle_data["grasps/transformations"])
        # grasp_successes = pickle_data["grasps/successes"]
        # success_grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if grasp_successes[i]]
        # for render_i in range(len(rendering_pcs)):
        #     rendering_pc = rendering_pcs[render_i]
        #     rendering_camera_T = rendering_camera_Ts[render_i]

        #     world_pc = transform_pcd(rendering_pc, np.linalg.inv(rendering_camera_T))

        #     # sample_grasp_Ts = random.sample(success_grasp_Ts, 50)
        #     sample_grasp_Ts = random.sample(grasp_Ts, 50)
        #     grasp_viser.vis_grasp_scene(sample_grasp_Ts, pc=world_pc, mesh=mesh, z_direction=True, max_grasp_num=50)
        #     # grasp_viser.vis_grasp_scene(success_grasp_Ts, pc=world_pc, mesh=None, z_direction=True)
        #     grasp_viser.wait_for_reset()

        # 采样点云
        for pc_num in sampled_pc_num:
            sampled_pc = mesh.sample_points_uniformly(number_of_points=pc_num)
            sampled_pc = np.array(sampled_pc.points)
            pickle_data["sampled_pc_{}".format(pc_num)] = sampled_pc
        # 保存数据
        save_path = os.path.join(data_save_dir, "{}.pickle".format(pickle_fname))
        with open(save_path, "wb") as f:
            pkl.dump(pickle_data, f)

    pass

    