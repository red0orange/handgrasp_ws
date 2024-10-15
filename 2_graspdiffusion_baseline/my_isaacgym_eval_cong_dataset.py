import os
import sys
import shutil
import random
import json
proj_dir = os.path.dirname(os.path.abspath(__file__))
mesh_root = os.path.join(proj_dir, 'data/obj_ShapeNetSem/models-OBJ/models')

import pickle
from tqdm import tqdm
import numpy as np
import open3d as o3d

from roboutils.vis.viser_grasp import ViserForGrasp

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = cur_dir


def export_isaacgym_eval_data():
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


    # 将结果转换为 IsaacGym 格式
    grasp_viser = ViserForGrasp()
    isaacgym_eval_data_dict = {}
    for i, pickle_path in tqdm(enumerate(valid_pkl_data_paths), total=len(valid_pkl_data_paths)):
        pickle_data = pickle.load(open(pickle_path, 'rb'))
        mesh_fname = os.path.basename(pickle_data["mesh/file"])
        mesh_path = os.path.join(mesh_root, mesh_fname)
        mesh_scale = pickle_data["mesh/scale"]

        pickle_fname = os.path.basename(pickle_path).split(".")[0]
        pickle_data = pickle.load(open(pickle_path, "rb"))

        obj_cat = pickle_fname.split("_")[1]
        mesh_fname = os.path.basename(pickle_data["mesh/file"])
        mesh_scale = pickle_data["mesh/scale"]
        mesh_path = os.path.join(mesh_root, mesh_fname)

        grasp_Ts = list(pickle_data["grasps/transformations"])
        grasp_successes = pickle_data["grasps/successes"]
        success_grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if grasp_successes[i]]

        # # for debug vis
        # rendering_pcs = pickle_data["rendering/point_clouds"]
        # rendering_camera_Ts = pickle_data["rendering/camera_poses"]
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # mesh.scale(mesh_scale, center=np.zeros(3))
        # sample_grasp_Ts = random.sample(grasp_Ts, 50)
        # grasp_viser.vis_grasp_scene(sample_grasp_Ts, mesh=mesh, z_direction=True, max_grasp_num=50)
        # grasp_viser.wait_for_reset()
            
        # export
        isaacgym_eval_data_dict[i] = {}
        isaacgym_eval_data_dict[i]['mesh_path'] = mesh_path
        isaacgym_eval_data_dict[i]['mesh_scale'] = mesh_scale
        isaacgym_eval_data_dict[i]['grasp_Ts'] = success_grasp_Ts
        isaacgym_eval_data_dict[i]['mesh_T'] = np.eye(4)

        isaacgym_eval_data_dict[i]['ori_cong_pickle_name'] = os.path.basename(pickle_path)
        isaacgym_eval_data_dict[i]['all_grasp_Ts'] = grasp_Ts
        isaacgym_eval_data_dict[i]['all_grasp_success'] = grasp_successes
        pass
    np.save(os.path.join(save_dataset_root_dir, 'cong_isaacgym_eval_data.npy'), isaacgym_eval_data_dict)


def get_each_category_rep_data():
    ori_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/cong_isaacgym_eval_data.npy"
    ori_data = np.load(ori_data_path, allow_pickle=True).item()
    ori_data_ids = list(ori_data.keys())

    obj_cat_data_dict = {}
    for data_id, data in ori_data.items():
        ori_cong_pickle_name = data['ori_cong_pickle_name']
        obj_cat = ori_cong_pickle_name.split("_")[1]
        if obj_cat not in obj_cat_data_dict:
            obj_cat_data_dict[obj_cat] = []
        obj_cat_data_dict[obj_cat].append(data_id)
    
    rep_data_ids = []
    for obj_cat, data_ids in obj_cat_data_dict.items():
        obj_cat_data_dict[obj_cat] = random.shuffle(data_ids)
        rep_data_ids.append(data_ids[0])

    rep_data = {}
    for data_id in rep_data_ids:
        rep_data[data_id] = ori_data[data_id]
    save_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/cong_cat_rep_isaacgym_eval_data.npy"
    np.save(save_path, rep_data)


def get_train_relabel_data():
    ori_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/cong_isaacgym_eval_data.npy"
    data_split_json_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/split.json"
    split_data = json.load(open(data_split_json_path, 'r'))
    train_pickle_paths = split_data['train']

    ori_data = np.load(ori_data_path, allow_pickle=True).item()
    ori_data_ids = list(ori_data.keys())

    train_data = {}
    for data_id, data in ori_data.items():
        ori_cong_pickle_name = data['ori_cong_pickle_name']
        if os.path.basename(ori_cong_pickle_name) in train_pickle_paths:
            train_data[data_id] = data
    print("Train Data Size:", len(train_data))
    save_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/cong_train_isaacgym_eval_data.npy"
    np.save(save_path, train_data)


def isaacgym_eval_data_to_graspldm_dataset(eval_data_path, start_idx):
    isaacgym_dir = os.path.join(os.path.dirname(proj_dir), "3rd_isaacgym_evaluation")
    sys.path.append(isaacgym_dir)
    from isaacgym_eval import IsaacGymGraspEva, CONGIsaacGymGraspEva

    cache_dir = os.path.join(proj_dir, 'data/IsaacGymCache')
    # output_dir = os.path.join(proj_dir, 'data/grasp_CONG_graspldm/CONG_ori_eval_results')
    output_dir = os.path.join(proj_dir, 'data/grasp_CONG_graspldm/CONG_train_eval_results')
    # output_dir = os.path.join(proj_dir, 'data/grasp_CONG_graspldm/CONG_rep_eval_results')
    # test_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/cong_isaacgym_eval_data.npy"
    evaluator = CONGIsaacGymGraspEva(eval_data_path, output_dir, cache_dir, n_envs=50, start_idx=start_idx)
    evaluator.eval(debug_vis=False)


def debug_vis_eval_results():
    grasp_viser = ViserForGrasp()

    debug_vis_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/CONG_rep_eval_results/constrained_ToyFigure_3ba8803f914ac8d767ff608a5fbe6aa8_0_0034366994172005437.npy"
    data = np.load(debug_vis_path, allow_pickle=True).item()

    mesh_path = data['mesh_path']
    mesh_scale = data['mesh_scale']
    grasp_Ts = data['grasp_Ts']
    mesh_T = data['mesh_T']

    isaacgym_eval_res = data["eva_result_success"]
    isaacgym_eval_success_grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if isaacgym_eval_res[i]]
    print(len(isaacgym_eval_success_grasp_Ts) / len(grasp_Ts))

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.scale(mesh_scale, center=np.zeros(3))
    mesh.transform(mesh_T)

    mesh_center = np.mean(np.array(mesh.vertices), axis=0)
    mesh.translate(-mesh_center)

    grasp_viser.vis_grasp_scene(grasp_Ts, mesh=mesh, z_direction=True, max_grasp_num=50)
    grasp_viser.wait_for_reset()
    grasp_viser.vis_grasp_scene(isaacgym_eval_success_grasp_Ts, mesh=mesh, z_direction=True, max_grasp_num=50)
    grasp_viser.wait_for_reset()
    pass


def transfere_eval_results():
    ori_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/data"
    save_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/eval_data"
    eval_results_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/CONG_train_eval_results"
    split_json_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/split.json"

    split_data = json.load(open(split_json_path, 'r'))

    valid_pickle_paths = split_data['valid']
    data_files = [os.path.join(ori_dir, i) for i in valid_pickle_paths]
    for data_file in data_files:
        save_path = os.path.join(save_dir, os.path.basename(data_file))
        shutil.copy(data_file, save_path)
    pass

    train_pickle_paths = split_data['train']
    data_files = [os.path.join(ori_dir, i) for i in train_pickle_paths]
    eval_result_files = [os.path.join(eval_results_dir, i.split(".")[0]+".npy") for i in train_pickle_paths]

    for data_file, eval_result_file in zip(data_files, eval_result_files):
        data = pickle.load(open(data_file, 'rb'))
        eval_result = np.load(eval_result_file, allow_pickle=True).item()

        ori_grasp_success = data['grasps/successes']
        eval_grasp_success = eval_result['eva_result_success']

        data['grasps/ori_successes'] = ori_grasp_success.copy()
        ori_grasp_success[ori_grasp_success == 1] = eval_grasp_success
        data['grasps/successes'] = ori_grasp_success

        save_path = os.path.join(save_dir, os.path.basename(data_file))
        pickle.dump(data, open(save_path, 'wb'))
        pass


def select_my_valid_intances_based_on_valid_results():
    eval_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_499_20240926-191802_detectiondiffusion/isaacgym_eval_results.npy"
    eval_result_txt_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_499_20240926-191802_detectiondiffusion/isaacgym_eval_result.txt"
    save_txt_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_499_20240926-191802_detectiondiffusion/selected_good_instances.txt"

    eval_data = np.load(eval_data_path, allow_pickle=True).item()
    with open(eval_result_txt_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    success_cases = [int(line.split(" ")[0]) for line in lines]
    all_num = [int(line.split(" ")[1]) for line in lines]
    success_ratios = [success_cases[i] / all_num[i] for i in range(len(success_cases))]

    sorted_success_ids = np.argsort(success_ratios)[::-1]
    valid_good_ids = sorted_success_ids[:400]

    selected_valid_data = [eval_data[i] for i in valid_good_ids]
    selected_valid_mesh_names = [os.path.basename(selected_valid_data[i]['mesh_path']).split(".")[0] for i in range(len(selected_valid_data))]
    np.savetxt(save_txt_path, selected_valid_mesh_names, fmt='%s')

    ori_split_json_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/split.json"
    save_split_json_path = os.path.join(os.path.dirname(save_txt_path), "selected_valid_split.json")
    ori_json_data = json.load(open(ori_split_json_path, 'r'))

    save_json_data = {}
    save_json_data['train'] = ori_json_data['train']
    save_json_data['valid'] = []

    for valid_pickle_name in ori_json_data['valid']:
        mesh_name = os.path.basename(valid_pickle_name).split("_")[2]
        if mesh_name in selected_valid_mesh_names:
            save_json_data['valid'].append(valid_pickle_name)
    json.dump(save_json_data, open(save_split_json_path, 'w'))


if __name__ == '__main__':
    # export_isaacgym_eval_data()

    # get_train_relabel_data()

    # eval_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/cong_train_isaacgym_eval_data.npy"
    # start_idx = 0
    # isaacgym_eval_data_to_graspldm_dataset(eval_data_path, start_idx)

    # eval_data_path = sys.argv[1]
    # start_idx = int(sys.argv[2])
    # isaacgym_eval_data_to_graspldm_dataset(eval_data_path, start_idx)

    # debug_vis_eval_results()

    # split_cong_eval_data()

    # get_each_category_rep_data()

    # transfere_eval_results()

    select_my_valid_intances_based_on_valid_results()
