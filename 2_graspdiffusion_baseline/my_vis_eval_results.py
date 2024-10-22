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


def parse_string(input_str):
    parts = input_str.split(' ', 2)  # 将字符串分割为三部分
    first_number = int(parts[0])     # 第一个数
    second_number = int(parts[1])    # 第二个数
    # 将第三部分的字符串（列表形式）转换为浮点数列表
    number_list = list(map(float, parts[2].strip('[]').split()))
    return first_number, second_number, number_list


if __name__ == "__main__":
    # pickle_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/scorenet_with_energy_sort_eval.pkl"
    # eval_results_txt_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/scorenet_with_energy_eval_results.txt"

    # pickle_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/scorenet_with_random_sort_eval.pkl"
    # eval_results_txt_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/scorenet_with_random_eval_results.txt"

    pickle_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/eval_oakink_only_s.pkl"
    eval_results_txt_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/eval_oakink_only_s_isaacgym_eval_results.txt"

    # @note 读取评估结果 txt 文件
    with open(eval_results_txt_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    results = [parse_string(line) for line in lines]
    
    success_cases = [int(line[0]) for line in results]
    all_num = [int(line[1]) for line in results]
    success_ratios = [success_cases[i] / all_num[i] for i in range(len(success_cases))]
    success_instances = [list(map(bool, line[2])) for line in results]

    # @note 读取 pickle 文件
    with open(pickle_path, 'rb') as f:
        eval_data = pickle.load(f)
    
    # 逐个查看
    viser_grasp = ViserForGrasp()
    for data_idx in range(len(eval_data)):
        data = eval_data[data_idx]
        xyz = data['xyz']
        grasp_Ts = data['grasp_Ts']

        success_case = success_cases[data_idx]
        all_num_case = all_num[data_idx]
        success_ratio = success_ratios[data_idx]
        success_instance = success_instances[data_idx]

        # grasp_colors = []
        # for grasp_i in range(len(grasp_Ts)):
        #     if success_instance[grasp_i]:
        #         grasp_colors.append([0, 255, 0])
        #     else:
        #         grasp_colors.append([0, 120, 0])
        grasp_colors = None
        
        viser_grasp.vis_grasp_scene(grasp_Ts, pc=xyz, grasp_colors=grasp_colors, mesh=None, z_direction=True)
        viser_grasp.wait_for_reset()
    pass