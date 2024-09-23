import os
import shutil
import random


if __name__ == '__main__':
    target_num = 300
    random.seed(0)

    dataset_path = '/home/huangdehao/Projects/handgrasp_ws/2_graspnet_baseline/data/grasp_CONG'
    output_path = '/home/huangdehao/Projects/handgrasp_ws/2_graspnet_baseline/data/grasp_CONG_small'

    data_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    selected_data_paths = random.sample(data_paths, target_num)
    for path in selected_data_paths:
        data_fname = os.path.basename(path)
        shutil.copy(path, os.path.join(output_path, data_fname))


    