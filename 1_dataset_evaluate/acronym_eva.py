import os
CURFILEDIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from tqdm import tqdm

from AcronymDataset import AcronymDataset
from IsaacSimEva import IsaacSimEva


if __name__ == '__main__':
    cache_dir = "/home/red0orange/Projects/handgrasp_ws/0_Data/IsaacSimGraspEvaCache"
    grasps_dir = "/home/red0orange/Projects/handgrasp_ws/0_Data/grasp_Acronym/grasps"
    meshes_dir = "/home/red0orange/Projects/handgrasp_ws/0_Data/grasp_Acronym/meshes"

    dataset = AcronymDataset(grasps_dir, meshes_dir)

    exp_name = "Acronym_Eva"
    # data_dict = {}
    # for i in tqdm(range(len(dataset)), total=len(dataset), desc="Converting data_dict"):
    #     dataset_item = dataset[i]

    #     grasp_Ts, grasp_success, mesh_path, mesh_scale, _ = dataset_item

    #     data_dict[i] = {
    #         "grasp_Ts": grasp_Ts,
    #         "mesh_path": mesh_path,
    #         "mesh_scale": mesh_scale,
    #         "mesh_T": np.eye(4),
    #     }
    # np.save("tmp.npy", data_dict)

    # @note for debug
    data_dict = np.load("tmp.npy", allow_pickle=True).item()
    
    # @note 开始评估
    isaac_sim_eva = IsaacSimEva(exp_name, data_dict, cache_dir=cache_dir, debug=False, rewrite_json=True)
    isaac_sim_eva.eval()

    pass