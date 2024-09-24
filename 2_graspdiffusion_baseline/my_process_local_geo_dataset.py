import os
import tqdm

import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader

from dataset._AcronymDataset import _AcronymDataset
from scipy.spatial import KDTree

from roboutils.vis.grasp import draw_scene, get_gripper_keypoints


if __name__ == "__main__":
    # test
    # data = np.load("/home/red0orange/Projects/grasp_diffusion_ws/data/my_local_geometry_train/data.npy", allow_pickle=True)
    # #############################

    # data_dir = "/home/red0orange/Projects/grasp_diffusion_ws/data"
    # dataset = _AcronymDataset(data_dir=data_dir, mode='train', n_pointcloud=2048, n_density=80, n_coords=400, augmented_rotation=True, center_pcl=True, split=True, partial=False, use_loca_geo_cache=True, fix_local_geo=True)

    # data_list = []
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # total_steps = len(data_loader)
    # pbar = tqdm.tqdm(total=total_steps)
    # for i, data in enumerate(data_loader):
    #     ori_xyz, xyz, T, _, _ = data
    #     pass
    # #########################


    # data_save_path = "/home/red0orange/Projects/grasp_diffusion_ws/data/my_local_geometry_train/data.npy"
    data_save_path = "/home/red0orange/Projects/grasp_diffusion_ws/data/my_local_geometry_test/data.npy"

    data_dir = "/home/red0orange/Projects/grasp_diffusion_ws/data"
    dataset = _AcronymDataset(
        data_dir=data_dir, mode='test', n_pointcloud=2048, n_density=80, n_coords=400, augmented_rotation=True, center_pcl=True, split=True, partial=False, 
        use_loca_geo_cache=False, fix_local_geo=True
    )

    data_list = []
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=24)

    total_steps = len(data_loader)
    pbar = tqdm.tqdm(total=total_steps)
    for i, data in enumerate(data_loader):
        ori_xyz, xyz, T, _, _ = data

        for batch_i in range(ori_xyz.shape[0]):
            data_dict = dict(
                ori_xyz=ori_xyz[batch_i].numpy(),
                xyz=xyz[batch_i].numpy(),
                T=T[batch_i].numpy(),
            )
            data_list.append(data_dict)

        pbar.update(1) 
        # break
    pbar.close()

    np.save(data_save_path, data_list)
    pass


