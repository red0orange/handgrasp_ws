import os
import time
import glob
from os.path import join as opj
import pickle as pkl

import h5py
import numpy as np
import trimesh
import pickle

import torch
from torch.utils.data import Dataset, DataLoader


class AcronymGrasps():
    def __init__(self, filename, std_scale: float = 1.0):
        """Read and process grasps data. If it is to augment the data files with priors,
        create priors and update local files.
        Note: if it is to use priors but there are no priors, one set of augmented priors will be created.

        Args:
            is_to_use_priors (bool):  flag whether we will read prior grasps, and whether they will be used.
            is_to_augment_priors (bool):  flag whether we will create new random noise prior grasps and save them in the
                original data file. This is to force creating a new set of priors.
            std_scale (float): if we are to augment new priors, the scale of std of Gaussian noises to add.
        """

        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            with h5py.File(filename, "r") as data:
                self.mesh_fname = data["object/file"][()].decode('utf-8')
                self.mesh_type = self.mesh_fname.split('/')[1]
                self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
                self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.grasps, self.success = self.load_grasps(filename)
        # self.prior_grasps, self.prior_success = self.load_prior_grasps(filename)
        self.prior_grasps, self.prior_success = self.grasps, self.success
        good_idxs = np.argwhere(self.success == 1)[:, 0]
        bad_idxs = np.argwhere(self.success == 0)[:, 0]
        self.good_grasps = self.grasps[good_idxs, ...]
        self.bad_grasps = self.grasps[bad_idxs, ...]

        # Work with priors
        self.filename = filename

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            with h5py.File(filename, "r") as data:
                T = np.array(data["grasps/transforms"])
                success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def load_prior_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            with h5py.File(filename, "r") as data:
                T = np.array(data["prior_grasp/transforms"])
                success = np.array(data["prior_grasp/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def create_prior_grasps(self, std_scale: float = 1.0):
        """Create a numpy.ndarray of prior grasps equal to the same num of good grasps.
            - We first calculate the population std, sigma, for each element in grasp tensor across all good grasps
            - Then add Gaussian noise to each element following N(0, std_scale * sigma)
            Overall, we create a same num of Gaussian noise augmented prior grasps as original grasps.

        Args:
            std_scale (float): Standard dev multiplier for random Gaussian noises to added to good grasps.

        Returns:
            np.ndarray: Homogeneous matrices describing the prior grasp poses.
        """
        good_grasps = self.good_grasps
        grasp_stds = torch.Tensor(good_grasps.std(axis=0))
        # Use torch since its easier
        noises = std_scale * torch.normal(
            torch.zeros(good_grasps.shape), grasp_stds
        )
        prior_grasps = good_grasps + noises.numpy()
        return prior_grasps

    def load_mesh(self, data_dir):
        mesh_path_file = os.path.join(data_dir, self.mesh_fname)
        with open(mesh_path_file, "r") as file:
            mesh = trimesh.load(file, file_type='obj', force='mesh')

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh


class AcronymDataset(Dataset):
    def __init__(self, grasps_dir, meshes_dir):
        self.grasps_dir = grasps_dir
        self.meshes_dir = meshes_dir

        self.ori_grasps_files = sorted(glob.glob(self.grasps_dir + '/*.h5'))
        self.grasps_files = []
        self.meshes_files = []
        for grasp_file in self.ori_grasps_files:
            grasp_name = os.path.basename(grasp_file)
            obj_cat, obj_name, _ = grasp_name.split('_')
            mesh_file = os.path.join(self.meshes_dir, obj_cat, obj_name + '.obj')
            if os.path.exists(mesh_file):
                self.grasps_files.append(grasp_file)
                self.meshes_files.append(mesh_file)
        pass

    def __len__(self):
        return len(self.grasps_files)
    
    def __getitem__(self, index):
        grasp_file = self.grasps_files[index]
        mesh_file = self.meshes_files[index]
        grasp_obj = AcronymGrasps(grasp_file)

        grasps = grasp_obj.grasps
        success = grasp_obj.success        
        mesh_scale = grasp_obj.mesh_scale

        return grasps, success, mesh_file, mesh_scale, grasp_obj


########################################
# for grasp diffusion
########################################
# from scipy.spatial.transform import Rotation
# from scipy.stats import special_ortho_group
# from scipy.spatial import KDTree
# from utils.rotate_rep import matrix_to_rotation_6d_np
# from roboutils.proj_llm_robot.pose_transform import update_pose
# from roboutils.vis.grasp import draw_scene, get_gripper_keypoints
# from roboutils.render_partial_pcd import Pytorch3dPartialPCDRenderer
# from roboutils.render_partial_pcd import Open3dPartialPCDRenderer

# def interpolate_3d_points(p1, p2, n):
#     """
#     在两个3D点p1和p2之间进行线性插值，包括两端点共n个点。

#     参数:
#         p1 (array-like): 第一个点的坐标，格式为[x1, y1, z1]。
#         p2 (array-like): 第二个点的坐标，格式为[x2, y2, z2]。
#         n (int): 需要生成的总点数，包括两个端点。

#     返回:
#         np.array: 包含n个点的坐标的数组。
#     """
#     # 将输入的p1和p2转换为Numpy数组，以便进行向量运算
#     p1 = np.array(p1)
#     p2 = np.array(p2)
    
#     # 生成从0到1的n个等间隔数值，包括0和1
#     t = np.linspace(0, 1, n)
    
#     # 计算插值点
#     # 每个点都是p1和p2的线性组合，系数是t和1-t
#     points = (1 - t)[:, None] * p1 + t[:, None] * p2
    
#     return points


# def visualize_ray(mesh, ray_origins, ray_directions, locations, index_ray, index_tri):
#     # stack rays into line segments for visualization as Path3D
#     ray_visualize = trimesh.load_path(
#         np.hstack((ray_origins, ray_origins + ray_directions)).reshape(-1, 2, 3)
#     )

#     # make mesh transparent- ish
#     mesh.visual.face_colors = [100, 100, 100, 100]

#     # create a visualization scene with rays, hits, and mesh
#     scene = trimesh.Scene([mesh, ray_visualize, trimesh.points.PointCloud(locations)])

#     # display the scene
#     scene.show()
#     pass


# class _AcronymDataset(Dataset):
#     def __init__(self, data_dir, mode='train', n_pointcloud=2048, n_density=80, n_coords=400,
#                  augmented_rotation=True, center_pcl=True, split=True, partial=True, fix_local_geo=False, use_loca_geo_cache=True):
#         # self.class_type = [
#         #     'Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
#         #     'Plate', 'ScrewDriver', 'WineBottle', 'Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
#         #     'Book', 'Books', 'Camera', 'CerealBox', 'Cookie', 'Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
#         #     'PillBottle', 'Plant', 'PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
#         #     'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan', 'Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
#         #     'ToyFigure', 'Wallet', 'WineGlass',
#         #     'Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick'
#         # ]

#         self.data_dir = data_dir
#         self.type = mode
#         self.grasps_dir = os.path.join(self.data_dir, 'grasps')
#         self.all_class_types = os.listdir(self.grasps_dir)

#         train_ratio = 0.8
#         test_ratio = 0.2

#         self.train_class_types = self.all_class_types[:int(len(self.all_class_types) * train_ratio)]
#         self.test_class_types = self.all_class_types[int(len(self.all_class_types) * train_ratio):]

#         if self.type == 'train':
#             self.train_grasp_files = []
#             train_err_cnt = 0
#             for class_type_i in self.train_class_types:
#                 cls_grasps_files = sorted(glob.glob(self.grasps_dir + '/' + class_type_i + '/*.h5'))

#                 for grasp_file in cls_grasps_files:
#                     g_obj = AcronymGrasps(grasp_file)

#                     grasp_name = os.path.basename(grasp_file)
#                     if grasp_name in ["Sword_f96883ba158a3ed47a88a2ad67bfd073_7.748767827814483e-05.h5", "TV_907b90beaf5e8edfbaf98569d8276423_7.669068045770162e-07.h5"]:
#                         train_err_cnt += 1
#                         continue

#                     # mesh_type = grasp_name.split('_')[0]
#                     # filename = grasp_name.split('_')[1]
#                     # sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename + '.json')
#                     # if not os.path.exists(sdf_file):
#                     #     continue

#                     ## Grasp File ##
#                     if g_obj.good_grasps.shape[0] > 0:
#                         self.train_grasp_files.append(grasp_file)
#         else:
#             self.test_grasp_files = []
#             test_err_cnt = 0
#             for class_type_i in self.test_class_types:
#                 cls_grasps_files = sorted(glob.glob(self.grasps_dir + '/' + class_type_i + '/*.h5'))

#                 for grasp_file in cls_grasps_files:
#                     g_obj = AcronymGrasps(grasp_file)

#                     # grasp_name = os.path.basename(grasp_file)
#                     # mesh_type = grasp_name.split('_')[0]
#                     # filename = grasp_name.split('_')[1]
#                     # sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename + '.json')
#                     # if not os.path.exists(sdf_file):
#                     #     test_err_cnt += 1
#                     #     continue

#                     ## Grasp File ##
#                     if g_obj.good_grasps.shape[0] > 0:
#                         self.test_grasp_files.append(grasp_file)
                
#         print("type: {}, num: {}".format(self.type, len(self.train_grasp_files) if self.type == 'train' else len(self.test_grasp_files)))
#         print("type: {}, num: {}".format(self.type, train_err_cnt if self.type == 'train' else test_err_cnt))

#         if self.type == 'train':
#             self.len = len(self.train_grasp_files)
#         else:
#             self.len = len(self.test_grasp_files)

#         self.ori_n_pointcloud = 8096  # 点云数量固定为 2048 个点
#         self.n_pointcloud = n_pointcloud  # 点云数量固定为 2048 个点
#         self.ori_n_density = 800
#         self.n_density = n_density        # 抓取数量固定为 80 个
#         self.n_occ = n_coords

#         ## Variables on Data
#         self.se3 = False
#         self.one_object = False
#         self.augmented_rotation = augmented_rotation
#         self.center_pcl = center_pcl
#         self.fix_local_geo = fix_local_geo

#         self.use_loca_geo_cache = (self.fix_local_geo and use_loca_geo_cache)
#         if self.type == 'train':
#             self.cache_path = os.path.join(self.data_dir, 'my_local_geometry_train/data.npy')
#         else:
#             self.cache_path = os.path.join(self.data_dir, 'my_local_geometry_test/data.npy')
#         self.local_geo_cache = None
#         if os.path.exists(self.cache_path) and self.use_loca_geo_cache:
#             print('Loading local geo cache')
#             self.local_geo_cache = np.load(self.cache_path, allow_pickle=True)
#             print('Loaded local geo cache')

#         self.pre_grasp = False
#         self.scale = 8.

#         self.partial = partial
#         if self.partial:
#             # self.renderer = Pytorch3dPartialPCDRenderer()
#             self.renderer = Open3dPartialPCDRenderer()
        
#         self.debug_cnt = 0

#     def __len__(self):
#         return self.len

#     def set_test_data(self):
#         self.len = len(self.test_grasp_files)
#         self.type = 'test'

#     def _get_grasps(self, grasp_obj):
#         try:
#             rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.ori_n_density)
#         except:
#             print('lets see')
#         H_grasps = grasp_obj.good_grasps[rix, ...]
#         H_vae_prior_grasps = grasp_obj.prior_grasps[rix, ...]
#         return H_grasps, H_vae_prior_grasps

#     def _get_sdf(self, grasp_obj, grasp_file):

#         mesh_fname = grasp_obj.mesh_fname   # 物体的 mesh 文件名
#         mesh_scale = grasp_obj.mesh_scale   # 物体的 mesh 缩放比例

#         mesh_type = mesh_fname.split('/')[1]
#         mesh_name = mesh_fname.split('/')[-1]
#         filename = mesh_name.split('.obj')[0]
#         sdf_file = os.path.join(self.data_dir, 'sdf', mesh_type, filename + '.json')

#         with open(sdf_file, 'rb') as handle:
#             sdf_dict = pickle.load(handle)

#         loc = sdf_dict['loc']
#         scale = sdf_dict['scale']
#         xyz = (sdf_dict['xyz'] + loc) * scale * mesh_scale
#         rix = np.random.permutation(xyz.shape[0])
#         xyz = xyz[rix[:self.n_occ], :]
#         sdf = sdf_dict['sdf'][rix[:self.n_occ]] * scale * mesh_scale
#         return xyz, sdf

#     def _get_mesh_pcl(self, grasp_obj):
#         mesh = grasp_obj.load_mesh(self.data_dir)
#         return mesh.sample(self.ori_n_pointcloud), mesh

#     def _get_partial_mesh_pcl(self, grasp_obj):
#         mesh = grasp_obj.load_mesh(self.data_dir)
#         centroid = mesh.centroid
#         H = np.eye(4)
#         H[:3, -1] = -centroid
#         mesh.apply_transform(H)
#         ######################

#         # time0 = time.time()

#         # P = self.scan_pointcloud.get_hq_scan_view(mesh)
#         # np.save('mesh.npy', {'vertices': mesh.vertices, 'faces': mesh.faces})
#         random_rot = Rotation.random().as_matrix()
#         P = self.renderer.render_partial_pcd(mesh.vertices, mesh.faces, random_rot, debug=False)

#         # print('Sample takes {} s'.format(time.time() - time0))

#         P += centroid
#         try:
#             # @note n_pointcloud
#             rix = np.random.randint(low=0, high=P.shape[0], size=self.n_pointcloud)
#         except:
#             print("scan view point size is not enough")
#             return None, None

#         return P[rix, :], mesh

#     def _get_item(self, index):
#         # @note for debug
#         # if self.debug_cnt < 32 * 207:
#         #     self.debug_cnt += 1
#         #     return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

#         if self.one_object:
#             index = 0

#         ## Load Files ##
#         if self.type == 'train':
#             # data/grasps/Plant/Plant_30a5d8cfbf35ee1de664ce1de3ddc8b4_0.004493535817825254.h5
#             grasps_obj = AcronymGrasps(self.train_grasp_files[index])
#             file_name = self.train_grasp_files[index]
#         else:
#             grasps_obj = AcronymGrasps(self.test_grasp_files[index])
#             file_name = self.test_grasp_files[index]
        
#         ## 物体的 SDF
#         # xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

#         ## PointCloud
#         # pcl: 采样的点云 (400, 3)
#         # mesh: trimesh mesh
#         if self.partial:
#             pcl, mesh = self._get_partial_mesh_pcl(grasps_obj)
#             if pcl is None:
#                 random_idx = np.random.randint(0, self.len)
#                 return self._get_item(random_idx)
#             # assert np.min(pcl) >= -1000.0 and np.max(pcl) <= 1000.0
#             if np.min(pcl) < -1000.0 or np.max(pcl) > 1000.0:
#                 print('Error: ', np.min(pcl), np.max(pcl))
#                 random_idx = np.random.randint(0, self.len)
#                 return self._get_item(random_idx)
#         else:
#             pcl, mesh = self._get_mesh_pcl(grasps_obj)

#         ## Grasps good/bad
#         # H_graps 是从好的抓取中随机选择 80 个
#         # H_vae_prior_grasps 是不管是否是好的抓取，随机选择 80 个
#         H_grasps, _ = self._get_grasps(grasps_obj)

#         # 夹爪位置是 pre_grasp 或是 grasp
#         if not self.pre_grasp:
#             for i in range(H_grasps.shape[0]):
#                 H_grasps[i] = update_pose(H_grasps[i], translate=[0, 0, 0.09])
        
#         ## rescale, rotate and translate ##
#         # 统一缩放
#         pcl = pcl * self.scale
#         mesh.apply_scale(self.scale)
#         H_grasps[..., :3, -1] = H_grasps[..., :3, -1] * self.scale

#         if self.center_pcl:
#             ## translate ##
#             mean = np.mean(pcl, 0)
#             pcl = pcl - mean
#             mesh.apply_translation(-mean)
#             H_grasps[..., :3, -1] = H_grasps[..., :3, -1] - mean

#         if self.augmented_rotation and self.type != "test":
#             ## Random rotation ##
#             R = special_ortho_group.rvs(3)   # 看上去是想高斯采样，但是里面看不出来
#             H = np.eye(4)
#             H[:3, :3] = R
#             pcl = np.einsum('mn,bn->bm', R, pcl)
#             H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)   # 应用随机的旋转
#             mesh.apply_transform(H)
        
#         # @note 随机采样 2048 个点，保留原始点云
#         ori_pcl = pcl.copy()
#         random_idx = np.random.randint(0, pcl.shape[0], self.n_pointcloud)
#         pcl = pcl[random_idx, ...]
        
#         ori_H_grasps = H_grasps.copy()
#         random_idx = np.random.randint(0, H_grasps.shape[0], self.n_density)
#         H_grasps = H_grasps[random_idx, ...]

#         # @note local geo
#         # 一直尝试直到找到一片满足 n_density 数量抓取的 local geo
#         if self.fix_local_geo and np.random.rand() > 0.5:
#             if self.use_loca_geo_cache and (self.local_geo_cache is not None):
#                 random_idx = np.random.randint(0, len(self.local_geo_cache), 1)
#                 data = self.local_geo_cache[random_idx[0]]
#                 ori_pcl, pcl, H_grasps = data['ori_xyz'], data['xyz'], data['T']

#                 # draw_scene(pcl, H_grasps, z_direction=True, scale=1.0/self.scale, max_grasps=800)
#             else:
#                 ori_pcl_kdtree = KDTree(ori_pcl)

#                 ray_origins = []
#                 ray_directions = []
#                 left_fingers, right_fingers = [], []
#                 grasp_keypoints = get_gripper_keypoints(ori_H_grasps, scale=self.scale)

#                 ray_num = 10
#                 for i in range(grasp_keypoints.shape[0]):
#                     left_finger = grasp_keypoints[i, [2, 3], :]
#                     right_finger = grasp_keypoints[i, [5, 6], :]
#                     ray_direction = np.array(right_finger[0] - left_finger[0])[None, ...].repeat(ray_num, axis=0)
#                     ray_origin = interpolate_3d_points(left_finger[0], left_finger[1], ray_num)

#                     ray_directions.append(ray_direction)
#                     ray_origins.append(ray_origin)
#                     left_fingers.append(left_finger)
#                     right_fingers.append(right_finger)
#                     pass
                    
#                 locations = []
#                 index_rays = []
#                 index_tris = []
#                 max_distance = np.sqrt(np.linalg.norm(right_fingers[0][0] - left_fingers[0][0])**2 + np.linalg.norm(left_fingers[0][1] - left_fingers[0][0])**2)
#                 for i in range(len(ray_origins)):
#                     ray_origin = ray_origins[i]
#                     ray_direction = ray_directions[i]

#                     left_finger = left_fingers[i]
#                     right_finger = right_fingers[i]
#                     finger = np.concatenate([left_finger, right_finger], axis=0)

#                     location, index_ray, index_tri = mesh.ray.intersects_location(
#                         ray_origins[i], ray_directions[i], multiple_hits=True
#                     )

#                     if location.shape[0] == 0:
#                         locations.append(location)
#                         index_rays.append(index_ray)
#                         index_tris.append(index_tri)
#                         continue

#                     # 保证是在夹爪之间的点
#                     distances = location[:, None, :] - finger[None, :, :]
#                     distances = np.linalg.norm(distances, axis=-1)
#                     max_distances = np.max(distances, axis=1)

#                     valid_idx = max_distances < max_distance
#                     location = location[valid_idx, ...]
#                     index_ray = index_ray[valid_idx, ...]
#                     index_tri = index_tri[valid_idx, ...]

#                     # # vis
#                     # visualize_ray(mesh, ray_origins[i], ray_directions[i], location, index_ray, index_tri)

#                     locations.append(location)
#                     index_rays.append(index_ray)
#                     index_tris.append(index_tri)
#                 pass

#                 try_times = 10
#                 while try_times > 0:
#                     try_times -= 1
#                     random_id = np.random.randint(0, ori_pcl.shape[0], 1)
#                     cur_point = ori_pcl[random_id, :]

#                     distances, indices = ori_pcl_kdtree.query(cur_point, k=self.n_pointcloud)
#                     local_pcl = ori_pcl[indices[0]]

#                     # draw_scene(local_pcl, ori_H_grasps, z_direction=True, scale=1.0/self.scale, max_grasps=800)

#                     # 找到对应该 local geo 的抓取
#                     local_geo_H_grasps = []
#                     for grasp_i in range(ori_H_grasps.shape[0]):
#                         location = locations[grasp_i]   # (n, 3)
#                         if location.shape[0] == 0:
#                             continue

#                         distance = location[:, None, :] - local_pcl[None, :, :]
#                         distance = np.linalg.norm(distance, axis=-1)
#                         min_distance = np.min(distance, axis=1)

#                         distance = np.max(min_distance)
#                         # print(distance)

#                         if distance < 1e-1 * (self.scale / 8.0):  # @note 

#                             # import open3d as o3d
#                             # ori_pcl_o3d = o3d.geometry.PointCloud()
#                             # ori_pcl_o3d.points = o3d.utility.Vector3dVector(ori_pcl / self.scale + np.array([0, 0, 0.001]))
#                             # ori_pcl_o3d.paint_uniform_color([1.0, 0.0, 0.0])
#                             # draw_scene(local_pcl, ori_H_grasps[grasp_i][None, ...], z_direction=True, scale=1.0/self.scale, max_grasps=800, other_geometry=[ori_pcl_o3d])

#                             local_geo_H_grasps.append(ori_H_grasps[grasp_i])
                        
#                     if len(local_geo_H_grasps) >= self.n_density:
#                         random_idx = np.random.randint(0, len(local_geo_H_grasps), self.n_density)
#                         H_grasps = np.array(local_geo_H_grasps)[random_idx, ...]
#                         pcl = local_pcl

#                         # draw_scene(local_pcl, H_grasps, z_direction=True, scale=1.0/self.scale, max_grasps=800)

#                         # print('Found local geo')
#                         break
#                     else:
#                         print('Not found local geo, num: {}, try again: {}'.format(len(local_geo_H_grasps), try_times))

#         # 转换为其他 rep
#         # 弃用 (deprecated)，改为在 model forward 中转换，这里不做预处理
#         grasps_quat = []
#         grasps_trans = []
#         grasps_rot6d = []

#         # grasps_quat = []
#         # grasps_trans = []
#         # grasps_rot6d = []
#         # for i in range(H_grasps.shape[0]):
#         #     grasp = H_grasps[i]
#         #     quat = Rotation.from_matrix(grasp[:3, :3]).as_quat()
#         #     trans = grasp[:3, -1]
#         #     grasps_rot6d.append(matrix_to_rotation_6d_np(grasp[:3, :3]))
#         #     grasps_quat.append(quat)
#         #     grasps_trans.append(trans)
#         # grasps_quat = np.array(grasps_quat)
#         # grasps_trans = np.array(grasps_trans)
#         # grasps_rot6d = np.array(grasps_rot6d)

#         # # debug vis
#         # draw_scene(pcl, H_grasps, z_direction=True, scale=1.0/self.scale, max_grasps=30)

#         # print("current return idx: {}, local flag: {}".format(index, try_times))

#         return [file_name], ori_pcl, pcl, H_grasps, grasps_rot6d, grasps_trans

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         return self._get_item(index)


if __name__ == "__main__":
    pass