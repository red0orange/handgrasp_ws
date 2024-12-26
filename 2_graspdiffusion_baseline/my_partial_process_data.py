import os 
import glob
import copy

import numpy as np
import trimesh.transformations as tra
from partial_utils.scene_renderer import SceneRenderer


def load_scene_contacts(dataset_folder, test_split_only=False, num_test=None, scene_contacts_path='scene_contacts_new'):
    """
    Load contact grasp annotations from acronym scenes 

    Arguments:
        dataset_folder {str} -- folder with acronym data and scene contacts

    Keyword Arguments:
        test_split_only {bool} -- whether to only return test split scenes (default: {False})
        num_test {int} -- how many test scenes to use (default: {None})
        scene_contacts_path {str} -- name of folder with scene contact grasp annotations (default: {'scene_contacts_new'})

    Returns:
        list(dicts) -- list of scene annotations dicts with object paths and transforms and grasp contacts and transforms.
    """
    
    scene_contact_paths = sorted(glob.glob(os.path.join(dataset_folder, scene_contacts_path, '*')))
    if test_split_only:
        scene_contact_paths = scene_contact_paths[-num_test:]
    contact_infos = []
    for contact_path in scene_contact_paths:
        print(contact_path)
        try:
        # if True:
            npz = np.load(contact_path, allow_pickle=False)
            contact_info = {'scene_contact_points':npz['scene_contact_points'],
                            'obj_paths':npz['obj_paths'],
                            'obj_transforms':npz['obj_transforms'],
                            'obj_scales':npz['obj_scales'],
                            'grasp_transforms':npz['grasp_transforms']}
            contact_infos.append(contact_info)
        except:
            print('corrupt, ignoring..')
    return contact_infos


def load_contact_grasps(contact_list, num_pos_contacts=8000):
    """
    Loads fixed amount of contact grasp data per scene into tf CPU/GPU memory

    Arguments:
        contact_infos {list(dicts)} -- Per scene mesh: grasp contact information  
        data_config {dict} -- data config

    Returns:
        [tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_offsets, 
        tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs, 
        all_obj_paths, all_obj_transforms] -- tf.constants with per scene grasp data, object paths/transforms in scene
    """
    pos_contact_points = []
    pos_contact_dirs = []
    pos_finger_diffs = []
    pos_approach_dirs = []

    for i,c in enumerate(contact_list):
        contact_directions_01 = c['scene_contact_points'][:,0,:] - c['scene_contact_points'][:,1,:]
        all_contact_points = c['scene_contact_points'].reshape(-1,3)
        all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps)
        all_contact_directions = np.empty((contact_directions_01.shape[0]*2, contact_directions_01.shape[1],))
        all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_directions[1::2] = contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_suc = np.ones_like(all_contact_points[:,0])
        all_grasp_transform = c['grasp_transforms'].reshape(-1,4,4)
        all_approach_directions = all_grasp_transform[:,:3,2]

        pos_idcs = np.where(all_contact_suc>0)[0]
        if len(pos_idcs) == 0:
            continue
        print('total positive contact points ', len(pos_idcs))
        
        all_pos_contact_points = all_contact_points[pos_idcs]
        all_pos_finger_diffs = all_finger_diffs[pos_idcs//2]
        all_pos_contact_dirs = all_contact_directions[pos_idcs]
        all_pos_approach_dirs = all_approach_directions[pos_idcs//2]
        
        # Use all positive contacts then mesh_utils with replacement
        if num_pos_contacts > len(all_pos_contact_points)/2:
            pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
            pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2 - len(all_pos_contact_points) , replace=True) 
            pos_sampled_contact_idcs= np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2, replace=False)

        pos_contact_points.append(all_pos_contact_points[pos_sampled_contact_idcs,:])
        pos_contact_dirs.append(all_pos_contact_dirs[pos_sampled_contact_idcs,:])
        pos_finger_diffs.append(all_pos_finger_diffs[pos_sampled_contact_idcs])
        pos_approach_dirs.append(all_pos_approach_dirs[pos_sampled_contact_idcs])
    
    pos_contact_points = np.array(pos_contact_points)
    pos_contact_dirs = np.array(pos_contact_dirs)
    pos_finger_diffs = np.array(pos_finger_diffs)
    pos_approach_dirs = np.array(pos_approach_dirs)

    scene_idxs = np.arange(len(pos_contact_points))
    pos_contact_points = pos_contact_points.astype(np.float32)
    pos_contact_dirs = np.array(pos_contact_dirs, dtype=np.float32)
    epsilon = 1e-12
    l2_norm = np.sqrt(np.sum(np.square(pos_contact_dirs), axis=2, keepdims=True)) + epsilon
    normalized_pos_contact_dirs = pos_contact_dirs / l2_norm
    pos_finger_diffs = pos_finger_diffs.astype(np.float32)
    pos_approach_dirs = pos_approach_dirs.astype(np.float32)
    l2_norm = np.sqrt(np.sum(np.square(pos_approach_dirs), axis=1, keepdims=True)) + epsilon
    normalized_pos_approach_dirs = pos_approach_dirs / l2_norm


    pass
    # device = "/cpu:0" if 'to_gpu' in data_config['labels'] and not data_config['labels']['to_gpu'] else "/gpu:0"
    # print("grasp label device: ", device)

    # with tf.device(device):
    #     tf_scene_idcs = tf.constant(np.arange(0,len(pos_contact_points)), tf.int32)
    #     tf_pos_contact_points = tf.constant(np.array(pos_contact_points), tf.float32)
    #     tf_pos_contact_dirs =  tf.math.l2_normalize(tf.constant(np.array(pos_contact_dirs), tf.float32),axis=2)
    #     tf_pos_finger_diffs = tf.constant(np.array(pos_finger_diffs), tf.float32)
    #     tf_pos_contact_approaches =  tf.math.l2_normalize(tf.constant(np.array(pos_approach_dirs), tf.float32),axis=2)

    # return tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs



class PointCloudReader:
    """
    Class to load scenes, render point clouds and augment them during training

    Arguments:
        root_folder {str} -- acronym root folder
        batch_size {int} -- number of rendered point clouds per-batch

    Keyword Arguments:
        raw_num_points {int} -- Number of random/farthest point samples per scene (default: {20000})
        estimate_normals {bool} -- compute normals from rendered point cloud (default: {False})
        caching {bool} -- cache scenes in memory (default: {True})
        use_uniform_quaternions {bool} -- use uniform quaternions for camera sampling (default: {False})
        scene_obj_scales {list} -- object scales in scene (default: {None})
        scene_obj_paths {list} -- object paths in scene (default: {None})
        scene_obj_transforms {np.ndarray} -- object transforms in scene (default: {None})
        num_train_samples {int} -- training scenes (default: {None})
        num_test_samples {int} -- test scenes (default: {None})
        use_farthest_point {bool} -- use farthest point sampling to reduce point cloud dimension (default: {False})
        intrinsics {str} -- intrinsics to for rendering depth maps (default: {None})
        distance_range {tuple} -- distance range from camera to center of table (default: {(0.9,1.3)})
        elevation {tuple} -- elevation range (90 deg is top-down) (default: {(30,150)})
        pc_augm_config {dict} -- point cloud augmentation config (default: {None})
        depth_augm_config {dict} -- depth map augmentation config (default: {None})
    """
    def __init__(
        self,
        root_folder,
        batch_size=1,
        raw_num_points = 20000,
        estimate_normals = False,
        caching=True,
        use_uniform_quaternions=False,
        scene_obj_scales=None,
        scene_obj_paths=None,
        scene_obj_transforms=None,
        num_train_samples=None,
        num_test_samples=None,
        use_farthest_point = False,
        intrinsics = None,
        distance_range = (0.9,1.3),
        elevation = (30,150),
        pc_augm_config = None,
        depth_augm_config = None
    ):
        self._root_folder = root_folder
        self._batch_size = batch_size
        self._raw_num_points = raw_num_points
        self._caching = caching
        self._num_train_samples = num_train_samples
        self._num_test_samples = num_test_samples
        self._estimate_normals = estimate_normals
        self._use_farthest_point = use_farthest_point
        self._scene_obj_scales = scene_obj_scales
        self._scene_obj_paths = scene_obj_paths
        self._scene_obj_transforms = scene_obj_transforms
        self._distance_range = distance_range
        self._pc_augm_config = pc_augm_config
        self._depth_augm_config = depth_augm_config

        self._current_pc = None
        self._cache = {}

        self._renderer = SceneRenderer(caching=True, intrinsics=intrinsics)

        if use_uniform_quaternions:
            quat_path = os.path.join(self._root_folder, 'uniform_quaternions/data2_4608.qua')
            quaternions = [l[:-1].split('\t') for l in open(quat_path, 'r').readlines()]

            quaternions = [[float(t[0]),
                            float(t[1]),
                            float(t[2]),
                            float(t[3])] for t in quaternions]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._cam_orientations = []
            self._elevation = np.array(elevation)/180. 
            for az in np.linspace(0, np.pi * 2, 30):
                for el in np.linspace(self._elevation[0], self._elevation[1], 30):
                    self._cam_orientations.append(tra.euler_matrix(0, -el, az))
            self._coordinate_transform = tra.euler_matrix(np.pi/2, 0, 0).dot(tra.euler_matrix(0, np.pi/2, 0))

    def get_cam_pose(self, cam_orientation):
        """
        Samples camera pose on shell around table center 

        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix

        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """
        
        distance = self._distance_range[0] + np.random.rand()*(self._distance_range[1]-self._distance_range[0])

        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance
        extrinsics = cam_orientation.dot(extrinsics)

        cam_pose = extrinsics.dot(self._coordinate_transform)
        # table height
        cam_pose[2,3] += self._renderer._table_dims[2]
        cam_pose[:3,:2]= -cam_pose[:3,:2]
        return cam_pose

    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud

        Returns:
            np.ndarray -- augmented point cloud
        """
        
        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self.apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'], 
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            pc = provider.jitter_point_cloud(pc[np.newaxis, :, :], 
                                            sigma=self._pc_augm_config['sigma'], 
                                            clip=self._pc_augm_config['clip'])[0]
        
        
        return pc[:,:3]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config

        Arguments:
            depth {np.ndarray} -- depth map

        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth,(kernel,kernel),0)
            depth[depth_copy==0] = depth_copy[depth_copy==0]
                
        return depth

    def apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- noof cluster to remove
            occlusion_dropout_rate {float} -- prob of removal

        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc

        labels = farthest_points(pc, occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]
    
    def get_scene_batch(self, scene_idx=None, return_segmap=False, save=False):
        """
        Render a batch of scene point clouds

        Keyword Arguments:
            scene_idx {int} -- index of the scene (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})

        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        dims = 6 if self._estimate_normals else 3
        batch_data = np.empty((self._batch_size, self._raw_num_points, dims), dtype=np.float32)
        cam_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)

        if scene_idx is None:
            scene_idx = np.random.randint(0,self._num_train_samples)

        obj_paths = [os.path.join(self._root_folder, p) for p in self._scene_obj_paths[scene_idx]]
        mesh_scales = self._scene_obj_scales[scene_idx]
        obj_trafos = self._scene_obj_transforms[scene_idx]

        self.change_scene(obj_paths, mesh_scales, obj_trafos, visualize=False)

        batch_segmap, batch_obj_pcs = [], []
        for i in range(self._batch_size):            
            # 0.005s
            pc_cam, pc_normals, camera_pose, depth = self.render_random_scene(estimate_normals = self._estimate_normals)

            if return_segmap:
                segmap, _, obj_pcs = self._renderer.render_labels(depth, obj_paths, mesh_scales, render_pc=True)
                batch_obj_pcs.append(obj_pcs)
                batch_segmap.append(segmap)

            batch_data[i,:,0:3] = pc_cam[:,:3]
            if self._estimate_normals:
                batch_data[i,:,3:6] = pc_normals[:,:3]
            cam_poses[i,:,:] = camera_pose
            
        if save:
            K = np.array([[616.36529541,0,310.25881958 ],[0,616.20294189,236.59980774],[0,0,1]])
            data = {'depth':depth, 'K':K, 'camera_pose':camera_pose, 'scene_idx':scene_idx}
            if return_segmap:
                data.update(segmap=segmap)
            np.savez('results/{}_acronym.npz'.format(scene_idx), data)

        if return_segmap:
            return batch_data, cam_poses, scene_idx, batch_segmap, batch_obj_pcs
        else:
            return batch_data, cam_poses, scene_idx

    def render_random_scene(self, estimate_normals=False, camera_pose=None):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations

        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {[type]} -- camera pose to render the scene from. (default: {None})

        Returns:
            [pc, pc_normals, camera_pose, depth] -- [point cloud, point cloud normals, camera pose, depth]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._cam_orientations))
            camera_orientation = self._cam_orientations[viewing_index]
            camera_pose = self.get_cam_pose(camera_orientation)

        in_camera_pose = copy.deepcopy(camera_pose)

        # 0.005 s
        _, depth, _, camera_pose = self._renderer.render(in_camera_pose, render_pc=False)
        depth = self._augment_depth(depth)
        
        pc = self._renderer._to_pointcloud(depth)
        pc = regularize_pc_point_count(pc, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        pc = self._augment_pc(pc)
        
        pc_normals = estimate_normals_cam_from_pc(pc[:,:3], raw_num_points=self._raw_num_points) if estimate_normals else []

        return pc, pc_normals, camera_pose, depth

    def change_object(self, cad_path, cad_scale):
        """
        Change object in pyrender scene

        Arguments:
            cad_path {str} -- path to CAD model
            cad_scale {float} -- scale of CAD model
        """

        self._renderer.change_object(cad_path, cad_scale)

    def change_scene(self, obj_paths, obj_scales, obj_transforms, visualize=False):
        """
        Change pyrender scene

        Arguments:
            obj_paths {list[str]} -- path to CAD models in scene
            obj_scales {list[float]} -- scales of CAD models
            obj_transforms {list[np.ndarray]} -- poses of CAD models

        Keyword Arguments:
            visualize {bool} -- whether to update the visualizer as well (default: {False})
        """
        self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        if visualize:
            self._visualizer.change_scene(obj_paths, obj_scales, obj_transforms)



    def __del__(self):
        print('********** terminating renderer **************')


def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters

def reject_median_outliers(data, m=0.4, z_only=False):
    """
    Reject outliers with median absolute distance m

    Arguments:
        data {[np.ndarray]} -- Numpy array such as point cloud

    Keyword Arguments:
        m {[float]} -- Maximum absolute distance from median in m (default: {0.4})
        z_only {[bool]} -- filter only via z_component (default: {False})

    Returns:
        [np.ndarray] -- Filtered data without outliers
    """
    if z_only:
        d = np.abs(data[:,2:3] - np.median(data[:,2:3]))
    else:
        d = np.abs(data - np.median(data, axis=0, keepdims=True))

    return data[np.sum(d, axis=1) < m]

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates 
      
      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """
    
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def transform_pcd(pcd, T):
    return (T @ np.hstack((pcd, np.ones((pcd.shape[0], 1)))).T).T[:, :3]


def resample_point_cloud(point_cloud, num_points):
    n = point_cloud.shape[0]
    
    if n < num_points:
        # If there are fewer points, randomly duplicate points to expand
        indices = np.random.choice(n, size=num_points, replace=True)
    else:
        # If there are more points, randomly sample points to reduce
        indices = np.random.choice(n, size=num_points, replace=False)
    
    return point_cloud[indices]


if __name__ == '__main__':
    dataset_folder = '/home/red0orange/Data/contactgraspnet'
    scene_contacts_path = "scene_contacts"

    contact_infos = load_scene_contacts(dataset_folder, test_split_only=True, num_test=3000, scene_contacts_path=scene_contacts_path)

    # grasp_annots = load_contact_grasps(contact_infos)

    root_folder = dataset_folder
    batch_size = 4
    estimate_normals = False
    raw_num_points = 20000
    use_uniform_quaternions = False
    scene_obj_scales = [c['obj_scales'] for c in contact_infos]
    scene_obj_paths = [c['obj_paths'] for c in contact_infos]
    scene_obj_transforms = [c['obj_transforms'] for c in contact_infos]
    num_train_samples = int(len(contact_infos)*0.99)
    num_test_samples = len(contact_infos) - num_train_samples
    use_farthest_point = False
    intrinsics = "kinect_azure"
    elevation = [30, 150]
    distance_range = [0.9, 1.3]

    pcreader = PointCloudReader(
        root_folder=root_folder,
        batch_size=batch_size,
        raw_num_points=raw_num_points,
        estimate_normals=estimate_normals,
        use_uniform_quaternions=use_uniform_quaternions,
        scene_obj_transforms=scene_obj_transforms,
        scene_obj_paths=scene_obj_paths,
        scene_obj_scales=scene_obj_scales,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        use_farthest_point=use_farthest_point,
        intrinsics=intrinsics,
        elevation=elevation,
        distance_range=distance_range,
        pc_augm_config={},
        depth_augm_config={}
    )
    glcam_in_cvcam = np.array([[1,0,0,0],
                                [0,-1,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]]).astype(float)

    # # @note debug vis
    from roboutils.vis.viser_grasp import ViserForGrasp
    import open3d as o3d
    # viser_for_grasp = ViserForGrasp()
    # for batch_idx in range(pcreader._num_train_samples):
    #     batch_data, cam_poses, scene_idx, batch_segmap, batch_obj_pcs = pcreader.get_scene_batch(scene_idx=batch_idx, return_segmap=True)
    #     contact_info = contact_infos[scene_idx]

    #     grasp_Ts = contact_info['grasp_transforms']
    #     grasp_contact_points = contact_info['scene_contact_points']

    #     world_pcds = []
    #     for i in range(batch_size):
    #         pcd = batch_data[i,:,0:3]
    #         camera_T = cam_poses[i]
    #         pcd = transform_pcd(pcd, camera_T @ glcam_in_cvcam)
    #         world_pcds.append(pcd)
    #         colors = np.zeros((pcd.shape[0], 3))
    #         colors[:,0] = np.random.rand(pcd.shape[0])
    #         viser_for_grasp.add_pcd(pcd, colors=colors)
        
    #     for i in range(100):
    #         viser_for_grasp.add_grasp(grasp_Ts[i])
    #         contact_points = grasp_contact_points[i]
    #         colors =   np.zeros((contact_points.shape[0], 3))
    #         colors[:,1] = 1.0
    #         viser_for_grasp.add_pcd(contact_points, colors=colors)
            

    #     viser_for_grasp.wait_for_reset()

    #     pass

    # 导出数据
    from tqdm import tqdm
    from scipy.spatial import KDTree
    import pickle as pkl
    save_root = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_contactgn/data"

    viser_for_grasp = ViserForGrasp()
    save_idx = 0
    for batch_idx in tqdm(range(pcreader._num_train_samples), total=pcreader._num_train_samples):
        try:
            batch_data, cam_poses, scene_idx, batch_segmap, batch_obj_pcs = pcreader.get_scene_batch(scene_idx=batch_idx, return_segmap=True)
        except:
            print(f"Error in scene {batch_idx}")
            continue
        contact_info = contact_infos[scene_idx]

        grasp_Ts = contact_info['grasp_transforms']
        grasp_contact_points = contact_info['scene_contact_points']

        # 遍历每个 batch
        for i in range(batch_size):
            camera_scene_pcd = batch_data[i,:,0:3]
            camera_obj_pcs = batch_obj_pcs[i]
            camera_T = cam_poses[i]
            camera_T = camera_T @ glcam_in_cvcam

            world_scene_pcd = transform_pcd(camera_scene_pcd, camera_T)

            # 遍历每个物体
            for obj_idx, camera_obj_pc in camera_obj_pcs.items():
                camera_obj_pc = camera_obj_pc[..., 0:3]
                world_obj_pc = transform_pcd(camera_obj_pc, camera_T)
                # 计算距离
                kdtree = KDTree(world_obj_pc)
                query_dist, query_idx = kdtree.query(grasp_contact_points)
                query_dist = query_dist[:, 0]
                query_idx = query_idx[:, 0]

                valid_grasp_idx = np.where(query_dist < 0.01)[0]
                valid_grasp_Ts = grasp_Ts[valid_grasp_idx]
                valid_camera_grasp_Ts = np.array([np.linalg.inv(camera_T) @ valid_grasp_T for valid_grasp_T in valid_grasp_Ts])

                # # @note debug
                # viser_for_grasp.vis_grasp_scene(valid_grasp_Ts, world_obj_pc)
                # viser_for_grasp.wait_for_reset()
                # viser_for_grasp.vis_grasp_scene(valid_camera_grasp_Ts, camera_obj_pc)
                # viser_for_grasp.wait_for_reset()

                if len(valid_grasp_idx) < 10:
                    continue

                # 保存数据
                save_data_dict = {
                    # meta
                    "obj_idx": obj_idx,
                    "camera_T": camera_T,

                    # world
                    "world_scene_pcd": world_scene_pcd,
                    "world_obj_pc": world_obj_pc,
                    "world_grasp_Ts": valid_grasp_Ts,
                    "world_grasp_contact_points": grasp_contact_points[valid_grasp_idx],

                    # camera
                    "camera_scene_pcd": camera_scene_pcd,
                    "camera_obj_pc": camera_obj_pc,
                    "camera_grasp_Ts": valid_camera_grasp_Ts,

                    # for CONG format training
                    "sampled_pc_2048": resample_point_cloud(camera_obj_pc, 2048),
                    "grasps/transformations": valid_camera_grasp_Ts,
                    "grasps/successes": np.ones((valid_camera_grasp_Ts.shape[0],)),
                }
                save_path = os.path.join(save_root, f"{save_idx:06d}.pickle")
                save_idx += 1
                with open(save_path, "wb") as f:
                    pkl.dump(save_data_dict, f)