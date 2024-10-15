import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from gorilla.config import Config

from utils import *
from utils.rotate_rep import rotation_6d_to_matrix_np, matrix_to_rotation_6d_np
from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose

from scipy.spatial import KDTree


def transform_pcd(pcd, T):
    return (T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]


def eval_for_vis(work_dir):
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")
    # checkpoint_path = os.path.join(work_dir, "model_epoch_75.pth")
    
    GUIDE_W = 0.5
    DEVICE=torch.device('cuda')

    cfg = Config.fromfile(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = build_model(cfg).to(DEVICE)
    dataset = build_dataset(cfg)['test_set']
    # dataset = build_dataset(cfg)['train_set']
    rot6d_rep = cfg.hyper_params.rot6d_rep
    
    print("Loading checkpoint....")
    _, exten = os.path.splitext(checkpoint_path)
    if exten == '.t7':
        model.load_state_dict(torch.load(checkpoint_path))
    elif exten == '.pth':
        model.load_state_dict(torch.load(checkpoint_path))
        # check = torch.load(checkpoint_path)
        # model.load_state_dict(check['model_state_dict'])

    print("Evaluating")
    viser_for_grasp = ViserForGrasp()
    model.eval()

    # load oakink data
    oakink_data_root = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_Oakink"
    mesh_root = os.path.join(oakink_data_root, "meshes")
    data_root = os.path.join(oakink_data_root, "data")

    data_paths = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.pkl')]
    mesh_paths = [os.path.join(mesh_root, os.path.basename(f).split('.')[0] + '.obj') for f in data_paths]
    for data_path, mesh_path in tqdm(zip(data_paths, mesh_paths)):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        obj_mesh = o3d.io.read_triangle_mesh(mesh_path)
        obj_mesh.compute_vertex_normals()

        hand_verts = data['hand_verts']
        hand_faces = data['hand_faces']
        my_palm_T = data['palm_T']
        obj_verts = obj_mesh.sample_points_uniformly(number_of_points=2048)
        obj_verts = np.array(obj_verts.points)

        # 中心化
        obj_center = np.mean(obj_verts, axis=0)
        obj_verts -= obj_center
        hand_verts -= obj_center
        my_palm_T[:3, 3] -= obj_center

        # # @note debug vis
        # viser_for_grasp.add_pcd(obj_verts)
        # viser_for_grasp.add_pcd(hand_verts)
        # viser_for_grasp.add_grasp(my_palm_T)
        # viser_for_grasp.wait_for_reset()

        hand_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hand_verts), o3d.utility.Vector3iVector(hand_faces))
        hand_mesh.compute_vertex_normals()
        
        xyz = dataset.preprocess_infer_data(obj_verts)
        refine_palm_T = my_palm_T.copy()
        refine_palm_T[:3, 3] *= dataset.scale

        # @note begin infer
        xyz = torch.from_numpy(xyz).unsqueeze(0).float().cuda()

        # 找出与 palm_T 抓取中心最近的物体点的范围
        obj_kdtree = KDTree(obj_verts)
        palm_grasp_center = refine_palm_T[:3, 3]
        distances, indices = obj_kdtree.query(palm_grasp_center, k=96)

        # closest_pcs = obj_verts[indices]
        # viser_for_grasp.add_pcd(closest_pcs, colors=np.array([(255, 0, 0)]*closest_pcs.shape[0]))
        # viser_for_grasp.add_pcd(hand_verts, colors=np.array([(0, 255, 0)]*hand_verts.shape[0]))
        # # viser_for_grasp.add_pcd(obj_verts, colors=np.array([(0, 0, 255)]*obj_verts.shape[0]))
        # viser_for_grasp.wait_for_reset()

        refine_palm_T[:3, 3] = obj_verts[indices[0]]  # 距离最近的点作为抓取中心


        # # @note 无限制直接预测
        # pred = model.detect_and_sample(xyz, 10, guide_w=GUIDE_W, data_scale=dataset.scale)

        # @note 限制预测
        # 抓取位置选择与 palm_T 抓取中心最近的物体点的范围
        g_init = np.concatenate((matrix_to_rotation_6d_np(refine_palm_T[:3, :3]), refine_palm_T[:3, 3]), axis=0)
        g_init = torch.from_numpy(g_init).float().cuda()
        pred = model.refine_grasp_sample(xyz, 10, g_init)

        xyz = xyz.squeeze(0).cpu().numpy()

        num_pose = pred.shape[0]
        if not rot6d_rep:
            rotation = np.concatenate((R.from_quat(pred[:, :4]).as_matrix(), np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
            translation = np.expand_dims(np.concatenate((pred[:, 4:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
        else:
            rotation = rotation_6d_to_matrix_np(pred[:, :6])
            rotation = np.concatenate((rotation, np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
            translation = np.expand_dims(np.concatenate((pred[:, 6:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
        T = np.concatenate((rotation, translation), axis=2)

        vis_xyz, vis_T = xyz.copy(), T.copy()
        vis_xyz /= dataset.scale
        vis_T[:, :3, 3] /= dataset.scale


        rotate_T = update_pose(np.eye(4), rotate=np.pi/2, rotate_axis="y")
        vis_xyz = transform_pcd(vis_xyz, rotate_T)
        hand_mesh.transform(rotate_T)
        my_palm_T = rotate_T @ my_palm_T
        vis_T = [rotate_T @ t for t in vis_T]

        # debug vis
        print("Waiiting for viser visualization")
        viser_for_grasp.vis_grasp_scene(vis_T, pc=vis_xyz, max_grasp_num=50)
        viser_for_grasp.add_mesh(hand_mesh)
        viser_for_grasp.add_grasp(my_palm_T, z_direction=True)
        viser_for_grasp.wait_for_reset()


class OakinkGraspDataset(torch.utils.data.Dataset):
    def __init__(self, root, ori_dataset):
        self.data_root = os.path.join(root, "data")
        self.mesh_root = os.path.join(root, "meshes")

        self.ori_dataset = ori_dataset

        self.data_paths = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root) if f.endswith('.pkl')]
        self.mesh_paths = [os.path.join(self.mesh_root, os.path.basename(f).split('.')[0] + '.obj') for f in self.data_paths]
        pass
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        mesh_path = self.mesh_paths[idx]

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        obj_mesh = o3d.io.read_triangle_mesh(mesh_path)
        obj_mesh.compute_vertex_normals()

        hand_verts = data['hand_verts']
        hand_faces = data['hand_faces']
        my_palm_T = data['palm_T']
        obj_verts = obj_mesh.sample_points_uniformly(number_of_points=2048)
        obj_verts = np.array(obj_verts.points)

        # 中心化
        obj_center = np.mean(obj_verts, axis=0)
        obj_verts -= obj_center
        hand_verts -= obj_center
        my_palm_T[:3, 3] -= obj_center

        mesh_T = np.eye(4)
        mesh_T[:3, 3] = -obj_center

        # hand_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hand_verts), o3d.utility.Vector3iVector(hand_faces))
        # hand_mesh.compute_vertex_normals()

        obj_verts = self.ori_dataset.preprocess_infer_data(obj_verts)
        mesh_T[:3, 3] *= self.ori_dataset.scale

        return [data_path], [mesh_path], obj_verts, mesh_T, my_palm_T, hand_verts
    

def eval_for_isaacgym(work_dir):
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")
    # checkpoint_path = os.path.join(work_dir, "model_epoch_75.pth")
    
    GUIDE_W = 0.5
    DEVICE=torch.device('cuda')

    cfg = Config.fromfile(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = build_model(cfg).to(DEVICE)
    dataset = build_dataset(cfg)['test_set']
    # dataset = build_dataset(cfg)['train_set']
    rot6d_rep = cfg.hyper_params.rot6d_rep
    
    print("Loading checkpoint....")
    _, exten = os.path.splitext(checkpoint_path)
    if exten == '.t7':
        model.load_state_dict(torch.load(checkpoint_path))
    elif exten == '.pth':
        model.load_state_dict(torch.load(checkpoint_path))
        # check = torch.load(checkpoint_path)
        # model.load_state_dict(check['model_state_dict'])

    print("Evaluating")
    viser_for_grasp = ViserForGrasp()
    model.eval()

    # @note 加载 oakink 数据集
    oakink_data_root = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_Oakink"
    oakink_dataset = OakinkGraspDataset(oakink_data_root, dataset)
    dataloader = torch.utils.data.DataLoader(oakink_dataset, batch_size=64, shuffle=False, num_workers=16)

    # results = []
    # for data in tqdm(dataloader, total=len(dataloader)):
    #     file_paths, mesh_paths, obj_verts, mesh_T, my_palm_T, hand_verts = data[0], data[1], data[2], data[3], data[4], data[5]
    #     file_paths = file_paths[0]
    #     mesh_paths = mesh_paths[0]
    #     xyz = obj_verts

    #     xyz = xyz.float().cuda()
    #     pred = model.batch_detect_and_sample(xyz, 10, guide_w=GUIDE_W, data_scale=dataset.scale)
    #     xyz = xyz.cpu().numpy()

    #     batch_size = pred.shape[0]
    #     num_grasp = pred.shape[1]
    #     pred = pred.reshape(-1, pred.shape[-1])  # 先展平
    #     num_pose = pred.shape[0]
    #     if not rot6d_rep:
    #         raise NotImplementedError
    #     else:
    #         rotation = rotation_6d_to_matrix_np(pred[:, :6])
    #         rotation = np.concatenate((rotation, np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
    #         translation = np.expand_dims(np.concatenate((pred[:, 6:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
    #     grasp_Ts = np.concatenate((rotation, translation), axis=2)
    #     grasp_Ts = grasp_Ts.reshape(batch_size, num_grasp, 4, 4)

    #     for batch_i in range(batch_size):
    #         batch_i_filename = file_paths[batch_i]
    #         batch_i_mesh_path = mesh_paths[batch_i]
    #         batch_i_xyz = xyz[batch_i]
    #         batch_i_grasp_Ts = grasp_Ts[batch_i]
    #         batch_i_mesh_T = mesh_T[batch_i].cpu().numpy()

    #         vis_xyz, vis_grasp_Ts = batch_i_xyz.copy(), batch_i_grasp_Ts.copy()
    #         vis_xyz /= dataset.scale
    #         vis_grasp_Ts[:, :3, 3] /= dataset.scale
    #         batch_i_mesh_T[:3, 3] /= dataset.scale

    #         # # debug vis
    #         # viser_for_grasp.vis_grasp_scene(vis_grasp_Ts, pc=vis_xyz, max_grasp_num=50)
    #         # viser_for_grasp.wait_for_reset()

    #         data_dict = {
    #             'filename': batch_i_filename,
    #             'mesh_path': batch_i_mesh_path,
    #             'xyz': vis_xyz,
    #             'grasp_Ts': vis_grasp_Ts,
    #             'mesh_T': batch_i_mesh_T,
    #         }
    #         results.append(data_dict)
    # with open(os.path.join(work_dir, 'oakink_eval_results.pkl'), 'wb') as f:
    #     pickle.dump(results, f)
    # # exit()
    
    results = pickle.load(open(os.path.join(work_dir, 'oakink_eval_results.pkl'), 'rb'))
    
    # 将结果转换为 IsaacGym 格式
    isaacgym_eval_data_dict = {}
    for i, data_dict in tqdm(enumerate(results), total=len(results)):
        isaacgym_eval_data_dict[i] = {}

        obj_pc = data_dict['xyz']
        mesh_T = data_dict['mesh_T']
        mesh_path = data_dict['mesh_path']
        mesh_scale = 1.0

        isaacgym_eval_data_dict[i]['mesh_path'] = mesh_path
        isaacgym_eval_data_dict[i]['mesh_scale'] = mesh_scale
        isaacgym_eval_data_dict[i]['grasp_Ts'] = data_dict['grasp_Ts']
        isaacgym_eval_data_dict[i]['mesh_T'] = mesh_T

        # # for debug vis
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # mesh.scale(mesh_scale, center=np.zeros(3))
        # mesh.transform(mesh_T)
        # grasp_Ts = data_dict['grasp_Ts']
        # viser_for_grasp.vis_grasp_scene(grasp_Ts, mesh=mesh, pc=obj_pc, max_grasp_num=50)
        # viser_for_grasp.wait_for_reset()

    np.save(os.path.join(work_dir, 'oakink_isaacgym_eval_results.npy'), isaacgym_eval_data_dict)
    pass


if __name__ == "__main__":
    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion"

    eval_for_vis(work_dir)
    # eval_for_isaacgym(work_dir)
    pass