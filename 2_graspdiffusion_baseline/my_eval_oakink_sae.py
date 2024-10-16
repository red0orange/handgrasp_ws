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
from models.main_nets import GraspRefineScoreNet

from scipy.spatial import KDTree


def transform_pcd(pcd, T):
    return (T @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)).T[:, :3]


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
        my_palm_T[:3, 3] *= self.ori_dataset.scale

        return [data_path], [mesh_path], obj_verts, mesh_T, my_palm_T, hand_verts
    

def eval_for_isaacgym(work_dir):
    save_name = "eval_oakink_sae"
    config_file_path = os.path.join(work_dir, "config.py")
    checkpoint_path = os.path.join(work_dir, "current_model.t7")
    
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
    dataloader = torch.utils.data.DataLoader(oakink_dataset, batch_size=4, shuffle=False, num_workers=16)

    grasp_eva = GraspRefineScoreNet()
    results = []
    for data in tqdm(dataloader, total=len(dataloader)):
        file_paths, mesh_paths, obj_verts, mesh_T, my_palm_Ts, hand_verts = data[0], data[1], data[2], data[3], data[4], data[5]
        file_paths = file_paths[0]
        mesh_paths = mesh_paths[0]
        xyz = obj_verts

        # 处理 palm_T
        refine_palm_Ts = []
        refine_palm_gs = []
        for i in range(len(my_palm_Ts)):
            cur_obj_verts = obj_verts[i].cpu().numpy()
            cur_hand_verts = hand_verts[i].cpu().numpy()
            my_palm_T = my_palm_Ts[i].cpu().numpy()

            obj_kdtree = KDTree(cur_obj_verts)
            palm_grasp_center = my_palm_T[:3, 3]
            distances, indices = obj_kdtree.query(palm_grasp_center, k=96)

            refine_palm_T = my_palm_T.copy()
            refine_palm_T[:3, 3] = cur_obj_verts[indices[0]]  # 距离最近的点作为抓取中心
            refine_palm_Ts.append(refine_palm_T)

            # # debug
            # closest_pcs = cur_obj_verts[indices]
            # viser_for_grasp.add_pcd(closest_pcs, colors=np.array([(255, 0, 0)]*closest_pcs.shape[0]))
            # viser_for_grasp.add_pcd(cur_hand_verts, colors=np.array([(0, 255, 0)]*cur_hand_verts.shape[0]))
            # # viser_for_grasp.add_pcd(obj_verts, colors=np.array([(0, 0, 255)]*obj_verts.shape[0]))
            # viser_for_grasp.add_grasp(my_palm_T, z_direction=True)
            # viser_for_grasp.add_grasp(refine_palm_T, z_direction=True)
            # print("wait for reset")
            # viser_for_grasp.wait_for_reset()

            g_init = np.concatenate((matrix_to_rotation_6d_np(refine_palm_T[:3, :3]), refine_palm_T[:3, 3]), axis=0)
            g_init = torch.from_numpy(g_init).float()
            refine_palm_gs.append(g_init)
        refine_palm_Ts = np.array(refine_palm_Ts)

        # 开始预测
        xyz = xyz.float().cuda()

        # 采样 100 个，每 100 个作为一次预测，从每 10 个预测中选出最佳的 1 个
        each_time_sample_num = 100
        eval_time_num = 10
        sample_num = each_time_sample_num * eval_time_num

        # 重复 g_init
        input_refine_palm_gs = torch.stack(refine_palm_gs, dim=0).unsqueeze(1).repeat(1, sample_num, 1)
        # input_refine_palm_gs = input_refine_palm_gs.reshape(-1, input_refine_palm_gs.shape[-1])

        # @note 无限制采样
        pred_gs = model.batch_detect_and_sample(xyz, sample_num, guide_w=GUIDE_W, data_scale=dataset.scale)

        # @note 根据约束选择
        selected_idx = []
        for batch_i in range(input_refine_palm_gs.shape[0]):
            cur_refine_palm_gs = input_refine_palm_gs[batch_i]
            cur_pred_gs = pred_gs[batch_i]

            cur_pred_gs = torch.from_numpy(cur_pred_gs).float()
            cur_selected_idx = grasp_eva.evaluate(cur_pred_gs, cur_refine_palm_gs)
            cur_selected_idx = cur_selected_idx.numpy()
            cur_selected_idx = np.where(cur_selected_idx)[0]
            selected_idx.append(cur_selected_idx)
        
        xyz = xyz.cpu().numpy()

        batch_size = pred_gs.shape[0]
        num_grasp = pred_gs.shape[1]
        pred_gs = pred_gs.reshape(-1, pred_gs.shape[-1])  # 先展平
        num_pose = pred_gs.shape[0]
        if not rot6d_rep:
            raise NotImplementedError
        else:
            rotation = rotation_6d_to_matrix_np(pred_gs[:, :6])
            rotation = np.concatenate((rotation, np.zeros((num_pose, 1, 3), dtype=np.float32)), axis=1)
            translation = np.expand_dims(np.concatenate((pred_gs[:, 6:], np.ones((num_pose, 1), dtype=np.float32)), axis=1), axis=2)
        grasp_Ts = np.concatenate((rotation, translation), axis=2)
        grasp_Ts = grasp_Ts.reshape(batch_size, num_grasp, 4, 4)

        for batch_i in range(batch_size):
            batch_i_filename = file_paths[batch_i]
            batch_i_mesh_path = mesh_paths[batch_i]
            batch_i_xyz = xyz[batch_i]
            batch_i_grasp_Ts = grasp_Ts[batch_i]
            batch_i_mesh_T = mesh_T[batch_i].cpu().numpy()
            batch_i_palm_T = my_palm_Ts[batch_i].cpu().numpy()
            batch_i_selected_idx = selected_idx[batch_i]

            vis_xyz, vis_grasp_Ts = batch_i_xyz.copy(), batch_i_grasp_Ts.copy()
            vis_palm_T = batch_i_palm_T.copy()
            vis_xyz /= dataset.scale
            vis_grasp_Ts[:, :3, 3] /= dataset.scale
            vis_palm_T[:3, 3] /= dataset.scale
            batch_i_mesh_T[:3, 3] /= dataset.scale

            # selected_grasp_Ts = vis_grasp_Ts[batch_i_selected_idx]

            # 
            selected_grasp_Ts = []
            for grasp_i in range(0, sample_num, each_time_sample_num):
                start_idx = grasp_i
                end_idx = grasp_i + each_time_sample_num

                valid_selected_idx = (batch_i_selected_idx >= start_idx) & (batch_i_selected_idx < end_idx)
                valid_selected_idx = batch_i_selected_idx[np.where(valid_selected_idx)[0]]

                if len(valid_selected_idx) == 0:
                    # @note 添加一个无效的抓取，一定是失败的
                    fake_grasp_T = np.eye(4)
                    fake_grasp_T[:3, 3] = np.array([0, 0, 0.1])
                    selected_grasp_Ts.append(fake_grasp_T)
                else:
                    cur_grasp_Ts = vis_grasp_Ts[valid_selected_idx]

                    # @note 随机选择
                    random_idx = np.random.choice(cur_grasp_Ts.shape[0], 1)
                    cur_grasp_Ts = cur_grasp_Ts[random_idx]
                    cur_grasp_T = cur_grasp_Ts[0]

                    selected_grasp_Ts.append(cur_grasp_T)
            selected_grasp_Ts = np.array(selected_grasp_Ts)

            # # debug vis
            # # viser_for_grasp.vis_grasp_scene(vis_grasp_Ts, pc=vis_xyz, max_grasp_num=1000)
            # # viser_for_grasp.wait_for_reset()
            # viser_for_grasp.vis_grasp_scene(selected_grasp_Ts, pc=vis_xyz, max_grasp_num=1000)
            # viser_for_grasp.add_grasp(vis_palm_T, z_direction=True)
            # viser_for_grasp.wait_for_reset()

            data_dict = {
                'filename': batch_i_filename,
                'mesh_path': batch_i_mesh_path,
                'xyz': vis_xyz,
                'grasp_Ts': selected_grasp_Ts,
                'mesh_T': batch_i_mesh_T,
            }
            results.append(data_dict)
    with open(os.path.join(work_dir, '{}.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(results, f)
    # # exit()
    
    results = pickle.load(open(os.path.join(work_dir, '{}.pkl'.format(save_name)), 'rb'))
    
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

    np.save(os.path.join(work_dir, '{}_isaacgym.npy'.format(save_name)), isaacgym_eval_data_dict)
    pass


if __name__ == "__main__":
    work_dir = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion"

    # eval_for_vis(work_dir)
    eval_for_isaacgym(work_dir)
    pass