import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(cur_dir), "3rd_isaacgym_evaluation"))
import hashlib
import json

from isaac_evaluation.grasp_quality_evaluation.any_obj_grasp_sucess import AnyGraspSuccessEvaluator

import torch
import numpy as np
import open3d as o3d

from roboutils.vis.viser_grasp import ViserForGrasp
from roboutils.proj_llm_robot.pose_transform import update_pose


def get_md5(file_path):
    # 创建 md5 对象
    md5_hash = hashlib.md5()
    
    # 打开文件以二进制模式读取
    with open(file_path, "rb") as file:
        # 读取并更新哈希对象，使用较小的块来处理大文件
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)
    
    # 获取最终的 MD5 哈希值
    return md5_hash.hexdigest()


class IsaacGymGraspEva(object):
    def __init__(self, data_npy_path, cache_dir, n_envs=20, device="cuda:0") -> None:
        """
            data_dict[i] = {
                "grasp_Ts": grasp_Ts,
                "mesh_path": mesh_path,
                "mesh_scale": mesh_scale,
                "mesh_T": np.eye(4),

                # for debug
                "grasp_success": grasp_success,
            }
        """
        self.cache_dir = cache_dir
        self.cache_obj_dir = os.path.join(self.cache_dir, "obj_cache")
        self.cache_output_dir = os.path.join(self.cache_dir, "output")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.cache_obj_dir, exist_ok=True)
        os.makedirs(self.cache_output_dir, exist_ok=True)

        self.exp_name = os.path.basename(data_npy_path).rsplit(".", maxsplit=1)[0]
        self.data_npy_path = data_npy_path
        self.data_dict = np.load(self.data_npy_path, allow_pickle=True).item()

        self.n_envs = n_envs
        self.device = device

        self.viser_grasp = ViserForGrasp()
        self.break_flag = False
        pass
    
    def eval(self, debug_vis=True):
        success_num_record = []
        num_record = []
        success_record = []
        for data_id, data in self.data_dict.items():
            obj_mesh_path = data['mesh_path']
            mesh_scale = data['mesh_scale']
            mesh_T = data['mesh_T']
            grasp_Ts = data['grasp_Ts']

            mesh_md5 = get_md5(obj_mesh_path)
            tmp_obj_mesh_path = os.path.join(self.cache_obj_dir, mesh_md5+".obj")

            # if not os.path.exists(tmp_obj_mesh_path):
            mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
            mesh = mesh.scale(mesh_scale, center=[0, 0, 0])
            mesh.transform(mesh_T)

            mesh_center = mesh.get_center()
            mesh.translate(-mesh_center)
            o3d.io.write_triangle_mesh(tmp_obj_mesh_path, mesh)

            for i, grasp_T in enumerate(grasp_Ts):
                grasp_Ts[i][:3, 3] -= mesh_center

            if debug_vis and (not self.break_flag):
                pc = np.array(mesh.vertices)
                print(f"Grasp num: {len(grasp_Ts)}")
                for i in range(len(grasp_Ts)):
                    print("current grasp id: ", i)
                    grasp_T = grasp_Ts[i]

                    # @note
                    grasp_T = update_pose(grasp_T, translate=[0, 0, 0.08])

                    self.viser_grasp.vis_grasp_scene(max_grasp_num=1, pc=pc, grasp_Ts=[grasp_T], mesh=mesh)
                    break_flag = self.viser_grasp.wait_for_reset()
                    if break_flag:
                        self.break_flag = True
                        break

                # viser_grasp.vis_grasp_scene(max_grasp_num=40, pc=pc, grasp_Ts=grasp_Ts, mesh=mesh)
                # viser_grasp.wait_for_reset()

            scales = [1.0] * len(grasp_Ts)
            grasp_evaluator = AnyGraspSuccessEvaluator(obj_mesh_path=tmp_obj_mesh_path, rotations=None, scales=scales, 
                                                    n_envs=self.n_envs, viewer=True, device=self.device, enable_rel_trafo=False)
            success_cases, success_flags = grasp_evaluator.eval_set_of_grasps(torch.tensor(grasp_Ts, device=self.device))

            # print(f"Success rate: {success_cases}/{len(grasp_Ts)}")
            num_record.append(len(grasp_Ts))
            success_num_record.append(success_cases)
            success_record.append(success_flags)

            data['eva_result_grasp_num'] = len(grasp_Ts)
            data['eva_result_success_num'] = success_cases
            data['eva_result_success'] = success_flags

            grasp_evaluator.grasping_env.kill()

        np.save(os.path.join(self.cache_output_dir, self.exp_name + ".npy"), data)
        return num_record, success_num_record, success_record


if __name__ == '__main__':
    cache_dir = "/home/red0orange/Projects/handgrasp_ws/0_Data/IsaacGymCache/obj_cache"
    os.makedirs(cache_dir, exist_ok=True)
    test_data_path = "/home/red0orange/Projects/handgrasp_ws/tmp.npy"

    evaluator = IsaacGymGraspEva(test_data_path, cache_dir)
    evaluator.eval()