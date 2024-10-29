import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(cur_dir), "3rd_isaacgym_evaluation"))
import hashlib
import json

from isaac_evaluation.grasp_quality_evaluation.any_obj_grasp_sucess import AnyGraspSuccessEvaluator

import torch
from tqdm import tqdm
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
        self.work_dir = os.path.dirname(data_npy_path)
        self.data_dict = np.load(self.data_npy_path, allow_pickle=True).item()

        # self.n_envs = n_envs
        self.device = device

        self.viser_grasp = ViserForGrasp()
        self.break_flag = False
        pass

    def eval_one_obj(self, data, debug_vis):
        obj_mesh_path = data['mesh_path']
        mesh_scale = data['mesh_scale']
        mesh_T = data['mesh_T']
        grasp_Ts = data['grasp_Ts']

        mesh_md5 = get_md5(obj_mesh_path)
        tmp_obj_mesh_path = os.path.join(self.cache_obj_dir, mesh_md5+".obj")

        mesh = None
        # if (not os.path.exists(tmp_obj_mesh_path)) or (not os.path.exists(os.path.join(self.cache_obj_dir, mesh_md5+"_center.npy"))):
        if True:
            mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
            mesh = mesh.scale(mesh_scale, center=[0, 0, 0])
            mesh.transform(mesh_T)

            mesh_center = mesh.get_center()
            mesh.translate(-mesh_center)
            o3d.io.write_triangle_mesh(tmp_obj_mesh_path, mesh)
            np.save(os.path.join(self.cache_obj_dir, mesh_md5+"_center.npy"), mesh_center)
        else:
            mesh_center = np.load(os.path.join(self.cache_obj_dir, mesh_md5+"_center.npy"))

        for i, grasp_T in enumerate(grasp_Ts):
            grasp_Ts[i][:3, 3] -= mesh_center

        if debug_vis and (not self.break_flag):
            mesh = o3d.io.read_triangle_mesh(tmp_obj_mesh_path)
            pc = np.array(mesh.vertices)

            # print(f"Grasp num: {len(grasp_Ts)}")
            # for i in range(len(grasp_Ts)):
            #     print("current grasp id: ", i)
            #     grasp_T = grasp_Ts[i]

            #     # @note
            #     grasp_T = update_pose(grasp_T, translate=[0, 0, 0.08])

            #     self.viser_grasp.vis_grasp_scene(max_grasp_num=1, pc=pc, grasp_Ts=[grasp_T], mesh=mesh)
            #     break_flag = self.viser_grasp.wait_for_reset()
            #     if break_flag:
            #         self.break_flag = True
            #         break

            self.viser_grasp.vis_grasp_scene(max_grasp_num=40, pc=pc, grasp_Ts=grasp_Ts, mesh=mesh)
            self.viser_grasp.wait_for_reset()

        scales = [1.0] * len(grasp_Ts)
        # try:
        if True:
            n_envs = min(len(grasp_Ts), 500)
            grasp_evaluator = AnyGraspSuccessEvaluator(obj_mesh_path=tmp_obj_mesh_path, rotations=None, scales=scales, 
                                                    n_envs=n_envs, viewer=False, device=self.device, enable_rel_trafo=False)
            success_cases, success_flags = grasp_evaluator.eval_set_of_grasps(torch.tensor(grasp_Ts, device=self.device))
        # except Exception as e:
        #     print(f"Error: {e}")

        # print(f"Success rate: {success_cases}/{len(grasp_Ts)}")

        data['eva_result_grasp_num'] = len(grasp_Ts)
        data['eva_result_success_num'] = success_cases
        data['eva_result_success'] = success_flags

        grasp_evaluator.grasping_env.kill()

        return success_cases, len(grasp_Ts), success_flags
    
    def eval(self, debug_vis=True, save_name="isaacgym_eval_result"):
        success_num_record = []
        num_record = []
        success_record = []
        tmp_txt_path = "{}.txt".format(save_name)
        tmp_txt_path = os.path.join(self.work_dir, tmp_txt_path)
        for data_id, data in tqdm(self.data_dict.items(), total=len(self.data_dict)):
            self.eval_one_obj(data, debug_vis)
            success_cases, total_num, success_flags = data['eva_result_success_num'], data['eva_result_grasp_num'], data['eva_result_success']

            success_num_record.append(success_cases)
            num_record.append(total_num)
            success_record.append(success_flags)

            print("================================")
            print("summary: {} / {}".format(sum(success_num_record), sum(num_record)))
            print("success rate: {}".format(sum(success_num_record) / sum(num_record)))
            print("================================")
            with open(tmp_txt_path, "a") as f:
                f.write("{} {} {}\n".format(success_cases, total_num, success_flags))

        np.save(os.path.join(self.cache_output_dir, self.exp_name + ".npy"), self.data_dict)
        return num_record, success_num_record, success_record


class CONGIsaacGymGraspEva(IsaacGymGraspEva):
    def __init__(self, data_npy_path, output_dir, cache_dir, n_envs=20, device="cuda:0", start_idx=0) -> None:
        super().__init__(data_npy_path, cache_dir, n_envs, device)
        self.output_dir = output_dir
        self.start_idx = start_idx
        os.makedirs(self.output_dir, exist_ok=True)

    def eval(self, debug_vis=True, rewrite=False):
        success_num_record = []
        num_record = []
        success_record = []
        for data_id, data in tqdm(self.data_dict.items(), total=len(self.data_dict)):
            if data_id < self.start_idx:
                continue
            ori_cong_pkl_name = data["ori_cong_pickle_name"]
            save_path = os.path.join(self.output_dir, ori_cong_pkl_name.split(".")[0] + ".npy")
            success_grasp_Ts = data["grasp_Ts"]
            grasp_num = len(success_grasp_Ts)

            if grasp_num == 0:
                print(f"Skip {ori_cong_pkl_name} because no grasp found")
                continue

            if os.path.exists(save_path) and not rewrite:
                print(f"Skip {ori_cong_pkl_name} because {save_path} exists")
                continue

            if ori_cong_pkl_name in ["constrained_Curtain_f0e5ec2c2b1a202c9e5d32eddde839e3_0_00414180060558533.pickle", "constrained_Bathtub_c609f4ec72f6094b322ed2ef5fc90e25_0_003123052179282923.pickle", "constrained_Vase_8fb8d64e04e0be16a383bc0337fb2972_0_005205584401434865.pickle", "constrained_TV_536f6eb94a60370eeeb828903de1947_0_0009229711140942286.pickle", "constrained_LightSwitch_2307c0f4e06b01c1ff643b6971c915c2_0_0569230054262237.pickle", "constrained_Basket_2294583b27469851da06470a491566ee_0_019064499391211836.pickle", "constrained_Bottle_c5eb3234b73037562825656dc457df78_0_008788526522791613.pickle", "constrained_FloorLamp_bd234f132e160fad8f045e3f1e9c8518_0_0030415802988862377.pickle", "constrained_Wii_a8dc576b875bc3cae28a83d476eddeca_0_0006153213536542015.pickle", "constrained_LightSwitch_d7bf02ee873e86c469d6e26572bd40e5_0_03486449970751047.pickle", "constrained_Whiteboard_1492b65a4dedd2612a384701382c9079_0_00349133232449086.pickle",
            "constrained_PowerSocket_a5d38fa50974f4713a619048371e969d_0_04097309596833642.pickle", 
            "constrained_PowerSocket_a5d38fa50974f4713a619048371e969d_0_04097309596833642.pickle"]:
                print("skip this one")
                continue

            # # @note 选 10 个代表出来测试
            # random_idx = np.random.choice(grasp_num, min(grasp_num, 10), replace=False)
            # data["grasp_Ts"] = np.array(success_grasp_Ts)[random_idx]
            # print(f"Random idx: {random_idx}")
            # print(data["grasp_Ts"])

            print(f"Start eval {ori_cong_pkl_name}")
            self.eval_one_obj(data, debug_vis)
            success_cases, total_num, success_flags = data['eva_result_success_num'], data['eva_result_grasp_num'], data['eva_result_success']

            success_num_record.append(success_cases)
            num_record.append(total_num)
            success_record.append(success_flags)

            print("================================")
            print("summary: {} / {}".format(sum(success_num_record), sum(num_record)))
            print("success rate: {}".format(sum(success_num_record) / sum(num_record)))
            print("================================")

            np.save(save_path, data)

        np.save(os.path.join(self.cache_output_dir, self.exp_name + ".npy"), data)
        return num_record, success_num_record, success_record
        pass


if __name__ == '__main__':
    cache_dir = "/home/red0orange/Projects/handgrasp_ws/0_Data/IsaacGymCache/obj_cache"
    os.makedirs(cache_dir, exist_ok=True)
    test_data_path = "/home/red0orange/Projects/handgrasp_ws/tmp.npy"

    evaluator = IsaacGymGraspEva(test_data_path, cache_dir)
    evaluator.eval()