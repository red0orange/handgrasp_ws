import os
import subprocess
import shutil
import json
import hashlib

from tqdm import tqdm
import numpy as np
import open3d as o3d
import trimesh

from roboutils.proj_llm_robot.pose_transform import update_pose
from roboutils.vis.viser_grasp import ViserForGrasp
import viser.transforms as tf


def T2sevendof(T):
    tf_se3 = tf.SE3.from_matrix(T)
    wxyz = tf_se3.wxyz_xyz[:4].tolist()
    xyz = tf_se3.wxyz_xyz[4:].tolist()
    return np.array([*xyz, wxyz[0], wxyz[1], wxyz[2], wxyz[3]])


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


def change_permissions(root_dir, permission=0o777):
    # 设置目录及其所有子目录和文件的权限
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 更改目录权限
        os.chmod(dirpath, permission)
        # 更改所有文件权限
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            os.chmod(file_path, permission)


def run_command(command):
    try:
        # 使用 subprocess.run 执行命令，捕获输出和错误信息
        result = subprocess.run(command, check=True, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 打印命令的标准输出
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，打印错误信息并退出程序
        print("Error:\n", e.stderr)
        raise SystemExit(f"Command failed with exit code {e.returncode}")


class IsaacSimEva:
    def __init__(self, exp_name, data_dict, cache_dir, debug=False, rewrite_obj=False, rewrite_json=False):
        """
        data_dict: dict
            {
                meta_info: dict
                    {
                        ''
                    },
                id: dict
                    {
                        'grasp_Ts': np.ndarray,
                        'mesh_path': str,
                        'mesh_scale': float,
                        'mesh_T': np.ndarray,
                    }
                ...
            }
        """
        self.exp_name = exp_name
        self.data_dict = data_dict

        self.usd_cache_dir = os.path.join(cache_dir, 'converted_usd')  # 被转换的usd文件，用 md5 命名，避免重复转换
        self.obj_to_convert_dir = os.path.join(cache_dir, 'obj_to_convert')  # 需要转换的 obj 文件
        self.obj_bak_dir = os.path.join(cache_dir, 'obj_bak')  # 需要转换的 obj 文件
        os.makedirs(self.usd_cache_dir, exist_ok=True)
        os.makedirs(self.obj_bak_dir, exist_ok=True)
        os.makedirs(self.obj_to_convert_dir, exist_ok=True)

        self.cache_dir = os.path.join(cache_dir, exp_name)
        self.json_save_dir = os.path.join(self.cache_dir, 'grasps_jsons')

        # @note debug 可视化检查数据
        if debug:
            viser_grasp = ViserForGrasp()
            for data_id, data in self.data_dict.items():
                grasp_Ts = data['grasp_Ts']
                mesh_path = data['mesh_path']
                mesh_scale = data['mesh_scale']
                mesh_T = data['mesh_T']

                mesh = o3d.io.read_triangle_mesh(mesh_path)
                mesh = mesh.scale(mesh_scale, center=[0, 0, 0])
                mesh = mesh.transform(mesh_T)
                pc = np.array(mesh.vertices)

                grasp_success = data['grasp_success']
                success_grasp_Ts = np.where(grasp_success)[0]
                success_grasp_Ts = [grasp_Ts[i] for i in success_grasp_Ts]
                print(f"Success grasp num: {len(success_grasp_Ts)}")
                for i in range(len(success_grasp_Ts)):
                    print("current grasp id: ", i)
                    grasp_T = success_grasp_Ts[i]
                    viser_grasp.vis_grasp_scene(max_grasp_num=1, pc=pc, grasp_Ts=[grasp_T], mesh=mesh)
                    viser_grasp.wait_for_reset()

                viser_grasp.vis_grasp_scene(max_grasp_num=40, pc=pc, grasp_Ts=grasp_Ts, mesh=mesh)
                viser_grasp.wait_for_reset()

        # @note 开始转换数据
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.json_save_dir, exist_ok=True)

        print("Begin to convert data_dict to IsaacSimGrasp format")
        load_cache = False

        # @note obj to usd
        to_convert_obj_paths = []
        usd_obj_names = [i.split('.')[0] for i in os.listdir(self.usd_cache_dir)]
        for data_id, data in tqdm(self.data_dict.items(), total=len(self.data_dict), desc="Saving obj for conversion"):
            mesh_path = data['mesh_path']
            mesh_scale = data['mesh_scale']
            mesh_T = data['mesh_T']

            md5 = get_md5(mesh_path)
            # 判断是否已经转换过
            if (not rewrite_obj) and (md5 in usd_obj_names):
                self.data_dict[data_id]['mesh_md5'] = md5
                self.data_dict[data_id]['mesh_usd_path'] = os.path.join(self.usd_cache_dir, f"{md5}.usd")
                continue

            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh = mesh.scale(mesh_scale, center=[0, 0, 0])
            mesh = mesh.transform(mesh_T)

            save_path = os.path.join(self.obj_bak_dir, f"{md5}.obj") 
            o3d.io.write_triangle_mesh(save_path, mesh)

            save_path = os.path.join(self.obj_to_convert_dir, f"{md5}.obj") 
            o3d.io.write_triangle_mesh(save_path, mesh)
            to_convert_obj_paths.append(save_path)

            # 绑定对应的 md5 和 usd 路径
            self.data_dict[data_id]['mesh_md5'] = md5
            self.data_dict[data_id]['mesh_usd_path'] = os.path.join(self.usd_cache_dir, f"{md5}.usd")

        # 调用 IsaacSim 转换 obj 为 usd
        to_convert_obj_names = os.listdir(self.obj_to_convert_dir)
        if len(to_convert_obj_names) > 0:
            print(f"Begin to convert {len(to_convert_obj_names)} obj files to usd")
            convert_command = f"bash /home/red0orange/Projects/handgrasp_ws/multigrippergrasp_examples/convert_obj_to_usd.sh"
            run_command(convert_command)

            # 删除 obj 文件
            for obj_name in to_convert_obj_names:
                os.remove(os.path.join(self.obj_to_convert_dir, obj_name))

        # @note 开始导出 IsaacSimGrasping 的 json 格式数据
        # change_permissions(self.cache_dir, permission=0o777)
        # h5_dofs = [1.37881, -1.37881, -1.37881, 1.37881]
        if rewrite_json:
            shutil.rmtree(self.json_save_dir)
            os.makedirs(self.json_save_dir)
        h5_dofs = [-1.37881, 1.37881, 1.37881, -1.37881]
        franka_dofs = [0, 0]
        gripper = "franka_panda"
        # gripper = "h5_hand"
        for data_id, data in tqdm(self.data_dict.items(), total=len(self.data_dict), desc="Saving grasps json"):
            object_name = data['mesh_md5']

            json_save_path = os.path.join(self.json_save_dir, f"{data_id}_{object_name}.json")
            if (not rewrite_json) and (os.path.exists(json_save_path)):
                continue

            grasp_Ts = data['grasp_Ts']
            # @note BUG 有可能四元数顺序不对，需要调整
            for i in range(len(grasp_Ts)):
                grasp_T = grasp_Ts[i]
                if gripper == "h5_hand":
                    grasp_T = update_pose(grasp_T, rotate=-np.pi/2, rotate_axis="z")
                    grasp_T = update_pose(grasp_T, rotate=-np.pi, rotate_axis="y")
                elif gripper == "franka_panda":
                    # grasp_T = update_pose(grasp_T, rotate=-np.pi/2, rotate_axis="z")
                    # grasp_T = update_pose(grasp_T, translate=[0, 0, -0.005])
                    pass
                else:
                    raise ValueError(f"Unknown gripper {gripper}")
                grasp_Ts[i] = grasp_T

            grasp_poses = [T2sevendof(T).tolist() for T in grasp_Ts]
            if gripper == "h5_hand":
                dofs = [h5_dofs for _ in range(len(grasp_Ts))]
            elif gripper == "franka_panda":
                dofs = [franka_dofs for _ in range(len(grasp_Ts))]
            else:
                raise ValueError(f"Unknown gripper {gripper}")

            cur_data_dict = {
                "object_id": object_name,
                "pose": grasp_poses,
                "dofs": dofs,
                "gripper": gripper,

                # "prior_success": 
            }

            json.dump(cur_data_dict, open(json_save_path, "w"))
            pass

    def eval(self):

        pass


if __name__ == '__main__':
    pass