import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
mesh_root = os.path.join(proj_dir, 'data/obj_ShapeNetSem/models-OBJ/models')
isaacgym_dir = os.path.join(os.path.dirname(proj_dir), "3rd_isaacgym_evaluation")
sys.path.append(isaacgym_dir)
from isaacgym_eval import IsaacGymGraspEva


if __name__ == '__main__':
    cache_dir = "/home/huangdehao/Projects/handgrasp_ws/2_graspdiff_baseline/data/IsaacGymCache"
    test_data_path = "/home/huangdehao/Projects/handgrasp_ws/2_graspdiff_baseline/log/epoch_199_20240923-232338_detectiondiffusion/isaacgym_eval_results.npy"
    evaluator = IsaacGymGraspEva(test_data_path, cache_dir, n_envs=10)
    evaluator.eval()