import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
mesh_root = os.path.join(proj_dir, 'data/obj_ShapeNetSem/models-OBJ/models')
isaacgym_dir = os.path.join(os.path.dirname(proj_dir), "3rd_isaacgym_evaluation")
sys.path.append(isaacgym_dir)
from isaacgym_eval import IsaacGymGraspEva


proj_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    cache_dir = os.path.join(proj_dir, 'data/IsaacGymCache')
    # test_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/isaacgym_eval_results.npy"
    # test_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/oakink_isaacgym_eval_results.npy"
    # test_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241019-115052_detectiondiffusion/eval_oakink_only_s_isaacgym.npy"
    # test_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_599_20241026-145515_grasp_diffusion_baseline/graspdiff_eval_cong_isaacgym.npy"
    # test_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_599_20241026-145515_grasp_diffusion_baseline/eval_cong_split_final_isaacgym.npy"
    test_data_path = "/home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/log_remote/epoch_299_20241008-111209_detectiondiffusion/contactgn_oakink_best_isaacgym.npy"
    save_name = "contactgn_oakink_best_isaacgym_results"
    # save_name = "eval_cong_fix_initial_results"
    # save_name = "eval_cong_split_final_results"
    # save_name = "eval_oakink_cg_norandom_results"
    # save_name = "eval_oakink_cg_results"
    # save_name = "eval_oakink_rdt_only_s_isaacgym_eval_results"
    # save_name = "tmp"
    evaluator = IsaacGymGraspEva(test_data_path, cache_dir, n_envs=10)
    evaluator.eval(debug_vis=False, save_name=save_name)
    # evaluator.debug_vis()