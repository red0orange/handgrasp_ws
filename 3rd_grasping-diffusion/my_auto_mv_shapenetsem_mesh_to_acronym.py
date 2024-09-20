import os
import shutil

import numpy as np


if __name__ == "__main__":
    shapenet_model_dir = "/home/huangdehao/Projects/grasping-diffusion/data/my_shapenetsem/models-OBJ/simplified"
    acronym_grasp_dir = "/home/huangdehao/Projects/grasping-diffusion/data/my_acronym/grasps"
    to_save_acronym_mesh_dir = "/home/huangdehao/Projects/grasping-diffusion/data/my_acronym/meshes"

    grasps_names = [i.split(".")[0] for i in os.listdir(acronym_grasp_dir)]
    for grasp_name in grasps_names:
        obj_cat, obj_name, obj_scale = grasp_name.split("_")

        obj_cat_dir = os.path.join(to_save_acronym_mesh_dir, obj_cat)
        os.makedirs(obj_cat_dir, exist_ok=True)

        if not os.path.exists(os.path.join(shapenet_model_dir, obj_name + ".obj")):
            print("Skipping", obj_name, "not found")
            continue

        model_path = os.path.join(shapenet_model_dir, obj_name + ".obj")
        save_path = os.path.join(obj_cat_dir, obj_name + ".obj")
        shutil.copy(model_path, save_path)
    pass