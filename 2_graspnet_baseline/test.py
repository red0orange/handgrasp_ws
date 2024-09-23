import os
import numpy as np
from tqdm import tqdm
from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer

from roboutils.vis.viser_grasp import ViserForGrasp

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data")
# cong_data_dir = os.path.join(data_dir, "grasp_CONG")
cong_data_dir = os.path.join(data_dir, "grasp_CONG_small")
shapenetsem_dir = os.path.join(data_dir, "obj_ShapeNetSem/models-OBJ/models")


def run_test(epoch=-1, name=""):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name

    opt.caching = True
    opt.validate = False
    opt.mesh_root_folder = shapenetsem_dir

    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(data)

        # # @note debug data
        # viser_grasp = ViserForGrasp()
        # obj_pc = data["pc"]
        # grasp_pc = data["grasp_rt"]
        # labels = data["labels"]
        # batch_size = obj_pc.shape[0]
        # for batch_i in range(batch_size):
        #     per_obj_pc = obj_pc[batch_i]
        #     per_grasp_pc = grasp_pc[batch_i]
        #     per_label = labels[batch_i]
        #     print("current label: ", per_label)

        #     pc_colors = np.zeros((per_obj_pc.shape[0] + per_grasp_pc.shape[0], 3), dtype=np.uint8)
        #     pc_colors[:per_obj_pc.shape[0], :] = np.array([127, 127, 127])
        #     pc = np.concatenate([per_obj_pc, per_grasp_pc], axis=0)

        #     viser_grasp.add_pcd(pc, colors=pc_colors)
        #     viser_grasp.wait_for_reset()

        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    import sys

    sys.argv.append('--dataset_root_folder')
    sys.argv.append(cong_data_dir)
    sys.argv.append('--arch')
    # sys.argv.append('vae')
    sys.argv.append('evaluator')
    sys.argv.append('--num_threads')
    sys.argv.append('0')

    checkpoint_name = "evaluator_lr_002_bs_64_scale_1_npoints_128_radius_02"

    run_test(name=checkpoint_name)
