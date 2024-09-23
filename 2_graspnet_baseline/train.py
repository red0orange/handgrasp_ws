import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer
from test import run_test
import threading

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data")
cong_data_dir = os.path.join(data_dir, "grasp_CONG")
shapenetsem_dir = os.path.join(data_dir, "obj_ShapeNetSem/models-OBJ/models")


def main():
    # @note 数据集路径、模型选择
    sys.argv.append('--dataset_root_folder')
    sys.argv.append(cong_data_dir)
    sys.argv.append('--arch')
    # sys.argv.append('vae')
    sys.argv.append('evaluator')

    opt = TrainOptions().parse()
    if opt == None:
        return
    opt.validate = False
    opt.caching = False
    opt.mesh_root_folder = shapenetsem_dir


    dataset = DataLoader(opt)
    dataset_size = len(dataset) * opt.num_grasps_per_object
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        # dataset.dataset.renderer.renderer = None
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                loss_types = []
                if opt.arch == "vae":
                    loss = [
                        model.loss, model.kl_loss, model.reconstruction_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "kl_loss", "reconstruction_loss",
                        "confidence loss"
                    ]
                elif opt.arch == "gan":
                    loss = [
                        model.loss, model.reconstruction_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "reconstruction_loss", "confidence_loss"
                    ]
                else:
                    loss = [
                        model.loss, model.classification_loss,
                        model.confidence_loss
                    ]
                    loss_types = [
                        "total_loss", "classification_loss", "confidence_loss"
                    ]
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data,
                                            loss_types)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size,
                                 loss_types)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest', epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest', epoch)
            model.save_network(str(epoch), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,
               time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch, name=opt.name)
            writer.plot_acc(acc, epoch)

    writer.close()


if __name__ == '__main__':
    main()
