import shutil
from tqdm import tqdm
import os
import collections
import numpy as np
from scipy.spatial import KDTree
from os.path import join as opj

import torch
from pytorch3d.ops import knn_points

from utils import *
from utils.utils import IOStream


DEVICE = torch.device('cuda')

# import warnings
# warnings.filterwarnings('error')


class MyTrainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.optimizer_dict = running["optim_dict"]
        self.optimizer = self.optimizer_dict.get("optimizer", None)
        self.scheduler = self.optimizer_dict.get("scheduler", None)
        self.epoch = 0
        self.bn_momentum = self.cfg.training_cfg.get('bn_momentum', None)

        # hyperparameters
        self.rot6d_rep = self.cfg.hyper_params.rot6d_rep
        pass

    def log(self, content):
        tmp_logger = IOStream(opj(self.cfg.log_dir, 'run.log'))
        tmp_logger.cprint(content)
        pass
              
    def train(self):
        # @note 训练的主函数
        self.model.train()
        self.log("Epoch(%d) begin training........" % self.epoch)
        
        new_log_dir = opj(os.path.dirname(self.cfg.base_log_dir), "epoch_{}_".format(self.epoch) + os.path.basename(self.cfg.base_log_dir))
        os.rename(self.cfg.log_dir, new_log_dir)
        self.cfg.log_dir = new_log_dir

        pbar = tqdm(self.train_loader)
        cnt = 0
        # for file_names, ori_xyz, xyz, T, _, _ in pbar:
        for data in pbar:
            # @note for debug
            # if cnt < 207:
            #     cnt += 1
            #     continue
            file_names = data[0]
            xyz = data[1]
            T = data[2]

            self.optimizer.zero_grad()
            xyz = xyz.float()

            xyz = xyz.to(DEVICE)
            T = T.to(DEVICE)
            
            # trained_model_type = "energy_based"
            trained_model_type = "score_based"
            loss = self.model(xyz, T, data_scale=self.train_loader.dataset.scale, method=trained_model_type)
            if torch.isnan(loss):
                self.log("##################### Error: Loss is nan")
                print('##################### Error: Loss is nan')
                raise ValueError('Loss is nan')

            loss.backward()
            
            loss_l = loss.item()
            pbar.set_description(f'Pose loss: {loss_l:.5f}')
            self.optimizer.step()
            
        if self.scheduler != None:
            self.scheduler.step()   
        if self.bn_momentum != None:
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch))
        
        outstr = f"\nEpoch {self.epoch}, Last Pose loss: {loss_l:.5f}"
        self.log(outstr)
        print('Saving checkpoint')
        torch.save(self.model.state_dict(), opj(self.cfg.log_dir, 'current_model.t7'))

        # 保存最新的模型
        if self.epoch % 5 == 0:
            torch.save(self.model.state_dict(), opj(self.cfg.log_dir, f'model_epoch_{self.epoch}.pth'))
            print(f'Saved model at epoch {self.epoch}')

            # 获取保存的模型文件
            saved_models = sorted([f for f in os.listdir(self.cfg.log_dir) if f.startswith('model_epoch_')], 
                                key=lambda x: int(x.split('_')[-1].split('.')[0]))

            # 如果保存的模型文件超过最大数量，则删除最早的文件
            if len(saved_models) > 5:
                os.remove(os.path.join(self.cfg.log_dir, saved_models[0]))
                print(f'Removed model {saved_models[0]}')

        self.epoch += 1

    def val(self):
       raise NotImplementedError
       
    def test(self):
        raise NotImplementedError

    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        
        while self.epoch < EPOCH:
            for key, running_epoch in workflow.items():
                epoch_runner = getattr(self, key)
                for _ in range(running_epoch):
                    epoch_runner()


def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)

class GraspDiffTrainer(MyTrainer):
    def __init__(self, cfg, running):
        self.cfg = cfg

        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.optimizer_dict = running["optim_dict"]
        self.optimizer = self.optimizer_dict.get("optimizer", None)
        self.scheduler = self.optimizer_dict.get("scheduler", None)
        self.loss_fn = running["loss_fn"]

        self.epoch = 0
        self.device = torch.device('cuda')
        pass

    def train(self):
        # @note 训练的主函数
        self.model.train()
        self.log("Epoch(%d) begin training........" % self.epoch)
        
        new_log_dir = opj(os.path.dirname(self.cfg.base_log_dir), "epoch_{}_".format(self.epoch) + os.path.basename(self.cfg.base_log_dir))
        os.rename(self.cfg.log_dir, new_log_dir)
        self.cfg.log_dir = new_log_dir

        pbar = tqdm(self.train_loader)
        cnt = 0
        for data in pbar:
            model_input = data[0]
            gt = data[1]

            model_input = dict_to_device(model_input, self.device)
            gt = dict_to_device(gt, self.device)

            losses, iter_info = self.loss_fn(self.model, model_input, gt)
            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_loss += single_loss

            # Optimize the model
            for optim in self.optimizer:
                optim.zero_grad()

            train_loss.backward()

            for optim in self.optimizer:
                optim.step()

            pbar.set_description(f'Pose loss: {train_loss.item():.5f}')

        
        # 打印 Epoch 最后的损失
        loss_str = '\nEpoch: {}/{} '.format(self.epoch, self.cfg.training_cfg.epoch)
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            loss_str += '{}: {:.3f} '.format(loss_name, single_loss)
            train_loss += single_loss
        self.log(loss_str)

        print('Saving checkpoint')
        torch.save(self.model.state_dict(), opj(self.cfg.log_dir, 'current_model.t7'))

        # 保存最新的模型
        if self.epoch % 5 == 0:
            torch.save(self.model.state_dict(), opj(self.cfg.log_dir, f'model_epoch_{self.epoch}.pth'))
            print(f'Saved model at epoch {self.epoch}')

            # 获取保存的模型文件
            saved_models = sorted([f for f in os.listdir(self.cfg.log_dir) if f.startswith('model_epoch_')], 
                                key=lambda x: int(x.split('_')[-1].split('.')[0]))

            # 如果保存的模型文件超过最大数量，则删除最早的文件
            if len(saved_models) > 5:
                os.remove(os.path.join(self.cfg.log_dir, saved_models[0]))
                print(f'Removed model {saved_models[0]}')

        self.epoch += 1