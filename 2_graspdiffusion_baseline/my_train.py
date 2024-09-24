import os
import sys
import subprocess
import shutil
import datetime
from os.path import join as opj
from gorilla.config import Config
from utils import *
import argparse
import torch


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()
    return args


def save_git_info(commit_info_file, uncommitted_changes_file):
    bash_script = opj(os.path.dirname(os.path.abspath(__file__)), "scripts/record_git_info.sh")
    result = subprocess.run([bash_script, commit_info_file, uncommitted_changes_file], capture_output=True, text=True)
    if result.returncode == 0:
        print("Git information saved successfully.")
    else:
        print("Error occurred while saving git information.")
        print(result.stderr)


if __name__ == "__main__":
    args = parse_args()
    config_file_path = args.config

    cfg = Config.fromfile(config_file_path)
    
    time_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S_')
    shutil.rmtree(cfg.log_dir, ignore_errors=True)
    cfg.log_dir = opj(os.path.dirname(cfg.log_dir), time_str + os.path.basename(cfg.log_dir))
    cfg.base_log_dir = cfg.log_dir
    os.mkdir(cfg.log_dir)
    
    # 记录配置文件、git commit id，缓存区
    shutil.copy(config_file_path, opj(cfg.log_dir, 'config.py'))
    save_git_info(opj(cfg.log_dir, 'commit_info.txt'), opj(cfg.log_dir, 'uncommitted_changes.txt'))
        
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))      # number of GPUs to use

    tmp_logger = IOStream(opj(cfg.log_dir, 'run.log'))
    tmp_logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))
    if cfg.get('seed') != None:     # set random seed
        set_random_seed(cfg.seed)
        tmp_logger.cprint('Set seed to %d' % cfg.seed)

    model = build_model(cfg).cuda()     # build the model from configuration

    print("Training from scratch!")

    dataset_dict = build_dataset(cfg)       # build the dataset
    loader_dict = build_loader(cfg, dataset_dict)       # build the loader
    optim_dict = build_optimizer(cfg, model)        # build the optimizer
    
    # construct the training process
    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        optim_dict=optim_dict,
    )

    task_trainer = MyTrainer(cfg, training)
    task_trainer.run()
