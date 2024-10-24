import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, MultiStepLR
from dataset import *
from models import *
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

# Pools of models, optimizers, weights initialization methods, schedulers
model_pool = {
    'scorebasedgraspingdiffusion': ScoreBasedGraspingDiffusion
}

optimizer_pool = {
    'sgd': SGD,
    'adam': Adam
}

init_pool = {
    'default_init': weights_init
}

scheduler_pool = {
    'step': StepLR,
    'cos': CosineAnnealingLR,
    'lr_lambda': LambdaLR,
    'multi_step': MultiStepLR
}


def build_model(cfg):
    """_summary_
    Function to build the model before training
    """
    if hasattr(cfg, 'model'):
        model_info = cfg.model
        weights_init = model_info.get('weights_init', None)
        device = model_info.get('device', torch.device('cuda'))
        model_name = model_info.type
        if model_name == 'scorebasedgraspingdiffusion':
            betas = model_info.get('betas', [1e-4, 0.02])
            n_T = model_info.get('n_T', 1000)
            drop_prob = model_info.get('drop_prob', 0.1)
            rot6d_rep = cfg.hyper_params.rot6d_rep
            training_method = cfg.hyper_params.training_method
            sigma = model_info.get('sigma', 25.0)
            unet_or_transformer = cfg.hyper_params.unet_or_transformer
            grasp_obj_joint_embed = cfg.hyper_params.grasp_obj_joint_embed
            model = ScoreBasedGraspingDiffusion(betas, n_T, device, drop_prob, rot6d_rep=rot6d_rep, unet_or_transformer=unet_or_transformer, training_method=training_method, sigma=sigma, grasp_obj_joint_embed=grasp_obj_joint_embed)
        else:
            raise ValueError("The model name does not exist!")
        if weights_init != None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        return model
    else:
        raise ValueError("Configuration does not have model config!")


def build_dataset(cfg):
    """_summary_
    Function to build the dataset
    """
    if hasattr(cfg, 'data'):
        data_info = cfg.data
        data_dir = data_info.data_dir
        dataset_name = data_info.type
        fix_local_geo = cfg.hyper_params.fix_local_geo
        if dataset_name == '3DAP':
            train_set = _3DAPDataset(data_dir, mode='train')
            test_set = _3DAPDataset(data_dir, mode='test')
        elif dataset_name == "Acronym":
            train_set = _AcronymDataset(data_dir, mode='train', fix_local_geo=fix_local_geo, use_loca_geo_cache=True)
            test_set = _AcronymDataset(data_dir, mode='test', fix_local_geo=fix_local_geo, use_loca_geo_cache=True)
        elif dataset_name == "CONG":
            split_json_path = None
            if hasattr(data_info, 'split_json_path'):
                split_json_path = data_info.split_json_path
            train_set = _CONGDataset(data_dir, mode='train', split_json_path=split_json_path)
            test_set = _CONGDataset(data_dir, mode='valid', split_json_path=split_json_path)
        elif dataset_name == "CONGDiff":
            split_json_path = None
            if hasattr(data_info, 'split_json_path'):
                split_json_path = data_info.split_json_path
            acronym_data_dir = data_info.acronym_data_dir
            train_set = _CONGDiffDataset(data_dir, acronym_data_dir=acronym_data_dir, mode='train', split_json_path=split_json_path)
            test_set = _CONGDiffDataset(data_dir, acronym_data_dir=acronym_data_dir, mode='valid', split_json_path=split_json_path)
        else:
            raise ValueError("The dataset name does not exist!")
        dataset_dict = dict(
            train_set=train_set,
            test_set=test_set
        )
        return dataset_dict
    else:
        raise ValueError("Configuration does not have data config!")


def build_loader(cfg, dataset_dict):
    """_summary_
    Function to build the loader
    """
    train_set = dataset_dict["train_set"]
    train_loader = DataLoader(train_set, batch_size=cfg.training_cfg.batch_size,
                              shuffle=True, drop_last=False, num_workers=cfg.training_cfg.num_workers)
    loader_dict = dict(
        train_loader=train_loader,
    )

    return loader_dict


def build_optimizer(cfg, model):
    """_summary_
    Function to build the optimizer
    """
    optimizer_info = cfg.optimizer
    optimizer_type = optimizer_info.type
    optimizer_info.pop('type')
    optimizer_cls = optimizer_pool[optimizer_type]
    optimizer = optimizer_cls(model.parameters(), **optimizer_info)
    scheduler_info = cfg.scheduler
    if scheduler_info:
        scheduler_name = scheduler_info.type
        scheduler_info.pop('type')
        scheduler_cls = scheduler_pool[scheduler_name]
        scheduler = scheduler_cls(optimizer, **scheduler_info)
    else:
        scheduler = None
    optim_dict = dict(
        scheduler=scheduler,
        optimizer=optimizer
    )
    return optim_dict