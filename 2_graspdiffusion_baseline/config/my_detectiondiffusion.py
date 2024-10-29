import os
import torch
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = 'detectiondiffusion'
seed = 1
log_dir = opj("./log/", exp_name)
try:
    os.makedirs(log_dir)
except:
    print('Logging Dir is already existed!')

hyper_params = dict(
    rot6d_rep=True,
    fix_local_geo=False,
    unet_or_transformer="transformer",
    training_method="score_based",
    grasp_obj_joint_embed=True,
)

scheduler = dict(
    type='lr_lambda',
    lr_lambda=PN2_Scheduler(init_lr=0.001, step=20,
                            decay_rate=0.5, min_lr=1e-5)
)

optimizer = dict(
    type='adam',
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-04,                  # @note 与 Loss nan 有关，把这个值改大
    weight_decay=1e-5,
)

model = dict(
    type='scorebasedgraspingdiffusion',
    device=torch.device('cuda'),
    betas=[1e-4, 0.02],
    n_T=1000,
    drop_prob=0.1,
    weights_init='default_init',
    sigma=0.2,
)

training_cfg = dict(
    model=model,
    batch_size=32,
    num_workers=16,
    epoch=600,
    gpu='0',
    workflow=dict(
        train=1,
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),
)

data = dict(
    type="CONG",
    data_dir="./data/grasp_CONG_graspldm",
    split_json_path="/home/huangdehao/Projects/handgrasp_ws/2_graspdiffusion_baseline/data/grasp_CONG_graspldm/selected_valid_split.json"
)