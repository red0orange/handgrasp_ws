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

from models.graspdiff.main_nets import load_graspdiff
from models.graspdiff.losses import get_loss_fn
from models.graspdiff.learning_rate import get_lr_scheduler


if __name__ == '__main__':
    model = load_graspdiff()
    loss_fn = get_loss_fn()
    lr_schedules = get_lr_scheduler()

    optimizer = torch.optim.Adam([
        {
            "params": model.vision_encoder.parameters(),
            "lr": lr_schedules[0].get_learning_rate(0),
        },
        {
            "params": model.feature_encoder.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        },
        {
            "params": model.decoder.parameters(),
            "lr": lr_schedules[2].get_learning_rate(0),
        },
    ])
    model.float()
    pass

