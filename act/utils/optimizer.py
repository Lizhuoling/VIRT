import torch
from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle

def make_scheduler(optimizer, cfg, last_epoch=-1):
    decay_steps = cfg['TRAIN']['DECAY_LR_ITER_STEPS']
    
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['TRAIN']['LR_DECAY']
        
        return max(cur_decay, cfg['TRAIN']['LR_CLIP'] / cfg['TRAIN']['LR'])
       
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    lr_warmup_scheduler = None
    if cfg['TRAIN']['LR_WARMUP']:
        lr_warmup_scheduler = CosineWarmupLR(optimizer, T_max=cfg['TRAIN']['WARMUP_STEPS'], eta_min=cfg['TRAIN']['LR'] / 10)

    return lr_scheduler, lr_warmup_scheduler


def make_optimizer(model, cfg):
    model_params = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg['TRAIN']['LR_BACKBONE'],
        },
    ]

    if cfg['TRAIN']['OPTIMIZER'] == 'adamw':
        optimizer = torch.optim.AdamW(model_params, lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'], betas=(0.9, 0.99))
    else:
        raise NotImplementedError
    
    return optimizer