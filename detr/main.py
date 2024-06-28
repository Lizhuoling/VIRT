# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path
import pdb

import numpy as np
import torch
from .models import build_vae, build_cnnmlp

import IPython
e = IPython.embed


def get_ACT_model(cfg):

    model = build_vae(cfg)
    model.cuda()

    return model


def get_CNNMLP_model(args_override):

    model = build_cnnmlp(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    return model

