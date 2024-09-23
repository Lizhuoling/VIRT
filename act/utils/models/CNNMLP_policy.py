import torch.nn as nn
from torch.nn import functional as F
from detr.models.detr_vae import get_CNNMLP_model
import pdb

class CNNMLPPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = get_CNNMLP_model(cfg)

    def __call__(self, qpos, image, actions=None, is_pad=None, task_instruction = None):

        if actions is not None: # training time
            a_hat = self.model(qpos, image, task_instruction, actions)
            l1_loss = F.l1_loss(actions, a_hat, reduction='mean')
            
            loss_dict = dict()
            loss_dict['dec_0'] = l1_loss.item()
            total_loss = l1_loss 
            loss_dict['loss'] = total_loss.item()
            return total_loss, loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, task_instruction) # no action, sample from prior
            return a_hat, None