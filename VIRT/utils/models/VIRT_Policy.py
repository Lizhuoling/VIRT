import pdb
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from VIRT.detr.models.VIRT import get_VIRT_model
import IPython
e = IPython.embed

class VIRT_Policy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = get_VIRT_model(cfg)
        self.model = model.cuda()

        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.status_cls_loss_weight = 10

    def __call__(self, image, past_action, end_obs, joint_obs, action=None, observation_is_pad = None, past_action_is_pad = None, action_is_pad = None, task_instruction_list = None, status = None):
        env_state = None
        if action is not None: 
            a_hat, a_hat_uncern, status_pred = self.model(image = image, past_action = past_action, end_obs = end_obs, joint_obs = joint_obs, env_state = env_state, action = action, \
                            observation_is_pad = observation_is_pad, past_action_is_pad = past_action_is_pad, action_is_pad = action_is_pad, task_instruction_list = task_instruction_list, status = status)
            loss_dict = dict()
            all_l1 = F.l1_loss(action.unsqueeze(0).expand(a_hat.shape[0], -1, -1, -1), a_hat, reduction='none') 
            expand_action_is_pad = action_is_pad[None, :, :, None].expand(all_l1.shape[0], -1, -1, all_l1.shape[3])    
            mask_l1 = (all_l1 * ~expand_action_is_pad).sum(dim = -1) 
            if self.cfg['POLICY']['USE_UNCERTAINTY']:
                a_hat_uncern = a_hat_uncern.squeeze(-1) 
                a_hat_uncern[action_is_pad[None].expand(a_hat_uncern.shape[0], -1, -1)] = a_hat_uncern[action_is_pad[None].expand(a_hat_uncern.shape[0], -1, -1)].detach()
                uncern_l1 = math.sqrt(2) * mask_l1 / a_hat_uncern.exp() + a_hat_uncern  
            else:
                uncern_l1 = mask_l1
            uncern_l1 = uncern_l1.sum(-1)   
            valid_count = torch.clip((~action_is_pad)[None].sum(dim = -1), min = 1)  
            l1 = (uncern_l1 / valid_count).mean(dim = -1)    
            total_loss = l1.sum()

            if self.cfg['POLICY']['STATUS_PREDICT']:
                for dec_cnt in range(status_pred.shape[0]):
                    status_pred_loss = self.status_cls_loss_weight * self.CrossEntropyLoss(status_pred[dec_cnt, :, :], status.long())
                    loss_dict[f'status_pred_{dec_cnt}'] = status_pred_loss.item()
                    total_loss = total_loss + status_pred_loss

            if self.cfg['POLICY']['USE_UNCERTAINTY']:
                l1_without_uncern = (mask_l1.sum(-1) / valid_count).mean(dim = -1)  
                total_l1_without_uncern = l1_without_uncern.sum()
                loss_dict['loss_without_uncern'] = total_l1_without_uncern.item()
            else:
                l1_without_uncern = l1
            for dec_id in range(l1.shape[0]):
                loss_dict[f'dec_{dec_id}'] = l1_without_uncern[dec_id].item()
            loss_dict['total_loss'] = total_loss.item()
            
            return total_loss, loss_dict
        else: 
            a_hat, _, status_pred = self.model(image = image, past_action = past_action, end_obs = end_obs, joint_obs = joint_obs, env_state = env_state, \
                            observation_is_pad = observation_is_pad, past_action_is_pad = past_action_is_pad, task_instruction_list = task_instruction_list, status = status)
            return a_hat, status_pred
