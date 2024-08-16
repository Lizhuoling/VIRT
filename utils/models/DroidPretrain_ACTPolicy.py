import pdb
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.models.DroidPretrain_ACT import get_DroidPretrain_ACT_model
import IPython
e = IPython.embed

class DroidPretrain_ACTPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feat_regu_loss_weight = 5

        model = get_DroidPretrain_ACT_model(cfg)
        self.model = model.cuda()

    def __call__(self, image, goal_image, past_action, action, end_obs, joint_obs, observation_is_pad, past_action_is_pad, action_is_pad, task_instruction_list):
        env_state = None
        if action is not None: # training or validation time
            a_hat, a_hat_uncern, reat_regu_loss = self.model(image = image, goal_image = goal_image, past_action = past_action, action = action, end_obs = end_obs, joint_obs = joint_obs, \
                            observation_is_pad = observation_is_pad, past_action_is_pad = past_action_is_pad, action_is_pad = action_is_pad, task_instruction_list = task_instruction_list)
            loss_dict = dict()
            all_l1 = F.l1_loss(action.unsqueeze(0).expand(a_hat.shape[0], -1, -1, -1), a_hat, reduction='none') # Left shape: (num_dec, B, num_query, num_action)
            expand_action_is_pad = action_is_pad[None, :, :, None].expand(all_l1.shape[0], -1, -1, all_l1.shape[3])    # action_is_pad shape: (B, num_query), expand_action_is_pad shape: (num_dec, B, num_query, num_action)
            mask_l1 = (all_l1 * ~expand_action_is_pad).sum(dim = -1) # Left shape: (num_dec, B, num_query)
            if self.cfg['POLICY']['USE_UNCERTAINTY']:
                a_hat_uncern = a_hat_uncern.squeeze(-1) # Left shape: (num_dec, B, num_query)
                a_hat_uncern[action_is_pad[None].expand(a_hat_uncern.shape[0], -1, -1)] = a_hat_uncern[action_is_pad[None].expand(a_hat_uncern.shape[0], -1, -1)].detach()
                uncern_l1 = math.sqrt(2) * mask_l1 / a_hat_uncern.exp() + a_hat_uncern  # Left shape: (num_dec, B, num_query)
            else:
                uncern_l1 = mask_l1
            uncern_l1 = uncern_l1.sum(-1)   # Left shape: (num_dec, B)
            valid_count = torch.clip((~action_is_pad)[None].sum(dim = -1), min = 1)  # Left shape: (num_dec,)
            l1 = (uncern_l1 / valid_count).mean(dim = -1)    # Left shape: (num_dec,)
            total_loss = l1.sum()

            if self.cfg['TRAIN']['FEATURE_REGULARIZATION']:
                reat_regu_loss = self.feat_regu_loss_weight * reat_regu_loss
                total_loss = total_loss + reat_regu_loss
                loss_dict['feat_regu'] = reat_regu_loss.item()

            if self.cfg['POLICY']['USE_UNCERTAINTY']:
                l1_without_uncern = (mask_l1.sum(-1) / valid_count).mean(dim = -1)  # Left shape: (num_dec,)
                total_l1_without_uncern = l1_without_uncern.sum()
                loss_dict['loss_without_uncern'] = total_l1_without_uncern.item()
            else:
                l1_without_uncern = l1
            for dec_id in range(l1.shape[0]):
                loss_dict[f'dec_{dec_id}'] = l1_without_uncern[dec_id].item()
            loss_dict['total_loss'] = total_loss.item()
            
            return total_loss, loss_dict
        else: # inference time
            raise Exception("Inference is not supported now.")

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
