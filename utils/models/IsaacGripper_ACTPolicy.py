import pdb
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.models.IsaacGripper_ACT import get_IsaacGripper_ACT_model
import IPython
e = IPython.embed

class IsaacGripper_ACTPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = get_IsaacGripper_ACT_model(cfg)
        self.model = model.cuda() # CVAE decoder
        if self.cfg['POLICY']['USE_VAE']:
            self.kl_weight = cfg['POLICY']['KL_WEIGHT']

    def __call__(self, image, past_action, end_obs, joint_obs, action=None, observation_is_pad = None, past_action_is_pad = None, action_is_pad = None, task_instruction_list = None):
        env_state = None
        if action is not None: # training or validation time
            a_hat, is_pad_hat, (mu, logvar) = self.model(image = image, past_action = past_action, end_obs = end_obs, joint_obs = joint_obs, env_state = env_state, action = action, \
                            observation_is_pad = observation_is_pad, past_action_is_pad = past_action_is_pad, action_is_pad = action_is_pad, task_instruction_list = task_instruction_list)
            loss_dict = dict()
            
            all_l1 = F.l1_loss(action, a_hat, reduction='none')
            l1 = (all_l1 * ~action_is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            if self.cfg['POLICY']['USE_VAE']:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            else:
                loss_dict['loss'] = l1
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(image = image, past_action = past_action, end_obs = end_obs, joint_obs = joint_obs, env_state = env_state, \
                            observation_is_pad = observation_is_pad, past_action_is_pad = past_action_is_pad, task_instruction_list = task_instruction_list)
            
            return a_hat

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
