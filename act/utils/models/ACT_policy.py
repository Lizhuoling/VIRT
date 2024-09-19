import pdb
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.models.detr_vae import get_ACT_model
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model = get_ACT_model(cfg)
        self.model = model.cuda() # CVAE decoder
        self.kl_weight = cfg['POLICY']['KL_WEIGHT']

    def __call__(self, qpos, image, actions, is_pad, task_instruction):
        if actions is not None: # training time
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos = qpos, image = image, task_instruction = task_instruction, actions = actions, is_pad = is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['dec_0'] = l1.item()
            loss_dict['kl'] = total_kld[0].item()
            total_loss = l1 + total_kld[0] * self.kl_weight
            loss_dict['loss'] = total_loss.item()

            return total_loss, loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, task_instruction) # no action, sample from prior
            return a_hat, None

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
