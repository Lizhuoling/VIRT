
"""
DETR model and criterion classes.
"""
import pdb
import copy
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from utils.models.external_detector import get_detector
from detr.util.grid_mask import GridMask

import numpy as np

import IPython
e = IPython.embed

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VIRT(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, state_dim, chunk_size, camera_names, cfg):
        """ Initializes the model.
        Parameters:
            backbones: Image backbone.
            roi_backbone:  ROI image backbone.
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        if self.cfg['POLICY']['STATUS_PREDICT']:
            query_num = 1 + chunk_size
        else:
            query_num = chunk_size
        self.query_embed = nn.Embedding(query_num, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, state_dim) 
        if self.cfg['POLICY']['USE_UNCERTAINTY']:
            self.uncern_head = nn.Linear(hidden_dim, 1)
        if self.cfg['POLICY']['STATUS_PREDICT']:
            self.status_head = nn.Linear(hidden_dim, self.cfg['POLICY']['STATUS_NUM'])

        if self.cfg['POLICY']['BACKBONE'] == 'dinov2_s':
            self.input_proj = nn.Linear(backbone.num_features, hidden_dim)
            img_token_len = self.cfg['DATA']['IMG_RESIZE_SHAPE'][1] * self.cfg['DATA']['IMG_RESIZE_SHAPE'][0] // 14 // 14
            self.image_pos_embed = nn.Embedding(img_token_len, hidden_dim)

        
        if 'past_action' in self.cfg['DATA']['INPUT_KEYS']:
            self.past_action_mlp = nn.Linear(9, hidden_dim)  
            self.past_action_pos_emb = nn.Embedding(self.cfg['DATA']['PAST_ACTION_LEN'], hidden_dim)
        if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS'] or 'observations/joint_observation' in self.cfg['DATA']['INPUT_KEYS']:
            self.obs_pos_emb = nn.Embedding(self.cfg['DATA']['PAST_OBSERVATION_LEN'], hidden_dim)
            if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS']:
                self.end_obs_mlp = nn.Linear(13, hidden_dim) 
                self.end_obs_pos_emb = nn.Embedding(1, hidden_dim)
            if 'observations/joint_observation' in self.cfg['DATA']['INPUT_KEYS']:
                self.joint_obs_pos_emb = nn.Embedding(1, hidden_dim)
                self.joint_obs_mlp = nn.Linear(9, hidden_dim) 
        
        if self.cfg["POLICY"]["EXTERNAL_DET"] != 'None':
            self.object_detector = get_detector(cfg)
            self.obj_box_mlp = nn.Linear(4, hidden_dim)
            roi_pos_num = len(self.cfg['DATA']['CAMERA_NAMES']) * self.cfg['DATA']['ROI_RESIZE_SHAPE'][0] * self.cfg['DATA']['ROI_RESIZE_SHAPE'][1] // 14 // 14
            self.roi_pos = nn.Embedding(roi_pos_num, hidden_dim)

        if self.cfg['POLICY']['GRID_MASK'] == True:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

    def forward(self, image, past_action, end_obs, joint_obs, env_state, action = None, observation_is_pad = None, past_action_is_pad = None, action_is_pad = None, task_instruction_list = None, status = None):
        """
        image: (batch, num_cam, channel, height, width)
        past_action: (batch, past_action_len, action_dim)
        end_obs: (batch, past_obs_len, end_obs_dim)
        joint_obs: (batch, past_obs_len, joint_obs_dim)
        env_state: None
        action: (batch, chunk_size, action_dim)
        is_pad: (batch, chunk_size)
        task_instruction_list: A list with the length of batch, each element is a string.
        """
        is_training = action is not None 
        bs, num_cam, in_c, in_h, in_w = image.shape
        
        if self.cfg["POLICY"]["EXTERNAL_DET"] != 'None':
            assert self.cfg['POLICY']['BACKBONE'] == 'dinov2_s', "ROI image backbone only supports DINOv2 now!"
            roi_box, roi_img = self.object_detector(image, task_instruction_list, status)  
            _, _, num_detbox, _, roi_h, roi_w = roi_img.shape
            assert num_detbox == 1
            roi_img = roi_img.view(bs * num_cam * num_detbox, 3, roi_h, roi_w)
            roi_box = roi_box.view(bs, num_cam * num_detbox, 4)
            roi_box_emb = self.obj_box_mlp(roi_box) 
            
        
        image = image.view(bs * num_cam, in_c, in_h, in_w)  

        if self.cfg['POLICY']['BACKBONE'] == 'dinov2_s':
            if is_training and self.cfg['POLICY']['GRID_MASK'] == True:
                image = self.grid_mask(image)
            features = self.backbone.forward_features(image)['x_norm_patchtokens']  
            proj_features = self.input_proj(features)
            src = proj_features.view(bs, -1, self.hidden_dim).permute(1, 0, 2)   
            image_pos_embed = self.image_pos_embed.weight[None, None, :, :].expand(bs, num_cam, -1, -1)  
            pos = image_pos_embed.reshape(bs, -1, self.hidden_dim).permute(1, 0, 2)  

            if self.cfg["POLICY"]["EXTERNAL_DET"] != 'None':
                roi_feature = self.backbone.forward_features(roi_img)['x_norm_patchtokens']  
                proj_roi_feature = self.input_proj(roi_feature).view(bs, num_cam, -1, self.hidden_dim)   
                proj_roi_feature = proj_roi_feature + roi_box_emb[:, :, None] 
                proj_roi_feature = proj_roi_feature.view(bs, -1, self.hidden_dim).permute(1, 0, 2)  
                src = torch.cat((src, proj_roi_feature), dim = 0)  
                roi_pos = self.roi_pos.weight[None, :, :].expand(bs, -1, -1).permute(1, 0, 2)  
                pos = torch.cat((pos, roi_pos), axis = 0)  
        else:
            raise NotImplementedError
        mask = torch.zeros((bs, src.shape[0]), dtype=torch.bool).to(image.device)   

        
        if 'past_action' in self.cfg['DATA']['INPUT_KEYS']:
            past_action_src = self.past_action_mlp(past_action).permute(1, 0, 2)   
            past_action_pos = self.past_action_pos_emb.weight[:, None, :].expand(-1, bs, -1)  
            src = torch.cat((src, past_action_src), dim = 0)  
            pos = torch.cat((pos, past_action_pos), dim = 0)  
            mask = torch.cat((mask, past_action_is_pad), dim = 1) 
        if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS']:
            end_obs_src = self.end_obs_mlp(end_obs).permute(1, 0, 2)    
            end_obs_pos = self.obs_pos_emb.weight[:, None, :].expand(-1, bs, -1) + self.end_obs_pos_emb.weight[:, None, :]  
            src = torch.cat((src, end_obs_src), dim = 0)  
            pos = torch.cat((pos, end_obs_pos), dim = 0)  
            mask = torch.cat((mask, observation_is_pad), dim = 1) 
        if 'observations/joint_observation' in self.cfg['DATA']['INPUT_KEYS']:
            joint_obs_src = self.joint_obs_mlp(joint_obs).permute(1, 0, 2)    
            joint_obs_pos = self.obs_pos_emb.weight[:, None, :].expand(-1, bs, -1) + self.joint_obs_pos_emb.weight[:, None, :]  
            src = torch.cat((src, joint_obs_src), dim = 0)  
            pos = torch.cat((pos, joint_obs_pos), dim = 0)  
            mask = torch.cat((mask, observation_is_pad), dim = 1) 
    
        query_emb = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)   
        hs = self.transformer(src, mask, query_emb, pos) 
        if self.cfg['POLICY']['STATUS_PREDICT']:
            status_hs = hs[:, :, 0] 
            hs = hs[:, :, 1:]
            status_pred = self.status_head(status_hs)  
            if not is_training: status_pred = status_pred[-1].argmax(dim = -1)  
        else:
            status_pred = None
        
        if not is_training: hs = hs[-1] 

        a_hat = self.action_head(hs)    
        if self.cfg['POLICY']['OUTPUT_MODE'] == 'relative': 
            cur_status = end_obs[:, -1, :7][:, None, :] 
            if is_training: cur_status = cur_status[None]   
            a_hat[..., :7] = a_hat[..., :7] + cur_status
        elif self.cfg['POLICY']['OUTPUT_MODE'] == 'absolute':
            pass
        else:
            raise NotImplementedError
        if self.cfg['POLICY']['USE_UNCERTAINTY']:
            a_hat_uncern = self.uncern_head(hs) 
            a_hat_uncern = torch.clamp(a_hat_uncern, min = self.cfg['POLICY']['UNCERTAINTY_RANGE'][0], max = self.cfg['POLICY']['UNCERTAINTY_RANGE'][1])
        else:
            a_hat_uncern = None
        
        return a_hat, a_hat_uncern, status_pred
    
def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(cfg):
    d_model = cfg['POLICY']['HIDDEN_DIM'] 
    dropout = cfg['POLICY']['DROPOUT'] 
    nhead = cfg['POLICY']['NHEADS'] 
    dim_feedforward = cfg['POLICY']['DIM_FEEDFORWARD'] 
    num_encoder_layers = cfg['POLICY']['ENC_LAYERS'] 
    normalize_before = False 
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def get_VIRT_model(cfg):
    
    
    
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    model = VIRT(
        backbone,
        transformer,
        state_dim=cfg['POLICY']['STATE_DIM'],
        chunk_size=cfg['POLICY']['CHUNK_SIZE'],
        camera_names=cfg['DATA']['CAMERA_NAMES'],
        cfg = cfg,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

