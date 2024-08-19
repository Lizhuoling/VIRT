# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model
import and criterion classes.
"""
import pdb
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from detr.util.grid_mask import GridMask

import numpy as np

import IPython
e = IPython.embed

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DroidPretrainDETR(nn.Module):
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
        if self.cfg['TRAIN']['FEATURE_REGULARIZATION']:
            self.init_backbone = [copy.deepcopy(backbone).cuda()]  # The list prevents the copied backbone from being registered.
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        query_num = chunk_size
        self.query_embed = nn.Embedding(query_num, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, state_dim) # Decode transformer output as action.
        if self.cfg['POLICY']['USE_UNCERTAINTY']:
            self.uncern_head = nn.Linear(hidden_dim, 1)

        # Camera image feature extraction.
        if self.cfg['POLICY']['BACKBONE'] == 'dinov2_s':
            self.input_proj = nn.Linear(backbone.num_features, hidden_dim)

            img_token_len = self.cfg['DATA']['IMG_RESIZE_SHAPE'][1] * self.cfg['DATA']['IMG_RESIZE_SHAPE'][0] // 14 // 14
            self.image_pos_embed = nn.Embedding(img_token_len, hidden_dim)

            goal_token_len = self.cfg['DATA']['ROI_RESIZE_SHAPE'][1] * self.cfg['DATA']['ROI_RESIZE_SHAPE'][0] // 14 // 14
            self.goal_pos_embed = nn.Embedding(goal_token_len, hidden_dim)
        else:
            raise NotImplementedError
        
        # Proprioception information encoding.
        if 'past_action' in self.cfg['DATA']['INPUT_KEYS']:
            self.past_action_mlp = nn.Linear(7, hidden_dim)  # Past action information encoding
            self.past_action_pos_emb = nn.Embedding(self.cfg['DATA']['PAST_ACTION_LEN'], hidden_dim)
        if 'observation/cartesian_position' in self.cfg['DATA']['INPUT_KEYS'] or 'observation/gripper_position' in self.cfg['DATA']['INPUT_KEYS'] or 'observation/joint_position' in self.cfg['DATA']['INPUT_KEYS']:
            self.obs_pos_emb = nn.Embedding(self.cfg['DATA']['PAST_OBSERVATION_LEN'], hidden_dim)
            if 'observation/cartesian_position' in self.cfg['DATA']['INPUT_KEYS'] or 'observation/gripper_position' in self.cfg['DATA']['INPUT_KEYS']:
                assert 'observation/cartesian_position' in self.cfg['DATA']['INPUT_KEYS'] and 'observation/gripper_position' in self.cfg['DATA']['INPUT_KEYS']
                self.end_obs_mlp = nn.Linear(7, hidden_dim) # End observation information encoding
                self.end_obs_pos_emb = nn.Embedding(1, hidden_dim)
            if 'observation/joint_position' in self.cfg['DATA']['INPUT_KEYS']:
                self.joint_obs_pos_emb = nn.Embedding(1, hidden_dim)
                self.joint_obs_mlp = nn.Linear(7, hidden_dim) # Joint observation information encoding

        if self.cfg["POLICY"]["USE_CLIP"]:
            self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.clip_tokenizer = AutoTokenizer.from_pretrained(cfg["POLICY"]["CLIP_PATH"])

        if self.cfg['POLICY']['GRID_MASK'] == True:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

    def forward(self, image, goal_image, past_action, action, end_obs, joint_obs, observation_is_pad, past_action_is_pad, action_is_pad, task_instruction_list):
        """
        image: (batch, num_cam, channel, height, width)
        goal_image: (batch, num_cam, channel, height, width)
        past_action: (batch, past_action_len, action_dim)
        action: (batch, chunk_size, action_dim)
        end_obs: (batch, past_obs_len, end_obs_dim)
        joint_obs: (batch, past_obs_len, joint_obs_dim)
        is_pad: (batch, chunk_size)
        task_instruction_list: A list with the length of batch, each element is a string.
        """
        is_training = action is not None # train or val
        bs, num_cam, in_c, in_h, in_w = image.shape
        _, _, _, goal_h, goal_w = goal_image.shape
            
        # Image observation features and position embeddings
        image = image.view(bs * num_cam, in_c, in_h, in_w)  # Left shape: (bs * num_cam, C, H, W)
        goal_image = goal_image.view(bs * num_cam, in_c, goal_h, goal_w)  # Left shape: (bs * num_cam, C, goal_h, goal_w)
        
        if self.cfg['POLICY']['BACKBONE'] == 'dinov2_s':
            if is_training and self.cfg['POLICY']['GRID_MASK'] == True:
                image = self.grid_mask(image)
                goal_image = self.grid_mask(goal_image)
            if self.cfg['TRAIN']['LR_BACKBONE'] > 0:
                features = self.backbone.forward_features(image)['x_norm_patchtokens']  # Left shape: (bs * num_cam, l, C)
                goal_feature = self.backbone.forward_features(goal_image)['x_norm_patchtokens']  # Left shape: (bs * num_cam, goal_l, C)
            else:
                with torch.no_grad():
                    features = self.backbone.forward_features(image)['x_norm_patchtokens']  # Left shape: (bs * num_cam, l, C)
                    goal_feature = self.backbone.forward_features(goal_image)['x_norm_patchtokens']  # Left shape: (bs * num_cam, goal_l, C)

            if self.cfg['TRAIN']['FEATURE_REGULARIZATION']:
                with torch.no_grad():
                    init_features = self.init_backbone[0].forward_features(image)['x_norm_patchtokens'] # Left shape: (bs * num_cam, l, C)
                    init_goal_feature = self.init_backbone[0].forward_features(goal_image)['x_norm_patchtokens']  # Left shape: (bs * num_cam, goal_l, C)
                feat_regu_loss = F.l1_loss(features, init_features, reduction='none').mean() + F.l1_loss(goal_feature, init_goal_feature, reduction='none').mean()
            else:
                feat_regu_loss = None

            features = self.input_proj(features)
            src = features.view(bs, -1, self.hidden_dim).permute(1, 0, 2)   # Left shape: (num_cam * l, B, C)
            image_pos_embed = self.image_pos_embed.weight[None, None, :, :].expand(bs, num_cam, -1, -1)  # (B, num_cam, l, C)
            pos = image_pos_embed.reshape(bs, -1, self.hidden_dim).permute(1, 0, 2)  # Left shape: (num_cam * l, B, C)

            goal_feature = self.input_proj(goal_feature)   # Left shape: (bs * num_cam, goal_l, C)
            goal_feature = goal_feature.view(bs, -1, self.hidden_dim).permute(1, 0, 2)  # Left shape: (num_cam * goal_l, B, C)
            goal_pos_embed = self.goal_pos_embed.weight[None, None, :, :].expand(bs, num_cam, -1, -1)  # (B, num_cam, goal_l, C)
            goal_pos = goal_pos_embed.reshape(bs, -1, self.hidden_dim).permute(1, 0, 2)  # Left shape: (num_cam * goal_l, B, C)

            src = torch.cat((src, goal_feature), dim = 0)  # Left shape: (l, B, C)
            pos = torch.cat((pos, goal_pos), axis = 0)  # Left shape: (num_cam * l, B, C)
        else:
            raise NotImplementedError
        mask = torch.zeros((bs, src.shape[0]), dtype=torch.bool).to(image.device)   # Left shape: (B, NHW)

        if self.cfg["POLICY"]["USE_CLIP"]:
            text_tokens = self.clip_tokenizer(task_instruction_list, padding=True, return_tensors="pt").to(image.device)
            task_instruction_emb = self.clip_text_model(**text_tokens).text_embeds.detach()  # Left shape: (bs, clip_text_len)
            task_instruction_src = task_instruction_emb[None]   # Left shape: (1, B, C)
            src = src + task_instruction_src  # Left shape: (L, B, C)
        
        # proprioception features
        if 'past_action' in self.cfg['DATA']['INPUT_KEYS']:
            past_action_src = self.past_action_mlp(past_action).permute(1, 0, 2)   # (past_action_len, B, C)
            past_action_pos = self.past_action_pos_emb.weight[:, None, :].expand(-1, bs, -1)  # (past_action_len, B, C)
            src = torch.cat((src, past_action_src), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, past_action_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, past_action_is_pad), dim = 1) # Left shape: (B, L)
        if 'observation/cartesian_position' in self.cfg['DATA']['INPUT_KEYS'] or 'observation/gripper_position' in self.cfg['DATA']['INPUT_KEYS']:
            end_obs_src = self.end_obs_mlp(end_obs).permute(1, 0, 2)    # (past_obs_len, B, C)
            end_obs_pos = self.obs_pos_emb.weight[:, None, :].expand(-1, bs, -1) + self.end_obs_pos_emb.weight[:, None, :]  # (past_obs_len, B, C)
            src = torch.cat((src, end_obs_src), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, end_obs_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, observation_is_pad), dim = 1) # Left shape: (B, L)
        if 'observation/joint_position' in self.cfg['DATA']['INPUT_KEYS']:
            joint_obs_src = self.joint_obs_mlp(joint_obs).permute(1, 0, 2)    # (past_obs_len, B, C)
            joint_obs_pos = self.obs_pos_emb.weight[:, None, :].expand(-1, bs, -1) + self.joint_obs_pos_emb.weight[:, None, :]  # (past_obs_len, B, C)
            src = torch.cat((src, joint_obs_src), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, joint_obs_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, observation_is_pad), dim = 1) # Left shape: (B, L)
    
        query_emb = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)   # Left shape: (num_query, B, C)
        hs = self.transformer(src, mask, query_emb, pos) # Left shape: (num_dec, B, num_query, C)
        
        if not is_training: hs = hs[-1] # Left shape: (B, num_query, C)

        a_hat = self.action_head(hs)    # left shape: (num_dec, B, num_query, action_dim)
        if self.cfg['POLICY']['OUTPUT_MODE'] == 'relative': # Only the robotic arm joint is controlled with relative signal, the gripper is still controlled absolutely.
            cur_status = end_obs[:, -1, :6][:, None, :] # left shape: (B, 1, action_dim)
            if is_training: cur_status = cur_status[None]   # left shape: (1, B, 1, action_dim)
            a_hat[..., :6] = a_hat[..., :6] + cur_status
        elif self.cfg['POLICY']['OUTPUT_MODE'] == 'absolute':
            pass
        else:
            raise NotImplementedError
        if self.cfg['POLICY']['USE_UNCERTAINTY']:
            a_hat_uncern = self.uncern_head(hs) # left shape: (num_dec, B, num_query, 1)
            a_hat_uncern = torch.clamp(a_hat_uncern, min = self.cfg['POLICY']['UNCERTAINTY_RANGE'][0], max = self.cfg['POLICY']['UNCERTAINTY_RANGE'][1])
        else:
            a_hat_uncern = None
        
        return a_hat, a_hat_uncern, feat_regu_loss
    
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
    d_model = cfg['POLICY']['HIDDEN_DIM'] # 256
    dropout = cfg['POLICY']['DROPOUT'] # 0.1
    nhead = cfg['POLICY']['NHEADS'] # 8
    dim_feedforward = cfg['POLICY']['DIM_FEEDFORWARD'] # 2048
    num_encoder_layers = cfg['POLICY']['ENC_LAYERS'] # 4 # TODO shared with VAE decoder
    normalize_before = False # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def get_DroidPretrain_ACT_model(cfg):
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    model = DroidPretrainDETR(
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

