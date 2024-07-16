# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import pdb
import torch
from torch import nn
from torch.autograd import Variable
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.Tensor(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class IsaacGripperDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, chunk_size, camera_names, cfg):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.cfg = cfg
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder  # VAE encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim) # Decode transformer output as action.
        self.is_pad_head = nn.Linear(hidden_dim, 1) # Predict which output action is invalid.
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)

        # VAE Encoder
        if self.cfg['POLICY']['USE_VAE']:
            self.latent_dim = 32 # final size of latent z # TODO tune
            self.cls_embed = nn.Embedding(1, hidden_dim) # cls token for VAE.
            self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
            self.register_buffer('vae_pos_table', get_sinusoid_encoding_table(1 + chunk_size, hidden_dim)) # [CLS], qpos, a_seq
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # Project VAE latent vector to embdding for Transformer
            self.vae_pos_embed = nn.Embedding(1, hidden_dim) # learned position embedding for vae mu and std

        # Camera image feature extraction.
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.backbones = nn.ModuleList(backbones)
        self.camera_view_pos_embed = nn.Embedding(len(camera_names), hidden_dim)

        # Proprioception information encoding.
        if 'past_action' in self.cfg['DATA']['INPUT_KEYS']:
            self.past_action_mlp = nn.Linear(9, hidden_dim)  # Past action information encoding
            self.past_action_pos_emb = nn.Embedding(self.cfg['DATA']['PAST_ACTION_LEN'], hidden_dim)
        if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS'] or 'observations/joint_observation' in self.cfg['DATA']['INPUT_KEYS']:
            self.obs_pos_emb = nn.Embedding(self.cfg['DATA']['PAST_OBSERVATION_LEN'], hidden_dim)
            if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS']:
                self.end_obs_mlp = nn.Linear(13, hidden_dim) # End observation information encoding
                self.end_obs_pos_emb = nn.Embedding(1, hidden_dim)
            if 'observations/joint_observation' in self.cfg['DATA']['INPUT_KEYS']:
                self.joint_obs_pos_emb = nn.Embedding(1, hidden_dim)
                self.joint_obs_mlp = nn.Linear(9, hidden_dim) # Joint observation information encoding

        if self.cfg["POLICY"]["USE_CLIP"]:
            self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.clip_tokenizer = AutoTokenizer.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.task_instruction_pos_emb = nn.Embedding(1, hidden_dim)

    def forward(self, image, past_action, end_obs, joint_obs, env_state, action = None, observation_is_pad = None, past_action_is_pad = None, action_is_pad = None, task_instruction_list = None):
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
        is_training = action is not None # train or val
        bs = image.shape[0]

        if self.cfg["POLICY"]["USE_CLIP"]:
            text_tokens = self.clip_tokenizer(task_instruction_list, padding=True, return_tensors="pt").to(image.device)
            with torch.no_grad():
                task_instruction_emb = self.clip_text_model(**text_tokens).text_embeds.detach()  # Left shape: (B, clip_text_len)

        ### Obtain latent z from action sequence
        if self.cfg['POLICY']['USE_VAE'] and is_training: # VAE encoder
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(action) # (bs, seq, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 1), False).to(image.device) # False: not a padding
            vae_is_pad = torch.cat([cls_joint_is_pad, action_is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.vae_pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=vae_is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]   # Left shape: (B, latent_dim)
            logvar = latent_info[:, self.latent_dim:]   # Left shape: (B, latent_dim)
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)[None]  # Left shape: (B, hidden_dim)

            vae_pos = self.vae_pos_embed.weight[:, None, :].expand(-1, bs, -1)  # Left shape: (1, B, C)
            vae_mask = torch.zeros((bs, 1), dtype=torch.bool).to(image.device)  # Left shape: (B, 1)
        elif self.cfg['POLICY']['USE_VAE'] and not is_training:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(image.device)
            latent_input = self.latent_out_proj(latent_sample)[None]    # Left shape: (1, B, C)

            vae_pos = self.vae_pos_embed.weight[:, None, :].expand(-1, bs, -1)  # Left shape: (1, B, C)
            vae_mask = torch.zeros((bs, 1), dtype=torch.bool).to(image.device)  # Left shape: (B, 1)
        else:
            mu = logvar = None
        
        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
            features = features[0] # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        img_src = torch.stack(all_cam_features, axis=2)   # Left shape: (B, C, N, H, W)
        cam_pos = torch.stack(all_cam_pos, axis=2)  # Left shape: (1, C, N, H, W)
        camera_view_pos = self.camera_view_pos_embed.weight.permute(1, 0)[None, :, :, None, None]  # (1, C, N, 1, 1)
        cam_pos = (cam_pos + camera_view_pos).expand(bs, -1, -1, -1, -1)  # (B, C, N, H, W)
        src = img_src.flatten(2).permute(2, 0, 1)   # Left shape: (NHW, B, C)
        pos = cam_pos.flatten(2).permute(2, 0, 1)  # Left shape: (NHW, B, C)
        mask = torch.zeros((bs, src.shape[0]), dtype=torch.bool).to(image.device)   # Left shape: (B, NHW)

        # proprioception features
        if 'past_action' in self.cfg['DATA']['INPUT_KEYS']:
            past_action_src = self.past_action_mlp(past_action).permute(1, 0, 2)   # (past_action_len, B, C)
            past_action_pos = self.past_action_pos_emb.weight[:, None, :].expand(-1, bs, -1)  # (past_action_len, B, C)
            src = torch.cat((src, past_action_src), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, past_action_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, past_action_is_pad), dim = 1) # Left shape: (B, L)
        if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS']:
            end_obs_src = self.end_obs_mlp(end_obs).permute(1, 0, 2)    # (past_obs_len, B, C)
            end_obs_pos = self.obs_pos_emb.weight[:, None, :].expand(-1, bs, -1) + self.end_obs_pos_emb.weight[:, None, :]  # (past_obs_len, B, C)
            src = torch.cat((src, end_obs_src), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, end_obs_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, observation_is_pad), dim = 1) # Left shape: (B, L)
        if 'observations/joint_observation' in self.cfg['DATA']['INPUT_KEYS']:
            joint_obs_src = self.joint_obs_mlp(joint_obs).permute(1, 0, 2)    # (past_obs_len, B, C)
            joint_obs_pos = self.obs_pos_emb.weight[:, None, :].expand(-1, bs, -1) + self.joint_obs_pos_emb.weight[:, None, :]  # (past_obs_len, B, C)
            src = torch.cat((src, joint_obs_src), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, joint_obs_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, observation_is_pad), dim = 1) # Left shape: (B, L)

        if self.cfg['POLICY']['USE_VAE']:
            src = torch.cat((src, latent_input), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, vae_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, vae_mask), dim = 1) # Left shape: (B, L)
        if self.cfg["POLICY"]["USE_CLIP"]:
            task_instruction_src = task_instruction_emb[None]   # Left shape: (1, B, C)
            task_instruction_pos = self.task_instruction_pos_emb.weight[:, None, :].expand(-1, bs, -1)   # Left shape: (1, B, C)
            task_instruction_mask = torch.zeros((bs, task_instruction_pos.shape[0]), dtype=torch.bool).to(image.device) # Left shape: (B, 1)
            src = torch.cat((src, task_instruction_src), dim = 0)  # Left shape: (L, B, C)
            pos = torch.cat((pos, task_instruction_pos), dim = 0)  # Left shape: (L, B, C)
            mask = torch.cat((mask, task_instruction_mask), dim = 1) # Left shape: (B, L)
    
        query_emb = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)   # Left shape: (num_query, B, C)
        hs = self.transformer(src, mask, query_emb, pos)[0] # Left shape: (B, num_query, C)

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

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


def get_IsaacGripper_ACT_model(cfg):
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(cfg)
    backbones.append(backbone)
    transformer = build_transformer(cfg)

    if cfg['POLICY']['USE_VAE']:
        vae_encoder = build_encoder(cfg)    # VAE encoder
    else:
        vae_encoder = None

    model = IsaacGripperDETR(
        backbones,
        transformer,
        vae_encoder,
        state_dim=cfg['POLICY']['STATE_DIM'],
        chunk_size=cfg['POLICY']['CHUNK_SIZE'],
        camera_names=cfg['DATA']['CAMERA_NAMES'],
        cfg = cfg,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

