"""
DETR model and criterion classes.
"""
import pdb
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer, CLIPTextModelWithProjection

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


class DETRVAE(nn.Module):
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
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)

        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.backbones = nn.ModuleList(backbones)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding
        
        if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS']:
            self.encoder_joint_proj = nn.Linear(13, hidden_dim)  # project qpos to embedding
            self.input_proj_robot_state = nn.Linear(13, hidden_dim)
        elif 'observations/qpos_obs' in self.cfg['DATA']['INPUT_KEYS']:
            self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            raise Exception('Not supported joint projection')
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+chunk_size, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        if self.cfg["POLICY"]["USE_CLIP"]:
            self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.clip_tokenizer = AutoTokenizer.from_pretrained(cfg["POLICY"]["CLIP_PATH"])

    def forward(self, qpos, image, task_instruction, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        task_instruction: bs
        actions: batch, seq, action_dim
        """
        
        is_training = actions is not None # train or val
        bs = image.shape[0]
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+2, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+2, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+2)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]   # Left shape: (B, latent_dim)
            logvar = latent_info[:, self.latent_dim:]   # Left shape: (B, latent_dim)
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)  # Left shape: (B, C)

        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
            features = features[0] # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)   # Left shape: (B, C, H, W)
        pos = torch.cat(all_cam_pos, axis=3)    # Left shape: (1, C, H, W)
        bs, ch, feat_h, feat_w =  src.shape
        src = src.permute(2, 3, 0, 1).reshape(feat_h * feat_w, bs, ch).contiguous() # Left shape: (L, B, C)
        pos = pos.permute(2, 3, 0, 1).reshape(feat_h * feat_w, 1, ch).expand(-1, bs, -1).contiguous() # Left shape: (L, B, C)

        src = torch.cat([latent_input[None], proprio_input[None], src], axis=0)  # Left shape: (L+2, B, C)
        pos = torch.cat([self.additional_pos_embed.weight[:, None].expand(-1, bs, -1), pos], axis=0)
        mask = torch.zeros((bs, src.shape[0]), dtype=torch.bool).to(src.device)

        if self.cfg["POLICY"]["USE_CLIP"]:
            text_tokens = self.clip_tokenizer(task_instruction, padding=True, return_tensors="pt").to(image.device)
            task_instruction_emb = self.clip_text_model(**text_tokens).text_embeds.detach()  # Left shape: (bs, clip_text_len)
            task_instruction_src = task_instruction_emb[None]   # Left shape: (1, B, C)
            src = src + task_instruction_src  # Left shape: (L, B, C)

        query_embed = self.query_embed.weight[:, None].expand(-1, bs, -1)   # Left shape: (num_query, B, C)
        hs = self.transformer(src, mask, query_embed, pos)
        hs = hs[-1] # Use the output of the last decoder
        
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names, cfg):
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
        self.camera_names = camera_names
        self.state_dim = state_dim
        self.chunk_size = self.cfg['POLICY']['CHUNK_SIZE']
        self.action_head = nn.Linear(2317, self.chunk_size * state_dim)

        self.backbones = nn.ModuleList(backbones)
        self.backbone_down_proj = nn.Sequential(
                nn.Conv2d(self.backbones[0].num_channels, 128, kernel_size=5),
                nn.Conv2d(128, 64, kernel_size=5),
                nn.Conv2d(64, 32, kernel_size=5)
            )

        if self.cfg["POLICY"]["USE_CLIP"]:
            self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.clip_tokenizer = AutoTokenizer.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.clip_mlp = nn.Linear(512, 1)

    def forward(self, qpos, image, task_instruction, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id])
            features = features[0] # take the last layer feature
            all_cam_features.append(self.backbone_down_proj(features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14

        if self.cfg["POLICY"]["USE_CLIP"]:
            text_tokens = self.clip_tokenizer(task_instruction, padding=True, return_tensors="pt").to(image.device)
            task_instruction_emb = self.clip_text_model(**text_tokens).text_embeds.detach()  # Left shape: (bs, clip_text_len)
            instruction_bias = self.clip_mlp(task_instruction_emb)
            features = features + instruction_bias
        
        a_hat = self.action_head(features)
        a_hat = a_hat.reshape(bs, self.chunk_size, self.state_dim)
        
        return a_hat


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


def get_ACT_model(cfg):
    backbones = []
    backbone = build_backbone(cfg)
    backbones.append(backbone)
    transformer = build_transformer(cfg)

    encoder = build_encoder(cfg)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=cfg['POLICY']['STATE_DIM'],
        chunk_size=cfg['POLICY']['CHUNK_SIZE'],
        camera_names=cfg['DATA']['CAMERA_NAMES'],
        cfg = cfg,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def get_CNNMLP_model(cfg):
    backbones = []
    backbone = build_backbone(cfg)
    backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=cfg['POLICY']['STATE_DIM'],
        camera_names=cfg['DATA']['CAMERA_NAMES'],
        cfg = cfg,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

