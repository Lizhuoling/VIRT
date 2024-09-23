import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
import numpy as np

from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

class DiffusionPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.camera_names = cfg['DATA']['CAMERA_NAMES']

        self.observation_horizon = 1
        self.action_horizon = 8
        self.prediction_horizon = cfg['POLICY']['CHUNK_SIZE']
        self.num_inference_timesteps = 10
        self.ema_power = 0.75

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = self.cfg['POLICY']['STATE_DIM']
        if 'observations/end_observation' in self.cfg['DATA']['INPUT_KEYS']:
            self.qpos_dim = 13
        elif 'observations/qpos_obs' in self.cfg['DATA']['INPUT_KEYS']:
            self.qpos_dim = 14
        self.obs_dim = self.feature_dimension * len(self.camera_names) + self.qpos_dim # camera features and proprio


        backbone = ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False})
        pool = SpatialSoftmax(**{'input_shape': [512, 8, 10], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0})
        linear = torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension)
        backbone = replace_bn_with_gn(backbone)
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbone': backbone,
                'pool': pool,
                'linear': linear,
                'noise_pred_net': noise_pred_net
            })
        })
        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        if self.cfg["POLICY"]["USE_CLIP"]:
            self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.clip_tokenizer = AutoTokenizer.from_pretrained(cfg["POLICY"]["CLIP_PATH"])
            self.clip_mlp = nn.Linear(512, 1)
        
        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))
    
    def __call__(self, qpos, image, actions=None, is_pad=None, task_instruction = None):
        '''
        Input:
            qpos shape: (B, qpos_dim)
            image shape: (B, num_cam, 3, img_h, img_w)
            actions shape: (B, preidiction_horizon, action_dim)
            is_pad shape: (B, preidiction_horizon)
        '''
        B = qpos.shape[0]

        if self.cfg["POLICY"]["USE_CLIP"]:
            text_tokens = self.clip_tokenizer(task_instruction, padding=True, return_tensors="pt").to(image.device)
            task_instruction_emb = self.clip_text_model(**text_tokens).text_embeds.detach()  # Left shape: (bs, clip_text_len)
            instruction_token = self.clip_mlp(task_instruction_emb) # Left shape: (bs, 1)

        if actions is not None: # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbone'](cam_image)
                pool_features = nets['policy']['pool'](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linear'](pool_features)
                all_features.append(out_features)
            
            obs_cond = torch.cat(all_features + [qpos], dim=1)  # Left shape: (B, C)
            if self.cfg["POLICY"]["USE_CLIP"]:
                obs_cond = obs_cond + instruction_token

            noise = torch.randn(actions.shape, device=obs_cond.device)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()

            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            total_loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict = dict(loss = total_loss.item())
            
            return total_loss, loss_dict
        
        else: # inference time
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model
            
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbone'](cam_image)
                pool_features = nets['policy']['pool'](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linear'](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction, None