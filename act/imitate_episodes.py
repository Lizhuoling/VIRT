import numpy as np
import os
import pdb
import time
import pickle
import datetime
import logging
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import IPython
e = IPython.embed

def main(args):
    # Initialize logger
    if comm.get_rank() == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    exp_start_time = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    rank = comm.get_rank()
    logger = setup_logger(args.save_dir, rank, file_name="log_{}.txt".format(exp_start_time))
    if comm.is_main_process():
        logger.info("Using {} GPUs".format(comm.get_world_size()))
        logger.info("Collecting environment info")
        logger.info(args) 
        logger.info("Loaded configuration file {}".format(args.config_name+'.yaml'))
    
    # Initialize cfg
    cfg = load_yaml_with_base(os.path.join('configs', args.config_name+'.yaml'))
    cfg['IS_EVAL'] = args.eval
    cfg['CKPT_DIR'] = args.save_dir
    cfg['DATASET_DIR'] = args.data_dir
    cfg['ONSCREEN_RENDER'] = args.onscreen_render
    cfg['IS_DEBUG'] = args.debug
    cfg['NUM_NODES'] = args.num_nodes
    cfg['EVAL']['REAL_ROBOT'] = args.real_robot
    
    if cfg['SEED'] >= 0:
        set_seed(cfg['SEED'])
    if cfg['IS_DEBUG']:
        cfg['TRAIN']['BATCH_SIZE'] = 2

    if cfg['IS_EVAL']:
        if args.load_dir != '':
            ckpt_paths = [args.load_dir]
        else:
            ckpt_paths = [os.path.join(cfg['CKPT_DIR'], 'policy_latest.ckpt')]
        results = []
        for ckpt_path in ckpt_paths:
            success_rate, avg_return = eval_bc(cfg, ckpt_path, save_episode=args.save_episode)
            results.append([ckpt_path.split('/')[-1], success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        exit()
    
    train_dataloader, val_dataloader, stats, = load_data(cfg)
    
    # save dataset stats
    if not os.path.isdir(cfg['CKPT_DIR']):
        os.makedirs(cfg['CKPT_DIR'])
    stats_path = os.path.join(cfg['CKPT_DIR'], f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    train_bc(train_dataloader, val_dataloader, cfg, load_dir = args.load_dir, load_pretrain = args.load_pretrain)

def make_policy(policy_class, cfg):
    if policy_class == 'ACT':
        policy = ACTPolicy(cfg)
    elif policy_class == 'IsaacGripper_ACT':
        policy = IsaacGripper_ACTPolicy(cfg)
    elif policy_class == 'DroidPretrain_ACT':
        policy = DroidPretrain_ACTPolicy(cfg)
    elif policy_class == 'AlohaGripper_ACT':
        policy = AlohaGripper_ACTPolicy(cfg)
    else:
        raise NotImplementedError

    return policy

def inference_get_images(ts, camera_names, inference_transforms):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image).float()
    curr_image, _, _, _ = inference_transforms(curr_image, None, None, None)
    return curr_image.cuda().unsqueeze(0)


def eval_bc(cfg, ckpt_path, save_episode=True):
    ckpt_dir = cfg['CKPT_DIR']
    ckpt_name = ckpt_path.split('/')[-1]
    policy_class = cfg['POLICY']['POLICY_NAME']
    
    # load policy and stats
    policy = make_policy(policy_class, cfg)
    loading_status = policy.load_state_dict(torch.load(ckpt_path)['model'], strict = True)
    print(loading_status)
    policy.cuda()
    policy.eval()
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if cfg['TASK_NAME'] == 'isaac_multicolorbox':
        from utils.inference.isaac_multicolorbox import IsaacMultiColorBoxTestEnviManager
        envi_manager = IsaacMultiColorBoxTestEnviManager(cfg, policy, stats)
    elif cfg['TASK_NAME'] == 'isaac_singlebox':
        from utils.inference.isaac_singlebox import IsaacSingleBoxTestEnviManager
        envi_manager = IsaacSingleBoxTestEnviManager(cfg, policy, stats)
    elif cfg['TASK_NAME'] == 'isaac_singlecolorbox':
        from utils.inference.isaac_singlecolorbox import IsaacSingleColorBoxTestEnviManager
        envi_manager = IsaacSingleColorBoxTestEnviManager(cfg, policy, stats)
    elif cfg['TASK_NAME'] == 'isaac_twoboxred':
        from utils.inference.isaac_twoboxred import IsaacTwoBoxRedTestEnviManager
        envi_manager = IsaacTwoBoxRedTestEnviManager(cfg, policy, stats)
    elif cfg['TASK_NAME'] == 'isaac_fiveboxred':
        from utils.inference.isaac_fiveboxred import IsaacFiveBoxRedTestEnviManager
        envi_manager = IsaacFiveBoxRedTestEnviManager(cfg, policy, stats)
    elif cfg['TASK_NAME'] in ['aloha_singleobjgrasp', 'aloha_beverage']:
        from utils.inference.aloha_singleobjgrasp import AlohaSingleObjGraspTestEnviManager
        envi_manager = AlohaSingleObjGraspTestEnviManager(cfg, policy, stats)
    else:
        raise NotImplementedError

    reward_info = envi_manager.inference()

    if reward_info != None:
        success_rate = reward_info['success_rate']
        avg_return = reward_info['average_reward']
        summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'

        print(summary_str)

        # save success rate to txt
        result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)

        return success_rate, avg_return
    else:
        return 0.0, 0.0


def forward_pass(data, policy, cfg):
    if cfg['POLICY']['POLICY_NAME'] == 'ACT':
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        return policy(qpos_data, image_data, action_data, is_pad)
    elif cfg['POLICY']['POLICY_NAME'] == 'IsaacGripper_ACT':
        image_data, past_action, action_data, end_observation, joint_observation, observation_is_pad, past_action_is_pad, action_is_pad, task_instruction_list, status = data
        
        image_data, past_action, action_data, end_observation, joint_observation, observation_is_pad, past_action_is_pad, action_is_pad, status = image_data.cuda(), past_action.cuda(), action_data.cuda(), \
            end_observation.cuda(), joint_observation.cuda(), observation_is_pad.cuda(), past_action_is_pad.cuda(), action_is_pad.cuda(), status.cuda()
        
        return policy(image = image_data, past_action = past_action, end_obs = end_observation, joint_obs = joint_observation, action = action_data, observation_is_pad = observation_is_pad, \
                      past_action_is_pad = past_action_is_pad, action_is_pad = action_is_pad, task_instruction_list = task_instruction_list, status = status)
    elif cfg['POLICY']['POLICY_NAME'] == 'DroidPretrain_ACT':
        image_data, goal_image_data, past_action, action_data, end_observation, joint_observation, observation_is_pad, past_action_is_pad, action_is_pad, task_instruction_list = data
        image_data, goal_image_data, past_action, action_data, end_observation, joint_observation, observation_is_pad, past_action_is_pad, action_is_pad = \
            image_data.cuda(), goal_image_data.cuda(), past_action.cuda(), action_data.cuda(), end_observation.cuda(), joint_observation.cuda(), observation_is_pad.cuda(), past_action_is_pad.cuda(), action_is_pad.cuda()
        
        return policy(image = image_data, goal_image = goal_image_data, past_action = past_action, action = action_data, end_obs = end_observation, joint_obs = end_observation,\
                    observation_is_pad = observation_is_pad, past_action_is_pad = past_action_is_pad, action_is_pad = action_is_pad, task_instruction_list = task_instruction_list)
    elif cfg['POLICY']['POLICY_NAME'] == 'AlohaGripper_ACT':
        image_data, past_action, action_data, effort_obs, qpos_obs, qvel_obs, observation_is_pad, past_action_is_pad, action_is_pad, task_instruction, status = data
        image_data, past_action, action_data, effort_obs, qpos_obs, qvel_obs, observation_is_pad, past_action_is_pad, action_is_pad, status = image_data.cuda(), past_action.cuda(), \
            action_data.cuda(), effort_obs.cuda(), qpos_obs.cuda(), qvel_obs.cuda(), observation_is_pad.cuda(), past_action_is_pad.cuda(), action_is_pad.cuda(), status.cuda()
        
        return policy(image = image_data, past_action = past_action, action = action_data, effort_obs = effort_obs, qpos_obs = qpos_obs, qvel_obs = qvel_obs, observation_is_pad = observation_is_pad, \
                      past_action_is_pad = past_action_is_pad, action_is_pad = action_is_pad, task_instruction = task_instruction, status = status)

def train_bc(train_dataloader, val_dataloader, cfg, load_dir = '', load_pretrain = ''):
    logger = logging.getLogger("grasp")

    num_iterations = cfg['TRAIN']['NUM_ITERATIONS']
    ckpt_dir = cfg['CKPT_DIR']
    seed = cfg['SEED']
    policy_class = cfg['POLICY']['POLICY_NAME']

    policy = make_policy(policy_class, cfg)
    if load_pretrain != '':
        load_dict = torch.load(load_pretrain)['model']
        filter_dict = {key:value for key, value in load_dict.items() if 'model.backbone' in key or 'model.transformer' in key}
        loading_status = policy.load_state_dict(filter_dict, strict = False)
    if load_dir != '':
        load_dict = torch.load(load_dir)
        loading_status = policy.load_state_dict(load_dict['model'], strict = True)
        start_iter = load_dict['iter']
    else:
        start_iter = 0
    policy.cuda()
    optimizer = make_optimizer(policy, cfg)
    scheduler, warmup_scheduler = make_scheduler(optimizer, cfg=cfg)
    
    main_thread = comm.get_rank() == 0
    if main_thread:
        tb_writer = SummaryWriter(os.path.join(ckpt_dir, 'tb/{}/'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))))

    if cfg['TRAIN']['LR_WARMUP']:
        assert warmup_scheduler is not None
        warmup_iters = cfg['TRAIN']['WARMUP_STEPS']
    else:
        warmup_iters = -1

    min_val_loss = np.inf
    end = time.time()
    train_meters = MetricLogger(delimiter=", ", )
    
    for data, iter_cnt in zip(train_dataloader, range(start_iter, num_iterations)):
        data_time = time.time() - end

        # training
        policy.train()
        optimizer.zero_grad()
        total_loss, loss_dict = forward_pass(data, policy, cfg)
        # backward
        total_loss.backward()
        optimizer.step()
        if iter_cnt < warmup_iters:
            warmup_scheduler.step(iter_cnt)
        else:
            scheduler.step(iter_cnt)

        train_meters.update(**loss_dict)
        batch_time = time.time() - end
        end = time.time()
        train_meters.update(time=batch_time, data=data_time)
        eta_seconds = train_meters.time.global_avg * (num_iterations - iter_cnt)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # log
        if main_thread and (iter_cnt % cfg['TRAIN']['LOG_INTERVAL'] == 0 or iter_cnt == num_iterations - 1):
            logger.info(
                train_meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f} \n",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iter_cnt,
                    meters=str(train_meters),
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

            tb_writer.add_scalar('loss/total_loss', total_loss.item(), iter_cnt)
            for dec_id in range(cfg['POLICY']['DEC_LAYERS']):
                tb_writer.add_scalar(f'loss/dec_{dec_id}', loss_dict[f'dec_{dec_id}'], iter_cnt)
            if cfg['POLICY']['USE_VAE']:
                tb_writer.add_scalar(f'loss/vae_loss', loss_dict['vae_loss'], iter_cnt)
            if cfg['POLICY']['USE_UNCERTAINTY']:
                tb_writer.add_scalar(f'loss/loss_without_uncern', loss_dict['loss_without_uncern'], iter_cnt)
            tb_writer.add_scalar('state/lr', optimizer.param_groups[0]["lr"], iter_cnt)

        # validation
        if cfg['EVAL']['DATA_EVAL_RATIO'] > 0 and main_thread and iter_cnt % cfg['EVAL']['EVAL_INTERVAL'] == 0 and iter_cnt != 0:
            logger.info("Start evaluation at iteration {}...".format(iter_cnt))
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []

                eval_total_iter_num = len(val_dataloader)
                if eval_total_iter_num > cfg['EVAL']['MAX_VAL_SAMPLE_NUM']:
                    eval_total_iter_num = cfg['EVAL']['MAX_VAL_SAMPLE_NUM']

                for eval_data, eval_iter_cnt in zip(val_dataloader, range(eval_total_iter_num)):
                    total_loss, loss_dict = forward_pass(eval_data, policy, cfg)
                    epoch_dicts.append(loss_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (iter_cnt, min_val_loss, deepcopy(policy.state_dict()))
            summary_string = 'Evaluation result:'
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            logger.info(summary_string)

        # Save checkpoint
        if main_thread and iter_cnt % cfg['TRAIN']['SAVE_CHECKPOINT_INTERVAL'] == 0 and iter_cnt != 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_latest.ckpt')
            if os.path.exists(ckpt_path):
                os.rename(ckpt_path, os.path.join(ckpt_dir, f'policy_previous.ckpt'))
            save_model(policy, ckpt_path, iter_cnt)
            
    if main_thread:
        ckpt_path = os.path.join(ckpt_dir, f'policy_latest.ckpt')
        save_model(policy, ckpt_path, iter_cnt)
        logger.info(f'Training finished!')

    comm.synchronize()
    tb_writer.close()
    for handler in logging.root.handlers:
        handler.close()

def save_model(model, save_path, iter_cnt):
    model_ckpt = model.state_dict()
    save_dict = {
        'iter': iter_cnt,
        'model': model_ckpt
    }
    torch.save(save_dict, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', action='store', type=str, help='configuration file name', required=True)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_dir', action='store', type=str, help='saving directory', required=True)
    parser.add_argument('--load_dir', action='store', type=str, default = '', help='The path to weight',)
    parser.add_argument('--load_pretrain', action='store', type=str, default = '', help='The path to pre-trained weight')
    parser.add_argument('--data_dir', action='store', type=str, help='dataset folder path')
    parser.add_argument('--real_robot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_episode', action='store_true')

    parser.add_argument('--num_nodes', default = 1, type = int, help = "The number of nodes.")
    
    args = parser.parse_args()

    if not args.real_robot:
        from isaacgym import gymapi
        from isaacgym import gymutil
        from isaacgym import gymtorch
        from isaacgym.torch_utils import quat_rotate, quat_conjugate, quat_mul
    import torch    # torch must be imported after isaacgym
    from torch.utils.tensorboard import SummaryWriter

    from constants import DT
    from constants import PUPPET_GRIPPER_JOINT_OPEN
    from utils.dataset_utils import load_data # data functions
    from utils.dataset_utils import sample_box_pose, sample_insertion_pose # robot functions
    from utils.dataset_utils import compute_dict_mean, detach_dict # helper functions
    from utils.utils import set_seed
    from utils.engine import launch
    from utils import comm
    from utils.optimizer import make_optimizer, make_scheduler
    from utils.check_point import DetectronCheckpointer
    from utils.metric_logger import MetricLogger
    from utils.logger import setup_logger
    from utils.models.ACT_policy import ACTPolicy
    from utils.models.IsaacGripper_ACTPolicy import IsaacGripper_ACTPolicy
    from utils.models.DroidPretrain_ACTPolicy import DroidPretrain_ACTPolicy
    from utils.models.AlohaGripper_ACTPolicy import AlohaGripper_ACTPolicy
    from visualize_episodes import save_videos
    from configs.utils import load_yaml_with_base

    launch(main, args)
