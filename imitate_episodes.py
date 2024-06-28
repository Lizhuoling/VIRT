import torch
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

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from dataset import load_data # data functions
from dataset import sample_box_pose, sample_insertion_pose # robot functions
from dataset import compute_dict_mean, detach_dict # helper functions
from utils.utils import set_seed
from utils.engine import launch
from utils import comm
from utils.optimizer import make_optimizer, make_scheduler
from utils.check_point import DetectronCheckpointer
from utils.metric_logger import MetricLogger
from utils.logger import setup_logger
from utils.transforms import build_transforms
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from configs.utils import load_yaml_with_base

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    # Initialize logger
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    exp_start_time = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    rank = comm.get_rank()
    logger = setup_logger(args.ckpt_dir, rank, file_name="log_{}.txt".format(exp_start_time))
    if comm.is_main_process():
        logger.info("Using {} GPUs".format(comm.get_world_size()))
        logger.info("Collecting environment info")
        logger.info(args) 
        logger.info("Loaded configuration file {}".format(args.config_name+'.yaml'))
    
    # Initialize cfg
    cfg = load_yaml_with_base(os.path.join('configs', args.config_name+'.yaml'))
    cfg['IS_EVAL'] = args.eval
    cfg['CKPT_DIR'] = args.ckpt_dir
    cfg['DATASET_DIR'] = args.data_dir
    cfg['ONSCREEN_RENDER'] = args.onscreen_render
    cfg['IS_DEBUG'] = args.debug
    cfg['NUM_NODES'] = args.num_nodes
    cfg['EVAL']['REAL_ROBOT'] = args.real_robot
    
    if cfg['SEED'] >= 0:
        set_seed(cfg['SEED'])

    if cfg['IS_EVAL']:
        ckpt_names = [f'policy_last.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(cfg, ckpt_name, save_episode=args.save_episode)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        exit()
    
    train_dataloader, val_dataloader, stats, _ = load_data(cfg)
    
    # save dataset stats
    if not os.path.isdir(cfg['CKPT_DIR']):
        os.makedirs(cfg['CKPT_DIR'])
    stats_path = os.path.join(cfg['CKPT_DIR'], f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    train_bc(train_dataloader, val_dataloader, cfg)

def make_policy(policy_class, cfg):
    if policy_class == 'ACT':
        policy = ACTPolicy(cfg)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(cfg)
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


def eval_bc(cfg, ckpt_name, save_episode=True):
    ckpt_dir = cfg['CKPT_DIR']
    state_dim = cfg['POLICY']['STATE_DIM']
    real_robot = cfg['EVAL']['REAL_ROBOT']
    policy_class = cfg['POLICY']['POLICY_NAME']
    onscreen_render = cfg['ONSCREEN_RENDER']
    camera_names = cfg['DATA']['CAMERA_NAMES']
    max_timesteps = cfg['POLICY']['EPISODE_LEN']
    task_name = cfg['TASK_NAME']
    temporal_agg = cfg['POLICY']['TEMPORAL_AGG']
    onscreen_cam = 'angle'
    
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, cfg)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process = lambda s_qpos: (s_qpos - stats['observations/qpos_mean'].numpy()) / stats['observations/qpos_std'].numpy()
    post_process = lambda a: a * stats['action_std'].numpy() + stats['action_mean'].numpy()
    inference_transforms = build_transforms(cfg)
    
    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = cfg['POLICY']['CHUNK_SIZE']
    if temporal_agg:
        query_frequency = 1
        num_queries = cfg['POLICY']['CHUNK_SIZE']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
    
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = inference_get_images(ts, camera_names, inference_transforms)

                ### query policy
                if cfg['POLICY']['POLICY_NAME'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif cfg['POLICY']['POLICY_NAME'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, cfg):
    logger = logging.getLogger("grasp")

    num_iterations = cfg['TRAIN']['NUM_ITERATIONS']
    ckpt_dir = cfg['CKPT_DIR']
    seed = cfg['SEED']
    policy_class = cfg['POLICY']['POLICY_NAME']

    policy = make_policy(policy_class, cfg)
    policy.cuda()
    optimizer = make_optimizer(policy, cfg)
    scheduler, warmup_scheduler = make_scheduler(optimizer, cfg=cfg)
    
    main_thread = comm.get_rank() == 0

    if cfg['TRAIN']['LR_WARMUP']:
        assert warmup_scheduler is not None
        warmup_iters = cfg['TRAIN']['WARMUP_STEPS']
    else:
        warmup_iters = -1

    min_val_loss = np.inf
    best_ckpt_info = None
    start_training_time = time.time()
    end = time.time()
    train_meters = MetricLogger(delimiter=", ", )

    for data, iter_cnt in zip(train_dataloader, range(num_iterations)):
        data_time = time.time() - end

        # training
        policy.train()
        optimizer.zero_grad()
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        if iter_cnt < warmup_iters:
            warmup_scheduler.step(iter_cnt)
        else:
            scheduler.step(iter_cnt)

        train_meters.update(**forward_dict)
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

        # validation
        if main_thread and iter_cnt % cfg['EVAL']['EVAL_INTERVAL'] == 0 and iter_cnt != 0:
            logger.info("Start evaluation at iteration {}...".format(iter_cnt))
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []

                eval_total_iter_num = len(val_dataloader)
                if eval_total_iter_num > cfg['EVAL']['MAX_VAL_SAMPLE_NUM']:
                    eval_total_iter_num = cfg['EVAL']['MAX_VAL_SAMPLE_NUM']

                for eval_data, eval_iter_cnt in zip(val_dataloader, range(eval_total_iter_num)):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
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
            ckpt_path = os.path.join(ckpt_dir, f'policy_iter{iter_cnt}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)

    if main_thread:
        ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
        torch.save(policy.state_dict(), ckpt_path)
        best_iter, min_val_loss, best_state_dict = best_ckpt_info
        logger.info(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at iteration {best_iter}')

    comm.synchronize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', action='store', type=str, help='configuration file name', required=True)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='saving directory', required=True)
    parser.add_argument('--data_dir', action='store', type=str, help='dataset folder path', required=True)
    parser.add_argument('--real_robot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_episode', action='store_true')

    parser.add_argument('--num_nodes', default = 1, type = int, help = "The number of nodes.")
    
    args = parser.parse_args()

    launch(main, args)
