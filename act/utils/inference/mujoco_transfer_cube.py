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
