print("Imports starting...")
from env.wrappers import make_pad_env
import numpy as np
from tqdm import tqdm
print("Imports done!")
import sys
import torch
torch.set_num_threads(2)
import re

assert len(sys.argv) == 6, "Specify env, task, num_data_pairs, reset_horizon, mode."

env_name, task_name, num_data_pairs, reset_horizon, mode = sys.argv[1:]

assert env_name in ['walker', 'cheetah', 'reacher'], "Invalid environment."

def get_env_dataset(env_name, task_name, num_data_pairs, reset_horizon, mode):
    i = reset_horizon
    env = make_pad_env(env_name, task_name, seed=0, episode_length=reset_horizon, frame_stack=3, action_repeat=4, mode=mode)
    for data_num in tqdm(range(num_data_pairs)):
        if i >= reset_horizon:
            curr_obs = env.reset()
            i = 0
        action = env.action_space.sample()
        next_obs, _, done, _ = env.step(action)
        data = (curr_obs, action, next_obs)
        curr_obs = next_obs
        video_dir = env_name
        if 'video' in mode:
            video_num = int(re.search(r'\d+$', mode).group())
            video_dir += f'_distracted/{video_num}'
        np.save("/data/sauravkadavath/sam/env_data/{}/{}.npy".format(video_dir, data_num), data)
        i = i + 1 if not done else reset_horizon

get_env_dataset(env_name, task_name, int(num_data_pairs), int(reset_horizon), mode)
