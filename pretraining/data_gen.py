print("Imports starting...")
from env.wrappers import make_pad_env
import numpy as np
from tqdm import tqdm
print("Imports done!")
import sys

assert len(sys.argv) == 5, "Specify env, task, num_data_pairs, reset_horizon."

env_name, task_name, num_data_pairs, reset_horizon = sys.argv[1:]

assert env_name in ['walker', 'cheetah', 'reacher'], "Invalid environment."

def get_env_dataset(env_name, task_name, num_data_pairs, reset_horizon):
    i = reset_horizon
    env = make_pad_env(env_name, task_name, seed=0, episode_length=reset_horizon, frame_stack=3, action_repeat=4, mode='train')
    for data_num in tqdm(range(num_data_pairs)):
        if i >= reset_horizon:
            curr_obs = env.reset()
            i = 0
        action = env.action_space.sample()
        next_obs, _, done, _ = env.step(action)
        data = (curr_obs, action, next_obs)
        curr_obs = next_obs
        np.save("/mnt/EnvData/{}/{}.npy".format(env_name, data_num), data)
        i = i + 1 if not done else reset_horizon

get_env_dataset(env_name, task_name, int(num_data_pairs), int(reset_horizon))
