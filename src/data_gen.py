from env.wrappers import make_pad_env
import numpy as np
from tqdm import tqdm

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
        np.save("/data/sauravkadavath/DeepRL_Pretraining_data/{}/{}".format(env_name, data_num), data)
        i = i + 1 if not done else reset_horizon

get_env_dataset('walker', 'walk', 200000, 100)
