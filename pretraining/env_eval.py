# IMPORTS
import random
import sys

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

# CONSTANTS
DATA_ROOT = "/global/scratch/brianyao/DeepRL_Pretraining/pretraining/env_data/{}/"
DATASET_SIZE = 200000
ENV_TO_ACTION_DIM = {
	'walker' : 6,
	'reacher' : 6,
	'cheetah' : 6,
}
OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}

# TRANSFORMS
class CenterCrop(nn.Module):
	"""Center-crop if observation is not already cropped"""
	def __init__(self, size):
		super().__init__()
		assert size == 84
		self.size = size

	def forward(self, x):
		assert len(x.shape) == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		elif x.size(-1) == 100:
			return x[:, :, 8:-8, 8:-8]
		else:
			return ValueError('unexepcted input size')


class NormalizeImg(nn.Module):
	"""Normalize observation"""
	def forward(self, x):
		return x/255.

# MODEL
class PixelEncoder(nn.Module):
	"""Convolutional encoder of pixel observations"""
	def __init__(self, obs_shape, action_dim, num_layers=11, num_filters=90, num_shared_layers=8, normalize=False):
		super().__init__()
		assert len(obs_shape) == 3

		self.num_layers = num_layers
		self.num_shared_layers = num_shared_layers

		preprocess_modules = [CenterCrop(size=84)]
		if normalize:
			preprocess_modules.append(NormalizeImg())
		self.preprocess = nn.Sequential(*preprocess_modules)

		self.convs = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(obs_shape[0], num_filters, 3, stride=2, groups=3),
				nn.BatchNorm2d(num_filters)
			)
		])

		for i in range(num_layers - 1):
			if i < (num_layers - 1) // 2:
				groups = 3
			else:
				groups = 1
			self.convs.append(nn.Sequential(
				nn.Conv2d(num_filters, num_filters, 3, stride=1, groups=groups),
				nn.BatchNorm2d(num_filters)
			))
		out_dim = OUT_DIM[num_layers]
		self.fc1 = nn.Linear(2 * num_filters * out_dim * out_dim, 1024)
		self.ln1 = nn.LayerNorm(1024)
		self.fc2 = nn.Linear(1024, 256)
		self.ln2 = nn.LayerNorm(256)
		self.output = nn.Linear(256, action_dim)
		self.dropout = nn.Dropout(0.5)

	def forward_conv(self, obs, detach=False):
		obs = self.preprocess(obs)
		conv = torch.relu(self.convs[0](obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()

		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, s0, s1, detach=False):
		h1 = self.forward_conv(s0, detach)
		h2 = self.forward_conv(s1, detach)
		h = torch.cat((h1, h2), 1)
		h = self.dropout(h)
		h_fc1 = torch.relu(self.ln1(self.fc1(h)))
		h_fc1 = self.dropout(h_fc1)
		h_fc2 = torch.relu(self.ln2(self.fc2(h_fc1)))
		h_fc2 = self.dropout(h_fc2)
		h_out = self.output(h_fc2)
		return h_out

# DATA
class EnvDataset(Dataset):
	"""Env dataset."""

	def __init__(self, root_dir, train):
		self.root_dir = root_dir
		self.train = train

	def __len__(self):
		return int(.95 * DATASET_SIZE) if self.train else int(.05 * DATASET_SIZE)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		if not self.train:
			idx += int(.95 * DATASET_SIZE)
		data = np.load(self.root_dir + "{}.npy".format(idx), allow_pickle=True)
		starts = torch.from_numpy(data[0].astype('float32'))
		actions = torch.from_numpy(data[1].astype('float32'))
		ends = torch.from_numpy(data[2].astype('float32'))
		return (starts, actions, ends)

if __name__ == '__main__':
	# Setup
	assert(len(sys.argv) == 3)
	env = sys.argv[1]
	checkpoint_path = sys.argv[2]

	# Data
	data_dir = DATA_ROOT.format(env)
	env_dataset_test = EnvDataset(data_dir, train=False)
	test_loader = DataLoader(env_dataset_test, batch_size=128, shuffle=False, num_workers=4)

	# Device / Model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("DEVICE: {}".format(device))
	model = PixelEncoder((9,100,100), ENV_TO_ACTION_DIM[env]).to(device)
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint)

	# Testing
	print("Starting Testing")
	model = model.eval()
	with torch.no_grad():
	    for j, data in enumerate(tqdm(test_loader)):
	        starts = data[0].to(device)
	        actions = data[1].to(device)
	        ends = data[2].to(device)
	        outputs = model(starts, ends)

	        for j in range(5):
	            _j = random.choice(list(range(starts.shape[0])))
	            print(actions[_j])
	            print(outputs[_j])
	            print(abs(outputs[_j] - actions[_j]))
	            print(torch.mean((outputs[_j] - actions[_j]) ** 2))
	            print()
