import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}

def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias


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


class PixelEncoder(nn.Module):
	"""Convolutional encoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4, normalize=True):
		super().__init__()
		assert len(obs_shape) == 3

		self.feature_dim = feature_dim
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
				k = 3
			else:
				groups = 3
				k = 3
			self.convs.append(nn.Sequential(
				nn.Conv2d(num_filters, num_filters, kernel_size=k, stride=1, groups=groups),
				nn.BatchNorm2d(num_filters)
			))
			

		out_dim = OUT_DIM[num_layers]
		# self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		# self.ln = nn.LayerNorm(self.feature_dim)

	def forward_conv(self, obs, detach=False):
		B, C, H, W = obs.shape
		obs = self.preprocess(obs)
		conv = torch.relu(self.convs[0](obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()

		split = torch.split(conv, conv.shape[1] // 3, dim=1)
		split = torch.stack(split, dim=0)
		split = split.reshape(3, B, -1)
		return split

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs, detach)
		out = h
		# h_fc = self.fc(h)
		# h_norm = self.ln(h_fc)
		# out = torch.tanh(h_norm)

		return out

	def copy_conv_weights_from(self, source, n=None):
		"""Tie n first convolutional layers"""
		if n is None:
			n = self.num_layers
		for i in range(n):
			# Tie Convs
			tie_weights(src=source.convs[i][0], trg=self.convs[i][0])
			# Tie BNs
			tie_weights(src=source.convs[i][1], trg=self.convs[i][1])


def make_encoder(
	obs_shape, feature_dim, num_layers, num_filters, num_shared_layers, normalize
):
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'

	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers

	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	
	encoder = PixelEncoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers, normalize
	)
	return encoder


class ClassifierFullGroupConvBigger(nn.Module):
	"""
	ClassifierFullGroupConvBigger wraps PixelEncoder
	"""

	def __init__(self, num_classes):
		super(ClassifierFullGroupConvBigger, self).__init__()

		self.num_classes = num_classes
		hidden_channels = 288
		enc_out_dim = 21 * 21 * hidden_channels // 3
		self.encoder = make_encoder(
			obs_shape=(9, 84, 84),
			feature_dim=enc_out_dim,
			num_layers=11,
			num_filters=hidden_channels,
			num_shared_layers=8,
			normalize=False
		)

		self.head1 = nn.Sequential(
			nn.Linear(enc_out_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, num_classes)
		)
		self.head2 = nn.Sequential(
			nn.Linear(enc_out_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, num_classes)
		)
		self.head3 = nn.Sequential(
			nn.Linear(enc_out_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, num_classes)
		)

	def forward(self, x, targets=None):
		"""
		obs: [B, 3, H, W]
		"""

		B, C, H, W = x.shape
		assert C == 3, f"Got C = {C}"
		assert B % 3 == 0, f"Got B = {B}"

		# Reshape input
		x = torch.split(x, B // 3)
		assert len(x) == 3
		x = torch.cat(x, dim=1)

		# torchvision.utils.save_image(x[0, 0:3].unsqueeze(0), "checkpoints/TEMP/1.png")
		# torchvision.utils.save_image(x[0, 3:6].unsqueeze(0), "checkpoints/TEMP/2.png")
		# torchvision.utils.save_image(x[0, 6:9].unsqueeze(0), "checkpoints/TEMP/3.png")
		# print(x.shape)

		h = self.encoder(x)
		h1, h2, h3 = h[0], h[1], h[2]

		y1s = self.head1(h1) # Channel 0-2 predictions
		y2s = self.head2(h2) # Channel 3-5 predictions
		y3s = self.head3(h3) # Channel 6-8 predictions

		preds = torch.cat([y1s, y2s, y3s], dim=0)

		if isinstance(targets, torch.Tensor):
			import random
			loss = F.cross_entropy(preds, targets)
		else:
			loss = None

		return preds, loss

if __name__ == "__main__":

	model = Classifier(num_classes=100).cuda()
	print(model)

	ex = torch.zeros((3, 3, 84, 84)).cuda()
	ret = model(ex)

