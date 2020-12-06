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
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		self.ln = nn.LayerNorm(self.feature_dim)

	def forward_conv(self, obs, detach=False):
		obs = self.preprocess(obs)
		conv = torch.relu(self.convs[0](obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()

		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs, detach)
		h_fc = self.fc(h)
		h_norm = self.ln(h_fc)
		out = torch.tanh(h_norm)

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
	obs_shape, feature_dim, num_layers, num_filters, num_shared_layers, encoder_checkpoint, normalize=True,
):
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
	
	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers

	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	
	encoder = PixelEncoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers, normalize
	)

	if encoder_checkpoint is not None:
		cleaned_checkpoint = get_state_dict_sam(encoder_checkpoint)

		# No strict to ensure we dont't error for not loading fc and ln
		encoder.load_state_dict(cleaned_checkpoint, strict=False)
		print(f"Successfully loaded checkpoint {encoder_checkpoint}")
	return encoder

def get_state_dict_saurav(encoder_checkpoint: str):
	cleaned_checkpoint = dict()
	exclude_keys = [
		'head1.weight', 'head1.bias', 
		'head2.weight', 'head2.bias', 
		'head3.weight', 'head3.bias', 
		'encoder.fc.weight', 'encoder.fc.bias', 
		'encoder.ln.weight', 'encoder.ln.bias']
	checkpoint = torch.load(encoder_checkpoint)['state_dict']
	for k, v in checkpoint.items():
		if k not in exclude_keys:
			if k.startswith('encoder.'):
				k = k[8:]
			cleaned_checkpoint[k] = v
	return cleaned_checkpoint


def get_state_dict_sam(encoder_checkpoint: str):
	"""
	Get the state dict from Sam's original saved checkpoints (should also work for Brian's)
	"""
	cleaned_checkpoint = dict()
	exclude_keys = [
		'fc1.weight', 'fc1.bias',
		'ln1.weight', 'ln1.bias',
		'fc2.weight', 'fc2.bias',
		'ln2.weight', 'ln2.bias',
		'output.weight','output.bias'
	]
	checkpoint = torch.load(encoder_checkpoint)
	for k, v in checkpoint.items():
		if k not in exclude_keys:
			cleaned_checkpoint[k] = v

	return cleaned_checkpoint
