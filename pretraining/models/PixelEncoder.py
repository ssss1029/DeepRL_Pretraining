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
		assert x.ndim == 4, 'input must be a 4D tensor'
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
	return PixelEncoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers, normalize
	)


class Classifier(nn.Module):
	"""
	Classifier wraps PixelEncoder
	"""

	def __init__(self, num_classes):
		super(Classifier, self).__init__()

		self.num_classes = num_classes
		enc_out_dim = 21 * 21 * 96
		self.encoder = make_encoder(
			obs_shape=(9, 84, 84),
			feature_dim=enc_out_dim,
			num_layers=11,
			num_filters=96,
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

		y1s = self.head1(h) # Channel 0-2 predictions
		y2s = self.head2(h) # Channel 3-5 predictions
		y3s = self.head3(h) # Channel 6-8 predictions

		preds = torch.cat([y1s, y2s, y3s], dim=0)

		if isinstance(targets, torch.Tensor):
			import random
			loss = F.cross_entropy(preds, targets)
		else:
			loss = None

		return preds, loss

if __name__ == "__main__":

	import torch
	import torch.nn as nn
	from torch.autograd import Variable

	from collections import OrderedDict
	import numpy as np


	def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
		result, params_info = summary_string(
			model, input_size, batch_size, device, dtypes)
		print(result)

		return params_info


	def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
		if dtypes == None:
			dtypes = [torch.FloatTensor]*len(input_size)

		summary_str = ''

		def register_hook(module):
			def hook(module, input, output):
				class_name = str(module.__class__).split(".")[-1].split("'")[0]
				module_idx = len(summary)

				m_key = "%s-%i" % (class_name, module_idx + 1)
				summary[m_key] = OrderedDict()
				summary[m_key]["input_shape"] = list(input[0].size())
				summary[m_key]["input_shape"][0] = batch_size
				if isinstance(output, (list, tuple)):
					summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
				else:
					summary[m_key]["output_shape"] = list(output.size())
					summary[m_key]["output_shape"][0] = batch_size

				params = 0
				if hasattr(module, "weight") and hasattr(module.weight, "size"):
					params += torch.prod(torch.LongTensor(list(module.weight.size())))
					summary[m_key]["trainable"] = module.weight.requires_grad
				if hasattr(module, "bias") and hasattr(module.bias, "size"):
					params += torch.prod(torch.LongTensor(list(module.bias.size())))
				summary[m_key]["nb_params"] = params

			if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
			):
				hooks.append(module.register_forward_hook(hook))

		# multiple inputs to the network
		if isinstance(input_size, tuple):
			input_size = [input_size]

		# batch_size of 2 for batchnorm
		x = [torch.rand(3, *in_size).type(dtype).to(device=device)
			for in_size, dtype in zip(input_size, dtypes)]

		# create properties
		summary = OrderedDict()
		hooks = []

		# register hook
		model.apply(register_hook)

		# make a forward pass
		# print(x.shape)
		model(*x)

		# remove these hooks
		for h in hooks:
			h.remove()

		summary_str += "----------------------------------------------------------------" + "\n"
		line_new = "{:>20}  {:>25} {:>15}".format(
			"Layer (type)", "Output Shape", "Param #")
		summary_str += line_new + "\n"
		summary_str += "================================================================" + "\n"
		total_params = 0
		total_output = 0
		trainable_params = 0
		for layer in summary:
			# input_shape, output_shape, trainable, nb_params
			line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
			total_params += summary[layer]["nb_params"]

			total_output += np.prod(summary[layer]["output_shape"])
			if "trainable" in summary[layer]:
				if summary[layer]["trainable"] == True:
					trainable_params += summary[layer]["nb_params"]
			summary_str += line_new + "\n"

		# assume 4 bytes/number (float on cuda).
		total_input_size = abs(np.prod(sum(input_size, ()))
							* batch_size * 4. / (1024 ** 2.))
		total_output_size = abs(2. * total_output * 4. /
								(1024 ** 2.))  # x2 for gradients
		total_params_size = abs(total_params * 4. / (1024 ** 2.))
		total_size = total_params_size + total_output_size + total_input_size

		summary_str += "================================================================" + "\n"
		summary_str += "Total params: {0:,}".format(total_params) + "\n"
		summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
		summary_str += "Non-trainable params: {0:,}".format(total_params -
															trainable_params) + "\n"
		summary_str += "----------------------------------------------------------------" + "\n"
		summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
		summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
		summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
		summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
		summary_str += "----------------------------------------------------------------" + "\n"
		# return summary
		return summary_str, (total_params, trainable_params)


	model = Classifier(num_classes=100).cuda()
	print(model)
	# summary(model, batch_size=3, input_size=(3, 84, 84))

