# models.py - define model architectures.

import torch
import torchvision
import math

from torch import nn
from torch.nn import init


class Net(nn.Module):

	def __init__(self, upscale_factor):
		super(Net, self).__init__()

		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
		self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
		self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
		self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

		self._initialize_weights()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.pixel_shuffle(self.conv4(x))
		return x

	def _initialize_weights(self):
		init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv4.weight)
		

class ConvolutionalBlock(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
		super(ConvolutionalBlock, self).__init__()

		if activation is not None:
			activation = activation.lower()
			assert activation in {'prelu', 'leakyrelu', 'tanh'}

		# A container that will hold the layers in this convolutonal block
		layers = list()

		# A convolutional layer
		layers.append(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
					  stride=stride, padding=kernel_size // 2))

		# A batch normalization (BN) layer, if wanted
		if batch_norm is True:
			layers.append(nn.BatchNorm2d(num_features=out_channels))

		# An activation layer, if wanted 
		if activation == 'prelu':
			layers.append(nn.PReLU())
		elif activation == 'leakyrelu':
			layers.append(nn.LeakyReLU(0.2))
		elif activation == 'tanh':
			layers.append(nn.Tanh())

		# Put together the convolutional block as a sequence of the layers in this container
		self.conv_block = nn.Sequential(*layers)

	def forward(self, input):
		output = self.conv_block(input)  # (N, out_channels, w, h)

		return output


class SubPixelConvolutionalBlock(nn.Module):

	def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
		super(SubPixelConvolutionalBlock, self).__init__()

		# A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
		self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
							  kernel_size=kernel_size, padding=kernel_size // 2)

		# These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
		self.prelu = nn.PReLU()

	def forward(self, input):
		output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
		output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling_factor, h * scaling_factor)
		output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

		return output


class ResidualBlock(nn.Module):

	def __init__(self, kernel_size=3, n_channels=64):
		super(ResidualBlock, self).__init__()

		# The first convolutional block
		self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
											batch_norm=True, activation='prelu')

		# The second convolutional block 
		self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
											batch_norm=True, activation=None)

	def forward(self, input):
		residual = input  # (N, n_channels, w, h)
		output = self.conv_block1(input)  # (N, n_channels, w, h)
		output = self.conv_block2(output)  # (N, n_channels, w, h)
		output = output + residual  # (N, n_channels, w, h)

		return output


class SRResNet(nn.Module):

	def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
		super(SRResNet, self).__init__()

		# Scaling factor must be 2, 4, or 8
		scaling_factor = int(scaling_factor)
		assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

		# The first convolutional block
		self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
											  batch_norm=False, activation='prelu')

		# A sequence of n_blocks residual blocks, each containing a skip-connection across the block
		self.residual_blocks = nn.Sequential(
			*[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

		# Another convolutional block
		self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
											  kernel_size=small_kernel_size, 
											  batch_norm=True, activation=None)

		# Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
		n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
		self.subpixel_convolutional_blocks = nn.Sequential(
			*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels,
				scaling_factor=2) for i in range(n_subpixel_convolution_blocks)])

		# The last convolutional block
		self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
											  batch_norm=False, activation='tanh')

	def forward(self, lr_imgs):
		output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
		residual = output  # (N, n_channels, w, h)
		output = self.residual_blocks(output)  # (N, n_channels, w, h)
		output = self.conv_block2(output)  # (N, n_channels, w, h)
		output = output + residual  # (N, n_channels, w, h)
		output = self.subpixel_convolutional_blocks(output)  # (N, n_channels, w * scaling_factor, h * scaling_factor)
		sr_imgs = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)

		return sr_imgs


class Generator(nn.Module):

	def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
		super(Generator, self).__init__()

		# The generator is simply an SRResNet, as above
		self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
							n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

	def initialize_with_srresnet(self, srresnet_checkpoint):
		srresnet = torch.load(srresnet_checkpoint)['model']
		self.net.load_state_dict(srresnet.state_dict())

		print("\nLoaded weights from pre-trained SRResNet.\n")

	def forward(self, lr_imgs):
		sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

		return sr_imgs


class Discriminator(nn.Module):

	def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
		super(Discriminator, self).__init__()

		in_channels = 3

		# A series of convolutional blocks
		# The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
		# The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
		# The first convolutional block is unique because it does not employ batch normalization
		conv_blocks = list()
		for i in range(n_blocks):
			out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
			conv_blocks.append(
				ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
								   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='leakyrelu'))
			in_channels = out_channels
		self.conv_blocks = nn.Sequential(*conv_blocks)

		# An adaptive pool layer that resizes it to a standard size
		# For the default input size of 96 and 8 convolutional blocks, this will have no effect
		self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

		self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

		self.leaky_relu = nn.LeakyReLU(0.2)

		self.fc2 = nn.Linear(1024, 1)

		# Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

	def forward(self, imgs):
		batch_size = imgs.size(0)
		output = self.conv_blocks(imgs)
		output = self.adaptive_pool(output)
		output = self.fc1(output.view(batch_size, -1))
		output = self.leaky_relu(output)
		logit = self.fc2(output)

		return logit


class TruncatedVGG19(nn.Module):

	def __init__(self, i, j):
		super(TruncatedVGG19, self).__init__()

		# Load the pre-trained VGG19 available in torchvision
		vgg19 = torchvision.models.vgg19(pretrained=True)

		maxpool_counter = 0
		conv_counter = 0
		truncate_at = 0

		# Iterate through the convolutional section ("features") of the VGG19
		for layer in vgg19.features.children():
			truncate_at += 1

			# Count the number of maxpool layers and the convolutional layers after each maxpool
			if isinstance(layer, nn.Conv2d):
				conv_counter += 1
			if isinstance(layer, nn.MaxPool2d):
				maxpool_counter += 1
				conv_counter = 0

			# Break if we reach the jth convolution after the (i - 1)th maxpool
			if maxpool_counter == i - 1 and conv_counter == j:
				break

		# Check if conditions were satisfied
		assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (i, j)

		# Truncate to the jth convolution (+ activation) before the ith maxpool layer
		self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

	def forward(self, input):
		output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

		return output