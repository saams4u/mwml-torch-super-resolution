# datasets.py - load, preprocess, split, tokenize, etc. data.

import torch
import json
import os

from torch.utils.data import Dataset
from PIL import Image
from utils import ImageTransforms


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
	img = Image.open(filepath).convert('YCbCr')
	y, _, _ = img.split()
	return y


class DatasetFromFolder(Dataset):
	
	def __init__(self, img_dir, input_transform=None, target_transform=None):
		super(DatasetFromFolder, self).__init__()
		self.image_filenames = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if is_image_file(x)]

		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		input = load_img(self.image_filenames[index])
		target = input.copy()
		if self.input_transform:
			input = self.input_transform(input)
		if self.target_transform:
			target = self.target_transform(target)

		return input, target

	def __len__(self):
		return len(self.image_filenames)


class SRDataset(Dataset):

	def __init__(self, data_folder, split, crop_size, scaling_factor, lr_img_type, hr_img_type, test_data_name=None):
		self.data_folder = data_folder
		self.split = split.lower()
		self.crop_size = int(crop_size)
		self.scaling_factor = int(scaling_factor)
		self.lr_img_type = lr_img_type
		self.hr_img_type = hr_img_type
		self.test_data_name = test_data_name

		assert self.split in {'train', 'test'}
		if self.split == 'test' and self.test_data_name is None:
			raise ValueError("Please provide the name of the test dataset!")
		assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
		assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

		# If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
		# (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
		if self.split == 'train':
			assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"	

		# Read list of image-paths
		if self.split == 'train':
			with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
				self.images = json.load(j)
		else:
			with open(os.path.join(data_folder, self.test_data_name + '_images.json'), 'r') as j:
				self.images = json.load(j)

		# Select the correct set of transforms
		self.transform = ImageTransforms(split=self.split, crop_size=self.crop_size,
			scaling_factor=self.scaling_factor, lr_img_type=self.lr_img_type,
			hr_img_type=self.hr_img_type)

	def __getitem__(self, i):
		img = Image.open(self.images[i], mode='r')
		img = img.convert('RGB')
		if img.width <= 96 or img.height <= 96:
			print(self.images[i], img.width, img.height)
		lr_img, hr_img = self.transform(img)

		return lr_img, hr_img

	def __len__(self):
		return len(self.images)