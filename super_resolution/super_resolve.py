from torchvision.transforms import ToTensor
from __future__ import print_function

from utils import *
from PIL import Image, ImageDraw, ImageFont

import argparse
import torch
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')

parser.add_argument('--input_image', default='leon.png', type=str, required=False, help='input image to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

parser.add_argument('--n_epoch', type=int, default=34, help='epoch number to load from model')

parser.add_argument('--model', type=str, default='model_net_{opt.n_epoch}.pth', required=False, help='model file to use')
# parser.add_argument('--model', type=str, default='model_srgan_{opt.n_epoch}.pth.tar', required=False, help='model file to use')
# parser.add_argument('--model', type=str, default='model_srresnet_{opt.n_epoch}.pth.tar', required=False, help='model file to use')

parser.add_argument('--output_filename', default='../resources/images/{opt.net}_{opt.input_image}', type=str, help='where to save the output image')
# parser.add_argument('--output_filename', default='../resources/images/{opt.srgan}_{opt.input_image}', type=str, help='where to save the output image')
# parser.add_argument('--output_filename', default='../resources/images/{opt.srresnet}_{opt.input_image}', type=str, help='where to save the output image')

print(opt)

## Sample Image -> leon.png"
img = Image.open(opt.input_image).convert('YCbCr')
y, cb = cr = img.split()

model = torch.load(opt.model)
# model = torch.load(opt.srgan)
# model = torch.load(opt.srresnet)

img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

if opt.cuda:
	model = model.cuda()
	input = input.cuda()

out = model(input)
out= out.cpu()

out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img.clip(0, 255)
out_img_y = Image.fromarray(np.unint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)

out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
out_img.save(opt.output_filename)

print('output image from Net model saved to ', opt.output_filename)
# print('output image from SRResNet model saved to ', opt.output_filename)
# print('output image from SRGAN model saved to ', opt.output_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## => Static Model checkpoints
srgan_checkpoint = "./model_srgan.pth.tar"
srresnet_checkpoint = "./model_srresnet.pth.tar"

# Load models
# net = torch.load(opt.net)['model'].to(device)
# net.eval()

# srresnet = torch.load(opt.srresnet)['model'].to(device)
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()

# srgan_generator = torch.load(opt.srgan)['generator'].to(device)
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()


def visualize_sr(img, halve=False):
	# Load image, downsample to obtain low-res version
	hr_img = Image.open(img, mode="r")
	hr_img = hr_img.convert('RGB')

	if halve:
		hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
								Image.LANCZOS)
	lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
								Image.BICUBIC)

	# Bicubic Upsampling 
	bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

	# Super resolution (SR) with Net
	# sr_img_net = net(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
	# sr_img_net = sr_img_net.squeeze(0).cpu().detach()
	# sr_img_net = convert_image(sr_img_net, source='[-1, 1]', target='pil')

	# Super resolution (SR) with SRResNet
	sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
	sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
	sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

	# Super-resolution (SR) with SRGAN
	sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
	sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
	sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

	# Create grid
	margin = 40
	grid_img = Image.new('RGB', (2 * hr_img.width + 3 * margin, 2 * hr_img.height + 3 * margin), (255, 255, 255))

	# Font
	draw = ImageDraw.Draw(grid_img)

	try:
		font = ImageFont.truetype("../resources/fonts/calibril.ttf", size=23)
		print("Loading Images from Static Checkpoints for SRGAN and SRResNet Models as Compared to Bicubic and High-Res Outcomes...")
		# Otherwise, use any TTF font of your choice
	except OSError:
		print(
			"Defaulting to a terrible font. To use a font of your choice, include the link to its TFF file in the function.")
		font = ImageFont.load_default()

	# Place bicubic-upsampled image
	grid_img.paste(bicubic_img, (margin, margin))
	text_size = font.getsize("Bicubic")
	draw.text(xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
			  font=font, fill='black')

	# Place Net image
	# grid_img.paste(sr_img_net, (2 * margin + bicubic_img.width, margin))
	# text_size = font.getsize("Net")
	# draw.text(
	# 	xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
	# 	text="Net", font=font, fill='black')

	# Place SRResNet image
	grid_img.paste(sr_img_srresnet, (2 * margin + bicubic_img.width, margin))
	text_size = font.getsize("SRResNet")
	draw.text(
		xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
		text="SRResNet", font=font, fill='black')

	# Place SRGAN image
	grid_img.paste(sr_img_srgan, (margin, 2 * margin + sr_img_srresnet.height))
	text_size = font.getsize("SRGAN")
	draw.text(
		xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, 2 * margin + sr_img_srresnet.height - text_size[1] - 5],
		text="SRGAN", font=font, fill='black')

	# Place original HR image
	grid_img.paste(hr_img, (2 * margin + bicubic_img.width, 2 * margin + sr_img_srresnet.height))
	text_size = font.getsize("Original HR")
	draw.text(xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2,
			      2 * margin + sr_img_srresnet.height - text_size[1] - 1], text="Original HR", font=font, fill='black')

	# Display grid
	grid_img.show()

	return grid_img


visualize_sr(opt.input_image)