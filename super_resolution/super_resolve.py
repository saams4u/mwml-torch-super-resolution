import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
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

	# Super resolution
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


visualize_sr("../input_data/Set14/image_SRF_4/img_002_SRF_4_LR.png")