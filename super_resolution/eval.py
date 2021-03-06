# eval.py - predict (infer) inputs (single/batch).

from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset

from data import get_test_set
from models import Net

from opt_for_train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = "../output_lists"
test_data_names = ["test"]

# data_folder = '../output_lists'
# test_data_names = ["image_SRF_4"]

# Static Model checkpoints
net_checkpoint = "models/model_net_35.pth"
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

## => Load model, Net, SRResNet or SRGAN
net = torch.load(net_checkpoint)['model'].to(device)
net.eval()
model = net

# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# srresnet.eval()
# model = srresnet

# srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
# srgan_generator.eval()
# model = srgan_generator

# Evaluate
for test_data_name in test_data_names:
	print("\nFor %s:\n" % test_data_name)

	## => Custom dataloaders for 
	# test_dataset = SRDataset(data_folder,
	# 						 split='test',
	# 						 crop_size=0,
	# 						 scaling_factor=4,
	# 						 lr_img_type='imagenet-norm',
	# 						 hr_img_type='[-1, 1]',
	# 						 test_data_name=test_data_name)
	# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
	# 										  pin_memory=True)

	test_dataset = get_test_set(TrainingOptions.upscale_factor)

	test_loader = DataLoader(
		dataset=test_dataset,
		num_workers=DeviceOptions.threads,
		batch_size=DeviceOptions.testBatchSize,
		shuffle=False)

	# Keep track of the PSNRs and the SSIMs across batches
	PSNRs = AverageMeter()
	SSIMs = AverageMeter()

	# Prohibit gradient computation explicitly because I had some problems with memory
	with torch.no_grad():
		# Batches
		for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
			# Move to default device
			lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
			hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

			# Forward prop.
			sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

			# Calculate PSNR and SSIM
			sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
				0)  # (w, h), in y-channel
			hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(
				0)  # (w, h), in y-channel
			psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
										   data_range=255.)
			ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
										   data_range=255.)
			PSNRs.update(psnr, lr_imgs.size(0))
			SSIMs.update(ssim, lr_imgs.size(0))

	# Print average PSNR and SSIM
	print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
	print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

print("\n")