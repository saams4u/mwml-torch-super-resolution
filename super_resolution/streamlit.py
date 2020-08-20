import streamlit as st

from opt_for_train import TrainingOptions, DeviceOptions, load_model, open_image, super_resolve, resize_naive


st.sidebar.markdown("## Training Options")

training_opts = TrainingOptions(
	upscale_factor=st.sidebar.slider('Upscale Factor', value=4, min_value=1, max_value=5),
	nEpochs=st.sidebar.slider('Training Epochs', value=35, min_value=1, max_value=100),
	lr=0.001,
	seed=123,
)

st.sidebar.markdown("## Hardware Options")

device_opts = DeviceOptions(
	cuda=st.sidebar.checkbox('Use CUDA', value=False),
	threads=st.sidebar.slider('Dataloader Threads', value=4, min_value=1, max_value=16),
	batchSize=st.sidebar.slider('Training Batch Size', value=4, min_value=1, max_value=256),
	testBatchSize=st.sidebar.slider('Testing Batch Size', value=100, min_value=1, max_value=256),
)

st.code(f"Using: upscale={training_opts.upscale_factor}x, "
		f"nEpochs={training_opts.nEpochs}")

model = load_model(training_opts, device_opts)


# => Set device as either CUDA or CPU
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# => Model checkpoints
	# srgan_checkpoint = "./checkpoint_srgan.pth.tar"
	# srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# => Load model, either the SRResNet or the SRGAN
	# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
	# srresnet.eval()
	# model = srresnet

	# srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
	# srgan_generator.eval()
	# model = srgan_generator


input_image = st.file_uploader("Upload an image", ["png", "jpg"], encoding=None)

if input_image is None:
	input_image = '../resources/images/leon.png'

st.image(open_image(input_image))

st.write('Super Resolution:')
st.image(super_resolve(model, input_image, False))

st.write('Naive Upscale:')
st.image(resize_naive(input_image, training_opts.upscale_factor))