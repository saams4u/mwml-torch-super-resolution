# create_data.py - create and process input data into outputted JSON lists.

from utils import create_data

if __name__ == '__main__':
	# create_data(train_folders=['../input_data/train2014', '../input_data/val2014'],
	# 			test_folders=['../input_data/BSD100/image_SRF_4', '../input_data/Set5/image_SRF_4', '../input_data/Set14/image_SRF_4'],
	# 			min_size=100,
	# 			output_folder='../output_lists')

	create_data(train_folders=['../input_data/BSDS300/bsds300_train'],
				test_folders=['../input_data/BSDS300/bsds300_test'],
				min_size=100,
				output_folder='../output_lists')