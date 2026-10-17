# -*- coding: utf-8 -*-

import glob
import os
from PIL import Image
from tqdm import tqdm
import torch
import sys
import cv2
import numpy as np
from collections import OrderedDict
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.registry import ARCH_REGISTRY
import pyiqa

def imread2tensor(img):
    img_tensor = torch.from_numpy(np.float32(img).transpose(2, 0, 1) / 255.)
    return img_tensor


def main(input_dir, device):
	device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

	# set up IQA model
	print(pyiqa.list_models())
	iqa_model_topiq = pyiqa.create_metric('topiq_nr', device=device)
	print('topiq_nr', iqa_model_topiq.lower_better)
	iqa_model_mus = pyiqa.create_metric('musiq', device=device)
	print('musiq', iqa_model_mus.lower_better)
	iqa_model_qal = pyiqa.create_metric('qalign', device=device) 
	print('qalign', iqa_model_qal.lower_better)

	input_file = input_dir
	save_file = os.path.join(input_dir, 'real_metrics_cmmq.txt')

	if os.path.isfile(input_file):
		input_paths = [input_file]
	else:
		input_dir = os.path.join(input_file, '*', '*.png')
		fn = os.path.join(input_file, '*')
		fns = sorted(os.listdir(input_file))
		print(fns)
		input_paths = sorted(glob.glob(input_dir, recursive = True))

	sf = open(save_file, 'a')
	sf.write(f'input address:\t{input_file}\n')
	p = sf.tell()

	avg_score_mus = 0
	avg_score_qal = 0
	avg_score_topiq = 0

	test_img_num = len(input_paths)
	tqdm_input_paths = tqdm(input_paths)

	for idx, img_path in enumerate(tqdm_input_paths):
		img_name = os.path.basename(img_path)
		tar_img = cv2.imread(img_path, 1)

		H, W, C = tar_img.shape


		pre_img_mus = 0
		pre_img_qal = 0
		pre_img_topiq = 0

		img = tar_img

		img = torch.from_numpy(img.copy()).permute(2,0,1).unsqueeze(0) / 255.
      
		score_topiq = iqa_model_topiq(img).mean().item()
		pre_img_topiq += score_topiq
		torch.cuda.empty_cache()

		score_qal = iqa_model_qal(img).mean().item()
		pre_img_qal += score_qal
		torch.cuda.empty_cache()

		score_mus = iqa_model_mus(img).mean().item()
		pre_img_mus += score_mus
		torch.cuda.empty_cache()

		avg_score_mus += pre_img_mus 
		avg_score_qal += pre_img_qal
		avg_score_topiq += pre_img_topiq

		sf.write('%s  \t musiq: %.3f\n topiq: %.3f\n qalign: %.3f\n' % 
		         (img_name, pre_img_mus, pre_img_topiq, pre_img_qal))

	avg_score_mus /= test_img_num
	avg_score_qal /= test_img_num
	avg_score_topiq /= test_img_num

	print('Average musiq score with %s images is: %.3f \n' % (test_img_num, avg_score_mus))
	print('Average qal score with %s images is: %.3f \n' % (test_img_num, avg_score_qal))
	print('Average topiq score with %s images is: %.3f \n' % (test_img_num, avg_score_topiq))

	sf.seek(p)
	sf.write('Average musiq score with %s images is: %.3f \n' % (test_img_num, avg_score_mus))
	sf.write('Average qal score with %s images is: %.3f \n' % (test_img_num, avg_score_qal))
	sf.write('Average topiq score with %s images is: %.3f \n' % (test_img_num, avg_score_topiq))
	sf.close()

if __name__ == '__main__':
	with torch.no_grad():
		main()

# pip install git+https://github.com/chaofengc/IQA-PyTorch.git
# Ubuntu >= 18.04
# Python >= 3.8
# Pytorch >= 1.8.1
# CUDA >= 10.1 (if use GPU)
# 缺少 version python set_up.py develop
