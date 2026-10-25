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

class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction

        # =========== define metric model ===============
        net_opts = OrderedDict()
        # load default setting first
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        # then update with custom setting
        net_opts.update(kwargs)
        network_type = net_opts.pop('type')
        self.net = ARCH_REGISTRY.get(network_type)(**net_opts)
        self.net = self.net.to(self.device)
        self.net.eval()

    def to(self, device):
        self.net.to(device)
        self.device = torch.device(device)
        return self
    
    def forward(self, target, ref=None, **kwargs):
        with torch.set_grad_enabled(self.as_loss):
            if 'fid' in self.metric_name:
                output = self.net(target, ref, device=self.device, **kwargs)
            else:
                if not torch.is_tensor(target):
                    target = imread2tensor(target)
                    target = target.unsqueeze(0)
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        ref = imread2tensor(ref)
                        ref = ref.unsqueeze(0)
                if self.metric_mode == 'FR':
                    output = self.net(target.to(self.device), ref.to(self.device), **kwargs)
                elif self.metric_mode == 'NR':
                    output = self.net(target.to(self.device), **kwargs)
        return output


def imread2tensor(img):
    img_tensor = torch.from_numpy(np.float32(img).transpose(2, 0, 1) / 255.)
    return img_tensor


def main(input_dir, device):
	device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

	# set up IQA model
	print(pyiqa.list_models())
	iqa_model_topiq = pyiqa.create_metric('topiq_nr', device=device)
	print('topiq_nr', iqa_model_topiq.lower_better)
	iqa_model_man = pyiqa.create_metric('maniqa', device=device) 
	print('maniqa', iqa_model_man.lower_better)
	iqa_model_mus = pyiqa.create_metric('musiq', device=device)
	print('musiq', iqa_model_mus.lower_better)

	input_file = input_dir
	save_file = os.path.join(input_dir, 'metrics_real.txt')

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
	avg_score_man = 0
	avg_score_topiq = 0

	test_img_num = len(input_paths)
	tqdm_input_paths = tqdm(input_paths)

	for idx, img_path in enumerate(tqdm_input_paths):
		img_name = os.path.basename(img_path)
		tar_img = cv2.imread(img_path, 1)

		H, W, C = tar_img.shape

		ref_img = None

		pre_img_man = 0
		pre_img_mus = 0
		pre_img_topiq = 0

		img = tar_img
		img = torch.from_numpy(img.copy()).permute(2,0,1).unsqueeze(0) / 255.
      
		score_topiq = iqa_model_topiq(img).mean().item()
		pre_img_topiq += score_topiq
		torch.cuda.empty_cache()

		score_man = iqa_model_man(img).mean().item()
		pre_img_man += score_man
		torch.cuda.empty_cache()

		score_mus = iqa_model_mus(img).mean().item()
		pre_img_mus += score_mus
		torch.cuda.empty_cache()

		avg_score_mus += pre_img_mus 
		avg_score_man += pre_img_man
		avg_score_topiq += pre_img_topiq

		sf.write('%s  \t musiq: %.3f\n maniqa: %.3f\n topiq: %.3f\n' % 
		         (img_name, pre_img_mus, pre_img_man, pre_img_topiq))


	avg_score_mus /= test_img_num
	avg_score_man /= test_img_num
	avg_score_topiq /= test_img_num

	print('Average musiq score with %s images is: %.3f \n' % (test_img_num, avg_score_mus))
	print('Average maniqa score with %s images is: %.3f \n' % (test_img_num, avg_score_man))
	print('Average topiq score with %s images is: %.3f \n' % (test_img_num, avg_score_topiq))

	sf.seek(p)
	sf.write('Average musiq score with %s images is: %.3f \n' % (test_img_num, avg_score_mus))
	sf.write('Average maniqa score with %s images is: %.3f \n' % (test_img_num, avg_score_man))
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