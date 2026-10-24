import sys

sys.path.append('.')

import warnings

warnings.filterwarnings('ignore')

import shutil
import math
import os
import cv2
import torch
import os.path as osp
import numpy as np
from argparse import ArgumentParser
import glob

import nriqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def crop_center(hr, ph, pw):
    ih, iw = hr.shape[0:2]
    lr_patch_h, lr_patch_w = ph, pw
    ph = ih // 2 - lr_patch_h // 2
    pw = iw // 2 - lr_patch_w // 2

    return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

def lpips_norm(img, range):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (range / 2.) - 1
	return torch.Tensor(img).to(device)

def calc_lpips(out, target, loss_fn_alex, range):
	lpips_out = lpips_norm(out, range)
	lpips_target = lpips_norm(target, range)
	LPIPS = loss_fn_alex(lpips_out, lpips_target)
	return LPIPS.detach().cpu().item()

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--dataset_dir", type=str, default="./", help="train data path")
    parser.add_argument('--save_dir', type=str, default='test_results/', help='where to save image results')
    args = parser.parse_args()

    # args.save_dir = os.path.join(args.log_dir, args.save_dir)
    # print(args.log_dir)
    
    nriqa.main(args.save_dir, 0)