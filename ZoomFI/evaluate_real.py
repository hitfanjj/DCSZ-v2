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
from skimage.metrics import structural_similarity as SSIM

from config.parser import parse_args
import nriqa

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./', help='where to evalute image results')
    args = parse_args(parser)
    
    nriqa.main(args.data_dir, 0)





    

