import sys
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from argparse import ArgumentParser
import nriqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='test_results/', help='where to save image results')
    args = parser.parse_args()
    
    nriqa.main(args.save_dir, 0)