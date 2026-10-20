import torch
import os
import numpy as np
import glob

def evaluate_ewarp(img_img, pre_img_img, flow_model):
    with torch.no_grad():
        backward_flow = flow_model(img_img * 255, pre_img_img * 255, iters=args.iters, test_mode=True)['flow'][-1]
    w_pre_img_img, img_mask = get_backwarp(pre_img_img, backward_flow)
    msg = ((w_pre_img_img - img_img) ** 2).mean()
    return msg

dataroot = ""
ids = os.listdir(dataroot)
mse_total = 0
for id in ids:
    files = sorted(glob.glob(os.path.join(dataroot, id, '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0].split('_')[-1]))
    
    
    
    