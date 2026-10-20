import sys
sys.path.append('../core')
import torch
import torch.nn as nn
import scipy.stats as st
import os
import numpy as np
import glob
from warplayer import getwarp
import torch.nn.functional as F
from raft import RAFT
from utils.utils import load_ckpt

import argparse
from config.parser import parse_args

import matplotlib.pyplot as plt 

import cv2
import math

from PIL import Image
from torchvision import transforms
import random

def prepare_image(image, resize=True):
    if resize and min(image.size) > 256:
        image = transforms.Resize(256)(image)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

def crop_center(hr, ph, pw):
    ih, iw = hr.shape[0:2]
    lr_patch_h, lr_patch_w = ph, pw
    ph = ih // 2 - lr_patch_h // 2
    pw = iw // 2 - lr_patch_w // 2

    return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

parser = argparse.ArgumentParser(description="finetune FT model")
parser.add_argument('--cfg', default='config/eval/kitti-S.json', help='experiment configure file name', required=True, type=str)
parser.add_argument('--model', default='../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-S.pth', help='checkpoint path', required=True, type=str)
parser.add_argument("--dataroot", type=str, default="./", help="data path")
parser.add_argument("--save_dir", type=str, default="../dataset/DCSZ_dataset/DCSZ_syn", help="save path")
parser.add_argument('--UW_size', default=0.6, type=float, help='world size')
args = parse_args(parser)

def convert(param):
    return {
    k.replace("searaft.", ""): v
        for k, v in param.items()
        if "searaft." in k
    }

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = RAFT(args).to(device)
load_ckpt(model, args.model)
x = torch.load("./99_flownet.pkl", map_location='cuda')
model.load_state_dict(convert(x), strict=True)

dataroot = args.dataroot
idsori = os.listdir(dataroot)
mse_total = 0
ids = []

for id in idsori:
    if ".txt" not in id:
        ids.append(id)
print(ids)
for id in ids:
    files = sorted(glob.glob(os.path.join(dataroot, id, '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0].split('_')[-1]))
    w_img = cv2.imread(files[-1])
    w_img = cv2.resize(w_img, (1280,1664), interpolation=cv2.INTER_LINEAR)
    w_img_d = cv2.resize(w_img, (320,416), interpolation=cv2.INTER_LINEAR)
    w_img_d = torch.from_numpy(w_img_d).permute(2,0,1).unsqueeze(0).to(device)
    save_dir = os.path.join(args.save_dir, id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '031.png')
    cv2.imwrite(save_path, w_img)
    w_img = torch.from_numpy(w_img).permute(2,0,1).unsqueeze(0).to(device)
    # print(w_img.shape)
    
    for i in range(len(files)-1):
        img_img = cv2.imread(files[i])
        img_img = cv2.resize(img_img, (1280,1664), interpolation=cv2.INTER_LINEAR)
        
        if args.UW_size != 0.6:
            scale = 1.0 / (0.7 + 0.3 * (i/31))  
            shape = img_img.shape
            I1 = cv2.resize(img_img, (int(img_img.shape[1]*scale), int(img_img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
            img_img = crop_center(I1, shape[0], shape[1])
        img_img_d = cv2.resize(img_img, (320,416), interpolation=cv2.INTER_LINEAR)
        img_img_d = torch.from_numpy(img_img_d).permute(2,0,1).unsqueeze(0).to(device)
        
        img_img = torch.from_numpy(img_img).permute(2,0,1).unsqueeze(0).to(device)
        # img_img_d = F.interpolate(img_img, (416,304), mode="bilinear", align_corners=True)
        
        with torch.no_grad():
            backward_flow = model(w_img_d / 255., img_img_d / 255., iters=args.iters, test_mode=True)['flow'][-1]
            backward_flow = F.interpolate(backward_flow, (1664,1280), mode="bilinear", align_corners=True) * 4
        w_pre_img_img, img_mask = getwarp(img_img, backward_flow)
        
        w_pre_img_img = w_pre_img_img[0].permute(1,2,0).cpu().numpy()
        
        save_path = os.path.join(save_dir, '{:03d}.png'.format(i))
        cv2.imwrite(save_path, w_pre_img_img)