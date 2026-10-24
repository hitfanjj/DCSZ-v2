import sys
sys.path.append('core')
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.zoomfi import * 
from model.zoomfi_small import * 

import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
from losses.vgg import VGGLoss
from raft import RAFT
from utils.utils import load_ckpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math

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
    

class Model:
    def __init__(self, args, model_size, local_rank=-1):
        if model_size=='full':
            self.flownet = IFNet_m(args)
        else:
            self.flownet = IFNet_m_small(args)
        self.args = args
        #导入SEA-RAFT
        load_ckpt(self.flownet.searaft, args.model)
        self.device()
        nn.init.zeros_(self.flownet.maskblock1.lastconv.weight)
        nn.init.zeros_(self.flownet.maskblock1.lastconv.bias)
        self.load_pretrained_model("./pretrained/RIFE/", rank=-1)
        print('pretrained RIFE loaded')

        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3)
        
        self.epe = EPE()
        self.lap = LapLoss()  
        self.sobel = SOBEL()
        self.vgg = VGGLoss().cuda()
        print("DDP check", local_rank)
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)
        
    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_pretrained_model(self, path, rank=0, suffix=None):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if suffix is None:
            load_path = '{}/flownet.pkl'.format(path)
        else:
            load_path = '{}/{}_flownet.pkl'.format(path, suffix)

        state_dict = torch.load(load_path)

        if rank <0:
            self.flownet.load_state_dict(convert(state_dict), strict=False)
        else:
           self.flownet.load_state_dict(state_dict, strict=False) 

    def load_model(self, path, rank=0, suffix=None):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if suffix is None:
            load_path = path
        else:
            load_path = '{}/{}.pkl'.format(path, suffix)

        if rank <0:
            self.flownet.load_state_dict(convert(torch.load(load_path)), strict=True)
        else:
           x = torch.load(load_path, map_location='cuda')['model_state_dict']
           #print(x.keys())
           self.flownet.load_state_dict(x, strict=True)
           
        
    def save_model(self, path, rank, epoch, suffix=None):
        if rank == 0:
            if suffix is None:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.flownet.state_dict(),
                        'optimizer_state_dict': self.optimG.state_dict()
                    }, '{}/flownet.pkl'.format(path))
            else:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.flownet.state_dict(),
                        'optimizer_state_dict': self.optimG.state_dict()
                    }, '{}/{}_flownet.pkl'.format(path, suffix))

    def inference(self, img0, img1, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merge, warped_img0_f, warped_img1= self.flownet(imgs, timestep=timestep)
        return merge, mask, warped_img0_f, warped_img1#, mask_tea, merged_teacher

    def update(self, imgs, gt, epoch, timestep=0.5, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        
        flow, mask, merge, warped_img0_f, warped_img1 = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1], timestep=timestep)

        loss_l1 = (self.lap(merge, gt)).mean()
        loss_vgg = (self.vgg(merge, gt)).mean()

        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_vgg
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]

        return merge, {'loss_l1': loss_l1, 'loss_vgg':loss_vgg}
