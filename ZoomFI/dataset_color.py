import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as tra
import glob

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
import math
import torch.nn as nn
import torch.nn.functional as F
 
def crop_center(hr, ph, pw):
    ih, iw = hr.shape[0:2]
    lr_patch_h, lr_patch_w = ph, pw
    ph = ih // 2 - lr_patch_h // 2
    pw = iw // 2 - lr_patch_w // 2
    return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]
 
def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

class DZDataset(Dataset):
    def __init__(self, dataset_name, data_root, if_crop=False, color_model="./" , batch_size=1):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.data_root = data_root
        self.if_crop = if_crop

        self.train_cache = {}
        
        self.color_model = Agent().eval()
        xx = torch.load(color_model)
        self.color_model.load_state_dict(xx, strict=True)
        
        aditional_ids = os.listdir(data_root)
        for aditional_id in aditional_ids:
            dl_ids = os.listdir(os.path.join(data_root, aditional_id))
            for dl_id in dl_ids:
                self.train_cache[dl_id] = []
                files = sorted(glob.glob(os.path.join(data_root, aditional_id, dl_id, '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0].split('_')[-1]))
            
                for file in files:
                    self.train_cache[dl_id].append(file)

        self.load_data()
        print(len(list(self.meta_data.keys())))
    def __len__(self):
        return 40000 * 1
        
    def load_data(self):
        self.meta_data = self.train_cache

    def crop_center(self, hr, ph, pw):
        ih, iw = hr.shape[0:2]
        lr_patch_h, lr_patch_w = ph, pw
        ph = ih // 2 - lr_patch_h // 2
        pw = iw // 2 - lr_patch_w // 2
        return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]
    
    def crop_center_gray(self, hr, ph, pw):
        ih, iw = hr.shape[0:2]
        lr_patch_h, lr_patch_w = ph, pw
        ph = ih // 2 - lr_patch_h // 2
        pw = iw // 2 - lr_patch_w // 2
        return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w]
    
    def metric_gamma(self, inp, gamma):
        return inp.sign() * torch.pow(inp.abs().clamp(0.0001,1), gamma)
    
    def __getitem__(self, index): 
        index = index % len(self.meta_data.keys())

        img0s = []
        gts = []
        img1s = []
        timesteps = []
        
        img0 = cv2.imread(self.meta_data[list(self.meta_data.keys())[index]][0])
        if img0 is None:
            print("img None:", self.meta_data[list(self.meta_data.keys())[index]][0])
        img0 = cv2.resize(img0, (1216,1632), interpolation=cv2.INTER_LINEAR) / 255.
        
        shape = img0.shape
        if self.if_crop:
            scale = 0.8 / 0.65
            img0_c = cv2.resize(img0, (int(img0.shape[1]*scale), int(img0.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
            img0_c = crop_center(img0_c, shape[0], shape[1])
        else:
            img0_c = img0
        
        img1_c = cv2.imread(self.meta_data[list(self.meta_data.keys())[index]][-1])
        img1_c = cv2.resize(img1_c, (1216,1632), interpolation=cv2.INTER_LINEAR) / 255.

        ii = random.randint(1,  32 - 2)
        timestep = ii / 31
        t = ii / 31
        gt = cv2.imread(self.meta_data[list(self.meta_data.keys())[index]][ii])
        gt = cv2.resize(gt, (1216,1632), interpolation=cv2.INTER_LINEAR) / 255.
        
        shape = gt.shape
        if self.if_crop:
            scale = (0.8 + 0.2 * ii/31) / (0.65 + 0.35 * ii/31)
            gt_c = cv2.resize(gt, (int(gt.shape[1]*scale), int(gt.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
            gt_c = crop_center(gt_c, shape[0], shape[1])
        else:
            gt_c = gt
        
        img0s.append(img0_c)
        img1s.append(img1_c)
        gts.append(gt_c)
        timesteps.append(timestep)
       
        img0s = np.array(img0s)
        img1s = np.array(img1s)
        gts = np.array(gts)

        img1s = torch.from_numpy(img1s).permute(0, 3, 1, 2)
        if self.if_crop:
            scale = 0.8 / 0.6
        else:
            scale = 1.0 / 0.6
        scale = 0.8 / 0.6
        I0 = cv2.resize(img0_c, (int(img0_c.shape[1]*scale), int(img0_c.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        shape = img0_c.shape
        I0 = crop_center(I0, shape[0], shape[1])
        I0 = torch.from_numpy(I0).unsqueeze(0).permute(0, 3, 1, 2).float()
  
        
        img1s = self.color_model.inference(I0, img1s, 1)["output"]
        img1s = img1s.permute(0, 2, 3, 1).numpy()
        
        gts = torch.from_numpy(gts).permute(0, 3, 1, 2)
        
        gts = self.color_model.inference(I0, gts, t)["output"]
        
        gts = gts.permute(0, 2, 3, 1).numpy()
        if self.dataset_name == 'train':
            if random.uniform(0, 1) < 0.5:  
                img0s = img0s[:, :, :, ::-1]
                img1s = img1s[:, :, :, ::-1]
                gts = gts[:, :, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0s = img0s[:, ::-1]
                img1s = img1s[:, ::-1]
                gts = gts[:, ::-1]

            if random.uniform(0, 1) < 0.5:
                img0s = img0s[:, :, ::-1]
                img1s = img1s[:, :, ::-1]
                gts = gts[:, :, ::-1]
                
        img0s = torch.from_numpy(img0s.copy()).permute(0, 3, 1, 2).float()
        img1s = torch.from_numpy(img1s.copy()).permute(0, 3, 1, 2).float()
        gts = torch.from_numpy(gts.copy()).permute(0, 3, 1, 2).float()
        
        timesteps = torch.tensor(timesteps).reshape(gts.shape[0], 1, 1, 1)

        return torch.cat((img0s, img1s, gts), 1), timesteps
    

def tanh01(x):
    return (torch.tanh(x) + 1.0) * 0.5

def mapping(params: torch.Tensor, value_range):
    """
    Linear map from [0,1] to [l,r].
    params: (N, K, ...), entries in [0,1]
    value_range: (l, r)
    """
    l, r = value_range
    return l + (r - l) * params

class CCMFilter(nn.Module):
    """
    3x3 color correction matrix with row normalization.
    Each element in cfg.ccm_range; rows normalized to sum 1.
    """
    def __init__(self, ccm_range = (0, 2.0)):
        super().__init__()
        self.name = 'CCMFilter'
        self.range = ccm_range
        self.param_num = 9  # 3x3

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        ccm = mapping(params[:, :9], self.range).view(1, 3, 3)      # allow negative entries
        ccm = ccm / (ccm.sum(dim=-1, keepdim=True) + 1e-8)       # row-normalize
        # Apply: out[n,h,w,:] = ccm[n] @ img[n,h,w,:]
        x = img.permute(0, 2, 3, 1)[..., None, :]  # (N,H,W,1,3)
        M = ccm[:, None, None]                     # (N,1,1,3,3)
        out = torch.sum(x * M, dim=-1)             # (N,H,W,3)
        out = out.permute(0, 3, 1, 2)
        return torch.clamp(out, 0.0, 1.0)
    
    def get_param(self, params):
        ccm = mapping(params, self.range).view(1, 3, 3)      # allow negative entries
        ccm = ccm / (ccm.sum(dim=-1, keepdim=True) + 1e-8)       # row-normalize
        return torch.flatten(ccm)
    
class CCMFilter_wb(nn.Module):
    """
    3x3 color correction matrix with row normalization.
    Each element in cfg.ccm_range; rows normalized to sum 1.
    """
    def __init__(self, ccm_range = (0, 2.0)):
        super().__init__()
        self.name = 'CCMFilter'
        self.range = ccm_range
        self.param_num = 12  # 3x3
        self.oneccm = torch.tensor([[1,0,0,0,1,0,0,0,1]])

    def forward(self, img, params, t):
        # print("params ccm : ", params.shape)
        ccm = mapping(params[:, :9], self.range).view(1, 3, 3)      # allow negative entries
        ccmb = mapping(params[:, 9:], (-0.1, 0.1)).view(1, 1, 1, 3) * t      # allow negative entries
        # print("ccm : ", ccm)
        ccm = ccm / (ccm.sum(dim=-1, keepdim=True) + 1e-8)       # row-normalize
        ccm = ccm * t + (1-t) * self.oneccm.view(1, 3, 3)
        # Apply: out[n,h,w,:] = ccm[n] @ img[n,h,w,:]
        x = img.permute(0, 2, 3, 1)[..., None, :]  # (N,H,W,1,3)
        M = ccm[:, None, None]                     # (N,1,1,3,3)
        # b = ccmb[:, None, None]                     # (N,1,1,1,3)
        out = torch.sum(x * M, dim=-1)             # (N,H,W,3)
        out = out + ccmb
        # print("out : ", out.shape)
        out = out.permute(0, 3, 1, 2)
        return torch.clamp(out, 0.0, 1.0)
    
    def get_param(self, params):
        ccm = mapping(params, self.range).view(1, 3, 3)      # allow negative entries
        ccmb = mapping(params, (-0.1, 0.1)).view(1, 3, 3)      # allow negative entries
        ccm = ccm / (ccm.sum(dim=-1, keepdim=True) + 1e-8)       # row-normalize
        return torch.flatten(ccm), torch.flatten(ccmb)
    
def metric_gamma(inp, gamma):
        return inp.sign() * torch.pow(torch.clamp(inp.abs(), 1e-6), gamma)
    
class GammaFilter_fc(nn.Module):
    """Gamma correction: output = img^gamma, gamma in [1/gamma_range, gamma_range]."""
    def __init__(self, g_range=3):
        super().__init__()
        self.name = 'GammaFilter'
        self.range = (1.0 / g_range, g_range)
        self.param_num = 9

    def forward(self, img, params, t, gamma_uw):
        gamma = mapping(params[:, :3], self.range) * t + (1-t) * gamma_uw  # (N,3)
        alpha = mapping(params[:, 3:6], (0.5,1.5)) * t + (1-t)  # (N,3)
        beta = mapping(params[:, 6:9], (-0.1,0.1)) * t  # (N,3)
        outr = (metric_gamma(alpha[:, 0] * img[:, 0:1] + beta[:, 0], gamma[:, 0]) - metric_gamma(beta[:, 0], gamma[:, 0])) / (metric_gamma(alpha[:, 0] + beta[:, 0], gamma[:, 0]) - metric_gamma(beta[:, 0], gamma[:, 0]))
        outg = (metric_gamma(alpha[:, 1] * img[:, 1:2] + beta[:, 1], gamma[:, 1]) - metric_gamma(beta[:, 1], gamma[:, 1])) / (metric_gamma(alpha[:, 1] + beta[:, 1], gamma[:, 1]) - metric_gamma(beta[:, 1], gamma[:, 1]))
        outb = (metric_gamma(alpha[:, 2] * img[:, 2:3] + beta[:, 2], gamma[:, 2]) - metric_gamma(beta[:, 2], gamma[:, 2])) / (metric_gamma(alpha[:, 2] + beta[:, 2], gamma[:, 2]) - metric_gamma(beta[:, 2], gamma[:, 2]))
        return torch.cat((outr, outg, outb), dim=1)

    def get_param(self, params):
        gamma = mapping(params[:, :3], self.range)  # (N,3)
        alpha = mapping(params[:, 3:6], (0.5,1.5))  # (N,3)
        beta = mapping(params[:, 6:9], (-0.1,0.1))  # (N,3)
        return torch.flatten(gamma), torch.flatten(alpha), torch.flatten(beta)
    
class nGammaFilter_fc(nn.Module):
    """Gamma correction: output = img^gamma, gamma in [1/gamma_range, gamma_range]."""
    def __init__(self, g_range=3):
        super().__init__()
        self.name = 'GammaFilter'
        self.range = (1.0 / g_range, g_range)
        self.param_num = 9

    def forward(self, img, params, t):
        gamma = mapping(params[:, :3], self.range)  # (N,3)
        alpha = mapping(params[:, 3:6], (0.5,1.5)) * t + (1-t)  # (N,3)
        beta = mapping(params[:, 6:9], (-0.1,0.1)) * t  # (N,3)
        outr = (metric_gamma(img[:, 0:1] * (metric_gamma(alpha[:, 0] + beta[:, 0], gamma[:, 0]) - metric_gamma(beta[:, 0], gamma[:, 0])) + metric_gamma(beta[:, 0], gamma[:, 0]), 1/gamma[:, 0]) - beta[:, 0]) / alpha[:, 0]
        outg = (metric_gamma(img[:, 1:2] * (metric_gamma(alpha[:, 1] + beta[:, 1], gamma[:, 1]) - metric_gamma(beta[:, 1], gamma[:, 1])) + metric_gamma(beta[:, 1], gamma[:, 1]), 1/gamma[:, 1]) - beta[:, 1]) / alpha[:, 1]
        outb = (metric_gamma(img[:, 2:3] * (metric_gamma(alpha[:, 2] + beta[:, 2], gamma[:, 2]) - metric_gamma(beta[:, 2], gamma[:, 2])) + metric_gamma(beta[:, 2], gamma[:, 2]), 1/gamma[:, 2]) - beta[:, 2]) / alpha[:, 2]
        return torch.cat((outr, outg, outb), dim=1)

    def get_param(self, params):
        gamma = mapping(params[:, :3], self.range)  # (N,3)
        alpha = mapping(params[:, 3:6], (0.5,1.5))  # (N,3)
        beta = mapping(params[:, 6:9], (-0.1,0.1))  # (N,3)
        return gamma, torch.flatten(alpha), torch.flatten(beta)
    
class GammaFilter(nn.Module):
    """Gamma correction: output = img^gamma, gamma in [1/gamma_range, gamma_range]."""
    def __init__(self, g_range=3):
        super().__init__()
        self.name = 'GammaFilter'
        self.range = (1.0 / g_range, g_range)
        self.param_num = 3

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        gamma = mapping(params[:, :1], self.range)  # (N,1)
        alpha = mapping(params[:, 1:2], (0.5,1.5))  # (N,1)
        beta = mapping(params[:, 2:3], (-0.1,0.1))  # (N,1)
        return (metric_gamma(alpha * img + beta, gamma) - metric_gamma(beta, gamma)) / (metric_gamma(alpha + beta, gamma) - metric_gamma(beta, gamma))

    def get_param(self, params):
        gamma = mapping(params[:, :1], self.range)  # (N,1)
        alpha = mapping(params[:, 1:2], (0.5,1.5))  # (N,1)
        beta = mapping(params[:, 2:3], (-0.1,0.1))  # (N,1)
        return gamma, beta, alpha
    
class nGammaFilter(nn.Module):
    """Gamma correction: output = img^gamma, gamma in [1/gamma_range, gamma_range]."""
    def __init__(self, g_range=3):
        super().__init__()
        self.name = 'GammaFilter'
        self.range = (1.0 / g_range, g_range)
        self.param_num = 3

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        gamma = mapping(params[:, :1], self.range)  # (N,1)
        alpha = mapping(params[:, 1:2], (0.5,1.5))  # (N,1)
        beta = mapping(params[:, 2:3], (-0.1,0.1))  # (N,1)
        return (metric_gamma(img * (metric_gamma(alpha + beta, gamma) - metric_gamma(beta, gamma)) + metric_gamma(beta, gamma), 1/gamma) - beta) / alpha

    def get_param(self, params):
        gamma = mapping(params[:, :1], self.range)  # (N,1)
        alpha = mapping(params[:, 1:2], (0.5,1.5))  # (N,1)
        beta = mapping(params[:, 2:3], (-0.1,0.1))  # (N,1)
        return gamma, beta, alpha

class ImprovedWhiteBalanceFilter(nn.Module):
    """
    Per-channel scale with gray-world normalization.
    scale_c in [1/wb_range, wb_range], then row-normalize by luma weights.
    """
    def __init__(self, wb_range = 1.1):
        super().__init__()
        self.name = 'ImprovedWhiteBalanceFilter'
        self.range = (1.0 / wb_range, wb_range)
        self.param_num = 3

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        scale = mapping(params, self.range)                  # (N,3)
        denom = 1e-6 + 0.27*scale[:, 0] + 0.67*scale[:, 1] + 0.06*scale[:, 2]
        scale = scale / denom
        return img * scale.view(1, 3, 1, 1)   

    def get_param(self, params):
        return mapping(params, self.range)                  # (N,3)
    
class nImprovedWhiteBalanceFilter(nn.Module):
    """
    Per-channel scale with gray-world normalization.
    scale_c in [1/wb_range, wb_range], then row-normalize by luma weights.
    """
    def __init__(self, wb_range = 1.1):
        super().__init__()
        self.name = 'ImprovedWhiteBalanceFilter'
        self.range = (1.0 / wb_range, wb_range)
        self.param_num = 3

    def forward(self, img: torch.Tensor, params: torch.Tensor):
        scale = mapping(params, self.range)                  # (N,3)
        denom = 1e-6 + 0.27*scale[:, 0] + 0.67*scale[:, 1] + 0.06*scale[:, 2]
        scale = scale / denom
        return img / scale.view(1, 3, 1, 1)   

    def get_param(self, params):
        return mapping(params, self.range)                  # (N,3)

class ImgOnlyParamAgent(nn.Module):
    def __init__(self, output_dim, img_channels=3, img_base=32, latent_dim=128):
        super().__init__()
        ch = img_base
        # self.enc0 = nn.Sequential(
        #     nn.Conv2d(img_channels, ch, 3, 2, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(ch, ch, 3, 2, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(ch, ch, 3, 2, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        self.down_sample = nn.AdaptiveAvgPool2d((64, 64))
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, ch, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc1 = nn.Linear(ch * 8, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim * 2, output_dim),
        )

    def forward(self, img):
        img = self.down_sample(img)
        feat = self.enc3(self.enc2(self.enc1(img)))
        gap = F.adaptive_avg_pool2d(feat, 1).view(img.size(0), -1)
        gmp = F.adaptive_max_pool2d(feat, 1).view(img.size(0), -1)
        hidden = F.relu(self.fc1(torch.cat([gap, gmp], dim=1)), inplace=True)
        return tanh01(self.decoder(self.fc2(hidden)))
    
class Agent(nn.Module):
    def __init__(self, output_path="./"):
        super().__init__()
        self.output_path = output_path
        # self.filter = nn.ModuleList([nGammaFilter_fc(), nImprovedWhiteBalanceFilter(), CCMFilter(), CCMFilter_wb(), CCMFilter(), ImprovedWhiteBalanceFilter(), GammaFilter_fc()])
        self.filter = nn.ModuleList([nGammaFilter_fc(), CCMFilter_wb(), GammaFilter_fc()])
        # self.filter = nn.ModuleList([nGammaFilter_fc(), CCMFilter(), GammaFilter_fc()])
        # self.filter = nn.ModuleList([nGammaFilter(), CCMFilter(), GammaFilter()])
        self.param_num = 30
        self.module_cnt = len(self.filter)
        self.eos_token = self.module_cnt
        self.max_bri = 0.9
        self.param_net = ImgOnlyParamAgent(output_dim=self.param_num, img_channels=3)

    def apply_filter(self, image, params, t):
        output = image.unsqueeze(0)
        penalty = output.new_tensor(0.0)
        # print("params : ", params.shape)
        output = self.filter[0](output, params[:, 0:9], t)
        gamma, _, _ = self.filter[0].get_param(params[:, 0:9])
        output = self.filter[1](output, params[:, 9:21], t)
        output = self.filter[2](output, params[:, 21:30], t, gamma)
        retouch_mean = torch.mean(output, dim=(1, 2, 3)).unsqueeze(-1)
        if torch.any((retouch_mean < 0.01) | (retouch_mean > self.max_bri)):
            penalty = penalty + (
                (
                    torch.clamp(0.01 - retouch_mean, min=0) * 100
                    + torch.clamp(retouch_mean - self.max_bri, min=0)
                )
                * 10
            ).mean()
        return output, penalty

    def forward(self, x, inp, t, writer=None, is_val=False, save_path=""):
        batch_size = x.shape[0]
        params = self.param_net(x)
        penalty = x.new_zeros(batch_size)
        output = []
        # print("params : ", params.shape)
        for index, image in enumerate(inp):
            filtered, stop_penalty = self.apply_filter(image, params[index:index+1], t)
            penalty[index] += stop_penalty
            output.append(filtered)
        early_stop_penalty = x.new_zeros(batch_size)
        return {
            "output": torch.cat(output, dim=0),
            "penalty": penalty,
            "early_stop_penalty": early_stop_penalty,
        }

    @torch.no_grad()
    def inference(self, x, inp, t, save_path="", x_origin=None):
        source = inp
        source = source.to(next(self.parameters()).device)
        params = self.param_net(x)
        outputs = []
        params_all = []
        for index, image in enumerate(source):
            current = image.unsqueeze(0)
            selected_params = []
            output = self.filter[0](current, params[:, 0:9], t)
            gamma, _, _ = self.filter[0].get_param(params[:, 0:9])
            output = self.filter[1](output, params[:, 9:21], t)
            output = self.filter[2](output, params[:, 21:30], t, gamma)

            outputs.append(output)
            params_all.append(params)
        return {
            "output": torch.cat(outputs, dim=0),
            "params_return": params_all
        }
        