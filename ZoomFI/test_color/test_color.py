import sys
sys.path.append('../core')
import torch
import torch.nn as nn
import scipy.stats as st
import os
import numpy as np
import glob
import torch.nn.functional as F

import argparse
from config.parser import parse_args
from swd import *

import cv2
import math

from PIL import Image
from torchvision import transforms
import random

def color_space_transform(input_color, fromSpace2toSpace):
    """
    Transforms inputs between different color spaces
    :param input_color: tensor of colors to transform (with NxCxHxW layout)
    :param fromSpace2toSpace: string describing transform
    :return: transformed tensor (with NxCxHxW layout)
    """
    dim = input_color.size()
    device = input_color.device

    # Assume D65 standard illuminant
    reference_illuminant = torch.tensor([[[0.950428545]], [[1.000000000]], [[1.088900371]]]).to(device)
    inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.000000000]], [[0.918357670]]]).to(device)

    if fromSpace2toSpace == "srgb2linrgb":
        limit = 0.04045
        transformed_color = torch.where(
            input_color > limit,
            torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
            input_color / 12.92
        )  # clamp to stabilize training

    elif fromSpace2toSpace == "linrgb2srgb":
        limit = 0.0031308
        transformed_color = torch.where(
            input_color > limit,
            1.055 * torch.pow(torch.clamp(input_color, min=limit), (1.0 / 2.4)) - 0.055,
            12.92 * input_color
        )

    elif fromSpace2toSpace in ["linrgb2xyz", "xyz2linrgb"]:
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        if fromSpace2toSpace == "linrgb2xyz":
            a11 = 10135552 / 24577794
            a12 = 8788810 / 24577794
            a13 = 4435075 / 24577794
            a21 = 2613072 / 12288897
            a22 = 8788810 / 12288897
            a23 = 887015 / 12288897
            a31 = 1425312 / 73733382
            a32 = 8788810 / 73733382
            a33 = 70074185 / 73733382
        else:
            # Constants found by taking the inverse of the matrix
            # defined by the constants for linrgb2xyz
            a11 = 3.241003275
            a12 = -1.537398934
            a13 = -0.498615861
            a21 = -0.969224334
            a22 = 1.875930071
            a23 = 0.041554224
            a31 = 0.055639423
            a32 = -0.204011202
            a33 = 1.057148933
        A = torch.Tensor([[a11, a12, a13],
                          [a21, a22, a23],
                          [a31, a32, a33]])

        input_color = input_color.view(dim[0], dim[1], dim[2] * dim[3])  # NC(HW)

        transformed_color = torch.matmul(A.to(device), input_color)
        transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])

    elif fromSpace2toSpace == "xyz2ycxcz":
        input_color = torch.mul(input_color, inv_reference_illuminant)
        y = 116 * input_color[:, 1:2, :, :] - 16
        cx = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        cz = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
        transformed_color = torch.cat((y, cx, cz), 1)

    elif fromSpace2toSpace == "ycxcz2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        cx = input_color[:, 1:2, :, :] / 500
        cz = input_color[:, 2:3, :, :] / 200

        x = y + cx
        z = y - cz
        transformed_color = torch.cat((x, y, z), 1)

        transformed_color = torch.mul(transformed_color, reference_illuminant)

    elif fromSpace2toSpace == "xyz2lab":
        input_color = torch.mul(input_color, inv_reference_illuminant)
        delta = 6 / 29
        delta_square = delta * delta
        delta_cube = delta * delta_square
        factor = 1 / (3 * delta_square)

        clamped_term = torch.pow(torch.clamp(input_color, min=delta_cube), 1.0 / 3.0).to(dtype=input_color.dtype)
        div = (factor * input_color + (4 / 29)).to(dtype=input_color.dtype)
        input_color = torch.where(input_color > delta_cube, clamped_term, div)  # clamp to stabilize training

        L = 116 * input_color[:, 1:2, :, :] - 16
        a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])

        transformed_color = torch.cat((L, a, b), 1)

    elif fromSpace2toSpace == "lab2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        a = input_color[:, 1:2, :, :] / 500
        b = input_color[:, 2:3, :, :] / 200

        x = y + a
        z = y - b

        xyz = torch.cat((x, y, z), 1)
        delta = 6 / 29
        delta_square = delta * delta
        factor = 3 * delta_square
        xyz = torch.where(xyz > delta, torch.pow(xyz, 3), factor * (xyz - 4 / 29))

        transformed_color = torch.mul(xyz, reference_illuminant)

    elif fromSpace2toSpace == "srgb2xyz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
    elif fromSpace2toSpace == "srgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "linrgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "srgb2lab":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "linrgb2lab":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "ycxcz2linrgb":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
    elif fromSpace2toSpace == "lab2srgb":
        transformed_color = color_space_transform(input_color, 'lab2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
    elif fromSpace2toSpace == "ycxcz2lab":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    else:
        sys.exit('Error: The color transform %s is not defined!' % fromSpace2toSpace)

    return transformed_color

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

def gauss_kernel(kernlen=84, nsig=12, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x1, x2):
        return torch.mean(x1 - x2)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_deltae(img_img, pre_img_img, cl, blur_rgb):
    img_rgb1 = img_img / 255.
    img_rgb2 = pre_img_img / 255.
    # img_rgb2 = pre_img_img.cpu() / 255.
    img_rgb1 = blur_rgb(img_rgb1)#.to(device)
    img_rgb2 = blur_rgb(img_rgb2)#.to(device)
    img_rgb1 = color_space_transform(img_rgb1, 'srgb2lab')
    img_rgb2 = color_space_transform(img_rgb2, 'srgb2lab')
    
    msg = cl(img_rgb1, img_rgb2).cpu().numpy()
    return msg

parser = argparse.ArgumentParser(description="test color")
parser.add_argument('--crop_img_path', default='./', help='cropped image path', required=True, type=str)
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

idsori = os.listdir(dataroot)
mse_total = 0
mse_total1 = 0
mse_total2 = 0
ids = []

blur_rgb = Blur(3).to(device)
cl = ColorLoss()

for id in idsori:
    if ".txt" not in id:
        ids.append(id)
print(ids)
for id in ids:
    files = sorted(glob.glob(os.path.join(dataroot, id, '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0].split('_')[-1]))
    y = []
    x = []
    for i in range(len(files)-1):
        img_img = cv2.imread(files[0])
        pre_img_img = cv2.imread(files[i+1])
        img_img = torch.from_numpy(img_img).permute(2,0,1).unsqueeze(0).to(device)
        pre_img_img = torch.from_numpy(pre_img_img).permute(2,0,1).unsqueeze(0).to(device)
        y.append(evaluate_deltae(img_img, pre_img_img, cl, blur_rgb))
        x.append(i)
    
    l = []
    x1 = np.array(x)
    y1 = np.array(y)
    y_max = 0
    y_min = 100
    m = 0
    mid_results = []
    coefficients = np.polyfit(x1, y1, 3)
    
    ccc = []
    ppc = []
    for k in range(len(y1)):
        ppp = coefficients[0] * (x1[k] ** 3) + coefficients[1] * (x1[k] ** 2) + coefficients[2] * (x1[k] ** 1) + coefficients[3]
        cha = np.abs(ppp - y1[k])
        ppc.append(ppp)
        ccc.append(cha)
    ccc = np.array(ccc)
    mean_i = np.mean(ccc)
    max_i = np.max(ccc)
    mse_total += max_i
    mse_total1 += mean_i
    print(f"max_i {id} : ", max_i)
    print(f"mean_i {id} : ", mean_i)
    print(f"coefficients {id} : ", coefficients)
    
mse_total = mse_total / len(ids)
print("mse_total_max : ", mse_total)
mse_total1 = mse_total1 / len(ids)
print("mse_total_mean : ", mse_total1)