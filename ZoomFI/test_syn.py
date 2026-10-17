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
from LDCSZ_model import Model
from skimage.metrics import structural_similarity as SSIM
import lpips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate(I0, I1, timestep):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    mid, mask, warped_img0_f, warped_img1 = model.inference(I0, I1, timestep=timestep)
    mid = mid[0]
    mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    mid = (mid * 255.).astype(np.uint8)
    imgs.append(mid)
    return imgs

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
    parser.add_argument('--cfg', default='config/eval/kitti-M.json', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', default='../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-M.pth', help='checkpoint path', required=True, type=str)
    parser.add_argument("--log_dir", type=str, default="./FI/checkpoints/RIFE/DR-RIFE-vgg/train_sdi_log", help="log path")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/DCSZ_dataset/DCSZ_syn", help="train data path")
    parser.add_argument('--save_dir', type=str, default='./syn_results_8/', help='where to save image results')
    args = parse_args(parser)

    args.save_dir = os.path.join(args.log_dir, args.save_dir)
    

    model = Model(args)
    model.load_model(args.log_dir, rank=0)

    model.eval()
    model.device()

    save_file = os.path.join(args.log_dir, 'metrics_syn_new_l.txt')
    sf = open(save_file, 'a')
    sf.write(f'{args.save_dir}\n')
    p = sf.tell()

    loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').cuda()    
    psnr_all_list = []
    ssim_all_list = []
    lpips_all_list = []

    data_dir = os.path.join(args.dataset_dir, "test_new")
    ids = os.listdir(data_dir)
    for id in ids:
        psnr_list = []
        ssim_list = []
        lpips_list = []
        files = sorted(glob.glob(os.path.join(os.path.join(data_dir, id), '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0]))
        print(os.path.join(data_dir, id), len(files))
        uw_image = cv2.imread(files[0])
        wide_image = cv2.imread(files[-1])
        uw_image = cv2.resize(uw_image, (1216,1632), interpolation=cv2.INTER_LINEAR)
        wide_image = cv2.resize(wide_image, (1216,1632), interpolation=cv2.INTER_LINEAR)
        os.makedirs(args.save_dir, exist_ok=True)
        
        I1 = wide_image

        shape = uw_image.shape

        start = 1
        end = 31

        I0 = uw_image


        save_img = I0
        os.makedirs(os.path.join(args.save_dir, id), exist_ok=True)
        save_path = osp.join(args.save_dir, id, '{:03d}.png'.format(start - 1))
        cv2.imwrite(save_path, save_img)

        save_img = I1
        os.makedirs(os.path.join(args.save_dir, id), exist_ok=True)
        save_path = osp.join(args.save_dir, id, '{:03d}.png'.format(end))
        cv2.imwrite(save_path, save_img)


        for ii in range(start, end, 1):
            timestep = ii / 31
            gt_image = cv2.imread(files[ii])
            gt_image = cv2.resize(gt_image, (1216,1632), interpolation=cv2.INTER_LINEAR)
            
            gif_imgs = [I0, I1]

            gif_imgs_temp = [gif_imgs[0], ]
            for i, (img_start, img_end) in enumerate(zip(gif_imgs[:-1], gif_imgs[1:])):
                interp_imgs = interpolate(img_start, img_end, timestep=timestep)
                gif_imgs_temp += interp_imgs
                gif_imgs_temp += [img_end, ]
            gif_imgs = gif_imgs_temp

            save_img = gif_imgs[1]

            save_path = osp.join(args.save_dir, id, '{:03d}.png'.format(ii))
            cv2.imwrite(save_path, save_img)

            pred = gif_imgs[1]
            gt = gt_image
            
            psnr = -10 * math.log10(np.mean((gt/255. - pred/255.) * (gt/255. - pred/255.)))
            psnr_list.append(psnr)
            psnr_all_list.append(psnr)

            ssim = SSIM(pred, gt, win_size=11, data_range=255, channel_axis=2, gaussian_weights=True)
            ssim_list.append(ssim)
            ssim_all_list.append(ssim)

            lpips_ = calc_lpips(pred, gt, loss_fn_alex_v1, 255)
            lpips_list.append(lpips_)
            lpips_all_list.append(lpips_)
            sf.write(f'{id}, {ii}, PSNR:{np.mean(psnr_list)}, SSIM:{np.mean(ssim_list)} LPIPS:{np.mean(lpips_list)}\n')

        sf.write(f'{id}, PSNR:{np.mean(psnr_list)}, SSIM:{np.mean(ssim_list)} LPIPS:{np.mean(lpips_list)}\n')
        sf.write('\n')
    
    sf.write(f'Average, PSNR:{np.mean(psnr_all_list)}, SSIM:{np.mean(ssim_all_list)} LPIPS:{np.mean(lpips_all_list)}\n')
    print('Average', 'PSNR:', np.mean(psnr_all_list), 'SSIM:', np.mean(ssim_all_list), 'LPIPS:', np.mean(lpips_all_list))
    sf.close()
    





    

