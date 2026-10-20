import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .vgg19_loss import PerceptualLoss

# from torchmetrics import PeakSignalNoiseRatio
# from torchmetrics.functional import structural_similarity_index_measure as ssim

# PSNR = PeakSignalNoiseRatio().cuda()
# vgg = PerceptualLoss().cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False
            
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss


def vgg_loss(example, prediction):
    return vgg(prediction['image'], example['y'])[0]

def style_loss(example, prediction):
    return vgg(prediction['image'], example['y'])[1]

def perceptual_loss(example, prediction):
    return sum(vgg(prediction['image'], example['y']))

def l1_loss(example, prediction):
    return torch.mean(torch.abs(prediction['image'] - example['y']))

def l1_warped_loss(example, prediction):
    loss = torch.zeros(1, dtype=torch.float32)
    if 'x0_warped' in prediction:
        loss += torch.mean(torch.abs(prediction['x0_warped'] - example['y']))
    if 'x1_warped' in prediction:
        loss += torch.mean(torch.abs(prediction['x1_warped'] - example['y']))
    return loss

def l2_loss(example, prediction):
    return torch.mean(torch.square(prediction['image'] - example['y']))

def ssim_loss(example, prediction):
    return ssim(prediction['image'], example['y']) # to do : max_val=1.0

def psnr_loss(example, prediction):
    return PSNR(prediction['image'], example['y'])

def get_loss(loss_name):
    if loss_name == 'l1':
        return l1_loss
    elif loss_name == 'l2':
        return l2_loss
    elif loss_name == 'ssim':
        return ssim_loss
    elif loss_name == 'vgg':
        return vgg_loss
    elif loss_name == 'style':
        return style_loss
    elif loss_name == 'psnr':
        return psnr_loss
    elif loss_name == 'l1_warped':
        return l1_warped_loss
    elif loss_name == 'perceptual':
        return perceptual_loss
    else:
        raise ValueError('Invalid loss function %s' % loss_name)

def get_loss_op(loss_name):
    loss = get_loss(loss_name)
    return lambda example, prediction: loss(example, prediction)

def get_weight_op(weight_schedule):
    return lambda iterations: weight_schedule(iterations)

def create_losses(loss_names, loss_weight=None):
    losses = dict()
    for name in (loss_names): # to do : loss_weight
        """#unique_values = np.unique(weight_schedule.values)
        #if len(unique_values) == 1: #and unique_values[0] == 1.0:
        #    weighted_name = name
        #else:
        #    weighted_name = 'k*' + name
        #losses[weighted_name] = (get_loss_op(name), get_weight_op(weight_schedule)) # to do
        #print(f"name {str(name)}")"""
        losses[name] = (get_loss_op(name))
    return losses

def training_losses(loss_names, loss_weights=None, loss_weight_schedules=None, loss_weight_parameters=None):
    weight_schedules = [] # to do
    """if not loss_weights:
        for weight_schedule, weight_parameters in zip(loss_weight_schedules, loss_weight_parameters):
            weight_schedules.append(weight_schedule(**weight_parameters))
    else:
        for loss_weight in loss_weights:
            weight_parameters = {
                'boundaries': [0],
                'values': 2 * [loss_weight,]
            }
            weight_schedules.append(torch.optim.lr_scheduler.ConstantLR(optimizer)) # to do : lr parameter"""
    return create_losses(loss_names, weight_schedules)

if __name__ == '__main__':
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (3, 3, 256, 256))).float().to(device)
    ternary_loss = Ternary()
    print(ternary_loss(img0, img1).shape)
