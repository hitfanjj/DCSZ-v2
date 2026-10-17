import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *

# change:
from raft import RAFT

from model.softsplat import softsplat

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            # conv(c, c),
            # conv(c, c),
            # conv(c, c),
            # conv(c, c),
            # conv(c, c),
            # conv(c, c),
            conv(c, c),
        )
        # self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

        self.apply(self._init_weights)

        # # change：增加一层卷积，训练时初始参数置0
        # self.zeroconv = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, flowb, scale=1):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        #if flow != None:
        flowb = F.interpolate(flowb, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flowb), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)

        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flowb = tmp[:, :4] * scale * 2
        # flowf = tmp[:, 4:8] * scale * 2
        mask = tmp[:, 4:5]
        return flowb, mask

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

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

class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()
        self.alpha = nn.Parameter(-torch.ones(1, 1, 1, 1))

    def forward(self, ten_first, ten_second, tenFlow):
        return self.alpha * F.l1_loss(ten_first, warp(ten_second, tenFlow), reduction='none').mean(1, keepdim=True)
    
class forward_feat_Block(nn.Module):
    def __init__(self, in_planes, c=64):
        super(forward_feat_Block, self).__init__()
        self.convblock = nn.Sequential(
            conv(in_planes, c),
            conv(c, c),
            conv(c, in_planes)
        )
        self.alpha = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        x = self.convblock(x)
        return self.alpha * x

class IFNet_m(nn.Module):
    def __init__(self, args):
        super(IFNet_m, self).__init__()
        self.searaft = RAFT(args).to(device)
        self.args = args

        self.maskblock1 = IFBlock(27, c=128)
        # self.maskblock1 = IFBlock(17, c=128)
        self.contextnet = Contextnet()
        self.unet = Unet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = Metric()

    def forward(self, x, scale=[1,1,1], timestep=0.5, returnflow=False):
        #t = timestep#.item()
        #print(timestep)
        timestep = (x[:, :1].clone() * 0 + 1) * timestep
        img0 = x[:, :3]
        b,c,h,w = img0.shape
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        flow = None 
        loss_distill = 0
        #with torch.no_grad():
        x0d = F.interpolate(img0, scale_factor=0.25, mode="bilinear", align_corners=True)
        x1d = F.interpolate(img1, scale_factor=0.25, mode="bilinear", align_corners=True)
        coarse_flow01, _ = calc_flow(self.args, self.searaft, x0d, x1d)
        coarse_flow10, _ = calc_flow(self.args, self.searaft, x1d, x0d)
        coarse_flow01 = F.interpolate(coarse_flow01, scale_factor=4, mode="bilinear", align_corners=True) * 4
        coarse_flow10 = F.interpolate(coarse_flow10, scale_factor=4, mode="bilinear", align_corners=True) * 4

        coarse_flowt1 = coarse_flow01 * (1.0-timestep)
        coarse_flowt0 = coarse_flow10 * timestep
        # coarse_flow0t = coarse_flow01.clone().detach() * timestep.clone().detach()
        # coarse_flow1t = coarse_flow10.clone().detach() * (1-timestep.clone().detach())
        # coarse_flow0t = coarse_flow01.clone().detach() * timestep.clone().detach()
        coarse_flow1t = coarse_flow10.clone().detach() * (1-timestep.clone().detach())
        coarse_flow0t = coarse_flow01.clone().detach() * timestep.clone().detach()
        # coarse_flow1t = coarse_flow10 * (1.0-timestep)
        # coarse_flow0t = coarse_flow01 * timestep
        #coarse_flow = torch.cat((coarse_flow0t,coarse_flowt1),1)
        #warped_img0 = warp(img0, coarse_flowt0)

        z0 = self.metric(img0, img1, coarse_flow01)
        z1 = self.metric(img1, img0, coarse_flow10)
        
        warped_img0_f = forward_warp0(img0, coarse_flow0t, z0)
        warped_img1_f = forward_warp0(img1, coarse_flow1t, z1)
        warped_img0_f = FunctionSoftsplat(img0, coarse_flow0t, None, 'average')
        warped_img1_f = FunctionSoftsplat(img1, coarse_flow1t, None, 'average')
        warped_img0 = warp(img0, coarse_flowt0) 
        warped_img1 = warp(img1, coarse_flowt1)
        flow = torch.cat((coarse_flowt0, coarse_flowt1), 1)
        flow_f = torch.cat((coarse_flow0t, coarse_flow1t), 1)
        
        flow_d, mask = self.maskblock1(torch.cat((img0, img1, timestep, warped_img0, warped_img1, warped_img0_f, warped_img1_f, flow_f), dim=1), flow)
        # flow_d, mask = self.maskblock1(torch.cat((img0, img1, timestep, warped_img0, warped_img1), dim=1), flow)
        # flow_d, mask = self.maskblock1(torch.cat((img0, img1, timestep, warped_img0, warped_img1, warped_img0_f, warped_img1_f), dim=1), flow)
        #flow_d, mask = self.maskblock(torch.cat((img0, img1, timestep, warped_img0, warped_img1), dim=1), torch.cat((img0, img1, timestep, warped_img0_f, warped_img1_f, flow_f), dim=1), flow)
        mask = torch.sigmoid(mask)
        flow = flow_d + flow
        #warped_img0_f = forward_warp0(img0, flow[:,:2], z0)
        warped_img0 = warp(img0, flow[:,:2])
        warped_img1 = warp(img1, flow[:,2:4]) 

        c0 = self.contextnet(img0, flow[:,:2])
        c1 = self.contextnet(img1, flow[:,2:4])

        # feat, merge0 = self.maskblock1(torch.cat((img0, img1, timestep, warped_img0_f, warped_img1), dim=1), flow, scale=1)

        merge = warped_img0 * mask + warped_img1 * (1-mask)

        for i in range(3):
            merged.append(merge)

        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merge = torch.clamp(merge + res, 0, 1)
        # return warped_img0, warped_img1
        return flow, mask, merge, warped_img0, warped_img1
 

def get_z(flow_fix):
    z = torch.where(flow_fix == 0, 0, 1).detach().sum(1, keepdim=True) / 2
    #print(z.shape)
    zt0, zt1 = z[:1], z[1:]
    return zt0,zt1

def forward_warp0(tenIn, tenFlow, z=None):
    # if z is None:
    #     z = torch.ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]).to(tenIn.device)
    # else:
    #     z = torch.where(z == 0, -20, 1)
    #out = softsplat1.FunctionSoftsplat(tenInput=tenIn, tenFlow=tenFlow, tenMetric=z, strType='softmax')
    out = softsplat(tenIn, tenFlow, tenMetric=z, strMode='soft')
    #out = softsplat(tenIn, tenFlow, tenMetric=z, strMode='avg')
    return out
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


