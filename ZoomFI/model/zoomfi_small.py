import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine_zoomfi import *
from model.warplayer_withmask import getwarp

# change:
from raft import RAFT
from model.softmax_splatting.softsplat import FunctionSoftsplat

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
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 9, 4, 2, 1)

        self.apply(self._init_weights)

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
        flow_d2 = tmp[:, 5:9] * scale * 2
        return flowb, flow_d2, mask

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

class IFNet_m_small(nn.Module):
    def __init__(self, args):
        super(IFNet_m_small, self).__init__()
        self.searaft = RAFT(args).to(device)
        self.args = args

        self.maskblock1 = IFBlock(27, c=128)
        # self.maskblock1 = IFBlock(17, c=128)
        self.contextnet = Contextnet()
        self.unet = Unet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.metric = Metric()

    def forward(self, x, scale=[1,1,1], timestep=0.5, returnflow=False):
        timestep = (x[:, :1].clone() * 0 + 1) * timestep
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        merged = []
        flow = None 
        x0d = F.interpolate(img0, scale_factor=0.125, mode="bilinear", align_corners=True)
        x1d = F.interpolate(img1, scale_factor=0.125, mode="bilinear", align_corners=True)
        timestep_4 = (x0d[:, :1].clone() * 0 + 1) * timestep[0,0,0,0]
        
        coarse_flow, _ = calc_flow(self.args, self.searaft, torch.cat((x0d, x1d), dim=0), torch.cat((x1d, x0d), dim=0))
        coarse_flow01 = coarse_flow[:1]
        coarse_flow10 = coarse_flow[1:]
        
        coarse_flow1t = coarse_flow10.clone().detach() * (1-timestep_4.clone().detach())
        coarse_flow0t = coarse_flow01.clone().detach() * timestep_4.clone().detach()

        coarse_flowt0 = FunctionSoftsplat(coarse_flow10 * timestep_4, coarse_flow10.clone().detach() * (1 - timestep_4), None, 'average')
        mask0 = torch.where(coarse_flowt0 == 0, 0, 1)[:, :1]
        coarse_flowt0 = torch.where(coarse_flowt0 == 0, coarse_flow10 * timestep_4, coarse_flowt0)
        coarse_flowt1 = FunctionSoftsplat(coarse_flow01 * (1 - timestep_4), coarse_flow01.clone().detach() * timestep_4, None, 'average')
        warped_img0_f = FunctionSoftsplat(x0d, coarse_flow0t, None, 'average')
        warped_img1_f = FunctionSoftsplat(x1d, coarse_flow1t, None, 'average')
        warped_img0 = warp(x0d, coarse_flowt0) * mask0 + (1-mask0) * warped_img0_f
        warped_img1, _ = getwarp(x1d, coarse_flowt1)
        flow = torch.cat((coarse_flowt0, coarse_flowt1), 1)
        flow_f = torch.cat((coarse_flow0t, coarse_flow1t), 1)
        
        flow_d, flow_d2, mask = self.maskblock1(torch.cat((x0d, x1d, timestep_4, warped_img0, warped_img1, warped_img0_f, warped_img1_f, flow_f), dim=1), flow)
        mask = torch.sigmoid(mask)
        flow = flow_d + flow + flow_d2 * (1-mask0)
        flow1 = F.interpolate(flow[:,:2], scale_factor=8, mode="bilinear", align_corners=True) * 8
        flow2 = F.interpolate(flow[:,2:4], scale_factor=8, mode="bilinear", align_corners=True) * 8
        flow = torch.cat((flow1, flow2), 1)
        warped_img0_final = warp(img0, flow[:,:2])
        warped_img1_final = warp(img1, flow[:,2:4]) 

        c0 = self.contextnet(img0, flow[:,:2])
        c1 = self.contextnet(img1, flow[:,2:4])
        
        mask = F.interpolate(mask, scale_factor=8, mode="bilinear", align_corners=True)

        merge = warped_img0_final * mask + warped_img1_final * (1-mask)

        for i in range(3):
            merged.append(merge)

        tmp = self.unet(img0, img1, warped_img0_final, warped_img1_final, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merge = torch.clamp(merge + res, 0, 1)
        return flow, mask, merge, warped_img0, warped_img1
 
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x