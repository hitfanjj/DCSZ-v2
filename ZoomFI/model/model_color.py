import sys
sys.path.append('./searaft/core')
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from model.zoomfi import * 
from model.zoomfi_small import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, args, model_size, load_root=None, local_rank=-1):
        if model_size=='full':
            self.flownet = IFNet_m(args)
        else:
            self.flownet = IFNet_m_small(args)
        self.args = args
        self.device()
        if load_root != None:
            self.load_model(load_root, rank=1) 
            print('pretrained loaded')

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
            self.flownet.load_state_dict(convert(torch.load(load_path)), strict=False)
        else:
           x = torch.load(load_path, map_location='cuda')['model_state_dict']
           #print(x.keys())
           self.flownet.load_state_dict(x, strict=False)
           
        
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
        return merge, mask, warped_img0_f, warped_img1
    