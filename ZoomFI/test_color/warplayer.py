import torch
backwarp_tenGrid = {}
backwarp_tenPartial = {}

def getwarp(tenInput, tenFlow):
    tenInput_shape = tenInput.shape
    width = tenInput_shape[-1]
    height = tenInput_shape[-2]
    channel = tenInput_shape[-3]
    frame = tenInput_shape[-4]
    tenInput = tenInput.view(-1, channel, height, width)

    device = tenInput.device
    key = str(tenFlow.shape) + str(device)
    if key not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[key] = torch.cat([tenHor, tenVer], 1).to(device)

    if key not in backwarp_tenPartial:
        backwarp_tenPartial[key] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[key]], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[key] + tenFlow).permute(0, 2, 3, 1),
                                                mode='bilinear', padding_mode='zeros', align_corners=True)

    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0

    tenOutput = tenOutput[:, :-1, :, :].view(-1, channel, height, width)
    tenMask = tenMask.view(-1, 1, height, width)

    return tenOutput.contiguous() * tenMask.contiguous(), tenMask.contiguous()