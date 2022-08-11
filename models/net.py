
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class CovDecoder(nn.Module):
    def __init__(self, config_model):
        super(CovDecoder, self).__init__()

        nch_in = config_model.nch_in
        nch_h = config_model.nch_h 
        nch_out = config_model.nch_out 
        filter_size = config_model.filter_size 
        in_size = config_model.in_size
        out_size = config_model.out_size
        self.layerNo = config_model.layerNo
        layerNo = config_model.layerNo
        scale_x,scale_y = (out_size[0]/in_size[0])**(1./(layerNo+1)), (out_size[1]/in_size[1])**(1./(layerNo+1))
        hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (layerNo+1))] + [out_size]
        self.bn = nn.ModuleList([nn.BatchNorm2d(nch_h, affine=True) for i in range(self.layerNo+2)])
        #self.bn = nn.ModuleList([nn.InstanceNorm2d(nch_h, affine=True) for i in range(self.layerNo+2)])
        self.up = nn.ModuleList([nn.Upsample(size=hidden_size[i], mode='nearest') for i in range(self.layerNo+1)])
        self.conv = nn.ModuleList([nn.Conv2d(nch_h, nch_h, kernel_size=filter_size, padding=1, bias=True)\
                                    for i in range(self.layerNo+1)])
        self.conv_0 = nn.Conv2d(nch_in, nch_h, kernel_size=filter_size, padding=1,bias=True)
        self.conv_end = nn.Conv2d(nch_h, nch_out, kernel_size=1, padding=0, bias=True)
    
    def forward(self, x):
        x = self.up[0](x)
        x = self.conv_0(x)
        x = F.relu(x)
        x = self.bn[0](x)
        for layer in range(self.layerNo):
            x = self.up[layer+1](x)
            x = self.conv[layer](x)
            x = F.relu(x)
            x = self.bn[layer+1](x)
        x = self.conv[layer+1](x)
        x = F.relu(x)
        x = self.bn[layer+2](x)
        x = self.conv_end(x)
        return x