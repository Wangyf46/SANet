import torch.nn as nn
import torch 
import numpy as np
from torchvision import models
import ipdb

class SANet(nn.Module):
    def __init__(self, bn = False):
        super(SANet, self).__init__()
        self.seen = 0  ## TODO
        self.frontend_feat = [16, 'M', 32, 'M', 32, 'M', 16]
        self.backend_feat = [(64,9), 'T', (32,7), 'T', (16,5), 'T',
                             (16,3), (16,5)] 
        self.frontend = make_layers_frontend(self.frontend_feat)
        self.backend = make_layers_backend(self.backend_feat, 64)
        self.output_layer = Conv2d(16, 1, 1, same_padding = True)# TODO
        self._initialize_weights()        

    def forward(self, x): 
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers_frontend(cfg, in_channels = 3):
    layers = []
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            if i == 0:
                layers += [scale_aggregation_moudule(in_channels, v, 
                                                     isFirstLayer = True)] 
            else:
                layers += [scale_aggregation_moudule(in_channels, v)]   # TODO
            in_channels = 4 * v   # concat
    return  nn.Sequential(*layers)


def make_layers_backend(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'T':
            layers += [nn.ConvTranspose2d(in_channels, in_channels,
                                          kernel_size = 2, stride = 2)] # TODO
        else:
            out_channels = v[0]
            kernel_size = v[1]
            layers += [Conv2d(in_channels, out_channels, kernel_size, 
                              same_padding = True, In = True)]
            in_channels = out_channels # TODO
    return nn.Sequential(*layers)


class scale_aggregation_moudule(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 instance_norm = True, isFirstLayer = False):
        super(scale_aggregation_moudule, self).__init__()

        if isFirstLayer: # TODO
            self.branch1 = Conv2d(in_channels, out_channels, 1, 
                                  same_padding = True, In = instance_norm) 
            self.branch2 = Conv2d(in_channels, out_channels, 3, 
                                  same_padding = True, In = instance_norm)
            self.branch3 = Conv2d(in_channels, out_channels, 5, 
                                  same_padding = True, In = instance_norm)
            self.branch4 = Conv2d(in_channels, out_channels, 7, 
                                  same_padding = True, In = instance_norm)
        else:   
            inner_channels = int(in_channels / 2)
            self.branch1 = Conv2d(in_channels, out_channels, 1, 
                                  same_padding = True, In = instance_norm)
            self.branch2 = nn.Sequential(Conv2d(in_channels, inner_channels, 
                                   1, same_padding = True, In = instance_norm),
                                   Conv2d(inner_channels, out_channels, 3, 
                                     same_padding = True, In = instance_norm))
            self.branch3 = nn.Sequential(Conv2d(in_channels, inner_channels, 
                                   1, same_padding = True, In = instance_norm),
                                   Conv2d(inner_channels, out_channels, 5,
                                     same_padding = True, In = instance_norm))
            self.branch4 = nn.Sequential(Conv2d(in_channels,  inner_channels, 
                                   1, same_padding = True, In = instance_norm),
                                   Conv2d(inner_channels, out_channels, 7,
                                     same_padding = True, In = instance_norm))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        output = torch.cat((branch1, branch2, branch3, branch4), 1)
        return output

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, 
                 relu = True, same_padding = False, bn = False, In = False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding = padding)
        self.bn = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0, 
                                 affine = True) if bn else None     ## todo
        self.In = nn.InstanceNorm2d(out_channels, affine = True, 
                                   track_running_stats = True) if In else None
        self.relu = nn.ReLU(inplace = True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.In is not None:
            x = self.In(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
