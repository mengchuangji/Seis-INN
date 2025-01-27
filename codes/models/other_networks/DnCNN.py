#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by chuangji meng 2022.9.29

import torch.nn as nn
import torch
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dep=20, num_filters=64, slope=0.2):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 20
            num_filters (int): number of filters in each layer, Default 64
        '''
        super(DnCNN, self).__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(slope, inplace=True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(slope, inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)

    def forward(self, x):
        y=x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        out = self.conv_last(x)

        return y-out


class DnCNN_R(nn.Module):
    def __init__(self, in_channels, out_channels, dep=20, num_filters=64, slope=0.2):

        super(DnCNN_R, self).__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(slope, inplace=True)
        mid_layer = []
        for ii in range(1, dep - 1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(slope, inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)


    def forward(self, x):
        y=x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        out = self.conv_last(x)
        return torch.cat((y-out[:,:1,:,:],out[:,1:,:,:]),1)

    # def forward(self, x):
    #     y = x
    #     out = self.dncnn(x)
    #     return y - out

