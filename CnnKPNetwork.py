import torch
from torch import nn
import numpy as np
import time
import torch
from torch import nn
from model.layers import Conv, Residual, Hourglass, Merge, Pool, upsample_recon

class CNNForKP(nn.Module):
    def __init__(self, inp_h, inp_w, num_of_kp):
        super(CNNForKP, self).__init__()

        self.inp_h = inp_h
        self.inp_W = inp_w
        self.num_of_kp = num_of_kp

        self.pre = nn.Sequential(
            Conv(3, 8, 3, 1, bn=True, relu=True),
            Conv(8, 16, 3, 1, bn=True, relu=True),
            Conv(16, 32, 3, 1, bn=True, relu=True),
            Conv(32, 64, 3, 1, bn=True, relu=True),
            Residual(64, 128),
            Residual(128, 128),
            Residual(128, num_of_kp)
        )


    def forward(self, imgs):
        x = imgs
        x = self.pre(x)

        return x
