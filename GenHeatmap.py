import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import Conv, Residual, Hourglass, Merge, Pool, upsample_recon

class GenHeatmap(nn.Module):
    def __init__(self, num_of_kp):
        super(GenHeatmap, self).__init__()
        #inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True
        ''''
        self.pre = nn.Sequential(
            Conv(3, 32, 3, 1, bn=True, relu=True),
            Conv(32, 32, 3, 1, bn=True, relu=True),
            Conv(32, 64, 3, 2, bn=True, relu=True),
            Conv(64, 64, 3, 1, bn=True, relu=True),
            Conv(64, 128, 3, 2, bn=True, relu=True),
            Conv(128, 128, 3, 1, bn=True, relu=True),
            Conv(128, 128, 2, 2, bn=True, relu=False),
            Conv(128, 128, 2, 2, bn=True, relu=False),
            Conv(128, 128, 2, 2, bn=False, relu=False)
        )
        '''
        self.conv3_32 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv32_32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv32_64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv64_64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv64_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv128_128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv128_128_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=2, bias=True)
        self.conv128_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=2, bias=True)
        self.conv128_128_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=2, bias=True)
        self.conv128_numKP = nn.Conv2d(in_channels=128, out_channels=num_of_kp, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)

    def forward(self, aefe_input):
        img = aefe_input
        out = self.conv3_32(img)
        out = self.conv32_32(out)
        out = self.conv32_64(out)
        out = self.conv64_64(out)
        out = self.conv64_128(out)
        out = self.conv128_128(out)
        out = self.conv128_128_1(out)
        out = self.conv128_128_2(out)
        out = self.conv128_128_3(out)
        out = self.conv128_numKP(out)

        return out

