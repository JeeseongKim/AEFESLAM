import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import Conv, Residual, Hourglass, Merge, Pool, upsample_recon

class L2net_R2D2(nn.Module):
    def __init__(self, num_of_kp):
        super(L2net_R2D2, self).__init__()
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
        self.conv128_2 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)

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
        out = torch.square(out)
        out = self.conv128_2(out)
        out = torch.softmax(out)


        return out


class GCNv2(torch.nn.Module):
    def __init__(self):
        super(GCNv2, self).__init__()
        self.elu = torch.nn.ELU(inplace=True)

        #self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)

        self.conv4_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        # Descriptor
        self.convF_1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.convF_2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Detector
        self.convD_1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.convD_2 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.pixel_shuffle = torch.nn.PixelShuffle(16)

    def forward(self, x):

        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))

        x = self.elu(self.conv3_1(x))
        x = self.elu(self.conv3_2(x))

        x = self.elu(self.conv4_1(x))
        x = self.elu(self.conv4_2(x))

        # Descriptor xF
        #xF = self.elu(self.convF_1(x))
        #desc = self.convF_2(xF)
        #dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        #desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        # Detector xD
        xD = self.elu(self.convD_1(x))
        det = self.convD_2(xD).sigmoid()
        det = self.pixel_shuffle(det)

        #return desc, det
        return det