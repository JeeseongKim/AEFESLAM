import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import Conv, Residual, Hourglass, Merge, Pool, upsample_recon

class FeatureDesc(torch.nn.Module):
    def __init__(self, img_width, img_height, feature_dimension):
        super(FeatureDesc, self).__init__()
        self.elu = torch.nn.ELU(inplace=True)
        hw = img_width * img_height
        self.conv_hw_2049 = nn.Conv2d(in_channels=hw, out_channels=2049, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv_2049_2049 = nn.Conv2d(in_channels=2049, out_channels=2049, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv_2049_1024 = nn.Conv2d(in_channels=2049, out_channels=1024, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv_1024_1024 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv_1024_512 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv_512_256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        #self.conv_256_256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv_256_fd = nn.Conv2d(in_channels=256, out_channels=feature_dimension, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv_fd_fd_1 = nn.Conv2d(in_channels=feature_dimension, out_channels=feature_dimension, kernel_size=2, stride=1, padding=1, dilation=2, bias=True)
        self.conv_fd_fd_2 = nn.Conv2d(in_channels=feature_dimension, out_channels=feature_dimension, kernel_size=2, stride=1, padding=1, dilation=2, bias=True)
        self.conv_fd_fd_3 = nn.Conv2d(in_channels=feature_dimension, out_channels=feature_dimension, kernel_size=2, stride=1, padding=1, dilation=2, bias=True)


    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        x = x.permute(0, 2, 1).cuda()
        x = x.unsqueeze(3)

        out = self.conv_hw_2049(x)
        out = self.conv_2049_2049(out)
        out = self.conv_2049_1024(out)
        out = self.conv_1024_1024(out)
        out = self.conv_1024_512(out)
        out = self.conv_512_256(out)
        #out = self.conv_256_256(out)
        out = self.conv_256_fd(out)
        out = self.conv_fd_fd_1(out)
        out = self.conv_fd_fd_2(out)
        out = self.conv_fd_fd_3(out) #(b,f,k,1)

        out = out.squeeze(3) #(b, f, k)
        out = out.permute(0, 2, 1).cuda() #(b,k,f)

        dn =torch.norm(out, p=2, dim=2)
        desc = out.div(torch.unsqueeze(dn, 2)) #(b,k,f) # Divide by norm to normalize.

        return desc