from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import math
import torch
from torch import nn
import torch.nn.functional as F

from misc import NestedTensor
from position_encoding import *


class DETR(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()

        #Resnet50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        #Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_voters, hidden_dim))

        #spatial positional encodings
        self.position_embedding = build_position_encoding(hidden_dim)

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.linear_class = nn.Linear(hidden_dim, num_voters)

        self.kp_embed = MLP(hidden_dim, hidden_dim, 2, 3)

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        #channel: 2048 -> 256 (hidden_dim)
        h = self.conv(x) #(b, hidden, h/8, w/8) = (b, 256, 12, 39)

        #positional encoding
        H, W = h.shape[-2:]
        pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1), self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)], dim=-1).flatten(0, 1).unsqueeze(1)
        #pos = (hw, 1, 256) = (468, 1, 256)#pos = self.position_embedding(inputs, ) #What is mask????????

        # propagate through transformer
        src = pos + 0.1 * h.flatten(2).permute(2, 0, 1)  # encoder input

        attention_map_1 = src.permute(1, 0, 2)
        attention_map_2 = torch.transpose(attention_map_1, 1, 2)
        attention_map = torch.matmul(attention_map_1, attention_map_2)
        #attention map generation
        #attention_1 = (pos + 0.1*h.flatten(2).permute(2, 0, 1)).squeeze(1) #(468,256)
        #attention_map = torch.matmul(attention_1, torch.transpose(attention_1, 0, 1))

        #propagate through transformer
        #src = pos + 0.1 * h.flatten(2).permute(2, 0, 1) #encoder input

        trg = self.query_pos.unsqueeze(1) #(200 = voters, 1, 256)

        h = self.transformer(src, trg).transpose(0, 1) #Highlight #(1, voters = 200, 256)
        #decoder_out = self.linear_class(h)
        outputs_kp = self.kp_embed(h).sigmoid()
        outputs_kp[:, :, 0] = torch.round(outputs_kp[:, :, 0] * inputs.shape[3])
        outputs_kp[:, :, 1] = torch.round(outputs_kp[:, :, 1] * inputs.shape[2])

        return attention_map, h, outputs_kp

class simple_rsz(nn.Module):
    def __init__(self, inp_channel, oup_channel):
        super(simple_rsz, self).__init__()
        self.hidden_dim = oup_channel
        self.inp_channel = inp_channel
        self.linear = nn.Linear(self.inp_channel, self.hidden_dim)

    def forward(self, inputs):
        x = self.linear(inputs)

        return x

class MakeDesc(nn.Module):
    def __init__(self, inp_channel, oup_channel):
        super(MakeDesc, self).__init__()
        self.desc_sz = oup_channel
        self.inp_channel = inp_channel
        self.linear = nn.Linear(self.inp_channel, self.desc_sz)

    def forward(self, inputs):
        x = self.linear(inputs)

        return x

class Recon_MakeDesc(nn.Module):
    def __init__(self, inp_channel, oup_channel):
        super(Recon_MakeDesc, self).__init__()
        self.desc_sz = inp_channel
        self.hidden_dim = oup_channel
        self.linear = nn.Linear(self.desc_sz, self.hidden_dim)

    def forward(self, inputs):
        x = self.linear(inputs)

        return x

class Recon_Detection(nn.Module):
    def __init__(self, inp_channel, oup_channel):
        super(Recon_Detection, self).__init__()
        self.hidden_dim = oup_channel
        self.kp = inp_channel #2
        self.linear_2_32 = nn.Linear(self.kp, 32)
        self.linear_32_64 = nn.Linear(32, 64)
        self.linear_64_128 = nn.Linear(64, 128)
        self.linear_128_256 = nn.Linear(128, self.hidden_dim)

    def forward(self, inputs):
        x = self.linear_2_32(inputs)
        x = self.linear_32_64(x)
        x = self.linear_64_128(x)
        x = self.linear_128_256(x)

        return x




class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x