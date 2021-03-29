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

from misc import NestedTensor
from position_encoding import *


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()

        #Resnet50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        #Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        #spatial positional encodings
        self.position_embedding = build_position_encoding(hidden_dim)

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        self.linear_class = nn.Linear(hidden_dim, num_classes)

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
        h = self.conv(x)

        #positional encoding
        H, W = h.shape[-2:]
        pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1), self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)], dim=-1).flatten(0, 1).unsqueeze(1)
        #pos = self.position_embedding(inputs, ) #What is mask????????

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

        trg = self.query_pos.unsqueeze(1)

        h = self.transformer(src, trg).transpose(0, 1)
        decoder_out = self.linear_class(h)

        return attention_map, decoder_out

