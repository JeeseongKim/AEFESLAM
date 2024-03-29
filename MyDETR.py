from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
#from torchvision.models import resnet50
from model.resnet import *
import torchvision.transforms as T
import math
import torch
from torch import nn
import torch.nn.functional as F
import copy
from typing import Optional, List

from torch import nn, Tensor
from misc import NestedTensor
from position_encoding import *
import matplotlib.pyplot as plt

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, input, cur_batch, h, w):
        x = input
        mask = torch.ones((cur_batch, h, w), dtype=torch.bool).cuda()
        for m in mask:
            m[: input.shape[1], :input.shape[2]] = False
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return m, pos

class DETR_backbone(nn.Module):
    def __init__(self, hidden_dim=256):
        super(DETR_backbone, self).__init__()

        # Resnet50 backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, inputs):
        # x = self.backbone.conv1(inputs)

        x = self.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x


class DETR2(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR2, self).__init__()

        #Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_voters, hidden_dim)

        self.linear_class = MLP(hidden_dim, hidden_dim, 2, 3)

        self.kp_embed = MLP(hidden_dim, hidden_dim, 2, 3)


    def forward(self, x, my_width, my_height):

        '''
        src = sequence to the encoder (S, N, E)
        tgt = sequence to the decoder (T, N, E)
        output = (T, N, E)
        S: source sequence length (hw)
        T: target sequence length (num_voter)
        N: batch size
        E: feature number (hidden_dim)
        '''


        cur_batch = x.shape[0]
        trg_tmp = self.query_pos.weight
        trg = trg_tmp.unsqueeze(1).repeat(1, cur_batch, 1)

        h = self.transformer(x.permute(1, 0, 2), trg).transpose(0, 1) #Highlight #(1, voters = 200, 256)

        hh = self.linear_class(h)

        #kp_sigmoid = torch.nn.Sigmoid()
        #kp = kp_sigmoid(hh)
        #kp[:, :, 0] = torch.round(kp[:, :, 0] * my_width).float()
        #kp[:, :, 1] = torch.round(kp[:, :, 1] * my_height).float()

        return hh

class DETR(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()

        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        #Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        #self.transformer = Transformer

        #output positional encodings (object queries)
        #self.query_pos = nn.Parameter(torch.rand(num_voters, hidden_dim))
        self.query_pos = nn.Embedding(num_voters, hidden_dim)

        #self.linear_class = nn.Linear(hidden_dim, 160)
        #self.linear_class = MLP(hidden_dim, hidden_dim, 160, 3)
        self.linear_class = MLP(hidden_dim, hidden_dim, 2, 3)

        self.kp_embed = MLP(hidden_dim, hidden_dim, 2, 3)


    def forward(self, x):

        #channel: 2048 -> 256 (hidden_dim)
        h = self.input_proj(x) #(b, 256, 12, 40)

        '''
        src = sequence to the encoder (S, N, E)
        tgt = sequence to the decoder (T, N, E)
        output = (T, N, E)
        S: source sequence length (hw)
        T: target sequence length (num_voter)
        N: batch size
        E: feature number (hidden_dim)
        '''

        #positional encoding
        positional_encoding = PositionEmbeddingSine()
        mask, pos = positional_encoding(x, x.shape[0], x.shape[2], x.shape[3])

        src = (h.flatten(2) + pos.flatten(2)).permute(2, 0, 1) #(hw, b, 256)
        src = src.permute(1, 0, 2)
        src_t = torch.transpose(src, 1, 2)
        attention_map = torch.matmul(src, src_t)
        attention_score = torch.sum(attention_map, dim=1) #(b, hw=480)
        attention_score = attention_score/math.sqrt(h.shape[1])
        #attention_score = attention_score * 1e-4 * 0.01
        #attention_score = torch.softmax(attention_score, dim=1)

        #getattentionMap = h.flatten(2).permute(2, 0, 1) #src
        #src = getattentionMap

        #attention_map_1 = getattentionMap.permute(1, 2, 0)
        #attention_map_2 = torch.transpose(getattentionMap, 1, 2)
        #attention_map = torch.matmul(attention_map_1, attention_map_2).permute(2, 1, 0)

        #batch = 2
        #trg = self.query_pos.weight.unsqueeze(1) #(200 = voters, 1, 256)
        cur_batch = x.shape[0]
        trg_tmp = self.query_pos.weight
        trg = trg_tmp.unsqueeze(1).repeat(1, cur_batch, 1)

        h = self.transformer(src.permute(1, 0, 2), trg).transpose(0, 1) #Highlight #(1, voters = 200, 256)

        #h = self.transformer(src, mask, self.query_pos, pos[-1])
        hh = self.linear_class(h)

        #kp = torch.sigmoid(hh)
        #kp[:, :, 0] = torch.round(kp[:, :, 0] * x.shape[3]).float()
        #kp[:, :, 1] = torch.round(kp[:, :, 1] * x.shape[2]).float()

        return attention_map, attention_score, h, hh

class DETR_origin(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR_origin, self).__init__()

        #Resnet50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        #x = self.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        #x = self.backbone.maxpool(x)

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

class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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
        self.linear1 = nn.Linear(self.inp_channel, 2048)
        #self.linear2 = nn.Linear(4096, 2048)
        #self.linear3 = nn.Linear(2048, 1024)
        #self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(2048, 512)
        self.linear6 = nn.Linear(512, 256)

    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        #x = F.relu(self.linear2(x))
        #x = F.relu(self.linear3(x))
        #x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))

        return x

class Recon_MakeDesc(nn.Module):
    def __init__(self, inp_channel, oup_channel):
        super(Recon_MakeDesc, self).__init__()
        self.desc_sz = inp_channel
        self.hidden_dim = oup_channel
        self.linear1 = nn.Linear(self.desc_sz, oup_channel)
        #self.linear1 = nn.Linear(self.desc_sz, 256)
        #self.linear2 = nn.Linear(256, 1024)
        #self.linear3 = nn.Linear(256, 4096)
        #self.linear4 = nn.Linear(4096, oup_channel)

    def forward(self, inputs):
        x = (self.linear1(inputs))
        #x = (self.linear2(x))
        #x = (self.linear3(x))
        #x = (self.linear4(x))

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
        self.linear_256_256_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_256_256_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_256_480 = nn.Linear(self.hidden_dim, 480)

    def forward(self, inputs):
        x = self.linear_2_32(inputs)
        x = self.linear_32_64(x)
        x = self.linear_64_128(x)
        x = self.linear_128_256(x)
        x = self.linear_256_256_1(x)
        x1 = self.linear_256_256_2(x)
        x2 = self.linear_256_480(x1)

        return x1,  x2


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