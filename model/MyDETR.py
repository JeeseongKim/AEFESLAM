from torchvision.models import resnet50
import torch.nn.functional as F
import copy
from typing import Optional, List

from torch import nn, Tensor
from model.position_encoding import *
from model.layers import weights_init

#from model.AEFE_Transformer import *
from model.DETR_transformer import *
from model.utils import *
from model.dcn import DeformableConv2d

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        x = x
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

        return pos

class DETR_backbone(nn.Module):
    def __init__(self, hidden_dim=256):
        super(DETR_backbone, self).__init__()

        # Resnet50 backbone
        self.backbone = resnet50(pretrained=False)
        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

    def forward(self, inputs):
        # x = self.backbone.conv1(inputs)
        inputs = inputs.permute() #inputs = Rk + posRk (b,k,48,160)

        x = self.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.upsample(x)
        # x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.upsample(x)
        x = self.backbone.layer3(x)
        x = self.upsample(x)
        x = self.backbone.layer4(x)
        x = self.upsample(x)

        return x

class ENCinp_1(nn.Module):
    def __init__(self, my_height, my_width, dim1=2048, dim2=256):
        super(ENCinp_1, self).__init__()

        # Resnet50 backbone
        #self.backbone = resnet50(pretrained=False)
        #del self.backbone.fc
        #self.conv1 = nn.Conv2d(my_height*my_width, dim1, 1)
        #self.conv2 = nn.Conv2d(dim1, dim2, 1)
        #self.conv3 = nn.Conv2d(dim2, dim2, 1)

        self.linear1 = torch.nn.Linear(my_height*my_width, dim1)
        self.linear2 = torch.nn.Linear(dim1, dim2)
        self.linear3 = torch.nn.Linear(dim2, dim2)

    #def forward(self, inputs, pos):
    def forward(self, inputs):
        self.linear1.apply(weights_init)
        self.linear1.apply(weights_init)
        self.linear1.apply(weights_init)

        # x = self.backbone.conv1(inputs)
        inputs = inputs.flatten(2) #inputs = Rk + posRk (b,k,7680)
        #inputs = inputs.permute(0, 2, 1)
        #inputs = inputs.unsqueeze(3)

        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        #out = F.relu(self.conv3(x))
        out = x

        #input_pos = pos.flatten(2)
        #xx = self.linear1(input_pos)
        #xx = self.linear2(xx)
        #xx = self.linear3(xx)
        #pos_out = xx

        #return out.squeeze(3)
        #return out, pos_out
        return inputs, out


class ENCinp(nn.Module):
    def __init__(self, my_height, my_width, dim1=2048, dim2=256):
        super(ENCinp, self).__init__()

        # Resnet50 backbone
        self.backbone = resnet50(pretrained=False)
        del self.backbone.fc

        self.conv1 = nn.Conv2d(my_height*my_width, dim1, 1)
        self.conv2 = nn.Conv2d(dim1, dim2, 1)
        self.conv3_kp = nn.Conv2d(dim2, dim2, 1)
        self.conv3_f = nn.Conv2d(dim2, dim2, 1)

    def forward(self, inputs):
        # x = self.backbone.conv1(inputs)
        inputs = inputs.flatten(2) #inputs = Rk + posRk (b,k,7680)
        inputs = inputs.permute(0, 2, 1)
        inputs = inputs.unsqueeze(3)

        x = self.conv1(inputs)
        x = self.conv2(x)

        x_kp = F.relu(self.conv3_kp(x))
        x_f = F.relu(self.conv3_f(x))

        #x_kp = self.backbone.bn1(x_kp)
        #x_kp = self.backbone.relu(x_kp)

        #x_f = self.backbone.bn1(x_f)
        #x_f = self.backbone.relu(x_f)

        return x_kp.squeeze(3), x_f.squeeze(3)


class DETR4kp(nn.Module):
    def __init__(self, num_voters, hidden_dim=200, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR4kp, self).__init__()

        #Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_voters, hidden_dim)

        self.linear_class = MLP(hidden_dim, hidden_dim, 2, 3)

        #self.kp_embed = MLP(hidden_dim, hidden_dim, 2, 3)


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

        h = self.transformer(x.permute(2, 0, 1), trg).transpose(0, 1)

        hh = self.linear_class(h)

        # kp = torch.sigmoid(Hk_kp)
        kp = 1 / (1 + torch.exp(-1 * hh))
        kp[:, :, 0] = torch.round(kp[:, :, 0] * my_width).float()
        kp[:, :, 1] = torch.round(kp[:, :, 1] * my_height).float()

        return hh, kp


class DETR4kpNf(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR4kpNf, self).__init__()

        #Transformer
        self.transformer_kp = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.transformer_f = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_voters, hidden_dim)

        self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)
        self.linear_class_f = MLP(hidden_dim, hidden_dim, 256, 3)


    def forward(self, enc_kp, enc_f, my_height, my_width):

        '''
        src = sequence to the encoder (S, N, E)
        tgt = sequence to the decoder (T, N, E)
        output = (T, N, E)
        S: source sequence length (hw)
        T: target sequence length (num_voter)
        N: batch size
        E: feature number (hidden_dim)
        '''

        cur_batch = enc_kp.shape[0]

        trg_tmp = self.query_pos.weight
        trg = trg_tmp.unsqueeze(1).repeat(1, cur_batch, 1)

        input_kp = enc_kp.permute(2, 0, 1)
        input_f = enc_f.permute(2, 0, 1)

        h_kp = self.transformer_kp(input_kp, trg)
        h_f = self.transformer_f(input_f, trg)

        hh_kp = self.linear_class_kp(h_kp)
        hh_f = self.linear_class_f(h_f)

        desc = 1 / (1 + torch.exp(-1 * hh_f))

        kp = 1 / (1 + torch.exp(-10 * hh_kp))
        kp[:, :, 0] = torch.round(kp[:, :, 0] * my_width).float()
        kp[:, :, 1] = torch.round(kp[:, :, 1] * my_height).float()

        kp = kp.permute(1, 0, 2)
        desc = desc.permute(1, 0, 2)

        return kp, desc

class DETR_1E2D(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR_1E2D, self).__init__()

        #Transformer
        #self.transformer_kp = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        #self.transformer_f = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos_kp = nn.Embedding(num_voters, hidden_dim)
        self.query_pos_f = nn.Embedding(num_voters, hidden_dim)
        #self.query_pos = nn.Embedding(num_voters, hidden_dim)

        self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)
        self.linear_class_f = MLP(hidden_dim, hidden_dim, 256, 3)

        self.STE = StraightThroughEstimator()
        #self.linear = torch.nn.Linear(2, 2)

    def forward(self, encoder_input, my_height, my_width):


        '''
        src = sequence to the encoder (S, N, E)
        tgt = sequence to the decoder (T, N, E)
        output = (T, N, E)
        S: source sequence length (hw)
        T: target sequence length (num_voter)
        N: batch size
        E: feature number (hidden_dim)
        '''
        #self.transformer.apply(weights_init)
        self.linear_class_kp.apply(weights_init)
        self.linear_class_f.apply(weights_init)

        cur_batch = encoder_input.shape[0]

        #trg_tmp = self.query_pos.weight
        #trg = trg_tmp.unsqueeze(1).repeat(1, cur_batch, 1)

        trg_tmp_kp = self.query_pos_kp.weight
        trg_kp = trg_tmp_kp.unsqueeze(1).repeat(1, cur_batch, 1)
        trg_tmp_f = self.query_pos_f.weight
        trg_f = trg_tmp_f.unsqueeze(1).repeat(1, cur_batch, 1)

        input = encoder_input.permute(1, 0, 2)

        #enc_ouput, h_kp, h_f = self.transformer(src=input, tgt1=trg, tgt2=trg)
        enc_ouput, h_kp, h_f = self.transformer(src=input, tgt1=trg_kp, tgt2=trg_f)

        hh_kp = self.linear_class_kp(h_kp)
        hh_f = self.linear_class_f(h_f)

        myKP = hh_kp.permute(1, 0, 2)
        myDesc = hh_f.permute(1, 0, 2)

        #initial_param = torch.tensor([5, 5]).float()
        #my_param = torch.abs(self.linear(initial_param.cuda()))
        desc = torch.bernoulli(1 / (1 + torch.exp(-3 * myDesc)))
        desc = self.STE(desc)
        #desc = 1 / (1 + torch.exp(-10 * myDesc))
        #desc = torch.sigmoid(myDesc)

        #kp = 1 / (1 + torch.exp(-10 * myKP))
        kp = 1 / (1 + torch.exp(-3 * myKP))
        #kp = torch.sigmoid(myKP)
        kp[:, :, 0] = torch.round(kp[:, :, 0] * my_width).float()
        kp[:, :, 1] = torch.round(kp[:, :, 1] * my_height).float()

        #kp = kp.permute(1, 0, 2)
        #desc = desc.permute(1, 0, 2)

        return kp, desc

class DETR_1E1D(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR_1E1D, self).__init__()

        #Transformer
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_voters, hidden_dim)

        #self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)
        #self.linear_class_kp = torch.nn.Linear(256, 2)
        #self.linear_class_kp.apply(weights_init)

        self.linear_class_kp1 = torch.nn.Linear(256, 128)
        self.linear_class_kp2 = torch.nn.Linear(128, 64)
        self.linear_class_kp3 = torch.nn.Linear(64, 32)
        self.linear_class_kp4 = torch.nn.Linear(32, 16)
        self.linear_class_kp5 = torch.nn.Linear(16, 2)

        self.linear_class_kp1.apply(weights_init)
        self.linear_class_kp2.apply(weights_init)
        self.linear_class_kp3.apply(weights_init)
        self.linear_class_kp4.apply(weights_init)
        self.linear_class_kp5.apply(weights_init)

        #self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)
        #self.linear_class_f = MLP(hidden_dim, hidden_dim, 256, 3)

        #self.STE = StraightThroughEstimator()
        #self.linear = torch.nn.Linear(2, 2)

    def forward(self, encoder_input):

        '''
        src = sequence to the encoder (S, N, E)
        tgt = sequence to the decoder (T, N, E)
        output = (T, N, E)
        S: source sequence length (hw)
        T: target sequence length (num_voter)
        N: batch size
        E: feature number (hidden_dim)
        '''

        #self.transformer.apply(weights_init)
        #self.linear_class_kp.apply(weights_init)
        #self.linear_class_f.apply(weights_init)

        cur_batch = encoder_input.shape[0]
        trg_tmp = self.query_pos.weight
        trg = trg_tmp.unsqueeze(1).repeat(1, cur_batch, 1)
        input = encoder_input.permute(2, 0, 1)

        enc_ouput, h_kp = self.transformer(src=input, tgt1=trg)

        #hh_kp = self.linear_class_kp(h_kp)

        hh_kp = self.linear_class_kp1(h_kp)
        hh_kp = self.linear_class_kp2(hh_kp)
        hh_kp = self.linear_class_kp3(hh_kp)
        hh_kp = self.linear_class_kp4(hh_kp)
        hh_kp = self.linear_class_kp5(hh_kp)

        hh_kp = F.leaky_relu(hh_kp)
        myKP = hh_kp.permute(1, 0, 2)
        kp = 1 / (1 + torch.exp(-3 * myKP))

        return kp


class DETR_1E1D(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR_1E1D, self).__init__()

        #Transformer
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_voters, hidden_dim)

        #self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)
        #self.linear_class_kp = torch.nn.Linear(256, 2)
        #self.linear_class_kp.apply(weights_init)

        self.linear_class_kp1 = torch.nn.Linear(256, 128)
        self.linear_class_kp2 = torch.nn.Linear(128, 64)
        self.linear_class_kp3 = torch.nn.Linear(64, 32)
        self.linear_class_kp4 = torch.nn.Linear(32, 16)
        self.linear_class_kp5 = torch.nn.Linear(16, 2)

        self.linear_class_kp1.apply(weights_init)
        self.linear_class_kp2.apply(weights_init)
        self.linear_class_kp3.apply(weights_init)
        self.linear_class_kp4.apply(weights_init)
        self.linear_class_kp5.apply(weights_init)

        #self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)
        #self.linear_class_f = MLP(hidden_dim, hidden_dim, 256, 3)

        #self.STE = StraightThroughEstimator()
        #self.linear = torch.nn.Linear(2, 2)

    def forward(self, encoder_input):

        '''
        src = sequence to the encoder (S, N, E)
        tgt = sequence to the decoder (T, N, E)
        output = (T, N, E)
        S: source sequence length (hw)
        T: target sequence length (num_voter)
        N: batch size
        E: feature number (hidden_dim)
        '''

        #self.transformer.apply(weights_init)
        #self.linear_class_kp.apply(weights_init)
        #self.linear_class_f.apply(weights_init)

        cur_batch = encoder_input.shape[0]
        trg_tmp = self.query_pos.weight
        trg = trg_tmp.unsqueeze(1).repeat(1, cur_batch, 1)
        input = encoder_input.permute(2, 0, 1)

        enc_ouput, h_kp = self.transformer(src=input, tgt1=trg)

        #hh_kp = self.linear_class_kp(h_kp)

        hh_kp = self.linear_class_kp1(h_kp)
        hh_kp = self.linear_class_kp2(hh_kp)
        hh_kp = self.linear_class_kp3(hh_kp)
        hh_kp = self.linear_class_kp4(hh_kp)
        hh_kp = self.linear_class_kp5(hh_kp)

        hh_kp = F.leaky_relu(hh_kp)
        myKP = hh_kp.permute(1, 0, 2)
        kp = 1 / (1 + torch.exp(-3 * myKP))

        return kp

class DETR_KPnDesc(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR_KPnDesc, self).__init__()
        self.nheads = nheads
        #Transformer
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, return_intermediate_dec=True)
        #self.transformer.apply(weights_init)
        self.query_embed = nn.Embedding(num_voters, hidden_dim)
        #torch.nn.init.xavier_normal_(self.query_embed.weight)

        self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)

        self.linear_class_desc = nn.Linear(hidden_dim, 256)

        self.input_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        #self.input_proj = DeformableConv2d(256, hidden_dim, kernel_size=1)
        #self.get_answer_desc = torch.nn.Linear(nheads, 1)

        #self.linear_class_kp = torch.nn.Linear(256, 2)
        #torch.nn.init.xavier_normal_(self.linear_class_kp.weight)
        #self.linear_class_kp.apply(weights_init)

        #self.linear_class_desc = torch.nn.Linear(256, 256)

        #torch.nn.init.xavier_normal_(self.linear_class_desc.weight)
        #self.linear_class_desc.apply(weights_init)

        #self.STE = StraightThroughEstimator()
        #self.linear = torch.nn.Linear(2, 2)

        #self.kp_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        #self.kp_head = MaskHeadSmallConv(hidden_dim + nheads, [256, 256, 256], hidden_dim)
    def MySigmoid(self, x):
        return torch.exp(x)/(torch.exp(x)+1)

    def forward(self, src, pos):
        hs = self.transformer(src=self.input_proj(src), query_embed=self.query_embed.weight, pos_embed=pos)[0]

        #myKP = self.MySigmoid(self.linear_class_kp(hs))
        myKP = self.linear_class_kp(hs).sigmoid()

        #hs_1 = self.linear_class_kp_1(hs)
        #hs_2 = self.linear_class_kp_2(hs_1)
        #hs_3 = self.linear_class_kp_3(hs_2)
        #myKP = self.linear_class_kp_4(hs_3).sigmoid()

        #myKP = self.linear_class_kp(hs)
        #multi_kp = torch.cat([myKP[0], myKP[1], myKP[2], myKP[3]], dim=0)
        #final_kp = self.get_answer(multi_kp.permute(1, 2, 0)).permute(2, 0, 1).sigmoid()

        myDesc = torch.tanh(self.linear_class_desc(hs))
        #myDesc = self.linear_class_desc(hs)
        #multi_desc = torch.cat([myDesc[0], myDesc[1], myDesc[2], myDesc[3]], dim=0)
        # desc = 1 / (1 + torch.exp(-1 * myDesc))
        #desc = self.get_answer(multi_desc.permute(1, 2, 0)).permute(2, 0, 1)
        #final_desc = torch.tanh(desc)

        '''
        #myKP = self.linear_class_kp(h_kp)
        p_myKP = myKP.permute(1, 3, 0, 2)
        v_myKP = p_myKP.view(p_myKP.shape[0], p_myKP.shape[1], p_myKP.shape[2]*p_myKP.shape[3])
        final_kp = torch.sigmoid(self.get_answer_kp(v_myKP).permute(0, 2, 1))

        #myDesc = self.linear_class_desc(h_kp)
        p_myDesc = myDesc.permute(1, 3, 0, 2)
        v_myDesc = p_myDesc.view(p_myDesc.shape[0], p_myDesc.shape[1], p_myDesc.shape[2] * p_myDesc.shape[3])
        desc = self.get_answer_desc(v_myDesc).permute(0, 2, 1)
        final_desc = torch.tanh(desc) #-1~1
        '''

        #return final_kp, final_desc
        return myKP[-1], myDesc[-1]

class DETR_KPnDesc_only(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR_KPnDesc_only, self).__init__()
        self.nheads = nheads

        #Transformer
        self.transformer = Transformer_only(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, return_intermediate_dec=True)
        self.query_embed = nn.Embedding(num_voters, hidden_dim)
        self.linear_class_kp = MLP(hidden_dim, hidden_dim, 2, 3)
        self.linear_class_desc = nn.Linear(hidden_dim, 256)
        self.input_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)

    def MySigmoid(self, x):
        return torch.exp(x)/(torch.exp(x)+1)

    def forward(self, src):
        hs = self.transformer(src=self.input_proj(src), query_embed=self.query_embed.weight)[0]
        myKP = self.linear_class_kp(hs).sigmoid()
        myDesc = torch.tanh(self.linear_class_desc(hs))

        return myKP[-1], myDesc[-1]


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class DETR4f(nn.Module):
    def __init__(self, num_voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR4f, self).__init__()

        #Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        #output positional encodings (object queries)
        self.query_pos = nn.Embedding(num_voters, hidden_dim)

        self.linear_class = MLP(hidden_dim, hidden_dim, 256, 3)


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

        h = self.transformer(x.permute(2, 0, 1), trg).transpose(0, 1) #Highlight #(1, voters = 200, 256)

        hh = self.linear_class(h)

        desc = 1 / (1 + torch.exp(-1 * hh))

        #kp_sigmoid = torch.nn.Sigmoid()
        #kp = kp_sigmoid(hh)
        #kp[:, :, 0] = torch.round(kp[:, :, 0] * my_width).float()
        #kp[:, :, 1] = torch.round(kp[:, :, 1] * my_height).float()

        return hh, desc

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

'''
class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer_1 = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_layer_2 = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm_1 = nn.LayerNorm(d_model)
        decoder_norm_2 = nn.LayerNorm(d_model)

        self.decoder_1 = TransformerDecoder(decoder_layer_1, num_decoder_layers, decoder_norm_1, return_intermediate=return_intermediate_dec)
        self.decoder_2 = TransformerDecoder(decoder_layer_2, num_decoder_layers, decoder_norm_2, return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
    '''

'''
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed_1, query_embed_2):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        #bs = src.shape[1]
        query_embed_1 = query_embed_1.unsqueeze(1).repeat(1, bs, 1)
        query_embed_2 = query_embed_2.unsqueeze(1).repeat(1, bs, 1)
        #query_embed_1 = query_embed_1
        #query_embed_2 = query_embed_2

        #mask = mask.flatten(1)

        tgt_1 = torch.zeros_like(query_embed_1)
        tgt_2 = torch.zeros_like(query_embed_2)

        memory_1 = self.encoder(src)
        memory_2 = self.encoder(src)

        hs_1 = self.decoder_1(tgt_1, memory_1)
        hs_2 = self.decoder_2(tgt_2, memory_2)

        return hs_1.transpose(1, 2), memory_1.permute(1, 2, 0).view(bs, c, h, w),  hs_2.transpose(1, 2), memory_2.permute(1, 2, 0).view(bs, c, h, w)
'''

'''
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
'''

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