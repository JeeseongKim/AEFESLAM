import torch
from torch import nn
from model.layers import *
from model.Heatmap import HeatmapLoss
from model.layers import weights_init
from model.dcn import DeformableConv2d
from torchvision.models import resnet50

class StackedHourglassForKP_DCN(torch.nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassForKP_DCN, self).__init__()
        self.nstack = nstack
        self.pool = Pool(2, 2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.pre = nn.Sequential(
            DConv(3, 64, 7, 2, bn=True, relu=True),
            Residual_dcn(64, 128),
            Residual_dcn(128, 128),
            Residual_dcn(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass_dcn(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual_dcn(inp_dim, inp_dim),
                DConv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                DConv(inp_dim, oup_dim, 1, relu=False, bn=False), #256 -> 200
            )for i in range(nstack)
         ])

        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])

    def forward(self, imgs):
        x = imgs #(b,3,w,h)
        x = self.pre(x) #(b,192,96,128) #(b, 192, 57, 192)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature) #--> heatmap prediction
            append(preds)
            if (i < self.nstack - 1):
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        heatmap = torch.stack(combined_hm_preds, 1)[:, self.nstack - 1, :, :, :]
        heatmap = self.pool(heatmap)

        return heatmap

class StackedHourglassForKP(torch.nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassForKP, self).__init__()

        self.nstack = nstack
        '''
        self.pre1 = Conv(3, 64, 7, 2, bn=True, relu=True)

        self.pre2 = Residual(64, 128)
        #self.pre3 = Pool(2, 2)
        #self.pre4 = Residual(128, 128)
        self.pre5 = Residual(128, inp_dim)
        '''
        self.pool = Pool(2, 2)
        #self.final_conv = Conv(256, 256, 7, 2, bn=True, relu=True)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.pre = nn.Sequential(
            #Conv(3, 64, 3, 1, bn=True, relu=True),
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            #Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
                #Hourglass(1, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                Conv(inp_dim, oup_dim, 1, relu=False, bn=False), #256 -> 200
            )for i in range(nstack)
         ])

        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])
        #self.relu = torch.nn.ReLU()
        #self.backbone = resnet50()
        #del self.backbone.fc

    def forward(self, imgs):
        x = imgs #(b,3,w,h)
        x = self.pre(x) #(b,192,96,128) #(b, 192, 57, 192)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature) #--> heatmap prediction
            append(preds)
            if (i < self.nstack - 1):
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        heatmap = torch.stack(combined_hm_preds, 1)[:, self.nstack - 1, :, :, :]
        heatmap = self.pool(heatmap)
        #heatmap = self.relu(heatmap)
        return heatmap

class StackedHourglassForKP_1(torch.nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassForKP_1, self).__init__()

        self.nstack = nstack

        self.pool = Pool(2, 2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        '''
        self.pre = nn.Sequential(
            Residual(3, 3),
            Residual(3, 128),
            Pool(2, 2),
            Residual(128, 128),
            Conv(128, inp_dim, 7, 2, bn=True, relu=True),
        )
        '''
        self.pre = nn.Sequential(
            Conv(3, 128, 3, 1, bn=True, relu=True),
            Pool(2, 2),
            Residual(128, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim),
        )
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
                #Hourglass(1, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                Conv(inp_dim, oup_dim, 1, relu=False, bn=False), #256 -> 200
            )for i in range(nstack)
         ])

        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])
        #self.relu = torch.nn.ReLU()
        #self.backbone = resnet50()
        #del self.backbone.fc

    def forward(self, imgs):
        x = imgs #(b,3,w,h)
        x = self.pre(x) #(b,192,96,128) #(b, 192, 57, 192)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature) #--> heatmap prediction
            append(preds)
            if (i < self.nstack - 1):
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        heatmap = torch.stack(combined_hm_preds, 1)[:, self.nstack - 1, :, :, :]
        return heatmap

class StackedHourglassImgRecon(nn.Module):
    def __init__(self, num_of_kp, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassImgRecon, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(2 * num_of_kp, 128, 3, 1, bn=True, relu=True),
            #Conv(2 * num_of_kp, 256, 3, 1, bn=True, relu=True),
            #Conv(256, 256, 3, 1, bn=True, relu=True),
            #Conv(256, 128, 3, 1, bn=True, relu=True),
            #Conv(128, 128, 3, 1, bn=True, relu=True),
            #Residual(128, 128),
            Residual(128, inp_dim), #inp_dim = 208 (img width)
            Residual(inp_dim, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                #Hourglass(4, inp_dim, bn, increase),
                Hourglass(2, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                #Conv(inp_dim, oup_dim, 1, relu=False, bn=False), #(256 -> 3)
                Conv(inp_dim, 128, 1, relu=False, bn=False),
                ##nn.BatchNorm2d(128),
                #Conv(128, 128, 1, relu=False, bn=False),
                Conv(128, 64, 1, relu=False, bn=False),
                #Conv(64, 64, 1, relu=False, bn=False),
                #Conv(64, 32, 1, relu=False, bn=False),
                #Conv(32, 32, 1, relu=False, bn=False),
                #Conv(32, 16, 1, relu=False, bn=False),
                #Conv(16, 16, 1, relu=False, bn=False),
                #Conv(16, 8, 1, relu=True, bn=False),
                #Conv(8, 8, 1, relu=False, bn=False),
                #Conv(8, 3, 1, relu=True, bn=False)
                Conv(64, 3, 1, relu=True, bn=False)
                #nn.BatchNorm2d(3)
        ) for i in range(nstack)
        ])

        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])

    def forward(self, concat_recon):
        x = concat_recon #(b, 2n, 96, 128)
        x = self.pre(x) #(b, 256, 96, 128)
        #model_upsample = upsample_recon(2, mode='bilinear', align_corners=True)
        #x = model_upsample(x) #(b,256,192,256)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)  # --> heatmap prediction
            append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return torch.stack(combined_hm_preds, 1)

class StackedHourglassImgRecon_DETR(nn.Module):
    def __init__(self, input_channel, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassImgRecon_DETR, self).__init__()
        self.nstack = nstack

        #self.pre1 = Conv(2 * 5 * num_of_kp, 64, 7, 2, bn=True, relu=True)
        #self.pre1 = Conv(256, 64, 7, 2, bn=True, relu=True)

        #self.pre1 = Conv(input_channel, 64, 7, 2, bn=True, relu=True)
        #self.pre2 = Residual(64, 128)
        #self.pre4 = Residual(128, inp_dim)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.pre = nn.Sequential(
            Conv(input_channel, 128, 3, 1, bn=True, relu=True),
            self.upsample,
            Residual(128, 128),
            self.upsample,
            Residual(128, inp_dim), #inp_dim = 208 (img width)
            #self.upsample,
            Residual(inp_dim, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
                #Hourglass(1, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                #Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                Conv(inp_dim, oup_dim, 1, relu=False, bn=False), #256 -> 200
            )for i in range(nstack)
         ])


        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])

    def forward(self, concat_recon):
        x = concat_recon

        #x = self.pre1(x)
        #x = self.pre2(x)
        #x = self.pre4(x)
        #x = self.upsample(x)

        x = self.pre(x) #(b,160,16,16)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)  # --> heatmap prediction
            append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return torch.stack(combined_hm_preds, 1)

class StackedHourglassImgRecon_DETR_dcn(nn.Module):
    def __init__(self, input_channel, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassImgRecon_DETR_dcn, self).__init__()
        self.nstack = nstack
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.pre = nn.Sequential(
            DConv(input_channel, 128, 3, 1, bn=True, relu=True),
            self.upsample,
            Residual_dcn(128, 128),
            self.upsample,
            Residual_dcn(128, inp_dim), #inp_dim = 208 (img width)
            #self.upsample,
            Residual_dcn(inp_dim, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass_dcn(4, inp_dim, bn, increase),
                #Hourglass(1, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                #Residual(inp_dim, inp_dim),
                DConv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                DConv(inp_dim, oup_dim, 1, relu=False, bn=False), #256 -> 200
            )for i in range(nstack)
         ])


        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])

    def forward(self, concat_recon):
        x = concat_recon

        #x = self.pre1(x)
        #x = self.pre2(x)
        #x = self.pre4(x)
        #x = self.upsample(x)

        x = self.pre(x) #(b,160,16,16)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)  # --> heatmap prediction
            append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return torch.stack(combined_hm_preds, 1)