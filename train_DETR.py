import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
import time
from model.layers import *
from StackedHourglass import *
from DetectionConfidenceMap import *
from loss import *
from utils import *
from GenDescriptorMap import *
import os
import numpy as np
from tqdm import tqdm
import torchvision
import random
import seaborn
import matplotlib.pyplot as plt
from torchvision import transforms
import warnings
import pytorch_msssim
warnings.filterwarnings("ignore")
from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader

from DETR_origin import *
from DETR_backbone import *
from misc import *
from position_encoding import *
from MyDETR import *

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
num_of_kp = 200
voters = 200
num_queries = voters

hidden_dim = 256
feature_dimension = 256 #32

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 100
batch_size = 1 #8 #4

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 4

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5  # 1e-2#1e-5 #1e-5 #5e-4
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_DETR = DETR(num_voters=voters, hidden_dim=hidden_dim, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR = nn.DataParallel(model_DETR).cuda()
    optimizer_DETR = torch.optim.AdamW(model_DETR.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_simple_rsz = simple_rsz(inp_channel=468, oup_channel=hidden_dim).cuda()
    model_simple_rsz = nn.DataParallel(model_simple_rsz).cuda()
    optimizer_simple_rsz = torch.optim.AdamW(model_simple_rsz.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_MakeDesc = MakeDesc(inp_channel=hidden_dim, oup_channel=feature_dimension).cuda()
    model_MakeDesc = nn.DataParallel(model_MakeDesc).cuda()
    optimizer_MakeDesc = torch.optim.AdamW(model_MakeDesc.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_recon_MakeDesc = Recon_MakeDesc(inp_channel=feature_dimension, oup_channel=hidden_dim).cuda()
    model_recon_MakeDesc = nn.DataParallel(model_recon_MakeDesc).cuda()
    optimizer_Recon_MakeDesc = torch.optim.AdamW(model_recon_MakeDesc.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_ReconDetection = Recon_Detection(inp_channel=2, oup_channel=hidden_dim)
    model_ReconDetection = nn.DataParallel(model_ReconDetection).cuda()
    optimizer_ReconDetection = torch.optim.AdamW(model_ReconDetection.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(num_of_kp=num_of_kp, nstack=num_nstack,inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False,increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    #optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate,weight_decay=weight_decay)
    optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=learning_rate,weight_decay=weight_decay)

    ###################################################################################################################
    #call checkpoint
    ###################################################################################################################

    dataset = my_dataset_originalImg(my_width=1226, my_height=370)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)

        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)
            cur_batch = aefe_input.shape[0]

            ##########################################ENCODER##########################################
            theta = random.uniform(-10, 10)  # rotating theta
            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            #tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            attention_map, Hk, kp = model_DETR(aefe_input)

            attention_score_col = torch.sum(attention_map, dim=1).unsqueeze(1) #(b, 1, 468)
            rsz_attention_score_col = model_simple_rsz(attention_score_col) #(b, 1, hidden_dim=256)
            DescMap = torch.mul(rsz_attention_score_col, Hk) #(b,v,256) == Hk.shape

            fk = model_MakeDesc(DescMap) #(b, 200, 256)

            ReconDescMap = model_recon_MakeDesc(fk) #(b, 200, 256)

            ReconDetectionMap = model_ReconDetection(kp) #kp = (b,v,2) #DetectionMap (b, 200, 256)
            FeatureMap = ReconDescMap * ReconDetectionMap
            recon_concat = torch.cat([ReconDetectionMap, FeatureMap], dim=1)
            recon_concat = torch.reshape(recon_concat, [cur_batch, 2*num_of_kp, 16, 16])
            recon_img = model_StackedHourglassImgRecon(recon_concat)

            # Define Loss Functions!
            #separation loss
            #fn_loss_separation = loss_separation(kp).cuda()
            #cur_sep_loss = fn_loss_separation()

            #recon_Dk loss
            #cur_dk_loss = F.mse_loss(ReconDetectionMap, reconDk_kp)

            #recon Fk loss

















##########################################################################################################################
if __name__ == '__main__':
    torch.cuda.empty_cache()
    if not os.path.exists("SaveKPImg"):
        os.makedirs("SaveKPImg")
    if not os.path.exists("SavetfKPImg"):
        os.makedirs("SavetfKPImg")
    if not os.path.exists("SaveReconstructedImg"):
        os.makedirs("SaveReconstructedImg")
    if not os.path.exists("SaveHeatMapImg"):
        os.makedirs("SaveHeatMapImg")
    if not os.path.exists("SaveModelCKPT"):
        os.makedirs("SaveModelCKPT")

    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################