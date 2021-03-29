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
num_queries = 100
hidden_dim = 256
feature_dimension = 256 #32

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 100
batch_size = 2 #8 #4

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

    model_DETR = DETR(num_classes=301, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR = nn.DataParallel(model_DETR).cuda()
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_DETR.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ###################################################################################################################
    """
    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_feature_descriptor.module.load_state_dict(checkpoint['model_feature_descriptor'])
        model_detection_map_kp.module.load_state_dict(checkpoint['model_detection_map_kp'])
        model_dec_feature_descriptor.module.load_state_dict(checkpoint['model_dec_feature_descriptor'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_Wk_.load_state_dict(checkpoint['optimizer_Wk_'])
        optimizer_reconDetectionkp.load_state_dict(checkpoint['optimizer_reconDetectionkp'])
        optimizer_decfeatureDescriptor.load_state_dict(checkpoint['optimizer_decfeatureDescriptor'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])
    """
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
            tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            attention_map, decoder_out = model_DETR(aefe_input)



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