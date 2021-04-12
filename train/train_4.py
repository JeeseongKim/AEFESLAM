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

import matplotlib.pyplot as plt
from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader

from DETR_origin import *
from DETR_backbone import *
from misc import *
from position_encoding import *
from MyDETR import *


warnings.filterwarnings("ignore")
from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader

import visdom

vis = visdom.Visdom()

plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))

#plot_transf = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation Loss'))
#plot_matching = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Feature Matching Loss'))
#plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosine Similarity Loss'))
plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))

plot_ok = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Ok reconstruction loss'))
plot_wk = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Wk reconstruction loss'))
plot_recon = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction Loss'))
#########################################parameter#########################################
num_of_kp = 200
feature_dimension = 256 #32

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 300
batch_size = 4

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 8

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5  # 1e-2#1e-5 #1e-5 #5e-4

lr_drop = 200
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=1e-3, weight_decay=1e-4)

    model_AttentionMap = AttentionMap(hidden_dim=256)
    model_AttentionMap = nn.DataParallel(model_AttentionMap).cuda()
    optimizer_AttentionMap = torch.optim.AdamW(model_AttentionMap.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_get_kp = getKP_attention()
    model_get_kp = nn.DataParallel(model_get_kp).cuda()
    optimizer_get_kp = torch.optim.AdamW(model_get_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_feature_descriptor = Linear(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor).cuda()
    optimizer_Wk_ = torch.optim.AdamW(model_feature_descriptor.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_detection_map_kp = ReconDetectionMapWithKP(img_width=my_width, img_height=my_height, num_of_kp=num_of_kp)
    model_detection_map_kp = nn.DataParallel(model_detection_map_kp).cuda()
    optimizer_reconDetectionkp = torch.optim.AdamW(model_detection_map_kp.parameters(), lr=learning_rate,weight_decay=weight_decay)

    model_dec_feature_descriptor = dec_Linear(feature_dimension=feature_dimension, img_width=my_height, img_height=my_width)
    model_dec_feature_descriptor = nn.DataParallel(model_dec_feature_descriptor).cuda()
    optimizer_decfeatureDescriptor = torch.optim.AdamW(model_dec_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack,inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False,increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=1e-3, weight_decay=1e-4)

    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_StackedHourglass_kp, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_AttentionMap, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_Wk_, lr_drop)
    lr_scheduler_optimizer4 = torch.optim.lr_scheduler.StepLR(optimizer_reconDetectionkp, lr_drop)
    lr_scheduler_optimizer5 = torch.optim.lr_scheduler.StepLR(optimizer_decfeatureDescriptor, lr_drop)
    lr_scheduler_optimizer6 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)
    lr_scheduler_optimizer7 = torch.optim.lr_scheduler.StepLR(optimizer_get_kp, lr_drop)

    ###################################################################################################################

    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        print("---Loading Checkpoint---")
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_AttentionMap.module.load_state_dict(checkpoint['model_AttentionMap'])
        model_get_kp.module.load_state_dict(checkpoint['model_get_kp'])
        model_feature_descriptor.module.load_state_dict(checkpoint['model_feature_descriptor'])
        model_detection_map_kp.module.load_state_dict(checkpoint['model_detection_map_kp'])
        model_dec_feature_descriptor.module.load_state_dict(checkpoint['model_dec_feature_descriptor'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_AttentionMap.load_state_dict(checkpoint['optimizer_AttentionMap'])
        optimizer_get_kp.load_state_dict(checkpoint['optimizer_get_kp'])
        optimizer_Wk_.load_state_dict(checkpoint['optimizer_Wk_'])
        optimizer_reconDetectionkp.load_state_dict(checkpoint['optimizer_reconDetectionkp'])
        optimizer_decfeatureDescriptor.load_state_dict(checkpoint['optimizer_decfeatureDescriptor'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])

        lr_scheduler_optimizer1.load_state_dict(checkpoint['lr_scheduler_optimizer1'])
        lr_scheduler_optimizer2.load_state_dict(checkpoint['lr_scheduler_optimizer2'])
        lr_scheduler_optimizer3.load_state_dict(checkpoint['lr_scheduler_optimizer3'])
        lr_scheduler_optimizer4.load_state_dict(checkpoint['lr_scheduler_optimizer4'])
        lr_scheduler_optimizer5.load_state_dict(checkpoint['lr_scheduler_optimizer5'])
        lr_scheduler_optimizer6.load_state_dict(checkpoint['lr_scheduler_optimizer6'])
        lr_scheduler_optimizer7.load_state_dict(checkpoint['lr_scheduler_optimizer7'])

    ###################################################################################################################

    dataset = my_dataset(my_width, my_height)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0

        running_sep_loss = 0

        running_ok_loss = 0
        running_Rk_loss = 0
        running_recon_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)
            cur_batch = aefe_input.shape[0]

            ##########################################ENCODER##########################################
            #theta = random.uniform(-10, 10)  # rotating theta
            #my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            #tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            #Rk = model_StackedHourglassForKP(aefe_input).sum(dim=1)
            Rk = model_StackedHourglassForKP(aefe_input)[:, num_nstack-1, :, :, :]
            #tf_Rk = model_StackedHourglassForKP(tf_aefe_input).sum(dim=1)

            attention_map, self_attention_feature_maps, kp_SAmap = model_AttentionMap(aefe_input) #(4, 256, 480) #convolutional feature maps

            #######################pytorch multihead attention#######################
            # z = h.permute(1, 0, 2) #key, query, value vector (256, 4, 480)
            #multihead_attn = nn.MultiheadAttention(embed_dim=480, num_heads=8, dropout=0.1).cuda()
            #attn_output, attn_output_weights = multihead_attn(z, z, z)

            #attn_src = attn_output.permute(1, 0, 2)
            #attn_src_t = torch.transpose(attn_src, 1, 2)
            #attention_map = torch.matmul(attn_src_t, attn_src)

            #attn_output = (256, 4, 480)
            #attn_output_weights = (4, 256, 256)
            #attention_map, self_attention_map = model_AttentionMap(aefe_input) #(b,480,480)
            ##########################################################################

            # keypoint extraction
            #model_get_kp = getKP_attention()
            Ok, kp, zeta, softmask = model_get_kp(Rk, my_height, my_width, kp_SAmap)

            # descriptor generation
            Wk_2d = Rk.view(cur_batch, num_of_kp, my_height * my_width)  # (b,k,h*w)
            fk_pre = model_feature_descriptor(Wk_2d)  # (b, k, h*w) -> (b,k,f)
            ww = (softmask * Rk).sum(dim=[2, 3]).unsqueeze(2)
            #fk = (speed_sigmoid_5(ww) * speed_sigmoid_5(fk_pre))  # (b, k, f)
            #fk = F.relu(ww * fk_pre)  # (b, k, f)
            #fk = 1/(1+torch.exp(-1*ww*fk_pre))
            fk = torch.sigmoid(ww*fk_pre)

            #my_feature = torch.cat([kp, fk], dim=2)
            #my_tf_feature = torch.cat([tf_kp, tf_fk], dim=2)

            #feature matching loss
            #fn_matching_loss = loss_matching(theta, my_feature, my_tf_feature, cur_batch, num_of_kp, my_width, my_height)
            #cur_transf_loss, cur_matching_loss = fn_matching_loss()

            #concentration loss
            #Rk_con_loss = loss_concentration(Rk).cuda()
            #tf_Rk_con_loss = loss_concentration(tf_Rk).cuda()
            #softmask_con_loss = loss_concentration(softmask).cuda()

            # similarity loss btw Dk and tf_Dk
            #fn_loss_cosim = loss_cosim(Ok, tf_Ok).cuda()
            #fn_loss_cosim = loss_cosim(Rk, tf_Rk).cuda()
            #cur_cosim_loss = fn_loss_cosim()

            # separation loss btw extracted kp
            fn_loss_separation = loss_separation(kp).cuda()
            cur_sep_loss = fn_loss_separation()

            # tf loss btw kp tf_kp
            #fn_loss_transformation = loss_transformation(theta, kp, tf_kp, cur_batch, num_of_kp, my_width, my_height).cuda()
            #cur_transf_loss = fn_loss_transformation()

            ##########################################DECODER##########################################
            # recon Rk with kp
            reconOk_kp = model_detection_map_kp(kp)
            #reconOk_kp = torch.softmax(reconOk_kp, dim=1)

            # reconDk_kp normalization
            #reconDk_kp_min = torch.min(torch.min(reconOk_kp, dim=2)[0], dim=2)[0]
            #reconDk_kp_max = torch.max(torch.max(reconOk_kp, dim=2)[0], dim=2)[0]
            #reconDk_kp_my_max_min = torch.cat([reconDk_kp_min.unsqueeze(2), reconDk_kp_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
            #reconDk_kp = (reconOk_kp - (reconDk_kp_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((reconDk_kp_my_max_min[:, :, 1] - reconDk_kp_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))

            reconWk = model_dec_feature_descriptor(fk)  # (b, 16, feature_dimension)
            reconWk = reconWk.view(cur_batch, num_of_kp, my_height, my_width)

            reconFk_fk = F.relu(reconWk) * reconOk_kp

            #reconDk_con_loss = loss_concentration(reconOk_kp).cuda()

            #cur_con_loss = Rk_con_loss() + tf_Rk_con_loss() + softmask_con_loss() + reconDk_con_loss()
            #cur_con_loss = Rk_con_loss() + tf_Rk_con_loss() + reconDk_con_loss()

            # recon dk loss
            #cur_dk_loss = F.mse_loss(Dk, reconDk_kp)
            cur_Ok_loss = F.mse_loss(Ok, reconOk_kp)

            # recon Wk loss
            cur_Rk_loss = F.mse_loss(Rk, reconWk)

            #concat_recon = torch.cat((reconDk_kp, reconFk_fk), 1)  # (b, 2k, h, w) channel-wise concatenation
            concat_recon = torch.cat((reconOk_kp, reconFk_fk), 1)  # (b, 2k, h, w) channel-wise concatenation
            reconImg = model_StackedHourglassImgRecon(concat_recon)  # (b, 8, 3, h,  w)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            #cur_recon_loss = pytorch_msssim.msssim(reconImg, aefe_input)
            criterion = SSIM()
            cur_recon_loss_ssim = criterion(reconImg, aefe_input)
            cur_recon_loss = cur_recon_loss_l2*5 + cur_recon_loss_ssim

            #loss parameter
            param_loss_sep = 1.0

            param_loss_ok = 0.3
            param_loss_Rk = 10.0
            param_loss_recon = 1.0

            my_sep_loss = param_loss_sep * cur_sep_loss

            my_ok_loss = param_loss_ok * cur_Ok_loss
            my_Rk_loss = param_loss_Rk * cur_Rk_loss
            my_recon_loss = param_loss_recon * cur_recon_loss

            loss = my_sep_loss + my_ok_loss + my_Rk_loss + my_recon_loss

            #print("Sep: ", my_sep_loss.item(), ", Con: ", my_con_loss.item(), ", Cosim: ", my_cosim_loss.item(), ", Trans: ", my_transf_loss.item(),  ", Matching: ", my_matching_loss.item(), ", Dk: ", my_dk_loss.item(), ", Wk: ", my_wk_loss.item(), ", Recon:", my_recon_loss.item())
            print("Sep: ", my_sep_loss.item(), ", O_k: ", my_ok_loss.item(), ", Rk: ", my_Rk_loss.item(), ", Recon:", my_recon_loss.item())

            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_AttentionMap.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_reconDetectionkp.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()

            loss.backward()
            '''
            for name, param in model_StackedHourglassForKP.named_parameters():
                #print(name, torch.max(abs(param.grad)))
                print(name, torch.isfinite(param.grad).all())

            for name, param in model_AttentionMap.named_parameters():
                #print(name, torch.max(abs(param.grad)))
                print(name, torch.isfinite(param.grad).all())

            for name, param in model_get_kp.named_parameters():
                #print(name, torch.max(abs(param.grad)))
                print(name, torch.isfinite(param.grad).all())

            for name, param in model_feature_descriptor.named_parameters():
                #print(name, torch.max(abs(param.grad)))
                print(name, torch.isfinite(param.grad).all())

            for name, param in model_detection_map_kp.named_parameters():
                #print(name, torch.max(abs(param.grad)))
                print(name, torch.isfinite(param.grad).all())

            for name, param in model_dec_feature_descriptor.named_parameters():
                #print(name, torch.max(abs(param.grad)))
                print(name, torch.isfinite(param.grad).all())

            for name, param in model_StackedHourglassImgRecon.named_parameters():
                #print(name, torch.max(abs(param.grad)))
                print(name, torch.isfinite(param.grad).all())
            '''

            optimizer_StackedHourglass_kp.step()
            optimizer_AttentionMap.step()
            optimizer_Wk_.step()
            optimizer_reconDetectionkp.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()

            #torch.autograd.set_detect_anomaly(True)

            lr_scheduler_optimizer1.step()
            lr_scheduler_optimizer2.step()
            lr_scheduler_optimizer3.step()
            lr_scheduler_optimizer4.step()
            lr_scheduler_optimizer5.step()
            lr_scheduler_optimizer6.step()

            running_loss = running_loss + loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()
            running_ok_loss = running_ok_loss + my_ok_loss.item()
            running_Rk_loss = running_Rk_loss + my_Rk_loss.item()
            running_recon_loss = running_recon_loss + my_recon_loss.item()

            if (((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
            #if (((epoch + 1) % 2 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, kp, epoch + 1, cur_filename)
                #fn_save_tfkpimg = savetfKPimg()
                #fn_save_tfkpimg(tf_aefe_input, tf_kp, epoch + 1, cur_filename)
                img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch + 1))
                save_image(reconImg, img_save_filename)

        #if (epoch != 0) and ((epoch+1) % 5 == 0):
        torch.save({
            'model_StackedHourglassForKP': model_StackedHourglassForKP.module.state_dict(),
            'model_AttentionMap': model_AttentionMap.module.state_dict(),
            'model_get_kp': model_get_kp.module.state_dict(),
            'model_feature_descriptor': model_feature_descriptor.module.state_dict(),
            'model_detection_map_kp': model_detection_map_kp.module.state_dict(),
            'model_dec_feature_descriptor': model_dec_feature_descriptor.module.state_dict(),
            'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

            'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
            'optimizer_AttentionMap': optimizer_AttentionMap.state_dict(),
            'optimizer_get_kp': optimizer_get_kp.state_dict(),
            'optimizer_Wk_': optimizer_Wk_.state_dict(),
            'optimizer_reconDetectionkp': optimizer_reconDetectionkp.state_dict(),
            'optimizer_decfeatureDescriptor': optimizer_decfeatureDescriptor.state_dict(),
            'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),

            'lr_scheduler_optimizer1': lr_scheduler_optimizer1.state_dict(),
            'lr_scheduler_optimizer2': lr_scheduler_optimizer2.state_dict(),
            'lr_scheduler_optimizer3': lr_scheduler_optimizer3.state_dict(),
            'lr_scheduler_optimizer4': lr_scheduler_optimizer4.state_dict(),
            'lr_scheduler_optimizer5': lr_scheduler_optimizer5.state_dict(),
            'lr_scheduler_optimizer6': lr_scheduler_optimizer6.state_dict(),
            'lr_scheduler_optimizer7': lr_scheduler_optimizer7.state_dict(),

        }, "./SaveModelCKPT/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')

        #vis.line(Y=[running_transf_loss], X=np.array([epoch]), win=plot_transf, update='append')
        #vis.line(Y=[running_matching_loss], X=np.array([epoch]), win=plot_matching, update='append')
        #vis.line(Y=[running_cosim_loss], X=np.array([epoch]), win=plot_cosim, update='append')
        vis.line(Y=[running_sep_loss], X=np.array([epoch]), win=plot_sep, update='append')

        vis.line(Y=[running_ok_loss], X=np.array([epoch]), win=plot_ok, update='append')
        vis.line(Y=[running_Rk_loss], X=np.array([epoch]), win=plot_wk, update='append')
        vis.line(Y=[running_recon_loss], X=np.array([epoch]), win=plot_recon, update='append')

        #saveLossData = 'epoch\t{}\tAll_Loss\t{:.4f} \tTrans\t{:.4f} \tMatching\t{:.4f}\tCosim\t{:.4f} \tSep\t{:.4f}\tVk\t{:.4f}\tWk\t{:.4f}\tRecon\t{:.4f}\n'.format(
        #    epoch, running_loss, running_transf_loss, running_matching_loss, running_cosim_loss, running_sep_loss, running_vk_loss, running_wk_loss, running_recon_loss)

        #saveLossTxt.write(saveLossData)

        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))


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

    print("!!!!!This is train_4.py!!!!!")
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################