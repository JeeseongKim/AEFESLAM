import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
import time
from model.layers import *
from model.StackedHourglass import *
from model.DetectionConfidenceMap import *
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
from misc import *
from model.MyDETR import *

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

plot_vk = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Vk reconstruction loss'))
plot_wk = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Wk reconstruction loss'))
plot_recon = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction Loss'))
torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
num_of_kp = 200
feature_dimension = 256 #32

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 100
batch_size = 4

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 4

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5 #1e-5  # 1e-2#1e-5 #1e-5 #5e-4

voters = num_of_kp
hidden_dim = 256

lr_drop = 200
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer_StackedHourglass_kp = torch.optim.Adam(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_DETR_Backbone = DETR_backbone(hidden_dim=hidden_dim).cuda()
    model_DETR_Backbone = nn.DataParallel(model_DETR_Backbone).cuda()
    optimizer_DETR_Backbone = torch.optim.AdamW(model_DETR_Backbone.parameters(), lr=1e-5, weight_decay=1e-4)

    model_DETR = DETR(num_voters=voters, hidden_dim=hidden_dim, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR = nn.DataParallel(model_DETR).cuda()
    optimizer_DETR = torch.optim.AdamW(model_DETR.parameters(), lr=1e-4, weight_decay=1e-4)

    model_feature_descriptor = Linear(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor).cuda()
    optimizer_Wk_ = torch.optim.AdamW(model_feature_descriptor.parameters(), lr=1e-6, weight_decay=5e-6)
    #optimizer_Wk_ = torch.optim.Adam(model_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_detection_map_kp = ReconDetectionMapWithKP(img_width=my_width, img_height=my_height, num_of_kp=num_of_kp)
    model_detection_map_kp = nn.DataParallel(model_detection_map_kp).cuda()
    #optimizer_reconDetectionkp = torch.optim.AdamW(model_detection_map_kp.parameters(), lr=5e-5, weight_decay=1e-4)
    optimizer_reconDetectionkp = torch.optim.AdamW(model_detection_map_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer_reconDetectionkp = torch.optim.Adam(model_detection_map_kp.parameters(), lr=learning_rate,weight_decay=weight_decay)

    model_dec_feature_descriptor = dec_Linear(feature_dimension=feature_dimension, img_width=my_height, img_height=my_width)
    model_dec_feature_descriptor = nn.DataParallel(model_dec_feature_descriptor).cuda()
    optimizer_decfeatureDescriptor = torch.optim.AdamW(model_dec_feature_descriptor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer_decfeatureDescriptor = torch.optim.Adam(model_dec_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack,inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=1e-3, weight_decay=1e-4)
    #optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=learning_rate,weight_decay=weight_decay)

    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_StackedHourglass_kp, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_DETR_Backbone, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_DETR, lr_drop)
    lr_scheduler_optimizer4 = torch.optim.lr_scheduler.StepLR(optimizer_Wk_, lr_drop)
    lr_scheduler_optimizer5 = torch.optim.lr_scheduler.StepLR(optimizer_reconDetectionkp, lr_drop)
    lr_scheduler_optimizer6 = torch.optim.lr_scheduler.StepLR(optimizer_decfeatureDescriptor, lr_drop)
    lr_scheduler_optimizer7 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)

    ###################################################################################################################
    if os.path.exists("./SaveModelCKPT/train_model.pth"):
    #if os.path.exists("./SaveModelCKPT/210401.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_feature_descriptor.module.load_state_dict(checkpoint['model_feature_descriptor'])
        model_detection_map_kp.module.load_state_dict(checkpoint['model_detection_map_kp'])
        model_dec_feature_descriptor.module.load_state_dict(checkpoint['model_dec_feature_descriptor'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])
        model_DETR.module.load_state_dict(checkpoint['model_DETR'])
        model_DETR_Backbone.module.load_state_dict(checkpoint['model_DETR_Backbone'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_Wk_.load_state_dict(checkpoint['optimizer_Wk_'])
        optimizer_reconDetectionkp.load_state_dict(checkpoint['optimizer_reconDetectionkp'])
        optimizer_decfeatureDescriptor.load_state_dict(checkpoint['optimizer_decfeatureDescriptor'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])
        optimizer_DETR.load_state_dict(checkpoint['optimizer_DETR'])
        optimizer_DETR_Backbone.load_state_dict(checkpoint['optimizer_DETR_Backbone'])

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

        #running_transf_loss = 0
        #running_matching_loss = 0
        #running_cosim_loss = 0

        running_sep_loss = 0

        running_vk_loss = 0
        running_wk_loss = 0
        running_recon_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)
            cur_batch = aefe_input.shape[0]

            ##########################################ENCODER##########################################
            #theta = random.uniform(-10, 10)  # rotating theta
            #my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            #tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            Rk = model_StackedHourglassForKP(aefe_input).sum(dim=1)
            #tf_Rk = model_StackedHourglassForKP(tf_aefe_input).sum(dim=1)

            x = model_DETR_Backbone(aefe_input)
            attention_map, attention_score, Hk = model_DETR(x)
            #tf_attention_map, tf_attention_score, tf_Hk = model_DETR(tf_aefe_input)

            # keypoint extraction
            fn_DetectionConfidenceMap2keypoint = getKP_DETR()
            #Ok, tf_Ok, Vk, tf_Vk, kp, tf_kp, zeta, tf_zeta = fn_DetectionConfidenceMap2keypoint(Rk, tf_Rk, my_height, my_width, attention_score, tf_attention_score, Hk, tf_Hk)
            Ok, Vk, kp, zeta = fn_DetectionConfidenceMap2keypoint(Rk, my_height, my_width, attention_score, Hk)
            #tf_Ok, tf_Vk, tf_kp, tf_zeta = fn_DetectionConfidenceMap2keypoint(tf_Rk, my_height, my_width, tf_attention_score, tf_Hk)

            # descriptor generation
            Wk = Ok.view(cur_batch, num_of_kp, my_height * my_width)  # (b,k,h*w)
            fk_pre = model_feature_descriptor(Wk)  # (b, k, h*w) -> (b,k,f)
            #fk = fk_pre * zeta.unsqueeze(2)
            fk = F.relu(fk_pre) * zeta.unsqueeze(2)

            #tf_Wk = tf_Ok.view(cur_batch, num_of_kp, my_height * my_width)  # (b,k,h*w)
            #tf_fk_pre = model_feature_descriptor(tf_Wk)  # (b, k, h*w) -> (b,k,f)
            #tf_fk = tf_fk_pre * tf_zeta.unsqueeze(2)
            #tf_fk = F.relu(tf_fk_pre) * tf_zeta.unsqueeze(2)

            #my_feature = torch.cat([kp, fk], dim=2)
            #my_tf_feature = torch.cat([tf_kp, tf_fk], dim=2)

            ## Encoder Loss
            #feature matching loss
            #fn_matching_loss = loss_matching(theta, my_feature, my_tf_feature, cur_batch, num_of_kp, my_width, my_height)
            #cur_transf_loss, cur_matching_loss = fn_matching_loss()

            # similarity loss btw Dk and tf_Dk
            #fn_loss_cosim = loss_cosim(Vk, tf_Vk).cuda()
            #fn_loss_cosim = loss_cosim(Ok, tf_Ok).cuda()
            #cur_cosim_loss = fn_loss_cosim()

            # separation loss btw extracted kp
            fn_sep_kp = loss_separation(kp).cuda()
            #fn_sep_tf_kp = loss_separation(tf_kp).cuda()
            sep_kp = fn_sep_kp()
            #sep_tf_kp = fn_sep_tf_kp()

            #cur_sep_loss = sep_kp + sep_tf_kp
            cur_sep_loss = sep_kp

            ##########################################DECODER##########################################
            # recon Rk with kp
            #recon_Vk = model_detection_map_kp(kp)
            recon_Ok = model_detection_map_kp(kp)
            #recon_tf_Vk = model_detection_map_kp(tf_kp)
            #recon_tf_Ok = model_detection_map_kp(tf_kp)

            #fn_gaussian_recon = ReconDetectionConfidenceMap_torch(kp, my_height, my_width)
            #recon_Ok = fn_gaussian_recon.sample()

            recon_Wk_2d = model_dec_feature_descriptor(fk, recon_Ok)
            #recon_tf_Wk_2d = model_dec_feature_descriptor(tf_fk, recon_tf_Ok)

            recon_Wk = recon_Wk_2d.view(cur_batch, num_of_kp, my_height, my_width)
            #recon_tf_Wk = recon_tf_Wk_2d.view(cur_batch, num_of_kp, my_height, my_width)

            #recon_Fk = F.relu(recon_Wk) * recon_Vk
            recon_Fk = F.relu(recon_Wk) * recon_Ok

            #concat_recon = torch.cat((recon_Vk, recon_Fk), 1)  # (b, 2k, h, w) channel-wise concatenation
            concat_recon = torch.cat((recon_Ok, recon_Fk), 1)  # (b, 2k, h, w) channel-wise concatenation
            reconImg = model_StackedHourglassImgRecon(concat_recon)  # (b, 8, 3, h,  w)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            ##Decoder Loss
            #cur_Vk_loss = F.mse_loss(Vk, recon_Vk) #+ F.mse_loss(tf_Vk, recon_tf_Vk)
            cur_Vk_loss = F.mse_loss(Ok, recon_Ok) #+ F.mse_loss(tf_Vk, recon_tf_Vk)
            cur_Wk_loss = F.mse_loss(Wk, recon_Wk_2d) #+ F.mse_loss(tf_Wk, recon_tf_Wk_2d)

            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            criterion = SSIM()
            cur_recon_loss_ssim = criterion(reconImg, aefe_input)
            cur_recon_loss = cur_recon_loss_l2 + cur_recon_loss_ssim

            #loss parameter
            #p_transf = 0.1
            #p_matching = 100000.0
            #p_cosim = 2.0
            p_sep = 1.0
            p_vk = 10.0
            p_wk = 500000.0
            p_recon = 1.0

            #my_transf_loss = p_transf * cur_transf_loss
            #my_matching_loss = p_matching * cur_matching_loss
            #my_cosim_loss = p_cosim * cur_cosim_loss
            my_sep_loss = p_sep * cur_sep_loss

            my_vk_loss = p_vk * cur_Vk_loss
            my_wk_loss = p_wk * cur_Wk_loss
            my_recon_loss = p_recon * cur_recon_loss

            #loss = my_transf_loss + my_matching_loss + my_cosim_loss + my_sep_loss + my_vk_loss + my_wk_loss + my_recon_loss
            loss =my_sep_loss + my_recon_loss + my_vk_loss + my_wk_loss

            #print("Trans: ", my_transf_loss.item(), ", Matching: ", my_matching_loss.item(), ", Cosim: ", my_cosim_loss.item(),  ", Sep: ", my_sep_loss.item(), ", Vk: ", my_vk_loss.item(),  ", Wk: ", my_wk_loss.item(), ", Recon:", my_recon_loss.item())
            print("Sep: ", my_sep_loss.item(), ", Vk: ", my_vk_loss.item(),  ", Wk: ", my_wk_loss.item(), ", Recon:", my_recon_loss.item())

            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_reconDetectionkp.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()
            optimizer_DETR.zero_grad()
            optimizer_DETR_Backbone.zero_grad()

            #if not (torch.isfinite(my_transf_loss) or torch.isfinite(my_matching_loss) or torch.isfinite(my_cosim_loss) or torch.isfinite(my_sep_loss) or torch.isfinite(my_vk_loss) or torch.isfinite(my_wk_loss) or torch.isfinite(my_recon_loss)):
            #    print("WARNING: not-finite loss, ending training")
            #    exit(1)

            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_Wk_.step()
            optimizer_reconDetectionkp.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()
            optimizer_DETR.step()
            optimizer_DETR_Backbone.step()

            lr_scheduler_optimizer1.step()
            lr_scheduler_optimizer2.step()
            lr_scheduler_optimizer3.step()
            lr_scheduler_optimizer4.step()
            lr_scheduler_optimizer5.step()
            lr_scheduler_optimizer6.step()
            lr_scheduler_optimizer7.step()

            running_loss = running_loss + loss.item()

            #running_transf_loss = running_transf_loss + my_transf_loss.item()
            #running_matching_loss = running_matching_loss + my_matching_loss.item()
            #running_cosim_loss = running_cosim_loss + my_cosim_loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()

            running_vk_loss = running_vk_loss + my_vk_loss.item()
            running_wk_loss = running_wk_loss + my_wk_loss.item()
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
            'model_feature_descriptor': model_feature_descriptor.module.state_dict(),
            'model_detection_map_kp': model_detection_map_kp.module.state_dict(),
            'model_dec_feature_descriptor': model_dec_feature_descriptor.module.state_dict(),
            'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),
            'model_DETR': model_DETR.module.state_dict(),
            'model_DETR_Backbone': model_DETR_Backbone.module.state_dict(),

            'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
            'optimizer_Wk_': optimizer_Wk_.state_dict(),
            'optimizer_reconDetectionkp': optimizer_reconDetectionkp.state_dict(),
            'optimizer_decfeatureDescriptor': optimizer_decfeatureDescriptor.state_dict(),
            'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),
            'optimizer_DETR': optimizer_DETR.state_dict(),
            'optimizer_DETR_Backbone': optimizer_DETR_Backbone.state_dict(),

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

        vis.line(Y=[running_vk_loss], X=np.array([epoch]), win=plot_vk, update='append')
        vis.line(Y=[running_wk_loss], X=np.array([epoch]), win=plot_wk, update='append')
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

    print("!!!!!This is train_5.py!!!!!")
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################