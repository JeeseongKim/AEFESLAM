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

warnings.filterwarnings("ignore")

from torch.utils.data import TensorDataset, DataLoader

import visdom

vis = visdom.Visdom()

plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))
plot_recon = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction Loss'))
plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))
plot_con = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration Loss'))
plot_transf = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation Loss'))
plot_detection = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Detection Map Loss'))
plot_wk = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Weight(Desc) Loss'))
plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosine Similarity Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
num_of_kp = 150
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
weight_decay = 1e-5  # 1e-2#1e-5 #1e-5 #5e-4
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_feature_descriptor = Linear(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor).cuda()
    optimizer_Wk_ = torch.optim.AdamW(model_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    #model_detection_map_kp = ReconDetectionMapWithKP_Res(img_width=my_width, img_height=my_height, num_of_kp=num_of_kp)
    model_detection_map_kp = ReconDetectionMapWithKP(img_width=my_width, img_height=my_height, num_of_kp=num_of_kp)
    model_detection_map_kp = nn.DataParallel(model_detection_map_kp).cuda()
    optimizer_reconDetectionkp = torch.optim.AdamW(model_detection_map_kp.parameters(), lr=learning_rate,weight_decay=weight_decay)

    model_dec_feature_descriptor = dec_Linear(feature_dimension=feature_dimension, img_width=my_height, img_height=my_width)
    model_dec_feature_descriptor = nn.DataParallel(model_dec_feature_descriptor).cuda()
    optimizer_decfeatureDescriptor = torch.optim.AdamW(model_dec_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack,inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False,increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate,weight_decay=weight_decay)

    ###################################################################################################################
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
    ###################################################################################################################

    dataset = my_dataset(my_width, my_height)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        running_recon_loss = 0
        running_sep_loss = 0
        running_con_loss = 0
        running_transf_loss = 0
        running_detection_loss = 0
        running_wk_loss = 0
        running_cosim_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)
            cur_batch = aefe_input.shape[0]

            ##########################################ENCODER##########################################
            theta = random.uniform(-5, 5)  # rotating theta
            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            #Rk = model_StackedHourglassForKP(aefe_input).sum(dim=1)
            #tf_Rk = model_StackedHourglassForKP(tf_aefe_input).sum(dim=1)

            Rk = model_StackedHourglassForKP(aefe_input)[:, num_nstack - 1, :, :, :]
            tf_Rk = model_StackedHourglassForKP(tf_aefe_input)[:, num_nstack-1, :, :, :]

            # keypoint extraction
            fn_DetectionConfidenceMap2keypoint = DetectionConfidenceMap2keypoint()
            Dk, kp, zeta, tf_Dk, tf_kp = fn_DetectionConfidenceMap2keypoint(Rk.clone(), tf_Rk, my_height, my_width) #num_of_kp = 2*num_of_kp
            kp = kp.float()
            tf_kp = tf_kp.float()

            #softmask
            fn_softmask = create_softmask()
            softmask = fn_softmask(Dk, zeta)  # (b,k,96,128)

            # descriptor generation
            Wk_cal = Rk * Dk  # (b,k,h,w)
            Wk = Wk_cal.view(cur_batch, num_of_kp, my_height * my_width)  # (b,k,h*w)
            fk_pre = model_feature_descriptor(Wk)  # (b, k, h*w) -> (b,k,f)
            ww = (softmask * Rk).sum(dim=[2, 3]).unsqueeze(2)
            fk = (ww * F.relu(fk_pre))  # (b, k, f)

            #concentration loss
            Rk_con_loss = loss_concentration(Rk).cuda()
            tf_Rk_con_loss = loss_concentration(tf_Rk).cuda()
            softmask_con_loss = loss_concentration(softmask).cuda()

            # similarity loss btw Dk and tf_Dk
            fn_loss_cosim = loss_cosim(Dk, tf_Dk).cuda()
            #fn_loss_cosim = loss_cosim(Rk, tf_Rk).cuda()
            cur_cosim_loss = fn_loss_cosim()

            # separation loss btw extracted kp
            fn_loss_separation = loss_separation(kp).cuda()
            cur_sep_loss = fn_loss_separation()

            # tf loss btw kp tf_kp
            fn_loss_transformation = loss_transformation(theta, kp, tf_kp, cur_batch, num_of_kp, my_width, my_height).cuda()
            cur_transf_loss = fn_loss_transformation()

            ##########################################DECODER##########################################
            # recon Rk with kp
            reconRk_kp = model_detection_map_kp(kp)

            # reconDk_kp normalization
            #reconDk_kp_min = torch.min(torch.min(reconRk_kp, dim=2)[0], dim=2)[0]
            #reconDk_kp_max = torch.max(torch.max(reconRk_kp, dim=2)[0], dim=2)[0]
            #reconDk_kp_my_max_min = torch.cat([reconDk_kp_min.unsqueeze(2), reconDk_kp_max.unsqueeze(2)],dim=2)  # (b,k,2) 2: min, max
            #reconDk_kp = (reconRk_kp - (reconDk_kp_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((reconDk_kp_my_max_min[:, :, 1] - reconDk_kp_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
            reconDk_kp = torch.sigmoid(reconRk_kp)

            reconWk = model_dec_feature_descriptor(fk)  # (b, 16, feature_dimension)
            reconWk = reconWk.view(cur_batch, num_of_kp, my_height, my_width)
            reconFk_fk = F.relu(reconWk) * reconDk_kp

            reconDk_con_loss = loss_concentration(reconDk_kp).cuda()

            cur_con_loss = Rk_con_loss() + tf_Rk_con_loss() + softmask_con_loss() + reconDk_con_loss()

            # recon dk loss
            cur_dk_loss = F.mse_loss(Dk, reconDk_kp)

            # recon Wk loss
            cur_Wk_loss = F.mse_loss(Wk_cal, reconWk)

            concat_recon = torch.cat((reconDk_kp, reconFk_fk), 1)  # (b, 2k, h, w) channel-wise concatenation
            reconImg = model_StackedHourglassImgRecon(concat_recon)  # (b, 8, 3, h,  w)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            cur_recon_loss = F.mse_loss(reconImg, aefe_input)

            #loss parameter
            param_loss_sep = 5
            param_loss_con = 0.033
            param_loss_recon = 10.0
            param_loss_transf = 0.005
            param_loss_dk = 25.0  # 10.0
            param_loss_wk = 20.0
            param_loss_cosim = 0.05

            my_sep_loss = param_loss_sep * cur_sep_loss
            my_con_loss = param_loss_con * cur_con_loss
            my_cosim_loss = param_loss_cosim * cur_cosim_loss
            my_transf_loss = param_loss_transf * cur_transf_loss
            my_dk_loss = param_loss_dk * cur_dk_loss
            my_wk_loss = param_loss_wk * cur_Wk_loss
            my_recon_loss = param_loss_recon * cur_recon_loss

            loss = my_sep_loss + my_cosim_loss + my_transf_loss + my_dk_loss + my_wk_loss + my_recon_loss + my_con_loss

            print("Sep: ", my_sep_loss.item(), ", Con: ", my_con_loss.item(), ", Cosim: ", my_cosim_loss.item(), ", Trans: ", my_transf_loss.item(), ", Dk: ", my_dk_loss.item(), ", Wk: ", my_wk_loss.item(), ", Recon:", my_recon_loss.item())
            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_reconDetectionkp.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()

            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_Wk_.step()
            optimizer_reconDetectionkp.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()

            running_loss = running_loss + loss.item()
            running_recon_loss = running_recon_loss + my_recon_loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()
            running_con_loss = running_con_loss + my_con_loss.item()
            running_transf_loss = running_transf_loss + my_transf_loss.item()
            running_detection_loss = running_detection_loss + my_dk_loss.item()
            running_wk_loss = running_wk_loss + my_wk_loss.item()
            running_cosim_loss = running_cosim_loss + my_cosim_loss.item()

            if (((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
            #if (((epoch + 1) % 2 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, kp, epoch + 1, cur_filename)
                #fn_save_tfkpimg = savetfKPimg()
                #fn_save_tfkpimg(tf_aefe_input, tf_kp, epoch + 1, cur_filename)
                img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch + 1))
                save_image(reconImg, img_save_filename)
                if (epoch != 0):
                    torch.save({
                        'model_StackedHourglassForKP': model_StackedHourglassForKP.module.state_dict(),
                        'model_feature_descriptor': model_feature_descriptor.module.state_dict(),
                        'model_detection_map_kp': model_detection_map_kp.module.state_dict(),
                        'model_dec_feature_descriptor': model_dec_feature_descriptor.module.state_dict(),
                        'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

                        'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
                        'optimizer_Wk_': optimizer_Wk_.state_dict(),
                        'optimizer_reconDetectionkp': optimizer_reconDetectionkp.state_dict(),
                        'optimizer_decfeatureDescriptor': optimizer_decfeatureDescriptor.state_dict(),
                        'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),
                    }, "./SaveModelCKPT/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')
        vis.line(Y=[running_recon_loss], X=np.array([epoch]), win=plot_recon, update='append')
        vis.line(Y=[running_sep_loss], X=np.array([epoch]), win=plot_sep, update='append')
        vis.line(Y=[running_con_loss], X=np.array([epoch]), win=plot_con, update='append')
        vis.line(Y=[running_transf_loss], X=np.array([epoch]), win=plot_transf, update='append')
        vis.line(Y=[running_detection_loss], X=np.array([epoch]), win=plot_detection, update='append')
        vis.line(Y=[running_wk_loss], X=np.array([epoch]), win=plot_wk, update='append')
        vis.line(Y=[running_cosim_loss], X=np.array([epoch]), win=plot_cosim, update='append')

        saveLossData = 'epoch\t{}\tAll_Loss\t{:.4f} \tRecon\t{:.4f} \tCosim\t{:.4f}\tSep\t{:.4f} \tTrans\t{:.4f}\tDk\t{:.4f}\tWk\t{:.4f}\tConc\t{:.4f}\n'.format(
            epoch, running_loss, running_recon_loss, running_cosim_loss, running_sep_loss, running_transf_loss,
            running_detection_loss, running_wk_loss, running_con_loss)
        saveLossTxt.write(saveLossData)

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

    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################