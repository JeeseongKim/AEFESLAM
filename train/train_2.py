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
plot_transf = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation Loss'))
plot_detection = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Detection Map Loss'))
plot_score = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Score Map Loss'))
plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosine Similarity Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
num_of_kp = 200
feature_dimension = 256

my_width = 160 #272 #96 #272 #208
my_height = 48 #80 #32 #80 #64

input_width = my_width

num_epochs = 100
batch_size = 2

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp #number of my keypoints

num_nstack = 4

learning_rate = 1e-4#1e-3#1e-4 #1e-3
weight_decay = 1e-5#1e-2#1e-5 #1e-5 #5e-4
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    #optimizer_StackedHourglass_kp = torch.optim.Adam(model_StackedHourglassForKP.parameters(), lr=1e-3, weight_decay=2e-4)
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=5e-4, weight_decay=5e-5)

    model_descriptor_map = FeatureDesc(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_descriptor_map = nn.DataParallel(model_descriptor_map).cuda()
    optimizer_descriptor_map = torch.optim.AdamW(model_descriptor_map.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_detection_map_kp = ReconDetectionMapWithKP(img_width=my_width, img_height=my_height)
    model_detection_map_kp = nn.DataParallel(model_detection_map_kp).cuda()
    #optimizer_reconDetection = torch.optim.Adam(model_score_map.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_reconDetectionkp = torch.optim.AdamW(model_detection_map_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_score_map_f = ReconDetectionMapWithF(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_score_map_f = nn.DataParallel(model_score_map_f).cuda()
    #optimizer_reconDetection = torch.optim.Adam(model_score_map.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_reconScoref = torch.optim.AdamW(model_score_map_f.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    #optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ###################################################################################################################
    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_descriptor_map.module.load_state_dict(checkpoint['model_descriptor_map'])
        model_detection_map_kp.module.load_state_dict(checkpoint['model_detection_map_kp'])
        model_score_map_f.module.load_state_dict(checkpoint['model_score_map_f'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_descriptor_map.load_state_dict(checkpoint['optimizer_descriptor_map'])
        optimizer_reconDetectionkp.load_state_dict(checkpoint['optimizer_reconDetectionkp'])
        optimizer_reconScoref.load_state_dict(checkpoint['optimizer_reconScoref'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])
    ###################################################################################################################

    dataset = my_dataset(my_width, my_height)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start) #7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        running_recon_loss = 0
        running_sep_loss = 0
        running_transf_loss = 0
        running_detection_loss = 0
        running_score_loss = 0
        running_cosim_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            time0 = time.time()
            input_img, cur_filename, kp_img = data

            aefe_input = input_img.cuda() #(b, 3, height, width)
            cur_batch = aefe_input.shape[0]
            #print("Data loading", time.time() - time0)
            ##########################################ENCODER##########################################
            time1 = time.time()
            theta = random.uniform(-5, 5) #rotating theta
            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input) #randomly rotated image
            #print("TF", time.time() - time1)

            time2 = time.time()
            Rk = model_StackedHourglassForKP(aefe_input).sum(dim=1)
            tf_Rk = model_StackedHourglassForKP(tf_aefe_input).sum(dim=1)
            #print("Heatmap", time.time() - time2)

            #keypoint extraction
            time3 = time.time()
            fn_DetectionConfidenceMap2keypoint = DetectionConfidenceMap2keypoint()
            Dk, kp, zeta, tf_Dk, tf_kp = fn_DetectionConfidenceMap2keypoint(Rk, tf_Rk, cur_batch)
            #print("KP extract", time.time()-time3)

            #similarity loss btw Dk and tf_Dk
            time4 = time.time()
            fn_loss_cosim = loss_cosim(Dk, tf_Dk).cuda()
            cur_cosim_loss = fn_loss_cosim()
            #separation loss btw extracted kp
            fn_loss_separation = loss_separation(kp).cuda()
            cur_sep_loss = fn_loss_separation()
            #tf loss btw kp tf_kp
            fn_loss_transformation = loss_transformation(theta, kp, tf_kp, cur_batch, num_of_kp, my_width, my_height).cuda()
            cur_transf_loss = fn_loss_transformation()
            #print("loss time", time.time() - time4)

            #descriptor generation
            time5 = time.time()
            fk = model_descriptor_map(Rk) #(b,k,f)
            my_relu = torch.nn.ReLU()
            fk = my_relu(fk)
            #print("descriptor", time.time()-time5)
            ##########################################DECODER##########################################
            time6 = time.time()
            #recon Rk with kp
            reconRk_kp = model_detection_map_kp(kp)
            #reconDk_kp normalization
            reconDk_kp_min = torch.min(torch.min(reconRk_kp, dim=2)[0], dim=2)[0]
            reconDk_kp_max = torch.max(torch.max(reconRk_kp, dim=2)[0], dim=2)[0]
            reconDk_kp_my_max_min = torch.cat([reconDk_kp_min.unsqueeze(2), reconDk_kp_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
            reconDk_kp = (reconRk_kp - (reconDk_kp_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((reconDk_kp_my_max_min[:, :, 1] - reconDk_kp_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
            #print("recon kp", time.time()-time6)

            time7 = time.time()
            #reconRk with f
            reconRk_f = model_score_map_f(fk)
            ## reconDk_f normalization
            #reconDk_f_min = torch.min(torch.min(reconRk_f, dim=2)[0], dim=2)[0]
            #reconDk_f_max = torch.max(torch.max(reconRk_f, dim=2)[0], dim=2)[0]
            #reconDk_f_my_max_min = torch.cat([reconDk_f_min.unsqueeze(2), reconDk_f_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
            #reconDk_f = (reconRk_f - (reconDk_f_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((reconDk_f_my_max_min[:, :, 1] - reconDk_f_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
            #print("recon f", time.time()-time7)
            
            cur_rk_loss = F.mse_loss(reconRk_f, Rk)

            #triplet loss
            #time8 = time.time()
            #triplet_loss = torch.nn.TripletMarginWithDistanceLoss(reduction='mean')
            #cur_dk_loss = triplet_loss(Dk, reconDk_kp, reconDk_f)
            cur_dk_loss = F.mse_loss(Dk, reconDk_kp)
            #print("Triplet loss", time.time() - time8)

            time9 = time.time()
            concat_recon = torch.cat((reconDk_kp, reconRk_f), 1)  # (b, 2k, h, w) channel-wise concatenation
            reconImg = model_StackedHourglassImgRecon(concat_recon)  # (b, 8, 3, h,  w)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)
            #print("imt recon", time.time()-time9)

            time10 = time.time()
            cur_recon_loss = F.mse_loss(reconImg, aefe_input)
            #cur_recon_loss = F.mse_loss(reconImg, aefe_input)
            #print("recon loss", time.time()-time10)

            time11 = time.time()
            param_loss_sep = 1.0
            param_loss_recon = 5.0
            param_loss_transf = 0.1
            param_loss_dk = 50.0  # 10.0
            param_loss_rk = 2.0
            param_loss_cosim = 1e-4

            my_sep_loss = param_loss_sep * cur_sep_loss
            my_cosim_loss = param_loss_cosim * cur_cosim_loss
            my_transf_loss = param_loss_transf * cur_transf_loss
            my_dk_loss = param_loss_dk * cur_dk_loss
            my_rk_loss = param_loss_rk * cur_rk_loss
            my_recon_loss = param_loss_recon * cur_recon_loss

            loss = my_sep_loss + my_cosim_loss + my_transf_loss + my_dk_loss + my_rk_loss + my_recon_loss

            print("Sep: ", my_sep_loss.item(), ", Cosim: ", my_cosim_loss.item(), ", Trans: ", my_transf_loss.item(), ", Dk: ", my_dk_loss.item(), ", Rk: ", my_rk_loss.item(), ", Recon:", my_recon_loss.item())
            #print("loss compute", time.time() - time11)
            # ================Backward================
            time12 = time.time()
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_descriptor_map.zero_grad()
            optimizer_reconDetectionkp.zero_grad()
            optimizer_reconScoref.zero_grad()
            optimizer_ImgRecon.zero_grad()
            #print("optimizer_zero_Grad", time.time() - time12)

            time13 = time.time()
            loss.backward()
            #print("loss backward", time.time() - time13)

            time14 = time.time()
            optimizer_StackedHourglass_kp.step()
            optimizer_descriptor_map.step()
            optimizer_reconDetectionkp.step()
            optimizer_reconScoref.step()
            optimizer_ImgRecon.step()
            #print("optimizer_step", time.time()-time14)

            running_loss = running_loss + loss.item()
            running_recon_loss = running_recon_loss + my_recon_loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()
            running_transf_loss = running_transf_loss + my_transf_loss.item()
            running_detection_loss = running_detection_loss + my_dk_loss.item()
            running_score_loss = running_score_loss + my_rk_loss.item()
            running_cosim_loss = running_cosim_loss + my_cosim_loss.item()

            if (((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, kp, epoch + 1, cur_filename)
                fn_save_tfkpimg = savetfKPimg()
                fn_save_tfkpimg(tf_aefe_input, tf_kp, epoch + 1, cur_filename)
                img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch + 1))
                save_image(reconImg, img_save_filename)
                if (epoch != 0):
                    torch.save({
                        'model_StackedHourglassForKP': model_StackedHourglassForKP.module.state_dict(),
                        'model_descriptor_map': model_descriptor_map.module.state_dict(),
                        'model_detection_map_kp': model_detection_map_kp.module.state_dict(),
                        'model_score_map_f': model_score_map_f.module.state_dict(),
                        'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

                        'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
                        'optimizer_descriptor_map': optimizer_descriptor_map.state_dict(),
                        'optimizer_reconDetectionkp': optimizer_reconDetectionkp.state_dict(),
                        'optimizer_reconScoref': optimizer_reconScoref.state_dict(),
                        'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),
                    }, "./SaveModelCKPT/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')
        vis.line(Y=[running_recon_loss], X=np.array([epoch]), win=plot_recon, update='append')
        vis.line(Y=[running_sep_loss], X=np.array([epoch]), win=plot_sep, update='append')
        vis.line(Y=[running_transf_loss], X=np.array([epoch]), win=plot_transf, update='append')
        vis.line(Y=[running_detection_loss], X=np.array([epoch]), win=plot_detection, update='append')
        vis.line(Y=[running_score_loss], X=np.array([epoch]), win=plot_score, update='append')
        vis.line(Y=[running_cosim_loss], X=np.array([epoch]), win=plot_cosim, update='append')

        
        saveLossData = 'epoch\t{}\tAll_Loss\t{:.4f} \tRecon\t{:.4f} \tCosim\t{:.4f}Sep\t{:.4f} \tTrans\t{:.4f} \tDk\tRk\n'.format(epoch, running_loss, running_recon_loss, running_cosim_loss, running_sep_loss, running_transf_loss,running_detection_loss, running_score_loss)
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