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
from patchnet import *
from GenHeatmap import *
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
plot_conc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration Loss'))
plot_transf = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation Loss'))
plot_detection = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Detection Map Loss'))
plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosine Similarity Loss'))
#plot_score = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Score Map Loss'))
plot_weightW = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Descriptor weight Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
num_of_kp = 2000
feature_dimension = 32

my_width = 160 #272 #96 #272 #208
my_height = 48 #80 #32 #80 #64

input_width = my_width

num_epochs = 1000
batch_size = 8

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp #number of my keypoints

num_nstack = 6

learning_rate = 1e-4 #1e-3#1e-4 #1e-3
weight_decay = 1e-5 #1e-5#1e-2#1e-5 #1e-5 #5e-4
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    #model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    #model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    #optimizer_StackedHourglass_kp = torch.optim.Adam(model_StackedHourglassForKP.parameters(), lr=1e-3, weight_decay=2e-4)
    #optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_L2net = L2net_R2D2(num_of_kp=num_of_kp)
    model_L2net = nn.DataParallel(model_L2net).cuda()
    optimizer_L2 = torch.optim.AdamW(model_L2net.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_feature_descriptor = Linear(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor).cuda()
    #optimizer_Wk_ = torch.optim.Adam(model_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)
    optimizer_Wk_ = torch.optim.AdamW(model_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_score_map = LinearReconScoreMap(img_width=my_width, img_height=my_height, num_of_kp=num_of_kp)
    model_score_map = nn.DataParallel(model_score_map).cuda()
    #optimizer_reconDetection = torch.optim.Adam(model_score_map.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_reconDetection = torch.optim.AdamW(model_score_map.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_dec_feature_descriptor = dec_Linear(feature_dimension=feature_dimension, img_width=my_height, img_height=my_width)
    model_dec_feature_descriptor = nn.DataParallel(model_dec_feature_descriptor).cuda()
    #optimizer_decfeatureDescriptor = torch.optim.Adam(model_dec_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)
    optimizer_decfeatureDescriptor = torch.optim.AdamW(model_dec_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    #optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ###################################################################################################################
    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        #model_L2net.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_L2net.module.load_state_dict(checkpoint['model_L2net'])
        model_feature_descriptor.module.load_state_dict(checkpoint['model_feature_descriptor'])
        model_score_map.module.load_state_dict(checkpoint['model_score_map'])
        model_dec_feature_descriptor.module.load_state_dict(checkpoint['model_dec_feature_descriptor'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        #optimizer_L2.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_L2.load_state_dict(checkpoint['optimizer_L2'])
        optimizer_Wk_.load_state_dict(checkpoint['optimizer_Wk_'])
        optimizer_reconDetection.load_state_dict(checkpoint['optimizer_reconDetection'])
        optimizer_decfeatureDescriptor.load_state_dict(checkpoint['optimizer_decfeatureDescriptor'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])
    ###################################################################################################################

    #my_transform = torchvision.transforms.RandomAffine((-60, 60), translate=(0.3, 0.3), scale=None, shear=None, resample=0, fillcolor=0)
    #dataset = my_dataset(transform=my_transform)
    dataset = my_dataset(my_width, my_height)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start) #7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        running_recon_loss = 0
        running_sep_loss = 0
        running_conc_loss = 0
        running_transf_loss = 0
        running_detection_loss = 0
        running_cosim_loss = 0
        #running_score_loss = 0
        running_descriptorW_loss= 0
        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data

            aefe_input = input_img.cuda() #(b, 3, height, width)
            cur_batch = aefe_input.shape[0]
            ##########################################ENCODER##########################################
            theta = random.uniform(-5, 5)
            #TFmatrix = make_transformation_M(theta, 0, 0)
            my_transform = torchvision.transforms.RandomAffine(degrees=(theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input) #randomly rotated image

            combined_hm_preds = model_L2net(aefe_input)
            tf_combined_hm_preds = model_L2net(tf_aefe_input)

            fn_DetectionConfidenceMap2keypoint = YesTF_GradKP_DetectionConfidenceMap2keypoint()
            DetectionMap, keypoints, zeta, tf_DetectionMap, tf_keypoints = fn_DetectionConfidenceMap2keypoint(combined_hm_preds, tf_combined_hm_preds)

            #fn_save_kpimg = saveKPimg()
            #fn_save_kpimg(kp_img, keypoints, epoch + 1, cur_filename)

            #fn_save_tfkpimg = savetfKPimg()
            #fn_save_tfkpimg(tf_aefe_input, tf_keypoints, epoch + 1, cur_filename)

            fn_loss_cosim = loss_cosim(DetectionMap, tf_DetectionMap).cuda()
            cur_cosim_loss = fn_loss_cosim()

            fn_softmask = create_softmask()
            softmask = fn_softmask(DetectionMap, zeta)  # (b,k,96,128)
            softmask_min = torch.min(torch.min(softmask, dim=2)[0], dim=2)[0]
            softmask_max = torch.max(torch.max(softmask, dim=2)[0], dim=2)[0]
            softmask_my_max_min = torch.cat([softmask_min.unsqueeze(2), softmask_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
            softmask = (softmask - (softmask_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((softmask_my_max_min[:, :, 1] - softmask_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
            #softmask = F.normalize(softmask, p=2, dim=1)

            fn_loss_concentration_Rk = loss_concentration(combined_hm_preds).cuda()
            fn_loss_concentration_softmask = loss_concentration(softmask).cuda()
            fn_loss_separation = loss_separation(keypoints).cuda()

            fn_loss_transformation = loss_transformation(theta, keypoints, tf_keypoints, cur_batch, num_of_kp, my_width, my_height).cuda()
            cur_transf_loss = fn_loss_transformation()

            cur_conc_loss = fn_loss_concentration_Rk() + fn_loss_concentration_softmask()
            #cur_conc_loss = fn_loss_concentration_Rk()
            cur_sep_loss = fn_loss_separation()

            fn_relu = torch.nn.ReLU().cuda()
            leakyrelu4descriptors = torch.nn.LeakyReLU(0.01).cuda()

            Wk_raw = combined_hm_preds * DetectionMap #(b,k,h,w)
            Wk_rsz = Wk_raw.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2] * Wk_raw.shape[3]) #(b,k,h*w)
            Wk_ = model_feature_descriptor(Wk_rsz) #(b, k, h*w) -> (b,k,f)
            #mul_all_torch = (softmask*fn_relu(combined_hm_preds)).sum(dim=[2, 3]).unsqueeze(2)
            mul_all_torch = (softmask*combined_hm_preds).sum(dim=[2, 3]).unsqueeze(2)
            my_descriptor = (mul_all_torch * fn_relu(Wk_)) #(b, k, f)
            #print("descriptor time", time.time() - start4)

            ##########################################DECODER##########################################
            reconScoreMap = model_score_map(keypoints, DetectionMap)

            #reconDetectionMap = F.softplus(reconScoreMap)
            #reconDetectionMap = F.normalize(reconDetectionMap, p=2, dim=1)
            reconDetection_min = torch.min(torch.min(reconScoreMap, dim=2)[0], dim=2)[0]
            reconDetection_max = torch.max(torch.max(reconScoreMap, dim=2)[0], dim=2)[0]
            reconDetection_my_max_min = torch.cat([reconDetection_min.unsqueeze(2), reconDetection_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
            reconDetectionMap = (reconScoreMap - (reconDetection_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((reconDetection_my_max_min[:, :, 1] - reconDetection_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))

            cur_detection_loss = F.mse_loss(DetectionMap, reconDetectionMap)

            til_Wk = model_dec_feature_descriptor(my_descriptor) #(b, 16, feature_dimension)
            dec_Wk = til_Wk.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2], Wk_raw.shape[3])
            #dec_Wk = F.softplus(dec_Wk) #changed: relu -> softplus
            dec_Wk = F.relu(dec_Wk)  # changed: relu -> softplus
            reconFeatureMap = dec_Wk * reconDetectionMap

            concat_recon = torch.cat((reconDetectionMap, reconFeatureMap), 1) #(b, 2n, 96, 128) channel-wise concatenation
            reconImg = model_StackedHourglassImgRecon(concat_recon) #(b, 8, 3, 192, 256)
            reconImg = reconImg[:, num_nstack-1, :, :, :] #(b,3,192,256)

            cur_recon_loss = F.mse_loss(reconImg, aefe_input.detach())

            cur_descriptorW_loss = F.mse_loss(Wk_raw, dec_Wk)

            param_loss_con = 0.1
            param_loss_sep = 1.0 #1e-2 #1.0
            param_loss_recon = 5.0
            param_loss_transf = 1e-1
            param_loss_detecionmap = 10.0 #10.0
            param_loss_descriptorW = 0.3
            param_loss_cosim = 2e-5

            #loss = param_loss_con * cur_conc_loss.cuda() + param_loss_sep * cur_sep_loss.cuda() + param_loss_recon * cur_recon_loss.cuda() + param_loss_transf * cur_transf_loss.cuda() + param_loss_detecionmap * cur_detection_loss.cuda() + param_loss_descriptorW * cur_descriptorW_loss.cuda()
            #loss = param_loss_con * cur_conc_loss.cuda() + param_loss_sep * cur_sep_loss.cuda() + param_loss_recon * cur_recon_loss.cuda() + param_loss_detecionmap * cur_detection_loss.cuda() + param_loss_descriptorW * cur_descriptorW_loss.cuda()
            #loss = param_loss_con * cur_conc_loss.cuda() + param_loss_sep * cur_sep_loss.cuda() + param_loss_recon * cur_recon_loss.cuda() + param_loss_detecionmap * cur_detection_loss.cuda() + param_loss_descriptorW * cur_descriptorW_loss.cuda() + param_loss_score * cur_score_loss.cuda(0)
            loss = param_loss_con * cur_conc_loss.cuda() + param_loss_sep * cur_sep_loss.cuda() + param_loss_recon * cur_recon_loss.cuda() + param_loss_transf * cur_transf_loss.cuda() + param_loss_detecionmap * cur_detection_loss.cuda() + param_loss_descriptorW * cur_descriptorW_loss.cuda() + param_loss_cosim * cur_cosim_loss.cuda()
            #loss = torch.tensor(loss, dtype=float, requires_grad=True)
            #loss = loss.cuda()
            #loss = 1000.0 * cur_conc_loss.cuda() + 1.0 * cur_sep_loss.cuda() + 1.0 * cur_recon_loss.cuda()

            print("concentration loss: ", param_loss_con * cur_conc_loss.item(), ",", "separation loss: ", param_loss_sep * cur_sep_loss.item(), ",", "cur_recon_loss: ", param_loss_recon * cur_recon_loss.item(),
                  ",", "transformation_loss: ", param_loss_transf * cur_transf_loss.item(), ",", "detection map loss", param_loss_detecionmap * cur_detection_loss.cuda().item(), ",", "descriptorW_loss", param_loss_descriptorW * cur_descriptorW_loss.item(), ",", "Cosim_loss", param_loss_cosim * cur_cosim_loss.item())

            # ================Backward================
            optimizer_L2.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()
            optimizer_reconDetection.zero_grad()

            loss.backward()

            optimizer_L2.step()
            optimizer_Wk_.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()
            optimizer_reconDetection.step()

            running_loss = running_loss + loss.item()
            running_recon_loss = running_recon_loss + param_loss_recon * cur_recon_loss.item()
            running_sep_loss = running_sep_loss + param_loss_sep * cur_sep_loss.item()
            running_conc_loss = running_conc_loss + param_loss_con * cur_conc_loss.item()
            running_transf_loss = running_transf_loss + param_loss_transf * cur_transf_loss.item()
            running_detection_loss = running_detection_loss + param_loss_detecionmap * cur_detection_loss.item()
            running_descriptorW_loss = running_descriptorW_loss + param_loss_descriptorW * cur_descriptorW_loss.item()
            running_cosim_loss = running_cosim_loss + param_loss_cosim * cur_cosim_loss.item()
            #running_score_loss = running_score_loss + param_loss_score * cur_score_loss.item()

            if (((epoch+1) % 5 == 0) or (epoch==0) or (epoch+1==num_epochs)):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, keypoints, epoch+1, cur_filename)
                fn_save_tfkpimg = savetfKPimg()
                fn_save_tfkpimg(tf_aefe_input, tf_keypoints, epoch + 1, cur_filename)
                img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch+1))
                save_image(reconImg, img_save_filename)
                if(epoch != 0):
                    torch.save({
                                #'model_StackedHourglassForKP': model_L2net.module.state_dict(),
                                'model_L2net': model_L2net.module.state_dict(),
                                'model_feature_descriptor': model_feature_descriptor.module.state_dict(),
                                'model_score_map': model_score_map.module.state_dict(),
                                'model_dec_feature_descriptor': model_dec_feature_descriptor.module.state_dict(),
                                'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

                                #'optimizer_StackedHourglass_kp': optimizer_L2.state_dict(),
                                'optimizer_L2': optimizer_L2.state_dict(),
                                'optimizer_Wk_': optimizer_Wk_.state_dict(),
                                'optimizer_decfeatureDescriptor': optimizer_decfeatureDescriptor.state_dict(),
                                'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),
                                'optimizer_reconDetection': optimizer_reconDetection.state_dict(),
                                }, "./SaveModelCKPT/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')
        vis.line(Y=[running_conc_loss], X=np.array([epoch]), win=plot_conc, update='append')
        vis.line(Y=[running_recon_loss], X=np.array([epoch]), win=plot_recon, update='append')
        vis.line(Y=[running_sep_loss], X=np.array([epoch]), win=plot_sep, update='append')
        vis.line(Y=[running_transf_loss], X=np.array([epoch]), win=plot_transf, update='append')
        vis.line(Y=[running_detection_loss], X=np.array([epoch]), win=plot_detection, update='append')
        vis.line(Y=[running_descriptorW_loss], X=np.array([epoch]), win=plot_weightW, update='append')
        vis.line(Y=[running_cosim_loss], X=np.array([epoch]), win=plot_cosim, update='append')
        #vis.line(Y=[running_score_loss], X=np.array([epoch]), win=plot_score, update='append')

        #saveLossData = 'epoch\t{}\trunning_loss_all\t{:.4f} \trunning_conc_loss\t{:.4f} \trunning_recon_loss\t{:.4f} \trunning_sep_loss\t{:.4f} \trunning_transf_loss\t{:.4f}\trunning_detection_loss\t{:.4f}\n'.format(epoch, running_loss, running_conc_loss, running_recon_loss, running_sep_loss, running_transf_loss, running_detection_loss, running_descriptorW_loss)
        #saveLossData = 'epoch\t{}\trunning_loss_all\t{:.4f} \trunning_conc_loss\t{:.4f} \trunning_recon_loss\t{:.4f} \trunning_sep_loss\t{:.4f} \trunning_detection_loss\t{:.4f} \trunning_score_loss\t{:.4f}\n'.format(epoch, running_loss, running_conc_loss, running_recon_loss, running_sep_loss, running_detection_loss, running_descriptorW_loss, running_score_loss)
        saveLossData = 'epoch\t{}\trunning_loss_all\t{:.4f} \trunning_conc_loss\t{:.4f} \trunning_recon_loss\t{:.4f} \trunning_sep_loss\t{:.4f} \trunning_detection_loss\n'.format(epoch, running_loss, running_conc_loss, running_recon_loss, running_sep_loss, running_detection_loss, running_descriptorW_loss)

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