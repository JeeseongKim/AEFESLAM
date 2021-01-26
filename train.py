import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import time
from model.layers import Linear, dec_Linear
from StackedHourglass import StackedHourglassForKP, StackedHourglassImgRecon
from DetectionConfidenceMap import DetectionConfidenceMap2keypoint, ReconDetectionConfidenceMap, create_softmask, LinearReconScoreMap, DetectionConfidenceMap2keypoint_test
from loss import loss_concentration, loss_separation, loss_transformation
from utils import my_dataset, saveKPimg, make_transformation_M
import os
import numpy as np
from parallel import DataParallelModel, DataParallelCriterion
import seaborn
from tqdm import tqdm
import argparse
import torchvision
import random


import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import TensorDataset, DataLoader

import visdom
vis = visdom.Visdom()

#plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))
#plot_recon = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction Loss'))
#plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))
#plot_conc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration Loss'))
#plot_transf = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
h0 = 480
w0 = 640

num_of_kp = 345
feature_dimension = 32

my_width = 272 #208
my_height = 80 #64

input_width = my_width

num_epochs = 100
batch_size = 2

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp #number of my keypoints

num_nstack = 3

learning_rate = 1e-4 #1e-3
weight_decay = 1e-5 #1e-5 #5e-4
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP)
    model_StackedHourglassForKP.cuda()
    optimizer_StackedHourglass_kp = torch.optim.Adam(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_feature_descriptor = Linear(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor)
    model_feature_descriptor.cuda()
    optimizer_Wk_ = torch.optim.Adam(model_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_score_map = LinearReconScoreMap(img_width=my_width, img_height=my_height, num_of_kp=num_of_kp)
    model_score_map = nn.DataParallel(model_score_map)
    model_score_map.cuda()
    optimizer_reconDetection = torch.optim.Adam(model_score_map.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_dec_feature_descriptor = dec_Linear(feature_dimension=feature_dimension, img_width=my_height, img_height=my_width)
    model_dec_feature_descriptor = nn.DataParallel(model_dec_feature_descriptor)
    model_dec_feature_descriptor.cuda()
    optimizer_decfeatureDescriptor = torch.optim.Adam(model_dec_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon)
    model_StackedHourglassImgRecon.cuda()
    optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ###################################################################################################################

    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_feature_descriptor.module.load_state_dict(checkpoint['model_feature_descriptor'])
        model_score_map.module.load_state_dict(checkpoint['model_score_map'])
        model_dec_feature_descriptor.module.load_state_dict(checkpoint['model_dec_feature_descriptor'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_Wk_.load_state_dict(checkpoint['optimizer_Wk_'])
        optimizer_reconDetection.load_state_dict(checkpoint['optimizer_reconDetection'])
        optimizer_decfeatureDescriptor.load_state_dict(checkpoint['optimizer_decfeatureDescriptor'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])


    ###################################################################################################################

    """
    Batchnorm2d = torch.nn.BatchNorm2d(num_of_kp)
    Batchnorm2d = nn.DataParallel(Batchnorm2d)
    Batchnorm2d.cuda()

    recon_Batchnorm2d = torch.nn.BatchNorm2d(num_of_kp * 2)
    recon_Batchnorm2d = nn.DataParallel(recon_Batchnorm2d)
    recon_Batchnorm2d.cuda()
    """
    #my_transform = torchvision.transforms.RandomAffine((-60, 60), translate=(0.3, 0.3), scale=None, shear=None, resample=0, fillcolor=0)
    #dataset = my_dataset(transform=my_transform)
    dataset = my_dataset(my_width, my_height)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    save_loss = []
    print("***", time.time() - model_start) #7.26 ~ 63
    my_iteration = 0

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        running_recon_loss = 0
        running_sep_loss = 0
        running_conc_loss = 0
        running_transf_loss = 0

        for i, data in enumerate(tqdm(train_loader)):
            start = time.time()
            input_img, cur_filename, kp_img = data
            #print("data loader time: ", time.time() - start)
            aefe_input = input_img.cuda() #(b, 3, height, width)

            cur_batch = aefe_input.shape[0]

            ##########################################ENCODER##########################################
            theta = random.uniform(-60, 60)
            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input) #randomly rotated image

            combined_hm_preds = model_StackedHourglassForKP(aefe_input)[:, num_nstack-1, :, :, :]
            tf_combined_hm_preds = model_StackedHourglassForKP(tf_aefe_input)[:, num_nstack - 1, :, :, :]

            #save heat map
            #if (epoch % 10 == 0):
            #heatmap_save_filename = ("SaveHeatMapImg/heatmap_%s_epoch_%s.jpg" % (cur_filename, epoch))
            #seaborn.heatmap(combined_hm_preds[0,0,:,:].detach().cpu().clone().numpy())
            #save_image(combined_hm_preds, heatmap_save_filename)
            #plt.savefig(heatmap_save_filename)

            #combined_hm_preds = Batchnorm2d(combined_hm_preds)
            start2 = time.time()

            fn_DetectionConfidenceMap2keypoint = DetectionConfidenceMap2keypoint()
            DetectionMap, keypoints, zeta, tf_keypoints = fn_DetectionConfidenceMap2keypoint(combined_hm_preds, tf_combined_hm_preds, cur_batch)

            #print("encoding: ", time.time()-start2)

            start3 = time.time()
            fn_softmask = create_softmask()
            softmask = fn_softmask(DetectionMap, zeta)  # (b,k,96,128)
            #print("softmask_creation: ", time.time() - start3)

            start_loss = time.time()
            fn_loss_concentration = loss_concentration().cuda()
            fn_loss_separation = loss_separation().cuda()
            fn_loss_transformation = loss_transformation().cuda()

            cur_conc_loss = fn_loss_concentration(softmask)
            cur_sep_loss = fn_loss_separation(keypoints)
            cur_transf_loss = fn_loss_transformation(theta, keypoints, tf_keypoints, cur_batch, num_of_kp)
            #print("loss computation time:", time.time() - start_loss)

            start4 = time.time()
            get_descriptors = combined_hm_preds
            fn_relu = torch.nn.ReLU().cuda()
            #leakyrelu4descriptors = torch.nn.LeakyReLU(0.01).cuda()
            #get_descriptors = fn_relu(get_descriptors) #(b,k,96,128)

            Wk_raw = get_descriptors * DetectionMap
            Wk_rsz = Wk_raw.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2] * Wk_raw.shape[3])
            Wk_ = model_feature_descriptor(Wk_rsz) #(b, k, 96*128) -> (b,k,128)
            mul_all_torch = (softmask*fn_relu(get_descriptors)).sum(dim=[2, 3]).unsqueeze(2)
            my_descriptor = (mul_all_torch * fn_relu(Wk_)) #(b, k, 128)
            #print("descriptor time", time.time() - start4)

            ##########################################DECODER##########################################
            start5 = time.time()
            #########
            #fn_reconDetectionMap = ReconDetectionConfidenceMap()
            #reconDetectionMap = fn_reconDetectionMap(keypoints, DetectionMap)
            ##########
            reconScoreMap = model_score_map(keypoints, DetectionMap)
            softmax = torch.nn.Softmax(dim=1)
            reconDetectionMap = softmax(reconScoreMap)
            #reconDetectionMap = fn_relu(reconDetectionMap)
            #print("recon Detection Map time:", time.time() - start5)

            start6 = time.time()
            til_Wk = model_dec_feature_descriptor(my_descriptor) #(b, 16, feature_dimension)
            dec_Wk = til_Wk.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2], Wk_raw.shape[3])
            dec_Wk = fn_relu(dec_Wk)
            reconFeatureMap = dec_Wk * reconDetectionMap
            #print("recon Feature Map time: ", time.time() - start6)

            start7 = time.time()
            concat_recon = torch.cat((reconDetectionMap, reconFeatureMap), 1) #(b, 2n, 96, 128) channel-wise concatenation
            #concat_recon = recon_Batchnorm2d(concat_recon)

            reconImg = model_StackedHourglassImgRecon(concat_recon) #(b, 8, 3, 192, 256)
            reconImg = reconImg[:, num_nstack-1, :, :, :] #(b,3,192,256)
            #print("reconstruction time: ", time.time() - start7)

            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()
            optimizer_reconDetection.zero_grad()

            start8 = time.time()
            cur_recon_loss = F.mse_loss(reconImg, aefe_input.detach())

            param_loss_con = 50.0
            param_loss_sep = 0.1
            param_loss_recon = 1.0
            param_loss_transf = 1e-6
            loss = param_loss_con * cur_conc_loss.cuda() + param_loss_sep * cur_sep_loss.cuda() + param_loss_recon * cur_recon_loss.cuda() + param_loss_transf * cur_transf_loss.cuda()
            #loss = 1000.0 * cur_conc_loss.cuda() + 1.0 * cur_sep_loss.cuda() + 1.0 * cur_recon_loss.cuda()

            #print("loss computation time: ", time.time() - start8)

            print("\n", "Raw Loss=>", "concentration loss: ", cur_conc_loss.item(), ",", "separation loss: ", cur_sep_loss.item(), ",", "cur_recon_loss: ", cur_recon_loss.item(), "cur_transf_loss: ", cur_transf_loss.item())
            print("Computing Loss=>", "concentration loss: ", param_loss_con * cur_conc_loss.item(), ",", "separation loss: ", param_loss_sep * cur_sep_loss.item(), ",", "cur_recon_loss: ", param_loss_recon * cur_recon_loss.item(), "cur_transf_loss: ", param_loss_transf * cur_transf_loss.item())
            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_Wk_.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()
            optimizer_reconDetection.step()

            running_loss = running_loss + loss.item()
            running_recon_loss = running_recon_loss + param_loss_recon * cur_recon_loss.item()
            running_sep_loss = running_sep_loss + param_loss_sep * cur_sep_loss.item()
            running_conc_loss = running_conc_loss + param_loss_con * cur_conc_loss.item()
            running_transf_loss = running_transf_loss + param_loss_transf * cur_transf_loss.item()

            print("Running Loss=>", "All loss", running_loss, "concentration loss: ", running_conc_loss, ",", "separation loss: ", running_sep_loss, ",", "reconstruction loss: ", running_recon_loss, "transformation loss: ", running_transf_loss)

            start13 = time.time()
            if (epoch % 5 == 0):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, keypoints, epoch, cur_filename)
                img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch))
                save_image(reconImg, img_save_filename)
                if(epoch != 0):
                    torch.save({
                                'model_StackedHourglassForKP': model_StackedHourglassForKP.module.state_dict(),
                                'model_feature_descriptor': model_feature_descriptor.module.state_dict(),
                                'model_score_map': model_score_map.module.state_dict(),
                                'model_dec_feature_descriptor': model_dec_feature_descriptor.module.state_dict(),
                                'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

                                'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
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

        saveLossData = 'epoch\t{}\trunning_loss_all\t{:.4f} \trunning_conc_loss\t{:.4f} \trunning_recon_loss\t{:.4f} \trunning_sep_loss\t{:.4f} \trunning_transf_loss\t{:.4f}\n'.format(epoch, running_loss, running_conc_loss, running_recon_loss, running_sep_loss, running_transf_loss)

        saveLossTxt.write(saveLossData)

        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))

def test():
    num_of_kp = 345
    feature_dimension = 32

    my_width = 272  # 208
    my_height = 80  # 64
    input_width = my_width

    stacked_hourglass_inpdim_kp = input_width
    stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

    num_nstack = 3

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP)
    model_StackedHourglassForKP.cuda()
    optimizer_StackedHourglass_kp = torch.optim.Adam(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_feature_descriptor = Linear(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor)
    model_feature_descriptor.cuda()
    optimizer_Wk_ = torch.optim.Adam(model_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_feature_descriptor.module.load_state_dict(checkpoint['model_feature_descriptor'])
        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_Wk_.load_state_dict(checkpoint['optimizer_Wk_'])

    dataset = my_dataset(my_width, my_height)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    final_keypoints = []
    final_descriptor = []

    for i, data in enumerate(tqdm(test_loader)):
        input_img, cur_filename, kp_img = data
        aefe_input = input_img.cuda()  # (b, 3, height, width)
        cur_batch = aefe_input.shape[0]
        combined_hm_preds = model_StackedHourglassForKP(aefe_input)[:, num_nstack - 1, :, :, :]

        # save heat map
        # if (epoch % 10 == 0):
        # heatmap_save_filename = ("SaveHeatMapImg/heatmap_%s_epoch_%s.jpg" % (cur_filename, epoch))
        # seaborn.heatmap(combined_hm_preds[0,0,:,:].detach().cpu().clone().numpy())
        # save_image(combined_hm_preds, heatmap_save_filename)
        # plt.savefig(heatmap_save_filename)

        fn_DetectionConfidenceMap2keypoint_test = DetectionConfidenceMap2keypoint_test()
        DetectionMap, keypoints, zeta = fn_DetectionConfidenceMap2keypoint_test(combined_hm_preds, cur_batch)

        fn_softmask = create_softmask()
        softmask = fn_softmask(DetectionMap, zeta)  # (b,k,96,128)

        get_descriptors = combined_hm_preds.cuda(1)
        fn_relu = torch.nn.ReLU().cuda(1)

        Wk_raw = get_descriptors * DetectionMap.cuda(1)
        Wk_rsz = Wk_raw.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2] * Wk_raw.shape[3])
        Wk_ = model_feature_descriptor(Wk_rsz)  # (b, k, 96*128) -> (b,k,128)
        mul_all_torch = (softmask.cuda(1) * fn_relu(get_descriptors)).sum(dim=[2, 3]).unsqueeze(2)
        my_descriptor = (mul_all_torch * fn_relu(Wk_).cuda(1))  # (b, k, 128)
        final_keypoints.append(keypoints)
        final_descriptor.append(my_descriptor)

    final_keypoints = torch.from_numpy(np.array(final_keypoints))
    final_descriptor = torch.from_numpy(np.array(final_descriptor))

    return final_keypoints, final_descriptor
##########################################################################################################################
if __name__ == '__main__':
    torch.cuda.empty_cache()
    if not os.path.exists("SaveKPImg"):
        os.makedirs("SaveKPImg")
    if not os.path.exists("SaveReconstructedImg"):
        os.makedirs("SaveReconstructedImg")
    if not os.path.exists("SaveHeatMapImg"):
        os.makedirs("SaveHeatMapImg")
    if not os.path.exists("SaveModelCKPT"):
        os.makedirs("SaveModelCKPT")

    train()
    #test()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################