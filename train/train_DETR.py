import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
import time
from model.layers import *
from model import *
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

from misc import *
from position_encoding import *
torch.multiprocessing.set_start_method('spawn', force=True)

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
voters = 200
num_queries = voters

hidden_dim = 256
feature_dimension = 256 #32

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 300
batch_size = 2 #4 #8 #4

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 4

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5  # 1e-2#1e-5 #1e-5 #5e-4
lr_drop = 200
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_DETR_Backbone = DETR_backbone(hidden_dim=hidden_dim).cuda()
    model_DETR_Backbone = nn.DataParallel(model_DETR_Backbone).cuda()
    optimizer_DETR_Backbone = torch.optim.AdamW(model_DETR_Backbone.parameters(), lr=1e-5, weight_decay=1e-4)

    model_DETR = DETR(num_voters=voters, hidden_dim=hidden_dim, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR = nn.DataParallel(model_DETR).cuda()
    optimizer_DETR = torch.optim.AdamW(model_DETR.parameters(), lr=1e-4, weight_decay=1e-4)

    model_simple_rsz = simple_rsz(inp_channel=480, oup_channel=hidden_dim).cuda()
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

    model_into_Img = IntoImg(hidden_dim=hidden_dim, img_width=my_width, img_height=my_height)
    model_into_Img = nn.DataParallel(model_into_Img).cuda()
    optimizer_intoImg = torch.optim.AdamW(model_into_Img.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(num_of_kp=num_of_kp, nstack=num_nstack,inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    #optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate,weight_decay=weight_decay)
    optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=learning_rate,weight_decay=weight_decay)

    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_DETR_Backbone, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_DETR, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_simple_rsz, lr_drop)
    lr_scheduler_optimizer4 = torch.optim.lr_scheduler.StepLR(optimizer_MakeDesc, lr_drop)
    lr_scheduler_optimizer5 = torch.optim.lr_scheduler.StepLR(optimizer_Recon_MakeDesc, lr_drop)
    lr_scheduler_optimizer6 = torch.optim.lr_scheduler.StepLR(optimizer_ReconDetection, lr_drop)
    lr_scheduler_optimizer7 = torch.optim.lr_scheduler.StepLR(optimizer_intoImg, lr_drop)
    lr_scheduler_optimizer8 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)


    ###################################################################################################################
    #call checkpoint
    if os.path.exists("./SaveModelCKPT/train_model.pth"):
    #if os.path.exists("./SaveModelCKPT/210401.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_DETR_Backbone.module.load_state_dict(checkpoint['model_DETR_Backbone'])
        model_DETR.module.load_state_dict(checkpoint['model_DETR'])
        model_simple_rsz.module.load_state_dict(checkpoint['model_simple_rsz'])
        model_MakeDesc.module.load_state_dict(checkpoint['model_MakeDesc'])
        model_recon_MakeDesc.module.load_state_dict(checkpoint['model_recon_MakeDesc'])
        model_ReconDetection.module.load_state_dict(checkpoint['model_ReconDetection'])
        model_into_Img.module.load_state_dict(checkpoint['model_into_Img'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_DETR_Backbone.load_state_dict(checkpoint['optimizer_DETR_Backbone'])
        optimizer_DETR.load_state_dict(checkpoint['optimizer_DETR'])
        optimizer_simple_rsz.load_state_dict(checkpoint['optimizer_simple_rsz'])
        optimizer_MakeDesc.load_state_dict(checkpoint['optimizer_MakeDesc'])
        optimizer_Recon_MakeDesc.load_state_dict(checkpoint['optimizer_Recon_MakeDesc'])
        optimizer_ReconDetection.load_state_dict(checkpoint['optimizer_ReconDetection'])
        optimizer_intoImg.load_state_dict(checkpoint['optimizer_intoImg'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])

        lr_scheduler_optimizer1.load_state_dict(checkpoint['lr_scheduler_optimizer1'])
        lr_scheduler_optimizer2.load_state_dict(checkpoint['lr_scheduler_optimizer2'])
        lr_scheduler_optimizer3.load_state_dict(checkpoint['lr_scheduler_optimizer3'])
        lr_scheduler_optimizer4.load_state_dict(checkpoint['lr_scheduler_optimizer4'])
        lr_scheduler_optimizer5.load_state_dict(checkpoint['lr_scheduler_optimizer5'])
        lr_scheduler_optimizer6.load_state_dict(checkpoint['lr_scheduler_optimizer6'])
        lr_scheduler_optimizer7.load_state_dict(checkpoint['lr_scheduler_optimizer7'])
        lr_scheduler_optimizer8.load_state_dict(checkpoint['lr_scheduler_optimizer8'])
    ###################################################################################################################

    dataset = my_dataset(my_width=my_width, my_height=my_height)
    #dataset = my_dataset_originalImg(my_width=my_width, my_height=my_height)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
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

            x = model_DETR_Backbone(aefe_input) #(b, 2048, 12, 40)
            attention_map, attention_score, H, Hk_kp = model_DETR(x) #(b, hw=480, hw=480), (b, hw=480), (b, 200, 256)(b, 200, 2)

            kp = get_kp(Hk_kp, my_height, my_width)

            rsz_attention_score_col = model_simple_rsz(attention_score.unsqueeze(1)) #(b, 1, 480) -> (b, 1, hidden_dim=256)
            DescMap = torch.mul(rsz_attention_score_col, H) #(b,1,256) * (b, 200, 256) = (b,200,256)

            fk = F.relu(model_MakeDesc(DescMap)) #(b, 200, 256)

            ReconDescMap = F.relu(model_recon_MakeDesc(fk)) #(b, 200, 256)

            ReconAttentionMap_tmp,  ReconAttentionMap = model_ReconDetection(kp) #kp = (b,v,2) #DetectionMap (b, 200, 256)

            attention_map_1 = ReconAttentionMap
            attention_map_2 = torch.transpose(attention_map_1, 1, 2)
            RECON_attention_map = torch.matmul(attention_map_2, attention_map_1)

            FeatureMap = ReconDescMap * ReconAttentionMap_tmp
            recon_concat = torch.cat([ReconAttentionMap_tmp, FeatureMap], dim=1) #(b,200,256) -> (b,200,512) -> (b,200,48*160=7680)

            img_recon_concat = model_into_Img(recon_concat)
            img_recon_concat_4d = img_recon_concat.reshape(cur_batch, 2*num_of_kp, my_height, my_width)
            reconImg = model_StackedHourglassImgRecon(img_recon_concat_4d)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            # Define Loss Functions!
            #separation loss
            fn_loss_separation = loss_separation(kp).cuda()
            cur_sep_loss = fn_loss_separation()

            #Recon Attention Map loss
            cur_sk_loss = F.mse_loss(RECON_attention_map, attention_map)

            #Recon Descriptor Loss
            cur_fk_loss = F.mse_loss(DescMap, ReconDescMap)

            #Recon img loss
            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            criterion = SSIM()
            cur_recon_loss_ssim = criterion(reconImg, aefe_input)
            cur_recon_loss = cur_recon_loss_l2*10 + cur_recon_loss_ssim

            #loss parameter
            #p_transf = 0.1
            #p_matching = 100000.0
            #p_cosim = 2.0
            p_sep = 1.0
            p_vk = 5*1e-5
            p_wk = 1e-6
            p_recon = 1.0

            #my_transf_loss = p_transf * cur_transf_loss
            #my_matching_loss = p_matching * cur_matching_loss
            #my_cosim_loss = p_cosim * cur_cosim_loss
            my_sep_loss = p_sep * cur_sep_loss

            my_sk_loss = p_vk * cur_sk_loss
            my_fk_loss = p_wk * cur_fk_loss
            my_recon_loss = p_recon * cur_recon_loss

            #loss = my_transf_loss + my_matching_loss + my_cosim_loss + my_sep_loss + my_vk_loss + my_wk_loss + my_recon_loss
            loss = my_sep_loss + my_recon_loss + my_sk_loss + my_fk_loss

            #print("Trans: ", my_transf_loss.item(), ", Matching: ", my_matching_loss.item(), ", Cosim: ", my_cosim_loss.item(),  ", Sep: ", my_sep_loss.item(), ", Vk: ", my_vk_loss.item(),  ", Wk: ", my_wk_loss.item(), ", Recon:", my_recon_loss.item())
            print("Sep: ", my_sep_loss.item(), ", Sk: ", my_sk_loss.item(),  ", fk: ", my_fk_loss.item(), ", Recon:", my_recon_loss.item())

            # ================Backward================
            optimizer_DETR_Backbone.zero_grad()
            optimizer_DETR.zero_grad()
            optimizer_simple_rsz.zero_grad()
            optimizer_MakeDesc.zero_grad()
            optimizer_Recon_MakeDesc.zero_grad()
            optimizer_ReconDetection.zero_grad()
            optimizer_intoImg.zero_grad()
            optimizer_ImgRecon.zero_grad()

            #if not (torch.isfinite(my_transf_loss) or torch.isfinite(my_matching_loss) or torch.isfinite(my_cosim_loss) or torch.isfinite(my_sep_loss) or torch.isfinite(my_vk_loss) or torch.isfinite(my_wk_loss) or torch.isfinite(my_recon_loss)):
            #    print("WARNING: not-finite loss, ending training")
            #    exit(1)

            loss.backward()
            #my_sk_loss.backward(retain_graph=True)
            #my_fk_loss.backward(retain_graph=True)
            #my_recon_loss.backward(retain_graph=True)
            #my_sep_loss.backward()

            optimizer_DETR_Backbone.step()
            optimizer_DETR.step()
            optimizer_simple_rsz.step()
            optimizer_MakeDesc.step()
            optimizer_Recon_MakeDesc.step()
            optimizer_ReconDetection.step()
            optimizer_intoImg.step()
            optimizer_ImgRecon.step()

            lr_scheduler_optimizer1.step()
            lr_scheduler_optimizer2.step()
            lr_scheduler_optimizer3.step()
            lr_scheduler_optimizer4.step()
            lr_scheduler_optimizer5.step()
            lr_scheduler_optimizer6.step()
            lr_scheduler_optimizer7.step()
            lr_scheduler_optimizer8.step()

            running_loss = running_loss + loss.item()

            #running_transf_loss = running_transf_loss + my_transf_loss.item()
            #running_matching_loss = running_matching_loss + my_matching_loss.item()
            #running_cosim_loss = running_cosim_loss + my_cosim_loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()

            running_vk_loss = running_vk_loss + my_sk_loss.item()
            running_wk_loss = running_wk_loss + my_fk_loss.item()
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
            'model_DETR_Backbone': model_DETR_Backbone.module.state_dict(),
            'model_DETR': model_DETR.module.state_dict(),
            'model_simple_rsz': model_simple_rsz.module.state_dict(),
            'model_MakeDesc': model_MakeDesc.module.state_dict(),
            'model_recon_MakeDesc': model_recon_MakeDesc.module.state_dict(),
            'model_ReconDetection': model_ReconDetection.module.state_dict(),
            'model_into_Img': model_into_Img.module.state_dict(),
            'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

            'optimizer_DETR_Backbone': optimizer_DETR_Backbone.state_dict(),
            'optimizer_DETR': optimizer_DETR.state_dict(),
            'optimizer_simple_rsz': optimizer_simple_rsz.state_dict(),
            'optimizer_MakeDesc': optimizer_MakeDesc.state_dict(),
            'optimizer_Recon_MakeDesc': optimizer_Recon_MakeDesc.state_dict(),
            'optimizer_ReconDetection': optimizer_ReconDetection.state_dict(),
            'optimizer_intoImg': optimizer_intoImg.state_dict(),
            'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),

            'lr_scheduler_optimizer1': lr_scheduler_optimizer1.state_dict(),
            'lr_scheduler_optimizer2': lr_scheduler_optimizer2.state_dict(),
            'lr_scheduler_optimizer3': lr_scheduler_optimizer3.state_dict(),
            'lr_scheduler_optimizer4': lr_scheduler_optimizer4.state_dict(),
            'lr_scheduler_optimizer5': lr_scheduler_optimizer5.state_dict(),
            'lr_scheduler_optimizer6': lr_scheduler_optimizer6.state_dict(),
            'lr_scheduler_optimizer7': lr_scheduler_optimizer7.state_dict(),
            'lr_scheduler_optimizer8': lr_scheduler_optimizer8.state_dict(),

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

    print("!!!!!This is train_DETR.py!!!!!")
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################