##Train DETR4 has 1 encoder for keypoints and descriptors, and 2 different decoders which shares target(=voters)

from model.StackedHourglass import *
from loss import *
from utils import *
from GenDescriptorMap import *
import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
# from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader
from misc import *
# from position_encoding import *
from model.MyDETR import *
from model.DetectionConfidenceMap import *
from model.AEFE_Transformer import *

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom

vis = visdom.Visdom()

plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))
plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))

plot_fundamental = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Fundamental loss (KP)'))
plot_kp_matching = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Matching loss (KP)'))
plot_desc_matching = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Matching loss (Desc)'))

plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosim Loss'))

plot_recon_L2 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction L2 Loss'))
plot_recon_L1 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction L1 Loss'))
plot_recon_SSIM = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction SSIM Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)

#########################################parameter#########################################
num_of_kp = 300
voters = num_of_kp
num_queries = voters

hidden_dim = 256
feature_dimension = 256  # 32

my_width = 320 #160  # 272 #96 #272 #208
my_height = 96 #48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 300
batch_size = 2  # 8 #4

stacked_hourglass_inpdim_kp = input_width
# stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 2

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5  # 1e-2#1e-5 #1e-5 #5e-4
lr_drop = 200
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor

######################################################################################################################################################################################
def train():
    model_start = time.time()

    # model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=256, oup_dim=256, bn=False, increase=0).cuda()
    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=128, oup_dim=256, bn=False,increase=0).cuda()
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_DETR_kp = DETR_KPnDesc(num_voters=voters, hidden_dim=256, nheads=4, num_encoder_layers=4, num_decoder_layers=4).cuda()
    model_DETR_kp = nn.DataParallel(model_DETR_kp).cuda()
    optimizer_DETR_kp = torch.optim.AdamW(model_DETR_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=256, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=1280, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_StackedHourglass_kp, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_DETR_kp, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)

    ###################################################################################################################
    # call checkpoint

    if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth"):
        # if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/210421_test.pth"):
        # if os.path.exists("./SaveModelCKPT/210401.pth"):
        print("-----Loading Checkpoint-----")

        checkpoint = torch.load("/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth")
        # checkpoint = torch.load("/home/jsk/AEFE_SLAM/SaveModelCKPT/210421_test.pth")

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_DETR_kp.module.load_state_dict(checkpoint['model_DETR_kp'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_DETR_kp.load_state_dict(checkpoint['optimizer_DETR_kp'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])

        lr_scheduler_optimizer1.load_state_dict(checkpoint['lr_scheduler_optimizer1'])
        lr_scheduler_optimizer2.load_state_dict(checkpoint['lr_scheduler_optimizer2'])
        lr_scheduler_optimizer3.load_state_dict(checkpoint['lr_scheduler_optimizer3'])

    ###################################################################################################################

    dataset = my_dataset(my_width=my_width, my_height=my_height)
    # dataset = my_dataset(my_width=640, my_height=192)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0

        running_sep_loss = 0

        running_fundamental_loss = 0
        running_kp_match_loss = 0
        running_desc_match_loss = 0

        running_cosim_loss = 0

        running_recon_loss_l2 = 0
        running_recon_loss_l1 = 0
        running_recon_loss_ssim = 0

        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)

            theta = random.uniform(-10, 10)
            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            ##########################################ENCODER##########################################

            Rk = model_StackedHourglassForKP(aefe_input)
            tf_Rk = model_StackedHourglassForKP(tf_aefe_input)

            cur_batch = Rk.shape[0]

            #Rk_flatten = Rk.flatten(2)
            #tf_Rk_flatten = tf_Rk.flatten(2)

            #kp, desc = model_DETR_kp(Rk_flatten, Rk)
            #tf_kp, tf_desc = model_DETR_kp(tf_Rk_flatten, tf_Rk)
            kp, desc = model_DETR_kp(Rk)
            tf_kp, tf_desc = model_DETR_kp(tf_Rk)

            kp = kp.view(cur_batch, num_of_kp, 2)
            tf_kp = tf_kp.view(cur_batch, num_of_kp, 2)

            desc = desc.view(cur_batch, num_of_kp, feature_dimension)
            tf_desc = tf_desc.view(cur_batch, num_of_kp, feature_dimension)

            #desc = torch.bernoulli(desc)
            #tf_desc = torch.bernoulli(tf_desc)

            desc = torch.sign(desc)
            tf_desc = torch.sign(tf_desc)

            MySTE = StraightThroughEstimator()
            desc = MySTE(desc)
            tf_desc = MySTE(tf_desc)
            ##########################################DECODER##########################################
            fn_ReconKp = ReconWithKP(Rk.shape[2], Rk.shape[3])
            recon_kp_1 = fn_ReconKp(kp, 0.1)
            recon_kp_2 = fn_ReconKp(kp, 0.5)
            recon_kp_3 = fn_ReconKp(kp, 1.0)
            recon_kp_4 = fn_ReconKp(kp, 3.0)
            recon_kp_5 = fn_ReconKp(kp, 5.0)
            #recon_kp_5 = fn_ReconKp(kp, 10.0)

            #recon_tf_kp = fn_ReconKp(tf_kp, 1.0)  # (b,200,48,160)

            kp[:, :, 0] = kp[:, :, 0] * my_width
            kp[:, :, 1] = kp[:, :, 1] * my_height

            tf_kp[:, :, 0] = tf_kp[:, :, 0] * my_width
            tf_kp[:, :, 1] = tf_kp[:, :, 1] * my_height

            kp = kp.int()
            tf_kp = tf_kp.int()

            my_feature = torch.cat([kp, desc], dim=2)
            my_tf_feature = torch.cat([tf_kp, tf_desc], dim=2)

            #reconInput = (recon_kp.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1)
            reconInput_1 = (recon_kp_1.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_2 = (recon_kp_2.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_3 = (recon_kp_3.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_4 = (recon_kp_4.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_5 = (recon_kp_5.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            #tf_reconInput = (recon_tf_kp.unsqueeze(2) * tf_desc.unsqueeze(3).unsqueeze(4)).mean(1)

            reconInput = torch.cat([reconInput_1, reconInput_2, reconInput_3, reconInput_4, reconInput_5], dim=1)
            reconImg = model_StackedHourglassImgRecon(reconInput)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            #tf_reconImg = model_StackedHourglassImgRecon(tf_reconInput)
            #tf_reconImg = tf_reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            ##############################################LOSS#############################################
            # Define Loss Functions!

            # separation loss
            fn_loss_separation = loss_separation(kp).cuda()
            tf_fn_loss_separation = loss_separation(tf_kp).cuda()
            cur_sep_loss = fn_loss_separation() + tf_fn_loss_separation()
            #cur_sep_loss = fn_loss_separation()

            # similarity loss btw Dk and tf_Dk
            fn_loss_cosim = loss_cosim(Rk, tf_Rk).cuda()
            cur_cosim_loss = fn_loss_cosim()

            # Encoder Loss
            # feature matching loss
            fn_hungarian_matcher = HungarianMatcher(cost_kp=1.0, cost_desc=1)
            match = fn_hungarian_matcher(-theta, my_feature, my_tf_feature, my_width, my_height)

            fn_matching_loss = matcher_criterion()
            loss_fundamental, loss_kp_match, loss_desc = fn_matching_loss(match, kp, tf_kp, desc, tf_desc)

            # Reconstruction Loss
            criterion = SSIM()
            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            cur_recon_loss_l1 = F.l1_loss(reconImg, aefe_input)
            cur_recon_loss_ssim = (1 - criterion(reconImg, aefe_input))

            p_sep_loss = 0.5
            p_fundamental_loss = 5
            p_kp_loss = 1.0
            p_desc_loss = 5.0
            p_cosim_loss = 1.0
            p_recon_img_l2 = 2.0
            p_recon_img_l1 = 2.0
            p_recon_img_ssim = 1.0

            my_sep_loss = p_sep_loss * cur_sep_loss
            my_fundamental_loss = p_fundamental_loss * loss_fundamental
            my_kp_loss = p_kp_loss * loss_kp_match
            my_desc_loss = p_desc_loss * loss_desc
            my_cosim_loss = p_cosim_loss * cur_cosim_loss
            my_recon_loss_l2 = p_recon_img_l2 * cur_recon_loss_l2
            my_recon_loss_l1 = p_recon_img_l1 * cur_recon_loss_l1
            my_recon_loss_ssim = p_recon_img_ssim * cur_recon_loss_ssim

            loss = (my_sep_loss + my_fundamental_loss + my_kp_loss + my_desc_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)

            print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", Fundamental: ", '%.4f' % my_fundamental_loss.item(), ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
                  ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
            #      ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(),
            #      ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_DETR_kp.zero_grad()
            optimizer_ImgRecon.zero_grad()

            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_DETR_kp.step()
            optimizer_ImgRecon.step()

            running_loss = running_loss + loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()
            running_fundamental_loss = running_fundamental_loss + my_fundamental_loss.item()
            running_kp_match_loss = running_kp_match_loss + my_kp_loss.item()
            running_desc_match_loss = running_desc_match_loss + my_desc_loss.item()
            running_cosim_loss = running_cosim_loss + my_cosim_loss.item()
            running_recon_loss_l2 = running_recon_loss_l2 + my_recon_loss_l2.item()
            running_recon_loss_l1 = running_recon_loss_l1 + my_recon_loss_l1.item()
            running_recon_loss_ssim = running_recon_loss_ssim + my_recon_loss_ssim.item()

            if (((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                # if (((epoch + 1) % 2 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                # print("epoch: ", epoch)
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, kp, epoch + 1, cur_filename)
                fn_save_tfkpimg = savetfKPimg()
                fn_save_tfkpimg(tf_aefe_input, tf_kp, epoch + 1, cur_filename)
                img_save_filename = ("/home/jsk/AEFE_SLAM/SaveReconstructedImg/%s_ep_%s.jpg" % (cur_filename, epoch + 1))
                #tf_img_save_filename = ("/home/jsk/AEFE_SLAM/SaveTFReconstructedImg/%s_ep_%s.jpg" % (cur_filename, epoch + 1))
                save_image(reconImg, img_save_filename)
                #save_image(tf_reconImg, tf_img_save_filename)

        # if (epoch != 0) and ((epoch+1) % 5 == 0):
        torch.save({
            'epoch': epoch,

            'model_StackedHourglassForKP': model_StackedHourglassForKP.module.state_dict(),
            'model_DETR_kp': model_DETR_kp.module.state_dict(),
            'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

            'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
            'optimizer_DETR_kp': optimizer_DETR_kp.state_dict(),
            'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),

            'lr_scheduler_optimizer1': lr_scheduler_optimizer1.state_dict(),
            'lr_scheduler_optimizer2': lr_scheduler_optimizer2.state_dict(),
            'lr_scheduler_optimizer3': lr_scheduler_optimizer3.state_dict(),

            'loss': loss,

        }, "/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')

        vis.line(Y=[running_sep_loss], X=np.array([epoch]), win=plot_sep, update='append')

        vis.line(Y=[running_fundamental_loss], X=np.array([epoch]), win=plot_fundamental, update='append')
        vis.line(Y=[running_kp_match_loss], X=np.array([epoch]), win=plot_kp_matching, update='append')
        vis.line(Y=[running_desc_match_loss], X=np.array([epoch]), win=plot_desc_matching, update='append')

        vis.line(Y=[running_cosim_loss], X=np.array([epoch]), win=plot_cosim, update='append')

        vis.line(Y=[running_recon_loss_l2], X=np.array([epoch]), win=plot_recon_L2, update='append')
        vis.line(Y=[running_recon_loss_l1], X=np.array([epoch]), win=plot_recon_L1, update='append')
        vis.line(Y=[running_recon_loss_ssim], X=np.array([epoch]), win=plot_recon_SSIM, update='append')

        # saveLossData = 'epoch\t{}\tAll_Loss\t{:.4f} \tTrans\t{:.4f} \tMatching\t{:.4f}\tCosim\t{:.4f} \tSep\t{:.4f}\tVk\t{:.4f}\tWk\t{:.4f}\tRecon\t{:.4f}\n'.format(
        #    epoch, running_loss, running_transf_loss, running_matching_loss, running_cosim_loss, running_sep_loss, running_vk_loss, running_wk_loss, running_recon_loss)

        # saveLossTxt.write(saveLossData)

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
    if not os.path.exists("SaveTFReconstructedImg"):
        os.makedirs("SaveTFReconstructedImg")
    if not os.path.exists("SaveHeatMapImg"):
        os.makedirs("SaveHeatMapImg")
    if not os.path.exists("SaveModelCKPT"):
        os.makedirs("SaveModelCKPT")

    print("!!210512!!><")
    print("!!!!!This is train_allnew2.py!!!!!")
    train()

##########################################################################################################################