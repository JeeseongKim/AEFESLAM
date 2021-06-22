##Train DETR4 has 1 encoder for keypoints and descriptors, and 2 different decoders which shares target(=voters)

from model.StackedHourglass import *
from model.loss import *
from model.utils import *
from model.GenDescriptorMap import *
import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
# from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader
from model.misc import *
# from position_encoding import *
from model.MyDETR import *
from model.DetectionConfidenceMap import *
#from model.AEFE_Transformer import *
from model.DETR_transformer import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom

vis = visdom.Visdom()

plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))
plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))

#plot_fundamental = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Fundamental loss (KP)'))
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
batch_size = 1  # 8 #4

stacked_hourglass_inpdim_kp = input_width
# stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 4

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5  # 1e-2#1e-5 #1e-5 #5e-4
lr_drop = 200
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor

######################################################################################################################################################################################
def generate_square_subsequent_mask(height, width):
    #mask = (torch.triu(torch.ones(height, width)) == 1).transpose(0, 1)
    mask = (torch.triu(torch.ones(width, height)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.bool()
    return ~mask

def train():
    #print("1")
    #torch.cuda.set_device(0)
    #print("-1-")
    #torch.distributed.init_process_group(backend='nccl', init_method='file:///mnt/nfs/sharedfile', world_size=2, rank=0)
    #print("-2-")
    #world_size = 2

    model_start = time.time()
    #print("111")
    # model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=256, oup_dim=256, bn=False, increase=0).cuda()
    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=128, oup_dim=256, bn=False,increase=0).cuda()
    #model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=128, oup_dim=256, bn=False,increase=0).to(rank)
    #model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #print("222")
    model_DETR_kp = DETR_KPnDesc(num_voters=voters, hidden_dim=256, nheads=4, num_encoder_layers=4, num_decoder_layers=4).cuda()
    #model_DETR_kp = DETR_KPnDesc(num_voters=voters, hidden_dim=256, nheads=2, num_encoder_layers=4, num_decoder_layers=4).cuda()
    #model_DETR_kp = nn.DataParallel(model_DETR_kp).cuda()
    #model_DETR_kp = DDP(model_DETR_kp, delay_allreduce=True)
    optimizer_DETR_kp = torch.optim.AdamW(model_DETR_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #print("333")
    #model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=256, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=1280, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0).cuda()
    #model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=1280, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0).to(rank)
    #model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    #model_StackedHourglassImgRecon = DDP(model_StackedHourglassImgRecon, delay_allreduce=True)
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #print("444")
    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_StackedHourglass_kp, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_DETR_kp, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)

    ###################################################################################################################

    # call checkpoint
    MyCKPT = "/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth"
    #if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/noMatchingLoss.pth"):
    if os.path.exists(MyCKPT):
    # if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/210421_test.pth"):
    # if os.path.exists("./SaveModelCKPT/210401.pth"):
        print("-----Loading Checkpoint-----")

        checkpoint = torch.load(MyCKPT)
        #checkpoint = torch.load("/home/jsk/AEFE_SLAM/SaveModelCKPT/noMatchingLoss.pth")
        # checkpoint = torch.load("/home/jsk/AEFE_SLAM/SaveModelCKPT/210421_test.pth")

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        #model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        #model_DETR_kp.module.load_state_dict(checkpoint['model_DETR_kp'])
        #model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])
        model_StackedHourglassForKP.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_DETR_kp.load_state_dict(checkpoint['model_DETR_kp'])
        model_StackedHourglassImgRecon.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_DETR_kp.load_state_dict(checkpoint['optimizer_DETR_kp'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])

        lr_scheduler_optimizer1.load_state_dict(checkpoint['lr_scheduler_optimizer1'])
        lr_scheduler_optimizer2.load_state_dict(checkpoint['lr_scheduler_optimizer2'])
        lr_scheduler_optimizer3.load_state_dict(checkpoint['lr_scheduler_optimizer3'])

    ###################################################################################################################
    print("---preparing dataset---")
    dataset = my_dataset(my_width=my_width, my_height=my_height)
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
            torch.autograd.set_detect_anomaly(True)
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)

            if(epoch < 1):
                #theta = 0
                theta = -1.5
            if(epoch >= 1) and(epoch < 50):
                theta = random.uniform(-3, 3)
                #theta = -20
            if(epoch >= 50) and (epoch < 100):
                theta = random.uniform(-5, 5)
            if (epoch >= 100) and (epoch < 200):
                theta = random.uniform(-7, 7)
            if (epoch >= 200) and (epoch < 300):
                theta = random.uniform(-10, 10)

            #theta = random.uniform(-60, 60)
            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            ##########################################ENCODER##########################################
            cur_batch = aefe_input.shape[0]

            time1 = time.time()
            Rk = model_StackedHourglassForKP(aefe_input)
            tf_Rk = model_StackedHourglassForKP(tf_aefe_input)
            #print("featuremap: ", time.time()-time1)

            time2 = time.time()
            position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
            mask = generate_square_subsequent_mask(Rk.shape[2], Rk.shape[3]).cuda()
            mask = mask.repeat(cur_batch, 1, 1)
            pos = position_embedding(aefe_input, mask)
            tf_pos = position_embedding(tf_aefe_input, mask)
            #pos = pos.repeat(cur_batch, 1, 1, 1)
            #tf_pos = tf_pos.repeat(cur_batch, 1, 1, 1)
            #print("PE and Mask: ", time.time() - time2)

            time3 = time.time()
            kp, desc = model_DETR_kp(Rk, mask, pos)
            tf_kp, tf_desc = model_DETR_kp(tf_Rk, mask, tf_pos)
            #print("DETR: ", time.time() - time3)

            #kp = kp.squeeze(1)
            #desc = desc.squeeze(1)
            #tf_kp = tf_kp.squeeze(1)
            #tf_desc = tf_desc.squeeze(1)
            ##For binary descriptors##
            #desc = torch.sign(desc)
            #tf_desc = torch.sign(tf_desc)
            #MySTE = StraightThroughEstimator()
            #desc = MySTE(desc)
            #tf_desc = MySTE(tf_desc)

            ##########################################DECODER##########################################
            time4 = time.time()
            fn_ReconKp = ReconWithKP(Rk.shape[2], Rk.shape[3])
            #reconstruction sigma = 0.1, 0.5, 1.0, 3.0, 5,
            #reconstruction sigma = 0.05, 0.1, 0.3, 0.5, 0.7
            recon_kp_1 = fn_ReconKp(kp, 0.1)
            recon_kp_2 = fn_ReconKp(kp, 0.5)
            recon_kp_3 = fn_ReconKp(kp, 1.0)
            recon_kp_4 = fn_ReconKp(kp, 3.0)
            recon_kp_5 = fn_ReconKp(kp, 5.0)
            #print("Decoder: ", time.time() - time4)

            kp[:, :, 0] = kp[:, :, 0] * my_width
            kp[:, :, 1] = kp[:, :, 1] * my_height

            tf_kp[:, :, 0] = tf_kp[:, :, 0] * my_width
            tf_kp[:, :, 1] = tf_kp[:, :, 1] * my_height

            kp = kp.int()
            tf_kp = tf_kp.int()

            my_feature = torch.cat([kp, desc], dim=2)
            my_tf_feature = torch.cat([tf_kp, tf_desc], dim=2)

            time5 = time.time()
            reconInput_1 = (recon_kp_1.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_2 = (recon_kp_2.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_3 = (recon_kp_3.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_4 = (recon_kp_4.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk
            reconInput_5 = (recon_kp_5.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk

            reconInput = torch.cat([reconInput_1, reconInput_2, reconInput_3, reconInput_4, reconInput_5], dim=1)
            reconImg = model_StackedHourglassImgRecon(reconInput)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)
            #print("Recon: ", time.time() - time5)

            #tf_reconImg = model_StackedHourglassImgRecon(tf_reconInput)
            #tf_reconImg = tf_reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            ##############################################LOSS#############################################
            # Define Loss Functions!

            # separation loss
            time6 = time.time()
            fn_loss_separation = loss_separation().cuda()
            cur_sep_loss = fn_loss_separation(kp) + fn_loss_separation(tf_kp)
            #print("SepLoss: ", time.time()-time6)

            # similarity loss btw Dk and tf_Dk
            time7 = time.time()
            fn_loss_cosim = loss_cosim().cuda()
            cur_cosim_loss = fn_loss_cosim(Rk, tf_Rk)
            #print("Cosim: ", time.time()-time7)

            time8 = time.time()
            fn_hungarian_matcher = HungarianMatcherNLoss()
            loss_kp_match, loss_desc = fn_hungarian_matcher(-theta, my_feature, my_tf_feature, my_width, my_height)
            #print("HungarianMatching: ", time.time()-time8)

            # NCC loss
            #fn_ncc_loss = NCC_loss()
            #loss_ncc = fn_ncc_loss(aefe_input, tf_aefe_input, match, kp, tf_kp)

            # Reconstruction Loss
            time9 = time.time()
            criterion = SSIM()
            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            cur_recon_loss_l1 = F.l1_loss(reconImg, aefe_input)
            cur_recon_loss_ssim = (1 - criterion(reconImg, aefe_input))
            #print("ReconLoss: ", time.time()-time9)

            p_sep_loss = 0.5
            #p_fundamental_loss = 5
            p_kp_loss = 1.0
            p_desc_loss = 15.0
            p_cosim_loss = 10.0
            p_recon_img_l2 = 2.0
            p_recon_img_l1 = 2.0
            p_recon_img_ssim = 1.0

            my_sep_loss = p_sep_loss * cur_sep_loss
            #my_fundamental_loss = p_fundamental_loss * loss_fundamental
            my_kp_loss = p_kp_loss * loss_kp_match
            my_desc_loss = p_desc_loss * loss_desc
            my_cosim_loss = p_cosim_loss * cur_cosim_loss
            my_recon_loss_l2 = p_recon_img_l2 * cur_recon_loss_l2
            my_recon_loss_l1 = p_recon_img_l1 * cur_recon_loss_l1
            my_recon_loss_ssim = p_recon_img_ssim * cur_recon_loss_ssim

            #loss = (my_sep_loss + my_fundamental_loss + my_kp_loss + my_desc_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            if(epoch < 5):
                print("No Matching Loss")
                loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
                #loss = (my_sep_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            else:
                print("Computing Matching Loss")
                loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
                #loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)

            #loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)

            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", Fundamental: ", '%.4f' % my_fundamental_loss.item(), ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
            #      ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
                  ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
            #      ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(),
            #      ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            # ================Backward================
            time9 = time.time()
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_DETR_kp.zero_grad()
            optimizer_ImgRecon.zero_grad()

            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_DETR_kp.step()
            optimizer_ImgRecon.step()
            #print("Optimizer: ", time.time()-time9)
            running_loss = running_loss + loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()
            #running_fundamental_loss = running_fundamental_loss + my_fundamental_loss.item()
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

            #'model_StackedHourglassForKP': model_StackedHourglassForKP.module.state_dict(),
            #'model_DETR_kp': model_DETR_kp.module.state_dict(),
            #'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),
            'model_StackedHourglassForKP': model_StackedHourglassForKP.state_dict(),
            'model_DETR_kp': model_DETR_kp.state_dict(),
            'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.state_dict(),

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

        #vis.line(Y=[running_fundamental_loss], X=np.array([epoch]), win=plot_fundamental, update='append')
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
    #if not os.path.exists("SaveTFReconstructedImg"):
    #    os.makedirs("SaveTFReconstructedImg")
    #if not os.path.exists("SaveHeatMapImg"):
    #    os.makedirs("SaveHeatMapImg")
    #if not os.path.exists("SaveModelCKPT"):
    #    os.makedirs("SaveModelCKPT")

    print("!!210618!!")
    print("!!!!!This is train_allnew2.py!!!!!")
    train()

##########################################################################################################################