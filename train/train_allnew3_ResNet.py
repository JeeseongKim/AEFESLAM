##Train DETR4 has 1 encoder for keypoints and descriptors, and 2 different decoders which shares target(=voters)

from model.StackedHourglass import *
from model.loss import *
from model.utils import *
from model.GenDescriptorMap import *
import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader
from model.misc import *
from model.MyDETR import *
from model.DetectionConfidenceMap import *
from model.DETR_transformer import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import model.DETR_backbone import *

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom

vis = visdom.Visdom()

plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))
plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))

#plot_fundamental = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Fundamental loss (KP)'))
plot_cost_loss = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cost loss'))
plot_kp_matching = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Matching loss (KP)'))
plot_desc_matching = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Matching loss (Desc)'))

plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosim Loss'))

plot_recon_L2 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction L2 Loss'))
plot_recon_L1 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction L1 Loss'))
plot_recon_SSIM = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction SSIM Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)

#########################################parameter#########################################
num_of_kp = 200
voters = num_of_kp
num_queries = voters

hidden_dim = 256
feature_dimension = 256  # 32

#kitti
#my_width = 320 #160  # 272 #96 #272 #208
#my_height = 96 #48  # 80 #32 #80 #64

#oxford
my_width = 192 #320
my_height = 128 #224

input_width = my_width

num_epochs = 500
batch_size = 1 #4  # 8 #4

stacked_hourglass_inpdim_kp = input_width
# stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 2

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5 # 1e-2#1e-5 #1e-5 #5e-4
lr_drop = 400
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
    model_start = time.time()

    model_FeatureMap = ResNetBackbone(hidden_dim=256).cuda()
    model_FeatureMap = nn.DataParallel(model_FeatureMap).cuda()
    optimizer_FeatureMap = torch.optim.AdamW(model_FeatureMap.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #model_DETR_kp = DETR_KPnDesc(num_voters=voters, hidden_dim=256, nheads=4, num_encoder_layers=4, num_decoder_layers=4).cuda()
    model_DETR_kp = DETR_KPnDesc_only(num_voters=voters, hidden_dim=256, nheads=4, num_encoder_layers=4, num_decoder_layers=4).cuda()
    #model_DETR_kp = DETR_KPnDesc(num_voters=voters, hidden_dim=256, nheads=2, num_encoder_layers=4, num_decoder_layers=4).cuda()
    model_DETR_kp = nn.DataParallel(model_DETR_kp).cuda()
    optimizer_DETR_kp = torch.optim.AdamW(model_DETR_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=256, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=1280, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0).cuda() #kitti
    #model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR_dcn(input_channel=1280, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0).cuda() #kitti
    #model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(input_channel=512, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0).cuda() #kitti
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_StackedHourglass_kp, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_DETR_kp, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)

    ###################################################################################################################

    # call checkpoint
    MyCKPT = "/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth"
    #if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/noMatchingLoss.pth"):
    if os.path.exists(MyCKPT):
        print("-----Loading Checkpoint-----")

        checkpoint = torch.load(MyCKPT)
        save_epoch = checkpoint['epoch']
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

    if os.path.exists(MyCKPT):
        print("!!")
        start_epoch = save_epoch+1
        print("start_epoch: ", start_epoch)
    else:
        start_epoch = 0

    #for epoch in tqdm(range(num_epochs)):
    for epoch in tqdm(range(start_epoch, num_epochs, 1)):
        print("\n===epoch=== ", epoch)
        running_loss = 0

        running_sep_loss = 0

        running_cost_loss = 0
        running_kp_match_loss = 0
        running_desc_match_loss = 0

        running_cosim_loss = 0

        running_recon_loss_l2 = 0
        running_recon_loss_l1 = 0
        running_recon_loss_ssim = 0

        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)

            if(epoch < 5):
                theta = 0
            elif (epoch >= 5) and (epoch < 15):
                theta = 0.5
            elif (epoch >= 15) and (epoch < 30):
                theta = -0.5
            elif (epoch >= 30) and (epoch < 50):
                theta = 1.0
            elif (epoch >= 50) and (epoch < 70):
                theta = -1.0
            elif (epoch >= 70) and (epoch < 90):
                theta = 1.5
            elif (epoch >= 90) and (epoch < 110):
                theta = -1.5
            elif (epoch >= 110) and (epoch < 130):
                theta = 2.0
            elif (epoch >= 130) and (epoch < 150):
                theta = -2.0
            elif (epoch >= 130) and (epoch < 160):
                theta = 2.5
            elif (epoch >= 160) and (epoch < 200):
                theta = -2.5
            elif (epoch >= 200) and (epoch < 300):
                theta = 3.0
            elif (epoch >= 300) and (epoch < 400):
                theta = -3.0
            elif (epoch >= 400):
                theta = random.uniform(-3, 3)
            '''
            elif (epoch >= 5) and (epoch < 250):
                theta = random.uniform(-1, 1)
            elif (epoch >= 250) and (epoch < 500):
                theta = random.uniform(-3, 3)
            '''

            '''
            elif(epoch >= 5) and(epoch < 50):
                theta = random.uniform(-1, 1)
            elif(epoch >= 50) and (epoch < 100):
                theta = random.uniform(-3, 3)
            elif(epoch >= 100) and (epoch < 200):
                theta = random.uniform(-5, 5)
            elif(epoch >= 200) and (epoch < 300):
                theta = random.uniform(-10, 10)
            '''

            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input)  # randomly rotated image

            ##########################################ENCODER##########################################
            cur_batch = aefe_input.shape[0]

            Rk_origin = model_StackedHourglassForKP(aefe_input)
            tf_Rk_origin = model_StackedHourglassForKP(tf_aefe_input)

            pool4 = torch.nn.AvgPool2d(2, stride=4)
            pool_aefe = pool4(aefe_input)
            pool_tf_aefe = pool4(tf_aefe_input)

            Rk = (pool_aefe.unsqueeze(1) * Rk_origin.unsqueeze(2)).mean(dim=2)
            tf_Rk = (pool_tf_aefe.unsqueeze(1) * tf_Rk_origin.unsqueeze(2)).mean(dim=2)

            '''
            position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
            mask = generate_square_subsequent_mask(Rk.shape[2], Rk.shape[3]).cuda()
            mask = mask.repeat(cur_batch, 1, 1)
            pos = position_embedding(aefe_input, mask)
            tf_pos = position_embedding(tf_aefe_input, mask)
            '''
            #kp, desc = model_DETR_kp(Rk, pos)
            #tf_kp, tf_desc = model_DETR_kp(tf_Rk, tf_pos)
            kp, desc = model_DETR_kp(Rk)
            tf_kp, tf_desc = model_DETR_kp(tf_Rk)

            ##For binary descriptors##
            desc = torch.sign(desc)
            tf_desc = torch.sign(tf_desc)
            MySTE = StraightThroughEstimator()
            desc = MySTE(desc)
            tf_desc = MySTE(tf_desc)

            ##########################################DECODER##########################################
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

            reconInput_1 = (recon_kp_1.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk_origin
            reconInput_2 = (recon_kp_2.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk_origin
            reconInput_3 = (recon_kp_3.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk_origin
            reconInput_4 = (recon_kp_4.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk_origin
            reconInput_5 = (recon_kp_5.unsqueeze(2) * desc.unsqueeze(3).unsqueeze(4)).mean(1) * Rk_origin

            reconInput = torch.cat([reconInput_1, reconInput_2, reconInput_3, reconInput_4, reconInput_5], dim=1)
            reconImg = model_StackedHourglassImgRecon(reconInput)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            #tf_reconImg = model_StackedHourglassImgRecon(tf_reconInput)
            #tf_reconImg = tf_reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            ##############################################LOSS#############################################
            # Define Loss Functions!

            # separation loss
            fn_loss_separation = loss_separation().cuda()
            cur_sep_loss = fn_loss_separation(kp) + fn_loss_separation(tf_kp)

            # similarity loss btw Dk and tf_Dk
            fn_loss_cosim = loss_cosim().cuda()
            cur_cosim_loss = fn_loss_cosim(Rk_origin, tf_Rk_origin)

            fn_hungarian_matcher = HungarianMatcherNLoss()
            loss_kp_match, loss_desc, loss_cost = fn_hungarian_matcher(-theta, my_feature, my_tf_feature, my_width, my_height)

            # NCC loss
            #fn_ncc_loss = NCC_loss()
            #loss_ncc = fn_ncc_loss(aefe_input, tf_aefe_input, match, kp, tf_kp)

            # Reconstruction Loss
            criterion = SSIM()
            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            cur_recon_loss_l1 = F.l1_loss(reconImg, aefe_input)
            cur_recon_loss_ssim = (1 - criterion(reconImg, aefe_input))

            p_sep_loss = 0.5
            p_kp_loss = 1.0 #1.0
            p_desc_loss = 3 * 1e-2
            p_cost_loss = 1e-4
            p_cosim_loss = 10.0
            p_recon_img_l2 = 2.0
            p_recon_img_l1 = 2.0
            p_recon_img_ssim = 1.0

            my_sep_loss = p_sep_loss * cur_sep_loss
            my_cost_loss = p_cost_loss * loss_cost
            my_kp_loss = p_kp_loss * loss_kp_match
            my_desc_loss = p_desc_loss * loss_desc
            my_cosim_loss = p_cosim_loss * cur_cosim_loss
            my_recon_loss_l2 = p_recon_img_l2 * cur_recon_loss_l2
            my_recon_loss_l1 = p_recon_img_l1 * cur_recon_loss_l1
            my_recon_loss_ssim = p_recon_img_ssim * cur_recon_loss_ssim

            #loss = (my_sep_loss + my_fundamental_loss + my_kp_loss + my_desc_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            loss = my_sep_loss + my_kp_loss + my_desc_loss + my_cost_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim
            '''
            if(epoch < 5):
                print("No Matching Loss")
                loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
                #loss = (my_sep_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            else:
                print("Computing Matching Loss")
                loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
                #loss = (my_sep_loss + my_kp_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
                #loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            '''
            #loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_kp_loss + my_desc_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)
            #loss = (my_sep_loss + my_cosim_loss + my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)

            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", Fundamental: ", '%.4f' % my_fundamental_loss.item(), ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
            #      ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", cost_loss: ", '%.4f' % my_cost_loss.item(),  ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", KP_matching:", '%.4f' % my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
            #print("Sep Loss: ", '%.4f' % my_sep_loss.item(), ", KP_matching:", my_kp_loss.item(), ", Desc_matching:", '%.4f' % my_desc_loss.item(), ", Cosim:", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(),
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
            running_cost_loss = running_cost_loss + my_cost_loss.item()
            running_kp_match_loss = running_kp_match_loss + my_kp_loss.item()
            running_desc_match_loss = running_desc_match_loss + my_desc_loss.item()
            running_cosim_loss = running_cosim_loss + my_cosim_loss.item()
            running_recon_loss_l2 = running_recon_loss_l2 + my_recon_loss_l2.item()
            running_recon_loss_l1 = running_recon_loss_l1 + my_recon_loss_l1.item()
            running_recon_loss_ssim = running_recon_loss_ssim + my_recon_loss_ssim.item()

            if (((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, kp, epoch + 1, cur_filename)
                fn_save_tfkpimg = savetfKPimg()
                fn_save_tfkpimg(tf_aefe_input, tf_kp, epoch + 1, cur_filename)
                img_save_filename = ("/home/jsk/AEFE_SLAM/SaveReconstructedImg/%s_ep_%s.jpg" % (cur_filename, epoch + 1))
                #tf_img_save_filename = ("/home/jsk/AEFE_SLAM/SaveTFReconstructedImg/%s_ep_%s.jpg" % (cur_filename, epoch + 1))
                save_image(reconImg, img_save_filename)
                #save_image(tf_reconImg, tf_img_save_filename)

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

        vis.line(Y=[running_cost_loss], X=np.array([epoch]), win=plot_cost_loss, update='append')
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

    print("!!210707!!")
    print("!!!!!This is train_allnew3 with ResNet Module.py!!!!!")
    train()

##########################################################################################################################