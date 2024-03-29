##Train DETR4 has 1 encoder for keypoints and descriptors, and 2 different decoders which shares target(=voters)

from model.StackedHourglass import *
from loss import *
from utils import *
from GenDescriptorMap import *
import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
#from IQA_pytorch import SSIM
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
plot_conc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration loss'))
plot_trans_kp = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation loss (KP)'))
plot_trans_desc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation loss (Desc)'))
plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosim loss'))
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

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 300
batch_size = 4  # 8 #4

stacked_hourglass_inpdim_kp = input_width
#stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

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

    #model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0).cuda()
    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=256, oup_dim=num_of_kp, bn=False, increase=0).cuda()
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    # optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=1e-3, weight_decay=5e-4)
    #optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=1e-3, weight_decay=2e-4)
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_ENC_inp = ENCinp_1(my_height=my_height, my_width=my_width, dim1=2048, dim2=256).cuda()
    model_ENC_inp = nn.DataParallel(model_ENC_inp).cuda()
    #optimizer_ENC_inp = torch.optim.AdamW(model_ENC_inp.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_ENC_inp = torch.optim.AdamW(model_ENC_inp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_DETR_kp_f = DETR_1E2D(num_voters=voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR_kp_f = nn.DataParallel(model_DETR_kp_f).cuda()
    #optimizer_DETR_kp_f = torch.optim.AdamW(model_DETR_kp_f.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_DETR_kp_f = torch.optim.AdamW(model_DETR_kp_f.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_recon_s = linear_s(inp_dim=feature_dimension, my_height=my_height, my_width=my_width)
    model_recon_s = nn.DataParallel(model_recon_s).cuda()
    #optimizer_recon_s = torch.optim.AdamW(model_recon_s.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizer_recon_s = torch.optim.AdamW(model_recon_s.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    #optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=1e-3, weight_decay=2e-4)
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_StackedHourglass_kp, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_ENC_inp, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_DETR_kp_f, lr_drop)
    lr_scheduler_optimizer4 = torch.optim.lr_scheduler.StepLR(optimizer_recon_s, lr_drop)
    lr_scheduler_optimizer5 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)

    ###################################################################################################################
    # call checkpoint

    if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth"):
    #if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/210421_test.pth"):
        # if os.path.exists("./SaveModelCKPT/210401.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth")
        #checkpoint = torch.load("/home/jsk/AEFE_SLAM/SaveModelCKPT/210421_test.pth")

        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_ENC_inp.module.load_state_dict(checkpoint['model_ENC_inp'])
        model_DETR_kp_f.module.load_state_dict(checkpoint['model_DETR_kp_f'])
        model_recon_s.module.load_state_dict(checkpoint['model_recon_s'])
        model_StackedHourglassImgRecon.module.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_ENC_inp.load_state_dict(checkpoint['optimizer_ENC_inp'])
        optimizer_DETR_kp_f.load_state_dict(checkpoint['optimizer_DETR_kp_f'])
        optimizer_recon_s.load_state_dict(checkpoint['optimizer_recon_s'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])

        lr_scheduler_optimizer1.load_state_dict(checkpoint['lr_scheduler_optimizer1'])
        lr_scheduler_optimizer2.load_state_dict(checkpoint['lr_scheduler_optimizer2'])
        lr_scheduler_optimizer3.load_state_dict(checkpoint['lr_scheduler_optimizer3'])
        lr_scheduler_optimizer4.load_state_dict(checkpoint['lr_scheduler_optimizer4'])
        lr_scheduler_optimizer5.load_state_dict(checkpoint['lr_scheduler_optimizer5'])

    ###################################################################################################################

    dataset = my_dataset(my_width=my_width, my_height=my_height)
    #dataset = my_dataset(my_width=640, my_height=192)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        running_sep_loss = 0
        running_conc_loss = 0
        running_homo_kp_loss = 0
        running_homo_desc_loss = 0
        running_cosim_loss = 0
        #running_recon_loss = 0
        running_recon_loss_l2 = 0
        running_recon_loss_l1 = 0
        running_recon_loss_ssim = 0
        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)
            cur_batch = aefe_input.shape[0]

            theta = random.uniform(-5, 5)
            my_transform = torchvision.transforms.RandomAffine((theta, theta), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
            tf_aefe_input = my_transform(aefe_input) #randomly rotated image
            ##########################################ENCODER##########################################
            Rk = model_StackedHourglassForKP(aefe_input)[:, num_nstack - 1, :, :, :]
            Rk_tf = model_StackedHourglassForKP(tf_aefe_input)[:, num_nstack - 1, :, :, :]
            #Rk = model_StackedHourglassForKP(aefe_input).sum(dim=1)
            Dk = torch.softmax(Rk, dim=1)  # (b,k=200, 48,160)
            Dk_tf = torch.softmax(Rk_tf, dim=1)  # (b,k=200, 48,160)

            Sk = aefe_input.sum(dim=1).unsqueeze(1) * Dk
            tf_sk = tf_aefe_input.sum(dim=1).unsqueeze(1) * Dk_tf

            #positional_encoding = PositionEmbeddingSine()
            #mask, pos = positional_encoding(Dk, cur_batch, my_height, my_width)

            #linearized_Dk, pe_Dk = model_ENC_inp(Dk, pos)  # (b,256,200)
            Sk_flatten, transformer_input = model_ENC_inp(Sk)  # (b,256,200)
            tf_sk_flatten, tf_transformer_input = model_ENC_inp(tf_sk)  # (b,256,200)

            kp, desc = model_DETR_kp_f(transformer_input, my_height, my_width)
            kp_tf, desc_tf = model_DETR_kp_f(tf_transformer_input, my_height, my_width)

            ##########################################DECODER##########################################
            fn_ReconKp = ReconWithKP(my_height, my_width)
            #recon_kp_1 = fn_ReconKp(kp, 0.1)  # (b,200,48,160)
            recon_kp_1 = fn_ReconKp(kp, 0.2)  # (b,200,48,160)
            recon_kp_2 = fn_ReconKp(kp, 0.5)  # (b,200,48,160)
            recon_kp_3 = fn_ReconKp(kp, 1.0)  # (b,200,48,160)
            #recon_kp_3 = fn_ReconKp(kp, 3.0)  # (b,200,48,160)
            recon_kp_4 = fn_ReconKp(kp, 5.0)  # (b,200,48,160)
            recon_kp_5 = fn_ReconKp(kp, 10.0)  # (b,200,48,160)
            #recon_kp_5 = fn_ReconKp(kp, 20.0)  # (b,200,48,160)

            # kpNdesc = torch.cat([kp, desc], dim=2)
            Recon_Desc = model_recon_s(desc)

            #Recon_feature_1 = F.elu(Recon_Desc * recon_kp_1)
            #Recon_feature_2 = F.elu(Recon_Desc * recon_kp_2)
            #Recon_feature_3 = F.elu(Recon_Desc * recon_kp_3)
            #Recon_feature_4 = F.elu(Recon_Desc * recon_kp_4)
            #Recon_feature_5 = F.elu(Recon_Desc * recon_kp_5)
            #Recon_feature_1 = F.relu(Recon_Desc * recon_kp_1)
            #Recon_feature_2 = F.relu(Recon_Desc * recon_kp_2)
            #Recon_feature_3 = F.relu(Recon_Desc * recon_kp_3)
            #Recon_feature_4 = F.relu(Recon_Desc * recon_kp_4)
            #Recon_feature_5 = F.relu(Recon_Desc * recon_kp_5)
            Recon_feature_1 = (Recon_Desc * recon_kp_1)
            Recon_feature_2 = (Recon_Desc * recon_kp_2)
            Recon_feature_3 = (Recon_Desc * recon_kp_3)
            Recon_feature_4 = (Recon_Desc * recon_kp_4)
            Recon_feature_5 = (Recon_Desc * recon_kp_5)

            recon_input = torch.cat(
                [recon_kp_1, Recon_feature_1, recon_kp_2, Recon_feature_2, recon_kp_3, Recon_feature_3, recon_kp_4,
                 Recon_feature_4, recon_kp_5, Recon_feature_5], dim=1)
            reconImg = model_StackedHourglassImgRecon(recon_input)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            ##############################################LOSS#############################################
            # Define Loss Functions!
            # concnetration_loss
            fn_loss_conc = loss_concentration(Rk)
            fn_loss_conc_tf = loss_concentration(Rk_tf)
            cur_conc_loss = fn_loss_conc() + fn_loss_conc_tf()

            # separation loss
            fn_loss_separation = loss_separation(kp).cuda()
            fn_loss_separation_tf = loss_separation(kp_tf).cuda()
            cur_sep_loss = fn_loss_separation() + fn_loss_separation_tf()

            # transformation loss
            my_feature = torch.cat([kp, desc], dim=2)
            my_tf_feature = torch.cat([kp_tf, desc_tf], dim=2)
            fn_loss_homo = loss_matching(theta, my_feature, my_tf_feature, cur_batch, num_of_kp, my_width, my_height)
            cur_homo_kp_loss, cur_homo_desc_loss = fn_loss_homo()

            # cosim loss
            fn_loss_cosim = loss_cosim(Rk, Rk_tf).cuda()
            cur_cosim_loss = fn_loss_cosim()

            # Recon img loss
            #std_color = 0.05
            #n_reconImg = torch.sigmoid(reconImg)
            #n_inputImg = torch.sigmoid(aefe_input)
            #cur_recon_loss_l2 = F.mse_loss(n_reconImg, n_inputImg)
            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            cur_recon_loss_l1 = F.l1_loss(reconImg, aefe_input)
            criterion = SSIM()
            #cur_recon_loss_ssim = criterion(n_reconImg, n_inputImg)
            cur_recon_loss_ssim = 1 - criterion(reconImg, aefe_input)
            #tmp_recon_loss = (cur_recon_loss_l2 * 5 + cur_recon_loss_ssim)
            cur_recon_loss_l2 = cur_recon_loss_l2
            cur_recon_loss_ssim = cur_recon_loss_ssim
            #cur_recon_loss = cur_recon_loss_l2 + cur_recon_loss_ssim
            #cur_recon_loss = (1 / (std_color ** 2)) * tmp_recon_loss + math.log(2 * math.pi * std_color * std_color)

            p_sep = 0.5
            p_conc = 0.25
            p_homo_kp = 0.02
            p_homo_desc = 2.0
            p_cosim = 1.0
            p_recon_img_l2 = 5.0
            p_recon_img_l1 = 4.0
            p_recon_img_ssim = 1.0

            my_sep_loss = p_sep * cur_sep_loss
            my_conc_loss = p_conc * cur_conc_loss
            my_homo_kp_loss = p_homo_kp * cur_homo_kp_loss
            my_homo_desc_loss = p_homo_desc * cur_homo_desc_loss
            my_cosim_loss = p_cosim * cur_cosim_loss
            my_recon_loss_l2 = p_recon_img_l2 * cur_recon_loss_l2
            my_recon_loss_l1 = p_recon_img_l1 * cur_recon_loss_l1
            my_recon_loss_ssim = p_recon_img_ssim * cur_recon_loss_ssim


            # loss = my_transf_loss + my_matching_loss + my_cosim_loss + my_sep_loss + my_vk_loss + my_wk_loss + my_recon_loss
            # loss = my_sep_loss + my_recon_loss + my_recon_kp_loss + my_recon_f_loss
            loss = (my_sep_loss + my_conc_loss + my_homo_kp_loss + my_homo_desc_loss + my_cosim_loss + my_recon_loss_l2 + my_recon_loss_ssim)

            # print("Trans: ", my_transf_loss.item(), ", Matching: ", my_matching_loss.item(), ", Cosim: ", my_cosim_loss.item(),  ", Sep: ", my_sep_loss.item(), ", Vk: ", my_vk_loss.item(),  ", Wk: ", my_wk_loss.item(), ", Recon:", my_recon_loss.item())
            # print("Sep: ", my_sep_loss.item(), ", recon_KP: ", my_recon_kp_loss.item(),  ", recon_f: ", my_recon_f_loss.item(), ", Recon:", my_recon_loss.item())
            # print("Sep: ", my_sep_loss.item(),  ", recon_s: ", my_recon_s_loss.item(), ", Recon:", my_recon_loss.item())
            print("Sep: ", '%.4f' % my_sep_loss.item(), ", Conc : ", '%.4f' % my_conc_loss.item(), ", homo_kp : ", '%.4f' % my_homo_kp_loss.item(), ", homo_desc : ", '%.4f' % my_homo_desc_loss.item(), ", Cosim : ", '%.4f' % my_cosim_loss.item(), ", Recon_L2:", '%.4f' % my_recon_loss_l2.item(), ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(), ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_ENC_inp.zero_grad()
            optimizer_DETR_kp_f.zero_grad()
            optimizer_recon_s.zero_grad()
            optimizer_ImgRecon.zero_grad()

            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_ENC_inp.step()
            optimizer_DETR_kp_f.step()
            optimizer_recon_s.step()
            optimizer_ImgRecon.step()

            lr_scheduler_optimizer1.step()
            lr_scheduler_optimizer2.step()
            lr_scheduler_optimizer3.step()
            lr_scheduler_optimizer4.step()
            lr_scheduler_optimizer5.step()

            running_loss = running_loss + loss.item()
            running_sep_loss = running_sep_loss + my_sep_loss.item()
            running_conc_loss = running_conc_loss + my_conc_loss.item()
            running_homo_kp_loss = running_homo_kp_loss + my_homo_kp_loss.item()
            running_homo_desc_loss = running_homo_desc_loss + my_homo_desc_loss.item()
            running_cosim_loss = running_cosim_loss + my_cosim_loss.item()
            #running_recon_loss = running_recon_loss + my_recon_loss.item()
            running_recon_loss_l2 = running_recon_loss_l2 + my_recon_loss_l2.item()
            running_recon_loss_l1 = running_recon_loss_l1 + my_recon_loss_l1.item()
            running_recon_loss_ssim = running_recon_loss_ssim + my_recon_loss_ssim.item()

            if (((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                # if (((epoch + 1) % 2 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                # print("epoch: ", epoch)
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, kp, epoch + 1, cur_filename)
                # fn_save_tfkpimg = savetfKPimg()
                # fn_save_tfkpimg(tf_aefe_input, tf_kp, epoch + 1, cur_filename)
                img_save_filename = ("/home/jsk/AEFE_SLAM/SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (
                cur_filename, epoch + 1))
                save_image(reconImg, img_save_filename)

        # if (epoch != 0) and ((epoch+1) % 5 == 0):
        torch.save({
            'model_StackedHourglassForKP': model_StackedHourglassForKP.module.state_dict(),
            'model_ENC_inp': model_ENC_inp.module.state_dict(),
            'model_DETR_kp_f': model_DETR_kp_f.module.state_dict(),
            'model_recon_s': model_recon_s.module.state_dict(),
            'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.module.state_dict(),

            'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
            'optimizer_ENC_inp': optimizer_ENC_inp.state_dict(),
            'optimizer_DETR_kp_f': optimizer_DETR_kp_f.state_dict(),
            'optimizer_recon_s': optimizer_recon_s.state_dict(),
            'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),

            'lr_scheduler_optimizer1': lr_scheduler_optimizer1.state_dict(),
            'lr_scheduler_optimizer2': lr_scheduler_optimizer2.state_dict(),
            'lr_scheduler_optimizer3': lr_scheduler_optimizer3.state_dict(),
            'lr_scheduler_optimizer4': lr_scheduler_optimizer4.state_dict(),
            'lr_scheduler_optimizer5': lr_scheduler_optimizer5.state_dict(),

        }, "/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')
        vis.line(Y=[running_sep_loss], X=np.array([epoch]), win=plot_sep, update='append')
        vis.line(Y=[running_conc_loss], X=np.array([epoch]), win=plot_conc, update='append')
        vis.line(Y=[running_homo_kp_loss], X=np.array([epoch]), win=plot_trans_kp, update='append')
        vis.line(Y=[running_homo_desc_loss], X=np.array([epoch]), win=plot_trans_desc, update='append')
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
    if not os.path.exists("SaveHeatMapImg"):
        os.makedirs("SaveHeatMapImg")
    if not os.path.exists("SaveModelCKPT"):
        os.makedirs("SaveModelCKPT")

    print("!!!!!This is train_DETR4.py!!!!!")
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################