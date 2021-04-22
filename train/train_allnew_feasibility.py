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
# plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))
# plot_conc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration loss'))
# plot_trans_kp = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation loss (KP)'))
# plot_trans_desc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Transformation loss (Desc)'))
# plot_cosim = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Cosim loss'))
plot_recon_L2 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction L2 Loss'))
plot_recon_L1 = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction L1 Loss'))
plot_recon_SSIM = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction SSIM Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)

#########################################parameter#########################################
num_of_kp = 200
voters = 200
num_queries = voters

hidden_dim = 256
feature_dimension = 256  # 32

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 300
batch_size = 2  # 8 #4

stacked_hourglass_inpdim_kp = input_width
# stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 1

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
    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=128, oup_dim=128, bn=False, increase=0).cuda()
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # model_DETR_kp = DETR_1E1D(num_voters=voters, hidden_dim=256, nheads=1, num_encoder_layers=2, num_decoder_layers=2).cuda()
    model_DETR_kp = DETR_1E1D(num_voters=voters, hidden_dim=128, nheads=1, num_encoder_layers=2, num_decoder_layers=2).cuda()
    model_DETR_kp = nn.DataParallel(model_DETR_kp).cuda()
    optimizer_DETR_kp = torch.optim.AdamW(model_DETR_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=128, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ###################################################################################################################
    # call checkpoint
    '''
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
    '''
    ###################################################################################################################

    dataset = my_dataset(my_width=my_width, my_height=my_height)
    # dataset = my_dataset(my_width=640, my_height=192)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        #running_sep_loss = 0
        running_recon_loss_l2 = 0
        running_recon_loss_l1 = 0
        running_recon_loss_ssim = 0
        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)

            ##########################################ENCODER##########################################
            Rk = model_StackedHourglassForKP(aefe_input)[:, num_nstack - 1, :, :, :]
            Rk_flatten = Rk.flatten(2)
            kp = model_DETR_kp(Rk_flatten, my_height, my_width)

            ##########################################DECODER##########################################
            fn_ReconKp = ReconWithKP(my_height, my_width)
            # recon_kp_1 = fn_ReconKp(kp, 0.1)  # (b,200,48,160)
            # recon_kp_1 = fn_ReconKp(kp, 0.2)  # (b,200,48,160)
            # recon_kp_2 = fn_ReconKp(kp, 0.5)  # (b,200,48,160)
            recon_kp_3 = fn_ReconKp(kp, 1.0)  # (b,200,48,160)
            # recon_kp_3 = fn_ReconKp(kp, 3.0)  # (b,200,48,160)
            # recon_kp_4 = fn_ReconKp(kp, 5.0)  # (b,200,48,160)
            # recon_kp_5 = fn_ReconKp(kp, 10.0)  # (b,200,48,160)
            # recon_kp_5 = fn_ReconKp(kp, 20.0)  # (b,200,48,160)

            # n_recon_kp_1 = recon_kp_1.unsqueeze(1)
            # n_recon_kp_2 = recon_kp_2.unsqueeze(1)
            n_recon_kp_3 = recon_kp_3.unsqueeze(1)
            # n_recon_kp_4 = recon_kp_4.unsqueeze(1)
            # n_recon_kp_5 = recon_kp_5.unsqueeze(1)

            n_Rk = Rk.unsqueeze(2)

            # RECON_1 = n_recon_kp_1 * n_Rk
            # RECON_2 = n_recon_kp_2 * n_Rk
            RECON_3 = n_recon_kp_3 * n_Rk
            m_RECON_3 = torch.mean(RECON_3, dim=2)
            # RECON_4 = n_recon_kp_4 * n_Rk
            # RECON_5 = n_recon_kp_5 * n_Rk

            # recon_input = torch.cat(
            #    [recon_kp_1, Recon_feature_1, recon_kp_2, Recon_feature_2, recon_kp_3, Recon_feature_3, recon_kp_4,
            #     Recon_feature_4, recon_kp_5, Recon_feature_5], dim=1)

            # recon_input = torch.cat([RECON_1, RECON_2, RECON_3, RECON_4, RECON_5], dim=1)
            reconImg = model_StackedHourglassImgRecon(m_RECON_3)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            ##############################################LOSS#############################################
            # Define Loss Functions!
            cur_recon_loss_l2 = F.mse_loss(reconImg, aefe_input)
            cur_recon_loss_l1 = F.l1_loss(reconImg, aefe_input)
            criterion = SSIM()
            cur_recon_loss_ssim = 1 - criterion(reconImg, aefe_input)
            cur_recon_loss_l2 = cur_recon_loss_l2
            cur_recon_loss_ssim = cur_recon_loss_ssim

            p_recon_img_l2 = 2.0
            p_recon_img_l1 = 2.0
            p_recon_img_ssim = 1.0

            my_recon_loss_l2 = p_recon_img_l2 * cur_recon_loss_l2
            my_recon_loss_l1 = p_recon_img_l1 * cur_recon_loss_l1
            my_recon_loss_ssim = p_recon_img_ssim * cur_recon_loss_ssim

            loss = (my_recon_loss_l1 + my_recon_loss_l2 + my_recon_loss_ssim)

            print("Recon_L2:", '%.4f' % my_recon_loss_l2.item(), ", Recon_SSIM:", '%.4f' % my_recon_loss_ssim.item(),
                  ", Recon_L1:", '%.4f' % my_recon_loss_l1.item())

            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_DETR_kp.zero_grad()
            optimizer_ImgRecon.zero_grad()

            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_DETR_kp.step()
            optimizer_ImgRecon.step()

            running_loss = running_loss + loss.item()
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
        '''
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
        '''
        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')
        vis.line(Y=[running_recon_loss_l2], X=np.array([epoch]), win=plot_recon_L2, update='append')
        vis.line(Y=[running_recon_loss_l1], X=np.array([epoch]), win=plot_recon_L1, update='append')
        vis.line(Y=[running_recon_loss_ssim], X=np.array([epoch]), win=plot_recon_SSIM, update='append')


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

    print("!!!!!This is train_allnew.py!!!!!")
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################