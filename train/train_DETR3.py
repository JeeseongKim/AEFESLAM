##Train DETR3 has 2 encoders for keypoints and descriptors, and 1 decoder which shares target(=voters)

from model.StackedHourglass import *
from loss import *
from utils import *
from GenDescriptorMap import *
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader
from misc import *
#from position_encoding import *
from model.MyDETR import *
from model.DetectionConfidenceMap import *

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom
vis = visdom.Visdom()

plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))
plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))
plot_conc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration loss'))
plot_recon = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction Loss'))

torch.multiprocessing.set_start_method('spawn', force=True)

#########################################parameter#########################################
num_of_kp = 200
voters = num_of_kp
num_queries = voters

hidden_dim = 256
feature_dimension = 256 #32

my_width = 160  # 272 #96 #272 #208
my_height = 48  # 80 #32 #80 #64

input_width = my_width

num_epochs = 300
batch_size = 4 #8 #4

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

num_nstack = 8

learning_rate = 1e-4  # 1e-3#1e-4 #1e-3
weight_decay = 1e-5  # 1e-2#1e-5 #1e-5 #5e-4
lr_drop = 200
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
def train():
    model_start = time.time()

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0).cuda()
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP).cuda()
    #optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizer_StackedHourglass_kp = torch.optim.AdamW(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_ENC_inp = ENCinp(my_height=my_height, my_width=my_width, dim1=2048, dim2=256).cuda()
    model_ENC_inp = nn.DataParallel(model_ENC_inp).cuda()
    optimizer_ENC_inp = torch.optim.AdamW(model_ENC_inp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_DETR_kp_f = DETR4kpNf(num_voters=voters, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR_kp_f = nn.DataParallel(model_DETR_kp_f).cuda()
    optimizer_DETR_kp_f = torch.optim.AdamW(model_DETR_kp_f.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_recon_s = linear_s(inp_dim=feature_dimension, my_height=my_height, my_width=my_width)
    model_recon_s = nn.DataParallel(model_recon_s).cuda()
    #optimizer_recon_s = torch.optim.AdamW(model_recon_s.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizer_recon_s = torch.optim.AdamW(model_recon_s.parameters(),lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon_DETR(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon).cuda()
    #optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizer_ImgRecon = torch.optim.AdamW(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)


    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_StackedHourglass_kp, lr_drop)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_ENC_inp, lr_drop)
    lr_scheduler_optimizer3 = torch.optim.lr_scheduler.StepLR(optimizer_DETR_kp_f, lr_drop)
    lr_scheduler_optimizer4 = torch.optim.lr_scheduler.StepLR(optimizer_recon_s, lr_drop)
    lr_scheduler_optimizer5 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)

    '''
    model_DETR_f = DETR4f(num_voters=voters, hidden_dim=hidden_dim, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR_f = nn.DataParallel(model_DETR_f).cuda()
    optimizer_DETR_f = torch.optim.AdamW(model_DETR_f.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_Attention_f = AttentionMap_f().cuda()
    model_Attention_f = nn.DataParallel(model_Attention_f).cuda()
    optimizer_Attentionf = torch.optim.AdamW(model_Attention_f.parameters(), lr=learning_rate,weight_decay=weight_decay)

    model_DETR_kp = DETR4kp(num_voters=voters, hidden_dim=200, nheads=8, num_encoder_layers=6, num_decoder_layers=6).cuda()
    model_DETR_kp = nn.DataParallel(model_DETR_kp).cuda()
    optimizer_DETR_kp = torch.optim.AdamW(model_DETR_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)


    #model_recon_kp = linear_kp(inp_dim=2, img_width=my_width, img_height=my_height)
    #model_recon_kp = nn.DataParallel(model_recon_kp).cuda()
    #optimizer_recon_kp = torch.optim.AdamW(model_recon_kp.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #model_recon_f = linear_f(inp_dim=feature_dimension, REimg_height=12, REimg_width=40)
    #model_recon_f = nn.DataParallel(model_recon_f).cuda()
    #optimizer_recon_f = torch.optim.AdamW(model_recon_f.parameters(), lr=learning_rate, weight_decay=weight_decay)
    '''

    '''
    
    lr_scheduler_optimizer6 = torch.optim.lr_scheduler.StepLR(optimizer_recon_f, lr_drop)
    lr_scheduler_optimizer7 = torch.optim.lr_scheduler.StepLR(optimizer_ImgRecon, lr_drop)
    '''
    ###################################################################################################################
    #call checkpoint

    if os.path.exists("/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth"):
    #if os.path.exists("./SaveModelCKPT/210401.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("/home/jsk/AEFE_SLAM/SaveModelCKPT/train_model.pth")

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
    #dataset = my_dataset(my_width=1226, my_height=370)
    #dataset = my_dataset_originalImg(my_width=my_width, my_height=my_height)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("***", time.time() - model_start)  # 7.26 ~ 63

    saveLossTxt = open("SaveLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        running_sep_loss = 0
        running_conc_loss = 0
        running_recon_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            input_img, cur_filename, kp_img = data
            aefe_input = input_img.cuda()  # (b, 3, height, width)
            cur_batch = aefe_input.shape[0]

            ##########################################ENCODER##########################################
            #Rk = model_StackedHourglassForKP(aefe_input)[:, num_nstack - 1, :, :, :]
            Rk = model_StackedHourglassForKP(aefe_input).sum(dim=1)
            #Maxpool4 = torch.nn.MaxPool2d(4, 4)
            #Rk_pooled = Maxpool4(Rk) #(b,k,12,40)

            # positional encoding
            #positional_encoding = PositionEmbeddingSine()
            #_, pos_Rk = positional_encoding(Rk, Rk.shape[0], Rk.shape[2], Rk.shape[3])

            #transformer_inp = Rk + pos_Rk #(b,k=200, 48,160)
            transformer_inp = torch.softmax(Rk, dim=1) #(b,k=200, 48,160)
            Enc_kp, Enc_f = model_ENC_inp(transformer_inp) #(b,256,200)

            kp, desc = model_DETR_kp_f(Enc_kp, Enc_f, my_height, my_width)

            ##########################################DECODER##########################################
            fn_ReconKp = ReconWithKP(my_height, my_width)
            recon_kp_1 = fn_ReconKp(kp, 0.2) #(b,200,48,160)
            recon_kp_2 = fn_ReconKp(kp, 0.4) #(b,200,48,160)
            recon_kp_3 = fn_ReconKp(kp, 0.6) #(b,200,48,160)
            recon_kp_4 = fn_ReconKp(kp, 0.8) #(b,200,48,160)
            recon_kp_5 = fn_ReconKp(kp, 1.0) #(b,200,48,160)

            #kpNdesc = torch.cat([kp, desc], dim=2)
            Recon_Desc = model_recon_s(desc)

            Recon_feature_1 = F.elu(Recon_Desc * recon_kp_1)
            Recon_feature_2 = F.elu(Recon_Desc * recon_kp_2)
            Recon_feature_3 = F.elu(Recon_Desc * recon_kp_3)
            Recon_feature_4 = F.elu(Recon_Desc * recon_kp_4)
            Recon_feature_5 = F.elu(Recon_Desc * recon_kp_5)

            recon_input = torch.cat([recon_kp_1, Recon_feature_1 , recon_kp_2, Recon_feature_2, recon_kp_3, Recon_feature_3, recon_kp_4, Recon_feature_4, recon_kp_5, Recon_feature_5], dim=1)
            reconImg = model_StackedHourglassImgRecon(recon_input)
            reconImg = reconImg[:, num_nstack - 1, :, :, :]  # (b,3,192,256)

            # Define Loss Functions!
            #concnetration_loss
            fn_loss_conc = loss_concentration(Rk)
            cur_conc_loss = fn_loss_conc()

            #separation loss
            fn_loss_separation = loss_separation(kp).cuda()
            cur_sep_loss = fn_loss_separation()

            #Recon img loss
            std_color = 0.05
            n_reconImg = torch.sigmoid(reconImg)
            n_inputImg = torch.sigmoid(aefe_input)

            cur_recon_loss_l2 = F.mse_loss(n_reconImg, n_inputImg)
            criterion = SSIM()

            cur_recon_loss_ssim = criterion(n_reconImg, n_inputImg)
            tmp_recon_loss = (cur_recon_loss_l2*35 + cur_recon_loss_ssim)

            cur_recon_loss = (1/(std_color ** 2))*tmp_recon_loss + math.log(2 * math.pi * std_color * std_color)

            p_sep = 1.0
            p_recon_conc = 1.0
            p_recon_img = 0.003


            my_sep_loss = p_sep * cur_sep_loss
            my_conc_loss = p_recon_conc * cur_conc_loss
            my_recon_loss = p_recon_img * cur_recon_loss

            #loss = my_transf_loss + my_matching_loss + my_cosim_loss + my_sep_loss + my_vk_loss + my_wk_loss + my_recon_loss
            #loss = my_sep_loss + my_recon_loss + my_recon_kp_loss + my_recon_f_loss
            loss = my_sep_loss + my_recon_loss + my_conc_loss

            #print("Trans: ", my_transf_loss.item(), ", Matching: ", my_matching_loss.item(), ", Cosim: ", my_cosim_loss.item(),  ", Sep: ", my_sep_loss.item(), ", Vk: ", my_vk_loss.item(),  ", Wk: ", my_wk_loss.item(), ", Recon:", my_recon_loss.item())
            #print("Sep: ", my_sep_loss.item(), ", recon_KP: ", my_recon_kp_loss.item(),  ", recon_f: ", my_recon_f_loss.item(), ", Recon:", my_recon_loss.item())
            #print("Sep: ", my_sep_loss.item(),  ", recon_s: ", my_recon_s_loss.item(), ", Recon:", my_recon_loss.item())
            print("Sep: ", my_sep_loss.item(), ", Conc: ", my_conc_loss.item(), ", Recon:", my_recon_loss.item())

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
            running_recon_loss = running_recon_loss + my_recon_loss.item()

            if (((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
            #if (((epoch + 1) % 2 == 0) or (epoch == 0) or (epoch + 1 == num_epochs)):
                #print("epoch: ", epoch)
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, kp, epoch + 1, cur_filename)
                #fn_save_tfkpimg = savetfKPimg()
                #fn_save_tfkpimg(tf_aefe_input, tf_kp, epoch + 1, cur_filename)
                img_save_filename = ("/home/jsk/AEFE_SLAM/SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch + 1))
                save_image(reconImg, img_save_filename)

        #if (epoch != 0) and ((epoch+1) % 5 == 0):
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

    print("!!!!!This is train_DETR3.py!!!!!")
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################