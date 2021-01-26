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

def test():
    num_of_kp = 345
    feature_dimension = 32

    my_width = 272  # 208
    my_height = 80  # 64
    input_width = my_width

    stacked_hourglass_inpdim_kp = input_width
    stacked_hourglass_oupdim_kp = num_of_kp  # number of my keypoints

    num_nstack = 3

    learning_rate = 1e-4  # 1e-3
    weight_decay = 1e-5  # 1e-5 #5e-4

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP)
    model_StackedHourglassForKP.cuda()

    model_feature_descriptor = Linear(img_width=my_width, img_height=my_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor)
    model_feature_descriptor.cuda()

    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.module.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_feature_descriptor.module.load_state_dict(checkpoint['model_feature_descriptor'])

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

    test()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################