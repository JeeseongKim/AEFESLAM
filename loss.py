## loss function

import torch
from torch import nn
import math
import time
import torch.nn.functional as F
from utils import my_dataset, saveKPimg, make_transformation_M

class loss_concentration(nn.Module):
    def __init__(self, softmask):
        super(loss_concentration, self).__init__()
        dmap_sz_0, dmap_sz_1, _, _ = softmask.shape  # (b, k, 96, 128)
        var = 0

        for b in range(dmap_sz_0):
            for k in range(dmap_sz_1):
                var = var + torch.var(softmask[b, k, :, :])

        self.conc_loss = var ** 0.5

    def forward(self):
        return self.conc_loss

class loss_separation(nn.Module):
    def __init__(self, keypoints):
        super(loss_separation, self).__init__()
        sep_loss = 0
        kp_sz_0, kp_sz_1, _ = keypoints.shape
        self. scale_param = 2e-9 #1e-9
        #scale_param = 1e-3

        for i in range(kp_sz_1):
            cur_loss = F.mse_loss(keypoints[:, i, :].unsqueeze(1), keypoints)
            #sep_loss = sep_loss + torch.exp(-1*cur_loss)
            #sep_loss = sep_loss + torch.exp(-1 * self. scale_param * cur_loss)
            sep_loss = sep_loss + cur_loss

        #self.sep_loss_output = torch.exp(-1 * self. scale_param * sep_loss)
        self.sep_loss_output = 50/sep_loss
        #self.sep_loss_output = sep_loss

    def forward(self):
        return self.sep_loss_output

class loss_transformation(nn.Module):
    def __init__(self, theta, keypoints, tf_keypoints, cur_batch, num_of_kp):
        super(loss_transformation, self).__init__()

        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp)
        all_kp[:, 0, :] = keypoints[:, :, 0]
        all_kp[:, 1, :] = keypoints[:, :, 1]
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp)
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(b, 2, k)

        get_my_tf_keypoint = torch.zeros_like(tf_keypoints) #(b,k,2)
        get_my_tf_keypoint[:, 0, :] = cal_tf_keypoint[:, :, 0]

        for i in range(num_of_kp):
            get_my_tf_keypoint[:, i, :] = cal_tf_keypoint[:, :, i]

        self.transf_loss = F.mse_loss(tf_keypoints, get_my_tf_keypoint)

    def forward(self):
        return self.transf_loss
