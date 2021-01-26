## loss function

import torch
from torch import nn
import math
import time
import torch.nn.functional as F
from utils import my_dataset, saveKPimg, make_transformation_M

class loss_concentration(nn.Module):
    def __init__(self):
        super(loss_concentration, self).__init__()

    def forward(self, softmask):
        dmap_sz_0, dmap_sz_1, _, _ = softmask.shape #(b, k, 96, 128)
        var = 0
        for b in range (dmap_sz_0):
            for k in range (dmap_sz_1):
                var = var + torch.var(softmask[b, k, :, :])
        conc_loss = var ** 0.5
        #var_x = torch.var(torch.var(softmask, 1))
        #var_y = torch.var(torch.var(softmask, 0))
        #var_x = torch.mean(torch.var(softmask, 1))
        #var_y = torch.mean(torch.var(softmask, 0))

        #conc_loss = torch.tensor(2 * torch.tensor(math.pi) * torch.tensor(math.e) * ((var_x + var_y)**0.5))

        return conc_loss

class loss_separation(nn.Module):
    def __init__(self):
        super(loss_separation, self).__init__()
        #self.std_sep = 0.06  # 0.04 ~ 0.08
        #self.inv_std_sep = 1 / (self.std_sep ** 2)

    def forward(self, keypoints):
        sep_loss = 0
        kp_sz_0, kp_sz_1, _ = keypoints.shape

        for i in range(kp_sz_1):
            cur_loss = F.mse_loss(keypoints[:, i, :].unsqueeze(1), keypoints)
            #cur_loss = torch.sqrt(((keypoints[:, i, :].unsqueeze(1) - keypoints) ** 2).sum())
            sep_loss = sep_loss + cur_loss
            #sep_loss = sep_loss + torch.exp(- 0.05 * cur_loss)

        #print(sep_loss)
        #sep_loss_output = sep_loss
        scale_param = 1e-9
        sep_loss_output = torch.exp(-1 * scale_param * sep_loss)

        return sep_loss_output

class loss_transformation(nn.Module):
    def __init__(self):
        super(loss_transformation, self).__init__()

    def forward(self, theta, keypoints, tf_keypoints, cur_batch, num_of_kp):
        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp)
        all_kp[:, 0, :] = keypoints[:, :, 0]
        all_kp[:, 1, :] = keypoints[:, :, 1]
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp)
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(4, 2, 200)

        get_my_tf_keypoint = torch.zeros_like(tf_keypoints) #(4,200,2)

        get_my_tf_keypoint[:, 0, :] = cal_tf_keypoint[:, :, 0]

        for i in range(num_of_kp):
            get_my_tf_keypoint[:, i, :] = cal_tf_keypoint[:, :, i]

        transf_loss = F.mse_loss(tf_keypoints, get_my_tf_keypoint)

        return transf_loss
