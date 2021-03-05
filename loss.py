## loss function

import torch
from torch import nn
import math
import time
import torch.nn.functional as F
from utils import my_dataset, saveKPimg, make_transformation_M
import numpy as np
import pdb

class loss_concentration(nn.Module):
    def __init__(self, softmask):
        super(loss_concentration, self).__init__()
        dmap_sz_0, dmap_sz_1, _, _ = softmask.shape  # (b, k, 96, 128)
        var = 0

        for b in range(dmap_sz_0):
            for k in range(dmap_sz_1):
                var = var + torch.var(softmask[b, k, :, :])

        #self.conc_loss = torch.exp(0.5*var) ** 0.5
        self.conc_loss = 2 * (var ** 0.5)
        #self.conc_loss = (var ** 2)


    def forward(self):
        return self.conc_loss

class loss_separation(nn.Module):
    def __init__(self, keypoints):
        super(loss_separation, self).__init__()
        sep_loss = 0
        kp_sz_0, kp_sz_1, _ = keypoints.shape
        self. scale_param = 0.0001#1e-9
        #scale_param = 1e-3

        for i in range(kp_sz_1):
            cur_loss = F.mse_loss(keypoints[:, i, :].unsqueeze(1), keypoints)
            #sep_loss = sep_loss + torch.exp(-1*cur_loss)
            #sep_loss = sep_loss + torch.exp(-1 * self. scale_param * cur_loss)
            sep_loss = sep_loss + cur_loss

        self.sep_loss_output = torch.exp(-1 * self. scale_param * sep_loss)
        #self.sep_loss_output = 200/sep_loss
        #self.sep_loss_output = sep_loss
        #self.sep_loss_output = (1+torch.tanh(1e-5*sep_loss))*(1-torch.tanh(1e-5*sep_loss))
        #self.sep_loss_output = (1-sep_loss)

    def forward(self):
        return self.sep_loss_output

class loss_transformation(nn.Module):
    def __init__(self, theta, keypoints, tf_keypoints, cur_batch, num_of_kp, my_width, my_height):
        super(loss_transformation, self).__init__()

        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp)
        all_kp[:, 0, :] = keypoints[:, :, 0] - (0.5 * my_width)
        all_kp[:, 1, :] = -keypoints[:, :, 1] + (0.5 * my_height)
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp)
        cal_tf_keypoint = torch.tensor(cal_tf_keypoint, dtype=torch.int64)
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(b, 2, k)

        get_my_tf_keypoint = cal_tf_keypoint.permute(0, 2, 1)
        '''
        get_my_tf_keypoint = torch.zeros_like(tf_keypoints) #(b,k,2)
        for i in range(num_of_kp):
            get_my_tf_keypoint[:, i, :] = cal_tf_keypoint[:, :, i]
        '''
        o_tf_keypoints = torch.zeros_like(tf_keypoints)
        o_tf_keypoints[:, :, 0] = tf_keypoints[:, :, 0] - (0.5 * my_width)
        o_tf_keypoints[:, :, 1] = -tf_keypoints[:, :, 1] + (0.5 * my_height)
        '''
        for b in range(cur_batch):
            for k in range(num_of_kp):
                tf_x = get_my_tf_keypoint[b, k, 0]
                tf_y = get_my_tf_keypoint[b, k, 1]
                if ((tf_x < (-0.5 * my_width)) or (tf_x > (0.5 * my_width)) or (tf_y < (-0.5*my_height)) or (tf_y > (0.5*my_height))):
                    o_tf_keypoints[b, k, :] = 0
                    get_my_tf_keypoint[b, k, :] = 0
        '''
        '''
        if ((tf_keypoints[:, :, 0] < (-0.5 * my_width)) or (tf_keypoints[:, :, 0] > (0.5 * my_width)) or (tf_keypoints[:, :, 1] < (0.5*my_height)) or (tf_keypoints[:, :, 1] > (0.5*my_height))):
            tf_keypoints[:, :, :] = 0
            get_my_tf_keypoint[:, :, :] = 0
        '''
        self.transf_loss = F.mse_loss(o_tf_keypoints, get_my_tf_keypoint.cuda())

    def forward(self):
        return self.transf_loss

class loss_cosim(nn.Module):
    def __init__(self, DetectionMap, tf_DetectionMap):
        super(loss_cosim, self).__init__()
        cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.cosim_loss = torch.sum(cosim(DetectionMap, tf_DetectionMap))

    def forward(self):
        return self.cosim_loss

class APLoss(nn.Module):
    """ differentiable AP loss, through quantization.

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: list of query AP (for each n in {1..N})
                 Note: typically, you want to minimize 1 - mean(AP)
    """

    def __init__(self, nq=25, min=0, max=1, euc=False):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        self.euc = euc
        gap = max - min
        assert gap > 0

        # init quantizer = non-learnable (fixed) convolution
        self.quantizer = q = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
        a = (nq - 1) / gap
        # 1st half = lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight.data[:nq] = -a
        q.bias.data[:nq] = torch.from_numpy(a * min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # 2nd half = lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight.data[nq:] = a
        q.bias.data[nq:] = torch.from_numpy(np.arange(2 - nq, 2, 1) - a * min)  # b = 1 - a*(min+x)
        # first and last one are special: just horizontal straight line
        q.weight.data[0] = q.weight.data[-1] = 0
        q.bias.data[0] = q.bias.data[-1] = 1

    def compute_AP(self, x, label):
        N, M = x.shape
        if self.euc:  # euclidean distance in same range than similarities
            x = 1 - torch.sqrt(2.001 - 2 * x)

        # quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # nb of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP
        return ap

    def forward(self, x, label):
        assert x.shape == label.shape  # N x M
        return self.compute_AP(x, label)
