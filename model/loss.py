## loss function

from model.DetectionConfidenceMap import *
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import cv2

class loss_concentration(nn.Module):
    def __init__(self, my_map):
        super(loss_concentration, self).__init__()
        dmap_sz_0, dmap_sz_1, _, _ = my_map.shape  # (b, k, 96, 128)

        #var = torch.var(my_map, dim=[2, 3]).sum(dim=[0, 1])
        h_var = torch.var(my_map, dim=2).sum(dim=[0, 1])
        w_var = torch.var(my_map, dim=3).sum(dim=[0, 1])
        var = h_var.sum() + w_var.sum()

        max_min_diff = torch.max(my_map) - torch.min(my_map)
        #self.conc_loss = torch.exp(0.5*var) ** 0.5

        self.conc_loss = 0.01 * (var ** 0.5) + 2 * torch.exp(-0.1 * max_min_diff)
        #self.conc_loss = (var ** 2)


    def forward(self):
        return self.conc_loss

class loss_separation(nn.Module):
    def __init__(self):
        super(loss_separation, self).__init__()

    def forward(self, keypoints):
        sep_loss = 0
        kp_sz_0, num_of_kp, _ = keypoints.shape
        keypoints = keypoints.float()

        for i in range(num_of_kp):
            #cur_loss = F.mse_loss(keypoints[:, i, :].unsqueeze(1), keypoints)
            cur_loss = torch.cdist(keypoints[:, i, :].unsqueeze(1), keypoints, p=2)
            sep_loss = sep_loss + torch.mean(cur_loss)

        #self.sep_loss_output = torch.exp(-0.0001 * sep_loss)
        #self.sep_loss_output = 0.01 * ((10*sep_loss) ** 0.5)
        #self.sep_loss_output = 200/sep_loss
        #self.sep_loss_output = (1+torch.tanh(1e-2 * sep_loss))*(1-torch.tanh(1e-2 * sep_loss))

        #self.sep_loss_output = 2*(1-torch.sigmoid(1e-4*sep_loss))
        #self.sep_loss_output = 4 * torch.sigmoid(1e-6 * sep_loss) * (1 - torch.sigmoid(1e-6 * sep_loss))
        #sep_loss_output = torch.exp(-5*1e-5*sep_loss)
        #sep_loss_output = torch.exp(-1e-3*sep_loss)
        sep_loss_output = torch.exp(-1e-4*sep_loss)
        #self.sep_loss_output = 4 * torch.sigmoid(1e-9 * sep_loss) * (1 - torch.sigmoid(1e-9 * sep_loss))

        return sep_loss_output

class loss_transformation(nn.Module):
    def __init__(self, theta, keypoints, tf_keypoints, cur_batch, num_of_kp, my_width, my_height):
        super(loss_transformation, self).__init__()

        tf_keypoints = tf_keypoints.cuda()
        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp)
        all_kp[:, 0, :] = keypoints[:, :, 0] - (0.5 * my_width)
        all_kp[:, 1, :] = -keypoints[:, :, 1] + (0.5 * my_height)
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp)
        cal_tf_keypoint = torch.round(cal_tf_keypoint).float()
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(b, 2, k)
        get_my_tf_keypoint = cal_tf_keypoint.permute(0, 2, 1).cuda()

        o_tf_keypoints = torch.zeros_like(tf_keypoints)
        o_tf_keypoints[:, :, 0] = get_my_tf_keypoint[:, :, 0] + (0.5 * my_width)
        o_tf_keypoints[:, :, 1] = -get_my_tf_keypoint[:, :, 1] + (0.5 * my_height)

        self.my_transf_loss = F.mse_loss(o_tf_keypoints, tf_keypoints)
        self.transf_loss = self.my_transf_loss

    def forward(self):
        return self.transf_loss

class loss_transformation_3kp(nn.Module):
    def __init__(self, theta, keypoints, tf_keypoints, cur_batch, num_of_kp, my_width, my_height):
        super(loss_transformation_3kp, self).__init__()
        num_of_kp = num_of_kp * 3

        tf_keypoints = tf_keypoints.cuda()
        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp)
        all_kp[:, 0, :] = keypoints[:, :, 0] - (0.5 * my_width)
        all_kp[:, 1, :] = -keypoints[:, :, 1] + (0.5 * my_height)
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp)
        cal_tf_keypoint = torch.round(cal_tf_keypoint).float()
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(b, 2, k)

        get_my_tf_keypoint = cal_tf_keypoint.permute(0, 2, 1).cuda()
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
        self.my_transf_loss = F.mse_loss(o_tf_keypoints, get_my_tf_keypoint)
        self.transf_loss = self.my_transf_loss

    def forward(self):
        return self.transf_loss

class loss_cosim(nn.Module):
    def __init__(self):
        super(loss_cosim, self).__init__()

    def forward(self, DetectionMap, tf_DetectionMap):

        cosim = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        my_val = (cosim(DetectionMap, tf_DetectionMap)).mean()
        #cosim_loss = torch.exp(-1.0 * my_val)
        cosim_loss = 1 - my_val

        return cosim_loss


class loss_matching(nn.Module):
    def __init__(self, theta, my_feature, my_tf_feature, cur_batch, num_of_kp, my_width, my_height):
        super(loss_matching, self).__init__()
        self. my_feature = my_feature
        self.my_tf_feature = my_tf_feature

        tf_keypoints = my_tf_feature[:, :, 0:2].cuda()
        keypoints = my_feature[:, :, 0:2].cuda()

        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp)
        all_kp[:, 0, :] = keypoints[:, :, 0] - (0.5 * my_width)
        all_kp[:, 1, :] = -keypoints[:, :, 1] + (0.5 * my_height)
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp)
        cal_tf_keypoint = torch.round(cal_tf_keypoint).float()
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(b, 2, k)

        get_my_tf_keypoint = cal_tf_keypoint.permute(0, 2, 1).cuda()

        o_tf_keypoints = torch.zeros_like(tf_keypoints)
        o_tf_keypoints[:, :, 0] = get_my_tf_keypoint[:, :, 0] + (0.5 * my_width)
        o_tf_keypoints[:, :, 1] = -get_my_tf_keypoint[:, :, 1] + (0.5 * my_height)

        self.my_transf_loss = F.mse_loss(o_tf_keypoints, tf_keypoints)

    def hamming_distance(self, fk, tf_fk):
        dist = torch.cdist(fk, tf_fk, p=0)

        return dist

    def forward(self):
        fk = self.my_feature[:, :, 2:]
        tf_fk = self.my_tf_feature[:, :, 2:]
        #my_dist = self.hamming_distance(fk, tf_fk)
        #my_dist = torch.sum(my_dist, dim=2)
        #matching_loss = torch.mean(my_dist)
        matching_loss = F.mse_loss(fk, tf_fk)
        #my_matching_loss = 0.01 * ((0.005*matching_loss) ** 0.3)
        my_matching_loss = matching_loss
        #return self.my_transf_loss , self.matching_loss
        return self.my_transf_loss, my_matching_loss

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


# Calculate the one-dimensional Gaussian distribution vector
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# Create a Gaussian kernel, obtained by matrix multiplication of two one-dimensional Gaussian distribution vectors
# You can set the channel parameter to expand to 3 channels
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# Calculate SSIM
# Use the formula of SSIM directly, but when calculating the average value, instead of directly calculating the pixel average value, normalized Gaussian kernel convolution is used instead.
# The formula Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y] is used when calculating variance and covariance .
# As mentioned earlier, the above expectation operation is replaced by Gaussian kernel convolution.
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_kp: float = 1, cost_desc: float = 1):
        super().__init__()
        self.cost_kp = cost_kp
        self.cost_desc = cost_desc
        assert cost_kp != 0 or cost_desc != 0 , "all costs cant be 0"

    @torch.no_grad()
    def forward(self, theta, my_feature, my_tf_feature, my_width, my_height):

        cur_batch, num_of_kp, _ = my_feature.shape

        my_kp = my_feature[:, :, 0:2]
        my_tf_kp = my_tf_feature[:, :, 0:2]
        my_desc = my_feature[:, :, 2:]
        my_tf_desc = my_tf_feature[:, :, 2:]

        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp)
        all_kp[:, 0, :] = my_kp[:, :, 0] - (0.5 * my_width)
        all_kp[:, 1, :] = -my_kp[:, :, 1] + (0.5 * my_height)
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp)
        cal_tf_keypoint = torch.round(cal_tf_keypoint).float()
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(b, 2, k)

        get_my_tf_keypoint = cal_tf_keypoint.permute(0, 2, 1).cuda()

        o_tf_keypoints = torch.zeros_like(my_tf_kp)
        o_tf_keypoints[:, :, 0] = get_my_tf_keypoint[:, :, 0] + (0.5 * my_width)
        o_tf_keypoints[:, :, 1] = -get_my_tf_keypoint[:, :, 1] + (0.5 * my_height)

        predicted_tf_kp = my_tf_kp
        cal_tf_kp = o_tf_keypoints
        cost_kp = torch.cdist(predicted_tf_kp.flatten(0, 1), cal_tf_kp.flatten(0, 1), p=1).cuda()

        '''
        ##Filter out negative kps
        flatten_pred_tf = predicted_tf_kp.flatten(0, 1)
        flatten_cal_tf = cal_tf_kp.flatten(0, 1)

        flatten_desc = my_desc.flatten(0, 1)
        flatten_tf_desc = my_tf_desc.flatten(0, 1)

        keep_idx =[]
        for idx in range(num_of_kp * cur_batch):
                if(flatten_cal_tf[idx, 0] >= 0) and (flatten_cal_tf[idx, 1] >= 0):
                    keep_idx.append(idx)

        Learned_Cal_tf_kp = torch.zeros(len(keep_idx), 4)
        Filtered_desc = torch.zeros(len(keep_idx), 512)
        for iidx in range(len(keep_idx)):
            Learned_Cal_tf_kp[iidx, 0:2] = flatten_pred_tf[keep_idx[iidx], :]
            Learned_Cal_tf_kp[iidx, 2:4] = flatten_cal_tf[keep_idx[iidx], :]
            Filtered_desc[iidx, 0:256] = flatten_desc[keep_idx[iidx], :]
            Filtered_desc[iidx, 256:512] = flatten_tf_desc[keep_idx[iidx], :]

        filtered_learned_tf_kp = Learned_Cal_tf_kp[:, 0:2]
        filtered_cal_tf_kp = Learned_Cal_tf_kp[:, 2:4]

        filtered_my_desc = Filtered_desc[:, 0:256]
        filtered_my_tf_desc = Filtered_desc[:, 256:512]

        #cost_kp = torch.cdist(predicted_tf_kp.flatten(0, 1), cal_tf_kp.flatten(0, 1), p=1).cuda()
        cost_kp = torch.cdist(filtered_learned_tf_kp, filtered_cal_tf_kp, p=1).cuda()
        cost_desc = torch.cdist(filtered_my_desc, filtered_my_tf_desc, p=2).cuda()  # p = norm
        '''
        '''
        pts1 = my_kp.int().detach().cpu().numpy()
        pts2 = my_tf_kp.int().detach().cpu().numpy()

        pts1 = pts1.reshape(pts1.shape[0]*pts1.shape[1], pts1.shape[2])
        pts2 = pts2.reshape(pts2.shape[0]*pts2.shape[1], pts2.shape[2])

        FundamentalMatrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

        ppts1 = torch.ones(cur_batch*num_of_kp, 3)
        ppts2 = torch.ones(cur_batch*num_of_kp, 3)

        ppts1[:, 0] = torch.tensor(pts1[:, 0])
        ppts1[:, 1] = torch.tensor(pts1[:, 1])
        ppts1[:, 2] = 1.0

        ppts2[:, 0] = torch.tensor(pts2[:, 0])
        ppts2[:, 1] = torch.tensor(pts2[:, 1])
        ppts2[:, 2] = 1.0

        t_ppts1 = ppts1.permute(1, 0)
        tensor_FundamentalM = torch.tensor(FundamentalMatrix)
        aaa = torch.matmul(tensor_FundamentalM.float(), t_ppts1.float())
        #t_ppts2 = ppts2.permute(1, 0)
        bbb = torch.matmul(ppts2.float(), aaa)

        cost_kp = torch.abs(bbb).cuda()
        '''

        flatten_my_desc = my_desc.flatten(0, 1)
        flatten_my_tf_desc = my_tf_desc.flatten(0, 1)
        cost_desc = torch.cdist(flatten_my_desc, flatten_my_tf_desc, p=2).cuda() #p = norm

        '''
        # We flatten to compute the cost matrices in a batch
        flatten_p_tf_kp = predicted_tf_kp.flatten(0, 1)
        flatten_cal_tf_kp = cal_tf_kp.flatten(0, 1)

        #flatten_my_desc = my_desc.flatten(0, 1)
        #flatten_my_tf_desc = my_tf_desc.flatten(0, 1)

        cost_kp = torch.cdist(flatten_p_tf_kp, flatten_cal_tf_kp, p=1)
        #cost_desc = torch.cdist(flatten_my_desc, flatten_my_tf_desc, p=1)
        '''

        # Final cost matrix
        C = self.cost_kp * cost_kp + self.cost_desc * cost_desc
        C = C.view(cur_batch, num_of_kp, -1).cpu()
        sizes = num_of_kp
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        match = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        #return match, cal_tf_kp
        return match



class HungarianMatcherNLoss(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_kp: float = 1, cost_desc: float = 1):
        super().__init__()
        self.cost_kp = cost_kp
        self.cost_desc = cost_desc
        assert cost_kp != 0 or cost_desc != 0 , "all costs cant be 0"

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @torch.no_grad()
    def forward(self, theta, my_feature, my_tf_feature, my_width, my_height):

        cur_batch, num_of_kp, _ = my_feature.shape

        my_kp = my_feature[:, :, 0:2]
        my_tf_kp = my_tf_feature[:, :, 0:2]
        my_desc = my_feature[:, :, 2:]
        my_tf_desc = my_tf_feature[:, :, 2:]

        make_transformation = make_transformation_M()
        my_tfMatrix = make_transformation(theta, 0, 0)

        all_kp = torch.zeros(cur_batch, 4, num_of_kp) #mid = origin
        all_kp[:, 0, :] = my_kp[:, :, 0] - (0.5 * my_width)
        all_kp[:, 1, :] = -my_kp[:, :, 1] + (0.5 * my_height)
        all_kp[:, 3, :] = 1.0

        cal_tf_keypoint = torch.matmul(my_tfMatrix, all_kp) #mid = origin
        cal_tf_keypoint = torch.round(cal_tf_keypoint).float()
        cal_tf_keypoint = cal_tf_keypoint[:, 0:2, :] #(b, 2, k)

        get_my_tf_keypoint = cal_tf_keypoint.permute(0, 2, 1).cuda() #mid = origin

        o_tf_keypoints = torch.zeros_like(my_tf_kp) #edge = origin
        o_tf_keypoints[:, :, 0] = get_my_tf_keypoint[:, :, 0] + (0.5 * my_width)
        o_tf_keypoints[:, :, 1] = -get_my_tf_keypoint[:, :, 1] + (0.5 * my_height)

        predicted_tf_kp = my_tf_kp #edge = origin
        cal_tf_kp = o_tf_keypoints #edge = origin

        ##Filter out negative kps
        flatten_tf = my_kp.flatten(0, 1)
        flatten_pred_tf = predicted_tf_kp.flatten(0, 1)
        flatten_cal_tf = cal_tf_kp.flatten(0, 1)

        flatten_desc = my_desc.flatten(0, 1)
        flatten_tf_desc = my_tf_desc.flatten(0, 1)

        keep_idx = []
        for idx in range(num_of_kp * cur_batch):
            if(flatten_cal_tf[idx, 0] >= 0) and (flatten_cal_tf[idx, 1] >= 0):
                keep_idx.append(idx)
        
        Learned_Cal_tf_kp = torch.zeros(len(keep_idx), 4)
        Filtered_desc = torch.zeros(len(keep_idx), 512)
        for iidx in range(len(keep_idx)):
            Learned_Cal_tf_kp[iidx, 0:2] = flatten_pred_tf[keep_idx[iidx], :] #edge=origin
            Learned_Cal_tf_kp[iidx, 2:4] = flatten_cal_tf[keep_idx[iidx], :] #edge=origin
            #Learned_Cal_tf_kp[iidx, 4:6] = flatten_tf[keep_idx[iidx], :] #edge=origin
            Filtered_desc[iidx, 0:256] = flatten_desc[keep_idx[iidx], :]
            Filtered_desc[iidx, 256:512] = flatten_tf_desc[keep_idx[iidx], :]

        '''
        Learned_Cal_tf_kp = torch.zeros(len(flatten_cal_tf), 4)
        Learned_Cal_tf_kp[:, 0:2] = flatten_pred_tf
        Learned_Cal_tf_kp[:, 2:4] = flatten_cal_tf
        '''
        filtered_learned_tf_kp = Learned_Cal_tf_kp[:, 0:2]
        filtered_cal_tf_kp = Learned_Cal_tf_kp[:, 2:4]
        #filtered_learned_kp = Learned_Cal_tf_kp[:, 4:6]

        filtered_my_desc = Filtered_desc[:, 0:256]
        filtered_my_tf_desc = Filtered_desc[:, 256:512]

        #cost_kp = torch.cdist(predicted_tf_kp.flatten(0, 1), cal_tf_kp.flatten(0, 1), p=1).cuda()
        #cost_kp = torch.cdist(filtered_learned_tf_kp, filtered_cal_tf_kp, p=2).cuda()
        #cost_desc = torch.cdist(filtered_my_desc, filtered_my_tf_desc, p=2).cuda()  # p = norm
        cost_kp = torch.cdist(filtered_learned_tf_kp, filtered_cal_tf_kp, p=1).cuda()
        cost_desc = torch.cdist(filtered_my_desc, filtered_my_tf_desc, p=1).cuda()  # p = norm

        # Final cost matrix
        ratio = cost_kp.sum()/cost_desc.sum()
        #C = self.cost_kp * cost_kp + self.cost_desc * cost_desc
        C = 1 * cost_kp + ratio * cost_desc
        #C = cost_kp
        C = C.cpu()
        indices = linear_sum_assignment(C)
        match = indices

        MyCost = C[match[0], match[1]].sum().cuda()
        #src_idx = self._get_src_permutation_idx(indices)
        #tgt_idx = self._get_tgt_permutation_idx(indices)

        ##compute loss with match
        src_kp = filtered_learned_tf_kp[match[0]] #predicted transformationed kp
        cal_target_kp = filtered_cal_tf_kp[match[1]] #original predicted kp cal to transformationed kp
        #src_kp = filtered_learned_tf_kp[match[1]] #predicted transformationed kp
        #cal_target_kp = filtered_cal_tf_kp[match[0]] #original predicted kp cal to transformationed kp
        #target_kp = filtered_learned_kp[match[1]] #original predicted

        #loss_kp = torch.nn.functional.mse_loss(src_kp.float(), cal_target_kp.float())
        loss_kp_match = F.l1_loss(src_kp.float(), cal_target_kp.float(), reduction='none')
        loss_kp = loss_kp_match.sum()/len(keep_idx)
        #loss_kp_match = torch.cdist(src_kp.float(), cal_target_kp.float(), p=2)

        # loss_kp = 10 * loss_fundamental + 1 * loss_kp_match

        src_desc = filtered_my_desc[match[0]]
        target_desc = filtered_my_tf_desc[match[1]]
        #src_desc = filtered_my_desc[match[1]]
        #target_desc = filtered_my_tf_desc[match[0]]

        loss_desc_match = F.l1_loss(src_desc.float(), target_desc.float(), reduction='none')
        loss_desc = loss_desc_match.sum() / len(keep_idx)
        #loss_desc = torch.nn.functional.mse_loss(src_desc, target_desc)

        #return loss_kp_match, loss_desc
        return loss_kp, loss_desc, MyCost


class matcher_criterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    #def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
    def __init__(self):
        super().__init__()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, match, kp, tf_kp, desc, tf_desc):

        idx_src = self._get_src_permutation_idx(match)
        idx_trg = self._get_tgt_permutation_idx(match)

        src_kp = kp[idx_src]
        target_kp = tf_kp[idx_trg]

        '''
        pts1 = src_kp.int().detach().cpu().numpy()
        pts2 = target_kp.int().detach().cpu().numpy()

        FundamentalMatrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

        ppts1 = torch.ones(kp.shape[0] * kp.shape[1], kp.shape[2] + 1)
        ppts2 = torch.ones(kp.shape[0] * kp.shape[1], kp.shape[2] + 1)

        pts1 = torch.tensor(pts1)
        pts2 = torch.tensor(pts2)

        ppts1[:, 0] = pts1[:, 0]
        ppts1[:, 1] = pts1[:, 1]
        ppts1[:, 2] = 1.0

        ppts2[:, 0] = pts2[:, 0]
        ppts2[:, 1] = pts2[:, 1]
        ppts2[:, 2] = 1.0

        FundamentalMatrix_ = torch.tensor(FundamentalMatrix)

        aaa = torch.matmul(ppts1.float(), FundamentalMatrix_.float())
        t_ppts2 = ppts2.permute(1, 0)
        bbb = torch.matmul(aaa, t_ppts2.float())
        tmp_result = torch.abs(bbb)
        '''
        #loss_kp = F.mse_loss(src_kp, target_kp)
        # loss_fundamental = torch.trace(tmp_result).cuda()
        loss_kp_match = torch.nn.functional.mse_loss(src_kp.float(), target_kp.float())

        #loss_kp = 10 * loss_fundamental + 1 * loss_kp_match

        src_desc = desc[idx_src]
        target_desc = tf_desc[idx_trg]

        loss_desc = torch.nn.functional.mse_loss(src_desc, target_desc)

        return loss_kp_match, loss_desc


class NCC_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, aefe_input, tf_aefe_input, match, kp, tf_kp):
        idx_src = self._get_src_permutation_idx(match)
        idx_trg = self._get_tgt_permutation_idx(match)

        src_kp = kp[idx_src]
        target_kp = tf_kp[idx_trg]

        window_sz = 5
        u1 = src_kp[:, 0]
        v1 = src_kp[:, 1]
        u2 = target_kp[:, 0]
        v2 = target_kp[:, 1]

        ncc_input_src = aefe_input[:, :, u1-window_sz:u1+window_sz, v1-window_sz:v1+window_sz]
        ncc_input_trg = tf_aefe_input[:, :, u2-window_sz:u2+window_sz, v2-window_sz:v2+window_sz]

        ncc = (ncc_input_src * ncc_input_trg)/(torch.abs(ncc_input_src)*torch.abs(ncc_input_trg))

        return ncc

