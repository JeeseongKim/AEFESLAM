import torch
from torch import nn
import numpy as np
import time
from model.layers import *
from model.resnet import *

import torch.nn.functional as F
import math
from GenHeatmap import *

def dev_sigmoid_100(x):
    out = 4 * torch.sigmoid(100 * x) * (1 - torch.sigmoid(100 * x))

    return out

def speed_sigmoid_5(x):
    out = 1 / (1 + torch.exp(-5 * x))

    return out

def speed_sigmoid_5_trans_x_1(x):
    out = 1 / (1 + torch.exp(-5 * (x-1)))

    return out

def speed_sigmoid_50(x):
    out = 1 / (1 + torch.exp(-50 * x))

    return out


def speed_sigmoid_100(x):
    out = 1 / (1 + torch.exp(-100 * x))

    return out


def speed_order4(x):
    out = (-10**5)*(x**4) + (10**4)*(x**3) + x

    return out

def tanh_05(x):
    out = (1 / (1 + torch.exp(-0.5 * x))) * (1 - (1 / (1 + torch.exp(-0.5 * x))))

    return out

def tanh_5(x):
    out = (1 / (1 + torch.exp(-5 * x))) * (1 - (1 / (1 + torch.exp(-5 * x))))

    return out


class DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self):
        super(DetectionConfidenceMap2keypoint, self).__init__()
        self.elu = torch.nn.ELU(inplace=True)

    def forward(self, Rk, tf_Rk, my_height, my_width):
        #_, inp_channel, img_height, img_width = Rk.shape
        #keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()
        #tf_keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()

        #Rk = self.elu(Rk)
        #tf_Rk = self.elu(tf_Rk)

        Dk_min = torch.min(torch.min(Rk, dim=2)[0], dim=2)[0]
        Dk_max = torch.max(torch.max(Rk, dim=2)[0], dim=2)[0]
        my_max_min = torch.cat([Dk_min.unsqueeze(2), Dk_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
        Dk = (Rk - (my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((my_max_min[:, :, 1] - my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
        #Dk = torch.sigmoid(Rk)
        #Dk = speed_order4(Rk)
        #Dk = speed_sigmoid_50(Rk)
        #Dk = tanh_5(Rk)


        tf_Dk_min = torch.min(torch.min(tf_Rk, dim=2)[0], dim=2)[0]
        tf_Dk_max = torch.max(torch.max(tf_Rk, dim=2)[0], dim=2)[0]
        tf_my_max_min = torch.cat([tf_Dk_min.unsqueeze(2), tf_Dk_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
        tf_Dk = (tf_Rk - (tf_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((tf_my_max_min[:, :, 1] - tf_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
        #tf_Dk = torch.sigmoid(tf_Rk)
        #tf_Dk = speed_order4(tf_Rk)
        #tf_Dk = speed_sigmoid_50(tf_Rk)
        #tf_Dk = tanh_5(tf_Rk)

        #map_val_all = torch.softmax(Rk, dim=1)
        #tf_map_val_all = torch.softmax(tf_Rk, dim=1)

        get_zeta = Dk.sum([2, 3]) #(b, k)
        tf_get_zeta = tf_Dk.sum([2, 3]) #(b, k)

        x_indices = torch.arange(0, my_width).repeat(Rk.shape[0], Rk.shape[1], my_height, 1).cuda()
        y_indices = torch.arange(0, my_height).repeat(Rk.shape[0], Rk.shape[1], my_width, 1).permute(0, 1, 3, 2).cuda()

        get_kp_x = (Dk * x_indices).sum(dim=[2, 3])
        get_kp_y = (Dk * y_indices).sum(dim=[2, 3])

        tf_get_kp_x = (tf_Dk * x_indices).sum(dim=[2, 3])
        tf_get_kp_y = (tf_Dk * y_indices).sum(dim=[2, 3])

        kp = torch.cat([(torch.round(get_kp_x/get_zeta)).unsqueeze(2).float(), (torch.round(get_kp_y/get_zeta)).unsqueeze(2).float()], dim=2)
        tf_kp = torch.cat([(torch.round(tf_get_kp_x/tf_get_zeta)).unsqueeze(2), (torch.round(tf_get_kp_y/tf_get_zeta)).unsqueeze(2)], dim=2)

        return Dk, tf_Dk, kp, tf_kp, get_zeta, tf_get_zeta

class DetectionConfidenceMap2keypoint_3kp(nn.Module):
    def __init__(self):
        super(DetectionConfidenceMap2keypoint_3kp, self).__init__()
        self.elu = torch.nn.ELU(inplace=True)

    def forward(self, Rk, tf_Rk, my_height, my_width):
        #_, inp_channel, img_height, img_width = Rk.shape
        #keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()
        #tf_keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()

        #Rk = self.elu(Rk)
        #tf_Rk = self.elu(tf_Rk)

        #Dk_min = torch.min(torch.min(Rk, dim=2)[0], dim=2)[0]
        #Dk_max = torch.max(torch.max(Rk, dim=2)[0], dim=2)[0]
        #my_max_min = torch.cat([Dk_min.unsqueeze(2), Dk_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
        #Dk = (Rk - (my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((my_max_min[:, :, 1] - my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
        Dk = torch.sigmoid(Rk)

        #tf_Dk_min = torch.min(torch.min(tf_Rk, dim=2)[0], dim=2)[0]
        #tf_Dk_max = torch.max(torch.max(tf_Rk, dim=2)[0], dim=2)[0]
        #tf_my_max_min = torch.cat([tf_Dk_min.unsqueeze(2), tf_Dk_max.unsqueeze(2)], dim=2)  # (b,k,2) 2: min, max
        #tf_Dk = (tf_Rk - (tf_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3))) / ((tf_my_max_min[:, :, 1] - tf_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))
        tf_Dk = torch.sigmoid(tf_Rk)

        #map_val_all = torch.softmax(Rk, dim=1)
        #tf_map_val_all = torch.softmax(tf_Rk, dim=1)

        get_zeta = Dk.sum([2, 3]) #(b, k)
        tf_get_zeta = tf_Dk.sum([2, 3]) #(b, k)

        x_indices = torch.arange(0, my_width).repeat(Rk.shape[0], Rk.shape[1], my_height, 1).cuda()
        y_indices = torch.arange(0, my_height).repeat(Rk.shape[0], Rk.shape[1], my_width, 1).permute(0, 1, 3, 2).cuda()

        get_kp_x = (Dk * x_indices).sum(dim=[2, 3])
        get_kp_y = (Dk * y_indices).sum(dim=[2, 3])

        tf_get_kp_x = (tf_Dk * x_indices).sum(dim=[2, 3])
        tf_get_kp_y = (tf_Dk * y_indices).sum(dim=[2, 3])

        kp = torch.cat([(torch.round(get_kp_x/get_zeta)).unsqueeze(2), (torch.round(get_kp_y/get_zeta)).unsqueeze(2)], dim=2)
        tf_kp = torch.cat([(torch.round(tf_get_kp_x/tf_get_zeta)).unsqueeze(2), (torch.round(tf_get_kp_y/tf_get_zeta)).unsqueeze(2)], dim=2)

        cur_batch = Rk.shape[0]
        num_of_kp = Rk.shape[1]

        kp1 = torch.zeros_like(kp)
        kp2 = torch.zeros_like(kp)
        tf_kp1 = torch.zeros_like(kp)
        tf_kp2 = torch.zeros_like(kp)

        for b in range(cur_batch):
            for k in range(num_of_kp):
                my_h = kp[b, k, 1] #keypoint y = height
                my_w = kp[b, k, 0] #keypoint x = width
                kp1[b, k, :] = kp[b, k, :] + kp[b, k, :] * Dk[b, k, my_h.long(), my_w.long()]
                kp2[b, k, :] = kp[b, k, :] - kp[b, k, :] * Dk[b, k, my_h.long(), my_w.long()]

                my_tf_h = tf_kp[b, k, 1] #keypoint y = height
                my_tf_w = tf_kp[b, k, 0] #keypoint x = width
                tf_kp1[b, k, :] = tf_kp[b, k, :] + tf_kp[b, k, :] * tf_Dk[b, k, my_tf_h.long(), my_tf_w.long()]
                tf_kp2[b, k, :] = tf_kp[b, k, :] - tf_kp[b, k, :] * tf_Dk[b, k, my_tf_h.long(), my_tf_w.long()]

        kp1 = kp1.int()
        kp2 = kp2.int()
        tf_kp1 = tf_kp1.int()
        tf_kp2 = tf_kp2.int()

        keypoint = torch.cat([kp, kp1, kp2], dim=1)
        tf_keypoint = torch.cat([tf_kp, tf_kp1, tf_kp2], dim=1)

        return Dk, tf_Dk, keypoint, tf_keypoint, get_zeta, tf_get_zeta

class DetectionConfidenceMap2keypoint_2(nn.Module):
    def __init__(self, my_width, my_height):
        super(DetectionConfidenceMap2keypoint_2, self).__init__()
        self.elu = torch.nn.ELU(inplace=True)
        self.my_width = my_width
        self.my_height = my_height


    def nms_fast(self, kp, dist_thresh=4):
        H = self.my_height
        W = self.my_width

        in_corners = kp.detach().cpu().numpy()

        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.

        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :], axis=2)
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def forward(self, Rk, tf_Rk):
        Dk = torch.sigmoid(Rk)
        tf_Dk = torch.sigmoid(tf_Rk)

        get_zeta = Dk.sum([2, 3]) #(b, k)
        tf_get_zeta = tf_Dk.sum([2, 3]) #(b, k)

        x_indices = torch.arange(0, self.my_width).repeat(Rk.shape[0], Rk.shape[1], self.my_height, 1).cuda()
        y_indices = torch.arange(0, self.my_height).repeat(Rk.shape[0], Rk.shape[1], self.my_width, 1).permute(0, 1, 3, 2).cuda()

        get_kp_x = (Dk * x_indices).sum(dim=[2, 3])
        get_kp_y = (Dk * y_indices).sum(dim=[2, 3])

        tf_get_kp_x = (tf_Dk * x_indices).sum(dim=[2, 3])
        tf_get_kp_y = (tf_Dk * y_indices).sum(dim=[2, 3])

        kp = torch.cat([(torch.round(get_kp_x/get_zeta)).unsqueeze(2), (torch.round(get_kp_y/get_zeta)).unsqueeze(2)], dim=2)
        tf_kp = torch.cat([(torch.round(tf_get_kp_x/tf_get_zeta)).unsqueeze(2), (torch.round(tf_get_kp_y/tf_get_zeta)).unsqueeze(2)], dim=2)

        cur_batch = Rk.shape[0]
        num_of_kp = Rk.shape[1]

        kp1 = torch.zeros_like(kp)
        kp2 = torch.zeros_like(kp)
        tf_kp1 = torch.zeros_like(kp)
        tf_kp2 = torch.zeros_like(kp)

        idx = 0
        score_kp = torch.zeros([cur_batch, num_of_kp])
        score_kp1 = torch.zeros([cur_batch, num_of_kp])
        score_kp2 = torch.zeros([cur_batch, num_of_kp])

        score_tf_kp = torch.zeros([cur_batch, num_of_kp])
        score_tf_kp1 = torch.zeros([cur_batch, num_of_kp])
        score_tf_kp2 = torch.zeros([cur_batch, num_of_kp])

        for b in range(cur_batch):
            for k in range(num_of_kp):
                my_h = kp[b, k, 1] #keypoint y = height
                my_w = kp[b, k, 0] #keypoint x = width
                kp1[b, k, :] = kp[b, k, :] + kp[b, k, :] * Dk[b, k, my_h.long(), my_w.long()]
                kp2[b, k, :] = kp[b, k, :] - kp[b, k, :] * Dk[b, k, my_h.long(), my_w.long()]
                kp1 = kp1.int()
                kp2 = kp2.int()

                score_kp[b, k] = Dk[b, k, my_h.long(), my_w.long()]
                h1 = kp1[b, k, 1]
                w1 = kp1[b, k, 0]
                score_kp1[b, k] = Dk[b, k, h1.long(), w1.long()]
                h2 = kp2[b, k, 1]
                w2 = kp2[b, k, 0]
                score_kp2[b, k] = Dk[b, k, h2.long(), w2.long()]

                my_tf_h = tf_kp[b, k, 1] #keypoint y = height
                my_tf_w = tf_kp[b, k, 0] #keypoint x = width
                tf_kp1[b, k, :] = tf_kp[b, k, :] + tf_kp[b, k, :] * tf_Dk[b, k, my_tf_h.long(), my_tf_w.long()]
                tf_kp2[b, k, :] = tf_kp[b, k, :] - tf_kp[b, k, :] * tf_Dk[b, k, my_tf_h.long(), my_tf_w.long()]
                tf_kp1 = tf_kp1.int()
                tf_kp2 = tf_kp2.int()

                score_tf_kp[b, k] = tf_Dk[b, k, my_tf_h.long(), my_tf_w.long()]
                tf_h1 = tf_kp1[b, k, 1]
                tf_w1 = tf_kp1[b, k, 0]
                score_tf_kp1[b, k] = tf_Dk[b, k, tf_h1.long(), tf_w1.long()]
                tf_h2 = tf_kp2[b, k, 1]
                tf_w2 = tf_kp2[b, k, 0]
                score_tf_kp2[b, k] = Dk[b, k, tf_h2.long(), tf_w2.long()]

                idx = idx+1

        kp_w_score = torch.cat([kp, score_kp.unsqueeze(2).cuda()], dim=2)
        kp1_w_score = torch.cat([kp1, score_kp1.unsqueeze(2).cuda()], dim=2)
        kp2_w_score = torch.cat([kp2, score_kp2.unsqueeze(2).cuda()], dim=2)

        tf_kp_w_score = torch.cat([tf_kp, score_tf_kp.unsqueeze(2).cuda()], dim=2)
        tf_kp1_w_score = torch.cat([tf_kp1, score_tf_kp1.unsqueeze(2).cuda()], dim=2)
        tf_kp2_w_score = torch.cat([tf_kp2, score_tf_kp2.unsqueeze(2).cuda()], dim=2)

        keypoint = torch.cat([kp_w_score, kp1_w_score, kp2_w_score], dim=1) #(b, 3k, 3)
        tf_keypoint = torch.cat([tf_kp_w_score, tf_kp1_w_score, tf_kp2_w_score], dim=1)

        keypoint = keypoint.float()
        tf_keypoint = tf_keypoint.float()

        keypoint = self.nms_fast(keypoint)
        tf_keypoint = self.nms_fast(tf_keypoint)

        return Dk, keypoint, get_zeta, tf_Dk, tf_keypoint

class noTF_DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self):
        super(noTF_DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, cur_batch):
        _, inp_channel, img_height, img_width = combined_hm_preds.shape
        keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()

        R_k = combined_hm_preds  # scoremap
        #map_val_all = R_k
        Dk_min = torch.min(torch.min(R_k, dim=2)[0], dim=2)[0]
        Dk_max = torch.max(torch.max(R_k, dim=2)[0], dim=2)[0]
        my_max_min = torch.cat([Dk_min.unsqueeze(2), Dk_max.unsqueeze(2)], dim=2) #(b,k,2) 2: min, max
        map_val_all = (R_k - (my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3)))/(my_max_min[:, :, 1].unsqueeze(2).unsqueeze(3))

        get_zeta = map_val_all.sum([2, 3])  # (b, k)

        get_kp_x = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()  # (b, k)
        get_kp_y = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()  # (b, k)

        for i in range(img_height):
            for j in range(img_width):
                cur_val = map_val_all[:, :, i, j]
                get_kp_x = get_kp_x + j * cur_val  # (b,k)
                get_kp_y = get_kp_y + i * cur_val  # (b,k)

        R_k_shape_0 = R_k.shape[0]
        for b in range(R_k_shape_0):
            for k in range(inp_channel):
                keypoint[b, k, 0] = int(torch.round((get_kp_x[b, k] / get_zeta[b, k])))
                keypoint[b, k, 1] = int(torch.round((get_kp_y[b, k] / get_zeta[b, k])))

        return map_val_all, keypoint, get_zeta

class modified_DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self):
        super(modified_DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, tf_combined_hm_preds, cur_batch):
        _, inp_channel, img_height, img_width = combined_hm_preds.shape
        #keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()
        #tf_keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()

        R_k = combined_hm_preds #scoremap
        tf_R_k = tf_combined_hm_preds #scoremap (transformed)

        softmax = torch.nn.Softmax(dim=1)
        map_val_all = softmax(R_k) #detection map
        tf_map_val_all = softmax(tf_R_k) #detection map (transformed)

        #map_val_all = torch.abs(R_k)
        #tf_map_val_all = torch.abs(tf_R_k)

        get_zeta = map_val_all.sum([2, 3]) #(b, k)
        #tf_get_zeta = tf_map_val_all.sum([2, 3]) #(b, k)

        find_col, indices_C = torch.max(map_val_all, dim=2)
        my_col = torch.argmax(find_col, dim=2)
        find_row, indices_R = torch.max(map_val_all, dim=3)
        my_row = torch.argmax(find_row, dim=2)
        keypoint = torch.cat([my_row.unsqueeze(2), my_col.unsqueeze(2)], dim=2)

        find_col_tf, indices_C_tf = torch.max(tf_map_val_all, dim=2)
        my_col_tf = torch.argmax(find_col_tf, dim=2)
        find_row_tf, indices_R_tf = torch.max(tf_map_val_all, dim=3)
        my_row_tf = torch.argmax(find_row_tf, dim=2)
        tf_keypoint = torch.cat([my_row_tf.unsqueeze(2), my_col_tf.unsqueeze(2)], dim=2)

        keypoint = keypoint.to(torch.float)
        tf_keypoint = tf_keypoint.to(torch.float)

        return map_val_all, keypoint, get_zeta, tf_keypoint

class noTF_maxKP_DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self):
        super(noTF_maxKP_DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, cur_batch):
        _, inp_channel, img_height, img_width = combined_hm_preds.shape

        R_k = combined_hm_preds  # scoremap

        #Dk_min = torch.min(torch.min(R_k, dim=2)[0], dim=2)[0]
        #Dk_max = torch.max(torch.max(R_k, dim=2)[0], dim=2)[0]
        #my_max_min = torch.cat([Dk_min.unsqueeze(2), Dk_max.unsqueeze(2)], dim=2) #(b,k,2) 2: min, max
        #map_val_all = (R_k - (my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3)))/(my_max_min[:, :, 1].unsqueeze(2).unsqueeze(3))

        #Dk_min = torch.min(torch.min(torch.abs(R_k), dim=2)[0], dim=2)[0]
        #Dk_max = torch.max(torch.max(torch.abs(R_k), dim=2)[0], dim=2)[0]
        #my_max_min = torch.cat([Dk_min.unsqueeze(2), Dk_max.unsqueeze(2)], dim=2) #(b,k,2) 2: min, max
        #map_val_all = (torch.abs(R_k) - (my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3)))/(my_max_min[:, :, 1].unsqueeze(2).unsqueeze(3))

        map_val_all = F.normalize(R_k, p=2, dim=1)

        get_zeta = map_val_all.sum([2, 3])  # (b, k)

        my_col = torch.argmax(torch.max(map_val_all, dim=2)[0], dim=2)
        my_row = torch.argmax(torch.max(map_val_all, dim=3)[0], dim=2)
        keypoint = torch.cat([my_row.unsqueeze(2), my_col.unsqueeze(2)], dim=2)

        keypoint = keypoint.to(torch.float)

        return map_val_all, keypoint, get_zeta

class GCNv2_extractKP(nn.Module):
    def __init__(self):
        super(GCNv2_extractKP, self).__init__()

    def forward(self, combined_hm_preds, tf_combined_hm_preds, num_of_kp):
        cur_batch, _, img_height, img_width = combined_hm_preds.shape

        O_k = combined_hm_preds.squeeze(1) #confidence map (b,1,h,w) -> (b,hmw)
        tf_O_k = tf_combined_hm_preds.squeeze(1) #confidence map (b,1,h,w)-> (b,hmw)

        get_zeta = O_k.sum([1, 2])  # (b)
        tf_get_zeta = tf_O_k.sum([1, 2])  # (b)

        get_kp_x = torch.zeros(cur_batch).cuda()
        get_kp_y = torch.zeros(cur_batch).cuda()
        tf_get_kp_x = torch.zeros(cur_batch).cuda()
        tf_get_kp_y = torch.zeros(cur_batch).cuda()

        keypoint = torch.ones(cur_batch, num_of_kp, 2).cuda()
        tf_keypoint = torch.ones(cur_batch, num_of_kp, 2).cuda()

        for i in range(img_height):
            for j in range(img_width):
                cur_val = O_k[:, i, j]
                get_kp_x = get_kp_x + j * cur_val  # (b,k)
                get_kp_y = get_kp_y + i * cur_val  # (b,k)

                tf_cur_val = tf_O_k[:, i, j]
                tf_get_kp_x = tf_get_kp_x + j * tf_cur_val  # (b,k)
                tf_get_kp_y = tf_get_kp_y + i * tf_cur_val  # (b,k)

        for b in range(cur_batch):
            for k in range(num_of_kp):
                keypoint[b, k, 0] = int(torch.round((get_kp_x[b, k] / get_zeta[b, k])))
                keypoint[b, k, 1] = int(torch.round((get_kp_y[b, k] / get_zeta[b, k])))

                tf_keypoint[b, k, 0] = int(torch.round((tf_get_kp_x[b, k] / tf_get_zeta[b, k])))
                tf_keypoint[b, k, 1] = int(torch.round((tf_get_kp_y[b, k] / tf_get_zeta[b, k])))


class YesTF_GradKP_DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self):
        super(YesTF_GradKP_DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, tf_combined_hm_preds):
        _, inp_channel, img_height, img_width = combined_hm_preds.shape

        R_k = combined_hm_preds  # scoremap
        tf_R_k = tf_combined_hm_preds  # scoremap (transformed)

        Dk_min = torch.min(torch.min(R_k, dim=2)[0], dim=2)[0]
        Dk_max = torch.max(torch.max(R_k, dim=2)[0], dim=2)[0]
        my_max_min = torch.cat([Dk_min.unsqueeze(2), Dk_max.unsqueeze(2)], dim=2) #(b,k,2) 2: min, max
        map_val_all = (R_k - (my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3)))/((my_max_min[:, :, 1]-my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))

        tf_Dk_min = torch.min(torch.min(tf_R_k, dim=2)[0], dim=2)[0]
        tf_Dk_max = torch.max(torch.max(tf_R_k, dim=2)[0], dim=2)[0]
        tf_my_max_min = torch.cat([tf_Dk_min.unsqueeze(2), tf_Dk_max.unsqueeze(2)], dim=2) #(b,k,2) 2: min, max
        tf_map_val_all = (tf_R_k - (tf_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3)))/((tf_my_max_min[:, :, 1]-tf_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))

        get_zeta = map_val_all.sum([2, 3])  # (b, k)
        tf_get_zeta = tf_map_val_all.sum([2, 3])  # (b, k)

        get_kp_x = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()  # (b, k)
        get_kp_y = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()  # (b, k)
        tf_get_kp_x = torch.zeros(tf_R_k.shape[0], tf_R_k.shape[1]).cuda()  # (b, k)
        tf_get_kp_y = torch.zeros(tf_R_k.shape[0], tf_R_k.shape[1]).cuda()  # (b, k)

        cur_batch = combined_hm_preds.shape[0]
        keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()
        tf_keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()

        for i in range(img_height):
            for j in range(img_width):
                cur_val = map_val_all[:, :, i, j]
                get_kp_x = get_kp_x + j * cur_val  # (b,k)
                get_kp_y = get_kp_y + i * cur_val  # (b,k)

                tf_cur_val = tf_map_val_all[:, :, i, j]
                tf_get_kp_x = tf_get_kp_x + j * tf_cur_val  # (b,k)
                tf_get_kp_y = tf_get_kp_y + i * tf_cur_val  # (b,k)
        '''
        my_kp = torch.cat((get_kp_x.unsqueeze(2), get_kp_y.unsqueeze(2)), 2)
        keypoint = torch.round(my_kp * (1/get_zeta).unsqueeze(2))

        tf_my_kp = torch.cat((tf_get_kp_x.unsqueeze(2), tf_get_kp_y.unsqueeze(2)), 2)
        tf_keypoint = torch.round(tf_my_kp * (1/tf_get_zeta).unsqueeze(2))

        '''

        R_k_shape_0 = R_k.shape[0]
        for b in range(R_k_shape_0):
            for k in range(inp_channel):
                keypoint[b, k, 0] = int(torch.round((get_kp_x[b, k] / get_zeta[b, k])))
                keypoint[b, k, 1] = int(torch.round((get_kp_y[b, k] / get_zeta[b, k])))

                tf_keypoint[b, k, 0] = int(torch.round((tf_get_kp_x[b, k] / tf_get_zeta[b, k])))
                tf_keypoint[b, k, 1] = int(torch.round((tf_get_kp_y[b, k] / tf_get_zeta[b, k])))


        return map_val_all, keypoint, get_zeta, tf_map_val_all, tf_keypoint

class DetectionConfidenceMap2keypoint_test(nn.Module):
    def __init__(self):
        super(DetectionConfidenceMap2keypoint_test, self).__init__()

    def forward(self, combined_hm_preds, cur_batch):
        start_ = time.time()
        _, inp_channel, img_height, img_width = combined_hm_preds.shape
        keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()
        tf_keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()

        R_k = combined_hm_preds #scoremap

        softmax = torch.nn.Softmax(dim=1)
        map_val_all = softmax(R_k) #detection map

        get_zeta = R_k.sum([2, 3]) #(b, k)
        get_kp_x = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()#(b, k)
        get_kp_y = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()#(b, k)

        for i in range(img_height):
            for j in range(img_width):
                cur_val = map_val_all[:, :, i, j]
                get_kp_x = get_kp_x + j * cur_val #(b,k)
                get_kp_y = get_kp_y + i * cur_val #(b,k)

        R_k_shape_0 = R_k.shape[0]
        for b in range(R_k_shape_0):
            for k in range(inp_channel):
                keypoint[b, k, 0] = int(torch.round((get_kp_x[b, k] / get_zeta[b, k])))
                keypoint[b, k, 1] = int(torch.round((get_kp_y[b, k] / get_zeta[b, k])))

        return map_val_all, keypoint, get_zeta

class ReconDetectionConfidenceMap(nn.Module):
    def __init__(self):
        super(ReconDetectionConfidenceMap, self).__init__()

        self.epsilon_scale = 1e-5
        self.correlation = 0.3
        self.reconDetectionMap_std_x = 5
        self.reconDetectionMap_std_y = 5

        self.cov = torch.zeros(2, 2)
        self.cov[0, 0] = self.reconDetectionMap_std_x ** 2
        self.cov[0, 1] = self.correlation * self.reconDetectionMap_std_x * self.reconDetectionMap_std_y
        self.cov[1, 0] = self.cov[0, 1]
        self.cov[1, 1] = self.reconDetectionMap_std_y ** 2

        self.det_cov = np.linalg.det(self.cov)
        self.inv_cov = np.linalg.inv(self.cov)

        self.inv_std_xx = 1 / self.reconDetectionMap_std_x / self.reconDetectionMap_std_x
        self.inv_std_yy = 1 / self.reconDetectionMap_std_y / self.reconDetectionMap_std_y
        self.reconDetectionMap_std_xy = self.reconDetectionMap_std_x * self.reconDetectionMap_std_y
        self.inv_reconDetectionMap_std_xy = 1/self.reconDetectionMap_std_xy

        self.gaussian_distribution_den = 1 / np.sqrt((2 * np.pi)**2 * self.det_cov)
        self.gaussian_distribution_num_1 = -1 / (2 * (1 - (self.correlation ** 2))) #scalar

    def forward(self, keypoints, DetectionMap):
        reconScoreMap = torch.ones(DetectionMap.shape[0], DetectionMap.shape[1], DetectionMap.shape[2], DetectionMap.shape[3]).cuda()
        _, keypoints_num_sz, _ = keypoints.shape
        _, _, DetectionMap_sz_2, DetectionMap_sz_3 = DetectionMap.shape

        #i, j = np.mgrid[0:DetectionMap_sz_2:1, 0:DetectionMap_sz_3:1]
        #data_ij = torch.tensor(np.dstack((i, j)))  # (64,208,2)
        #data_ij = data_ij.view(data_ij.shape[0]*data_ij.shape[1], data_ij.shape[2]) #(64*208=13312, 2)
        #data_ij = data_ij.unsqueeze(0).unsqueeze(0)
        #my_mu = keypoints.unsqueeze(2).detach().cpu().clone().numpy()
        #my_test = data_ij - my_mu

        #my_gaussian = torch.distributions.multivariate_normal.MultivariateNormal(loc=keypoints.detach().cpu().clone().numpy(), covariance_matrix=self.cov)
        #my_gaussian.sample() #(6,200,2)
        for i in range(DetectionMap_sz_2): #height (y)
            for j in range(DetectionMap_sz_3): #width (x)
                #cur_y = (i - keypoints[:, :, 1]).unsqueeze(2)
                #cur_x = (j - keypoints[:, :, 0]).unsqueeze(2)
                #cur = torch.cat((cur_x, cur_y), 2)
                #reconScoreMap[:, :, i, j] = self.gaussian_distribution_den * torch.exp(-0.5 * (((cur[:, :, 0] ** 2) * self.cov[0, 0] ) + ((cur[:, :, 1] ** 2) * self.cov[1, 1] ) + (2 * cur[:, :, 0] * cur[:, :, 1] * self.cov[0, 1])))
                gaussian_distribution_num_2 = ((i - keypoints[:, :, 1])**2 * self.inv_std_yy) + ((j - keypoints[:, :, 0])**2 * self.inv_std_xx) - (2*self.correlation*(i - keypoints[:, :, 1])*(j - keypoints[:, :, 0])*self.inv_reconDetectionMap_std_xy)
                reconScoreMap[:, :, i, j] = self.gaussian_distribution_den * torch.exp(self.gaussian_distribution_num_1 * gaussian_distribution_num_2)

        reconDetectionMap_denum = reconScoreMap.sum(1) #(b, 96, 128)
        reconDetectionMap = reconScoreMap /(reconDetectionMap_denum.unsqueeze(1) + self.epsilon_scale)

        return reconDetectionMap

class create_softmask (nn.Module):
    def __init__(self):
        super(create_softmask, self).__init__()

    def forward(self, DetectionMap, zeta):
        softmask = torch.zeros_like(DetectionMap)
        #detectionmap_sz_0, detectionmap_sz_1, _, _ = DetectionMap.shape
        """
        for b in range(detectionmap_sz_0):
            for k in range(detectionmap_sz_1):
                softmask[b, k, :, :] = DetectionMap[b, k, :, :] / zeta[b, k]
        """
        softmask = DetectionMap / zeta.unsqueeze(2).unsqueeze(3)

        return softmask

class ReconDetectionMapWithKP_3kp(nn.Module):
    def __init__(self, img_width, img_height, num_of_kp):
        super(ReconDetectionMapWithKP_3kp, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_of_kp = num_of_kp

        self.linear_2_16 = torch.nn.Linear(2, 16)
        self.linear_16_64 = torch.nn.Linear(16, 64)
        self.linear_64_256 = torch.nn.Linear(64, 256)
        self.linear_256_1024 = torch.nn.Linear(256, 1024)
        self.linear_1024_4096 = torch.nn.Linear(1024, 4096)
        self.linear_4096_end = torch.nn.Linear(4096, self.img_width * self.img_height)

        self.linear_300_256 = torch.nn.Linear(num_of_kp*3, 256)
        self.linear_256_kpnum = torch.nn.Linear(256, num_of_kp)

        '''
        self.pre = nn.Sequential(
            #Conv(2, 64, 3, 1, bn=True, relu=True),
            #Conv(64, 1024, 3, 1, bn=True, relu=True),
            Conv(200, 1024, 3, 1, bn=True, relu=True),
            Conv(1024, 4096, 3, 1, bn=True, relu=True),
            Residual(1024, 4096),
            ##Pool(2, 2),
            #Residual(4096, 4096),
            Residual(4096, img_width * img_height)
        )
        '''
    def forward(self, keypoints):
        out = self.linear_2_16(keypoints)
        #out = F.relu(out)
        out = self.linear_16_64(out)
        #out = F.relu(out)
        out = self.linear_64_256(out)
        #out = F.relu(out)
        out = self.linear_256_1024(out)
        #out = F.relu(out)
        out = self.linear_1024_4096(out)
        #out = F.relu(out)
        out = self.linear_4096_end(out)
        out = out.permute(0, 2, 1)

        out = self.linear_300_256(out)
        out = self.linear_256_kpnum(out)
        out = out.permute(0, 2, 1)
        out = out.cuda()
        #out = self.pre(keypoints)

        out = out.view(keypoints.shape[0], self.num_of_kp, self.img_height, self.img_width) #(b,300,h,2)

        return out

class ReconDetectionMapWithKP(nn.Module):
    def __init__(self, img_width, img_height, num_of_kp):
        super(ReconDetectionMapWithKP, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_of_kp = num_of_kp

        self.linear_2_16 = torch.nn.Linear(2, 16)
        self.linear_16_64 = torch.nn.Linear(16, 64)
        self.linear_64_256 = torch.nn.Linear(64, 256)
        self.linear_256_1024 = torch.nn.Linear(256, 1024)
        self.linear_1024_4096 = torch.nn.Linear(1024, 4096)
        self.linear_4096_end = torch.nn.Linear(4096, self.img_width * self.img_height)

    def forward(self, keypoints):
        out = F.relu(self.linear_2_16(keypoints))
        out = F.relu(self.linear_16_64(out))
        out = F.relu(self.linear_64_256(out))
        out = F.relu(self.linear_256_1024(out))
        out = F.relu(self.linear_1024_4096(out))
        out = F.relu(self.linear_4096_end(out))
        out = out.view(keypoints.shape[0], self.num_of_kp, self.img_height, self.img_width) #(b,300,h,2)

        return out

class ReconDetectionMapWithKP_Res(nn.Module):
    def __init__(self, img_width, img_height, num_of_kp):
        super(ReconDetectionMapWithKP_Res, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_of_kp = num_of_kp

        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(self.num_of_kp, 32*self.num_of_kp) #num_of_kp, 32*num_of_kp

        #self.net = ResNet(Bottleneck, [3, 4, 23, 3])
        self.net = ResNet(BasicBlock, [3, 4, 6, 3])

    def forward(self, keypoints):
        x = F.relu(self.linear1(keypoints)) #(b,k,3)
        x = x.permute(0, 2, 1).unsqueeze(3) #(b,3,k)
        #x = x.permute(0, 2, 1) #(b,3,k)
        #x = F.relu(self.linear2(x)).unsqueeze(3) #(b,3,32k,1)

        out = self.net(x, self.img_height, self.img_width, keypoints.shape[1])

        return out


class ReconDetectionMapWithF(nn.Module):
    def __init__(self, img_width, img_height, feature_dimension):
        super(ReconDetectionMapWithF, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.fd = feature_dimension
        self.linear_fd_256 = torch.nn.Linear(self.fd, 256)
        self.linear_256_256 = torch.nn.Linear(256, 256)
        self.linear_256_1024 = torch.nn.Linear(256, 1024)
        self.linear_1024_4096 = torch.nn.Linear(1024, 4096)
        self.linear_4096_end = torch.nn.Linear(4096, self.img_width * self.img_height)

    def forward(self, fk):
        #fk (b,k,f=256)

        out = self.linear_fd_256(fk)
        out = F.relu(out)
        out = self.linear_256_256(out)
        out = F.relu(out)
        out = self.linear_256_1024(out)
        out = F.relu(out)
        out = self.linear_1024_4096(out)
        out = F.relu(out)
        out = self.linear_4096_end(out)

        out = out.view(fk.shape[0], fk.shape[1], self.img_height, self.img_width)

        return out


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)

        return maxima.nonzero().t()[2:4]

class extract_multiscale(nn.Module):
    def __init__(self):
        super(extract_multiscale, self).__init__()

    def forward(self, img, scale_f=2 ** 0.25, min_scale=0.0, max_scale=1, min_size=256, max_size=1024, verbose=False):

        old_bm = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # speedup

        # extract keypoints at multiple scales
        B, three, H, W = img.shape
        assert B == 1 and three == 3, "should be a batch with a single RGB image"

        assert max_scale <= 1
        s = 1.0  # current scale factor

        X, Y, S, C, Q, D = [], [], [], [], [], []
        while s + 0.001 >= max(min_scale, min_size / max(H, W)):
            if s - 0.001 <= min(max_scale, max_size / max(H, W)):
                nh, nw = img.shape[2:]
                if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
                # extract descriptors
                with torch.no_grad():
                    res = L2net_R2D2(img)

                # get output and reliability map
                descriptors = res['descriptors'][0]
                reliability = res['reliability'][0]
                repeatability = res['repeatability'][0]

                # normalize the reliability for nms
                # extract maxima and descs
                y, x = detector(**res)  # nms
                c = reliability[0, 0, y, x]
                q = repeatability[0, 0, y, x]
                d = descriptors[0, :, y, x].t()
                n = d.shape[0]

                # accumulate multiple scales
                X.append(x.float() * W / nw)
                Y.append(y.float() * H / nh)
                S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
                C.append(c)
                Q.append(q)
                D.append(d)
            s /= scale_f

            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

        # restore value
        torch.backends.cudnn.benchmark = old_bm

        Y = torch.cat(Y)
        X = torch.cat(X)
        S = torch.cat(S)  # scale
        scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
        XYS = torch.stack([X, Y, S], dim=-1)
        D = torch.cat(D)
        return XYS, D, scores


class extract_keypoints(nn.Module):
    def __init__(self):
        super(extract_keypoints, self).__init__()

    def forward(self, MyHeatMap):

        detector = NonMaxSuppression(MyHeatMap)

        while args.images:
            img_path = args.images.pop(0)

            if img_path.endswith('.txt'):
                args.images = open(img_path).read().splitlines() + args.images
                continue

            print(f"\nExtracting features for {img_path}")
            img = Image.open(img_path).convert('RGB')
            W, H = img.size
            img = norm_RGB(img)[None]
            if iscuda: img = img.cuda()

            # extract keypoints/descriptors for a single image
            xys, desc, scores = extract_multiscale(net, img, detector,scale_f=args.scale_f,min_scale=args.min_scale,max_scale=args.max_scale,min_size=args.min_size,max_size=args.max_size,verbose=True)

            xys = xys.cpu().numpy()
            desc = desc.cpu().numpy()
            scores = scores.cpu().numpy()
            idxs = scores.argsort()[-args.top_k or None:]

            outpath = img_path + '.' + args.tag
            print(f"Saving {len(idxs)} keypoints to {outpath}")
            np.savez(open(outpath, 'wb'),imsize=(W, H),keypoints=xys[idxs],descriptors=desc[idxs],scores=scores[idxs])