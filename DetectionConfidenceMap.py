import torch
from torch import nn
import numpy as np
import time
from model.layers import Conv, Residual
import torch.nn.functional as F
import math

class DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self):
        super(DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, tf_combined_hm_preds, cur_batch):
        _, inp_channel, img_height, img_width = combined_hm_preds.shape
        keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()
        tf_keypoint = torch.ones(cur_batch, inp_channel, 2).cuda()

        R_k = combined_hm_preds #scoremap
        tf_R_k = tf_combined_hm_preds #scoremap (transformed)

        #softmax = torch.nn.Softmax(dim=1)
        #map_val_all = softmax(R_k) #detection map
        #tf_map_val_all = softmax(tf_R_k) #detection map (transformed)

        map_val_all = torch.abs(R_k)
        tf_map_val_all = torch.abs(tf_R_k)

        get_zeta = map_val_all.sum([2, 3]) #(b, k)
        tf_get_zeta = tf_map_val_all.sum([2, 3]) #(b, k)

        get_kp_x = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()#(b, k)
        get_kp_y = torch.zeros(R_k.shape[0], R_k.shape[1]).cuda()#(b, k)
        tf_get_kp_x = torch.zeros(tf_R_k.shape[0], tf_R_k.shape[1]).cuda()#(b, k)
        tf_get_kp_y = torch.zeros(tf_R_k.shape[0], tf_R_k.shape[1]).cuda()#(b, k)

        for i in range(img_height):
            for j in range(img_width):
                cur_val = map_val_all[:, :, i, j]
                get_kp_x = get_kp_x + j * cur_val #(b,k)
                get_kp_y = get_kp_y + i * cur_val #(b,k)

                tf_cur_val = tf_map_val_all[:, :, i, j]
                tf_get_kp_x = tf_get_kp_x + j * tf_cur_val #(b,k)
                tf_get_kp_y = tf_get_kp_y + i * tf_cur_val #(b,k)
        """
        my_kp = torch.cat((get_kp_x.unsqueeze(2), get_kp_y.unsqueeze(2)), 2)
        inv_get_zeta = (1/get_zeta).unsqueeze(2)
        keypoint = torch.round(my_kp * inv_get_zeta)

        tf_my_kp = torch.cat((tf_get_kp_x.unsqueeze(2), tf_get_kp_y.unsqueeze(2)), 2)
        inv_get_tf_zeta = (1/tf_get_zeta).unsqueeze(2)
        tf_keypoint = torch.round(tf_my_kp * inv_get_tf_zeta)

        """
        R_k_shape_0 = R_k.shape[0]
        for b in range(R_k_shape_0):
            for k in range(inp_channel):
                keypoint[b, k, 0] = int(torch.round((get_kp_x[b, k] / get_zeta[b, k])))
                keypoint[b, k, 1] = int(torch.round((get_kp_y[b, k] / get_zeta[b, k])))

                tf_keypoint[b, k, 0] = int(torch.round((tf_get_kp_x[b, k] / tf_get_zeta[b, k])))
                tf_keypoint[b, k, 1] = int(torch.round((tf_get_kp_y[b, k] / tf_get_zeta[b, k])))

        return map_val_all, keypoint, get_zeta, tf_keypoint

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

class YesTF_maxKP_DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self):
        super(YesTF_maxKP_DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, tf_combined_hm_preds, tf_corner_P):
        _, inp_channel, img_height, img_width = combined_hm_preds.shape

        R_k = combined_hm_preds  # scoremap
        tf_R_k = tf_combined_hm_preds  # scoremap (transformed)

        #map_val_all = F.normalize(R_k, p=2, dim=1)
        #tf_map_val_all = F.normalize(tf_R_k, p=2, dim=1)

        Dk_min = torch.min(torch.min(R_k, dim=2)[0], dim=2)[0]
        Dk_max = torch.max(torch.max(R_k, dim=2)[0], dim=2)[0]
        my_max_min = torch.cat([Dk_min.unsqueeze(2), Dk_max.unsqueeze(2)], dim=2) #(b,k,2) 2: min, max
        map_val_all = (R_k - (my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3)))/((my_max_min[:, :, 1]-my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))

        tf_Dk_min = torch.min(torch.min(tf_R_k, dim=2)[0], dim=2)[0]
        tf_Dk_max = torch.max(torch.max(tf_R_k, dim=2)[0], dim=2)[0]
        tf_my_max_min = torch.cat([tf_Dk_min.unsqueeze(2), tf_Dk_max.unsqueeze(2)], dim=2) #(b,k,2) 2: min, max
        tf_map_val_all = (tf_R_k - (tf_my_max_min[:, :, 0].unsqueeze(2).unsqueeze(3)))/((tf_my_max_min[:, :, 1]-tf_my_max_min[:, :, 0]).unsqueeze(2).unsqueeze(3))

        get_zeta = map_val_all.sum([2, 3])  # (b, k)
        #tf_get_zeta = tf_map_val_all.sum([2, 3])  # (b, k)

        my_col = torch.argmax(torch.max(map_val_all, dim=2)[0], dim=2)
        my_row = torch.argmax(torch.max(map_val_all, dim=3)[0], dim=2)
        keypoint = torch.cat([my_col.unsqueeze(2), my_row.unsqueeze(2)], dim=2)
        keypoint = keypoint.to(torch.float)

        tf_my_col = torch.argmax(torch.max(tf_map_val_all, dim=2)[0], dim=2)
        tf_my_row = torch.argmax(torch.max(tf_map_val_all, dim=3)[0], dim=2)
        tf_keypoint = torch.cat([tf_my_col.unsqueeze(2), tf_my_row.unsqueeze(2)], dim=2)
        tf_keypoint = tf_keypoint.to(torch.float)

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
        detectionmap_sz_0, detectionmap_sz_1, _, _ = DetectionMap.shape
        """
        for b in range(detectionmap_sz_0):
            for k in range(detectionmap_sz_1):
                softmask[b, k, :, :] = DetectionMap[b, k, :, :] / zeta[b, k]
        """
        softmask = DetectionMap / zeta.unsqueeze(2).unsqueeze(3)

        return softmask

class LinearReconScoreMap(nn.Module):
    def __init__(self, img_width, img_height, num_of_kp):
        super(LinearReconScoreMap, self).__init__()
        self.linear_2_16 = torch.nn.Linear(2, 16)
        self.linear_16_64 = torch.nn.Linear(16, 64)
        self.linear_64_256 = torch.nn.Linear(64, 256)
        self.linear_256_1024 = torch.nn.Linear(256, 1024)
        self.linear_1024_4096 = torch.nn.Linear(1024, 4096)
        self.linear_4096_end = torch.nn.Linear(4096, img_width * img_height)
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
    def forward(self, keypoints, DetectionMap):
        out = self.linear_2_16(keypoints)
        out = self.linear_16_64(out)
        out = self.linear_64_256(out)
        out = self.linear_256_1024(out)
        out = self.linear_1024_4096(out)
        out = self.linear_4096_end(out)

        #out = self.pre(keypoints)

        out = out.view(DetectionMap.shape[0], DetectionMap.shape[1], DetectionMap.shape[2], DetectionMap.shape[3])

        return out