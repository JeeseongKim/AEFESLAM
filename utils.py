import torch
from torch import nn
import random
import math
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms
import torchvision
import cv2
#cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torchvision.utils import save_image

class make_transformation_M(nn.Module):
    def __init__(self):
        super(make_transformation_M, self).__init__()

    def forward(self, r_theta, t_x, t_y):
        dx = random.uniform(-t_x, t_x)
        dy = random.uniform(-t_y, t_y)
        theta = random.uniform(-r_theta, r_theta)

        transformation_g = torch.zeros(4, 4)
        transformation_g[0, 0] = math.cos(theta)
        transformation_g[0, 1] = -math.sin(theta)
        transformation_g[0, 3] = dx

        transformation_g[1, 0] = math.sin(theta)
        transformation_g[1, 1] = math.cos(theta)
        transformation_g[1, 3] = dy

        transformation_g[2, 2] = 1.0

        transformation_g[3, 3] = 1.0

        return transformation_g

class my_dataset(Dataset):
    #def __init__(self, transform=None):
    def __init__(self, my_width, my_height):
        self.dataset_img = []
        self.dataset_filename = []
        self.kp_img = []
        self.input_height = my_height
        self.input_width = my_width
        #lg dataset(96,128))

        for filename in (sorted(glob.glob('./Kitti/sequences/00/image_3/*.png'))):
        #for filename in (sorted(glob.glob('./Kitti/sequences/05/image_2/*.png'))):
        #for filename in (sorted(glob.glob('./Kitti_tmp/sequences/00/image_2/*.png'))):
        #for filename in (sorted(glob.glob('./data/*.jpg'))):
            im = Image.open(filename)

            img_rsz = cv2.resize(np.array(im), (self.input_width, self.input_height)) #opencv image: (h,w,C), tensor image: (c, h, w)
            img_tensor_input = transforms.ToTensor()(img_rsz)  # (3,192,256)
            self.dataset_img.append(img_tensor_input)

            #img_rsz_fn = torchvision.transforms.Resize((my_height, my_width), 2)
            #img_rsz = img_rsz_fn(im)
            #img_rsz = transforms.ToTensor()(img_rsz)
            #self.dataset_img.append(img_rsz)

            #self.dataset_filename.append(filename.split('.')[1].split('/')[2])
            self.dataset_filename.append(filename.split('/')[5].split('.')[0])
            self.kp_img.append(img_rsz)

        self.len = len(self.dataset_img)
        a = torch.utils.data.get_worker_info()

    def __getitem__(self, index):
        return self.dataset_img[index], self.dataset_filename[index], self.kp_img[index]

    def __len__(self):
        return len(self.dataset_img)

class saveKPimg(nn.Module):
    def __init__(self):
        super(saveKPimg, self).__init__()

    def forward(self, kp_img, keypoints, epoch, cur_filename):
        batch_size, kp_num, _ = keypoints.shape
        for b in range(batch_size):
            cur_img = kp_img[b, :, :, :].numpy()
            #cur_img = torchvision.transforms.ToPILImage()(kp_img[b, :, :, :])
            cur_kp = keypoints[b, :, :]
            for i in range(kp_num):
                if(i==0):
                    kpimg = cv2.circle(cur_img, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
                else:
                    kpimg = cv2.circle(kpimg, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
            save_kpimg = transforms.ToTensor()(kpimg).unsqueeze(0) # (1,3,192,256)
            img_save_filename = ("SaveKPImg/%s_epoch_%s.jpg" % (cur_filename[b], epoch))
            save_image(save_kpimg, img_save_filename)


class savetfKPimg(nn.Module):
    def __init__(self):
        super(savetfKPimg, self).__init__()

    def forward(self, tf_aefe_input, tf_keypoints, epoch, cur_filename):
        batch_size, kp_num, _ = tf_keypoints.shape
        for b in range(batch_size):
            tf_kpimg = torchvision.transforms.ToPILImage()(tf_aefe_input[b, :, :, :])
            n_tf_kpimg = np.array(tf_kpimg)
            #cur_img = tf_kpimg.numpy()
            #tf_kpimg = tf_aefe_input.cpu().detach().numpy()
            cur_kp = tf_keypoints[b, :, :]
            for i in range(kp_num):
                n_tf_kpimg = cv2.circle(n_tf_kpimg, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
                #if(i == 0):
                #    #tf_kpimg = cv2.circle(cur_img, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
                #    tf_kpimg = cv2.circle(tf_kpimg, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
                #else:
                #    tf_kpimg = cv2.circle(tf_kpimg, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
            save_kpimg = transforms.ToTensor()(n_tf_kpimg).unsqueeze(0) # (1,3,192,256)
            img_save_filename = ("SavetfKPImg/%s_epoch_%s.jpg" % (cur_filename[b], epoch))
            save_image(save_kpimg, img_save_filename)

class multivariate_normal(nn.Module):
    def __init__(self):
        super(multivariate_normal, self).__init__()

    def forward(self, x, d, mean, covariance):
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))