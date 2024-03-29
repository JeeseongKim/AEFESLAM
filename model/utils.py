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
cv2.ocl.setUseOpenCL(False)
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

class make_transformation_M(nn.Module):
    def __init__(self):
        super(make_transformation_M, self).__init__()

    def forward(self, r_theta, t_x, t_y):
        radian_theta = r_theta * 3.141592/180   #degree
        dx = random.uniform(-t_x, t_x)
        dy = random.uniform(-t_y, t_y)

        transformation_g = torch.zeros(4, 4)
        transformation_g[0, 0] = math.cos(radian_theta)
        transformation_g[0, 1] = -math.sin(radian_theta)
        transformation_g[0, 3] = dx

        transformation_g[1, 0] = math.sin(radian_theta)
        transformation_g[1, 1] = math.cos(radian_theta)
        transformation_g[1, 3] = dy

        transformation_g[2, 2] = 1.0

        transformation_g[3, 3] = 1.0

        return transformation_g

def get_kp (H, my_height, my_width):

    #kp = torch.sigmoid(Hk_kp)
    kp = 1/(1+torch.exp(-1*H))
    kp[:, :, 0] = torch.round(kp[:, :, 0] * my_width).float()
    kp[:, :, 1] = torch.round(kp[:, :, 1] * my_height).float()

    return kp



class my_dataset_originalImg(Dataset):
    #def __init__(self, transform=None):
    def __init__(self, my_width, my_height):
        self.dataset_img = []
        self.dataset_filename = []
        self.kp_img = []
        self.input_height = my_height
        self.input_width = my_width
        #lg dataset(96,128))

        #for filename in (sorted(glob.glob('./dataset/Kitti/sequences/00/image_2/*.png'))):
        #for filename in (sorted(glob.glob('./dataset/Kitti/sequences/05/image_2/*.png'))):
        for filename in (sorted(glob.glob('./dataset/Kitti/oxbuild_images/*.jpg'))):
        #for filename in (sorted(glob.glob('./dataset/Kitti_tmp/sequences/05/image_2/*.png'))):
        #for filename in (sorted(glob.glob('./dataset/LG/*.jpg'))):
            im = Image.open(filename)

            img_rsz = cv2.resize(np.array(im), (self.input_width, self.input_height)) #opencv image: (h,w,C), tensor image: (c, h, w)
            img_tensor_input = transforms.ToTensor()(img_rsz)# (3,192,256)
            self.dataset_img.append(img_tensor_input)

            #self.dataset_filename.append(filename.split('.')[1].split('/')[2])
            self.dataset_filename.append(filename.split('/')[5].split('.')[0])
            self.kp_img.append(img_rsz)

        self.len = len(self.dataset_img)
        a = torch.utils.data.get_worker_info()

    def __getitem__(self, index):
        return self.dataset_img[index], self.dataset_filename[index], self.kp_img[index]

    def __len__(self):
        return len(self.dataset_img)

class my_dataset(Dataset):
    #def __init__(self, transform=None):
    def __init__(self, my_width, my_height):
        self.dataset_img = []
        self.dataset_filename = []
        self.kp_img = []
        self.input_height = my_height
        self.input_width = my_width
        #lg dataset(96,128))

        #for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/Kitti/sequences/00/image_2/*.png'))):
        #for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/Kitti/sequences/05/image_2/*.png'))):
        #for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/Kitti_tmp/sequences/05/image_2/*.png'))):
        #for filename in (sorted(glob.glob('./data/*.jpg'))):
        #for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/oxbuild_images/*.jpg'))):
        #for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/oxbuild_tmp/*.jpg'))):
        #for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/oxbuild_part/*.jpg'))):
        for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/oxbuild_debug/*.jpg'))):
        #for filename in (sorted(glob.glob('/home/jsk/AEFE_SLAM/dataset/oxbuild_ox/*.jpg'))):
            im = Image.open(filename)
            if(im.height < im.width):
                img_rsz = cv2.resize(np.array(im), (self.input_width, self.input_height)) #opencv image: (h,w,C), tensor image: (c, h, w)
                img_tensor_input = transforms.ToTensor()(img_rsz)# (3,192,256)
                self.dataset_img.append(img_tensor_input)

                #self.dataset_filename.append(filename.split('/')[8].split('.')[0])  # KITTI
                self.dataset_filename.append(filename.split('/')[6].split('.')[0])  # oxford

                #self.dataset_filename.append(filename.split('/')[3]) #oxfor building
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
            cur_kp = keypoints[b, :, :]
            for i in range(kp_num):
                if(i==0):
                    kpimg = cv2.circle(cur_img, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
                else:
                    kpimg = cv2.circle(kpimg, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
            save_kpimg = transforms.ToTensor()(kpimg).unsqueeze(0) # (1,3,192,256)
            img_save_filename = ("/home/jsk/AEFE_SLAM/SaveKPImg/%s_ep_%s.jpg" % (cur_filename[b], epoch))
            save_image(save_kpimg, img_save_filename)


class savetfKPimg(nn.Module):
    def __init__(self):
        super(savetfKPimg, self).__init__()

    def forward(self, tf_aefe_input, tf_keypoints, epoch, cur_filename):
        batch_size, kp_num, _ = tf_keypoints.shape
        for b in range(batch_size):
            cur_img = tf_aefe_input[b, :, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            cur_img = np.ascontiguousarray(cur_img)
            cur_kp = tf_keypoints[b, :, :]
            for i in range(kp_num):
                if(i==0):
                    kpimg = cv2.circle(cur_img, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
                else:
                    kpimg = cv2.circle(kpimg, tuple(cur_kp[i, :]), 2, (255, 0, 0), -1)
            save_kpimg = transforms.ToTensor()(kpimg).unsqueeze(0) # (1,3,192,256)
            img_save_filename = ("/home/jsk/AEFE_SLAM/SavetfKPImg/%s_ep_%s.jpg" % (cur_filename[b], epoch))
            save_image(save_kpimg, img_save_filename)

class multivariate_normal(nn.Module):
    def __init__(self):
        super(multivariate_normal, self).__init__()

    def forward(self, x, d, mean, covariance):
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x