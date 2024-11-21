import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import h5py as h5
from . import BaseDataSetImg, BaseDataSetH5


class LQGTSNDatasetSeis(BaseDataSetImg):
    def __init__(self, opt, im_list, length, pch_size=(128,128), noiseL_B=(0,75)):
        super(LQGTSNDatasetSeis, self).__init__(im_list, length)
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT, self.paths_Noisy = None, None, None
        self.random_scale_list = [1]
        self.pch_size = pch_size
        self.noiseL_B = noiseL_B #[0, 75]


    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)
        GT_path, Noisy_path, LQ_path = None, None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        # GT_path = self.paths_GT[index] #如果是切分好的
        im_ori = self.im_list[ind_im]
        img_GT = self.crop_patch(im_ori)  # mcj

        C = img_GT.shape[2]

        img_index = str(index)  # 命名为img的name

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)


        #  data augmentation
        if self.opt['phase'] == 'train':
            img_GT = util.random_augmentation(img_GT)[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, C = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if C==1:
                    img_GT=img_GT[:,:,np.newaxis]


            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)
            #
            # H, W, C = img_LQ.shape
            # LQ_size = GT_size // scale
            #
            # # randomly crop
            # rnd_h = random.randint(0, max(0, H - LQ_size))
            # rnd_w = random.randint(0, max(0, W - LQ_size))
            # img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            # rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            # img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            #
            # # augmentation - flip, rotate
            # img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
            #                               self.opt['use_rot'])
            if self.opt['noise_mode'] == 'S':
                noise = np.random.normal(0, self.opt['sigma'] / 255, img_GT.shape)
            if self.opt['noise_mode'] == 'B':
                # noiseL_B = [0, 75]
                stdN = np.random.uniform(self.noiseL_B[0], self.noiseL_B[1])
                noise = np.random.normal(0, stdN / 255, img_GT.shape)
            if self.opt['noise_mode'] == 'GB':
                # generate sigmaMap
                sigma_map = self.generate_sigma()
                # generate noise
                noise = torch.randn(img_GT.shape).numpy() * sigma_map
            img_Noisy = img_GT + noise


        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT.transpose((2, 0, 1)))).float()
        img_Noisy = torch.from_numpy(np.ascontiguousarray(img_Noisy.transpose((2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ.transpose((2, 0, 1)))).float()

        # return im_noisy, im_gt
        return  {'LQ': img_LQ, 'Noisy':img_Noisy, 'GT': img_GT, 'img_index': img_index}

    def generate_sigma(self):
        pch_size = self.pch_size[0]
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size/4, pch_size/4*3)
        kernel = util.gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(self.noiseL_B[0]/255.0, self.noiseL_B[1]/255.0)
        down = random.uniform(self.noiseL_B[0]/255.0, self.noiseL_B[1]/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min()) *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

class LQGTSNDatasetSeis_CL_NE(BaseDataSetH5):
    def __init__(self, opt, CL_h5_file, length, NE_h5_file, pch_size=(128,128), noiseL_B=(0,75)):
        super(LQGTSNDatasetSeis_CL_NE, self).__init__(CL_h5_file, length)
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT, self.paths_Noisy = None, None, None
        self.random_scale_list = [1]
        self.pch_size = pch_size
        self.noiseL_B = noiseL_B #[0, 75]
        self.NE_h5_file = NE_h5_file
        with h5.File(self.NE_h5_file, 'r') as h5_file:
            self.NE_keys = list(h5_file.keys())
            self.NE_num_images = len(self.NE_keys)

    def crop_patch(self, img):
        H, W, C = img.shape
        ind_H = random.randint(0, H-self.pch_size[0])
        ind_W = random.randint(0, W - self.pch_size[0])
        img = np.array(img[ind_H:ind_H + self.pch_size[0], ind_W:ind_W + self.pch_size[0], :])
        return img

    def __getitem__(self, index):
        num_images = self.num_images
        ind_gt = random.randint(0, num_images-1)
        ne_num_images = self.NE_num_images
        ind_ne = random.randint(0, ne_num_images - 1)

        GT_path, Noisy_path, LQ_path = None, None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        # GT_path = self.paths_GT[index] #如果是切分好的
        with h5.File(self.h5_path, 'r') as h5_file:
            im_gt= np.array(h5_file[self.keys[ind_gt]])
        img_GT = self.crop_patch(im_gt)  # mcj

        with h5.File(self.NE_h5_file, 'r') as h5_file:
            im_noise = np.array(h5_file[self.NE_keys[ind_ne]])
        im_noise = self.crop_patch(im_noise)
        im_noise_max = abs(im_noise).max()


        C = img_GT.shape[2]

        img_index = str(index)  # 命名为img的name

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)


        #  data augmentation
        if self.opt['phase'] == 'train':
            img_GT = util.random_augmentation(img_GT)[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, C = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if C==1:
                    img_GT=img_GT[:,:,np.newaxis]


            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)
            #
            # H, W, C = img_LQ.shape
            # LQ_size = GT_size // scale
            #
            # # randomly crop
            # rnd_h = random.randint(0, max(0, H - LQ_size))
            # rnd_w = random.randint(0, max(0, W - LQ_size))
            # img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            # rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            # img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            #
            # # augmentation - flip, rotate
            # img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
            #                               self.opt['use_rot'])
            if self.opt['noise_mode'] == 'S':
                noise = np.random.normal(0, self.opt['sigma'] / 255, img_GT.shape)
            if self.opt['noise_mode'] == 'B':
                # noiseL_B = [0, 75]
                stdN = np.random.uniform(self.noiseL_B[0], self.noiseL_B[1])
                noise = np.random.normal(0, stdN / 255, img_GT.shape)
            if self.opt['noise_mode'] == 'GB':
                # generate sigmaMap
                sigma_map = self.generate_sigma()
                # generate noise
                noise = torch.randn(img_GT.shape).numpy() * sigma_map
            if self.opt['noise_mode'] == 'SYN_S':
                noise = (im_noise / im_noise_max) * (self.noiseL_B[1] / 255)
            if self.opt['noise_mode'] == 'SYN_B':
                stdN = np.random.uniform(self.noiseL_B[0], self.noiseL_B[1])
                noise = (im_noise / im_noise_max) * (stdN / 255)
            if self.opt['noise_mode'] == 'SYN_GB':
                # generate sigmaMap
                sigma_map = self.generate_sigma()
                # generate noise
                noise = (im_noise / im_noise_max) * sigma_map
            img_Noisy = img_GT + noise


        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT.transpose((2, 0, 1)))).float()
        img_Noisy = torch.from_numpy(np.ascontiguousarray(img_Noisy.transpose((2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ.transpose((2, 0, 1)))).float()

        # return im_noisy, im_gt
        return  {'LQ': img_LQ, 'Noisy':img_Noisy, 'GT': img_GT, 'img_index': img_index}

    def generate_sigma(self):
        pch_size = self.pch_size[0]
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size/4, pch_size/4*3)
        kernel = util.gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(self.noiseL_B[0]/255.0, self.noiseL_B[1]/255.0)
        down = random.uniform(self.noiseL_B[0]/255.0, self.noiseL_B[1]/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min()) *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]


