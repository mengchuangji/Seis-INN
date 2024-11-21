import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import h5py as h5
from . import BaseDataSetH5


class GTRNDatasetSeis(BaseDataSetH5):
    def __init__(self, opt, h5_file, length, pch_size=(128,128)):
        super(GTRNDatasetSeis, self).__init__(h5_file, length)
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_GT, self.paths_Noisy =None, None
        self.random_scale_list = [1]
        self.pch_size = pch_size


    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)
        GT_path, Noisy_path= None, None
        GT_size = self.opt['GT_size']

        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            H, W, C2 = imgs_sets.shape
            C = int(C2 / 2)


            # get GT image
            # GT_path = self.paths_GT[index] #如果是切分好的
            img_index= str(index) # 命名为img的name
            # img_GT= np.array(imgs_sets[:, :, :C])
            if self.opt['phase'] == 'train':
                img_GT, img_Noisy = self.crop_patch(imgs_sets)
            else:
                img_Noisy = np.array(imgs_sets[:, :, :C])
                img_GT = np.array(imgs_sets[:, :, C:])


            # get Noisy image
            # Noisy_path = self.paths_Noisy[index]
            # img_Noisy = np.array(imgs_sets[:, :, C:])

        #  data augmentation
        # if self.opt['phase'] == 'train':
        #     img_GT, img_Noisy = util.random_augmentation(img_GT, img_Noisy)



        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT.transpose((2, 0, 1)))).float()
        img_Noisy = torch.from_numpy(np.ascontiguousarray(img_Noisy.transpose((2, 0, 1)))).float()

        # return im_noisy, im_gt
        return  {'Noisy':img_Noisy, 'GT': img_GT, 'img_index': img_index}


class GTRNDatasetSeisTest(BaseDataSetH5):

    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            H, W, C2 = imgs_sets.shape
            C = int(C2 / 2)
            img_index= str(index) # 命名为img的name
            img_Noisy = np.array(imgs_sets[:, :, :C])
            img_GT = np.array(imgs_sets[:, :, C:])

        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT.transpose((2, 0, 1)))).float()
        img_Noisy = torch.from_numpy(np.ascontiguousarray(img_Noisy.transpose((2, 0, 1)))).float()
        # return im_noisy, im_gt
        return  {'Noisy':img_Noisy, 'GT': img_GT, 'img_index': img_index}



