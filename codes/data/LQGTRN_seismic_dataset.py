import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import h5py as h5
from . import BaseDataSetH5


class LQGTRNDatasetSeis(BaseDataSetH5):
    def __init__(self, opt, h5_file, length, pch_size=(128,128)):
        super(LQGTRNDatasetSeis, self).__init__(h5_file, length)
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT, self.paths_Noisy = None, None, None
        self.random_scale_list = [1]
        self.pch_size = pch_size


    def __getitem__(self, index):
        num_images = self.num_images
        if self.opt['phase'] == 'train':
            ind_im = random.randint(0, num_images-1)
        else:
            ind_im = index
        GT_path, Noisy_path, LQ_path = None, None, None
        scale = self.opt['scale']
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

            # modcrop in the validation / test phase
            if self.opt['phase'] != 'train':
                img_GT = util.modcrop(img_GT, scale)

            # get Noisy image
            # Noisy_path = self.paths_Noisy[index]
            # img_Noisy = np.array(imgs_sets[:, :, C:])

            # modcrop in the validation / test phase
            if self.opt['phase'] != 'train':
                img_Noisy = util.modcrop(img_Noisy, scale)

        # #  data augmentation   self.crop_patch已经增广过
        # if self.opt['phase'] == 'train':
        #     img_GT, img_Noisy = util.random_augmentation(img_GT, img_Noisy)

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
                img_Noisy = cv2.resize(np.copy(img_Noisy), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if C==1:
                    img_GT=img_GT[:,:,np.newaxis]
                    img_Noisy=img_Noisy[:,:,np.newaxis]

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
                img_Noisy = cv2.resize(np.copy(img_Noisy), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            img_Noisy = img_Noisy[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT, img_Noisy = util.augment([img_LQ, img_GT, img_Noisy], self.opt['use_flip'],
                                          self.opt['use_rot'])


        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT.transpose((2, 0, 1)))).float()
        img_Noisy = torch.from_numpy(np.ascontiguousarray(img_Noisy.transpose((2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ.transpose((2, 0, 1)))).float()

        # return im_noisy, im_gt
        return  {'LQ': img_LQ, 'Noisy':img_Noisy, 'GT': img_GT, 'img_index': img_index}


