'''create dataset and dataloader'''
import cv2
import logging
import torch
import torch.utils.data
import random
import numpy as np
import torch.utils.data as uData
import h5py as h5
import os
from math import ceil


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQGTRN':
        from data.LQGTRN_dataset import LQGTRNDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGTSN':
        from data.LQGTSN_dataset import LQGTSNDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGTRN_Seis_train':
        from data.LQGTRN_seismic_dataset import LQGTRNDatasetSeis as Seis
        # dataset_opt['dataroot_h5'] ='/home/shendi_mcj/datasets/seismic/hdf5/expertXJ10_aug'
        # dataset_opt['dataroot_h5'] ='/home/shendi_mcj/datasets/seismic/hdf5/XJ10sc3'
        dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
        # if dataset_opt['dataset_type'] == 'expert_aug2ep1e3':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2ep1e3_Patches_256_expert.hdf5')
        # # elif dataset_opt['dataset_type'] == 'orthofxdecon_aug2epItvc':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2epItvc_Patches_256_orthofxdecon.hdf5')
        # elif dataset_opt['dataset_type'] == 'orthofxdecon_aug2ep5e3':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2ep5e3_Patches_256_orthofxdecon.hdf5')  # 10 or 30
        # elif dataset_opt['dataset_type']== 'orthofxdecon_aug1epItvd':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug1epItvd_Patches_256_orthofxdecon.hdf5')
        # elif dataset_opt['dataset_type']== 'orthofxdecon_aug2ep1e3':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2ep1e3_Patches_256_orthofxdecon.hdf5')
        if dataset_opt['dataset_type'] == 'expertXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'expertXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'orthofxdeconXJ10PK10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_PK_trn_s80_90_s21_30_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type']== 'orthofxdeconXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type']== 'orthofxdeconXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type']== 'orthofxdeconPK10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PK_trn_s21_30_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'orthodmssaPK10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PK_trn_s21_30_Patches_256_orthodmssa.hdf5')
        elif dataset_opt['dataset_type']== 'fxdeconXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256_fxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'fxdeconXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256_fxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'orthofxdecon_marmousi35_XJ_ns80_110':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'marmousi35_trn_XJ_s80_110_noise_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'marmousi35_gn005':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'marmousi35_trn_gaussian005_noise_Patches_256_1.hdf5')
        elif dataset_opt['dataset_type'] == 'orthofxdecon_marmousi35_gn005':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'marmousi35_trn_gaussian005_noise_Patches_256_orthofxdecon_1.hdf5')

        # elif dataset_opt['dataset_type'] == 'PankeSR':
        #     dataset_opt['dataroot_h5']='/home/shendi_mcj/datasets/seismic/20221126'
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PankeSR_Trn_Patches_128_stride64.hdf5')
        if dataset_opt['data_aug']:
            # dataset_opt['data_mul_aft_aug'] = ceil(dataset_opt['data_mul'] * (dataset_opt['fake_ratio'] + 1))
            dataset = Seis(dataset_opt, h5_file=train_h5, length=dataset_opt['data_mul_aft_aug'] * dataset_opt['batch_size'], pch_size=[dataset_opt['GT_size'],dataset_opt['GT_size']]) #6000
        else:
            dataset = Seis(dataset_opt, h5_file=train_h5, length=dataset_opt['data_mul'] * dataset_opt['batch_size'],pch_size=[dataset_opt['GT_size'], dataset_opt['GT_size']])  # 6000
    elif mode == 'LQGTRN_Seis_val':
        from data.LQGTRN_seismic_dataset import LQGTRNDatasetSeis as Seis
        if dataset_opt['dataset_type'] == 'PankeSR_val':
            dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/20221126'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PankeSR_Trn_Patches_128_stride64.hdf5')
        elif dataset_opt['dataset_type'] == 'overthrust_salt_tst_XJ_ns120':
            dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'overthrust_salt_tst_XJ_s120_noise_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'marmousi20mp_tst_XJ_ns110_120_noise':
            dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'],'marmousi20mp_tst_XJ_s110_120_noise_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'marmousi35_tst_XJ_ns110_120':
            dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'],'marmousi35_tst_XJ_s110_120_noise_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'marmousi35_tst_XJ_ns80_110':
            dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'],'marmousi35_tst_XJ_s80_110_noise_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'marmousi35_tst_gn005':
            dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'marmousi35_trn_gaussian005_noise_Patches_256_1.hdf5')
        else:
            dataset_opt['dataroot_h5']='/home/shendi_mcj/datasets/seismic/hdf5'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_tst_s110_120_Patches_256_stride_256.hdf5')
        dataset = Seis(dataset_opt, h5_file=test_h5, length= dataset_opt['batch_size'])
    elif mode == 'GTRN_Seis_train':
        from data.GTRN_seismic_dataset import GTRNDatasetSeis as Seis
        dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
        if dataset_opt['dataset_type'] == 'expertXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'expertXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'orthofxdeconXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'orthofxdeconXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'fxdecon':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256_fxdecon.hdf5')


        # dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5/expertXJ10_aug'
        # if dataset_opt['dataset_type'] == 'expert_aug2ep1e3':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2ep1e3_Patches_256_expert.hdf5')

        # dataset_opt['dataroot_h5'] ='/home/shendi_mcj/datasets/seismic/hdf5/XJ10sc3'
        # if dataset_opt['dataset_type'] == 'orthofxdecon_aug2epItvd':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2epItvd_Patches_256_orthofxdecon.hdf5')
        # elif dataset_opt['dataset_type'] == 'orthofxdecon_aug2epItvc':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2epItvc_Patches_256_orthofxdecon.hdf5')
        # elif dataset_opt['dataset_type'] == 'orthofxdecon_aug2ep5e3':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2ep5e3_Patches_256_orthofxdecon.hdf5')  # 10 or 30
        # elif dataset_opt['dataset_type']== 'orthofxdecon_aug1epItvd':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug1epItvd_Patches_256_orthofxdecon.hdf5')
        # elif dataset_opt['dataset_type']== 'orthofxdecon_aug2ep1e3':
        #     train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ10_aug2ep1e3_Patches_256_orthofxdecon.hdf5')
        if dataset_opt['data_aug']:
            dataset = Seis(dataset_opt, h5_file=train_h5,
                           length=dataset_opt['data_mul_aft_aug'] * dataset_opt['batch_size'],
                           pch_size=[dataset_opt['GT_size'], dataset_opt['GT_size']])  # 6000
        else:
            dataset = Seis(dataset_opt, h5_file=train_h5, length=dataset_opt['data_mul'] * dataset_opt['batch_size'], pch_size=[dataset_opt['GT_size'],dataset_opt['GT_size']])

        #6000
    elif mode == 'GTRN_Seis_val':
        from data.GTRN_seismic_dataset import GTRNDatasetSeisTest as Seis
        dataset_opt['dataroot_h5']='/home/shendi_mcj/datasets/seismic/hdf5'
        test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_tst_s110_120_Patches_256_stride_256.hdf5')
        dataset = Seis(h5_path=test_h5)
    elif mode == 'LQGTSN_Seis_train':
        from data.LQGTSN_seismic_dataset import LQGTSNDatasetSeis as Seis
        from data.prepare_data.mat.bia2small_mat import generate_patch_from_mat
        marmousi_im_dir='/home/shendi_mcj/datasets/seismic/marmousi'
        marmousi_im_list = generate_patch_from_mat(dir=marmousi_im_dir, pch_size=128, stride=[96, 96])
        from data.prepare_data.segy.big2small_segy import generate_patch_from_segy_1by1
        open_segy_dir='/home/shendi_mcj/datasets/seismic/train'
        open_im_list = generate_patch_from_segy_1by1(dir=open_segy_dir,pch_size=(128,128),stride=(96, 96),jump=2,agc=True,train_data_num=100000,aug_times=[],scales = [])
        train_im_list = np.concatenate([marmousi_im_list, open_im_list], axis=0)
        # train_im_list = train_im_list.astype(np.float32)
        dataset = Seis(dataset_opt, im_list=train_im_list, length=dataset_opt['data_mul'] * dataset_opt['batch_size'],
                       pch_size=[dataset_opt['GT_size'], dataset_opt['GT_size']], noiseL_B=[0,75])
    # elif mode == 'LQGTSN_Seis_train1':
    #     from data.LQGTSN_seismic_dataset import LQGTSNDatasetSeis as Seis
    #     # from data.prepare_data.mat.bia2small_mat import generate_patch_from_mat
    #     # marmousi_im_dir = '/home/shendi_mcj/datasets/seismic/marmousi'
    #     # marmousi_im_list = generate_patch_from_mat(dir=marmousi_im_dir, pch_size=64, stride=[64, 64])
    #     # from data.prepare_data.segy.big2small_segy import generate_patch_from_segy_1by1
    #     # open_segy_dir = '/home/shendi_mcj/datasets/seismic/train'
    #     open_segy_dir = '/home/shendi_mcj/datasets/seismic/fielddata/train/clean'
    #     from data.prepare_data.segy.get_patch import datagenerator
    #     open_im_list = datagenerator(data_dir=open_segy_dir, patch_size=(64, 64), stride=(32, 32),
    #                                      train_data_num=100000,
    #                                      download=False, datasets=0, aug_times=0,
    #                                      scales=[1],
    #                                      verbose=True, jump=3, agc=dataset_opt['agc'])
    #     # train_im_list = np.concatenate([marmousi_im_list, open_im_list], axis=0)
    #     # train_im_list = train_im_list.astype(np.float32)
    #     train_im_list=open_im_list
    #     dataset = Seis(dataset_opt, im_list=train_im_list, length=dataset_opt['data_mul'] * dataset_opt['batch_size'],
    #                    pch_size=[dataset_opt['GT_size'], dataset_opt['GT_size']], noiseL_B=[0, 75])
    # elif mode == 'LQGTSN_Seis_CL_NE':
    #     from data.LQGTSN_seismic_dataset import LQGTSNDatasetSeis_CL_NE as Seis
    #     # CL_h5_file = '/home/shendi_mcj/datasets/seismic/hdf5/marmousi_opensegy_agc_clean_patches_128_stride_96.hdf5'
    #     CL_h5_file = '/home/shendi_mcj/datasets/seismic/hdf5/marF20F40MpZp_os19AnMoSe_clean_patches_128_stride_96.hdf5'
    #     NE_h5_file = '/home/shendi_mcj/datasets/seismic/hdf5/XJ_PK_noise_Patches_256_orthofxdecon.hdf5'
    #     dataset = Seis(dataset_opt, CL_h5_file=CL_h5_file, length=dataset_opt['data_mul'] * dataset_opt['batch_size'], NE_h5_file=NE_h5_file,
    #                    pch_size=[dataset_opt['GT_size'], dataset_opt['GT_size']], noiseL_B=[0, 75])
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))


    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

def create_dataset_aug(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQGTRN':
        from data.LQGTRN_dataset import LQGTRNDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGTSN':
        from data.LQGTSN_dataset import LQGTSNDataset as D
        dataset = D(dataset_opt)
    elif mode == 'LQGTRN_Seis_train':
        from data.LQGTRN_seismic_dataset import LQGTRNDatasetSeis as Seis
        dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
        if dataset_opt['dataset_type'] == 'expertXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'expertXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256.hdf5')
        if dataset_opt['dataset_type']== 'orthofxdeconXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type']== 'orthofxdeconXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type']== 'orthofxdeconPK10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PK_trn_s21_30_Patches_256_orthofxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'orthodmssaPK10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PK_trn_s21_30_Patches_256_orthodmssa.hdf5')
        elif dataset_opt['dataset_type']== 'fxdecon':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256_fxdecon.hdf5')
        elif dataset_opt['dataset_type'] == 'PankeSR':
            dataset_opt['dataroot_h5']='/home/shendi_mcj/datasets/seismic/20221126'
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PankeSR_Trn_Patches_128_stride64.hdf5')
        dataset = Seis(dataset_opt, h5_file=train_h5, length=dataset_opt['data_mul'] * dataset_opt['batch_size'], pch_size=[dataset_opt['GT_size'],dataset_opt['GT_size']]) #6000
    elif mode == 'LQGTRN_Seis_val':
        from data.LQGTRN_seismic_dataset import LQGTRNDatasetSeis as Seis
        if dataset_opt['dataset_type'] == 'PankeSR_val':
            dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/20221126'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'PankeSR_Trn_Patches_128_stride64.hdf5')
        else:
            dataset_opt['dataroot_h5']='/home/shendi_mcj/datasets/seismic/hdf5'
            test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_tst_s110_120_Patches_256_stride_256.hdf5')
        dataset = Seis(dataset_opt, h5_file=test_h5, length= dataset_opt['batch_size'])
    elif mode == 'GTRN_Seis_train':
        from data.GTRN_seismic_dataset import GTRNDatasetSeis as Seis
        dataset_opt['dataroot_h5'] = '/home/shendi_mcj/datasets/seismic/hdf5'
        if dataset_opt['dataset_type'] == 'expertXJ10':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_90_Patches_256.hdf5')
        elif dataset_opt['dataset_type'] == 'expertXJ30':
            train_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_trn_s80_110_Patches_256.hdf5')
        dataset = Seis(dataset_opt, h5_file=train_h5, length=dataset_opt['data_mul'] * dataset_opt['batch_size'], pch_size=[dataset_opt['GT_size'],dataset_opt['GT_size']]) #6000
    elif mode == 'GTRN_Seis_val':
        from data.GTRN_seismic_dataset import GTRNDatasetSeisTest as Seis
        dataset_opt['dataroot_h5']='/home/shendi_mcj/datasets/seismic/hdf5'
        test_h5 = os.path.join(dataset_opt['dataroot_h5'], 'XJ_tst_s110_120_Patches_256_stride_256.hdf5')
        dataset = Seis(h5_path=test_h5)
    elif mode == 'LQGTSN_Seis_train':
        from data.LQGTSN_seismic_dataset import LQGTSNDatasetSeis as Seis
        from data.prepare_data.mat.bia2small_mat import generate_patch_from_mat
        marmousi_im_dir='/home/shendi_mcj/datasets/seismic/marmousi'
        marmousi_im_list = generate_patch_from_mat(dir=marmousi_im_dir, pch_size=128, stride=[96, 96])
        from data.prepare_data.segy.big2small_segy import generate_patch_from_segy_1by1
        open_segy_dir='/home/shendi_mcj/datasets/seismic/train'
        open_im_list = generate_patch_from_segy_1by1(dir=open_segy_dir,pch_size=(128,128),stride=(96, 96),jump=2,agc=True,train_data_num=100000,aug_times=[],scales = [])
        train_im_list = np.concatenate([marmousi_im_list, open_im_list], axis=0)
        # train_im_list = train_im_list.astype(np.float32)
        dataset = Seis(dataset_opt, im_list=train_im_list, length=dataset_opt['data_mul'] * dataset_opt['batch_size'],
                       pch_size=[dataset_opt['GT_size'], dataset_opt['GT_size']], noiseL_B=[0,75])
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))


    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

# Base Datasets
class BaseDataSetH5(uData.Dataset):
    def __init__(self, h5_path, length=None):
        '''
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetH5, self).__init__()
        self.h5_path = h5_path
        self.length = length
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, imgs_sets):
        H, W, C2 = imgs_sets.shape
        C = int(C2/2)
        ind_H = random.randint(0, H-self.pch_size[0])
        ind_W = random.randint(0, W-self.pch_size[0])
        im_noisy = np.array(imgs_sets[ind_H:ind_H+self.pch_size[0], ind_W:ind_W+self.pch_size[0], :C])
        im_gt = np.array(imgs_sets[ind_H:ind_H+self.pch_size[0], ind_W:ind_W+self.pch_size[0], C:])
        return im_gt, im_noisy

class BaseDataSetImg(uData.Dataset):
    def __init__(self, im_list, length, pch_size=(128,128)):
        '''
        Args:
            im_list (list): path of each image
            length (int): length of Datasets
            pch_size (int): patch size of the cropped patch from each image
        '''
        super(BaseDataSetImg, self).__init__()
        self.im_list = im_list
        self.length = length
        self.pch_size = pch_size
        self.num_images = len(im_list)

    def __len__(self):
        return self.length

    def crop_patch(self, im):
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size[0] or W < self.pch_size[0]:
            H = max(self.pch_size[0], H)
            W = max(self.pch_size[0], W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.pch_size[0])
        ind_W = random.randint(0, W-self.pch_size[0])
        pch = im[ind_H:ind_H+self.pch_size[0], ind_W:ind_W+self.pch_size[0]]
        return pch
