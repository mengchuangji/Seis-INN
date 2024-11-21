import os
import scipy
# import cv2
import scipy.io as sio
import h5py
from data.util import read_img_array
import logging
import argparse
import numpy as np
import options.options as option
import utils.util as util
from models import create_model
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_InvDN_mcj.yml', help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
# args = parser.parse_args()
# opt = option.parse(args.opt, is_train=True)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

def XJ_test(model, opt):
    dataset_dir = opt['name']
    out_dir = os.path.join('../experiments', dataset_dir)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'XJ_test') # dataset name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # load info
    data_dir='/home/shendi_mcj/datasets/seismic/fielddata'
    im = '00-L120.sgy'
    from utils.readsegy import readsegy
    #[600:728,:128]  [600:728,:128] [72:200,372:500](2ndpaper)
    x_max = abs(readsegy(data_dir, '00-L120.sgy')).max()
    original=readsegy(data_dir,'00-L120.sgy')#[50:50+64,200:200+64]#[600:728,:128]#[72:200,372:500]#[600:856,:128]
    # x_max=max(abs(original.max()),abs(original.min()))
    groundtruth = readsegy(data_dir, '00-L120-Y.sgy')#[50:50+64,200:200+64]#[600:728,:128]#[72:200,372:500]#[600:856,:128]
    noise = readsegy(data_dir, '00-L120-N.sgy')#[50:50+64,200:200+64]#[600:728,:128]#[72:200,372:500]#[600:856,:128]




    # process datax
    x = original
    x = x / x.max()
    # sio.savemat(('../results/XJ_h72_w372_s128_noisy.mat'), {'data': x[:, :]})
    groundtruth = groundtruth/x_max
    noise = noise / x_max
    from utils.compare import compare_SNR
    snr_y = compare_SNR(groundtruth, x)
    print('bdfore snr_y= {1:2.2f}dB'.format('test', snr_y))
    psnr_y = compare_psnr(groundtruth, x,data_range=2)
    print('before psnr_y=', '{:.4f}'.format(psnr_y))
    y_ssim = compare_ssim(groundtruth, x)
    print('before ssim=', '{:.4f}'.format(y_ssim))
    ##################################
    import time
    start_time = time.time()
    x_ = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1]).type(torch.FloatTensor)
    # data = Inoisy[k].unsqueeze(dim=0)
    model.feed_test_data(x_)
    if opt['self_ensemble']:
        model.test(opt['self_ensemble'])
    elif opt['mc_ensemble']:
        model.MC_test()
    else:
        model.test()

    img = model.fake_H.detach().float().cpu()  # 1,1,256,256
    denoised = img.cpu().numpy()
    denoised = denoised.squeeze()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    snr_x_ = compare_SNR(groundtruth, denoised)
    print('after snr_x_= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(groundtruth, denoised,data_range=2)
    print('after psnr_x =', '{:.4f}'.format(psnr_x_))
    y_ssim = compare_ssim(groundtruth, denoised)
    print('after ssim=', '{:.4f}'.format(y_ssim))

    from  utils.plotfunction import show_GT_NY_Dn_Rn_GTn_Resi,show_GT_NY_Dn_Rn
    # show_GT_NY_Dn_Rn_GTn_Resi(groundtruth,x,denoised)
    show_GT_NY_Dn_Rn(groundtruth,x,denoised,dpi=100,figsize=(12,3))
    import matplotlib.pyplot as plt

    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(x, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('fake_noisy')
    plt.axis('off')

    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(groundtruth, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('fake_noisy')
    plt.axis('off')
    plt.show()

    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(denoised, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('denoised')
    plt.axis('off')
    plt.show()
    # sio.savemat(('../results/XJ_h72_w372_s128_denoised.mat'), {'data': denoised[:, :]})
    dir, test_name, sample, eps = 'sc3', 'XJ_r600c0s128', 0, '0'
    # sio.savemat(('./output/' + dir + '/xp_' + test_name + '_' + str(sample) + '.mat'), {'data': denoised})

    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(x-denoised, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('Gaussion_noisy')
    plt.axis('off')
    plt.show()

    # fn = (x-denoised).copy().flatten()
    # import matplotlib.pyplot as plt
    # plt.figure(dpi=300, figsize=(3, 3))
    # from scipy.stats import norm
    # import seaborn as sns
    # # sns.distplot(a=gn, color='green',
    # #              hist_kws={"edgecolor": 'white'})
    # sns.distplot(a=fn, fit=norm, color='blue',
    #              hist_kws={"edgecolor": 'white'})
    # plt.axis('off')
    # plt.show()

    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(noise, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('Clean_data')
    plt.axis('off')
    plt.show()

    # plt.figure(dpi=300, figsize=(3, 3))
    # gn=0.1*np.random.normal(0, 1, groundtruth.shape)
    # plt.imshow(gn, cmap=plt.cm.seismic,vmin=-1, vmax=1)
    # # plt.title('Clean_data')
    # plt.axis('off')
    # plt.show()
    #
    # plt.figure(dpi=300, figsize=(3, 3))
    # plt.imshow(denoised + gn, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # # plt.title('denoised_gauss')
    # plt.axis('off')
    # plt.show()

    # gn_ = gn.copy().flatten()
    # import matplotlib.pyplot as plt
    # plt.figure(dpi=300, figsize=(3, 3))
    # import seaborn as sns
    # sns.distplot(a=gn_, fit=norm, color='green',
    #              hist_kws={"edgecolor": 'white'})
    # plt.axis('off')
    plt.show()

def AKLD(model, opt):
    dataset_dir = opt['name']
    out_dir = os.path.join('../experiments', dataset_dir)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'AKDL') # dataset name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # load info
    data_dir='/home/shendi_mcj/datasets/seismic/fielddata'
    data_dir_ = 'E:\博士期间资料\田亚军\\2021.6.07'
    im = '00-L120.sgy'
    from utils.readsegy import readsegy
    #[600:728,:128]  [600:728,:128] [72:200,372:500](2ndpaper)
    original=readsegy(data_dir,'00-L120.sgy')#[600:728,:128]#[72:200,372:500]#[600:856,:128]
    x_max=max(abs(original.max()),abs(original.min()))
    groundtruth = readsegy(data_dir, '00-L120-Y.sgy')#[600:728,:128]#[72:200,372:500]#[600:856,:128]
    noise = readsegy(data_dir, '00-L120-N.sgy')#[600:728,:128]#[72:200,372:500]#[600:856,:128]


    # process datax
    x = original
    x = x / x.max()
    # sio.savemat(('../results/XJ_h72_w372_s128_noisy.mat'), {'data': x[:, :]})
    groundtruth = groundtruth/x_max
    noise = noise / x_max

    ##################################
    import time
    start_time = time.time()
    x_ = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1]).type(torch.FloatTensor)
    groundtruth_ = torch.from_numpy(groundtruth).view(1, -1, x.shape[0], x.shape[1]).type(torch.FloatTensor)
    model.feed_test_data(x_)
    # calculate AKLD and generate L sample
    L = 50
    AKLD = 0
    # # torch array
    # sigma_real = util.estimate_sigma_gauss(x_, groundtruth_)
    # for i in range(L):
    #     # outputs_pad = sample_generator(net, im_gt_pad)
    #     # im_noisy_fake = padunet.pad_inverse(outputs_pad)
    #
    #     model.generateFakeNoisy(epsilon=1e-4) # 2e-4
    #     im_noisy_fake = model.FakeNoisy.detach().float().cpu()  # 1,3,256,256
    #     # im_noisy_fake.clamp_(0.0, 1.0)
    #     sigma_fake = util.estimate_sigma_gauss(im_noisy_fake, groundtruth_)
    #     kl_dis = util.kl_gauss_zero_center(sigma_fake, sigma_real) #.to(sigma_fake.device)
    #     AKLD += kl_dis
    #     # # save generate image
    #     # Idenoised_crop = util.tensor2img_Real(im_noisy_fake)  # 3,256,256  # uint8
    #     # Idenoised_crop = np.transpose(Idenoised_crop, (1, 2, 0))
    #     # # DenoisedBlocksSrgb[i][k] = Idenoised_crop
    #     # save_file = os.path.join(out_dir, '%d_%02d.PNG' % (1, i))
    #     # cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop, cv2.COLOR_RGB2BGR))
    #     # print('[%d/%d] is done\n' % (i + 1, L))
    # AKLD /= L
    # print("AKLD value: {:.3f}".format(AKLD))
    # numpy array
    sigma_real= util.sigma_estimate_gauss_numpy(x, groundtruth,7,3)
    for i in range(L):
        model.generateFakeNoisy(epsilon=1e-2) # 2e-4
        im_noisy_fake = model.FakeNoisy.detach().float().cpu().numpy().squeeze()  #
        # denoised = img.cpu().numpy()
        # denoised = denoised.squeeze()
        sigma_fake = util.sigma_estimate_gauss_numpy(im_noisy_fake, groundtruth,7,3)
        kl_dis = util.kl_gauss_zero_center_numpy(sigma_fake, sigma_real)
        AKLD += kl_dis
    AKLD /= L
    print("AKLD value: {:.3f}".format(AKLD))


    # plt.figure(dpi=300, figsize=(3, 3))
    # plt.imshow(x, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # # plt.title('fake_noisy')
    # plt.axis('off')









def Panke_test(model, opt):
    dataset_dir = opt['name']
    out_dir = os.path.join('../experiments', dataset_dir)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'Panke_test') # dataset name
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # load info
    data_dir='/home/shendi_mcj/datasets/seismic/test'
    data_dir1 = '/home/shendi_mcj/datasets/seismic/PankeData'
    # im = 'PANKE-INline443.sgy'
    from utils.readsegy import readsegy
    # original=readsegy(data_dir,'PANKE-INline443.sgy')[:1000,:500]#2348,784
    original = readsegy(data_dir1, 'pk-00-L21-40-t400-4000.sgy')[:,15902-795:15902]
    x_max=max(abs(original.max()),abs(original.min()))


    # process data
    x = original[:1000, :500]  #[800:928, :128]
    x = x / x_max

    ##################################
    import time
    start_time = time.time()
    x_ = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1]).type(torch.FloatTensor)
    # data = Inoisy[k].unsqueeze(dim=0)
    model.feed_test_data(x_)
    if opt['self_ensemble']:
        model.test(opt['self_ensemble'])
    elif opt['mc_ensemble']:
        model.MC_test()
    else:
        model.test()

    img = model.fake_H.detach().float().cpu()  # 1,1,256,256
    denoised = img.cpu().numpy()
    denoised = denoised.squeeze()
    elapsed_time = time.time() - start_time


    from  utils.plotfunction import show_GT_NY_Dn_Rn_GTn_Resi,show_NY_Dn_Rn
    # show_GT_NY_Dn_Rn_GTn_Resi(groundtruth,x,denoised)
    show_NY_Dn_Rn(x,denoised)


def main():
    opt['model'] = 'InvDN_C'
    opt['scale']= 4
    opt['gpu_ids']=[7]
    opt['network_G']['scale']= 4
    opt['network_G']['block_num'] = [8,8]
    opt['split_channel'] = 3
    # opt['self_ensemble']=True
    # opt['mc_ensemble']=True

    # opt['path']['pretrain_model_G']='/home/shendi_mcj/code/InvDN-mcj/pretrained/XJ_32/8.0_G.pth'
    # default
    # opt['path']['pretrain_model_G']='/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4/models/10.0_G.pth'
    # #  test pixel_criterion_back l2 or l1
    # opt['path']['pretrain_model_G']='/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_l2/models/20.0_G.pth'
    # # test GT_size 32 or 64
    # opt['path']['pretrain_model_G']= '/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_GTsize64/models/20.0_G.pth'
    # # test block_num 8 or 16
    # opt['path']['pretrain_model_G']='/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_IB16/models/20.0_G.pth'
    # # test scale x4 or x8
    # opt['path']['pretrain_model_G']='/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x2_IB8/models/20.0_G.pth'
    # #test syn GB
    # opt['path']['pretrain_model_G']='/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_syn_GB/models/20000_G.pth'

    #test syn l1 l2
    # opt['path']['pretrain_model_G']='/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_IB8_syn_GB_l2/models/7.0_G.pth'

    # test XJ_clean+gauss
    # opt['path']['pretrain_model_G'] = '/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_syn_GB_trainXJclean/models/55000_G.pth'

    #2022.10.21
    # opt['path']['pretrain_model_G'] ='/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_syn_agc_B_75/models/146000_G.pth'

    # orfx_XJ10
    # opt['path']['pretrain_model_G'] = '../experiments/InvDN_ResUnit_x4_orfx_XJ10/models/11_G.pth'

    # orfx_XJ30
    # opt['path']['pretrain_model_G'] = '/home/shendi_mcj/code/InvDN-mcj/experiments/InvDN_ResUnit_x4_orfx_XJ30_1/models/2_G.pth'
 ###############################################################################################

    # 2022.11.5 orfx_XJ10_sc3 8.38/35.74/0.8287/3
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/InvDN_ResUnit_x4_orfx_XJ10_sc3/models/3_G.pth'

    # 2022.11.5 orfx_XJ30_sc3  8.53/35.89/0.8335/9
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/InvDN_ResUnit_x4_orfx_XJ30_sc3/models/9_G.pth'

    # 2023.1.17 orfx_XJ10_aug2ep1e4 8.44dB/35.80/0.8298/26
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/InvDN_ResUnit_x4_XJ10_aug2ep1e4/models/26_G.pth'

    # 2023.1.17 orfx_XJ10_aug2ep2e4  8.50/35.86/0.8326/16     v0  8.54dB/35.84/0.8190
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/InvDN_ResUnit_x4_XJ10_aug2ep2e4_v0/models/18_G.pth'

    # 2023.1.17 orfx_XJ10_aug2ep5e4  8.46dB/35.82/0.8323/8
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/InvDN_ResUnit_x4_XJ10_aug2ep2e4/models/8_G.pth'

    # 2023.1.17 orfx_XJ10_aug2ep1e3 8.45dB/35.79/0.8265/11
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/InvDN_ResUnit_x4_XJ10_aug2ep1e3/models/11_G.pth'

    # 2023.1.17 orfx_XJ10_aug2ep12510e4 8.42dB/35.77/0.8274/7
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/InvDN_ResUnit_x4_XJ10_aug2ep12510e4/models/7_G.pth'

    # 2023.1.1 orfx_PK10_sc3 5 10 13
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_PK10/sc3/models/5_G.pth'

    # 2023.1.2 ordm_PK10_sc3 5 10 13
    # opt['path']['pretrain_model_G'] = '/home/shendi_mcj/code/InvDN-mcj/experiments/ordm_PK10/sc3/models/10_G.pth'

    # 2023.1.12 InvDN_ResUnit_x4_expert_XJ10  8.75/36.11/0.8395/10   8.72dB/36.070.8375/11/_2
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/InvDN_ResUnit_x4_expert_XJ10_2/models/11_G.pth'

    # 2023.1.19 InvDN_ResUnit_x4_XJ10_aug2ep2e4   8.74dB/36.10/0.8394/20     v0
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/InvDN_ResUnit_x4_XJ10_aug2ep2e4/models/20_G.pth'

    # 2023.1.20 InvDN_ResUnit_x4_XJ10_aug2ep5e4   8.76dB/36.13/0.8402/20     v0
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/InvDN_ResUnit_x4_XJ10_aug2ep5e4/models/20_G.pth'

    # 2023.1.12 InvDN_ResUnit_x4_expert_XJ30 8.75/36.12/0.8407/12  8.77/36.12/0.8379/60
    # opt['path']['pretrain_model_G'] ='../experiments/expert_XJ/InvDN_ResUnit_x4_expert_XJ30/models/12_G.pth'

    # 2023.1.15 InvDN_ResUnit_x4_orfx_XJ10PK10 8.33/35.69/0.8291/60
    # opt['path']['pretrain_model_G'] ='../experiments/orfx_XJPK/InvDN_ResUnit_x4_XJ10PK10_5000/models/60_G.pth'

    # 2023.3.8 InvDN_ResUnit_x4_fx_XJ10 7.98/35.34/0.8241
    opt['path']['pretrain_model_G'] = '../experiments/fx_XJ/InvDN_ResUnit_x4_fx_XJ10/models/5_G.pth'

    model = create_model(opt)
    XJ_test(model, opt)
    # Panke_test(model, opt)
    # AKLD(model, opt)

def save_hist(x, root):
    x = x.flatten()
    import matplotlib.pyplot as plt
    plt.figure()
    n, bins, patches = plt.hist(x, bins=128, density=1)
    # plt.savefig(root)
    # plt.close()
    plt.show()

if __name__ == "__main__":
    main()