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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_CNN_denoiser_mcj.yml', help='Path to options YMAL file.')
opt = option.parse_(parser.parse_args().opt, is_train=False)
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

def plot_simi(img,dpi,figsize):
    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=0, vmax=1, cmap=plt.cm.jet)
    # plt.title('fake_noisy')
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()

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
    x_max=abs(readsegy(data_dir,'00-L120.sgy')).max()
    original=readsegy(data_dir,'00-L120.sgy')#[50:50+64,200:200+64]#[600:728,:128]#[72:200,372:500]#[600:856,:128]
    # x_max=abs(original).max()
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
    # from utils.PadUnet import PadUnet
    # padunet = PadUnet(x_, dep_U=5)
    # x_pad = padunet.pad()
    model.feed_test_data(x_)
    if opt['self_ensemble']:
        model.test(opt['self_ensemble'])
    elif opt['mc_ensemble']:
        model.MC_test()
    else:
        model.test()
    # model.fake_H = padunet.pad_inverse(model.fake_H)
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

    # plt.figure(dpi=300, figsize=(3, 3))
    # plt.imshow(x, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # # plt.title('fake_noisy')
    # plt.axis('off')
    #
    # plt.figure(dpi=300, figsize=(3, 3))
    # plt.imshow(groundtruth, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # # plt.title('fake_noisy')
    # plt.axis('off')
    # plt.show()
    #
    # plt.figure(dpi=300, figsize=(3, 3))
    # plt.imshow(denoised, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # # plt.title('denoised')
    # plt.axis('off')
    # plt.show()
    # sio.savemat(('../results/XJ_h72_w372_s128_denoised.mat'), {'data': denoised[:, :]})
    dir, test_name, sample, eps = 'sc3', 'XJ_r600c0s128', 0, '0'
    # sio.savemat(('./output/' + dir + '/xp_' + test_name + '_' + str(sample) + '.mat'), {'data': denoised})

    # plt.figure(dpi=300, figsize=(3, 3))
    # plt.imshow(x-denoised, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # # plt.title('Gaussion_noisy')
    # plt.axis('off')
    # plt.show()

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

    # plt.figure(dpi=300, figsize=(3, 3))
    # plt.imshow(noise, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # # plt.title('Clean_data')
    # plt.axis('off')
    # plt.show()

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

    from utils.localsimi import localsimi
    simi = localsimi(denoised, x-denoised, rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
    plot_simi(simi.squeeze(), 300, (5, 8.76))
    energy_simi = np.sum(simi ** 2) / simi.size
    print("energy_simi=", energy_simi)
    plt.show()



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
    opt['model'] = 'DnCNN'
    opt['gpu_ids']=[0]
    opt['split_channel'] =3
    opt['self_ensemble']=False

    # 2023 DnCNN_orfx_XJ10 8.33dB/35.69/0.8301/60/0.0202
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/DnCNN_orfx_XJ10/models/60.0_DnCNN.pth'

    # 2023 DnCNN_orfx_XJ30 8.39dB/35.75/0.8313/60  0.0195
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/DnCNN_orfx_XJ30/models/60.0_DnCNN.pth'

    # 2023.1.18 DnCNN_XJ10_aug2ep1e3 8.38dB/35.75/0.8312/60  v2 8.31dB/35.67/0.8291/60
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/DnCNN_XJ10_aug2ep1e3_v2/models/60.0_DnCNN.pth'

    # 2023 DnCNN_XJ10_aug2ep5e4    8.37/35.74/0.8309/60/0.0219     8.52/35.88/0.8292/3/0.0383
    # opt['path']['pretrain_model_G'] = '/home/shendi_mcj/code/InvDN-mcj/experiments/orfx_XJ/DnCNN_XJ10_aug2ep5e4/models/60.0_DnCNN.pth'

    # 2023 DnCNN_XJ10_aug2ep2e4 3  8.36dB/35.72/0.8305/60/0.0211 8.52dB/35.89/0.8332/3/0.0397
    # opt['path']['pretrain_model_G'] = '/home/shendi_mcj/code/InvDN-mcj/experiments/orfx_XJ/DnCNN_XJ10_aug2ep2e4/models/3.0_DnCNN.pth'

    # 2023.1.18 DnCNN_XJ10_aug2ep1e4 8.33dB/35.70/0.8299/60
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_XJ/DnCNN_XJ10_aug2ep1e4/models/60.0_DnCNN.pth'

    # 2023 DnCNN_XJ10_aug2ep12510e4  8.36dB/35.72/0.8306/60/0.0210  8.54/35.90/0.8323/3/0.0373
    opt['path']['pretrain_model_G'] = '/home/shendi_mcj/code/InvDN-mcj/experiments/orfx_XJ/DnCNN_XJ10_aug2ep12510e4/models/60.0_DnCNN.pth'

    # 2023.1.14 DnCNN_expert_XJ10  9.07dB/36.44/0.8492/60/0.1381  9.20/36.55/0.8488/13
    # opt['path']['pretrain_model_G'] ='../experiments/expert_XJ/DnCNN_expert_XJ10/models/60.0_DnCNN.pth'

    # 2023.1.14 DnCNN_expert_XJ30  9.28dB/36.65/0.8534/60/0.1417
    # opt['path']['pretrain_model_G'] ='../experiments/expert_XJ/DnCNN_expert_XJ30/models/60.0_DnCNN.pth'

    # 2023.1.14 DnCNN_XJ10_aug2ep2e4  9.20db/36.57/0.8507/0.1517
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/DnCNN_XJ10_aug2ep2e4/models/60.0_DnCNN.pth'

    # 2023.1.15 DnCNN_XJ10_aug2ep5e4  9.21/36.58/0.8510/60/0.1525 9.22/36.58/0.8511/45/0.1502
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/DnCNN_XJ10_aug2ep5e4/models/60.0_DnCNN.pth'

    # 2023.1.15 DnCNN_XJ10_aug2ep1e3  9.22/36.58/0.8513/60  9.23/36.59/0.8515/45
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/DnCNN_XJ10_aug2ep1e3/models/45.0_DnCNN.pth'

    # 2023.1.15 DnCNN_XJ10_aug2ep12510e4 9.21/36.57/0.8509
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/DnCNN_XJ10_aug2ep12510e4/models/60.0_DnCNN.pth'

    # 2023.1.15 DnCNN_XJ10_aug2ep12510e3  效果差 9.13/36.50/0.8476
    # opt['path']['pretrain_model_G'] = '../experiments/expert_XJ/DnCNN_XJ10_aug2ep12510e3/models/60.0_DnCNN.pth'

    model = create_model(opt)
    XJ_test(model, opt)
    # Panke_test(model, opt)


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