import os
import cv2
import scipy
import torch
import scipy.io as sio
import h5py
from data.util import read_img_array
import logging
import argparse
import numpy as np
import options.options as option
import utils.util as util
from models import create_model
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

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


# import matplotlib.pyplot as plt
def plot_(img,dpi,figsize,fontsize,save_dir,name,save):
    save_obj = os.path.join(save_dir, name)
    if save:
        sio.savemat((save_obj + '.mat'), {'data': img})
    # Set font
    fpath = '/home/shendi_mcj/fonts/times.ttf'
    prop = fm.FontProperties(fname=fpath, size=fontsize)
    fig, (ax0) = plt.subplots(nrows=1, dpi=dpi,figsize=figsize)
    plt.imshow(img, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('fake_noisy')
    ax0.set_xlabel("Trace", fontsize=fontsize, fontproperties=prop)
    ax0.set_ylabel("Time(s)", fontsize=fontsize, fontproperties=prop)
    # ax0.set_title('trace compare', fontsize=fontsize)  # trace='+str(trace_num_1)
    ax0.tick_params(direction='in')
    ax0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax0.xaxis.set_label_position('top')
    ax0.set_xticks(np.arange(0, 501, 100))
    ax0.set_xticklabels(np.arange(0, 501, 100), fontdict={'size': fontsize}, fontproperties=prop)
    ax0.set_yticks(np.arange(0, 1001, 200)) #np.arange(0, 3504/1000, 0.5)
    ax0.set_yticklabels(['{:g}'.format (i) for i in np.arange(0, 2002/1000, 0.4)], fontdict={'size': fontsize}, fontproperties=prop)
    plt.tight_layout()
    if save:
        plt.savefig(save_obj+'.jpg', bbox_inches='tight')
    plt.show()

def plot_cmap(img,dpi,figsize,cmap):
    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=0, vmax=1, cmap=cmap)
    # plt.title('fake_noisy')
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()

def generate_test(model, opt):

    # load info
    data_dir='/home/shendi_mcj/datasets/seismic/test'
    data_dir1 = '/home/shendi_mcj/datasets/seismic/PankeData'
    # data_dir1_='E:\博士期间资料\田亚军\田博给的数据\盘客数据'
    # im = 'PANKE-INline443.sgy'
    from utils.readsegy import readsegy
    # original=readsegy(data_dir,'PANKE-INline443.sgy')#[:1000,:500]#2348,784
    original = readsegy(data_dir1, 'pk-00-L21-40-t400-4000.sgy')[:, 15902 - 796:15902]
    x_max=max(abs(original.max()),abs(original.min()))
    im='panke'


    # process data
    x = original[:1000, :500]#[0:64, :64]  #[:1000, :796] [:1000, :500][872:1000, :128]
    x = x / x_max

    import time
    start_time = time.time()
    x_ = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1]).type(torch.FloatTensor)
    # model.feed_test_data(x_)
    model.noisy_H = x_
    # generate field data
    # model.generateFakeNoisy(epsilon=0)#2e-4
    # fake_noisy = model.FakeNoisy.detach().float().cpu()  # 1,1,256,256
    # fake_noisy = fake_noisy.cpu().numpy().squeeze()
    # elapsed_time = time.time() - start_time
    # print(' %10s : %2.4f second' % (im, elapsed_time))
    # generate clean data
    # model.feed_test_data(x_)
    if opt['self_ensemble']:
        model.test(opt['self_ensemble'])
    elif opt['mc_ensemble']:
        model.MC_test()
    else:
        model.test()
    img = model.fake_H.detach().float().cpu()  # 1,1,256,256
    denoised = img.cpu().numpy()
    denoised = denoised.squeeze()


    # ssim_FN_RN, ssim_map_FN_RN = compare_ssim(x, fake_noisy, full=True)
    # print('ssim_FN_RN =', '{:.4f}'.format(ssim_FN_RN))
    # plot_cmap(ssim_map_FN_RN, 300, (5, 10), cmap=plt.cm.jet)

    from  utils.plotfunction import show_GT_GN_FN_RN,show_FN_RN,show_GC_GN_FN_RN
    # show_FN_RN(fake_noisy,x)
    # sigma_true = util.estimate_sigma_gauss(x_, img).squeeze().numpy()
    # Gaussion_noisy = denoised + np.random.normal(0, 1, denoised.shape)*sigma_true/sigma_true.max() #* 0.1
    # ssim_GN_RN, ssim_map_GN_RN = compare_ssim(x, Gaussion_noisy.astype(x.dtype), full=True)
    # print('ssim_GN_RN =', '{:.4f}'.format(ssim_GN_RN))
    # plot_cmap(ssim_map_GN_RN, 300, (5, 10), cmap=plt.cm.jet)
    # show_GC_GN_FN_RN(denoised, Gaussion_noisy, fake_noisy, x)
    # # calculate AKLD and generate L sample
    # sigma_real = util.estimate_sigma_gauss(x_, img)
    # L = 50
    # AKLD_G = 0
    # for i in range(L):
    #     # Gaussion_noisy = groundtruth + np.random.normal(0, 1, groundtruth.shape) * sigma_true.squeeze()[:, :] * (
    #     #             0.44 / sigma_true.max())
    #     Gaussion_noisy = denoised + np.random.normal(0, 1, img.shape) * sigma_true.squeeze()[:,:] / sigma_true.max()
    #     Gaussion_noisy_ = torch.from_numpy(Gaussion_noisy).view(1, -1, x.shape[0], x.shape[1]).type(torch.FloatTensor)
    #     sigma_fake = util.estimate_sigma_gauss(Gaussion_noisy_, img)
    #     kl_dis = util.kl_gauss_zero_center(sigma_fake, sigma_real)
    #     AKLD_G += kl_dis
    # AKLD_G /= L
    # print("AKLD_G value: {:.4f}".format(AKLD_G))


    from utils.plot_trace_compare import plot_trace_compare
    # plot_trace_compare(x=groundtruth, y=x,yf=fake_noisy)
    #
    save_dir = './output/INN_denoise'
    # # dir, test_name, sample, eps = 'sc3', 'XJ_r600c0s128', 0, '0'
    plot_(denoised, 300, (5, 9), 14, save_dir, 'INN_orfx_PK30_dn', True)
    # # sio.savemat(('./output/' + dir + '/x_' + test_name +'.mat'), {'data': denoised})
    plot_(x-denoised, 300, (5, 9), 14, save_dir, 'INN_orfx_PK30_n', True)
    # # sio.savemat(('./output/' + dir + '/n_' + test_name +'.mat'), {'data': x-denoised})
    # plot_(Gaussion_noisy, 300, (5, 10))
    # # sio.savemat(('./output/' + dir + '/yg_' + test_name +'.mat'), {'data':Gaussion_noisy})
    # plot_(fake_noisy,300,(5, 10))
    # # sio.savemat(('./output/'+dir+'/yf_'+test_name+'_'+eps+'_'+str(sample)+'.mat'), {'data': fake_noisy})
    # plot_(x, 300, (5, 9), 14, save_dir, 'PK_c500_r1000', True)
    # # sio.savemat(('./output/' + dir + '/y_' + test_name +'.mat'), {'data': x})
    #
    from utils.localsimi import localsimi
    # simi = localsimi(denoised, x-denoised, rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
    # plot_cmap(simi.squeeze(), 300, (5, 10), cmap=plt.cm.jet)
    # energy_simi = np.sum(simi ** 2) / simi.size
    # print("energy_simi=", energy_simi)
    #
    # yf_simi = localsimi(x, fake_noisy, rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
    # plot_cmap(yf_simi.squeeze(), 300, (5, 10), cmap=plt.cm.jet)
    # energy_yf_simi = np.sum(yf_simi ** 2) / yf_simi.size
    # print("energy_yf_simi=", energy_yf_simi)


    # from utils import wigb
    # denoised_ = denoised.copy()
    # fake_noisy_=fake_noisy.copy()
    # x_=x.copy()
    # wigb.wigb(denoised_, figsize=(10, 6), linewidth=1)
    # wigb.wigb(fake_noisy_, figsize=(10, 6), linewidth=1)
    # wigb.wigb(x_, figsize=(10, 6), linewidth=1)

def generate_aug_dataset(model, opt):
    import h5py as h5
    import numpy as np

    def show(x, method):
        import matplotlib.pyplot as plt
        noise = x.squeeze()
        # residual = gain(residual, 0.004, 'agc', 0.05, 1)
        # plt.xticks([])  # 去掉横坐标值
        # plt.yticks([])  # 去掉纵坐标值
        plt.imshow(noise, vmin=-1, vmax=1, cmap='gray')
        # io.savemat(('../../noise/shot_' + method + '_n.mat'), {'data': noise[:, :, np.newaxis]})
        # plt.title('removed noise')
        plt.tight_layout()
        plt.show()

    XJ_hdf5_path = '/home/shendi_mcj/datasets/seismic/hdf5/XJ_trn_s80_110_Patches_256_orthofxdecon.hdf5'
    hdf5_path_list = [XJ_hdf5_path]
    XJ_aug_hdf5_path = '/home/shendi_mcj/datasets/seismic/hdf5/XJ10sc3/XJ10_aug1epItvd_Patches_256_orthofxdecon.hdf5'
    noisy_list = []
    gt_list=[]
    num_patch = 0
    aug_time= 1
    aug_time_ratio=1
    epsilon_change=True
    epsilon_interval=False
    contain_original_data= True
    for hdf_path in hdf5_path_list:
        with h5.File(hdf_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            num_images = len(keys)
            for ii in range(len(keys)):
                if (ii + 1) % 10 == 0:
                    print('    The {:d} original images'.format(ii + 1))
                if ii  % int(1/aug_time_ratio) == 0:
                    imgs_sets = h5_file[keys[ii]]
                    H, W, C2 = imgs_sets.shape
                    # minus the bayer patter channel
                    C = int(C2 / 2)
                    im_noisy = np.array(imgs_sets[:, :, :C])
                    im_gt = np.array(imgs_sets[:, :, C:])
                    if ii == 0:
                        show(im_noisy, method='orthofxdecon')
                    for jj in range(aug_time):
                        x_ = torch.from_numpy(im_noisy).view(1, -1, im_noisy .shape[0], im_noisy .shape[1]).type(torch.FloatTensor)
                        # model.feed_test_data(x_)
                        model.noisy_H = x_
                        if epsilon_change:
                            epsilon_list=[1e-2,7e-3,5e-3,3e-3,1e-3] #[3e-3,2e-3,1e-3,5e-4,1e-4]
                            idx=np.random.randint(0,5,size=1)[0]
                            epsilon=epsilon_list[idx]
                            model.generateFakeNoisy(epsilon=epsilon)  # 2e-4
                        elif epsilon_interval:
                            epsilon_list=[i * 0.001 for i in range(0, 10, 1)] #[i * 0.0001 for i in range(0, 30, 1)]
                            idx = np.random.randint(0, 10, size=1)[0]
                            epsilon = epsilon_list[idx]
                            model.generateFakeNoisy(epsilon=epsilon)  # 2e-4
                        else:
                            model.generateFakeNoisy(epsilon=1e-3)  # 2e-4
                        fake_noisy = model.FakeNoisy.detach().float().cpu()  # 1,1,256,256
                        fake_noisy = np.expand_dims(fake_noisy.cpu().numpy().squeeze(),axis=2)
                        fkn_max=abs(fake_noisy).max()
                        noisy_list.append(fake_noisy/fkn_max)
                        gt_list.append(im_gt/fkn_max)
                        num_patch += 1
                if contain_original_data:
                    noisy_list.append(im_noisy)
                    gt_list.append(im_gt)
                    num_patch += 1
        print('Single Dataset Finish!\n')

    num_patch_c = 0
    with h5.File(XJ_aug_hdf5_path, 'w') as h5_file:
        for jj in range(len(noisy_list)):
            if (jj + 1) % 10 == 0:
                print('The {:d} noise images'.format(jj + 1))
            pch_noisy = noisy_list[jj]
            pch_dn = gt_list[jj]
            pch_imgs = np.concatenate((pch_noisy, pch_dn), axis=2)
            h5_file.create_dataset(name=str(num_patch_c), shape=pch_imgs.shape,
                                   dtype=pch_imgs.dtype, data=pch_imgs)
            num_patch_c += 1
    print('Total {:d} small noise data in noise dataset'.format(num_patch_c))
    print('Finish!\n')

def generate_latentZ_visualize(model, opt):
    import h5py as h5
    import numpy as np
    def gaussian_batch(device,dims):
        return torch.randn(tuple(dims)).to(device)
    def show(x, method):
        import matplotlib.pyplot as plt
        noise = x.squeeze()
        # residual = gain(residual, 0.004, 'agc', 0.05, 1)
        # plt.xticks([])  # 去掉横坐标值
        # plt.yticks([])  # 去掉纵坐标值
        plt.imshow(noise, vmin=-1, vmax=1, cmap='gray')
        # io.savemat(('../../noise/shot_' + method + '_n.mat'), {'data': noise[:, :, np.newaxis]})
        # plt.title('removed noise')
        plt.tight_layout()
        plt.show()

    XJ_hdf5_path = '/home/shendi_mcj/datasets/seismic/hdf5/XJ_trn_s80_110_Patches_256_orthofxdecon.hdf5'
    hdf5_path_list = [XJ_hdf5_path]
    # XJ_aug_hdf5_path = 'E:\博士期间资料\田亚军\\2021.6.07\original\XJ_PK_noise_Patches_256_orthofxdecon.hdf5'
    Z_list = []
    Z_HF_list=[]
    num_patch = 0
    aug_time=1
    for hdf_path in hdf5_path_list:
        with h5.File(hdf_path, 'r') as h5_file:
            keys = list(h5_file.keys())
            num_images = len(keys)
            for ii in range(len(keys)):
                if (ii + 1) % 10 == 0:
                    print('    The {:d} original images'.format(ii + 1))
                imgs_sets = h5_file[keys[ii]]
                H, W, C2 = imgs_sets.shape
                # minus the bayer patter channel
                C = int(C2 / 2)
                im_noisy = np.array(imgs_sets[:, :, :C])
                im_gt = np.array(imgs_sets[:, :, C:])
                if ii == 0:
                    show(im_noisy, method='orthofxdecon')

                x_ = torch.from_numpy(im_noisy).view(1, -1, im_noisy .shape[0], im_noisy .shape[1]).type(torch.FloatTensor)
                # model.feed_test_data(x_)
                model.noisy_H = x_
                test_opt = opt['test']
                s_c = 3
                if test_opt and test_opt['gaussian_scale'] != None:
                    gaussian_scale = test_opt['gaussian_scale']
                output = model.netG(x=model.noisy_H)
                Z = output[:, s_c:, :, :].detach().float().cpu().numpy().squeeze().flatten()
                Z_HF = gaussian_batch(device=model.device, dims=output[:, s_c:, :, :].shape).detach().float().cpu().numpy().squeeze().flatten()
                Z_list.append(Z)
                Z_HF_list.append(Z_HF)
                num_patch += 1
        print('Single Dataset Finish!\n')
    print('Total {:d} latent variant '.format(num_patch))

    # visualize Z and Z_HF
    epsilon = 3e-3
    Z_HF_f_list = [i+epsilon *np.random.normal(0, 1, i.shape).flatten() for i in Z_list]
    Z_label=np.zeros((len(Z_list[:])))
    Z_HF_label = np.ones((len(Z_HF_list[:])))
    Z_HF_f_label = np.ones((len(Z_HF_f_list[:])))+1
    y=np.concatenate((Z_label,Z_HF_label,Z_HF_f_label), axis=0)
    # Z_HF_list = [np.random.normal(0,1,i.shape).flatten() for i in Z_HF_list]
    # Z_list = [np.random.normal(0, 1, i.shape).flatten() for i in Z_list]
    X=np.array(Z_list[:]+Z_HF_list[:]+Z_HF_f_list[:])




    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)  # (560,3)

    ex_variance = np.var(X_pca, axis=0)  # (3,1)
    ex_variance_ratio = ex_variance / np.sum(ex_variance)

    Xax = X_pca[:, 0]  # (569,1)
    Yax = X_pca[:, 1]
    Zax = X_pca[:, 2]

    cdict = {0: 'red', 1: 'green', 2: 'blue'}
    labl = {0: '${\\bf{z}}_{hf}$', 1: '${\\bf{z}}_{hf}^{new}$', 2: '${\\bf{z}}_{hf}^{f}$, $\epsilon$=3e-3'}
    marker = {0: '*', 1: 'o', 2: 's'}
    alpha = {0: .1, 1: .1, 2: .1}

    fig = plt.figure(dpi=750,figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    # Set font
    fpath = '/home/shendi_mcj/fonts/times.ttf'
    from matplotlib import font_manager as fm
    prop = fm.FontProperties(fname=fpath, size=12)


    fig.patch.set_facecolor('white')
    for l in np.unique(y):  # [0,1]
        ix = np.where(y == l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=20,
                   label=labl[l], marker=marker[l], alpha=alpha[l])
    # for loop ends
    ax.set_xlabel("First Principal Component", fontsize=12,fontproperties=prop)
    ax.set_ylabel("Second Principal Component", fontsize=12,fontproperties=prop)
    ax.set_zlabel("Third Principal Component", fontsize=12,fontproperties=prop)

    ax.legend()
    plt.show()
    print('Finish!\n')





def main():
    opt['model'] = 'InvDN_C'
    opt['scale'] = 4
    opt['gpu_ids'] = [7]
    opt['split_channel']=3 #1
    opt['network_G']['scale'] = 4
    opt['network_G']['block_num'] = [8, 8]


    # 2023.1.1 orfx_PK10_sc3 5 10 13
    # opt['path']['pretrain_model_G'] = '../experiments/orfx_PK10/sc3/models/5_G.pth'
    # 2023.1.2 ordm_PK10_sc3 5 10 13
    opt['path']['pretrain_model_G'] = '../experiments/ordm_PK10/sc3/models/10_G.pth'

    # 2023.1.15 InvDN_ResUnit_x4_orfx_XJ10PK10 8.33/35.69/0.8291/60
    # opt['path']['pretrain_model_G'] ='../experiments/orfx_XJPK/InvDN_ResUnit_x4_XJ10PK10_5000/models/60_G.pth'

    model = create_model(opt)
    generate_test(model, opt)
    # generate_latentZ_visualize(model, opt)
    # generate_aug_dataset(model, opt)

if __name__ == "__main__":
    main()