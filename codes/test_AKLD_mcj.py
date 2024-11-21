import os
import scipy
import torch
import cv2
import scipy.io as sio
import h5py
from data.util import read_img_array
import logging
import argparse
import numpy as np
import options.options as option
import utils.util as util
from models import create_model


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/generate/generate_InvDN_mcj.yml', help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
# args = parser.parse_args()
# opt = option.parse(args.opt, is_train=True)
opt = option.dict_to_nonedict(opt)


util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'generate_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

def generate_noisy(model, opt):
    dataset_dir = opt['name']

    out_dir = os.path.join('../experiments', dataset_dir)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'generate_fake_noisy')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # # load info
    # files = scipy.io.loadmat(os.path.join(opt['datasets']['generate']['dataroot_Noisy'], 'BenchmarkNoisyBlocksSrgb.mat'))
    # imgArray = files['BenchmarkNoisyBlocksSrgb']
    # nImages = 1 #40
    # nBlocks = 1  #imgArray.shape[1]  #32
    # # DenoisedBlocksSrgb = np.empty_like(imgArray)
    # # process data
    # Inoisy = read_img_array(imgArray[0][0])
    # Inoisy = torch.from_numpy(np.transpose(Inoisy, (2, 0, 1))).type(torch.FloatTensor)
    # data = Inoisy.unsqueeze(dim=0)

    # read the images
    from scipy.io import loadmat
    im_noisy_real = loadmat('./test_data/SIDD/noisy.mat')['im_noisy']
    im_gt = loadmat('./test_data/SIDD/gt.mat')['im_gt']

    # calculate AKLD and generate L sample
    L = 50
    AKLD = 0
    from skimage import img_as_float32, img_as_ubyte
    im_noisy_real = torch.from_numpy(img_as_float32(im_noisy_real).transpose([2, 0, 1])).unsqueeze(0).cuda()
    im_gt = torch.from_numpy(img_as_float32(im_gt).transpose([2, 0, 1])).unsqueeze(0).cuda()
    sigma_real = util.estimate_sigma_gauss(im_noisy_real, im_gt)

    # #pad  if Unet need pading
    # padunet = util.PadUNet(im_gt, dep_U=5)
    # im_gt = padunet.pad()

    # feed noisy and GT to model Class
    model.noisy_H=im_noisy_real
    model.real_H=im_gt

    for i in range(L):
        # outputs_pad = sample_generator(net, im_gt_pad)
        # im_noisy_fake = padunet.pad_inverse(outputs_pad)

        model.generateFakeNoisy(epsilon=2e-3) # 2e-4
        im_noisy_fake = model.fake_Noisy#.detach().float().cpu()  # 1,3,256,256
        im_noisy_fake.clamp_(0.0, 1.0)
        sigma_fake = util.estimate_sigma_gauss(im_noisy_fake, im_gt)
        kl_dis = util.kl_gauss_zero_center(sigma_fake, sigma_real)
        AKLD += kl_dis
        # save generate image
        Idenoised_crop = util.tensor2img_Real(im_noisy_fake)  # 3,256,256  # uint8
        Idenoised_crop = np.transpose(Idenoised_crop, (1, 2, 0))
        # DenoisedBlocksSrgb[i][k] = Idenoised_crop
        save_file = os.path.join(out_dir, '%d_%02d.PNG' % (1, i))
        cv2.imwrite(save_file, cv2.cvtColor(Idenoised_crop, cv2.COLOR_RGB2BGR))
        print('[%d/%d] is done\n' % (i + 1, L))
    AKLD /= L
    print("AKLD value: {:.3f}".format(AKLD))

def main():
    model = create_model(opt)
    generate_noisy(model, opt)

if __name__ == "__main__":
    main()