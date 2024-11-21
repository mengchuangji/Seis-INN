import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from models.other_networks.UNet import UNet
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, Gradient_Loss, SSIM_Loss, Gradient_Loss_chn1
import models.networks as networks
from math import ceil
import numpy as np

logger = logging.getLogger('base')

class Unet_model_aug(BaseModel):
    def __init__(self, opt):
        super(Unet_model_aug, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.s_c = opt['split_channel']
        self.model_generator = networks.define_G(opt).to(self.device)
        self.load_model_generator()

        self.net = UNet(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2).to(self.device)

        if opt['dist']:
            self.net = DistributedDataParallel(self.net, device_ids=[torch.cuda.current_device()])
        else:
            self.net = DataParallel(self.net)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.net.train()

            # loss
            self.MSE_loss = ReconstructionLoss(losstype='l2')
            self.Grad_loss = Gradient_Loss_chn1() #mcj Gradient_Loss()
            self.SSIM_loss = SSIM_Loss()


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.real_H = data['GT'].to(self.device)  # GT
        self.noisy_H = data['Noisy'].to(self.device)  # Noisy
    def feed_data_aug(self, data):
        NReal = ceil(self.opt['datasets']['train']['batch_size'] / (1 + self.opt['datasets']['train']['fake_ratio']))
        self.real_H = data['GT'].to(self.device)  # GT
        self.noisy_H = data['Noisy'].to(self.device)  # Noisy
        model_generator=self.model_generator
        if (type(self.opt['datasets']['train']['epsilon']).__name__ == 'list'):
            epsilon_arr=self.opt['datasets']['train']['epsilon']
            len_arr = len(epsilon_arr)
            idx = np.random.randint(0, len_arr, size=1)[0]
            epsilon = epsilon_arr[idx]
        else:
            epsilon = self.opt['datasets']['train']['epsilon']
        with torch.autograd.no_grad():
            # self.real_H[NReal:, ] = self.INN_denoiser(model_generator, self.noisy_H[NReal:, ])
            # self.noisy_H[NReal:, ] = self.sample_generator(model_generator,self.noisy_H[NReal:, ],epsilon)
            denoised = self.INN_denoiser(model_generator, self.noisy_H[NReal:, ])
            fake_noisy = self.sample_generator(model_generator, self.noisy_H[NReal:, ], epsilon)
            field_noise = fake_noisy - denoised
            self.noisy_H[NReal:, ] = self.real_H[NReal:, ] + field_noise

        self.optimizer_G.zero_grad()

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def feed_test_data(self, data):
        self.noisy_H = data.to(self.device)  # Noisy

    def loss(self, out, y):
        l_forw_fit = self.MSE_loss(out, y)

        return l_forw_fit # + l_forw_grad + l_forw_SSIM



    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        # forward
        self.output = self.net(x=self.noisy_H)

        l_fit = self.loss(self.output, self.real_H)

        # total loss
        loss = l_fit
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_fit'] = l_fit.item()

    def sample_generator(self,model_generator,x,epsilon):
        gaussian_scale = 1
        output = model_generator(x=x)
        # C=self.input.shape[1]
        Z_dist = output[:, self.s_c:, :, :] + epsilon * self.gaussian_batch(output[:, self.s_c:, :, :].shape)
        y_forw = torch.cat(
            (output[:, :self.s_c, :, :], Z_dist), dim=1)
        FakeNoisy = model_generator(x=y_forw, rev=True)  # [:, :self.s_c, :, :]
        return FakeNoisy

    def INN_denoiser(self,model_generator,x):
        output = model_generator(x=x)
        gaussian_scale = 1
        y_forw = torch.cat((output[:, :self.s_c, :, :], gaussian_scale * self.gaussian_batch(output[:, self.s_c:, :, :].shape)),
                           dim=1)
        fake_H = model_generator(x=y_forw, rev=True)[:, :self.s_c, :, :]
        return fake_H

    def test(self, self_ensemble=False):
        self.input = self.noisy_H


        self.net.eval()
        with torch.no_grad():
            if self_ensemble:
                forward_function = self.net.forward
                self.fake_H = self.forward_x8(self.input, forward_function)
            else:
                self.fake_H = self.net(x=self.input)
        self.net.train()

    def MC_test(self, sample_num=16, self_ensemble=False):
        self.input = self.noisy_H

        self.net.eval()
        with torch.no_grad():
            if self_ensemble:
                forward_function = self.net.forward
                self.fake_H = self.Multi_forward_x8(self.input, forward_function, sample_num)
            else:
                output = self.net(x=self.input)
                C=self.input.shape[1]
                fake_Hs = []
                for i in range(sample_num):
                    fake_Hs.append(self.net(output))
                fake_H = torch.cat(fake_Hs, dim=0)
                self.fake_H = fake_H.mean(dim=0, keepdim=True)

        self.net.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Denoised'] = self.fake_H.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['Noisy'] = self.noisy_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.net, self.opt['path']['strict_load'])

    def load_model_generator(self):
        load_path_model_generator= self.opt['path']['model_generator']
        if load_path_model_generator is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_model_generator))
            self.load_network(load_path_model_generator, self.model_generator, self.opt['path']['strict_load'])


    def save(self, iter_label):
        self.save_network(self.net, 'Unet', iter_label)

    def forward_x8(self, x, forward_function):
        C=x.shape[1]
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        noise_list = [x]
        for tf in 'v', 'h', 't':
            noise_list.extend([_transform(t, tf) for t in noise_list])

        lr_list = [forward_function(aug) for aug in noise_list]
        back_list = []
        for data in lr_list:
            back_list.append(data)
        sr_list = [forward_function(data) for data in back_list]

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def Multi_forward_x8(self, x, forward_function, sample_num=16):
        # C = x.shape[1]
        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        noise_list = [x]
        for tf in 'v', 'h', 't':
            noise_list.extend([_transform(t, tf) for t in noise_list])

        lr_list = [forward_function(aug) for aug in noise_list]
        sr_list = []
        for data in lr_list:
            fake_Hs = []
            for i in range(sample_num):
                fake_Hs.append(self.net(x=data, rev=True))
            fake_H = torch.cat(fake_Hs, dim=0)
            fake_H = fake_H.mean(dim=0, keepdim=True)
            sr_list.append(fake_H)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output