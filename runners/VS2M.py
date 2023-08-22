#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: yuchun   time: 2020/7/10
from collections import namedtuple
from runners.com_psnr import quality
from models import *
from models.fcn import fcn
from models.skip import skip
from models.losses import *
from models.noise import *
from utils.image_io import *
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
Result = namedtuple("Result", ['recon', 'psnr', 'step'])

class VS2M(object):
    def __init__(self, rank, image_noisy, image_clean, beta, num_iter, lr=0.001):
        self.beta = beta
        self.rank = rank
        self.channel = image_noisy.shape[2]
        self.image_size = image_noisy.shape[0]
        self.now_rank = self.rank
        self.image = np.reshape(image_noisy, (image_noisy.shape[0] * image_noisy.shape[1], image_noisy.shape[2]), order="F")
        self.image_clean = image_clean
        self.num_iter = num_iter
        self.image_net = None
        self.mask_net = None
        self.mse_loss = None
        self.learning_rate = lr
        self.parameters = None
        self.current_result = None
        self.input_depth = 1
        self.output_depth = 1
        self.exp_weight = 0.98
        self.best_result = None
        self.best_result_av = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.done = False
        self.ambient_out = None
        self.total_loss = None
        self.post = None
        self._init_all()
        self.out_avg = 0
        self.save_every = 1000
        self.o = torch.zeros((self.image_clean.shape[0] * self.image_clean.shape[1], self.image_clean.shape[2])).type(
            torch.cuda.FloatTensor)
        self.previous = np.zeros(self.image_clean.shape)


    def _init_images(self):
        self.original_image = self.image.copy()
        image = self.image
        self.image_torch = np_to_torch(image).type(torch.cuda.FloatTensor)
        self.image_torch = self.image_torch.squeeze(0)

    def _init_nets(self):
        pad = 'reflection'
        data_type = torch.cuda.FloatTensor
        self.image_net = []
        self.parameters = []
        for i in range(self.rank):
            net = skip(self.input_depth, self.output_depth,  num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up = [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [0, 0, 4, 4, 4, 4],
                           filter_size_down = [7, 7, 5, 5, 3, 3], filter_size_up = [7, 7, 5, 5, 3, 3],
                           upsample_mode='bilinear', downsample_mode='avg',
                           need_sigmoid=False, pad=pad, act_fun='LeakyReLU').type(data_type)
            self.parameters = [p for p in net.parameters()] + self.parameters
            self.image_net.append(net)
        self.mask_net = []
        for i in range(self.rank):
            net = fcn(self.image_clean.shape[2], self.image_clean.shape[2], num_hidden=[128, 256, 256, 128]).type(data_type)
            self.parameters = self.parameters + [p for p in net.parameters()]
            self.mask_net.append(net)

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.sp_loss = SPLoss().type(data_type)
        self.kl_loss = KLLoss().type(data_type)
        self.tv_loss = TVLoss3d().type(data_type)

    def _init_inputs(self):
        original_noise = torch_to_np(get_noise1(1, 'noise', (self.input_depth, self.image_clean.shape[0], self.image_clean.shape[1]), noise_type='u',
                                                                     var=10/10.).type(torch.cuda.FloatTensor).detach())
        self.image_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :]


        original_noise = torch_to_np(get_noise2(1, 'noise', self.image.shape[1], noise_type='u', var=10/ 10.).type(torch.cuda.FloatTensor).detach())
        self.mask_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :]
        self.mask_net_inputs = self.mask_net_inputs

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_loss()
        self._init_optimizer()

    def reinit(self):
        self._init_nets()
        self._init_optimizer()

    def optimize(self, image_noisy, image_clean, at, mask, iteration, logger, avg, update):
        self.num_iter = iteration
        # self.mask = torch.from_numpy(mask).cuda()
        self.image = np.reshape(image_noisy, (image_noisy.shape[0] * image_noisy.shape[1], image_noisy.shape[2]), order="F")
        self.out_avg = avg
        self.image_clean = image_clean
        self._init_images()
        for j in range(self.num_iter + 1):
            self.optimizer.zero_grad()
            self._optimization_closure(at, j)
            self._obtain_current_result(j)
            self._plot_closure(j, logger)
            if update:
                self.optimizer.step()
            else:
                break
        return self.current_result_av.recon, self.best_result_av.step, self.best_result_av.recon, self.best_result_av.psnr

    def _optimization_closure(self, at, j):
        at = at[0,0,0,0]
        m = 0
        M = self.image_net_inputs
        out = self.image_net[0](M)
        for i in range(1, self.now_rank):
            out = torch.cat((out, self.image_net[i](M)), 0)
        out = out[:, :, :self.image_clean.shape[0], :self.image_clean.shape[1]]
        self.image_out = out[m, :, :, :].squeeze().reshape((-1, 1))
        for m in range(1, self.now_rank):
            self.image_out = torch.cat((self.image_out, out[m, :, :, :].squeeze().reshape((-1, 1))), 1)
        self.image_out_np = torch_to_np(self.image_out)

        M = self.mask_net_inputs
        out = self.mask_net[0](M)
        for i in range(1, self.now_rank):
            out = torch.cat((out, self.mask_net[i](M)), 0)
        self.mask_out = out.squeeze(1)
        self.mask_out_np = torch_to_np(self.mask_out)
        self.image_com = self.image_out.mm(self.mask_out)
        self.image_com_np = np.matmul(self.image_out_np, self.mask_out_np)
        self.image_com_np = np.reshape(self.image_com_np, self.image_clean.shape, order='F')
        self.out_avg = self.out_avg * self.exp_weight + self.image_com_np * (1 - self.exp_weight)

        self.image_com_rescale = self.image_com * 2 - 1.0

        self.out_avg_rescale = self.out_avg * 2 - 1.0

        self.et = (self.image_torch - at.sqrt() * self.image_com_rescale) / (1 - at).sqrt()
        self.mean = torch.mean(self.et)
        self.var = torch.var(self.et)

        self.loss1 = self.mse_loss(self.image_com_rescale * at.sqrt(), self.image_torch)
        self.loss2 = self.kl_loss(self.et)
        self.image_com_rescale = torch.reshape(self.image_com_rescale, (self.image_size,self.image_size,self.channel)).permute(2,0,1).unsqueeze(0)
        self.loss3 = self.tv_loss(self.image_com_rescale)
        self.total_loss = self.loss1 + self.beta * self.loss3
        self.total_loss.backward(retain_graph=True)
        self.res = np.sqrt(np.sum(np.square(self.out_avg - self.previous)) / np.sum(np.square(self.previous)))
        self.previous = self.out_avg

    def _obtain_current_result(self, step):
        self.psnr = quality(self.image_clean, np.clip(self.image_com_np.astype(np.float64),0,1))
        self.psnr_av = quality(self.image_clean, np.clip(self.out_avg.astype(np.float64),0,1))
        self.current_result = Result(recon=np.clip(self.image_com_np,0,1),  psnr=self.psnr, step=step)
        self.current_result_av = Result(recon=np.clip(self.out_avg,0,1),  psnr=self.psnr_av, step=step)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result
        if self.best_result_av is None or self.best_result_av.psnr < self.current_result_av.psnr:
            self.best_result_av = self.current_result_av

    def _plot_closure(self, step, logger):
        logger.info('--------->Iteration %05d  kl_loss %f tol_loss %f   current_psnr: %f  max_psnr %f  current_psnr_av: %f max_psnr_av: %f mean: %f var: %f res: %f  step: %f'   % (step, self.loss2.item(), self.total_loss.item(),
                                                                                self.current_result.psnr, self.best_result.psnr,
                                                                                self.current_result_av.psnr, self.best_result_av.psnr, self.mean, self.var, self.res, self.current_result_av.step ))



