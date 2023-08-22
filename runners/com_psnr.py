#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: yuchun   time: 2020/7/7
import numpy as np
import scipy.io as sio
import math



def quality (imagery1, imagery2):
    imagery1 = imagery1 * 255
    imagery2 = imagery2 * 255
    Nway = imagery1.shape
    psnr = np.zeros((Nway[2], 1))
    # ssim = psnr
    for i in range(Nway[2]):
        psnr[i] = psnr_index(imagery1[:,:,i],imagery2[:,:,i])
        # ssim[i] = ssim_index(imagery1[:,:,i],imagery2[:,:,i])
    psnr = np.mean(psnr)
    # ssim = np.mean(ssim)
    return psnr

def psnr_index (x, y):
    mse = np.mean((x-y)**2)
    p = 10 * math.log10(255.0**2/mse)
    return p
