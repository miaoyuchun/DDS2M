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

# data = scipy.io.loadmat("images//Case3//wdc_h.mat")
# img_clean = data['img_clean']
#
# data = scipy.io.loadmat("best_result.mat")
# pred_avg = data['pred_avg']
#
# print(quality(img_clean*255, pred_avg*255))

def ssim(imagery1,imagery2):
    [m, n, k] = imagery1.shape
    [mm, nn, kk] = imagery1.shape
    m = min(m, mm)
    n = min(n, nn)
    k = min(k, kk)
    imagery1 = imagery1[0:m, 0:n, 0:k]
    imagery2 = imagery2[0:m, 0:n, 0:k]
    ssim = 0
    for i in range(k):
        ssim = ssim + compare_ssim(imagery1[:, :, i], imagery2[:, :, i])
    ssim = ssim/k
    return ssim

def sam(T, H):
    assert T.ndim ==3 and T.shape == H.shape
    t1 = sum(T*H)
    t2 = sum(T*T)
    t3 = sum(H*H)
    sam_all = np.arccos(t1/np.sqrt(t2 * t3 + 1e-10))
    return sam_all.mean()
